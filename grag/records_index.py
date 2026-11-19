"""Phase 5 (records mini-phase): an index of the corpus's own record sentences, built by code ($0, no LLM).

Why: a record often sits as one line inside a long chunk ("Alzarri Joseph (Mumbai Indians) took the best bowling
figures in the IPL." inside a 2019 IPL chunk of match boxes). Dense and BM25 search miss it, only SPLADE finds it,
and RRF ranks it 46th, below the reranker's top 20. As its own short passage it is easy to find and easy to read.

    every article (main + the 9 fallback biographies) -> each line under its heading path
      keep a line that claims a record (highest / lowest / best figures / most runs|wickets|sixes ... / fastest /
      youngest / oldest / record / triple or double century / "first ... to"), whole if <= MAX_WORDS words (so
      "Karun Nair scoring 303 not out. This was India's highest total ..." stays together), else each matching
      sentence with the one before it; score tables (mostly numbers) are skipped
      tag it from where it comes, by code, the same rule for every line:
        [IPL records · Twenty20]            a line from an IPL article, or one that names the IPL
        [International records · Test]      otherwise; the format from the line, then its headings, then the
        [International records · ODI] ...   article title (Ashes -> Test, Champions Trophy -> ODI, ...)
    -> Qdrant collection cricket_records: dense (bge) + BM25 + SPLADE over "title › headings" + the tagged line

Nothing is written by hand: no record is typed in, so no tune answer is put into the library.
Output: data/records/records.jsonl and the Qdrant collection. Embeddings are cached in data/records/cache/.

Run from graph_rag_in_depth/ (Qdrant running):
    uv run grag/records_index.py
"""

import json
import re
from collections import Counter

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

import grag_bridge
import chunks
import fallback_index
import ingest

RECORDS = "cricket_records"
OUT = grag_bridge.ROOT / "data" / "records" / "records.jsonl"
MAX_WORDS = 80
RECORD = re.compile(r"\b(highest|lowest|best(?: bowling)? figures|most (?:runs|wickets|sixes|fours|centuries|hundreds|"
                    r"catches|dismissals)|fastest|slowest|youngest|oldest|records?|triple[- ]centur(?:y|ies)|"
                    r"double[- ]centur(?:y|ies))\b|\bfirst\b[^.]{0,40}\bto\b", re.I)
IPL = re.compile(r"\bIPL\b|Indian Premier League")
FORMATS = [("Twenty20", IPL), ("T20I", re.compile(r"\bT20Is?\b|Twenty20 International|T20 World Cup|World Twenty20")),
           ("ODI", re.compile(r"\bODIs?\b|One Day International|Cricket World Cup|Champions Trophy")),
           ("Test", re.compile(r"\bTests?\b|Test cricket|Test Championship|Ashes")),
           ("Twenty20", re.compile(r"Twenty20|\bT20\b"))]


def fmt(*texts: str) -> str | None:
    for text in texts:
        for name, pat in FORMATS:
            if pat.search(text):
                return name
    return None


def tag(line: str, path: tuple[str, ...]) -> str:
    title, heads = path[0], " › ".join(path[1:])
    if IPL.search(title) or IPL.search(line):
        return "[IPL records · Twenty20]"
    f = fmt(line, heads, title)
    return f"[International records · {f}]" if f else "[International records]"


def is_table(line: str) -> bool:
    words = line.split()
    return len(words) > 30 and sum(bool(re.search(r"\d", w)) for w in words) / len(words) > 0.3


def record_lines(body: str) -> list[str]:
    out = []
    for line in body.splitlines():
        line = line.strip().lstrip("-• ").strip()
        if len(line) < 30 or is_table(line) or not RECORD.search(line):
            continue
        if len(line.split()) <= MAX_WORDS:
            out.append(line)
            continue
        sents = re.split(r"(?<=[.!?])\s+", line)
        out += [" ".join(sents[max(0, i - 1):i + 1]) for i, s in enumerate(sents) if RECORD.search(s)]
    return out


def build_chunks() -> list[chunks.Chunk]:
    out, seen = [], set()
    for d in chunks.included_docs() + fallback_index.left_out():
        for n, (path, body) in enumerate(chunks.sections((chunks.TEXT_DIR / f"{d}.txt").read_text())):
            for line in record_lines(body):
                if line in seen:
                    continue
                seen.add(line)
                out.append(chunks.Chunk(f"{d}::rec::{len(out):05d}", d, path, f"{tag(line, path)} {line}",
                                        len(line.split())))
    return out


def index(cs: list[chunks.Chunk], collection: str = RECORDS) -> None:
    """Corrective #5's fallback builder (fallback_index.build), over the header_match text, with its own cache."""
    ingest.CACHE = OUT.parent / "cache"
    texts = [ingest.VARIANTS["header_match"](c) for c in cs]
    dense = TextEmbedding(chunks.EMBED_MODEL)
    vectors = {
        ingest.DENSE: ingest.cached(f"dense_{chunks.EMBED_MODEL}", texts, lambda ts: dense.passage_embed(ts, batch_size=8)),
        ingest.BM25: list(SparseTextEmbedding(ingest.BM25_MODEL, avg_len=ingest.avg_words(texts)).embed(texts)),
        ingest.SPLADE: ingest.cached(f"splade_{ingest.SPLADE_MODEL}", texts,
                                     lambda ts: SparseTextEmbedding(ingest.SPLADE_MODEL).embed(ts, batch_size=8)),
    }
    client = QdrantClient(url=ingest.QDRANT_URL, timeout=300)
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection,
        vectors_config={ingest.DENSE: models.VectorParams(size=len(vectors[ingest.DENSE][0]),
                                                          distance=models.Distance.COSINE)},
        sparse_vectors_config={ingest.BM25: models.SparseVectorParams(modifier=models.Modifier.IDF),
                               ingest.SPLADE: models.SparseVectorParams()})
    client.upload_points(collection, points=(models.PointStruct(
        id=ingest.point_id(c.chunk_id),
        vector={ingest.DENSE: vectors[ingest.DENSE][i].tolist(), ingest.BM25: ingest.sparse(vectors[ingest.BM25][i]),
                ingest.SPLADE: ingest.sparse(vectors[ingest.SPLADE][i])},
        payload={"chunk_id": c.chunk_id, "doc_id": c.doc_id, "path": list(c.path), "text": c.text})
        for i, c in enumerate(cs)), batch_size=32, wait=True)


def load(path=OUT) -> list[chunks.Chunk]:
    return [chunks.Chunk(r["chunk_id"], r["doc_id"], tuple(r["path"]), r["text"], r["n_tokens"])
            for r in map(json.loads, path.read_text().splitlines())]


def main():
    cs = build_chunks()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps({"chunk_id": c.chunk_id, "doc_id": c.doc_id, "path": list(c.path),
                                       "text": c.text, "n_tokens": c.n_tokens}, ensure_ascii=False) + "\n" for c in cs))
    print(f"record lines: {len(cs)}  by tag: {Counter(c.text.split(']')[0] + ']' for c in cs).most_common()}")
    for key in ("Alzarri Joseph", "175 not out", "Karun Nair scoring 303"):
        print(f"  {key!r}: {[c.text[:150] for c in cs if key in c.text]}")
    index(cs)
    print(f"Qdrant {RECORDS}: {QdrantClient(url=ingest.QDRANT_URL).count(RECORDS).count} points")


if __name__ == "__main__":
    main()
