"""Phase 5 (records mini-phase): the record index's best passages get guaranteed seats in the LLM rerank pool.

    question ─┬─ f6a lists -> RRF -> fused list                                   [as before]
              ├─ (multi) graph walk chunks                                         [as before]
              └─ records: dense + BM25 + SPLADE on cricket_records -> RRF -> top N_RECORDS   (new)
    rerank pool (20) = graph chunks + record passages + the fused list -> top 5 (+ graph chunks on multi)
    -> CRAG (linked grading) -> Self-RAG -> answer; the IRCoT fallback stays as it is

On the single route this replaces f6a's search with the same search plus the record seats; the none route's
"unsure -> single" path gets them too. Self-RAG's retry and the IRCoT fallback search as before.
install() goes after graph_route.install(...).
"""

from functools import partial

import baseline
import crag
import graph_route
import ladder
import rag_fusion
import records_index
import retrieve
import routes
import stacked
from ingest import BM25, SPLADE

N_RECORDS = 3
LOG: dict[str, list[str]] = {}
STATE = {"collection": records_index.RECORDS}
_ORIGINAL_SINGLE = stacked.F6A_ROUTES["single"]
NOTED = records_index.OUT.parent / "records_noted.jsonl"
CUSTOM = records_index.OUT.parent / "records_custom.jsonl"


def _use(noted: bool) -> None:
    """Plain record lines, or the ones with a scope note (records_notes.py); crag._by_id lets labelled() and the
    grader read them."""
    path, coll = (NOTED, "cricket_records_noted") if noted else (records_index.OUT, records_index.RECORDS)
    cs = records_index.load(path)
    if noted and CUSTOM.exists():
        cs += records_index.load(CUSTOM)  # hand-written lines (records_notes.CUSTOM)
    crag._by_id.update({c.chunk_id: c for c in cs})
    STATE["collection"] = coll


_use(False)


def records_search(question: str, k: int = N_RECORDS) -> list:
    coll = STATE["collection"]
    lists = {"dense": retrieve.dense_search(question, 20, coll),
             "bm25": retrieve.sparse_search(question, 20, BM25, coll),
             "splade": retrieve.sparse_search(question, 20, SPLADE, coll)}
    hits = retrieve.rrf_fuse(lists)[:k]
    LOG[question] = [h.chunk_id for h in hits]
    return hits


def single_with_records(question: str, k: int = graph_route.TOP_K):
    routes._start()
    opts = crag._f5a(rerank=False)
    opts.pop("rerank")
    list_opts, fuse_opts = rag_fusion.split(opts)
    fused = rag_fusion.fuse(rag_fusion.fusion_lists(question, **list_opts), **fuse_opts)
    recs = records_search(question)
    ids = {h.chunk_id for h in recs}
    pool = recs + [h for h in fused if h.chunk_id not in ids][:rag_fusion.RERANK_POOL - len(recs)]
    hits = rag_fusion.llm_rerank(question, pool, ladder.labelled)[:k]
    routes.USAGE.update(searches=1, thoughts=[], added=[[h.chunk_id for h in hits]])
    return routes.answer_from(question, hits, [])


def install(on: bool = True, noted: bool = False) -> None:
    LOG.clear()
    _use(noted)
    single = single_with_records if on else _ORIGINAL_SINGLE
    baseline.ROUTE_FNS["single"] = single
    stacked.F6A_ROUTES["single"] = single  # Self-RAG's "unsure -> single" reads this one
    graph_route.EXTRA_SEATS[:] = [records_search] if on else []
