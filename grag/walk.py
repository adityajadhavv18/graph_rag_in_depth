"""Phase 3: the graph walk that replaces IRCoT's search -> reason -> search loop (free: Neo4j + local bge, no LLM).

    1. SEEDS   which match is the question about?
               a) local bge: the question against the 2,156 match names ("2011 Cricket World Cup final",
                  "Indian cricket team in England in 2025 › Test series › 1st Test (20–24 June 2025)"),
                  with "first Test" written "1st Test" on both sides; the best one. Tour matches and
                  "unofficial Tests" only when the question says so (bge reads "1st unofficial Test" as "1st Test")
               b) the matches behind f6a's fused top SEED_SEARCH chunks (Neo4j DESCRIBED_IN), after it
    2. HOP 1   Neo4j: the first seed with a link -[HELD_AT | PLAYER_OF_MATCH | TOP_BATTER | TOP_BOWLER]-> a ground
               or player that has its own article. Which link: local bge, the question without its match words
               against a plain sentence for each ("the player of the match: Mitchell Starc"), sprint 09's scored
               arrows (with the whole question, "the 2024 IPL final" pulled chain-07 to the ground: 0.611 vs 0.605)
    3. HOP 2   that article's chunks, ranked by local bge (Qdrant, filtered to that article) against the target's
               name + the question without its match words ("Mitchell Starc: Which school did player match
               attend?"): the match words point back at hop 1 and drown out what is asked (Starc's school
               chunk: rank 32 with the whole question, 1 with this). The best N_CHUNKS, plus the chunk holding
               the line that made the hop ("Venue: Wankhede Stadium, Mumbai"), so the answer step sees both hops

It starts only from match names, never from a single ordinary word (sprint 09). A match listed inside a ground's
own article may hop to that article: it was played there.
"""

import re
from functools import cache

import numpy as np
from qdrant_client import models

import grag_bridge
import crag
import retrieve
from fallback_index import FALLBACK
from load_neo4j import driver
from rag_fusion import NOTE

NAME_VECS = grag_bridge.ROOT / "data" / "graph" / "match_name_vecs.npz"
HOP1 = ["HELD_AT", "PLAYER_OF_MATCH", "TOP_BATTER", "TOP_BOWLER"]
SEED_NAMES = 1
SEED_MIN = 0.0  # every multi question's best name scores 0.67-0.82 (dry run); no cut
SEED_SEARCH = 3
N_CHUNKS = 3
LINK_TEXT = {"HELD_AT": "the ground or stadium that hosted the match: {}", "PLAYER_OF_MATCH": "the player of the match: {}",
             "TOP_BATTER": "the top run-scorer in the match: {}", "TOP_BOWLER": "the best bowler in the match: {}"}
MATCH_WORDS = {"final", "ipl", "test", "odi", "t20i", "tour", "series", "world", "cup", "championship", "league",
               "premier", "indian", "the", "of", "in", "s"}
SIDE_GAMES = re.compile(r"unofficial|Tour matches|warm-up", re.I)
ORDINAL = {"first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th", "sixth": "6th"}


def ordinals(text: str) -> str:
    """"first Test" / "First Test" -> "1st Test", so the question and the match names spell it the same way."""
    return re.sub(r"\b(first|second|third|fourth|fifth|sixth)(?=\s+(?:unofficial\s+)?(?:Test|ODI|T20I|match)\b)",
                  lambda m: ORDINAL[m.group(1).lower()], text, flags=re.I)


@cache
def _session():
    return driver().session()  # one per process; questions run one after another


@cache
def _names() -> tuple[list[str], list[str], np.ndarray]:
    rows = _session().run("MATCH (m:CricketMatch) RETURN m.id AS id, m.name AS name ORDER BY m.id").data()
    ids, names = [r["id"] for r in rows], [r["name"] for r in rows]
    if NAME_VECS.exists():
        saved = np.load(NAME_VECS, allow_pickle=False)
        if list(saved["ids"]) == ids:
            return ids, names, saved["vecs"]
    vecs = np.array(list(retrieve._dense().embed([ordinals(n) for n in names], batch_size=8)))
    np.savez(NAME_VECS, ids=np.array(ids), vecs=vecs)
    return ids, names, vecs


def hop2_query(question: str, seed_name: str, target: str) -> str:
    """"Mitchell Starc: Which school did player match attend?" (the seed's words, years and ordinals removed)."""
    drop = {w.lower() for w in re.findall(r"[\w–]+", seed_name)} | MATCH_WORDS
    words = [w for w in re.findall(r"[\w'–]+|[?,]", ordinals(question))
             if w.lower().strip("'s") not in drop and not re.fullmatch(r"\d{4}(–\d{2})?|\d(st|nd|rd|th)", w)]
    return f"{target}: {' '.join(words)}"


def _hit(chunk_id: str, score: float = 0.0) -> retrieve.Hit:
    c = crag._by_id[chunk_id]
    return retrieve.Hit(c.chunk_id, c.doc_id, list(c.path), c.text, score)


def _rank_chunks(question_vec: list[float], docs: list[str], k: int) -> list[retrieve.Hit]:
    """The article chunks closest to the question, over the main and the fallback library."""
    flt = models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=docs))])
    hits = []
    for coll in (NOTE, FALLBACK):
        pts = retrieve._qdrant().query_points(coll, query=question_vec, using=retrieve.DENSE, query_filter=flt,
                                              limit=k).points
        hits += retrieve._hits(pts)
    return sorted(hits, key=lambda h: -h.score)[:k]


def seeds(question: str, fused: list) -> dict:
    qv = next(retrieve._dense().query_embed(ordinals(question)))
    ids, names, vecs = _names()
    sims = vecs @ qv
    side_ok = bool(SIDE_GAMES.search(question)) or "tour match" in question.lower()
    top = [i for i in np.argsort(-sims) if side_ok or not SIDE_GAMES.search(names[i])][:SEED_NAMES]
    by_name = [(ids[i], names[i], round(float(sims[i]), 3)) for i in top if sims[i] >= SEED_MIN]
    by_search = _session().run(
        "MATCH (m:CricketMatch)-[:DESCRIBED_IN]->(c:CricketChunk) WHERE c.id IN $ids "
        "RETURN DISTINCT m.id AS id, m.name AS name", ids=[h.chunk_id for h in fused[:SEED_SEARCH]]).data()
    return {"vector": qv.tolist(), "by_name": by_name, "by_search": [(r["id"], r["name"]) for r in by_search]}


def walk(question: str, fused: list) -> tuple[list[retrieve.Hit], dict]:
    """(graph chunks for the answer step, log). No chunks when no seed leads to an article."""
    s = seeds(question, fused)
    names = {i: n for i, n, *_ in s["by_name"]} | dict(s["by_search"])
    seed_ids = list(dict.fromkeys([i for i, *_ in s["by_name"]] + [i for i, _ in s["by_search"]]))
    log = {"by_name": s["by_name"], "by_search": s["by_search"], "hops": [], "chunks": []}
    if not seed_ids:
        return [], log
    hops = _session().run(
        "MATCH (m:CricketMatch)-[r]->(e:CricketEntity) WHERE m.id IN $ids AND type(r) IN $types "
        "AND e.doc_id IS NOT NULL "
        "RETURN m.id AS match, type(r) AS rel, r.line AS line, r.chunk AS box, e.name AS to, e.doc_id AS doc",
        ids=seed_ids, types=HOP1).data()
    if not hops:
        return [], log
    first = min(seed_ids.index(h["match"]) for h in hops)  # the first seed that leads anywhere
    hops = [h for h in hops if h["match"] == seed_ids[first]]
    texts = [LINK_TEXT[h["rel"]].format(h["to"]) for h in hops]
    asked = hop2_query(question, names[seed_ids[first]], "").lstrip(": ")  # the question without its match words
    scores = np.array(list(retrieve._dense().embed(texts, batch_size=8))) @ next(retrieve._dense().query_embed(asked))
    best = hops[int(np.argmax(scores))]
    log["hops"] = [(h["rel"], h["to"], round(float(x), 3)) for h, x in zip(hops, scores)]
    query = hop2_query(question, names[best["match"]], best["to"])
    qv = next(retrieve._dense().query_embed(query)).tolist()
    article = _rank_chunks(qv, [best["doc"]], N_CHUNKS)
    box = [_hit(best["box"])] if best["box"] else []
    out = list({h.chunk_id: h for h in box + article}.values())
    log["chunks"] = [h.chunk_id for h in out]
    log["via"] = (best["match"], best["rel"], best["to"], best["line"])
    log["hop2_query"] = query
    return out, log
