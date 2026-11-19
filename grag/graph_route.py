"""Phase 3: the multi route with the graph walk in place of IRCoT. Everything around it is unchanged.

    question (router: multi)
      ├─ f6a's lists (4 keyword rewrites + the question; dense / BM25 / SPLADE) -> RRF -> fused list
      └─ graph walk (walk.py) -> up to 4 graph chunks
    LLM rerank pool (20) = the graph chunks (a guaranteed seat) + the fused list   [same reranker]
    passages = rerank top 5 + any graph chunk not already among them (<= 9; IRCoT's median pool is 9)
    -> routes.answer_from: CRAG 3f -> Self-RAG -> answer                            [unchanged]

A question whose walk finds nothing gets plain f6a (the single route's search), as Aditya chose.
install() puts this route in baseline.ROUTE_FNS["multi"], on top of Self-RAG's install.

LINKED grading (Phase 4, "B"): CRAG's grader reads one passage at a time against the whole question, so a hop-2
passage looks useless ("Starc attended Homebush Boys High School" says nothing about the 2024 IPL final: 0). With
LINKED on, each graph passage is graded against the question with the link that brought it filled in:
    "Which school did the player of the match in the 2024 IPL final attend? (Known from the match record: in
     2024 Indian Premier League final, the player of the match was Mitchell Starc.)"
Every other passage is graded exactly as before; CRAG's prompt, bars and fallback are unchanged.

ADD ("add, don't replace", Phase 4): the passages are f6a's own top 5 (the fused list's top 20, LLM-reranked:
exactly what the single route reads) followed by the graph chunks, instead of graph chunks taking seats in the
rerank pool. The graph can then only add evidence, never push f6a's out (twin-03: three Mitchell Johnson
biography chunks took top-5 seats and pushed out the over-by-over passage the base had graded 10).
Tried on tune, it lost (0.929 -> 0.904): graph chunks at positions 6-9 are read after f6a's look-alikes. Off.

FALLBACK (Phase 4): graph first; if CRAG throws every passage away (INCORRECT), the question runs through the old
IRCoT loop instead (its search -> reason -> search finds passages a walk can't reach, e.g. twin-03's last over,
which IRCoT found on its second search). The graph attempt's tokens and time are still counted.
"""

from functools import partial

from eval_harness import PipelineOutput

import llm_cache

from collections import defaultdict

import grader

import baseline
import crag
import ladder
import rag_fusion
import routes
import self_rag
import walk

TOP_K = 5
LOG: dict[str, dict] = {}
LINKED = {"on": False}
ADD = {"on": False}
FALLBACK = {"on": False}
EXTRA_SEATS: list = []  # more searches whose hits get guaranteed rerank seats (records_route.py)
IRCOT = partial(routes.search_loop, search="f6a")  # Adaptive #4's multi route, unchanged
HOP_FACT = {"HELD_AT": "the match was played at {}", "PLAYER_OF_MATCH": "the player of the match was {}",
            "TOP_BATTER": "{} was the top run-scorer", "TOP_BOWLER": "{} was the best bowler"}
_HOP_Q: dict[str, str] = {}  # graph passage text (as the grader reads it) -> the question it is graded against
_grade = grader.grade


def linked_grade(question: str, passages: list[str]) -> tuple[list[int], list]:
    """grader.grade, except graph passages are graded against their linked question (same prompt, same scale)."""
    groups = defaultdict(list)
    for i, p in enumerate(passages):
        groups[_HOP_Q.get(p, question)].append(i)
    if len(groups) == 1 and question in groups:
        return _grade(question, passages)
    scores, replies = [0] * len(passages), []
    for q, idx in groups.items():
        s, r = _grade(q, [passages[i] for i in idx])
        replies += r
        for i, x in zip(idx, s):
            scores[i] = x
    return scores, replies


def graph_search(question: str, k: int = TOP_K) -> list | None:
    opts = crag._f5a(rerank=False)
    opts.pop("rerank")
    list_opts, fuse_opts = rag_fusion.split(opts)
    fused = rag_fusion.fuse(rag_fusion.fusion_lists(question, **list_opts), **fuse_opts)
    graph, log = walk.walk(question, fused)
    LOG[question] = log
    if not graph:
        log["fallback"] = True
        return None
    ids = {h.chunk_id for h in graph}
    if ADD["on"]:  # f6a's own top k (same rerank call as f6a, so cached), then the graph's chunks
        pool = fused[:rag_fusion.RERANK_POOL]
    else:  # the graph chunks (and any extra searches' hits) take guaranteed seats in the rerank pool
        seats = list({h.chunk_id: h for h in graph + [x for fn in EXTRA_SEATS for x in fn(question)]}.values())
        sid = {h.chunk_id for h in seats}
        pool = seats + [h for h in fused if h.chunk_id not in sid][:rag_fusion.RERANK_POOL - len(seats)]
    top = rag_fusion.llm_rerank(question, pool, ladder.labelled)[:k]
    shown = {h.chunk_id for h in top}
    out = top + [h for h in graph if h.chunk_id not in shown]
    log["passages"] = [h.chunk_id for h in out]
    return out


def graph_multi(question: str):
    routes._start()
    hits = graph_search(question)
    if hits is None:  # plain f6a, or the single route's search when records are on
        return baseline.ROUTE_FNS["single"](question)
    routes.USAGE.update(searches=1, thoughts=[], added=[[h.chunk_id for h in hits]])
    log = LOG[question]
    if LINKED["on"]:
        _, rel, to, _ = log["via"]
        known = f"{question} (Known from the match record: in {log['via_name']}, {HOP_FACT[rel].format(to)}.)"
        _HOP_Q.update({ladder.labelled(h): known for h in hits if h.chunk_id in set(log["chunks"])})
        log["linked_question"] = known
    try:
        out = routes.answer_from(question, hits, [])
    finally:
        _HOP_Q.clear()
    if not (FALLBACK["on"] and crag.LOG[question]["action"] == "INCORRECT"):
        return out
    log["ircot_fallback"] = True
    spent_ms = llm_cache.CACHED_MS["ms"]  # search_loop resets the counters; keep the graph attempt's time
    again = IRCOT(question)
    llm_cache.CACHED_MS["ms"] += spent_ms
    return PipelineOutput(answer=again.answer, contexts=again.contexts, context_ids=again.context_ids,
                          tokens_in=out.tokens_in + again.tokens_in, tokens_out=out.tokens_out + again.tokens_out)


FINAL = {"linked": True, "add": False, "fallback": True}  # tune 0.942: graph + linked grading + IRCoT fallback


def install(selfrag_config: dict | None, linked: bool = False, add: bool = False, fallback: bool = False) -> None:
    self_rag.install(selfrag_config)
    LOG.clear()
    LINKED["on"], ADD["on"], FALLBACK["on"] = linked, add, fallback
    grader.grade = linked_grade if linked else _grade
    baseline.ROUTE_FNS["multi"] = graph_multi
