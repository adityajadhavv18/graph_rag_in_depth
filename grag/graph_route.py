"""Phase 3: the multi route with the graph walk in place of IRCoT. Everything around it is unchanged.

    question (router: multi)
      ├─ f6a's lists (4 keyword rewrites + the question; dense / BM25 / SPLADE) -> RRF -> fused list
      └─ graph walk (walk.py) -> up to 4 graph chunks
    LLM rerank pool (20) = the graph chunks (a guaranteed seat) + the fused list   [same reranker]
    passages = rerank top 5 + any graph chunk not already among them (<= 9; IRCoT's median pool is 9)
    -> routes.answer_from: CRAG 3f -> Self-RAG -> answer                            [unchanged]

A question whose walk finds nothing gets plain f6a (the single route's search), as Aditya chose.
install() puts this route in baseline.ROUTE_FNS["multi"], on top of Self-RAG's install.
"""

import baseline
import crag
import ladder
import rag_fusion
import routes
import self_rag
import walk

TOP_K = 5
LOG: dict[str, dict] = {}


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
    pool = graph + [h for h in fused if h.chunk_id not in ids][:rag_fusion.RERANK_POOL - len(graph)]
    top = rag_fusion.llm_rerank(question, pool, ladder.labelled)[:k]
    shown = {h.chunk_id for h in top}
    out = top + [h for h in graph if h.chunk_id not in shown]
    log["passages"] = [h.chunk_id for h in out]
    return out


def graph_multi(question: str):
    routes._start()
    hits = graph_search(question)
    if hits is None:
        return routes.one_search(question, search="f6a")
    routes.USAGE.update(searches=1, thoughts=[], added=[[h.chunk_id for h in hits]])
    return routes.answer_from(question, hits, [])


def install(selfrag_config: dict | None) -> None:
    self_rag.install(selfrag_config)
    LOG.clear()
    baseline.ROUTE_FNS["multi"] = graph_multi
