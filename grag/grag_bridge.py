"""Reuse Self-RAG #6 as-is (and through it Corrective #5, Adaptive #4, RAG Fusion #3 and Contextual RAG #2): router,
f6a search, CRAG 3f with its fallback library, answer prompt v2, and the Self-RAG layer (sure check, candidates,
gpt-4o-mini critic, retry).

Self-RAG's folder goes on sys.path; its selfrag_bridge adds Corrective's, which adds the rest. Every module there is
imported by bare name, so ours must not reuse any of their names:
    Contextual #2: chunks, ingest, retrieve, hyde, llm_rerank, notes, ladder, answer
    Fusion #3:     contextual, rewrite, rag_fusion, diagnose, fusion_ladder, fusion_answer
    Adaptive #4:   bridge, bar, golden_check, llm_cache, routes, route_eval, loop_knobs, router, router_eval,
                   end_to_end, stacked, answer_tuning, test_run
    Corrective #5: crag_bridge, crag, crag_eval, grader, grader_eval, baseline, chunk_labels, crag_golden_check,
                   fallback_index, laya_eval, model_check, test_once, test_relabel, compare_archs
    Self-RAG #6:   selfrag_bridge, replay_3f, selfrag_golden, base_tune, self_rag, smoke, selfrag_tune

Self-RAG's LLM cache is copied once into this build's data/cache/llm.json and llm_cache is pointed at the copy:
replays are free and new calls land here. The system GraphRAG plugs into is the full Self-RAG one (tune 0.897);
GraphRAG replaces only the IRCoT loop on the router's multi route.
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAGS = ROOT.parent
SELFRAG = RAGS / "self_rag_in_depth" / "selfrag"
sys.path.append(str(SELFRAG))

import selfrag_bridge  # noqa: E402  (Self-RAG's bridge: adds Corrective, Adaptive, Fusion, Contextual; installs 3f)
import llm_cache  # noqa: E402
import selfrag_tune  # noqa: E402

CACHE = ROOT / "data" / "cache" / "llm.json"
if not CACHE.exists():
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(selfrag_bridge.CACHE, CACHE)
llm_cache.CACHE = CACHE

SELFRAG_CONFIG = selfrag_tune.CONFIG  # {"sure": True, "retries": 1}
SELFRAG_RESULTS = SELFRAG / "results"
