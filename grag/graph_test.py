"""The one test run: Self-RAG #6 (IRCoT on the multi route) and the final GraphRAG (graph_route.FINAL: graph walk +
linked grading + IRCoT fallback; records off) on Self-RAG's 24 test questions, which neither system has seen.

It saves grag/results/test_once.json and refuses to run again once that file exists.

Run from graph_rag_in_depth/ (Qdrant + kg-neo4j running; OpenAI key sourced), once:
    uv run grag/graph_test.py
"""

import json
import sys

import grag_bridge
from eval_harness import load_golden

import crag_eval
import graph_route
import graph_tune
import selfrag_golden as sg
import selfrag_tune
from base_tune import KINDS

RESULT = grag_bridge.ROOT / "grag" / "results" / "test_once.json"


def by_kind(rows: dict) -> dict:
    return {k: f"{sum(r['correct'] for r in rows.values() if r['kind'] == k):g}/"
               f"{sum(r['kind'] == k for r in rows.values())}" for k in KINDS + ("solo",)
            if any(r["kind"] == k for r in rows.values())}


def main():
    if RESULT.exists():
        sys.exit(f"The test has already been run: {RESULT}")
    golden = {g.id: g for g in load_golden(sg.golden_name("test"))}
    base = selfrag_tune.run(golden, split="test")  # Self-RAG #6 as it is: IRCoT on multi
    graph = graph_tune.run(golden, split="test", **graph_route.FINAL)
    out = {}
    for name, rows in (("selfrag_ircot", base), ("graphrag_final", graph)):
        s = crag_eval.summary(rows, None)
        s["by_kind"] = by_kind(rows)
        multi = [i for i in golden if base[i]["route"] == "multi"]
        s["multi"] = f"{sum(rows[i]['correct'] for i in multi):g}/{len(multi)}"
        out[name] = {"summary": s, "questions": rows}
    out["changed"] = {i: f"{base[i]['correct']:g} -> {graph[i]['correct']:g}" for i in golden
                      if base[i]["correct"] != graph[i]["correct"]}
    out["ircot_fallback"] = [i for i, r in graph.items() if (r.get("walk") or {}).get("ircot_fallback")]
    RESULT.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    b, g = out["selfrag_ircot"]["summary"], out["graphrag_final"]["summary"]
    print(f"\nTEST (24): Self-RAG #6 {b['correct']:.3f} -> GraphRAG {g['correct']:.3f} (1 vote); "
          f"{b['correct3']:.3f} -> {g['correct3']:.3f} (3 votes)")
    print(f"routes {b['routes']}; multi questions: {b['multi']} -> {g['multi']}")
    print(f"by kind  Self-RAG {b['by_kind']}\n         GraphRAG {g['by_kind']}")
    print(f"$/question {b['cost_per_question']:.5f} -> {g['cost_per_question']:.5f}; "
          f"p50 {b['p50_ms'] / 1000:.1f} -> {g['p50_ms'] / 1000:.1f} s")
    print(f"IRCoT fallback used on: {out['ircot_fallback'] or 'none'}")
    print(f"changed: {out['changed'] or 'none'}")


if __name__ == "__main__":
    main()
