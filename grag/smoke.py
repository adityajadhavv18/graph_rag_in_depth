"""Phase 3: the full system with the graph walk in place of IRCoT, on 5 tune questions the router sends to multi,
against the base's saved answers (phase1.json, Self-RAG #6 = 0.897):

    chain-02, chain-04, chain-07   the base gets them wrong (wrong ground, wrong ground, refused)
    multi-01, pair-01              the base gets them right: nothing should break

Results: grag/results/phase3_smoke.json.

Run from graph_rag_in_depth/ (Qdrant + kg-neo4j running; OpenAI key sourced):
    uv run grag/smoke.py
"""

import json
import time

import grag_bridge
from eval_harness import load_golden, settings
from eval_harness.metrics import cost_usd, is_refusal

import baseline
import graph_route
import route_eval
import self_rag
import selfrag_golden as sg

IDS = ("chain-02", "chain-04", "chain-07", "multi-01", "pair-01")
RESULTS = grag_bridge.ROOT / "grag" / "results"


def main():
    golden = {g.id: g for g in load_golden(sg.golden_name("tune"))}
    base = json.loads((RESULTS / "phase1.json").read_text())["questions"]
    graph_route.install(grag_bridge.SELFRAG_CONFIG)
    picks, rows = {}, {}
    pipeline = baseline.routed(picks)
    for i in IDS:
        g = golden[i]
        start = time.perf_counter()
        out = pipeline(g.question)
        secs = time.perf_counter() - start
        if g.question_type == "no_answer":
            correct = float(is_refusal(out.answer))
        else:
            correct = 0.0 if is_refusal(out.answer) else route_eval.grade(g.question, g.ground_truth, out.answer)
        walk_log = graph_route.LOG.get(g.question, {})
        rows[i] = {"base": base[i]["correct"], "now": correct, "route": picks[g.question], "answer": out.answer,
                   "base_answer": base[i]["answer"], "cost": cost_usd(out.tokens_in, out.tokens_out, settings.llm_model),
                   "secs": round(secs, 1), "walk": walk_log, "selfrag_picked": self_rag.LOG.get(g.question, {}).get("picked")}
        via = walk_log.get("via")
        print(f"\n{i}: base {base[i]['correct']:g} -> now {correct:g}  route {picks[g.question]}  "
              f"walk {via[1] + ' -> ' + via[2] if via else 'fallback f6a'}  ${rows[i]['cost']:.4f}  {secs:.1f} s")
        print(f"  answer: {out.answer[:200]}")
        print(f"  base:   {base[i]['answer'][:200]}")
    (RESULTS / "phase3_smoke.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nright: base {sum(r['base'] for r in rows.values()):g}/5 -> now {sum(r['now'] for r in rows.values()):g}/5; "
          f"${sum(r['cost'] for r in rows.values()):.4f} (original-call cost, cached calls included)")


if __name__ == "__main__":
    main()
