"""Phase 5: the final GraphRAG system (graph_route.FINAL) plus the record seats, on 5 tune questions, against the
final system's saved answers (phase4_tune_linked_fallback.json, 0.942):

    alike-04 (single), alike-05 (single), trap-05 (multi)   record questions, wrong now
    twin-04 (multi), alike-02 (single)                      right now, with look-alike record lines in their seats

Results: grag/results/phase5_records_smoke.json.

Run from graph_rag_in_depth/ (Qdrant + kg-neo4j running; OpenAI key sourced):
    uv run grag/records_smoke.py
"""

import json
import sys

import grag_bridge
from eval_harness import load_golden, settings
from eval_harness.metrics import cost_usd, is_refusal

import baseline
import crag
import graph_route
import records_route
import route_eval
import selfrag_golden as sg

IDS = ("alike-04", "alike-05", "trap-05", "twin-04", "alike-02")
RESULTS = grag_bridge.ROOT / "grag" / "results"


def main():
    golden = {g.id: g for g in load_golden(sg.golden_name("tune"))}
    before = json.loads((RESULTS / "phase4_tune_linked_fallback.json").read_text())["questions"]
    graph_route.install(grag_bridge.SELFRAG_CONFIG, **graph_route.FINAL)
    noted = "--noted" in sys.argv
    records_route.install(True, noted=noted)
    picks, rows = {}, {}
    pipeline = baseline.routed(picks)
    for i in IDS:
        g = golden[i]
        out = pipeline(g.question)
        right = 0.0 if is_refusal(out.answer) else route_eval.grade(g.question, g.ground_truth, out.answer)
        recs = records_route.LOG.get(g.question, [])
        rows[i] = {"before": before[i]["correct"], "now": right, "route": picks[g.question], "answer": out.answer,
                   "records": [crag._by_id[c].text[:140] for c in recs], "crag": crag.LOG.get(g.question),
                   "cost": cost_usd(out.tokens_in, out.tokens_out, settings.llm_model)}
        print(f"\n{i} [{picks[g.question]}]: before {before[i]['correct']:g} -> now {right:g}   "
              f"CRAG {(crag.LOG.get(g.question) or {}).get('action')}")
        print(f"  answer: {out.answer[:200]}")
    (RESULTS / ("phase5_records_smoke_noted.json" if noted else "phase5_records_smoke.json")).write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nright: before {sum(r['before'] for r in rows.values()):g}/5 -> now {sum(r['now'] for r in rows.values()):g}/5")


if __name__ == "__main__":
    main()
