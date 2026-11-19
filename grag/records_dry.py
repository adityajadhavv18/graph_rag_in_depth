"""Phase 5, free check: the records search alone on the 69 tune questions that search (no LLM, $0).

For the three record questions, is the line that answers it among the records' top 3? For every question, which
record lines would get rerank seats: these are what could mislead the questions that are right now.

Results: grag/results/phase5_records_dry.json.

Run from graph_rag_in_depth/ (Qdrant running):
    OPENAI_API_KEY=sk-dummy uv run grag/records_dry.py
"""

import json

from eval_harness import load_golden

import grag_bridge
import crag
import records_route
import selfrag_golden as sg

RESULTS = grag_bridge.ROOT / "grag" / "results"
TARGETS = {"alike-04": "Alzarri Joseph", "alike-05": "175 not out", "trap-05": "303"}


def main():
    import sys
    records_route.install(True, noted="--noted" in sys.argv)
    golden = {g.id: g for g in load_golden(sg.golden_name("tune"))}
    final = json.loads((RESULTS / "phase4_tune_linked_fallback.json").read_text())["questions"]
    rows = {}
    for i, r in final.items():
        if r["route"] == "none":
            continue
        hits = records_route.records_search(golden[i].question)
        rows[i] = {"route": r["route"], "kind": r["kind"], "correct_now": r["correct"],
                   "records": [crag._by_id[h.chunk_id].text[:160] for h in hits]}
        if i in TARGETS:
            rows[i]["target_found"] = any(TARGETS[i] in crag._by_id[h.chunk_id].text for h in hits)
    (RESULTS / ("phase5_records_dry_noted.json" if "--noted" in sys.argv else "phase5_records_dry.json")).write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    for i in TARGETS:
        print(f"{i}: answer line in records top 3: {'✓' if rows[i]['target_found'] else '✗'}")
        for t in rows[i]["records"]:
            print(f"    {t}")
    print(f"\nquestions searched: {len(rows)}; each gets {records_route.N_RECORDS} record seats")



if __name__ == "__main__":
    main()
