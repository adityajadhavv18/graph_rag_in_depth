"""Phase 4: the full system with the graph walk in place of IRCoT on the multi route, on Self-RAG's tune set (78),
run like Self-RAG's selfrag_tune.py and graded with the 1-vote and 3-vote graders, then compared question by question
with the base's replay (phase1.json: Self-RAG #6 with IRCoT, 0.897).

The none and single routes are unchanged and replay from the cache; only the 39 multi questions make new calls.
Decision rule (Aditya's): the graph replaces IRCoT if the multi questions score higher than IRCoT's and the total
stays >= 0.897.

--linked turns on solution B (graph passages graded against the question with their link filled in), --add
"add, don't replace" (f6a's own top 5 + the graph chunks) and --fallback the IRCoT loop when CRAG rejects every
graph passage, see graph_route.py. --add and --fallback runs are compared with the --linked run, --linked with
the plain graph run.

Results: grag/results/phase4_tune.json, phase4_tune_linked.json, phase4_tune_linked_add.json,
phase4_tune_linked_fallback.json.

Run from graph_rag_in_depth/ (Qdrant + kg-neo4j running; OpenAI key sourced):
    uv run grag/graph_tune.py [--linked] [--add] [--fallback]
    uv run grag/graph_tune.py --linked --fallback --records   (Phase 5: + record seats with scope notes and the
                                                              custom line; trap-05 reported apart)
"""

import sys

import json
import statistics

import grag_bridge
from eval_harness import load_golden

import answer_tuning
import baseline
import crag
import crag_eval
import graph_route
import route_eval
import self_rag
import selfrag_golden as sg
from base_tune import KINDS

RESULTS = grag_bridge.ROOT / "grag" / "results"


def run(golden: dict, split: str = "tune", linked: bool = False, add: bool = False, fallback: bool = False,
        records: bool = False) -> dict:
    graph_route.install(grag_bridge.SELFRAG_CONFIG, linked=linked, add=add, fallback=fallback)
    if records:
        import records_route
        records_route.install(True, noted=True)
    picks = {}
    rows = route_eval.run_route("graph", golden, baseline.routed(picks), golden_set=sg.golden_name(split))
    for i, row in rows.items():
        q = golden[i].question
        log = self_rag.LOG.get(q, {})
        row["route"], row["kind"] = picks[q], sg.kind(i)
        row["correct3"] = answer_tuning.grade3(golden[i], row["answer"])
        row["crag"] = crag.LOG.get(q)
        row["selfrag"] = {k: log.get(k) for k in ("sure", "picked", "good", "retries", "candidates")}
        row["walk"] = graph_route.LOG.get(q) if picks[q] == "multi" else None
        if records:
            row["records"] = records_route.LOG.get(q)
    return rows


def score(rows: dict, ids: list[str], key: str = "correct") -> float:
    return sum(rows[i][key] for i in ids)


def main():
    golden = {g.id: g for g in load_golden(sg.golden_name("tune"))}
    base = json.loads((RESULTS / "phase1.json").read_text())
    b, bq = base["summary"], base["questions"]
    linked, add, fallback = "--linked" in sys.argv, "--add" in sys.argv, "--fallback" in sys.argv
    records = "--records" in sys.argv
    rows = run(golden, linked=linked, add=add, fallback=fallback, records=records)
    s = crag_eval.summary(rows, None)
    by_kind = lambda rs: {k: f"{sum(r['correct'] for r in rs.values() if r['kind'] == k):g}/"
                             f"{sum(r['kind'] == k for r in rs.values())}" for k in KINDS}
    multi = [i for i in golden if bq[i]["route"] == "multi"]
    multi_now = {i: rows[i] for i in multi}
    s["by_kind"] = by_kind(rows)
    s["multi"] = {"ircot": score(bq, multi), "graph": score(rows, multi), "n": len(multi),
                  "ircot_by_kind": by_kind({i: bq[i] for i in multi}), "graph_by_kind": by_kind(multi_now)}
    s["fallback"] = sum(bool((r["walk"] or {}).get("fallback")) for r in multi_now.values())
    s["ircot_fallback"] = [i for i, r in multi_now.items() if (r["walk"] or {}).get("ircot_fallback")]
    s["route_changed"] = [i for i in golden if rows[i]["route"] != bq[i]["route"]]
    s["changed"] = {i: f"{bq[i]['correct']:g} -> {r['correct']:g}" for i, r in rows.items()
                    if r["correct"] != bq[i]["correct"]}
    s["multi_p50_ms"] = {"ircot": statistics.median(bq[i]["ms"] for i in multi),
                         "graph": statistics.median(rows[i]["ms"] for i in multi)}
    s["multi_cost"] = {"ircot": sum(bq[i]["cost"] for i in multi) / len(multi),
                       "graph": sum(rows[i]["cost"] for i in multi) / len(multi)}
    prev = ("phase4_tune_linked_fallback.json" if records else
            "phase4_tune_linked.json" if (add or fallback) else "phase4_tune.json")
    if records:
        import records_notes
        keep = [i for i in golden if i not in records_notes.EXCLUDED]
        p = json.loads((RESULTS / prev).read_text())["questions"]
        s["excluding_custom"] = {"ids": list(records_notes.EXCLUDED), "n": len(keep),
                                 "before": sum(p[i]["correct"] for i in keep) / len(keep),
                                 "now": sum(rows[i]["correct"] for i in keep) / len(keep)}
    if linked or add or fallback:
        g = json.loads((RESULTS / prev).read_text())["questions"]
        s["vs_graph"] = {i: f"{g[i]['correct']:g} -> {r['correct']:g}" for i, r in rows.items()
                         if r["correct"] != g[i]["correct"]}
        s["crag_actions"] = {name: {a: sum((src[i]["crag"] or {}).get("action") == a for i in multi)
                                    for a in ("CORRECT", "AMBIGUOUS", "INCORRECT")}
                             for name, src in (("ircot", bq), ("previous", g), ("now", rows))}
    out = ("phase5_tune_records.json" if records else
           "phase4_tune" + "_linked" * linked + "_add" * add + "_fallback" * fallback + ".json")
    (RESULTS / out).write_text(json.dumps({"summary": s, "questions": rows}, indent=2, ensure_ascii=False))
    m = s["multi"]
    print(f"\nright (78): IRCoT base {b['correct']:.3f} -> graph {s['correct']:.3f} (1 vote); "
          f"{b['correct3']:.3f} -> {s['correct3']:.3f} (3 votes)")
    print(f"multi (39): IRCoT {m['ircot']:g} -> graph {m['graph']:g}")
    print(f"  by kind IRCoT {m['ircot_by_kind']}\n          graph {m['graph_by_kind']}")
    print(f"fallback to plain f6a: {s['fallback']}; route changed: {s['route_changed'] or 'none'}")
    print(f"multi $/question {s['multi_cost']['ircot']:.5f} -> {s['multi_cost']['graph']:.5f}; "
          f"multi p50 {s['multi_p50_ms']['ircot'] / 1000:.1f} -> {s['multi_p50_ms']['graph'] / 1000:.1f} s; "
          f"all p50 {b['p50_ms'] / 1000:.1f} -> {s['p50_ms'] / 1000:.1f} s")
    print(f"changed vs IRCoT base: {s['changed'] or 'none'}")
    print(f"IRCoT fallback used on: {s['ircot_fallback'] or 'none'}")
    if records:
        e = s["excluding_custom"]
        print(f"without {e['ids']} ({e['n']} questions): {e['before']:.3f} -> {e['now']:.3f}")
    if linked or add or fallback:
        print(f"changed vs previous step ({prev}): {s['vs_graph'] or 'none'}")
        print(f"CRAG actions on multi: {s['crag_actions']}")


if __name__ == "__main__":
    main()
