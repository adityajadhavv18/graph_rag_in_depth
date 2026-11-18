"""Phase 4, solution B on the problem it targets: CRAG's grader scores on chain-07 with the graph route, without and
with linked grading. Without it, every passage scored 0 (INCORRECT -> discarded -> "I don't know").

Run from graph_rag_in_depth/ (Qdrant + kg-neo4j running; OpenAI key sourced):
    uv run grag/linked_check.py [ids...]          (default: chain-07)
    uv run grag/linked_check.py --add twin-03    (B alone vs B + "add, don't replace")
"""

import sys

import grag_bridge
from eval_harness import load_golden
from eval_harness.metrics import is_refusal

import baseline
import crag
import graph_route
import route_eval
import selfrag_golden as sg


def main():
    add = "--add" in sys.argv
    ids = [a for a in sys.argv[1:] if not a.startswith("--")] or ["chain-07"]
    golden = {g.id: g for g in load_golden(sg.golden_name("tune"))}
    setups = [(True, False), (True, True)] if add else [(False, False), (True, False)]
    for linked, with_add in setups:
        graph_route.install(grag_bridge.SELFRAG_CONFIG, linked=linked, add=with_add)
        pipeline = baseline.routed({})
        for i in ids:
            g = golden[i]
            out = pipeline(g.question)
            right = 0.0 if is_refusal(out.answer) else route_eval.grade(g.question, g.ground_truth, out.answer)
            c, w = crag.LOG[g.question], graph_route.LOG.get(g.question, {})
            graph = set(w.get("chunks", []))
            marks = [f"{s}{'*' if cid in graph else ''}" for cid, s in zip(w.get("passages", []), c["scores"])]
            print(f"\n{i} linked={linked} add={with_add}: CRAG {c['action']}  scores {marks}  (* = graph passage)  right {right:g}")
            if linked:
                print(f"  graded against: {w.get('linked_question')}")
            print(f"  answer: {out.answer[:200]}")


if __name__ == "__main__":
    main()
