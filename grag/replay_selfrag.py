"""Phase 1: the base GraphRAG plugs into. Self-RAG #6's full system on its tune set (78), run live through the bridge:

    few-shot router -> none (+ sure check) / single (f6a) / multi (IRCoT) -> CRAG 3f -> Self-RAG layer -> answer

and graded with the 1-vote and 3-vote graders. Every LLM call should come from the cache; run it with a dummy OpenAI
key and any uncached call fails instead of spending. It must match Self-RAG's saved run (phase4_tune.json, 0.897)
question by question: score, route, CRAG action, Self-RAG pick and answer text, or the reuse is broken.

Results go to grag/results/phase1.json.

Run from graph_rag_in_depth/ (Qdrant running):
    OPENAI_API_KEY=sk-dummy uv run grag/replay_selfrag.py
"""

import json

import grag_bridge
from eval_harness import load_golden

import crag_eval
import selfrag_golden as sg
import selfrag_tune

RESULTS = grag_bridge.ROOT / "grag" / "results" / "phase1.json"
SAVED = grag_bridge.SELFRAG_RESULTS / "phase4_tune.json"
FIELDS = ("correct", "correct3", "route", "answer")


def main():
    golden = {g.id: g for g in load_golden(sg.golden_name("tune"))}
    rows = selfrag_tune.run(golden)
    saved = json.loads(SAVED.read_text())
    now = crag_eval.summary(rows, None)
    then = saved["summary"]

    action = lambda row: row["crag"]["action"] if row["crag"] else None
    picked = lambda row: (row["selfrag"] or {}).get("picked")
    differs = {}
    for i in golden:
        old = saved["questions"][i]
        d = [f for f in FIELDS if rows[i][f] != old[f]]
        d += ["action"] * (action(rows[i]) != action(old)) + ["picked"] * (picked(rows[i]) != picked(old))
        if d:
            differs[i] = d
    RESULTS.write_text(json.dumps({"summary": now, "saved_summary": then, "differs": differs, "questions": rows},
                                  indent=2, ensure_ascii=False))

    print("\n| | now | Self-RAG saved |\n|---|---|---|")
    for key in ("correct", "correct3"):
        print(f"| {key} | {now[key]:.3f} | {then[key]:.3f} |")
    print(f"| routes | {now['routes']} | {then['routes']} |")
    print(f"| $/question (original calls) | {now['cost_per_question']:.5f} | {then['cost_per_question']:.5f} |")
    print(f"| p50 s | {now['p50_ms'] / 1000:.1f} | {then['p50_ms'] / 1000:.1f} |")
    print(f"\nquestions that differ: {differs or 'none'}")


if __name__ == "__main__":
    main()
