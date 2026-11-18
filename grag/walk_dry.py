"""Phase 3, free check: the walk alone on the 39 tune questions the router sends to multi (no LLM, $0).

For each question: the name seed and its cosine, the search seeds, the hop the walk took and its chunks. For the 8
chain questions, is the second hop's gold chunk (phase2_coverage.json) among the walk's chunks, and was it in
IRCoT's pool (phase1.json)? f6a's rewrites come from the cache, so run it with a dummy key.

Results: grag/results/phase3_dry.json.

Run from graph_rag_in_depth/ (Qdrant + kg-neo4j running):
    OPENAI_API_KEY=sk-dummy uv run grag/walk_dry.py
"""

import json

from eval_harness import load_golden

import grag_bridge
import crag
import rag_fusion
import selfrag_golden as sg
import walk

RESULTS = grag_bridge.ROOT / "grag" / "results"


def main():
    golden = {g.id: g for g in load_golden(sg.golden_name("tune"))}
    base = json.loads((RESULTS / "phase1.json").read_text())["questions"]
    gold = {i: r["hop2_chunks"] for i, r in json.loads((RESULTS / "phase2_coverage.json").read_text()).items()}
    multi = [i for i, r in base.items() if r["route"] == "multi"]
    opts = crag._f5a(rerank=False)
    opts.pop("rerank")
    list_opts, fuse_opts = rag_fusion.split(opts)

    rows = {}
    for i in multi:
        q = golden[i].question
        fused = rag_fusion.fuse(rag_fusion.fusion_lists(q, **list_opts), **fuse_opts)
        chunks, log = walk.walk(q, fused)
        ircot = {c for added in base[i]["added"] for c in added}
        row = {"kind": base[i]["kind"], "ircot_correct": base[i]["correct"], **log}
        if i in gold:
            row["gold_in_walk"] = bool(set(gold[i]) & set(log["chunks"]))
            row["gold_in_ircot"] = bool(set(gold[i]) & ircot)
        rows[i] = row
    (RESULTS / "phase3_dry.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    print("| id | IRCoT | name seed (cos) | walk via | chunks | gold: walk / IRCoT |\n|---|---|---|---|---|---|")
    for i, r in rows.items():
        seed = f"{r['by_name'][0][1][:55]} ({r['by_name'][0][2]})" if r["by_name"] else "-"
        via = f"{r['via'][1]} -> {r['via'][2]}" if r.get("via") else "fallback f6a"
        g = f"{'✓' if r['gold_in_walk'] else '✗'} / {'✓' if r['gold_in_ircot'] else '✗'}" if "gold_in_walk" in r else ""
        print(f"| {i} | {r['ircot_correct']:g} | {seed} | {via} | {len(r['chunks'])} | {g} |")
    chains = [r for r in rows.values() if "gold_in_walk" in r]
    print(f"\nwalk found a path: {sum(bool(r['chunks']) for r in rows.values())}/{len(rows)}")
    print(f"chain hop-2 gold: walk {sum(r['gold_in_walk'] for r in chains)}/{len(chains)}, "
          f"IRCoT pool {sum(r['gold_in_ircot'] for r in chains)}/{len(chains)}")


if __name__ == "__main__":
    main()
