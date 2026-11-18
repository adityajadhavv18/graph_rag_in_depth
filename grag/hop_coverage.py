"""Phase 2: does the graph hold both hops of every chain question in Self-RAG's tune set? (free, no LLM)

For each chain question I name, by hand, the match its first hop starts from (the article and section it is
described in). The check then asks Neo4j:
    hop 1   that match node exists, and HELD_AT / PLAYER_OF_MATCH leads to the question's answer article
    hop 2   that article's HAS_CHUNK reaches a chunk holding the golden evidence for the second hop
This tests the graph, not the walk: Phase 3 still has to find the match from the question's words.

Results: grag/results/phase2_coverage.json.

Run from graph_rag_in_depth/ (kg-neo4j running):
    uv run grag/hop_coverage.py
"""

import json
import re

from eval_harness import load_golden

import grag_bridge
import ladder
import selfrag_golden as sg
from load_neo4j import driver

RESULTS = grag_bridge.ROOT / "grag" / "results" / "phase2_coverage.json"
# chain id -> (article the first hop is in, section of that match or None for a final, link to follow)
CHAINS = {
    "chain-01": ("2023_world_test_championship_final", None, "HELD_AT"),
    "chain-02": ("2011_cricket_world_cup_final", None, "HELD_AT"),
    "chain-03": ("2025_indian_premier_league_final", None, "HELD_AT"),
    "chain-04": ("indian_cricket_team_in_england_in_2025", "1st Test", "HELD_AT"),
    "chain-05": ("2025-26_ashes_series", "First Test", "HELD_AT"),
    "chain-06": ("indian_cricket_team_in_australia_in_2024-25", "2nd Test", "HELD_AT"),
    "chain-07": ("2024_indian_premier_league_final", None, "PLAYER_OF_MATCH"),
    "chain-08": ("indian_cricket_team_in_australia_in_2024-25", "1st Test", "HELD_AT"),
}


def fragments(evidence: str) -> list[str]:
    return [f.strip() for f in re.split(r"\|\||\.\.\.", evidence) if len(f.strip()) > 8]


def main():
    golden = {g.id: g for g in load_golden(sg.golden_name("tune"))}
    by_id = ladder._chunks_by_id()
    rows = {}
    with driver() as d, d.session() as s:
        for cid, (doc, section, rel) in CHAINS.items():
            g = golden[cid]
            target = s.run("MATCH (e:CricketEntity {doc_id: $doc}) RETURN e.name AS n", doc=g.source_doc).single()["n"]
            matches = s.run("MATCH (m:CricketMatch {doc_id: $doc}) RETURN m.id AS id, m.name AS name", doc=doc).data()
            if section:
                matches = [m for m in matches if re.search(rf"› {re.escape(section)}( ›| \()", m["name"])]
            else:
                matches = [m for m in matches if m["id"].endswith("::final")]
            hop1 = s.run(f"MATCH (m:CricketMatch)-[r:{rel}]->(e:CricketEntity) WHERE m.id IN $ids "
                         "RETURN m.name AS match, e.name AS to, r.line AS line",
                         ids=[m["id"] for m in matches]).data()
            reached = s.run("MATCH (:CricketEntity {name: $t})-[:HAS_CHUNK]->(c) RETURN c.id AS id", t=target).data()
            frags = fragments(g.evidence)
            gold = [c["id"] for c in reached if any(f in by_id[c["id"]].text for f in frags)]
            rows[cid] = {"question": g.question, "match_found": [m["name"] for m in matches],
                         "hop1": hop1, "target": target, "hop1_ok": any(h["to"] == target for h in hop1),
                         "hop2_chunks": gold, "hop2_ok": bool(gold)}
    RESULTS.write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    for cid, r in rows.items():
        h = next((x for x in r["hop1"] if x["to"] == r["target"]), r["hop1"][0] if r["hop1"] else None)
        print(f"{cid}  hop1 {'✓' if r['hop1_ok'] else '✗'}  hop2 {'✓' if r['hop2_ok'] else '✗'}  "
              f"{(h or {}).get('match', 'no match')} -> {(h or {}).get('to')}  [{len(r['hop2_chunks'])} gold chunk(s)]")
    both = sum(r["hop1_ok"] and r["hop2_ok"] for r in rows.values())
    print(f"\nboth hops in the graph: {both}/{len(rows)}")


if __name__ == "__main__":
    main()
