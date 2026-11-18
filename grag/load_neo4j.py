"""Phase 2: load data/graph/graph.json into Neo4j (KG_basics's kg-neo4j container, `docker start kg-neo4j`).

The container also holds KG_basics's practice graph, so every cricket node carries the label Cricket plus one of
CricketEntity (keyed by name), CricketMatch (keyed by id) and CricketChunk (keyed by id). A reload deletes only
:Cricket nodes. Each link keeps the line it came from and the chunk that line sits in.

Run from graph_rag_in_depth/:
    uv run grag/load_neo4j.py
"""

import json
from collections import defaultdict

from neo4j import GraphDatabase

import grag_bridge

GRAPH = grag_bridge.ROOT / "data" / "graph" / "graph.json"
ENV = grag_bridge.RAGS / "KG_basics" / ".env"
LABEL = {"entity": ("CricketEntity", "name"), "match": ("CricketMatch", "id"), "chunk": ("CricketChunk", "id")}
TYPES = {"HELD_AT", "PLAYER_OF_MATCH", "WON_BY", "TEAM", "UMPIRE", "TOP_BATTER", "TOP_BOWLER", "PART_OF",
         "DESCRIBED_IN", "HAS_CHUNK", "MENTIONS"}
BATCH = 2000


def driver():
    env = dict(line.split("=", 1) for line in ENV.read_text().splitlines() if "=" in line and not line.startswith("#"))
    return GraphDatabase.driver(env["NEO4J_URI"].strip(), auth=(env["NEO4J_USER"].strip(), env["NEO4J_PASSWORD"].strip()))


def batches(rows: list):
    for i in range(0, len(rows), BATCH):
        yield rows[i:i + BATCH]


def main():
    g = json.loads(GRAPH.read_text())
    with driver() as d, d.session() as s:
        while s.run("MATCH (n:Cricket) WITH n LIMIT 10000 DETACH DELETE n RETURN count(*) AS n").single()["n"]:
            pass
        for label, key in LABEL.values():
            s.run(f"CREATE CONSTRAINT {label.lower()}_{key} IF NOT EXISTS FOR (n:{label}) REQUIRE n.{key} IS UNIQUE")
        for rows in batches(g["nodes"]["entities"]):
            s.run("UNWIND $rows AS r MERGE (n:Cricket:CricketEntity {name: r.name}) "
                  "SET n.kinds = r.kinds, n.doc_id = r.doc_id", rows=rows)
        for rows in batches(g["nodes"]["matches"]):
            s.run("UNWIND $rows AS r MERGE (n:Cricket:CricketMatch {id: r.id}) "
                  "SET n.name = r.name, n.doc_id = r.doc_id, n.date = r.date, n.label = r.label", rows=rows)
        for rows in batches(g["nodes"]["chunks"]):
            s.run("UNWIND $rows AS r MERGE (n:Cricket:CricketChunk {id: r.id}) SET n.doc_id = r.doc_id", rows=rows)

        groups = defaultdict(list)
        for l in g["links"]:
            groups[(l["type"], l["src"][0], l["dst"][0])].append(
                {"src": l["src"][1], "dst": l["dst"][1], "line": l["line"], "chunk": l["chunk"]})
        for (type_, src, dst), links in groups.items():
            assert type_ in TYPES
            (sl, sk), (dl, dk) = LABEL[src], LABEL[dst]
            for rows in batches(links):
                s.run(f"UNWIND $rows AS r MATCH (a:{sl} {{{sk}: r.src}}) MATCH (b:{dl} {{{dk}: r.dst}}) "
                      f"MERGE (a)-[x:{type_}]->(b) SET x.line = r.line, x.chunk = r.chunk", rows=rows)

        nodes = s.run("MATCH (n:Cricket) RETURN count(n) AS n").single()["n"]
        rels = s.run("MATCH (:Cricket)-[r]->(:Cricket) RETURN count(r) AS n").single()["n"]
        other = s.run("MATCH (n) WHERE NOT n:Cricket RETURN count(n) AS n").single()["n"]
    print(f"Neo4j: {nodes} cricket nodes, {rels} cricket links (json: "
          f"{sum(map(len, g['nodes'].values()))} nodes, {len(g['links'])} links); KG_basics nodes kept: {other}")


if __name__ == "__main__":
    main()
