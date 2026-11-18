"""Phase 2: the cricket graph, built by code alone from the corpus's regular lines ($0, no LLM).

Every article prints "Key: value" lines that code can read exactly:
  - a tour or series prints one box per match:   Match: 20–24 June 2025 | England v India
                                                  England: 465 (100.4 overs); Ollie Pope 106 (137); Jasprit Bumrah 5/83
                                                  Result: England won by 5 wickets
                                                  Venue: Headingley, Leeds
                                                  Player of the match: Ben Duckett (Eng)
                                                  Umpires: Chris Gaffaney (NZ) and Sharfuddoula (Ban)
  - a final prints the same facts once, in its lead: Event / Date / Venue / Player of the match / Umpires
  - a ground or player article starts with its own facts (Capacity, Opened, Born, ...), which stay in its chunks

So the graph is:
  (Match) -[HELD_AT]-> (ground)            (Match) -[PLAYER_OF_MATCH]-> (player)
  (Match) -[WON_BY]-> (team)               (Match) -[TEAM]-> (team)          (Match) -[UMPIRE]-> (umpire)
  (Match) -[TOP_BATTER / TOP_BOWLER]-> (player)                              (Match) -[PART_OF]-> (article)
  (Match) -[DESCRIBED_IN]-> (Chunk)        (article) -[HAS_CHUNK]-> (Chunk)  (Chunk) -[MENTIONS]-> (article)

Names are merged in code onto the 149 article titles (included + the 9 fallback biographies): "Headingley, Leeds",
"Mitchell Starc (KKR)" and "M.A. Chidambaram Stadium, Chennai" land on the articles "Headingley Cricket Ground",
"Mitchell Starc" and "M. A. Chidambaram Stadium", so a match's ground or player leads straight into that article's
chunks. Every link keeps the line it came from and the chunk that line sits in.

Output: data/graph/graph.json (nodes + links), loaded into Neo4j by load_neo4j.py.

Run from graph_rag_in_depth/:
    uv run grag/extract_code.py
"""

import json
import re
from collections import Counter, defaultdict

import grag_bridge
import chunks
import fallback_index

OUT = grag_bridge.ROOT / "data" / "graph" / "graph.json"
KEY = re.compile(r"^((?:\(H\) )?[A-Z][^:]{0,40}): (.+)$")  # IPL marks the home side "(H) Mumbai Indians"
FACT_KEYS = {"Match", "Result", "Venue", "Player of the match", "Players of the match", "Umpires", "Event", "Date",
             "Attendance", "Toss"}
GROUND_WORD = re.compile(r"Stadium|Ground|Oval|Gardens|Park|Kotla")
GROUND_SUFFIX = re.compile(r"\s+(Cricket Ground|Stadium|Ground|Oval)$")
# Venue names that differ from the article title (old names, sponsors, short forms), checked in the corpus
VENUE_ALIASES = {"kenningtonoval": "The Oval", "theovalkennington": "The Oval", "sardarpatelstadium": "Narendra Modi Stadium",
                 "moterastadium": "Narendra Modi Stadium", "optusstadium": "Perth Stadium", "chepauk": "M. A. Chidambaram Stadium",
                 "machidambaramstadiumchepauk": "M. A. Chidambaram Stadium", "lordscricketground": "Lord's"}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def strip_brackets(s: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", s).strip()


def split_people(s: str) -> list[str]:
    return [strip_brackets(p) for p in re.split(r";| and |, ", s) if strip_brackets(p)]


class Graph:
    def __init__(self, titles: dict[str, str]):
        self.titles = titles  # doc_id -> article title
        self.by_norm = {norm(t): t for t in titles.values()}
        for t in titles.values():  # "Headingley Cricket Ground" is also "Headingley" (for venues only)
            short = GROUND_SUFFIX.sub("", t)
            if short != t and len(short) > 3:
                self.by_norm.setdefault(norm(short), t)
        self.by_norm.update(VENUE_ALIASES)
        self.entities: dict[str, dict] = {}
        self.matches: dict[str, dict] = {}
        self.links: list[dict] = []
        self.unmapped_venues = Counter()

    def entity(self, name: str, kind: str, venue: bool = False) -> str:
        if venue:
            head = name.split(",")[0].strip()
            name = self.by_norm.get(norm(name)) or self.by_norm.get(norm(head)) or head
            if name == head and norm(head) not in self.by_norm:
                self.unmapped_venues[head] += 1
        else:
            name = self.by_norm.get(norm(name), name)
        e = self.entities.setdefault(name, {"name": name, "kinds": set(), "doc_id": None})
        e["kinds"].add(kind)
        return name

    def link(self, type_: str, src: tuple, dst: tuple, line: str = "", chunk: str | None = None):
        self.links.append({"type": type_, "src": src, "dst": dst, "line": line, "chunk": chunk})


def boxes_in(lines: list[str], lead: bool) -> list[list[str]]:
    """Each match's fact lines. A box starts at "Match:" and runs while lines look like "Key: value". A final's
    lead holds its facts as an infobox, mixed with stray lines, so there every "Key: value" line in the first 40
    counts, if they include a Venue and an Event or a Player of the match."""
    out, cur = [], None
    for line in lines:
        if line.startswith("Match: "):
            cur = [line]
            out.append(cur)
        elif cur is not None and KEY.match(line) and len(cur) < 16:
            cur.append(line)
        else:
            cur = None
    if lead and not out:
        keys = [l for l in lines[:40] if KEY.match(l)]
        names = {KEY.match(l).group(1) for l in keys}
        if "Venue" in names and names & {"Event", "Player of the match", "Players of the match"}:
            out.append(keys)
    return out


def add_match(g: Graph, mid: str, name: str, doc: str, lines: list[str], chunk_of) -> dict:
    m = g.matches.setdefault(mid, {"id": mid, "name": name, "doc_id": doc, "date": None, "label": None,
                                   "box_chunks": []})
    m["box_chunks"] = sorted(set(m["box_chunks"]) | {c for c in map(chunk_of, lines) if c})
    g.link("PART_OF", ("match", mid), ("entity", g.entity(g.titles[doc], "article")))
    teams, venue = None, None
    for line in lines:
        key, value = KEY.match(line).groups()
        c = chunk_of(line)
        if key == "Match":
            date, _, teams = value.partition(" | ")
            m["date"] = m["date"] or date
            for t in re.split(r" v | vs ", teams):
                g.link("TEAM", ("match", mid), ("entity", g.entity(strip_brackets(t), "team")), line, c)
        elif key == "Date":
            m["date"] = m["date"] or value
        elif key == "Venue" or (key == "Result" and " won" not in value and GROUND_WORD.search(value)):
            venue = value  # the 2012 IPL article prints the ground in its "Result:" line
            g.link("HELD_AT", ("match", mid), ("entity", g.entity(value, "ground", venue=True)), line, c)
        elif key in ("Player of the match", "Players of the match"):
            for p in split_people(value):
                g.link("PLAYER_OF_MATCH", ("match", mid), ("entity", g.entity(p, "player")), line, c)
        elif key == "Umpires":
            for p in split_people(value):
                g.link("UMPIRE", ("match", mid), ("entity", g.entity(p, "umpire")), line, c)
        elif key == "Result":
            won = re.match(r"(.+?) won\b", value)
            if won:
                g.link("WON_BY", ("match", mid), ("entity", g.entity(won.group(1).strip(), "team")), line, c)
        elif key == "Event":
            g.link("PART_OF", ("match", mid), ("entity", g.entity(value, "event")), line, c)
        elif key not in FACT_KEYS:
            # scorecard, one line:  "India A: 557 (125.1 overs); Karun Nair 204 (281); Josh Hull 3/72 (20 overs)"
            # or IPL's two lines:   "Deccan Chargers: 110 all out (18.4 overs)"
            #                       "Deccan Chargers: Andrew Symonds 32 (39); Murali Kartik 3/17 (3.4 overs)"
            parts = [p.strip() for p in value.split(";")]
            if re.match(r"\d", value):
                g.link("TEAM", ("match", mid), ("entity", g.entity(strip_brackets(key), "team")), line, c)
                parts = parts[1:]
            if parts and (b := re.match(r"(.+?) \d+\*? \(", parts[0])):
                g.link("TOP_BATTER", ("match", mid), ("entity", g.entity(b.group(1), "player")), line, c)
            if len(parts) > 1 and (w := re.match(r"(.+?) \d+/\d+", parts[1])):
                g.link("TOP_BOWLER", ("match", mid), ("entity", g.entity(w.group(1), "player")), line, c)
    if lines[0].startswith("Match: ") and venue:  # the label Contextual #2 gave this match's chunks
        date, _, teams = lines[0][len("Match: "):].partition(" | ")
        m["label"] = f"{chunks._with_year(date, g.titles[doc])} | {teams} | {venue}"
    return m


def build() -> Graph:
    docs = chunks.included_docs() + fallback_index.left_out()
    all_chunks = chunks.load_chunks() + fallback_index.fallback_chunks(fallback_index.left_out())
    by_section = defaultdict(list)
    for c in all_chunks:
        by_section[(c.doc_id, int(c.chunk_id.split("::")[1][1:]))].append(c)
    titles = {d: (chunks.TEXT_DIR / f"{d}.txt").read_text().splitlines()[0].lstrip("# ").strip() for d in docs}
    g = Graph(titles)

    for d in docs:
        g.entity(titles[d], "article")
        g.entities[titles[d]]["doc_id"] = d
        secs = chunks.sections((chunks.TEXT_DIR / f"{d}.txt").read_text())
        found = []  # (match id, its fact lines, section number)
        for n, (path, body) in enumerate(secs):
            for k, lines in enumerate(boxes_in(body.splitlines(), lead=(n == 0))):
                found.append((f"{d}::s{n:03d}::m{k}", " › ".join(path), lines, n))
        # a final with a lead infobox and one "Match:" box describes one match: merge them into the final
        # (only boxes in the lead or under "Match": a "Route to the final" box is another match)
        is_final = "final" in titles[d].lower() and found and not found[0][2][0].startswith("Match: ")
        for mid, name, lines, n in found:
            path = secs[n][0]
            same = is_final and (n == 0 or (len(path) > 1 and path[1].lower().startswith(("match", "final"))))
            if same:
                mid, name = f"{d}::final", titles[d]
            secs_chunks = by_section[(d, n)]
            chunk_of = lambda line, cs=secs_chunks: next((c.chunk_id for c in cs if line in c.text), None)
            m = add_match(g, mid, name if same else f"{name} ({lines[0].split(' | ')[0][7:]})", d, lines, chunk_of)
            m["sections"] = sorted(set(m.get("sections", [])) | {n})

        for c in [c for (doc, _), cs in by_section.items() if doc == d for c in cs]:
            g.link("HAS_CHUNK", ("entity", titles[d]), ("chunk", c.chunk_id))

    # a match listed inside a ground's own article prints no venue: it was played there
    grounds = {l["dst"][1] for l in g.links if l["type"] == "HELD_AT" and g.entities[l["dst"][1]]["doc_id"]}
    has_venue = {l["src"][1] for l in g.links if l["type"] == "HELD_AT"}
    for mid, m in g.matches.items():
        if mid not in has_venue and titles[m["doc_id"]] in grounds:
            g.link("HELD_AT", ("match", mid), ("entity", titles[m["doc_id"]]), "(listed in the ground's article)",
                   (m["box_chunks"] or [None])[0])

    # which chunks tell each match's story: the chunks its box sits in, Contextual #2's match label, and for a
    # final the whole article
    for mid, m in g.matches.items():
        for cid in m["box_chunks"]:
            g.link("DESCRIBED_IN", ("match", mid), ("chunk", cid))
    for c in all_chunks:
        for mid, m in g.matches.items():
            if (m["label"] and m["label"] in c.match) or (mid.endswith("::final") and m["doc_id"] == c.doc_id):
                g.link("DESCRIBED_IN", ("match", mid), ("chunk", c.chunk_id))

    # a chunk naming another article's title (whole words, case kept) links to it
    pats = {t: re.compile(rf"(?<![\w']){re.escape(t)}(?![\w'])") for t in titles.values() if len(t) > 4}
    for c in all_chunks:
        for t, p in pats.items():
            if titles[c.doc_id] != t and p.search(c.text):
                g.link("MENTIONS", ("chunk", c.chunk_id), ("entity", t))
    g.chunk_docs = {c.chunk_id: c.doc_id for c in all_chunks}
    return g


def main():
    g = build()
    nodes = {"entities": [{**e, "kinds": sorted(e["kinds"])} for e in g.entities.values()],
             "matches": list(g.matches.values()),
             "chunks": [{"id": i, "doc_id": d} for i, d in g.chunk_docs.items()]}
    seen, links = set(), []
    for l in g.links:  # one link per (type, src, dst): the first line that made it
        key = (l["type"], tuple(l["src"]), tuple(l["dst"]))
        if key not in seen:
            seen.add(key)
            links.append(l)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"nodes": nodes, "links": links}, ensure_ascii=False))

    held = [l for l in links if l["type"] == "HELD_AT"]
    on_article = sum(g.entities[l["dst"][1]]["doc_id"] is not None for l in held)
    print(f"nodes: {len(nodes['entities'])} entities, {len(nodes['matches'])} matches, {len(nodes['chunks'])} chunks")
    print(f"links: {len(links)}  " + ", ".join(f"{t} {n}" for t, n in Counter(l['type'] for l in links).most_common()))
    print(f"HELD_AT reaching a ground article: {on_article}/{len(held)}")
    print(f"venues with no article (top 15): {g.unmapped_venues.most_common(15)}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
