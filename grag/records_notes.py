"""Phase 5: a scope note on every record line, so look-alikes say what they are limited to (gpt-4o-mini, ~$0.06).

    "Shubman Gill (Gujarat Titans) made the highest individual score of 129 runs from 60 balls in IPL Playoffs."
    -> Scope: IPL (Twenty20) · playoffs only · overall record: no · 2023

gpt-4o-mini reads ONLY each line and its heading, 10 lines per call, and fills fixed fields (competition, stage,
place, holder group, overall, when). Code writes the Scope line from those fields, so the model never writes free
text and cannot add a fact from memory. The noted lines go to data/records/records_noted.jsonl and a separate Qdrant
collection, cricket_records_noted; the plain index stays as it is.

CUSTOM: hand-written record lines that the corpus does not state, added at Aditya's request (2026-11-19). The only
one: Karun Nair's 303* as the highest Test score by an Indian against England (the corpus gives the 303* and
"second Indian to score a triple century", never the record). It answers tune question trap-05, so trap-05 is
reported apart from the rest in every run that uses it.

Run from graph_rag_in_depth/ (Qdrant running; OpenAI key sourced):
    uv run grag/records_notes.py              (notes for every line, then the collection, with the custom lines)
    uv run grag/records_notes.py --custom     (only re-add the custom lines and rebuild; no LLM)
"""

import sys

import json
import random

import grag_bridge
import llm_cache
import records_index
import self_rag

NOTED = grag_bridge.ROOT / "data" / "records" / "records_noted.jsonl"
COLLECTION = "cricket_records_noted"
CUSTOM_FILE = grag_bridge.ROOT / "data" / "records" / "records_custom.jsonl"
CUSTOM = [{"chunk_id": "custom::rec::00001", "doc_id": "custom_records",
           "path": ["Custom records (hand-written)", "India v England"],
           "text": "Scope: Test · India v England · by an Indian batter · overall record: yes · December 2016\n"
                   "[International records · Test] Karun Nair's 303 not out against England at Chennai in December "
                   "2016 is the highest individual Test score by an Indian batter against England.",
           "n_tokens": 30, "fields": {}}]
EXCLUDED = ("trap-05",)  # answered by a custom line: reported apart
PER_CALL = 10
FIELDS = ("competition", "stage", "place", "holder_group", "overall", "when")
PROMPT = """You label cricket record statements. For each numbered statement, read ONLY its heading and its text.
Do not add anything you know from elsewhere; if the text does not say it, leave the field empty.

Fields:
- competition: the competition or format the record belongs to, e.g. "IPL (Twenty20)", "Test", "ODI", "T20I",
  "all Twenty20 cricket", "World Cup (ODI)", or ""
- stage: what the record covers, e.g. "all-time", "season", "playoffs only", "final", "series", "single match",
  "career", "tournament", or ""
- place: a ground or country the record is limited to, or ""
- holder_group: a group the record is limited to, e.g. "by an Indian captain", "on debut", "by a wicketkeeper",
  "away from home", or ""
- overall: "yes" if the text says it is the record for the whole competition or format, "yes, at the time" if it
  says it was then the record, "no" if it is limited (to a place, stage, team, group or match), "not a record" if
  the text states no record
- when: the year or date the text gives, or ""

Reply in JSON: {{"labels": [{{"n": 1, "competition": "...", "stage": "...", "place": "...", "holder_group": "...",
"overall": "...", "when": "..."}}, ...]}} with one entry per statement, in order.

Statements:
{statements}"""


def line_of(text: str) -> str:
    return text.split("] ", 1)[1] if text.startswith("[") else text


def scope(f: dict) -> str:
    parts = [f.get("competition"), f.get("stage"), f.get("place") and f"only at {f['place']}", f.get("holder_group"),
             f.get("overall") and f"overall record: {f['overall']}", f.get("when")]
    parts = [str(p).strip() for p in parts if p and str(p).strip()]
    return "Scope: " + " · ".join(parts) if parts else ""


def prompts_for(cs: list) -> list[str]:
    out = []
    for i in range(0, len(cs), PER_CALL):
        block = cs[i:i + PER_CALL]
        statements = "\n".join(f"[{n}] Heading: {' › '.join(c.path)}\n    Text: {line_of(c.text)}"
                               for n, c in enumerate(block, start=1))
        out.append(PROMPT.format(statements=statements))
    return out


def build_collection() -> None:
    CUSTOM_FILE.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in CUSTOM))
    records_index.index(records_index.load(NOTED) + records_index.load(CUSTOM_FILE), COLLECTION)
    print(f"Qdrant {COLLECTION} built ({len(CUSTOM)} custom line)")


def main():
    if "--custom" in sys.argv:
        return build_collection()
    cs = records_index.load()
    prompts = prompts_for(cs)
    self_rag.warm(prompts, json_mode=True)  # 8 threads; the cache is written from this thread
    noted, missing = [], 0
    for i, p in enumerate(prompts):
        block = cs[i * PER_CALL:(i + 1) * PER_CALL]
        try:
            labels = {int(x.get("n", 0)): x for x in json.loads(llm_cache.chat(p, json_mode=True).text)["labels"]}
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            labels = {}
        for n, c in enumerate(block, start=1):
            s = scope(labels.get(n, {}))
            missing += not s
            noted.append({"chunk_id": c.chunk_id, "doc_id": c.doc_id, "path": list(c.path),
                          "text": f"{s}\n{c.text}" if s else c.text, "n_tokens": c.n_tokens,
                          "fields": {k: labels.get(n, {}).get(k, "") for k in FIELDS}})
    NOTED.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in noted))
    print(f"noted {len(noted) - missing}/{len(noted)} lines in {len(prompts)} calls")
    build_collection()


if __name__ == "__main__":
    main()
