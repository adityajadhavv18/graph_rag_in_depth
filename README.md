# GraphRAG #7: cricket on Wikipedia

Ask "how long did it take to build the stadium that hosted the 2011 Cricket World Cup final, and when did it open?" and Self-RAG #6 answers "five years, opened in 2020". The facts are real, but they belong to the Narendra Modi Stadium: the 2011 final was at the Wankhede. Its multi-step search (IRCoT) reasoned its way to the wrong ground. The corpus says plainly where the final was played, on a line that reads `Venue: Wankhede Stadium, Mumbai`. GraphRAG turns lines like that into links, so the second step follows a link instead of a guess.

I replaced Self-RAG #6's IRCoT loop on the multi route with a walk over a cricket graph, kept everything else, and measured it on the same questions.

**Result on tune (78 questions): Self-RAG #6 0.897 → GraphRAG 0.942** (0.936 with the 3-vote grader). On the 39 multi-step questions it went from 34 to 37.5, and the two-step chain questions from 4 of 8 to 7.5. A multi question costs $0.0043 instead of $0.0072, and its median time is 12.2 s instead of 20.4 s. I built the graph with code alone, for $0.

```
                         ┌──────────────────────┐
                         │   User's question    │
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ ROUTER (Adaptive #4) │   unchanged
                         └──┬────────┬────────┬─┘
              ┌─────────────┘        │        └──────────────────────┐
              ▼                      ▼                               ▼
   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────────────────────────┐
   │ NONE             │   │ SINGLE           │   │ MULTI: GRAPH WALK (new, replaces IRCoT)│
   │ memory + "are    │   │ f6a search,      │   │ 1 seed: which match? (local bge)     │
   │ you sure?"       │   │ top 5            │   │ 2 hop 1: match -[HELD_AT / PLAYER_OF_ │
   │ (unchanged)      │   │ (unchanged)      │   │   MATCH / …]-> ground or player       │
   └────────┬─────────┘   └────────┬─────────┘   │   (Neo4j)                            │
            │                      │             │ 3 hop 2: that article's best chunks   │
            │                      │             │   (local bge)                        │
            │                      │             │ + f6a's fused list → LLM rerank        │
            │                      │             │ passages: rerank top 5 + graph chunks │
            │                      │             └──────────────────┬───────────────────┘
            │                      └──────────────┬─────────────────┘
            │                                     ▼
            │                ┌─────────────────────────────────────────────┐
            │                │ CRAG (Corrective #5)                        │
            │                │ + LINKED GRADING (new): a graph passage is  │
            │                │   graded with the link that brought it      │
            │                └──────────────────────┬──────────────────────┘
            │                                       │ all passages rejected on the graph route?
            │                                       │ → IRCOT FALLBACK (new): the old loop runs instead
            │                                       ▼
            │                ┌─────────────────────────────────────────────┐
            │                │ SELF-RAG (#6), unchanged                    │
            │                └──────────────────────┬──────────────────────┘
            ▼                                       ▼
                         ┌──────────────────────┐
                         │        ANSWER        │
                         └──────────────────────┘
```

Think of the "see also" signposts in a library. Search is a librarian who brings you the pages that *sound like* your question. The graph is the signposts: 2011 final → see Wankhede Stadium → see "Construction". The page about building the Wankhede never mentions the 2011 final, so it never sounds like the question. Following the signposts gets you there anyway.

- **GraphRAG** (Edge et al., Microsoft, 2024, "From Local to Global: A Graph RAG Approach to Query-Focused Summarization"): an LLM extracts entities and relationships from every chunk into a graph. *Local search* maps a question to entities and hands their neighbourhood's source text to the LLM. My retrieval is local search. The paper's other half, community summaries for global questions, is not part of this build.
- **IRCoT** (Trivedi et al., 2023, "Interleaving Retrieval with Chain-of-Thought Reasoning"): the loop GraphRAG replaces. It searches, writes one reasoning sentence, and searches again with that sentence.
- **Self-RAG #6** ([self_rag_in_depth](https://github.com/adityajadhavv18/self_rag_in_depth)): the whole system underneath. It rests on Corrective #5 (CRAG), Adaptive #4 (router, IRCoT), RAG Fusion #3 (f6a search) and Contextual #2 (the corpus, chunks and index).

## Same system, one step swapped

I reused Self-RAG #6's final system as it is: the 140 cricket Wikipedia articles and 4,650 chunks of [contextual-rag-in-depth](https://github.com/adityajadhavv18/contextual-rag-in-depth), Adaptive #4's router, Fusion #3's f6a search, CRAG with its fallback library of 9 left-out biographies, answer prompt v2, and Self-RAG's checker. `grag/grag_bridge.py` puts Self-RAG's code on the import path, and `grag/graph_route.py` replaces one entry, the multi route.

**No new questions.** The ruler is Self-RAG's tune set (`golden_cricket_selfrag_tune_v1`, 78 questions). The router sends 39 of them to multi: 16 old, 8 chain, 6 pair, 3 gap, 3 trap and 3 look-alike. The none (9) and single (30) routes don't change, so their answers replay from the cache, word for word.

## The graph: built by code, for $0

Every article prints regular "Key: value" lines:

```
Match: 20–24 June 2025 | England v India           (a tour's match box)
England: 465 (100.4 overs); Ollie Pope 106 (137); Jasprit Bumrah 5/83 (24.4 overs)
Result: England won by 5 wickets
Venue: Headingley, Leeds
Player of the match: Ben Duckett (Eng)
Umpires: Chris Gaffaney (NZ) and Sharfuddoula (Ban)
```

A final prints the same facts once, at the top of its article. A ground or player article starts with its own facts: Capacity, Opened, Born. Regular lines are safest read by code (sprint 09's lesson), and the chain questions' first hop always sits on one. So `grag/extract_code.py` reads them with regular expressions. No LLM is involved.

| Link | From the line | Count |
|---|---|---|
| (Match) -[HELD_AT]-> (ground) | `Venue:` | 2,154 |
| (Match) -[PLAYER_OF_MATCH]-> (player) | `Player of the match:` | 1,899 |
| (Match) -[TOP_BATTER / TOP_BOWLER]-> (player) | scorecard lines | 4,363 / 4,319 |
| (Match) -[TEAM / WON_BY / UMPIRE]-> | `Match:`, `Result:`, `Umpires:` | 4,286 / 1,943 / 4,119 |
| (Match) -[PART_OF]-> (series article) | the article it is in | 2,193 |
| (Match) -[DESCRIBED_IN]-> (Chunk) | the chunks that tell its story | 6,348 |
| (article) -[HAS_CHUNK]-> (Chunk) | | 4,956 |
| (Chunk) -[MENTIONS]-> (article) | a chunk names another article's title | 5,387 |

**Names are merged in code onto the 149 article titles**, which include the 9 fallback biographies. "Headingley, Leeds", "M.A. Chidambaram Stadium, Chennai" and "Mitchell Starc (KKR)" become the articles "Headingley Cricket Ground", "M. A. Chidambaram Stadium" and "Mitchell Starc". A match's ground or player therefore leads straight into that article's chunks. Every link keeps the line it came from and the chunk that line sits in.

Three quirks of the source text needed handling:
- IPL boxes mark the home side as "(H) Mumbai Indians".
- The 2012 IPL article prints the ground in its `Result:` line.
- A match listed inside a ground's own article prints no venue, so it links to that ground.

After these fixes, 2,154 of 2,156 matches have a venue.

**The store is Neo4j**, KG_basics's `kg-neo4j` container. Every cricket node carries a `Cricket` label, so my KG_basics practice graph stays untouched. It holds 8,893 nodes (1,781 entities, 2,156 matches and 4,956 chunks) and 41,967 links. It loads in 22 s and uses 638 MB of memory.

**Coverage check** (`grag/hop_coverage.py`): for each of the 8 chain questions, I named the first hop's match by hand. The graph leads from it to the article the golden set names as the answer's source, and that article holds the golden evidence chunk word for word. **8 of 8.**

## The walk: Neo4j for the links, local bge for the words

Neo4j never reads the question. The Cypher queries are fixed templates that take a match id. The parts that need language understanding go to bge-small, the local embedding model f6a's dense search already uses, so the walk calls no LLM.

1. **Seed.** bge compares the question with the 2,156 match names ("2011 Cricket World Cup final", "Indian cricket team in England in 2025 › Test series › 1st Test") and takes the closest. "first Test" is written "1st Test" on both sides. The matches behind f6a's top 3 chunks come next.
2. **Hop 1.** Neo4j returns the seed's links to grounds or players that have an article. bge picks one by comparing the question with a plain sentence for each, such as "the player of the match: Mitchell Starc".
3. **Hop 2.** bge ranks that article's chunks and the walk keeps the best 3. It adds the chunk holding the line that made the hop (`Venue: …`), so the answer step sees both hops.

The walk's chunks get guaranteed seats in the LLM reranker's pool of 20, next to f6a's fused list. The answer step reads the reranker's top 5 plus any graph chunk not already among them, at most 9 passages. IRCoT's median pool was 9.

A free dry run on the 39 multi questions shaped three choices (`grag/walk_dry.py`, no LLM):

| Choice | What went wrong before it |
|---|---|
| Hop 2 searches with the target's name + **the question without its match words** ("Mitchell Starc: Which school did player match attend?") | The match words point back at hop 1 and drown out what is asked. Starc's school chunk ranked **32nd** in his article with the whole question and **1st** with this. Over the 8 chains, the gold chunk moved from ranks up to 32 into the top 3 every time |
| The link is also chosen with the question without its match words | "the 2024 IPL final" pulled chain-07 to the ground (0.611) instead of the player of the match (0.605) |
| Tour matches and "unofficial Tests" are seeds only when asked for; a match listed inside a ground's article may hop to that article | bge read "1st unofficial Test" as "1st Test"; chain-02's seed was the 2011 final as listed in the Wankhede article |

## Two fixes around the walk

**Linked grading.** CRAG's grader reads one passage at a time against the whole question. Take "Which school did the player of the match in the 2024 IPL final attend?": the walk put Starc's school passage first, but that passage never mentions the IPL final. All five passages scored 0, CRAG threw them away, and the answer was "I don't know". So each graph passage is now graded against the question with its link filled in:

```
Which school did the player of the match in the 2024 IPL final attend? (Known from the match record:
in 2024 Indian Premier League final, the player of the match was Mitchell Starc.)
```

Starc's passage went from 0 to 10, and the answer became "Homebush Boys High School". f6a's passages are graded as before. The prompt, the bars and the fallback are unchanged.

**IRCoT fallback.** When CRAG rejects every passage on the graph route, the question runs through the old IRCoT loop instead. Some answers sit where only re-searching reaches. On twin-03 ("who bowled Mumbai's last over in the 2017 IPL final?"), the walk followed the final to Mitchell Johnson's biography, which doesn't describe the over. IRCoT found the right passage on its second search. The fallback fired on 3 of 39 questions. All 3 answers came back word for word as IRCoT's, at $0 from the cache, and the graph attempt's tokens and time still count.

## Run it

From `graph_rag_in_depth/`, next to working `self_rag_in_depth/`, `corrective_rag_in_depth/`, `adaptive_rag_in_depth/`, `rag_fusion_in_depth/`, `contextual_rag_in_depth/`, `eval_harness/` and `KG_basics/`. Qdrant and Neo4j run in Docker (`docker start qdrant kg-neo4j`). The OpenAI key is needed for anything not yet cached.

```bash
OPENAI_API_KEY=sk-dummy uv run grag/replay_selfrag.py   # Phase 1: Self-RAG's tune score, from the cache
uv run grag/extract_code.py                    # Phase 2: the graph from the regular lines ($0)
uv run grag/load_neo4j.py                      # Phase 2: into Neo4j
uv run grag/hop_coverage.py                    # Phase 2: both hops of every chain question in the graph?
OPENAI_API_KEY=sk-dummy uv run grag/walk_dry.py   # Phase 3: the walk alone on the 39 multi questions ($0)
uv run grag/smoke.py                           # Phase 3: the graph route on 5 questions
uv run grag/linked_check.py                    # Phase 4: linked grading on chain-07
uv run grag/graph_tune.py --linked --fallback  # Phase 4: the final system on the 78 tune questions
uv run grag/graph_tune.py --linked --fallback --records   # Phase 5: + the records index
```

## Results

Scores are the share of questions answered right, marked by Contextual #2's answer checker (1 = right, 0.5 = half, 0 = wrong), with a 3-vote checker beside it. "I don't know" counts as 0 on an answerable question and as right on a no-answer one.

### Phase 1: the reuse works

Self-RAG #6's full system, run through the bridge on its 78 tune questions, scored **0.897**. That matches Self-RAG's saved run question by question (score, route, CRAG action, Self-RAG's pick and answer text) at $0: a dummy API key proved every call came from the cache.

### Phase 3: the walk alone

| On the 39 multi questions | |
|---|---|
| a path found | 39 / 39 |
| chain questions: second hop's gold chunk reached | **walk 7 / 8**, IRCoT's pool 5 / 8 |
| smoke test, 5 questions (chain-02, chain-04, chain-07 + two right ones) | 2 / 5 → 4 / 5 |

### Phase 4: the graph on tune

| Tune 78 | right (1 vote / 3 votes) | multi (39) | chain (8) | fixed / broken vs IRCoT | multi $/question | multi p50 |
|---|---|---|---|---|---|---|
| Self-RAG #6 (IRCoT) | 0.897 / 0.897 | 34 | 4 | – | $0.0072 | 20.4 s |
| graph walk | 0.917 / 0.910 | 35.5 | 6.5 | 3 / 2 | $0.0038 | 15.2 s |
| + linked grading | 0.929 / 0.923 | 36.5 | 7.5 | 4 / 2 | $0.0038 | 13.1 s |
| **+ IRCoT fallback (final)** | **0.942 / 0.936** | **37.5** | **7.5** | **4 / 1** | **$0.0043** | **12.2 s** |
| + "add, don't replace" (tried, off) | 0.904 / 0.904 | 34.5 | 6 | 2 / 2 | $0.0046 | 15.4 s |

Over all 78 questions, the final system costs $0.0034 per question instead of $0.0049, and its median time is 11.1 s instead of 15.4 s. Each multi question saves IRCoT's one to three reasoning calls.

**"Add, don't replace"** gave the answer step f6a's own top 5 with the graph chunks *after* them, so that the graph could only add. It lost. Graph chunks read in positions 6 to 9 come after f6a's look-alikes. On chain-02, CRAG kept the right Wankhede chunk at position 7, but the answer took "five years to build" from the Narendra Modi chunk at position 1. Where the graph's passages sit matters as much as whether they are there.

What changed, question by question (final system vs Self-RAG #6):

| Question | IRCoT → graph | Why |
|---|---|---|
| chain-02 | 0 → 1 | 2011 final -[HELD_AT]-> Wankhede → "built in about 13 months"; IRCoT went to the Narendra Modi Stadium |
| chain-03 | 0 → 1 | 2025 IPL final -[HELD_AT]-> Narendra Modi Stadium → its opening chunk; with IRCoT's passages, Self-RAG's checker rejected the right answer |
| chain-04 | 0 → 1 | 1st Test of India's 2025 tour -[HELD_AT]-> Headingley → 18,350; IRCoT read Edgbaston passages and refused |
| chain-07 | 0 → 1 | 2024 IPL final -[PLAYER_OF_MATCH]-> Mitchell Starc → his school, kept by linked grading |
| twin-03 | 1 → 1 | CRAG rejected every graph passage, so the IRCoT fallback ran and gave IRCoT's answer |
| chain-06 | 1 → 0.5 | "the 1st Test at the ground that hosted the 2nd Test": the seed was the 1st Test, so the answer added "the 2nd Test was at Perth" |

chain-06's seeds scored 0.758 (1st Test) and 0.755 (2nd Test). Near-ties like this are common: 9 of the 39 multi questions have two seeds from the same article within 0.005. Walking both would touch 7 right answers to win at most half a point back, so the walk takes one.

Still wrong in the final system, as in Self-RAG #6: trap-05, alike-04, alike-05 and none-11. All four are records or look-alike questions, and three of them go to the single route, which the graph does not touch.

### Phase 5: a records index (tried, off)

Record questions fail for another reason. "Alzarri Joseph (Mumbai Indians) took the best bowling figures in the IPL" sits as one line inside a long chunk of 2019 match boxes. Only SPLADE finds that chunk, in 3 of f6a's 15 lists, so RRF ranks it 46th and the reranker, which reads 20, never sees it. Elsewhere the corpus is full of look-alikes such as "the highest individual score in an IPL match *at Eden Gardens*".

I built a second Qdrant collection from the corpus's own record sentences:
- **Lines:** 2,246 of them, picked by code, each whole line kept with its heading.
- **Tags:** each is tagged by where it comes from, such as `[IPL records · Twenty20]` or `[International records · Test]`.
- **Scope notes:** gpt-4o-mini filled fixed fields from each line alone (competition, stage, place, holder, overall, when), and code wrote them as a note, e.g. `Scope: IPL (Twenty20) · playoffs · overall record: no · 2023`.
- **Use:** the top 3 got guaranteed seats in the reranker's pool on the single and multi routes.

The corpus never states that Karun Nair's 303* is the highest Test score by an Indian against England. It gives the 303* and "the second Indian to score a triple century". I added that one record as a **hand-written line** (`records_notes.CUSTOM`). It answers trap-05, so trap-05 is reported apart.

| Tune 78 | right | without trap-05 (77) |
|---|---|---|
| final system | 0.942 | 0.955 |
| + records, scope notes and the hand-written line | 0.949 | **0.948** |

It fixed alike-04 (Alzarri Joseph, 6/12) and trap-05 (only through the hand-written line), and it broke two right answers:
- **pron-03:** a Mitchell Johnson record line reached the passages, and the answer took its score, 4/143, instead of 1/35.
- **chain-06:** the new seats pushed the Adelaide passage out of the top 5, and the answer became "Perth, 2018".

On the other 77 questions it lost ground, so it stays off. alike-05 stayed wrong with its scope note in plain view. Gill's line says "playoffs · overall record: no", and gpt-4o-mini still answered "Gill's 129 is the highest IPL score".

### Spend

The graph cost $0 to build. The whole build spent about $0.43 on gpt-4o-mini. The Phase 4 runs took $0.19, the records notes $0.09 and the records runs $0.15.

The 24 test questions have not been run.

## Lessons

- **Links fix the second hop when the first hop sits on a regular line.** Venue and player of the match are printed the same way in every article, so code builds the graph for free, and the walk reaches what IRCoT's wording-driven searches missed (7 of 8 chains vs 5).
- **The walk's words decide where it lands.** The question's match words point back at the hop already taken. Taking them out before choosing the link and the chunks moved every chain's gold chunk into the top 3.
- **A grader that reads one passage at a time cannot see a second hop.** A passage that answers the second step never mentions the first. Telling the grader the link that brought it fixed this without changing CRAG.
- **Order matters.** The same graph chunks helped when they were reranked into the top 5 and hurt when they were read after f6a's look-alikes (0.929 → 0.904).
- **Keep the old road as a detour.** The graph walk is better on chains, but re-searching still reaches some passages a link can't. Falling back to IRCoT only when CRAG rejects everything fixed twin-03 at no cost.
- **Extra seats are not free.** Record lines misled two questions that had nothing to do with records, and a scope note next to a look-alike did not stop gpt-4o-mini from using it.

## When I'd reach for it

When questions **chain through facts the corpus prints in a fixed form**: venue, award, captain, team. The second step's page should never mention the first. Here the graph built by code costs nothing, the walk needs no LLM, and it replaces a reasoning loop with links that are faster, cheaper and more often right. It does not help one-hop questions, records or look-alikes, which have no link to follow. Keep a search loop as the fallback for answers that sit away from any link.

## Files

| File | Purpose |
|---|---|
| `grag/grag_bridge.py` | Puts Self-RAG #6's code (and through it Corrective's, Adaptive's, Fusion's and Contextual's) on the import path; copies Self-RAG's LLM cache |
| `grag/replay_selfrag.py` | Phase 1: Self-RAG's tune score through the bridge, from the cache |
| `grag/extract_code.py` | Phase 2: the graph from the corpus's regular lines; names merged onto article titles |
| `grag/load_neo4j.py` | Phase 2: loads the graph into Neo4j under `Cricket` labels |
| `grag/hop_coverage.py` | Phase 2: both hops of every chain question in the graph? |
| `grag/walk.py` | The walk: seed, hop 1, hop 2 |
| `grag/graph_route.py` | The multi route with the walk; linked grading; IRCoT fallback; `FINAL` is the final setup |
| `grag/walk_dry.py` | Phase 3: the walk alone, free |
| `grag/smoke.py` | Phase 3: the graph route on 5 questions |
| `grag/linked_check.py` | Phase 4: linked grading on the questions it targets |
| `grag/graph_tune.py` | Phase 4 and 5: tune runs (`--linked`, `--fallback`, `--add`, `--records`) |
| `grag/records_index.py`, `records_notes.py`, `records_route.py` | Phase 5: the records index, its scope notes and hand-written line, and its rerank seats |
| `grag/records_dry.py`, `records_smoke.py` | Phase 5: free check and 5-question test |
| `grag/results/*.json` | Every phase's numbers, per question |
