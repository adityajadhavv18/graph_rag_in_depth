# GraphRAG #7: cricket on Wikipedia

Ask "how long did it take to build the stadium that hosted the 2011 Cricket World Cup final, and when did it open?" and Self-RAG #6 answers "five years, opened in 2020". The facts are real, but they belong to the Narendra Modi Stadium: the 2011 final was at the Wankhede. Its multi-step search (IRCoT) reasoned its way to the wrong ground. The corpus says plainly where the final was played, on a line that reads `Venue: Wankhede Stadium, Mumbai`. GraphRAG turns lines like that into links, so the second step follows a link instead of a guess.

I replaced Self-RAG #6's IRCoT loop on the multi route with a walk over a cricket graph, kept everything else, and measured it on the same questions.

| | Tune (78) | Test (24, unseen) |
|---|---|---|
| Self-RAG #6 (IRCoT on multi) | 0.897 | 0.875 |
| **GraphRAG** (graph walk + linked grading + IRCoT fallback) | **0.942** | **0.812** |
| multi-step questions | 34 → 37.5 of 39 | 10 → 8.5 of 11 |
| two-step chain questions | 4 → 7.5 of 8 | 2 → 1 of 3 |
| $ per question | $0.0049 → $0.0034 | $0.0051 → $0.0031 |
| median time | 15.4 s → 11.1 s | 18.5 s → 11.4 s |

**On tune, the graph won clearly. On the 24 unseen test questions it lost**, by 1.5 questions out of 24. In one of the test questions, bge took "the 2025 Champions Trophy final" for "the 2025 IPL final", and the walk followed the wrong match to a confident dead end. It was cheaper and faster on both sets. The walk's choices were shaped on 8 tune chains, and on new chains a single wrong seed decides the answer. So the next build, Agentic RAG, gets **both the graph walk and IRCoT as tools**, and the agent chooses. I built the graph with code alone, for $0.

## Architecture

```
                         ┌──────────────────────┐
                         │   User's question    │
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ ROUTER (Adaptive #4) │   few-shot gpt-4o-mini, one word; unchanged
                         └──┬────────┬────────┬─┘
              ┌─────────────┘        │        └───────────────────────┐
              ▼                      ▼                                ▼
   ┌──────────────────┐   ┌──────────────────┐   ┌────────────────────────────────────────┐
   │ NONE             │   │ SINGLE           │   │ MULTI: GRAPH WALK (new, replaces IRCoT) │
   │ memory + "are    │   │ f6a search       │   │                                        │
   │ you sure?"       │   │ (Fusion #3):     │   │ 1 SEED   local bge: question vs 2,156  │
   │ (Self-RAG #6)    │   │ 4 rewrites +     │   │          match names → best match      │
   │ unsure → SINGLE ─┼──▶│ question; dense /│   │ 2 HOP 1  Neo4j: match -[HELD_AT /       │
   └────────┬─────────┘   │ BM25 / SPLADE;   │   │          PLAYER_OF_MATCH / TOP_…]->     │
            │             │ RRF; LLM rerank  │   │          ground or player; bge picks    │
            │             │ → top 5          │   │          the link                      │
            │             └────────┬─────────┘   │ 3 HOP 2  bge ranks that article's      │
            │                      │             │          chunks → best 3 + the hop line │
            │                      │             │ f6a's fused list + graph chunks (seats)│
            │                      │             │ → LLM rerank → top 5 + graph chunks    │
            │                      │             │ no path → plain f6a                    │
            │                      │             └───────────────────┬────────────────────┘
            │                      └───────────────┬─────────────────┘
            │                                      ▼
            │               ┌──────────────────────────────────────────────┐
            │               │ CRAG (Corrective #5): grade each passage     │
            │               │ 10 / 5 / 0; fallback library of 9 biographies│
            │               │ + LINKED GRADING (new): a graph passage is   │
            │               │   graded with the link that brought it       │
            │               └──────────────────────┬───────────────────────┘
            │                                      │ graph route and every passage rejected?
            │                                      │──► IRCOT FALLBACK (new): Adaptive's loop
            │                                      │    (search → reason → search, ≤ 3) runs instead
            │                                      ▼
            │               ┌──────────────────────────────────────────────┐
            │               │ SELF-RAG (#6): candidates + A_all → checker  │
            │               │ → pick → one retry; unchanged                │
            │               └──────────────────────┬───────────────────────┘
            ▼                                      ▼
                         ┌──────────────────────┐
                         │        ANSWER        │
                         └──────────────────────┘
```

Think of the "see also" signposts in a library. Search is a librarian who brings you the pages that *sound like* your question. The graph is the signposts: 2011 final → see Wankhede Stadium → see "Construction". The page about building the Wankhede never mentions the 2011 final, so it never sounds like the question. Following the signposts gets you there anyway, as long as you started at the right shelf.

- **GraphRAG** (Edge et al., Microsoft, 2024, "From Local to Global: A Graph RAG Approach to Query-Focused Summarization"): an LLM extracts entities and relationships from every chunk into a graph. *Local search* maps a question to entities and hands their neighbourhood's source text to the LLM. My retrieval is local search over a graph built by code. The paper's other half, community summaries for global questions, is not part of this build.
- **IRCoT** (Trivedi et al., 2023, "Interleaving Retrieval with Chain-of-Thought Reasoning"): the loop the walk replaces, kept as the fallback.
- **CRAG** (Yan et al., 2024) and **Self-RAG** (Asai et al., 2023): the passage grader and the answer checker around it, unchanged apart from linked grading.
- **Self-RAG #6** ([self_rag_in_depth](https://github.com/adityajadhavv18/self_rag_in_depth)): the whole system underneath. It rests on Corrective #5, Adaptive #4, RAG Fusion #3 and Contextual #2.

## Same system, one step swapped

I reused Self-RAG #6's final system as it is: the 140 cricket Wikipedia articles and 4,650 chunks of [contextual-rag-in-depth](https://github.com/adityajadhavv18/contextual-rag-in-depth), Adaptive #4's router, Fusion #3's f6a search, CRAG with its fallback library of 9 left-out biographies, answer prompt v2, and Self-RAG's checker. `grag/grag_bridge.py` puts Self-RAG's code on the import path, and `grag/graph_route.py` replaces one entry, the multi route.

**No new questions.** The rulers are Self-RAG's golden sets:
- **Tune:** 78 questions. The router sends 39 to multi: 16 old, 8 chain, 6 pair, 3 gap, 3 trap and 3 look-alike.
- **Test:** 24 questions, never run before this build. The router sends 11 to multi, 10 to single and 3 to none.

The none and single routes don't change, so their answers replay from the cache, word for word.

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

```
                    (Entity: series article)
                              ▲
                              │ PART_OF
   (Chunk) ◄──DESCRIBED_IN── (Match) ──HELD_AT──────────► (Entity: ground) ──HAS_CHUNK──► (Chunk)
                              │  ──PLAYER_OF_MATCH──────► (Entity: player) ──HAS_CHUNK──► (Chunk)
                              │  ──TOP_BATTER / TOP_BOWLER──► (Entity: player)
                              │  ──TEAM / WON_BY──────────► (Entity: team)
                              │  ──UMPIRE─────────────────► (Entity: umpire)
   (Chunk) ──MENTIONS──► (Entity with an article)
   every link keeps {line it came from, chunk that line sits in, source: code}
```

| Link | From the line | Count |
|---|---|---|
| HELD_AT | `Venue:` | 2,154 |
| PLAYER_OF_MATCH | `Player of the match:` | 1,899 |
| TOP_BATTER / TOP_BOWLER | scorecard lines | 4,363 / 4,319 |
| TEAM / WON_BY / UMPIRE | `Match:`, `Result:`, `Umpires:` | 4,286 / 1,943 / 4,119 |
| PART_OF | the article the match is in | 2,193 |
| DESCRIBED_IN | the chunks that tell the match's story | 6,348 |
| HAS_CHUNK | an article's chunks | 4,956 |
| MENTIONS | a chunk names another article's title | 5,387 |

**Names are merged in code onto the 149 article titles**, which include the 9 fallback biographies. "Headingley, Leeds", "M.A. Chidambaram Stadium, Chennai" and "Mitchell Starc (KKR)" become the articles "Headingley Cricket Ground", "M. A. Chidambaram Stadium" and "Mitchell Starc". A match's ground or player therefore leads straight into that article's chunks.

Three quirks of the source text needed handling:
- IPL boxes mark the home side as "(H) Mumbai Indians".
- The 2012 IPL article prints the ground in its `Result:` line.
- A match listed inside a ground's own article prints no venue, so it links to that ground.

After these fixes, 2,154 of 2,156 matches have a venue.

**The store is Neo4j**, KG_basics's `kg-neo4j` container. Every cricket node carries a `Cricket` label beside `CricketEntity`, `CricketMatch` or `CricketChunk`, so my KG_basics practice graph stays untouched. It holds 8,893 nodes (1,781 entities, 2,156 matches and 4,956 chunks) and 41,967 links. It loads in 22 s and uses 638 MB of memory on an 8 GB Mac.

**Coverage check** (`grag/hop_coverage.py`): for each of the 8 tune chain questions, I named the first hop's match by hand. The graph leads from it to the article the golden set names as the answer's source, and that article holds the golden evidence chunk word for word. **8 of 8.**

## The walk: Neo4j for the links, local bge for the words

Neo4j never reads the question. The Cypher queries are fixed templates that take a match id. The parts that need language understanding go to bge-small, the local embedding model f6a's dense search already uses, so the walk calls no LLM.

**A worked example, chain-07: "Which school did the player of the match in the 2024 IPL final attend?"**

```
1 SEED    bge: question vs 2,156 match names        → "2024 Indian Premier League final"   (cosine 0.789)
2 HOP 1   Neo4j: that match's links to articles, each scored by bge against
          the question without its match words ("Which school did player match attend?"):
            the player of the match: Mitchell Starc            0.688  ◄ chosen
            the ground that hosted the match: M. A. Chidambaram Stadium   0.629
            the best bowler / top run-scorer: Pat Cummins       0.562 / 0.556
3 HOP 2   bge ranks Starc's 52 chunks against "Mitchell Starc: Which school did player match attend?"
          → mitchell_starc::s001::00 first ("... attended Homebush Boys High School ...")
          + the chunk with "Player of the match: Mitchell Starc (KKR)"
4 RERANK  graph chunks get seats next to f6a's fused list → Starc's school chunk is passage [1]
5 CRAG    linked grading: "… (Known from the match record: in 2024 Indian Premier League final,
          the player of the match was Mitchell Starc.)" → Starc's chunk 0 → 10, kept
6 ANSWER  "Mitchell Starc, attended Homebush Boys High School [1]"
```

The three choices behind steps 1 to 3 came from a free dry run on the 39 tune multi questions (`grag/walk_dry.py`, no LLM):

| Choice | What went wrong before it |
|---|---|
| Hop 2 searches with the target's name + **the question without its match words** | The match words point back at hop 1 and drown out what is asked. Starc's school chunk ranked **32nd** in his article with the whole question and **1st** with this. Over the 8 tune chains, the gold chunk moved from ranks up to 32 into the top 3 every time |
| The link is also chosen with the question without its match words | "the 2024 IPL final" pulled chain-07 to the ground (0.611) instead of the player of the match (0.605) |
| Tour matches and "unofficial Tests" are seeds only when asked for; a match listed inside a ground's article may hop to that article | bge read "1st unofficial Test" as "1st Test"; chain-02's seed was the 2011 final as listed in the Wankhede article |

The walk's chunks get guaranteed seats in the LLM reranker's pool of 20, next to f6a's fused list. The answer step reads the reranker's top 5 plus any graph chunk not already among them, at most 9 passages. IRCoT's median pool was 9.

## Two fixes around the walk

**Linked grading.** CRAG's grader reads one passage at a time against the whole question. A passage that answers the second hop never mentions the first. Starc's school passage says nothing about the IPL final, so it scored 0 with the other four, CRAG threw them all away, and the answer was "I don't know". Each graph passage is now graded against the question with its link filled in (step 5 above). f6a's passages are graded as before. The prompt, the bars and the fallback are unchanged.

**IRCoT fallback.** When CRAG rejects every passage on the graph route, the question runs through the old IRCoT loop instead. Some answers sit where only re-searching reaches. On twin-03 ("who bowled Mumbai's last over in the 2017 IPL final?"), the walk followed the final to Mitchell Johnson's biography, which doesn't describe the over. IRCoT found the right passage on its second search. The graph attempt's tokens and time still count.

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
uv run grag/graph_test.py                      # Test: both systems on the 24 test questions (runs once)
```

## Results

Scores are the share of questions answered right, marked by Contextual #2's answer checker (1 = right, 0.5 = half, 0 = wrong), with a 3-vote checker beside it. "I don't know" counts as 0 on an answerable question and as right on a no-answer one.

### Phase 1: the reuse works

Self-RAG #6's full system, run through the bridge on its 78 tune questions, scored **0.897**. That matches Self-RAG's saved run question by question (score, route, CRAG action, Self-RAG's pick and answer text) at $0: a dummy API key proved every call came from the cache.

### Phase 3: the walk alone

| On the 39 tune multi questions | |
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

Over all 78 questions, the final system costs $0.0034 per question instead of $0.0049, and its median time is 11.1 s instead of 15.4 s. Each multi question saves IRCoT's one to three reasoning calls. The IRCoT fallback fired on 3 of 39: twin-03, which it fixed, and gap-02 and gap-06, which were right either way. All 3 answers came back word for word as IRCoT's.

**"Add, don't replace"** gave the answer step f6a's own top 5 with the graph chunks *after* them, so that the graph could only add. It lost. Graph chunks read in positions 6 to 9 come after f6a's look-alikes. On chain-02, CRAG kept the right Wankhede chunk at position 7, but the answer took "five years to build" from the Narendra Modi chunk at position 1.

What changed on tune, question by question (final system vs Self-RAG #6):

| Question | IRCoT → graph | Why |
|---|---|---|
| chain-02 | 0 → 1 | 2011 final -[HELD_AT]-> Wankhede → "built in about 13 months"; IRCoT went to the Narendra Modi Stadium |
| chain-03 | 0 → 1 | 2025 IPL final -[HELD_AT]-> Narendra Modi Stadium → its opening chunk; with IRCoT's passages, Self-RAG's checker rejected the right answer |
| chain-04 | 0 → 1 | 1st Test of India's 2025 tour -[HELD_AT]-> Headingley → 18,350; IRCoT read Edgbaston passages and refused |
| chain-07 | 0 → 1 | the worked example above |
| twin-03 | 1 → 1 | CRAG rejected every graph passage, so the IRCoT fallback ran and gave IRCoT's answer |
| chain-06 | 1 → 0.5 | "the 1st Test at the ground that hosted the 2nd Test": the seed was the 1st Test (0.758 vs 0.755), so the answer added "the 2nd Test was at Perth" |

Near-tie seeds like chain-06's are common: 9 of the 39 multi questions have two seeds from the same article within 0.005.

### Phase 5: a records index (tried, off)

Record questions fail for another reason. "Alzarri Joseph (Mumbai Indians) took the best bowling figures in the IPL" sits as one line inside a long chunk of 2019 match boxes. Only SPLADE finds that chunk, in 3 of f6a's 15 lists, so RRF ranks it 46th and the reranker, which reads 20, never sees it. Elsewhere the corpus is full of look-alikes such as "the highest individual score in an IPL match *at Eden Gardens*".

I built a second Qdrant collection from the corpus's own record sentences:
- **Lines:** 2,246 of them, picked by code, each whole line kept with its heading.
- **Tags:** each is tagged by where it comes from, such as `[IPL records · Twenty20]` or `[International records · Test]`.
- **Scope notes:** gpt-4o-mini filled fixed fields from each line alone (competition, stage, place, holder, overall, when), and code wrote them as a note, e.g. `Scope: IPL (Twenty20) · playoffs · overall record: no · 2023`.
- **Use:** the top 3 got guaranteed seats in the reranker's pool on the single and multi routes.

The corpus never states that Karun Nair's 303* is the highest Test score by an Indian against England. It gives the 303* and "the second Indian to score a triple century". I added that one record as a **hand-written line** (`records_notes.CUSTOM`). It answers tune question trap-05, so trap-05 is reported apart.

| Tune 78 | right | without trap-05 (77) |
|---|---|---|
| final system | 0.942 | 0.955 |
| + records, scope notes and the hand-written line | 0.949 | **0.948** |

It fixed alike-04 (Alzarri Joseph, 6/12) and trap-05 (only through the hand-written line), and it broke two right answers:
- **pron-03:** a Mitchell Johnson record line reached the passages, and the answer took its score, 4/143, instead of 1/35.
- **chain-06:** the new seats pushed the Adelaide passage out of the top 5, and the answer became "Perth, 2018".

On the other 77 questions it lost ground, so it stays off and is not part of the test run. alike-05 stayed wrong with its scope note in plain view: Gill's line says "playoffs · overall record: no", and gpt-4o-mini still answered "Gill's 129 is the highest IPL score".

### Test: both systems on the 24 unseen questions

`grag/graph_test.py` ran Self-RAG #6 and the final GraphRAG once on Self-RAG's test set, which neither had seen. It saved the results and refuses to run again.

| Test 24 | Self-RAG #6 (IRCoT) | **GraphRAG (final)** |
|---|---|---|
| right (1 vote / 3 votes) | 0.875 / 0.896 | **0.812 / 0.812** |
| multi (11) | 10 | 8.5 |
| chain (3) / pair (5) | 2 / 5 | 1 / 4.5 |
| look-alike, sure, gap, none, solo | 5/5, 2/3, 1/2, 3/3, 3/3 | the same |
| IRCoT fallback fired | – | 0 times |
| $ per question / p50 time | $0.0051 / 18.5 s | $0.0031 / 11.4 s |

| Question | IRCoT → graph | Why |
|---|---|---|
| chain-22 | 1 → 0 | "the player of the match in the 2025 **Champions Trophy** final": the seed was the 2025 **IPL** final, the walk went to Virat Kohli, and the answer was "I don't know". CRAG accepted one passage, so the fallback never fired |
| chain-21 | 1 → 0.5 | the walk was right (2013 Champions Trophy final → Ravindra Jadeja), but hop 2's chunks missed his birthplace: "born in India" instead of Navagam Ghed, Jamnagar |
| pair-22 | 1 → 0.5 | a graph passage gave Root's 54 "off 37 balls" instead of 36 |
| chain-23 | 0 → 0.5 | 2024 IPL final -[HELD_AT]-> M. A. Chidambaram Stadium → 38,000, nearer than IRCoT's 39,000 (the truth is 38,200) |

The test set is small and only 4 answers changed, but they point the same way. When the seed or the hop-2 chunk is wrong, the walk has no second chance. IRCoT's reasoning steps can still recover. The fallback catches only the case where CRAG rejects everything, and a wrong path often leaves one passage CRAG accepts.

### Spend

The graph cost $0 to build. The whole build spent about $0.57 on gpt-4o-mini:

| Item | Cost |
|---|---|
| Phase 4 runs | $0.19 |
| Records notes | $0.09 |
| Records runs | $0.15 |
| Test | $0.13 |

## What goes into the Agentic track

The decision rule I set before building was that the graph replaces IRCoT if it wins on tune. It won on tune and lost on test. So the next build, Agentic RAG, keeps **both the graph walk and IRCoT as tools**, and an agent chooses between them per question. The results already show what that choice can lean on:
- **Seed confidence:** the walk was right when its seed was clear, and wrong when it confused two finals (chain-22) or two near-tie Tests (chain-06).
- **Hop-2 evidence:** a path that reaches the right article but not the fact (chain-21) needs a search, not another link.
- **Question shape:** chains through venue and player of the match suit the walk. Pairs, records and look-alikes don't.

## Lessons

- **Links fix the second hop when the first hop sits on a regular line.** Venue and player of the match are printed the same way in every article, so code builds the graph for free. On tune, the walk reached second-hop evidence that IRCoT's searches missed (7 of 8 chains vs 5).
- **A walk shaped on 8 questions is fitted to them.** Its choices (one seed, the stripped question, the best 3 chunks) all came from the tune chains. On new chains it was beaten 1 to 2. A single seed is a single point of failure: when bge confuses the 2025 Champions Trophy final with the 2025 IPL final, everything after it is confidently wrong.
- **The walk's words decide where it lands.** The question's match words point back at the hop already taken. Taking them out before choosing the link and the chunks moved every tune chain's gold chunk into the top 3.
- **A grader that reads one passage at a time cannot see a second hop.** Telling the grader the link that brought a passage fixed this without changing CRAG.
- **Order matters.** The same graph chunks helped when they were reranked into the top 5 and hurt when they were read after f6a's look-alikes (0.929 → 0.904).
- **A fallback is only as good as its trigger.** Running IRCoT when CRAG rejects everything fixed twin-03, but a wrong path that leaves one acceptable passage never triggers it (chain-22 on test).
- **Extra seats are not free.** Record lines misled two questions that had nothing to do with records, and a scope note next to a look-alike did not stop gpt-4o-mini from using it.

## When I'd reach for it

When questions **chain through facts the corpus prints in a fixed form** (venue, award, captain, team), the second step's page never mentions the first, and the first step can be pinned down reliably. The graph built by code costs nothing, the walk needs no LLM, and each question is about 40% cheaper and faster than a reasoning loop. Where the starting point is ambiguous (similar match names, two events in one question), I'd use the walk as one tool beside a search loop, not as a replacement for it.

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
| `grag/graph_test.py` | The one test run: both systems on the 24 test questions |
| `grag/results/*.json` | Every phase's numbers, per question (`test_once.json` is the test) |
