These are the various parts of the architecture we'll need to build. This is what I'm thinking so far. Add onto this doc as we go, but don't rewrite or delete anything written by Jian (me). Subagents will build each component. Need tests (pytest and smoke tests) for each component.

-----JIAN PLANS-----

**Foundation**
All architecture should be linked. Outputs, CLI commands, and docs should be human readable (especially takehome reviewer friendly).
Parallelize as much as possible, including subagent spawning. We'll use async gather for calling external APIs.
There should be a final run.py file that runs inference on the haiku agent mentioned later and evals then writes results to a single jsonl file for that run.

**Data Gen**
We need a data generation pipeline for 400 examples where the tool should be used (400 is a buffer we'll trim it to 384) and 68 where the tool should not be used. This subagent should be and opus-4-7 subagent with high effort that is allowed to search the web and come up with a large list of topics for TV shows and Sports past March 2025. I'm thinking like a list of ~100 TV shows and sporting events. This subagent will need to verify that these shows and sporting events have wikipedia pages and enough information in the leader for synthetic data. TV and sports are broad umbrella terms -- I'd count anime, k-dramas, cartoons, etc. as falling under TV shows and pro wrestling and other stuff under sports. This subagent is allowed to get pretty jiggy with its research and should not restrict itself to only things up until its 

We'll use two API calls to opus-4-7 with low effort for both

For 400: Given the same tool as the Haiku agent we're evaling. Opus will be instructed to use the tool to generate an answer FIRST, then a synthetic user prompt that would result in this answer with the tool call. This gives us a golden/ground truth answer to eval against. Synthetic user prompts will be revolved around facts that might have changed since March 2025 and warrant a wikipedia search.

For 68: Have Opus generate 68 answers and prompts for unchangeable general knowledge facts (Who wrote).

**Eval**
Two graders:

1. Programmatic: checks if a tool was used at all. Boolean output.
2. LLM judge: Judges haiku response given a golden answer. It can be sonnet-4-6 with low effort. Structured output with 1-3 sentence reasoning FIRST, then boolean value for whether or not the answer was correct.

We'll need to track accuracy, precision, recall, tp, tn, fp, fn stats. Also need to have a way of comaparing different runs with McNeMar's for statistical significance.

**Haiku agent**
- Single tool that returns top-k article lead sections. Needs exponential backoff to avoid rate limiting. Simple tool description
- Tool calls and responses are tracked
- Easy prompt iteration and tool description iteration.
-----JIAN PLANS------

-----CLAUDE AMENDMENTS (2026-05-04, post-discussion)-----
Captures decisions made in conversation. Jian's block above is the source of truth for original intent; this section adds detail and resolves open questions. **All subagents must read this entire file before starting work, and must conform strictly to the "Pre-spawn contracts" section. If something is unspecified, pause and ask — do not invent schema.**

## Architecture

End-to-end pipeline. All artifacts land on disk so a reviewer can inspect each stage independently.

```
[Subagent A: topic research]
        │
        ▼
data/seeds.jsonl  (~200 verified topics, tagged tv|sports, sliced 0..23)
        │
        ├─────────────────────────────────────┐
        ▼                                     ▼
[datagen/gen_tool_required.py]        [datagen/gen_stable.py]
24 async Opus 4.7 (medium) workers     single Opus 4.7 (low) call
each handles 16 seeds sequentially     no tool, generates 80 → trim 68
imports wiki/client.py for the tool
retry-until-384 success
        │                                     │
        ▼                                     ▼
data/tool_required.jsonl              data/tool_not_required.jsonl
        │                                     │
        └─────────────────┬───────────────────┘
                          ▼
                      [run.py]
              load both, tag tool_required_label,
              call agent.haiku_agent.run_agent(question, tool_description_id, system_prompt_id)
              bounded async concurrency, imports wiki/client.py
                          │
                          ▼
              runs/<ts>.inference.jsonl
                          │
                          ▼
                  [eval/grade.py]
              programmatic (tool called?) + Sonnet 4.6 judge (answer correct?)
                          │
                          ▼
              runs/<ts>.results.jsonl   +   runs/<ts>.summary.json
                          │
                          ▼
                 [eval/compare.py]
              McNemar's between two run files
```

### Directory layout

```
.
├── PLANS.md, README.md, run.py
├── agent/
│   ├── haiku_agent.py        # inference loop
│   └── prompts.py            # TOOL_DESCRIPTIONS + SYSTEM_PROMPTS registries
├── wiki/
│   └── client.py             # search_wikipedia + SQLite cache  (SHARED MODULE)
├── datagen/
│   ├── gen_tool_required.py  # 24 async workers
│   └── gen_stable.py         # single call, 80 → 68
├── eval/
│   ├── grade.py              # programmatic + LLM judge
│   ├── stats.py              # accuracy / precision / recall / TP-TN-FP-FN
│   └── compare.py            # McNemar's
├── tests/                    # pytest + smoke for every module
├── data/                     # seeds.jsonl, tool_required.jsonl, tool_not_required.jsonl
├── cache/wikipedia.sqlite    # shared by agent + datagen
└── runs/<timestamp>.{inference,results}.jsonl   runs/<timestamp>.summary.json
```

`wiki/client.py` is the single shared module. Both `agent/haiku_agent.py` (subagent C) and `datagen/gen_tool_required.py` (subagent B) **import** its `search_wikipedia` function — same code, same cache, no re-implementation anywhere.

## Pre-spawn contracts

These are locked before any subagent spawns. Subagents conform exactly.

### `data/seeds.jsonl`  (one topic per line)
```json
{"title": "Severance (TV series)", "wikipedia_url": "https://en.wikipedia.org/wiki/Severance_(TV_series)", "category": "tv", "post_march_2025_fact": "Season 2 finale aired in [date] with [event].", "topic_slice": 0}
```
`category` ∈ {"tv","sports"}. `topic_slice` is int 0..23.

### `data/tool_required.jsonl`
```json
{"id": "tr_0000", "seed_title": "Severance (TV series)", "user_question": "...", "golden_answer": "...", "datagen_search_queries": ["..."], "datagen_n_tool_calls": 3}
```

### `data/tool_not_required.jsonl`
```json
{"id": "tnr_0000", "user_question": "...", "golden_answer": "..."}
```

### `search_wikipedia` tool

Python signature: `search_wikipedia(query: str) -> dict`

Return shape:
```json
{"title": "string|null", "url": "string|null", "lead": "string", "error": "string|null"}
```
- Success: title/url populated, lead is plaintext (no wiki markup), error=null.
- No-hit / failure: title=null, url=null, lead="", error="<short reason>". Never raises.
- Cache: SQLite at `cache/wikipedia.sqlite`, key=`(endpoint, query)`. Cache hits bypass network.
- Backoff: exponential + jitter on 429/5xx.
- Anthropic tool spec: input_schema = `{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}`. Tool **description text** is iterable and lives in `agent/prompts.py:TOOL_DESCRIPTIONS`.

### Inference trace  (`runs/<ts>.inference.jsonl`)
```json
{
  "question_id": "tr_0042",
  "tool_required_label": true,
  "user_question": "...",
  "golden_answer": "...",
  "messages": "<full Anthropic messages array, including tool_use & tool_result blocks>",
  "final_answer": "...",
  "n_tool_calls": 3,
  "tool_description_id": "baseline",
  "system_prompt_id": "baseline",
  "model": "claude-haiku-4-5",
  "latency_ms": 1234
}
```

### Graded record  (`runs/<ts>.results.jsonl`)
Inference trace plus:
```json
{
  "tool_called": true,
  "tool_use_correct": true,
  "answer_correct": true,
  "judge_reasoning": "<1-3 sentences from Sonnet — REASONING FIELD COMES FIRST in the structured-output schema>"
}
```

### Run summary  (`runs/<ts>.summary.json`)
```json
{
  "run_id": "<timestamp>",
  "tool_description_id": "baseline|v1|...",
  "system_prompt_id": "baseline|v1|...",
  "n": 452,
  "tool_use":           {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "tp": 0, "fp": 0, "tn": 0, "fn": 0},
  "answer_correctness": {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "tp": 0, "fp": 0, "tn": 0, "fn": 0},
  "n_tool_calls_avg": 0.0
}
```

### McNemar comparison  (`eval/compare.py` output)
```json
{"run_a": "...", "run_b": "...", "metric": "answer_correct", "b": 0, "c": 0, "p_value": 0.0, "delta_accuracy": 0.0}
```
`b` and `c` are discordant pair counts. Always report b/c alongside p — don't oversell significance for small discordant counts.

## Decisions

- **Iteration axes: tool description AND system prompt.** Both are iterable. Baseline anchor is `system_prompt_id="baseline"` (empty string `""`) + `tool_description_id="baseline"` (one neutral sentence). Subsequent runs may vary either or both. Caveat: when both vary in a single run, attribution of lift to one axis vs the other is muddy — prefer one-at-a-time A/Bs when reading deltas. McNemar's works pairwise regardless.
- **k=1.** Multi-hop allowed via repeated tool calls. Hard cap 5 calls per turn.
- **Sample sizes** from Cochran's (infinite pop, p=0.5): 384 = 95% CI ±5%, 68 = 90% CI ±10%.
- **24-worker dedup model.** 24 async Opus workers × 16 seeds each = 384. Each worker is a single growing-context conversation; intra-worker dedup happens via context. Programmatic global near-duplicate filter at the end.
- **User-style framing.** Opus phrases user_question colloquially (see subagent B prompt for full spec).
- **Topic scope:** TV + sports broadly defined (anime, k-dramas, cartoons, streaming, reality / pro wrestling, esports, niche sports). Post-March-2025 facts only. **No politics.**
- **MediaWiki endpoint: LOCKED** — Action API single-call pattern using `generator=search` chained into `prop=extracts`. One round trip, returns title + url + plaintext lead. Concretely:
  ```
  GET https://en.wikipedia.org/w/api.php
      ?action=query
      &generator=search&gsrsearch=<query>&gsrlimit=1
      &prop=extracts|info
      &exintro=1&explaintext=1
      &inprop=url
      &format=json&formatversion=2
  ```
  Map response: `query.pages[0].title` → `title`, `query.pages[0].fullurl` → `url`, `query.pages[0].extract` → `lead`. If `query.pages` is empty/missing → return the no-hit error shape.
  Why this over the alternatives: (a) single call beats search-then-extract two-call patterns; (b) REST `/page/summary/{title}` doesn't do search and needs an exact title (we'd need a search call anyway); (c) Core REST `/search/page` returns only a short highlighted excerpt, not the lead. User-Agent must be set per Wikimedia etiquette: `wikipedia-tool/0.1 (https://github.com/jouyang97/wikipedia-tool; jianouyang001@gmail.com)`. Disambiguation pages: if the top hit is a disambiguation page, the extract will read list-shaped; let the agent decide to re-query rather than filtering at the client level (keeps the tool dumb and predictable).
- **Models.** Under test: Haiku 4.5 (`claude-haiku-4-5`). Data-gen workers: Opus 4.7 (`claude-opus-4-7`) **medium** effort. Stable-set generator and topic researcher: Opus 4.7 — researcher = high effort, stable-set = low effort. LLM judge: Sonnet 4.6 (`claude-sonnet-4-6`) low effort.
- **Metrics:** tool-use correctness (binary, expected 1.0 on the final prompt), answer correctness (LLM judge), avg `n_tool_calls`. McNemar's for run-vs-run.
- **No human-review JSON keys.** Spot-checking happens out-of-band.

## Subagents

Four subagents. **Subagent A produces data; B/C/D write code.** Each:
- Reads PLANS.md fully before starting.
- Conforms to "Pre-spawn contracts" exactly.
- Ships pytest + a smoke test for every module it writes (per Jian's directive at the top of this file).
- Pauses and asks rather than inventing schema if something is unspecified.

**Spawn order:** A first (slowest, web-bound). B and D can spawn in parallel right after, since they only depend on contracts. C waits on the MediaWiki endpoint decision (Claude pins it while A is running).

### Subagent A — Topic research  (Opus 4.7, high effort, web search enabled, single instance)
Outputs `data/seeds.jsonl`. Only data-emitting subagent.

### Subagent B — Data-gen scripts  (code-writer)
Outputs `datagen/gen_tool_required.py`, `datagen/gen_stable.py`, tests.

### Subagent C — Haiku agent + Wikipedia client  (code-writer)
Outputs `wiki/client.py`, `agent/haiku_agent.py`, `agent/prompts.py`, tests.

### Subagent D — Eval / grader / stats / compare  (code-writer)
Outputs `eval/grade.py`, `eval/stats.py`, `eval/compare.py`, tests.

### `run.py`  (Jian and Main Agent owns; not a subagent task)
Glue: load both datasets → tag with `tool_required_label` → call `run_agent` per question (bounded async) → call grader → write the three runs/ files. CLI: `--tool-description-id baseline|v1|...` and `--system-prompt-id baseline|v1|...` (both default to `baseline`).

## Subagent prompts (review before spawning)

### Subagent A — Topic research

```
Read PLANS.md fully before starting. Your output is data/seeds.jsonl conforming to the seed schema in "Pre-spawn contracts." Your own knowledge cutoff is January 2026.

Goal: ~100 TV-show topics + ~100 sporting-event topics with significant, factual, verifiable developments AFTER March 2025. The eval will use these to force the agent under test to use a Wikipedia search tool.

Definitions are intentionally broad — see PLANS.md "Topic scope." NO POLITICS.

For each candidate:
1. Use web search to identify a specific post-March-2025 development (a season premiered, a tournament was won, a cast change happened, a record was broken, an upset occurred, other things you can think of).
2. Verify the entity has an English Wikipedia article (en.wikipedia.org/wiki/<Title>).
3. Verify the article's LEAD SECTION (intro paragraphs before the first heading) contains enough about that development for a 2-4 sentence Q&A. If the fact is only in the body, skip.
4. Tag with category ("tv" | "sports") and a topic_slice 0..23. Distribute slices roughly evenly across both categories so each downstream worker gets a comparable mix.

Aim for diversity: mix genres, regions (US/UK/Korea/Japan/Latin America/etc.), mainstream and niche.

Quality over quantity. If you can only verify 80 of one bucket, return 80 — orchestrator rebalances.

Output: write JSONL to data/seeds.jsonl. One topic per line. No prose. No padding.
```

### Subagent B — Data-gen scripts

```
Read PLANS.md fully before starting. Conform exactly to the schemas and tool signature in "Pre-spawn contracts." Ship pytest + smoke tests.

Deliverables:
- datagen/gen_tool_required.py
- datagen/gen_stable.py
- tests/test_gen_tool_required.py   (mocked Anthropic + mocked tool)
- tests/test_gen_stable.py

gen_tool_required.py:
- CLI: --seeds data/seeds.jsonl --out data/tool_required.jsonl --target 384
- Load seeds; partition by topic_slice 0..23.
- 24 async workers via asyncio.gather. Each worker handles its slice's ~16 seeds SEQUENTIALLY in a SINGLE growing Opus 4.7 (medium effort) conversation. Growing context = intra-worker dedup.
- Each worker uses the search_wikipedia tool — IMPORT from wiki.client. DO NOT re-implement the tool or the cache.
- Per seed, prompt Opus to: (1) search Wikipedia (1+ tool calls allowed, max 5) to ground a factual answer, (2) emit one JSON line per the tool_required.jsonl schema, (3) on inability to ground, emit {"seed_title": "...", "skip_reason": "..."} so the orchestrator can replace it.
- Orchestrator pulls replacement seeds from a held-back pool (or loops back through skips with retries) until 384 successes.
- After 384 successes: programmatic near-duplicate filter on user_question (e.g., normalized 5-gram Jaccard > 0.5 → drop later occurrence). If that drops below 384, retry-until-384 again.
- Pass these USER-QUESTION FRAMING instructions verbatim into the Opus prompt:
    Phrase user_question like an actual curious person typing it, NOT like a search query. Vary register: casual ("yo did X really happen?"), context-setting ("I heard last month that..."), opinion-bait ("is X overrated now?"), follow-up-shaped ("wait, didn't they replace Y?"), comparative ("how does the new season compare to the old one?"). Do NOT phrase like "When did season 3 of X premiere?" — that's search-engine-shaped and trivially retrievable. The question must still be fully answerable from the searches you ran.
- Write to --out as JSONL.

gen_stable.py:
- CLI: --out data/tool_not_required.jsonl --target 68
- Single Opus 4.7 (low effort) call, NO tool.
- Prompt Opus for 80 stable evergreen Q/A pairs per the "Topic scope" rules in PLANS.md (no politics, no post-March-2025, no obscure trivia, colloquial phrasing).
- Trim/dedup to 68. Write JSONL.
```

### Subagent C — Haiku agent + Wikipedia client

```
Read PLANS.md fully before starting. Conform exactly to "Pre-spawn contracts." The MediaWiki endpoint decision is pinned in PLANS.md before you spawn — use it. Ship pytest + smoke tests (no real network in tests; use requests-mock or equivalent).

Deliverables:
- wiki/client.py
- agent/haiku_agent.py
- agent/prompts.py
- tests/test_wiki_client.py
- tests/test_haiku_agent.py   (mocked Anthropic + mocked tool)

wiki/client.py:
- search_wikipedia(query: str) -> dict per the tool signature contract.
- SQLite cache at cache/wikipedia.sqlite, key (endpoint, query). Forward-compatible schema.
- Exponential backoff with jitter on 429/5xx.
- Returns {"title": None, "url": None, "lead": "", "error": "..."} on failure. NEVER raises.
- Sets a descriptive User-Agent per Wikipedia API etiquette (include contact info or repo URL).

agent/haiku_agent.py:
- run_agent(question: str, tool_description_id: str, system_prompt_id: str = "baseline", model: str = "claude-haiku-4-5") -> InferenceTrace dict per contract.
- System prompt text pulled from agent.prompts.SYSTEM_PROMPTS[system_prompt_id]. The "baseline" entry is the empty string "" — agent passes it straight through to the Anthropic SDK.
- Single tool with input_schema {"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}; description pulled from agent.prompts.TOOL_DESCRIPTIONS[tool_description_id].
- Loop until model returns a final assistant message with no tool_use blocks, OR n_tool_calls hits 5 (hard stop, then ask the model for a final answer with the tool removed).
- Track every tool_use and tool_result block in messages[]. Compute n_tool_calls and latency_ms. Echo tool_description_id and system_prompt_id into the returned trace.

agent/prompts.py:
- TOOL_DESCRIPTIONS: dict[str, str]. Seed minimally with:
    "baseline": one short, neutral sentence — e.g. "Search Wikipedia for an article matching the query."
    "v1": placeholder string — Jian iterates here.
- SYSTEM_PROMPTS: dict[str, str]. Seed minimally with:
    "baseline": "" (empty string)
    "v1": placeholder string — Jian iterates here.
```

### Subagent D — Eval / grader / stats / compare

```
Read PLANS.md fully before starting. Conform exactly to "Pre-spawn contracts." Ship pytest + smoke tests.

Deliverables:
- eval/grade.py
- eval/stats.py
- eval/compare.py
- tests/test_grade.py
- tests/test_stats.py
- tests/test_compare.py

grade.py:
- grade_one(trace: dict) -> graded record per contract.
- Programmatic: tool_called = any tool_use block in trace["messages"]; tool_use_correct = (tool_called == trace["tool_required_label"]).
- LLM judge: Sonnet 4.6, low effort, structured output schema = {"reasoning": str, "correct": bool}. THE REASONING FIELD MUST COME FIRST in both the schema definition and the system/user prompt — order matters for quality. Reasoning is 1–3 sentences.
- grade_batch(traces) -> list[graded] with bounded async concurrency.

stats.py:
- compute_metrics(graded_records) -> summary dict per contract.
- Tool-use metrics computed across the FULL set (positive class = tool_required_label is True).
- Answer-correctness metrics across the FULL set (positive class = answer_correct is True; precision/recall here is mostly accuracy-flavored — include for completeness).
- Compute avg n_tool_calls.

compare.py:
- compare_runs(run_a_results_jsonl, run_b_results_jsonl, metric: str) -> dict per contract.
- Pair on question_id. McNemar's exact test (binomial on min(b,c)). Report b, c, p_value, delta_accuracy.
- CLI: python -m eval.compare <a> <b> --metric answer_correct
```

-----END CLAUDE AMENDMENTS-----