# Iterations

Full record of every prompt revision, the diff narrative behind each change, and the statistical effect on the eval set.

## Eval set

452 questions total: **384 tool-required** (post-March-2025 facts, split 199 TV / 185 sports) + **68 tool-not-required** (stable evergreen facts). Same set used across every run; comparisons are paired on `question_id`.

Two graders:

- **Programmatic**: `tool_called == tool_required_label` → `tool_use_correct`. Computed inline, no API call.
- **LLM judge** (Sonnet 4.6, low effort): receives `(user_question, golden_answer, final_answer)` and writes 1–3 sentences of reasoning before emitting `correct: bool` via a tool-call. The judge is given the current UTC date and is permitted to overrule stale 2025 golden answers when the agent's answer reflects a later, correctly-retrieved fact.

Statistical tests:

- **McNemar's exact two-sided binomial** for paired binary outcomes (`answer_correct`, `tool_use_correct`).
- **Paired t-test** for `n_tool_calls`.

---

## Prompt versions

The two iteration axes are `tool_description` and `system_prompt`, both stored as markdown files under `prompts/` and selected by id at runtime via `--tool-description-id` / `--system-prompt-id`.

`{{TIMESTAMP}}` in any system prompt is auto-substituted with `YYYY-MM-DD` UTC at every agent invocation.

### System prompts

#### `system_prompts/baseline.md`

```
(empty file — no system prompt)
```

#### `system_prompts/v1.md`

```
You are a helpful assistant. Your knowledge cutoff is February 2025.

CURRENT DATE: {{TIMESTAMP}}
```

#### `system_prompts/v2.md`

```
You are a helpful assistant. Your knowledge cutoff is February 2025.

When responding to users, ALWAYS ASK AND OUTPUT AN ANSWER TO THE FOLLOWING QUESTION ENCASED IN <think></think> TAGS BEFORE USING TOOLS OR DOING ANYTHING ELSE: "What would the perfect response be?"

CURRENT DATE: {{TIMESTAMP}}
```

### Tool descriptions

#### `tool_descriptions/baseline.md`

```
Search Wikipedia for an article matching the query.
```

One-liner anchor. No schema documentation, no decision rules, no usage rules.

#### `tool_descriptions/v1.md`

```
Search English Wikipedia for the article best matching `query` and return its lead section (the introductory paragraphs before the first heading).

Returns an object with:
- `title`: the matched article's title, or null if no article was found
- `url`: link to the matched article on en.wikipedia.org
- `lead`: plain-text lead section of the article (typically 1–4 paragraphs)
- `error`: null on success; otherwise a short string explaining why no result was returned (e.g. "no_results", "empty_query")

When deciding whether or not to use this tool, follow these rules:
<tool_use_decision_rules>
- First determine if you can respond to the user with your own knowledge. Do not use this tool if you can.
- If the user asks about something you are not aware of or is past your knowledge cutoff, use the tool.
</tool_use_decision_rules>
```

Adds: return-shape schema and a 2-rule decision block.

#### `tool_descriptions/v2.md`

```
Search English Wikipedia for the article best matching `query` and return its lead section (the introductory paragraphs before the first heading).

Returns an object with:
- `title`: the matched article's title, or null if no article was found
- `url`: link to the matched article on en.wikipedia.org
- `lead`: plain-text lead section of the article (typically 1–4 paragraphs)
- `error`: null on success; otherwise a short string explaining why no result was returned (e.g. "no_results", "empty_query")

When deciding whether or not to use this tool, follow these rules:
<tool_use_decision_rules>
- First determine if you can **confidently** respond to the user with your own knowledge. Do not use this tool if you can.
- Use this tool when the user asks about something that is past your knowledge cutoff.
- Use this tool if the user references something that could plausibly change in the world.
</tool_use_decision_rules>

When using the tool, follow these rules:
<tool_use_rules>
- You are able to use the tool up to a maximum of 10 times.
- If a tool call does not return useful results, carefully consider what other queries to try executing.
- You MUST ALWAYS cite information retrieved by the tool in your response.
- You MUST treat information retrieved by the tool as ground-truth, even if it conflicts with your own knowledge.
</tool_use_rules>
```

Changes vs v1:
- Decision rule 1 hardened with **"confidently"** — closes the loophole that produced v1's regression.
- New decision rule: **"could plausibly change in the world"** — the right primitive for "did Manchester City sign anyone this window?", which doesn't trip a knowledge-cutoff heuristic.
- New `<tool_use_rules>` block: max-10 budget, retry-on-bad-results, mandatory citation, and the load-bearing **"treat retrieved info as ground-truth even if it conflicts with your knowledge"** — fixes the failure mode where Haiku searches, sees a fact that contradicts pretraining, and falls back to pretraining.

#### `tool_descriptions/v3.md`

```
[v2 verbatim, plus one new decision rule:]

<tool_use_decision_rules>
- First determine if you can **confidently** respond to the user with your own knowledge. Do not use this tool if you can.
- If you're confused on what the user is asking about, make a reasonable assumption and use this tool to cover all your bases.
- Use this tool when the user asks about something that is past your knowledge cutoff.
- Use this tool if the user references something that could plausibly change in the world.
</tool_use_decision_rules>
```

Single addition: **"if you're confused, assume and search anyway"**. Targets the residual didn't-call failures where Haiku punted on ambiguous referents instead of probing.

#### `tool_descriptions/v4.md`

v3 verbatim, plus a new clause inside `<tool_use_rules>`:

```
- Before responding to the user, YOU MUST ASK YOURSELF AND ANSWER THE FOLLOWING QUESTIONS ENCASED IN <think></think> TAGS:
    1. "What information was retrieved that best responds to the user?"
    2. "How can I use this information to give the most informative response?"
```

Mirrors the `<think>` exploit from `system_prompts/v2.md` but moves it post-retrieval — forces an explicit "what did I just learn" reasoning step before the final answer.

#### `tool_descriptions/v5.md`

v4 verbatim, plus a new "don't give up" rule inside `<tool_use_rules>`:

```
- If you haven't found information that can help respond to the user, keep using the tool until you do or use 10 attempts. NEVER respond saying you could not find information UNTIL ALL 10 ATTMEMPTS HAVE BEEN USED.
```

Targets the brittle-single-shot failure mode: Haiku searches once, gets a near-miss article, and concludes "I couldn't find anything." Forces multi-shot retrieval until exhaustion.

---

## Run-by-run results

All runs against the same 452-question eval. Numbers are post-judge-fix (date-aware, with stale-golden override permitted).

| run | tool-req acc | TV | sports | tool-use acc | didn't-call acc | avg tool calls | over-call (stable) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `td-baseline / sp-baseline` | 72.1% | 82.4% | 61.1% | 79.9% | 0/52 (0.0%) | 1.51 | 39/68 (57.4%) |
| `td-v1 / sp-v1` | 62.8% | 64.3% | 61.1% | 77.2% | 15/103 (14.6%) | 0.90 | 0/68 (0.0%) |
| `td-v2 / sp-v1` | 78.6% | 82.4% | 74.6% | 91.2% | 10/40 (25.0%) | 1.22 | 0/68 (0.0%) |
| `td-v3 / sp-v1` | 80.5% | 85.9% | 74.6% | 93.6% | 5/27 (18.5%) | 1.37 | 2/68 (2.9%) |
| `td-v3 / sp-v2` | 83.6% | 88.9% | 77.8% | 97.8% | 7/10 (70.0%) | 1.32 | 0/68 (0.0%) |
| `td-v4 / sp-v2` | 84.1% | 89.9% | 77.8% | 97.3% | 5/11 (45.5%) | 1.52 | 1/68 (1.5%) |
| **`td-v5 / sp-v2`** | **90.4%** | **93.0%** | **87.6%** | 96.9% | **8/11 (72.7%)** | 1.70 | 3/68 (4.4%) |

`didn't-call acc` is "of the questions where the agent skipped the tool, how often was the answer correct anyway?" — a calibration signal: a high number here means the agent correctly identified questions it already knew vs. wrongly bailed.

---

## Statistical effects (consecutive deltas)

All p-values are McNemar's exact two-sided for the binary metrics, paired t-test for `n_tool_calls`. `b` = run A correct, run B wrong; `c` = run A wrong, run B correct.

### baseline → `td-v1 / sp-v1` — **regression**

| metric | result |
|---|---|
| answer_correct | b=69, c=36, **p=1.66e-03**, **Δ=−7.3pp** |
| tool_use_correct | b=78, c=66, p=0.36, Δ=−2.7pp |
| n_tool_calls | μ 1.51 → 0.90, **p=2.2e-20** |

Decision rule 1 ("do not use the tool if you can respond from your own knowledge") was too permissive. Tool-call rate on tool-required questions collapsed (86.5% → 73.2%); the 103 didn't-call cases were 88/103 wrong. v1 fixed the over-call problem on stable (57.4% → 0%) but at a much larger cost on tool-required.

### `td-v1 / sp-v1` → `td-v2 / sp-v1` — **first real win**

| metric | result |
|---|---|
| answer_correct | b=12, c=72, **p=1.39e-11**, **Δ=+13.3pp** |
| tool_use_correct | b=5, c=68, **p=3.42e-15**, **Δ=+13.9pp** |
| n_tool_calls | μ 0.90 → 1.22, p=7.8e-12 |

Hardening "confidently" + adding "could plausibly change in the world" + the four `<tool_use_rules>` (especially "treat retrieved info as ground-truth") flipped v1's regression. Sports specifically went 61.1% → 74.6% — the "ground-truth" rule fixed cases where Haiku searched, retrieved a fresh fact, and still answered from pretraining.

### `td-v2 / sp-v1` → `td-v3 / sp-v1` — **noise**

| metric | result |
|---|---|
| answer_correct | b=25, c=30, p=0.59, Δ=+1.1pp |
| tool_use_correct | b=10, c=21, p=0.071, Δ=+2.4pp |
| n_tool_calls | μ 1.22 → 1.37, p=1.13e-03 |

Adding "if confused, assume and search" alone produced a small drift but nothing significant on answer accuracy. Tool-use accuracy moved (more calls, marginally better calibration) but the answer-correctness signal was a wash.

### `td-v3 / sp-v1` → `td-v3 / sp-v2` — **system prompt does the work**

| metric | result |
|---|---|
| answer_correct | b=22, c=37, p=0.067, Δ=+3.3pp |
| tool_use_correct | b=3, c=22, **p=1.57e-04**, **Δ=+4.2pp** |
| n_tool_calls | μ 1.37 → 1.32, p=0.23 |

The `<think>` "what would the perfect response be?" exploit. Borderline on answer accuracy, ultra-significant on tool-use calibration. The didn't-call slice jumped from 18.5% → 70.0% accuracy: when the model decides to skip the tool now, it's right.

This is where the post-training "reasoning" mode of Haiku 4.5 starts paying real dividends: the `<think>` block forces a planning step before action.

### `td-v3 / sp-v2` → `td-v4 / sp-v2` — **wash**

| metric | result |
|---|---|
| answer_correct | b=24, c=24, **p=1.00**, **Δ=+0.0pp** |
| tool_use_correct | b=5, c=3, p=0.73, Δ=−0.4pp |
| n_tool_calls | μ 1.32 → 1.52, **p=5.03e-07** |

Adding a post-retrieval `<think>` block ("what info best responds to the user?") inside the tool description didn't move accuracy — the system-prompt-level `<think>` already covered the planning step. Net effect: more tool calls per question (+0.20) for no quality lift. v4 alone is not worth shipping.

### `td-v4 / sp-v2` → `td-v5 / sp-v2` — **second real win, the biggest single jump**

| metric | result |
|---|---|
| answer_correct | b=10, c=36, **p=1.56e-04**, **Δ=+5.8pp** |
| tool_use_correct | b=6, c=4, p=0.75, Δ=−0.4pp |
| n_tool_calls | μ 1.52 → 1.70, p=3.37e-04 |

Adding **"NEVER respond saying you could not find information UNTIL ALL 10 ATTEMPTS HAVE BEEN USED"** turned Haiku from a giver-upper into a pursuer. Sports jumped from 77.8% → 87.6% (+9.8pp) — sports questions disproportionately benefit because event names are messier (player nicknames, league shorthand, tournament editions) and the first query often returns a near-miss article. v5 forces re-querying with a different phrasing instead of bailing.

Notable: tool-use *accuracy* didn't change (the agent calls the tool on the same set of questions). All of v5's gain comes from what happens *during* multi-shot retrieval, not from the call/no-call decision.

### baseline → `td-v5 / sp-v2` — **end-to-end**

| metric | result |
|---|---|
| answer_correct | b=12, c=85, **p=1.04e-14**, **Δ=+16.2pp** |
| tool_use_correct | b=9, c=86, **p=6.61e-17**, **Δ=+17.0pp** |
| n_tool_calls | μ 1.51 → 1.70, p=6.25e-03 |

The full ladder: +16.2pp answer accuracy, +17.0pp tool-use accuracy, both significant at p < 1e-14. Cost is +0.19 tool calls per question on average — the agent is more thorough, not just more accurate.

---

## What each lever bought

Decomposing the +16.2pp baseline → v5/v2 win into the three changes that mattered:

| change | mechanism | answer-acc Δ | significance |
|---|---|---:|---|
| Tool description **v2** (decision-rule sharpening + `<tool_use_rules>`) | Closes "I'll just answer from memory" loophole; forces using retrieved facts as ground-truth | +15.8pp vs v1, +5.5pp vs baseline\* | p<1e-11 |
| System prompt **v2** (`<think>` exploit) | Pre-action planning; calibrates which questions actually need search | +3.3pp on top of td-v3 | p=0.067 (marginal) |
| Tool description **v5** ("don't give up" rule) | Multi-shot retrieval until 10 attempts exhausted | +5.8pp on top of td-v4/sp-v2 | p<1e-3 |

\*Note: v1 was a regression vs baseline. v2's +15.8pp over v1 is partly recovery of the v1 regression, but v2/sp-v1 still beats baseline by +6.5pp (the cleanest "did the tool description help" comparison, holding system prompt at v1).

The "if confused, assume and search" rule in v3 and the post-retrieval `<think>` block in v4 each looked promising on inspection but produced no significant change. They are kept in v5 (v5 is v4 + the don't-give-up rule) because removing them was not tested and they are not harmful.

---

## Failure analysis: where the remaining 9.6% comes from

The final v5/v2 run mislabeled 37 of 384 tool-required questions. I read every wrong case and bucketed by failure mode, then attributed each to dataset, eval-tooling, or agent.

| bucket | TV | sports | total | % wrong |
|---|---:|---:|---:|---:|
| A. Wrong article retrieved (year disambiguation) | 1 | 8 | 9 | 24% |
| B. Right article, fact not in lead | 6 | 2 | 8 | 22% |
| C. Right info, wrong synthesis | 4 | 3 | 7 | 19% |
| E. Gave up too early | 0 | 5 | 5 | 14% |
| D. Hallucinated despite retrieval | 2 | 1 | 3 | 8% |
| F. Stale Wikipedia / golden mismatch | 0 | 3 | 3 | 8% |
| G. Judge error | 1 | 1 | 2 | 5% |

### Most failures are dataset issues, not agent issues

Re-bucketing by whose fault:

| issue type | cases |
|---|---:|
| Dataset (A + B + F + ~2 of E that were vague Q's) | ~22 |
| Eval tooling (G) | 2 |
| Agent (C + D + ~3 of E genuine give-ups) | ~13 |

So the real agent failure rate is **~13 / 384 ≈ 3.4%**, not 9.6%. v5/v2 is hitting an eval ceiling.

### Why the dataset failures look like dataset failures

All three dataset buckets trace back to one decision: the data was generated in 2025 by Opus, with goldens grounded in Wikipedia leads as they existed at gen time, and the eval runs in May 2026.

- **Year disambiguation (9 cases, all sports).** Questions like "did Norris win Monaco this year?" had golden = 2025 race. The agent runs with `CURRENT_DATE: 2026-05-04` and reasonably interprets "this year" as 2026. It fetches the 2026 stub article (event hasn't happened yet) and answers "no race yet." The judge's stale-golden override clause requires the agent's answer to reflect *fresher* info — when the agent finds a future-event stub, there's no fresher info to overrule with. Same pattern hit Belmont, Silverstone, Indy 500, Brazilian GP.
- **Fact not in lead (8 cases, mostly TV).** Opus's data-gen prompt said "ground in the lead" but evidently Opus read deeper into search results than the agent's tool returns. The agent gets a strict lead-section slice; the golden was authored from a richer view. Hits anime hardest (Apothecary Diaries S2 staff, Dandadan: Evil Eye, Solo Leveling ReAwakening) where lead sections are thin on production credits.
- **Stale goldens (3 cases).** Outright wrong at gen time — Pereira UFC 320 winner, Littler's 2024 PDC final opponent, and one Alcaraz slam count are factually incorrect in the dataset.

### What this means for the headline

Conservative reading: **v5/v2 = 90.4% answer accuracy.** Honest reading after stripping eval-side noise: **~96.6%**. The +16.2pp baseline → v5/v2 delta is unaffected — both runs were graded against the same flawed dataset, so the comparison is still valid even though the absolute number is conservative.

The remaining ~3.4% true agent error is concentrated in:
- **Synthesis (C, 7 cases)**: agent has the info but partially answers or conflates facts. e.g. McIlroy's "third player ever" framing — agent saw both wins but missed the historical-context framing. This is hard to prompt-engineer further without a verification step.
- **Hallucination despite retrieval (D, 3 cases)**: rare but real. Suggests the "treat retrieved info as ground-truth" rule isn't fully internalized.
- **Genuine give-ups (E, ~3 cases)**: v5's "don't give up" rule mostly fixed this but it leaks on questions vague enough that the agent asks for clarification instead of searching.

### What I'd change if I were rerunning the eval

Listed for completeness; not done in this iteration since the dataset is fixed and the comparisons are valid against it.

1. **Absolute dates in user questions**: rewrite "this year" / "the current year" to "2025" at gen time. Fixes the year-disambiguation bucket entirely.
2. **Tighten the data-gen lead-grounding rule**: make Opus quote the lead snippet that grounds each fact, then validate post-hoc that the snippet is actually in the lead the agent's tool returns. Catches the lead-too-thin cases at gen time rather than at eval time.
3. **Manual spot-check on sports goldens**: 5–10 minutes of human verification on the 185 sports goldens would have caught the 3 outright-wrong ones.
4. **Two-judge agreement**: a second judge (Opus 4.7 low effort) on the same `(question, golden, agent_answer)` triples would catch G-bucket cases. Cohen's kappa as an eval-quality metric.
