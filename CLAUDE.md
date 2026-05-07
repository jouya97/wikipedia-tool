# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

This is an Anthropic prompt-engineering take-home (see `TAKEHOME.md`). The deliverable is a system that uses Claude + a `search_wikipedia(query: str)` tool to answer user questions, plus an eval suite that measures how well it works. The code does not exist yet — `PLANS.md` is the design and the source of truth for what to build.

The primary evaluation criteria are **prompt quality** and **eval design**, not production-grade search infrastructure.

## Hard constraints (from TAKEHOME.md)

- Must use an Anthropic model via the Anthropic API. The agent under test is **Claude Haiku** (see `PLANS.md`); data generation uses **Opus 4.7 (low effort)**; LLM-judge uses **Sonnet 4.6 (low effort)**.
- **Do not use hosted search / RAG tools** (no Anthropic `web_search` tool type, no OpenAI browsing, no Perplexity, etc.). The Wikipedia integration must be built from scratch — MediaWiki API, a dump, or a local index are all fine.
- Outputs and CLI commands should be human-readable and reviewer-friendly — the Anthropic team must be able to run the prototype and see it work immediately (demo mode / sample queries).

## Planned architecture (from PLANS.md)

Three linked components — keep them wired end-to-end so a reviewer can run the whole pipeline. A single `run.py` should drive Haiku-agent inference + eval and write results to one jsonl per run.

1. **Haiku agent** — single `search_wikipedia` tool that returns top-k article lead sections. Exponential backoff on the Wikipedia call to avoid rate limiting. Tool calls and responses must be tracked per question (the programmatic grader needs this). Prompt and tool description should be easy to iterate on.
2. **Data generation** — Opus 4.7 (low effort) for both passes, but a separate **high-effort** Opus subagent (allowed to browse the web) first brainstorms ~100 TV shows and ~100 sporting events post-March 2025, broadly defined (anime, k-dramas, cartoons, pro wrestling, etc.), and verifies each has a Wikipedia page with enough lead-section content for synthetic data.
   - **400 tool-required examples** (buffer; trim to 384): Opus is given the same tool as Haiku, generates an answer *using the tool first*, then writes a synthetic user prompt that would produce that answer. Prompts target facts that may have changed since March 2025. This yields a golden answer for grading.
   - **68 tool-not-required examples**: Opus generates Q&A pairs for stable general-knowledge facts (no tool use expected).
3. **Eval** — two graders run over the agent's outputs:
   - **Programmatic grader**: boolean — was the tool called at all? Used to compute tool-use precision/recall against the dataset's ground-truth labels (tool-required vs. not).
   - **LLM judge**: Sonnet 4.6 (low effort), structured output with **1-3 sentence reasoning first, then a boolean** for answer correctness vs. the golden answer. Order matters — reasoning before verdict.
   - Track: accuracy, precision, recall, TP/TN/FP/FN. Support cross-run comparison via McNemar's test for statistical significance.

Parallelize external API calls (data gen, inference, judging) with `asyncio.gather`. Spawn subagents in parallel where the work is independent.

## Working with PLANS.md

`PLANS.md` is Jian's living design doc. Append to it as the design evolves, but **do not rewrite or delete anything inside the `-----JIAN PLANS-----` block** — that's the user's authored content.

## Setup

`.env` (gitignored — already present locally) holds the `ANTHROPIC_API_KEY`. Any new entry-point script should load it before instantiating the SDK client.

When adding code, document the run commands (install, generate data, run agent, run eval, demo mode) in `README.md` so a reviewer can execute the prototype without spelunking. Each component should ship with pytest tests and a smoke test.
