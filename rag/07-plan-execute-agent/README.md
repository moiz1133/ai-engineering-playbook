# Plan-and-Execute Research Agent

## What This Is

This project is a minimal implementation of the Plan-and-Execute agent pattern for research tasks. Given a topic on the command line, it plans a fixed set of sub-questions, researches each one sequentially with web search, and synthesizes everything into a cited markdown report. It's built with plain Python and the OpenAI SDK — no agent framework — specifically so the pattern itself stays visible in the code rather than hidden behind abstractions.

## The Pattern

Plan-and-Execute separates a research task into three strict phases: **plan** (one LLM call turns the topic into a fixed list of sub-questions before any research happens), **execute** (each sub-question is researched sequentially — search, then summarize — using exactly the plan produced in phase one), and **synthesize** (a final LLM call turns all the collected findings into one coherent, cited report). The defining property is that the plan is committed to upfront and never revised mid-run — contrast this with ReAct, which interleaves reasoning and action and can dynamically add, drop, or reorder steps as it learns from each result. Plan-and-Execute trades that adaptability for predictability: you always know up front exactly how many searches will run and what they'll be about, which makes cost, latency, and behavior far easier to reason about than a dynamically re-planning agent.

## Setup

```bash
git clone <this-repo-url>
cd plan-execute-agent
pip install -r requirements.txt
cp .env.example .env   # then fill in your OPENAI_API_KEY (TAVILY_API_KEY optional)
python -m src.main "Impact of HNSW on production RAG systems"
```

## Usage

```bash
python -m src.main "Impact of HNSW on production RAG systems"
python -m src.main "Impact of HNSW on production RAG systems" --verbose
```

Expected output structure:

1. A Rich-rendered table showing the plan (each sub-question, its search query, and why it matters) — printed before any research begins.
2. Progress lines as each step executes in order: which sub-question is running, how many results came back, how long it took, and the 1-2 sentence summary generated for that step (`--verbose` additionally lists each result's title and URL).
3. A final panel with the path to the saved report, e.g. `outputs/report_20260724_092058.md`.

Search defaults to DuckDuckGo (`ddgs` package) since it needs no API key. Set `SEARCH_PROVIDER=tavily` and `TAVILY_API_KEY` in `.env` to switch providers — `src/tools/web_search.py` dispatches on that one setting.

## Self-Reflection (Optional)

Pass `--reflect` and, after the report is synthesized, the agent critiques its own output before saving it: a critic LLM judges whether the report actually answers the topic, then the agent either accepts it, revises it with what it already has, or runs a few targeted new searches and re-synthesizes — all without ever calling the planner again or touching the original plan.

```bash
python -m src.main "Impact of HNSW on production RAG systems" --reflect
```

A real reflection run against that exact topic produced [`outputs/reflections/reflection_20260725_124204.json`](outputs/reflections/reflection_20260725_124204.json), paired with the revised report at [`outputs/report_20260725_124204.md`](outputs/report_20260725_124204.md). The critic never accepted the report as fully sufficient across 3 iterations (real research reports on a broad topic rarely reach 0.8 confidence with only 4 search results per sub-question), so it chose "re-execute" every time — running 3 new targeted queries per iteration, folding in 12 new results each round, and pushing the report from a 20-source draft to a 40-source, more evidence-backed final version. Confidence moved 0.40 → 0.55 → 0.55 across the three iterations; a snippet from iteration 1:

```json
{
  "iteration": 1,
  "critique": {
    "is_sufficient": false,
    "missing_information": [
      "No specific measured latency values or recall rates for HNSW compared to exact nearest neighbor search at production scale.",
      "No detailed examples of known failure modes or limitations of HNSW in production RAG deployments."
    ],
    "additional_queries": [
      "HNSW latency recall tradeoff benchmarks in production environments",
      "Case studies on HNSW failure modes in production RAG systems",
      "Comparative analysis of HNSW and IVF indexing methods in RAG systems"
    ],
    "confidence": 0.4,
    "overall_assessment": "The report provides a general overview of HNSW but fails to deliver specific, quantitative insights and real-world examples necessary to fully address the research questions."
  },
  "decision": "re-execute",
  "improvement_notes": "Ran 3 additional queries, incorporated 12 new result(s)."
}
```

The loop takes one of three paths after each critique, based on `is_sufficient`, `confidence`, `weak_sections`, and `additional_queries`:

- **Accept** — the report is sufficient and the critic is confident (`confidence >= 0.8`). Done.
- **Revise** — either the report is sufficient but confidence is still low, or the critic flagged specific weak sections without proposing any new searches. The reviser LLM improves the report using only information it already has.
- **Re-execute** — the report is insufficient and the critic proposed concrete search queries. Those queries run through the same `web_search.py` tool the executor uses, the new results get their own citation numbers (appended after the existing sources, never renumbering what's already cited), and the reviser incorporates them into a new draft.

The loop is capped at `MAX_REFLECTION_ITERATIONS = 3` (`src/config.py`) — a deliberate choice, not an oversight. Self-reflection over an LLM's own output can in principle keep finding "one more gap" indefinitely; a hard iteration cap (plus a rough token-budget guard) keeps cost and latency bounded and predictable, at the cost of not guaranteeing a "perfect" report. If the cap is hit, the best report so far is saved anyway, with the reflection log noting `"stop_reason": "max_iterations"`.

## Sample Report

A real report generated by this exact codebase is committed at [`outputs/report_20260724_092058.md`](outputs/report_20260724_092058.md), for the topic *"Impact of HNSW on production RAG systems"*. It ran the full pipeline for real: a 5-step plan, 20 live DuckDuckGo searches (4 results per step), 5 LLM-generated step summaries, and one synthesis call that produced 5 cited sections plus a 18-source, deduplicated Sources list — no hand-edited content.

## What This Deliberately Does NOT Include

- **ReAct-style dynamic re-planning** — the plan is fixed after Phase 1 and never revised based on what search turns up. This is the entire point of Plan-and-Execute as a pattern; adding it back would just turn this into a worse ReAct implementation.
- **Memory** — no short-term or long-term memory across runs. Each invocation is a fresh, stateless research task.
- **Multi-agent orchestration** — one agent, three sequential phases, no sub-agents or delegation.
- **A vector database** — nothing is embedded or indexed. Search results are used once, in-context, for a single synthesis call.
- **Any agent framework** (LangChain, LlamaIndex, LangGraph, CrewAI, AutoGPT, DSPy) — the pattern is implemented directly so it's visible in the code, not hidden behind a framework's abstractions.
- **Multi-agent debate** — reflection here is a single agent critiquing its own output, not multiple agents arguing with each other over a report.
- **Dynamic re-planning during reflection** — `--reflect` can add new searches and revise the report, but the original Plan from Phase 1 is never touched or regenerated; only searches and synthesis are revised.

Every omission above is a deliberate scope boundary: this project exists to show Plan-and-Execute in its cleanest form, and each of these additions would either duplicate what the pattern already does or dilute the point being demonstrated.
