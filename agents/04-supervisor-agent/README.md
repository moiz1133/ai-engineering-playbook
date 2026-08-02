# supervisor-agent

## What This Is

`supervisor-agent` is a supervisor/worker multi-agent system built entirely from
scratch: a supervisor decomposes a task into sub-tasks, assigns each to a specialist
worker, runs the independent ones in parallel via `asyncio.gather`, and assembles all
outputs into one coherent final result. Every orchestration decision -- which workers
to use, when they run in parallel versus sequentially, how their outputs get merged --
is plain, readable Python, not a call into a framework's black box.

## Architecture

```
User task (string)
    |
    v
[Supervisor: Decomposer]  --  LLM call
    -> Analyzes the task
    -> Decides which workers are needed and in what role
    -> Produces a WorkPlan: list of SubTasks, each assigned to a worker
    |
    v
[Parallel Worker Execution]  --  asyncio.gather runs Researcher + Analyst + Writer simultaneously
    -> Researcher: searches the web (Tavily), synthesizes cited findings
    -> Analyst:    pure reasoning -- patterns, implications, trade-offs, risks
    -> Writer:     drafts structured markdown sections
    |
    v
[Critic]  --  runs ONLY after the above finish, since it reviews their actual output
    -> Reads Researcher/Analyst/Writer outputs, flags gaps, recommends fixes
    (skipped entirely for tasks that don't need a quality pass)
    |
    v
[Assembler]  --  LLM call
    -> Synthesizes every worker's output into one coherent document
    -> Resolves conflicts, closes gaps the Critic flagged, adds citations
    -> Saves to outputs/ as markdown + a full JSON run log
    |
    v
Final output file
```

Not every task uses all four workers -- see "The Supervisor's Decomposition Logic"
below for how that's decided.

## The Four Workers

**Researcher** searches the live web via Tavily and turns raw results into a cited
summary with a Key Facts list and numbered sources. It's used whenever a task needs
current information the model can't reliably know from training alone -- statistics,
recent events, named case studies.

**Analyst** does pure reasoning with no external tools: it weighs trade-offs, surfaces
risks and implications, and self-reports its own confidence in its output. It's used
for tasks that need structured judgment applied to information already at hand, not
new facts.

**Writer** drafts the actual readable markdown -- whatever structure fits the task,
decided by the model itself rather than a fixed template. It runs on almost every task,
since nearly every task ultimately needs a written result.

**Critic** is the odd one out: it reads the other workers' *completed* outputs and
reviews them for quality, accuracy, and gaps, with an explicit instruction to be
genuinely critical rather than agreeable. It only runs on complex or high-stakes tasks,
and only after the others finish, since there's nothing to critique before they have.

## The Supervisor's Decomposition Logic

The Decomposer is a single LLM call that reads the task and returns a `WorkPlan`: which
workers to use, specific (never generic) instructions for each, and a written
`supervisor_reasoning` explaining the choice. It follows four encoded rules: simple
factual questions get Researcher + Writer only; tasks that require weighing options get
Researcher + Analyst + Writer; complex or high-stakes tasks get all four, with the
Critic reviewing before assembly; and purely creative or reasoning tasks (no need for
current facts) get Analyst + Writer, skipping research entirely. The plan is fixed once
decomposition finishes -- there is no dynamic re-planning mid-run.

## Parallelism

Researcher, Analyst, and Writer don't depend on each other's output, so they're
launched together with a single `asyncio.gather(...)` call and run concurrently --
while one worker is waiting on a network response (Tavily, or the OpenAI API), the
Python event loop is free to make progress on the others instead of idling. The Critic
is the deliberate exception: it depends on what the other three produced, so it's
awaited separately, after the gather completes. The CLI makes this concrete by printing
how much wall-clock time the parallel phase actually took versus what the same workers
would have cost running one after another -- see the real numbers below.

## Setup

```bash
cd agents/04-supervisor-agent
pip install -r requirements.txt
cp .env.example .env
# edit .env: set OPENAI_API_KEY and TAVILY_API_KEY
```

## Usage

```bash
python -m src.main "Analyze the business case for building AI triage systems in Pakistani private hospitals in 2026"
```

Sample console output structure:

```
Task received: [task]

Phase 1: Decomposition
  Decomposing task... done
  Workers assigned: Researcher, Analyst, Writer, Critic
  Plan: [supervisor_reasoning excerpt]

Phase 2: Worker Execution
  [Researcher] Starting...
  [Analyst] Starting...
  [Writer] Starting...
  OK [Analyst] Complete (6.1s, 912 tokens, confidence: 0.85)
  OK [Researcher] Complete (9.6s, 1225 tokens, confidence: 0.70)
  OK [Writer] Complete (10.0s, 1136 tokens, confidence: 0.70)
  [Critic] Reviewing outputs...
  OK [Critic] Complete (5.1s, 2957 tokens, confidence: 0.60)

Phase 3: Assembly
  Assembling final output... done (13.0s)

Output saved: outputs/task_20260803_002551.md
Run log saved: outputs/log_20260803_002551.json

Summary:
  Total time: 36.9s (would be 52.0s without parallelism)
  Total tokens: 6,230
  Est. cost: $0.0019
  Workers used: 4/4
```

That "would be X without parallelism" figure is measured, not guessed: the CLI tracks
the actual wall-clock span of the parallel worker phase and compares it against the sum
of each worker's individual execution time, then applies the difference to the total
run time -- the decomposition, Critic, and Assembly phases (which are sequential either
way) are held constant in the comparison.

Set `PARALLEL_WORKERS = False` in `src/config.py` to run the same workers strictly
sequentially, if you want to see the difference directly rather than via the estimate.

## Sample Output

A real, unedited run is committed at
[`outputs/task_20260803_002551.md`](outputs/task_20260803_002551.md), with its full
run log at [`outputs/log_20260803_002551.json`](outputs/log_20260803_002551.json). That
run used all four workers on a deliberately high-stakes prompt (a fintech startup's
build-vs-buy decision on fraud detection) and took 36.9s wall-clock against an estimated
52.0s had the parallel workers run one after another -- a real ~29% reduction from
parallelism alone on this run, using 6,230 tokens for an estimated $0.0019.

## What This Deliberately Does NOT Include

- **No multi-agent framework** (no LangGraph, CrewAI, AutoGen, Semantic Kernel). Every
  orchestration decision -- decomposition, parallel dispatch, dependency handling,
  synthesis -- is explicit Python, because the point of this project is to show that
  logic, not hide it behind a framework's abstractions.
- **No memory or state persistence across runs.** Each invocation is a fresh, isolated
  run; nothing is remembered between calls to `python -m src.main`.
- **No dynamic re-planning.** The Decomposer's `WorkPlan` is fixed the moment it's
  produced -- workers don't get to request new sub-tasks or change the plan mid-run,
  even if their output reveals the original plan was wrong.
- **No human-in-the-loop.** The entire run, from decomposition through assembly, is
  unattended once started.
- **No web UI.** CLI only.
- **No vector database.** The Researcher's "memory" of search results lives only for
  the duration of one call -- there's no persistent retrieval store.
- **No more than four worker types**, and no CLI commands beyond running a task.
