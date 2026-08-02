# stress-tests

## What This Is

`stress-tests` is a systematic stress-test suite for AI agent tools, verifying that 24 distinct failure conditions across three tools (web search, database, code execution) are handled gracefully. "Handled gracefully" is a precise, checked contract — no unhandled exceptions, no silent corruption, no cryptic errors — not a vague aspiration. Every scenario runs against the real tool classes from this repo's `production-tools` project with deterministic mocks standing in for the network/database/Docker dependency, so this project tests real error-handling code, not a simulation of it.

## Why Graceful Handling Matters

A demo tool that only ever gets called with valid inputs against a healthy backend never has to prove anything about failure handling — it just works, until it doesn't. Production tools break constantly and in different ways than demos do: networks partition, rate limits get hit, databases go down mid-query, containers get OOM-killed, APIs return malformed responses. When a tool doesn't handle these gracefully, the failure doesn't stay contained — an unhandled `KeyError` from a malformed API response looks identical to a bug in the calling code, a hung connection with no timeout ties up resources indefinitely, and a raw database exception leaking up to an LLM-facing tool call can leak schema details or internal infrastructure names to an end user. Verifying graceful handling *before* these conditions happen in production, rather than discovering the gaps from an incident, is the entire value proposition of this project.

## The 24 Scenarios

| # | Scenario | Tool | Failure Injected | Expected Graceful Behavior |
|---|---|---|---|---|
| S1 | `web_search_timeout` | web_search | Response takes 20s vs. 15s timeout | `ToolTimeoutError` within the timeout window |
| S2 | `web_search_network_error` | web_search | DNS/connection failure | `ToolExecutionError` with network context |
| S3 | `web_search_rate_limit` | web_search | HTTP 429 with Retry-After | `ToolExecutionError` after retries exhaust |
| S4 | `web_search_empty_results` | web_search | Valid response, zero hits | Successful empty-list return, not an error |
| S5 | `web_search_malformed_response` | web_search | Unrecognized JSON shape | Typed handling, not `KeyError`/`AttributeError` |
| S6 | `web_search_partial_results` | web_search | 3 of 5 results missing `url` | All 5 returned, missing fields defaulted |
| S7 | `web_search_auth_failure` | web_search | HTTP 401 | Typed error, no leaked API key |
| S8 | `web_search_cascading_retry` | web_search | Fail, fail, then succeed | Transparent success on the working attempt |
| D1 | `database_empty_results` | database | Valid query, zero rows | Successful empty-rows return, not an error |
| D2 | `database_connection_refused` | database | DB unreachable | `ToolExecutionError` with connection context |
| D3 | `database_query_timeout` | database | Query exceeds 30s `statement_timeout` | `ToolTimeoutError` within the timeout window |
| D4 | `database_syntax_error` | database | Malformed SQL | Typed error identifying the bad query |
| D5 | `database_permission_denied` | database | Real `DELETE` query | `ToolSecurityError` before the DB is ever touched |
| D6 | `database_rows_truncated` | database | 500 rows available, `max_rows=100` | 100 rows returned, `truncated=True` |
| D7 | `database_null_values` | database | NULL in several columns | `None` values returned cleanly |
| D8 | `database_type_mismatch` | database | Unexpected column type | Raw value returned as-is, no coercion/crash |
| C1 | `code_syntax_error` | code_execution | Invalid Python syntax | Non-zero exit code *returned*, not raised |
| C2 | `code_runtime_error` | code_execution | Uncaught exception (`1/0`) | Non-zero exit code *returned*, not raised |
| C3 | `code_execution_timeout` | code_execution | Infinite loop vs. timeout | `timed_out=True`, container killed cleanly |
| C4 | `code_memory_exceeded` | code_execution | OOM-killed container | `exit_code=137` returned, no tool crash |
| C5 | `code_infinite_loop` | code_execution | Loop with prior output, then timeout | Partial stdout preserved through the kill |
| C6 | `code_blocked_import` | code_execution | `import subprocess` | `ToolSecurityError` before Docker is touched |
| C7 | `code_stderr_only` | code_execution | `exit_code=0`, stderr only | Treated as success, stderr surfaced as info |
| C8 | `code_empty_output` | code_execution | Runs successfully, prints nothing | Clean success result, no error |

**Real result from an actual run: 22/24 passed.** The two failures are genuine findings in `production-tools`, not test artifacts — see [Reading the Report](#reading-the-report).

## What "Graceful" Means

Every scenario checks all six of these criteria explicitly (`grace_criteria_met` in the result):

1. **No unhandled exception** propagates out of the tool call.
2. **Typed error** — a specific custom exception class, never bare `Exception`.
3. **Human-readable message** — tells the *user* what failed, not the engineer (no leaked internals, no raw driver jargon).
4. **Structured error response** — never `None`, never an empty string.
5. **Consistent state** — no partial writes, no leaked connections/containers after the failure.
6. **Recovery is possible** — the same tool instance can be called again successfully right after.

A scenario only passes if all six are true.

## Setup

```bash
git clone <this-repo-url>
cd agents/03-stress-tests
pip install -r requirements.txt
python -m src.runner
```

No environment variables, no real network/DB/Docker access are required — everything is mocked. `production-tools` should be checked out as a sibling directory (`../01-production-tools`) for scenarios to exercise the real tool classes; see [Integration with production-tools](#integration-with-production-tools).

## Running the Suite

```bash
python -m src.runner                       # all 24 scenarios
python -m src.runner --suite web_search     # one suite: web_search | database | code
python -m src.runner --suite database
python -m src.runner --suite code
python -m src.runner --scenario web_search_timeout   # exactly one scenario by name
```

Each run prints a progress bar as scenarios execute, a color-coded pass/fail per scenario, a full results table, a per-tool summary, and — if anything failed — a highlighted breakdown of what went wrong and why.

## Reading the Report

Each run saves a JSON report to `reports/stress_test_YYYYMMDD_HHMMSS.json`:

```json
{
  "run_timestamp": "...",
  "total_scenarios": 24,
  "passed": 22,
  "failed": 2,
  "by_tool": {
    "web_search": {"total": 8, "passed": 6},
    "database": {"total": 8, "passed": 8},
    "code_execution": {"total": 8, "passed": 8}
  },
  "scenarios": [ /* all 24 ScenarioResult objects, including grace_criteria_met breakdowns and notes */ ]
}
```

A real run (`reports/stress_test_20260728_173308.json`, committed as a sample) found exactly two genuine gaps in `production-tools`, both worth fixing there:

- **`web_search_network_error` (S2) failed** on the `human_readable_message` criterion: the tool's actual error is `"Tavily request failed: Name resolution failed"` — it leaks the internal service name and raw DNS jargon instead of a clean, user-facing message.
- **`web_search_partial_results` (S6) failed** on nearly every criterion: `SearchResult.url` is typed as `str` (not `Optional[str]`), so a single result missing its URL raises a raw Pydantic `ValidationError` that discards *all* results in the batch, not just the malformed one — exactly the "NOT graceful" failure mode this scenario was written to catch.

Every other scenario — including `database_syntax_error` and `web_search_auth_failure`, which both surfaced a real classification nuance (production-tools uses the same `ToolExecutionError` for both connection-layer and caller-input-layer failures) — passed, with the nuance recorded in that scenario's `notes` field rather than silently ignored.

## Integration with production-tools

Scenarios inject failures into the **real** tool classes two different ways, depending on what each tool's constructor actually supports:

- **`database` and `code_execution`**: genuine constructor injection. `PostgresQueryTool(pool=...)` and `CodeExecutorTool(docker_client=...)` both accept their dependency directly, so `MockDatabaseClient`/`MockCodeExecutor` are passed straight in — no monkeypatching.
- **`web_search`**: `WebSearchTool` has no constructor injection point for its HTTP client (it constructs `httpx.AsyncClient(timeout=...)` fresh inside a private function on every call), so `MockSearchClient` is monkeypatched in as a drop-in replacement for `httpx.AsyncClient` itself, for the scenario's duration only.

**A real architectural wrinkle, solved**: both this project and `production-tools` name their own top-level package `src`, so they can never both be import-resolvable under that name simultaneously. `src/config.py`'s `import_production_module()` temporarily evicts this project's own `src.*` modules from `sys.modules`, adds `production-tools` to `sys.path`, imports what's needed, then restores everything — verified for real (not just written and assumed to work) before any scenario code was built on top of it.

If `production-tools` isn't present at `../01-production-tools` (or `PRODUCTION_TOOLS_DIR` in `.env`), every scenario reports itself as cleanly unavailable rather than failing with an import error or silently duplicating the real tool's error-handling logic as a fallback.

## What This Does NOT Cover

- **Load or performance testing** — this suite verifies *correctness* of failure handling for one call at a time, not behavior under concurrent load. That's a separate, legitimate concern with different tooling.
- **Chaos engineering** — no random fault injection, no production traffic replay. Every failure here is deterministic and hand-picked because it's a known, specific failure mode worth verifying.
- **Multi-agent failure modes** — this project tests one tool call at a time in isolation, not how failures cascade or compound across an agentic system calling multiple tools in sequence.

This is intentionally scoped to single-tool graceful handling — proving each tool fails safely on its own is the prerequisite for reasoning about any of the above, not a replacement for them.
