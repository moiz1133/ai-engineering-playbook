# production-tools

## What This Is

`production-tools` is a standalone Python package providing six genuinely production-quality tools for AI agents: web search, sandboxed code execution, file reading, read-only Postgres querying, REST API calling, and a safe calculator. Every tool shares one small interface (`BaseTool`), validates its inputs and outputs with Pydantic, and ships with a real security control for its specific risk — not a placeholder, an actual one. This is a tools library, not an agent: there's no framework, no orchestration, and no planning logic anywhere in this repo.

## The Six Tools

| Tool | Production-quality concern it demonstrates |
|---|---|
| `web_search` | Retry/backoff on rate limits (tenacity), hard timeout, graceful "no results" handling (never raises on a zero-result search) |
| `code_executor` | The highest-risk tool: real Docker sandboxing (no network, read-only fs, memory/CPU caps, non-root user), with a denylist as a first-pass filter only |
| `file_reader` | Path traversal protection that also catches symlink escapes, since resolving a path follows symlinks before the containment check runs |
| `postgres_query` | Read-only enforcement layered three ways: statement-type check, keyword denylist (catches data-modifying CTEs that the type check alone misses), and stacked-statement rejection |
| `rest_api` | SSRF protection that resolves DNS and checks every redirect hop, not just the URL the caller supplied |
| `calculator` | Why "just use eval()" is never acceptable, even for arithmetic -- an AST whitelist instead |

## The BaseTool Interface

Every tool inherits from `BaseTool` (`src/base.py`): a `name`, an LLM-facing `description`, a Pydantic `input_schema`, a Pydantic `output_schema`, and an async `execute(inputs) -> output`. Callers use `await tool.run(**kwargs)`, which validates raw keyword arguments against `input_schema` before `execute()` ever runs -- invalid inputs raise `pydantic.ValidationError` at the boundary, never reach tool logic, and every tool always returns a validated Pydantic model, never a raw dict or string. `to_openai_schema()` and `to_anthropic_schema()` turn `input_schema` into the exact shape each provider's tool-calling API expects, so any tool here plugs into a real LLM conversation with zero adapter code.

## Setup

```bash
git clone <this-repo-url>
cd agents/01-production-tools
pip install -r requirements.txt
cp .env.example .env   # fill in TAVILY_API_KEY / POSTGRES_* / OPENAI_API_KEY as needed
docker build -t production-tools-sandbox:latest -f docker/sandbox.Dockerfile .
```

Not every tool needs every credential: `calculator` and `file_reader` need nothing at all; `web_search` needs `TAVILY_API_KEY`; `postgres_query` needs a reachable Postgres; `code_executor` needs a running Docker daemon plus the sandbox image built above; `rest_api` needs nothing but does need real network access to whatever URL you call.

## Usage

```python
import asyncio
from src.tools.calculator import CalculatorTool

async def main():
    result = await CalculatorTool().run(expression="sqrt(144) + 8")
    print(result)  # result=20 expression='sqrt(144) + 8'

asyncio.run(main())
```

Run the full six-tool walkthrough with `python examples/basic_usage.py` -- it calls every tool directly and prints each structured response with Rich. Tools whose dependency isn't configured (no `TAVILY_API_KEY`, no Docker daemon, no reachable Postgres) fail through their own real error handling rather than being skipped or faked, which is itself part of the demonstration.

## Wiring to an LLM

`examples/with_llm.py` is a minimal OpenAI function-calling loop: list all six tools via `.to_openai_schema()`, send a query to `gpt-4o-mini` with those tools available, dispatch whichever tool the model calls, feed the structured result back, and print the final answer. A real run against `"What is the square root of 144, plus 8?"`:

```
User: What is the square root of 144, plus 8?
Tool Call: calculator({'expression': 'sqrt(144) + 8'})
Tool Result: calculator: {"result":20,"expression":"sqrt(144) + 8"}
Assistant (final answer): The square root of 144 is 12, and when you add 8 to it, the result is 20.
```

The orchestration is intentionally thin -- one dispatch loop, no framework, no multi-step planning -- because the point of this example is proving the tool schemas and execution model work with a real LLM, not showcasing an agent architecture.

## Security Considerations

- **`code_executor` (sandboxing)**: containers run with no network (`network_disabled=True`), a read-only root filesystem except a `tmpfs` `/tmp`, a 256MB memory cap, a 0.5-core CPU cap, and a non-root user baked into the image. Every container is force-removed after execution, win or lose -- no state ever persists between runs. The regex denylist (`import subprocess`, `os.system`, `__import__("os")`, etc.) is explicitly a first-pass filter, not the real defense; it's checked and documented as such in the code.
- **`file_reader` (path traversal)**: absolute paths and any `..` path segment are rejected outright, and every remaining path is resolved and checked against the base directory with `Path.is_relative_to()`. Because `.resolve()` follows symlinks before that check runs, a symlink planted inside the base directory that points outside it is caught by the same check, not a separate one.
- **`rest_api` (SSRF)**: the host is checked against a denylist (`localhost`, `127.0.0.1`, `169.254.169.254`, `0.0.0.0`) and an optional allowlist, then DNS-resolved so a public-looking hostname that resolves to a private/loopback/link-local address is still blocked -- name-based blocking alone misses DNS rebinding. Every redirect hop is re-validated the same way before being followed (max 5 redirects), since SSRF-via-redirect (an allowed URL that 302s to an internal address) is a real, common bypass of URL-only checks. Response bodies are streamed with a hard size cap to bound memory use.
- **`postgres_query` (read-only enforcement)**: `sqlparse` must classify the statement as `SELECT`, multiple stacked statements are rejected outright, and a keyword denylist scans the raw SQL text as defense-in-depth -- this last check exists specifically because a data-modifying CTE (`WITH x AS (DELETE FROM t RETURNING *) SELECT * FROM x`) is classified as `SELECT` by the statement-type check alone. Queries always run parameterized; the tool never string-formats caller-supplied values into SQL.
- **`calculator` (no eval)**: expressions are parsed with `ast.parse` and evaluated by walking the tree against an explicit whitelist of node types, operators, function names, and constants. Anything outside that whitelist -- including attempts like `__import__("os")` -- raises `ToolSecurityError` before any code-like construct is ever evaluated.

## What This Deliberately Does NOT Include

- **Any agent framework** (LangChain, LangGraph, CrewAI, AutoGPT, DSPy, LlamaIndex) -- these are plain async Python classes, usable from any framework or none.
- **Orchestration or planning logic** -- `examples/with_llm.py` shows one tool-call round trip, not a planning loop; deciding *which* tools to call in what order is the embedding agent's job, not this library's.
- **Memory or state management** -- every tool call is independent; nothing here remembers a previous call.
- **Multi-tool coordination** -- no tool calls another tool, and there's no mechanism for them to.

The value here is six tools that are each genuinely well-built for their specific risk, not a demonstration of agent architecture -- that's what makes this a tools library rather than an agent.
