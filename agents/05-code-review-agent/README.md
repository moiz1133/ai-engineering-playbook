# code-review-agent

## What This Is

`code-review-agent` is an AutoGen-based code review system where three specialist
agents -- Critic, Fixer, and Tester -- communicate in a group chat to iteratively
review and improve a piece of code. The Critic finds issues, the Fixer applies them,
and the Tester writes and reasons through tests against the result, cycling until both
the Critic and Tester independently approve or a round limit is hit. Every review in
this repo's `outputs/` directory is the real output of that conversation, not a
scripted or edited transcript.

## Why AutoGen Here

The five prior projects in this repo (`rag/07-plan-execute-agent`,
`agents/01-production-tools` through `agents/04-supervisor-agent`) were all built with
plain Python and asyncio specifically to understand what an orchestration framework
would otherwise be hiding -- decomposition, parallel dispatch, dependency handling,
termination logic, all written out explicitly. This project makes the opposite choice
deliberately: a group chat where three peers read a shared conversation and take turns
responding to whatever was said most recently is *exactly* what AutoGen's `GroupChat` /
`GroupChatManager` primitives are built for, and reimplementing that message-passing
and speaker-selection machinery by hand would just be rebuilding a worse version of
what the framework already does well. One real wrinkle surfaced while building this:
`pip install pyautogen` (as commonly documented) now resolves to Microsoft's rewritten,
async `autogen-agentchat` package, which has no `import autogen` namespace and none of
the classic `AssistantAgent(llm_config=...)` / `GroupChat` / `GroupChatManager` /
`UserProxyAgent` API this project's group-chat pattern depends on. `requirements.txt`
pins `ag2==0.9.10` instead -- the last release of the community-maintained fork that
still ships that classic synchronous API under `import autogen`, verified by installing
it and inspecting the actual class signatures before writing any code against them.

## The Three Agents

**Critic** is a systematic, review-only reader of the code: it finds bugs, security
issues, performance problems, missing error handling, and style violations, always in
the same structured format, and always ends with an explicit `APPROVE` or `NEEDS WORK`
verdict plus a severity rating. It never writes or suggests fixed code -- its only job
is finding problems precisely enough that the Fixer doesn't have to guess what "add
null checks" means.

**Fixer** reads the Critic's most recent findings and produces the complete corrected
file -- never a diff, never a partial snippet -- addressing every CRITICAL and HIGH
severity issue while deliberately preserving the code's original intent rather than
opportunistically refactoring things the Critic didn't flag. It reports back exactly
what it changed and why, plus what it intentionally left for later.

**Tester** writes a pytest suite for the Fixer's latest code covering the happy path,
meaningful edge cases, and a regression test named after each bug the Critic found.
Since there's no runtime available inside the group chat, the Tester mentally traces
through the code to predict each test's outcome, and only recommends `APPROVE` when
both the code's critical/high issues look genuinely resolved and its own test suite
looks sound.

## The Review Cycle

```
        +-----------+
        |  Critic   |  reviews the current code, posts structured findings
        +-----------+
              |
              v
        +-----------+
        |  Fixer    |  addresses CRITICAL/HIGH issues, posts complete fixed code
        +-----------+
              |
              v
        +-----------+
        |  Tester   |  writes pytest cases, predicts outcomes, posts a verdict
        +-----------+
              |
              v
   [ GroupChatManager checks: did Critic AND Tester both just say APPROVE? ]
              |
       no ----+---- yes
       |             |
       v             v
   back to        conversation
    Critic          ends
  (next round)
```

One round is a full Critic -> Fixer -> Tester pass. Speaker order is enforced by
`speaker_selection_method="round_robin"` over exactly `[Critic, Fixer, Tester]` -- the
`UserProxyAgent` that starts the conversation is deliberately left out of that rotation
(see the comment in `src/group_chat/manager.py` for why including it would insert a
spurious fourth turn into every cycle).

## Termination Conditions

The conversation ends the moment BOTH of these are true at once: the most recent
Critic message says `APPROVE`, and the most recent Tester message recommends `APPROVE`.
Critic approval alone is deliberately not enough -- a Fixer's change can look right to
a static reviewer while still breaking under the Tester's traced-through test cases, so
both have to independently agree before the cycle stops (`src/group_chat/termination.py`).
If that never happens, AutoGen's own `max_round` budget (sized to guarantee
`MAX_ROUNDS` full cycles complete before it can fire) cuts the conversation off, and
the last Tester message stands as the final state -- verdict `MAX_ROUNDS_HIT` rather
than `APPROVED`.

## Setup

```bash
cd agents/05-code-review-agent
pip install -r requirements.txt
cp .env.example .env
# edit .env: set OPENAI_API_KEY
```

## Usage

```bash
# Review a file
python -m src.main --file examples/sample_code/buggy_fibonacci.py

# Review code directly
python -m src.main --code "def add(a, b): return a + b"

# Review with a custom round limit
python -m src.main --file examples/sample_code/insecure_auth.py --rounds 4
```

Each run prints the live AutoGen conversation as it happens (every agent's full
message, exactly as generated -- nothing is hidden or summarized during the run),
followed by a Rich summary once the chat ends:

```
Code Review Multi-Agent System
Source: insecure_auth.py

Starting group chat...
----------------------------------------
[... full live Critic / Fixer / Tester conversation printed by AutoGen ...]
----------------------------------------
APPROVED after 1 rounds (16.5s)

Summary:
  Rounds: 1 of 3 max
  Issues found: 6
  Tests generated: 5
  Tokens used: 4,612

Saved: outputs/review_insecure_auth_20260804_182431.md
```

## Sample Reviews

Four real, unedited reviews are committed in [`outputs/`](outputs/) -- one per file in
`examples/sample_code/`, generated by actually running `python -m examples.run_demo`
against all four:

| File | Verdict | Rounds | Issues Found | Tests Generated |
|---|---|---|---|---|
| `buggy_fibonacci.py` | APPROVED | 1 | 4 | 5 |
| `insecure_auth.py` | APPROVED | 1 | 6 | 5 |
| `slow_query.py` | APPROVED | 1 | 6 | 5 |
| `missing_tests.py` | APPROVED | 1 | 6 | 5 |

Worth reading in full: [`outputs/review_insecure_auth_20260804_182431.md`](outputs/review_insecure_auth_20260804_182431.md)
-- the Fixer's response replaces MD5 with salted PBKDF2-HMAC, swaps the `==` password
comparison for `hmac.compare_digest` to close the timing-attack gap, and switches the
session token to `os.urandom`, all genuinely correct fixes for the bugs the Critic
found, not superficial edits.

## What This Does NOT Include

- **No actual code execution.** The Tester reasons about what its tests would do by
  tracing through the code, it never runs them -- `code_execution_config=False` is set
  explicitly on the `UserProxyAgent` so AutoGen never executes anything either. This
  keeps the system's only real capability "read code, write text," a deliberately
  narrower and safer surface than letting an LLM-controlled loop execute arbitrary code.
- **No CI/CD integration.** This is a standalone CLI, not a GitHub Action or pre-commit
  hook -- wiring it into a pipeline is a separate, legitimate follow-on project.
- **No PR or GitHub workflow.** It reviews a file or a code string given directly on
  the command line; it doesn't open, comment on, or fetch pull requests.
- **No persistent memory across review sessions.** Every invocation starts a fresh
  group chat with no knowledge of past reviews, even of the same file.
- **No web search or external tools.** All three agents reason purely from the code
  and conversation history they're given.
- **No human-in-the-loop interruption.** `human_input_mode="NEVER"` throughout -- once
  started, a review runs to completion unattended.
- **No more than three specialist agents**, and no CLI commands beyond running a review.
