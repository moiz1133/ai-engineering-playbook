# memory-assistant

## What This Is

`memory-assistant` is a personal assistant agent built from scratch with three distinct memory types: working, episodic, and procedural. It demonstrates how memory-augmented agents actually work by keeping each memory type's storage, lifetime, and purpose genuinely separate rather than collapsing them into one generic "memory" abstraction. There's no agent framework here — just plain Python, Pydantic, SQLite, and ChromaDB, so the pattern itself stays visible.

## The Three Memory Types

**Working memory** is the current conversation's context: a bounded sliding window of the last N messages (default 20), held in a plain Python `deque` and never persisted. It exists purely to give the LLM immediate context of what's being discussed right now, without paying context-window cost on old conversations. It resets every session — if something matters beyond the current conversation, it has to be extracted into episodic or procedural memory before the session ends, because working memory itself carries nothing forward.

**Episodic memory** is durable, natural-language facts about the user — "User has a daughter named Zara," "User works at Afiniti as a Senior Software Engineer." It's backed by ChromaDB and retrieved by semantic similarity to the current query, not exact match, because facts are relevant based on meaning ("what does the user do for work?" should retrieve a fact about their job even if it never uses the word "work"). Not every message becomes a fact: a `FactExtractor` deliberately decides what's durable and worth remembering versus what's transient chatter.

**Procedural memory** is structured, learned preferences about how the assistant should behave — `response_style: concise`, not "the user said they like short answers on Tuesday." It's backed by SQLite as key/value rows with a category and a confidence score, and it's surfaced into the system prompt on every turn without needing any retrieval step, because there are typically only a handful of preferences and they apply universally rather than being conditionally relevant like facts are.

## The Difference That Matters

This is where most implementations blur two genuinely different concepts together. The test: does this describe **who the user is** (episodic) or **how the assistant should behave** (procedural)?

| | Episodic (facts) | Procedural (preferences) |
|---|---|---|
| Answers | "What do I know about this person?" | "How should I behave with this person?" |
| Example | "User's daughter Zara turned 4" | "response_style: concise" |
| Format | Free natural language | Structured key/value |
| Storage | ChromaDB (semantic search) | SQLite (exact lookup) |
| Retrieved | Conditionally, by relevance to the query | Always, every turn |
| Changes when | Something new happens in the user's life | The user corrects or restates how they want to be treated |

"Told me on Tuesday they like short answers" is a fact about a past event — it belongs in episodic memory. "prefers_concise_responses: true" is a standing instruction for future behavior — it belongs in procedural memory. The same underlying preference can show up as a passing remark (episodic-shaped) or as a stored rule (procedural-shaped); the `FactExtractor` runs two separate extraction passes specifically so this distinction gets made deliberately rather than by accident.

## Setup

```bash
git clone <this-repo-url>
cd agents/02-memory-assistant
pip install -r requirements.txt
cp .env.example .env   # fill in your OPENAI_API_KEY
python -m src.main --session-id your_name
```

## Usage

```
$ python -m src.main --session-id abdul_moiz

Assistant Memory System
Session: abdul_moiz
Working memory: 0 messages | Episodic: 3 facts | Procedural: 1 preferences

You: My daughter Zara just turned 4 today.
Assistant: Happy birthday to Zara! Turning 4 is such a fun age...
[Fact extracted: "User has a daughter named Zara who turned 4 today"]

You: /memory show
                    Episodic Facts
+-------------------------------------------+
| Fact                          | Accessed  |
|--------------------------------+----------|
| User has a daughter named Zara| 0         |
+-------------------------------------------+
                Procedural Preferences
+------------------------------------------------+
| Key            | Value  | Category      | Conf |
|-----------------+--------+---------------+------|
| response_style  | concise| communication | 0.95 |
+------------------------------------------------+

You: /memory forget "Zara"
Removed 1 fact(s) matching '"Zara"':
  - User has a daughter named Zara who turned 4 today

You: /exit
Session ended. Facts saved to episodic memory.
```

Other slash commands: `/memory clear working` (reset working memory only), `/memory clear all` (wipe everything, with a confirmation prompt).

## The Multi-Session Demo

```bash
python examples/multi_session_demo.py
```

Session 1 tells the assistant three things across separate turns: a name/job fact, a communication preference, and a current-learning fact. Session 2 constructs a **brand-new `Assistant` instance** — exactly what a fresh process on a different day would create — and asks an unrelated question: *"Can you explain how HNSW works?"* A real run produces:

- **Procedural memory working**: the reply is concise, technical, and bulleted, matching the stored `response_style: concise, technical` preference — without that preference ever being restated in session 2.
- **Episodic memory working**: the reply connects the answer to the user's stated RAG learning and employer ("...making it suitable for production systems like those you might encounter at Afiniti... enhance your implementation of efficient retrieval mechanisms in your projects") — facts session 2 was never directly told, only retrieved by semantic search against session 1's stored facts.
- **Persistence proven structurally, not just asserted**: session 2 starts with `Working memory: 0 messages | Episodic: 3 facts | Procedural: 1 preferences` — zero working memory (it never carries over) but the full episodic/procedural state from session 1, because those live on disk (`./data/chroma_memory/`, `./data/procedural.db`), not in the Python process.

This demo also surfaced two real bugs during actual runs, both fixed:
1. **Exact-duplicate facts**: the same fact ("User's name is Abdul") got extracted twice across two turns, because the extractor was shown the assistant's reply as "context" and re-derived the name from the assistant repeating it back. Fixed with an exact-match dedup check in `Assistant._extract_and_store()`.
2. **Synthesized facts**: a still-subtler case — the extractor combined two already-known facts ("works at Afiniti" + "learning about RAG") into a new, unstated-but-true fact ("User works on RAG systems at Afiniti"), even after three rounds of explicit prompt instructions telling it not to. The fix that actually worked was structural, not prompt engineering: `FactExtractor.extract_facts()` no longer shows the assistant's response to the LLM at all, so it structurally cannot pull a "fact" from text the user never wrote. `assistant_response` stays a parameter (for interface symmetry with `extract_preferences()`) but is deliberately unused in the fact-extraction prompt.

## Architecture

```
                          ┌─────────────────────┐
                          │   Assistant.chat()   │
                          └──────────┬───────────┘
                                     │
        ┌────────────────┬──────────┼──────────┬────────────────┐
        │                │                      │                │
        ▼                ▼                      ▼                ▼
 WorkingMemory      EpisodicMemory         ProceduralMemory   FactExtractor
 (deque, RAM)      (ChromaDB, semantic)     (SQLite, exact)   (LLM, post-turn)
        │                │                      │                │
        │ last N msgs    │ search(query)        │ list_preferences()
        │                │ → relevant facts     │ → all preferences
        ▼                ▼                      ▼
 ┌─────────────────────────────────────────────────────────┐
 │  prompts/assistant.txt                                    │
 │  # User Preferences: {procedural_preferences}              │
 │  # Relevant facts:    {episodic_facts}                      │
 │  # Current conversation: {working_memory_messages}           │
 │  # Latest user message:  {user_message}                       │
 └─────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                               LLM response
                                     │
                    (after responding, in the background)
                                     ▼
                  FactExtractor.extract_facts() / extract_preferences()
                       │                              │
                       ▼                              ▼
                EpisodicMemory.add_fact()   ProceduralMemory.set_preference()
```

Working memory feeds the prompt directly every turn. Episodic memory is searched fresh each turn and only the relevant subset is injected. Procedural memory is small enough to inject in full, every time. After the response is generated, the same turn is run back through the extractors to decide what (if anything) should be written into episodic or procedural memory for next time.

## What This Deliberately Does NOT Include

- **Any agent framework** (LangChain, LangGraph, CrewAI, Letta, Zep, mem0) — the three memory types are built from scratch specifically so the pattern is visible, not hidden behind someone else's abstraction.
- **A web UI or REST API** — this is a CLI tool; the memory architecture is the point, not the interface.
- **Multi-user support** — `session_id` tags which facts came from which conversation; it is not authentication, and all sessions share the same episodic/procedural stores.
- **Memory consolidation** — related facts are never automatically merged or summarized into fewer entries. Over a long enough history, episodic memory will accumulate some redundancy; deliberately deciding when and how to consolidate related facts is a real problem, and left for a follow-up rather than solved here.
- **Automatic forgetting policies** — nothing decays or expires on its own. The only way to remove a memory is explicit: `/memory forget`, `episodic.forget()`, or `procedural.forget_preference()`.
- **Reflection or meta-memory** — the assistant doesn't reason about its own memory quality, re-evaluate old facts, or maintain memory about its memory.

Three memory types, built cleanly and kept genuinely distinct, are enough to build something that feels personalized — that's the whole thesis of this project. Adding more concepts on top would blur exactly the distinction it's trying to demonstrate.
