"""Async generator that streams response tokens from an OpenAI chat completion call."""

from __future__ import annotations

from typing import AsyncIterator

from openai import AsyncOpenAI

from src.config import GENERATION_MODEL


async def stream_answer(client: AsyncOpenAI, prompt: str) -> AsyncIterator[str]:
    """Yield response tokens as they arrive from the streaming chat completion.

    Failures are yielded as a plain-text error message rather than raised,
    since the HTTP response has already started streaming by the time a
    generation error could occur -- there is no status code left to change.
    """
    try:
        stream = await client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as e:
        yield f"\n[ERROR] Generation failed: {e}\n"
