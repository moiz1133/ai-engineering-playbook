"""The Assembler: synthesizes all worker outputs into one coherent final markdown result."""

from __future__ import annotations

import logging
from pathlib import Path

from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import ASSEMBLER_MODEL, MAX_RETRIES, OPENAI_API_KEY
from src.schemas import WorkerOutput, WorkPlan

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "assembler.txt"
_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


def _format_worker_outputs(worker_outputs: list[WorkerOutput]) -> str:
    """Renders every worker's output, labeled, with status and sources, for the assembler prompt."""
    sections = []
    for output in worker_outputs:
        label = output.worker_type.value.capitalize()
        if output.status == "failed":
            sections.append(f"### {label} Output -- STATUS: FAILED\n(error: {output.error})")
            continue
        sources_block = ""
        if output.sources:
            sources_block = "\nSources:\n" + "\n".join(f"- {s}" for s in output.sources)
        sections.append(
            f"### {label} Output -- STATUS: {output.status.upper()} (confidence: {output.confidence:.2f})\n"
            f"{output.content}{sources_block}"
        )
    return "\n\n".join(sections)


class Assembler:
    """Synthesizes every completed worker's output into a single coherent final document."""

    @retry(
        reraise=True,
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    async def _call_llm(self, prompt: str) -> str:
        response = await _client.chat.completions.create(
            model=ASSEMBLER_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise editor who synthesizes multiple sources into one coherent document."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content or ""

    async def assemble(self, original_task: str, plan: WorkPlan, worker_outputs: list[WorkerOutput]) -> str:
        """Synthesize all worker outputs into a single, coherent final result."""
        successful = [o for o in worker_outputs if o.status != "failed"]
        if not successful:
            logger.warning("All workers failed for plan_id=%s; assembling a failure notice instead.", plan.plan_id)
            return (
                f"# {original_task}\n\n"
                "## Executive Summary\n"
                "Every worker assigned to this task failed to produce output. No result could be assembled.\n"
            )

        prompt = _PROMPT_PATH.read_text(encoding="utf-8").format(
            original_task=original_task,
            supervisor_reasoning=plan.supervisor_reasoning,
            worker_outputs=_format_worker_outputs(worker_outputs),
        )
        final_content = await self._call_llm(prompt)
        logger.info("Assembled final output for plan_id=%s (%d chars)", plan.plan_id, len(final_content))
        return final_content
