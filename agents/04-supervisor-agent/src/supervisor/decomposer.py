"""The LLM call that turns a raw task string into a validated WorkPlan."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import MAX_RETRIES, OPENAI_API_KEY, SUPERVISOR_MODEL
from src.schemas import SubTask, WorkPlan

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "decomposer.txt"
_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


class Decomposer:
    """Analyzes a task and decides which workers to use, in what role, with what instructions."""

    @retry(
        reraise=True,
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    async def _call_llm(self, task: str) -> dict:
        prompt = _PROMPT_PATH.read_text(encoding="utf-8").format(task=task)
        response = await _client.chat.completions.create(
            model=SUPERVISOR_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)

    async def decompose(self, task: str) -> WorkPlan:
        """Analyze the task and decide which workers to use, in what role, with what
        specific instructions. Assigns real UUIDs for plan_id and every sub_task's
        task_id, remapping any depends_on references the LLM produced accordingly."""
        raw = await self._call_llm(task)

        raw_sub_tasks = raw.get("sub_tasks", [])
        id_map: dict[str, str] = {}
        for raw_task in raw_sub_tasks:
            old_id = str(raw_task.get("task_id", ""))
            id_map[old_id] = str(uuid.uuid4())

        sub_tasks = []
        for raw_task in raw_sub_tasks:
            old_id = str(raw_task.get("task_id", ""))
            old_depends_on = raw_task.get("depends_on", []) or []
            sub_tasks.append(
                SubTask(
                    task_id=id_map[old_id],
                    worker_type=raw_task["worker_type"],
                    instruction=raw_task["instruction"],
                    context=raw_task.get("context", ""),
                    priority=raw_task.get("priority", 1),
                    depends_on=[id_map[d] for d in old_depends_on if d in id_map],
                )
            )

        plan = WorkPlan(
            original_task=raw.get("original_task", task),
            plan_id=str(uuid.uuid4()),
            sub_tasks=sub_tasks,
            supervisor_reasoning=raw.get("supervisor_reasoning", ""),
            estimated_complexity=raw.get("estimated_complexity", "moderate"),
        )
        logger.info(
            "Decomposed task into %d sub-tasks (complexity=%s): %s",
            len(plan.sub_tasks), plan.estimated_complexity,
            [st.worker_type.value for st in plan.sub_tasks],
        )
        return plan
