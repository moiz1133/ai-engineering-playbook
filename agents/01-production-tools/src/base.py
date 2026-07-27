"""BaseTool: the shared abstraction every tool implements -- Pydantic-validated input/output, async execution,
and schema generation for OpenAI and Anthropic tool-calling. Get this interface right and every tool below is
just implementation detail behind it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Type

from pydantic import BaseModel


class BaseTool(ABC):
    """Base class every tool inherits from. Subclasses set name/description/input_schema/output_schema class
    attributes and implement execute(). Input validation happens automatically via Pydantic before execute()
    ever runs -- invalid inputs raise pydantic.ValidationError from run(), never reach tool logic.
    """

    name: str
    description: str
    input_schema: Type[BaseModel]
    output_schema: Type[BaseModel]

    @abstractmethod
    async def execute(self, inputs: BaseModel) -> BaseModel:
        """Run the tool against already-validated `inputs` and return a validated output model.

        Example:
            result = await tool.execute(tool.input_schema(query="HNSW algorithm"))
        """
        raise NotImplementedError

    async def run(self, **kwargs: Any) -> BaseModel:
        """Validate raw keyword arguments against input_schema, then execute(). The entry point most callers use.

        Example:
            result = await tool.run(query="HNSW algorithm", max_results=3)
        """
        validated = self.input_schema(**kwargs)
        return await self.execute(validated)

    @staticmethod
    def _clean_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
        """Strip Pydantic-generated 'title' keys, which just add noise to an LLM-facing tool schema."""
        schema = dict(schema)
        schema.pop("title", None)
        properties = schema.get("properties")
        if properties:
            schema["properties"] = {
                key: {k: v for k, v in value.items() if k != "title"} for key, value in properties.items()
            }
        return schema

    def to_openai_schema(self) -> Dict[str, Any]:
        """Return this tool's schema in OpenAI's function-calling format.

        Example:
            tools = [t.to_openai_schema() for t in all_tools]
            client.chat.completions.create(model="gpt-4o-mini", tools=tools, messages=messages)
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._clean_schema(self.input_schema.model_json_schema()),
            },
        }

    def to_anthropic_schema(self) -> Dict[str, Any]:
        """Return this tool's schema in Anthropic's tool-use format.

        Example:
            tools = [t.to_anthropic_schema() for t in all_tools]
            client.messages.create(model="claude-...", tools=tools, messages=messages)
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self._clean_schema(self.input_schema.model_json_schema()),
        }
