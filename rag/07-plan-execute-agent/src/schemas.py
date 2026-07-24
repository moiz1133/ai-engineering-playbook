"""Pydantic data models passed between the planner, executor, and report phases."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    step_number: int
    sub_question: str
    search_query: str
    rationale: str


class Plan(BaseModel):
    topic: str
    steps: List[PlanStep] = Field(default_factory=list)


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class StepResult(BaseModel):
    step: PlanStep
    search_results: List[SearchResult] = Field(default_factory=list)
    summary: str


class ReportData(BaseModel):
    """Everything needed to render and save the final markdown report."""

    topic: str
    plan: Plan
    step_results: List[StepResult]
    sources: List[str] = Field(default_factory=list)  # deduplicated URLs; index+1 == citation number
    markdown_body: str = ""  # LLM-synthesized title/summary/TOC/sections, no Sources section
    full_markdown: str = ""  # markdown_body + the appended Sources section
    output_path: Optional[str] = None
