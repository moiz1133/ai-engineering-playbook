"""Pydantic data models shared across the three memory types and the fact/preference extractors."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """One turn in the current conversation, held only in working memory."""

    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EpisodicFact(BaseModel):
    """A durable, natural-language fact about the user, stored in episodic memory (ChromaDB)."""

    fact_id: str
    content: str
    source: str
    session_id: str
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0


class Preference(BaseModel):
    """A structured, key/value behavioral preference, stored in procedural memory (SQLite)."""

    key: str
    value: str
    category: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    created_at: datetime
    updated_at: datetime


class ExtractedFact(BaseModel):
    """One candidate episodic fact identified by FactExtractor.extract_facts(), before the confidence gate."""

    content: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    category: Literal["personal", "professional", "preference", "temporal", "other"]


class ExtractedPreference(BaseModel):
    """One candidate procedural preference identified by FactExtractor.extract_preferences(), before the confidence gate."""

    key: str
    value: str
    category: str
    confidence: float = Field(..., ge=0.0, le=1.0)
