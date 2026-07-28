"""Procedural memory: structured, learned preferences about how the user wants the assistant to behave.

Stored in SQLite as key/value rows, not natural language -- "response_style:
concise" is a preference; "told me on Tuesday they like short answers" is a
fact and belongs in episodic memory instead. Keeping preferences structured
is what lets them be surfaced into the system prompt cheaply on every turn,
without re-reasoning over raw conversation history.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.config import PROCEDURAL_DB_PATH
from src.memory.base import BaseMemory
from src.schemas import Preference

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS preferences (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    category TEXT,
    confidence REAL,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
"""


class ProceduralMemory(BaseMemory):
    """SQLite-backed store of structured user preferences (key -> value, category, confidence)."""

    def __init__(self, db_path: str = PROCEDURAL_DB_PATH) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def set_preference(self, key: str, value: str, category: str, confidence: float) -> None:
        """Insert a new preference or update an existing one with the same key.

        Example:
            procedural.set_preference("response_style", "concise", "communication", 0.9)
        """
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO preferences (key, value, category, confidence, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                category = excluded.category,
                confidence = excluded.confidence,
                updated_at = excluded.updated_at
            """,
            (key, value, category, confidence, now, now),
        )
        self._conn.commit()
        logger.info("procedural_memory: set key=%r category=%r confidence=%.2f", key, category, confidence)

    def get_preference(self, key: str) -> Optional[str]:
        """Return the stored value for key, or None if it isn't set.

        Example:
            style = procedural.get_preference("response_style")
        """
        row = self._conn.execute("SELECT value FROM preferences WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def list_preferences(self, category: Optional[str] = None) -> List[Preference]:
        """Return all preferences, optionally filtered by category, for prompt building.

        Example:
            comms = procedural.list_preferences(category="communication")
        """
        if category is not None:
            rows = self._conn.execute(
                "SELECT * FROM preferences WHERE category = ? ORDER BY key", (category,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM preferences ORDER BY key").fetchall()
        return [
            Preference(
                key=r["key"],
                value=r["value"],
                category=r["category"] or "",
                confidence=r["confidence"],
                created_at=datetime.fromisoformat(r["created_at"]),
                updated_at=datetime.fromisoformat(r["updated_at"]),
            )
            for r in rows
        ]

    def update_confidence(self, key: str, delta: float) -> None:
        """Adjust an existing preference's confidence by delta, clamped to [0.0, 1.0].

        Example:
            procedural.update_confidence("response_style", 0.05)  # user confirmed it again
        """
        row = self._conn.execute("SELECT confidence FROM preferences WHERE key = ?", (key,)).fetchone()
        if row is None:
            raise KeyError(f"No preference stored for key: {key!r}")
        new_confidence = max(0.0, min(1.0, row["confidence"] + delta))
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute("UPDATE preferences SET confidence = ?, updated_at = ? WHERE key = ?", (new_confidence, now, key))
        self._conn.commit()

    def forget_preference(self, key: str) -> None:
        """Explicitly delete a preference.

        Example:
            procedural.forget_preference("response_style")
        """
        self._conn.execute("DELETE FROM preferences WHERE key = ?", (key,))
        self._conn.commit()

    def clear(self) -> None:
        """Wipe all stored preferences."""
        self._conn.execute("DELETE FROM preferences")
        self._conn.commit()

    def count(self) -> int:
        """Return the number of stored preferences."""
        row = self._conn.execute("SELECT COUNT(*) AS n FROM preferences").fetchone()
        return row["n"]

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
