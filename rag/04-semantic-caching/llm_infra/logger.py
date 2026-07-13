"""Structured JSONL logging for llm_infra components.

WHAT: every significant event (cache hit/miss, routing decision, circuit
      breaker trip, LLM call completion) is appended as one JSON object per line
WHY: JSONL is trivially greppable and parseable line-by-line, and appending a
     line never requires rewriting the whole file — ideal for high-volume
     structured logs where each event is independent, unlike a single JSON
     array that requires a full read-modify-write on every append

THREAD SAFETY: this class is NOT thread-safe — concurrent appends from multiple
threads can interleave partial writes to the log file. For threaded use, guard
calls to log() with a threading.Lock(); for asyncio, use asyncio.Lock().
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict


class JSONLLogger:
    def __init__(self, log_path: str = "llm_infra_events.log.jsonl"):
        self.log_path = log_path

    def log(self, event: str, **fields: Any) -> Dict:
        """Append one structured JSON line and return the record that was written."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return record
