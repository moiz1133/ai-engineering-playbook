"""Working memory: the current conversation's context. In-memory only, reset every session."""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, List

from src.config import WORKING_MEMORY_MAX_MESSAGES
from src.memory.base import BaseMemory
from src.schemas import Message

logger = logging.getLogger(__name__)


class WorkingMemory(BaseMemory):
    """A bounded sliding window over the current session's messages. Deliberately ephemeral -- never persisted.

    If information matters beyond this session, it must be extracted into
    episodic or procedural memory before the session ends -- working memory
    itself carries nothing forward.
    """

    def __init__(self, max_messages: int = WORKING_MEMORY_MAX_MESSAGES) -> None:
        self.max_messages = max_messages
        self._messages: Deque[Message] = deque(maxlen=max_messages)

    def add(self, role: str, content: str) -> None:
        """Append a new message, silently dropping the oldest if the window is already full.

        Example:
            working.add("user", "My daughter Zara just turned 4.")
        """
        message = Message(role=role, content=content)
        self._messages.append(message)
        logger.debug("working_memory: added role=%s size=%d/%d", role, len(self._messages), self.max_messages)

    def get_context(self) -> List[Message]:
        """Return the current window of messages, oldest first.

        Example:
            for m in working.get_context():
                print(m.role, m.content)
        """
        return list(self._messages)

    def clear(self) -> None:
        """Reset working memory to empty."""
        self._messages.clear()

    def count(self) -> int:
        """Return the number of messages currently held in the window."""
        return len(self._messages)
