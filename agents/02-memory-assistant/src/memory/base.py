"""BaseMemory: the minimal shared interface across working, episodic, and procedural memory.

Deliberately thin -- the three memory types are genuinely different in
lifetime, storage, and purpose, and shouldn't be forced into a common shape
beyond "you can clear it and count what's in it."
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseMemory(ABC):
    """Shared interface implemented by WorkingMemory, EpisodicMemory, and ProceduralMemory."""

    @abstractmethod
    def clear(self) -> None:
        """Wipe all stored memories."""
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored memories."""
        raise NotImplementedError
