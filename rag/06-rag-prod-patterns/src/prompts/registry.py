"""Loads prompt templates from versions/*.txt at startup and serves them by version name."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from src.config import DEFAULT_PROMPT_VERSION

VERSIONS_DIR = Path(__file__).parent / "versions"


class PromptRegistry:
    """Holds every prompt template found in versions/, keyed by filename stem (e.g. "v1")."""

    def __init__(self, versions_dir: Path = VERSIONS_DIR, default_version: str = DEFAULT_PROMPT_VERSION) -> None:
        self.default_version = default_version
        self._templates: Dict[str, str] = {}
        for path in sorted(versions_dir.glob("*.txt")):
            self._templates[path.stem] = path.read_text(encoding="utf-8")

    def get(self, version_name: str) -> str:
        """Return the raw template string for the given version name."""
        if version_name not in self._templates:
            raise KeyError(f"Unknown prompt version: {version_name!r}. Available: {self.list_versions()}")
        return self._templates[version_name]

    def list_versions(self) -> List[str]:
        """Return all loaded version names."""
        return sorted(self._templates.keys())
