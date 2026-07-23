"""Minimal sanity checks for prompt versioning and the health endpoint. No LLM or ChromaDB calls."""

from fastapi.testclient import TestClient

from src.main import app
from src.prompts.registry import PromptRegistry


def test_registry_loads_both_versions() -> None:
    registry = PromptRegistry()
    assert set(registry.list_versions()) == {"v1", "v2"}


def test_registry_get_returns_different_strings() -> None:
    registry = PromptRegistry()
    assert registry.get("v1") != registry.get("v2")


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert set(body["prompt_versions_loaded"]) == {"v1", "v2"}
