"""Episodic memory: durable, natural-language facts about the user, retrieved by semantic search.

Not every message becomes a fact -- FactExtractor decides what's worth storing
here. What IS stored persists across all future sessions until explicitly
forgotten, and is retrieved by meaning rather than exact match, which is why
this is backed by embeddings/ChromaDB while procedural memory (structured
preferences) is backed by SQLite instead.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import List

import chromadb
from openai import OpenAI

from src.config import EMBEDDING_MODEL, EPISODIC_MEMORY_COLLECTION, EPISODIC_MEMORY_DIR, EPISODIC_TOP_K
from src.memory.base import BaseMemory
from src.schemas import EpisodicFact

logger = logging.getLogger(__name__)


class EpisodicMemory(BaseMemory):
    """ChromaDB-backed store of durable user facts, retrieved by semantic similarity to the current query."""

    def __init__(
        self,
        persist_dir: str = EPISODIC_MEMORY_DIR,
        collection_name: str = EPISODIC_MEMORY_COLLECTION,
    ) -> None:
        self._client = OpenAI()
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        self._collection = chroma_client.get_or_create_collection(name=collection_name)

    def _embed(self, text: str) -> List[float]:
        response = self._client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
        return response.data[0].embedding

    def add_fact(self, content: str, source: str, session_id: str) -> EpisodicFact:
        """Embed and store a new durable fact about the user.

        Example:
            episodic.add_fact(
                "User's daughter Zara turned 4",
                source="extracted from conversation",
                session_id="abdul_moiz",
            )
        """
        now = datetime.now(timezone.utc)
        fact_id = str(uuid.uuid4())
        embedding = self._embed(content)

        self._collection.add(
            ids=[fact_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[
                {
                    "source": source,
                    "session_id": session_id,
                    "created_at": now.isoformat(),
                    "last_accessed": now.isoformat(),
                    "access_count": 0,
                }
            ],
        )
        logger.info("episodic_memory: added fact_id=%s session_id=%s", fact_id, session_id)
        return EpisodicFact(
            fact_id=fact_id,
            content=content,
            source=source,
            session_id=session_id,
            created_at=now,
            last_accessed=now,
            access_count=0,
        )

    def search(self, query: str, top_k: int = EPISODIC_TOP_K) -> List[EpisodicFact]:
        """Semantic search for facts relevant to query. Updates last_accessed/access_count on every match.

        Example:
            facts = episodic.search("What does the user do for work?")
        """
        if self.count() == 0:
            return []

        embedding = self._embed(query)
        n_results = min(top_k, self.count())
        results = self._collection.query(query_embeddings=[embedding], n_results=n_results)

        facts: List[EpisodicFact] = []
        for fact_id, content, metadata in zip(results["ids"][0], results["documents"][0], results["metadatas"][0]):
            new_access_count = metadata.get("access_count", 0) + 1
            now = datetime.now(timezone.utc)
            updated_metadata = {**metadata, "last_accessed": now.isoformat(), "access_count": new_access_count}
            self._collection.update(ids=[fact_id], metadatas=[updated_metadata])

            facts.append(
                EpisodicFact(
                    fact_id=fact_id,
                    content=content,
                    source=metadata["source"],
                    session_id=metadata["session_id"],
                    created_at=datetime.fromisoformat(metadata["created_at"]),
                    last_accessed=now,
                    access_count=new_access_count,
                )
            )
        return facts

    def forget(self, fact_id: str) -> None:
        """Explicitly delete a fact by ID.

        Example:
            episodic.forget(fact.fact_id)
        """
        self._collection.delete(ids=[fact_id])
        logger.info("episodic_memory: forgot fact_id=%s", fact_id)

    def list_all(self) -> List[EpisodicFact]:
        """Return every stored fact, for debugging and demos.

        Example:
            for f in episodic.list_all():
                print(f.content)
        """
        if self.count() == 0:
            return []
        results = self._collection.get()
        return [
            EpisodicFact(
                fact_id=fact_id,
                content=content,
                source=metadata["source"],
                session_id=metadata["session_id"],
                created_at=datetime.fromisoformat(metadata["created_at"]),
                last_accessed=datetime.fromisoformat(metadata["last_accessed"]),
                access_count=metadata.get("access_count", 0),
            )
            for fact_id, content, metadata in zip(results["ids"], results["documents"], results["metadatas"])
        ]

    def clear(self) -> None:
        """Wipe all stored facts."""
        all_ids = self._collection.get()["ids"]
        if all_ids:
            self._collection.delete(ids=all_ids)

    def count(self) -> int:
        """Return the number of stored facts."""
        return self._collection.count()
