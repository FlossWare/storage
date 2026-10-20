"""Dependency-free in-memory storage implementation."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .types import ChunkRecord, DocumentRecord, EmbeddingRecord


class InMemoryStorage:
    """Reference implementation of document, chunk, and embedding storage."""

    def __init__(self) -> None:
        self.documents: dict[str, DocumentRecord] = {}
        self.chunks: dict[str, ChunkRecord] = {}
        self.embeddings: dict[str, EmbeddingRecord] = {}

    async def put_document(self, document: DocumentRecord) -> None:
        self.documents[document.id] = document

    async def get_document(self, document_id: str) -> DocumentRecord | None:
        return self.documents.get(document_id)

    async def delete_document(self, document_id: str) -> bool:
        document = self.documents.pop(document_id, None)
        if document is None:
            return False
        for chunk_id in list(document.chunk_ids):
            self.chunks.pop(chunk_id, None)
            self.embeddings.pop(chunk_id, None)
        return True

    async def put_chunk(self, chunk: ChunkRecord) -> None:
        self.chunks[chunk.id] = chunk

    async def get_chunk(self, chunk_id: str) -> ChunkRecord | None:
        return self.chunks.get(chunk_id)

    async def get_chunks_for_document(self, document_id: str) -> list[ChunkRecord]:
        return sorted(
            (c for c in self.chunks.values() if c.document_id == document_id),
            key=lambda c: (c.sequence, c.id),
        )

    async def delete_chunk(self, chunk_id: str) -> bool:
        existed = self.chunks.pop(chunk_id, None) is not None
        self.embeddings.pop(chunk_id, None)
        return existed

    async def put_embedding(self, embedding: EmbeddingRecord) -> None:
        _validate_vector(embedding.vector)
        self.embeddings[embedding.chunk_id] = embedding

    async def get_embedding(self, chunk_id: str) -> EmbeddingRecord | None:
        return self.embeddings.get(chunk_id)

    async def delete_embedding(self, chunk_id: str) -> bool:
        return self.embeddings.pop(chunk_id, None) is not None

    async def search(
        self, vector: Sequence[float], limit: int = 10
    ) -> list[tuple[str, float]]:
        _validate_vector(vector)
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a positive integer")
        for embedding in self.embeddings.values():
            if len(embedding.vector) != len(vector):
                raise ValueError(
                    "query vector dimensions must match stored embedding dimensions"
                )
        scored = [
            (e.chunk_id, _cosine(vector, e.vector)) for e in self.embeddings.values()
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:limit]


def _validate_vector(vector: Sequence[float]) -> None:
    if not vector:
        raise ValueError("embedding vectors must not be empty")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in vector
    ):
        raise ValueError("embedding vectors must contain finite numeric values")
    if not any(value != 0 for value in vector):
        raise ValueError("cosine similarity requires a non-zero vector")


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)
