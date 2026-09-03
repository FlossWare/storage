"""Storage contracts and records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


@dataclass
class DocumentRecord:
    id: str
    content: str
    chunk_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    created_at: str = ""


@dataclass
class ChunkRecord:
    id: str
    document_id: str
    content: str
    sequence: int = 0
    token_count: int = 0
    start_offset: int = 0
    end_offset: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingRecord:
    id: str
    chunk_id: str
    vector: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentStore(Protocol):
    async def put_document(self, document: DocumentRecord) -> None: ...
    async def get_document(self, document_id: str) -> DocumentRecord | None: ...
    async def delete_document(self, document_id: str) -> bool: ...


class ChunkStore(Protocol):
    async def put_chunk(self, chunk: ChunkRecord) -> None: ...
    async def get_chunk(self, chunk_id: str) -> ChunkRecord | None: ...
    async def get_chunks_for_document(self, document_id: str) -> list[ChunkRecord]: ...
    async def delete_chunk(self, chunk_id: str) -> bool: ...


class EmbeddingStore(Protocol):
    async def put_embedding(self, embedding: EmbeddingRecord) -> None: ...
    async def get_embedding(self, chunk_id: str) -> EmbeddingRecord | None: ...
    async def delete_embedding(self, chunk_id: str) -> bool: ...
    async def search(self, vector: Sequence[float], limit: int = 10) -> list[tuple[str, float]]: ...
