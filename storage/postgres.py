"""PostgreSQL/pgvector storage adapter.

The adapter owns its small, namespaced storage schema. It does not assume or
rewrite tables belonging to the existing FlossWare knowledge corpus.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from .types import ChunkRecord, DocumentRecord, EmbeddingRecord


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _identifier(value: str, label: str) -> str:
    if not _IDENTIFIER.fullmatch(value) or len(value) > 45:
        raise ValueError(f"{label} must be a simple SQL identifier (max 45 chars)")
    return value


def _jsonb(value: Any) -> Any:
    """Wrap JSON values for psycopg without making psycopg a base dependency."""
    try:
        from psycopg.types.json import Jsonb
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise RuntimeError(
            "PostgresStorage requires the 'postgres' optional dependencies"
        ) from exc
    return Jsonb(value)


class PostgresStorage:
    """Async document, chunk, and pgvector store backed by PostgreSQL.

    Supply a pgvector dimension explicitly. Different embedding dimensions can
    use different embedding tables while sharing the document and chunk tables.
    """

    def __init__(
        self,
        pool: Any,
        *,
        embedding_dimensions: int,
        schema: str = "loom_storage",
        embedding_table: str | None = None,
    ) -> None:
        if not isinstance(embedding_dimensions, int) or not 1 <= embedding_dimensions <= 2000:
            raise ValueError("embedding_dimensions must be between 1 and 2000")
        self.pool = pool
        self.schema = _identifier(schema, "schema")
        self.embedding_dimensions = embedding_dimensions
        self.embedding_table = _identifier(
            embedding_table or f"embeddings_{embedding_dimensions}",
            "embedding_table",
        )

    @classmethod
    async def connect(
        cls,
        dsn: str,
        *,
        embedding_dimensions: int,
        schema: str = "loom_storage",
        embedding_table: str | None = None,
        min_size: int = 1,
        max_size: int = 10,
    ) -> "PostgresStorage":
        """Create a configured async connection pool using optional drivers."""
        try:
            from pgvector.psycopg import register_vector_async
            from psycopg_pool import AsyncConnectionPool
        except ImportError as exc:  # pragma: no cover - exercised without extra
            raise RuntimeError(
                "PostgresStorage.connect requires storage[postgres]"
            ) from exc

        async def configure(connection: Any) -> None:
            await register_vector_async(connection)

        pool = AsyncConnectionPool(
            conninfo=dsn,
            min_size=min_size,
            max_size=max_size,
            configure=configure,
            open=False,
        )
        await pool.open()
        return cls(
            pool,
            embedding_dimensions=embedding_dimensions,
            schema=schema,
            embedding_table=embedding_table,
        )

    async def initialize(self) -> None:
        """Create the adapter-owned tables and cosine index idempotently."""
        schema = self.schema
        embedding_table = self.embedding_table
        index_name = _identifier(f"idx_{embedding_table}_cosine", "index name")
        async with self.pool.connection() as connection:
            await connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await connection.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {schema}.documents (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    chunk_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    provenance JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    content_hash TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT ''
                )
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {schema}.chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL
                        REFERENCES {schema}.documents(id) ON DELETE CASCADE,
                    content TEXT NOT NULL,
                    sequence INTEGER NOT NULL DEFAULT 0,
                    token_count INTEGER NOT NULL DEFAULT 0,
                    start_offset INTEGER NOT NULL DEFAULT 0,
                    end_offset INTEGER NOT NULL DEFAULT 0,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    provenance JSONB NOT NULL DEFAULT '{{}}'::jsonb
                )
                """
            )
            await connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_chunks_document_sequence
                ON {schema}.chunks (document_id, sequence, id)
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {schema}.{embedding_table} (
                    id TEXT PRIMARY KEY,
                    chunk_id TEXT NOT NULL UNIQUE
                        REFERENCES {schema}.chunks(id) ON DELETE CASCADE,
                    vector vector({self.embedding_dimensions}) NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
                )
                """
            )
            await connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {index_name}
                ON {schema}.{embedding_table}
                USING hnsw (vector vector_cosine_ops)
                """
            )

    async def close(self) -> None:
        await self.pool.close()

    async def put_document(self, document: DocumentRecord) -> None:
        async with self.pool.connection() as connection:
            await connection.execute(
                f"""
                INSERT INTO {self.schema}.documents
                    (id, content, chunk_ids, metadata, provenance, content_hash, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    chunk_ids = EXCLUDED.chunk_ids,
                    metadata = EXCLUDED.metadata,
                    provenance = EXCLUDED.provenance,
                    content_hash = EXCLUDED.content_hash,
                    created_at = EXCLUDED.created_at
                """,
                (
                    document.id,
                    document.content,
                    _jsonb(document.chunk_ids),
                    _jsonb(document.metadata),
                    _jsonb(document.provenance),
                    document.content_hash,
                    document.created_at,
                ),
            )

    async def get_document(self, document_id: str) -> DocumentRecord | None:
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                f"""
                SELECT id, content, chunk_ids, metadata, provenance, content_hash, created_at
                FROM {self.schema}.documents WHERE id = %s
                """,
                (document_id,),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return DocumentRecord(
            id=row[0],
            content=row[1],
            chunk_ids=list(row[2]),
            metadata=dict(row[3]),
            provenance=dict(row[4]),
            content_hash=row[5],
            created_at=row[6],
        )

    async def delete_document(self, document_id: str) -> bool:
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                f"DELETE FROM {self.schema}.documents WHERE id = %s",
                (document_id,),
            )
        return cursor.rowcount > 0

    async def put_chunk(self, chunk: ChunkRecord) -> None:
        async with self.pool.connection() as connection:
            await connection.execute(
                f"""
                INSERT INTO {self.schema}.chunks
                    (id, document_id, content, sequence, token_count, start_offset,
                     end_offset, metadata, provenance)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    document_id = EXCLUDED.document_id,
                    content = EXCLUDED.content,
                    sequence = EXCLUDED.sequence,
                    token_count = EXCLUDED.token_count,
                    start_offset = EXCLUDED.start_offset,
                    end_offset = EXCLUDED.end_offset,
                    metadata = EXCLUDED.metadata,
                    provenance = EXCLUDED.provenance
                """,
                (
                    chunk.id,
                    chunk.document_id,
                    chunk.content,
                    chunk.sequence,
                    chunk.token_count,
                    chunk.start_offset,
                    chunk.end_offset,
                    _jsonb(chunk.metadata),
                    _jsonb(chunk.provenance),
                ),
            )

    async def get_chunk(self, chunk_id: str) -> ChunkRecord | None:
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                f"""
                SELECT id, document_id, content, sequence, token_count, start_offset,
                       end_offset, metadata, provenance
                FROM {self.schema}.chunks WHERE id = %s
                """,
                (chunk_id,),
            )
            row = await cursor.fetchone()
        return _chunk_from_row(row) if row is not None else None

    async def get_chunks_for_document(self, document_id: str) -> list[ChunkRecord]:
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                f"""
                SELECT id, document_id, content, sequence, token_count, start_offset,
                       end_offset, metadata, provenance
                FROM {self.schema}.chunks
                WHERE document_id = %s
                ORDER BY sequence, id
                """,
                (document_id,),
            )
            rows = await cursor.fetchall()
        return [_chunk_from_row(row) for row in rows]

    async def delete_chunk(self, chunk_id: str) -> bool:
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                f"DELETE FROM {self.schema}.chunks WHERE id = %s",
                (chunk_id,),
            )
        return cursor.rowcount > 0

    async def put_embedding(self, embedding: EmbeddingRecord) -> None:
        self._validate_vector(embedding.vector)
        async with self.pool.connection() as connection:
            await connection.execute(
                f"""
                INSERT INTO {self.schema}.{self.embedding_table}
                    (id, chunk_id, vector, metadata)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    id = EXCLUDED.id,
                    vector = EXCLUDED.vector,
                    metadata = EXCLUDED.metadata
                """,
                (
                    embedding.id,
                    embedding.chunk_id,
                    embedding.vector,
                    _jsonb(embedding.metadata),
                ),
            )

    async def get_embedding(self, chunk_id: str) -> EmbeddingRecord | None:
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                f"""
                SELECT id, chunk_id, vector, metadata
                FROM {self.schema}.{self.embedding_table} WHERE chunk_id = %s
                """,
                (chunk_id,),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return EmbeddingRecord(
            id=row[0],
            chunk_id=row[1],
            vector=[float(value) for value in row[2]],
            metadata=dict(row[3]),
        )

    async def delete_embedding(self, chunk_id: str) -> bool:
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                f"DELETE FROM {self.schema}.{self.embedding_table} WHERE chunk_id = %s",
                (chunk_id,),
            )
        return cursor.rowcount > 0

    async def search(
        self, vector: Sequence[float], limit: int = 10
    ) -> list[tuple[str, float]]:
        self._validate_vector(vector)
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                f"""
                SELECT chunk_id, 1 - (vector <=> %s) AS similarity
                FROM {self.schema}.{self.embedding_table}
                ORDER BY vector <=> %s, chunk_id
                LIMIT %s
                """,
                (list(vector), list(vector), limit),
            )
            rows = await cursor.fetchall()
        return [(row[0], float(row[1])) for row in rows]

    def _validate_vector(self, vector: Sequence[float]) -> None:
        if len(vector) != self.embedding_dimensions:
            raise ValueError(
                f"expected {self.embedding_dimensions} vector dimensions, got {len(vector)}"
            )
        if any(not isinstance(value, (int, float)) for value in vector):
            raise ValueError("embedding vectors must contain numeric values")


def _chunk_from_row(row: tuple[Any, ...]) -> ChunkRecord:
    return ChunkRecord(
        id=row[0],
        document_id=row[1],
        content=row[2],
        sequence=row[3],
        token_count=row[4],
        start_offset=row[5],
        end_offset=row[6],
        metadata=dict(row[7]),
        provenance=dict(row[8]),
    )
