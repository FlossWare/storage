import asyncio
import json

import pytest

from storage import (
    ChunkRecord,
    DocumentRecord,
    EmbeddingRecord,
    OrientDBProvenanceStore,
    PostgresStorage,
    ProvenanceRecord,
)


class FakeCursor:
    rowcount = 1

    async def fetchone(self):
        return None

    async def fetchall(self):
        return []


class FakeConnection:
    def __init__(self):
        self.statements = []

    async def execute(self, statement, parameters=None):
        self.statements.append((statement, parameters))
        return FakeCursor()


class FakePool:
    def __init__(self):
        self.connection_instance = FakeConnection()
        self.closed = False

    class _ConnectionContext:
        def __init__(self, connection):
            self.connection = connection

        async def __aenter__(self):
            return self.connection

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    def connection(self):
        return self._ConnectionContext(self.connection_instance)

    async def close(self):
        self.closed = True


def test_postgres_initializes_namespaced_pgvector_tables_and_index():
    async def run():
        pool = FakePool()
        store = PostgresStorage(pool, embedding_dimensions=3)
        await store.initialize()

        sql = "\n".join(
            statement for statement, _ in pool.connection_instance.statements
        )
        assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
        assert "CREATE SCHEMA IF NOT EXISTS loom_storage" in sql
        assert "loom_storage.documents" in sql
        assert "loom_storage.chunks" in sql
        assert "loom_storage.embeddings_3" in sql
        assert "vector(3)" in sql
        assert "USING hnsw (vector vector_cosine_ops)" in sql

    asyncio.run(run())


def test_postgres_rejects_invalid_dimensions_identifiers_and_vectors():
    pool = FakePool()
    with pytest.raises(ValueError, match="embedding_dimensions"):
        PostgresStorage(pool, embedding_dimensions=0)
    with pytest.raises(ValueError, match="schema"):
        PostgresStorage(pool, embedding_dimensions=3, schema="loom; DROP SCHEMA public")
    store = PostgresStorage(pool, embedding_dimensions=3)
    with pytest.raises(ValueError, match="dimensions"):
        store._validate_vector([1.0, 0.0])
    with pytest.raises(ValueError, match="finite numeric"):
        store._validate_vector([1.0, float("nan"), 0.0])


def test_orientdb_uses_parameterized_sql_and_basic_auth(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return b'{"result": []}'

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("storage.orientdb.urlopen", fake_urlopen)
    store = OrientDBProvenanceStore(
        "http://localhost:2480/", "loom", "user", "password"
    )
    record = ProvenanceRecord(
        id="p-1",
        subject_id="subject' OR 1=1",
        relation="derived_from",
        object_id="source-1",
        metadata={"source": "test"},
    )

    asyncio.run(store.put_provenance(record))

    request = captured["request"]
    payload = json.loads(request.data)
    assert request.full_url == "http://localhost:2480/command/loom/sql"
    assert request.get_header("Authorization").startswith("Basic ")
    assert ":subjectId" in payload["command"]
    assert "subject' OR 1=1" not in payload["command"]
    assert payload["parameters"]["subjectId"] == "subject' OR 1=1"
    assert payload["parameters"]["recordId"] == "p-1"


def test_orientdb_reads_provenance_records(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(
                {
                    "result": [
                        {
                            "recordId": "p-2",
                            "subjectId": "chunk-1",
                            "relation": "derived_from",
                            "objectId": "document-1",
                            "metadata": {"offset": 12},
                            "createdAt": "2026-10-19T12:00:00+00:00",
                        }
                    ]
                }
            ).encode()

    monkeypatch.setattr(
        "storage.orientdb.urlopen", lambda request, timeout: FakeResponse()
    )
    store = OrientDBProvenanceStore("http://localhost:2480", "loom", "user", "password")
    record = asyncio.run(store.get_provenance("p-2"))
    assert record == ProvenanceRecord(
        id="p-2",
        subject_id="chunk-1",
        relation="derived_from",
        object_id="document-1",
        metadata={"offset": 12},
        created_at="2026-10-19T12:00:00+00:00",
    )


def test_live_postgres_pgvector_round_trip_when_configured():
    import os
    import uuid

    dsn = os.getenv("FLOSSWARE_STORAGE_POSTGRES_DSN")
    dimensions = os.getenv("FLOSSWARE_STORAGE_EMBEDDING_DIMENSIONS")
    if not dsn or not dimensions:
        pytest.skip("set FLOSSWARE_STORAGE_POSTGRES_DSN and embedding dimensions")

    async def run():
        store = await PostgresStorage.connect(dsn, embedding_dimensions=int(dimensions))
        suffix = uuid.uuid4().hex
        document_id = f"loom-doc-{suffix}"
        chunk_id = f"loom-chunk-{suffix}"
        try:
            await store.initialize()
            document = DocumentRecord(
                id=document_id,
                content="Loom storage integration test",
                chunk_ids=[chunk_id],
                metadata={"test": True},
                provenance={"source": "integration-test"},
                content_hash=suffix,
            )
            chunk = ChunkRecord(
                id=chunk_id,
                document_id=document_id,
                content="Loom storage integration test",
                provenance={"document_id": document_id},
            )
            vector = [0.0] * (int(dimensions) - 1) + [1.0]
            embedding = EmbeddingRecord(
                id=f"loom-embedding-{suffix}",
                chunk_id=chunk_id,
                vector=vector,
                metadata={"model": "integration-test"},
            )
            await store.put_document(document)
            await store.put_chunk(chunk)
            await store.put_embedding(embedding)
            assert await store.get_document(document_id) == document
            assert await store.get_chunk(chunk_id) == chunk
            assert await store.get_embedding(chunk_id) == embedding
            assert (await store.search(vector, limit=1))[0][0] == chunk_id
        finally:
            await store.delete_document(document_id)
            await store.close()

    asyncio.run(run())


def test_live_orientdb_provenance_round_trip_when_configured():
    import os
    import uuid

    base_url = os.getenv("FLOSSWARE_STORAGE_ORIENTDB_URL")
    database = os.getenv("FLOSSWARE_STORAGE_ORIENTDB_DATABASE")
    username = os.getenv("FLOSSWARE_STORAGE_ORIENTDB_USERNAME")
    password = os.getenv("FLOSSWARE_STORAGE_ORIENTDB_PASSWORD")
    if not all((base_url, database, username, password)):
        pytest.skip(
            "set FLOSSWARE_STORAGE_ORIENTDB_URL, _DATABASE, _USERNAME and _PASSWORD"
        )

    async def run():
        store = OrientDBProvenanceStore(base_url, database, username, password)
        suffix = uuid.uuid4().hex
        record = ProvenanceRecord(
            id=f"loom-provenance-{suffix}",
            subject_id=f"loom-chunk-{suffix}",
            relation="derived_from",
            object_id=f"loom-document-{suffix}",
            metadata={"test": True},
        )
        try:
            await store.initialize()
            await store.put_provenance(record)
            recovered = await store.get_provenance(record.id)
            assert recovered is not None
            assert recovered.id == record.id
            assert recovered.subject_id == record.subject_id
            assert recovered.relation == record.relation
            assert recovered.object_id == record.object_id
            assert recovered.metadata == record.metadata
            assert await store.get_provenance_for_subject(record.subject_id) == [
                recovered
            ]
        finally:
            await store.delete_provenance(record.id)

    asyncio.run(run())


def test_orientdb_initializes_native_graph_classes_and_indexes(monkeypatch):
    commands = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return b'{"result": 1}'

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data)
        commands.append(payload["command"])
        return FakeResponse()

    monkeypatch.setattr("storage.orientdb.urlopen", fake_urlopen)
    store = OrientDBProvenanceStore("http://localhost:2480", "loom", "user", "password")
    asyncio.run(store.initialize())

    sql = "\\n".join(commands)
    assert "CREATE CLASS LoomEntity IF NOT EXISTS EXTENDS V" in sql
    assert "CREATE CLASS LoomProvenance IF NOT EXISTS EXTENDS E" in sql
    assert "LoomEntity_entityId" in sql
    assert "LoomProvenance_recordId" in sql
