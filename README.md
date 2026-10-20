# storage

Reusable persistence capability for FlossWare documents, chunks, embeddings, provenance, and other typed records.

This repository owns storage contracts and replaceable persistence implementations. Database-specific adapters remain behind those contracts and do not leak into callers.

## Architectural boundary

Storage answers **where and how durable state is persisted**. It does not own retrieval ranking, Knowledge semantics, RAG composition, model invocation, or Loom orchestration.

```text
application / Loom capability
          |
          v
       storage contract
          |
          +--> in-memory reference implementation
          +--> PostgreSQL + pgvector document/chunk/embedding adapter
          +--> OrientDB provenance adapter
```

## Backends

### In-memory reference

`InMemoryStorage` is dependency-free and intended for tests and lightweight deployments.

### PostgreSQL + pgvector

Install the optional driver dependencies:

```sh
pip install -e ".[postgres]"
```

```python
from storage import PostgresStorage

store = await PostgresStorage.connect(
    dsn="postgresql://user:password@host:5433/database",
    embedding_dimensions=1536,  # must match the embedding model
)
await store.initialize()
```

The adapter creates its own `loom_storage` schema with document and chunk tables, plus a dimension-specific embedding table (for example, `embeddings_1536`) and HNSW cosine index. It does not alter or assume the layout of existing corpus tables. Different embedding dimensions can use different embedding tables.

### OrientDB provenance

OrientDB stores provenance as native graph edges between `LoomEntity` vertices, independently of document text and embeddings:

```python
from storage import OrientDBProvenanceStore, ProvenanceRecord

graph = OrientDBProvenanceStore(
    base_url="http://host:2480",
    database="database",
    username="configured-user",
    password="configured-password",
)
await graph.initialize()
await graph.put_provenance(
    ProvenanceRecord(
        id="chunk-42-derived-from-document-7",
        subject_id="chunk-42",
        relation="derived_from",
        object_id="document-7",
        metadata={"start_offset": 120, "end_offset": 260},
    )
)
```

The adapter uses OrientDB's parameterized REST SQL command endpoint. Credentials are supplied at runtime and are never stored in provenance records.

## Live integration tests

The normal test suite uses a fake connection pool and mocked HTTP responses. Live tests run only when their environment variables are configured.

PostgreSQL/pgvector:

- `FLOSSWARE_STORAGE_POSTGRES_DSN`
- `FLOSSWARE_STORAGE_EMBEDDING_DIMENSIONS`

OrientDB:

- `FLOSSWARE_STORAGE_ORIENTDB_URL`
- `FLOSSWARE_STORAGE_ORIENTDB_DATABASE`
- `FLOSSWARE_STORAGE_ORIENTDB_USERNAME`
- `FLOSSWARE_STORAGE_ORIENTDB_PASSWORD`

Run unit and contract tests with `pip install -e ".[postgres,test]" && pytest -q`. After setting all six variables, run `bash scripts/test-live-backends.sh` to exercise both live services. Without those variables the live tests are skipped; passing unit tests alone does not certify a live database deployment.

## Contracts

- `DocumentStore`, `ChunkStore`, and `EmbeddingStore` describe persistence of canonical records.
- `ProvenanceStore` persists typed subject/relation/object records.
- `DocumentRecord`, `ChunkRecord`, `EmbeddingRecord`, and `ProvenanceRecord` are the shared data contracts.

Knowledge defines domain semantics and provenance; storage provides persistence mechanisms. RAG and Loom implementations should depend on these interfaces rather than directly on database clients.
