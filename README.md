# storage

Reusable persistence capability for FlossWare documents, chunks, embeddings, evidence, and other typed records.

This repository owns storage contracts and replaceable persistence implementations. Database-specific adapters must remain behind those contracts and must not leak into callers.

## Architectural boundary

Storage answers **where and how durable state is persisted**. It does not own retrieval ranking, Knowledge semantics, RAG composition, model invocation, or Loom orchestration.

```text
application / Loom capability
          |
          v
       storage contract
          |
          +--> memory
          +--> filesystem
          +--> database
          +--> vector / graph adapters
```

`knowledge` defines durable Knowledge semantics; `storage` provides persistence mechanisms that Knowledge and other capabilities may use.

The initial reference implementation is dependency-free and in-memory for tests and lightweight deployments.
