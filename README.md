# storage

Storage capability for FlossWare documents, chunks, and embeddings.

This repository owns persistence contracts and a dependency-free in-memory reference implementation. Database-specific adapters belong behind these contracts and should not leak into callers.
