import asyncio

from storage import ChunkRecord, DocumentRecord, EmbeddingRecord, InMemoryStorage


def test_document_chunk_and_embedding_lifecycle():
    async def run():
        store = InMemoryStorage()
        doc = DocumentRecord(id="d1", content="hello", chunk_ids=["c1"])
        await store.put_document(doc)
        await store.put_chunk(ChunkRecord(id="c1", document_id="d1", content="hello"))
        await store.put_embedding(
            EmbeddingRecord(id="e1", chunk_id="c1", vector=[1.0, 0.0])
        )
        assert await store.get_document("d1") == doc
        assert (await store.get_chunks_for_document("d1"))[0].id == "c1"
        assert (await store.search([1.0, 0.0]))[0] == ("c1", 1.0)
        assert await store.delete_document("d1")
        assert await store.get_chunk("c1") is None
        assert await store.get_embedding("c1") is None

    asyncio.run(run())



def test_embedding_validation_and_search_contract():
    import pytest

    async def run():
        store = InMemoryStorage()
        with pytest.raises(ValueError, match="must not be empty"):
            await store.put_embedding(
                EmbeddingRecord(id="empty", chunk_id="empty", vector=[])
            )
        with pytest.raises(ValueError, match="finite numeric"):
            await store.put_embedding(
                EmbeddingRecord(id="nan", chunk_id="nan", vector=[float("nan")])
            )
        with pytest.raises(ValueError, match="non-zero"):
            await store.put_embedding(
                EmbeddingRecord(id="zero", chunk_id="zero", vector=[0.0, 0.0])
            )

        await store.put_embedding(
            EmbeddingRecord(id="e1", chunk_id="c1", vector=[1.0, 0.0])
        )
        with pytest.raises(ValueError, match="dimensions"):
            await store.search([1.0])
        with pytest.raises(ValueError, match="limit"):
            await store.search([1.0, 0.0], limit=0)
        with pytest.raises(ValueError, match="finite numeric"):
            await store.search([float("inf"), 0.0])

    asyncio.run(run())
