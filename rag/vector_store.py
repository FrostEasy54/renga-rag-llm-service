from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from rag.document_loader import DocumentChunk, DocumentLoader


_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
_COLLECTION_NAME = "construction_norms"


class VectorStore:

    def __init__(self, persist_dir: Path = Path("chroma_db")) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._embedding_fn = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=_EMBEDDING_MODEL
            )
        )
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=self._embedding_fn, # type: ignore
            metadata={"hnsw:space": "cosine"},
        )


    def index_documents(self, knowledge_base_dir: Path) -> int:
        loader = DocumentLoader()
        chunks = loader.load_directory(knowledge_base_dir)

        if not chunks:
            print("[VectorStore] No documents found to index.")
            return 0

        ids, texts, metadatas = self._prepare_batch(chunks)
        batch_size = 100
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self._collection.upsert(
                ids=ids[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end], # type: ignore
            )

        print(f"[VectorStore] Indexed {len(chunks)} chunks from {knowledge_base_dir}")
        return len(chunks)

    def is_empty(self) -> bool:
        """Return True if no documents have been indexed yet."""
        return self._collection.count() == 0


    def query(self, query_text: str, n_results: int = 4) -> list[str]:
        """
        Return the top-n most semantically relevant norm chunks for the query.

        Returns an empty list if the collection is empty, rather than raising.
        """
        if self.is_empty():
            return []

        results = self._collection.query(
            query_texts=[query_text],
            n_results=min(n_results, self._collection.count()),
        )
        return results["documents"][0] if results["documents"] else []

    @staticmethod
    def _prepare_batch(
        chunks: list[DocumentChunk],
    ) -> tuple[list[str], list[str], list[dict]]:
        """Convert chunks into the parallel lists ChromaDB's upsert expects."""
        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict] = []

        for chunk in chunks:
            chunk_id = f"{chunk.source}__chunk_{chunk.chunk_index}"
            ids.append(chunk_id)
            texts.append(chunk.text)
            metadatas.append(
                {
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                    **chunk.metadata,
                }
            )

        return ids, texts, metadatas