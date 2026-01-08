from typing import List, Dict
import numpy as np


class SemanticRetriever:
    """
    Retrieves most relevant text chunks using cosine similarity
    """

    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    def retrieve(
        self,
        question_embedding: np.ndarray,
        chunk_embeddings: np.ndarray,
        metadata: List[Dict]
    ) -> List[Dict]:
        """
        Finds top-k most relevant chunks

        Args:
            question_embedding: shape (embedding_dim,)
            chunk_embeddings: shape (num_chunks, embedding_dim)
            metadata: list of chunk dicts

        Returns:
            List of top-k chunk dicts with similarity score
        """

        # Cosine similarity since embeddings are normalized
        similarities = np.dot(chunk_embeddings, question_embedding)

        # Get top-k indices
        top_k_idx = np.argsort(similarities)[-self.top_k:][::-1]

        results = []

        for idx in top_k_idx:
            chunk_info = metadata[idx].copy()
            chunk_info["score"] = float(similarities[idx])
            results.append(chunk_info)

        return results

if __name__ == "__main__":
    from src.ingestion.pdf_loader import PDFLoader
    from src.chunking.text_chunker import TextChunker
    from src.embeddings.embedder import TextEmbedder

    loader = PDFLoader()
    docs = loader.load_pdfs(["data/uploads/sample.pdf"])

    chunker = TextChunker(chunk_size=300, overlap=50)
    chunks = chunker.chunk_documents(docs)

    embedder = TextEmbedder()
    embed_result = embedder.embed_chunks(chunks)

    question = "What dataset was used in the experiment?"

    q_emb = embedder.embed_question(question)

    retriever = SemanticRetriever(top_k=3)
    top_chunks = retriever.retrieve(
        q_emb,
        embed_result["embeddings"],
        embed_result["metadata"]
    )

    print("\nTop Retrieved Chunks:\n")
    for c in top_chunks:
        print("Doc:", c["doc_id"])
        print("Score:", round(c["score"], 4))
        print(c["text"][:400])
        print("-" * 60)
