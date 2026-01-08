from typing import List, Dict

from transformers import pipeline


class AnswerExtractor:
    """
    Extracts answers from text using an extractive QA model
    """

    def __init__(self, model_name: str = "distilbert-base-cased-distilled-squad"):
        self.qa_pipeline = pipeline(
            "question-answering",
            model=model_name,
            tokenizer=model_name
        )

    def extract_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict]
    ) -> Dict:
        """
        Runs QA model on retrieved chunks and returns best answer

        Args:
            question: user question
            retrieved_chunks: list of chunk dicts with text

        Returns:
            {
                "answer": str,
                "score": float,
                "doc_id": str
            }
        """

        best_answer = {
            "answer": "",
            "score": 0.0,
            "doc_id": None
        }

        for chunk in retrieved_chunks:
            result = self.qa_pipeline(
                question=question,
                context=chunk["text"]
            )

            if result["score"] > best_answer["score"]:
                best_answer = {
                    "answer": result["answer"],
                    "score": float(result["score"]),
                    "doc_id": chunk["doc_id"]
                }

        return best_answer

if __name__ == "__main__":
    from src.ingestion.pdf_loader import PDFLoader
    from src.chunking.text_chunker import TextChunker
    from src.embeddings.embedder import TextEmbedder
    from src.retrieval.retriever import SemanticRetriever

    loader = PDFLoader()
    docs = loader.load_pdfs(["data/uploads/sample.pdf"])

    chunker = TextChunker()
    chunks = chunker.chunk_documents(docs)

    embedder = TextEmbedder()
    embed_result = embedder.embed_chunks(chunks)

    question = "What dataset was used?"

    q_emb = embedder.embed_question(question)

    retriever = SemanticRetriever(top_k=3)
    top_chunks = retriever.retrieve(
        q_emb,
        embed_result["embeddings"],
        embed_result["metadata"]
    )

    extractor = AnswerExtractor()
    answer = extractor.extract_answer(question, top_chunks)

    print("\nFinal Answer:")
    print(answer)
