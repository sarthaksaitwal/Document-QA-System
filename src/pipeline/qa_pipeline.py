from typing import List, Dict

from src.ingestion.pdf_loader import PDFLoader
from src.chunking.text_chunker import TextChunker
from src.embeddings.embedder import TextEmbedder
from src.retrieval.retriever import SemanticRetriever
from src.qa.answer_extractor import AnswerExtractor


class DocumentQAPipeline:
    """
    End-to-end pipeline for document question answering
    """

    def __init__(
        self,
        chunk_size: int = 300,
        overlap: int = 50,
        top_k: int = 3
    ):
        self.loader = PDFLoader()
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        self.embedder = TextEmbedder()
        self.retriever = SemanticRetriever(top_k=top_k)
        self.answer_extractor = AnswerExtractor()

    def run(self, file_paths: List[str], question: str) -> Dict:
        """
        Executes full QA pipeline

        Args:
            file_paths: list of uploaded PDF paths
            question: user question

        Returns:
            {
                "answer": str,
                "score": float,
                "doc_id": str
            }
        """

        # Step 1: Load PDFs
        documents = self.loader.load_pdfs(file_paths)

        # Step 2: Chunk documents
        chunks = self.chunker.chunk_documents(documents)

        # Step 3: Embed chunks
        embed_result = self.embedder.embed_chunks(chunks)

        # Step 4: Embed question
        q_emb = self.embedder.embed_question(question)

        # Step 5: Retrieve relevant chunks
        top_chunks = self.retriever.retrieve(
            q_emb,
            embed_result["embeddings"],
            embed_result["metadata"]
        )

        # Step 6: Extract answer
        answer = self.answer_extractor.extract_answer(question, top_chunks)

        return answer

if __name__ == "__main__":
    pipeline = DocumentQAPipeline(top_k=3)

    files = ["data/uploads/sample.pdf"]

    question = "What dataset was used in the study?"

    answer = pipeline.run(files, question)

    print("\nFINAL ANSWER:")
    print(answer)
