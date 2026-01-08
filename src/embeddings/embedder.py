from typing import List,Dict
import numpy as np

from sentence_transformers import SentenceTransformer

class TextEmbedder:
    """
    Generates embeddings for text chunks using Sentence-BERT
    """

    def __init__(self,model_name:str="all-MiniLM-L6-v2"):
        self.model=SentenceTransformer(model_name)
    
    def embed_chunks(self,chunks:List[Dict])->Dict:
        """
        Converts text chunks into embeddings

        Args:
            chunks: [
                {
                    "doc_id": str,
                    "chunk_id": int,
                    "text": str
                }
            ]

        Returns:
            {
                "embeddings": np.ndarray,
                "metadata": chunks
            }
        """
        texts=[chunk['text'] for chunk in chunks]
        embeddings=self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return {
            "embeddings":embeddings,
            "metadata":chunks
        }

    def embed_question(self,question:str)->np.ndarray:
        """
        Converts question into embedding
        """

        embedding=self.model.encode(
            question,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding
    
if __name__ == "__main__":
    from src.ingestion.pdf_loader import PDFLoader
    from src.chunking.text_chunker import TextChunker

    loader = PDFLoader()
    docs = loader.load_pdfs(["data/uploads/sample.pdf"])

    chunker = TextChunker(chunk_size=300, overlap=50)
    chunks = chunker.chunk_documents(docs)

    embedder = TextEmbedder()
    result = embedder.embed_chunks(chunks)

    print("Embedding shape:", result["embeddings"].shape)
    print("First vector (first 10 dims):", result["embeddings"][0][:10])
    # print(result)
