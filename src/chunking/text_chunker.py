from typing import List,Dict

class TextChunker:
    """
    Splits document text into overlapping chunks
    """

    def  __init__(self,chunk_size=300,overlap=50):
        self.chunk_size=chunk_size
        self.overlap=overlap

    def chunk_documents(self,documents:List[Dict])->List[Dict]:
        """
        Splits each document into chunks

        Args:
            documents: [
                {
                    "doc_id": str,
                    "text": str
                }
            ]

        Returns:
            chunks: [
                {
                    "doc_id": str,
                    "chunk_id": int,
                    "text": str
                }
            ]
        """

        all_chunks=[]

        for doc in documents:
            doc_id=doc['doc_id']
            text=doc['text']

            words=text.split()
            print(words)
            start=0
            chunk_id=0

            while start < len(words):
                end = start + self.chunk_size
                chunk_words = words[start:end]
                print(chunk_words)
                chunk_text = " ".join(chunk_words)
                print(chunk_text)

                all_chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "text": chunk_text
                })

                chunk_id += 1
                start += self.chunk_size - self.overlap

        return all_chunks 
    
if __name__ == "__main__":
    from src.ingestion.pdf_loader import PDFLoader

    loader = PDFLoader()
    docs = loader.load_pdfs(["data/uploads/sample.pdf"])

    chunker = TextChunker(chunk_size=300, overlap=50)
    chunks = chunker.chunk_documents(docs)

