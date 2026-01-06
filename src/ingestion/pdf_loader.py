import os
import PyPDF2
from typing import List,Dict

class PDFLoader:
    def __init__(self):
        pass

    def load_pdfs(self,file_paths:List[str])->List[Dict]:
        """
        Reads multiple PDF files and extracts text

        Args:
            file_paths: list of PDF file paths

        Returns:
            List of dictionaries:
            [
              {
                "doc_id": filename,
                "text": extracted_text
              }
            ]
        """

        document = []
        for path in file_paths:
            text = self.extract_text_from_pdf(path)
            document.append({
                "doc_id":os.path.basename(path),
                "text":text
            })
        
        return document

    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extracts text from a single PDF file
        """

        extracted_text = []

        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text.append(page_text)

        return "\n".join(extracted_text)

