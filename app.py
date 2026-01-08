import os
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename

from src.pipeline.qa_pipeline import DocumentQAPipeline
from src.ingestion.pdf_loader import PDFLoader
from src.chunking.text_chunker import TextChunker
from src.embeddings.embedder import TextEmbedder
from src.retrieval.retriever import SemanticRetriever
from src.qa.answer_extractor import AnswerExtractor

UPLOAD_FOLDER = "data/uploads"
ALLOWED_EXTENSIONS = {"pdf"}

app = Flask(__name__)
app.secret_key = "doc_qa_secret_key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global session cache (demo purpose)
SESSION_STORE = {}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    doc_id = None

    if request.method == "POST":

        # ----------------------------
        # Document Upload
        # ----------------------------
        if "documents" in request.files:
            files = request.files.getlist("documents")
            saved_paths = []

            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                    file.save(path)
                    saved_paths.append(path)

            if saved_paths:
                # Process documents only once
                loader = PDFLoader()
                docs = loader.load_pdfs(saved_paths)

                chunker = TextChunker()
                chunks = chunker.chunk_documents(docs)

                embedder = TextEmbedder()
                embed_result = embedder.embed_chunks(chunks)

                SESSION_STORE["embeddings"] = embed_result["embeddings"]
                SESSION_STORE["metadata"] = embed_result["metadata"]

                session["docs_ready"] = True

            return redirect(url_for("index"))

        # ----------------------------
        # Question Handling
        # ----------------------------
        if "question" in request.form and session.get("docs_ready"):
            question = request.form["question"]

            embedder = TextEmbedder()
            q_emb = embedder.embed_question(question)

            retriever = SemanticRetriever(top_k=3)
            top_chunks = retriever.retrieve(
                q_emb,
                SESSION_STORE["embeddings"],
                SESSION_STORE["metadata"]
            )

            extractor = AnswerExtractor()
            result = extractor.extract_answer(question, top_chunks)

            answer = result["answer"]
            doc_id = result["doc_id"]

    return render_template(
        "index.html",
        answer=answer,
        doc_id=doc_id,
        docs_ready=session.get("docs_ready", False)
    )


if __name__ == "__main__":
    app.run(debug=True)
