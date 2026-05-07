from fastapi import FastAPI, UploadFile, File
import shutil
import os
from services.parser import extract_text_from_pdf
from services.chunker import chunk_text
from services.embedder import get_embeddings , model
from db.faiss_store import search , initialize ,add_embeddings
from services.llm import generate_answer
from services.audio import transcribe_audio
from services.image import extract_text_from_image
import numpy as np
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
@app.get("/files")
async def get_files():
    try:
        # Check if directory exists
        if not os.path.exists(UPLOAD_DIR):
            return {"files": []}

        # List only files (ignore folders)
        files = [
            f for f in os.listdir(UPLOAD_DIR)
            if os.path.isfile(os.path.join(UPLOAD_DIR, f))
        ]

        return {"files": files}

    except Exception as e:
        return {"files": [], "error": str(e)}

@app.on_event("startup")
def startup_event():
    initialize()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Step 1: Extract text
    if file.filename.endswith(".mp3") or file.filename.endswith(".wav"):
        extracted_text = transcribe_audio(file_path)
    elif file.filename.endswith((".png", ".jpg", ".jpeg")):
        extracted_text = extract_text_from_image(file_path)
        print("EXTRACTED TEXT:", extracted_text)
    else:
        extracted_text = extract_text_from_pdf(file_path)

    # Step 2: Chunk text
    chunks = chunk_text(extracted_text)

    # Step 3: Generate embeddings
    embeddings = get_embeddings(chunks)

    # Step 4: Store in FAISS
    add_embeddings(embeddings, chunks, file.filename)

    return {
     "filename": file.filename,
     "chunks": len(chunks),
     "message": "Embeddings created and stored"
}

class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def query_rag(request: QueryRequest):
    query = request.query

    # Step 1: Convert query → embedding
    query_embedding = model.encode([query])

    # Step 2: Search FAISS
    results = search(query_embedding, top_k=3)
    results = sorted(
        results,
        key=lambda r: keyword_match_score(query, r["text"]),
        reverse=True
    )
    if not results:
        return {
            "answer": "No relevant information found.",
            "sources": []
       }
    context = "\n\n".join([
        f"Chunk {i+1}:\n{r['text']}"
        for i, r in enumerate(results)
    ])
    # Step 4: Generate answer
    answer = generate_answer(query, context)
    score = float(results[0]["score"])
    confidence = max(0, min(1, score))  # clamp between 0–1
    return {
        "query": query,
        "answer": answer,
        "confidence": confidence,
        "sources": [
            {
                "fileName": r.get("source", "Unknown"),
                "snippet": r["text"][:200],
                "score": float(r["score"])
            }
            for r in results
        ]
    }
def keyword_match_score(query, text):
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())
    return len(query_words.intersection(text_words))
