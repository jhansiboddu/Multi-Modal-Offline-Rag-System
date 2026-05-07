import faiss
import numpy as np
import pickle
import os
from rank_bm25 import BM25Okapi

INDEX_FILE = "./faiss.index"
DATA_FILE = "./chunks.pkl"
os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
# Global variables (simple approach)
index = None
stored_chunks = []

def initialize_keyword_search(chunks):
    global bm25, documents

    documents = [chunk.split() for chunk in chunks]
    bm25 = BM25Okapi(documents)

def search_keyword(query, top_k=3):
    global bm25, documents

    if bm25 is None:
        return []

    query_tokens = query.split()
    scores = bm25.get_scores(query_tokens)

    ranked = sorted(
        list(enumerate(scores)),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    return ranked

def initialize():
    global index, stored_data

    if os.path.exists(INDEX_FILE) and os.path.exists(DATA_FILE):
        index = faiss.read_index(INDEX_FILE)

        with open(DATA_FILE, "rb") as f:
            stored_data = pickle.load(f)

        print("✅ Loaded existing FAISS index")
    else:
        index = None
        stored_data = []
        print("⚠️ No existing index found")

def save():
    global index, stored_data

    if index is not None:
        faiss.write_index(index, INDEX_FILE)

        with open(DATA_FILE, "wb") as f:
            pickle.dump(stored_data, f)

        print("💾 Index saved")


def add_embeddings(embeddings, chunks, filename):
    global index, stored_data

    embeddings = np.array(embeddings)

    if index is None:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    # Store metadata
    for i, chunk in enumerate(chunks):
        stored_data.append({
            "text": chunk,
            "source": filename,
            "chunk_id": len(stored_data)
        })

    save()

def search(query_embedding, top_k=3):
    global index, stored_data

    if index is None:
        raise ValueError("No data available. Upload documents first.")

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        distance = distances[0][i]
        similarity = 1 / (1 + distance)

        results.append({
            "text": stored_data[idx]["text"],
            "source": stored_data[idx]["source"],
            "score": float(similarity)
        })

    return results