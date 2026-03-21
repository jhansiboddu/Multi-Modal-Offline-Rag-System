from sentence_transformers import SentenceTransformer

# Load model once (important)
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings