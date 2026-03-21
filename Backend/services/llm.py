import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def generate_answer(query, context):
    prompt = f"""
    You are a strict assistant.
    Use ONLY the information from the provided chunks.
    Rules:
    - Do NOT add external knowledge
    - Do NOT guess
    - Do NOT combine unrelated steps
    - If answer is not clearly present, say: "I don't know based on the provided documents."
    Answer ONLY using relevant chunks.
    Context:{context}
    Question:{query}
    Answer:
    """
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "tinyllama",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]