import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

# def generate_answer(query, context):
#     prompt = f"""
#     You are a strict assistant.
#     Use ONLY the information from the provided chunks.
#     Rules:
#     IMPORTANT:
#     - Use bullet points where possible
#     - Use headings if needed
#     - Keep it clean and structured
#     - Do NOT dump raw text
#     - Do NOT add external knowledge
#     - Do NOT guess
#     - Do NOT combine unrelated steps
#     - If answer is not clearly present, say: "I don't know based on the provided documents."
#     Answer ONLY using relevant chunks.
#     Context:{context}
#     Question:{query}
#     Answer:
#     """
#     response = requests.post(
#         OLLAMA_URL,
#         json={
#             "model": "tinyllama",
#             "prompt": prompt,
#             "stream": False
#         }
#     )

#     return response.json()["response"]

def generate_answer(query, context):
    prompt = f"""
Answer ONLY using the provided chunks.

- Quote or closely follow the wording
- Do NOT generalize
- Do NOT add external knowledge
- If unsure, say: "I don't know"

Chunks:
{context}

Question:
{query}

Answer:
"""
#     prompt = f"""
# You are a strict question-answering assistant.

# You MUST follow these rules:
# - Answer ONLY using the provided context
# - Do NOT use outside knowledge
# - Do NOT guess
# - If answer is not clearly present, say:
#   "I don't know based on the provided documents."

# Formatting rules:
# - Be clear and structured
# - Use bullet points if multiple points
# - Keep it concise
# - Do NOT dump raw text
# - Do NOT say "based on the context"

# Context:
# {context}

# Question:
# {query}

# Answer:
# """

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }
    )

    print("RAW RESPONSE:", response.text)  # DEBUG

    if response.status_code != 200:
        return f"LLM Error: {response.text}"

    try:
        data = response.json()
    except:
        return f"Invalid JSON response: {response.text}"

    if "response" in data:
        return data["response"]
    else:
        return f"Unexpected LLM output: {data}"