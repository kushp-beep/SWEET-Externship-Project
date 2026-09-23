import os
import ollama
import numpy as np

DOCUMENT_PATH = "carnatic_music.txt"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"
TOP_K = 4

def load_document(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def chunk_document(text):
    raw_chunks = text.split("\n\n")
    chunks = [c.strip() for c in raw_chunks if c.strip()]
    return chunks

def get_embedding(text):
    """Fetches embedding vector from Ollama and normalizes it."""
    # Supports both ollama.embeddings and ollama.embed
    try:
        res = ollama.embeddings(model=EMBED_MODEL, prompt=text)
        vec = np.array(res["embedding"], dtype=np.float32)
    except AttributeError:
        res = ollama.embed(model=EMBED_MODEL, input=text)
        vec = np.array(res["embeddings"][0], dtype=np.float32)

    # L2 Normalization for fast dot-product similarity
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def embed_all_chunks(chunks):
    chunk_embeddings = []
    for chunk in chunks:
        vector = get_embedding(chunk)
        chunk_embeddings.append((chunk, vector))
    return chunk_embeddings

def retrieve_relevant_chunks(question, chunk_embeddings, top_k=TOP_K):
    """Calculates cosine similarity via dot product and retrieves top_k chunks."""
    question_vector = get_embedding(question)

    scored_chunks = []
    for chunk, vector in chunk_embeddings:
        # Dot product of normalized unit vectors = Cosine Similarity
        similarity = float(np.dot(question_vector, vector))
        scored_chunks.append((chunk, similarity))

    # Sort chunks by similarity score in descending order
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    # Take top_k items
    top_chunks = scored_chunks[:top_k]
    return top_chunks

def generate_answer(question, top_chunks):
    """Passes top retrieved chunks to llama3.2 with strict system instructions."""
    context_text = "\n\n".join([chunk for chunk, score in top_chunks])

    prompt = f"""Answer the question using ONLY the context provided below. If the answer cannot be determined from the context, state that clearly.

    Context:
    {context_text}

    Question: {question}"""

    response = ollama.chat(
        model=LLM_MODEL, messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

def main():
    if not os.path.exists(DOCUMENT_PATH):
        print(f"Error: {DOCUMENT_PATH} not found.")
        return

    print("Loading document and indexing vector embeddings...")
    text = load_document(DOCUMENT_PATH)
    chunks = chunk_document(text)

    if not chunks:
        print("Error: Document contains no text.")
        return

    # Index embeddings once at startup
    chunk_embeddings = embed_all_chunks(chunks)

    print(f"Loaded and indexed {len(chunks)} paragraph chunks from '{DOCUMENT_PATH}'.")
    print("\n=== Local Document Q&A System Ready ===")
    print("Type your question below (or 'exit' to quit).\n")

    while True:
        try:
            question = input("Question: ").strip()
            if not question or question.lower() == "exit":
                print("Exiting application.")
                break

            top_chunks = retrieve_relevant_chunks(question, chunk_embeddings)
            answer = generate_answer(question, top_chunks)

            print("\n" + "=" * 50)
            print("ANSWER:")
            print(answer)
            print("-" * 50)
            print("SOURCES / RETRIEVED CONTEXT:")
            for chunk, similarity in top_chunks:
                snippet = chunk.replace("\n", " ")[:120]
                print(f" - [Score: {similarity:.4f}] \"{snippet}...\"")
            print("=" * 50 + "\n")

        except KeyboardInterrupt:
            print("\nExiting application.")
            break


if __name__ == "__main__":
    main()