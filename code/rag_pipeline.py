import os
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

# --- LOADING ---
def load_texts(data_dir):
    """Loads all .txt files from the data directory into a list of strings."""
    all_chunks = []
    for fn in os.listdir(data_dir):
        if fn.endswith('.txt'):
            path = os.path.join(data_dir, fn)
            with open(path, encoding="utf-8") as f:
                all_chunks.append(f.read())
    return all_chunks

# --- CHUNKING ---
def chunk_texts(texts, max_length=300):
    """Splits a list of strings into smaller text chunks of max_length char."""
    chunks = []
    for text in texts:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        for para in paragraphs:
            for i in range(0, len(para), max_length):
                chunk = para[i:i + max_length]
                if len(chunk) > 30:
                    chunks.append(chunk)
    return chunks

# --- RETRIEVER SETUP ---
def fit_tfidf(chunks):
    """Fits and returns TF-IDF vectorizer and matrix from text chunks."""
    vectorizer = TfidfVectorizer()
    emb = vectorizer.fit_transform(chunks)
    return vectorizer, emb

def retrieve(query, vectorizer, emb, chunks, top_k=3):
    """Returns top-k chunks most relevant to a query."""
    query_vec = vectorizer.transform([query])
    similarities = (emb * query_vec.T).toarray().flatten()
    top_indices = similarities.argsort()[::-1][:top_k]
    return [(chunks[i], similarities[i]) for i in top_indices]

# --- GENERATOR ---
generator = pipeline("text-generation", model="distilgpt2")

def generate_answer(context_chunks, user_query):
    """Generate a short, clean answer (without showing prompt/context)."""
    context = "\n".join(context_chunks)
    prompt = (
        "You are a concise assistant for Sanskrit stories.\n"
        "Answer in simple Hindi or English in 1-2 lines using ONLY the context.\n"
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_query}\n"
        "Answer: "
    )

    # Generate: prompt + answer, then cut off the prompt part
    result = generator(
        prompt,
        max_length=len(prompt) + 80,   # prompt length + ~80 chars answer
        num_return_sequences=1,
        do_sample=False,
        temperature=0.3,
    )
    full_text = result[0]["generated_text"]
    answer_only = full_text[len(prompt):].strip()
    return answer_only

# --- PIPELINE CONSTRUCTOR: For import and testing ---
def build_pipeline(data_dir="../data", chunk_size=300):
    """Builds and returns everything needed for retrieval/generation."""
    texts = load_texts(data_dir)
    chunks = chunk_texts(texts, max_length=chunk_size)
    vectorizer, emb = fit_tfidf(chunks)
    return vectorizer, emb, chunks

# --- MAIN INTERFACE: Only runs if launched as script, not on import ---
def main():
    DATA_DIR = "../data"
    texts = load_texts(DATA_DIR)
    chunks = chunk_texts(texts)
    vectorizer, emb = fit_tfidf(chunks)

    print("\nWelcome to the Sanskrit RAG CLI. Type your question (or 'exit' to quit):\n")
    while True:
        query = input("Enter question (or 'exit'): ")
        if query.lower().strip() == "exit":
            break

        top_chunks = retrieve(query, vectorizer, emb, chunks)
        chunk_texts_only = [c for c, score in top_chunks]

        # Debug ke liye chunks dekhne hain to neeche line uncomment kar sakte ho:
        # print("\nTop relevant chunks:\n", "\n---\n".join(chunk_texts_only))

        answer = generate_answer(chunk_texts_only, query)
        print("\nAnswer:\n", answer)
        print("\n" + "=" * 40)

if __name__ == "__main__":
    main()
