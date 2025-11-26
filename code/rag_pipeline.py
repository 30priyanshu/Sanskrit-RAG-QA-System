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
                chunk = para[i:i+max_length]
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
    """Generates an answer given relevant context chunks and a user query."""
    context = "\n".join(context_chunks)
    prompt = f"Answer this question based on the following context:\n{context}\nQuestion: {user_query}\nAnswer:"
    result = generator(prompt, max_length=60, num_return_sequences=1)
    return result[0]['generated_text']

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
        print("\nTop relevant chunks:\n", "\n---\n".join(chunk_texts_only))
        answer = generate_answer(chunk_texts_only, query)
        print("\nGenerated answer:\n", answer)
        print("\n" + "="*40)

if __name__ == "__main__":
    main()
