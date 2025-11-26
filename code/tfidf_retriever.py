import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from transformers import pipeline

DATA_DIR = "../data"
CHUNK_SIZE = 300

# 1. Chunking logic
def chunk_text(text, max_length):
    chunks = []
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    for para in paragraphs:
        for i in range(0, len(para), max_length):
            chunk = para[i:i+max_length]
            if len(chunk) > 30:
                chunks.append(chunk)
    return chunks

# 2. Loading and vectorizing
all_chunks = []
for fn in os.listdir(DATA_DIR):
    if fn.endswith('.txt'):
        path = os.path.join(DATA_DIR, fn)
        with open(path, encoding="utf-8") as f:
            text = f.read()
            chunks = chunk_text(text, CHUNK_SIZE)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(all_chunks)  # Matrix for retrieval

# 3. The retrieve function (must be above main block!)
def retrieve(query, top_k=3):
    query_vec = vectorizer.transform([query])
    similarities = (X * query_vec.T).toarray().flatten()
    top_indices = similarities.argsort()[::-1][:top_k]
    return [(i, all_chunks[i], similarities[i]) for i in top_indices]

# 4. The generator
generator = pipeline("text-generation", model="distilgpt2")

def generate_answer(context_chunks, user_query):
    combined_context = "\n".join(context_chunks)
    prompt = f"Answer this question based on the following context:\n{combined_context}\nQuestion: {user_query}\nAnswer:"
    result = generator(prompt, max_length=60, num_return_sequences=1)
    return result[0]['generated_text']

# 5. Main interactive loop
if __name__ == "__main__":
    while True:
        query = input("Enter your question in Sanskrit or English (or 'exit'): ")
        if query.lower().strip() == "exit":
            break
        results = retrieve(query, top_k=3)
        context_chunks = [c for idx, c, score in results]
        print("\nTop matches:")
        for c in context_chunks:
            print(c)
        print("\n---")
        answer = generate_answer(context_chunks, query)
        print("Generated answer:\n", answer)
        print("\n---\n")
