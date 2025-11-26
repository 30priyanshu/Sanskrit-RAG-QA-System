import os

DATA_DIR = "../data"   # Your data folder
CHUNK_SIZE = 300       # Number of characters in a chunk (adjust as needed)

def chunk_text(text, max_length):
    chunks = []
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    for para in paragraphs:
        # Break long paragraphs into smaller chunks
        for i in range(0, len(para), max_length):
            chunk = para[i:i+max_length]
            if len(chunk) > 30:   # Only keep meaningful chunks
                chunks.append(chunk)
    return chunks

all_chunks = []
for fn in os.listdir(DATA_DIR):
    if fn.endswith('.txt'):
        path = os.path.join(DATA_DIR, fn)
        with open(path, encoding="utf-8") as f:
            text = f.read()
            chunks = chunk_text(text, CHUNK_SIZE)
            # Store file name and chunk id for traceability
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    'source': fn,
                    'chunk_id': i,
                    'text': chunk
                })

# Let's print out what we did
for c in all_chunks:
    print(f"--- {c['source']} (chunk {c['chunk_id']}) ---")
    print(c['text'])
    print('')
