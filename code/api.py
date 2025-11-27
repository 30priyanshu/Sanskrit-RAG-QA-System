from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline import build_pipeline, retrieve, generate_answer

app = FastAPI()

# CORS so that browser (HTML/JS UI) can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # for dev; tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryPayload(BaseModel):
    question: str

# Build RAG pipeline once at startup
vectorizer, emb, chunks = build_pipeline()

@app.post("/query")
async def query_endpoint(payload: QueryPayload):
    question = payload.question
    results = retrieve(question, vectorizer, emb, chunks, top_k=3)
    context_chunks = [c for c, score in results]
    answer = generate_answer(context_chunks, question)
    return {"answer": answer}
