import streamlit as st
from rag_pipeline import build_pipeline, retrieve, generate_answer

# Initialize/Cache the pipeline (so it doesn't reload for every interaction)
@st.cache_resource
def get_pipeline():
    return build_pipeline()

st.title("Sanskrit RAG QA System")

# User query form
user_query = st.text_input("Enter your question (Sanskrit or English):")

if user_query:
    vectorizer, emb, chunks = get_pipeline()
    results = retrieve(user_query, vectorizer, emb, chunks, top_k=3)
    context_chunks = [c for c, score in results]

    st.subheader("Top Retrieved Chunks")
    for idx, chunk in enumerate(context_chunks, 1):
        st.markdown(f"**{idx}.** {chunk}")

    answer = generate_answer(context_chunks, user_query)
    st.subheader("Generated Answer")
    st.write(answer)
