Sanskrit RAG QA System
A Retrieval-Augmented Generation (RAG) project for answering questions from Sanskrit literature using local Python. The system ingests Sanskrit text files, retrieves context using TF-IDF, and generates answers using a lightweight language model.
It supports both command-line and web UI (HTML/JS) usage.

Features
Load multiple Sanskrit/English text files as knowledge base

Document text is split into smart “chunks” for efficient retrieval

Fast, CPU-friendly retrieval using TF-IDF vectorizer

Answer generation using a local, small language model (distilgpt2)

Modular codebase for research, learning, and extension

Automated test suite to ensure retrieval and generation correctness

Optional web UI (HTML/CSS/JS) for user-friendly interaction

Folder Structure
text
RAG_Sanskrit_<YourName>/
├── code/
│   ├── rag_pipeline.py     # Main pipeline: loading, retrieval, generation
│   ├── test_rag.py         # Automated test cases for pipeline
│   ├── api.py              # (Optional) FastAPI backend for web UI/API use
│   ├── app.py              # (Optional) Streamlit web demo
│   └── ...                 # Any helper scripts
├── data/
│   ├── kalidasa_story_1.txt
│   ├── kalidasa_story_2.txt
│   └── ...                 # Any other documents
├── report/
│   └── Project_Report.pdf  # Your architecture and results writeup
├── web/
│   ├── index.html          # Simple HTML chat UI
│   ├── chat.css
│   └── chat.js
├── README.md
└── requirements.txt
Setup
Clone/download the repository

(Recommended) Use a virtual environment:

text
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

text
pip install -r requirements.txt
Example requirements.txt:

text
scikit-learn
transformers
fastapi
uvicorn
streamlit
How to Run
1. Command-line RAG Pipeline
text
cd code
python rag_pipeline.py
Type questions in Sanskrit or English. The system retrieves relevant context and generates an answer.

2. Run Automated Tests
text
python test_rag.py
Runs several test queries and prints if retrieval/generation is correct.

3. Web UI (HTML+JS Frontend)
Start API backend:

text
uvicorn api:app --reload
Open web/index.html in your browser

Ask questions in the chat interface.

4. (Optional) Streamlit Demo
text
streamlit run app.py
Interact with the system in your browser using Streamlit UI.

Example
Question: Who was Kalidasa?
Top Retrieved Context:
कालिदासः आरम्भे जडः, ... सः तदनन्तरम् कुमारसंभवः, मेघदूतः, ... रचयामास।
Generated Answer:
Kalidasa was a famous Sanskrit poet and playwright known for works like Meghaduta.

Customize / Extend
Add more .txt files to the data/ folder for richer answers.

Tweak chunk size or retrieval method by editing rag_pipeline.py.

Swap models for more accurate/faster answers (see HuggingFace/transformers docs).

Customize the web UI via CSS or JavaScript as you wish.

Report/Documentation
See /report/Project_Report.pdf for architecture diagrams, evaluation results, and limitations.

Credits
RAG pipeline design: Based on [open-source templates]​

Chat/UI inspiration: Scaler, W3Schools, YouTube RAG demos