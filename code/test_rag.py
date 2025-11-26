from rag_pipeline import retrieve, generate_answer, build_pipeline

vectorizer, emb, chunks = build_pipeline()

test_cases = [
    {
        "query": "Who was Kalidasa?",
        "expected_phrase": "poet"
    },
    {
        "query": "यक्षः कः आसीत्?",
        "expected_phrase": "कुबेरस्य सेवकः"
    }
]

for case in test_cases:
    results = retrieve(case["query"], vectorizer, emb, chunks)
    context_chunks = [c for c, score in results]
    print(f"Query: {case['query']}")
    print("Retrieved:", context_chunks)
    answer = generate_answer(context_chunks, case["query"])
    print("Generated answer:", answer)
    print("Check for expected:", case["expected_phrase"] in answer)
    print("-----\n")
