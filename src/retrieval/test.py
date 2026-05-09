from query import retrieve

if __name__ == "__main__":
    q = "how to install transformers"
    results, debug = retrieve(q)

    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print("Score:", r["rerank_score"])
        print(r["text"][:300])

    print("\n--- DEBUG INFO ---")
    print(debug)