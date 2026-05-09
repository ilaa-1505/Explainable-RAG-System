import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

client = PersistentClient(path="embeddings/")
collection = client.get_or_create_collection(name="rag_docs")


def embed_and_store(input_path="processed/chunks.json"):
    with open(input_path, "r") as f:
        chunks = json.load(f)

    documents, metadatas, ids = [], [], []

    for i, chunk in enumerate(tqdm(chunks)):
        documents.append("passage: " + chunk["text"])
        metadatas.append(chunk["metadata"])
        ids.append(f"chunk_{i}")

    client.delete_collection(name="rag_docs")
    collection = client.get_or_create_collection(name="rag_docs")       

    embeddings = model.encode(
        documents,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings.tolist()
    )


if __name__ == "__main__":
    embed_and_store()