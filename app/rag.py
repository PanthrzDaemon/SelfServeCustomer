from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# Project paths
BASE_DIR = Path(__file__).parent.parent
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"


# Embedding model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ChromaDB
client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_or_create_collection(
    name="cloudflow_knowledge"
)


def load_documents():
    """Read all Markdown documents from the knowledge base."""

    documents = []
    metadatas = []
    ids = []

    for file_path in DOCUMENTS_DIR.glob("*.md"):
        text = file_path.read_text(encoding="utf-8")

        documents.append(text)
        metadatas.append({
            "source": file_path.name
        })
        ids.append(file_path.stem)

    return documents, metadatas, ids


def ingest_documents():
    """Add knowledge-base documents to ChromaDB."""

    documents, metadatas, ids = load_documents()

    if not documents:
        return {
            "success": False,
            "message": "No documents found."
        }

    embeddings = embedding_model.encode(
        documents
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    return {
        "success": True,
        "documents_added": len(documents)
    }


def search_knowledge(query, top_k=3):
    """Search the knowledge base."""

    query_embedding = embedding_model.encode(
        [query]
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    return results


if __name__ == "__main__":

    print("Ingesting documents...")

    result = ingest_documents()

    print(result)

    print("\nSearching knowledge base...")

    results = search_knowledge(
        "How much storage does the Pro plan provide?"
    )

    print("\n--- SEARCH RESULTS ---")

    for i, document in enumerate(results["documents"][0]):
        source = results["metadatas"][0][i]["source"]

        print(f"\nSource: {source}")
        print(document[:500])