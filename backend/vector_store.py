import os
import chromadb
from typing import List, Dict
import uuid

# configure persistence directory
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "chroma_db")

# Persistent Chroma client
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


class VectorStore:
    def __init__(self, collection_name: str = "web_docs"):
        self.collection_name = collection_name
        try:
            # ✅ Disable default embedding function (we provide our own embeddings)
            self.collection = client.get_collection(
                name=collection_name,
                embedding_function=None
            )
            print(f"[VectorStore] Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = client.create_collection(
                name=collection_name,
                embedding_function=None
            )
            print(f"[VectorStore] Created new collection: {collection_name}")

    def reset(self):
        """Delete and recreate the collection."""
        try:
            client.delete_collection(self.collection_name)
            print(f"[VectorStore] Deleted collection: {self.collection_name}")
        except Exception:
            print(f"[VectorStore] No collection found: {self.collection_name}")

        self.collection = client.create_collection(
            name=self.collection_name,
            embedding_function=None
        )
        print(f"[VectorStore] Created new collection: {self.collection_name}")

    def add_documents(self, docs: List[Dict], embeddings: List[List[float]]):
        ids, metadatas, texts = [], [], []

        for d in docs:
            cid = d.get("chunk_id") or str(uuid.uuid4())
            ids.append(cid)
            metadatas.append({
                "url": d["url"],
                "title": d.get("title", ""),
                "chunk_id": cid,
            })
            texts.append(d["text"])

        self.collection.add(
            ids=ids,
            metadatas=metadatas,
            documents=texts,
            embeddings=embeddings
        )
        print(f"[VectorStore] Added {len(ids)} documents")

    def query(self, query_embeddings, n_results: int = 5):
        """Query similar documents from the vector store."""
        res = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=["metadatas", "documents", "distances"]
        )
        if not res or not res.get("ids") or not res["ids"][0]:
            return []

        hits = []
        for i in range(len(res["ids"][0])):
            hits.append({
                "id": res["ids"][0][i],
                "metadata": res["metadatas"][0][i],
                "document": res["documents"][0][i],
                "score": float(res["distances"][0][i]),
                "citation": f'{res["metadatas"][0][i].get("title", "")} ({res["metadatas"][0][i].get("url", "")})'
            })
        return hits

    def get_all_metadatas(self) -> List[Dict]:
        """
        Return all metadata entries from the collection.
        Useful for listing which URLs are indexed.
        """
        try:
            # ✅ Fetch all items (n_results = very large number)
            res = self.collection.get(include=["metadatas"])
            return res.get("metadatas", [])
        except Exception as e:
            print(f"[VectorStore] get_all_metadatas error: {e}")
            return []
