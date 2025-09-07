import uuid
from typing import Dict, List
from .web_loader import fetch_website_content
from .embeddings import EmbeddingProvider
from .vector_store import VectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter


class Engine:
    def __init__(self):
        self.store = VectorStore()
        self.embedder = EmbeddingProvider()
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    def reset_index(self):
        """Reset the vector store (wipe all indexed data)."""
        self.store.reset()

    def index_url(self, url: str) -> Dict:
        """
        Fetch the page, chunk it, get embeddings, add to vector store.
        Returns a dict with url, title, n_chunks, status.
        """
        fetched = fetch_website_content(url)
        title = fetched.get("title", url)
        content = fetched.get("content", "")

        if content.startswith("__FETCH_ERROR__"):
            return {"url": url, "title": title, "status": "error", "detail": content}

        # chunk the content
        chunks = self.splitter.split_text(content)

        docs = []
        texts = []
        for i, c in enumerate(chunks):
            cid = f"{uuid.uuid4()}"
            docs.append({
                "url": url,
                "title": title,
                "chunk_id": cid,
                "text": c
            })
            texts.append(c)

        # embed
        embeddings = self.embedder.embed_texts(texts)

        # add to vector store
        self.store.add_documents(docs=docs, embeddings=embeddings)

        return {"url": url, "title": title, "status": "success", "n_chunks": len(chunks)}

    def retrieve(self, query: str, top_k: int = 5):
        """Retrieve relevant chunks from the vector store."""
        q_emb = self.embedder.embed_texts([query])[0]
        hits = self.store.query(query_embeddings=[q_emb], n_results=top_k)
        return hits

    def list_indexed(self) -> List[str]:
        """Return a list of unique URLs currently stored in the vector DB."""
        try:
            metas = self.store.get_all_metadatas()
            urls = {m.get("url") for m in metas if m.get("url")}
            return sorted(list(urls))
        except Exception:
            return []


# Create a single global engine instance for use in router
engine = Engine()
