import os
from typing import List
import numpy as np
from dotenv import load_dotenv
load_dotenv()

# Use OpenAI embeddings; otherwise fallback to sentence-transformers
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

if OPENAI_KEY:
    import openai
    openai.api_key = OPENAI_KEY

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

class EmbeddingProvider:
    def __init__(self, model_name_openai="text-embedding-3-small", st_model_name="all-MiniLM-L6-v2"):
        self.use_openai = bool(OPENAI_KEY)
        self.st_model_name = st_model_name
        if not self.use_openai:
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers is required when OPENAI_API_KEY is not set.")
            self.st_model = SentenceTransformer(st_model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self.use_openai:
            # call OpenAI embeddings (batch in 1 call if small number)
            # model name can be changed to gpt-4o-mini-embeddings or text-embedding-3-small
            resp = openai.Embedding.create(model="text-embedding-3-small", input=texts)
            vectors = [r['embedding'] for r in resp['data']]
            return vectors
        else:
            # sentence-transformers returns numpy arrays
            embs = self.st_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return embs.tolist()
