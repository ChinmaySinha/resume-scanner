# app/embeddings.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from app.config import settings

_model = None

def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model

def embed_text(text: str):
    model = get_embedding_model()
    if not text or text.strip() == "":
        d = model.get_sentence_embedding_dimension()
        return np.zeros(d, dtype=float)
    vec = model.encode([text], convert_to_numpy=True)[0]
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def embed_list(texts: List[str]):
    model = get_embedding_model()
    emb = model.encode(texts, convert_to_numpy=True)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms
