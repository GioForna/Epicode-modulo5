import numpy as np

from .config import EMBEDDING_MODEL
from .knowledge import load_knowledge_base


def dish_to_text(info):
    return (
        f"{info['nome']}. {info['descrizione']} "
        f"Ingredienti: {', '.join(info['ingredienti'])}. "
        f"Caratteristiche: {', '.join(info['tag'])}."
    )


def cosine_similarity(query_vec, matrix):
    query_vec = query_vec / np.linalg.norm(query_vec)
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix @ query_vec


class MoodSearch:
    """Ricerca per umore: trova i piatti semanticamente piu' vicini a una frase libera."""

    def __init__(self, kb=None, model=None, model_name=EMBEDDING_MODEL):
        self.kb = kb if kb is not None else load_knowledge_base()
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name)
        self.model = model
        self.labels = list(self.kb.keys())
        self.texts = [dish_to_text(self.kb[k]) for k in self.labels]
        self.embeddings = np.asarray(self.model.encode(self.texts))

    def search(self, query, top_k=3):
        query_vec = np.asarray(self.model.encode([query]))[0]
        scores = cosine_similarity(query_vec, self.embeddings)
        best = np.argsort(scores)[::-1][:top_k]
        return [
            {"label": self.labels[i], "nome": self.kb[self.labels[i]]["nome"], "score": float(scores[i])}
            for i in best
        ]
