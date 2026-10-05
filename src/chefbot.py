from .knowledge import format_card, load_knowledge_base, lookup
from .semantic_search import MoodSearch


class ChefBot:
    """Unisce classificatore, knowledge base e ricerca semantica."""

    def __init__(self, model=None, class_names=None, search=None):
        self.kb = load_knowledge_base()
        self._model = model
        self._class_names = class_names
        self._search = search

    @property
    def model(self):
        if self._model is None:
            from .vision import load_trained_model

            self._model = load_trained_model()
        return self._model

    @property
    def class_names(self):
        if self._class_names is None:
            from .vision import load_class_names

            self._class_names = load_class_names()
        return self._class_names

    @property
    def search(self):
        if self._search is None:
            self._search = MoodSearch(self.kb)
        return self._search

    def recognize(self, image, top_k=3):
        from .vision import predict

        preds = predict(self.model, image, self.class_names, top_k=top_k)
        label, confidence = preds[0]
        info = lookup(label, self.kb)
        return {"label": label, "confidence": confidence, "top_k": preds, "info": info}

    def dish_card(self, image):
        result = self.recognize(image)
        return format_card(result["info"], result["confidence"])

    def mood_search(self, query, top_k=3):
        results = self.search.search(query, top_k=top_k)
        for r in results:
            r["info"] = lookup(r["label"], self.kb)
        return results
