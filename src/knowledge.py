import json
from functools import lru_cache

from .config import KNOWLEDGE_BASE_PATH


@lru_cache(maxsize=1)
def load_knowledge_base(path=KNOWLEDGE_BASE_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize_label(label):
    return label.strip().lower().replace(" ", "_").replace("-", "_")


def lookup(label, kb=None):
    """Restituisce i metadati del piatto predetto, None se non presente nel database."""
    kb = kb if kb is not None else load_knowledge_base()
    key = normalize_label(label)
    if key not in kb:
        return None
    return {"label": key, **kb[key]}


def format_card(info, confidence=None):
    if info is None:
        return "Piatto non presente nel database."

    nutr = info["nutrienti_g"]
    lines = [f"## {info['nome']}"]
    if confidence is not None:
        lines.append(f"**Confidenza:** {confidence:.1%}")
    lines += [
        f"**Origine:** {info['origine']}  |  **Categoria:** {info['categoria']}",
        "",
        info["descrizione"],
        "",
        "### Ingredienti principali",
        ", ".join(info["ingredienti"]),
        "",
        f"### Valori nutrizionali (porzione da {info['porzione_g']} g)",
        f"- Calorie stimate: **{info['calorie_kcal']} kcal**",
        f"- Proteine: {nutr['proteine']} g",
        f"- Carboidrati: {nutr['carboidrati']} g",
        f"- Grassi: {nutr['grassi']} g",
        f"- Fibre: {nutr['fibre']} g",
    ]
    return "\n".join(lines)
