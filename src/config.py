from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

KNOWLEDGE_BASE_PATH = DATA_DIR / "knowledge_base.json"
MODEL_PATH = MODELS_DIR / "chefbot_classifier.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
HISTORY_PATH = MODELS_DIR / "history.json"

# Subset di Food101 usato per il classificatore (nomi TFDS)
SELECTED_CLASSES = [
    "apple_pie",
    "baklava",
    "bruschetta",
    "caesar_salad",
    "caprese_salad",
    "ceviche",
    "chicken_curry",
    "churros",
    "greek_salad",
    "guacamole",
    "hamburger",
    "ice_cream",
    "lasagna",
    "miso_soup",
    "pancakes",
    "pizza",
    "ramen",
    "spaghetti_carbonara",
    "sushi",
    "tiramisu",
]

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
