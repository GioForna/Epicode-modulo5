# ChefBot AI - Dallo scatto alla ricetta

Progetto finale del Modulo 5 (Epicode).

ChefBot è l'assistente digitale di una startup Food-Tech che collega i pixel di una foto alla conoscenza testuale su un piatto:

1. **Riconoscimento visivo** - classifica la foto di un piatto tra 20 classi di Food101 con transfer learning (EfficientNetB0 / MobileNetV2).
2. **Knowledge retrieval** - dalla predizione recupera una scheda con ingredienti, calorie stimate, macronutrienti e descrizione da un database JSON.
3. **Ricerca per umore** - l'utente scrive una frase (es. *"voglio qualcosa di fresco e leggero"*) e il sistema propone il piatto più vicino semanticamente usando gli embeddings di Sentence-Transformers e la similarità coseno.

## Struttura

```
├── app.py                      # demo Gradio (foto -> scheda, ricerca per umore)
├── train.py                    # training da riga di comando
├── data/
│   └── knowledge_base.json     # "dizionario della conoscenza" dei 20 piatti
├── models/                     # modello addestrato e nomi delle classi
├── notebooks/
│   └── ChefBot_AI.ipynb        # notebook completo (training, valutazione, demo)
└── src/
    ├── config.py               # classi scelte, path e iperparametri
    ├── vision.py               # dataset Food101, modello, training, predizione
    ├── knowledge.py            # lookup nel database e scheda del piatto
    ├── semantic_search.py      # embeddings e similarità coseno
    └── chefbot.py              # pipeline che unisce le tre parti
```

## Classi scelte

`apple_pie`, `baklava`, `bruschetta`, `caesar_salad`, `caprese_salad`, `ceviche`, `chicken_curry`, `churros`, `greek_salad`, `guacamole`, `hamburger`, `ice_cream`, `lasagna`, `miso_soup`, `pancakes`, `pizza`, `ramen`, `spaghetti_carbonara`, `sushi`, `tiramisu`

Sono piatti molto diversi tra loro (dolci, insalate, zuppe, primi, street food, cucina giapponese...) così che anche la ricerca semantica abbia senso.

## 1. Riconoscimento visivo

- Dataset: Food101 da `tensorflow_datasets` (split `train` e `validation`), filtrato sulle 20 classi con una `StaticHashTable` che rimappa le etichette 0-100 in 0-19.
- Immagini ridimensionate a 224x224, pipeline `tf.data` con `cache`, `shuffle` e `prefetch`.
- Modello: backbone **EfficientNetB0** pre-addestrato su ImageNet con pesi congelati + data augmentation + classificatore personalizzato (`GlobalAveragePooling → BatchNorm → Dropout → Dense(256) → Dropout → Dense(20, softmax)`).
- Training in due fasi:
  1. solo il classificatore (backbone congelato, Adam lr 1e-3);
  2. fine-tuning degli ultimi 30 layer del backbone con lr 1e-5 (le BatchNorm restano congelate).
- Callback: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`.
- Valutazione: accuracy, top-3 accuracy, classification report e matrice di confusione.

Nota: la versione TFDS di Food101 ha solo gli split `train` e `validation`, quindi lo split `validation` viene usato come test set.

## 2. Knowledge base

`data/knowledge_base.json` mappa ogni etichetta a:

| campo | contenuto |
|---|---|
| `nome`, `origine`, `categoria` | informazioni generali |
| `descrizione` | breve descrizione del piatto |
| `ingredienti` | ingredienti principali |
| `porzione_g`, `calorie_kcal` | calorie stimate per una porzione |
| `nutrienti_g` | proteine, carboidrati, grassi, fibre |
| `tag` | caratteristiche (dolce, fresco, leggero, piccante...) usate anche dalla ricerca semantica |

I valori nutrizionali sono stime indicative per una porzione media.

```python
from src.knowledge import lookup, format_card

info = lookup("baklava")
print(format_card(info, confidence=0.93))
```

## 3. Ricerca per umore

Per ogni piatto viene costruito un testo (nome + descrizione + ingredienti + tag) e trasformato in un vettore con il modello multilingua `paraphrase-multilingual-MiniLM-L12-v2`. La frase dell'utente viene codificata allo stesso modo e confrontata con tutti i piatti tramite similarità coseno.

```python
from src.semantic_search import MoodSearch

search = MoodSearch()
search.search("dolce con miele")
```

## Come eseguire

### Google Colab (consigliato)

Aprire `notebooks/ChefBot_AI.ipynb` su Colab con runtime GPU ed eseguire tutte le celle: il notebook clona il repository, installa le dipendenze, scarica Food101, addestra il modello e lancia la demo.

### In locale

```bash
pip install -r requirements.txt

# training (il primo avvio scarica Food101, circa 5 GB)
python train.py --backbone efficientnet --epochs 10 --fine-tune-epochs 5

# demo
python app.py
```

Il modello viene salvato in `models/chefbot_classifier.keras` insieme a `models/class_names.json` e `models/history.json`.

## Tecnologie

TensorFlow / Keras, TensorFlow Datasets, Sentence-Transformers, scikit-learn, Matplotlib / Seaborn, Gradio.
