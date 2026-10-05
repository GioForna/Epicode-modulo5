import argparse
import json

import numpy as np
from sklearn.metrics import classification_report

from src.config import HISTORY_PATH, MODEL_PATH, MODELS_DIR, SELECTED_CLASSES
from src import vision


def parse_args():
    parser = argparse.ArgumentParser(description="Training del classificatore ChefBot su un subset di Food101")
    parser.add_argument("--backbone", choices=["efficientnet", "mobilenet"], default="efficientnet")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=5, help="0 per lasciare il backbone congelato")
    parser.add_argument("--fine-tune-layers", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-dir", default=None, help="cartella in cui TFDS scarica Food101")
    return parser.parse_args()


def main():
    args = parse_args()
    MODELS_DIR.mkdir(exist_ok=True)

    train_ds, test_ds = vision.get_datasets(data_dir=args.data_dir, batch_size=args.batch_size)

    model, base = vision.build_model(len(SELECTED_CLASSES), backbone=args.backbone)
    vision.compile_model(model, learning_rate=1e-3)
    model.summary()

    print("\n== Fase 1: backbone congelato ==")
    history = model.fit(train_ds, validation_data=test_ds, epochs=args.epochs, callbacks=vision.get_callbacks())
    full_history = {k: list(map(float, v)) for k, v in history.history.items()}

    if args.fine_tune_epochs > 0:
        print(f"\n== Fase 2: fine-tuning degli ultimi {args.fine_tune_layers} layer ==")
        vision.unfreeze_top_layers(base, args.fine_tune_layers)
        vision.compile_model(model, learning_rate=1e-5)
        start = len(full_history["loss"])
        history_ft = model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=start + args.fine_tune_epochs,
            initial_epoch=start,
            callbacks=vision.get_callbacks(),
        )
        for k, v in history_ft.history.items():
            full_history.setdefault(k, []).extend(map(float, v))

    model.save(MODEL_PATH)
    vision.save_class_names()
    with open(HISTORY_PATH, "w") as f:
        json.dump(full_history, f, indent=2)

    loss, acc, top3 = model.evaluate(test_ds, verbose=0)
    print(f"\nTest accuracy: {acc:.4f} | top-3 accuracy: {top3:.4f} | loss: {loss:.4f}")

    y_true, y_pred = [], []
    for images, labels in test_ds:
        y_true.append(labels.numpy())
        y_pred.append(np.argmax(model.predict_on_batch(images), axis=1))
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    print(classification_report(y_true, y_pred, target_names=SELECTED_CLASSES, digits=3))


if __name__ == "__main__":
    main()
