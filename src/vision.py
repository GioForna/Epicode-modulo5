import json

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

from .config import (
    BATCH_SIZE,
    CLASS_NAMES_PATH,
    IMG_SIZE,
    MODEL_PATH,
    SEED,
    SELECTED_CLASSES,
)

AUTOTUNE = tf.data.AUTOTUNE


def load_food101(data_dir=None):
    (ds_train, ds_test), ds_info = tfds.load(
        "food101",
        split=["train", "validation"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
        data_dir=data_dir,
    )
    return ds_train, ds_test, ds_info


def build_label_table(all_class_names, selected_classes=SELECTED_CLASSES):
    """Mappa gli id originali di Food101 (0-100) sugli id del subset (0-N), -1 per le classi escluse."""
    original_ids = [all_class_names.index(c) for c in selected_classes]
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(original_ids, dtype=tf.int64),
            values=tf.range(len(original_ids), dtype=tf.int64),
        ),
        default_value=-1,
    )
    return table


def resize_image(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return tf.cast(image, tf.uint8), label


def prepare_dataset(ds, table, training=False, batch_size=BATCH_SIZE, cache=True):
    ds = ds.map(lambda img, lbl: (img, table.lookup(tf.cast(lbl, tf.int64))), num_parallel_calls=AUTOTUNE)
    ds = ds.filter(lambda img, lbl: lbl >= 0)
    ds = ds.map(resize_image, num_parallel_calls=AUTOTUNE)
    if cache:
        ds = ds.cache(cache if isinstance(cache, str) else "")
    if training:
        ds = ds.shuffle(2000, seed=SEED)
    return ds.batch(batch_size).prefetch(AUTOTUNE)


def get_datasets(data_dir=None, selected_classes=SELECTED_CLASSES, batch_size=BATCH_SIZE, cache=True):
    ds_train, ds_test, ds_info = load_food101(data_dir)
    all_names = ds_info.features["label"].names
    table = build_label_table(all_names, selected_classes)
    train = prepare_dataset(ds_train, table, training=True, batch_size=batch_size, cache=cache)
    test = prepare_dataset(ds_test, table, training=False, batch_size=batch_size, cache=cache)
    return train, test


def build_model(num_classes, backbone="efficientnet", dropout=0.3, weights="imagenet"):
    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )

    if backbone == "efficientnet":
        base = keras.applications.EfficientNetB0(
            include_top=False, weights=weights, input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        # EfficientNet in Keras include gia' il rescaling interno: accetta pixel in [0, 255]
        preprocess = layers.Identity(name="preprocess")
    elif backbone == "mobilenet":
        base = keras.applications.MobileNetV2(
            include_top=False, weights=weights, input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        preprocess = layers.Rescaling(1.0 / 127.5, offset=-1, name="preprocess")
    else:
        raise ValueError(f"Backbone non supportato: {backbone}")

    base.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    x = augmentation(inputs)
    x = preprocess(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    model = keras.Model(inputs, outputs, name=f"chefbot_{backbone}")
    return model, base


def compile_model(model, learning_rate=1e-3):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc")],
    )
    return model


def unfreeze_top_layers(base, n_layers=30):
    """Fine-tuning: sblocca gli ultimi n layer del backbone lasciando congelate le BatchNorm."""
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers[-n_layers:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def get_callbacks(model_path=MODEL_PATH, patience=3):
    return [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=1, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(str(model_path), monitor="val_accuracy", save_best_only=True),
    ]


def save_class_names(class_names=SELECTED_CLASSES, path=CLASS_NAMES_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(class_names), f, indent=2)


def load_class_names(path=CLASS_NAMES_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_trained_model(path=MODEL_PATH):
    return keras.models.load_model(path)


def preprocess_image(image):
    """Accetta un path, una PIL.Image o un array e restituisce un batch (1, IMG_SIZE, IMG_SIZE, 3)."""
    if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
        image = keras.utils.load_img(image)
    image = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
    image = tf.image.resize(image[..., :3], (IMG_SIZE, IMG_SIZE))
    return tf.expand_dims(tf.cast(image, tf.uint8), 0)


def predict(model, image, class_names, top_k=3):
    probs = model.predict(preprocess_image(image), verbose=0)[0]
    best = np.argsort(probs)[::-1][:top_k]
    return [(class_names[i], float(probs[i])) for i in best]
