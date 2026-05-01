"""
train_explicit.py
-----------------
Retrain the radar-HAR CNN using an explicit subject-id split.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
import tensorflow as tf

from model import build_cnn, compile_model
from radar_utils import ACTIVITY_NAMES, NUM_CLASSES

# User-provided explicit splits
TRAIN_PERSONS = [1, 2, 4, 5, 6, 7, 9, 10, 13, 15, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 62, 63, 64, 66, 67, 68, 69, 70, 71]
VAL_PERSONS = [3, 11, 12, 16, 33, 44, 72]
TEST_PERSONS = [8, 14, 20, 42, 58, 61, 65]

def _load_dataset(path: str | Path):
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    persons = data["persons"].astype(np.int32)
    if "filenames" in data.files:
        filenames = np.asarray(data["filenames"])
    else:
        filenames = np.asarray([f"sample_{i}" for i in range(len(y))])
    if X.ndim == 3:
        X = X[..., np.newaxis]  # add channel dim
    return X, y, persons, filenames

def _split_explicit(persons: np.ndarray):
    """Return three boolean masks using the hardcoded subject lists."""
    train_mask = np.isin(persons, TRAIN_PERSONS)
    val_mask = np.isin(persons, VAL_PERSONS)
    test_mask = np.isin(persons, TEST_PERSONS)
    
    # Check if any person is missing or in multiple splits (just for sanity)
    unassigned = set(np.unique(persons)) - set(TRAIN_PERSONS) - set(VAL_PERSONS) - set(TEST_PERSONS)
    if unassigned:
        print(f"Warning: These persons in dataset are not in any split list: {unassigned}")
        
    return train_mask, val_mask, test_mask

def _plot_history(history, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history.history["loss"], label="train")
    if "val_loss" in history.history:
        ax[0].plot(history.history["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()

    ax[1].plot(history.history["accuracy"], label="train")
    if "val_accuracy" in history.history:
        ax[1].plot(history.history["val_accuracy"], label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

def _plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    labels = [ACTIVITY_NAMES[i + 1] for i in range(NUM_CLASSES)]
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalised)")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            txt = f"{cm[i, j]}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

def _augment(x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    max_shift = tf.shape(x)[1] // 8
    shift = tf.random.uniform([], -max_shift, max_shift + 1, dtype=tf.int32)
    x = tf.roll(x, shift=shift, axis=1)
    if tf.random.uniform([]) < 0.5:
        h = tf.shape(x)[0]
        mask_h = tf.random.uniform([], 1, tf.maximum(h // 10, 2), dtype=tf.int32)
        start = tf.random.uniform([], 0, h - mask_h, dtype=tf.int32)
        mask = tf.concat([
            tf.ones([start, tf.shape(x)[1], tf.shape(x)[2]], dtype=x.dtype),
            tf.zeros([mask_h, tf.shape(x)[1], tf.shape(x)[2]], dtype=x.dtype),
            tf.ones([h - start - mask_h, tf.shape(x)[1], tf.shape(x)[2]], dtype=x.dtype),
        ], axis=0)
        x = x * mask
    return x, y

def train(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {args.dataset} ...")
    X, y, persons, filenames = _load_dataset(args.dataset)
    print(f"  X={X.shape}  y={y.shape}  unique_persons={len(np.unique(persons))}")

    train_mask, val_mask, test_mask = _split_explicit(persons)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print("\nSplit summary (Explicit split):")
    print(f"  train   : {len(X_train):>5} samples / {len(set(persons[train_mask].tolist())):>3} persons")
    print(f"  val     : {len(X_val):>5} samples / {len(set(persons[val_mask].tolist())):>3} persons")
    print(f"  test    : {len(X_test):>5} samples / {len(set(persons[test_mask].tolist())):>3} persons")

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(buffer_size=len(X_train), seed=args.seed)
        .map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    tf.keras.utils.set_random_seed(args.seed)
    model = build_cnn(input_shape=X.shape[1:], num_classes=NUM_CLASSES, dropout=args.dropout)
    compile_model(model, learning_rate=args.lr)
    model.summary()

    ckpt_path = out_dir / "best_model_explicit.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(ckpt_path), monitor="val_accuracy",
            save_best_only=True, mode="max", verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=12,
            restore_best_weights=True, mode="max"),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    final_model_path = out_dir / "radar_cnn_explicit.keras"
    model.save(final_model_path)
    print(f"\nSaved final model to {final_model_path}")
    print(f"Best-val model saved to {ckpt_path}")

    # Evaluations
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\nValidation accuracy: {val_acc:.4f}   loss: {val_loss:.4f}")

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}   loss: {test_loss:.4f}")

    y_val_pred = np.argmax(model.predict(val_ds, verbose=0), axis=1)
    y_test_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)

    _plot_history(history, out_dir / "training_curves_explicit.png")
    _plot_confusion(y_val, y_val_pred, out_dir / "confusion_matrix_val_explicit.png")
    _plot_confusion(y_test, y_test_pred, out_dir / "confusion_matrix_test_explicit.png")

    with open(out_dir / "metrics_explicit.json", "w") as f:
        json.dump({
            "val_accuracy": float(val_acc),
            "val_loss": float(val_loss),
            "test_accuracy": float(test_acc),
            "test_loss": float(test_loss),
            "history": {k: [float(v) for v in vs] for k, vs in history.history.items()}
        }, f, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="processed/full_dataset.npz")
    ap.add_argument("--out-dir", default="models_explicit")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train(args)

if __name__ == "__main__":
    main()
