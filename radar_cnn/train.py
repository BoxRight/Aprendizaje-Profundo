"""
train.py
--------
Train the radar-HAR CNN on the cached spectrogram dataset produced by
``preprocess.py``.

The dataset is split into three disjoint subsets **by person id** so
that no subject ever appears in more than one of the splits:

* **train**     – used to fit the network.
* **validation** – monitored during training (early stopping, LR
  schedule, best-checkpoint selection).
* **hold-out test** – completely reserved.  It is **never** passed to
  ``model.fit`` and is only used for the final report.  The hold-out
  spectrograms together with their labels and source filenames are
  saved on disk so that the user can run the model against the same
  reserved subset later (see ``evaluate.py``).

Usage::

    python train.py
    python train.py --epochs 50 --val-frac 0.15 --holdout-frac 0.2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

from model import build_cnn, compile_model
from radar_utils import ACTIVITY_NAMES, NUM_CLASSES


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


def _split_by_person(persons: np.ndarray,
                     y: np.ndarray,
                     val_frac: float,
                     holdout_frac: float,
                     seed: int
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return three boolean masks ``(train, val, holdout)`` such that no
    person appears in more than one subset.

    Persons are shuffled deterministically (using ``seed``) and then sliced
    into the three groups.  The hold-out group is selected first so that
    successive training runs always reserve the same subjects for it (as
    long as the seed and the dataset are the same).
    """
    if val_frac < 0 or holdout_frac < 0 or val_frac + holdout_frac >= 1:
        raise ValueError(
            "val_frac and holdout_frac must be non-negative and "
            "their sum must be < 1.")

    rng = np.random.default_rng(seed)
    unique_persons = np.unique(persons)
    rng.shuffle(unique_persons)

    n_total = len(unique_persons)
    n_holdout = max(1, int(round(n_total * holdout_frac))) if holdout_frac > 0 else 0
    n_val = max(1, int(round(n_total * val_frac))) if val_frac > 0 else 0
    if n_holdout + n_val >= n_total:
        raise ValueError("Not enough unique persons for the requested splits.")

    holdout_persons = set(unique_persons[:n_holdout].tolist())
    val_persons = set(unique_persons[n_holdout:n_holdout + n_val].tolist())

    holdout_mask = np.array([p in holdout_persons for p in persons])
    val_mask = np.array([p in val_persons for p in persons])
    train_mask = ~(holdout_mask | val_mask)

    # Safety: every class must appear at least once in val and holdout so
    # that metrics are well defined.  If a class is missing, move one
    # sample over from the train split.
    def _ensure_class_coverage(target_mask: np.ndarray) -> None:
        for c in range(NUM_CLASSES):
            if not np.any(target_mask & (y == c)):
                candidates = np.where(train_mask & (y == c))[0]
                if candidates.size > 0:
                    idx = int(rng.choice(candidates))
                    train_mask[idx] = False
                    target_mask[idx] = True

    if n_val > 0:
        _ensure_class_coverage(val_mask)
    if n_holdout > 0:
        _ensure_class_coverage(holdout_mask)

    return train_mask, val_mask, holdout_mask


def _plot_history(history, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # noqa: WPS433 lazy import
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
        import matplotlib.pyplot as plt  # noqa: WPS433
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
    """Light augmentation: random time-shifts and frequency-masking."""
    # Random time shift (roll along width axis)
    max_shift = tf.shape(x)[1] // 8
    shift = tf.random.uniform([], -max_shift, max_shift + 1, dtype=tf.int32)
    x = tf.roll(x, shift=shift, axis=1)

    # Frequency masking (zero out a small horizontal band)
    if tf.random.uniform([]) < 0.5:
        h = tf.shape(x)[0]
        mask_h = tf.random.uniform([], 1, tf.maximum(h // 10, 2), dtype=tf.int32)
        start = tf.random.uniform([], 0, h - mask_h, dtype=tf.int32)
        mask = tf.concat([
            tf.ones([start, tf.shape(x)[1], tf.shape(x)[2]], dtype=x.dtype),
            tf.zeros([mask_h, tf.shape(x)[1], tf.shape(x)[2]], dtype=x.dtype),
            tf.ones([h - start - mask_h, tf.shape(x)[1], tf.shape(x)[2]],
                    dtype=x.dtype),
        ], axis=0)
        x = x * mask
    return x, y


def _save_holdout(out_dir: Path,
                  X: np.ndarray, y: np.ndarray,
                  persons: np.ndarray, filenames: np.ndarray) -> None:
    """Persist the held-out test subset so it can be re-used by ``evaluate.py``."""
    npz_path = out_dir / "holdout_test.npz"
    list_path = out_dir / "holdout_files.txt"

    np.savez_compressed(
        npz_path,
        X=X.astype(np.float32),
        y=y.astype(np.int64),
        persons=persons.astype(np.int32),
        filenames=np.asarray(filenames),
        class_names=np.array([ACTIVITY_NAMES[i + 1]
                              for i in range(NUM_CLASSES)]),
    )

    unique_files = sorted(set(str(f) for f in filenames))
    list_path.write_text("\n".join(unique_files) + "\n")

    mb = npz_path.stat().st_size / (1024 * 1024)
    print(f"Held-out test subset:")
    print(f"  spectrograms : {len(X)}")
    print(f"  source files : {len(unique_files)}")
    print(f"  saved to     : {npz_path}  ({mb:.1f} MB)")
    print(f"  file listing : {list_path}")


def train(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {args.dataset} ...")
    X, y, persons, filenames = _load_dataset(args.dataset)
    print(f"  X={X.shape}  y={y.shape}  unique_persons={len(np.unique(persons))}")

    train_mask, val_mask, holdout_mask = _split_by_person(
        persons, y, args.val_frac, args.holdout_frac, args.seed)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_hold, y_hold = X[holdout_mask], y[holdout_mask]

    print("\nSplit summary (by person id):")
    print(f"  train   : {len(X_train):>5} samples / "
          f"{len(set(persons[train_mask].tolist())):>3} persons")
    print(f"  val     : {len(X_val):>5} samples / "
          f"{len(set(persons[val_mask].tolist())):>3} persons")
    print(f"  holdout : {len(X_hold):>5} samples / "
          f"{len(set(persons[holdout_mask].tolist())):>3} persons   "
          "(reserved, NOT used for training)")
    print(f"  holdout persons -> {sorted(set(persons[holdout_mask].tolist()))}")

    _save_holdout(out_dir,
                  X_hold, y_hold,
                  persons[holdout_mask],
                  filenames[holdout_mask])

    # tf.data pipelines: only train + validation are seen during fit().
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
    holdout_ds = (
        tf.data.Dataset.from_tensor_slices((X_hold, y_hold))
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    tf.keras.utils.set_random_seed(args.seed)
    model = build_cnn(input_shape=X.shape[1:], num_classes=NUM_CLASSES,
                      dropout=args.dropout)
    compile_model(model, learning_rate=args.lr)
    model.summary()

    ckpt_path = out_dir / "best_model.keras"
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

    final_model_path = out_dir / "radar_cnn.keras"
    model.save(final_model_path)
    print(f"\nSaved final model to {final_model_path}")
    print(f"Best-val model saved to {ckpt_path}")

    # Final evaluation on the validation split (used during training)
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\nValidation accuracy: {val_acc:.4f}   loss: {val_loss:.4f}")

    # Final evaluation on the held-out test subset (never seen by the model)
    hold_loss, hold_acc = model.evaluate(holdout_ds, verbose=0)
    print(f"Hold-out test accuracy: {hold_acc:.4f}   loss: {hold_loss:.4f}")

    y_val_pred = np.argmax(model.predict(val_ds, verbose=0), axis=1)
    y_hold_pred = np.argmax(model.predict(holdout_ds, verbose=0), axis=1)

    _plot_history(history, out_dir / "training_curves.png")
    _plot_confusion(y_val, y_val_pred, out_dir / "confusion_matrix_val.png")
    _plot_confusion(y_hold, y_hold_pred, out_dir / "confusion_matrix_holdout.png")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "val_accuracy": float(val_acc),
                "val_loss": float(val_loss),
                "holdout_accuracy": float(hold_acc),
                "holdout_loss": float(hold_loss),
                "n_train_samples": int(len(X_train)),
                "n_val_samples": int(len(X_val)),
                "n_holdout_samples": int(len(X_hold)),
                "holdout_persons": sorted(set(int(p) for p
                                              in persons[holdout_mask])),
                "history": {k: [float(v) for v in vs]
                            for k, vs in history.history.items()},
                "class_names": [ACTIVITY_NAMES[i + 1]
                                for i in range(NUM_CLASSES)],
            },
            f,
            indent=2,
        )
    print(f"Metrics and plots saved under {out_dir}/")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="processed/dataset.npz",
                    help="Path to the preprocessed .npz file")
    ap.add_argument("--out-dir", default="models",
                    help="Directory to store model weights and plots")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="Fraction of persons used as the validation split "
                         "during training")
    ap.add_argument("--holdout-frac", type=float, default=0.20,
                    help="Fraction of persons reserved as the held-out test "
                         "subset (never used for training)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
