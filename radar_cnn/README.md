# Radar CNN — Doppler spectrogram classifier (INSHEP / Glasgow `.dat`)

This package trains a **small 2D CNN** on **log-magnitude Doppler spectrograms** built with **fixed preprocessing** \(P\): range FFT, **fixed range bin or band** (no adaptive max-energy by default), **fixed slow-time length** before STFT, fixed STFT parameters, then resize to **H×W**.

## What the model learns

The classifier estimates **\(P_{\text{exp}}(Y \mid X)\)** under this fixed \(P\), where **\(Y\)** is the **script / activity label** in the corpus (which recording was performed), not a clinically adjudicated “real fall” in the wild. See:

- [`../Chuy/03_causal_graph_falls.md`](../Chuy/03_causal_graph_falls.md) — phenomenon vs experimental DAG  
- [`../Chuy/04_nn_inputs_exclusions.md`](../Chuy/04_nn_inputs_exclusions.md) — inputs to exclude (metadata, filename tokens, selection-induced leakage)

**Subject ID** is used **only for splitting**, never as a model input.

## Data format

Glasgow-style `.dat`: **four header floats** (`fc` Hz, `Tsweep` ms, `NTS`, `Bw` Hz), then **interleaved real/imaginary** beat samples. See [`../Chuy/01_dataset_inventory.md`](../Chuy/01_dataset_inventory.md).

## Install

```bash
pip install -r requirements.txt
```

## Configuration

Edit [`../configs/default.yaml`](../configs/default.yaml). Important **causal defaults**:

| Setting | Role |
|--------|------|
| `range_bin_mode: fixed` | Avoids label-dependent bin selection (use `max_energy` only as an ablation). |
| `fixed_num_chirps` | Same slow-time length for all clips (mitigates 5 s vs 10 s walking cue). |
| STFT params | Fixed \(P\) across train/val/test. |

**Normalization:** mean/std of log-spectrogram pixels computed on **training files only**, applied to val/test.

## Train (subject-wise 80/10/10, shuffled before split)

```bash
python -m radar_cnn.train \
  --data_root /path/to/extracted/dat/files \
  --config configs/default.yaml \
  --output_dir runs/exp1 \
  --cache_dir runs/exp1/cache
```

**Best checkpoint:** `runs/exp1/best.pt` is saved when **`val_macro_f1`** on the validation set improves (default). Use `--best_metric val_loss` to restore loss-based selection.

**Early stopping:** default `--patience 8` stops training if `best_metric` does not improve for 8 epochs. Use `--patience 0` to always run the full `--epochs` (previous behavior).

**Shuffle-label pipeline test:** same command with `--shuffle_labels`. Training labels are permuted among samples (class histogram unchanged); validation/test labels stay true. Expect chance-level validation macro-F1 if inputs are not leaking label information.

**Batch normalization:** convolutional blocks already use `BatchNorm2d`. Optional `training.use_head_bn: true` in the YAML adds `BatchNorm1d` on the GAP feature vector before the linear classifier (can help with very small batches).

Run metadata (shuffle mode, patience, epochs run, early stop) is written to **`runs/exp1/train_meta.json`**.

Checkpoint payload includes `spec_cfg`, `mean`, `std`, `seed`, `split_fractions`, `cnn_base`, `use_head_bn`, `best_metric`, `shuffle_labels`.

Each epoch appends a row to **`runs/exp1/metrics.csv`**. Plot curves:

```bash
python -m radar_cnn.plot_metrics --csv runs/exp1/metrics.csv --out runs/exp1/metrics.png
```

Install `matplotlib` if needed: `pip install matplotlib`.

## Evaluate (global + per-subject CSV)

Uses the **same seed and split fractions** stored in the checkpoint so splits match training. Prints **accuracy**, **macro-F1**, **confusion matrix**, **classification_report**, and a **per-subject** table with macro-F1 and file counts (mean/std of subject macro-F1s on stdout).

Run **validation** and **test** to assess calibration vs generalization:

```bash
python -m radar_cnn.evaluate \
  --checkpoint runs/exp1/best.pt \
  --data_root /path/to/extracted/dat/files \
  --split val \
  --output_csv runs/exp1/eval_val_per_subject.csv

python -m radar_cnn.evaluate \
  --checkpoint runs/exp1/best.pt \
  --data_root /path/to/extracted/dat/files \
  --split test \
  --output_csv runs/exp1/eval_test_per_subject.csv
```

Interpretation: large **std** of per-subject macro-F1 or a few subjects near chance while others are near perfect suggests **heterogeneity** or **shortcut** behavior—review confusion matrices (especially fall vs high-motion classes) alongside these tables.

## Leave-one-subject-out (optional, expensive)

```bash
python -m radar_cnn.loso \
  --data_root /path/to/dat/files \
  --config configs/default.yaml \
  --epochs 15 \
  --max_folds 5
```

`--max_folds` limits subjects for debugging.

## Negative controls

```bash
python -m radar_cnn.controls --data_root /path/to/dat --mode subject
python -m radar_cnn.controls --data_root /path/to/dat --mode chirp_len
```

High accuracy on **subject** or **chirp length** alone indicates strong non-physics structure in the labels (see [`../Chuy/04_nn_inputs_exclusions.md`](../Chuy/04_nn_inputs_exclusions.md)).

## Caching

If `--cache_dir` is set, spectrograms are written under `{cache_dir}/{split_name}/`. Populate cache **after** splits are fixed (this implementation caches per split name).

## Module map

| Module | Role |
|--------|------|
| `io_dat.py` | Header + IQ load |
| `spectrogram.py` | Range FFT, fixed bin/band, temporal crop/pad, STFT, resize |
| `labels.py` | Parse `PxxAxxRxx` |
| `splits.py` | Discover files, subject-wise split |
| `dataset.py` | PyTorch dataset + train statistics |
| `model.py` | `SmallRadarCNN` |
| `train.py` | Weighted CE, augmentation |
| `evaluate.py` | Metrics + per-subject table |
| `loso.py` | Leave-one-subject-out |
| `controls.py` | Negative-control baselines |
