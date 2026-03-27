"""PyTorch Dataset: load .dat -> log spectrogram; optional disk cache per split."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from radar_cnn.io_dat import read_glasgow_dat
from radar_cnn.labels import activity_to_binary_label, activity_to_label, parse_filename
from radar_cnn.spectrogram import SpectrogramConfig, compute_log_spectrogram


def _cache_key(path: Path, spec_cfg: SpectrogramConfig) -> str:
    d = {
        "path": str(path.resolve()),
        "range_bin_mode": spec_cfg.range_bin_mode,
        "fixed_range_bin": spec_cfg.fixed_range_bin,
        "range_band": spec_cfg.range_band,
        "fixed_num_chirps": spec_cfg.fixed_num_chirps,
        "stft_nperseg": spec_cfg.stft_nperseg,
        "stft_noverlap": spec_cfg.stft_noverlap,
        "spec_hw": (spec_cfg.spec_height, spec_cfg.spec_width),
    }
    s = json.dumps(d, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:24]


def spectrogram_to_sequence_features(
    spec: np.ndarray,
    seq_len: int,
    frame_reduce: str = "mean",
) -> np.ndarray:
    """
    Convert (H, W) spectrogram to sequence (T, F) by slicing W into T chunks.
    F is H; each timestep pools one width chunk.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    if spec.ndim != 2:
        raise ValueError(f"Expected 2D spectrogram, got shape {spec.shape}")

    h, w = spec.shape
    edges = np.linspace(0, w, num=seq_len + 1, dtype=int)
    out = np.zeros((seq_len, h), dtype=np.float32)
    for t in range(seq_len):
        s = edges[t]
        e = edges[t + 1]
        if e <= s:
            e = min(w, s + 1)
        chunk = spec[:, s:e]
        if frame_reduce == "max":
            out[t] = np.max(chunk, axis=1)
        else:
            out[t] = np.mean(chunk, axis=1)
    return out


class RadarSpectrogramDataset(Dataset):
    """
    Returns (tensor 1xHxW, label int64, subject_id int for eval grouping).
    Normalization (mean/std) applied externally if train_stats provided.
    """

    def __init__(
        self,
        paths: list[Path | str],
        spec_cfg: SpectrogramConfig,
        train_mean: float | None = None,
        train_std: float | None = None,
        cache_dir: Path | str | None = None,
        split_name: str = "train",
        label_override: np.ndarray | None = None,
        binary_labels: bool = False,
    ) -> None:
        self.paths = [Path(p) for p in paths]
        self.spec_cfg = spec_cfg
        self.train_mean = train_mean
        self.train_std = train_std
        self.binary_labels = bool(binary_labels)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.split_name = split_name
        if label_override is not None:
            lo = np.asarray(label_override, dtype=np.int64).reshape(-1)
            if len(lo) != len(self.paths):
                raise ValueError("label_override length must match paths")
            self.label_override = lo
        else:
            self.label_override = None
        if self.cache_dir:
            self.cache_dir = self.cache_dir / split_name
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.paths)

    def _load_spec(self, path: Path) -> np.ndarray:
        if self.cache_dir:
            key = _cache_key(path, self.spec_cfg)
            cpath = self.cache_dir / f"{key}.npy"
            if cpath.exists():
                return np.load(cpath)

        data, hdr = read_glasgow_dat(str(path))
        tc = hdr["tsweep_s"]
        spec = compute_log_spectrogram(data, tc, self.spec_cfg)

        if self.cache_dir:
            key = _cache_key(path, self.spec_cfg)
            cpath = self.cache_dir / f"{key}.npy"
            np.save(cpath, spec)

        return spec

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        spec = self._load_spec(path)
        x = torch.from_numpy(spec).float().unsqueeze(0)
        if self.train_mean is not None and self.train_std is not None:
            x = (x - self.train_mean) / (self.train_std + 1e-8)

        if self.label_override is not None:
            y = int(self.label_override[idx])
        else:
            _, activity, _ = parse_filename(path)
            if self.binary_labels:
                y = activity_to_binary_label(activity)
            else:
                y = activity_to_label(activity)
        sid, _, _ = parse_filename(path)

        return x, torch.tensor(y, dtype=torch.long), sid


class RadarSequenceDataset(Dataset):
    """
    Returns (sequence tensor TxF, label int64, subject_id int).
    Sequence is derived from the same cached spectrogram path used by CNN runs.
    """

    def __init__(
        self,
        paths: list[Path | str],
        spec_cfg: SpectrogramConfig,
        seq_len: int,
        frame_reduce: str = "mean",
        binary_labels: bool = True,
        train_mean: float | None = None,
        train_std: float | None = None,
        cache_dir: Path | str | None = None,
        split_name: str = "train",
        label_override: np.ndarray | None = None,
    ) -> None:
        if frame_reduce not in {"mean", "max"}:
            raise ValueError("frame_reduce must be 'mean' or 'max'")
        self.paths = [Path(p) for p in paths]
        self.spec_cfg = spec_cfg
        self.seq_len = int(seq_len)
        self.frame_reduce = frame_reduce
        self.binary_labels = bool(binary_labels)
        self.train_mean = train_mean
        self.train_std = train_std
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.split_name = split_name
        if label_override is not None:
            lo = np.asarray(label_override, dtype=np.int64).reshape(-1)
            if len(lo) != len(self.paths):
                raise ValueError("label_override length must match paths")
            self.label_override = lo
        else:
            self.label_override = None
        if self.cache_dir:
            self.cache_dir = self.cache_dir / split_name
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.paths)

    def _load_spec(self, path: Path) -> np.ndarray:
        if self.cache_dir:
            key = _cache_key(path, self.spec_cfg)
            cpath = self.cache_dir / f"{key}.npy"
            if cpath.exists():
                return np.load(cpath)

        data, hdr = read_glasgow_dat(str(path))
        tc = hdr["tsweep_s"]
        spec = compute_log_spectrogram(data, tc, self.spec_cfg)

        if self.cache_dir:
            key = _cache_key(path, self.spec_cfg)
            cpath = self.cache_dir / f"{key}.npy"
            np.save(cpath, spec)
        return spec

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        spec = self._load_spec(path)
        if self.train_mean is not None and self.train_std is not None:
            spec = (spec - self.train_mean) / (self.train_std + 1e-8)

        x = spectrogram_to_sequence_features(spec, self.seq_len, self.frame_reduce)
        x_t = torch.from_numpy(x).float()

        if self.label_override is not None:
            y = int(self.label_override[idx])
        else:
            _, activity, _ = parse_filename(path)
            if self.binary_labels:
                y = activity_to_binary_label(activity)
            else:
                y = activity_to_label(activity)
        sid, _, _ = parse_filename(path)
        return x_t, torch.tensor(y, dtype=torch.long), sid


def compute_train_statistics(
    paths: list[Path],
    spec_cfg: SpectrogramConfig,
    cache_dir: Path | None,
    max_files: int | None = None,
    verbose: bool = False,
) -> tuple[float, float]:
    """
    Mean and std of log-spectrogram pixels over training files (single pass).

    This pass reads each .dat (slow: large ASCII), builds spectrogram, aggregates
    global mean/var. Use cache_dir so subsequent epochs reuse .npy caches.
    """
    acc_sum = 0.0
    acc_sq = 0.0
    n_pix = 0
    use_paths = paths[: max_files] if max_files else paths
    n_total = len(use_paths)
    if verbose:
        print(
            f"[train_stats] Computing mean/std over {n_total} files "
            f"(first full pass; use --cache_dir to speed reruns).",
            file=sys.stderr,
            flush=True,
        )
    ds_temp = RadarSpectrogramDataset(
        use_paths,
        spec_cfg,
        train_mean=None,
        train_std=None,
        cache_dir=cache_dir,
        split_name="train_stats",
    )
    pbar = tqdm(
        range(len(ds_temp)),
        desc="train_stats (global mean/std)",
        unit="file",
        disable=not sys.stdout.isatty(),
    )
    for i in pbar:
        path = ds_temp.paths[i]
        if verbose:
            pbar.set_postfix(file=path.name[:48], refresh=False)
        spec, _, _ = ds_temp[i]
        x = spec.squeeze().numpy()
        acc_sum += float(x.sum())
        acc_sq += float(np.square(x).sum())
        n_pix += x.size
        if verbose and (i % max(1, n_total // 20) == 0 or i == n_total - 1):
            print(
                f"[train_stats] {i + 1}/{n_total}  {path}  "
                f"running_mean≈{acc_sum / max(n_pix, 1):.4f}",
                file=sys.stderr,
                flush=True,
            )
    mean = acc_sum / max(n_pix, 1)
    var = acc_sq / max(n_pix, 1) - mean * mean
    std = float(np.sqrt(max(var, 1e-12)))
    return mean, std


def compute_train_statistics_sequence(
    paths: list[Path],
    spec_cfg: SpectrogramConfig,
    seq_len: int,
    frame_reduce: str,
    cache_dir: Path | None,
    max_files: int | None = None,
    verbose: bool = False,
) -> tuple[float, float]:
    """
    Mean/std over sequence features (T,F) built from training spectrograms.
    """
    acc_sum = 0.0
    acc_sq = 0.0
    n_pix = 0
    use_paths = paths[: max_files] if max_files else paths
    n_total = len(use_paths)
    if verbose:
        print(
            f"[train_stats_seq] Computing mean/std over {n_total} files.",
            file=sys.stderr,
            flush=True,
        )

    ds_temp = RadarSequenceDataset(
        use_paths,
        spec_cfg,
        seq_len=seq_len,
        frame_reduce=frame_reduce,
        binary_labels=True,
        train_mean=None,
        train_std=None,
        cache_dir=cache_dir,
        split_name="train_stats_seq",
    )
    pbar = tqdm(
        range(len(ds_temp)),
        desc="train_stats_seq (global mean/std)",
        unit="file",
        disable=not sys.stdout.isatty(),
    )
    for i in pbar:
        x, _, _ = ds_temp[i]
        a = x.numpy()
        acc_sum += float(a.sum())
        acc_sq += float(np.square(a).sum())
        n_pix += a.size
        if verbose and (i % max(1, n_total // 20) == 0 or i == n_total - 1):
            print(
                f"[train_stats_seq] {i + 1}/{n_total} running_mean≈{acc_sum / max(n_pix, 1):.4f}",
                file=sys.stderr,
                flush=True,
            )
    mean = acc_sum / max(n_pix, 1)
    var = acc_sq / max(n_pix, 1) - mean * mean
    std = float(np.sqrt(max(var, 1e-12)))
    return mean, std
