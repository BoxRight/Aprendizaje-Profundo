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
from radar_cnn.labels import activity_to_label, parse_filename
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
    ) -> None:
        self.paths = [Path(p) for p in paths]
        self.spec_cfg = spec_cfg
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
        x = torch.from_numpy(spec).float().unsqueeze(0)
        if self.train_mean is not None and self.train_std is not None:
            x = (x - self.train_mean) / (self.train_std + 1e-8)

        if self.label_override is not None:
            y = int(self.label_override[idx])
        else:
            _, activity, _ = parse_filename(path)
            y = activity_to_label(activity)
        sid, _, _ = parse_filename(path)

        return x, torch.tensor(y, dtype=torch.long), sid


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
