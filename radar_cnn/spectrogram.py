"""Fixed preprocessing P: range FFT, fixed bin/band, temporal crop, STFT, resize to H×W."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.ndimage import zoom
from scipy.signal import stft, windows


@dataclass
class SpectrogramConfig:
    """All parameters fixed a priori (no per-file max-energy bin in default path)."""

    # Range FFT
    range_bin_mode: Literal["fixed", "band", "max_energy"] = "fixed"
    fixed_range_bin: int = 32  # label-agnostic default (middle of positive half for Ns=128 -> bins 0..63)
    range_band: tuple[int, int] = (28, 36)  # inclusive, used if range_bin_mode == "band"

    # Slow-time length before STFT (fixes 5s vs 10s walking leakage)
    fixed_num_chirps: int = 5000  # ~5 s at 1 ms/chirp

    # STFT (slow-time = chirp index)
    stft_nperseg: int = 128
    stft_noverlap: int = 96
    stft_window: Literal["hann", "hamming"] = "hann"

    # Output log-magnitude grid
    spec_height: int = 128
    spec_width: int = 128
    log_eps: float = 1e-6

def _window(name: str, n: int) -> np.ndarray:
    if name == "hann":
        return windows.hann(n, sym=False)
    if name == "hamming":
        return windows.hamming(n, sym=False)
    raise ValueError(name)


def _temporal_fix_length(sig: np.ndarray, n_target: int) -> np.ndarray:
    """Center crop or zero-pad complex slow-time signal to n_target samples."""
    n = sig.shape[0]
    if n == n_target:
        return sig
    if n > n_target:
        start = (n - n_target) // 2
        return sig[start : start + n_target].copy()
    pad_total = n_target - n
    pl = pad_total // 2
    pr = pad_total - pl
    return np.pad(sig, (pl, pr), mode="constant", constant_values=0)


def range_fft_positive(data_chirps: np.ndarray) -> tuple[np.ndarray, int]:
    """
    data_chirps: (num_chirps, Ns)
    Returns range_fft (num_chirps, Nb), Nb = Ns//2
    """
    ns = data_chirps.shape[1]
    win = _window("hann", ns)
    tmp = np.fft.fft(data_chirps * win[None, :], axis=1)
    nb = ns // 2
    return tmp[:, :nb], nb


def slow_time_at_range(
    range_fft: np.ndarray,
    cfg: SpectrogramConfig,
    nb: int,
) -> np.ndarray:
    """Complex slow-time series at fixed bin, band mean, or max-energy (ablation)."""
    if cfg.range_bin_mode == "max_energy":
        power = np.mean(np.abs(range_fft) ** 2, axis=0)
        bi = int(np.argmax(power))
        bi = max(0, min(nb - 1, bi))
        return range_fft[:, bi].astype(np.complex64)

    if cfg.range_bin_mode == "band":
        b0, b1 = cfg.range_band
        b0 = max(0, min(nb - 1, b0))
        b1 = max(b0, min(nb - 1, b1))
        return np.mean(range_fft[:, b0 : b1 + 1], axis=1).astype(np.complex64)

    # fixed
    bi = cfg.fixed_range_bin
    bi = max(0, min(nb - 1, bi))
    return range_fft[:, bi].astype(np.complex64)


def compute_log_spectrogram(
    data_chirps: np.ndarray,
    tc_s: float,
    cfg: SpectrogramConfig,
) -> np.ndarray:
    """
    Returns float32 array (H, W) log-magnitude spectrogram.
    """
    range_fft, nb = range_fft_positive(data_chirps)
    slow = slow_time_at_range(range_fft, cfg, nb)
    slow = _temporal_fix_length(slow, cfg.fixed_num_chirps)

    fs_slow = 1.0 / tc_s
    win = _window(cfg.stft_window, cfg.stft_nperseg)
    _, _, Z = stft(
        slow,
        fs=fs_slow,
        window=win,
        nperseg=cfg.stft_nperseg,
        noverlap=cfg.stft_noverlap,
        return_onesided=False,
    )
    mag = np.abs(Z).astype(np.float32)
    log_spec = np.log(mag + cfg.log_eps)

    h0, w0 = log_spec.shape
    zh = cfg.spec_height / h0
    zw = cfg.spec_width / w0
    if zh != 1.0 or zw != 1.0:
        log_spec = zoom(log_spec, (zh, zw), order=1).astype(np.float32)

    # Ensure exact size (zoom can be off by 1)
    log_spec = _crop_pad_2d(log_spec, cfg.spec_height, cfg.spec_width)
    return log_spec


def _crop_pad_2d(a: np.ndarray, h: int, w: int) -> np.ndarray:
    H, W = a.shape
    out = np.zeros((h, w), dtype=np.float32)
    sh = min(H, h)
    sw = min(W, w)
    oh = (H - sh) // 2 if H >= h else 0
    ow = (W - sw) // 2 if W >= w else 0
    dh = (h - sh) // 2 if H < h else 0
    dw = (w - sw) // 2 if W < w else 0
    src = a[oh : oh + sh, ow : ow + sw]
    out[dh : dh + sh, dw : dw + sw] = src
    return out
