"""Load YAML config into SpectrogramConfig + training dict."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from radar_cnn.spectrogram import SpectrogramConfig


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def spectrogram_from_dict(d: dict[str, Any]) -> SpectrogramConfig:
    return SpectrogramConfig(
        range_bin_mode=d.get("range_bin_mode", "fixed"),
        fixed_range_bin=int(d.get("fixed_range_bin", 32)),
        range_band=tuple(d.get("range_band", (28, 36))),
        fixed_num_chirps=int(d.get("fixed_num_chirps", 5000)),
        stft_nperseg=int(d.get("stft_nperseg", 128)),
        stft_noverlap=int(d.get("stft_noverlap", 96)),
        stft_window=d.get("stft_window", "hann"),
        spec_height=int(d.get("spec_height", 128)),
        spec_width=int(d.get("spec_width", 128)),
        log_eps=float(d.get("log_eps", 1e-6)),
    )
