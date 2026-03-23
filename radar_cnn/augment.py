"""Training-time augmentation on log-spectrogram tensors."""

from __future__ import annotations

import torch


def augment_spectrogram(
    x: torch.Tensor,
    amp_log_gain_range: float = 0.15,
    noise_std: float = 0.02,
) -> torch.Tensor:
    """
    x: (N,1,H,W) or (1,H,W)
    Multiplicative amplitude in log domain ~ additive shift; Gaussian noise.
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)
    device = x.device
    gain = (torch.rand(x.size(0), 1, 1, 1, device=device) * 2 - 1) * amp_log_gain_range
    x = x + gain
    noise = torch.randn_like(x) * noise_std
    return x + noise
