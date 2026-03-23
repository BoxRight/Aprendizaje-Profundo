"""Negative-control baselines: predict class from non-radar covariates only."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from radar_cnn.io_dat import read_glasgow_dat
from radar_cnn.labels import activity_to_label, parse_filename
from radar_cnn.splits import discover_dat_files, subject_train_val_test


class SubjectOnlyDataset(Dataset):
    """subject index (0..S-1), label."""

    def __init__(
        self,
        paths: list[Path],
        subject_to_idx: dict[int, int],
    ) -> None:
        self.paths = paths
        self.subject_to_idx = subject_to_idx

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        sid, act, _ = parse_filename(p)
        y = activity_to_label(act)
        u = self.subject_to_idx[sid]
        return torch.tensor(u, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class ChirpLenOnlyDataset(Dataset):
    """Normalized clip length (num chirps) only."""

    def __init__(self, paths: list[Path], max_chirps: float) -> None:
        self.paths = paths
        self.max_chirps = max_chirps

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        data, _ = read_glasgow_dat(str(p))
        nc = float(data.shape[0])
        x = torch.tensor([nc / self.max_chirps], dtype=torch.float32)
        _, act, _ = parse_filename(p)
        y = activity_to_label(act)
        return x, torch.tensor(y, dtype=torch.long)


class SubjectEmbeddingModel(nn.Module):
    def __init__(self, num_subjects: int, num_classes: int = 6, dim: int = 64) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_subjects, dim)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.fc(self.emb(idx))


class MLPScalar(nn.Module):
    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--mode", type=str, choices=["subject", "chirp_len"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-2)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    all_files = discover_dat_files(args.data_root)
    train_p, val_p, _ = subject_train_val_test(
        all_files,
        parse_filename,
        seed=args.seed,
        fractions=(0.8, 0.1, 0.1),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "subject":
        subs = sorted({parse_filename(p)[0] for p in train_p + val_p})
        sub_to_idx = {s: i for i, s in enumerate(subs)}
        train_ds = SubjectOnlyDataset(train_p, sub_to_idx)
        val_ds = SubjectOnlyDataset(val_p, sub_to_idx)
        model = SubjectEmbeddingModel(len(subs), num_classes=6, dim=64).to(device)
    else:
        max_c = 0
        for p in train_p + val_p:
            d, _ = read_glasgow_dat(str(p))
            max_c = max(max_c, d.shape[0])
        train_ds = ChirpLenOnlyDataset(train_p, max_c)
        val_ds = ChirpLenOnlyDataset(val_p, max_c)
        model = MLPScalar(num_classes=6).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    for _ in range(args.epochs):
        model.train()
        for batch in train_loader:
            if args.mode == "subject":
                x, y = batch
                x = x.to(device)
            else:
                x, y = batch
                x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            if args.mode == "subject":
                x, y = batch
                x = x.to(device)
            else:
                x, y = batch
                x = x.to(device)
            y = y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    print(f"Negative control [{args.mode}] val accuracy: {correct/total:.4f} (n={total})")
    print("High accuracy here suggests label structure predictable without radar.")


if __name__ == "__main__":
    main()
