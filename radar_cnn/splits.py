"""Subject-wise splits with shuffled file order."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np


def discover_dat_files(data_root: str | Path) -> list[Path]:
    root = Path(data_root)
    files = sorted(root.rglob("*.dat"))
    return [p for p in files if p.is_file()]


def group_files_by_subject(
    paths: list[Path],
    parse_fn,
) -> dict[int, list[Path]]:
    by_sub: dict[int, list[Path]] = defaultdict(list)
    for p in paths:
        sid, _, _ = parse_fn(p)
        by_sub[sid].append(p)
    return dict(by_sub)


def subject_train_val_test(
    paths: list[Path],
    parse_fn,
    seed: int,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> tuple[list[Path], list[Path], list[Path]]:
    """
    Shuffle files first, then split *subjects* so no subject appears in more than one split.
    """
    assert abs(sum(fractions) - 1.0) < 1e-6
    rng = random.Random(seed)
    shuffled = paths[:]
    rng.shuffle(shuffled)

    by_sub = group_files_by_subject(shuffled, parse_fn)
    subjects = sorted(by_sub.keys())
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = int(round(n * fractions[0]))
    n_val = int(round(n * fractions[1]))
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0
    # adjust if rounding
    while n_train + n_val + n_test > n:
        n_train -= 1
    while n_train + n_val + n_test < n:
        n_train += 1

    sub_train = set(subjects[:n_train])
    sub_val = set(subjects[n_train : n_train + n_val])
    sub_test = set(subjects[n_train + n_val :])

    train, val, test = [], [], []
    for p in shuffled:
        sid, _, _ = parse_fn(p)
        if sid in sub_train:
            train.append(p)
        elif sid in sub_val:
            val.append(p)
        else:
            test.append(p)

    return train, val, test


def subject_ids_in_paths(paths: list[Path], parse_fn) -> set[int]:
    return {parse_fn(p)[0] for p in paths}


def assert_disjoint_subject_splits(
    train_paths: list[Path],
    val_paths: list[Path],
    test_paths: list[Path],
    parse_fn,
) -> None:
    """Fail fast if the same subject id appears in more than one split."""
    st = subject_ids_in_paths(train_paths, parse_fn)
    sv = subject_ids_in_paths(val_paths, parse_fn)
    se = subject_ids_in_paths(test_paths, parse_fn)
    if st & sv or st & se or sv & se:
        raise AssertionError(
            f"Subject overlap between splits: train∩val={st & sv}, train∩test={st & se}, val∩test={sv & se}"
        )


def split_class_counts(paths: list[Path], parse_fn, label_fn) -> dict[int, int]:
    from collections import Counter

    c: Counter[int] = Counter()
    for p in paths:
        _, act, _ = parse_fn(p)
        c[label_fn(act)] += 1
    return dict(c)
