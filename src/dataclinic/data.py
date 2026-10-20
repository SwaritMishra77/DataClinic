"""Bundled datasets and contamination processes; clean labels are kept private."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Cohort:
    dataset: str
    mode: str
    seed: int
    train_ids: np.ndarray
    test_ids: np.ndarray
    x_train: np.ndarray
    y_noisy: np.ndarray
    y_private: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    corrupted: np.ndarray


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name == "digits":
        x, y = load_digits(return_X_y=True)
    elif name == "breast_cancer":
        x, y = load_breast_cancer(return_X_y=True)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.int64)


def contaminate(
    x: np.ndarray, y: np.ndarray, mode: str, rate: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Contaminate exactly round(rate*n) TRAIN labels. Never inspect test data."""
    if not 0 < rate < 0.5:
        raise ValueError("noise_rate must be in (0, 0.5)")
    if mode not in {"uniform", "paired", "boundary"}:
        raise ValueError(f"Unknown corruption mode: {mode}")
    rng = np.random.default_rng(seed)
    n = len(y)
    k = round(n * rate)
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError("Need at least two classes")

    if mode == "boundary":
        # Select images with a close rival class in the TRAIN set. Truth is used
        # only by the corruption generator and the later simulated annotator.
        scales = np.maximum(np.std(x, axis=0), 1.0)
        z = x / scales
        distances = cdist(z, z, metric="euclidean")
        different = y[:, None] != y[None, :]
        rival = np.where(different, distances, np.inf)
        nearest = np.argmin(rival, axis=1)
        rival_distance = rival[np.arange(n), nearest]
        within = np.where(~different, distances, np.inf)
        np.fill_diagonal(within, np.inf)
        own_distance = np.min(within, axis=1)
        ambiguity = own_distance / np.maximum(rival_distance, 1e-12)
        # Tiny random tie break; no test-set signal or model score is used.
        chosen = np.argsort(-(ambiguity + rng.uniform(0, 1e-9, n)))[:k]
        replacements = y[nearest[chosen]]
    else:
        # On binary data the paired setting is one-way contamination, unlike
        # uniform flipping. It is feasible only if that class has >=k rows.
        candidates = (np.flatnonzero(y == classes[0])
                      if mode == "paired" and len(classes) == 2 else np.arange(n))
        if len(candidates) < k:
            raise ValueError("Not enough source-class samples for paired noise")
        chosen = rng.choice(candidates, k, replace=False)
        if mode == "uniform":
            # Uniform over every *other* class, including binary datasets.
            old = np.searchsorted(classes, y[chosen])
            offset = rng.integers(1, len(classes), size=k)
            replacements = classes[(old + offset) % len(classes)]
        else:
            # Predeclared deterministic confusions. Binary classes swap.
            if len(classes) == 2:
                mapping = {int(classes[0]): classes[1], int(classes[1]): classes[0]}
            elif np.array_equal(classes, np.arange(10)):
                pairs = (6, 7, 7, 8, 9, 6, 5, 1, 3, 4)
                mapping = {int(i): int(pairs[i]) for i in range(10)}
            else:
                mapping = {int(c): classes[(i + 1) % len(classes)] for i, c in enumerate(classes)}
            replacements = np.array([mapping[int(label)] for label in y[chosen]])

    noisy = y.copy()
    noisy[chosen] = replacements
    changed = noisy != y
    if changed.sum() != k:
        raise AssertionError("Corruption generator failed to change exactly k labels")
    return noisy, changed


def make_cohort(
    name: str, mode: str, seed: int, rate: float = 0.2, test_size: float = 0.25
) -> Cohort:
    x, y = load_dataset(name)
    all_ids = np.arange(len(y))
    train, test = train_test_split(
        all_ids, test_size=test_size, stratify=y, random_state=seed
    )
    x_train, x_test = x[train], x[test]
    private = y[train]
    noisy, corrupted = contaminate(x_train, private, mode, rate, seed + 10007)
    return Cohort(
        name, mode, seed, train, test, x_train, noisy, private,
        x_test, y[test], corrupted
    )
