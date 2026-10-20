"""Predeclared label-review rankings; none can access the clean labels."""

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


METHODS = (
    "random", "oof_logistic", "neighbors", "oof_forest",
    "training_margin", "rank_fusion",
)


def make_logistic(seed: int):
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=0.2, max_iter=700, random_state=seed)
    )


def make_forest(seed: int, trees: int):
    return RandomForestClassifier(
        n_estimators=trees, min_samples_leaf=2, n_jobs=1, random_state=seed
    )


def oof_self_probability(estimator, x: np.ndarray, y: np.ndarray, folds: int, seed: int):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    probability = cross_val_predict(
        estimator, x, y, cv=cv, method="predict_proba", n_jobs=1
    )
    return probability[np.arange(len(y)), y]


def neighbor_score(x: np.ndarray, y: np.ndarray, k: int = 7):
    # Leave-one-out nearest neighbors; the example never votes for itself.
    scaled = StandardScaler().fit_transform(x)
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(y))).fit(scaled)
    indices = nn.kneighbors(scaled, return_distance=False)
    votes = np.empty(len(y), dtype=float)
    for i, row in enumerate(indices):
        others = row[row != i][:k]
        votes[i] = np.mean(y[others] == y[i])
    return 1.0 - votes


def training_margin_score(x: np.ndarray, y: np.ndarray, seed: int, epochs: int):
    # A compact neural network records a class margin after every epoch.
    # This is an adaptation of training-dynamics ranking, not exact AUM.
    scaled = StandardScaler().fit_transform(x)
    net = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
        alpha=1e-3, batch_size=64, random_state=seed
    )
    classes = np.unique(y)
    margins = np.zeros(len(y), dtype=float)
    row = np.arange(len(y))
    for _ in range(epochs):
        net.partial_fit(scaled, y, classes=classes)
        probabilities = net.predict_proba(scaled)
        own = probabilities[row, y]
        rival = probabilities.copy()
        rival[row, y] = -np.inf
        margins += own - rival.max(axis=1)
    return -margins / epochs


def rank_examples(
    x: np.ndarray, y: np.ndarray, seed: int, folds: int = 3,
    epochs: int = 20, trees: int = 60
) -> dict[str, np.ndarray]:
    if epochs < 1 or trees < 1:
        raise ValueError("epochs and trees must be positive")
    if np.min(np.bincount(y)) < folds:
        raise ValueError("Each noisy class needs at least cv_folds examples")
    rng = np.random.default_rng(seed + 123457)
    rankings = {
        "random": rng.random(len(y)),
        "oof_logistic": 1.0 - oof_self_probability(
            make_logistic(seed), x, y, folds, seed
        ),
        "neighbors": neighbor_score(x, y),
        "oof_forest": 1.0 - oof_self_probability(
            make_forest(seed, trees), x, y, folds, seed
        ),
        "training_margin": training_margin_score(x, y, seed, epochs),
    }
    # Equal-weight rank fusion is fixed in advance, with no fitted weights.
    components = ("oof_logistic", "neighbors", "training_margin")
    rankings["rank_fusion"] = np.mean(
        [rankdata(rankings[name], method="average") / len(y) for name in components],
        axis=0,
    )
    for name, score in rankings.items():
        if len(score) != len(y) or not np.isfinite(score).all():
            raise ValueError(f"Invalid score from {name}")
    return rankings
