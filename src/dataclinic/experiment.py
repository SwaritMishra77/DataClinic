"""Paired comparisons of error ranking and simulated human label review."""

from dataclasses import dataclass
from pathlib import Path
import json
import platform
import time

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, log_loss

from . import __version__
from .data import Cohort, make_cohort
from .methods import METHODS, make_forest, make_logistic, rank_examples


@dataclass(frozen=True)
class Config:
    datasets: tuple[str, ...]
    modes: tuple[str, ...]
    seeds: tuple[int, ...]
    noise_rate: float
    test_size: float
    cv_folds: int
    dynamics_epochs: int
    budgets: tuple[int, ...]
    forest_trees: int
    secondary_budget: int

    @classmethod
    def from_file(cls, path: str | Path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("datasets", "modes", "seeds", "budgets"):
            raw[key] = tuple(raw[key])
        config = cls(**raw)
        if not config.datasets or not config.modes or not config.seeds:
            raise ValueError("Datasets, modes and seeds must be nonempty")
        if any(b < 1 for b in config.budgets) or len(set(config.budgets)) != len(config.budgets):
            raise ValueError("Budgets must be positive and distinct")
        if config.secondary_budget not in config.budgets:
            raise ValueError("secondary_budget must also appear in budgets")
        return config


def evaluate(model, x_train, y_train, x_test, y_test) -> dict[str, float]:
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    proba = model.predict_proba(x_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "log_loss": float(log_loss(y_test, proba, labels=model.classes_)),
    }


def review(cohort: Cohort, order: np.ndarray, budget: int) -> tuple[np.ndarray, int]:
    """A simulated annotator reveals TRUE labels only for inspected rows."""
    inspected = order[: min(budget, len(order))]
    corrected = cohort.y_noisy.copy()
    corrected[inspected] = cohort.y_private[inspected]
    found = int(cohort.corrupted[inspected].sum())
    return corrected, found


def case_study(cohort: Cohort, rankings: dict[str, np.ndarray], limit: int = 30):
    if cohort.dataset != "digits" or cohort.seed != 11:
        return None
    entries = {}
    for name, scores in rankings.items():
        order = np.argsort(-scores, kind="stable")[:limit]
        entries[name] = [
            {
                "sample_id": int(cohort.train_ids[i]),
                "pixels": cohort.x_train[i].reshape(8, 8).astype(int).tolist(),
                "recorded_label": int(cohort.y_noisy[i]),
                "reviewed_label": int(cohort.y_private[i]),
                "injected_error": bool(cohort.corrupted[i]),
                "rank": rank + 1,
                "score": round(float(scores[i]), 4),
            }
            for rank, i in enumerate(order)
        ]
    return entries


def one_cohort(cohort: Cohort, config: Config):
    rankings = rank_examples(
        cohort.x_train, cohort.y_noisy, cohort.seed,
        config.cv_folds, config.dynamics_epochs, config.forest_trees
    )
    baseline = {
        "logistic": evaluate(make_logistic(cohort.seed), cohort.x_train,
                             cohort.y_noisy, cohort.x_test, cohort.y_test),
        "forest": evaluate(make_forest(cohort.seed, config.forest_trees), cohort.x_train,
                           cohort.y_noisy, cohort.x_test, cohort.y_test),
    }
    keys = {
        "dataset": cohort.dataset,
        "mode": cohort.mode,
        "seed": cohort.seed,
        "train_n": len(cohort.y_noisy),
        "test_n": len(cohort.y_test),
        "errors_n": int(cohort.corrupted.sum()),
    }
    ranking_rows, review_rows = [], []
    oracle = np.r_[np.flatnonzero(cohort.corrupted), np.flatnonzero(~cohort.corrupted)]
    for method, score in [*rankings.items(), ("oracle_ceiling", None)]:
        order = oracle if score is None else np.argsort(-score, kind="stable")
        ranking_rows.append({
            **keys, "method": method,
            "average_precision": 1.0 if score is None else float(
                average_precision_score(cohort.corrupted, score)
            ),
        })
        for budget in config.budgets:
            fixed, found = review(cohort, order, budget)
            selected_n = min(budget, len(order))
            for model_name in ("logistic", "forest"):
                if model_name == "forest" and budget != config.secondary_budget:
                    continue
                model = (make_logistic(cohort.seed) if model_name == "logistic"
                         else make_forest(cohort.seed, config.forest_trees))
                after = evaluate(model, cohort.x_train, fixed, cohort.x_test, cohort.y_test)
                before = baseline[model_name]
                review_rows.append({
                    **keys, "method": method, "model": model_name,
                    "budget": budget, "reviewed_n": selected_n,
                    "found_n": found, "precision_at_budget": found / selected_n,
                    "recall_at_budget": found / keys["errors_n"],
                    "base_accuracy": before["accuracy"],
                    "after_accuracy": after["accuracy"],
                    "delta_accuracy": after["accuracy"] - before["accuracy"],
                    "base_macro_f1": before["macro_f1"],
                    "after_macro_f1": after["macro_f1"],
                    "delta_macro_f1": after["macro_f1"] - before["macro_f1"],
                    "base_log_loss": before["log_loss"],
                    "after_log_loss": after["log_loss"],
                    "delta_log_loss": after["log_loss"] - before["log_loss"],
                })
    return ranking_rows, review_rows, case_study(cohort, rankings)


def run(config: Config, output: str | Path):
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    ranking_rows, review_rows, cases = [], [], {}
    started = time.monotonic()
    for dataset in config.datasets:
        for mode in config.modes:
            for seed in config.seeds:
                print(f"{dataset}/{mode}/seed={seed}", flush=True)
                cohort = make_cohort(dataset, mode, seed, config.noise_rate, config.test_size)
                ranks, reviews, example = one_cohort(cohort, config)
                ranking_rows.extend(ranks)
                review_rows.extend(reviews)
                if example is not None:
                    cases[mode] = example

    ranking = pd.DataFrame(ranking_rows)
    review_data = pd.DataFrame(review_rows)
    ranking.to_csv(destination / "ranking.csv", index=False, float_format="%.8f")
    review_data.to_csv(destination / "review.csv", index=False, float_format="%.8f")
    (destination / "case_studies.json").write_text(
        json.dumps(cases, indent=2), encoding="utf-8"
    )
    metadata = {
        "version": __version__, "configuration": config.__dict__,
        "runtime_seconds": round(time.monotonic() - started, 2),
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
        },
        "cohorts": len(config.datasets) * len(config.modes) * len(config.seeds),
        "note": "Clean training labels were used only to inject errors, simulate review, and score rankings.",
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return ranking, review_data
