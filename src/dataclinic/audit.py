"""Check that saved outputs match the fixed protocol and each other."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import make_cohort
from .methods import METHODS
from .report import summarize


def audit_outputs(result_dir: str | Path, docs_dir: str | Path) -> dict:
    root = Path(result_dir)
    config = json.loads((root / "metadata.json").read_text(encoding="utf-8"))["configuration"]
    rank = pd.read_csv(root / "ranking.csv")
    review = pd.read_csv(root / "review.csv")
    summary = pd.read_csv(root / "summary.csv")
    rank_summary = pd.read_csv(root / "ranking_summary.csv")
    names = (*METHODS, "oracle_ceiling")
    expected_cohorts = len(config["datasets"]) * len(config["modes"]) * len(config["seeds"])
    expected_rank_rows = expected_cohorts * len(names)
    expected_review_rows = expected_cohorts * len(names) * (len(config["budgets"]) + 1)
    assert len(rank) == expected_rank_rows, "Missing or extra ranking rows"
    assert len(review) == expected_review_rows, "Missing or extra review rows"
    assert not rank.isna().any().any() and not review.isna().any().any(), "Missing values"
    assert not rank.duplicated(["dataset", "mode", "seed", "method"]).any()
    assert not review.duplicated(["dataset", "mode", "seed", "method", "model", "budget"]).any()
    assert set(rank.method) == set(names) == set(review.method)
    assert np.allclose(review.precision_at_budget, review.found_n / review.reviewed_n)
    assert np.allclose(review.recall_at_budget, review.found_n / review.errors_n)
    assert np.allclose(review.delta_accuracy, review.after_accuracy - review.base_accuracy,
                       atol=2e-8)
    assert review.precision_at_budget.between(0, 1).all()
    assert rank.average_precision.between(0, 1).all()
    assert review.found_n.le(review.reviewed_n).all()
    assert review.found_n.le(review.errors_n).all()
    oracle = review[review.method == "oracle_ceiling"]
    assert np.array_equal(oracle.found_n.to_numpy(),
                          np.minimum(oracle.reviewed_n.to_numpy(), oracle.errors_n.to_numpy()))
    for _, rows in review.groupby(["dataset", "mode", "seed", "model"]):
        assert rows.base_accuracy.nunique() == 1, "Baselines vary across methods"
        assert rows.base_log_loss.nunique() == 1
    expected_summary, expected_ranks = summarize(rank, review)
    keys = ["dataset", "mode", "model", "budget", "method"]
    keys_rank = ["dataset", "mode", "method"]
    pd.testing.assert_frame_equal(
        summary.sort_values(keys).reset_index(drop=True),
        expected_summary.sort_values(keys).reset_index(drop=True),
        check_exact=False, atol=2e-7, rtol=0
    )
    pd.testing.assert_frame_equal(
        rank_summary.sort_values(keys_rank).reset_index(drop=True),
        expected_ranks.sort_values(keys_rank).reset_index(drop=True),
        check_exact=False, atol=2e-7, rtol=0
    )
    cases = json.loads((root / "case_studies.json").read_text(encoding="utf-8"))
    if "digits" in config["datasets"] and 11 in config["seeds"]:
        for mode in config["modes"]:
            cohort = make_cohort("digits", mode, 11, config["noise_rate"], config["test_size"])
            lookup = {int(idx): pos for pos, idx in enumerate(cohort.train_ids)}
            for method in METHODS:
                entries = cases[mode][method]
                assert len(entries) == 30
                for entry in entries:
                    pos = lookup[entry["sample_id"]]
                    assert entry["pixels"] == cohort.x_train[pos].reshape(8, 8).astype(int).tolist()
                    assert entry["recorded_label"] == int(cohort.y_noisy[pos])
                    assert entry["reviewed_label"] == int(cohort.y_private[pos])
                    assert entry["injected_error"] == bool(cohort.corrupted[pos])
    html = (Path(docs_dir) / "index.html").read_text(encoding="utf-8")
    assert "__DATACLINIC_DATA__" not in html and "case_studies" not in html[:500]
    assert (root / "precision_at_100.png").stat().st_size > 1000
    assert (root / "detection_vs_accuracy.png").stat().st_size > 1000
    return {"cohorts": expected_cohorts, "ranking_rows": len(rank),
            "review_rows": len(review), "case_study_modes": len(cases)}
