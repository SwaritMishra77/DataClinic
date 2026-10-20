"""Tests focus on leakage boundaries and real pipeline behavior."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from dataclinic.audit import audit_outputs
from dataclinic.data import contaminate, make_cohort
from dataclinic.experiment import Config, review, run
from dataclinic.methods import rank_examples
from dataclinic.report import write_report


class DataClinicTests(unittest.TestCase):
    def test_split_and_corruption_boundaries(self):
        for name in ("digits", "breast_cancer"):
            for mode in ("uniform", "paired", "boundary"):
                c = make_cohort(name, mode, 11)
                self.assertEqual(len(np.intersect1d(c.train_ids, c.test_ids)), 0)
                self.assertEqual(int(c.corrupted.sum()), round(.2 * len(c.train_ids)))
                self.assertTrue(np.array_equal(c.corrupted, c.y_noisy != c.y_private))
                again = make_cohort(name, mode, 11)
                self.assertTrue(np.array_equal(c.y_noisy, again.y_noisy))

    def test_review_is_budgeted_and_never_changes_unselected_rows(self):
        c = make_cohort("digits", "boundary", 11)
        order = np.arange(len(c.y_noisy))[::-1]
        corrected, found = review(c, order, 25)
        self.assertTrue(np.array_equal(corrected[order[25:]], c.y_noisy[order[25:]]))
        self.assertTrue(np.array_equal(corrected[order[:25]], c.y_private[order[:25]]))
        self.assertEqual(found, int(c.corrupted[order[:25]].sum()))
        self.assertTrue(np.array_equal(c.y_noisy, make_cohort("digits", "boundary", 11).y_noisy))

    def test_binary_paired_is_distinct_from_uniform(self):
        c = make_cohort("breast_cancer", "paired", 23)
        self.assertTrue(np.all(c.y_private[c.corrupted] == 0))
        self.assertTrue(np.all(c.y_noisy[c.corrupted] == 1))
        u = make_cohort("breast_cancer", "uniform", 23)
        self.assertTrue(np.any(u.y_private[u.corrupted] == 1))

    def test_rankers_only_need_noisy_training_input(self):
        c = make_cohort("digits", "uniform", 11)
        original = rank_examples(c.x_train, c.y_noisy, 11, epochs=2, trees=10)
        self.assertEqual(len(original), 6)
        for score in original.values():
            self.assertEqual(len(score), len(c.y_noisy))
            self.assertTrue(np.isfinite(score).all())
        copy = rank_examples(c.x_train, c.y_noisy, 11, epochs=2, trees=10)
        for key in original:
            self.assertTrue(np.array_equal(original[key], copy[key]))

    def test_end_to_end_artifacts_match_rows(self):
        with TemporaryDirectory() as temp:
            out, docs = Path(temp) / "results", Path(temp) / "site"
            cfg = Config.from_file(Path(__file__).resolve().parents[1] / "configs/smoke.json")
            ranks, reviews = run(cfg, out)
            write_report(ranks, reviews, out, docs)
            self.assertEqual(audit_outputs(out, docs)["review_rows"], 21)
            self.assertEqual(len(ranks), 7)
            self.assertEqual(len(reviews), 7 * (len(cfg.budgets) + 1))
            self.assertTrue((docs / "index.html").exists())
            self.assertNotIn("__DATACLINIC_DATA__", (docs / "index.html").read_text())
            self.assertFalse(pd.read_csv(out / "review.csv").isna().any().any())
            cases = json.loads((out / "case_studies.json").read_text())
            self.assertEqual(len(cases["uniform"]["rank_fusion"]), 30)


if __name__ == "__main__":
    unittest.main()
