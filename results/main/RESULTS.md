# DataClinic results

Generated from the committed experiment CSV files. These are completed runs, not projected outcomes.

- Cohorts: 30 (5 split/noise seeds).
- Noise rate: 20% of training labels; clean test labels untouched.
- Runtime in the recorded environment: 52.5 seconds.
- The table uses 100 reviews, with the logistic outcome model. Changes are percentage points of test accuracy relative to the same seed's contaminated baseline.
- The oracle row is a hindsight ceiling for finding errors, not necessarily the highest possible test accuracy; it is not a deployable method.

## Matched review comparison

| Dataset | Noise | Ranking | Precision @100 | Test accuracy change | AP |
|---|---|---|---:|---:|---:|
| digits | uniform | Random review | 16.0% | +0.36 pp | 0.198 |
| digits | uniform | Out-of-fold logistic | 97.0% | +0.13 pp | 0.944 |
| digits | uniform | Neighbour agreement | 97.2% | +1.38 pp | 0.965 |
| digits | uniform | Out-of-fold forest | 100.0% | +0.67 pp | 0.990 |
| digits | uniform | Neural training margin | 100.0% | +0.31 pp | 0.988 |
| digits | uniform | Fixed rank fusion | 100.0% | +0.09 pp | 0.993 |
| digits | uniform | Hindsight ceiling | 100.0% | +1.38 pp | 1.000 |
| digits | paired | Random review | 16.0% | +0.53 pp | 0.198 |
| digits | paired | Out-of-fold logistic | 82.2% | +1.78 pp | 0.741 |
| digits | paired | Neighbour agreement | 90.2% | +2.36 pp | 0.841 |
| digits | paired | Out-of-fold forest | 90.6% | +2.00 pp | 0.844 |
| digits | paired | Neural training margin | 98.8% | +1.73 pp | 0.951 |
| digits | paired | Fixed rank fusion | 96.2% | +1.56 pp | 0.936 |
| digits | paired | Hindsight ceiling | 100.0% | +3.82 pp | 1.000 |
| digits | boundary | Random review | 20.8% | +1.02 pp | 0.202 |
| digits | boundary | Out-of-fold logistic | 95.0% | +1.42 pp | 0.871 |
| digits | boundary | Neighbour agreement | 95.4% | +2.31 pp | 0.885 |
| digits | boundary | Out-of-fold forest | 95.8% | +2.93 pp | 0.892 |
| digits | boundary | Neural training margin | 98.4% | +1.33 pp | 0.899 |
| digits | boundary | Fixed rank fusion | 98.4% | +1.29 pp | 0.932 |
| digits | boundary | Hindsight ceiling | 100.0% | +3.47 pp | 1.000 |
| breast_cancer | uniform | Random review | 21.0% | +0.70 pp | 0.212 |
| breast_cancer | uniform | Out-of-fold logistic | 78.8% | +2.66 pp | 0.917 |
| breast_cancer | uniform | Neighbour agreement | 76.4% | +2.24 pp | 0.869 |
| breast_cancer | uniform | Out-of-fold forest | 78.8% | +2.24 pp | 0.929 |
| breast_cancer | uniform | Neural training margin | 80.6% | +3.08 pp | 0.963 |
| breast_cancer | uniform | Fixed rank fusion | 81.6% | +2.80 pp | 0.970 |
| breast_cancer | uniform | Hindsight ceiling | 85.0% | +2.80 pp | 1.000 |
| breast_cancer | paired | Random review | 21.2% | +7.55 pp | 0.213 |
| breast_cancer | paired | Out-of-fold logistic | 37.4% | +15.24 pp | 0.385 |
| breast_cancer | paired | Neighbour agreement | 46.2% | +17.34 pp | 0.426 |
| breast_cancer | paired | Out-of-fold forest | 41.2% | +15.38 pp | 0.386 |
| breast_cancer | paired | Neural training margin | 36.2% | +14.97 pp | 0.355 |
| breast_cancer | paired | Fixed rank fusion | 37.4% | +15.94 pp | 0.368 |
| breast_cancer | paired | Hindsight ceiling | 85.0% | +24.20 pp | 1.000 |
| breast_cancer | boundary | Random review | 18.0% | +2.10 pp | 0.197 |
| breast_cancer | boundary | Out-of-fold logistic | 70.2% | +4.90 pp | 0.844 |
| breast_cancer | boundary | Neighbour agreement | 64.4% | +5.03 pp | 0.712 |
| breast_cancer | boundary | Out-of-fold forest | 72.0% | +5.17 pp | 0.843 |
| breast_cancer | boundary | Neural training margin | 74.6% | +5.45 pp | 0.882 |
| breast_cancer | boundary | Fixed rank fusion | 73.6% | +5.17 pp | 0.859 |
| breast_cancer | boundary | Hindsight ceiling | 85.0% | +6.29 pp | 1.000 |

## What the comparisons show

On digit labels with random flips, fixed rank fusion found 100% of the first 100 reviewed examples to be errors, but logistic test accuracy changed by only +0.09 pp. Neighbour review found 97.2% and changed accuracy by +1.38 pp. With the forest outcome model, fusion's change was -0.13 pp. Finding errors and improving a particular classifier are separate outcomes.
The binary class-specific experiment is deliberately severe: one source class supplies 20% of all training rows as changed labels. Even random review changed test accuracy by +7.55 pp; neighbour review changed it by +17.34 pp. These are controlled stress-test gains, not expected gains for an ordinary dataset.

## How to interpret this

A high error-finding score does not guarantee a useful model improvement. Check both columns.
Runs reuse a small bundled dataset under several controlled corruptions; the seeds are paired but not independent populations. No significance or real-world deployment claim is made.
The boundary condition selects genuinely difficult images for corruption. A ranking can also flag difficult *correct* images, which should not be automatically deleted or relabelled.
The binary dataset's paired condition flips only one source class; its uniform condition flips either class.
The mean and standard deviation for all budgets, the second outcome model, and every seed are in `summary.csv`, `ranking.csv`, and `review.csv`.
Training labels are revealed only by the simulated reviewer after a method selects them. The review simulation is not an automatic relabelling algorithm.

## Replication

From the repository root: `python -m dataclinic reproduce --config configs/reproduce.json --output results/main --docs docs`.
The command overwrites the derived outputs and rebuilds the standalone review page.
See `docs/METHOD.md` for the complete protocol and its limitations.
