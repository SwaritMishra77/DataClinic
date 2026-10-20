# Experimental method

## Question

With a fixed number of human reviews, which method finds more incorrect training labels, and do those corrections improve the classifier on a clean held-out set?

## Dataset and split

`load_digits` contains 1,797 handwritten 8 × 8 images and ten classes. `load_breast_cancer` contains 569 numerical samples and two classes. The latter is only a general small-tabular cross-check; **no clinical inference or deployment is supported**. Both are bundled in scikit-learn. For each of five seeds (11, 23, 37, 51, 79), a stratified 75/25 split is created before any labels are modified. Train indices and test indices are disjoint.

All preprocessing for a ranking method is fit on its training fold or the full *training* set, as appropriate. Nothing is learned from test features or labels. The original training labels are retained privately by the simulation. The ranking algorithms only receive features and the contaminated training labels.

## Controlled label errors

Exactly `round(0.20 × training examples)` labels change in each experiment. The generator and the simulated reviewer may see the private originals. The methods and fitted classifiers cannot.

1. **Uniform:** select training rows at random and choose another class uniformly.
2. **Paired:** select rows at random and replace their class with a fixed confusable class. On the binary dataset only class 0 is selected and flipped to class 1, creating a one-way class-specific error process.
3. **Boundary:** compute nearest same-class and different-class distances within the training set, after feature scaling. Select rows with the highest ratio of own-class to rival-class distance and give each the rival's class. This deliberately changes more ambiguous examples. It is a controlled stress test, not a claim that the original label is ambiguous or wrong.

The 20% prevalence is an experiment setting, not an estimate of error rates in real datasets. Paired corruption on binary data may select a large share of the smaller source class. Every scenario is generated afresh within each seed's training partition.

## Ranking methods

Each produces one suspicion score per training example; descending scores get reviewed first.

| Method | Score | Access to held-out labels? |
|---|---|---|
| Random | Reproducible pseudorandom score | No |
| Out-of-fold logistic | 1 minus the predicted probability of the recorded label from 3-fold cross-fitted logistic regression | No |
| Neighbours | 1 minus the share of the seven nearest *other* training examples with the recorded label | No |
| Out-of-fold forest | Same cross-fitting idea, using a random forest | No |
| Neural training margin | Negative mean, over 20 epochs, of the predicted probability margin for the recorded label against its closest rival; 64- and 32-unit hidden layers | No |
| Rank fusion | Arithmetic mean of the normalised ranks of out-of-fold logistic, neighbours, and neural margin; equal weights fixed before evaluation | No |
| Oracle ceiling | All known injected errors first, for a hindsight upper bound only | **Yes** |

The neural margin is inspired by training-dynamics research. It is not a reproduction of AUM's threshold-sample algorithm or a new theoretical estimator. Raw neural scores are not treated as calibrated probabilities. Out-of-fold means each scored example was absent from the particular fitting fold that predicted it. Neighbour scoring excludes the example itself but does not cross-fit the neighbours; this difference is deliberate and recorded.

## Review and measurement

For budgets 25, 50, 100, and 200, an annotator simulator reveals the clean label only for selected examples. It corrects wrong labels and leaves already correct ones untouched. The outcome classifier is trained from scratch with the corrected training labels. Both before and after use the exact same untouched test set in a seed.

Detection metrics: average precision across the full ranking, precision and recall at each review budget. Outcome metrics: held-out accuracy, macro F1, and multiclass log loss before and after review, plus the paired change. Logistic regression is evaluated at all budgets. A random forest is evaluated at budget 100 as a second outcome model. The oracle is a ceiling, not included when comparing usable review strategies.

Every cell averages five paired seed runs. The standard deviation and all individual observations are saved. The runs share a dataset and should not be interpreted as five independent populations or a formal confidence interval. We do not pick a method using the test scores and then report that method as an unbiased future winner.

## Reproduction and files

`configs/reproduce.json` fixes all settings. `ranking.csv` has one row per dataset, scenario, seed, and ranking method. `review.csv` adds budgets and outcome models. `summary.csv` and `ranking_summary.csv` aggregate those rows. `case_studies.json` holds the first 30 digit queue entries per method for seed 11, including the original label only for display after simulated review. `docs/index.html` is generated from these files. `metadata.json` records versions, settings, and runtime.

The code uses `random_state` or local NumPy generators everywhere. For reproducible numerical behavior, set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` before running. Exact execution time and the last decimal of floating-point values can vary by machine.

## Limits

- These are small, bundled datasets. Results cannot be extrapolated to large language or vision models.
- Simulated errors have known ground truth. Real annotation disagreements often do not.
- Boundary noise is constructed using the clean training labels. Only the *generator* uses them; the test has no access to them. The setup is a stress test, not an estimate of deployment frequency.
- The reviewer is an oracle for the selected example. Real reviewers can disagree and cost different amounts of time.
- Detection at a fixed prevalence is easier than detection when errors are rare. Accuracy may barely change despite good detection.
- No method is guaranteed to win across the scenarios or outcome models. Difficult, correct examples should not be purged automatically.

