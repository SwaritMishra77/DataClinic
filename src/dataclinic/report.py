"""Summaries and static figures built exclusively from saved experiment output."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .methods import METHODS


NAMES = {
    "random": "Random review",
    "oof_logistic": "Out-of-fold logistic",
    "neighbors": "Neighbour agreement",
    "oof_forest": "Out-of-fold forest",
    "training_margin": "Neural training margin",
    "rank_fusion": "Fixed rank fusion",
    "oracle_ceiling": "Hindsight ceiling",
}


def summarize(ranking: pd.DataFrame, review: pd.DataFrame):
    group = ["dataset", "mode", "model", "budget", "method"]
    summary = review.groupby(group, as_index=False).agg(
        seeds=("seed", "nunique"),
        mean_precision=("precision_at_budget", "mean"),
        std_precision=("precision_at_budget", "std"),
        mean_found=("found_n", "mean"),
        mean_base_accuracy=("base_accuracy", "mean"),
        mean_after_accuracy=("after_accuracy", "mean"),
        mean_delta_accuracy=("delta_accuracy", "mean"),
        std_delta_accuracy=("delta_accuracy", "std"),
        mean_delta_macro_f1=("delta_macro_f1", "mean"),
        mean_delta_log_loss=("delta_log_loss", "mean"),
    )
    rank_summary = ranking.groupby(["dataset", "mode", "method"], as_index=False).agg(
        seeds=("seed", "nunique"),
        mean_average_precision=("average_precision", "mean"),
        std_average_precision=("average_precision", "std"),
    )
    return summary.fillna(0), rank_summary.fillna(0)


def plot_comparison(summary: pd.DataFrame, destination: Path):
    frame = summary.loc[
        (summary.dataset == "digits") & (summary.model == "logistic") &
        (summary.budget == 100) & (summary.method != "oracle_ceiling")
    ].copy()
    modes = [m for m in ("uniform", "paired", "boundary") if m in frame["mode"].values]
    if not modes:
        return
    fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 4.3), sharey=True)
    if len(modes) == 1:
        axes = [axes]
    palette = {"random": "#99918b", "oof_logistic": "#e36243",
               "neighbors": "#c39b27", "oof_forest": "#804fc0",
               "training_margin": "#375f8d", "rank_fusion": "#144f42"}
    for ax, mode in zip(axes, modes):
        sub = frame[frame["mode"] == mode].set_index("method")
        methods = [m for m in METHODS if m in sub.index]
        vals = [sub.loc[m, "mean_precision"] for m in methods]
        ax.barh([NAMES[m] for m in methods], vals,
                color=[palette[m] for m in methods])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_title(mode.title())
        ax.set_xlabel("Errors found / 100 reviews")
        ax.grid(axis="x", alpha=.16)
        ax.set_axisbelow(True)
    fig.suptitle("Digit labels: precision at a 100-example review budget", fontsize=13)
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight", facecolor="#faf7f2")
    plt.close(fig)


def plot_tradeoff(summary: pd.DataFrame, destination: Path):
    frame = summary.loc[
        (summary.dataset == "digits") & (summary.model == "logistic") &
        (summary.budget == 100) & (summary.method != "oracle_ceiling")
    ]
    modes = [m for m in ("uniform", "paired", "boundary") if m in frame["mode"].values]
    if not modes:
        return
    fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 4.6), sharey=True)
    if len(modes) == 1:
        axes = [axes]
    colors = ("#99918b", "#e36243", "#c39b27", "#804fc0", "#375f8d", "#144f42")
    for ax, mode in zip(axes, modes):
        sub = frame[frame["mode"] == mode].set_index("method")
        for method, color in zip(METHODS, colors):
            if method in sub.index:
                row = sub.loc[method]
                ax.scatter(row.mean_precision, row.mean_delta_accuracy * 100,
                           s=100, c=color, label=NAMES[method], edgecolor="#26252a", linewidth=.5)
        ax.axhline(0, color="#3a3736", linestyle="--", alpha=.5)
        ax.set_xlim(0, 1.03)
        ax.set_title(mode.title())
        ax.set_xlabel("Errors found / reviews")
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Change in test accuracy (pp)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, -.07))
    fig.suptitle("More detected errors need not mean more model improvement", fontsize=13)
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight", facecolor="#faf7f2")
    plt.close(fig)


def write_results_markdown(summary: pd.DataFrame, rank_summary: pd.DataFrame, metadata: dict, path: Path):
    lines = [
        "# DataClinic results", "",
        "Generated from the committed experiment CSV files. These are completed runs, not projected outcomes.", "",
        f"- Cohorts: {metadata['cohorts']} ({len(metadata['configuration']['seeds'])} split/noise seeds).",
        f"- Noise rate: {metadata['configuration']['noise_rate']:.0%} of training labels; clean test labels untouched.",
        f"- Runtime in the recorded environment: {metadata['runtime_seconds']:.1f} seconds.",
        "- The table uses 100 reviews, with the logistic outcome model. Changes are percentage points of test accuracy relative to the same seed's contaminated baseline.",
        "- The oracle row is a hindsight ceiling for finding errors, not necessarily the highest possible test accuracy; it is not a deployable method.",
        "", "## Matched review comparison", "",
        "| Dataset | Noise | Ranking | Precision @100 | Test accuracy change | AP |",
        "|---|---|---|---:|---:|---:|",
    ]
    rows = summary[(summary.model == "logistic") & (summary.budget == 100)]
    merged = rows.merge(rank_summary, on=["dataset", "mode", "method"])
    for dataset in metadata["configuration"]["datasets"]:
        for mode in metadata["configuration"]["modes"]:
            for method in (*METHODS, "oracle_ceiling"):
                selected = merged[(merged.dataset == dataset) & (merged["mode"] == mode) &
                                  (merged.method == method)]
                if selected.empty:
                    continue
                row = selected.iloc[0]
                lines.append(
                    f"| {dataset} | {mode} | {NAMES[method]} | "
                    f"{row.mean_precision:.1%} | {100 * row.mean_delta_accuracy:+.2f} pp | "
                    f"{row.mean_average_precision:.3f} |"
                )
    lines.extend([
        "", "## What the comparisons show", "",
    ])
    def cell(dataset: str, mode: str, model: str, method: str):
        subset = summary[(summary.dataset == dataset) & (summary["mode"] == mode) &
                         (summary.model == model) & (summary.budget == 100) &
                         (summary.method == method)]
        return None if subset.empty else subset.iloc[0]

    fusion = cell("digits", "uniform", "logistic", "rank_fusion")
    neighbour = cell("digits", "uniform", "logistic", "neighbors")
    forest_fusion = cell("digits", "uniform", "forest", "rank_fusion")
    if fusion is not None and neighbour is not None and forest_fusion is not None:
        lines.append(
            "On digit labels with random flips, fixed rank fusion found "
            f"{fusion.mean_precision:.0%} of the first 100 reviewed examples to be errors, "
            f"but logistic test accuracy changed by only {100*fusion.mean_delta_accuracy:+.2f} pp. "
            f"Neighbour review found {neighbour.mean_precision:.1%} and changed accuracy by "
            f"{100*neighbour.mean_delta_accuracy:+.2f} pp. "
            f"With the forest outcome model, fusion's change was {100*forest_fusion.mean_delta_accuracy:+.2f} pp. "
            "Finding errors and improving a particular classifier are separate outcomes."
        )
    paired_random = cell("breast_cancer", "paired", "logistic", "random")
    paired_neighbour = cell("breast_cancer", "paired", "logistic", "neighbors")
    if paired_random is not None and paired_neighbour is not None:
        lines.append(
            "The binary class-specific experiment is deliberately severe: one source class supplies "
            "20% of all training rows as changed labels. Even random review changed test accuracy by "
            f"{100*paired_random.mean_delta_accuracy:+.2f} pp; neighbour review changed it by "
            f"{100*paired_neighbour.mean_delta_accuracy:+.2f} pp. These are controlled stress-test gains, "
            "not expected gains for an ordinary dataset."
        )
    lines.extend([
        "", "## How to interpret this", "",
        "A high error-finding score does not guarantee a useful model improvement. Check both columns.",
        "Runs reuse a small bundled dataset under several controlled corruptions; the seeds are paired but not independent populations. No significance or real-world deployment claim is made.",
        "The boundary condition selects genuinely difficult images for corruption. A ranking can also flag difficult *correct* images, which should not be automatically deleted or relabelled.",
        "The binary dataset's paired condition flips only one source class; its uniform condition flips either class.",
        "The mean and standard deviation for all budgets, the second outcome model, and every seed are in `summary.csv`, `ranking.csv`, and `review.csv`.",
        "Training labels are revealed only by the simulated reviewer after a method selects them. The review simulation is not an automatic relabelling algorithm.",
        "", "## Replication", "",
        "From the repository root: `python -m dataclinic reproduce --config configs/reproduce.json --output results/main --docs docs`.",
        "The command overwrites the derived outputs and rebuilds the standalone review page.",
        "See `docs/METHOD.md` for the complete protocol and its limitations.", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(ranking: pd.DataFrame, review: pd.DataFrame,
                 result_dir: Path, docs_dir: Path):
    summary, rank_summary = summarize(ranking, review)
    summary.to_csv(result_dir / "summary.csv", index=False, float_format="%.8f")
    rank_summary.to_csv(result_dir / "ranking_summary.csv", index=False, float_format="%.8f")
    plot_comparison(summary, result_dir / "precision_at_100.png")
    plot_tradeoff(summary, result_dir / "detection_vs_accuracy.png")
    metadata = json.loads((result_dir / "metadata.json").read_text(encoding="utf-8"))
    write_results_markdown(summary, rank_summary, metadata, result_dir / "RESULTS.md")
    template = (Path(__file__).resolve().parents[2] / "demo" / "template.html")
    if not template.exists():
        raise FileNotFoundError(f"Missing project demo template: {template}")
    cases = json.loads((result_dir / "case_studies.json").read_text(encoding="utf-8"))
    payload = {"summary": summary.round(8).to_dict(orient="records"),
               "cases": cases, "metadata": metadata}
    docs_dir.mkdir(parents=True, exist_ok=True)
    html = template.read_text(encoding="utf-8")
    if html.count("__DATACLINIC_DATA__") != 1:
        raise ValueError("Demo template must have exactly one data marker")
    html = html.replace("__DATACLINIC_DATA__", json.dumps(payload, separators=(",", ":")))
    (docs_dir / "index.html").write_text(html, encoding="utf-8")
