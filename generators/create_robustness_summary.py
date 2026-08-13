import pandas as pd
from pathlib import Path


def get_project_root():
    return (
        Path(__file__)
        .resolve()
        .parent.parent
    )


def load_results(filename):
    path = (
        get_project_root()
        / "results"
        / filename
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Results file not found: {path}"
        )

    return pd.read_csv(path)


def main():
    print("=" * 80)
    print("PaceMate-AI Robustness Experiment Summary")
    print("=" * 80)

    root = get_project_root()

    print()
    print("Loading robustness results...")

    discrimination = load_results(
        "repeated_discrimination_robustness.csv"
    )

    threshold = load_results(
        "repeated_threshold_f1_robustness.csv"
    )

    ablation = load_results(
        "repeated_longitudinal_ablation.csv"
    )

    print()
    print(
        f"Discrimination rows: {len(discrimination)}"
    )

    print(
        f"Threshold/F1 rows:   {len(threshold)}"
    )

    print(
        f"Ablation rows:        {len(ablation)}"
    )

    # --------------------------------------------------
    # Experiment 1A
    # --------------------------------------------------

    discrimination_summary = (
        discrimination
        .groupby("target")
        .agg(
            validation_roc_auc_mean=(
                "validation_roc_auc",
                "mean",
            ),
            validation_roc_auc_std=(
                "validation_roc_auc",
                "std",
            ),
            test_roc_auc_mean=(
                "test_roc_auc",
                "mean",
            ),
            test_roc_auc_std=(
                "test_roc_auc",
                "std",
            ),
            validation_pr_auc_mean=(
                "validation_pr_auc",
                "mean",
            ),
            validation_pr_auc_std=(
                "validation_pr_auc",
                "std",
            ),
            test_pr_auc_mean=(
                "test_pr_auc",
                "mean",
            ),
            test_pr_auc_std=(
                "test_pr_auc",
                "std",
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------
    # Experiment 1B
    # --------------------------------------------------

    threshold_summary = (
        threshold
        .groupby("target")
        .agg(
            threshold_mean=(
                "validation_threshold",
                "mean",
            ),
            threshold_std=(
                "validation_threshold",
                "std",
            ),
            validation_f1_mean=(
                "validation_f1",
                "mean",
            ),
            validation_f1_std=(
                "validation_f1",
                "std",
            ),
            test_precision_mean=(
                "test_precision",
                "mean",
            ),
            test_precision_std=(
                "test_precision",
                "std",
            ),
            test_recall_mean=(
                "test_recall",
                "mean",
            ),
            test_recall_std=(
                "test_recall",
                "std",
            ),
            test_f1_mean=(
                "test_f1",
                "mean",
            ),
            test_f1_std=(
                "test_f1",
                "std",
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------
    # Experiment 1C
    # --------------------------------------------------

    ablation_summary = (
        ablation
        .groupby("target")
        .agg(
            roc_auc_improvement_mean=(
                "roc_auc_improvement",
                "mean",
            ),
            roc_auc_improvement_std=(
                "roc_auc_improvement",
                "std",
            ),
            pr_auc_improvement_mean=(
                "pr_auc_improvement",
                "mean",
            ),
            pr_auc_improvement_std=(
                "pr_auc_improvement",
                "std",
            ),
            brier_difference_mean=(
                "brier_difference",
                "mean",
            ),
            brier_difference_std=(
                "brier_difference",
                "std",
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------
    # Combine
    # --------------------------------------------------

    summary = (
        discrimination_summary
        .merge(
            threshold_summary,
            on="target",
            how="inner",
        )
        .merge(
            ablation_summary,
            on="target",
            how="inner",
        )
    )

    output_path = (
        root
        / "results"
        / "day9_robustness_summary.csv"
    )

    summary.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("DAY 9 ROBUSTNESS SUMMARY")
    print("=" * 80)

    print(
        summary.round(4).to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved to: {output_path}"
    )

    print()
    print("=" * 80)
    print("ROBUSTNESS SUMMARY COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()