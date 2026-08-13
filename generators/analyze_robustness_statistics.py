import pandas as pd
from pathlib import Path
from scipy.stats import ttest_1samp


TARGETS = [
    "flare_risk",
    "dizziness_risk",
    "fatigue_risk",
    "fainting_risk",
    "need_to_hydrate",
    "need_to_rest",
]


def get_project_root():
    return (
        Path(__file__)
        .resolve()
        .parent.parent
    )


def main():
    print("=" * 80)
    print("PaceMate-AI Robustness Statistical Analysis")
    print("=" * 80)

    root = get_project_root()

    input_path = (
        root
        / "results"
        / "repeated_longitudinal_ablation.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {input_path}"
        )

    data = pd.read_csv(input_path)

    print()
    print(
        f"Loaded {len(data)} robustness observations."
    )

    results = []

    for target in TARGETS:

        target_data = data[
            data["target"] == target
        ]

        roc_values = (
            target_data["roc_auc_improvement"]
            .dropna()
        )

        pr_values = (
            target_data["pr_auc_improvement"]
            .dropna()
        )

        brier_values = (
            target_data["brier_difference"]
            .dropna()
        )

        roc_test = ttest_1samp(
            roc_values,
            0.0,
        )

        pr_test = ttest_1samp(
            pr_values,
            0.0,
        )

        brier_test = ttest_1samp(
            brier_values,
            0.0,
        )

        results.append(
            {
                "target": target,

                "roc_auc_improvement_mean":
                    roc_values.mean(),

                "roc_auc_improvement_std":
                    roc_values.std(),

                "roc_auc_t_statistic":
                    roc_test.statistic,

                "roc_auc_p_value":
                    roc_test.pvalue,

                "pr_auc_improvement_mean":
                    pr_values.mean(),

                "pr_auc_improvement_std":
                    pr_values.std(),

                "pr_auc_t_statistic":
                    pr_test.statistic,

                "pr_auc_p_value":
                    pr_test.pvalue,

                "brier_difference_mean":
                    brier_values.mean(),

                "brier_difference_std":
                    brier_values.std(),

                "brier_t_statistic":
                    brier_test.statistic,

                "brier_p_value":
                    brier_test.pvalue,
            }
        )

    results_df = pd.DataFrame(
        results
    )

    output_path = (
        root
        / "results"
        / "robustness_statistical_analysis.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("STATISTICAL ROBUSTNESS RESULTS")
    print("=" * 80)

    print(
        results_df.round(6).to_string(
            index=False
        )
    )

    print()
    print(
        f"Results saved to: {output_path}"
    )

    print()
    print("=" * 80)
    print("STATISTICAL ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()