import os

import pandas as pd


RESULTS_DIR = "results"


def load_error_analysis():
    path = os.path.join(
        RESULTS_DIR,
        "model_error_analysis.csv",
    )

    return pd.read_csv(path)


def create_results_table(error_analysis):
    rows = []

    for error_type in [
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ]:
        group = error_analysis[
            error_analysis["error_type"] == error_type
        ]

        if group.empty:
            continue

        rows.append(
            {
                "error_type": error_type,
                "count": len(group),
                "mean_probability": group["probability"].mean(),
                "median_probability": group["probability"].median(),
                "mean_threshold": group["threshold"].mean(),
                "mean_sleep_hours": group["sleep_hours"].mean(),
                "mean_water_intake_ml": group["water_intake_ml"].mean(),
                "mean_resting_hr": group["resting_hr"].mean(),
                "mean_hrv": group["hrv"].mean(),
                "mean_symptom_severity": group[
                    "symptom_severity"
                ].mean(),
                "mean_previous_symptom_severity": group[
                    "previous_symptom_severity"
                ].mean(),
                "mean_symptom_3day_mean": group[
                    "symptom_3day_mean"
                ].mean(),
                "mean_symptom_7day_mean": group[
                    "symptom_7day_mean"
                ].mean(),
                "mean_previous_hrv": group[
                    "previous_hrv"
                ].mean(),
                "mean_hrv_3day_mean": group[
                    "hrv_3day_mean"
                ].mean(),
                "mean_hrv_7day_mean": group[
                    "hrv_7day_mean"
                ].mean(),
                "mean_symptom_change": group[
                    "symptom_change"
                ].mean(),
                "mean_hrv_change": group[
                    "hrv_change"
                ].mean(),
                "mean_sleep_deficit": group[
                    "sleep_deficit"
                ].mean(),
                "mean_hydration_deficit": group[
                    "hydration_deficit"
                ].mean(),
            }
        )

    return pd.DataFrame(rows)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    error_analysis = load_error_analysis()

    results_table = create_results_table(
        error_analysis
    )

    output_path = os.path.join(
        RESULTS_DIR,
        "error_analysis_results_table.csv",
    )

    results_table.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        "PaceMate-AI Error Analysis Results Table"
    )
    print()
    print(results_table.to_string(index=False))
    print()
    print(
        f"Results saved to: {output_path}"
    )


if __name__ == "__main__":
    main()
