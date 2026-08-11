import os
import pandas as pd


RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def load_data():
    error_table_path = os.path.join(
        RESULTS_DIR,
        "error_analysis_results_table.csv",
    )

    error_summary_path = os.path.join(
        RESULTS_DIR,
        "error_analysis_summary.txt",
    )

    error_table = pd.read_csv(error_table_path)

    with open(error_summary_path, "r", encoding="utf-8") as file:
        error_summary = file.read()

    return error_table, error_summary


def create_report(error_table, error_summary):
    total_count = error_table["count"].sum()

    true_negatives = int(
        error_table.loc[
            error_table["error_type"] == "true_negative",
            "count",
        ].iloc[0]
    )

    false_positives = int(
        error_table.loc[
            error_table["error_type"] == "false_positive",
            "count",
        ].iloc[0]
    )

    false_negatives = int(
        error_table.loc[
            error_table["error_type"] == "false_negative",
            "count",
        ].iloc[0]
    )

    true_positives = int(
        error_table.loc[
            error_table["error_type"] == "true_positive",
            "count",
        ].iloc[0]
    )

    error_count = false_positives + false_negatives
    error_rate = error_count / total_count

    fp_row = error_table[
        error_table["error_type"] == "false_positive"
    ].iloc[0]

    fn_row = error_table[
        error_table["error_type"] == "false_negative"
    ].iloc[0]

    tp_row = error_table[
        error_table["error_type"] == "true_positive"
    ].iloc[0]

    tn_row = error_table[
        error_table["error_type"] == "true_negative"
    ].iloc[0]

    report = f"""# PaceMate-AI Model Error Analysis Report

## 1. Purpose

This analysis examined the prediction errors produced by the PaceMate-AI machine-learning models on the held-out test data.

The purpose was to determine how false positives and false negatives differ from correct predictions and to identify feature patterns associated with different prediction outcomes.

The analysis complements the earlier feature-importance analysis and longitudinal feature-ablation experiment.

Rather than evaluating the models only through aggregate performance metrics, this analysis examines the characteristics of individual prediction outcomes.

## 2. Dataset

The analysis included {total_count:,} test examples across the prediction tasks.

The prediction outcomes were:

- True negatives: {true_negatives:,}
- False positives: {false_positives:,}
- False negatives: {false_negatives:,}
- True positives: {true_positives:,}

The overall error rate was {error_rate:.4f} ({error_rate * 100:.2f}%).

False positives represented cases where the model predicted a positive outcome while the observed target was negative.

False negatives represented cases where the model predicted a negative outcome while the observed target was positive.

## 3. Prediction Probability Analysis

False-positive predictions had a mean predicted probability of {fp_row["mean_probability"]:.4f} and a median predicted probability of {fp_row["median_probability"]:.4f}.

False-negative predictions had a mean predicted probability of {fn_row["mean_probability"]:.4f} and a median predicted probability of {fn_row["median_probability"]:.4f}.

The higher probability values among false positives indicate that these errors generally occurred when the model assigned substantial confidence to a positive prediction even though the observed target was negative.

False negatives occurred at lower predicted probabilities, indicating that the model often assigned relatively modest probabilities to examples that were actually positive.

This difference is important because it suggests that false positives and false negatives arise from different prediction conditions rather than representing identical types of model error.

## 4. Symptom-Related Feature Patterns

Symptom-related variables showed substantial differences across prediction outcome groups.

The mean current symptom severity was:

- True negatives: {tn_row["mean_symptom_severity"]:.4f}
- False positives: {fp_row["mean_symptom_severity"]:.4f}
- False negatives: {fn_row["mean_symptom_severity"]:.4f}
- True positives: {tp_row["mean_symptom_severity"]:.4f}

The mean previous symptom severity was:

- True negatives: {tn_row["mean_previous_symptom_severity"]:.4f}
- False positives: {fp_row["mean_previous_symptom_severity"]:.4f}
- False negatives: {fn_row["mean_previous_symptom_severity"]:.4f}
- True positives: {tp_row["mean_previous_symptom_severity"]:.4f}

True-positive examples had the highest average symptom severity, while true-negative examples had substantially lower symptom severity.

False positives also showed elevated symptom severity compared with true negatives.

This suggests that the model may identify elevated symptom patterns as evidence for a positive outcome, but some cases with elevated symptoms do not correspond to positive targets.

## 5. HRV-Related Feature Patterns

HRV-related variables also differed substantially between prediction outcome groups.

The mean current HRV was:

- True negatives: {tn_row["mean_hrv"]:.4f}
- False positives: {fp_row["mean_hrv"]:.4f}
- False negatives: {fn_row["mean_hrv"]:.4f}
- True positives: {tp_row["mean_hrv"]:.4f}

The mean previous HRV was:

- True negatives: {tn_row["mean_previous_hrv"]:.4f}
- False positives: {fp_row["mean_previous_hrv"]:.4f}
- False negatives: {fn_row["mean_previous_hrv"]:.4f}
- True positives: {tp_row["mean_previous_hrv"]:.4f}

The true-positive group had the lowest average HRV among the four prediction outcome groups.

False-positive examples had higher HRV than true positives but lower HRV than true negatives.

These patterns are consistent with the earlier feature-importance analysis, in which HRV-related longitudinal features repeatedly appeared among important predictors.

## 6. Hydration-Related Patterns

Hydration deficit also differed substantially between prediction outcome groups.

Mean hydration deficit was:

- True negatives: {tn_row["mean_hydration_deficit"]:.4f}
- False positives: {fp_row["mean_hydration_deficit"]:.4f}
- False negatives: {fn_row["mean_hydration_deficit"]:.4f}
- True positives: {tp_row["mean_hydration_deficit"]:.4f}

The largest difference among the analyzed outcome groups occurred for hydration deficit.

True negatives had the most negative average hydration deficit, while true positives had a less negative average value.

Because these values are derived from synthetic data, this pattern should be interpreted as a characteristic of the generated dataset rather than evidence of a clinical relationship.

## 7. Longitudinal Features

The error analysis also examined historical and rolling features.

Features included:

- Previous symptom severity
- Three-day symptom mean
- Seven-day symptom mean
- Previous HRV
- Three-day HRV mean
- Seven-day HRV mean
- Symptom change
- HRV change
- Sleep deficit
- Hydration deficit

The longitudinal variables showed systematic differences between prediction outcome groups.

In particular, previous symptom severity and rolling symptom averages were consistently higher in false-positive and true-positive cases than in true-negative cases.

The HRV-related longitudinal features showed the opposite general pattern, with lower HRV values occurring in groups containing more positive predictions.

These findings provide additional support for retaining longitudinal information within the PaceMate-AI feature architecture.

## 8. Relationship to the Longitudinal Ablation Study

The error analysis complements the longitudinal feature-ablation experiment conducted previously.

The ablation study demonstrated that removing longitudinal features substantially reduced predictive performance across all six prediction targets.

The current analysis provides a more detailed view of why those features may be useful.

Historical symptom severity, rolling symptom averages, previous HRV, and rolling HRV measurements differ systematically between prediction outcome groups.

Together, these analyses provide two complementary forms of evidence:

1. The ablation experiment demonstrates that longitudinal features improve predictive performance.

2. The error analysis demonstrates that longitudinal features vary meaningfully across correct and incorrect prediction groups.

This strengthens the methodological justification for using longitudinal features rather than relying exclusively on same-day measurements.

## 9. Error Characteristics

The results indicate that false positives and false negatives have different characteristics.

False positives generally occurred among examples with elevated symptom severity and intermediate-to-high predicted probabilities.

False negatives generally had lower predicted probabilities than false positives and showed intermediate symptom and HRV values.

This suggests that some errors occur near overlapping regions of the feature space where positive and negative examples are difficult to distinguish.

The results therefore motivate additional investigation into threshold selection, calibration, and class imbalance.

## 10. Visualizations

Four research visualizations were generated from the error-analysis results.

### Prediction Outcomes

`results/figures/prediction_outcomes.png`

This visualization shows the number of test examples in each prediction outcome category.

### Error Probability Distribution

`results/figures/error_probability_distribution.png`

This visualization compares the predicted-probability distributions of false positives and false negatives.

### Error Group Feature Comparison

`results/figures/error_group_feature_comparison.png`

This visualization compares selected feature means across true-negative, false-positive, false-negative, and true-positive groups.

### Longitudinal Error Feature Patterns

`results/figures/longitudinal_error_feature_patterns.png`

This visualization focuses specifically on longitudinal features and compares their mean values across prediction outcome groups.

## 11. Scientific Interpretation

The error analysis demonstrates that model mistakes are not uniformly distributed across the feature space.

Correct and incorrect predictions exhibit different symptom, HRV, hydration, and longitudinal feature patterns.

The strongest differences were observed in symptom-related variables, HRV-related variables, and hydration deficit.

These results are consistent with the broader PaceMate-AI modeling strategy, which incorporates both current measurements and historical measurements.

However, these relationships should be interpreted as associations within the generated dataset.

They do not demonstrate that any individual feature causes a particular clinical outcome.

## 12. Limitations

The dataset used in this analysis is synthetic.

Therefore, the observed feature relationships, error patterns, and model behavior cannot establish clinical effectiveness.

The analysis also includes highly imbalanced prediction targets, meaning that accuracy alone is not sufficient for evaluating performance.

The error-group comparisons describe associations between features and prediction outcomes and should not be interpreted as causal relationships.

Future evaluation should include additional calibration analysis, threshold analysis, subgroup analysis, and validation using appropriately collected real-world data if such data become available.

## 13. Conclusion

The PaceMate-AI error analysis provides a more detailed evaluation of model behavior beyond aggregate performance metrics.

The results show that false positives and false negatives have distinct probability and feature profiles.

Symptom history, HRV-related variables, and hydration-related variables differed substantially across prediction outcome groups.

The findings complement the previous feature-importance and longitudinal-ablation experiments and provide additional evidence that temporal information is an important component of the PaceMate-AI architecture.

Because the current dataset is synthetic, these findings should be considered evidence about the machine-learning pipeline rather than evidence of clinical effectiveness.

"""

    return report


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    error_table, error_summary = load_data()

    report = create_report(
        error_table,
        error_summary,
    )

    output_path = os.path.join(
        RESULTS_DIR,
        "error_analysis_report.md",
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report)

    print()
    print("PaceMate-AI Error Analysis Report")
    print()
    print(f"Report saved to: {output_path}")
    print()
    print("Report generation complete.")


if __name__ == "__main__":
    main()