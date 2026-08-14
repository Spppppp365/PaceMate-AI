# PaceMate-AI Final Research Results Inventory

## Dataset

- Participants: 500 synthetic participants
- Observation period: 180 days per participant
- Total generated observations: 90,000
- Modeling observations: 89,000
- Prediction targets: 6
- Final longitudinal model features: 32
- Same-day comparison features: 6
- Training participants: 350
- Validation participants: 75
- Test participants: 75
- Participant-level splitting was used so participants were not shared across training, validation, and test sets.

## Prediction Targets

The model predicts six binary outcomes:

- flare_risk
- dizziness_risk
- fatigue_risk
- fainting_risk
- need_to_hydrate
- need_to_rest

## Primary Experimental Questions

### Question 1

Does longitudinal information improve prediction compared with same-day information?

The final longitudinal models used 32 features, while the same-day comparison models used 6 features.

Primary metrics:

- ROC-AUC
- PR-AUC
- Brier score

Longitudinal models were compared with same-day models across five participant-level experimental repeats using seeds 42 through 46.

### Question 2

Do longitudinal-model improvements generalize across unseen participants?

Primary evidence:

- Five repeated participant-level experiments
- Different random seeds
- 350 training participants per repeat
- 75 validation participants per repeat
- 75 test participants per repeat
- Mean and standard deviation of longitudinal-versus-same-day improvements
- Statistical tests of the repeated improvements
- Final held-out test-set performance

### Question 3

Can decision thresholds improve the usefulness of predictions?

Thresholds were selected using validation data and then evaluated on unseen test participants.

The threshold was not re-tuned on the test set.

Primary evidence:

- Validation-selected thresholds
- Test precision
- Test recall
- Test F1
- Repeated threshold/F1 experiments across five participant-level splits

### Question 4

Are model probabilities reasonably calibrated?

Calibration was evaluated using:

- Brier score
- Predicted probability versus observed outcome comparisons
- Probability-alignment analysis

### Question 5

Where does the model fail?

Error analysis examined:

- False positives
- False negatives
- Prediction probability distributions
- Feature differences between outcome groups
- Longitudinal feature patterns associated with errors

## Final Model Results

The final test-set evaluation was performed on 13,350 observations from 75 previously unseen participants.

| Target            | Threshold | Positive Rate | Accuracy | Precision | Recall |     F1 | ROC-AUC | PR-AUC |  Brier |
| ----------------- | --------: | ------------: | -------: | --------: | -----: | -----: | ------: | -----: | -----: |
| flare_risk       |      0.55 |        0.1845 |   0.8784 |    0.7526 | 0.5075 | 0.6062 |  0.9216 | 0.7221 | 0.0843 |
| dizziness_risk   |      0.25 |        0.3016 |   0.6032 |    0.4159 | 0.7809 | 0.5428 |  0.7145 | 0.4883 | 0.1862 |
| fatigue_risk     |      0.30 |        0.3743 |   0.5864 |    0.5021 | 0.7867 | 0.6130 |  0.7158 | 0.5716 | 0.2035 |
| fainting_risk    |      0.20 |        0.0240 |   0.9604 |    0.2815 | 0.4174 | 0.3363 |  0.9516 | 0.2599 | 0.0195 |
| need_to_hydrate |      0.35 |        0.3029 |   0.8120 |    0.6528 | 0.8103 | 0.7231 |  0.8927 | 0.7582 | 0.1225 |
| need_to_rest    |      0.35 |        0.3849 |   0.7346 |    0.6142 | 0.8352 | 0.7078 |  0.8369 | 0.7421 | 0.1598 |

## Final Test-Set Confusion-Matrix Results

| Target            | True Negatives | False Positives | False Negatives | True Positives |
| ----------------- | -------------: | --------------: | --------------: | -------------: |
| flare_risk       |         10,476 |             411 |           1,213 |          1,250 |
| dizziness_risk   |          4,909 |           4,415 |             882 |          3,144 |
| fatigue_risk     |          4,455 |           3,898 |           1,066 |          3,931 |
| fainting_risk    |         12,687 |             342 |             187 |            134 |
| need_to_hydrate |          7,563 |           1,743 |             767 |          3,277 |
| need_to_rest    |          5,515 |           2,696 |             847 |          4,292 |

## Final Test-Set Probability Results

| Target            | Actual Positive Rate | Mean Predicted Probability | Predicted Positive Rate |
| ----------------- | -------------------: | -------------------------: | ----------------------: |
| flare_risk       |               0.1845 |                     0.1788 |                  0.1244 |
| dizziness_risk   |               0.3016 |                     0.2971 |                  0.5662 |
| fatigue_risk     |               0.3743 |                     0.3684 |                  0.5864 |
| fainting_risk    |               0.0240 |                     0.0213 |                  0.0357 |
| need_to_hydrate |               0.3029 |                     0.2959 |                  0.3760 |
| need_to_rest    |               0.3849 |                     0.3791 |                  0.5234 |

## Longitudinal Feature Ablation

Five repeated participant-level experiments compared the 32-feature longitudinal models with 6-feature same-day models.

For every target and every repeat:

- Longitudinal features increased ROC-AUC.
- Longitudinal features increased PR-AUC.
- Longitudinal features decreased Brier score.

### Mean Longitudinal Improvement

| Target            | ROC-AUC Improvement Mean | ROC-AUC Improvement SD | PR-AUC Improvement Mean | PR-AUC Improvement SD | Brier Difference Mean | Brier Difference SD |
| ----------------- | -----------------------: | ---------------------: | ----------------------: | --------------------: | --------------------: | ------------------: |
| flare_risk       |                  +0.2885 |                 0.0301 |                 +0.4658 |                0.0283 |               -0.0664 |              0.0095 |
| dizziness_risk   |                  +0.1582 |                 0.0088 |                 +0.1433 |                0.0132 |               -0.0282 |              0.0015 |
| fatigue_risk     |                  +0.1777 |                 0.0132 |                 +0.1593 |                0.0179 |               -0.0352 |              0.0022 |
| fainting_risk    |                  +0.1613 |                 0.0208 |                 +0.1858 |                0.0194 |               -0.0042 |              0.0011 |
| need_to_hydrate |                  +0.1020 |                 0.0072 |                 +0.1789 |                0.0246 |               -0.0432 |              0.0034 |
| need_to_rest    |                  +0.2101 |                 0.0126 |                 +0.2399 |                0.0239 |               -0.0702 |              0.0019 |

Negative Brier differences indicate lower Brier scores for the longitudinal models.

## Statistical Analysis of Longitudinal Improvements

Paired statistical tests were performed across the five repeated participant-level experiments.

### ROC-AUC Improvement

| Target            | Mean Improvement | t-statistic |  p-value |
| ----------------- | ---------------: | ----------: | -------: |
| flare_risk       |          +0.2885 |     21.4255 | 0.000028 |
| dizziness_risk   |          +0.1582 |     40.0523 | 0.000002 |
| fatigue_risk     |          +0.1777 |     30.0190 | 0.000007 |
| fainting_risk    |          +0.1613 |     17.3689 | 0.000064 |
| need_to_hydrate |          +0.1020 |     31.8867 | 0.000006 |
| need_to_rest    |          +0.2101 |     37.1953 | 0.000003 |

### PR-AUC Improvement

| Target            | Mean Improvement | t-statistic |  p-value |
| ----------------- | ---------------: | ----------: | -------: |
| flare_risk       |          +0.4658 |     36.8359 | 0.000003 |
| dizziness_risk   |          +0.1433 |     24.2543 | 0.000017 |
| fatigue_risk     |          +0.1593 |     19.8766 | 0.000038 |
| fainting_risk    |          +0.1858 |     21.3966 | 0.000028 |
| need_to_hydrate |          +0.1789 |     16.2859 | 0.000083 |
| need_to_rest    |          +0.2399 |     22.4779 | 0.000023 |

### Brier Score Difference

| Target            | Mean Difference | t-statistic |    p-value |
| ----------------- | --------------: | ----------: | ---------: |
| flare_risk       |         -0.0664 |    -15.6578 |   0.000097 |
| dizziness_risk   |         -0.0282 |    -40.9004 |   0.000002 |
| fatigue_risk     |         -0.0352 |    -35.1620 |   0.000004 |
| fainting_risk    |         -0.0042 |     -8.2472 |   0.001179 |
| need_to_hydrate |         -0.0432 |    -28.7275 |   0.000009 |
| need_to_rest    |         -0.0702 |    -84.1451 |  0.00000012 |

These repeated experiments provide evidence that the observed longitudinal improvements were consistent across the five participant-level splits.

## Threshold Robustness

Decision thresholds were selected independently using validation data in each repeated participant-level experiment.

The selected thresholds varied across splits, demonstrating that threshold choice was not identical across participant samples.

### Repeated Test-Set Threshold Results

| Target            | Mean Test Precision | Mean Test Recall | Mean Test F1 |
| ----------------- | ------------------: | ---------------: | -----------: |
| flare_risk       |              0.6244 |           0.7486 |       0.6803 |
| dizziness_risk   |              0.4309 |           0.7628 |       0.5501 |
| fatigue_risk     |              0.4945 |           0.7704 |       0.6020 |
| fainting_risk    |              0.2669 |           0.5193 |       0.3431 |
| need_to_hydrate |              0.6276 |           0.8044 |       0.7042 |
| need_to_rest    |              0.6258 |           0.8247 |       0.7116 |

These repeated results show that threshold-selected classification performance varied across participant splits rather than remaining identical across experiments.

## Threshold Selection by Repeated Split

Validation-selected thresholds varied as follows:

| Target            | Thresholds Selected Across Five Repeats |
| ----------------- | --------------------------------------- |
| flare_risk       | 0.55, 0.60, 0.60, 0.65, 0.55            |
| dizziness_risk   | 0.45, 0.40, 0.45, 0.45, 0.40            |
| fatigue_risk     | 0.40, 0.45, 0.40, 0.40, 0.40            |
| fainting_risk    | 0.75, 0.55, 0.70, 0.65, 0.60            |
| need_to_hydrate | 0.55, 0.45, 0.55, 0.50, 0.50            |
| need_to_rest    | 0.45, 0.45, 0.45, 0.45, 0.45            |

The final thresholds used for the final held-out test evaluation were determined separately and are reported in the Final Model Results table above.

## Robustness Findings

Repeated participant-level experiments demonstrated that:

- Longitudinal features improved ROC-AUC for all six prediction targets in every repeated experiment.
- Longitudinal features improved PR-AUC for all six prediction targets in every repeated experiment.
- Longitudinal features reduced Brier score for all six prediction targets in every repeated experiment.
- The magnitude of improvement varied across participant-level splits.
- Threshold-selected precision, recall, and F1 varied across participant-level splits.
- Threshold selection was relatively stable for some targets and more variable for others, particularly fainting_risk.

## Important Negative Findings

- Historical variability features did not improve flare-risk performance.
- Validation improvements did not always translate into equivalent test-set F1 improvements.
- Prediction quality differed substantially between targets.
- Fainting-risk had very strong ROC-AUC (0.9516) but substantially weaker PR-AUC (0.2599), precision (0.2815), recall (0.4174), and F1 (0.3363).
- The weak positive-class metrics for fainting_risk occurred in the context of a highly imbalanced target, with an actual positive rate of only 2.40%.
- The final test-set threshold for fainting_risk was 0.20, producing high overall accuracy (0.9604) but relatively limited positive-class precision, recall, and F1.
- Threshold-selected predictions for dizziness_risk, fatigue_risk, and need_to_rest produced relatively high recall but also substantially more false positives than the flare_risk model.

## Target-Level Interpretation

### flare_risk

The final flare-risk model showed strong discrimination with ROC-AUC of 0.9216 and PR-AUC of 0.7221.

At the final threshold of 0.55:

- Precision was 0.7526.
- Recall was 0.5075.
- F1 was 0.6062.
- Accuracy was 0.8784.
- Brier score was 0.0843.

Longitudinal information produced the largest mean ROC-AUC improvement among the six targets, with a mean ROC-AUC increase of 0.2885. It also produced the largest mean PR-AUC improvement, with a mean PR-AUC increase of 0.4658.

### dizziness_risk

The final dizziness-risk model had ROC-AUC of 0.7145 and PR-AUC of 0.4883.

At the final threshold of 0.25:

- Precision was 0.4159.
- Recall was 0.7809.
- F1 was 0.5428.
- Accuracy was 0.6032.
- Brier score was 0.1862.

The model favored recall at the selected threshold, resulting in 3,144 true positives and 4,415 false positives on the test set.

### fatigue_risk

The final fatigue-risk model had ROC-AUC of 0.7158 and PR-AUC of 0.5716.

At the final threshold of 0.30:

- Precision was 0.5021.
- Recall was 0.7867.
- F1 was 0.6130.
- Accuracy was 0.5864.
- Brier score was 0.2035.

The model again favored recall at the selected threshold, with 3,931 true positives and 3,898 false positives.

### fainting_risk

The final fainting-risk model had ROC-AUC of 0.9516, indicating strong ranking discrimination.

However, the positive class was highly imbalanced, with an actual positive rate of only 2.40%.

At the final threshold of 0.20:

- Precision was 0.2815.
- Recall was 0.4174.
- F1 was 0.3363.
- Accuracy was 0.9604.
- PR-AUC was 0.2599.
- Brier score was 0.0195.

The difference between the high ROC-AUC and much weaker positive-class metrics demonstrates why ROC-AUC alone is insufficient for evaluating this target.

### need_to_hydrate

The final hydration model had ROC-AUC of 0.8927 and PR-AUC of 0.7582.

At the final threshold of 0.35:

- Precision was 0.6528.
- Recall was 0.8103.
- F1 was 0.7231.
- Accuracy was 0.8120.
- Brier score was 0.1225.

This was one of the strongest overall target results, combining high discrimination with relatively strong precision, recall, and F1.

### need_to_rest

The final rest model had ROC-AUC of 0.8369 and PR-AUC of 0.7421.

At the final threshold of 0.35:

- Precision was 0.6142.
- Recall was 0.8352.
- F1 was 0.7078.
- Accuracy was 0.7346.
- Brier score was 0.1598.

Longitudinal information produced a mean ROC-AUC improvement of 0.2101 and a mean PR-AUC improvement of 0.2399 relative to same-day features.

## Calibration

Calibration was evaluated using Brier scores and probability-alignment analyses.

The final test-set Brier scores were:

- flare_risk: 0.0843
- dizziness_risk: 0.1862
- fatigue_risk: 0.2035
- fainting_risk: 0.0195
- need_to_hydrate: 0.1225
- need_to_rest: 0.1598

In the repeated longitudinal ablation experiments, the longitudinal model had a lower Brier score than the same-day model for every target in every repeat.

The mean Brier-score differences were negative for all six targets, indicating improved probability error for the longitudinal models.

## Error Analysis

The project included dedicated analyses of model errors.

Generated error-analysis outputs included:

- False-positive and false-negative analysis
- Prediction probability distributions
- Feature comparisons between error/outcome groups
- Longitudinal feature patterns associated with prediction errors

These analyses were used to identify where the models performed well and where prediction quality was weaker.

## Limitations

### Synthetic Data

The dataset and prediction targets are synthetic.

Therefore:

- Model performance cannot be interpreted as clinical effectiveness.
- The results cannot establish clinical utility.
- The results cannot establish that the model would perform similarly on real patients.
- External validation using real-world data would be required.
- Synthetic relationships may be easier for a machine-learning model to learn than relationships in real-world clinical data.

### Participant-Level Generalization

Although participant-level splitting was used to prevent the same participant from appearing in multiple partitions, the participants themselves were synthetically generated.

Therefore, successful generalization to unseen synthetic participants does not establish generalization to unseen real patients.

### Class Imbalance

Fainting-risk was strongly imbalanced, with an actual positive rate of only 2.40% in the final test set.

This caused ROC-AUC to appear substantially stronger than positive-class precision, recall, F1, and PR-AUC.

### Threshold Stability

Validation-selected thresholds varied across repeated participant-level splits.

Therefore, the exact threshold values should not be interpreted as universally optimal clinical decision thresholds.

### Clinical Interpretation

PaceMate-AI is a research prototype and not a clinically validated diagnostic or treatment system.

The model results demonstrate predictive behavior within the synthetic experimental environment but do not establish medical safety, diagnostic accuracy, or clinical usefulness.

## Figures

The final analysis produced the following figures:

- model_discrimination_comparison.png
- model_classification_metrics.png
- model_brier_score_comparison.png
- model_probability_alignment.png
- prediction_outcomes.png
- error_probability_distribution.png
- error_group_feature_comparison.png
- longitudinal_error_feature_patterns.png

## Final Research Conclusion

Across five repeated participant-level experiments, longitudinal information consistently improved discrimination and probability quality for all six prediction targets compared with same-day information alone.

The mean improvements in ROC-AUC, PR-AUC, and Brier score were positive in the expected directions for every target, and the corresponding repeated-split statistical tests produced small p-values for all three measures across all six targets.

The final held-out test results demonstrated that performance varied substantially by prediction target. The strongest ROC-AUC values were observed for fainting-risk and flare-risk. Need-to-hydrate and need-to-rest produced strong combinations of ROC-AUC, PR-AUC, precision, recall, and F1, while fainting-risk remained substantially weaker on positive-class metrics because of severe class imbalance.

These findings support the research hypothesis that longitudinal information can provide additional predictive information beyond same-day measurements within this synthetic PaceMate-AI dataset. However, because the data and targets are synthetic, the results should be interpreted as evidence from a controlled machine-learning experiment rather than evidence of clinical effectiveness.

Real-world data, external validation, prospective evaluation, and clinical safety assessment would be required before drawing conclusions about clinical utility. 