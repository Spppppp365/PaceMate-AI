# PaceMate AI Project Journal

## August 3, 2026

### Goal
Rebuild and improve my previous POTS symptom prediction machine learning project.

### Starting Point

I have:
- Research paper
- Project concept
- Previous prototype experience

The original implementation was lost, so this rebuild will create a cleaner, better documented version.

### Today's Work

- Created project structure
- Set up GitHub workflow
- Created documentation system

### Decisions

The new version will focus on:
- Multi-output predictions
- Better documentation
- Reproducible machine learning pipeline

### Next Steps

- Organize dataset
- Build preprocessing pipeline
- Train baseline model
## Day 2

### Goal
Build the first version of the machine learning pipeline.

### Tasks
- Organize dataset
- Explore data
- Create preprocessing pipeline
- Train first model

## Day 2: Dataset Construction, Feature Engineering, and Baseline ML

### Completed

* Generated 500 synthetic participant profiles.
* Generated 90,000 longitudinal daily observations across 180 days per participant.
* Validated participant and daily datasets for missing values, duplicate IDs, participant counts, and expected ranges.
* Generated 89,500 next-day prediction target rows.
* Created six prediction targets:

  * `flare_risk`
  * `dizziness_risk`
  * `fatigue_risk`
  * `fainting_risk`
  * `need_to_hydrate`
  * `need_to_rest`
* Adjusted hydration and rest target thresholds after reviewing initial target distributions.
* Created historical features using previous-day values and 3-day/7-day rolling averages.
* Added trend and deficit features including HRV change, resting heart-rate change, symptom change, sleep deficit, and hydration deficit.
* Validated the training dataset:

  * 89,000 rows
  * 36 columns
  * 0 missing values
  * 0 duplicate participant/day rows
  * 500 participants
  * 178 training examples per participant
  * No future-derived feature columns detected
* Split the dataset at the participant level to prevent participant leakage:

  * 350 training participants
  * 75 validation participants
  * 75 test participants
  * No participant overlap between splits.
* Trained the first Random Forest baseline model for `flare_risk`.

### Baseline Flare-Risk Results

Validation set:

* Accuracy: 0.8553
* Precision: 0.5857
* Recall: 0.8942
* F1 Score: 0.7078
* ROC-AUC: 0.9413
* PR-AUC: 0.7897

Unseen participant test set:

* Accuracy: 0.8397
* Precision: 0.5420
* Recall: 0.8465
* F1 Score: 0.6609
* ROC-AUC: 0.9213
* PR-AUC: 0.7228

The most important features in the baseline model were previous symptom severity, 7-day symptom average, 3-day symptom average, and symptom change.

### Research Notes

The dataset is synthetic and the prediction targets are rule-generated. These results therefore demonstrate the behavior of the modeling pipeline rather than clinical effectiveness.

Participant-level splitting was used so that participants in the validation and test sets were not present in the training set.

The baseline model provides an initial benchmark for subsequent model development and comparison.

## Day 3: Feature Expansion, Model Tuning, and Threshold Analysis

### Goal

Improve the Day 2 flare-risk baseline model while maintaining strict participant-level separation between training, validation, and unseen test participants.

The main questions for Day 3 were:

* Can historical variability features improve prediction?
* Can Random Forest hyperparameters improve performance?
* Can adjusting the classification threshold improve the precision/recall tradeoff?
* Do improvements observed on the validation set generalize to completely unseen participants?

### Feature Engineering Experiment

Added eight historical variability features:

* hrv_3day_std
* hrv_7day_std
* symptom_3day_std
* symptom_7day_std
* sleep_3day_std
* sleep_7day_std
* water_3day_std
* water_7day_std

These features measure recent variability rather than only recent average values.

The resulting training dataset contained:

* 89,000 rows
* 44 total columns
* 32 model features

All eight new variability features were checked for missing values. None contained missing values in the final training dataset.

### Feature Engineering Results

The expanded model was compared with the Day 2 baseline.

Validation results:

* Baseline F1: 0.7078
* Expanded-feature F1: 0.7058
* Baseline PR-AUC: 0.7897
* Expanded-feature PR-AUC: 0.7886
* Baseline ROC-AUC: 0.9413
* Expanded-feature ROC-AUC: 0.9409

Unseen test participant results:

* Baseline F1: 0.6609
* Expanded-feature F1: 0.6593
* Baseline PR-AUC: 0.7228
* Expanded-feature PR-AUC: 0.7208
* Baseline ROC-AUC: 0.9213
* Expanded-feature ROC-AUC: 0.9204

The additional variability features did not improve predictive performance. This was retained as an experimental result because negative results are important to document during model development.

### Hyperparameter Tuning

Six Random Forest configurations were tested on the validation participants.

Baseline configuration:

* 300 trees
* Maximum depth: 12
* Minimum samples per leaf: 5

Configurations tested:

* Baseline: 300 trees, depth 12, minimum leaf 5
* Deeper: 300 trees, depth 16, minimum leaf 5
* More trees: 500 trees, depth 12, minimum leaf 5
* Smaller leaf: 300 trees, depth 12, minimum leaf 3
* Larger leaf: 300 trees, depth 12, minimum leaf 8
* Deeper and smaller leaf: 300 trees, depth 16, minimum leaf 3

Validation results:

Baseline:

* Precision: 0.5851
* Recall: 0.8892
* F1: 0.7058
* ROC-AUC: 0.9409
* PR-AUC: 0.7886

Deeper:

* Precision: 0.6108
* Recall: 0.8678
* F1: 0.7170
* ROC-AUC: 0.9415
* PR-AUC: 0.7899

More trees:

* Precision: 0.5866
* Recall: 0.8900
* F1: 0.7072
* ROC-AUC: 0.9410
* PR-AUC: 0.7890

Smaller leaf:

* Precision: 0.5878
* Recall: 0.8850
* F1: 0.7064
* ROC-AUC: 0.9407
* PR-AUC: 0.7880

Larger leaf:

* Precision: 0.5818
* Recall: 0.8942
* F1: 0.7049
* ROC-AUC: 0.9409
* PR-AUC: 0.7897

Deeper and smaller leaf:

* Precision: 0.6205
* Recall: 0.8491
* F1: 0.7170
* ROC-AUC: 0.9411
* PR-AUC: 0.7887

The best validation F1 was 0.7170, achieved by both deeper configurations.

The deeper configuration was selected because it achieved the same F1 while producing slightly better PR-AUC and ROC-AUC.

Selected configuration:

* n_estimators = 300
* max_depth = 16
* min_samples_leaf = 5
* class_weight = balanced

### Improved Model Evaluation

The selected deeper model was evaluated on completely unseen test participants.

Results:

* Accuracy: 0.8524
* Precision: 0.5705
* Recall: 0.8080
* F1 Score: 0.6688
* ROC-AUC: 0.9209
* PR-AUC: 0.7194

The deeper model improved validation F1 compared with the expanded-feature baseline, but the improvement did not translate into a higher test-set F1 score.

This demonstrates why the unseen participant test set is important: validation improvements do not necessarily generalize to new participants.

### Decision Threshold Experiment

The Random Forest classification threshold was varied to examine the precision/recall tradeoff.

Threshold 0.30:

* Precision: 0.5057
* Recall: 0.9538
* F1: 0.6609

Threshold 0.35:

* Precision: 0.5295
* Recall: 0.9350
* F1: 0.6762

Threshold 0.40:

* Precision: 0.5571
* Recall: 0.9167
* F1: 0.6931

Threshold 0.45:

* Precision: 0.5828
* Recall: 0.8953
* F1: 0.7060

Threshold 0.50:

* Precision: 0.6108
* Recall: 0.8678
* F1: 0.7170

Threshold 0.55:

* Precision: 0.6406
* Recall: 0.8315
* F1: 0.7236

Threshold 0.60:

* Precision: 0.6695
* Recall: 0.7872
* F1: 0.7236

The threshold of 0.55 was selected because it achieved the best validation F1 while retaining more recall than the 0.60 threshold.

This experiment showed that the model's operating point can be changed without retraining the underlying Random Forest.

### Final Unseen-Test Evaluation

The final model used:

* 32 features
* 300 trees
* Maximum depth: 16
* Minimum samples per leaf: 5
* Balanced class weighting
* Decision threshold: 0.55

The final model was evaluated on the completely unseen test participants.

Final results:

* Accuracy: 0.8624
* Precision: 0.6006
* Recall: 0.7584
* F1 Score: 0.6704
* ROC-AUC: 0.9209
* PR-AUC: 0.7194

Confusion matrix:

9645  1242
595   1868

Compared with the Day 2 baseline, the final threshold reduced false positives from 1,762 to 1,242 but increased false negatives from 378 to 595.

Therefore, the threshold adjustment produced a more conservative model: it generated fewer false alarms but missed more actual positive flare-risk examples.

### What Was Learned

Several important findings came from Day 3.

First, adding historical variability features did not automatically improve prediction. Although these features are conceptually meaningful, their addition produced slightly worse validation and test metrics in this synthetic dataset.

Second, increasing Random Forest depth improved validation performance. The best validation F1 increased from 0.7058 to 0.7170. However, this improvement did not translate into a higher unseen-test F1 score, demonstrating the importance of evaluating generalization to participants that were never used during training.

Third, decision threshold tuning significantly changed the precision/recall balance. A higher threshold reduced false positives while increasing false negatives.

Fourth, ROC-AUC and PR-AUC remained relatively stable across the experiments. This suggests that the experiments primarily changed the model's operating point rather than dramatically changing its underlying ability to rank higher-risk versus lower-risk observations.

Finally, because this project uses synthetic data and rule-generated targets, these results should be interpreted as evidence about the machine learning pipeline and experimental methodology rather than evidence of clinical effectiveness.

### Research Interpretation

The Day 3 experiments demonstrate an important part of the model-development process: not every technically reasonable improvement improves generalization.

The final model achieved strong discrimination according to ROC-AUC, but its precision and recall tradeoff remains a key consideration for future development.

For a health-related risk-support application, false negatives and false positives have different consequences. Future work should therefore evaluate threshold selection according to the intended use case rather than relying exclusively on F1 score.

### Day 3 Conclusion

Day 3 successfully expanded and tested the PaceMate-AI flare-risk modeling pipeline.

The project now has:

* Participant-level train, validation, and test separation
* 32 model features
* Historical variability features
* Hyperparameter experimentation
* Decision-threshold experimentation
* Unseen-participant evaluation
* Documented precision/recall tradeoffs
* Reproducible model-selection experiments

The model is not being treated as clinically validated. The current results establish a stronger experimental baseline for the next stage of development.
