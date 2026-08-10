# PaceMate-AI Project Journal

## Day 1 — Project Setup

### Goal

Rebuild and improve my previous POTS symptom prediction machine learning project.

### Starting Point

I had:

- A research paper from the previous project
- The original project concept
- Experience with an earlier prototype
- A goal of rebuilding the project with a cleaner and more reproducible machine learning pipeline

The original implementation was no longer available, so I started the project again from the beginning.

### Work Completed

- Created the PaceMate-AI project repository.
- Set up the project directory structure.
- Set up the GitHub workflow.
- Created the project documentation system.
- Began organizing the project around reproducible data generation, feature engineering, model training, and evaluation.

### Initial Project Direction

The rebuilt project would focus on predicting multiple daily POTS-related outcomes instead of only one outcome.

The six prediction targets were:

- `flare_risk`
- `dizziness_risk`
- `fatigue_risk`
- `fainting_risk`
- `need_to_hydrate`
- `need_to_rest`

The project would use participant-level dataset splitting so that the same participant would never appear in both the training and test sets.

### Initial Plan

1. Generate participant data.
2. Generate longitudinal daily observations.
3. Generate prediction targets.
4. Engineer historical features.
5. Validate the dataset.
6. Split participants into training, validation, and test groups.
7. Train baseline models.
8. Evaluate the models on participants that were not used during training.

## Day 2 — Dataset Construction and Baseline Model

### Goal

Build the first complete version of the machine learning dataset and establish a baseline flare-risk model.

### Participant Dataset

Generated 500 synthetic participant profiles.

The participant data included variables such as:

- Age
- Biological sex
- Height
- Weight
- Years since diagnosis
- Baseline symptom severity
- Medication group
- Compression garment use
- Mobility aid use
- Athlete status
- Caffeine sensitivity
- Baseline resting heart rate
- Baseline HRV
- Baseline water goal
- Baseline sleep goal

The participant generator used a fixed random seed so that the synthetic dataset could be reproduced.

### Daily Observations

Generated 90,000 longitudinal daily observations.

This represented:

- 500 participants
- 180 days per participant

The daily observation dataset included:

- `participant_id`
- `day`
- `sleep_hours`
- `water_intake_ml`
- `resting_hr`
- `hrv`
- `dizziness`
- `fatigue`
- `brain_fog`
- `symptom_severity`
- `activity_level`
- `stress_level`

### Target Generation

Generated 89,500 target rows.

The six prediction targets were:

- `flare_risk`
- `dizziness_risk`
- `fatigue_risk`
- `fainting_risk`
- `need_to_hydrate`
- `need_to_rest`

The target generator produced next-day prediction targets, which is why the number of target rows was smaller than the 90,000 daily observations.

The target distributions were:

### Flare Risk

- Negative: 73,250
- Positive: 16,250
- Positive rate: approximately 18.2%

### Dizziness Risk

- Negative: 63,146
- Positive: 26,354
- Positive rate: approximately 29.4%

### Fatigue Risk

- Negative: 57,052
- Positive: 32,448
- Positive rate: approximately 36.3%

### Fainting Risk

- Negative: 87,317
- Positive: 2,183
- Positive rate: approximately 2.4%

### Need to Hydrate

- Negative: 63,564
- Positive: 25,936
- Positive rate: approximately 29.0%

### Need to Rest

- Negative: 56,119
- Positive: 33,381
- Positive rate: approximately 37.3%

### Feature Engineering

Created historical features using previous observations and rolling windows.

The engineered features included:

- Previous-day sleep
- Previous-day water intake
- Previous-day resting heart rate
- Previous-day HRV
- Previous-day symptom severity
- Three-day sleep average
- Three-day water average
- Three-day HRV average
- Three-day symptom average
- Seven-day sleep average
- Seven-day water average
- Seven-day HRV average
- Seven-day symptom average
- HRV change
- Resting heart-rate change
- Symptom change
- Sleep deficit
- Hydration deficit

The resulting training dataset contained 89,000 rows and 36 columns.

The dataset was checked for:

- Missing values
- Duplicate participant/day combinations
- Participant counts
- Expected observation counts
- Future-derived features

No missing values or duplicate participant/day combinations were found, and no future-derived feature columns were detected.

### Participant-Level Dataset Split

The data was split by participant rather than randomly by row.

This was done to prevent the same participant’s history from appearing in both training and evaluation data.

The split was:

- 350 training participants
- 75 validation participants
- 75 test participants

The resulting row counts were:

- Training: 62,300
- Validation: 13,350
- Test: 13,350

There was no participant overlap between the three datasets.

This was an important part of the experimental design because a random row-level split could allow information from the same participant to appear in both training and testing.

### Baseline Flare-Risk Model

The first machine learning model was a Random Forest model predicting `flare_risk`.

### Validation Results

- Accuracy: 0.8553
- Precision: 0.5857
- Recall: 0.8942
- F1 Score: 0.7078
- ROC-AUC: 0.9413
- PR-AUC: 0.7897

### Unseen Test Participant Results

- Accuracy: 0.8397
- Precision: 0.5420
- Recall: 0.8465
- F1 Score: 0.6609
- ROC-AUC: 0.9213
- PR-AUC: 0.7228

The model performed well at ranking higher-risk observations above lower-risk observations, but performance decreased when evaluated on completely unseen participants.

### Baseline Feature Importance

The most important features included:

- Previous symptom severity
- Seven-day symptom average
- Three-day symptom average
- Symptom change

This suggested that recent symptom history was highly influential in the synthetic flare-risk prediction task.

### Day 2 Conclusion

Day 2 established the first complete reproducible version of the PaceMate-AI machine learning pipeline.

At this point the project had:

- 500 synthetic participants
- 180 days of observations per participant
- 90,000 daily observations
- Six prediction targets
- Historical feature engineering
- Participant-level train/validation/test splitting
- A baseline Random Forest model
- An unseen-participant test set
- Initial performance benchmarks

The results were treated as machine learning pipeline results rather than evidence of clinical effectiveness because the data and targets were synthetic.

## Day 3 — Feature Experiments, Model Tuning, and Threshold Selection

### Goal

Improve the flare-risk model and determine whether the improvements would generalize to participants who were completely excluded from model training.

The main questions were:

1. Would additional historical variability features improve prediction?
2. Would changing Random Forest hyperparameters improve performance?
3. Could changing the classification threshold improve the precision/recall tradeoff?
4. Would improvements seen on the validation participants also appear on the unseen test participants?

### Experiment 1 — Historical Variability Features

Added eight additional historical variability features:

- `hrv_3day_std`
- `hrv_7day_std`
- `symptom_3day_std`
- `symptom_7day_std`
- `sleep_3day_std`
- `sleep_7day_std`
- `water_3day_std`
- `water_7day_std`

These features were designed to represent recent variability rather than only recent averages.

The expanded dataset contained:

- 89,000 rows
- 44 total columns
- 32 model features

The new features were checked for missing values, and none contained missing values in the final dataset.

### Validation Comparison

Baseline model:

- F1: 0.7078
- PR-AUC: 0.7897
- ROC-AUC: 0.9413

Expanded-feature model:

- F1: 0.7058
- PR-AUC: 0.7886
- ROC-AUC: 0.9409

### Unseen Test Comparison

Baseline:

- F1: 0.6609
- PR-AUC: 0.7228
- ROC-AUC: 0.9213

Expanded-feature model:

- F1: 0.6593
- PR-AUC: 0.7208
- ROC-AUC: 0.9204

### Result

The additional variability features did not improve performance.

The experiment was retained as a documented negative result because it showed that adding features that appear theoretically useful does not necessarily improve a model.

### Experiment 2 — Random Forest Hyperparameter Tuning

Six Random Forest configurations were tested on the validation participants.

### Baseline Configuration

- 300 trees
- Maximum depth: 12
- Minimum samples per leaf: 5

### Configurations Tested

1. Baseline

   * 300 trees
   * Depth 12
   * Minimum leaf 5
2. Deeper

   * 300 trees
   * Depth 16
   * Minimum leaf 5
3. More trees

   * 500 trees
   * Depth 12
   * Minimum leaf 5
4. Smaller leaf

   * 300 trees
   * Depth 12
   * Minimum leaf 3
5. Larger leaf

   * 300 trees
   * Depth 12
   * Minimum leaf 8
6. Deeper and smaller leaf

   * 300 trees
   * Depth 16
   * Minimum leaf 3

All models used balanced class weighting.

### Validation Results

Baseline:

- Precision: 0.5851
- Recall: 0.8892
- F1: 0.7058
- ROC-AUC: 0.9409
- PR-AUC: 0.7886

Deeper:

- Precision: 0.6108
- Recall: 0.8678
- F1: 0.7170
- ROC-AUC: 0.9415
- PR-AUC: 0.7899

More trees:

- Precision: 0.5866
- Recall: 0.8900
- F1: 0.7072
- ROC-AUC: 0.9410
- PR-AUC: 0.7890

Smaller leaf:

- Precision: 0.5878
- Recall: 0.8850
- F1: 0.7064
- ROC-AUC: 0.9407
- PR-AUC: 0.7880

Larger leaf:

- Precision: 0.5818
- Recall: 0.8942
- F1: 0.7049
- ROC-AUC: 0.9409
- PR-AUC: 0.7897

Deeper and smaller leaf:

- Precision: 0.6205
- Recall: 0.8491
- F1: 0.7170
- ROC-AUC: 0.9411
- PR-AUC: 0.7887

### Model Selection

The best validation F1 score was 0.7170.

Two configurations achieved this:

- Deeper
- Deeper and smaller leaf

The deeper configuration was selected because it achieved the same F1 while having slightly better ROC-AUC and PR-AUC.

The selected configuration was:

- 300 trees
- Maximum depth: 16
- Minimum samples per leaf: 5
- Balanced class weighting

### Experiment 3 — Evaluation of the Tuned Model on Unseen Participants

The selected deeper model was evaluated on the 75 participants that had never been used for training.

Results:

- Accuracy: 0.8524
- Precision: 0.5705
- Recall: 0.8080
- F1 Score: 0.6688
- ROC-AUC: 0.9209
- PR-AUC: 0.7194

The model improved validation performance compared with the earlier model, but the improvement did not carry over to the unseen test participants.

This demonstrated that selecting a model based on validation performance does not guarantee that the same improvement will appear on completely new participants.

### Experiment 4 — Decision Threshold Testing

The next experiment changed the probability threshold used to convert the model’s predicted probability into a positive or negative prediction.

The underlying Random Forest model was not retrained.

The validation threshold results were:

Threshold 0.30:

- Precision: 0.5057
- Recall: 0.9538
- F1: 0.6609

Threshold 0.35:

- Precision: 0.5295
- Recall: 0.9350
- F1: 0.6762

Threshold 0.40:

- Precision: 0.5571
- Recall: 0.9167
- F1: 0.6931

Threshold 0.45:

- Precision: 0.5828
- Recall: 0.8953
- F1: 0.7060

Threshold 0.50:

- Precision: 0.6108
- Recall: 0.8678
- F1: 0.7170

Threshold 0.55:

- Precision: 0.6406
- Recall: 0.8315
- F1: 0.7236

Threshold 0.60:

- Precision: 0.6695
- Recall: 0.7872
- F1: 0.7236

The threshold of 0.55 was selected because it achieved the best validation F1 while retaining more recall than the 0.60 threshold.

This demonstrated that the operating point of the model could be changed without retraining the Random Forest.

## Day 3 Final Flare-Risk Model

The final flare-risk model selected at the end of Day 3 used:

- 32 features
- 300 Random Forest trees
- Maximum depth of 16
- Minimum samples per leaf of 5
- Balanced class weighting
- Classification threshold of 0.55

The final model was then evaluated on the completely unseen test participants.

### Final Test Results

- Accuracy: 0.8624
- Precision: 0.6006
- Recall: 0.7584
- F1 Score: 0.6704
- ROC-AUC: 0.9209
- PR-AUC: 0.7194

### Confusion Matrix

- True negatives: 9,645
- False positives: 1,242
- False negatives: 595
- True positives: 1,868

Compared with the earlier 0.50 operating point, the 0.55 threshold produced fewer false positives but more false negatives.

The model therefore became more conservative about classifying an observation as high risk.

## Day 3 — Multi-Target Model Development

After establishing the flare-risk pipeline, the project was expanded to production models for all six prediction targets.

The targets were:

- `flare_risk`
- `dizziness_risk`
- `fatigue_risk`
- `fainting_risk`
- `need_to_hydrate`
- `need_to_rest`

Separate training and calibration work was performed for the secondary targets.

Production model files were created for:

- Flare risk
- Dizziness risk
- Fatigue risk
- Fainting risk
- Hydration need
- Rest need

The final evaluation used the same completely unseen test participant group:

- 75 participants
- 13,350 test rows
- 32 model features

## Multi-Target Production Evaluation

### Flare Risk

Threshold:

- 0.55

Results:

- Accuracy: 0.8784
- Precision: 0.7526
- Recall: 0.5075
- F1 Score: 0.6062
- ROC-AUC: 0.9216
- PR-AUC: 0.7221
- Brier Score: 0.084342
- Positive rate: 0.1845
- Mean predicted probability: 0.1788

The direct probability check for the flare model produced:

- Mean probability: 0.178774
- Brier score: 0.084342
- ROC-AUC: 0.921615
- PR-AUC: 0.722091

The mean predicted probability was close to the actual test positive rate, which provided evidence that the final flare model’s average predicted probabilities were reasonably aligned with the observed test prevalence.

### Dizziness Risk

Threshold:

- 0.25

Results:

- Accuracy: 0.6032
- Precision: 0.4159
- Recall: 0.7809
- F1 Score: 0.5428
- ROC-AUC: 0.7145
- PR-AUC: 0.4883
- Brier Score: 0.186172
- Positive rate: 0.3016
- Mean predicted probability: 0.2971

The model showed relatively high recall at the selected threshold, but its overall discrimination was weaker than the flare-risk model.

### Fatigue Risk

Threshold:

- 0.30

Results:

- Accuracy: 0.6282
- Precision: 0.5021
- Recall: 0.7867
- F1 Score: 0.6130
- ROC-AUC: 0.7158
- PR-AUC: 0.5716
- Brier Score: 0.203489
- Positive rate: 0.3743
- Mean predicted probability: 0.3684

The model prioritized recall at the selected threshold, identifying a larger proportion of positive fatigue-risk observations.

### Fainting Risk

Threshold:

- 0.20

Results:

- Accuracy: 0.9604
- Precision: 0.2815
- Recall: 0.4174
- F1 Score: 0.3363
- ROC-AUC: 0.9516
- PR-AUC: 0.2599
- Brier Score: 0.019522
- Positive rate: 0.0240
- Mean predicted probability: 0.0213

Fainting risk was highly imbalanced, with only about 2.4% positive examples in the test set.

Because of this imbalance, accuracy alone was not a useful description of model performance.

The model had very strong ROC-AUC, but its precision and recall at the selected threshold were much weaker.

This target therefore requires additional work before it could be considered a strong practical prediction model.

### Need to Hydrate

Threshold:

- 0.35

Results:

- Accuracy: 0.8120
- Precision: 0.6528
- Recall: 0.8103
- F1 Score: 0.7231
- ROC-AUC: 0.8927
- PR-AUC: 0.7582
- Brier Score: 0.122473
- Positive rate: 0.3029
- Mean predicted probability: 0.2959

This was one of the stronger secondary models.

The model achieved relatively high precision and recall at the selected threshold.

### Need to Rest

Threshold:

- 0.35

Results:

- Accuracy: 0.7346
- Precision: 0.6142
- Recall: 0.8352
- F1 Score: 0.7078
- ROC-AUC: 0.8369
- PR-AUC: 0.7421
- Brier Score: 0.159831
- Positive rate: 0.3849
- Mean predicted probability: 0.3791

The rest model also performed reasonably well, particularly in recall.

## Multi-Target Threshold Evaluation

The final thresholds were evaluated again using the completely unseen test set.

### Final Thresholds

- Flare risk: 0.55
- Dizziness risk: 0.25
- Fatigue risk: 0.30
- Fainting risk: 0.20
- Need to hydrate: 0.35
- Need to rest: 0.35

### Flare Risk

- True negatives: 10,476
- False positives: 411
- False negatives: 1,213
- True positives: 1,250

### Dizziness Risk

- True negatives: 4,909
- False positives: 4,415
- False negatives: 882
- True positives: 3,144

### Fatigue Risk

- True negatives: 4,455
- False positives: 3,898
- False negatives: 1,066
- True positives: 3,931

### Fainting Risk

- True negatives: 12,687
- False positives: 342
- False negatives: 187
- True positives: 134

### Need to Hydrate

- True negatives: 7,563
- False positives: 1,743
- False negatives: 767
- True positives: 3,277

### Need to Rest

- True negatives: 5,515
- False positives: 2,696
- False negatives: 847
- True positives: 4,292

## Overall Findings

Several important findings came from the experiments.

First, adding historical variability features did not improve the flare-risk model. The additional features slightly decreased the main evaluation metrics.

Second, increasing Random Forest depth improved validation F1 from approximately 0.706 to 0.717, but this improvement did not translate into a better F1 score on completely unseen participants.

Third, threshold selection had a major effect on the precision and recall balance. A higher threshold generally reduced false positives while increasing false negatives.

Fourth, the six prediction targets did not perform equally well.

The hydration and rest models were among the stronger secondary models based on their combination of F1, PR-AUC, precision, and recall.

The flare-risk model showed strong discrimination based on ROC-AUC and PR-AUC.

The fainting model had very strong ROC-AUC but was affected by the severe class imbalance and had relatively weak precision, recall, and F1 at the selected threshold.

The dizziness and fatigue models were weaker than the flare, hydration, and rest models based on ROC-AUC and PR-AUC.

These differences are important because a multi-target system should not be treated as if every prediction target has the same level of reliability.

## Research Limitations

The dataset used in this project is synthetic.

The prediction targets are also generated from predefined rules rather than from real patient outcomes.

Therefore, the model results cannot be interpreted as evidence that PaceMate-AI can accurately predict real POTS symptoms in patients.

The current experiments demonstrate:

- Dataset construction
- Feature engineering
- Participant-level leakage prevention
- Model training
- Hyperparameter comparison
- Threshold selection
- Probability calibration
- Multi-target evaluation
- Generalization testing on unseen participants

They do not establish clinical validity.

Real-world validation with appropriately collected clinical data would be required before making clinical performance claims.

Another limitation is that the synthetic data-generating process may create relationships that are easier for machine learning models to detect than relationships in real-world patient data.

The strong performance of some models therefore needs to be interpreted in the context of how the synthetic targets were generated.

## Day 3 Conclusion

By the end of Day 3, PaceMate-AI had developed from a single baseline flare-risk model into a multi-target machine learning pipeline.

The project now included:

- 500 synthetic participants
- 90,000 daily observations
- 89,500 generated target rows
- 89,000 final training rows
- 62,300 training rows
- 13,350 validation rows
- 13,350 unseen test rows
- 32 model features
- Six prediction targets
- Participant-level dataset separation
- Random Forest model experimentation
- Hyperparameter tuning
- Historical variability feature experimentation
- Probability calibration
- Threshold tuning
- Production model generation
- Unified model evaluation
- Confusion-matrix analysis
- Precision and recall analysis
- ROC-AUC and PR-AUC evaluation
- Brier-score evaluation
- Documented negative experimental results

The most important lesson from these experiments was that improving a model on the validation set does not necessarily improve its performance on completely unseen participants.

The project therefore continued to use a separate participant-level test set as the main check of generalization.

The results establish a reproducible machine learning research pipeline, but they should not be interpreted as clinical validation.

