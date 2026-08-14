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

## Day 4

## Milestone: Production Model Validation, Calibration, Threshold Optimization, and Verification

Today I completed a major stage of the PaceMate-AI machine learning pipeline. The project now has calibrated production models for all six prediction targets, optimized decision thresholds, held-out test evaluation, and individual production-model verification scripts.

The participant-level dataset pipeline was successfully rerun from beginning to end. The project generated 500 participants and 90,000 daily observations covering 180 days per participant. The target-generation pipeline produced 89,500 target rows, and feature engineering produced 89,000 final training rows with 44 total columns.

The participant-level split was preserved throughout the modeling process:

- 350 participants for training
- 75 participants for validation
- 75 participants for testing
- 62,300 training rows
- 13,350 validation rows
- 13,350 test rows
- No participant overlap between training, validation, and test sets

This was important because the model must be evaluated on completely unseen participants rather than simply random rows from participants it has already seen.

I trained six separate prediction models for the PaceMate-AI targets:

- flare_risk
- dizziness_risk
- fatigue_risk
- fainting_risk
- need_to_hydrate
- need_to_rest

I then performed probability calibration experiments for the five secondary models that required calibration. Both sigmoid and isotonic calibration were tested against the original Random Forest probabilities.

The calibration experiments showed meaningful improvements in probability quality.

For dizziness risk, the Brier score improved from 0.207026 to 0.186172 using sigmoid calibration. ROC-AUC remained approximately 0.715, while PR-AUC remained approximately 0.488.

For fatigue risk, the Brier score improved from 0.212434 to 0.203489 using isotonic calibration. PR-AUC improved from 0.566625 to 0.571593.

For fainting risk, calibration produced the largest probability-quality improvement. The Brier score decreased from 0.041776 to 0.019522 using isotonic calibration. ROC-AUC remained approximately 0.952 and PR-AUC increased from 0.253716 to 0.259856.

For hydration risk, sigmoid calibration reduced the Brier score from 0.130671 to 0.122473. ROC-AUC was approximately 0.893 and PR-AUC was approximately 0.758.

For rest risk, isotonic calibration reduced the Brier score from 0.164215 to 0.159831. ROC-AUC was approximately 0.837 and PR-AUC was approximately 0.742.

Based on these experiments, I built the calibrated production models:

- Dizziness risk uses sigmoid calibration
- Fatigue risk uses isotonic calibration
- Fainting risk uses isotonic calibration
- Hydration risk uses sigmoid calibration
- Rest risk uses isotonic calibration
- Flare risk continues to use its dedicated production model

All six production models were successfully saved as joblib files in the models directory.

I then evaluated the unified production models on the completely unseen test participants.

The final test results were:

- Flare risk: ROC-AUC 0.9216, PR-AUC 0.7221, F1 0.6062, Brier score 0.0843
- Dizziness risk: ROC-AUC 0.7145, PR-AUC 0.4883, F1 0.5428, Brier score 0.1862
- Fatigue risk: ROC-AUC 0.7158, PR-AUC 0.5716, F1 0.6130, Brier score 0.2035
- Fainting risk: ROC-AUC 0.9516, PR-AUC 0.2599, F1 0.3363, Brier score 0.0195
- Hydration risk: ROC-AUC 0.8927, PR-AUC 0.7582, F1 0.7231, Brier score 0.1225
- Rest risk: ROC-AUC 0.8369, PR-AUC 0.7421, F1 0.7078, Brier score 0.1598

The models were also evaluated with decision thresholds rather than relying on the default 0.50 classification threshold. Threshold tuning was performed on the validation participants and the selected thresholds were then evaluated on the completely unseen test participants.

The selected thresholds are:

- Flare risk: 0.55
- Dizziness risk: 0.25
- Fatigue risk: 0.30
- Fainting risk: 0.20
- Hydration risk: 0.35
- Rest risk: 0.35

The validation-set threshold optimization produced the following F1 scores:

- Flare risk: 0.7236
- Dizziness risk: 0.5725
- Fatigue risk: 0.6245
- Fainting risk: 0.4404
- Hydration risk: 0.7298
- Rest risk: 0.7377

I then evaluated these fixed thresholds on the unseen test participants. This produced the final threshold evaluation results that are now saved in the results directory.

I also created individual production-model verification scripts for every production model. These scripts load the saved model, load the unseen test dataset, reproduce the expected feature set, generate probability predictions, verify that probabilities remain within the valid 0 to 1 range, and print representative predictions alongside the actual target values.

Verification successfully passed for:

- Dizziness risk
- Fatigue risk
- Fainting risk
- Hydration risk
- Rest risk
- Flare risk

The flare verification script initially did not exist, which caused a module-not-found error when I attempted to run it. I then created the missing verification script and successfully reran the verification.

The production models therefore now have both quantitative evaluation and basic artifact-level verification.

One important observation from today's work is that the models are not equally strong. Flare, hydration, and rest prediction currently show the strongest overall predictive performance. Dizziness and fatigue are substantially weaker, with ROC-AUC values around 0.71. Fainting has very strong ROC-AUC but a difficult class-imbalance problem because positive cases are rare. This means the model should not be presented as uniformly accurate across every target.

The calibration experiments also demonstrated why probability calibration is important. A model can have similar ROC-AUC before and after calibration while becoming substantially better at producing probabilities that correspond to actual observed frequencies. This is particularly important for PaceMate-AI because the intended system is based on risk estimates rather than only binary classifications.

At this point, the core machine-learning pipeline is substantially more complete than it was at the beginning of the summer. The project now has:

- Synthetic participant generation
- Longitudinal daily observation generation
- Target generation
- Historical feature engineering
- Participant-level train/validation/test splitting
- Six machine-learning targets
- Model training
- Calibration experiments
- Calibrated production models
- Threshold optimization
- Unseen-participant test evaluation
- Production-model verification
- Saved model artifacts
- Saved evaluation results

The next phase is to move beyond model training and turn these results into a complete research project. This includes improving and documenting the scientific methodology, performing deeper error and feature analysis, creating appropriate visualizations, developing the research paper, organizing the GitHub repository, creating the project website and portfolio presentation, producing a working demonstration, and preparing the competition materials.

My goal is to make PaceMate-AI a complete, reproducible, well-documented research project that can withstand serious technical and scientific scrutiny.

## Day 5

## Milestone: Feature Importance Analysis and Longitudinal Feature Ablation

Today the PaceMate-AI project continued its model analysis by investigating which features the trained models rely on most heavily and testing whether the longitudinal features provide meaningful predictive information.

The goal was to move beyond simply reporting model performance and determine why the models were able to make their predictions.

A secondary feature-importance analysis was created and run for three of the prediction targets:

- dizziness_risk
- fatigue_risk
- fainting_risk

The analysis used the trained Random Forest models and calculated feature importance values for all 32 model features.

For dizziness risk, the most important features included:

- symptom_7day_mean: 0.15279
- symptom_3day_mean: 0.11281
- previous_symptom_severity: 0.07098
- hrv_7day_mean: 0.03333
- hrv_3day_mean: 0.02939
- symptom_7day_std: 0.02734
- water_intake_ml: 0.02679
- sleep_7day_std: 0.02606
- hrv_7day_std: 0.02605
- water_7day_std: 0.02599

For fatigue risk, the strongest features were:

- symptom_7day_mean: 0.16164
- symptom_3day_mean: 0.11853
- previous_symptom_severity: 0.07372
- hrv_7day_mean: 0.02899
- symptom_7day_std: 0.02770
- sleep_7day_std: 0.02683
- water_7day_std: 0.02679
- hrv_7day_std: 0.02629
- water_3day_std: 0.02597
- water_intake_ml: 0.02564

For fainting risk, the strongest features were:

- symptom_3day_mean: 0.20195
- symptom_7day_mean: 0.16289
- previous_symptom_severity: 0.15733
- hrv_7day_mean: 0.07581
- hrv: 0.06172
- hrv_3day_mean: 0.06103
- previous_hrv: 0.04606
- symptom_change: 0.04372
- hrv_change: 0.01012
- hrv_7day_std: 0.00984

The feature-importance analysis showed that recent symptom history was consistently important across all three targets.

The fainting-risk model also relied more heavily on HRV-related features than the dizziness-risk and fatigue-risk models.

This provided additional evidence that the six prediction targets should not automatically be treated as identical prediction problems. Different targets can depend on different combinations of physiological and symptom-related variables.

The feature-importance results were saved to:

results/secondary_feature_importance.csv

The project then performed a longitudinal feature ablation study.

The purpose of this experiment was to directly test whether the historical and longitudinal features were actually contributing meaningful predictive information.

The full models used all 32 engineered features.

The comparison models used only six same-day features.

The dataset sizes were:

- Training rows: 62,300
- Validation rows: 13,350
- Test rows: 13,350
- Full feature count: 32
- Same-day feature count: 6

For flare risk, the full longitudinal model achieved:

- Accuracy: 0.8815
- Precision: 0.7501
- Recall: 0.5363
- F1: 0.6255
- ROC-AUC: 0.9211
- PR-AUC: 0.7247
- Brier score: 0.0847

The same-day-only model achieved:

- Accuracy: 0.8062
- Precision: 0.2961
- Recall: 0.0365
- F1: 0.0651
- ROC-AUC: 0.6169
- PR-AUC: 0.2541
- Brier score: 0.1511

The longitudinal features improved flare-risk ROC-AUC by 0.3041 and PR-AUC by 0.4705.

For dizziness risk, the full model achieved:

- Accuracy: 0.7109
- Precision: 0.5512
- Recall: 0.2218
- F1: 0.3163
- ROC-AUC: 0.7064
- PR-AUC: 0.4789
- Brier score: 0.1881

The same-day-only model achieved:

- Accuracy: 0.6859
- Precision: 0.3781
- Recall: 0.0643
- F1: 0.1100
- ROC-AUC: 0.5513
- PR-AUC: 0.3395
- Brier score: 0.2144

The longitudinal features improved dizziness-risk ROC-AUC by 0.1551 and PR-AUC by 0.1394.

For fatigue risk, the full model achieved:

- Accuracy: 0.6747
- Precision: 0.5889
- Recall: 0.4337
- F1: 0.4995
- ROC-AUC: 0.7098
- PR-AUC: 0.5638
- Brier score: 0.2054

The same-day-only model achieved:

- Accuracy: 0.6083
- Precision: 0.4295
- Recall: 0.1415
- F1: 0.2129
- ROC-AUC: 0.5228
- PR-AUC: 0.3985
- Brier score: 0.2413

The longitudinal features improved fatigue-risk ROC-AUC by 0.1870 and PR-AUC by 0.1654.

For fainting risk, the full model achieved:

- Accuracy: 0.9759
- Precision: 0.0000
- Recall: 0.0000
- F1: 0.0000
- ROC-AUC: 0.9434
- PR-AUC: 0.2400
- Brier score: 0.0198

The same-day-only model achieved:

- Accuracy: 0.9759
- Precision: 0.0000
- Recall: 0.0000
- F1: 0.0000
- ROC-AUC: 0.7745
- PR-AUC: 0.0582
- Brier score: 0.0237

The longitudinal features improved fainting-risk ROC-AUC by 0.1689 and PR-AUC by 0.1818.

Although the threshold-based classification metrics were zero for both models, the ranking metrics showed a substantial difference. This was especially important because fainting risk is highly imbalanced.

For need_to_hydrate, the full model achieved:

- Accuracy: 0.8130
- Precision: 0.7129
- Recall: 0.6405
- F1: 0.6747
- ROC-AUC: 0.8906
- PR-AUC: 0.7549
- Brier score: 0.1241

The same-day-only model achieved:

- Accuracy: 0.7349
- Precision: 0.5903
- Recall: 0.4080
- F1: 0.4825
- ROC-AUC: 0.7845
- PR-AUC: 0.5665
- Brier score: 0.1684

The longitudinal features improved hydration ROC-AUC by 0.1060 and PR-AUC by 0.1884.

For need_to_rest, the full model achieved:

- Accuracy: 0.7535
- Precision: 0.6898
- Recall: 0.6534
- F1: 0.6711
- ROC-AUC: 0.8341
- PR-AUC: 0.7372
- Brier score: 0.1618

The same-day-only model achieved:

- Accuracy: 0.6219
- Precision: 0.5143
- Recall: 0.3224
- F1: 0.3964
- ROC-AUC: 0.6150
- PR-AUC: 0.4896
- Brier score: 0.2327

The longitudinal features improved rest-risk ROC-AUC by 0.2190 and PR-AUC by 0.2475.

The ablation study provided strong experimental evidence that longitudinal information is an important part of the PaceMate-AI architecture.

The largest ROC-AUC improvement occurred for flare risk, with an improvement of 0.3041.

The largest PR-AUC improvement also occurred for flare risk, with an improvement of 0.4705.

Need_to_rest had the second-largest ROC-AUC improvement at 0.2190.

Need_to_hydrate had a particularly large PR-AUC improvement of 0.1884.

The results also helped explain why features such as previous symptom severity, three-day symptom averages, seven-day symptom averages, previous HRV, and HRV rolling averages repeatedly appeared among the most important predictors.

The ablation study was saved to:

results/longitudinal_feature_ablation.csv

The analysis script was saved to:

generators/ablation_longitudinal_features.py

The feature-importance analysis script was updated and retained in:

generators/analyze_feature_importance.py

At the end of the day's work, Git was checked and the working tree was clean.

Today's work strengthened the scientific methodology of PaceMate-AI because it moved the analysis beyond simply reporting model scores and tested a specific hypothesis about the importance of longitudinal information.

The main findings from today were:

- Recent symptom history was consistently among the most important predictors.
- HRV-related features were particularly important for the fainting-risk model.
- Longitudinal features substantially improved performance compared with same-day-only features.
- The benefit of longitudinal information was present across all six prediction targets.
- The largest longitudinal improvement occurred for flare risk.
- Fainting risk remains strongly affected by class imbalance and should not be judged using accuracy alone.
- Different prediction targets rely on different feature patterns.
- The longitudinal feature architecture is supported by an explicit ablation experiment rather than only theoretical reasoning.

The project remains based on synthetic data, so these findings demonstrate the behavior of the machine-learning pipeline rather than clinical effectiveness.

The next stage can focus on deeper model error analysis, examining false positives and false negatives, creating research-quality visualizations, and continuing to document the scientific methodology.

## Day 6

## Milestone: Model Error Analysis and Research-Quality Error Visualization

Today the PaceMate-AI project moved into a deeper stage of model evaluation by analyzing not only overall performance metrics, but also the specific types of predictions the model made correctly and incorrectly.

The goal was to understand the model's error patterns, compare false positives and false negatives, and determine whether the incorrect predictions showed identifiable differences in their underlying features.

### Model Error Analysis

A new model error analysis pipeline was created to examine the final flare-risk model's predictions on the unseen test participants.

The analysis classified every test prediction into one of four outcome groups:

- true_negative

- false_positive

- false_negative

- true_positive

The resulting error-analysis dataset was saved to:

results/model_error_analysis.csv

The analysis contained 80,100 test examples.

The prediction outcomes were:

- True negatives: 45,605

- False positives: 13,505

- False negatives: 4,962

- True positives: 16,028

The overall error rate was 0.2305.

### Prediction Probability Analysis

The predicted probability distributions were examined separately for false positives and false negatives.

False positives had:

- Mean predicted probability: 0.4742

- Median predicted probability: 0.4573

False negatives had:

- Mean predicted probability: 0.2233

- Median predicted probability: 0.2111

This showed that false positives generally occurred at substantially higher predicted probabilities than false negatives.

The error probability analysis was visualized in:

results/figures/error_probability_distribution.png

### Error Group Feature Analysis

The project compared feature values across the four prediction outcome groups.

For previous symptom severity, the mean values were:

- True negative: 5.2157

- False positive: 7.3810

- False negative: 6.6503

- True positive: 7.6376

For three-day symptom mean:

- True negative: 5.2403

- False positive: 7.3325

- False negative: 6.5368

- True positive: 7.5495

For seven-day symptom mean:

- True negative: 5.2706

- False positive: 7.1812

- False negative: 6.4756

- True positive: 7.4053

These results showed a consistent pattern in which higher recent symptom-severity levels were associated with the positive prediction groups.

The HRV features showed the opposite general pattern.

For mean HRV:

- True negative: 44.1716

- False positive: 37.8320

- False negative: 40.4609

- True positive: 33.0294

For previous HRV:

- True negative: 43.9849

- False positive: 37.8010

- False negative: 40.7420

- True positive: 33.4629

This indicated that lower HRV values were associated with higher-risk prediction groups in the synthetic dataset.

The complete feature-group analysis was saved to:

results/error_group_feature_analysis.csv

### Longitudinal Error Patterns

The analysis also examined changes and rolling statistics rather than only individual observations.

The mean symptom change values were:

- True negative: -0.0430

- False positive: 0.0803

- False negative: 0.1264

- True positive: 0.2214

The true-positive group therefore had the largest average symptom increase.

The mean HRV change values were:

- True negative: 0.2273

- False positive: -1.2579

- False negative: 0.3669

- True positive: -1.7688

The true-positive group therefore showed the largest average decrease in HRV.

The analysis also examined sleep deficit and hydration deficit.

Mean sleep deficit:

- True negative: 0.0276

- False positive: -0.1617

- False negative: -0.1529

- True positive: -0.1827

Mean hydration deficit:

- True negative: -214.5342

- False positive: -98.4542

- False negative: -133.9913

- True positive: -259.1870

Hydration deficit showed the largest numerical difference among the analyzed feature groups.

These patterns were visualized in:

results/figures/longitudinal_error_feature_patterns.png

### Research Visualizations

Four research visualizations were generated.

The first visualization shows the number of observations in each prediction outcome group:

results/figures/prediction_outcomes.png

The second visualization shows the predicted probability distributions for false positives and false negatives:

results/figures/error_probability_distribution.png

The third visualization compares selected feature means across true negatives, false positives, false negatives, and true positives:

results/figures/error_group_feature_comparison.png

The fourth visualization examines longitudinal feature patterns across the prediction outcome groups:

results/figures/longitudinal_error_feature_patterns.png

### Error Analysis Summary

A dedicated error-analysis summary was generated and saved to:

results/error_analysis_summary.txt

The summary reported:

- Test examples analyzed: 80,100

- True negatives: 45,605

- False positives: 13,505

- False negatives: 4,962

- True positives: 16,028

- Overall error rate: 0.2305

The summary also documented differences in predicted probability and longitudinal feature values across prediction outcome groups.

### Structured Error Analysis Results Table

A structured results table was created and saved to:

results/error_analysis_results_table.csv

The table contains counts, probability statistics, threshold statistics, and mean feature values for each error group.

Selected results included:

True negative:

- Count: 45,605

- Mean probability: 0.082670

- Mean symptom severity: 4.999395

- Mean previous symptom severity: 5.014593

- Mean HRV: 44.171582

- Mean previous HRV: 43.984914

- Mean hydration deficit: -258.824397

False positive:

- Count: 13,505

- Mean probability: 0.474239

- Mean symptom severity: 6.403288

- Mean previous symptom severity: 6.446486

- Mean HRV: 37.831988

- Mean previous HRV: 37.800963

- Mean hydration deficit: -116.936567

False negative:

- Count: 4,962

- Mean probability: 0.223271

- Mean symptom severity: 5.562475

- Mean previous symptom severity: 5.428859

- Mean HRV: 40.460903

- Mean previous HRV: 40.742040

- Mean hydration deficit: -191.121896

True positive:

- Count: 16,028

- Mean probability: 0.579188

- Mean symptom severity: 6.845845

- Mean previous symptom severity: 6.802702

- Mean HRV: 33.029386

- Mean previous HRV: 33.462940

- Mean hydration deficit: -144.458284

### Error Analysis Report

A complete Markdown research report was generated and saved to:

results/error_analysis_report.md

The report consolidates the error-analysis findings into a research-oriented format.

It documents:

- Error-group distributions

- Prediction probability patterns

- Feature differences between error groups

- Longitudinal feature patterns

- Interpretation of false positives and false negatives

- Limitations of the analysis

The report emphasizes that the dataset is synthetic and that the observed relationships cannot establish clinical validity or causal relationships.

### Technical Issues Resolved

During visualization development, the error-analysis visualization script initially expected a column named predicted_probability.

The actual dataset used the column:

probability

The visualization script was corrected to use the actual dataset schema.

A second compatibility issue occurred because the installed version of Matplotlib did not accept the labels argument in plt.boxplot().

The plotting code was corrected to use the compatible parameter.

After these corrections, the visualization pipeline executed successfully and generated all four expected figures.

The summary-generation script also initially produced a formatting error because duplicate column names caused a pandas Series to be returned where a scalar value was expected.

The summary-generation code was corrected so that the error-group statistics could be formatted correctly.

The corrected summary script then executed successfully and produced:

results/error_analysis_summary.txt

### GitHub Integration

All Day 6 error-analysis work was added to the Git repository.

The committed files included:

- generators/analyze_model_errors.py

- generators/compare_error_groups.py

- generators/create_error_analysis_report.py

- generators/create_error_analysis_results_table.py

- generators/create_error_analysis_summary.py

- generators/create_error_analysis_visualizations.py

- results/error_analysis_report.md

- results/error_analysis_results_table.csv

- results/error_analysis_summary.txt

- results/error_group_feature_analysis.csv

- results/model_error_analysis.csv

The four generated visualization files were also committed:

- results/figures/prediction_outcomes.png

- results/figures/error_probability_distribution.png

- results/figures/error_group_feature_comparison.png

- results/figures/longitudinal_error_feature_patterns.png

PROJECT_JOURNAL.md was also included in the commit.

The changes were successfully pushed to GitHub.

The final Git status showed:

On branch main

Your branch is up to date with origin/main.

nothing to commit, working tree clean

### Day 6 Findings

Today's analysis showed that the model's errors were not randomly distributed across the synthetic dataset.

Higher recent symptom severity was associated with both false-positive and true-positive predictions.

Lower HRV values were particularly common among true-positive predictions.

True-positive examples also showed larger average increases in symptom severity and larger average decreases in HRV.

False positives generally had substantially higher predicted probabilities than false negatives.

The error analysis therefore provided a more detailed understanding of where the model succeeds and where it makes mistakes.

An important finding was that false positives were not simply low-confidence predictions. Their average predicted probability was 0.4742, meaning many were relatively close to the classification threshold.

This suggests that threshold analysis and calibration remain important parts of understanding the model's behavior.

### Limitations

The entire error analysis is based on synthetic data.

The observed relationships between symptoms, HRV, hydration, and prediction outcomes reflect the synthetic data-generation process and model behavior.

They should not be interpreted as evidence that these variables have the same predictive relationships in real POTS patients.

The analysis also focuses primarily on the flare-risk model rather than treating all six prediction targets as equally analyzed.

Future work should extend error analysis to the remaining prediction targets where appropriate.

### Day 6 Conclusion

Day 6 expanded PaceMate-AI from model-performance evaluation into detailed error analysis.

The project now includes not only model metrics, calibration results, feature importance, and longitudinal ablation experiments, but also a structured analysis of false positives, false negatives, true positives, and true negatives.

The project now has:

- Model error-analysis data

- Error-group feature analysis

- Structured error-analysis results

- Error-analysis summary

- Research-oriented Markdown report

- Four error-analysis visualizations

- Longitudinal error-pattern analysis

- Prediction probability analysis

- Documented error-analysis limitations

- GitHub-tracked analysis scripts

- GitHub-tracked research outputs

The most important methodological result from Day 6 is that the project is now evaluating model behavior at multiple levels rather than relying on a single performance metric.

The workflow now examines:

- Model discrimination

- Probability quality

- Classification thresholds

- Feature importance

- Longitudinal feature contribution

- Prediction errors

- Error-group feature patterns

- Generalization to unseen participants

This provides a substantially stronger foundation for the research paper and future experimental work.

The next stage will build on this analysis by continuing the scientific documentation, refining the research presentation, organizing the repository, developing research-quality figures and tables, and preparing the project for external evaluation and competition submission.

## Day 7

## Milestone: Final Production Model Comparison and Research Visualization

Today the PaceMate-AI project completed another major stage of model evaluation by creating a unified comparison of all six final production models and generating research-quality visualizations of their performance.

The goal of today's work was to move from analyzing individual models to evaluating the complete multi-target system as a whole.

This is important because PaceMate-AI does not rely on a single prediction target. The system contains six separate prediction models, and their strengths and weaknesses need to be evaluated together.

The six final prediction targets are:

- flare_risk
- dizziness_risk
- fatigue_risk
- fainting_risk
- need_to_hydrate
- need_to_rest

### Final Production Model Comparison

A unified final production-model comparison was generated using the completely unseen test participants.

The comparison includes:

- Classification threshold
- Positive prediction rate
- Mean predicted probability
- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- PR-AUC
- Brier score
- True negatives
- False positives
- False negatives
- True positives

The final comparison was generated with:

generators/create_model_comparison.py

The resulting comparison was saved to:

results/final_model_comparison.csv

A human-readable summary was saved to:

results/final_model_comparison_summary.txt

### Final Production Model Results

The final results for each production model were:

### Flare Risk

Threshold:

- 0.55

Performance:

- Accuracy: 0.8784
- Precision: 0.7526
- Recall: 0.5075
- F1: 0.6062
- ROC-AUC: 0.9216
- PR-AUC: 0.7221
- Brier score: 0.0843

The flare-risk model demonstrated strong discrimination and the highest precision among the six models.

Its ROC-AUC of 0.9216 indicates strong ranking performance on the unseen participants.

The selected threshold of 0.55 produced a relatively conservative classification strategy, resulting in higher precision but lower recall.

### Dizziness Risk

Threshold:

- 0.25

Performance:

- Accuracy: 0.6032
- Precision: 0.4159
- Recall: 0.7809
- F1: 0.5428
- ROC-AUC: 0.7145
- PR-AUC: 0.4883
- Brier score: 0.1862

The dizziness-risk model prioritized recall at its selected threshold.

Its ROC-AUC and PR-AUC were substantially lower than those of the strongest models, indicating weaker overall discrimination.

### Fatigue Risk

Threshold:

- 0.30

Performance:

- Accuracy: 0.6282
- Precision: 0.5021
- Recall: 0.7867
- F1: 0.6130
- ROC-AUC: 0.7158
- PR-AUC: 0.5716
- Brier score: 0.2035

The fatigue-risk model also emphasized recall.

Its discrimination was similar to the dizziness-risk model, although its F1 and PR-AUC were somewhat stronger.

### Fainting Risk

Threshold:

- 0.20

Performance:

- Accuracy: 0.9604
- Precision: 0.2815
- Recall: 0.4174
- F1: 0.3363
- ROC-AUC: 0.9516
- PR-AUC: 0.2599
- Brier score: 0.0195

The fainting-risk model achieved the highest ROC-AUC of all six models at 0.9516.

It also achieved the lowest Brier score at 0.0195.

However, fainting risk is highly imbalanced, with only approximately 2.4% positive examples.

Because of this imbalance, the high accuracy and high ROC-AUC should not be interpreted as meaning that the model reliably identifies individual fainting-risk events.

The relatively low precision, recall, and F1 demonstrate that the threshold-based classification performance remains limited.

This remains one of the most important areas for future improvement.

### Need to Hydrate

Threshold:

- 0.35

Performance:

- Accuracy: 0.8120
- Precision: 0.6528
- Recall: 0.8103
- F1: 0.7231
- ROC-AUC: 0.8927
- PR-AUC: 0.7582
- Brier score: 0.1225

The hydration model produced the strongest overall F1 score of the six production models.

It also produced the strongest PR-AUC at 0.7582.

The combination of precision, recall, F1, ROC-AUC, and PR-AUC makes need_to_hydrate one of the strongest-performing targets in the current system.

### Need to Rest

Threshold:

- 0.35

Performance:

- Accuracy: 0.7346
- Precision: 0.6142
- Recall: 0.8352
- F1: 0.7078
- ROC-AUC: 0.8369
- PR-AUC: 0.7421
- Brier score: 0.1598

The rest model achieved the highest recall of all six models at 0.8352.

It also achieved the second-highest F1 score and a strong PR-AUC.

This indicates that the model was particularly effective at identifying positive rest-need examples at its selected operating threshold.

### Overall Model Comparison

The unified comparison showed that no single model was strongest across every metric.

The strongest F1 score was achieved by:

- need_to_hydrate: 0.7231

The strongest ROC-AUC was achieved by:

- fainting_risk: 0.9516

The strongest PR-AUC was achieved by:

- need_to_hydrate: 0.7582

The lowest Brier score was achieved by:

- fainting_risk: 0.0195

The highest recall was achieved by:

- need_to_rest: 0.8352

The highest precision was achieved by:

- flare_risk: 0.7526

These results reinforce the importance of evaluating multiple metrics rather than selecting a model based on accuracy alone.

For example, fainting risk had the highest ROC-AUC and lowest Brier score but had substantially weaker threshold-based precision, recall, and F1.

Conversely, need_to_hydrate had a lower ROC-AUC than fainting risk but produced the strongest F1 and PR-AUC combination.

This demonstrates that different metrics answer different questions about model behavior.

### Probability Alignment

The comparison also included the positive prediction rate and mean predicted probability for each target.

For flare risk:

- Positive prediction rate: 0.1845
- Mean predicted probability: 0.1788

For dizziness risk:

- Positive prediction rate: 0.3016
- Mean predicted probability: 0.2971

For fatigue risk:

- Positive prediction rate: 0.3743
- Mean predicted probability: 0.3684

For fainting risk:

- Positive prediction rate: 0.0240
- Mean predicted probability: 0.0213

For need_to_hydrate:

- Positive prediction rate: 0.3029
- Mean predicted probability: 0.2959

For need_to_rest:

- Positive prediction rate: 0.3849
- Mean predicted probability: 0.3791

The mean predicted probabilities were generally close to the observed positive rates.

This provides an additional indication that the probability outputs were reasonably aligned with the prevalence of the corresponding targets in the unseen test set.

However, this comparison alone does not establish perfect calibration.

### Final Confusion-Matrix Comparison

The unified analysis also preserved the complete confusion-matrix counts for each target.

Flare risk:

- True negatives: 10,476
- False positives: 411
- False negatives: 1,213
- True positives: 1,250

Dizziness risk:

- True negatives: 4,909
- False positives: 4,415
- False negatives: 882
- True positives: 3,144

Fatigue risk:

- True negatives: 4,455
- False positives: 3,898
- False negatives: 1,066
- True positives: 3,931

Fainting risk:

- True negatives: 12,687
- False positives: 342
- False negatives: 187
- True positives: 134

Need to hydrate:

- True negatives: 7,563
- False positives: 1,743
- False negatives: 767
- True positives: 3,277

Need to rest:

- True negatives: 5,515
- False positives: 2,696
- False negatives: 847
- True positives: 4,292

These values provide a more complete picture of the classification behavior than aggregate metrics alone.

### Research-Quality Model Visualizations

A dedicated visualization pipeline was created to present the final model comparison graphically.

The visualization script was:

generators/create_model_comparison_visualizations.py

Four research figures were generated.

The first figure compares model discrimination using ROC-AUC and PR-AUC:

results/figures/model_discrimination_comparison.png

The second figure compares the main classification metrics across all six models:

results/figures/model_classification_metrics.png

The third figure compares Brier scores across the models:

results/figures/model_brier_score_comparison.png

The fourth figure compares observed positive prediction rates with mean predicted probabilities:

results/figures/model_probability_alignment.png

These visualizations provide a more accessible way to communicate the differences between the six prediction targets.

They will also be useful for the research paper, project website, presentation, and competition materials.

### Scientific Interpretation

Today's comparison demonstrated that PaceMate-AI should be treated as a multi-target system rather than as one model with six interchangeable outputs.

The prediction targets have different levels of difficulty and different performance characteristics.

The strongest overall targets currently appear to be:

- need_to_hydrate
- need_to_rest
- flare_risk

The more challenging targets currently appear to be:

- dizziness_risk
- fatigue_risk
- fainting_risk

The fainting model requires special interpretation because of the severe class imbalance.

The results also demonstrate why multiple evaluation metrics are necessary.

Accuracy alone would make fainting risk appear extremely strong, but precision, recall, and F1 reveal that identifying individual positive cases remains difficult.

Similarly, ROC-AUC alone would not adequately communicate the practical classification behavior of the models.

PR-AUC, F1, precision, recall, and Brier score therefore provide important complementary information.

### Relationship to Previous Experiments

Today's model comparison builds directly on the previous work completed during Days 4–6.

Day 4 established calibrated production models and optimized thresholds.

Day 5 demonstrated that longitudinal features substantially improved prediction compared with same-day-only features.

Day 6 examined the detailed error patterns of the flare-risk model.

Day 7 now places the six final production models into a single evaluation framework.

This creates a progression from:

- Model construction
- Model calibration
- Feature analysis
- Longitudinal ablation
- Error analysis
- Multi-target comparison

The project is therefore moving from simply building models toward constructing a complete research methodology around the models.

### Research Limitations

The models continue to be evaluated entirely using synthetic data.

The six targets were generated according to predefined synthetic rules.

The relationships identified by the models therefore reflect the synthetic data-generating process and should not be interpreted as established clinical relationships.

The results demonstrate the technical behavior of the machine-learning pipeline but do not establish clinical effectiveness.

Additional limitations include class imbalance for fainting risk and weaker discrimination for dizziness and fatigue.

Future work should continue investigating these limitations rather than presenting the current system as uniformly accurate.

### Day 7 Conclusion

Day 7 completed a unified comparison of all six PaceMate-AI production models and produced four research-quality model-comparison visualizations.

The project now contains:

- Six final production models
- Calibrated probability outputs
- Optimized decision thresholds
- Unseen-participant evaluation
- Confusion-matrix analysis
- Feature-importance analysis
- Longitudinal feature ablation
- Detailed flare-risk error analysis
- Unified six-target model comparison
- Model discrimination visualization
- Classification-metric visualization
- Brier-score visualization
- Probability-alignment visualization
- Structured CSV comparison results
- Human-readable model comparison summary

The most important result from today's work is that the six targets have substantially different performance profiles.

Need_to_hydrate currently provides the strongest overall balance of F1 and PR-AUC.

Need_to_rest provides the highest recall.

Flare risk provides the highest precision among the six models.

Fainting risk provides the strongest ROC-AUC and lowest Brier score but remains difficult to classify because of severe class imbalance.

Dizziness and fatigue remain weaker prediction targets and should receive additional investigation before any claims of strong predictive performance are made.

The next phase of the project can now move toward consolidating the experimental results into a formal research methodology, creating publication-quality tables and figures, improving the research paper, organizing the repository, and preparing the project for external evaluation.

As throughout the project, all current findings should be interpreted as machine-learning experiments on synthetic data rather than evidence of clinical effectiveness.

## Day 8

## Milestone: Final Threshold Evaluation, Production Verification, and Project Documentation

Today the PaceMate-AI project completed another verification stage for the production machine-learning pipeline.

The main goal was to ensure that the final threshold-selection results were reproducible, that the evaluation scripts were synchronized with the latest model configuration, and that the project documentation accurately reflected the completed experimental work.

### Final Threshold Evaluation

The final production thresholds were evaluated using the completely unseen test participants.

The selected thresholds remained:

- Flare risk: 0.55
- Dizziness risk: 0.25
- Fatigue risk: 0.30
- Fainting risk: 0.20
- Need to hydrate: 0.35
- Need to rest: 0.35

The threshold-evaluation pipeline was rerun using the current production models and the held-out test dataset.

The resulting evaluation was saved to:

results/final_threshold_evaluation.csv

The evaluation confirmed the classification behavior of the final production thresholds on participants who were not used during model training or threshold selection.

### Threshold Evaluation Scripts

The threshold evaluation pipeline was reviewed and updated to ensure that the current production configuration was being evaluated consistently.

The relevant scripts included:

generators/evaluate_all_thresholds.py

generators/tune_all_thresholds.py

These scripts are responsible for evaluating candidate thresholds and determining the selected operating points for the production models.

The scripts were checked against the existing production-model and evaluation workflow so that threshold selection remained separated from the final unseen test evaluation.

This distinction is important because the test participants should not be used to select the thresholds.

The validation participants were used for threshold optimization, while the completely unseen test participants were reserved for final evaluation.

### Production Model Verification

The production model artifacts were also verified using the existing verification workflow.

The verification process checks that the saved models can be loaded successfully, that the expected feature set is available, and that the models can generate valid probability predictions on the unseen test dataset.

The probability outputs were checked to ensure that they remained within the valid range of 0 to 1.

This provides an additional artifact-level check beyond the performance metrics reported in the model evaluation.

### Reproducibility Check

The project was reviewed to ensure that the major stages of the machine-learning pipeline remained represented by executable scripts and saved outputs.

The current workflow includes:

- Participant generation
- Daily observation generation
- Target generation
- Feature engineering
- Dataset validation
- Participant-level dataset splitting
- Model training
- Hyperparameter experimentation
- Probability calibration
- Threshold optimization
- Production model generation
- Unseen-participant evaluation
- Feature importance analysis
- Longitudinal feature ablation
- Error analysis
- Model comparison
- Threshold verification
- Production artifact verification

The project therefore contains both the code used to perform the experiments and the resulting datasets, metrics, visualizations, and model artifacts.

### Project Journal Update

The PaceMate-AI project journal was updated to document the latest experimental and verification work.

The journal now records the progression of the project from the initial machine-learning pipeline through:

- Dataset construction
- Baseline modeling
- Feature experimentation
- Hyperparameter tuning
- Threshold selection
- Multi-target model development
- Probability calibration
- Production model creation
- Longitudinal feature ablation
- Feature importance analysis
- Error analysis
- Research-quality visualization
- Six-target model comparison
- Final threshold evaluation
- Production-model verification

Maintaining the journal alongside the code and experimental outputs provides a chronological record of the research process.

This also preserves negative experimental results rather than only documenting successful improvements.

### Scientific Significance

Today's work primarily strengthened the reproducibility and documentation of the project rather than introducing a new modeling technique.

The threshold-selection workflow now clearly separates model development, threshold optimization, and final test evaluation.

This prevents the unseen test participants from influencing the selected operating thresholds.

The production verification process also provides an additional check that the saved model artifacts continue to function correctly outside of the original training scripts.

Together, these steps make the machine-learning pipeline easier to reproduce and audit.

### Limitations

The project continues to rely entirely on synthetic data.

The prediction targets were generated using predefined rules rather than observations from real patients.

Therefore, the threshold results and model performance metrics demonstrate the behavior of the experimental machine-learning system but do not establish clinical effectiveness.

The test set is unseen with respect to model training and threshold selection, but it is still derived from the same synthetic data-generating process as the training and validation data.

Real-world validation would therefore be necessary to determine whether the observed performance generalizes to actual patient populations.

### Day 8 Conclusion

Day 8 strengthened the final PaceMate-AI pipeline through threshold evaluation, production-model verification, reproducibility checks, and project documentation.

The project now has a clearly documented separation between:

- Model training
- Validation
- Calibration
- Threshold selection
- Final unseen-participant testing
- Production artifact verification

At this stage, PaceMate-AI has developed into a substantially complete machine-learning research pipeline rather than only a model-training experiment.

The next phase can focus on consolidating the results into the formal research paper, creating publication-quality tables and figures, improving the scientific methodology section, organizing the GitHub repository, and preparing the project for external evaluation and competition submission.

All current results continue to be interpreted as experiments on synthetic data rather than evidence of clinical effectiveness. 

### Day 9 

- Today I completed the PaceMate-AI robustness analysis for the six prediction targets: flare_risk, dizziness_risk, fatigue_risk, fainting_risk, need_to_hydrate, and need_to_rest.

- I used data/training_dataset.csv as the input dataset.

- I first ran Experiment 1A, Repeated Discrimination Robustness.

- Experiment 1A used five participant-level splits with random seeds 42, 123, 456, 789, and 1000.

- For every split, participants were randomly shuffled and divided into 70% training, 15% validation, and 15% testing. With 500 total participants, each split contained 350 training participants, 75 validation participants, and 75 test participants.

- Participant overlap between the training, validation, and test groups was explicitly checked, and the experiment stopped with an error if overlap was detected.

- The Random Forest configuration for Experiment 1A used 300 estimators, max_depth=16, min_samples_leaf=5, random_state=42, n_jobs=-1, and class_weight="balanced".

- Features excluded from model training included participant_id, day, the current symptom variables dizziness, fatigue, brain_fog, and symptom_severity, and all six target columns.

- For every target and participant split, I calculated validation ROC-AUC, validation PR-AUC, test ROC-AUC, and test PR-AUC.

- The complete Experiment 1A results were saved to results/repeated_discrimination_robustness.csv.

- Across the five participant-level splits, the mean test ROC-AUC was 0.9216 for flare_risk, 0.7175 for dizziness_risk, 0.7144 for fatigue_risk, 0.9481 for fainting_risk, 0.8882 for need_to_hydrate, and 0.8432 for need_to_rest.

- The corresponding test ROC-AUC standard deviations were 0.0209 for flare_risk, 0.0193 for dizziness_risk, 0.0133 for fatigue_risk, 0.0160 for fainting_risk, 0.0191 for need_to_hydrate, and 0.0192 for need_to_rest.

- The mean test PR-AUC was 0.7118 for flare_risk, 0.4829 for dizziness_risk, 0.5593 for fatigue_risk, 0.2530 for fainting_risk, 0.7441 for need_to_hydrate, and 0.7413 for need_to_rest.

- The corresponding test PR-AUC standard deviations were 0.0270 for flare_risk, 0.0169 for dizziness_risk, 0.0107 for fatigue_risk, 0.0528 for fainting_risk, 0.0302 for need_to_hydrate, and 0.0104 for need_to_rest.

- I then ran Experiment 1B, Repeated Threshold/F1 Robustness.

- Experiment 1B used five repeated participant-level splits with split seeds 42, 43, 44, 45, and 46.

- Each repeat again used 350 training participants, 75 validation participants, and 75 test participants.

- The same Random Forest configuration was used: 300 estimators, max_depth=16, min_samples_leaf=5, random_state=42, n_jobs=-1, and class_weight="balanced".

- For each target, predicted probabilities were generated for the validation and test sets.

- I evaluated classification thresholds from 0.10 through 0.90 in increments of 0.05.

- The threshold producing the highest validation F1 score was selected using the validation data.

- The selected validation threshold was then applied to the held-out test probabilities to calculate test precision, test recall, and test F1.

- The complete Experiment 1B results were saved to results/repeated_threshold_f1_robustness.csv.

- The mean selected threshold was 0.59 for flare_risk, 0.43 for dizziness_risk, 0.41 for fatigue_risk, 0.65 for fainting_risk, 0.51 for need_to_hydrate, and 0.45 for need_to_rest.

- The mean test precision was 0.6244 for flare_risk, 0.4309 for dizziness_risk, 0.4945 for fatigue_risk, 0.2669 for fainting_risk, 0.6276 for need_to_hydrate, and 0.6258 for need_to_rest.

- The mean test recall was 0.7486 for flare_risk, 0.7628 for dizziness_risk, 0.7704 for fatigue_risk, 0.5193 for fainting_risk, 0.8044 for need_to_hydrate, and 0.8247 for need_to_rest.

- The mean test F1 score was 0.6803 for flare_risk, 0.5501 for dizziness_risk, 0.6020 for fatigue_risk, 0.3431 for fainting_risk, 0.7042 for need_to_hydrate, and 0.7116 for need_to_rest.

- I then ran Experiment 1C, Repeated Longitudinal Feature Ablation.

- The purpose of this experiment was to compare the complete feature set with a same-day-only feature set to determine how much predictive performance changed when longitudinal information was removed.

- The full feature set was created after excluding participant_id, day, the current symptom variables, and all target columns.

- The longitudinal features removed for the same-day-only model included columns beginning with previous_, sleep_3day_, sleep_7day_, water_3day_, water_7day_, hrv_3day_, hrv_7day_, symptom_3day_, and symptom_7day_.

- The exact longitudinal features hrv_change, resting_hr_change, symptom_change, sleep_deficit, and hydration_deficit were also removed.

- The resulting full feature set contained 32 features, while the same-day-only feature set contained 6 features.

- Experiment 1C used five participant-level splits with seeds 42, 43, 44, 45, and 46.

- For every target and repeat, I trained one Random Forest using the full feature set and another Random Forest using only the same-day features.

- The Random Forest models used 300 estimators, random_state=42, n_jobs=-1, max_depth=None, min_samples_split=2, and min_samples_leaf=1.

- The full and same-day-only models were evaluated on the held-out test participants using ROC-AUC, PR-AUC, and Brier score.

- The complete Experiment 1C results were saved to results/repeated_longitudinal_ablation.csv.

- Across the five repeats, the mean ROC-AUC improvement from the full longitudinal feature set compared with the same-day-only feature set was +0.2885 for flare_risk, +0.1582 for dizziness_risk, +0.1777 for fatigue_risk, +0.1613 for fainting_risk, +0.1020 for need_to_hydrate, and +0.2101 for need_to_rest.

- The mean PR-AUC improvement was +0.4658 for flare_risk, +0.1433 for dizziness_risk, +0.1593 for fatigue_risk, +0.1858 for fainting_risk, +0.1789 for need_to_hydrate, and +0.2399 for need_to_rest.

- The Brier difference was calculated as the full-model Brier score minus the same-day-only Brier score.

- The mean Brier difference was -0.0664 for flare_risk, -0.0282 for dizziness_risk, -0.0352 for fatigue_risk, -0.0042 for fainting_risk, -0.0432 for need_to_hydrate, and -0.0702 for need_to_rest.

- Because a lower Brier score represents better probabilistic prediction performance, the negative Brier differences indicate that the full feature model had lower Brier scores than the same-day-only model across the repeated experiments.

- I then performed the statistical analysis of the repeated longitudinal ablation results.

- The statistical analysis loaded results/repeated_longitudinal_ablation.csv and performed a one-sample t-test against zero for the repeated ROC-AUC improvements, PR-AUC improvements, and Brier-score differences for each of the six targets.

- For flare_risk, the mean ROC-AUC improvement was +0.288498 with p=0.000028, the mean PR-AUC improvement was +0.465775 with p=0.000003, and the mean Brier difference was -0.066374 with p=0.000097.

- For dizziness_risk, the mean ROC-AUC improvement was +0.158231 with p=0.000002, the mean PR-AUC improvement was +0.143284 with p=0.000017, and the mean Brier difference was -0.028213 with p=0.000002.

- For fatigue_risk, the mean ROC-AUC improvement was +0.177668 with p=0.000007, the mean PR-AUC improvement was +0.159338 with p=0.000038, and the mean Brier difference was -0.035167 with p=0.000004.

- For fainting_risk, the mean ROC-AUC improvement was +0.161298 with p=0.000064, the mean PR-AUC improvement was +0.185775 with p=0.000028, and the mean Brier difference was -0.004180 with p=0.001179.

- For need_to_hydrate, the mean ROC-AUC improvement was +0.101995 with p=0.000006, the mean PR-AUC improvement was +0.178903 with p=0.000083, and the mean Brier difference was -0.043176 with p=0.000009.

- For need_to_rest, the mean ROC-AUC improvement was +0.210122 with p=0.000003, the mean PR-AUC improvement was +0.239864 with p=0.000023, and the mean Brier difference was -0.070232 with p<0.000001.

- The statistical analysis results were saved to results/robustness_statistical_analysis.csv.

- Finally, I created the Day 9 combined robustness summary by loading the three experiment result files and grouping their results by target.

- The summary combined the mean and standard deviation of validation and test ROC-AUC and PR-AUC from Experiment 1A, the mean and standard deviation of thresholds and F1-related metrics from Experiment 1B, and the mean and standard deviation of the ROC-AUC improvement, PR-AUC improvement, and Brier difference from Experiment 1C.

- The final combined summary was saved to results/day9_robustness_summary.csv.

- Day 9 therefore completed three repeated robustness experiments: repeated discrimination robustness, repeated threshold/F1 robustness, and repeated longitudinal feature ablation.

- The results showed that model performance varied across participant-level splits but remained measurable across all six prediction targets.

- The longitudinal ablation consistently showed higher ROC-AUC and PR-AUC for the full feature model than for the same-day-only model across the five repeated splits.

- Day 9 also produced statistical tests examining whether the repeated longitudinal-feature improvements differed from zero.

- These analyses were performed on the PaceMate-AI project dataset and evaluate robustness and feature contribution within that dataset. They do not establish clinical validity or clinical effectiveness.  