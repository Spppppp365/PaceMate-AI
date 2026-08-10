# PaceMate-AI

PaceMate-AI is a machine learning research project focused on predicting daily symptom-related outcomes using longitudinal health data.

The project was rebuilt from an earlier prototype with an emphasis on reproducibility, participant-level evaluation, feature engineering, model comparison, and transparent documentation of both successful and unsuccessful experiments.

## Project Goal

The goal of PaceMate-AI is to investigate whether machine learning models can identify patterns in longitudinal symptom and lifestyle data that may be useful for daily symptom management.

The current project is a research prototype and is not a clinically validated medical device.

## Research Approach

The project uses synthetic longitudinal data to develop and evaluate a multi-target machine learning pipeline.

The pipeline includes:

- Synthetic participant generation
- Longitudinal daily observation generation
- Next-day target generation
- Historical feature engineering
- Participant-level dataset splitting
- Baseline model development
- Hyperparameter tuning
- Feature engineering experiments
- Probability calibration
- Decision threshold tuning
- Production model generation
- Unseen-participant evaluation
- Performance analysis
- Reproducible experiment documentation

## Dataset

The current synthetic dataset contains:

- 500 synthetic participants
- 180 days of observations per participant
- 90,000 daily observations
- 89,500 generated next-day target rows
- 89,000 final modeling rows
- 36 total dataset columns before the final feature expansion
- 32 model features after feature engineering

The dataset was divided at the participant level to prevent the same participant from appearing in multiple dataset splits.

The final split was:

- 350 training participants
- 75 validation participants
- 75 test participants
- 62,300 training rows
- 13,350 validation rows
- 13,350 test rows

The test participants were completely excluded from model training and model selection.

## Prediction Targets

The project currently predicts six outcomes:

- flare_risk
- dizziness_risk
- fatigue_risk
- fainting_risk
- need_to_hydrate
- need_to_rest

## Feature Engineering

The modeling dataset uses current and historical information to represent recent trends.

Features include:

- Previous-day sleep
- Previous-day water intake
- Previous-day resting heart rate
- Previous-day HRV
- Previous-day symptom severity
- Three-day rolling averages
- Seven-day rolling averages
- HRV change
- Resting heart-rate change
- Symptom change
- Sleep deficit
- Hydration deficit
- Historical variability measures

The feature engineering process was checked for missing values, duplicate participant/day combinations, and future-derived information.

No future observations were intentionally used to construct the prediction features.

## Model Development

Random Forest models were used as the primary modeling approach.

The flare-risk model was developed first as the baseline prediction task.

Multiple experiments were then performed to determine whether the model could be improved.

These experiments included:

- Historical variability feature expansion
- Random Forest hyperparameter tuning
- Decision threshold testing
- Probability calibration
- Secondary-target model tuning
- Production model construction
- Unified multi-target evaluation

## Model Selection

Several Random Forest configurations were compared during flare-risk development.

The main configurations varied:

- Number of trees
- Maximum tree depth
- Minimum samples per leaf
- Class weighting

The best validation F1 score was 0.7170.

The selected configuration used:

- 300 trees
- Maximum depth of 16
- Minimum samples per leaf of 5
- Balanced class weighting

The classification threshold was later selected separately from the model parameters.

## Threshold Selection

Decision thresholds were tested to study the tradeoff between precision and recall.

For the final multi-target models, the selected thresholds were:

- flare_risk: 0.55
- dizziness_risk: 0.25
- fatigue_risk: 0.30
- fainting_risk: 0.20
- need_to_hydrate: 0.35
- need_to_rest: 0.35

Threshold selection was performed separately for each prediction target because the targets have different distributions and different precision-recall tradeoffs.

## Final Production Evaluation

The final production models were evaluated on the 75 test participants that were not used during model training or model selection.

The test set contained 13,350 observations and 32 model features.

Final test performance:

### Flare Risk

- Accuracy: 0.8784
- Precision: 0.7526
- Recall: 0.5075
- F1 Score: 0.6062
- ROC-AUC: 0.9216
- PR-AUC: 0.7221
- Brier Score: 0.084342

### Dizziness Risk

- Accuracy: 0.6032
- Precision: 0.4159
- Recall: 0.7809
- F1 Score: 0.5428
- ROC-AUC: 0.7145
- PR-AUC: 0.4883
- Brier Score: 0.186172

### Fatigue Risk

- Accuracy: 0.6282
- Precision: 0.5021
- Recall: 0.7867
- F1 Score: 0.6130
- ROC-AUC: 0.7158
- PR-AUC: 0.5716
- Brier Score: 0.203489

### Fainting Risk

- Accuracy: 0.9604
- Precision: 0.2815
- Recall: 0.4174
- F1 Score: 0.3363
- ROC-AUC: 0.9516
- PR-AUC: 0.2599
- Brier Score: 0.019522

### Need to Hydrate

- Accuracy: 0.8120
- Precision: 0.6528
- Recall: 0.8103
- F1 Score: 0.7231
- ROC-AUC: 0.8927
- PR-AUC: 0.7582
- Brier Score: 0.122473

### Need to Rest

- Accuracy: 0.7346
- Precision: 0.6142
- Recall: 0.8352
- F1 Score: 0.7078
- ROC-AUC: 0.8369
- PR-AUC: 0.7421
- Brier Score: 0.159831

## Important Findings

The experiments produced several important findings.

First, adding historical variability features did not improve the flare-risk model. The expanded model performed slightly worse than the earlier feature set.

Second, increasing Random Forest depth improved validation performance, but the improvement did not translate into a higher F1 score on completely unseen participants.

Third, changing the classification threshold substantially changed the precision and recall tradeoff without retraining the underlying model.

Fourth, the six prediction targets did not perform equally well.

The hydration and rest models produced some of the strongest overall secondary-target results.

The flare-risk model showed strong discrimination based on ROC-AUC and PR-AUC.

The fainting model produced a very high ROC-AUC but was affected by severe class imbalance and had substantially weaker precision, recall, and F1 at the selected threshold.

The dizziness and fatigue models showed weaker discrimination than the flare, hydration, and rest models.

These differences demonstrate why a multi-target system should not assume that every prediction target has the same level of reliability.

## Reproducibility

The project is organized so that the major stages of the machine learning pipeline can be reproduced from the source code.

The repository contains scripts for:

- Generating participants
- Generating longitudinal observations
- Generating prediction targets
- Building training datasets
- Splitting participants into train, validation, and test sets
- Training models
- Calibrating models
- Tuning thresholds
- Evaluating models
- Analyzing model performance
- Building production models
- Verifying production models

The project also records model evaluation results and development decisions in the project journal.

Random seeds are used where appropriate during synthetic data generation and model development so that experiments can be repeated.

## Research Transparency

Not every experiment improved the model.

Experiments that produced worse results were retained and documented rather than removed.

This includes the historical variability feature experiment, which produced slightly worse performance.

The project also evaluates models on participants that were completely excluded from training and model selection.

This is intended to provide a more realistic measurement of generalization within the synthetic research environment.

## Limitations

The dataset is synthetic.

The prediction targets are generated using predefined rules rather than real patient outcomes.

As a result, the current model performance cannot be interpreted as evidence that PaceMate-AI can accurately predict symptoms in real patients.

The relationships learned by the models may also be easier to detect in synthetic data than in real-world clinical data.

The project therefore demonstrates a machine learning research and development pipeline rather than clinical effectiveness.

Real-world validation using appropriately collected clinical data would be required before making clinical performance claims.

## Future Work

Future development will focus on improving the scientific quality and robustness of the project.

Potential areas include:

- Additional model architectures
- More rigorous calibration analysis
- More extensive threshold analysis
- Cross-validation strategies
- Additional robustness testing
- Feature ablation studies
- Model interpretability
- External validation
- Evaluation on real-world data when appropriate data becomes available
- Improved documentation of the complete experimental pipeline

## Project Status

PaceMate-AI currently has a complete synthetic multi-target machine learning pipeline with participant-level dataset separation, feature engineering, model training, calibration, threshold tuning, production model generation, and unseen-participant evaluation.

The project is currently a research prototype and is not clinically validated.
