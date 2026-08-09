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
