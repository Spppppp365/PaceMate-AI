PaceMate-AI Experimental Methodology

1. Research Design

PaceMate-AI was developed as a reproducible machine-learning research pipeline for predicting multiple next-day POTS-related outcomes from longitudinal synthetic participant data.

The main experimental question was whether information from a participant's recent history could improve prediction compared with using same-day information alone.

The experimental pipeline consisted of:

- Synthetic participant generation

- Longitudinal daily observation generation

- Next-day target generation

- Dataset validation

- Historical feature engineering

- Participant-level dataset splitting

- Random Forest model training

- Historical variability feature experimentation

- Hyperparameter comparison

- Longitudinal-versus-same-day ablation experiments

- Validation-based threshold selection

- Repeated participant-level robustness experiments

- Calibration analysis

- Error analysis

- Final held-out test evaluation

Because the dataset and prediction targets were synthetic, the experiments were designed to evaluate machine-learning behavior and experimental methodology rather than clinical effectiveness.

2. Synthetic Participant Generation

The project began by generating 500 synthetic participant profiles.

The participant profiles contained baseline characteristics used to create longitudinal observations. These included variables such as:

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

The participant-generation process used a fixed random seed so that the synthetic participant population could be reproduced.

No real patient records were used to construct the experimental dataset.

3. Longitudinal Observation Generation

Each synthetic participant was assigned 180 days of simulated observations.

This produced:

- 500 participants

- 180 days per participant

- 90,000 total daily observations

The daily observation data included:

- participant_id

- day

- sleep_hours

- water_intake_ml

- resting_hr

- hrv

- dizziness

- fatigue

- brain_fog

- symptom_severity

- activity_level

- stress_level

The longitudinal structure was important because the project was specifically designed to test whether recent participant history provided predictive information beyond measurements from a single day.

4. Prediction Target Generation

Six binary prediction targets were generated:

- flare_risk

- dizziness_risk

- fatigue_risk

- fainting_risk

- need_to_hydrate

- need_to_rest

The targets represented next-day prediction outcomes.

Because each target required information from the following day, the target-generation process produced 89,500 target rows from the original 90,000 daily observations.

This reduction occurred because the final observation for each participant does not have a subsequent day available for generating a next-day target.

The targets were generated synthetically using predefined rules rather than being derived from real patient outcomes.

5. Dataset Validation

Before model training, the generated dataset was checked for structural and data-quality problems.

Validation included checks for:

- Missing values

- Duplicate participant/day combinations

- Participant counts

- Expected observation counts

- Future-derived features

The final modeling dataset contained 89,000 rows.

No missing values or duplicate participant/day combinations were identified, and no future-derived feature columns were detected in the validated dataset.

The resulting dataset therefore contained only information available at or before the prediction point for the longitudinal modeling experiments.

6. Longitudinal Feature Engineering

Historical features were created to represent information from a participant's recent trajectory rather than relying only on the current day's measurements.

The longitudinal feature set included previous-day measurements, rolling averages, changes over time, and deficit-based features.

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

Additional historical variability features were later tested:

- hrv_3day_std

- hrv_7day_std

- symptom_3day_std

- symptom_7day_std

- sleep_3day_std

- sleep_7day_std

- water_3day_std

- water_7day_std

These additional features were intended to capture recent variability rather than only recent averages.

The expanded longitudinal dataset contained 32 model features.

7. Same-Day Baseline

To determine whether longitudinal information actually contributed predictive value, a same-day comparison model was created using six same-day features.

The same-day model did not use the historical feature set.

The longitudinal model therefore used 32 features, while the same-day comparison used 6 features.

The primary comparison metrics were:

- ROC-AUC

- PR-AUC

- Brier score

The difference between the two feature sets allowed the experiment to directly test whether historical information improved prediction.

8. Participant-Level Dataset Splitting

The dataset was divided at the participant level rather than by randomly assigning individual rows.

The split consisted of:

- 350 training participants

- 75 validation participants

- 75 test participants

The corresponding modeling row counts were:

- Training: 62,300 rows

- Validation: 13,350 rows

- Test: 13,350 rows

There was no participant overlap between the three partitions.

Participant-level splitting was used to prevent observations from the same participant from appearing in both training and evaluation data.

This was particularly important because the dataset contained longitudinal information. A random row-level split could otherwise allow a model to learn participant-specific patterns from training rows and encounter the same participant again during evaluation.

The test participants were therefore completely excluded from model training.

9. Model Training

Random Forest classifiers were used as the primary machine-learning models.

The initial model development focused on flare_risk before the pipeline was expanded to the full set of six prediction targets.

The models used balanced class weighting to account for differences in positive and negative class frequencies.

The initial baseline Random Forest configuration used:

- 300 trees

- Maximum depth of 12

- Minimum samples per leaf of 5

- Balanced class weighting

The baseline model established an initial performance reference before additional feature and hyperparameter experiments were performed.

10. Historical Variability Experiment

Eight additional historical variability features were tested to determine whether short-term variability added predictive information beyond historical averages and changes.

The additional features were:

- hrv_3day_std

- hrv_7day_std

- symptom_3day_std

- symptom_7day_std

- sleep_3day_std

- sleep_7day_std

- water_3day_std

- water_7day_std

The expanded feature model was compared with the earlier baseline.

For flare_risk, the additional variability features did not improve performance.

The baseline model had:

- ROC-AUC: 0.9413

- PR-AUC: 0.7897

- F1: 0.7078

The expanded-feature model had:

- ROC-AUC: 0.9409

- PR-AUC: 0.7886

- F1: 0.7058

On the unseen test participants, the baseline model had:

- ROC-AUC: 0.9213

- PR-AUC: 0.7228

- F1: 0.6609

The expanded-feature model had:

- ROC-AUC: 0.9204

- PR-AUC: 0.7208

- F1: 0.6593

The additional variability features were therefore retained as a documented negative experimental result rather than being treated as an improvement.

11. Random Forest Hyperparameter Experiment

Six Random Forest configurations were compared during model development.

The configurations varied:

- Number of trees

- Maximum tree depth

- Minimum samples per leaf

The tested configurations included:

- Baseline

- 300 trees

- Depth 12

- Minimum leaf size 5

- Deeper

- 300 trees

- Depth 16

- Minimum leaf size 5

- More trees

- 500 trees

- Depth 12

- Minimum leaf size 5

- Smaller leaf

- 300 trees

- Depth 12

- Minimum leaf size 3

- Larger leaf

- 300 trees

- Depth 12

- Minimum leaf size 8

- Deeper and smaller leaf

- 300 trees

- Depth 16

- Minimum leaf size 3

All configurations used balanced class weighting.

The deeper configuration improved validation F1 compared with the baseline, but the improvement did not translate into a corresponding improvement on completely unseen participants.

This experiment reinforced the importance of evaluating model changes on participants that were not used during model development.

12. Longitudinal Ablation Experiment

The primary longitudinal experiment compared:

- A 32-feature longitudinal model

- A 6-feature same-day model

The comparison was repeated across five participant-level experimental splits.

The random seeds were:

- 42

- 43

- 44

- 45

- 46

For each repeat, participants were divided into:

- 350 training participants

- 75 validation participants

- 75 test participants

The same comparison procedure was applied to all six prediction targets.

The longitudinal model was evaluated against the same-day model using:

- ROC-AUC improvement

- PR-AUC improvement

- Brier score difference

This design tested whether the benefit of longitudinal information persisted across different unseen participant samples rather than appearing only in one split.

13. Repeated Robustness Experiments

Five repeated participant-level experiments were used to assess the stability of the longitudinal improvement.

For every target and every repeat:

- Longitudinal ROC-AUC was higher than same-day ROC-AUC.

- Longitudinal PR-AUC was higher than same-day PR-AUC.

- Longitudinal Brier score was lower than same-day Brier score.

The magnitude of these improvements varied between participant splits.

Mean and standard deviation were calculated across the five repeats for each target and each metric.

This allowed the analysis to distinguish consistent directional improvement from a result that might occur because of a single favorable participant split.

14. Statistical Analysis

The repeated longitudinal-versus-same-day differences were analyzed using paired statistical tests across the five participant-level experiments.

Separate tests were performed for:

- ROC-AUC improvement

- PR-AUC improvement

- Brier score difference

The resulting t-statistics and p-values were recorded for all six prediction targets.

The tests were used to evaluate whether the observed repeated differences were consistently different from zero across the five experimental repeats.

Because only five repeats were used, the statistical results were treated as supporting evidence for consistency rather than as a substitute for a much larger repeated-validation study.

15. Threshold Selection

Classification thresholds were selected using validation data rather than being optimized directly on the final test set.

This was done because the default probability threshold of 0.50 does not necessarily provide the desired precision/recall balance for every target.

For each repeated participant-level experiment:

- The model was trained using the training participants.

- Predictions were generated for the validation participants.

- A decision threshold was selected using validation performance.

- The selected threshold was then applied to the previously unseen test participants.

- Test precision, recall, and F1 were recorded.

The test set was not used to re-tune the threshold.

Threshold selection was repeated independently for each prediction target.

16. Threshold Robustness

Threshold selection was repeated across five participant-level splits.

The selected thresholds varied across targets and participant splits.

The repeated test-set results were summarized using:

- Precision

- Recall

- F1

This analysis evaluated whether threshold-based classification performance remained reasonably consistent when the participant sample changed.

Threshold variation was particularly noticeable for fainting_risk, while need_to_rest used the same selected threshold across all five repeated splits.

The threshold results were therefore interpreted as experiment-specific operating points rather than universal decision thresholds.

17. Calibration Analysis

Probability quality was evaluated separately from binary classification performance.

The primary calibration-related metric was the Brier score.

Probability-alignment analysis was also performed to compare predicted probabilities with observed outcomes.

A lower Brier score indicates lower squared probability error.

The longitudinal-versus-same-day experiment therefore used Brier score differences to determine whether historical information improved not only ranking performance but also the quality of predicted probabilities.

The longitudinal model produced a lower Brier score than the same-day model for every target in every repeated experiment.

18. Error Analysis

The project included dedicated analyses of prediction errors.

The error-analysis outputs examined:

- False positives

- False negatives

- Prediction probability distributions

- Feature differences between outcome/error groups

- Longitudinal feature patterns associated with prediction errors

Confusion matrices were also generated for the final test-set predictions.

These analyses were used to identify where models produced incorrect positive predictions, where positive outcomes were missed, and how prediction behavior differed between targets.

19. Final Held-Out Test Evaluation

After model development and threshold selection, the final models were evaluated on the held-out test participants.

The final test set contained:

- 75 previously unseen participants

- 13,350 modeling observations

The final evaluation reported:

- Accuracy

- Precision

- Recall

- F1

- ROC-AUC

- PR-AUC

- Brier score

- Confusion-matrix counts

The final thresholds were:

- flare_risk: 0.55

- dizziness_risk: 0.25

- fatigue_risk: 0.30

- fainting_risk: 0.20

- need_to_hydrate: 0.35

- need_to_rest: 0.35

These thresholds were determined separately from the final held-out test evaluation and were not selected by optimizing directly on the final test results.

20. Evaluation Metrics

Accuracy

Accuracy measured the proportion of all test observations that were classified correctly.

Precision

Precision measured the proportion of predicted positive observations that were actually positive.

Recall

Recall measured the proportion of actual positive observations that were correctly identified.

F1 Score

F1 combined precision and recall into a single metric using their harmonic mean.

ROC-AUC

ROC-AUC measured the model's ability to rank positive observations above negative observations across classification thresholds.

PR-AUC

PR-AUC measured the relationship between precision and recall across classification thresholds and was especially important for targets with class imbalance.

Brier Score

Brier score measured the squared error between predicted probabilities and binary outcomes.

Lower Brier scores indicate better probability accuracy.

21. Class Imbalance

Class distributions differed substantially between the six prediction targets.

Fainting_risk was particularly imbalanced, with an actual positive rate of approximately 2.4% in the final test set.

This imbalance was considered when interpreting model performance.

In particular, the final fainting-risk model produced a very high ROC-AUC while having substantially weaker positive-class precision, recall, F1, and PR-AUC.

This demonstrated why ROC-AUC was not used as the sole evaluation metric.

22. Reproducibility

The project was organized around reproducible data generation and model experimentation.

The experimental workflow was separated into stages for:

- Participant generation

- Daily observation generation

- Target generation

- Feature engineering

- Dataset splitting

- Model training

- Evaluation

- Robustness analysis

- Threshold analysis

- Error analysis

Synthetic participant generation used a fixed random seed, and the repeated robustness experiments explicitly recorded the seeds used for each participant-level repeat.

The project results and analysis outputs were stored in structured CSV and image files so that the numerical results could be inspected independently of the narrative documentation.

23. Methodological Limitations

The primary methodological limitation is that the entire dataset is synthetic.

The participant characteristics, daily observations, and prediction targets were generated computationally rather than collected from real patients.

The target-generation rules may therefore create relationships that are easier for machine-learning models to learn than relationships that would exist in real-world clinical data.

A second limitation is the relatively small number of repeated participant-level experiments used for the robustness analysis. Five repeats provide evidence that the observed direction of improvement was consistent across the tested splits, but they do not represent an exhaustive assessment of every possible participant split.

A third limitation is that threshold selection varied between participant splits. The selected thresholds should therefore be interpreted as experimental operating points rather than validated clinical decision thresholds.

A fourth limitation is the severe class imbalance for fainting_risk, which makes positive-class metrics and PR-AUC particularly important when interpreting that target.

Finally, the held-out participants were unseen synthetic participants rather than real patients. Generalization to unseen synthetic participants therefore does not establish generalization to real-world patients.

24. Methodological Summary

The final PaceMate-AI experiment was designed to test whether longitudinal information improved multi-target prediction while reducing the risk of participant-level information leakage.

The central comparison used 32 longitudinal features versus 6 same-day features and was repeated across five participant-level experimental splits.

The analysis combined discrimination metrics, probability-quality metrics, threshold-based classification metrics, repeated participant-level evaluation, statistical testing, calibration analysis, and error analysis.

This experimental structure allowed the project to evaluate not only whether the models performed well on one test set, but also whether the observed benefit of longitudinal information remained consistent when the unseen participants changed.

The resulting evidence supports the conclusion that, within the synthetic dataset and experimental framework, longitudinal information provided additional predictive information beyond same-day measurements.

The methodology does not establish clinical effectiveness or real-world patient performance.

PaceMate-AI Final Research Results Inventory

Dataset

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

Prediction Targets

The model predicts six binary outcomes:

- flare_risk
- dizziness_risk
- fatigue_risk
- fainting_risk
- need_to_hydrate
- need_to_rest

Primary Experimental Questions

Question 1

Does longitudinal information improve prediction compared with same-day information?

The final longitudinal models used 32 features, while the same-day comparison models used 6 features.

Primary metrics:

- ROC-AUC
- PR-AUC
- Brier score

Longitudinal models were compared with same-day models across five participant-level experimental repeats using seeds 42 through 46.

Question 2

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

Question 3

Can decision thresholds improve the usefulness of predictions?

Thresholds were selected using validation data and then evaluated on unseen test participants.

The threshold was not re-tuned on the test set.

Primary evidence:

- Validation-selected thresholds
- Test precision
- Test recall
- Test F1
- Repeated threshold/F1 experiments across five participant-level splits

Question 4

Are model probabilities reasonably calibrated?

Calibration was evaluated using:

- Brier score
- Predicted probability versus observed outcome comparisons
- Probability-alignment analysis
- Calibration analysis for the prediction targets

Question 5

Where does the model fail?

Error analysis examined:

- False positives
- False negatives
- Prediction probability distributions
- Feature differences between outcome groups
- Longitudinal feature patterns associated with errors

Final Model Results

The final test-set evaluation was performed on 75 previously unseen participants.

Target | Threshold | Positive Rate | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier
flare_risk | 0.55 | 0.1845 | 0.8784 | 0.7526 | 0.5075 | 0.6062 | 0.9216 | 0.7221 | 0.0843
dizziness_risk | 0.25 | 0.3016 | 0.6032 | 0.4159 | 0.7809 | 0.5428 | 0.7145 | 0.4883 | 0.1862
fatigue_risk | 0.30 | 0.3743 | 0.5864 | 0.5021 | 0.7867 | 0.6130 | 0.7158 | 0.5716 | 0.2035
fainting_risk | 0.20 | 0.0240 | 0.9604 | 0.2815 | 0.4174 | 0.3363 | 0.9516 | 0.2599 | 0.0195
need_to_hydrate | 0.35 | 0.3029 | 0.8120 | 0.6528 | 0.8103 | 0.7231 | 0.8927 | 0.7582 | 0.1225
need_to_rest | 0.35 | 0.3849 | 0.7346 | 0.6142 | 0.8352 | 0.7078 | 0.8369 | 0.7421 | 0.1598

Final Test-Set Confusion-Matrix Results

Target | True Negatives | False Positives | False Negatives | True Positives
flare_risk | 10,476 | 411 | 1,213 | 1,250
dizziness_risk | 4,909 | 4,415 | 882 | 3,144
fatigue_risk | 4,455 | 3,898 | 1,066 | 3,931
fainting_risk | 12,687 | 342 | 187 | 134
need_to_hydrate | 7,563 | 1,743 | 767 | 3,277
need_to_rest | 5,515 | 2,696 | 847 | 4,292

Final Test-Set Probability Results

Target | Actual Positive Rate | Mean Predicted Probability | Predicted Positive Rate
flare_risk | 0.1845 | 0.1788 | 0.1244
dizziness_risk | 0.3016 | 0.2971 | 0.5662
fatigue_risk | 0.3743 | 0.3684 | 0.5864
fainting_risk | 0.0240 | 0.0213 | 0.0357
need_to_hydrate | 0.3029 | 0.2959 | 0.3760
need_to_rest | 0.3849 | 0.3791 | 0.5234

Longitudinal Feature Ablation

Five repeated participant-level experiments compared the 32-feature longitudinal models with 6-feature same-day models.

For every target and every repeat:

- Longitudinal features increased ROC-AUC.
- Longitudinal features increased PR-AUC.
- Longitudinal features decreased Brier score.

Mean Longitudinal Improvement

Target | ROC-AUC Improvement Mean | ROC-AUC Improvement SD | PR-AUC Improvement Mean | PR-AUC Improvement SD | Brier Difference Mean | Brier Difference SD
flare_risk | +0.2885 | 0.0301 | +0.4658 | 0.0283 | -0.0664 | 0.0095
dizziness_risk | +0.1582 | 0.0088 | +0.1433 | 0.0132 | -0.0282 | 0.0015
fatigue_risk | +0.1777 | 0.0132 | +0.1593 | 0.0179 | -0.0352 | 0.0022
fainting_risk | +0.1613 | 0.0208 | +0.1858 | 0.0194 | -0.0042 | 0.0011
need_to_hydrate | +0.1020 | 0.0072 | +0.1789 | 0.0246 | -0.0432 | 0.0034
need_to_rest | +0.2101 | 0.0126 | +0.2399 | 0.0239 | -0.0702 | 0.0019

Negative Brier differences indicate lower Brier scores for the longitudinal models.

Statistical Analysis of Longitudinal Improvements

Paired statistical tests were performed across the five repeated participant-level experiments.

ROC-AUC Improvement

Target | Mean Improvement | t-statistic | p-value
flare_risk | +0.2885 | 21.4255 | 0.000028
dizziness_risk | +0.1582 | 40.0523 | 0.000002
fatigue_risk | +0.1777 | 30.0190 | 0.000007
fainting_risk | +0.1613 | 17.3689 | 0.000064
need_to_hydrate | +0.1020 | 31.8867 | 0.000006
need_to_rest | +0.2101 | 37.1953 | 0.000003

PR-AUC Improvement

Target | Mean Improvement | t-statistic | p-value
flare_risk | +0.4658 | 36.8359 | 0.000003
dizziness_risk | +0.1433 | 24.2543 | 0.000017
fatigue_risk | +0.1593 | 19.8766 | 0.000038
fainting_risk | +0.1858 | 21.3966 | 0.000028
need_to_hydrate | +0.1789 | 16.2859 | 0.000083
need_to_rest | +0.2399 | 22.4779 | 0.000023

Brier Score Difference

Target | Mean Difference | t-statistic | p-value
flare_risk | -0.0664 | -15.6578 | 0.000097
dizziness_risk | -0.0282 | -40.9004 | 0.000002
fatigue_risk | -0.0352 | -35.1620 | 0.000004
fainting_risk | -0.0042 | -8.2472 | 0.001179
need_to_hydrate | -0.0432 | -28.7275 | 0.000009
need_to_rest | -0.0702 | -84.1451 | 0.00000012

These repeated experiments provide evidence that the observed longitudinal improvements were consistent across the five participant-level splits.

Threshold Robustness

Decision thresholds were selected independently using validation data in each repeated participant-level experiment.

The selected thresholds varied across splits, demonstrating that threshold choice was not identical across participant samples.

Repeated Test-Set Threshold Results

Target | Mean Test Precision | Mean Test Recall | Mean Test F1
flare_risk | 0.6244 | 0.7486 | 0.6803
dizziness_risk | 0.4309 | 0.7628 | 0.5501
fatigue_risk | 0.4945 | 0.7704 | 0.6020
fainting_risk | 0.2669 | 0.5193 | 0.3431
need_to_hydrate | 0.6276 | 0.8044 | 0.7042
need_to_rest | 0.6258 | 0.8247 | 0.7116

These repeated results show that threshold-selected classification performance varied across participant splits rather than remaining identical across experiments.

Threshold Selection by Repeated Split

Validation-selected thresholds varied as follows:

Target | Thresholds Selected Across Five Repeats
flare_risk | 0.55, 0.60, 0.60, 0.65, 0.55
dizziness_risk | 0.45, 0.40, 0.45, 0.45, 0.40
fatigue_risk | 0.40, 0.45, 0.40, 0.40, 0.40
fainting_risk | 0.75, 0.55, 0.70, 0.65, 0.60
need_to_hydrate | 0.55, 0.45, 0.55, 0.50, 0.50
need_to_rest | 0.45, 0.45, 0.45, 0.45, 0.45

The final thresholds used for the final held-out test evaluation were determined separately and are reported in the Final Model Results table above.

Robustness Findings

Repeated participant-level experiments demonstrated that:

- Longitudinal features improved ROC-AUC for all six prediction targets in every repeated experiment.
- Longitudinal features improved PR-AUC for all six prediction targets in every repeated experiment.
- Longitudinal features reduced Brier score for all six prediction targets in every repeated experiment.
- The magnitude of improvement varied across participant-level splits.
- Threshold-selected precision, recall, and F1 varied across participant-level splits.
- Threshold selection was relatively stable for some targets and more variable for others, particularly fainting_risk.

Important Negative Findings

- Historical variability features did not improve flare-risk performance.
- Validation improvements did not always translate into equivalent test-set F1 improvements.
- Prediction quality differed substantially between targets.
- Fainting-risk had very strong ROC-AUC (0.9516) but substantially weaker PR-AUC (0.2599), precision (0.2815), recall (0.4174), and F1 (0.3363).
- The weak positive-class metrics for fainting_risk occurred in the context of a highly imbalanced target, with an actual positive rate of only 2.40%.
- The final test-set threshold for fainting_risk was 0.20, producing high overall accuracy (0.9604) but relatively limited positive-class precision, recall, and F1.
- Threshold-selected predictions for dizziness_risk, fatigue_risk, and need_to_rest produced relatively high recall but also substantially more false positives than the flare_risk model.

Target-Level Interpretation

flare_risk

The final flare-risk model showed strong discrimination with ROC-AUC of 0.9216 and PR-AUC of 0.7221.

At the final threshold of 0.55:

- Precision was 0.7526.
- Recall was 0.5075.
- F1 was 0.6062.
- Accuracy was 0.8784.
- Brier score was 0.0843.

Longitudinal information produced a mean ROC-AUC improvement of 0.2885 and mean PR-AUC increase of 0.4658 relative to same-day features.

dizziness_risk

The final dizziness-risk model had ROC-AUC of 0.7145 and PR-AUC of 0.4883.

At the final threshold of 0.25:

- Precision was 0.4159.
- Recall was 0.7809.
- F1 was 0.5428.
- Accuracy was 0.6032.
- Brier score was 0.1862.

The model favored recall at the selected threshold, resulting in 3,144 true positives and 4,415 false positives on the test set.

fatigue_risk

The final fatigue-risk model had ROC-AUC of 0.7158 and PR-AUC of 0.5716.

At the final threshold of 0.30:

- Precision was 0.5021.
- Recall was 0.7867.
- F1 was 0.6130.
- Accuracy was 0.5864.
- Brier score was 0.2035.

The model again favored recall at the selected threshold, with 3,931 true positives and 3,898 false positives.

fainting_risk

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

need_to_hydrate

The final hydration model had ROC-AUC of 0.8927 and PR-AUC of 0.7582.

At the final threshold of 0.35:

- Precision was 0.6528.
- Recall was 0.8103.
- F1 was 0.7231.
- Accuracy was 0.8120.
- Brier score was 0.1225.

This was one of the strongest overall target results, combining high discrimination with relatively strong precision, recall, and F1.

need_to_rest

The final rest model had ROC-AUC of 0.8369 and PR-AUC of 0.7421.

At the final threshold of 0.35:

- Precision was 0.6142.
- Recall was 0.8352.
- F1 was 0.7078.
- Accuracy was 0.7346.
- Brier score was 0.1598.

Longitudinal information produced a mean ROC-AUC improvement of 0.2101 and a mean PR-AUC improvement of 0.2399 relative to same-day features.

Calibration

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

Error Analysis

The project included dedicated analyses of model errors.

Generated error-analysis outputs included:

- False-positive and false-negative analysis
- Prediction probability distributions
- Feature comparisons between error/outcome groups
- Longitudinal feature patterns associated with prediction errors

These analyses were used to identify where the models performed well and where prediction quality was weaker.

Limitations

Synthetic Data

The dataset and prediction targets are synthetic.

Therefore:

- Model performance cannot be interpreted as clinical effectiveness.
- The results cannot establish clinical utility.
- The results cannot establish that the model would perform similarly on real patients.
- External validation using real-world data would be required.
- Synthetic relationships may be easier for a machine-learning model to learn than relationships in real-world clinical data.

Participant-Level Generalization

Although participant-level splitting was used to prevent the same participant from appearing in multiple partitions, the participants themselves were synthetically generated.

Therefore, successful generalization to unseen synthetic participants does not establish generalization to unseen real patients.

Class Imbalance

Fainting-risk was strongly imbalanced, with an actual positive rate of only 2.40% in the final test set.

This caused ROC-AUC to appear substantially stronger than positive-class precision, recall, F1, and PR-AUC.

Threshold Stability

Validation-selected thresholds varied across repeated participant-level splits.

Therefore, the exact threshold values should not be interpreted as universally optimal clinical decision thresholds.

Clinical Interpretation

PaceMate-AI is a research prototype and not a clinically validated diagnostic or treatment system.

The model results demonstrate predictive behavior within the synthetic experimental environment but do not establish medical safety, diagnostic accuracy, or clinical usefulness.

Figures

The final analysis produced the following figures:

- model_discrimination_comparison.png
- model_classification_metrics.png
- model_brier_score_comparison.png
- model_probability_alignment.png
- prediction_outcomes.png
- error_probability_distribution.png
- error_group_feature_comparison.png
- longitudinal_error_feature_patterns.png

Final Research Conclusion

Across five repeated participant-level experiments, longitudinal information consistently improved discrimination and probability quality for all six prediction targets compared with same-day information alone.

The mean improvements in ROC-AUC, PR-AUC, and Brier score were positive in the expected directions for every target, and the corresponding repeated-split statistical tests produced small p-values for all three measures across all six targets.

The final held-out test results demonstrated that performance varied substantially by prediction target. The strongest overall discrimination was observed for fainting-risk and flare-risk, while hydration and rest produced the strongest combinations of discrimination and threshold-based classification performance. Fainting-risk remained substantially more difficult to classify reliably at the positive-class level because of severe class imbalance.

These findings support the research hypothesis that longitudinal information can provide additional predictive information beyond same-day measurements within this synthetic PaceMate-AI dataset. However, because the data and targets are synthetic, the results should be interpreted as evidence from a controlled machine-learning experiment rather than evidence of clinical effectiveness.

Real-world data, external validation, prospective evaluation, and clinical safety assessment would be required before drawing conclusions about clinical utility.