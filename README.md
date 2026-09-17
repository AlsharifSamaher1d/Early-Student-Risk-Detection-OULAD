# Early Student Risk Detection Using OULAD

## An Explainable Machine Learning Approach for Early Academic Intervention

This project develops an **early-warning machine learning system** for identifying students who may be at risk of failing or withdrawing from an online course. Using the **Open University Learning Analytics Dataset (OULAD)**, the analysis combines student information, registration data, early Virtual Learning Environment (VLE) engagement, and assessment behavior to estimate student risk during the first weeks of a course.

The main objective is not only to predict risk accurately, but to make predictions **early, explainable, and actionable** so that academic advisors can provide timely support.

---

## Project Overview

Student disengagement and academic difficulty often become visible before a final course outcome is known. Learning analytics can help detect these warning signals early, but an effective intervention system must answer several questions:

- Can student risk be identified within the first few weeks of a course?
- Which early behavioral and academic signals are most informative?
- How much predictive performance is gained by waiting for additional weeks of data?
- Which machine learning model provides the strongest early-risk predictions?
- How should the classification threshold be selected when missing an at-risk student is more costly than generating an additional alert?
- Can individual predictions be explained clearly enough to support academic intervention?

This case study addresses these questions using a leakage-aware, group-aware modeling pipeline with **Logistic Regression, Random Forest, XGBoost, threshold optimization, and SHAP explainability**.

---

## Dataset

**Dataset:** Open University Learning Analytics Dataset (OULAD)

**Kaggle source:** https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad

The dataset contains information from an online-learning environment across several related tables rather than a single flat file.

### Dataset Scale

- **32,593** student-course records
- **22** course presentations
- More than **10.6 million** VLE interaction records
- **173,912** assessment submissions

### Main Tables Used

| File | Purpose |
|---|---|
| `studentInfo.csv` | Student demographics, prior education, studied credits, disability information, and final course result |
| `studentRegistration.csv` | Registration and unregistration information |
| `studentVle.csv` | Student interactions and clicks within the Virtual Learning Environment |
| `vle.csv` | Metadata describing VLE learning activities |
| `studentAssessment.csv` | Student assessment submissions and scores |
| `assessments.csv` | Assessment dates, types, and weights |
| `courses.csv` | Course presentation information |

The main student-course key is formed from:

`code_module + code_presentation + id_student`

---

## Target Definition

The original `final_result` variable is converted into a binary early-risk target:

| Target | Original Outcome | Meaning |
|---|---|---|
| **0 — Not At Risk** | Pass or Distinction | Student successfully completes the course |
| **1 — At Risk** | Fail or Withdrawn | Student fails or withdraws from the course |

### Target Distribution

- **At Risk:** 17,208 records (**52.8%**)
- **Not At Risk:** 15,385 records (**47.2%**)

The target is reasonably balanced. However, because the purpose of the system is intervention, evaluation focuses particularly on **At-Risk Recall, F1, PR-AUC, and ROC-AUC**, rather than Accuracy alone.

---

## Early-Warning Design

A central requirement of this project is that the model must only use information that would realistically be available at the selected intervention point.

Four early checkpoints are evaluated:

| Checkpoint | Cutoff Day |
|---|---:|
| Week 2 | 14 |
| Week 4 | 28 |
| Week 6 | 42 |
| Week 8 | 56 |

The main early-warning point is **Week 4 (Day 28)**. This checkpoint provides substantially more predictive information than Week 2 while still leaving enough time for academic intervention.

---

## Data Quality and Cleaning

Several data-quality issues are handled before modeling.

### Missing Values

`imd_band` contains **1,111 missing records (3.41%)** in `studentInfo`. Instead of dropping these students, missing values are retained as an **Unknown** category.

Missing assessment information is also handled carefully because the absence of an early submission can itself be an important risk signal. Missing-value indicators are therefore used where appropriate rather than treating every missing value as meaningless noise.

### Duplicate VLE Records

`studentVle` contains **787,170 exact duplicate rows**. These duplicates are removed before aggregating click-based features so that student engagement is not artificially inflated.

---

## Feature Engineering

Raw transactional learning-platform data is transformed into one student-course snapshot for each early checkpoint.

### 1. Engagement Features

Early VLE activity is summarized using features such as:

- `total_clicks`
- `active_study_days`
- `materials_touched`
- `last_activity_day`
- `avg_clicks_per_active_day`
- `days_since_last_activity`

These features describe how frequently and how recently a student interacts with the learning environment.

### 2. Inactivity Features

Engagement volume alone may not reveal periods of disengagement. The project therefore measures gaps between active study days using:

- `max_inactivity_gap`
- `avg_inactivity_gap`

These features help identify students whose study activity becomes irregular or stops for extended periods.

### 3. Engagement Trend

The pipeline includes early activity-trend features to capture whether engagement is increasing or decreasing as the course progresses.

### 4. Activity-Type Features

VLE interactions are also separated by common activity types. This provides more information than total clicks alone by showing how students interact with different learning resources.

### 5. Assessment and Submission Features

Early academic behavior is represented using features including:

- `avg_score`
- `min_score`
- `score_std`
- `assessments_completed`
- `avg_submission_day`
- `late_submission_rate`
- `assessment_weight_completed`
- `weighted_score_earned`

Assessment participation and assessment performance are treated as related but distinct signals. A student who has not submitted an early assessment may carry a different type of risk from a student who submitted but received a low score.

### 6. Student Information

The modeling snapshot also includes relevant student and course information such as prior education and course context. Demographic information is interpreted carefully and is not treated as a causal explanation of student outcomes.

---

## Exploratory Analysis

The exploratory analysis examines risk patterns across:

- Student demographics
- Prior education
- Socioeconomic groups
- VLE engagement
- Active study days
- Learning materials accessed
- Assessment participation
- Early assessment performance
- Submission timing
- Engagement trends

The analysis shows that **At-Risk students generally demonstrate weaker early engagement and weaker assessment participation/performance**. These patterns support combining behavioral and academic information rather than relying on a single feature.

---

## How Early Can Risk Be Detected?

A Logistic Regression pipeline is evaluated at Weeks 2, 4, 6, and 8 to measure how predictive performance changes as additional course information becomes available.

| Checkpoint | PR-AUC |
|---|---:|
| Week 2 | **0.813** |
| Week 4 | **0.861** |
| Week 6 | **0.874** |
| Week 8 | **0.878** |

Performance improves as more information becomes available. However, waiting longer also reduces the amount of time available for intervention.

**Week 4 is therefore selected as the main practical early-warning checkpoint:** it provides a meaningful improvement over Week 2 while preserving enough time for advisors to act.

---

## Leakage-Aware Train/Test Design

Some students appear in more than one course record. A standard random row-level split could therefore place records from the same student in both training and testing, producing an overly optimistic estimate of generalization.

To avoid this, the project uses **GroupShuffleSplit** with `id_student` as the grouping variable.

### Split

- **Training:** 26,122 records
- **Test:** 6,471 records

Each student appears entirely in either the training set or the test set.

This provides a more realistic estimate of model performance on previously unseen students.

---

## Models Compared

Three supervised classification models are evaluated:

### Logistic Regression

Used as an interpretable linear baseline with balanced class weighting.

### Random Forest

Used as a nonlinear bagging ensemble capable of modeling interactions between behavioral and academic features.

### XGBoost

Used as a gradient-boosting ensemble that sequentially improves predictions by learning from previous errors.

The models are evaluated using:

- **PR-AUC**
- **ROC-AUC**
- **Precision**
- **Recall**
- **F1-score**

Because the intervention objective emphasizes identifying students who genuinely need support, **PR-AUC and At-Risk Recall are particularly important**.

---

## Initial Model Comparison

Among the tested models, **XGBoost provides the strongest overall balance**.

Initial XGBoost performance includes:

- **PR-AUC:** 0.8807
- **At-Risk Recall:** 0.7345
- **At-Risk F1:** 0.7661

Random Forest and Logistic Regression remain competitive, but XGBoost is selected for further tuning because of its stronger overall precision-recall performance and At-Risk detection capability.

---

## XGBoost Optimization

XGBoost is tuned using **RandomizedSearchCV** with **StratifiedGroupKFold** cross-validation. Grouped cross-validation ensures that the same student is not shared between training and validation folds.

The search explores parameters including:

- Number of estimators
- Maximum tree depth
- Learning rate
- Row subsampling
- Column subsampling
- Minimum child weight
- L1 regularization
- L2 regularization

### Best Grouped Cross-Validation Result

**PR-AUC = 0.8782**

### Selected Parameters

- `n_estimators = 300`
- `max_depth = 4`
- `learning_rate = 0.04`
- `subsample = 0.85`
- `colsample_bytree = 0.85`
- `min_child_weight = 3`
- `reg_alpha = 0`
- `reg_lambda = 5`

The grouped cross-validation score is close to held-out performance, supporting the stability of the modeling approach.

---

## Decision Threshold Optimization

A default classification threshold of 0.50 is not automatically optimal for an intervention system.

For this project, **missing an At-Risk student is considered more costly than generating an additional alert**. Threshold selection therefore follows an operational objective:

1. Require **At-Risk Recall of at least 80%** on validation data.
2. Among eligible thresholds, choose the threshold with the strongest Precision, using F1 as an additional criterion.

A separate student-level validation split is created from the training data, with **zero student overlap** between fitting and validation subsets.

### Selected Threshold

**0.40**

On validation data, this threshold achieves approximately:

- **At-Risk Recall:** 0.818
- **Precision:** 0.753

The lower threshold intentionally accepts more false-positive alerts in exchange for missing fewer students who may genuinely require support.

---

## Final Model Performance

The tuned XGBoost model is refitted on the full training data and evaluated once on the untouched group-aware test set using the selected 0.40 threshold.

### Overall Results

| Metric | Final Result |
|---|---:|
| **PR-AUC** | **0.8806** |
| **ROC-AUC** | **0.8469** |
| **Accuracy** | **0.75** |
| **At-Risk Precision** | **0.74** |
| **At-Risk Recall** | **0.82** |
| **At-Risk F1** | **0.77** |

The final system identifies approximately **82% of At-Risk records** on the untouched test set.

---

## Confusion Matrix Interpretation

At the selected threshold of **0.40**:

- **2,782** At-Risk records are correctly identified
- **616** At-Risk records are missed
- **2,071** Not At Risk records are correctly classified
- **1,002** Not At Risk records are flagged as At Risk

This trade-off is intentional. In an early-intervention context, the system prioritizes detecting more students who may need support, even if advisors must review some additional false-positive alerts.

---

## Explainability with SHAP

Predictive performance alone is not sufficient for a student-support system. Advisors also need to understand why a student was flagged.

The final XGBoost model is interpreted using **SHAP (SHapley Additive exPlanations)**.

### Global Interpretability

The global SHAP analysis indicates that the model uses a combination of:

- Activity timing
- Early assessment performance
- Study frequency
- Submission behavior
- Course context
- Prior education

`last_activity_day` and `avg_score` are among the strongest global contributors.

This supports the main modeling conclusion that student risk is better represented by a combination of **behavioral and academic signals** than by any single variable.

### Individual Interpretability

For an individual student, a SHAP waterfall plot shows which features push the predicted probability toward or away from the At-Risk class.

One correctly identified At-Risk example has a predicted risk probability of approximately **0.85**. The explanation highlights early academic and submission-related signals contributing to the prediction.

For practical use, advisors should focus primarily on **actionable academic and engagement factors** rather than demographic characteristics.

---

## Key Findings

1. **Early detection is feasible.** Meaningful student-risk prediction is possible within the first four weeks of a course.

2. **Week 4 provides a practical intervention point.** PR-AUC improves from 0.813 in Week 2 to 0.861 in Week 4, while still leaving time to provide support.

3. **Engagement matters.** At-Risk students generally show fewer active study days, fewer clicks, and fewer learning materials accessed early in the course.

4. **Assessment participation and performance provide different signals.** Students who do not submit early work may be at risk, while low scores among students who do submit provide an additional academic warning signal.

5. **XGBoost provides the strongest overall model performance** among the tested approaches.

6. **Threshold selection should reflect the intervention objective.** A 0.40 threshold increases At-Risk Recall to approximately 82% rather than simply optimizing overall Accuracy.

7. **Explainability is essential.** SHAP makes individual predictions more transparent and supports targeted intervention.

8. **Actionable signals should drive intervention.** Low engagement, inactivity, non-submission, late submission, and weak early assessment performance are more useful for support planning than demographic characteristics alone.

---

## Practical Academic Intervention

A real early-warning system could generate its first risk assessment around **Week 4** and update risk probabilities periodically as new course information becomes available.

For each flagged student, an advisor could receive:

- Predicted **At-Risk probability**
- Recent engagement level
- Inactivity information
- Early assessment participation
- Assessment performance
- Submission behavior
- Main SHAP factors contributing to the prediction

The model is intended as a **decision-support tool**, not an automated decision maker. Its purpose is to help advisors identify students who may need attention and provide appropriate human support.

---

## Limitations

### Single Educational Context

OULAD represents one online-learning environment. Student behavior and course structure may differ across institutions, so external validation is required before deployment elsewhere.

### Historical Dataset

The dataset represents an earlier online-learning environment and may not fully reflect modern behavior involving mobile applications, video platforms, live sessions, and newer learning tools.

### VLE Activity Is a Proxy for Engagement

A large number of clicks does not necessarily indicate effective learning, while a student with fewer recorded clicks may still be studying successfully through other resources.

### Assessment Availability

The dataset provides submission timing but not the exact moment when every assessment grade became available to the student or advisor. Early-warning analysis therefore assumes that scores submitted within the cutoff are available at that checkpoint.

### Threshold Trade-Off

The 0.40 threshold increases At-Risk Recall but also generates more false-positive alerts. On the final test set, 1,002 Not At Risk records are flagged as At Risk and 616 At-Risk records are still missed.

### Risk Changes Over Time

Week 8 produces stronger predictive performance than Week 4, but waiting longer reduces the opportunity for early support. Week 4 is therefore a practical balance rather than a universally optimal cutoff.

### Demographic Associations Are Not Causal

Observed differences across demographic or socioeconomic groups should be interpreted as associations, not causal relationships. These variables can support fairness monitoring and population-level analysis but should not be the sole basis for individual intervention decisions.

---

## Future Work

Potential extensions include:

- **Dynamic risk monitoring:** update risk probabilities regularly rather than producing only one Week-4 prediction.
- **External validation:** evaluate the model on other institutions, courses, and more recent online-learning datasets.
- **Richer behavioral features:** model changes in engagement, assessment progression, and additional learning-platform interactions.
- **Advisor-capacity-aware thresholding:** adapt the operating threshold to the number of students advisors can realistically support.
- **Fairness evaluation:** compare error rates and predictive performance across demographic and socioeconomic groups.
- **Intervention effectiveness:** measure whether interventions triggered by the system actually improve student outcomes.
- **Model calibration:** evaluate whether predicted probabilities accurately represent observed levels of risk.
- **Deployment pipeline:** build an automated workflow that updates features and predictions as new learning-platform data arrives.

---

## Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **XGBoost**
- **SHAP**
- **Google Colab**

---

## Project Workflow

```text
OULAD Raw Tables
       |
       v
Data Quality & Cleaning
       |
       v
Binary Risk Target
       |
       v
Early Time Cutoffs
Weeks 2 / 4 / 6 / 8
       |
       v
Feature Engineering
Engagement + Inactivity + Assessments + Student Information
       |
       v
Group-Aware Train/Test Split
       |
       v
Model Comparison
Logistic Regression / Random Forest / XGBoost
       |
       v
Grouped XGBoost Optimization
       |
       v
Validation-Based Threshold Selection
       |
       v
Untouched Test Evaluation
       |
       v
SHAP Explainability
       |
       v
Advisor-Focused Early Intervention
```

---

## Repository Structure

```text
Early-Student-Risk-Detection-OULAD/
|
|-- SamaherAlsharif_OULAD_CaseStudy.ipynb
|-- README.md
```

The OULAD raw CSV files are not required to be stored in this repository. They can be obtained from the dataset source linked above and placed in the data directory expected by the notebook.

---

## Conclusion

This project demonstrates that student risk can be detected meaningfully during the **first four weeks of an online course** by combining engagement behavior, assessment information, and student context.

The final tuned XGBoost system achieves **PR-AUC = 0.8806**, **ROC-AUC = 0.8469**, and **At-Risk Recall = 0.82** at a validation-selected threshold of 0.40. SHAP explanations add transparency by showing which factors contribute to individual risk predictions.

The most important outcome is not simply maximizing Accuracy. The proposed system combines:

> **Early detection + strong At-Risk Recall + explainable predictions + actionable intervention signals**

This makes the model suitable as a foundation for a human-centered academic decision-support system in which predictions help advisors identify students who may need timely support.

