# Unveiling Academic Trajectories Through Machine Learning

A multiclass machine learning project for predicting student academic status — **Graduate, Dropout, or Enrolled** — using academic, demographic, and socioeconomic features.

The project compares linear and tree-based classification models, addresses class imbalance with SMOTE, and evaluates model performance using accuracy, precision, recall, F1-score, ROC/AUC, and confusion matrices.

## Project Overview

The goal is to analyze whether machine learning can distinguish different student academic trajectories from a structured student dataset.

The dataset contains:

- **4,424 student records**
- **37 features**
- Academic, demographic, and socioeconomic variables
- Target classes:
  - Graduate
  - Dropout
  - Enrolled

The original project uses the Kaggle student academic dataset.

## Methodology

The workflow consists of two main stages.

### 1. Data Preparation

The project performs:

- Exploratory data analysis
- Feature analysis using correlation and domain understanding
- Categorical encoding
- Numerical feature scaling
- Train/test splitting
- Class-imbalance handling with **SMOTE**

SMOTE is applied only to the training data to reduce the risk of information leakage.

### 2. Modelling & Validation

Five classification approaches are evaluated:

1. Logistic Regression
2. Support Vector Classifier (SVC)
3. Random Forest
4. LightGBM
5. Histogram-based Gradient Boosting (HistGradientBoosting)

The project also applies cross-validation and **RandomizedSearchCV** for hyperparameter tuning.

Model selection and comparison use:

- Accuracy
- Precision
- Recall
- F1-score
- ROC/AUC
- Confusion matrix

## Model Results

The reported evaluation results are:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| HistGradientBoosting | 0.955932 | 0.955959 | 0.955932 | 0.955537 |
| LightGBM (Tuned) | 0.955932 | 0.955927 | 0.955932 | 0.955553 |
| Random Forest | 0.954802 | 0.955424 | 0.954802 | 0.954191 |
| SVC (Tuned) | 0.850847 | 0.848505 | 0.850847 | 0.842047 |
| Logistic Regression | 0.794350 | 0.781923 | 0.794350 | 0.772743 |

The project reports that HistGradientBoosting and tuned LightGBM achieve very similar evaluation metrics, while Random Forest is also competitive. SVC and Logistic Regression obtain lower scores on the reported evaluation.

The ROC analysis reports AUC values of approximately:

- LightGBM: ~0.913
- HistGradientBoosting: ~0.912–0.913
- Random Forest: ~0.912–0.913
- Logistic Regression: ~0.910
- SVC: ~0.888

## Class Imbalance

The original target distribution is imbalanced, with the **Graduate** category representing the majority class. The project uses SMOTE to balance the training distribution and improve the opportunity for the models to learn the minority classes, particularly Dropout.

The poster specifically illustrates the target distribution before and after SMOTE.

## Key Predictive Features

The project identifies several variables as important predictors of academic status, including:

- GPA / academic performance
- Credit-related variables
- Age at enrollment
- Admission grades
- Scholarship status

The analysis frames these features as relevant predictors within the dataset rather than as universal determinants of student outcomes.

## Confusion Matrix & ROC Analysis

The confusion matrices are used to inspect class-specific prediction errors across Graduate, Dropout, and Enrolled.

The project reports that HistGradientBoosting, LightGBM, and Random Forest produce relatively few misclassifications across the classes, while SVC and Logistic Regression show more difficulty with the Dropout-related class predictions.

The ROC curves provide an additional view of multiclass discrimination, with the tree-based ensemble models showing AUC values around 0.912–0.913 in the reported evaluation.

## Repository Structure

```text
student-academic-trajectory-ml/
├── README.md
├── LICENSE
├── .gitignore
├── notebooks/
│   └── academic-trajectory-classification.ipynb
├── data/
│   └── students_dropout_academic_success.csv
└── docs/
    ├── project-poster.pdf
    └── presentation-link.md
```

## Tools & Libraries

The project uses Python-based machine learning and data analysis tools, including:

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- LightGBM
- Imbalanced-learn
- Jupyter Notebook

## Reproducibility

Create a Python environment and install the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn lightgbm imbalanced-learn jupyter
```

Then open:

```text
notebooks/academic-trajectory-classification.ipynb
```

The dataset is provided in:

```text
data/students_dropout_academic_success.csv
```

Run the notebook sequentially to reproduce the analysis.

## Presentation

The project presentation video is available through the Google Drive link provided for the project:

https://drive.google.com/file/d/1dEqXrtHIJl-gDNfjNUuUrzu8Pzyy8aEL/view?usp=drive_link

A copy of the project poster is also included in:

```text
docs/project-poster.pdf
```

## Academic Context

**Project:** Unveiling Academic Trajectories Through Machine Learning Analysis

**Authors:**
- M. Aufa Mumtaza Ibadillah
- Vanessa Santoso
- Zahra Annisa Afandi

**Institution:** Statistics Department, School of Computer Science, Bina Nusantara University, Jakarta, Indonesia.

## Limitations

- The dataset is a secondary dataset sourced from Kaggle.
- Model performance is specific to the dataset and evaluation procedure used in the project.
- Class imbalance can affect multiclass classification performance, particularly for the minority Dropout class.
- High test performance should not automatically be interpreted as equivalent performance on other universities, populations, or future cohorts.
- The models are analytical/predictive models and are not presented as a standalone decision system for actual student intervention.

## License

This repository is intended as an academic portfolio project. See `LICENSE` for licensing information.
