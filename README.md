
# Predicting Student Dropout Using Machine Learning

## Introduction

Student dropout is a significant issue in education systems worldwide, resulting in a loss of potential and contributing to socioeconomic disparities. Educational institutions face challenges in retaining students until they complete their courses. This project leverages machine learning to predict the likelihood of student dropouts, enabling timely interventions to improve student retention and success rates.

The project aligns with the United Nations Sustainable Development Goal 4: **Quality Education**, aiming to ensure inclusive and equitable quality education by identifying at-risk students early.

## Project Goal

The main objective of this project is to build machine learning models that can:
- Predict the likelihood of a student dropping out.
- Identify key features and factors that contribute to student dropout rates.
- Assist educational institutions in implementing early interventions for at-risk students.

## Dataset

- **Source**: UCI Machine Learning Repository
- **Original Dataset**: 4,424 instances with 36 features.
- **Preprocessed Dataset**:
  - Removed the "Enrolled" class to focus on a binary classification problem: **1 (Dropout)** vs. **0 (Graduate)**.
  - Final dataset contains **3,630 instances**.

## Data Preprocessing

- Standard scaling was applied to normalize features.
- Missing values were already handled by the dataset creators.
- SMOTE (Synthetic Minority Over-sampling Technique) was tested to address class imbalance but was not used in the final models.
- The target variable was encoded as a binary indicator, focusing on dropouts.

## Machine Learning Models

Two machine learning models were employed:

1. **Random Forest Classifier**
   - Chosen for its robustness against overfitting and its ability to handle large, imbalanced datasets.
   - Provides insights into feature importance.
   - Hyperparameters tuned using `GridSearchCV`: `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`.
   - Cross-validated to ensure generalizability.

2. **Logistic Regression**
   - Selected for its simplicity and efficiency in binary classification tasks.
   - Regularization applied to prevent overfitting.
   - Hyperparameters tuned using `GridSearchCV`: `C`, `penalty`, `max_iter`, and `intercept_scaling`.
   - Cross-validated for robust performance.

## Model Performance

- Both models achieved similar performance scores, but the **Random Forest** model demonstrated better recall, which is critical in identifying at-risk students.
- **Recall** was prioritized as false negatives (missed at-risk students) could lead to students dropping out without intervention, while false positives pose minimal risk.

| Metric       | Random Forest | Logistic Regression |
|--------------|---------------|---------------------|
| Accuracy     | High          | High                |
| Recall       | Higher        | Lower               |
| Precision    | Comparable    | Comparable          |

## Conclusion

- The **Random Forest** model was selected due to its superior recall score, making it more effective in identifying potential dropouts.
- The project demonstrates the potential of machine learning in aiding educational institutions to proactively address dropout risks.
- Future work may involve exploring additional features or advanced models, and testing on different datasets for further validation.

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Jupyter Notebook**:
   Open `AaronWallerML.ipynb` in Jupyter Notebook or JupyterLab and run all cells sequentially.

3. **Data Visualization**:
   The notebook includes various plots for data exploration and model evaluation (e.g., feature importance, ROC curves).

## Acknowledgments

- Data sourced from the UCI Machine Learning Repository.
- Project inspired by the need for quality education and aligning with UN SDG 4.
