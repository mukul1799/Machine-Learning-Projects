# Census Income Prediction Using Machine Learning

This project predicts whether an individual earns **more than $50,000 per year** using demographic and employment data from the U.S. Census dataset.

---

## Overview

The goal is to build a supervised machine learning classification model to predict income category:

- **0 → ≤ $50K**
- **1 → > $50K**

The project follows an end-to-end ML pipeline including data preprocessing, EDA, model building, evaluation, and optimization.

---

## Technologies Used

- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  

---

## Key Steps

- Data cleaning and preprocessing  
- Missing value handling and duplicate removal  
- Outlier treatment using IQR method  
- Label encoding for categorical variables  
- Model training and comparison  
- Hyperparameter tuning using GridSearchCV  

---

## Models Implemented

- Logistic Regression  
- Decision Tree  
- Random Forest  

---

## Results

| Model | Accuracy |
|------|------|
| Logistic Regression | 78.6% |
| Decision Tree | 78.8% |
| Random Forest | 84.2% |
| **Random Forest (Tuned)** | **85.17%** |

---

##Final Model

**Tuned Random Forest Classifier**  
Selected for its highest accuracy and strong generalization performance.
