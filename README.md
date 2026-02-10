# ML Assignment 2 – Obesity Classification

## a. Problem statement and project objective
This project implements six machine learning classification models on the Obesity dataset to predict whether a person is obese or not.
The objective is to compare multiple models using comprehensive evaluation metrics and identify the best-performing classifier.

## b. Dataset Description
- Dataset Name: Obesity Dataset
- Source: Kaggle
- Problem Type: Binary Classification
- Target Variable: Obese (0 = Not Obese, 1 = Obese)

## Target Mapping
- 0 → Insufficient_Weight / Normal_Weight
- 1 → Overweight_Level_I, Overweight_Level_II, Obesity_Type_I, Obesity_Type_II, Obesity_Type_III

## c. Models used
1. Logistic Regression  
2. Decision Tree  
3. KNN  
4. Naive Bayes  
5. Random Forest  
6. XGBoost  

## d. Models Performance Comparison
Accuracy – Overall classification correctness
AUC – Class separability performance
MCC (Matthews Correlation Coefficient) – Balanced evaluation metric
Precision (Weighted) – False positive control
Recall (Weighted) – False negative control
F1 Score (Weighted) – Harmonic balance of Precision and Recall


| Model                | Accuracy | AUC    | MCC    |
|----------------------|----------|--------|--------|
| Logistic Regression  | 0.9764   | 0.9953 | 0.9390 |
| Decision Tree        | 0.9764   | 0.9923 | 0.9389 |
| KNN                  | 0.9149   | 0.9448 | 0.7781 |
| Naive Bayes          | 0.8723   | 0.9292 | 0.6847 |
| Random Forest        | 0.9669   | 0.9945 | 0.9150 |
| XGBoost              | 0.9905   | 0.9998 | 0.9759 |

## Best Model
XGBoost achieved the highest performance across all evaluation metrics.
Its shown highest Accuracy, near-perfect AUC (excellent class separation), highest MCC (balanced predictive power), strong Precision, Recall, and F1.
This indicates superior generalization and robustness compared to other models.


## Live App: 
https://obesity-classification-app.streamlit.app/
---

## How to Run the Project

### Run Individual Model
```
python model/logistic_regression.py
```

### Run Streamlit App
```
streamlit run app.py
```
