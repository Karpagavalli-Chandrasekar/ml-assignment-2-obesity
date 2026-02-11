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
All comparison results are generated using probability threshold = 0.50 for fair evaluation.

- Accuracy – Overall classification correctness  
- AUC – Ability to separate classes  
- MCC (Matthews Correlation Coefficient) – Balanced performance metric  
- Precision (Weighted) – Control of false positives  
- Recall (Weighted) – Control of false negatives  
- F1 Score (Weighted) – Harmonic balance of Precision and Recall  


| Model | Accuracy | AUC | MCC | Precision | Recall | F1 |
|-------|----------|-----|-----|-----------|--------|-----|
| Logistic Regression | 0.9867 | 0.9994 | 0.9659 | 0.9868 | 0.9867 | 0.9867 |
| Decision Tree | 0.9735 | 0.9683 | 0.9323 | 0.9736 | 0.9735 | 0.9735 |
| KNN | 0.9034 | 0.9320 | 0.7477 | 0.9020 | 0.9034 | 0.9024 |
| Naive Bayes | 0.8655 | 0.9308 | 0.6680 | 0.8713 | 0.8655 | 0.8676 |
| Random Forest | 0.9735 | 0.9958 | 0.9333 | 0.9741 | 0.9735 | 0.9737 |
| XGBoost | 0.9867 | 0.9995 | 0.9658 | 0.9867 | 0.9867 | 0.9867 |


## e. Model Observations
| Model | Observation |
|-------|-------------|
| Logistic Regression | Logistic Regression performed exceptionally well with high Accuracy (0.9867) and near-perfect AUC (0.9994). This suggests that the dataset is largely linearly separable after preprocessing. The model shows balanced Precision and Recall, indicating very few false positives and false negatives. |
| Decision Tree | The Decision Tree achieved strong Accuracy (0.9735) but slightly lower AUC compared to ensemble methods. This may be due to its tendency to overfit training data when not heavily regularized. While performance is good, it is slightly less stable than ensemble approaches. |
| KNN | KNN showed comparatively lower performance (Accuracy 0.9034). Since KNN is distance-based, it is sensitive to feature scaling and local noise. Although scaling was applied, the model may struggle due to overlapping class regions in feature space. |
| Naive Bayes | Naive Bayes produced the lowest performance among all models. This is likely due to its strong assumption of conditional independence between features, which may not hold true for this dataset. As a result, its decision boundaries are less flexible. |
| Random Forest | Random Forest performed very well (Accuracy 0.9735, AUC 0.9958). As an ensemble of decision trees, it reduces variance and improves generalization compared to a single Decision Tree. It provides a good balance between bias and variance. |
| XGBoost | XGBoost achieved the best overall performance (Accuracy 0.9867, AUC 0.9995, MCC 0.9658). The boosting mechanism iteratively corrects errors from previous trees, leading to improved predictive power and class separation. This explains its superior performance compared to other models. |



## Best Model
XGBoost achieved the highest performance across all evaluation metrics. It showed the highest Accuracy, near-perfect AUC (excellent class separation), highest MCC (balanced predictive power), and strong Precision, Recall, and F1. This indicates superior generalization and robustness compared to other models.


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
