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
| Logistic Regression | 0.9886 | 0.9996 | 0.9709 | 0.9888 | 0.9886 | 0.9886 |
| Decision Tree | 0.9716 | 0.9919 | 0.9265 | 0.9717 | 0.9716 | 0.9713 |
| KNN | 0.9091 | 0.9481 | 0.7647 | 0.9084 | 0.9091 | 0.9087 |
| Naive Bayes | 0.8636 | 0.9298 | 0.6662 | 0.8709 | 0.8636 | 0.8661 |
| Random Forest | 0.9621 | 0.9924 | 0.9024 | 0.9620 | 0.9621 | 0.9620 |
| XGBoost | 0.9943 | 0.9999 | 0.9854 | 0.9943 | 0.9943 | 0.9943 |



## e. Model Observations
| Model | Observation |
|-------|-------------|
| Logistic Regression | Logistic Regression performed exceptionally well with high Accuracy (0.9886) and near-perfect AUC (0.9996). This suggests that the dataset is largely linearly separable after preprocessing. The model shows balanced Precision and Recall, indicating very few false positives and false negatives. |
| Decision Tree | The Decision Tree achieved strong Accuracy (0.9716) but slightly lower AUC compared to ensemble methods. This may be due to its tendency to overfit training data when not heavily regularized. While performance is good, it is slightly less stable than ensemble approaches. |
| KNN | KNN showed comparatively lower performance (Accuracy 0.9091). Since KNN is distance-based, it is sensitive to feature scaling and local noise. Although scaling was applied, the model may struggle due to overlapping class regions in feature space. |
| Naive Bayes | Naive Bayes produced the lowest performance among all models. This is likely due to its strong assumption of conditional independence between features, which may not hold true for this dataset. As a result, its decision boundaries are less flexible. |
| Random Forest | Random Forest performed very well (Accuracy 0.9621, AUC 0.9924). As an ensemble of decision trees, it reduces variance and improves generalization compared to a single Decision Tree. It provides a good balance between bias and variance. |
| XGBoost | XGBoost achieved the best overall performance (Accuracy 0.9943, AUC 0.9999, MCC 0.9854). The boosting mechanism iteratively corrects errors from previous trees, leading to improved predictive power and class separation. This explains its superior performance compared to other models. |



## Best Model
XGBoost achieved the highest performance across all evaluation metrics. It showed the highest Accuracy, near-perfect AUC (excellent class separation), highest MCC (balanced predictive power), and strong Precision, Recall, and F1. This indicates superior generalization and robustness compared to other models.


## Live App: 
https://obesitylevel-analysis.streamlit.app/
---

## How to Run the Project

### Run Individual Model
```
python -m model.logistic_regression
python -m model.decision_tree
python -m model.knn
python -m model.naive_bayes
python -m model.random_forest
python -m model.xgboost_model

```

### Run Streamlit App
```
streamlit run app.py
```
