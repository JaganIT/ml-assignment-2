# Machine Learning Assignment 2: Multi-Model Classification & Deployment

Name of the student: Jagannath Moguluru
Roll number: 2026ak05616
Subject Code: AIMLZG565
 

## a. Problem Statement
The objective of this assignment is to build, evaluate, compare, and deploy multiple machine learning classification algorithms using an end-to-end MLOps pipeline. The project covers training models, storing evaluation metrics, building an interactive Streamlit frontend, and deploying the solution live.

## b. Dataset Description
* **Dataset Name**: Breast Cancer Wisconsin (Diagnostic) Dataset (sourced from Scikit-Learn / UCI Repository).
* **Problem Type**: Binary Classification (Malignant vs. Benign).
* **Feature Count**: 30 numerical features ( satisfies requirement $\ge 12$ ).
* **Instance Count**: 569 instances ( satisfies requirement $\ge 500$ ).

## c. GitHub Repository Link
https://github.com/JaganIT/ml-assignment-2

## d. Models Used & Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.9825** | **0.9954** | **0.9861** | **0.9861** | **0.9861** | **0.9623** |
| **Decision Tree** | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| **KNN** | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| **Naive Bayes** | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| **Support Vector Machine** | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

### Observations on Model Performance
* **Logistic Regression**: Performed exceptionally well due to the linear separability of the scaled features, yielding high accuracy and balanced precision/recall.
* **Decision Tree**: Prone to slight overfitting compared to linear models, resulting in slightly lower accuracy and MCC scores.
* **KNN**: Performed strongly after feature scaling, capturing local patterns effectively with $K=5$.
* **Naive Bayes**: Showed robust probabilistic classification, handling feature independence assumptions well.
* **Support Vector Machine (SVM)**: Achieved top-tier performance matching Logistic Regression, benefiting from optimal margin hyperplanes on scaled data.

* **Overall Winner**: **Logistic Regression & Support Vector Machine (Tie)**, offering the highest accuracy, AUC, and MCC scores on the test set.

## e. Deployment Details
* **Live Streamlit App**: https://2026ak0561-ml-assignment-2.streamlit.app/
