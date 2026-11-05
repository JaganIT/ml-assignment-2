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
| **Logistic Regression** | 0.9649 | 0.9912 | 0.9577 | 0.9859 | 0.9716 | 0.9234 |
| **Decision Tree** | 0.9298 | 0.9215 | 0.9420 | 0.9296 | 0.9358 | 0.8502 |
| **KNN** | 0.9386 | 0.9754 | 0.9315 | 0.9718 | 0.9512 | 0.8665 |
| **Naive Bayes** | 0.9386 | 0.9850 | 0.9444 | 0.9577 | 0.9511 | 0.8659 |
| **Support Vector Machine** | 0.9649 | 0.9912 | 0.9577 | 0.9859 | 0.9716 | 0.9234 |

### Observations on Model Performance
* **Logistic Regression**: Performed exceptionally well due to the linear separability of the scaled features, yielding high accuracy and balanced precision/recall.
* **Decision Tree**: Prone to slight overfitting compared to linear models, resulting in slightly lower accuracy and MCC scores.
* **KNN**: Performed strongly after feature scaling, capturing local patterns effectively with $K=5$.
* **Naive Bayes**: Showed robust probabilistic classification, handling feature independence assumptions well.
* **Support Vector Machine (SVM)**: Achieved top-tier performance matching Logistic Regression, benefiting from optimal margin hyperplanes on scaled data.

* **Overall Winner**: **Logistic Regression & Support Vector Machine (Tie)**, offering the highest accuracy, AUC, and MCC scores on the test set.

## e. Deployment Details
* **Live Streamlit App**: https://2026ak0561-ml-assignment-2.streamlit.app/
