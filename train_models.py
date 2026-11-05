import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef
)

print("--- Training Models & Generating Test Data ---")

# Load Dataset (Meets criteria: 30 features, 569 instances)[cite: 3]
data = load_breast_cancer(as_frame=True)
df = data.frame

X = df.drop(columns=['target'])
y = df['target']

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Export test_data.csv for Streamlit upload requirement[cite: 3, 5]
df_test = pd.concat([X_test, y_test], axis=1)
df_test.to_csv("test_data.csv", index=False)
print("[INFO] 'test_data.csv' successfully saved!")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.pkl")

# Define 5 Classifiers[cite: 3]
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(probability=True, random_state=42)
}

for name, model in models.items():
    if name in ["Decision Tree", "Naive Bayes"]:
        model.fit(X_train, y_train)
    else:
        model.fit(X_train_scaled, y_train)
        
    filename = f"model/{name.lower().replace(' ', '_')}.pkl"
    joblib.dump(model, filename)

print("[INFO] All model artifacts saved successfully in 'model/' folder!")