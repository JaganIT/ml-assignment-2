import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

st.set_page_config(page_title="ML Classification App - BITS AIML", page_icon="📊", layout="wide")

st.title("🩺 Advanced ML Classification & Performance Analytics Web App")
st.markdown("**Course:** M.Tech AIML | **Assignment 2**[cite: 1]")
st.markdown("Demonstrating model evaluation metrics, interactive charts, and automated summary reports[cite: 1, 3, 5].")

# Sidebar
st.sidebar.header("Configuration & Inputs")
uploaded_file = st.sidebar.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

model_options = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Support Vector Machine": "model/support_vector_machine.pkl"
}

selected_model_name = st.sidebar.selectbox("Select Classification Model", list(model_options.keys()))

@st.cache_resource
def load_artifacts(model_path):
    model = joblib.load(model_path)
    scaler = joblib.load("model/scaler.pkl") if os.path.exists("model/scaler.pkl") else None
    return model, scaler

@st.cache_data
def load_default_test_data():
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    _, X_test, _, y_test = train_test_split(df.drop(columns=['target']), df['target'], test_size=0.2, random_state=42, stratify=y_test if 'y_test' in locals() else None)
    return pd.concat([X_test, y_test], axis=1)

# Dataset Loading Handling
if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.sidebar.success("Using uploaded test dataset.")
else:
    if os.path.exists("test_data.csv"):
        test_df = pd.read_csv("test_data.csv")
        st.sidebar.info("Loaded default `test_data.csv` from folder.")
    else:
        test_df = load_default_test_data()

st.subheader("📁 Test Dataset Preview")
st.dataframe(test_df.head(3))

if "target" not in test_df.columns:
    st.error("Uploaded dataset must contain the 'target' column!")
else:
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]
    
    model_path = model_options[selected_model_name]
    if os.path.exists(model_path):
        model, scaler = load_artifacts(model_path)
        
        # Inference Prep
        if selected_model_name in ["Decision Tree", "Naive Bayes"]:
            X_inference = X_test
        else:
            X_inference = scaler.transform(X_test) if scaler else X_test
            
        preds = model.predict(X_inference)
        try:
            probs = model.predict_proba(X_inference)[:, 1]
        except:
            probs = np.zeros(len(preds))
            
        # Metrics Calculation[cite: 3, 5]
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.0
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        mcc = matthews_corrcoef(y_test, preds)
        
        # Display Metrics Row
        st.markdown(f"### ⚡ Performance Metrics for: **{selected_model_name}**")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Accuracy", f"{acc:.4f}")
        c2.metric("AUC Score", f"{auc:.4f}")
        c3.metric("Precision", f"{prec:.4f}")
        c4.metric("Recall", f"{rec:.4f}")
        c5.metric("F1 Score", f"{f1:.4f}")
        c6.metric("MCC Score", f"{mcc:.4f}")
        
        st.markdown("---")
        
        # Section: Visualizations / Charts
        st.subheader("📊 Interactive Model Performance Charts")
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("**Confusion Matrix Heatmap**[cite: 5]")
            cm = confusion_matrix(y_test, preds)
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False)
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            st.pyplot(fig_cm)
            
        with chart_col2:
            st.markdown("**ROC Curve**")
            fig_roc, ax_roc = plt.subplots(figsize=(4, 3))
            if auc > 0:
                fpr, tpr, _ = roc_curve(y_test, probs)
                ax_roc.plot(fpr, tpr, color='orange', label=f'ROC (AUC = {auc:.2f})')
                ax_roc.plot([0, 1], [0, 1], color='navy', linestyle='--')
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)

        st.markdown("---")

        # Section: Automated Summary Report
        st.subheader("📄 Automated Executive Summary Report")
        with st.container():
            st.info(f"""
            **Evaluation Summary Report for {selected_model_name}**:
            * **Overall Classification Capability**: The model achieved an Accuracy of **{acc*100:.2f}%** and an MCC score of **{mcc:.4f}**, indicating a strong correlation between predicted and actual diagnostic outcomes.
            * **Precision vs. Recall Tradeoff**: Precision stands at **{prec:.4f}** (minimizing false positives), while Recall stands at **{rec:.4f}** (ensuring minimal false negatives). 
            * **Discriminative Power**: With an AUC score of **{auc:.4f}**, the model demonstrates robust separability between malignant and benign classes.
            """)
            
            # Classification report table dropdown/display
            with st.expander("View Detailed Classification Report Data"):
                report_dict = classification_report(y_test, preds, output_dict=True)
                st.dataframe(pd.DataFrame(report_dict).transpose())
    else:
        st.error(f"Model file for {selected_model_name} missing. Please run `train_models.py` first.")