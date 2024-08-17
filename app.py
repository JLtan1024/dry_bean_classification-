
import streamlit as st
import shap
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from streamlit_shap import st_shap
import openpyxl

# Load the dataset (assuming it's in the same directory)
customer = pd.read_excel("Dry_Bean_Dataset.xlsx")

# Preprocessing
X = customer.drop("Class", axis=1)
y = customer['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Model training
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# SHAP explainer
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

# Streamlit app
st.title("SHAP Analysis for Dry Bean Classification")

# Part 1: General SHAP Analysis
st.header("Part 1: General SHAP Analysis")
st.write("### Classification Report")
st.dataframe(classification_report(y_test, y_pred, output_dict=True))

# Summary plot
st.subheader("Summary Plot")
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_test, show=False)
st.pyplot(fig)

# Part 2: Individual Input Prediction & Explanation
st.header("Part 2: Individual Input Prediction & Explanation")

# Input fields for features
input_data = {}
for feature in X.columns:
    input_data[feature] = st.number_input(f"Enter {feature}:", value=float(X_test[feature].mean()), step=0.01)

# Create a DataFrame from input data
input_df = pd.DataFrame(input_data, index=[0])

# Make prediction
prediction = clf.predict(input_df)[0]
probability = clf.predict_proba(input_df)[0]  # Probabilities for all classes

# Display prediction
st.write(f"**Prediction:** {prediction}")
st.write("**Class Probabilities:**")
for i, prob in enumerate(probability):
    st.write(f"Class {i}: {prob:.2f}")

# SHAP explanation for the input
shap_values_input = explainer.shap_values(input_df)

# Force plot (for multiclass)
st.subheader("Force Plot")
st_shap(shap.force_plot(explainer.expected_value[clf.classes_.tolist().index(prediction)], shap_values_input[clf.classes_.tolist().index(prediction)], input_df), height=400, width=1000)

# Decision plot
st.subheader("Decision Plot")
st_shap(shap.decision_plot(explainer.expected_value[clf.classes_.tolist().index(prediction)], shap_values_input[clf.classes_.tolist().index(prediction)], X_test.columns))
