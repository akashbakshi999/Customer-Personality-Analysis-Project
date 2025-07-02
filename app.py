# app.py

import streamlit as st
import pandas as pd
import pickle
import os

st.title("🧠 Customer Personality Segmentation")
st.markdown("Upload your Excel data to segment customers using KMeans clustering.")

# Check if model files exist
if not os.path.exists("customer_model.pkl") or not os.path.exists("customer_scaler.pkl"):
    st.error("Model files not found. Please run model_training.py first.")
    st.stop()

# Load model and scaler
with open("customer_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("customer_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.write("### Uploaded Data", df.head())

    # Drop unneeded columns
    df.drop(columns=['ID', 'Z_CostContact', 'Z_Revenue'], errors='ignore', inplace=True)
    df.dropna(inplace=True)

    # Select only numeric columns
    X = df.select_dtypes(include=['int64', 'float64'])

    # Predict clusters
    X_scaled = scaler.transform(X)
    clusters = model.predict(X_scaled)

    df['Cluster'] = clusters
    st.write("### Clustered Data", df.head())

    st.download_button("📥 Download Clustered Results", df.to_csv(index=False), "clustered_results.csv", "text/csv")
