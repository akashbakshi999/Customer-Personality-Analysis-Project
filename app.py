# app.py

import streamlit as st
import pandas as pd
import pickle

# Load model and scaler
with open("customer_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("customer_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("🧠 Customer Personality Segmentation")
st.markdown("Upload customer Excel data to predict customer segments using KMeans clustering.")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)

    st.write("### Uploaded Data", data.head())

    # Drop unneeded columns
    cols_to_drop = ['ID', 'Z_CostContact', 'Z_Revenue']
    for col in cols_to_drop:
        if col in data.columns:
            data = data.drop(columns=col)

    # Drop missing values
    data = data.dropna()

    # Select numeric columns
    X = data.select_dtypes(include=['int64', 'float64'])

    # Scale and predict clusters
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)

    data['Cluster'] = predictions
    st.write("### Clustered Data", data.head())

    st.success("✅ Clustering completed!")
    st.download_button("📥 Download Results", data.to_csv(index=False), "clustered_customers.csv", "text/csv")
