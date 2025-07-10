# app.py
import streamlit as st
import pickle
import numpy as np

# Load scaler and model
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("kmeans_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🧑‍💼 Customer Personality Cluster Predictor")

# User Input
income = st.number_input("Income", value=50000)
age = st.number_input("Age", value=35)
children = st.number_input("Total Children at Home", min_value=0, value=1)
recency = st.slider("Days Since Last Purchase", min_value=0, max_value=100, value=30)
total_spent = st.number_input("Total Amount Spent", value=1000)

if st.button("Predict Customer Segment"):
    data = np.array([[income, age, children, recency, total_spent]])
    data_scaled = scaler.transform(data)
    cluster = model.predict(data_scaled)[0]
    st.success(f"Customer belongs to Cluster: {cluster}")
