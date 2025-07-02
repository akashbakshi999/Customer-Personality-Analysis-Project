import streamlit as st
import pickle
import numpy as np

# Load saved models
with open("customer_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("customer_kmeans.pkl", "rb") as f:
    kmeans = pickle.load(f)

st.title("Customer Personality Segment Predictor")

st.write("Enter the customer details below:")

income = st.number_input("Annual Income", min_value=0)
recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=100)
total_children = st.number_input("Total Children (Kidhome + Teenhome)", min_value=0, max_value=10)
age = st.number_input("Customer Age", min_value=18, max_value=100)
total_spent = st.number_input("Total Amount Spent", min_value=0)

if st.button("Predict Customer Segment"):
    input_data = np.array([[income, recency, total_children, age, total_spent]])
    input_scaled = scaler.transform(input_data)
    cluster = kmeans.predict(input_scaled)[0]
    
    st.success(f"Predicted Customer Segment: **Cluster {cluster}**")

    cluster_insights = {
        0: "Younger or lower-income moderate spenders.",
        1: "High-income, low-recency loyal customers.",
        2: "Affluent, older high spenders.",
        3: "Older, low-spending customers with more dependents."
    }

    st.info(f"Segment Insight: {cluster_insights.get(cluster, 'No insight available')}")
