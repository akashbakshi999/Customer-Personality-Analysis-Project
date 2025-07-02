# model_training.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle

# Load the Excel file
df = pd.read_excel("marketing_campaign1.xlsx")  # ← updated file name

# Drop irrelevant columns
df.drop(columns=['ID', 'Z_CostContact', 'Z_Revenue'], errors='ignore', inplace=True)

# Drop rows with missing values
df.dropna(inplace=True)

# Select numeric features only
X = df.select_dtypes(include=['int64', 'float64'])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit KMeans model
model = KMeans(n_clusters=4, random_state=42)
model.fit(X_scaled)

# Save model and scaler
with open("customer_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("customer_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved successfully.")
