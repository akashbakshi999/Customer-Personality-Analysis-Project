# model_training.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle

# Load data
df = pd.read_excel("marketing_campaign1.xlsx", sheet_name="marketing_campaign")

# Preprocess
df = df.dropna()
df['Age'] = 2025 - df['Year_Birth']
df['TotalChildren'] = df['Kidhome'] + df['Teenhome']
df['TotalSpend'] = df[['MntWines', 'MntFruits', 'MntMeatProducts',
                       'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)

features = df[['Income', 'Age', 'TotalChildren', 'Recency', 'TotalSpend']]

# Scale
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Train model
model = KMeans(n_clusters=4, random_state=42)
model.fit(scaled_features)

# Save model and scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("kmeans_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model and scaler saved.")
