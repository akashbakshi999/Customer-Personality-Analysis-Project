import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np
import datetime

# --- Configuration ---
MODEL_PATH = 'kmeans_model.pkl'
SCALER_PATH = 'scaler.pkl'
PCA_PATH = 'pca.pkl'
DATA_FILE = 'marketing_campaign1 (1) (1) (1).xlsx - marketing_campaign.csv' # Used for initial data loading and display

# --- Load Model, Scaler, and PCA ---
@st.cache_resource # Cache the model loading to avoid reloading on every rerun
def load_assets():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        pca = joblib.load(PCA_PATH)
        return model, scaler, pca
    except FileNotFoundError:
        st.error(f"Error: Model, scaler, or PCA file not found. Please ensure '{MODEL_PATH}', '{SCALER_PATH}', and '{PCA_PATH}' are in the same directory.")
        st.stop() # Stop the app if essential files are missing
    except Exception as e:
        st.error(f"An error occurred while loading assets: {e}")
        st.stop()

kmeans_model, scaler, pca_model = load_assets()

# --- Data Preprocessing Function (consistent with deployment script) ---
def preprocess_data(df_input):
    df = df_input.copy()

    # Drop irrelevant columns for clustering (consistent with training)
    df = df.drop(columns=['ID', 'Z_CostContact', 'Z_Revenue'], errors='ignore')

    # Handle missing 'Income' values
    if 'Income' in df.columns:
        df['Income'] = df['Income'].fillna(df['Income'].median())

    # Feature: Age of Customer
    if 'Year_Birth' in df.columns:
        df['Age'] = 2025 - df['Year_Birth']
        df = df[df['Age'] < 100]
        df = df[df['Age'] > 18]
        df = df.drop(columns=['Year_Birth'])

    # Feature: Total Children
    if 'Kidhome' in df.columns and 'Teenhome' in df.columns:
        df['Total_Children'] = df['Kidhome'] + df['Teenhome']
        df = df.drop(columns=['Kidhome', 'Teenhome'])

    # Feature: Customer Enrollment Days
    if 'Dt_Customer' in df.columns:
        df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y', errors='coerce')
        df.dropna(subset=['Dt_Customer'], inplace=True)
        latest_enrollment_date = df['Dt_Customer'].max()
        df['Customer_Enrollment_Days'] = (latest_enrollment_date - df['Dt_Customer']).dt.days
        df = df.drop(columns=['Dt_Customer'])

    # Feature: Total Spend
    product_cols = [col for col in df.columns if 'Mnt' in col]
    if product_cols:
        df['Total_Spend'] = df[product_cols].sum(axis=1)

    df.dropna(inplace=True) # Drop any remaining NaNs after feature engineering

    # Encode Categorical Features (consistent with training)
    # Ensure all possible categories from training are handled
    categorical_cols = ['Education', 'Marital_Status']
    for col in categorical_cols:
        if col in df.columns:
            # For consistent one-hot encoding, we need to ensure the columns match
            # during prediction. This. is a common challenge.
            # A robust way is to re-create the dummy variables and then align columns.
            df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)

    # Align columns with the columns used during model training
    # Get the columns that the scaler was fitted on
    training_columns = scaler.feature_names_in_
    processed_df = df.reindex(columns=training_columns, fill_value=0)

    # Ensure all columns are numeric
    processed_df = processed_df.select_dtypes(include=np.number)

    return processed_df, df_input.loc[processed_df.index] # Return original rows that were processed

# --- Streamlit UI Setup ---
st.set_page_config(layout="wide", page_title="Customer Personality Analysis")

st.title("🛍️ Customer Personality Analysis")
st.markdown("""
This application helps businesses understand their customer segments based on various attributes.
By analyzing customer behavior and demographics, companies can tailor their marketing strategies
and product offerings more effectively.
""")

# --- 1. Display Overall Data and Clusters ---
st.header("📊 Overall Customer Segmentation")
st.write("Loading and analyzing the marketing campaign data to identify customer segments.")

try:
    original_df = pd.read_csv(DATA_FILE)
    st.success(f"Successfully loaded '{DATA_FILE}'.")

    # Preprocess the original data
    processed_df_for_display, original_df_aligned = preprocess_data(original_df.copy())

    # Apply scaling
    scaled_features_for_display = scaler.transform(processed_df_for_display)
    scaled_df_for_display = pd.DataFrame(scaled_features_for_display, columns=processed_df_for_display.columns, index=processed_df_for_display.index)

    # Predict clusters
    clusters = kmeans_model.predict(scaled_df_for_display)
    original_df_aligned['Cluster'] = clusters

    # Display cluster characteristics
    st.subheader("Cluster Characteristics (Mean Values)")
    cluster_summary = original_df_aligned.groupby('Cluster').mean(numeric_only=True).round(2)
    st.dataframe(cluster_summary)
    st.info("Analyze the mean values for each cluster to understand their unique characteristics (e.g., average income, total spend, age).")

    st.subheader("Cluster Sizes")
    cluster_counts = original_df_aligned['Cluster'].value_counts().sort_index()
    st.dataframe(cluster_counts.to_frame(name='Count'))
    st.info("This shows the number of customers in each identified segment.")

    # Visualize clusters using PCA
    st.subheader("Cluster Visualization (PCA)")
    # Apply the *trained* PCA model to the *scaled* data
    principal_components_display = pca_model.transform(scaled_df_for_display)
    pca_df_display = pd.DataFrame(data=principal_components_display, columns=['PC1', 'PC2'])
    pca_df_display['Cluster'] = original_df_aligned['Cluster'].values # Ensure index alignment

    fig = px.scatter(pca_df_display, x='PC1', y='PC2', color='Cluster',
                     title='Customer Segments (PCA)',
                     labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
                     hover_data={'Cluster': True})
    st.plotly_chart(fig, use_container_width=True)
    st.info("This scatter plot shows customer segments in a 2D space. Customers within the same cluster are similar.")

except Exception as e:
    st.error(f"An error occurred during initial data processing and display: {e}")
    st.warning("Please ensure the `marketing_campaign1 (1) (1) (1).xlsx - marketing_campaign.csv` file is correctly formatted and available.")


# --- 2. Predict Segment for a New Customer ---
st.header("🔮 Predict Segment for a New Customer")
st.write("Enter hypothetical customer details to see which segment they belong to.")

with st.form("new_customer_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        income = st.number_input("Income ($)", min_value=0, value=50000)
        age_input = st.number_input("Age (Years)", min_value=18, max_value=99, value=35)
        kidhome = st.number_input("Number of Kids at Home", min_value=0, max_value=3, value=0)
    with col2:
        teenhome = st.number_input("Number of Teens at Home", min_value=0, max_value=3, value=0)
        recency = st.number_input("Days Since Last Purchase", min_value=0, max_value=100, value=20)
        mnt_wines = st.number_input("Amount Spent on Wines ($)", min_value=0, value=100)
    with col3:
        mnt_fruits = st.number_input("Amount Spent on Fruits ($)", min_value=0, value=10)
        mnt_meat = st.number_input("Amount Spent on Meat Products ($)", min_value=0, value=50)
        mnt_fish = st.number_input("Amount Spent on Fish Products ($)", min_value=0, value=5)
        mnt_sweet = st.number_input("Amount Spent on Sweet Products ($)", min_value=0, value=5)
        mnt_gold = st.number_input("Amount Spent on Gold Products ($)", min_value=0, value=15)
    
    education = st.selectbox("Education", ['Graduation', 'PhD', 'Master', '2n Cycle', 'Basic'])
    marital_status = st.selectbox("Marital Status", ['Married', 'Together', 'Single', 'Divorced', 'Widowed', 'Alone', 'Absurd', 'YOLO'])

    num_deals_purchases = st.number_input("Number of Purchases with Discount", min_value=0, value=1)
    num_web_purchases = st.number_input("Number of Web Purchases", min_value=0, value=2)
    num_catalog_purchases = st.number_input("Number of Catalog Purchases", min_value=0, value=1)
    num_store_purchases = st.number_input("Number of Store Purchases", min_value=0, value=3)
    num_web_visits_month = st.number_input("Number of Web Visits Last Month", min_value=0, value=5)
    accepted_cmp1 = st.checkbox("Accepted Campaign 1")
    accepted_cmp2 = st.checkbox("Accepted Campaign 2")
    accepted_cmp3 = st.checkbox("Accepted Campaign 3")
    accepted_cmp4 = st.checkbox("Accepted Campaign 4")
    accepted_cmp5 = st.checkbox("Accepted Campaign 5")
    complain = st.checkbox("Complained")

    submitted = st.form_submit_button("Predict Customer Segment")

    if submitted:
        # Create a DataFrame for the new customer
        new_customer_data = {
            'Income': income,
            'Recency': recency,
            'MntWines': mnt_wines,
            'MntFruits': mnt_fruits,
            'MntMeatProducts': mnt_meat,
            'MntFishProducts': mnt_fish,
            'MntSweetProducts': mnt_sweet,
            'MntGoldProds': mnt_gold,
            'NumDealsPurchases': num_deals_purchases,
            'NumWebPurchases': num_web_purchases,
            'NumCatalogPurchases': num_catalog_purchases,
            'NumStorePurchases': num_store_purchases,
            'NumWebVisitsMonth': num_web_visits_month,
            'AcceptedCmp1': int(accepted_cmp1),
            'AcceptedCmp2': int(accepted_cmp2),
            'AcceptedCmp3': int(accepted_cmp3),
            'AcceptedCmp4': int(accepted_cmp4),
            'AcceptedCmp5': int(accepted_cmp5),
            'Complain': int(complain),
            'Response': 0, # Assuming new customer hasn't responded to latest campaign yet
            'Education': education,
            'Marital_Status': marital_status,
            'Kidhome': kidhome,
            'Teenhome': teenhome,
            'Year_Birth': 2025 - age_input, # Convert age back to year_birth for processing
            'Dt_Customer': datetime.datetime.now().strftime('%d-%m-%Y') # Use current date for new customer
        }
        new_customer_df = pd.DataFrame([new_customer_data])

        try:
            # Preprocess the new customer data using the same function
            processed_new_customer_df, _ = preprocess_data(new_customer_df)

            # Scale the new customer data
            scaled_new_customer = scaler.transform(processed_new_customer_df)

            # Predict the cluster
            predicted_cluster = kmeans_model.predict(scaled_new_customer)[0]

            st.success(f"The new customer belongs to **Cluster {predicted_cluster}**!")
            st.write("You can now tailor your marketing efforts based on the characteristics of this segment.")

            # Optionally, show the processed data for the new customer
            # st.subheader("Processed New Customer Data (for debugging)")
            # st.dataframe(processed_new_customer_df)

        except Exception as e:
            st.error(f"An error occurred during prediction for the new customer: {e}")
            st.warning("Please ensure all input fields are valid and the model assets are correctly loaded.")

st.markdown("---")
st.markdown("Developed for Customer Personality Analysis.")
