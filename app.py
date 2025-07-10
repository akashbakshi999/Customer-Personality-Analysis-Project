pip install streamlit pandas numpy matplotlib seaborn scikit-learn
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import datetime

# --- Configuration ---
# Set Streamlit page configuration for a wide layout and a descriptive title
st.set_page_config(layout="wide", page_title="Customer Personality Analysis")

# --- Helper Functions ---

@st.cache_data
def load_data(file_path):
    """
    Loads the dataset from a CSV file.
    It first attempts to load with a tab separator, as this is common for marketing
    campaign datasets. If that fails, it tries with a comma separator.
    """
    try:
        df = pd.read_csv(file_path, sep='\t')
        st.success("Data loaded successfully with tab separator.")
    except Exception as e:
        st.warning(f"Could not load data with tab separator: {e}. Trying comma separator...")
        try:
            df = pd.read_csv(file_path, sep=',')
            st.success("Data loaded successfully with comma separator.")
        except Exception as e:
            st.error(f"Could not load data with comma separator either: {e}. Please check your file delimiter.")
            st.stop() # Stop the app if data cannot be loaded
    return df

@st.cache_data
def preprocess_data(df_input):
    """
    Performs data cleaning and feature engineering on the input DataFrame.
    This function is cached to speed up re-runs when input data doesn't change.
    """
    df = df_input.copy() # Work on a copy to avoid modifying the original cached DataFrame

    # 1. Handle Missing Values: Drop rows where 'Income' is missing, as it's a critical feature.
    initial_rows = df.shape[0]
    df.dropna(subset=['Income'], inplace=True)
    st.info(f"Dropped {initial_rows - df.shape[0]} rows with missing 'Income' values.")

    # 2. Feature Engineering: Date-related features
    # Convert 'Dt_Customer' to datetime objects
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y', errors='coerce')
    # Use a fixed current date (e.g., 2025-07-10) for consistent 'Enrollment_Days' calculation
    # This ensures reproducibility regardless of when the app is run.
    current_date = datetime.datetime(2025, 7, 10)
    df['Enrollment_Days'] = (current_date - df['Dt_Customer']).dt.days
    # Handle any remaining NaT from date conversion (e.g., invalid date formats)
    df.dropna(subset=['Enrollment_Days'], inplace=True)
    df['Enrollment_Days'] = df['Enrollment_Days'].astype(int) # Ensure integer type

    # 3. Feature Engineering: Age calculation
    # Calculate 'Age' assuming the current year is 2025
    df['Age'] = 2025 - df['Year_Birth']
    # Filter out unrealistic ages (e.g., very old or very young, likely data entry errors)
    df = df[(df['Age'] >= 18) & (df['Age'] <= 100)] # Keep ages between 18 and 100

    # 4. Feature Engineering: Total Spending
    # Sum up spending across all product categories
    df['Total_Spending'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + \
                           df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']

    # 5. Feature Engineering: Children and Family Size
    # Total number of children (kids + teens) at home
    df['Children'] = df['Kidhome'] + df['Teenhome']
    # Calculate 'Family_Size' based on marital status and number of children
    # Assuming 2 for married/together, 1 for others + number of children
    df['Family_Size'] = df['Children'] + df['Marital_Status'].apply(lambda x: 2 if x in ['Married', 'Together'] else 1)

    # 6. Feature Engineering: Is Parent (binary flag)
    # 1 if the customer has any children, 0 otherwise
    df['Is_Parent'] = df['Children'].apply(lambda x: 1 if x > 0 else 0)

    # 7. Handle Categorical Features: 'Marital_Status'
    # Group rare or similar marital status categories into 'Single' for simplification
    df['Marital_Status'] = df['Marital_Status'].replace(['Absurd', 'Alone', 'YOLO'], 'Single')
    # One-hot encode 'Marital_Status' to convert it into numerical format suitable for clustering.
    # `drop_first=True` prevents multicollinearity.
    df = pd.get_dummies(df, columns=['Marital_Status'], prefix='Marital_Status', drop_first=True)

    # 8. Handle Categorical Features: 'Education'
    # One-hot encode 'Education' levels
    df = pd.get_dummies(df, columns=['Education'], prefix='Education', drop_first=True)

    # 9. Drop Irrelevant Columns
    # Remove original columns that are no longer needed, have been transformed,
    # or are not relevant for personality analysis/clustering.
    columns_to_drop = [
        'ID', 'Year_Birth', 'Dt_Customer', # Transformed or identifier
        'Z_CostContact', 'Z_Revenue', # Seemingly constant or less relevant for customer personality
        'Kidhome', 'Teenhome' # Combined into 'Children'
    ]
    df.drop(columns=columns_to_drop, errors='ignore', inplace=True)

    return df

@st.cache_data
def perform_clustering(df_scaled, n_clusters):
    """
    Performs K-Means clustering on the scaled data.
    This function is cached to prevent re-running clustering if parameters don't change.
    """
    # Initialize KMeans with the selected number of clusters and a fixed random state for reproducibility.
    # n_init=10 specifies the number of times the K-Means algorithm will be run with different centroid seeds.
    # The final results will be the best output of n_init consecutive runs in terms of inertia.
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(df_scaled) # Fit the KMeans model to the scaled data
    df_clustered = df_scaled.copy()
    df_clustered['Cluster'] = kmeans.labels_ # Assign the cluster labels to the DataFrame
    return kmeans, df_clustered

# --- Main Streamlit App Logic ---
def main():
    st.title("Customer Personality Analysis App")
    st.markdown("This interactive application guides you through the process of customer personality analysis and segmentation using K-Means clustering.")

    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload your marketing campaign CSV file (e.g., marketing_campaign.csv)", type=["csv"])

    if uploaded_file is not None:
        # Load the data
        df = load_data(uploaded_file)

        st.header("1. Raw Data Overview")
        st.write("First 5 rows of the raw dataset:")
        st.write(df.head())
        st.write(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
        st.write("Basic descriptive statistics of numerical columns:")
        st.write(df.describe())

        st.header("2. Data Preprocessing & Feature Engineering")
        st.markdown("""
        This crucial step transforms the raw data into a suitable format for analysis and clustering. It includes:
        - **Handling Missing Values:** Rows with missing `Income` are removed.
        - **Date-based Features:** Calculating `Age` from `Year_Birth` and `Enrollment_Days` (days since customer enrollment).
        - **Spending Aggregation:** Summing up all product spending into `Total_Spending`.
        - **Family Features:** Creating `Children` (total kids/teens) and `Family_Size` (adults + children).
        - **Binary Flags:** `Is_Parent` (1 if customer has children, 0 otherwise).
        - **Categorical Encoding:** Grouping rare `Marital_Status` categories and then applying one-hot encoding to both `Marital_Status` and `Education` to convert them into numerical features.
        - **Column Removal:** Dropping original columns that are redundant or not directly used in the analysis.
        """)
        df_processed = preprocess_data(df.copy()) # Pass a copy to preprocess to preserve original df for EDA
        st.write("Processed Data Sample (first 5 rows):")
        st.write(df_processed.head())
        st.write(f"Processed Dataset shape: {df_processed.shape[0]} rows, {df.shape[1]} columns")
        st.write("Missing values after preprocessing:")
        st.write(df_processed.isnull().sum())

        st.header("3. Exploratory Data Analysis (EDA)")
        st.markdown("Visualizations to understand the distribution and relationships within your customer data.")

        # Plotting distributions of key demographic and spending features
        st.subheader("Distribution of Key Demographic and Spending Features")
        fig_demographics, axes_demographics = plt.subplots(1, 3, figsize=(20, 6))
        sns.histplot(df_processed['Age'], kde=True, ax=axes_demographics[0], color='skyblue')
        axes_demographics[0].set_title("Distribution of Age")
        sns.histplot(df_processed['Income'], kde=True, ax=axes_demographics[1], color='lightcoral')
        axes_demographics[1].set_title("Distribution of Income")
        sns.histplot(df_processed['Total_Spending'], kde=True, ax=axes_demographics[2], color='lightgreen')
        axes_demographics[2].set_title("Distribution of Total Spending")
        plt.tight_layout()
        st.pyplot(fig_demographics)

        # Plotting distributions of original categorical features (Marital Status and Education)
        st.subheader("Original Marital Status and Education Distribution")
        # Load a fresh copy of the original data for these specific plots to show original categories
        original_df_for_eda = load_data(uploaded_file)
        original_df_for_eda.dropna(subset=['Income'], inplace=True) # Apply same income drop for consistency
        # Apply the same marital status grouping for consistency in EDA plot
        original_df_for_eda['Marital_Status'] = original_df_for_eda['Marital_Status'].replace(['Absurd', 'Alone', 'YOLO'], 'Single')

        fig_categorical_orig, axes_categorical_orig = plt.subplots(1, 2, figsize=(15, 6))
        sns.countplot(x='Marital_Status', data=original_df_for_eda, ax=axes_categorical_orig[0], palette='pastel')
        axes_categorical_orig[0].set_title("Original Marital Status Distribution (Grouped)")
        axes_categorical_orig[0].tick_params(axis='x', rotation=45)

        sns.countplot(x='Education', data=original_df_for_eda, ax=axes_categorical_orig[1], palette='pastel')
        axes_categorical_orig[1].set_title("Original Education Distribution")
        axes_categorical_orig[1].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig_categorical_orig)


        st.header("4. Customer Segmentation (Clustering)")
        st.markdown("Here, we will use K-Means clustering to group customers into distinct segments based on their personality traits and behaviors.")

        # Define features to be used for clustering. These must be numerical.
        cluster_features = [
            'Age', 'Income', 'Total_Spending', 'Enrollment_Days',
            'Children', 'Family_Size', 'Is_Parent',
            'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts',
            'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
            'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
            'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp1',
            'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5',
            'Complain', 'Response'
        ]
        # Dynamically add the one-hot encoded marital status and education columns
        marital_status_cols = [col for col in df_processed.columns if col.startswith('Marital_Status_')]
        education_cols = [col for col in df_processed.columns if col.startswith('Education_')]
        cluster_features.extend(marital_status_cols)
        cluster_features.extend(education_cols)

        # Filter out any features from the list that might not exist in the processed DataFrame
        # (e.g., if a specific category was not present in the original data and thus no OHE column was created)
        cluster_features = [f for f in cluster_features if f in df_processed.columns]

        X = df_processed[cluster_features]

        st.subheader("Scaling Features")
        st.write("Features are scaled using `StandardScaler`. This is crucial for distance-based algorithms like K-Means, ensuring all features contribute equally regardless of their original scale.")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        st.write("Scaled features sample (first 5 rows):")
        st.write(df_scaled.head())

        st.subheader("Optimal Number of Clusters (Elbow Method)")
        st.write("The Elbow Method helps determine the optimal number of clusters (K). It plots the Sum of Squared Errors (SSE) against the number of clusters. The 'elbow point' on the graph, where the rate of decrease in SSE significantly slows down, is often considered the optimal K.")
        sse = []
        k_range = range(1, 11) # Test K from 1 to 10
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            sse.append(kmeans.inertia_) # Inertia is the SSE

        fig_elbow, ax_elbow = plt.subplots(figsize=(10, 6))
        ax_elbow.plot(k_range, sse, marker='o', linestyle='-', color='blue')
        ax_elbow.set_xlabel("Number of Clusters (K)")
        ax_elbow.set_ylabel("SSE (Sum of Squared Errors)")
        ax_elbow.set_title("Elbow Method for Optimal K")
        ax_elbow.grid(True)
        st.pyplot(fig_elbow)

        st.sidebar.subheader("Clustering Parameters")
        # Slider to allow user to select the number of clusters
        n_clusters = st.sidebar.slider("Select Number of Clusters (K) for K-Means", min_value=2, max_value=8, value=4, help="Choose K based on the Elbow Method plot above.")

        kmeans_model, df_clustered_scaled = perform_clustering(df_scaled, n_clusters)
        df_clustered_original = df_processed.copy()
        df_clustered_original['Cluster'] = kmeans_model.labels_ # Add cluster labels to the original (pre-scaled) processed dataframe
        st.subheader(f"Clustering Results with K = {n_clusters}")
        st.write("Data with assigned clusters (first 5 rows, showing key features and cluster):")
        st.write(df_clustered_original[['Age', 'Income', 'Total_Spending', 'Cluster']].head())
        st.write("Distribution of customers across the identified clusters:")
        st.write(df_clustered_original['Cluster'].value_counts().sort_index())

        st.header("5. Cluster Analysis & Insights")
        st.markdown("This section helps in understanding the distinct characteristics of each customer segment by examining the average values of key features within each cluster. This allows for targeted marketing strategies.")

        st.subheader("Average Characteristics per Cluster")
        # Calculate the mean of all numerical features for each cluster
        cluster_summary = df_clustered_original.groupby('Cluster')[cluster_features].mean().round(2)
        st.write(cluster_summary)

        st.subheader("Visualizing Cluster Characteristics")

        # Helper function to plot bar charts for cluster characteristics
        def plot_cluster_feature(df_summary, feature_name, title):
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x=df_summary.index, y=df_summary[feature_name], ax=ax, palette='viridis')
            ax.set_title(title)
            ax.set_xlabel("Cluster")
            ax.set_ylabel(feature_name)
            st.pyplot(fig)

        # Plotting key features to easily compare clusters
        plot_cluster_feature(cluster_summary, 'Income', "Average Income per Cluster")
        plot_cluster_feature(cluster_summary, 'Total_Spending', "Average Total Spending per Cluster")
        plot_cluster_feature(cluster_summary, 'Age', "Average Age per Cluster")
        plot_cluster_feature(cluster_summary, 'Recency', "Average Recency (Days Since Last Purchase) per Cluster")
        plot_cluster_feature(cluster_summary, 'Children', "Average Number of Children per Cluster")
        plot_cluster_feature(cluster_summary, 'NumWebPurchases', "Average Web Purchases per Cluster")
        plot_cluster_feature(cluster_summary, 'NumStorePurchases', "Average Store Purchases per Cluster")

        st.subheader("Key Insights and Marketing Strategies per Cluster:")
        st.write("Based on the average characteristics observed in the summary table and plots, here are some interpretations and potential marketing strategies for each customer cluster:")

        # Iterate through each cluster to provide insights
        for i in range(n_clusters):
            st.write(f"### Cluster {i}:")
            st.write(f"**Average Income:** ${cluster_summary.loc[i, 'Income']:,}")
            st.write(f"**Average Total Spending:** ${cluster_summary.loc[i, 'Total_Spending']:,}")
            st.write(f"**Average Age:** {cluster_summary.loc[i, 'Age']:.0f} years")
            st.write(f"**Average Recency (days since last purchase):** {cluster_summary.loc[i, 'Recency']:.0f} days")
            st.write(f"**Average Children:** {cluster_summary.loc[i, 'Children']:.1f}")
            st.write(f"**Average Web Purchases:** {cluster_summary.loc[i, 'NumWebPurchases']:.1f}")
            st.write(f"**Average Store Purchases:** {cluster_summary.loc[i, 'NumStorePurchases']:.1f}")
            st.write(f"**Average Accepted Campaigns (1-5):** {cluster_summary.loc[i, ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']].sum():.1f}")


            # Provide a generic insight based on key characteristics
            insight = "This cluster typically consists of "
            if cluster_summary.loc[i, 'Age'] < df_clustered_original['Age'].mean() - 5:
                insight += "younger "
            elif cluster_summary.loc[i, 'Age'] > df_clustered_original['Age'].mean() + 5:
                insight += "older "
            else:
                insight += "middle-aged "

            if cluster_summary.loc[i, 'Income'] > df_clustered_original['Income'].mean() * 1.2:
                insight += "high-income "
            elif cluster_summary.loc[i, 'Income'] < df_clustered_original['Income'].mean() * 0.8:
                insight += "lower-income "
            else:
                insight += "average-income "

            if cluster_summary.loc[i, 'Total_Spending'] > df_clustered_original['Total_Spending'].mean() * 1.2:
                insight += "high-spending customers. "
            elif cluster_summary.loc[i, 'Total_Spending'] < df_clustered_original['Total_Spending'].mean() * 0.8:
                insight += "low-spending customers. "
            else:
                insight += "average-spending customers. "

            if cluster_summary.loc[i, 'Children'] > 0.5:
                insight += "They are often families with children. "
            else:
                insight += "They are often individuals or couples without children. "

            if cluster_summary.loc[i, 'Recency'] < df_clustered_original['Recency'].mean() * 0.8:
                insight += "They have a recent purchase history, indicating high engagement. "
            else:
                insight += "They have a less recent purchase history, suggesting lower engagement. "

            st.info(insight)
            st.write("---")

    else:
        st.info("Please upload your `marketing_campaign.csv` file to begin the customer personality analysis. The app will attempt to detect the correct delimiter (tab or comma).")

# Run the main function when the script is executed
if __name__ == "__main__":
    main()
