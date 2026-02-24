import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier

class StrategyExtractor:
    def __init__(self, data_path: str = "data/trades_dataset.csv"):
        self.data_path = data_path
        self.df = None
        self.feature_cols = [
            'RSI_14', 'MACD', 'MACD_Hist', 'Stoch_k', 'SMA_9', 'SMA_20', 
            'EMA_9', 'EMA_21', 'BB_High', 'BB_Low', 'ATR_14', 'OBV', 'VWAP', 
            'dist_to_HOD_pct', 'dist_to_LOD_pct', 'dist_to_VWAP_pct'
        ]

    def load_data(self):
        """Loads historical trades from the dataset."""
        if not os.path.exists(self.data_path):
            print(f"Dataset {self.data_path} not found.")
            return False
        
        df = pd.read_csv(self.data_path)
        self.df = df
        if df is not None:
            print(f"Loaded {len(df)} trades for analysis.")
        return True

    def _preprocess_data(self):
        """Cleans NA values and scales features for distance-based ML algorithms."""
        df = self.df
        if df is None or df.empty:
            return None
            
        # Drop rows with NaN in critical feature columns
        df_clean = df.dropna(subset=self.feature_cols)
        
        if df_clean is None or len(df_clean) < 3: # Need minimum amount of data to cluster meaningfully
            print("Not enough complete data points to run ML models (minimum 3 required).")
            return None

        X = df_clean[self.feature_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return df_clean, X_scaled
        
    def perform_eda(self, output_dir: str = "data/plots"):
        """Phase 1: Generates Exploratory Data Analysis plots."""
        print("\n--- Phase 1: Exploratory Data Analysis ---")
        df = self.df
        if df is None:
            return
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Plot distribution of key distances (VWAP, HOD, LOD)
        plt.figure(figsize=(10, 6))
        cols_to_plot = ['dist_to_VWAP_pct', 'dist_to_HOD_pct', 'dist_to_LOD_pct']
        existing_cols = [c for c in cols_to_plot if c in df.columns]
        
        if existing_cols:
            df[existing_cols].plot(kind='box', title="Distribution of Price Distances at Entry")
            plt.savefig(f"{output_dir}/distance_distributions.png")
            plt.close()
            print(f"Saved distance distribution plot to {output_dir}/distance_distributions.png")

        # Correlation heatmap
        plt.figure(figsize=(12, 10))
        corr = df[self.feature_cols].corr()
        sns.heatmap(corr, annot=False, cmap='coolwarm')
        plt.title("Feature Correlation Matrix")
        plt.savefig(f"{output_dir}/correlation_matrix.png")
        plt.close()
        print(f"Saved correlation matrix plot to {output_dir}/correlation_matrix.png")

    def train_unsupervised_clusters(self, n_clusters: int = 2):
        """Phase 3: Clusters trades to find setup archetypes."""
        print(f"\n--- Phase 3: Unsupervised Clustering (k={n_clusters}) ---")
        result = self._preprocess_data()
        if result is None or result[0] is None or result[1] is None:
            return None
        
        df_clean, X_scaled = result

        n_clusters = min(n_clusters, len(df_clean))
        if n_clusters < 2:
           print("Not enough data to form meaningful clusters.")
           df_clean = df_clean.copy()
           df_clean['Cluster'] = 0
           return df_clean 

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        df_clean = df_clean.copy()
        df_clean['Cluster'] = clusters
        
        print("\nCluster Centroids (Mean values per setup):")
        numeric_df = df_clean.select_dtypes(include=[np.number])
        cluster_summary = df_clean.groupby('Cluster')[numeric_df.columns].mean()
        print(cluster_summary[['RSI_14', 'dist_to_VWAP_pct', 'dist_to_HOD_pct', 'dist_to_LOD_pct']])
        
        return df_clean

    def extract_rules(self, df_clustered):
        """Phase 4: Uses Decision Trees to extract human-readable rules from clusters."""
        print("\n--- Phase 4: Rule Extraction via Decision Trees ---")
        if df_clustered is None:
            return
            
        if 'Cluster' not in df_clustered.columns:
            return

        # We treat the unsupervised 'Cluster' as the target label for the Decision Tree
        X = df_clustered[self.feature_cols]
        y = df_clustered['Cluster']
        
        if len(y.unique()) < 2:
            print("Only 1 cluster detected. Need multiple clusters to extract distinct rules.")
            return

        # Train a short tree for interpretability
        dt = DecisionTreeClassifier(max_depth=3, random_state=42)
        dt.fit(X, y)

        rules = export_text(dt, feature_names=self.feature_cols)
        print("\nExtracted Trading Rules based on Trade Clusters:")
        print(rules)
        
        # Feature Importance Analysis
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        importances = pd.Series(rf.feature_importances_, index=self.feature_cols).sort_values(ascending=False)
        print("\nTop 5 Most Important Features Driving the Strategy:")
        print(importances.head(5))

    def run_pipeline(self):
        """Executes the complete strategy extraction pipeline."""
        if not self.load_data() or self.df is None:
            return
            
        self.perform_eda()
        df_clustered = self.train_unsupervised_clusters()
        
        if df_clustered is not None:
             self.extract_rules(df_clustered)
