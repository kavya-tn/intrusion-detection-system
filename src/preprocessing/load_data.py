"""
Downloads the Cleveland Heart Disease dataset, performs initial cleaning,
handles missing values (represented by '?'), assigns appropriate column names,
converts the target variable to binary (0 or 1), and saves the processed DataFrame.
"""
import os
import pandas as pd
import numpy as np
from urllib import request

# Define column names
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

# Data URL
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
RAW_DATA_PATH = "data/processed.cleveland.data"
PROCESSED_DATA_PATH = "data/processed_heart_disease.csv"

def download_data():
    """Downloads data if not present."""
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Downloading data from {DATA_URL}...")
        request.urlretrieve(DATA_URL, RAW_DATA_PATH)
        print(f"Data downloaded to {RAW_DATA_PATH}")
    else:
        print(f"Data already exists at {RAW_DATA_PATH}")

def load_and_process_data():
    """Loads, processes, and saves the heart disease data."""
    download_data()

    # Load data from the raw file, assign column names, and mark '?' as NaN
    df = pd.read_csv(RAW_DATA_PATH, names=COLUMNS, na_values="?")
    print(f"Raw data loaded. Shape: {df.shape}")

    # Simple missing value imputation strategy:
    # For numeric columns, fill NaN with the median of the column.
    # For categorical/object columns (if any had '?'), fill NaN with the mode.
    # Note: A more sophisticated strategy might be needed for a production system,
    # potentially involving different imputation methods per feature or using domain knowledge.
    print("Handling missing values...")
    for col in df.columns:
        if df[col].isnull().any():
            print(f"Column '{col}' has {df[col].isnull().sum()} missing values.")
            if pd.api.types.is_numeric_dtype(df[col]):
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"Filled NaN in '{col}' with median: {median_val}")
            else:
                mode_val = df[col].mode()[0] # mode() can return multiple values if they have same frequency
                df[col] = df[col].fillna(mode_val)
                print(f"Filled NaN in '{col}' with mode: {mode_val}")


    # Convert target column to binary: 0 for no heart disease, 1 for presence of heart disease.
    # The original dataset uses 0 for <50% diameter narrowing and values 1, 2, 3, 4 for >50% diameter narrowing.
    print("Converting 'target' column to binary (0 or 1)...")
    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

    # Save the processed DataFrame to a CSV file
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved to {PROCESSED_DATA_PATH}. Shape: {df.shape}")
    print(f"Target value counts:\n{df['target'].value_counts()}")

if __name__ == "__main__":
    print("Starting data loading and initial processing...")
    load_and_process_data()
    print("Data loading and initial processing finished.")
