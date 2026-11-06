
import pandas as pd
from pathlib import Path

# Locate the project folder and dataset
project_folder = Path(__file__).resolve().parent.parent

dataset_path = (
    project_folder
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# Load dataset
df = pd.read_csv(dataset_path)

print("=" * 50)
print("TELECOM CUSTOMER CHURN - DATASET INSPECTION")
print("=" * 50)

# 1. Dataset dimensions
print("\n1. DATASET SHAPE")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 2. First five records
print("\n2. FIRST FIVE CUSTOMERS")
print(df.head())

# 3. Column information
print("\n3. COLUMN INFORMATION")
df.info()

# 4. Missing values
print("\n4. MISSING VALUES")
print(df.isnull().sum())

# 5. Convert TotalCharges into numeric format
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\n5. MISSING VALUES AFTER CONVERSION")
print(df.isnull().sum())

# 6. Churn distribution
print("\n6. CUSTOMER CHURN DISTRIBUTION")
print(df["Churn"].value_counts())

print("\nCHURN PERCENTAGES")
print(df["Churn"].value_counts(normalize=True) * 100)

# 7. Numerical summary
print("\n7. NUMERICAL DATA SUMMARY")
print(df.describe())

print("\n" + "=" * 50)
print("DATASET INSPECTION COMPLETED")
print("=" * 50)
