
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Project paths
project_folder = Path(__file__).resolve().parent.parent

dataset_path = (
    project_folder
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

results_folder = project_folder / "results"
results_folder.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(dataset_path)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

sns.set_theme(style="whitegrid")

# GRAPH 1: Overall customer churn distribution
plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="Churn")

plt.title("Customer Churn Distribution")
plt.xlabel("Customer Churn")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig(results_folder / "01_churn_distribution.png")
plt.show()


# GRAPH 2: Churn by contract type
plt.figure(figsize=(9, 5))

sns.countplot(data=df, x="Contract", hue="Churn")

plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig(results_folder / "02_churn_by_contract.png")
plt.show()


# GRAPH 3: Monthly charges and customer churn
plt.figure(figsize=(8, 5))

sns.boxplot(data=df, x="Churn", y="MonthlyCharges")

plt.title("Monthly Charges by Customer Churn")
plt.xlabel("Customer Churn")
plt.ylabel("Monthly Charges")

plt.tight_layout()
plt.savefig(results_folder / "03_monthly_charges_by_churn.png")
plt.show()



# GRAPH 4: Customer tenure by churn
plt.figure(figsize=(8, 5))

sns.boxplot(data=df, x="Churn", y="tenure")

plt.title("Customer Tenure by Churn Status")
plt.xlabel("Customer Churn")
plt.ylabel("Tenure (Months)")

plt.tight_layout()
plt.savefig(results_folder / "04_churn_by_tenure.png", dpi=300)
plt.show()


# GRAPH 5: Churn by internet service
plt.figure(figsize=(9, 5))

sns.countplot(data=df, x="InternetService", hue="Churn")

plt.title("Customer Churn by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig(results_folder / "05_churn_by_internet_service.png", dpi=300)
plt.show()


# GRAPH 6: Correlation heatmap
numeric_df = df[
    ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
].dropna()

plt.figure(figsize=(8, 6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap of Numerical Features")

plt.tight_layout()
plt.savefig(results_folder / "06_correlation_heatmap.png", dpi=300)
plt.show()

print("All six EDA visualisations completed!")


print("EDA visualisations completed successfully!")
print("Graphs saved in:", results_folder)
