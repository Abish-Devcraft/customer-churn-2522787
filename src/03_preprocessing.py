
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

project_folder = Path(__file__).resolve().parent.parent

dataset_path = (
    project_folder
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

processed_folder = project_folder / "data" / "processed"
processed_folder.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(dataset_path)

print("=" * 50)
print("TELECOM CUSTOMER CHURN - DATA PREPROCESSING")
print("=" * 50)

print("\n1. ORIGINAL DATASET")
print("-------------------")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# --------------------------------------------------
# 3. CONVERT TOTALCHARGES TO NUMERIC
# --------------------------------------------------

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\n2. MISSING VALUES AFTER CONVERSION")
print("----------------------------------")

print(df.isnull().sum())

missing_count = df["TotalCharges"].isnull().sum()

print("\nMissing TotalCharges values:", missing_count)


# --------------------------------------------------
# 4. REMOVE CUSTOMER IDENTIFIER
# --------------------------------------------------

df = df.drop(columns=["customerID"])

print("\n3. CUSTOMER IDENTIFIER REMOVED")
print("------------------------------")
print("customerID column removed successfully.")


# --------------------------------------------------
# 5. ENCODE TARGET VARIABLE
# --------------------------------------------------

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

print("\n4. TARGET VARIABLE ENCODING")
print("---------------------------")
print("No  = 0")
print("Yes = 1")


# --------------------------------------------------
# 6. SEPARATE FEATURES AND TARGET
# --------------------------------------------------

X = df.drop(columns=["Churn"])
y = df["Churn"]

print("\n5. FEATURES AND TARGET")
print("----------------------")
print("Number of features:", X.shape[1])
print("Target variable: Churn")


# --------------------------------------------------
# 7. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n6. TRAIN-TEST SPLIT")
print("-------------------")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)


# --------------------------------------------------
# 8. CHECK MISSING VALUES
# --------------------------------------------------

print("\n7. MISSING VALUES IN TRAINING DATA")
print("----------------------------------")

print(X_train.isnull().sum())

print("\nMissing values in testing data:")
print(X_test.isnull().sum())


# --------------------------------------------------
# 9. SAVE PROCESSED DATASETS
# --------------------------------------------------

X_train.to_csv(
    processed_folder / "X_train.csv",
    index=False
)

X_test.to_csv(
    processed_folder / "X_test.csv",
    index=False
)

y_train.to_csv(
    processed_folder / "y_train.csv",
    index=False
)

y_test.to_csv(
    processed_folder / "y_test.csv",
    index=False
)


# --------------------------------------------------
# 10. DISPLAY CHURN DISTRIBUTION
# --------------------------------------------------

print("\n8. TRAINING CHURN DISTRIBUTION")
print("------------------------------")

print(y_train.value_counts())

print("\nTESTING CHURN DISTRIBUTION")
print("--------------------------")

print(y_test.value_counts())


# --------------------------------------------------
# 11. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 50)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 50)

print("Original records:", len(df))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Features:", X_train.shape[1])
print("Missing TotalCharges values retained:", missing_count)

print("\nProcessed datasets saved in:")
print(processed_folder)

print("\nAll preprocessing steps completed!")
