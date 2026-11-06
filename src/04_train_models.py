
import pandas as pd
import joblib

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

project_folder = Path(__file__).resolve().parent.parent

processed_folder = project_folder / "data" / "processed"
models_folder = project_folder / "models"
results_folder = project_folder / "results"

models_folder.mkdir(exist_ok=True)
results_folder.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. LOAD PREPROCESSED DATA
# --------------------------------------------------

print("=" * 60)
print("TELECOM CUSTOMER CHURN - MODEL TRAINING")
print("=" * 60)

X_train = pd.read_csv(processed_folder / "X_train.csv")
X_test = pd.read_csv(processed_folder / "X_test.csv")

y_train = pd.read_csv(processed_folder / "y_train.csv").squeeze()
y_test = pd.read_csv(processed_folder / "y_test.csv").squeeze()

print("\n1. DATA LOADED")
print("--------------")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)


# --------------------------------------------------
# 3. IDENTIFY FEATURE TYPES
# --------------------------------------------------

numerical_columns = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

print("\n2. FEATURE TYPES")
print("----------------")
print("Numerical features:", numerical_columns)
print("Categorical features:", categorical_columns)


# --------------------------------------------------
# 4. CREATE PREPROCESSING PIPELINES
# --------------------------------------------------

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])


# --------------------------------------------------
# 5. DEFINE MACHINE LEARNING MODELS
# --------------------------------------------------

models = {

    "KNN": (
        KNeighborsClassifier(),
        {
            "model__n_neighbors": [5, 7, 11],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2]
        }
    ),

    "Logistic Regression": (
        LogisticRegression(
            max_iter=2000,
            random_state=42
        ),
        {
            "model__C": [0.01, 0.1, 1, 10],
            "model__class_weight": [None, "balanced"]
        }
    ),

    "Random Forest": (
        RandomForestClassifier(
            random_state=42
        ),
        {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__class_weight": [None, "balanced_subsample"]
        }
    )
}


# --------------------------------------------------
# 6. CONFIGURE CROSS-VALIDATION
# --------------------------------------------------

cross_validation = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# --------------------------------------------------
# 7. TRAIN AND TUNE MODELS
# --------------------------------------------------

results = []
best_models = {}

for model_name, (model, parameters) in models.items():

    print("\n" + "=" * 60)
    print("TRAINING:", model_name)
    print("=" * 60)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameters,
        scoring="f1",
        cv=cross_validation,
        n_jobs=-1,
        refit=True,
        verbose=1
    )

    # Hyperparameter tuning using training data only
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Predict using the separate test dataset
    predictions = best_model.predict(X_test)

    probabilities = best_model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test, probabilities
    )

    # Store results
    results.append({
        "Model": model_name,
        "Best CV F1": grid_search.best_score_,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    })

    best_models[model_name] = best_model

    # Display results
    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("\nTest Evaluation:")
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC AUC  :", round(roc_auc, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["No Churn", "Churn"],
            zero_division=0
        )
    )

    # Save trained model
    safe_name = model_name.lower().replace(" ", "_")

    model_path = models_folder / f"{safe_name}_model.pkl"

    joblib.dump(best_model, model_path)

    print("Model saved:", model_path)


# --------------------------------------------------
# 8. CREATE MODEL COMPARISON TABLE
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))


# --------------------------------------------------
# 9. SAVE COMPARISON RESULTS
# --------------------------------------------------

comparison_path = results_folder / "model_comparison.csv"

results_df.to_csv(
    comparison_path,
    index=False
)

print("\nComparison results saved:", comparison_path)


# --------------------------------------------------
# 10. IDENTIFY BEST MODEL
# --------------------------------------------------

best_model_name = results_df.iloc[0]["Model"]

print("\n" + "=" * 60)
print("BEST MODEL BASED ON TEST F1 SCORE")
print("=" * 60)

print("Best model:", best_model_name)
print(
    "F1 Score:",
    round(results_df.iloc[0]["F1 Score"], 4)
)

print("\nMODEL TRAINING COMPLETED SUCCESSFULLY!")
