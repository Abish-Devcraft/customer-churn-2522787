import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

# Project paths
project_folder = Path(__file__).resolve().parent.parent

processed_folder = project_folder / "data" / "processed"
models_folder = project_folder / "models"
results_folder = project_folder / "results"

results_folder.mkdir(parents=True, exist_ok=True)

# Load test data
X_test = pd.read_csv(processed_folder / "X_test.csv")
y_test = pd.read_csv(processed_folder / "y_test.csv").squeeze()

# Models to evaluate
model_files = {
    "KNN": "knn_model.pkl",
    "Logistic Regression": "logistic_regression_model.pkl",
    "Random Forest": "random_forest_model.pkl"
}

sns.set_theme(style="whitegrid")

# Store ROC information
plt.figure(figsize=(9, 7))

# Evaluate each model
for model_name, filename in model_files.items():

    model_path = models_folder / filename

    with open(model_path, "rb") as file:
        model = joblib.load(model_path)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{model_name} Confusion Matrix:")
    print(cm)

    tn, fp, fn, tp = cm.ravel()

    print("True Negatives:", tn)
    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("True Positives:", tp)

    plt.figure(figsize=(6, 5))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Churn", "Churn"]
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")

    plt.savefig(
        results_folder / f"{safe_name}_confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(1)
    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {roc_auc:.3f})"
    )

# Final ROC graph
plt.figure(1)
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Model Comparison")
plt.legend()
plt.tight_layout()

plt.savefig(
    results_folder / "roc_curve_comparison.png",
    dpi=300
)

plt.close()

# Model performance comparison
comparison = pd.read_csv(results_folder / "model_comparison.csv")

metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]

comparison_plot = comparison.set_index("Model")[metrics]

comparison_plot.plot(
    kind="bar",
    figsize=(12, 7)
)

plt.title("Performance Comparison of Machine Learning Models")
plt.ylabel("Score")
plt.xlabel("Model")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(
    results_folder / "model_performance_comparison.png",
    dpi=300
)

plt.close()

print("\nMODEL EVALUATION GRAPHS GENERATED SUCCESSFULLY!")
print("Check the results folder.")