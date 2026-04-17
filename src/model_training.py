"""
Model Training & Evaluation
=============================
Train multiple ML models, compare them, and save the best one.

KEY ML CONCEPTS COVERED:
1. Train/Test Split — why you can't evaluate on training data
2. Multiple algorithms — comparing different approaches
3. Evaluation metrics — accuracy isn't enough for imbalanced data
4. Feature importance — which features drive predictions?
5. Model serialization — saving the model for later use

METRICS WE USE:
- Accuracy: % of all predictions that are correct (can be misleading!)
- Precision: of predicted churners, how many actually churned? (avoid false alarms)
- Recall: of actual churners, how many did we catch? (don't miss real churners)
- F1 Score: harmonic mean of precision & recall (balanced single metric)
- AUC-ROC: probability that model ranks a random churner above a random non-churner
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve
)
from xgboost import XGBClassifier

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


def load_engineered_data():
    """Load the feature-engineered dataset."""
    df = pd.read_csv(os.path.join(DATA_DIR, "engineered_dataset.csv"))
    X = df.drop(columns=["churned"])
    y = df["churned"]
    return X, y


def train_test_split_data(X, y):
    """
    Split data into training (80%) and testing (20%).

    WHY SPLIT? If you train AND evaluate on the same data, the model
    memorizes the answers — like studying with the answer key. The test
    set simulates "unseen future data" to estimate real-world performance.

    stratify=y ensures both sets have the same churn ratio (~28%).
    random_state makes the split reproducible.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape[0]} samples ({y_train.mean():.1%} churn)")
    print(f"Test set:  {X_test.shape[0]} samples ({y_test.mean():.1%} churn)")
    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):
    """
    Train 3 different models. Each sees the same training data.

    WHY MULTIPLE MODELS?
    No single algorithm is best for every problem. By training several
    and comparing results, we find what works best for OUR data.
    """
    models = {}

    # --- Model 1: Logistic Regression ---
    # The simplest classification model. Fits a linear boundary.
    # Good baseline — if this works well, you may not need anything fancier.
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    # class_weight="balanced" handles imbalanced data by upweighting the minority class
    lr.fit(X_train, y_train)
    models["Logistic Regression"] = lr

    # --- Model 2: Random Forest ---
    # Creates 200 random decision trees, each trained on a random subset.
    # Final prediction = majority vote. Handles non-linear patterns well.
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,       # Number of trees
        max_depth=15,           # Max tree depth (prevents overfitting)
        min_samples_split=10,   # Min samples to split a node
        class_weight="balanced",
        random_state=42,
        n_jobs=-1               # Use all CPU cores
    )
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf

    # --- Model 3: XGBoost ---
    # Gradient boosting: builds trees SEQUENTIALLY, each correcting
    # the errors of the previous one. Usually the most accurate.
    print("Training XGBoost...")
    # Calculate scale_pos_weight for imbalanced data
    scale = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,      # How much each tree contributes (lower = more conservative)
        scale_pos_weight=scale, # Handle class imbalance
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    models["XGBoost"] = xgb

    return models


def evaluate_models(models, X_test, y_test):
    """
    Evaluate all models on the held-out test set.
    Returns a comparison DataFrame with all metrics.
    """
    print("\n" + "=" * 60)
    print("MODEL EVALUATION (on test set)")
    print("=" * 60)

    results = []

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # Probability of churn

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "AUC-ROC": roc_auc_score(y_test, y_prob),
        }
        results.append(metrics)

        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

    results_df = pd.DataFrame(results).set_index("Model")
    print("\n=== MODEL COMPARISON ===")
    print(results_df.round(4).to_string())

    return results_df


def plot_results(models, X_test, y_test, results_df, feature_cols):
    """Generate comparison charts: ROC curves, confusion matrices, feature importance."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # --- Plot 1: ROC Curves ---
    ax = axes[0, 0]
    colors_map = {"Logistic Regression": "#3498db", "Random Forest": "#2ecc71", "XGBoost": "#e74c3c"}
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=colors_map[name], linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC=0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves", fontsize=14, fontweight="bold")
    ax.legend()

    # --- Plot 2: Metric Comparison Bar Chart ---
    ax = axes[0, 1]
    results_df.plot(kind="bar", ax=ax, rot=15, edgecolor="black")
    ax.set_title("Model Comparison: All Metrics", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="lower right")

    # --- Plot 3: Best Model Confusion Matrix ---
    best_model_name = results_df["F1 Score"].idxmax()
    best_model = models[best_model_name]
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    ax = axes[1, 0]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Retained", "Churned"],
                yticklabels=["Retained", "Churned"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix: {best_model_name}", fontsize=14, fontweight="bold")

    # --- Plot 4: Top 15 Feature Importance (best model) ---
    ax = axes[1, 1]
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=feature_cols)
    else:
        # Logistic Regression uses coefficients
        importances = pd.Series(np.abs(best_model.coef_[0]), index=feature_cols)

    top_features = importances.nlargest(15)
    top_features.sort_values().plot(kind="barh", ax=ax, color="#3498db", edgecolor="black")
    ax.set_title(f"Top 15 Features: {best_model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance")

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "model_results.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved results plot to: data/model_results.png")


def save_best_model(models, results_df):
    """Save the best performing model (by F1 Score)."""
    best_model_name = results_df["F1 Score"].idxmax()
    best_model = models[best_model_name]

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "best_model.joblib")
    joblib.dump(best_model, model_path)

    # Save metadata about which model was best
    metadata = {
        "model_name": best_model_name,
        "metrics": results_df.loc[best_model_name].to_dict(),
    }
    joblib.dump(metadata, os.path.join(MODEL_DIR, "model_metadata.joblib"))

    print(f"\nBest model: {best_model_name}")
    print(f"Saved to: {model_path}")
    return best_model_name


def run_training():
    """Execute the full training pipeline."""
    print("=" * 60)
    print("MODEL TRAINING PIPELINE")
    print("=" * 60)

    # Load data
    X, y = load_engineered_data()
    feature_cols = list(X.columns)

    # Split
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # Train
    models = train_models(X_train, y_train)

    # Evaluate
    results_df = evaluate_models(models, X_test, y_test)

    # Visualize
    plot_results(models, X_test, y_test, results_df, feature_cols)

    # Save best model
    best_name = save_best_model(models, results_df)

    # Cross-validation for the best model (more robust estimate)
    print(f"\n5-Fold Cross-Validation for {best_name}:")
    cv_scores = cross_val_score(models[best_name], X, y, cv=5, scoring="f1")
    print(f"  F1 scores: {cv_scores.round(4)}")
    print(f"  Mean F1:   {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return models, results_df


if __name__ == "__main__":
    run_training()
