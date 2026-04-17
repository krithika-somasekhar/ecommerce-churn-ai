"""
Feature Engineering
====================
Transform the clean dataset into ML-ready features.

WHAT IS FEATURE ENGINEERING?
Raw data often isn't the best input for ML models. Feature engineering
creates NEW columns that capture patterns more effectively. For example:
- "total_spend / total_transactions" = spend per transaction (spending intensity)
- "total_tickets / tenure_days" = complaint frequency (normalized by time)

Good feature engineering is often the difference between a mediocre
model and a great one. Domain knowledge drives the best features.

THIS MODULE:
1. Creates derived (computed) features from existing columns
2. Encodes categorical variables (text → numbers, because ML needs numbers)
3. Scales numeric features to similar ranges (important for some algorithms)
4. Returns X (features) and y (target) ready for model training
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import joblib

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


def create_derived_features(df):
    """
    Create NEW features from existing columns. Each feature is designed
    to capture a specific behavioral pattern related to churn.
    """
    df = df.copy()

    # --- Spending behavior ---
    # Spend per transaction: are they making big or small purchases?
    df["spend_per_transaction"] = np.where(
        df["total_transactions"] > 0,
        df["total_spend"] / df["total_transactions"],
        0
    )

    # Spend per day of tenure: how intensely do they use the platform?
    df["spend_per_day"] = np.where(
        df["tenure_days"] > 0,
        df["total_spend"] / df["tenure_days"],
        0
    )

    # --- Engagement intensity ---
    # Transactions per day: purchase frequency normalized by tenure
    df["transactions_per_day"] = np.where(
        df["tenure_days"] > 0,
        df["total_transactions"] / df["tenure_days"],
        0
    )

    # --- Support burden ---
    # Tickets per transaction: what fraction of purchases lead to complaints?
    df["tickets_per_transaction"] = np.where(
        df["total_transactions"] > 0,
        df["total_tickets"] / df["total_transactions"],
        0
    )

    # Complaint frequency: tickets per day of tenure
    df["tickets_per_day"] = np.where(
        df["tenure_days"] > 0,
        df["total_tickets"] / df["tenure_days"],
        0
    )

    # --- Review engagement ---
    # Review rate: what fraction of customers who buy also review?
    df["reviews_per_transaction"] = np.where(
        df["total_transactions"] > 0,
        df["total_reviews"] / df["total_transactions"],
        0
    )

    # Rating gap: difference between average and minimum rating
    # A large gap means inconsistent experience
    df["rating_gap"] = df["avg_rating"] - df["min_rating"]

    # --- Boolean flags ---
    # Has the customer ever filed a ticket?
    df["has_tickets"] = (df["total_tickets"] > 0).astype(int)

    # Has the customer ever left a review?
    df["has_reviews"] = (df["total_reviews"] > 0).astype(int)

    # Is the customer on a paid plan?
    df["is_paid"] = (df["subscription_tier"] != "Free").astype(int)

    print(f"  Created {10} derived features")
    return df


def encode_categoricals(df):
    """
    Convert categorical (text) columns to numbers.

    WHY? ML algorithms work with numbers, not strings.
    We use two techniques:
    1. Label encoding for ordinal categories (Free < Basic < Premium < Enterprise)
    2. One-hot encoding for nominal categories (gender, city — no natural order)
    """
    df = df.copy()

    # Ordinal encoding for subscription tier (there IS a natural order)
    tier_map = {"Free": 0, "Basic": 1, "Premium": 2, "Enterprise": 3}
    df["subscription_tier_encoded"] = df["subscription_tier"].map(tier_map)

    # One-hot encoding for gender (no natural order)
    gender_dummies = pd.get_dummies(df["gender"], prefix="gender", dtype=int)
    df = pd.concat([df, gender_dummies], axis=1)

    # City: too many categories for one-hot. We'll use frequency encoding:
    # replace each city with how often it appears (captures city "size")
    city_freq = df["city"].value_counts(normalize=True)
    df["city_frequency"] = df["city"].map(city_freq)

    print(f"  Encoded categorical features (tier, gender, city)")
    return df


def prepare_model_features(df):
    """
    Final step: select the features for modeling, split into X and y,
    and scale numeric features.

    WHY SCALE? Some algorithms (Logistic Regression, SVM) are sensitive
    to feature magnitude. If 'total_spend' ranges from 0-5000 but 'age'
    ranges from 18-75, the model might weight spend more just because
    of its scale. StandardScaler normalizes all features to mean=0, std=1.

    Note: Tree-based models (Random Forest, XGBoost) don't need scaling,
    but it doesn't hurt them, and it helps Logistic Regression.
    """
    # Features to use in the model
    feature_cols = [
        # Raw features
        "age", "tenure_days", "total_transactions", "total_spend",
        "avg_transaction_amount", "max_transaction_amount", "unique_categories",
        "days_since_last_purchase", "total_tickets", "resolved_tickets",
        "avg_resolution_days", "ticket_resolution_rate", "total_reviews",
        "avg_rating", "min_rating",
        # Derived features
        "spend_per_transaction", "spend_per_day", "transactions_per_day",
        "tickets_per_transaction", "tickets_per_day", "reviews_per_transaction",
        "rating_gap", "has_tickets", "has_reviews", "is_paid",
        # Encoded categoricals
        "subscription_tier_encoded", "city_frequency",
        "gender_Female", "gender_Male", "gender_Non-binary",
    ]

    X = df[feature_cols].copy()
    y = df["churned"].copy()

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=X.index)

    # Save the scaler — we'll need it later to transform new data for predictions
    os.makedirs(MODEL_DIR, exist_ok=True)
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"  Saved scaler to: {scaler_path}")

    # Also save the feature column names for reference
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_cols.joblib"))

    return X_scaled, y, feature_cols


def run_feature_engineering():
    """Execute the full feature engineering pipeline."""
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    # Load clean dataset
    df = pd.read_csv(os.path.join(DATA_DIR, "clean_dataset.csv"))
    print(f"Loaded clean dataset: {df.shape}")

    # Create derived features
    df = create_derived_features(df)

    # Encode categoricals
    df = encode_categoricals(df)

    # Prepare final feature matrix
    X, y, feature_cols = prepare_model_features(df)

    print(f"\nFinal feature matrix: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(f"Features: {feature_cols}")

    # Save the engineered dataset
    engineered = pd.concat([X, y], axis=1)
    engineered.to_csv(os.path.join(DATA_DIR, "engineered_dataset.csv"), index=False)
    print(f"\nSaved engineered dataset to: data/engineered_dataset.csv")

    return X, y, feature_cols


if __name__ == "__main__":
    run_feature_engineering()
