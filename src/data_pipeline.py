"""
Data Pipeline: Cleaning, Validation & Transformation
=====================================================
This module takes the raw CSV files and produces a CLEAN, MERGED dataset
ready for analysis and modeling.

In real-world projects, raw data is ALWAYS messy:
- Missing values, wrong types, duplicates, outliers
- Data spread across multiple tables that need joining

This pipeline handles all of that. The output is a single clean DataFrame
where each row is one customer with all their aggregated metrics.

PIPELINE STAGES:
1. Load    — Read raw CSVs
2. Validate — Check data types, ranges, required fields
3. Clean   — Handle missing values, fix types, remove duplicates
4. Merge   — Join all tables into one customer-level dataset
5. Save    — Export the clean dataset
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# Path configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


# ============================================================
# STAGE 1: LOAD RAW DATA
# ============================================================
def load_raw_data():
    """
    Load all CSV files into DataFrames.
    We parse dates at load time — this is a best practice because
    pandas can auto-detect date formats and store them efficiently.
    """
    print("Loading raw data...")

    customers = pd.read_csv(
        os.path.join(DATA_DIR, "customers.csv"),
        parse_dates=["signup_date"]
    )
    transactions = pd.read_csv(
        os.path.join(DATA_DIR, "transactions.csv"),
        parse_dates=["transaction_date"]
    )
    tickets = pd.read_csv(
        os.path.join(DATA_DIR, "support_tickets.csv"),
        parse_dates=["ticket_date"]
    )
    reviews = pd.read_csv(
        os.path.join(DATA_DIR, "reviews.csv"),
        parse_dates=["review_date"]
    )
    churn = pd.read_csv(
        os.path.join(DATA_DIR, "churn_labels.csv"),
        parse_dates=["last_active_date"]
    )

    print(f"  Loaded: {len(customers)} customers, {len(transactions)} transactions, "
          f"{len(tickets)} tickets, {len(reviews)} reviews")

    return customers, transactions, tickets, reviews, churn


# ============================================================
# STAGE 2: VALIDATE DATA QUALITY
# ============================================================
def validate_data(customers, transactions, tickets, reviews, churn):
    """
    Run data quality checks. In production, these would alert your team
    when something is wrong with the data pipeline.

    We check:
    - No duplicate customer IDs
    - All foreign keys (customer_id) reference existing customers
    - Numeric values are in reasonable ranges
    - No null values in critical fields
    """
    print("Validating data quality...")
    issues = []

    # Check 1: No duplicate customer IDs
    dupes = customers["customer_id"].duplicated().sum()
    if dupes > 0:
        issues.append(f"  WARN: {dupes} duplicate customer IDs found")

    # Check 2: All transaction customer_ids exist in customers table
    orphan_txns = ~transactions["customer_id"].isin(customers["customer_id"])
    if orphan_txns.sum() > 0:
        issues.append(f"  WARN: {orphan_txns.sum()} transactions reference non-existent customers")

    # Check 3: No negative transaction amounts
    neg_amounts = (transactions["amount"] < 0).sum()
    if neg_amounts > 0:
        issues.append(f"  WARN: {neg_amounts} negative transaction amounts found")

    # Check 4: Rating values between 1-5
    bad_ratings = ~reviews["rating"].between(1, 5)
    if bad_ratings.sum() > 0:
        issues.append(f"  WARN: {bad_ratings.sum()} reviews with invalid ratings")

    # Check 5: Required fields not null
    for col in ["customer_id", "age", "signup_date"]:
        nulls = customers[col].isnull().sum()
        if nulls > 0:
            issues.append(f"  WARN: {nulls} null values in customers.{col}")

    if issues:
        print("  Data quality issues found:")
        for issue in issues:
            print(issue)
    else:
        print("  All validation checks passed!")

    return len(issues) == 0


# ============================================================
# STAGE 3: CLEAN DATA
# ============================================================
def clean_data(customers, transactions, tickets, reviews, churn):
    """
    Handle data quality issues:
    - Fill missing values with sensible defaults
    - Remove exact duplicate rows
    - Clip outliers to reasonable ranges
    - Ensure correct data types
    """
    print("Cleaning data...")

    # Remove any exact duplicate rows
    customers = customers.drop_duplicates(subset=["customer_id"])
    transactions = transactions.drop_duplicates(subset=["transaction_id"])
    tickets = tickets.drop_duplicates(subset=["ticket_id"])
    reviews = reviews.drop_duplicates(subset=["review_id"])

    # Fill missing resolution_days (unresolved tickets) with -1 as a sentinel value
    # This is better than dropping the rows — we want to know a ticket was NOT resolved
    tickets["resolution_days"] = tickets["resolution_days"].fillna(-1).astype(int)

    # Clip transaction amounts to remove extreme outliers
    # 99.5th percentile cap — keeps 99.5% of data, removes extreme outliers
    upper_bound = transactions["amount"].quantile(0.995)
    transactions["amount"] = transactions["amount"].clip(upper=upper_bound)

    # Ensure age is in a valid range
    customers["age"] = customers["age"].clip(lower=18, upper=100)

    print(f"  Cleaned: {len(customers)} customers, {len(transactions)} transactions")
    return customers, transactions, tickets, reviews, churn


# ============================================================
# STAGE 4: MERGE INTO CUSTOMER-LEVEL DATASET
# ============================================================
def create_customer_dataset(customers, transactions, tickets, reviews, churn):
    """
    This is the KEY transformation step. We go from multiple tables to
    ONE row per customer with aggregated features.

    For each customer, we compute:
    - Transaction metrics: count, total spend, avg spend, days since last purchase
    - Support metrics: ticket count, resolution rate, avg resolution time
    - Review metrics: count, average rating
    - Account info: tenure (days since signup), subscription tier

    This "flattened" format is what ML models need — one row per observation
    with all features as columns.
    """
    print("Creating customer-level dataset...")

    reference_date = datetime(2025, 1, 1)  # Fixed reference date for consistency

    # --- Transaction aggregations ---
    txn_agg = transactions.groupby("customer_id").agg(
        total_transactions=("transaction_id", "count"),
        total_spend=("amount", "sum"),
        avg_transaction_amount=("amount", "mean"),
        max_transaction_amount=("amount", "max"),
        unique_categories=("category", "nunique"),
        last_transaction_date=("transaction_date", "max"),
    ).reset_index()

    txn_agg["days_since_last_purchase"] = (
        reference_date - txn_agg["last_transaction_date"]
    ).dt.days
    txn_agg = txn_agg.drop(columns=["last_transaction_date"])

    # --- Support ticket aggregations ---
    tkt_agg = tickets.groupby("customer_id").agg(
        total_tickets=("ticket_id", "count"),
        resolved_tickets=("status", lambda x: (x == "Resolved").sum()),
        avg_resolution_days=("resolution_days", lambda x: x[x >= 0].mean()),
    ).reset_index()

    tkt_agg["ticket_resolution_rate"] = (
        tkt_agg["resolved_tickets"] / tkt_agg["total_tickets"]
    ).round(3)
    tkt_agg["avg_resolution_days"] = tkt_agg["avg_resolution_days"].fillna(0).round(1)

    # --- Review aggregations ---
    rev_agg = reviews.groupby("customer_id").agg(
        total_reviews=("review_id", "count"),
        avg_rating=("rating", "mean"),
        min_rating=("rating", "min"),
    ).reset_index()

    rev_agg["avg_rating"] = rev_agg["avg_rating"].round(2)

    # --- Merge everything together ---
    # Start with customers, then LEFT JOIN each aggregation
    # LEFT JOIN ensures we keep ALL customers, even those with no transactions
    dataset = customers.copy()

    # Add customer tenure (how long they've been a customer)
    dataset["tenure_days"] = (reference_date - dataset["signup_date"]).dt.days

    dataset = dataset.merge(txn_agg, on="customer_id", how="left")
    dataset = dataset.merge(tkt_agg, on="customer_id", how="left")
    dataset = dataset.merge(rev_agg, on="customer_id", how="left")
    dataset = dataset.merge(churn[["customer_id", "churned"]], on="customer_id", how="left")

    # Fill NaN for customers who had no transactions/tickets/reviews
    fill_zero_cols = [
        "total_transactions", "total_spend", "avg_transaction_amount",
        "max_transaction_amount", "unique_categories", "days_since_last_purchase",
        "total_tickets", "resolved_tickets", "avg_resolution_days",
        "ticket_resolution_rate", "total_reviews", "avg_rating", "min_rating"
    ]
    for col in fill_zero_cols:
        dataset[col] = dataset[col].fillna(0)

    # Drop columns not needed for modeling
    dataset = dataset.drop(columns=["signup_date"])

    print(f"  Final dataset: {dataset.shape[0]} rows x {dataset.shape[1]} columns")
    return dataset


# ============================================================
# STAGE 5: SAVE CLEAN DATASET
# ============================================================
def save_clean_data(dataset):
    """Save the merged, clean dataset for modeling."""
    output_path = os.path.join(DATA_DIR, "clean_dataset.csv")
    dataset.to_csv(output_path, index=False)
    print(f"  Saved clean dataset to: {output_path}")
    return output_path


# ============================================================
# RUN THE FULL PIPELINE
# ============================================================
def run_pipeline():
    """Execute all pipeline stages in order."""
    print("=" * 60)
    print("DATA PIPELINE")
    print("=" * 60)

    # Stage 1: Load
    customers, transactions, tickets, reviews, churn = load_raw_data()

    # Stage 2: Validate
    validate_data(customers, transactions, tickets, reviews, churn)

    # Stage 3: Clean
    customers, transactions, tickets, reviews, churn = clean_data(
        customers, transactions, tickets, reviews, churn
    )

    # Stage 4: Merge
    dataset = create_customer_dataset(customers, transactions, tickets, reviews, churn)

    # Stage 5: Save
    save_clean_data(dataset)

    # Print a preview
    print("\nDataset preview:")
    print(dataset.head(3).to_string(index=False))
    print(f"\nColumn types:\n{dataset.dtypes.to_string()}")

    return dataset


if __name__ == "__main__":
    run_pipeline()
