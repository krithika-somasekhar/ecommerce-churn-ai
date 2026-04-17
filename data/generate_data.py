"""
Synthetic Data Generator for E-Commerce Churn Prediction
=========================================================
This script generates realistic fake data for our project. We create 5 CSV files:
1. customers.csv     - Customer profiles (demographics, subscription info)
2. transactions.csv  - Purchase history
3. support_tickets.csv - Customer service interactions
4. reviews.csv       - Product reviews with text (for AI sentiment analysis)
5. churn_labels.csv  - Whether each customer churned (our prediction target)

WHY SYNTHETIC DATA?
- We control the patterns, so we know what the model should find
- No privacy concerns — this data isn't real
- We can make it as large or complex as we want
- Generating data forces you to think about data modeling

KEY INSIGHT: We build in realistic correlations. Customers who churn tend to:
- Have fewer transactions and lower spend
- File more support tickets
- Leave more negative reviews
This mimics real-world behavior, giving our ML models signal to learn from.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Set seed for reproducibility — anyone running this gets the same data
np.random.seed(42)
random.seed(42)

# ============================================================
# CONFIGURATION
# ============================================================
NUM_CUSTOMERS = 5000
DATE_START = datetime(2023, 1, 1)
DATE_END = datetime(2024, 12, 31)

# Where to save the generated files
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# HELPER: Generate random dates within a range
# ============================================================
def random_dates(start, end, n):
    """Generate n random dates between start and end."""
    delta = (end - start).days
    return [start + timedelta(days=random.randint(0, delta)) for _ in range(n)]


# ============================================================
# 1. GENERATE CUSTOMERS
# ============================================================
def generate_customers(n):
    """
    Create customer profiles. Each customer gets:
    - A unique ID
    - Demographics (age, gender, location)
    - Account info (signup date, subscription tier)

    We also secretly assign a 'churn probability' to each customer
    based on their profile. This drives all downstream data generation.
    """
    print(f"Generating {n} customers...")

    # Subscription tiers — higher tiers have lower churn (they're more invested)
    tiers = ["Free", "Basic", "Premium", "Enterprise"]
    tier_weights = [0.30, 0.35, 0.25, 0.10]  # Distribution across tiers

    # Base churn probability per tier (Free users churn most)
    tier_churn_base = {"Free": 0.45, "Basic": 0.25, "Premium": 0.15, "Enterprise": 0.08}

    cities = [
        "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
        "Philadelphia", "San Antonio", "San Diego", "Dallas", "Austin",
        "Seattle", "Denver", "Boston", "Nashville", "Portland"
    ]

    customers = []
    for i in range(n):
        customer_id = f"CUST_{i+1:05d}"
        age = int(np.clip(np.random.normal(38, 12), 18, 75))
        gender = np.random.choice(["Male", "Female", "Non-binary"], p=[0.48, 0.48, 0.04])
        city = random.choice(cities)
        signup_date = random_dates(DATE_START, DATE_END - timedelta(days=180), 1)[0]
        tier = np.random.choice(tiers, p=tier_weights)

        # Calculate churn probability: base rate + age adjustment + noise
        # Younger and older customers churn slightly more (U-shaped curve)
        age_factor = 0.05 * abs(age - 35) / 35
        churn_prob = np.clip(tier_churn_base[tier] + age_factor + np.random.normal(0, 0.05), 0.02, 0.85)

        customers.append({
            "customer_id": customer_id,
            "age": age,
            "gender": gender,
            "city": city,
            "signup_date": signup_date.strftime("%Y-%m-%d"),
            "subscription_tier": tier,
            "_churn_prob": churn_prob  # Hidden column — used for generation, removed from final data
        })

    return pd.DataFrame(customers)


# ============================================================
# 2. GENERATE TRANSACTIONS
# ============================================================
def generate_transactions(customers_df):
    """
    Generate purchase records. Customers with HIGH churn probability
    get FEWER transactions and LOWER average spend. This is the core
    signal our ML model will learn.
    """
    print("Generating transactions...")

    categories = ["Electronics", "Clothing", "Home & Garden", "Books", "Food & Beverage",
                  "Health & Beauty", "Sports", "Toys", "Automotive", "Office Supplies"]

    transactions = []
    for _, cust in customers_df.iterrows():
        # Churny customers buy less frequently
        # A loyal customer might make 15-25 purchases; a churny one might make 3-8
        avg_transactions = int(np.clip(20 * (1 - cust["_churn_prob"]) + np.random.normal(0, 3), 1, 40))

        signup = datetime.strptime(cust["signup_date"], "%Y-%m-%d")

        for _ in range(avg_transactions):
            tx_date = random_dates(signup, DATE_END, 1)[0]

            # Churny customers spend less per transaction
            base_amount = np.random.lognormal(mean=3.5, sigma=0.8)  # ~$30 median
            loyalty_multiplier = 1.0 + 0.5 * (1 - cust["_churn_prob"])
            amount = round(base_amount * loyalty_multiplier, 2)

            transactions.append({
                "transaction_id": f"TXN_{len(transactions)+1:07d}",
                "customer_id": cust["customer_id"],
                "transaction_date": tx_date.strftime("%Y-%m-%d"),
                "amount": amount,
                "category": random.choice(categories),
                "quantity": random.randint(1, 5)
            })

    return pd.DataFrame(transactions)


# ============================================================
# 3. GENERATE SUPPORT TICKETS
# ============================================================
def generate_support_tickets(customers_df):
    """
    Customers with high churn probability file MORE support tickets
    and have LOWER resolution rates. Unhappy customers complain more!
    """
    print("Generating support tickets...")

    issue_types = ["Billing", "Product Quality", "Shipping Delay", "Account Issue",
                   "Return Request", "Technical Problem", "General Inquiry"]

    # Weights: churny customers get more "negative" issue types
    issue_weights_normal = [0.10, 0.10, 0.10, 0.10, 0.15, 0.15, 0.30]
    issue_weights_churny = [0.20, 0.25, 0.20, 0.15, 0.10, 0.05, 0.05]

    tickets = []
    for _, cust in customers_df.iterrows():
        # High churn prob → more tickets (1-8 tickets vs 0-2 for happy customers)
        avg_tickets = int(np.clip(8 * cust["_churn_prob"] + np.random.normal(0, 1), 0, 12))

        weights = issue_weights_churny if cust["_churn_prob"] > 0.35 else issue_weights_normal
        signup = datetime.strptime(cust["signup_date"], "%Y-%m-%d")

        for _ in range(avg_tickets):
            ticket_date = random_dates(signup, DATE_END, 1)[0]
            issue = np.random.choice(issue_types, p=weights)

            # Churny customers' tickets are less likely to be resolved
            resolved = np.random.random() > (cust["_churn_prob"] * 0.6)

            tickets.append({
                "ticket_id": f"TKT_{len(tickets)+1:07d}",
                "customer_id": cust["customer_id"],
                "ticket_date": ticket_date.strftime("%Y-%m-%d"),
                "issue_type": issue,
                "status": "Resolved" if resolved else random.choice(["Open", "Pending"]),
                "resolution_days": random.randint(1, 7) if resolved else None
            })

    return pd.DataFrame(tickets)


# ============================================================
# 4. GENERATE REVIEWS
# ============================================================
def generate_reviews(customers_df):
    """
    Product reviews with REALISTIC TEXT. This is crucial for our AI features:
    - Claude will analyze sentiment from the review text
    - The AI explainer will reference review themes

    Churny customers leave more negative reviews (lower stars, angrier text).
    We use template-based generation with randomized components for variety.
    """
    print("Generating reviews...")

    # Review text templates grouped by sentiment
    positive_reviews = [
        "Absolutely love this product! {reason}. Would definitely recommend to friends and family.",
        "Great quality for the price. {reason}. Very satisfied with my purchase.",
        "Exceeded my expectations! {reason}. Will be buying again soon.",
        "Perfect! Exactly what I was looking for. {reason}. Five stars all the way.",
        "Impressive quality and fast shipping. {reason}. Couldn't be happier.",
        "This is my third purchase and it never disappoints. {reason}. Loyal customer here!",
        "Best purchase I've made this year. {reason}. Worth every penny.",
        "Amazing product, amazing service. {reason}. Keep up the great work!",
    ]

    neutral_reviews = [
        "Decent product overall. {reason}. Not bad but nothing special either.",
        "It's okay for what you pay. {reason}. Might look for alternatives next time.",
        "Average quality. {reason}. Serves its purpose but could be improved.",
        "Mixed feelings about this one. {reason}. Some good aspects, some not so much.",
        "Does the job but {reason}. Expected a bit more for the price.",
        "Solid enough product. {reason}. No major complaints but room for improvement.",
    ]

    negative_reviews = [
        "Very disappointed with this purchase. {reason}. Would not recommend.",
        "Poor quality product. {reason}. Waste of money, returning immediately.",
        "Terrible experience. {reason}. Customer service was unhelpful when I complained.",
        "Not as described at all. {reason}. Feel completely ripped off.",
        "Broke after a week of use. {reason}. Cheaply made, avoid this product.",
        "Worst purchase I've ever made. {reason}. Demanding a full refund.",
        "Hugely disappointing. {reason}. The product photos are misleading.",
        "Complete garbage. {reason}. I've wasted so much time dealing with returns.",
    ]

    positive_reasons = [
        "The material feels premium", "Works exactly as advertised",
        "The design is sleek and modern", "Easy to set up and use",
        "Customer support was super helpful", "Arrived earlier than expected",
        "Better than the more expensive alternatives", "My whole family loves it",
    ]

    neutral_reasons = [
        "The packaging was nice at least", "It works but feels flimsy",
        "The color was slightly different from the photo", "Took a while to arrive",
        "Instructions could be clearer", "Size runs a bit small",
    ]

    negative_reasons = [
        "The stitching came apart immediately", "It doesn't match the description at all",
        "Arrived damaged with no protective packaging", "The battery life is terrible",
        "It stopped working after just a few uses", "Missing parts in the box",
        "The material feels cheap and flimsy", "Waited 3 weeks for delivery",
    ]

    products = [
        "Wireless Headphones", "Running Shoes", "Coffee Maker", "Laptop Stand",
        "Yoga Mat", "Backpack", "Smart Watch", "Water Bottle", "Desk Lamp",
        "Phone Case", "Bluetooth Speaker", "Kitchen Scale", "Sunglasses",
        "Portable Charger", "Notebook Set"
    ]

    reviews = []
    for _, cust in customers_df.iterrows():
        # Customers write 0-5 reviews each
        num_reviews = random.randint(0, 5)
        signup = datetime.strptime(cust["signup_date"], "%Y-%m-%d")

        for _ in range(num_reviews):
            review_date = random_dates(signup, DATE_END, 1)[0]

            # Star rating influenced by churn probability
            # Churny customers: skewed toward 1-3 stars
            # Happy customers: skewed toward 3-5 stars
            if cust["_churn_prob"] > 0.4:
                stars = np.random.choice([1, 2, 3, 4, 5], p=[0.30, 0.30, 0.25, 0.10, 0.05])
            elif cust["_churn_prob"] > 0.2:
                stars = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.15, 0.30, 0.30, 0.20])
            else:
                stars = np.random.choice([1, 2, 3, 4, 5], p=[0.02, 0.05, 0.13, 0.35, 0.45])

            # Select review text based on stars
            if stars >= 4:
                template = random.choice(positive_reviews)
                reason = random.choice(positive_reasons)
            elif stars == 3:
                template = random.choice(neutral_reviews)
                reason = random.choice(neutral_reasons)
            else:
                template = random.choice(negative_reviews)
                reason = random.choice(negative_reasons)

            review_text = template.format(reason=reason)

            reviews.append({
                "review_id": f"REV_{len(reviews)+1:07d}",
                "customer_id": cust["customer_id"],
                "product": random.choice(products),
                "rating": stars,
                "review_text": review_text,
                "review_date": review_date.strftime("%Y-%m-%d"),
            })

    return pd.DataFrame(reviews)


# ============================================================
# 5. GENERATE CHURN LABELS
# ============================================================
def generate_churn_labels(customers_df):
    """
    The final step: for each customer, flip a coin weighted by their
    churn probability. This is our PREDICTION TARGET — what the ML
    model will try to predict.

    We also record the last_active_date. Churned customers have an
    earlier last active date (they stopped using the platform).
    """
    print("Generating churn labels...")

    labels = []
    for _, cust in customers_df.iterrows():
        churned = int(np.random.random() < cust["_churn_prob"])

        # Churned customers were last active earlier
        if churned:
            last_active = random_dates(
                datetime.strptime(cust["signup_date"], "%Y-%m-%d"),
                DATE_END - timedelta(days=90),
                1
            )[0]
        else:
            last_active = random_dates(
                DATE_END - timedelta(days=30),
                DATE_END,
                1
            )[0]

        labels.append({
            "customer_id": cust["customer_id"],
            "churned": churned,
            "last_active_date": last_active.strftime("%Y-%m-%d"),
        })

    return pd.DataFrame(labels)


# ============================================================
# MAIN: Generate all data and save to CSV
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("E-Commerce Synthetic Data Generator")
    print("=" * 60)

    # Generate all datasets
    customers_df = generate_customers(NUM_CUSTOMERS)
    transactions_df = generate_transactions(customers_df)
    tickets_df = generate_support_tickets(customers_df)
    reviews_df = generate_reviews(customers_df)
    churn_df = generate_churn_labels(customers_df)

    # Remove the hidden churn probability column before saving
    customers_public = customers_df.drop(columns=["_churn_prob"])

    # Save to CSV
    customers_public.to_csv(os.path.join(OUTPUT_DIR, "customers.csv"), index=False)
    transactions_df.to_csv(os.path.join(OUTPUT_DIR, "transactions.csv"), index=False)
    tickets_df.to_csv(os.path.join(OUTPUT_DIR, "support_tickets.csv"), index=False)
    reviews_df.to_csv(os.path.join(OUTPUT_DIR, "reviews.csv"), index=False)
    churn_df.to_csv(os.path.join(OUTPUT_DIR, "churn_labels.csv"), index=False)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Customers:       {len(customers_public):,}")
    print(f"  Transactions:    {len(transactions_df):,}")
    print(f"  Support Tickets: {len(tickets_df):,}")
    print(f"  Reviews:         {len(reviews_df):,}")
    print(f"  Churn Rate:      {churn_df['churned'].mean():.1%}")
    print(f"\nFiles saved to: {OUTPUT_DIR}/")
    print("=" * 60)
