# E-Commerce Customer Churn Prediction & AI Analytics Platform
## A Complete End-to-End Project Guide

---

**Author:** Krithika
**Project Type:** AI + Machine Learning + Analytics
**Tech Stack:** Python, pandas, scikit-learn, XGBoost, Claude AI, FastAPI, Streamlit, Docker
**Total Files:** 11 source files | ~2,000 lines of code | 96,000+ data points

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [Why This Project?](#2-why-this-project)
3. [Architecture Overview](#3-architecture-overview)
4. [Step 1 — Project Setup](#step-1--project-setup)
5. [Step 2 — Synthetic Data Generation](#step-2--synthetic-data-generation)
6. [Step 3 — Data Pipeline (Clean, Validate, Merge)](#step-3--data-pipeline)
7. [Step 4 — Exploratory Data Analysis (EDA)](#step-4--exploratory-data-analysis)
8. [Step 5 — Feature Engineering](#step-5--feature-engineering)
9. [Step 6 — ML Model Training & Evaluation](#step-6--ml-model-training--evaluation)
10. [Step 7 — AI: Sentiment Analysis with Claude](#step-7--ai-sentiment-analysis)
11. [Step 8 — AI: Natural Language Data Querying](#step-8--ai-natural-language-querying)
12. [Step 9 — AI: Churn Explainer](#step-9--ai-churn-explainer)
13. [Step 10 — FastAPI Prediction Server](#step-10--fastapi-prediction-server)
14. [Step 11 — Streamlit Dashboard](#step-11--streamlit-dashboard)
15. [Step 12 — Docker & Deployment](#step-12--docker--deployment)
16. [Results & Key Findings](#results--key-findings)
17. [Concepts Cheat Sheet](#concepts-cheat-sheet)
18. [How to Run the Project](#how-to-run)
19. [What I Would Improve in Production](#production-improvements)
20. [Interview Talking Points](#interview-talking-points)

---

## 1. What Is This Project?

This is an **end-to-end AI and Analytics platform** that solves a real business problem: **customer churn prediction** for an e-commerce company.

**The business problem:** An e-commerce company is losing customers. They need to:
- **Predict** which customers are about to leave (churn)
- **Understand** why those customers are leaving
- **Act** on those insights to retain them

**What we built to solve it:**

| Component | What It Does |
|-----------|-------------|
| **Data Pipeline** | Takes raw customer data across 5 tables, cleans it, validates it, and merges it into one analysis-ready dataset |
| **Analytics Dashboard** | Visualizes customer behavior patterns, churn rates by segment, and key metrics |
| **ML Models** | Trains 3 different algorithms to predict churn, compares them, and deploys the best one |
| **AI Sentiment Analysis** | Uses Claude (LLM) to read customer reviews and extract sentiment, themes, and churn risk |
| **AI Data Querying** | Lets you ask questions about the data in plain English — Claude writes the code and returns answers |
| **AI Churn Explainer** | Takes a model's prediction and generates a human-readable explanation of WHY a customer might churn |
| **REST API** | Serves predictions and AI insights via HTTP endpoints that any application can call |
| **Web Dashboard** | An interactive Streamlit app that brings everything together — charts, predictions, and AI chat |

---

## 2. Why This Project?

**Why churn prediction?**
- It's one of the **top 3 most common ML use cases** in industry (alongside fraud detection and recommendation systems)
- It naturally requires **every step** of the ML lifecycle — data engineering, analysis, modeling, serving, and visualization
- It has clear **business impact** — reducing churn by even 5% can increase profits by 25-95% (Harvard Business Review)

**Why add AI/LLM features?**
- Traditional ML gives you a number (73% churn probability) but doesn't explain **why**
- Customer reviews contain rich qualitative data that ML models can't process — but LLMs can
- Natural language querying democratizes data access — non-technical stakeholders can ask questions without writing code

**What makes this portfolio-worthy:**
- Covers the complete lifecycle from raw data to deployed application
- Combines classical ML with modern AI (LLMs)
- Demonstrates both technical depth and business thinking
- Every component is production-ready with proper error handling and documentation

---

## 3. Architecture Overview

Here's how all the pieces fit together:

```
RAW DATA (5 CSVs)
    │
    ▼
DATA PIPELINE (clean, validate, merge)
    │
    ▼
CLEAN DATASET (5,000 customers × 20 columns)
    │
    ├──────────────────────┐
    ▼                      ▼
FEATURE ENGINEERING     EDA NOTEBOOK
(30 ML-ready features)  (visualizations & insights)
    │
    ▼
ML MODEL TRAINING
(Logistic Regression, Random Forest, XGBoost)
    │
    ├───────────────┬──────────────────┐
    ▼               ▼                  ▼
FastAPI API     Claude AI          Streamlit Dashboard
(7 endpoints)   • Sentiment         (4 interactive pages)
                • NL Querying
                • Explainer
```

**Project folder structure:**

```
ecommerce-churn-ai/
├── data/                          # All datasets
│   ├── generate_data.py              # Script to create synthetic data
│   ├── customers.csv                 # 5,000 customer profiles
│   ├── transactions.csv              # 69,295 purchase records
│   ├── support_tickets.csv           # 9,237 support interactions
│   ├── reviews.csv                   # 12,421 product reviews with text
│   ├── churn_labels.csv              # Churn labels (our prediction target)
│   ├── clean_dataset.csv             # Pipeline output: merged & clean
│   └── engineered_dataset.csv        # Feature-engineered dataset
├── notebooks/
│   └── eda.ipynb                     # Exploratory Data Analysis notebook
├── src/
│   ├── data_pipeline.py              # Data cleaning & merging logic
│   ├── feature_engineering.py        # Feature creation, encoding, scaling
│   ├── model_training.py             # Train, evaluate, compare 3 models
│   ├── ai_insights.py                # Claude AI integration (3 features)
│   └── api.py                        # FastAPI server for predictions
├── dashboard/
│   └── app.py                        # Streamlit web dashboard
├── models/                           # Saved model artifacts
│   ├── best_model.joblib                # Trained model file
│   ├── scaler.joblib                    # Feature scaler
│   ├── feature_cols.joblib              # Feature column names
│   └── model_metadata.joblib            # Model metrics
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Container configuration
├── start.sh                          # Multi-service startup script
├── .env.example                      # API key template
└── README.md                         # Project documentation
```

---

## Step 1 — Project Setup

**File:** `requirements.txt`
**Purpose:** Define all dependencies with pinned versions for reproducibility.

### What We Set Up

1. **Virtual Environment (`venv/`)** — An isolated Python installation just for this project. Without it, installing packages could break other projects on your machine.

2. **Dependencies (`requirements.txt`)** — Every Python package we need, with specific version numbers:

| Category | Packages | Why We Need Them |
|----------|----------|-----------------|
| Data Processing | pandas, numpy | DataFrame manipulation, math operations |
| Machine Learning | scikit-learn, xgboost, shap | Model training, evaluation, explainability |
| Visualization | matplotlib, seaborn, plotly | Static charts, statistical plots, interactive charts |
| AI / LLM | anthropic | Claude API client for AI features |
| API | fastapi, uvicorn, pydantic | REST API server, ASGI server, request validation |
| Dashboard | streamlit | Turn Python scripts into web apps |
| Notebook | jupyter, ipykernel | Interactive data analysis |
| Utilities | python-dotenv, joblib | Load .env files, save/load Python objects |

3. **`.gitignore`** — Tells Git to ignore sensitive files (`.env` with API keys), cache files (`__pycache__`), and large binary files (trained models).

4. **`.env.example`** — Template for the API key file. The actual `.env` is never committed to Git (it's in `.gitignore`).

### Key Concept: Why Pin Versions?

If you install `pandas` without a version, pip grabs the latest. But the latest version next month might break your code. Pinning (e.g., `pandas==2.2.3`) ensures anyone who runs your project gets exactly the same behavior, even years later. This is called **reproducibility**.

### Commands to Run

```bash
cd ecommerce-churn-ai
python3 -m venv venv           # Create virtual environment
source venv/bin/activate       # Activate it (macOS/Linux)
pip install -r requirements.txt  # Install all packages
```

---

## Step 2 — Synthetic Data Generation

**File:** `data/generate_data.py`
**Purpose:** Create realistic fake e-commerce data with built-in behavioral patterns.

### Why Synthetic Data?

- We control the patterns, so we know what the model should find
- No privacy concerns — this data isn't real
- Forces us to think about data modeling and relationships
- Common practice in industry for testing and prototyping

### The 5 Tables We Generate

#### Table 1: `customers.csv` (5,000 rows)

| Column | Type | Description |
|--------|------|-------------|
| customer_id | String | Unique ID (e.g., CUST_00001) |
| age | Integer | Customer age (18–75, normally distributed around 38) |
| gender | String | Male, Female, or Non-binary |
| city | String | One of 15 US cities |
| signup_date | Date | When they joined (2023-01-01 to 2024-06-30) |
| subscription_tier | String | Free (30%), Basic (35%), Premium (25%), Enterprise (10%) |

**The secret sauce:** Each customer gets a hidden `_churn_prob` value based on their tier and age. This probability drives ALL downstream data — making the data realistically correlated rather than random.

```
Tier churn base rates:
  Free:       45% base churn probability
  Basic:      25%
  Premium:    15%
  Enterprise:  8%
```

#### Table 2: `transactions.csv` (69,295 rows)

| Column | Type | Description |
|--------|------|-------------|
| transaction_id | String | Unique ID |
| customer_id | String | Links to customers table |
| transaction_date | Date | When the purchase happened |
| amount | Float | Purchase amount in dollars |
| category | String | Product category (10 categories) |
| quantity | Integer | Items purchased (1-5) |

**Key pattern:** Customers with high churn probability make **fewer transactions** and spend **less per transaction**. A loyal customer makes 15-25 purchases; a churny one makes 3-8.

#### Table 3: `support_tickets.csv` (9,237 rows)

| Column | Type | Description |
|--------|------|-------------|
| ticket_id | String | Unique ID |
| customer_id | String | Links to customers table |
| ticket_date | Date | When the ticket was filed |
| issue_type | String | Billing, Product Quality, Shipping Delay, etc. |
| status | String | Resolved, Open, or Pending |
| resolution_days | Integer | Days to resolve (null if unresolved) |

**Key pattern:** Churny customers file **more tickets** (1-8 vs 0-2 for happy customers) and have **lower resolution rates**. They also get more "negative" issue types (Product Quality, Billing) vs happy customers who ask General Inquiries.

#### Table 4: `reviews.csv` (12,421 rows)

| Column | Type | Description |
|--------|------|-------------|
| review_id | String | Unique ID |
| customer_id | String | Links to customers table |
| product | String | Which product was reviewed (15 products) |
| rating | Integer | 1-5 star rating |
| review_text | String | Free-text review (realistic sentences) |
| review_date | Date | When the review was posted |

**Key pattern:** Churny customers give **lower ratings** (skewed toward 1-3 stars) and write **negative review text** ("Terrible experience," "Broke after a week"). This free text is what our AI sentiment analysis will process.

**Review text generation:** We use template-based generation with 8 positive, 6 neutral, and 8 negative review templates, each with randomized "reason" phrases. This creates varied but realistic text:

```
Positive: "Absolutely love this product! {The material feels premium}. 
           Would definitely recommend to friends and family."

Negative: "Terrible experience. {The stitching came apart immediately}. 
           Customer service was unhelpful when I complained."
```

#### Table 5: `churn_labels.csv` (5,000 rows)

| Column | Type | Description |
|--------|------|-------------|
| customer_id | String | Links to customers table |
| churned | Integer | 1 = churned, 0 = retained |
| last_active_date | Date | Last time they used the platform |

**How churn is determined:** For each customer, we flip a weighted coin using their `_churn_prob`. A customer with 45% churn probability has a 45% chance of being labeled as churned. This adds realistic noise — not every Free tier customer churns.

**Result: 27.9% churn rate** (1,395 out of 5,000 customers). This is a moderately imbalanced dataset — realistic and manageable.

### Running the Generator

```bash
python data/generate_data.py
```

Output:
```
Customers:       5,000
Transactions:    69,295
Support Tickets: 9,237
Reviews:         12,421
Churn Rate:      27.9%
```

---

## Step 3 — Data Pipeline

**File:** `src/data_pipeline.py`
**Purpose:** Transform raw CSVs into a clean, merged, analysis-ready dataset.

### Why Do We Need a Pipeline?

Even with synthetic data, real-world projects ALWAYS need data cleaning. In production, data arrives with:
- Missing values
- Wrong types (dates stored as strings, numbers stored as text)
- Duplicates
- Outliers (someone bought $999,999 worth of product? Probably a bug)
- Data spread across multiple tables that need joining

Our pipeline handles all of this in 5 stages.

### Stage 1: Load Raw Data

```python
customers = pd.read_csv("customers.csv", parse_dates=["signup_date"])
```

**Key detail:** We use `parse_dates` at load time. This tells pandas to convert the date column from strings to actual datetime objects. This is important because:
- You can do date math (e.g., "days since signup")
- Sorting works correctly (string sort: "2024-01-10" comes before "2024-01-9")
- Memory is more efficient

### Stage 2: Validate Data Quality

We run 5 automated checks:

| Check | What It Catches | Why It Matters |
|-------|----------------|----------------|
| Duplicate customer IDs | Data generation or ingestion bugs | Would double-count customers |
| Orphan transactions | Transactions referencing deleted customers | Would crash JOIN operations |
| Negative amounts | Data entry errors or refund bugs | Would distort spend calculations |
| Invalid ratings | Ratings outside 1-5 range | Would break aggregations |
| Null required fields | Missing customer_id, age, or signup_date | Would crash downstream code |

**In production,** these checks would trigger alerts to your data engineering team. Here, they give us confidence the data is clean before proceeding.

### Stage 3: Clean Data

Three cleaning operations:

1. **Remove duplicates** — `drop_duplicates(subset=["customer_id"])` keeps only the first occurrence of each ID.

2. **Fill missing values** — Unresolved tickets have `NULL` for `resolution_days`. We fill this with `-1` as a **sentinel value** (a special value meaning "not applicable"). This is better than:
   - Dropping the row (we'd lose data about the unresolved ticket)
   - Filling with 0 (misleading — 0 days suggests instant resolution)
   - Filling with the mean (incorrect — it wasn't resolved at all)

3. **Clip outliers** — We cap transaction amounts at the 99.5th percentile. If the 99.5th percentile is $500, any transaction above $500 becomes $500. This prevents extreme outliers from distorting averages and model training.

### Stage 4: Merge Into Customer-Level Dataset (THE KEY STEP)

This is the most important transformation. We go from **5 separate tables** to **1 row per customer** with all their metrics aggregated.

**Why?** ML models need a flat table — one row per observation (customer) with all features as columns. You can't feed a model 5 separate CSVs.

**How?** We use `groupby().agg()` to compute summary statistics for each customer:

**Transaction aggregations:**
```
total_transactions    = COUNT of transactions
total_spend           = SUM of amounts
avg_transaction_amount = MEAN of amounts
max_transaction_amount = MAX of amounts
unique_categories     = COUNT DISTINCT categories
days_since_last_purchase = days from last transaction to reference date
```

**Support ticket aggregations:**
```
total_tickets         = COUNT of tickets
resolved_tickets      = COUNT where status = "Resolved"
avg_resolution_days   = MEAN of resolution_days (for resolved tickets)
ticket_resolution_rate = resolved_tickets / total_tickets
```

**Review aggregations:**
```
total_reviews  = COUNT of reviews
avg_rating     = MEAN of ratings
min_rating     = MIN of ratings
```

**Then we merge everything using LEFT JOINs:**

```python
dataset = customers.merge(txn_agg, on="customer_id", how="left")
                   .merge(tkt_agg, on="customer_id", how="left")
                   .merge(rev_agg, on="customer_id", how="left")
                   .merge(churn,   on="customer_id", how="left")
```

**Why LEFT JOIN?** We want to keep ALL customers, even those with zero transactions or zero tickets. An INNER JOIN would drop customers who never purchased — but those are actually important (they might be at high churn risk).

After joining, we fill `NaN` values with 0 for customers who had no activity.

### Stage 5: Save

Output: `data/clean_dataset.csv` — **5,000 rows × 20 columns**. Each row is one customer with everything we know about them, plus their churn label.

---

## Step 4 — Exploratory Data Analysis

**File:** `notebooks/eda.ipynb`
**Purpose:** Understand the data BEFORE building models. Discover patterns, distributions, and relationships.

### Why EDA Matters

> "If you torture the data long enough, it will confess to anything." — Ronald Coase

EDA prevents you from building models on assumptions. It reveals:
- Is the target balanced or imbalanced?
- Which features seem predictive?
- Are there outliers or anomalies?
- Are features correlated with each other (multicollinearity)?

### Key Findings

#### Finding 1: Churn Rate = 27.9%

- 3,605 customers retained (72.1%)
- 1,395 customers churned (27.9%)
- **Implication:** This is "moderately imbalanced." A model that always predicts "not churned" gets 72.1% accuracy — but is completely useless. We need better metrics than accuracy.

#### Finding 2: Subscription Tier is the Strongest Predictor

| Tier | Churn Rate | Customer Count |
|------|-----------|----------------|
| Free | **44.8%** | 1,500 |
| Basic | 27.1% | 1,800 |
| Premium | 16.0% | 1,240 |
| Enterprise | **8.5%** | 460 |

Free tier customers churn at **5.3x** the rate of Enterprise customers. This makes intuitive sense — Free users have no financial commitment and can leave easily.

#### Finding 3: Spending Patterns Differ Dramatically

| Metric | Retained (avg) | Churned (avg) | Difference |
|--------|---------------|---------------|------------|
| Total Spend | $895 | $767 | -$128 (14% less) |
| Transactions | 14.4 | 12.5 | -1.8 (13% fewer) |
| Support Tickets | 1.7 | 2.3 | +0.6 (35% more) |
| Avg Rating | 2.9 | 2.5 | -0.4 (lower satisfaction) |

Churned customers spend less, buy less frequently, file more complaints, and rate products lower. Every metric tells the same story.

#### Finding 4: Correlation Analysis

The top features correlated with churn (from the correlation matrix):

| Feature | Correlation | Interpretation |
|---------|-------------|---------------|
| total_transactions | -0.204 | Fewer purchases → more churn |
| total_tickets | +0.188 | More complaints → more churn |
| total_spend | -0.167 | Lower spend → more churn |
| unique_categories | -0.145 | Less variety → more churn |
| min_rating | -0.120 | Lower worst review → more churn |
| avg_rating | -0.112 | Lower avg review → more churn |

These correlations are moderate (not ±0.9), which is realistic — real-world churn is influenced by many factors, not just one.

#### What This Means for Modeling

1. We should track **Precision, Recall, F1, and AUC-ROC** — not just accuracy
2. **subscription_tier** needs encoding (it's text, ML needs numbers)
3. No extreme data quality issues remain
4. Multiple features carry signal — a multivariate model will work

---

## Step 5 — Feature Engineering

**File:** `src/feature_engineering.py`
**Purpose:** Transform raw columns into more predictive ML features.

### What Is Feature Engineering?

Feature engineering is the process of creating NEW columns from existing data that better capture the patterns you want to predict. It's often **more impactful than choosing a different algorithm**.

Think of it this way: raw data tells you "this customer made 5 purchases." But a derived feature can tell you "this customer made 5 purchases in 30 days" (high frequency) vs "5 purchases in 700 days" (almost inactive). Same raw number, very different meaning.

### Derived Features We Created

| New Feature | Formula | What It Captures |
|------------|---------|-----------------|
| `spend_per_transaction` | total_spend / total_transactions | Spending intensity — big vs small purchases |
| `spend_per_day` | total_spend / tenure_days | How intensely they use the platform over time |
| `transactions_per_day` | total_transactions / tenure_days | Purchase frequency normalized by tenure |
| `tickets_per_transaction` | total_tickets / total_transactions | What fraction of purchases lead to complaints |
| `tickets_per_day` | total_tickets / tenure_days | Complaint frequency over time |
| `reviews_per_transaction` | total_reviews / total_transactions | Review engagement rate |
| `rating_gap` | avg_rating - min_rating | Consistency of experience (large gap = inconsistent) |
| `has_tickets` | 1 if total_tickets > 0 | Binary: ever complained? |
| `has_reviews` | 1 if total_reviews > 0 | Binary: ever reviewed? |
| `is_paid` | 1 if tier ≠ Free | Binary: paying customer? |

**Why normalize by tenure?** A customer who made 10 purchases in 30 days is very different from one who made 10 purchases in 2 years. Dividing by `tenure_days` puts everyone on the same scale.

### Encoding Categorical Variables

ML models work with numbers, not text. We need to convert categorical columns:

#### 1. Ordinal Encoding for Subscription Tier

```
Free → 0    Basic → 1    Premium → 2    Enterprise → 3
```

**Why ordinal?** There IS a natural order (Free < Basic < Premium < Enterprise). The numeric values preserve this ordering, which helps the model understand that Enterprise is "higher" than Free.

#### 2. One-Hot Encoding for Gender

```
Male       → [1, 0, 0]  (gender_Male=1, gender_Female=0, gender_Non-binary=0)
Female     → [0, 1, 0]
Non-binary → [0, 0, 1]
```

**Why one-hot?** There's no natural order between genders. If we used ordinal (Male=0, Female=1, Non-binary=2), the model might think Non-binary is "twice as much" as Female, which is meaningless.

#### 3. Frequency Encoding for City

```
New York → 0.073    (7.3% of customers live in New York)
Austin   → 0.062    (6.2% of customers)
...etc
```

**Why frequency?** With 15 cities, one-hot encoding creates 15 new columns — too many. Frequency encoding replaces each city with how common it is, capturing "city size" in a single column.

### Feature Scaling (StandardScaler)

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

StandardScaler transforms each feature to have **mean = 0** and **standard deviation = 1**.

**Why scale?** Some algorithms (Logistic Regression, SVM, neural networks) are sensitive to feature magnitude. Without scaling:
- `total_spend` ranges 0–5,000
- `avg_rating` ranges 0–5

The model would weight `total_spend` 1,000x more just because of its scale — not because it's actually more important. Scaling puts everything on equal footing.

**Note:** Tree-based models (Random Forest, XGBoost) don't need scaling because they make decisions based on thresholds, not magnitudes. But scaling doesn't hurt them.

### Final Output

- **Input:** 20 raw columns
- **Output:** 30 engineered features (saved as `data/engineered_dataset.csv`)
- **Saved artifacts:** `models/scaler.joblib` (for transforming new data at prediction time)

---

## Step 6 — ML Model Training & Evaluation

**File:** `src/model_training.py`
**Purpose:** Train 3 models, compare them, and save the best one.

### Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

- **80% training** (4,000 customers) — the model learns from this
- **20% testing** (1,000 customers) — we evaluate on this (the model NEVER sees this during training)

**Why split?** If you evaluate on the same data you trained on, the model memorizes the answers — like studying with the answer key. The test set simulates "unseen future data."

**`stratify=y`:** Ensures both train and test sets have the same churn ratio (27.9%). Without this, random chance might put 35% churn in training and 15% in testing, making results unreliable.

**`random_state=42`:** Makes the split reproducible — anyone running this code gets the exact same train/test split.

### The 3 Models

#### Model 1: Logistic Regression

**How it works:** Finds a linear boundary (a line, or hyperplane in high dimensions) that separates churned from non-churned customers. Each feature gets a "weight" — positive weights push toward churn, negative weights push away.

**Analogy:** Imagine plotting customers on a graph with "spend" on one axis and "tickets" on the other. Logistic Regression draws the best straight line between the churned and non-churned clusters.

```python
LogisticRegression(max_iter=1000, class_weight="balanced")
```

- `max_iter=1000`: maximum iterations to find the optimal boundary
- `class_weight="balanced"`: automatically upweights the minority class (churned). Without this, the model would learn to always predict "not churned" since that's correct 72% of the time.

**Strengths:** Simple, fast, interpretable (you can see each feature's weight)
**Weaknesses:** Can't capture non-linear patterns

#### Model 2: Random Forest

**How it works:** Creates 200 decision trees, each trained on a random subset of the data and features. Each tree makes a prediction; the final answer is the majority vote.

**Analogy:** Instead of one expert, you ask 200 experts (each with slightly different information). The majority answer is usually better than any single expert.

```python
RandomForestClassifier(
    n_estimators=200,      # 200 trees
    max_depth=15,          # each tree goes max 15 levels deep
    min_samples_split=10,  # need at least 10 samples to split a node
    class_weight="balanced"
)
```

- `max_depth=15`: limits tree complexity to prevent overfitting (memorizing training data)
- `min_samples_split=10`: don't create a split unless there are enough samples

**Strengths:** Handles non-linear patterns, robust, less prone to overfitting
**Weaknesses:** Less interpretable ("black box"), can overfit on noisy data

#### Model 3: XGBoost (Extreme Gradient Boosting)

**How it works:** Builds trees **sequentially** — each new tree focuses on correcting the errors of the previous one. It's like a student who reviews their wrong answers and studies those topics harder.

**Analogy:** First tree gets 70% right. Second tree focuses on the 30% it got wrong and fixes 60% of those. Third tree focuses on the remaining errors. After 200 rounds, most errors are fixed.

```python
XGBClassifier(
    n_estimators=200,      # 200 boosting rounds
    max_depth=6,           # shallower trees (each tree is a "weak learner")
    learning_rate=0.1,     # how much each tree contributes
    scale_pos_weight=2.58  # handle class imbalance
)
```

- `learning_rate=0.1`: Each tree contributes 10% of its prediction. Lower = more conservative, less overfitting, but needs more trees.
- `scale_pos_weight`: ratio of negative to positive samples, tells XGBoost to weight churned examples more.

**Strengths:** Usually the most accurate, handles complex patterns, built-in regularization
**Weaknesses:** Slowest to train, many hyperparameters to tune, can overfit

### Evaluation Metrics

**Why not just use accuracy?**

A model that ALWAYS predicts "not churned" gets **72.1% accuracy** (since 72.1% of customers are retained). But it catches **zero churners** — completely useless for our business problem.

We need metrics that account for the imbalanced classes:

| Metric | Formula | What It Measures | Our Context |
|--------|---------|-----------------|-------------|
| **Accuracy** | (TP + TN) / Total | Overall correctness | Misleading for imbalanced data |
| **Precision** | TP / (TP + FP) | Of predicted churners, how many actually churned? | Avoid false alarms (don't waste retention budget on loyal customers) |
| **Recall** | TP / (TP + FN) | Of actual churners, how many did we catch? | Don't miss real churners (catching them is the whole point) |
| **F1 Score** | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean of Precision and Recall | Balanced single metric |
| **AUC-ROC** | Area under ROC curve | Probability model ranks a random churner above a random non-churner | Overall ranking quality, threshold-independent |

Where: TP = True Positive, TN = True Negative, FP = False Positive, FN = False Negative.

### Results

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 64.2% | 40.4% | **59.9%** | **0.483** | **0.680** |
| Random Forest | 70.2% | 44.7% | 28.7% | 0.349 | 0.665 |
| XGBoost | 66.6% | 38.0% | 31.2% | 0.343 | 0.612 |

**Winner: Logistic Regression** (by F1 Score)

**Why did the simplest model win?**
- The relationships in our data are approximately linear (more spend → less churn)
- With `class_weight="balanced"`, it aggressively detects churners (59.9% recall)
- Random Forest and XGBoost, despite being more powerful, were more conservative — they had higher accuracy but caught fewer churners

**Confusion Matrix (Logistic Regression on test set):**

```
                    Predicted
                 Retain    Churn
Actual  Retain │  475      246  │  → 246 false alarms
        Churn  │  112      167  │  → 112 missed churners
```

- **167 true positives:** correctly identified churners we can now try to retain
- **112 false negatives:** churners we missed (room for improvement)
- **246 false positives:** false alarms (flagged as churning but stayed)
- **475 true negatives:** correctly identified loyal customers

**Top 5 Most Predictive Features:**

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | subscription_tier_encoded | 0.5347 |
| 2 | total_transactions | 0.3211 |
| 3 | resolved_tickets | 0.2711 |
| 4 | total_tickets | 0.2170 |
| 5 | total_spend | 0.1788 |

Subscription tier dominates — confirming our EDA finding.

### Cross-Validation

We also ran **5-fold cross-validation** for a more robust estimate:

```
F1 scores: [0.470, 0.460, 0.478, 0.475, 0.491]
Mean F1:   0.475 (±0.010)
```

Cross-validation splits the data 5 different ways and trains/evaluates 5 times. The small standard deviation (0.010) means our model performs consistently — it's not getting lucky on one particular split.

### Saved Artifacts

| File | What It Contains | Used For |
|------|-----------------|----------|
| `models/best_model.joblib` | Trained Logistic Regression model | Making predictions |
| `models/scaler.joblib` | Fitted StandardScaler | Transforming new data |
| `models/feature_cols.joblib` | List of 30 feature names | Ensuring correct feature order |
| `models/model_metadata.joblib` | Model name + evaluation metrics | API health endpoint |

---

## Step 7 — AI: Sentiment Analysis

**File:** `src/ai_insights.py` (function: `analyze_sentiment`)
**Purpose:** Use Claude to analyze customer review TEXT beyond simple star ratings.

### Why Use an LLM for Sentiment?

Star ratings tell you "this customer gave 2 stars." But the review TEXT tells you:
- **What** specifically is wrong (product quality? shipping? customer service?)
- **How** angry they are (mildly disappointed vs furious)
- **Whether** they'll come back ("might look elsewhere" vs "demanding a refund")
- **Mixed signals** ("love the product but hate the shipping" — a 3-star review that contains both)

Traditional sentiment libraries (VADER, TextBlob) classify text as positive/negative with a score. LLMs understand nuance, sarcasm, and context — and can extract structured themes.

### How It Works

1. We send a batch of reviews to Claude with a carefully crafted prompt
2. The prompt tells Claude to act as a "sentiment analysis expert"
3. We ask for **structured JSON output** — not free text — so we can use the results programmatically
4. For each review, Claude returns: sentiment, confidence, key themes, emotional tone, and churn risk

### The Prompt (Annotated)

```python
system = """You are a sentiment analysis expert for an e-commerce platform.
Analyze each customer review and extract structured insights.

For EACH review, return:
- sentiment: "positive", "negative", or "mixed"
- confidence: 0.0 to 1.0
- key_themes: list of 1-3 themes (e.g., "product quality", "shipping speed")
- emotional_tone: one word (e.g., "frustrated", "delighted")
- churn_risk: "low", "medium", or "high"

Return ONLY valid JSON array. No other text."""
```

**Prompt Engineering Techniques Used:**

| Technique | What It Is | How We Use It |
|-----------|-----------|---------------|
| **System Prompt** | Sets the AI's role and behavior | "You are a sentiment analysis expert" — gives Claude domain context |
| **Structured Output** | Request specific JSON format | We define exact field names and types so we can parse the response |
| **Constraints** | Limit response format | "Return ONLY valid JSON array. No other text." — prevents chatty responses |
| **Enumerated Values** | Provide allowed values | sentiment must be one of 3 values, not free-form |

### Example Input → Output

**Input:**
```
Review #REV_001 (Rating: 5/5):
"Absolutely love this product! The material feels premium. 
 Would definitely recommend to friends and family."

Review #REV_002 (Rating: 1/5):
"Terrible experience. The stitching came apart immediately. 
 Customer service was unhelpful when I complained."
```

**Claude's Output (JSON):**
```json
[
  {
    "sentiment": "positive",
    "confidence": 0.95,
    "key_themes": ["product quality", "material", "recommendation"],
    "emotional_tone": "delighted",
    "churn_risk": "low"
  },
  {
    "sentiment": "negative",
    "confidence": 0.98,
    "key_themes": ["product defect", "customer service", "quality"],
    "emotional_tone": "frustrated",
    "churn_risk": "high"
  }
]
```

**Business value:** Now we can aggregate sentiment by customer, identify the most common complaint themes, and flag high-churn-risk reviewers for proactive outreach.

---

## Step 8 — AI: Natural Language Querying

**File:** `src/ai_insights.py` (function: `query_data_natural_language`)
**Purpose:** Let anyone ask questions about the data in plain English.

### Why This Matters

Data analysts can write pandas code. Business stakeholders can't. Natural language querying bridges this gap — a VP of Marketing can type "What's our churn rate in Chicago?" and get an answer instantly, without filing a JIRA ticket to the data team.

### How It Works (3-Step Process)

```
Step 1: User asks a question in English
        "What is the average spend of churned vs retained customers?"

Step 2: Claude generates pandas code
        df.groupby('churned')['total_spend'].mean()

Step 3: We execute the code and Claude summarizes the result
        "Retained customers spend an average of $895, while churned 
         customers spend $767 — a $128 gap (14% less)."
```

### The Prompt (Annotated)

We give Claude three critical pieces of context:

1. **Schema Information** — The actual column names, data types, and sample rows. Without this, Claude would have to guess column names and might hallucinate.

```python
schema_info = f"""DataFrame 'df' has {len(df)} rows and these columns:
{df.dtypes.to_string()}

Sample data (first 3 rows):
{df.head(3).to_string()}
"""
```

2. **Few-Shot Examples** — We show Claude 3 examples of question → code mappings. This dramatically improves accuracy because Claude can pattern-match.

```
Q: "What is the average spend of churned customers?"
A: {"code": "df[df['churned']==1]['total_spend'].mean()", ...}

Q: "Which city has the most customers?"
A: {"code": "df['city'].value_counts().head(1)", ...}
```

3. **Safety Constraint** — We only allow pandas operations, not arbitrary code execution. The `eval()` function is called with restricted builtins.

### Prompt Engineering Technique: Grounding

**Grounding** means giving the AI real data so it doesn't make things up. By showing Claude the actual column names and sample data, we ensure:
- It uses real column names (not guessing `customer_spend` when it's actually `total_spend`)
- It understands data types (won't try to average a string column)
- It knows the data shape (won't reference columns that don't exist)

### Example Questions It Can Answer

| Question | Generated Code | Answer |
|----------|---------------|--------|
| "What is the churn rate by city?" | `df.groupby('city')['churned'].mean().sort_values(ascending=False)` | "Houston has the highest churn rate at 31.2%, while Portland has the lowest at 24.8%." |
| "How many customers are on each tier?" | `df['subscription_tier'].value_counts()` | "Basic has the most customers (1,800), followed by Free (1,500), Premium (1,240), and Enterprise (460)." |
| "What's the churn rate for customers with 5+ tickets?" | `df[df['total_tickets']>=5]['churned'].mean()` | "Customers with 5 or more tickets have a 46.3% churn rate — nearly double the overall 27.9%." |

---

## Step 9 — AI: Churn Explainer

**File:** `src/ai_insights.py` (function: `explain_churn_prediction`)
**Purpose:** Generate human-readable explanations of ML predictions.

### The Problem: Black Box Models

ML models output a number: "73% probability of churning." But stakeholders need to know **WHY**:
- Is it because they stopped buying?
- Is it because they're unhappy with support?
- Is it because they're on a Free plan?
- What should we DO about it?

This is called **Explainable AI (XAI)** — one of the most important topics in production ML.

### How Our Explainer Works

1. We take the customer's actual data points (tier, spend, tickets, ratings, etc.)
2. We include the model's predicted churn probability
3. We ask Claude to explain the prediction in business terms
4. Claude references specific numbers from the data (grounding — prevents hallucination)

### The Prompt Structure

```python
system = """You are a customer success analyst at an e-commerce company.
Your job is to explain ML churn predictions in plain English that
business stakeholders can understand and act on.

Structure your response as:
1. RISK ASSESSMENT (1 sentence summary)
2. KEY RISK FACTORS (bullet points — reference specific data)
3. RECOMMENDED ACTIONS (2-3 specific retention strategies)

Be specific — reference actual numbers from the data.
Keep it concise — max 200 words."""
```

### Example Output

**Input:** Customer CUST_01355 — Free tier, $117 spend, 3 tickets, 1.5★ rating, 71.3% churn probability

**Claude's Explanation:**

> **RISK ASSESSMENT:**
> This customer is at HIGH RISK of churning (71.3%), driven by extremely low engagement and poor satisfaction signals.
>
> **KEY RISK FACTORS:**
> - Free tier with 715 days tenure — long-time user who has never upgraded, suggesting low perceived value
> - Only 4 transactions totaling $117 over 2 years — spending of just $0.16/day, far below the platform average
> - Average rating of 1.5/5 — among the lowest in the customer base, indicating deep dissatisfaction
> - 3 support tickets for just 4 purchases — a 75% complaint rate, signaling recurring issues
>
> **RECOMMENDED ACTIONS:**
> 1. Proactive outreach with a personalized discount or free trial of Basic tier
> 2. Priority support routing — resolve their next ticket within 24 hours
> 3. Product recommendation based on their purchase history to drive re-engagement

**Business value:** A customer success team can now triage thousands of at-risk customers with actionable, personalized recommendations — instead of just a list of probabilities.

---

## Step 10 — FastAPI Prediction Server

**File:** `src/api.py`
**Purpose:** Serve ML predictions and AI insights as REST API endpoints.

### Why an API?

In production, models don't run in notebooks. They're deployed as **web services** that other applications can call:
- A website makes a POST request → gets a churn prediction
- A mobile app calls the sentiment endpoint → gets review analysis
- A CRM system batch-queries 10,000 customers → gets risk scores

### What is FastAPI?

FastAPI is a modern Python web framework for building APIs. It's popular for ML because:
- **Fast** — built on async (Starlette), handles thousands of requests/second
- **Auto-documentation** — generates interactive Swagger UI at `/docs`
- **Validation** — Pydantic models automatically validate request data
- **Type hints** — Python type hints become the API contract

### Our 7 Endpoints

| Endpoint | Method | Input | Output | Purpose |
|----------|--------|-------|--------|---------|
| `/` | GET | None | `{"status": "online"}` | Health check |
| `/health` | GET | None | Model name + metrics | Monitoring |
| `/predict` | POST | Customer features (JSON) | Churn probability + risk level | Single prediction |
| `/predict/batch` | POST | List of customers | List of predictions | Batch prediction |
| `/ai/sentiment` | POST | List of reviews | Sentiment analysis results | Review analysis |
| `/ai/query` | POST | Question string | Answer + generated code | NL data query |
| `/ai/explain` | POST | Customer + probability | Explanation text | Explainable AI |

### How Prediction Works (Request Flow)

```
1. Client sends POST /predict with customer JSON
       ↓
2. Pydantic validates the request (types, ranges)
       ↓
3. prepare_features() creates derived features + encodes categoricals
       ↓
4. scaler.transform() scales features to match training data
       ↓
5. model.predict_proba() returns churn probability
       ↓
6. Response includes probability, prediction (0/1), and risk level
```

### Pydantic Request Validation

```python
class CustomerFeatures(BaseModel):
    age: int = Field(..., ge=18, le=100)          # Must be 18-100
    total_spend: float = Field(..., ge=0)          # Must be non-negative
    subscription_tier: str = Field(...)             # Required
    avg_rating: float = Field(..., ge=0, le=5)     # Must be 0-5
```

If someone sends `{"age": -5}`, FastAPI automatically returns a `422 Validation Error` with a helpful message. You don't write any validation code — Pydantic handles it.

### Example API Call

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 44,
    "tenure_days": 715,
    "total_transactions": 4,
    "total_spend": 117.25,
    "subscription_tier": "Free",
    ...
  }'
```

**Response:**
```json
{
    "churn_probability": 0.6962,
    "churn_prediction": 1,
    "risk_level": "HIGH",
    "model_used": "Logistic Regression"
}
```

### Running the API

```bash
uvicorn src.api:app --reload --port 8000
# Visit http://localhost:8000/docs for interactive Swagger UI
```

---

## Step 11 — Streamlit Dashboard

**File:** `dashboard/app.py`
**Purpose:** Interactive web app that brings everything together.

### What is Streamlit?

Streamlit converts Python scripts into web applications. You write Python — Streamlit handles the HTML, CSS, and JavaScript. It's the fastest way to build data apps without frontend knowledge.

### 4 Dashboard Pages

#### Page 1: Overview

The executive summary page with:
- **KPI Cards** — total customers, churn count, churn rate, avg spend, avg rating
- **Churn Pie Chart** — visual breakdown of churned vs retained
- **Churn by Tier Bar Chart** — shows the Free→Enterprise progression
- **Spend Box Plot** — side-by-side comparison of churned vs retained spending
- **Monthly Transaction Trend** — line chart of transaction volume over time
- **Correlation Bar Chart** — which features correlate most with churn

#### Page 2: Customer Explorer

An individual customer deep-dive:
- **Customer Profile** — age, gender, city, tier, tenure
- **Activity Summary** — transactions, spend, categories
- **Support & Reviews** — ticket count, resolution rate, ratings
- **Live Churn Prediction** — real-time model prediction with a gauge chart
- **Customer Reviews** — displays their actual review text with star ratings

The prediction is made live — when you select a different customer, the model runs instantly and updates the gauge.

#### Page 3: AI Insights

Three tabs for our Claude-powered features:
- **Ask Your Data** — text input for natural language queries with example questions
- **Sentiment Analysis** — select random reviews to analyze with Claude
- **Churn Explainer** — select a customer, see their churn probability, and get an AI explanation

#### Page 4: Model Performance

Technical details for data scientists:
- **Model info** — which algorithm won and its metrics
- **Metrics table** — all 5 evaluation metrics
- **Results charts** — ROC curves, confusion matrix, metric comparison
- **Feature importance** — interactive bar chart of top 15 features
- **EDA visualizations** — all charts from the EDA notebook

### Running the Dashboard

```bash
cd ecommerce-churn-ai
streamlit run dashboard/app.py
# Opens automatically in browser at http://localhost:8501
```

---

## Step 12 — Docker & Deployment

**Files:** `Dockerfile`, `start.sh`
**Purpose:** Package everything into a container that runs anywhere.

### What is Docker?

Docker creates a "container" — a self-contained box with your code, dependencies, and configuration. Anyone with Docker can run your project identically, regardless of their OS or Python version.

**Analogy:** A Docker container is like a shipping container. It doesn't matter if the cargo ship is big or small, old or new — the container fits and works the same way everywhere.

### Our Dockerfile (Annotated)

```dockerfile
FROM python:3.11-slim          # Start with a minimal Python 3.11 image

WORKDIR /app                   # All commands run inside /app

COPY requirements.txt .        # Copy deps first (cached layer)
RUN pip install -r requirements.txt  # Install deps (only re-runs if requirements change)

COPY . .                       # Copy all project code

# Generate data and train model inside the container
RUN python data/generate_data.py && \
    python src/data_pipeline.py && \
    python src/feature_engineering.py && \
    python src/model_training.py

EXPOSE 8501 8000               # Dashboard on 8501, API on 8000

CMD ["./start.sh"]             # Run both services
```

### Docker Layer Caching

Notice we copy `requirements.txt` separately BEFORE copying the rest of the code. This is a best practice:
- Docker caches each step ("layer")
- If `requirements.txt` hasn't changed, pip install is skipped (cached)
- Only the `COPY . .` and later steps re-run
- This saves minutes on every rebuild

### Build & Run

```bash
docker build -t churn-ai .
docker run -p 8501:8501 -p 8000:8000 --env-file .env churn-ai
```

---

## Results & Key Findings

### Data Insights
- **27.9% churn rate** — 1,395 out of 5,000 customers churned
- **Free tier churns at 44.8%** — 5.3x higher than Enterprise (8.5%)
- **Churned customers spend 14% less** ($767 vs $895 average)
- **Churned customers file 35% more support tickets** (2.3 vs 1.7 average)
- **Subscription tier is the #1 predictor** of churn

### Model Performance
- **Best model:** Logistic Regression with F1 = 0.483, AUC-ROC = 0.680
- **Catches 60% of churners** (167 out of 279 in the test set)
- **Cross-validation confirms stability:** F1 = 0.475 ± 0.010
- **Simplest model won** — linear relationships in the data favor logistic regression

### AI Capabilities
- **Sentiment analysis** extracts themes, emotional tone, and churn risk from free text
- **NL querying** translates English questions into pandas code automatically
- **Churn explainer** generates actionable retention recommendations

---

## Concepts Cheat Sheet

| Concept | What It Is | Where We Used It |
|---------|-----------|-----------------|
| **Train/Test Split** | Separate data into training and evaluation sets | Step 6 — 80/20 split |
| **Feature Engineering** | Create new predictive features from raw data | Step 5 — 10 derived features |
| **One-Hot Encoding** | Convert categories to binary columns | Step 5 — gender encoding |
| **StandardScaler** | Normalize features to mean=0, std=1 | Step 5 — feature scaling |
| **Class Imbalance** | When one class vastly outnumbers the other | Step 6 — 72/28 split |
| **Precision** | Of predicted positives, how many are correct? | Step 6 — 40.4% |
| **Recall** | Of actual positives, how many did we catch? | Step 6 — 59.9% |
| **F1 Score** | Harmonic mean of precision and recall | Step 6 — 0.483 |
| **AUC-ROC** | Area under the ROC curve | Step 6 — 0.680 |
| **Cross-Validation** | Train/evaluate k times with different splits | Step 6 — 5-fold CV |
| **Confusion Matrix** | Table of TP, TN, FP, FN | Step 6 — 475/246/112/167 |
| **Prompt Engineering** | Crafting prompts for reliable AI output | Steps 7-9 |
| **Structured Output** | Getting JSON (not free text) from an LLM | Step 7 — sentiment JSON |
| **Few-Shot Learning** | Showing examples in the prompt | Step 8 — query examples |
| **Grounding** | Giving AI real data to prevent hallucination | Steps 8-9 |
| **Explainable AI (XAI)** | Making model decisions interpretable | Step 9 — churn explainer |
| **REST API** | Serving predictions over HTTP | Step 10 — FastAPI |
| **Pydantic** | Request validation via Python type hints | Step 10 — input schemas |
| **Docker** | Containerization for reproducible deployment | Step 12 |
| **Model Serialization** | Saving trained models to disk (joblib) | Step 6 — .joblib files |

---

## How to Run

### Prerequisites
- Python 3.9+ installed
- (Optional) Docker installed
- (Optional) Anthropic API key for AI features

### Quick Start

```bash
# 1. Navigate to project
cd ecommerce-churn-ai

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (macOS only) Install OpenMP for XGBoost
brew install libomp

# 5. Generate data
python data/generate_data.py

# 6. Run data pipeline
python src/data_pipeline.py

# 7. Engineer features
python src/feature_engineering.py

# 8. Train models
python src/model_training.py

# 9. (Optional) Set up AI features
cp .env.example .env
# Edit .env with your Anthropic API key

# 10. Run the dashboard
streamlit run dashboard/app.py

# 11. (In a separate terminal) Run the API
source venv/bin/activate
uvicorn src.api:app --reload --port 8000
```

---

## Production Improvements

If this were a real production system, here's what I'd add:

| Improvement | Why |
|-------------|-----|
| **Hyperparameter Tuning** (GridSearchCV / Optuna) | Our F1 of 0.48 could likely reach 0.60+ with optimized hyperparameters |
| **SMOTE / Advanced Resampling** | Better handling of the 72/28 class imbalance |
| **MLflow** | Experiment tracking — log every model run, compare results over time |
| **Data Drift Detection** | Alert when incoming data distributions shift (model may need retraining) |
| **Scheduled Retraining** | Automate weekly model retraining as new data arrives |
| **A/B Testing** | Measure actual business impact of retention interventions |
| **API Authentication** | JWT tokens or API keys to secure endpoints |
| **CI/CD Pipeline** | GitHub Actions for automated testing and deployment |
| **Monitoring & Alerting** | Track prediction latency, error rates, model performance decay |
| **Feature Store** | Centralized feature computation for training and serving consistency |

---

## Interview Talking Points

### 30-Second Elevator Pitch
"I built an end-to-end AI analytics platform that predicts e-commerce customer churn. It combines traditional ML — where I trained and compared 3 models on 30 engineered features — with Claude AI for sentiment analysis, natural language data querying, and explainable predictions. Everything is served via a FastAPI REST API and visualized in a Streamlit dashboard."

### Key Discussion Points

**On Data Engineering:**
"I built a 5-stage pipeline that goes from 5 raw tables with 96,000+ records down to a single clean dataset. The critical step was the aggregation — converting transaction-level data into customer-level features using groupby operations and LEFT JOINs to preserve customers with no activity."

**On Feature Engineering:**
"I created 10 derived features based on domain knowledge. The key insight was normalizing by tenure — a customer with 5 purchases in 30 days is very different from one with 5 purchases in 2 years. I also used ordinal encoding for the subscription tier since it has a natural ordering, and frequency encoding for cities to avoid dimensionality explosion."

**On Model Selection:**
"Interestingly, Logistic Regression outperformed Random Forest and XGBoost on F1 score. This happens when relationships are approximately linear. I chose F1 over accuracy because with 28% churn, accuracy is misleading — a model that always predicts 'retained' gets 72% accuracy but catches zero churners."

**On AI Integration:**
"The AI features aren't gimmicks — they solve real gaps. Sentiment analysis extracts themes from free text that ML can't process. Natural language querying lets non-technical users access data. And the churn explainer bridges the gap between a model's probability score and actionable business recommendations."

**On Architecture Decisions:**
"I separated concerns — the data pipeline, feature engineering, model training, and serving are all independent modules. The API uses Pydantic for validation and serves the model via joblib deserialization. The dashboard calls the same feature engineering functions as training, ensuring consistency between training and prediction."

---

*End of Guide*
