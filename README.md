# E-Commerce Customer Churn Prediction & AI Analytics Platform

An end-to-end AI + Analytics project that predicts customer churn, analyzes sentiment with Claude AI, and serves insights through an interactive dashboard.

## What This Project Covers

| Area | Concepts |
|------|----------|
| **Data Engineering** | Synthetic data generation, data pipelines, cleaning, validation |
| **Analytics** | EDA, statistical analysis, correlation, visualization |
| **Machine Learning** | Feature engineering, model training (Logistic Regression, Random Forest, XGBoost), evaluation |
| **AI / LLM** | Sentiment analysis, natural language querying, explainable AI — all powered by Claude |
| **Backend** | REST API with FastAPI, model serving, request validation |
| **Frontend** | Interactive Streamlit dashboard with charts and AI chat |
| **DevOps** | Docker containerization, environment management |

## Project Structure

```
ecommerce-churn-ai/
├── data/                       # Datasets + generation script
│   ├── generate_data.py           # Creates synthetic e-commerce data
│   ├── customers.csv              # 5,000 customer profiles
│   ├── transactions.csv           # ~69,000 purchase records
│   ├── support_tickets.csv        # ~9,000 support interactions
│   ├── reviews.csv                # ~12,000 product reviews
│   ├── churn_labels.csv           # Churn labels (prediction target)
│   ├── clean_dataset.csv          # Merged & cleaned dataset
│   └── engineered_dataset.csv     # Feature-engineered dataset
├── notebooks/
│   └── eda.ipynb                  # Exploratory Data Analysis
├── src/
│   ├── data_pipeline.py           # Data cleaning & merging
│   ├── feature_engineering.py     # Feature creation & encoding
│   ├── model_training.py          # ML model training & evaluation
│   ├── ai_insights.py             # Claude AI: sentiment, NL queries, explainer
│   └── api.py                     # FastAPI prediction server
├── dashboard/
│   └── app.py                     # Streamlit analytics dashboard
├── models/                        # Saved models & artifacts
├── requirements.txt
├── Dockerfile
└── README.md
```

## Quick Start

### 1. Setup

```bash
# Clone the repo
cd ecommerce-churn-ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# (macOS only) Install OpenMP for XGBoost
brew install libomp
```

### 2. Generate Data & Train Model

```bash
# Generate synthetic dataset
python data/generate_data.py

# Run data pipeline (clean & merge)
python src/data_pipeline.py

# Engineer features
python src/feature_engineering.py

# Train ML models
python src/model_training.py
```

### 3. Set Up AI Features (Optional)

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your Anthropic API key
# Get your key at: https://console.anthropic.com/settings/keys
```

### 4. Run the Dashboard

```bash
cd ecommerce-churn-ai
streamlit run dashboard/app.py
```

Visit `http://localhost:8501` in your browser.

### 5. Run the API

```bash
uvicorn src.api:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive API docs.

### 6. Docker (Optional)

```bash
docker build -t churn-ai .
docker run -p 8501:8501 -p 8000:8000 --env-file .env churn-ai
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Model info & metrics |
| `/predict` | POST | Single churn prediction |
| `/predict/batch` | POST | Batch predictions |
| `/ai/sentiment` | POST | Review sentiment analysis |
| `/ai/query` | POST | Natural language data query |
| `/ai/explain` | POST | AI churn explanation |

## Tech Stack

- **Python 3.11+** — core language
- **pandas / numpy** — data processing
- **scikit-learn / XGBoost** — ML models
- **matplotlib / seaborn / plotly** — visualization
- **Claude API (Anthropic)** — AI features
- **FastAPI** — REST API
- **Streamlit** — dashboard
- **Docker** — containerization

## Key Learnings

1. **Data pipelines** are essential — clean data is the foundation of everything
2. **Feature engineering** often matters more than algorithm choice
3. **Accuracy is misleading** for imbalanced datasets — use F1, AUC-ROC instead
4. **LLMs add real value** when grounded in data (sentiment, explanations, NL queries)
5. **End-to-end thinking** — from raw data to deployed dashboard — is a critical skill
