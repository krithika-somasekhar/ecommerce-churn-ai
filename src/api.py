"""
FastAPI Prediction & AI Insights API
======================================
This module serves our ML model and AI features as REST API endpoints.

WHY AN API?
In production, models don't run in notebooks. They're deployed as web
services that other applications can call. FastAPI is the most popular
Python framework for ML APIs because:
- It's fast (async, based on Starlette)
- Auto-generates interactive docs (Swagger UI)
- Built-in request validation via Pydantic
- Easy to deploy (Docker, cloud, etc.)

ENDPOINTS:
  GET  /                     — Health check
  GET  /health               — Detailed health status
  POST /predict              — Get churn prediction for a customer
  POST /predict/batch        — Get predictions for multiple customers
  POST /ai/sentiment         — Analyze review sentiment
  POST /ai/query             — Ask a question about the data in English
  POST /ai/explain           — Get AI explanation of a churn prediction

HOW TO RUN:
  uvicorn src.api:app --reload --port 8000

THEN VISIT:
  http://localhost:8000/docs  — Interactive API documentation
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# Import our AI insights module
from src.ai_insights import analyze_sentiment, query_data_natural_language, explain_churn_prediction

# ============================================================
# LOAD MODEL & ARTIFACTS
# ============================================================
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

# Load saved model, scaler, and feature columns
model = joblib.load(os.path.join(MODEL_DIR, "best_model.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.joblib"))
model_metadata = joblib.load(os.path.join(MODEL_DIR, "model_metadata.joblib"))


# ============================================================
# PYDANTIC MODELS (Request/Response Schemas)
# ============================================================
# Pydantic models define the expected shape of API requests/responses.
# FastAPI uses them for automatic validation and documentation.

class CustomerFeatures(BaseModel):
    """Input features for churn prediction."""
    age: int = Field(..., ge=18, le=100, description="Customer age")
    tenure_days: int = Field(..., ge=0, description="Days since signup")
    total_transactions: int = Field(..., ge=0, description="Total number of purchases")
    total_spend: float = Field(..., ge=0, description="Total amount spent ($)")
    avg_transaction_amount: float = Field(..., ge=0, description="Average purchase amount")
    max_transaction_amount: float = Field(..., ge=0, description="Largest single purchase")
    unique_categories: int = Field(..., ge=0, description="Number of distinct product categories")
    days_since_last_purchase: int = Field(..., ge=0, description="Days since most recent purchase")
    total_tickets: int = Field(..., ge=0, description="Support tickets filed")
    resolved_tickets: int = Field(..., ge=0, description="Tickets that were resolved")
    avg_resolution_days: float = Field(..., ge=0, description="Average days to resolve a ticket")
    ticket_resolution_rate: float = Field(..., ge=0, le=1, description="Fraction of tickets resolved")
    total_reviews: int = Field(..., ge=0, description="Number of product reviews written")
    avg_rating: float = Field(..., ge=0, le=5, description="Average review rating given")
    min_rating: float = Field(..., ge=0, le=5, description="Lowest rating given")
    subscription_tier: str = Field(..., description="Free, Basic, Premium, or Enterprise")
    gender: str = Field(..., description="Male, Female, or Non-binary")
    city: str = Field(default="Unknown", description="Customer city")


class PredictionResponse(BaseModel):
    """Response from the prediction endpoint."""
    churn_probability: float
    churn_prediction: int
    risk_level: str
    model_used: str


class ReviewInput(BaseModel):
    """Input for sentiment analysis."""
    review_id: str
    review_text: str
    rating: int = Field(..., ge=1, le=5)


class QueryInput(BaseModel):
    """Input for natural language query."""
    question: str


class ExplainInput(BaseModel):
    """Input for churn explanation."""
    customer: CustomerFeatures
    churn_probability: float = Field(..., ge=0, le=1)


# ============================================================
# HELPER: Transform raw input into model features
# ============================================================
def prepare_features(customer: CustomerFeatures) -> np.ndarray:
    """
    Transform API input into the feature vector the model expects.
    This mirrors the feature engineering pipeline:
    1. Create derived features
    2. Encode categoricals
    3. Scale everything
    """
    data = customer.model_dump()

    # Derived features (same as feature_engineering.py)
    total_tx = data["total_transactions"]
    tenure = data["tenure_days"]

    derived = {
        "spend_per_transaction": data["total_spend"] / total_tx if total_tx > 0 else 0,
        "spend_per_day": data["total_spend"] / tenure if tenure > 0 else 0,
        "transactions_per_day": total_tx / tenure if tenure > 0 else 0,
        "tickets_per_transaction": data["total_tickets"] / total_tx if total_tx > 0 else 0,
        "tickets_per_day": data["total_tickets"] / tenure if tenure > 0 else 0,
        "reviews_per_transaction": data["total_reviews"] / total_tx if total_tx > 0 else 0,
        "rating_gap": data["avg_rating"] - data["min_rating"],
        "has_tickets": int(data["total_tickets"] > 0),
        "has_reviews": int(data["total_reviews"] > 0),
        "is_paid": int(data["subscription_tier"] != "Free"),
    }

    # Encode categoricals
    tier_map = {"Free": 0, "Basic": 1, "Premium": 2, "Enterprise": 3}
    encoded = {
        "subscription_tier_encoded": tier_map.get(data["subscription_tier"], 0),
        "city_frequency": 0.067,  # Default frequency (1/15 cities)
        "gender_Female": int(data["gender"] == "Female"),
        "gender_Male": int(data["gender"] == "Male"),
        "gender_Non-binary": int(data["gender"] == "Non-binary"),
    }

    # Build feature vector in the correct column order
    feature_dict = {
        "age": data["age"],
        "tenure_days": tenure,
        "total_transactions": total_tx,
        "total_spend": data["total_spend"],
        "avg_transaction_amount": data["avg_transaction_amount"],
        "max_transaction_amount": data["max_transaction_amount"],
        "unique_categories": data["unique_categories"],
        "days_since_last_purchase": data["days_since_last_purchase"],
        "total_tickets": data["total_tickets"],
        "resolved_tickets": data["resolved_tickets"],
        "avg_resolution_days": data["avg_resolution_days"],
        "ticket_resolution_rate": data["ticket_resolution_rate"],
        "total_reviews": data["total_reviews"],
        "avg_rating": data["avg_rating"],
        "min_rating": data["min_rating"],
        **derived,
        **encoded,
    }

    # Create DataFrame with correct column order and scale
    feature_vector = pd.DataFrame([feature_dict])[feature_cols]
    scaled = scaler.transform(feature_vector)
    return scaled


# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(
    title="Churn Prediction & AI Insights API",
    description="ML-powered churn prediction with AI-driven customer insights",
    version="1.0.0",
)


@app.get("/")
def root():
    """Health check — confirms the API is running."""
    return {"status": "online", "model": model_metadata["model_name"]}


@app.get("/health")
def health():
    """Detailed health status."""
    return {
        "status": "healthy",
        "model": model_metadata["model_name"],
        "metrics": model_metadata["metrics"],
        "features": len(feature_cols),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerFeatures):
    """
    Predict whether a customer will churn.

    Takes customer features as input, returns:
    - churn_probability (0-1)
    - churn_prediction (0 or 1)
    - risk_level (LOW / MEDIUM / HIGH)
    """
    features = prepare_features(customer)
    probability = float(model.predict_proba(features)[0, 1])
    prediction = int(probability >= 0.5)

    if probability > 0.6:
        risk = "HIGH"
    elif probability > 0.3:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return PredictionResponse(
        churn_probability=round(probability, 4),
        churn_prediction=prediction,
        risk_level=risk,
        model_used=model_metadata["model_name"],
    )


@app.post("/predict/batch")
def predict_batch(customers: list[CustomerFeatures]):
    """Predict churn for multiple customers at once."""
    results = []
    for customer in customers:
        features = prepare_features(customer)
        probability = float(model.predict_proba(features)[0, 1])
        results.append({
            "churn_probability": round(probability, 4),
            "churn_prediction": int(probability >= 0.5),
            "risk_level": "HIGH" if probability > 0.6 else "MEDIUM" if probability > 0.3 else "LOW",
        })
    return results


@app.post("/ai/sentiment")
def sentiment_analysis(reviews: list[ReviewInput]):
    """Analyze sentiment of customer reviews using Claude AI."""
    review_dicts = [r.model_dump() for r in reviews]
    results = analyze_sentiment(review_dicts)
    return {"results": results}


@app.post("/ai/query")
def natural_language_query(query: QueryInput):
    """Ask a question about the customer data in plain English."""
    result = query_data_natural_language(query.question)
    return result


@app.post("/ai/explain")
def explain_prediction(data: ExplainInput):
    """Get an AI-generated explanation of a churn prediction."""
    customer_dict = {
        "subscription_tier": data.customer.subscription_tier,
        "tenure_days": data.customer.tenure_days,
        "total_transactions": data.customer.total_transactions,
        "total_spend": data.customer.total_spend,
        "total_tickets": data.customer.total_tickets,
        "ticket_resolution_rate": data.customer.ticket_resolution_rate,
        "avg_rating": data.customer.avg_rating,
        "total_reviews": data.customer.total_reviews,
    }
    explanation = explain_churn_prediction(customer_dict, data.churn_probability)
    return {"explanation": explanation}
