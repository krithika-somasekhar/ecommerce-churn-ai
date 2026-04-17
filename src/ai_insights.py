"""
AI-Powered Insights (Claude Integration)
==========================================
This module adds LLM-powered intelligence to our analytics platform.

THREE AI FEATURES:
1. Sentiment Analysis — Analyze customer review text to extract sentiment,
   key themes, and emotional indicators that go beyond star ratings.
2. Natural Language Querying — Let users ask questions about the data
   in plain English. Claude translates to pandas code, executes it,
   and returns a human-readable answer.
3. Churn Explainer — Given a customer's data, Claude generates a
   plain-English explanation of WHY they might churn, referencing
   specific data points.

KEY AI/LLM CONCEPTS:
- Prompt Engineering: crafting prompts that get reliable, structured outputs
- System Prompts: setting the AI's role and behavior
- Structured Output: getting JSON back from an LLM (not just free text)
- Few-shot Examples: showing the AI examples of desired output
- Grounding: giving the AI real data so it doesn't hallucinate
"""

import os
import json
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def get_client():
    """Create an Anthropic API client."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        raise ValueError(
            "Please set your ANTHROPIC_API_KEY in the .env file.\n"
            "Get your key at: https://console.anthropic.com/settings/keys"
        )
    return Anthropic(api_key=api_key)


# ============================================================
# FEATURE 1: SENTIMENT ANALYSIS
# ============================================================
def analyze_sentiment(reviews: list[dict]) -> list[dict]:
    """
    Analyze sentiment of customer reviews using Claude.

    WHY USE AN LLM INSTEAD OF A SIMPLE SENTIMENT LIBRARY?
    - LLMs understand nuance, sarcasm, and context
    - They can extract THEMES (what specifically is good/bad)
    - They can detect mixed sentiment ("love the product, hate the shipping")
    - They return structured data we can use programmatically

    PROMPT ENGINEERING TECHNIQUES USED:
    - System prompt: defines Claude's role as a sentiment analyst
    - Structured output: we ask for JSON so we can parse it programmatically
    - Batch processing: we send multiple reviews at once (more efficient)

    Args:
        reviews: list of dicts with 'review_id', 'review_text', 'rating'

    Returns:
        list of dicts with sentiment analysis results
    """
    client = get_client()

    # Format reviews for the prompt
    reviews_text = "\n\n".join([
        f"Review #{r['review_id']} (Rating: {r['rating']}/5):\n\"{r['review_text']}\""
        for r in reviews
    ])

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system="""You are a sentiment analysis expert for an e-commerce platform.
Analyze each customer review and extract structured insights.

For EACH review, return:
- sentiment: "positive", "negative", or "mixed"
- confidence: 0.0 to 1.0
- key_themes: list of 1-3 themes (e.g., "product quality", "shipping speed", "customer service")
- emotional_tone: one word (e.g., "frustrated", "delighted", "disappointed", "satisfied")
- churn_risk: "low", "medium", or "high" — based on the review content

Return ONLY valid JSON array. No other text.""",
        messages=[{
            "role": "user",
            "content": f"Analyze these customer reviews:\n\n{reviews_text}"
        }]
    )

    # Parse the JSON response
    response_text = message.content[0].text
    # Extract JSON from response (handle potential markdown code blocks)
    if "```" in response_text:
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]

    results = json.loads(response_text.strip())
    return results


# ============================================================
# FEATURE 2: NATURAL LANGUAGE DATA QUERYING
# ============================================================
def query_data_natural_language(question: str, df: pd.DataFrame = None) -> dict:
    """
    Let users ask questions about the data in plain English.

    HOW IT WORKS:
    1. User asks: "What is the average spend of churned customers?"
    2. Claude sees the question + the DataFrame schema (column names, types, sample)
    3. Claude generates pandas code to answer the question
    4. We execute the code safely and return the result
    5. Claude then explains the result in plain English

    PROMPT ENGINEERING TECHNIQUES:
    - Few-shot examples: show Claude examples of question → code mappings
    - Schema grounding: give Claude the actual column names and types
    - Safety: we only execute pandas operations, not arbitrary code

    Args:
        question: natural language question about the data
        df: DataFrame to query (loads clean_dataset.csv if not provided)

    Returns:
        dict with 'answer', 'code', 'result'
    """
    if df is None:
        df = pd.read_csv(os.path.join(DATA_DIR, "clean_dataset.csv"))

    client = get_client()

    # Give Claude context about the data
    schema_info = f"""DataFrame 'df' has {len(df)} rows and these columns:
{df.dtypes.to_string()}

Sample data (first 3 rows):
{df.head(3).to_string()}

Key facts:
- 'churned' column: 1 = churned, 0 = retained
- 'subscription_tier': Free, Basic, Premium, Enterprise
- Numeric columns can be aggregated (sum, mean, median, etc.)
"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=f"""You are a data analyst assistant. The user will ask questions about an e-commerce customer dataset.

DATA SCHEMA:
{schema_info}

YOUR TASK:
1. Write a single pandas expression that answers the question
2. The DataFrame is available as 'df'
3. Return ONLY valid JSON with two fields:
   - "code": the pandas code (single expression, NO imports needed)
   - "explanation": plain English explanation of what the code does

EXAMPLES:
Q: "What is the average spend of churned customers?"
A: {{"code": "df[df['churned']==1]['total_spend'].mean()", "explanation": "Filters to churned customers and calculates their average total spend"}}

Q: "Which city has the most customers?"
A: {{"code": "df['city'].value_counts().head(1)", "explanation": "Counts customers per city and returns the top one"}}

Q: "What's the churn rate by subscription tier?"
A: {{"code": "df.groupby('subscription_tier')['churned'].mean().sort_values(ascending=False)", "explanation": "Groups by tier and calculates the percentage of customers who churned in each"}}

Return ONLY the JSON. No other text.""",
        messages=[{
            "role": "user",
            "content": question
        }]
    )

    response_text = message.content[0].text
    if "```" in response_text:
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]

    parsed = json.loads(response_text.strip())

    # Execute the generated code safely
    try:
        result = eval(parsed["code"], {"df": df, "pd": pd, "__builtins__": {}})
        result_str = str(result)
    except Exception as e:
        result_str = f"Error executing query: {e}"

    # Get a plain-English summary of the result
    summary_msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""The user asked: "{question}"

The pandas code `{parsed['code']}` returned:
{result_str}

Write a clear, concise 1-3 sentence answer to the user's question based on this result. Include specific numbers. Be conversational."""
        }]
    )

    return {
        "question": question,
        "code": parsed["code"],
        "explanation": parsed["explanation"],
        "raw_result": result_str,
        "answer": summary_msg.content[0].text
    }


# ============================================================
# FEATURE 3: CHURN EXPLAINER
# ============================================================
def explain_churn_prediction(customer_data: dict, churn_probability: float) -> str:
    """
    Generate a plain-English explanation of why a customer might churn.

    WHY THIS MATTERS:
    ML models are often "black boxes" — they output a probability but
    don't explain WHY. Business stakeholders need to understand the
    reasoning to take action. This is called "Explainable AI" (XAI).

    HOW IT WORKS:
    We give Claude the customer's actual data + the model's prediction,
    and ask it to explain the prediction in business terms. Claude
    identifies the key risk factors and suggests retention actions.

    PROMPT ENGINEERING TECHNIQUE: "Grounding"
    By providing real data points, we prevent Claude from hallucinating.
    Every claim it makes is traceable to an actual number.

    Args:
        customer_data: dict of customer features and values
        churn_probability: model's predicted churn probability (0-1)

    Returns:
        Human-readable explanation string
    """
    client = get_client()

    # Format customer data nicely
    data_summary = "\n".join([f"  - {k}: {v}" for k, v in customer_data.items()])

    risk_level = "HIGH" if churn_probability > 0.6 else "MEDIUM" if churn_probability > 0.3 else "LOW"

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system="""You are a customer success analyst at an e-commerce company.
Your job is to explain ML churn predictions in plain English that
business stakeholders can understand and act on.

Structure your response as:
1. RISK ASSESSMENT (1 sentence summary)
2. KEY RISK FACTORS (bullet points — reference specific data)
3. RECOMMENDED ACTIONS (2-3 specific retention strategies)

Be specific — reference actual numbers from the data.
Keep it concise — max 200 words.""",
        messages=[{
            "role": "user",
            "content": f"""Explain this churn prediction:

Customer Profile:
{data_summary}

Model Prediction: {churn_probability:.1%} probability of churning
Risk Level: {risk_level}

Why might this customer churn, and what should we do about it?"""
        }]
    )

    return message.content[0].text


# ============================================================
# DEMO / TESTING
# ============================================================
def demo():
    """Run a demo of all three AI features."""
    print("=" * 60)
    print("AI INSIGHTS DEMO")
    print("=" * 60)

    # --- Demo 1: Sentiment Analysis ---
    print("\n--- 1. SENTIMENT ANALYSIS ---")
    sample_reviews = [
        {
            "review_id": "REV_001",
            "review_text": "Absolutely love this product! The material feels premium. Would definitely recommend to friends and family.",
            "rating": 5
        },
        {
            "review_id": "REV_002",
            "review_text": "Terrible experience. The stitching came apart immediately. Customer service was unhelpful when I complained.",
            "rating": 1
        },
        {
            "review_id": "REV_003",
            "review_text": "It's okay for what you pay. The color was slightly different from the photo. Might look for alternatives next time.",
            "rating": 3
        }
    ]

    sentiments = analyze_sentiment(sample_reviews)
    print(json.dumps(sentiments, indent=2))

    # --- Demo 2: Natural Language Querying ---
    print("\n--- 2. NATURAL LANGUAGE QUERYING ---")
    questions = [
        "What is the average spend of churned vs retained customers?",
        "Which subscription tier has the highest churn rate?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        result = query_data_natural_language(q)
        print(f"Code: {result['code']}")
        print(f"Answer: {result['answer']}")

    # --- Demo 3: Churn Explainer ---
    print("\n--- 3. CHURN EXPLAINER ---")
    sample_customer = {
        "subscription_tier": "Free",
        "tenure_days": 120,
        "total_transactions": 3,
        "total_spend": 45.50,
        "total_tickets": 5,
        "ticket_resolution_rate": 0.4,
        "avg_rating": 2.0,
        "total_reviews": 3,
    }

    explanation = explain_churn_prediction(sample_customer, 0.78)
    print(explanation)


if __name__ == "__main__":
    demo()
