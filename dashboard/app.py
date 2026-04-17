"""
Streamlit Analytics Dashboard
===============================
Interactive web app that brings together analytics, ML predictions,
and AI-powered insights in one place.

WHAT IS STREAMLIT?
Streamlit turns Python scripts into web apps. You write Python,
and Streamlit handles the HTML/CSS/JavaScript. Perfect for data
apps, dashboards, and ML demos.

PAGES:
1. Overview — KPIs, churn distribution, key metrics
2. Customer Explorer — Browse individual customers, get predictions
3. AI Insights — Chat with your data, sentiment analysis
4. Model Performance — Model comparison, feature importance

HOW TO RUN:
  cd ecommerce-churn-ai
  streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import sys

# Add project root to path so we can import our modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.feature_engineering import create_derived_features, encode_categoricals

# ============================================================
# CONFIGURATION & DATA LOADING
# ============================================================
MODEL_DIR = os.path.join(project_root, "models")
DATA_DIR = os.path.join(project_root, "data")

# Page config — must be the first Streamlit command
st.set_page_config(
    page_title="Churn Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_data():
    """Load and cache all datasets. @cache_data prevents reloading on every interaction."""
    clean = pd.read_csv(os.path.join(DATA_DIR, "clean_dataset.csv"))
    reviews = pd.read_csv(os.path.join(DATA_DIR, "reviews.csv"))
    transactions = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"), parse_dates=["transaction_date"])
    return clean, reviews, transactions


@st.cache_resource
def load_model():
    """Load the trained model and scaler."""
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.joblib"))
    metadata = joblib.load(os.path.join(MODEL_DIR, "model_metadata.joblib"))
    return model, scaler, feature_cols, metadata


# Load everything
df, reviews_df, transactions_df = load_data()
model, scaler, feature_cols, model_metadata = load_model()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Overview", "Customer Explorer", "AI Insights", "Model Performance"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "E-Commerce Churn Prediction & AI Analytics Platform. "
    "Built with Python, scikit-learn, Claude AI, and Streamlit."
)


# ============================================================
# PAGE 1: OVERVIEW DASHBOARD
# ============================================================
if page == "Overview":
    st.title("E-Commerce Churn Analytics Dashboard")
    st.markdown("Real-time overview of customer health and churn risk.")

    # --- KPI Cards ---
    col1, col2, col3, col4, col5 = st.columns(5)

    total_customers = len(df)
    churned = df["churned"].sum()
    churn_rate = df["churned"].mean()
    avg_spend = df["total_spend"].mean()
    avg_tickets = df["total_tickets"].mean()

    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Churned", f"{churned:,}", delta=f"{churn_rate:.1%}", delta_color="inverse")
    col3.metric("Avg Spend", f"${avg_spend:,.0f}")
    col4.metric("Avg Tickets", f"{avg_tickets:.1f}")
    col5.metric("Avg Rating", f"{df['avg_rating'].mean():.1f}/5.0")

    st.markdown("---")

    # --- Row 1: Churn Distribution + Tier Analysis ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Distribution")
        fig = px.pie(
            values=df["churned"].value_counts().values,
            names=["Retained", "Churned"],
            color_discrete_sequence=["#2ecc71", "#e74c3c"],
            hole=0.4
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Churn Rate by Subscription Tier")
        tier_data = df.groupby("subscription_tier")["churned"].agg(["mean", "count"]).reset_index()
        tier_data.columns = ["Tier", "Churn Rate", "Count"]
        tier_order = ["Free", "Basic", "Premium", "Enterprise"]
        tier_data["Tier"] = pd.Categorical(tier_data["Tier"], categories=tier_order, ordered=True)
        tier_data = tier_data.sort_values("Tier")

        fig = px.bar(
            tier_data, x="Tier", y="Churn Rate",
            color="Churn Rate", color_continuous_scale=["#2ecc71", "#e74c3c"],
            text=tier_data["Churn Rate"].apply(lambda x: f"{x:.1%}")
        )
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # --- Row 2: Spend Distribution + Transactions Over Time ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Spend Distribution: Churned vs Retained")
        fig = px.box(
            df, x="churned", y="total_spend",
            color="churned", color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
            labels={"churned": "Churned", "total_spend": "Total Spend ($)"}
        )
        fig.update_layout(height=350, showlegend=False)
        fig.update_xaxes(ticktext=["Retained", "Churned"], tickvals=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Monthly Transactions")
        monthly = transactions_df.groupby(transactions_df["transaction_date"].dt.to_period("M")).size()
        monthly.index = monthly.index.astype(str)
        fig = px.line(
            x=monthly.index, y=monthly.values,
            labels={"x": "Month", "y": "Transactions"}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # --- Row 3: Feature Correlations ---
    st.subheader("Feature Correlations with Churn")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_with_churn = df[numeric_cols].corr()["churned"].drop("churned").sort_values()

    fig = px.bar(
        x=corr_with_churn.values, y=corr_with_churn.index,
        orientation="h",
        color=corr_with_churn.values,
        color_continuous_scale="RdBu_r",
        labels={"x": "Correlation with Churn", "y": "Feature"}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 2: CUSTOMER EXPLORER
# ============================================================
elif page == "Customer Explorer":
    st.title("Customer Explorer")
    st.markdown("Browse individual customers and get real-time churn predictions.")

    # Customer selector
    customer_ids = df["customer_id"].tolist()
    selected_id = st.selectbox("Select a Customer", customer_ids)

    customer = df[df["customer_id"] == selected_id].iloc[0]

    # --- Customer Profile ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Profile")
        st.write(f"**ID:** {customer['customer_id']}")
        st.write(f"**Age:** {customer['age']}")
        st.write(f"**Gender:** {customer['gender']}")
        st.write(f"**City:** {customer['city']}")
        st.write(f"**Tier:** {customer['subscription_tier']}")
        st.write(f"**Tenure:** {customer['tenure_days']} days")

    with col2:
        st.subheader("Activity")
        st.write(f"**Transactions:** {customer['total_transactions']}")
        st.write(f"**Total Spend:** ${customer['total_spend']:,.2f}")
        st.write(f"**Avg Transaction:** ${customer['avg_transaction_amount']:,.2f}")
        st.write(f"**Categories:** {customer['unique_categories']}")
        st.write(f"**Days Since Purchase:** {customer['days_since_last_purchase']}")

    with col3:
        st.subheader("Support & Reviews")
        st.write(f"**Tickets:** {int(customer['total_tickets'])}")
        st.write(f"**Resolution Rate:** {customer['ticket_resolution_rate']:.0%}")
        st.write(f"**Reviews:** {int(customer['total_reviews'])}")
        st.write(f"**Avg Rating:** {customer['avg_rating']:.1f}/5.0")
        actual = "Yes" if customer["churned"] == 1 else "No"
        st.write(f"**Actually Churned:** {actual}")

    st.markdown("---")

    # --- Churn Prediction ---
    st.subheader("ML Churn Prediction")

    # Prepare features for prediction
    cust_df = df[df["customer_id"] == selected_id].copy()
    cust_engineered = create_derived_features(cust_df)
    cust_engineered = encode_categoricals(cust_engineered)

    # Build feature vector
    feature_vector = cust_engineered[feature_cols].values
    scaled_features = scaler.transform(feature_vector)

    # Predict
    churn_prob = model.predict_proba(scaled_features)[0, 1]
    churn_pred = "Will Churn" if churn_prob >= 0.5 else "Will Stay"
    risk_level = "HIGH" if churn_prob > 0.6 else "MEDIUM" if churn_prob > 0.3 else "LOW"

    col1, col2, col3 = st.columns(3)
    col1.metric("Churn Probability", f"{churn_prob:.1%}")
    col2.metric("Prediction", churn_pred)

    risk_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
    col3.metric("Risk Level", f"{risk_color[risk_level]} {risk_level}")

    # Gauge chart for churn probability
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_prob * 100,
        title={"text": "Churn Risk Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#e74c3c" if churn_prob > 0.5 else "#f39c12" if churn_prob > 0.3 else "#2ecc71"},
            "steps": [
                {"range": [0, 30], "color": "#d5f5e3"},
                {"range": [30, 60], "color": "#fdebd0"},
                {"range": [60, 100], "color": "#fadbd8"},
            ],
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    # --- Customer Reviews ---
    st.subheader("Customer Reviews")
    cust_reviews = reviews_df[reviews_df["customer_id"] == selected_id]
    if len(cust_reviews) > 0:
        for _, rev in cust_reviews.iterrows():
            stars = "⭐" * int(rev["rating"])
            st.markdown(f"**{stars}** — *{rev['product']}* ({rev['review_date']})")
            st.markdown(f"> {rev['review_text']}")
            st.markdown("")
    else:
        st.info("This customer hasn't written any reviews.")


# ============================================================
# PAGE 3: AI INSIGHTS
# ============================================================
elif page == "AI Insights":
    st.title("AI-Powered Insights")
    st.markdown("Leverage Claude AI for sentiment analysis, data querying, and churn explanations.")

    tab1, tab2, tab3 = st.tabs(["Ask Your Data", "Sentiment Analysis", "Churn Explainer"])

    # --- Tab 1: Natural Language Querying ---
    with tab1:
        st.subheader("Ask Questions About Your Data")
        st.markdown("Type a question in plain English, and AI will query the data for you.")

        # Example questions
        st.markdown("**Example questions:**")
        examples = [
            "What is the average spend of churned vs retained customers?",
            "Which city has the highest churn rate?",
            "What's the churn rate for customers with more than 5 support tickets?",
            "How many customers are on each subscription tier?",
        ]
        for ex in examples:
            st.markdown(f"- *{ex}*")

        question = st.text_input("Your question:", placeholder="e.g., What is the churn rate by city?")

        if st.button("Ask AI", key="query_btn"):
            if question:
                with st.spinner("AI is analyzing your data..."):
                    try:
                        from src.ai_insights import query_data_natural_language
                        result = query_data_natural_language(question, df)
                        st.success(result["answer"])
                        with st.expander("See generated code"):
                            st.code(result["code"], language="python")
                            st.markdown(f"*{result['explanation']}*")
                        with st.expander("See raw result"):
                            st.text(result["raw_result"])
                    except Exception as e:
                        st.error(f"Error: {e}. Make sure ANTHROPIC_API_KEY is set in .env")
            else:
                st.warning("Please enter a question.")

    # --- Tab 2: Sentiment Analysis ---
    with tab2:
        st.subheader("Review Sentiment Analysis")
        st.markdown("Analyze customer reviews using Claude AI to extract sentiment, themes, and churn risk.")

        # Let user pick reviews to analyze
        num_reviews = st.slider("Number of reviews to analyze", 1, 10, 3)

        sample_reviews = reviews_df.sample(num_reviews, random_state=42)

        st.markdown("**Selected reviews:**")
        for _, rev in sample_reviews.iterrows():
            stars = "⭐" * int(rev["rating"])
            st.markdown(f"{stars} — *{rev['review_text'][:100]}...*")

        if st.button("Analyze Sentiment", key="sentiment_btn"):
            with st.spinner("Claude is analyzing reviews..."):
                try:
                    from src.ai_insights import analyze_sentiment
                    review_list = sample_reviews[["review_id", "review_text", "rating"]].to_dict("records")
                    results = analyze_sentiment(review_list)

                    for i, result in enumerate(results):
                        with st.expander(f"Review {i+1}: {result.get('sentiment', 'N/A').upper()}"):
                            col1, col2 = st.columns(2)
                            col1.metric("Sentiment", result.get("sentiment", "N/A"))
                            col2.metric("Churn Risk", result.get("churn_risk", "N/A"))
                            st.write(f"**Confidence:** {result.get('confidence', 'N/A')}")
                            st.write(f"**Emotional Tone:** {result.get('emotional_tone', 'N/A')}")
                            st.write(f"**Key Themes:** {', '.join(result.get('key_themes', []))}")
                except Exception as e:
                    st.error(f"Error: {e}. Make sure ANTHROPIC_API_KEY is set in .env")

    # --- Tab 3: Churn Explainer ---
    with tab3:
        st.subheader("AI Churn Explanation")
        st.markdown("Select a customer and get an AI-generated explanation of their churn risk.")

        customer_id = st.selectbox("Select Customer", df["customer_id"].tolist(), key="explain_select")
        customer = df[df["customer_id"] == customer_id].iloc[0]

        # Get prediction
        cust_df = df[df["customer_id"] == customer_id].copy()
        cust_engineered = create_derived_features(cust_df)
        cust_engineered = encode_categoricals(cust_engineered)
        feature_vector = cust_engineered[feature_cols].values
        scaled_features = scaler.transform(feature_vector)
        churn_prob = model.predict_proba(scaled_features)[0, 1]

        st.write(f"**Churn Probability:** {churn_prob:.1%}")

        if st.button("Generate AI Explanation", key="explain_btn"):
            with st.spinner("Claude is generating explanation..."):
                try:
                    from src.ai_insights import explain_churn_prediction
                    customer_data = {
                        "subscription_tier": customer["subscription_tier"],
                        "tenure_days": int(customer["tenure_days"]),
                        "total_transactions": int(customer["total_transactions"]),
                        "total_spend": float(customer["total_spend"]),
                        "total_tickets": int(customer["total_tickets"]),
                        "ticket_resolution_rate": float(customer["ticket_resolution_rate"]),
                        "avg_rating": float(customer["avg_rating"]),
                        "total_reviews": int(customer["total_reviews"]),
                    }
                    explanation = explain_churn_prediction(customer_data, churn_prob)
                    st.markdown(explanation)
                except Exception as e:
                    st.error(f"Error: {e}. Make sure ANTHROPIC_API_KEY is set in .env")


# ============================================================
# PAGE 4: MODEL PERFORMANCE
# ============================================================
elif page == "Model Performance":
    st.title("Model Performance")
    st.markdown("Detailed model evaluation and feature importance analysis.")

    # Model info
    st.subheader("Best Model")
    col1, col2 = st.columns(2)
    col1.metric("Algorithm", model_metadata["model_name"])
    metrics = model_metadata["metrics"]
    col2.metric("F1 Score", f"{metrics['F1 Score']:.4f}")

    # Metrics table
    st.subheader("Evaluation Metrics")
    metrics_df = pd.DataFrame([metrics]).T
    metrics_df.columns = ["Score"]
    st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)

    # Model results image
    results_img_path = os.path.join(DATA_DIR, "model_results.png")
    if os.path.exists(results_img_path):
        st.subheader("Detailed Results")
        st.image(results_img_path, use_container_width=True)

    # Feature importance
    st.subheader("Feature Importance")
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_cols)
    else:
        importances = pd.Series(np.abs(model.coef_[0]), index=feature_cols)

    top_features = importances.nlargest(15).sort_values(ascending=True)

    fig = px.bar(
        x=top_features.values, y=top_features.index,
        orientation="h",
        labels={"x": "Importance", "y": "Feature"},
        color=top_features.values,
        color_continuous_scale="Blues"
    )
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # EDA charts
    st.subheader("EDA Visualizations")
    chart_files = ["churn_distribution.png", "feature_distributions.png",
                   "correlation_matrix.png", "tier_analysis.png", "churn_comparisons.png"]
    for chart_file in chart_files:
        chart_path = os.path.join(DATA_DIR, chart_file)
        if os.path.exists(chart_path):
            st.image(chart_path, caption=chart_file.replace("_", " ").replace(".png", "").title(),
                     use_container_width=True)
