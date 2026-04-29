import os

import pandas as pd
import plotly.express as px
import streamlit as st

from src.predict import predict_transaction


st.set_page_config(page_title="Fraud Detection System", layout="wide")
st.title("Credit Card Fraud Detection System")
st.caption("Beginner-friendly, production-like demo with model comparison and explanations.")


@st.cache_data
def load_dashboard_data():
    path = "data/cleaned_creditcard.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    raw = "data/creditcard.csv"
    if os.path.exists(raw):
        return pd.read_csv(raw)
    return None


df_dash = load_dashboard_data()

tab1, tab2 = st.tabs(["Prediction", "Dashboard"])

with tab1:
    st.subheader("Real-time Transaction Prediction")
    st.write("Enter values for a transaction and click Predict.")

    col1, col2 = st.columns(2)
    with col1:
        time_val = st.number_input("Time", min_value=0.0, value=10000.0, step=1.0)
        amount_val = st.number_input("Amount", min_value=0.0, value=50.0, step=1.0)
    with col2:
        threshold = st.slider("Fraud Probability Threshold", 0.1, 0.9, 0.5, 0.05)

    st.markdown("### PCA Features (V1 to V28)")
    user_features = {}
    feature_cols = st.columns(4)
    for i in range(1, 29):
        with feature_cols[(i - 1) % 4]:
            user_features[f"V{i}"] = st.number_input(f"V{i}", value=0.0, step=0.1, format="%.4f")

    input_payload = {"Time": time_val, **user_features, "Amount": amount_val}

    if st.button("Predict", type="primary"):
        try:
            result = predict_transaction(input_payload, threshold=threshold)
            prob = result["fraud_probability"]
            if result["prediction"] == "Fraud":
                st.error(f"Prediction: FRAUD | Probability: {prob:.2%}")
                st.warning("ALERT: This transaction should be reviewed immediately.")
                st.info("Email simulation: risk-team@example.com has been notified.")
            else:
                st.success(f"Prediction: NOT FRAUD | Probability: {prob:.2%}")

            st.write(f"Model used: `{result['model_name']}`")
            st.markdown("### Why this was flagged")
            for reason in result["explanation"]:
                st.write(f"- {reason}")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

with tab2:
    st.subheader("Fraud Monitoring Dashboard")
    if df_dash is None:
        st.info("Run model training first and place dataset at data/creditcard.csv to view dashboard.")
    else:
        counts = df_dash["Class"].value_counts().rename(index={0: "Normal", 1: "Fraud"}).reset_index()
        counts.columns = ["Class", "Count"]
        fig_pie = px.pie(counts, names="Class", values="Count", title="Fraud vs Normal")
        st.plotly_chart(fig_pie, use_container_width=True)

        sampled = df_dash.sample(min(len(df_dash), 20000), random_state=42).copy()
        sampled["ClassLabel"] = sampled["Class"].map({0: "Normal", 1: "Fraud"})
        fig_hist = px.histogram(
            sampled,
            x="Amount",
            color="ClassLabel",
            barmode="overlay",
            nbins=80,
            title="Transaction Amount Distribution (Sample)",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.dataframe(counts, use_container_width=True)

