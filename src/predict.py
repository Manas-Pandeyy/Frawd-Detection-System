from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd


def load_artifacts() -> Tuple[object, object, List[str], str]:
    model = joblib.load("models/best_model.joblib")
    scaler = joblib.load("models/scaler.joblib")
    feature_names = joblib.load("models/feature_names.joblib")
    with open("models/best_model_name.txt", "r", encoding="utf-8") as fp:
        model_name = fp.read().strip()
    return model, scaler, feature_names, model_name


def predict_transaction(input_data: Dict[str, float], threshold: float = 0.5) -> Dict:
    model, scaler, feature_names, model_name = load_artifacts()

    row = pd.DataFrame([input_data], columns=feature_names)
    row_scaled = scaler.transform(row)

    fraud_probability = float(model.predict_proba(row_scaled)[0][1])
    pred_label = int(fraud_probability >= threshold)
    prediction = "Fraud" if pred_label == 1 else "Not Fraud"

    explanation = explain_prediction(model, model_name, row_scaled[0], feature_names)

    if pred_label == 1:
        print(f"ALERT: Suspicious transaction detected. Risk score = {fraud_probability:.4f}")
        print("Email simulation -> To: risk-team@example.com | Subject: Fraud Alert")

    return {
        "prediction": prediction,
        "fraud_probability": fraud_probability,
        "model_name": model_name,
        "explanation": explanation,
    }


def explain_prediction(model, model_name: str, scaled_values: np.ndarray, feature_names: List[str]) -> List[str]:
    # Beginner-friendly explanation:
    # Build a simple contribution score and show top factors.
    if model_name == "logistic_regression" and hasattr(model, "coef_"):
        contrib = model.coef_[0] * scaled_values
    elif hasattr(model, "feature_importances_"):
        contrib = model.feature_importances_ * np.abs(scaled_values)
    else:
        contrib = np.abs(scaled_values)

    top_idx = np.argsort(np.abs(contrib))[::-1][:5]
    reasons = [
        f"{feature_names[i]} had strong impact (score={contrib[i]:.4f})" for i in top_idx
    ]
    return reasons

