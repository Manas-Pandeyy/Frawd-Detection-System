import json
import os
from typing import Dict, Tuple

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

from src.preprocessing import basic_cleaning, load_data, split_and_scale


def evaluate_model(model, X_test, y_test) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    metrics = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "report": classification_report(y_test, y_pred, zero_division=0),
    }
    return metrics


def train_all_models(dataset_path: str = "data/creditcard.csv") -> Tuple[str, Dict]:
    os.makedirs("models", exist_ok=True)

    df = load_data(dataset_path)
    df = basic_cleaning(df)
    bundle = split_and_scale(df)

    # Handle imbalance only on training split.
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(bundle.X_train, bundle.y_train)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1500, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=250, max_depth=None, random_state=42, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        ),
    }

    comparison = {}
    best_model_name = None
    best_model_obj = None
    best_recall = -1.0

    for name, model in models.items():
        model.fit(X_train_balanced, y_train_balanced)
        metrics = evaluate_model(model, bundle.X_test, bundle.y_test)
        comparison[name] = metrics

        print(f"\nModel: {name}")
        print(metrics["report"])

        # Priority is recall for fraud detection.
        if metrics["recall"] > best_recall:
            best_recall = metrics["recall"]
            best_model_name = name
            best_model_obj = model

        joblib.dump(model, f"models/{name}.joblib")

    # Save shared artifacts used by prediction app.
    joblib.dump(bundle.scaler, "models/scaler.joblib")
    joblib.dump(bundle.feature_names, "models/feature_names.joblib")

    with open("models/model_comparison.json", "w", encoding="utf-8") as fp:
        json.dump(comparison, fp, indent=2)

    joblib.dump(best_model_obj, "models/best_model.joblib")
    with open("models/best_model_name.txt", "w", encoding="utf-8") as fp:
        fp.write(best_model_name)

    # Keep a cleaned copy for dashboard/analysis.
    df.to_csv("data/cleaned_creditcard.csv", index=False)

    print(f"\nBest model selected (by recall): {best_model_name}")
    return best_model_name, comparison


if __name__ == "__main__":
    train_all_models()

