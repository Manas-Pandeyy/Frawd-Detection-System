# End-to-End Fraud Detection System (Python + Streamlit)

This project is a complete beginner-friendly fraud detection system using an **imbalanced credit card dataset**.

It includes:
- Data preprocessing (missing values, scaling, normalization)
- EDA plots
- Imbalance handling using **SMOTE**
- Model training and comparison (**Logistic Regression**, **Random Forest**, **XGBoost**)
- Best model selection (based on **recall**)
- Real-time prediction with probability
- Streamlit UI dashboard + alert simulation + basic explanation

---

## 1) Project Structure

```text
FraudDetectionSystem/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── creditcard.csv   # add this file
├── models/
├── reports/
└── src/
    ├── __init__.py
    ├── preprocessing.py
    ├── eda.py
    ├── model_training.py
    ├── predict.py
    └── run_pipeline.py
```

---

## 2) Step-by-Step Commands (Windows PowerShell)

Run these commands exactly in order:

### Step 1: Open project folder
```powershell
cd D:\FraudDetectionSystem
```

### Step 2: Create virtual environment
```powershell
python -m venv .venv
```

### Step 3: Activate virtual environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install dependencies
```powershell
pip install -r requirements.txt
```

### Step 5: Add dataset
Place `creditcard.csv` inside:
```text
D:\FraudDetectionSystem\data\creditcard.csv
```

Dataset suggestion: Kaggle "Credit Card Fraud Detection".

### Step 6: Run EDA + preprocessing + model training
```powershell
python -m src.run_pipeline
```

This generates:
- EDA graphs in `reports/`
- trained models in `models/`
- model comparison file in `models/model_comparison.json`

### Step 7: Launch Streamlit app
```powershell
streamlit run app.py
```

Open the local URL shown in terminal (usually `http://localhost:8501`).

---

## 3) How Prediction Works

In Streamlit:
1. Enter transaction values (`Time`, `Amount`, `V1` to `V28`)
2. Click **Predict**
3. App shows:
   - Fraud / Not Fraud
   - Probability score
   - Alert simulation if fraud
   - Top features influencing decision

---

## 4) Model Comparison Criteria

Models are compared with:
- Precision
- Recall (**highest priority**)
- F1-score

Best model is selected by highest **Recall**, because in fraud detection, missing fraud is costly.

---

## 5) Notes

- `creditcard.csv` is highly imbalanced (fraud is very small percentage).
- SMOTE is applied only on training data (correct approach).
- Scaling is fit on training and reused for test + prediction.
- This is beginner-friendly but production-like in structure and flow.

