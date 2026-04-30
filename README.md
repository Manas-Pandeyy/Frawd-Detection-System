# End-to-End Fraud Detection System (Python + Streamlit)

This project is a complete beginner-friendly fraud detection system using an **imbalanced credit card dataset**.

Developed an end-to-end Fraud Detection System to identify suspicious financial transactions using Machine Learning techniques.

🔹 Data Preprocessing:
Performed data cleaning, handled missing values, and normalized numerical features using StandardScaler. Removed duplicate entries to improve data quality.

🔹 Handling Imbalanced Data:
Applied SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset and improve model performance on minority (fraud) class.

🔹 Exploratory Data Analysis (EDA):
Conducted in-depth EDA using Matplotlib and Seaborn to understand transaction patterns, fraud distribution, and feature relationships through visualizations.

🔹 Model Building:
Trained and compared multiple Machine Learning models:

* Logistic Regression
* Random Forest Classifier
* XGBoost Classifier

Evaluation was focused on Recall, Precision, and F1-score, with special emphasis on Recall to minimize false negatives (i.e., undetected fraud cases).

🔹 Model Selection:
Selected the best-performing model based on evaluation metrics and saved it using Joblib for future predictions.

🔹 Real-Time Prediction System:
Built a prediction pipeline where users can input transaction details (amount, time, etc.) and receive:

* Fraud / Not Fraud classification
* Probability score for better interpretability

🔹 User Interface (UI):
Developed an interactive web application using Streamlit:

* User-friendly input fields
* Real-time prediction button
* Clear output display with alerts

🔹 Advanced Features:

* Dashboard showing fraud vs non-fraud transactions
* Alert system for detected fraud cases
* Model explainability using feature importance

🔹 Tech Stack:

* Python, Pandas, NumPy
* Scikit-learn, XGBoost
* Matplotlib, Seaborn
* Streamlit (for UI)
* Joblib (model saving)

🔹 Outcome:
Successfully built a production-like ML system capable of detecting fraudulent transactions with high recall, ensuring minimal risk of missing fraud cases.

This project demonstrates strong understanding of data preprocessing, imbalance handling, model evaluation, and real-world deployment of machine learning systems.


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


Demo Link : https://frawd-detection-system-drcaxsymgfxxpk4nmfgjvr.streamlit.app/
