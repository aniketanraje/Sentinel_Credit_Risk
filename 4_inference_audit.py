"""
SYSTEM: Sentinel Credit Risk Auditor
MODULE: 4_inference_audit.py
ROLE: Batch Inference & Risk Scoring
STRATEGY: Automated Auditing of Model Decisions
"""

import sqlite3
import pandas as pd
import pickle
import os

# Paths
DB_PATH = "data/sentinel_production.db"
MODEL_PATH = "models/sentinel_optimized.pkl"
REPORT_PATH = "outputs/risk_audit_report.csv"

def run_audit():
    # 1. Load the "Alpha" Brain
    if not os.path.exists(MODEL_PATH):
        print("❌ Error: Optimized model not found!")
        return
        
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    # 2. Pull Data from SQLite
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM credit_data", conn)
    conn.close()

    # 3. Clean for Inference (Drop target and ID)
    features = df.drop(columns=['default_payment_next_month', 'id'], errors='ignore')

    # 4. Process the Batch (Milking the M3)
    print(f"🚀 Scanning {len(df)} accounts for credit risk...")
    df['risk_probability'] = model.predict_proba(features)[:, 1]
    df['prediction'] = model.predict(features)

    # 5. Extract "The Dangerous Ten"
    top_risk = df.sort_values(by='risk_probability', ascending=False).head(10)

    print("\n--- ⚠️ TOP 10 HIGH-RISK ACCOUNTS ---")
    print(top_risk[['id', 'limit_bal', 'age', 'risk_probability']])

    # 6. Save Audit
    os.makedirs('outputs', exist_ok=True)
    df.to_csv(REPORT_PATH, index=False)
    print(f"\n✅ Full audit saved to: {REPORT_PATH}")

if __name__ == "__main__":
    run_audit()