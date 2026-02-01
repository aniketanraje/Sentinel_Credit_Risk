
import sqlite3
import pandas as pd
import pickle

# Config
DB_PATH = 'data/sentinel_production.db'
MODEL_PATH = 'models/sentinel_optimized.pkl'

def flash_check():
    print("\n🔍 --- SENTINEL DB AUDIT: START ---")
    
    # 1. Load Model Requirements
    try:
        with open(MODEL_PATH, 'rb') as f:
            pipeline = pickle.load(f)
        expected = [c.lower() for c in pipeline.feature_names_in_]
        print(f"✅ Model Expects: {len(expected)} features (lowercase).")
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        return

    # 2. Inspect DB Schema
    conn = sqlite3.connect(DB_PATH)
    db_df = pd.read_sql("SELECT * FROM credit_data LIMIT 5", conn)
    actual_cols = list(db_df.columns)
    conn.close()

    print(f"📊 DB Table Found: {len(actual_cols)} columns.")
    
    # 3. The "Shit" List (Mismatches)
    missing = [c for c in expected if c not in [a.lower() for a in actual_cols]]
    extra = [c for c in actual_cols if c.lower() not in expected]
    
    print("\n--- RESULTS ---")
    if missing:
        print(f"🚨 MISSING FROM DB: {missing}")
    else:
        print("✅ No missing features.")

    if extra:
        print(f"⚠️ EXTRA TRASH IN DB (Will break XGBoost): {extra}")
    
    # 4. Data Type Check
    print("\n--- TYPE CHECK (First 5 Rows) ---")
    print(db_df.dtypes)
    
    # Check for strings that should be numbers
    string_cols = db_df.select_dtypes(include=['object']).columns.tolist()
    if string_cols:
        print(f"🚨 STRING ALERT: Columns {string_cols} are strings. Model needs Floats/Ints.")
    else:
        print("✅ Data types look numeric.")

    print("\n--- AUDIT COMPLETE ---\n")

if __name__ == "__main__":
    flash_check()