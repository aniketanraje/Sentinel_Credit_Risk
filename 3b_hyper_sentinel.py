"""
SYSTEM: Sentinel Credit Risk Auditor
MODULE: 3b_hyper_sentinel.py
ROLE: Hyperparameter Optimization (HPO)
"""
import os
import pickle
import sqlite3
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Setup
DB_PATH = "data/sentinel_production.db"
MODEL_PATH = "models/sentinel_optimized.pkl"

# 1. Load Data
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM credit_data", conn)
conn.close()

X = df.drop(columns=['default_payment_next_month', 'id'])
y = df['default_payment_next_month']
scale_weight = (y == 0).sum() / (y == 1).sum()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 2. Define Pipeline
base_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', XGBClassifier(eval_metric='logloss', scale_pos_weight=scale_weight, random_state=42))
])

# 3. Define Search Space (The "Pro" Grid)
param_dist = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [3, 4, 5, 6],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__subsample': [0.8, 0.9]
}

# 4. Execute Search
print("Initiating Search... Grab a coffee.")
search = RandomizedSearchCV(
    base_pipe, param_distributions=param_dist, 
    n_iter=10, cv=3, scoring='f1', n_jobs=-1, verbose=1
)
search.fit(X_train, y_train)

# 5. Save the "Alpha" Model
best_model = search.best_estimator_
print(f"Best Params: {search.best_params_}")
print("\n--- OPTIMIZED PERFORMANCE ---")
print(classification_report(y_test, best_model.predict(X_test)))

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(best_model, f)