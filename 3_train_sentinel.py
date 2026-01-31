"""
SYSTEM: Sentinel Credit Risk Auditor
MODULE: 3_train_sentinel.py
ROLE: Model Training & Hyperparameter Optimization
STRATEGY: XGBoost Gradient Boosting / Model Persistence (.pkl)
"""

import os 
import logging 
import sqlite3
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# 1. INFRASTRUCTURE SETUP
LOG_DIR = "logs"
MODEL_DIR = "models"
DATA_DIR = "data"
ARTIFACTS_DIR = "outputs"
DB_PATH = os.path.join(DATA_DIR, "sentinel_production.db")

for d in [LOG_DIR, MODEL_DIR, ARTIFACTS_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s |[%(levelname)s] | MODULE:TRAIN | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "sentinel_train.log")),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger("SentinelTrainer")


# 2. DATA INGESTION & SPLITTING

def load_and_split_data():
    """
    SE4ML: Fetches data and performs Stratified Train-Test Split.
    Stratifcation is non-negotiable due to class imbalance.
    """
    logger.info("Loading data from Production Vault")
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM credit_data"
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Feature Separation 
    target_col = "default_payment_next_month"
    X = df.drop(columns=[target_col, "id"]) # dropping ID to prevent data leakage
    y = df[target_col]

    # Calculate scale weight fr XGBoost (Imbalance Handling)
    # Formula: count(negative_examples) / count(positive_examples)
    scale_weight = (y == 0).sum() / (y == 1).sum()
    logger.info(f"Computed scale weight for XGBoost: {scale_weight:.2f}")

    # Stratified Split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Split Complete: Train Size: {X_train.shape[0]}, Test Size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, scale_weight

# 3. PIPELINE CONSTRUCTION 
def build_pipeline(scale_weight):
    """ 
    SE4ML: The Atomic Pipeline encapsulating Preprocessing and Model Training.
    1. StandardScaler for Feature Scaling: Normalizes 'LIMIT_BAL', 'AGE', etc.
    2. XGBClassifier: Gradient Boosting engine.
    """
   
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            scale_pos_weight=scale_weight,
            eval_metric='logloss',
            random_state=42
        ))
    ])

# 4. EVALUATION & ARTIFACT GENERATION
def evaluate_and_save(pipeline, X_test, y_test):
    """ 
    SE4ML: Evaluates the trained model and saves artifacts.
    Generates Classification Report, Confusion Matrix, and saves the model.
    """
    logger.info("Evaluating Model Performance")

    #1. Predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # 2. Metrics
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    logger.info(f"——— Model Performance ———")
    logger.info(f"ROC-AUC Score: {auc:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info("Classification Report:")
    logger.info("\n" + classification_report(y_test, y_pred))

    # 3. Visual Evidence: Confusion Matrix
    plt.figure(figsize=(8,6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                cbar=False)
    plt.title(f"Sentinel v1 Confusion Matrix (F1: {f1:.2f})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    save_viz_path = os.path.join(ARTIFACTS_DIR, "5_confusion_matrix.png")
    plt.savefig(save_viz_path)
    logger.info(f"Confusion Matrix saved at {save_viz_path}")   

    # 4. Serialization 

    model_path = os.path.join(MODEL_DIR, "sentinel_xgb_model_v1.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    logger.info(f"SUCCESS: Model Pipeline saved at {model_path}")

# 5. EXECUTION
if __name__ == "__main__":
    try:
        # Load 
        X_train, X_test, y_train, y_test, weight = load_and_split_data()
        
        # Build Pipeline
        pipeline = build_pipeline(scale_weight=weight)

        # Train
        logger.info("Engaging XGBoost Model Training")
        pipeline.fit(X_train, y_train)

        # Evaluate & Save Artifacts
        evaluate_and_save(pipeline, X_test, y_test)
        logger.info("Training Pipeline Completed Successfully.")

    except Exception as e:
        logger.error(f"Training Pipeline Failed: {e}")
        raise e