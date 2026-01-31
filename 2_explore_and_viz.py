"""
SYSTEM: Sentinel Credit Risk Auditor
MODULE: 2_explore_and_viz.py
ROLE: Exploratory Data Analysis (EDA) & Artifact Generation
STRATEGY: Statistical Integrity / Visualization of Feature Drift
"""
import logging
import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, skew

# 1. INFRASTRUCTURE & TELEMETRY 

LOG_DIR = "logs"
DATA_DIR = "data"
ARTIFACTS_DIR = "outputs"
DB_PATH = os.path.join(DATA_DIR, "sentinel_production.db")

# Ensure artifacts storage exists
if not os.path.exists(ARTIFACTS_DIR): os.makedirs(ARTIFACTS_DIR)
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s |[%(levelname)s] | MODULE:EDA | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR,"sentinel_eda.log")),
        logging.StreamHandler()
    ],
)

logger = logging.getLogger("Sentinel_EDA")

# 2. DATA LOADING (FROM VAULT)

def load_clean_data() -> pd.DataFrame:
    """
    SE4ML: Fetches validated data from the SQLite Prduction DB.
    We do not read CSVs to ensure only audit 'clean' data.
    """

    if not os.path.exists(DB_PATH):
        logger.critical(f"FATAL: Database not found at {DB_PATH}")
        raise FileNotFoundError(f"Production Database Missing!")

    conn = sqlite3.connect(DB_PATH)

    try:
        df = pd.read_sql_query("SELECT * FROM credit_data", conn)
        logger.info(f"Data successfully loaded from DB. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.critical(f"Error loading data from DB: {e}")
        raise e
    finally:
        conn.close()

# 3. SENTINEL VIZ LOGIC

def plot_target_imbalance(df: pd.DataFrame):
    """
    AUDIT 1: Class imbalance requires SMOTE or XGBoost scale_pos_weight.
    """
    plt.figure(figsize=(8,6))
    sns.set_theme(style="whitegrid")

    # calculate ratio

    counts = df['default_payment_next_month'].value_counts()
    imbalance_ratio = counts[0] / counts[1]
    logger.info(f"Class Imbalance Ratio: 0 (Safe):: {counts[0]} | 1 (Default):: {counts[1]} | Ratio:: {imbalance_ratio:.2f}")

    ax = sns.countplot(x='default_payment_next_month', 
                   data=df,
                   hue='default_payment_next_month', 
                   palette='viridis', 
                   legend=False)    
    plt.title(f"Target Variable Distribution (Imbalance Ratio {imbalance_ratio:.1f}:1)", fontsize=14)
    plt.xlabel("Default Payment (0=No, 1=Yes)", fontsize=12)
    plt.ylabel("Count", fontsize=12)

    save_path = os.path.join(ARTIFACTS_DIR, "1_target_imbalance.png")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Target imbalance plot saved to {save_path}")

def plot_correlation_heatmap(df: pd.DataFrame):
    """
    AUDIT 2: Feature Correlation Matrix.
    Justification: Identify multicollinearity among features
    confirm 'PAY_0' predictive power.
    """
    plt.figure(figsize=(12,10))
    # Select only numeic features for correlation
    corr = df.corr()
    # Focus on correlations with target
    target_corr = corr[['default_payment_next_month']].sort_values(by='default_payment_next_month', ascending=False)
    logger.info("Top 5 positive Correlations:\n" + str(target_corr.head(5)))

    # Plot full correlation heatmap
    sns.heatmap(corr, cmap="coolwarm",
               annot= False,
               fmt=".2f",
               linewidths=0.5,)
    plt.title("Feature Correlation Matrix (Spearman)", 
              fontsize=16)
    save_path = os.path.join(ARTIFACTS_DIR, "2_correlation_heatmap_matrix.png")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Artifact: Correlation heatmap matrix saved to {save_path}")
    

def audit_skewness_and_distribution(df: pd.DataFrame, col: str= "limit_bal"):
    """
    AUDIT 3: Statistical Normality & Skewness Check
    
    Justification: If Skew > 1, we must apply Log/ Box-Cox Transformations in training. 
    """
    plt.figure(figsize=(10,6))
    
    # 1. Calculate Statistics
    data_col = df[col]
    skewness = skew(data_col)

    # Shapiro-Wilk Test for Normality
    # Note: for N>5000, Shapiro p-value is almost always 0, 
    # rely more on skewness metric

    logger.info(f"Feature '{col}' Skewness: {skewness:.4f}")

    # 2. Plot Distribution
    sns.histplot(data_col, kde=True, color='navy', bins=50)
    plt.title(f"Distribution of '{col.upper()}' (Skewness: {skewness:.2f})", fontsize=14)

    save_path = os.path.join(ARTIFACTS_DIR, f"3_{col}_distribution.png")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Artifact: Distribution plot for '{col}' saved to {save_path}")

def plot_pay_status_impact(df: pd.DataFrame):
    """
    AUDIT 4: Behavioral Analysis (PAY_) vs. Default) 
    Justification: Verify the 'Recency' hypothesis (recent late payments predict default)
    """

    plt.figure(figsize=(10,6))

    # Group by PAY_0 status and calculate default rates
    # PAY_): -2=No consumption, -1=Paid in Full, 0=Revolving, 1-9=Months delay
    default_rate = df.groupby('pay_0')['default_payment_next_month'].mean() * 100  # percentage

    sns.barplot(x=default_rate.index, 
                y=default_rate.values, 
                palette='magma',
                hue=default_rate.index,
                legend=False)
    plt.title("Default Risk by Repayment Status (PAY_0)", fontsize=14)
    plt.xlabel("Repayment status (Sept 2005)")
    plt.ylabel("Probability of Default (%)")
    plt.axhline(y=df['default_payment_next_month'].mean()*100, 
                color='red', 
                linestyle='--', 
                label='Global Default Rate')
    plt.legend()

    save_path = os.path.join(ARTIFACTS_DIR, "4_pay_0_risk_impact.png")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Artifact: PAY_0 risk impact plot saved to {save_path}")   

# 4. RUN EDA PIPELINE

def run_sentinel_eda():
    logger.info("Initializing Sentinel EDA Pipeline")

    # 1. Load clean data
    try:
        df = load_clean_data()
    except Exception as e:
        return 

    # 2. Run Audits & Generate Artifacts
    logger.info("——— Starting EDA Audits & Artifact Generation ———")
    plot_target_imbalance(df)
    plot_correlation_heatmap(df)
    audit_skewness_and_distribution(df, col="limit_bal")
    audit_skewness_and_distribution(df, col="age")
    plot_pay_status_impact(df)
    logger.info("——— EDA Audits & Artifact Generation Completed ———")

if __name__ == "__main__":
    run_sentinel_eda()  