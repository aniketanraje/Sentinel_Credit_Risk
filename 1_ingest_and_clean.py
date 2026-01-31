"""
SYSTEM: Sentinel Credit Risk Auditor
MODULE: 1_ingest_and_clean.py
ROLE: Data Ingestion & Pydantic Validation (SE4ML)
STRATEGY: Defensive Programming / Data Contract Enforcement
"""

import os
import logging
import sqlite3
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from typing import Optional

# 1. INFRASTRUCTURE & TELEMETRY SETUP 

LOG_DIR = "logs"
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "sentinel_production.db")
RAW_DATA_PATH = os.path.join(DATA_DIR, "Credit_Card_Default.csv")

if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

# Configure 'Sentinel' Auditor 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s |[%(levelname)s] | MODULE:INGEST | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR,"sentinel_ingest.log")),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger("Sentinel")


# 2. THE DATA CONTRACT

class CreditRecord(BaseModel):
    """
    SE$ML: The Pydantic Model acts as the 'Gatekeeper'.
    Any row not meeting these strict physical criteria is rejcted 
    """

    id: int  # FIXED: lowercase to match cleaner output
    limit_bal: float = Field(ge=0, description="Credit limit must be non-negative")
    sex: int = Field(ge=1, le=2, description="1=male, 2=female")
    education: int = Field(ge=0, le=6, description="Education tier 0-6")
    marriage: int = Field(ge=0, le=3, description="Marital status 0-3")
    age: int = Field(ge=18, le=100, description="legal age range 18-100")

    # Repayment status (-2=NO consumption, -1=Paid in Full, 0=Revolving, 1-9=Months delaty)
    pay_0: int 
    pay_2: int
    pay_3: int
    pay_4: int
    pay_5: int
    pay_6: int


    # Bill amounts ( can be negative if overpaid)
    bill_amt1: float
    bill_amt2: float
    bill_amt3: float
    bill_amt4: float
    bill_amt5: float
    bill_amt6: float

    # Previous payment amounts (Must be non-negative)
    pay_amt1: float = Field(ge=0)
    pay_amt2: float = Field(ge=0)
    pay_amt3: float = Field(ge=0)
    pay_amt4: float = Field(ge=0)
    pay_amt5: float = Field(ge=0)
    pay_amt6: float = Field(ge=0)

    # Target (0=No Default, 1=Default)
    default_payment_next_month: int = Field(ge=0, le=1)


# 3. THE PIPELINE LOGIC 

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    SE4ML: Normalize messy CSV headers to strict snake_case to match Pydantic model
    1. Removes double quotes from header strings.
    2. Converts all to lowercase.
    3. Maps the dot-notation target to our snake_case contract.
    Args:
        df (pd.DataFrame): Raw dataframe with original column names
    Returns:
        pd.DataFrame: Dataframe with cleaned column names
    """
    # FIXED: Added quote and dot replacement
    df.columns = [col.replace('"', '').replace('.', '_').strip().lower() for col in df.columns]
    df.columns = df.columns.str.replace(" ", "_", regex=False)
    
    # Fix specific discrepance in dataset 
    if 'default_payment_next_month' not in df.columns:
        # FIXED: Removed the 's' from the source key to match actual CSV dots-to-underscore conversion
        df.rename(columns={'default_payment_next_month': 'default_payment_next_month'}, inplace=True)
    return df

def run_ingestion():
    logger.info("Initializing Sentinel Ingestion Pipeline")

    # 1. Validation: Check Raw Data Existence
    if not os.path.exists(RAW_DATA_PATH):
        logger.critical(f"FATAL: Raw data not found at {RAW_DATA_PATH}")
        return
    
    # 2. Loading Raw Data 
    try:
        # Added low_memory=False to ensure types are consistent
        df_raw = pd.read_csv(RAW_DATA_PATH, low_memory=False)
        logger.info(f"Raw data loaded with shape: {df_raw.shape}")
    except Exception as e:
        logger.critical(f"FATAL: Error loading raw data - {e}")
        return
    
    # 3. Sanitization 

    df_clean = clean_column_names(df_raw)

    # 4. Strict Pydantic Validation (The Loop)
    records = df_clean.to_dict(orient="records")
    validated_rows = []
    failed_rows = 0

    for i,row in enumerate(records):
        try:
            # The Gatekeeper: Instantiating the model forces validatiion
            valid_record = CreditRecord(**row)
            validated_rows.append(valid_record.model_dump())
        except ValidationError as ve:
            failed_rows += 1
            if failed_rows <= 5: # Log first 5 errors only
                logger.warning(f"Row {i} failed validation: {ve}")
    
    logger.info(f"Validation complete. {len(validated_rows)} records valid, {failed_rows} records failed.")

    # 5. Persistence (The Valut)
    if validated_rows:
        try:
            conn = sqlite3.connect(DB_PATH)
            df_final = pd.DataFrame(validated_rows)
            df_final.to_sql("credit_data", conn, if_exists="replace", index=False)
            conn.close()
            logger.info(f"SUCCESS: {len(df_final)} records ingested into database at {DB_PATH}")
        except Exception as e:
            logger.critical(f"FATAL: Error saving to database - {e}")
    else:
        logger.error("No valid record found. Database update aborted.")

if __name__ == "__main__":
    run_ingestion()