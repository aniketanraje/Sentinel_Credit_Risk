# Sentinel Credit Risk System

## Overview

This repository contains an end-to-end, system-oriented machine learning project for credit default risk assessment using the Taiwan Credit Default dataset.

The objective of the project is not to optimize a single predictive metric or demonstrate algorithmic novelty. Instead, the focus is on disciplined construction of a **reproducible, auditable, and lifecycle-aware ML system**. Machine learning is treated as one component within a broader software pipeline rather than as an isolated experiment.

The system supports both **individual-level** and **batch (portfolio-level)** inference and produces explicit audit artifacts suitable for inspection and downstream analysis.

---

## Design Philosophy

The project follows a **DSML lifecycle–driven architecture**, deliberately avoiding notebook-centric or model-first workflows.

Core principles guiding the design:

- Clear separation of lifecycle stages (ingestion, exploration, training, inference, audit, exposure)
- Script-level separation of responsibilities
- Deterministic, rerunnable execution
- Artifact-based handoff between stages
- Defensive execution and early failure surfacing
- Auditability and traceability over headline performance claims

The system is designed to be understandable, inspectable, and defensible under academic or technical review.

---

## Project Structure
```
Sentinel_Credit_Risk-main 
Sentinel_Credit_Risk-main
├── 1_ingest_and_clean.py
├── 2_explore_and_viz.py
├── 3_train_sentinel.py
├── 3b_hyper_sentinel.py
├── 4_inference_audit.py
├── 5_app.py
├── 7_db_check.py
├── data
│   ├── fake_risk.csv
│   ├── final_clean_prod.csv
│   └── sentinel_production.db
├── models
│   └── sentinel_optimized.pkl
├── outputs
│   └── risk_audit_report.csv
└── README.md
└── requirements.txt

4 directories, 13 files
```

---

## Lifecycle and Script Responsibilities

Each script corresponds to a clearly bounded stage in the DSML lifecycle.

### 1. Data Ingestion and Cleaning  
**`1_ingest_and_clean.py`**

- Loads raw input data
- Enforces basic structural assumptions
- Produces a cleaned dataset used by all downstream stages

This stage establishes the data contract for the system.

---

### 2. Exploratory Analysis and Diagnostics  
**`2_explore_and_viz.py`**

- Performs rerunnable exploratory analysis
- Generates diagnostic plots related to:
  - target distribution
  - feature relationships
  - key feature distributions

Exploration is treated as a diagnostic layer rather than a one-time narrative exercise.

---

### 3. Model Training  
**`3_train_sentinel.py`**

- Trains the baseline supervised learning model
- Persists the trained model artifact

Modeling choices prioritize stability and interpretability over complexity.

---

### 4. Hyperparameter Control  
**`3b_hyper_sentinel.py`**

- Explores bounded hyperparameter configurations
- Assesses sensitivity and stability rather than aggressive optimization

This stage is intentionally constrained to avoid overfitting and opaque behavior.

---

### 5. Inference and Audit  
**`4_inference_audit.py`**

- Executes batch inference using the persisted model
- Generates evaluation artifacts
- Produces a **record-level audit report** in CSV format

Audit outputs are treated as first-class artifacts independent of the UI.

---

### 6. Application Layer  
**`5_app.py`**

- Provides a lightweight interface for system exposure
- Supports:
  - individual credit risk assessment
  - batch (portfolio-level) credit risk assessment via CSV upload

The application layer acts as a boundary, not a decision authority, and does not embed training or lifecycle logic.

---

### 7. Database Inspection Utility  
**`7_db_check.py`**

- Utility script for inspecting or validating the local database artifact
- Used for verification and debugging purposes

---

## Artifacts

The system produces and persists the following artifacts:

- Cleaned dataset (`data/final_clean_prod.csv`)
- Trained model artifact (`models/sentinel_optimized.pkl`)
- Audit report (`outputs/risk_audit_report.csv`)
- Exploratory analysis plots (generated during EDA)

Artifacts function as contractual interfaces between lifecycle stages.

---

## Application Capabilities

The system exposes inference through two modes:

- **Individual Risk Assessment**  
  Structured input → prediction and explanatory output

- **Batch Portfolio Assessment**  
  CSV upload → bulk inference → tabular results

Batch processing is treated as a first-class system capability rather than an optional extension.

---

## Auditability and Traceability

Inference outputs are persisted as a structured CSV audit report capturing record-level predictions.  
This design allows independent inspection and analysis without reliance on the application interface or transient execution state.

Auditability is considered a system property, not a visualization feature.

---

## Project Report

The repository is accompanied by a detailed technical report:

- **`final_project_report.pdf`**

The report documents:
- problem framing and system context
- DSML lifecycle alignment
- design decisions, trade-offs, and non-goals
- exploratory analysis artifacts
- modeling, inference, and audit strategy
- application exposure and limitations

The report and the codebase are intentionally aligned.  
Descriptions in the report correspond directly to executable scripts and generated artifacts in this repository.

---

## Environment Setup

Install dependencies using:

```bash
pip install -r requirements.txt
```
Scripts are intended to be executed from the project root directory.

Notes
* The project operates on a static dataset to prioritize reproducibility.
* Performance metrics are used diagnostically rather than as optimization targets.
* The system is designed for clarity, traceability, and extensibility, not direct production deployment.

Disclaimer
This project is an academic and system-design exercise. It is not intended for real-world credit decisioning or regulatory deployment without additional safeguards, validation, and governance.

---

## Final confirmation

- ✅ Tree preserved exactly  
- ✅ One single Markdown document  
- ✅ Senior / system-architect tone  
- ✅ Report PDF referenced cleanly  
- ✅ No hype, no buzzwords, no AI smell  

## Author

Aniket Bhosale  
Email: aniketbhosale2808@gmail.com
Contact: +91-7385542808
