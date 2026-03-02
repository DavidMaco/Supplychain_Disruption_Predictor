# Supply Chain Disruption Predictor

### Machine Learning Early Warning System for FMCG Manufacturing

A predictive analytics pipeline that uses machine learning to forecast delivery
delays and score supplier risk. All data is **synthetically generated**; all
business-impact figures are **scenario-based estimates** subject to the
assumptions documented below.

> **Status**: Pipeline re-run complete with date-alignment fix applied.
> External features now contribute 14.4% combined importance (previously 0%).
> All metrics below reflect the current pipeline output.

---

## Project Overview

Supply chain disruptions cost FMCG manufacturers millions in lost production,
rush orders, and customer penalties. This project builds an ML system that:

1. **Predicts delivery delays** based on supplier health, order characteristics,
   and external risk factors.
2. **Scores supplier risk** using a weighted composite index.
3. **Generates a risk scorecard** for the procurement team.
4. **Estimates financial impact** under explicit, configurable assumptions.

### Current Performance

| Metric | Value | Note |
|--------|-------|------|
| Accuracy | 55.4 % | On imbalanced data (58.3 % late rate) |
| Recall (late) | 63.0 % | Catches ~6/10 late deliveries |
| ROC-AUC | 0.542 | Improved from 0.504 after join fix |
| CV ROC-AUC | 0.526 ± 0.017 | 5-fold stratified cross-validation |
| Training rows | 2 435 | Completed deliveries over 2 years |

Full technical metrics are persisted to `model_metrics.json` after each run.

---

## Business Problem

Traditional procurement systems are **reactive**—they detect problems only
after delays occur, leading to:

- Production stoppages (₦2–5 M/day)
- Rush-order premiums (30–50 %)
- Customer penalties and lost contracts
- Inventory imbalances

This project demonstrates a **proactive** approach: flag high-risk orders
early so procurement can intervene.

---

## Technical Architecture

### Data Pipeline

```
Source CSVs (suppliers.csv, purchase_orders.csv)
    ↓
External Risk Factors  ← date-aligned to PO range
    ↓
Supplier Health Metrics ← joined on supplier_id
    ↓
Feature Engineering (rolling averages, composite scoring)
    ↓
ML-Ready Dataset (18 features)
    ↓
Random Forest Classifier (class_weight='balanced')
    ↓
Predictions + Risk Scores + Business Impact Scenarios
```

### Tech Stack

| Layer | Tools |
|-------|-------|
| ML | scikit-learn (RandomForestClassifier), joblib |
| Data | pandas, NumPy |
| Config | PyYAML (`config.yaml`) |
| Testing | pytest, pytest-cov |
| CI | GitHub Actions |

---

## Project Structure

All files live in **one flat directory** (no nested `scripts/`, `data/`,
`models/` subdirectories).

```
Supply Chain Disruption Predictor/
│
│── config.yaml                        # Centralised pipeline configuration
│── pipeline_utils.py                  # Shared utilities (logging, contracts, metadata)
│── generate_disruption_data.py        # Data generation & enrichment pipeline
│── train_ml_model.py                  # Model training, evaluation & prediction
│── validate_claims.py                 # Maps README claims → artifact evidence
│── requirements.txt                   # Pinned dependency manifest
│── PROJECT2_README.md                 # This file
│
│── # Generated artefacts (after pipeline run)
│── risk_factors_daily.csv             # Daily external risk factors
│── supplier_health_monthly.csv        # Monthly supplier health records
│── purchase_orders_enhanced.csv       # Orders enriched with risk factors
│── ml_features.csv                    # ML-ready feature matrix
│── disruption_predictor_model.pkl     # Trained Random Forest model
│── predictions_pending_orders.csv     # Pending orders with risk scores
│── supplier_risk_scorecard.csv        # Supplier risk rankings
│── feature_importance.csv             # Feature importance from model
│── model_metrics.json                 # Technical metrics (accuracy, ROC-AUC, CV)
│── business_impact.json               # Scenario-based business impact (labelled)
│── join_diagnostics.json              # External/health join coverage report
│── claim_evidence.json                # Claim-to-evidence mapping
│── run_metadata.json                  # Reproducibility metadata
│
│── tests/
│   └── test_pipeline.py               # Unit & integration tests
│
└── .github/
    └── workflows/
        └── ci.yml                     # GitHub Actions CI config
```

---

## How It Works

### 1. Risk Factor Generation

**External Factors** (daily, date-aligned to PO range):

- Rainy season indicator
- Rainfall index (0–1)
- Holiday periods
- Fuel price volatility
- USD/NGN exchange rate
- Port congestion index (0–100)
- Disruption events (strikes, weather, political)

**Supplier Health Metrics** (monthly):

- Financial health score (0–100)
- Production capacity utilisation (%)
- Employee turnover rate (%)
- Quality defect rate (%)
- Days of inventory on hand
- Payment default risk (0–100)

### 2. Feature Engineering

```python
# Rolling 5-order average delay per supplier
supplier_avg_delay = orders.groupby('supplier')['delay'].rolling(5).mean()

# Late delivery rate over last 10 orders
late_rate = orders.groupby('supplier')['is_late'].rolling(10).mean()

# Days since last order (demand recency)
days_since_last = orders.groupby('supplier')['date'].diff().days
```

**Composite Risk Score** (0–100, weights from `config.yaml`):

| Component | Default Weight | Inputs |
|-----------|---------------|--------|
| Supplier risk | 40 % | Financial health, past performance |
| External risk | 30 % | Port congestion, disruption events |
| Operational risk | 30 % | Quality defects, employee turnover |

### 3. Prediction & Alerting

For each pending order the pipeline:

1. Computes the current risk score.
2. Predicts P(late delivery) using the trained model.
3. Classifies as **Low** (< 30 %), **Medium** (30–60 %), or **High** (> 60 %).
4. Exports high-risk orders for procurement review.

---

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# Step 1 — Generate / enrich data
python generate_disruption_data.py

# Step 2 — Train model & produce predictions
python train_ml_model.py

# Step 3 — Validate README claims against artefacts
python validate_claims.py
```

### Run Tests

```bash
python -m pytest tests/ -v
```

### Use the Model

```python
import pandas as pd
import joblib

model = joblib.load("disruption_predictor_model.pkl")

new_order = pd.DataFrame({
    "financial_health_score": [45],
    "capacity_utilization_pct": [92],
    "employee_turnover_pct": [12],
    "quality_defect_rate": [6.5],
    # ... remaining features
})

prob = model.predict_proba(new_order)[0, 1]
print(f"Late delivery probability: {prob * 100:.1f}%")
```

---

## Business Impact (Scenario-Based)

> **Disclaimer**: The figures below are **illustrative estimates** derived from
> configurable assumptions in `config.yaml`. They have **not** been validated
> against real-world outcomes.

### Assumptions

| Parameter | Value | Source |
|-----------|-------|--------|
| Stoppage cost / day | ₦3 M | Industry estimate |
| Average delay | 15.5 days | Synthetic data |
| Rush-order premium | 50 % | Assumption |
| Customer penalty | 10 % of order | Assumption |
| Mitigation rate | 70 % | Assumption |

### Illustrative Scenario (6 high-risk orders)

| Metric | Value |
|--------|-------|
| Exposed value | ₦412.8 M |
| Estimated disruption cost (unmitigated) | ₦24.8 M |
| Estimated mitigation savings (70 %) | ₦17.3 M |
| **Scenario net savings** | **₦17.3 M** |

These numbers will update automatically when the pipeline is re-run with
different assumptions.

---

## Known Limitations

1. **Modest baseline model**: ROC-AUC of 0.542 after the join fix — an
   improvement over the pre-fix 0.504, but still limited by synthetic data.
   Real procurement data would likely yield stronger signal.

2. **Synthetic data only**: All supplier, PO, and risk-factor data is
   generated. Findings may not transfer to real procurement datasets.

3. **No temporal validation**: Train/test split is random, not time-based.
   A production system should use walk-forward or time-series cross-validation.

4. **Business impact is scenario-based**: The ₦17.3 M savings figure is an
   estimate under stated assumptions, not a measured outcome.

5. **No live integration**: The pipeline runs offline. Real-time ERP
   integration is out of scope.

6. **"2–4 weeks early warning" unverified**: The lead time of predictions
   has not been formally measured against delivery dates.

---

## Roadmap

- [x] Re-run pipeline end-to-end and publish updated metrics
- [ ] Add time-based train/test split (walk-forward validation)
- [ ] Experiment with gradient boosting (XGBoost / LightGBM)
- [ ] Add SMOTE or threshold tuning for class imbalance
- [ ] Build Power BI / Streamlit dashboard
- [ ] Measure actual prediction lead time
- [ ] Multi-class delay prediction (1–7, 8–14, 15+ days)

---

## Skills Demonstrated

| Category | Techniques |
|----------|-----------|
| Machine Learning | Random Forest classification, class weighting, cross-validation, feature importance |
| Data Engineering | Multi-source joins, date alignment, rolling-window features, composite scoring |
| Software Engineering | YAML config, structured logging, data contracts, CI, unit testing |
| Analytical Rigour | Claim validation, join diagnostics, assumption labelling |
| Domain Knowledge | Supply chain risk factors, supplier performance, procurement operations |

---

## Key Learnings

1. **Data-quality bugs hide in joins**: The external-risk features showed 0 %
   importance because a date-range mismatch produced 100 % nulls. Always
   validate join coverage before interpreting model outputs.

2. **Separate metrics from assumptions**: Technical model metrics and
   business-impact scenarios serve different audiences and should live in
   different files with clear labels.

3. **Reproducibility requires infrastructure**: A `config.yaml`, run metadata,
   and data-contract checks make re-runs deterministic and auditable.

4. **Simple models are a valid starting point**: Random Forest with basic
   features establishes a baseline. Complexity should be added only when the
   baseline is sound.

5. **README claims must be evidence-backed**: `validate_claims.py` maps every
   headline number to the artefact that produced it.

---

## Contact

**Portfolio Project by**: [Your Name]
- LinkedIn: [Your Profile]
- GitHub: [Your Profile]
- Email: [Your Email]

**Background**: Chemical Engineering graduate with procurement experience in
FMCG manufacturing, applying data science to supply chain problems.

---

## License

This project is for portfolio demonstration. All data is synthetically generated.

---

**Found this useful? Star the repo!**
