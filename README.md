<div align="center">

<!-- Hero Section -->
<img src="https://img.shields.io/badge/🔗-Supply_Chain-0A66C2?style=for-the-badge" alt="Supply Chain">
<img src="https://img.shields.io/badge/🤖-Machine_Learning-FF6F00?style=for-the-badge" alt="ML">
<img src="https://img.shields.io/badge/📊-Predictive_Analytics-2E7D32?style=for-the-badge" alt="Analytics">

# 🏭 Supply Chain Disruption Predictor

### Machine Learning Early Warning System for FMCG Manufacturing

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Tests](https://img.shields.io/badge/tests-15%20passed-brightgreen?logo=pytest&logoColor=white)](tests/)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=github-actions&logoColor=white)](.github/workflows/ci.yml)
[![License: Portfolio](https://img.shields.io/badge/license-portfolio-lightgrey)](LICENSE)

<br>

> **A predictive analytics pipeline that forecasts delivery delays and scores supplier risk.**
> Built with synthetic data — all business-impact figures are scenario-based estimates.

<br>

| Accuracy | Recall (Late) | ROC-AUC | CV ROC-AUC | Training Rows |
|:--------:|:-------------:|:-------:|:----------:|:-------------:|
| **54.5 %** | **63.2 %** | **0.540** | **0.511 ± 0.038** | **2,326** |

</div>

---

## 📋 Table of Contents

- [Why This Matters](#-why-this-matters)
- [Architecture](#-architecture)
- [Additional Documentation](#additional-documentation)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [How It Works](#-how-it-works)
- [Business Impact (Scenario)](#-business-impact-scenario-based)
- [Known Limitations](#-known-limitations)
- [Key Learnings](#-key-learnings)
- [Skills Demonstrated](#-skills-demonstrated)
- [Roadmap](#-roadmap)
- [About](#-about)

---

## 🎯 Why This Matters

Traditional procurement systems are **reactive** — they detect problems only after delays occur.

<table>
<tr>
<td width="50%" valign="top">

### ❌ &nbsp; Without Prediction
- Production stoppages (₦2–5 M/day)
- Rush-order premiums (30–50 %)
- Customer penalties & lost contracts
- Inventory imbalances

</td>
<td width="50%" valign="top">

### ✅ &nbsp; With This System
- Flag high-risk orders **before** delays
- Enable proactive supplier engagement
- Reduce emergency procurement costs
- Data-driven risk scoring

</td>
</tr>
</table>

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     DATA GENERATION PIPELINE                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  suppliers.csv ─┐                                                │
│                 ├─► Supplier Health Metrics (monthly)             │
│                 │         │                                       │
│  purchase_      │         ▼                                       │
│  orders.csv ────┼─► Date-Aligned ──► Feature ──► ML-Ready        │
│                 │   External Risk    Engineering  Dataset         │
│                 │   Factors (daily)  (18 features)                │
│                 │         │                                       │
│  config.yaml ───┘         ▼                                       │
│                  Join Diagnostics                                 │
│                  (coverage validation)                            │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│                      ML TRAINING PIPELINE                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ML Features ──► Train/Test ──► Random Forest ──► Predictions    │
│                  Split (80/20)   (balanced)       Risk Scorecard  │
│                                      │           Business Impact  │
│                                      ▼                           │
│                              Model Metrics                       │
│                              Feature Importance                  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Layer | Tools |
|:------|:------|
| **Machine Learning** | scikit-learn (RandomForestClassifier), joblib |
| **Data Processing** | pandas, NumPy |
| **Configuration** | PyYAML — single `config.yaml` |
| **Testing** | pytest (15 tests), pytest-cov |
| **CI/CD** | GitHub Actions (Python 3.10 / 3.11 / 3.12) |
| **Hashing** | SHA-256 (reproducibility fingerprints) |

---

## Additional Documentation

- [Layman Guide](docs/PROJECT_EXPLAINED_FOR_EVERYONE.md)
- [Content Campaign Pack](docs/CONTENT_CAMPAIGN_PACK.md)

---

## 📁 Project Structure

```
Supply Chain Disruption Predictor/
│
├── config.yaml                      # Centralised pipeline configuration
├── pipeline_utils.py                # Logging, contracts, metadata, hashing
├── generate_disruption_data.py      # Data generation & enrichment pipeline
├── train_ml_model.py                # Model training, evaluation & prediction
├── validate_claims.py               # Maps README claims → artefact evidence
├── requirements.txt                 # Pinned dependency manifest
├── README.md                        # This file
│
├── # ─── Generated Artefacts (after pipeline run) ──────────────
├── risk_factors_daily.csv           # Daily external risk factors
├── supplier_health_monthly.csv      # Monthly supplier health records
├── purchase_orders_enhanced.csv     # Orders enriched with risk factors
├── ml_features.csv                  # ML-ready feature matrix
├── disruption_predictor_model.pkl   # Trained Random Forest model
├── predictions_pending_orders.csv   # Pending orders with risk scores
├── supplier_risk_scorecard.csv      # Supplier risk rankings
├── feature_importance.csv           # Feature importance from model
├── model_metrics.json               # Technical metrics (accuracy, ROC-AUC, CV)
├── business_impact.json             # Scenario-based impact (with disclaimer)
├── join_diagnostics.json            # External/health join coverage report
├── claim_evidence.json              # Claim-to-evidence mapping
├── run_metadata.json                # Reproducibility metadata
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py             # 15 unit & integration tests
│
└── .github/
    └── workflows/
        └── ci.yml                   # GitHub Actions CI config
```

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
# Step 1 — Generate & enrich data
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
import joblib, pandas as pd

model = joblib.load("disruption_predictor_model.pkl")

new_order = pd.DataFrame({
    "financial_health_score": [45],
    "capacity_utilization_pct": [92],
    "employee_turnover_pct":   [12],
    "quality_defect_rate":     [6.5],
    # ... remaining features (see feature_importance.csv for full list)
})

prob = model.predict_proba(new_order)[0, 1]
print(f"Late delivery probability: {prob * 100:.1f}%")
```

---

## ⚙️ How It Works

### 1 &nbsp; External Risk Factors <sub>(daily, date-aligned to PO range)</sub>

| Factor | Range | Description |
|:-------|:-----:|:------------|
| Rainfall index | 0 – 1 | Rainy vs dry season |
| Fuel price (₦) | Variable | Trend + volatility |
| USD/NGN rate | Variable | Exchange rate pressure |
| Port congestion | 0 – 100 | Includes seasonal spikes |
| Disruption events | Categorical | Strikes, weather, political |

### 2 &nbsp; Supplier Health Metrics <sub>(monthly per supplier)</sub>

| Metric | Range | Signal |
|:-------|:-----:|:-------|
| Financial health score | 0 – 100 | Stability indicator |
| Capacity utilisation | 60 – 95 % | Overload risk |
| Employee turnover | 2 – 15 % | Operational continuity |
| Quality defect rate | 0.5 – 8 % | Product reliability |
| Payment default risk | 0 – 30 % | Financial risk |

### 3 &nbsp; Feature Engineering

**18 features** produced, including:

- **Rolling averages**: 5-order average delay per supplier
- **Late delivery rate**: Over last 10 orders per supplier
- **Demand recency**: Days since last order
- **Order sizing**: Current order vs supplier average
- **Composite risk score** (0–100): Weighted blend of supplier (40 %), external (30 %), and operational (30 %) risk

### 4 &nbsp; Prediction & Risk Classification

| Risk Level | Threshold | Action |
|:-----------|:---------:|:-------|
| 🟢 **Low** | P(late) < 30 % | Standard monitoring |
| 🟡 **Medium** | 30 – 60 % | Enhanced tracking |
| 🔴 **High** | > 60 % | Procurement intervention |

---

## 💰 Business Impact (Scenario-Based)

> [!CAUTION]
> The figures below are **illustrative estimates** derived from configurable assumptions
> in `config.yaml`. They have **not** been validated against real-world outcomes.

<table>
<tr>
<td>

### Assumptions

| Parameter | Value |
|:----------|:------|
| Delay cost %/week | 3 % of order value |
| Average delay | 2 weeks |
| Mitigation success | 70 % |

</td>
<td>

### Scenario Output (6 high-risk orders)

| Metric | Value |
|:-------|------:|
| Exposed value | ₦875.2 M |
| Est. disruption cost | ₦52.5 M |
| **Potential savings (70 %)** | **₦36.8 M** |

</td>
</tr>
</table>

Numbers update automatically on each pipeline re-run. See `business_impact.json` for full output.

---

## ⚠️ Known Limitations

| # | Limitation | Impact |
|:-:|:-----------|:-------|
| 1 | **Modest baseline model** (ROC-AUC 0.540) | Limited by synthetic data; real data expected to improve |
| 2 | **Synthetic data only** | Findings may not transfer to real procurement datasets |
| 3 | **Random train/test split** | No temporal validation; production should use walk-forward |
| 4 | **Scenario-based savings** | ₦36.8 M is an estimate, not a measured outcome |
| 5 | **No live ERP integration** | Pipeline runs offline; real-time not in scope |
| 6 | **"Early warning" unverified** | Prediction lead time not formally measured |

---

## 🧠 Key Learnings

<details>
<summary><b>1. Data-quality bugs hide in joins</b></summary>

The external-risk features showed **0 % importance** because a date-range mismatch
produced 100 % nulls. After aligning the date range to PO dates, external features
now contribute **13.7 %** combined importance. Always validate join coverage before
interpreting model outputs.
</details>

<details>
<summary><b>2. Separate metrics from assumptions</b></summary>

Technical model metrics and business-impact scenarios serve different audiences and
should live in different files with clear labels. `model_metrics.json` vs
`business_impact.json` — the latter carries a `_disclaimer` field.
</details>

<details>
<summary><b>3. Reproducibility requires infrastructure</b></summary>

A `config.yaml`, run metadata with SHA-256 fingerprints, and data-contract checks
make re-runs deterministic and auditable. Every artefact is traceable.
</details>

<details>
<summary><b>4. Simple models are a valid starting point</b></summary>

Random Forest with basic features establishes a baseline. Complexity (XGBoost,
feature selection, ensembles) should be added only when the baseline is sound.
</details>

<details>
<summary><b>5. README claims must be evidence-backed</b></summary>

`validate_claims.py` maps every headline number to the artefact that produced it.
Run it after each pipeline execution to confirm figures are current.
</details>

---

## 🛠 Skills Demonstrated

<table>
<tr>
<td align="center" width="20%"><b>🤖<br>Machine Learning</b></td>
<td align="center" width="20%"><b>🔧<br>Data Engineering</b></td>
<td align="center" width="20%"><b>💻<br>Software Engineering</b></td>
<td align="center" width="20%"><b>📐<br>Analytical Rigour</b></td>
<td align="center" width="20%"><b>🏭<br>Domain Knowledge</b></td>
</tr>
<tr>
<td valign="top"><sub>Random Forest, class weighting, cross-validation, feature importance</sub></td>
<td valign="top"><sub>Multi-source joins, date alignment, rolling-window features, composite scoring</sub></td>
<td valign="top"><sub>YAML config, structured logging, data contracts, CI, unit testing, local RNG</sub></td>
<td valign="top"><sub>Claim validation, join diagnostics, assumption labelling, SHA-256 fingerprints</sub></td>
<td valign="top"><sub>Supply chain risk factors, supplier performance, procurement operations</sub></td>
</tr>
</table>

---

## 🗺 Roadmap

- [x] End-to-end pipeline with validated metrics
- [x] Date-alignment fix (external features now active)
- [x] Claim validation system
- [x] CI with multi-version Python matrix
- [ ] Time-based train/test split (walk-forward validation)
- [ ] Gradient boosting (XGBoost / LightGBM)
- [ ] SMOTE or threshold tuning for class imbalance
- [ ] Power BI / Streamlit dashboard
- [ ] Measure actual prediction lead time
- [ ] Multi-class delay prediction (1–7, 8–14, 15+ days)

---

## 👤 About

<table>
<tr>
<td>

**Portfolio project by David Maco**

A Chemical Engineering graduate with procurement experience in FMCG manufacturing,
applying data science to supply chain problems.

[![GitHub](https://img.shields.io/badge/GitHub-DavidMaco-181717?logo=github)](https://github.com/DavidMaco)

</td>
</tr>
</table>

---

<div align="center">

**All data is synthetically generated. This is a portfolio demonstration project.**

<sub>Built with Python · scikit-learn · pandas · pytest · GitHub Actions</sub>

<br>

⭐ **Found this useful? Star the repo!** ⭐

</div>
