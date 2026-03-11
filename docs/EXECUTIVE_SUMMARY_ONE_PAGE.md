# Executive Summary

## Supply Chain Disruption Predictor

### What this project is

The Supply Chain Disruption Predictor is a decision-support project designed to help a business identify which supplier orders are most likely to arrive late before the delay becomes operationally expensive.

It combines purchase order data, supplier condition signals, and external disruption signals to estimate risk on pending orders and rank suppliers by delivery-related risk.

### The business problem

Late supplier deliveries can create a chain of problems across a business:

- production delays
- emergency procurement
- higher logistics cost
- customer service issues
- planning instability
- avoidable financial exposure

In many organizations, these issues are managed reactively. The business notices them clearly only after delivery performance has already failed.

This project addresses that gap by creating an early-warning view of delivery risk.

### What the project does

The pipeline:

1. loads supplier and purchase order records
2. adds external risk context such as congestion and disruption conditions
3. adds supplier health context such as financial health and quality risk
4. engineers historical behavior features such as average delay and late-delivery rate
5. calculates a composite risk score
6. trains a machine learning model on completed deliveries
7. predicts the probability of lateness for pending orders
8. groups pending orders into low, medium, and high risk
9. creates a supplier risk scorecard
10. estimates scenario-based financial exposure under stated assumptions

### Why it matters

The value of the project is not just prediction. It is prioritization.

The outputs help a procurement or operations team decide:

- which pending orders need immediate follow-up
- which suppliers require closer review
- where contingency planning may be needed
- where potential disruption exposure is concentrated

This makes the project relevant to procurement, operations, planning, supply chain leadership, and stakeholders who need a clearer view of risk.

### Current output summary

Based on the latest saved artefacts in the repository:

- test accuracy is approximately 54.5%
- recall for late deliveries is approximately 63.2%
- ROC-AUC is approximately 0.540
- completed training sample size is 2,326 records
- 6 pending orders are currently classified as high risk in the scenario output
- potential savings are estimated at about NGN 36.8M under the stated assumptions

### Important interpretation note

This project uses synthetic data, and the financial impact figures are scenario-based estimates, not measured real-world savings.

That means the project should be understood as a strong prototype for decision support and analytics design, not as a finished production forecasting system.

### What this project demonstrates

This work demonstrates the ability to:

- translate a business problem into an analytics workflow
- combine data engineering and machine learning in one pipeline
- design interpretable features and risk logic
- build outputs that support practical decisions
- validate data quality and documentation claims
- communicate technical work clearly to non-technical stakeholders

### Bottom line

This project shows how predictive analytics can be used to move a business from reacting to supplier delays after they happen toward identifying risk earlier and acting more intelligently before disruption escalates.
