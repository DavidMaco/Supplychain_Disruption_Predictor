---
applyTo: "**"
---

# Output Quality — Supply Chain Disruption Predictor

Standards for prediction outputs, reports, and visualisations.

## Data Output Standards

- All prediction outputs must include a confidence or probability score
- Output files must have consistent column names documented in README
- Business impact summaries in `business_impact.json` must follow the schema in `docs/`

## Report Standards

- Human-readable without requiring technical context
- Include methodology summary and data sources

## Shared Rules

- No hardcoded file paths — use config or relative paths from project root
- No commented-out code in committed files
