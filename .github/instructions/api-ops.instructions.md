---
applyTo: "*.py"
---

# Python ML Application Standards — Supply Chain Disruption Predictor

Standards for the Python ML modules at the project root and submodules.

## Module Conventions

- `from __future__ import annotations` at the top of every module
- Config loaded from `config.yaml` or environment; no bare `os.environ` in ML logic
- Logging via `logger = logging.getLogger(__name__)`

## ML Model Standards

- Model paths must come from config, not hardcoded
- Validate input DataFrame shape and dtypes before model inference
- Handle out-of-distribution inputs gracefully—log a warning, do not raise silently

## Error Handling

```python
# CORRECT
try:
    predictions = model.predict(features)
except Exception as exc:
    logger.error("Prediction failed", extra={"error": str(exc)})
    raise RuntimeError("Prediction failed. Check input features.") from exc
```

## Test Requirements

- Every prediction function has at least one positive-case test
- Run: `python -m pytest tests/ -v --tb=short`
