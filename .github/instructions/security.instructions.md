---
applyTo: "**"
---

# Security Standards — Supply Chain Disruption Predictor

Non-negotiable security requirements. Violations fail the CI `standards-governance` scan.

## Secret Management

```python
# ALWAYS — fail loudly at startup
secret = os.environ.get("API_KEY")
if not secret:
    raise RuntimeError("API_KEY environment variable must be set")

# NEVER — hardcoded fallback
secret = os.environ.get("API_KEY", "dev-key")  # FORBIDDEN
```

- Rotate secrets via environment; never in source code
- No `.env` files committed; add to `.gitignore`

## Data Handling

- Validate all input data shapes before feeding to ML models
- Handle missing values and out-of-range inputs explicitly
- Do not log sensitive supply chain or business data at INFO level

## CI Enforcement

The `standards-governance` job runs a security pattern scan that will **fail the build** on:

1. `os.environ.get(VAR, "secret-fallback")` patterns
2. `raise HTTPException(..., detail=str(exc))` patterns
3. `allow_origins=["*"]` patterns
