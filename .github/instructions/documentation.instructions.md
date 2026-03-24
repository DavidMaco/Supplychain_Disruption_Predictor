---
applyTo: "**"
---

# Documentation Standards — Supply Chain Disruption Predictor

## When to Write Documentation

Write or update documentation when:

- Adding a new disruption prediction model or feature
- Changing input data format or schema
- Modifying environment variable or configuration requirements

Do **not** add docstrings or comments to code you did not change.

## Docstring Format (Python)

```python
def predict_disruption(features: pd.DataFrame, model_path: str) -> pd.Series:
    """Predict supply chain disruption probability for each record.

    Args:
        features: DataFrame of input features (n_samples x n_features).
        model_path: Path to the serialised model file.

    Returns:
        Series of disruption probability scores (0.0–1.0).

    Raises:
        FileNotFoundError: If `model_path` does not exist.
        ValueError: If `features` is empty.
    """
```

## README Updates

Every `README.md` must include:

1. One-sentence product description
2. Quickstart commands
3. Environment variable list

## Changelog

Format: `## [YYYY-MM-DD] — description`.

## Standards Docs Location

Implementation status lives in `docs/`:

- `STANDARDS_IMPLEMENTATION.md`
- `SKILLS_IMPLEMENTATION.md`
- `ROLLOUT_IMPLEMENTATION.md`
