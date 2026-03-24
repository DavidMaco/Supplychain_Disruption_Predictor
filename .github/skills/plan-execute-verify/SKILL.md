# Plan -> Execute -> Verify

**Skill domain:** Structured change execution for the Supply Chain Disruption Predictor.

## When to Use

Use whenever a task involves more than a single-line edit:

- Adding new ML models or prediction features
- Changing input data schema or feature engineering
- Modifying CI or test configuration

## Protocol

### Step 1 — Plan

```
Files to change: [list]
Tests to run: [list of commands]
Risk of regression: [low | medium | high]
Sub-tasks: [numbered list]
```

### Step 2 — Execute

- Change one concern at a time
- Read files before editing
- Apply security standards throughout

### Step 3 — Verify

| Layer | Commands |
|---|---|
| Python | `python -m pytest tests/ -v --tb=short` |
| Lint | `flake8 *.py --max-line-length=120 --ignore=E501,W503` |

Then run `get_errors` on every modified file.

### Step 4 — Complete

Mark each todo complete immediately after verification.

## Discipline Constraints

- Never mark a step complete if errors remain
- One pull request = one logical concern
