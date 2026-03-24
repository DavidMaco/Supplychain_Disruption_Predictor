---
applyTo: "**"
---

# Execution Workflow — Supply Chain Disruption Predictor

The required work pattern for every task in this repository.

## Phase 1 — Understand

1. Read the relevant instruction file for the layer being changed
2. Read files you will modify — understand context before making changes
3. Identify the correct verification gate

## Phase 2 — Plan

Write a step-by-step plan before touching any code.

## Phase 3 — Implement

- Make the smallest change that satisfies the requirement
- Do not refactor or add docstrings to code you didn't change
- Apply security standards throughout

## Phase 4 — Verify

| Layer | Gate |
|---|---|
| Python (3.10) | `python -m pytest tests/ -v --tb=short` |
| Python (3.11) | `python -m pytest tests/ -v --tb=short` |
| Python (3.12) | `python -m pytest tests/ -v --tb=short` |
| Lint | `flake8 *.py --max-line-length=120 --ignore=E501,W503` |

## Phase 5 — Validate

- Run `get_errors` on every modified file
- Fix all errors; do not mark a task complete with open errors

## Discipline Rules

- One concern per commit
- Do not add features not in the task scope
