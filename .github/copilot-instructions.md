# Supply Chain Disruption Predictor — Copilot Instructions

Python ML tool for predicting supply chain disruptions. Multi-version matrix
testing (Python 3.10 – 3.12), pytest, flake8.

## Verification Gates

```bash
python -m pytest tests/ -v --tb=short
flake8 *.py --max-line-length=120 --ignore=E501,W503
```

## Security Standards (non-negotiable)

- **Never** add `os.environ.get("VAR", "fallback-secret")` — raise `RuntimeError` if missing
- No secrets or credentials in source code
- No hardcoded external API endpoints — configure via environment or `config.yaml`

## Code Standards

### Python

- `from __future__ import annotations` on every module
- Settings loaded from `config.yaml` or environment; no bare `os.environ` in business logic
- Logger via `logging.getLogger(__name__)`
- All ML model loading must handle `FileNotFoundError` gracefully

## Instruction Files

| Scope | File |
|---|---|
| Python ML Application | `.github/instructions/api-ops.instructions.md` |
| Output & Reporting | `.github/instructions/frontend-quality.instructions.md` |
| Security | `.github/instructions/security.instructions.md` |
| Documentation | `.github/instructions/documentation.instructions.md` |
| Workflow | `.github/instructions/execution-workflow.instructions.md` |
