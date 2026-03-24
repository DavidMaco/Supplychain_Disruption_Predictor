# AGENTS.md — Supply Chain Disruption Predictor

Operational guide for AI agents working on this repository.

## Repository Layout

```text
*.py             Root-level ML pipeline and prediction modules
tests/           pytest test suite
config.yaml      Runtime configuration
business_impact.json  Business impact summary output
docs/            Project documentation
```

## Mandatory Workflow

### Before any code change

1. Read the relevant instruction file from `.github/instructions/`
2. Identify the verification gate for the layer being changed
3. Plan using the `plan-execute-verify` skill

### After any code change

1. Run the verification gate for the changed layer
2. Run `get_errors` on all modified files
3. Fix all errors before reporting done

## Verification Gates

### Python

```bash
python -m pytest tests/ -v --tb=short
flake8 *.py --max-line-length=120 --ignore=E501,W503
```

## Security Rules (blocking)

- No `os.environ.get("VAR", "fallback")` for secrets — raise `RuntimeError`
- No credentials in source code
- No hardcoded production data paths

## CI / Standards Governance

The `standards-governance` job in `.github/workflows/python-app.yml` enforces:

- All 12 standards baseline files present
- Security pattern scan

The `ci.yml` workflow runs the full test matrix across Python 3.10, 3.11, 3.12.

## Skills Available

| Skill | Path |
|---|---|
| Plan -> Execute -> Verify | `.github/skills/plan-execute-verify/SKILL.md` |
| Standards Review | `.github/skills/standards-review/SKILL.md` |
