---
description: Check difflow code style, syntax and imports of the core and plugin packages
allowed-tools: ["Bash"]
---

# Lint Code

Check code style and common issues in the difflow codebase.

## Instructions

1. Check for Python syntax errors:
```bash
python -m compileall -q src/ | head -20
```

2. Check that the core package and every plugin import:
```bash
for m in difflow difflow_bio difflow_ree difflow_cc difflow_gas difflow_power difflow_refinery; do
  python -c "import $m" 2>/dev/null && echo "ok   $m" || echo "FAIL $m"
done
```

3. If `ruff` is available, run `ruff check src/ tests/`.

4. Look for common issues in changed files (`git diff --name-only main`):
- Unused imports
- Missing type hints on public functions
- Public functions or `Params` fields without docstrings
- Python control flow on traced values in a unit's `__call__`

Report any issues found and suggest fixes.
