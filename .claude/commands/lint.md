# Lint Code

Check code style and common issues in the difflow codebase.

## Instructions

Run code quality checks:

1. Check for Python syntax errors:
```bash
python -m py_compile src/difflow/*.py src/difflow/**/*.py 2>&1 | head -20
```

2. Check imports are valid:
```bash
python -c "import difflow; print('✅ Main package imports successfully')"
python -c "import difflow_bio; print('✅ Bio plugin imports successfully')" 2>/dev/null || echo "⚠️ Bio plugin not available"
python -c "import difflow_ree; print('✅ REE plugin imports successfully')" 2>/dev/null || echo "⚠️ REE plugin not available"
```

3. Look for common issues:
- Unused imports
- Missing type hints on public functions
- Functions without docstrings

Report any issues found and suggest fixes.
