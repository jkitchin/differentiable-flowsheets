# Run Tests

Run the difflow test suite using pytest.

## Instructions

Run the full test suite:
```bash
pytest tests/ -v
```

If tests fail, analyze the failures and suggest fixes. Focus on:
1. Import errors (missing dependencies)
2. Assertion failures (logic bugs)
3. JAX tracing errors (compatibility issues)

Report a summary of test results including:
- Total tests passed/failed
- Any failing test names and brief error descriptions
- Suggested fixes for failures
