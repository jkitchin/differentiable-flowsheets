---
description: Run the difflow per-commit test suite and summarize failures
allowed-tools: ["Bash"]
---

# Run Tests

Run the difflow per-commit suite (parallel, skips `slow` and `release`):

```bash
make test
```

`make test-all` runs everything, including the slow and release tests.

If tests fail, analyze the failures and suggest fixes. Focus on:
1. Import errors (missing dependencies or optional extras)
2. Assertion failures (logic bugs)
3. JAX tracing errors (Python control flow on traced values)

Report a summary of test results including:
- Total tests passed/failed
- Any failing test names and brief error descriptions
- Suggested fixes for failures
