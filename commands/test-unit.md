---
description: Run the tests for one difflow unit or module (e.g. cstr, flash, refinery/test_treating)
argument-hint: "<module>"
allowed-tools: ["Bash"]
---

# Run Specific Unit Tests

Run tests for a specific unit operation or module.

## Arguments
- $ARGUMENTS: The unit or module name to test (e.g., "cstr", "pfr", "flash", "eos",
  or a plugin path such as "refinery/test_treating")

## Instructions

Find the matching test file(s) and run them:
```bash
ls tests/test_*$ARGUMENTS*.py tests/*/test_*$ARGUMENTS*.py tests/$ARGUMENTS.py 2>/dev/null
pytest <matched files> -v
```

If no argument is provided, list available test files:
```bash
ls tests/test_*.py tests/*/test_*.py
```

Report results including any failures and their causes.
