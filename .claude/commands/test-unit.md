# Run Specific Unit Tests

Run tests for a specific unit operation or module.

## Arguments
- $ARGUMENTS: The unit or module name to test (e.g., "cstr", "pfr", "flash", "distillation", "eos", "dynamic")

## Instructions

Run tests for the specified module:
```bash
pytest tests/test_$ARGUMENTS.py -v
```

If no argument provided, list available test files:
```bash
ls tests/test_*.py
```

Report results including any failures and their causes.
