# Run Example Notebook

Execute an example notebook to verify it works.

## Arguments
- $ARGUMENTS: The example number or name (e.g., "00", "03_optimization", "cstr")

## Instructions

Find and execute the matching example notebook:

```bash
# Find matching notebook
ls examples/*$ARGUMENTS*.ipynb 2>/dev/null || ls examples/**/*$ARGUMENTS*.ipynb 2>/dev/null
```

Execute the notebook using jupyter:
```bash
jupyter execute examples/<matched_notebook>.ipynb --timeout=300
```

If no argument provided, list available examples:
```bash
ls examples/*.ipynb
```

Report:
- Whether the notebook executed successfully
- Any errors encountered
- Key outputs or results from the notebook
