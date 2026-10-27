---
description: Execute a difflow example notebook and report whether it runs
argument-hint: "<number or name>"
allowed-tools: ["Bash"]
---

# Run Example Notebook

Execute an example notebook to verify it works.

## Arguments
- $ARGUMENTS: The example number or name (e.g., "00", "03_optimization", "cstr")

## Instructions

Find the matching example notebook:

```bash
ls examples/*$ARGUMENTS*.ipynb
```

Execute it:
```bash
jupyter execute examples/<matched_notebook>.ipynb --timeout=600
```

If no argument is provided, list available examples:
```bash
ls examples/*.ipynb
```

Report:
- Whether the notebook executed successfully
- Any errors encountered
- Key outputs or results from the notebook
