# TEOS-10 reference values for the seawater properties

`seawater_teos10.py` evaluates TEOS-10 (the international seawater standard)
with its reference code, GSW-Python, on a grid of 5-40 C and 0-70 g/kg, and
writes `seawater_teos10.json`. `examples/77_water_desalination_units.ipynb`
compares difflow_water's Sharqawy correlations with it. GSW is not a difflow
dependency: run the script in its own environment
(`python -m venv gswenv && gswenv/bin/pip install gsw`). A hand-run
cross-check, not a test.
