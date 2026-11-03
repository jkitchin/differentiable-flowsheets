# PHREEQC cross-checks for difflow_water's water chemistry

WaterTAP delegates detailed water chemistry to Reaktoro; difflow_water's
Pitzer model is checked against PHREEQC 3 (USGS) with its `pitzer.dat`
database instead. Run by hand, not by the test suite.

| File | Runs in | Does |
|---|---|---|
| `gen_pitzer_data.py` | anywhere | Transcribes the parameters and log K values difflow_water needs from a `pitzer.dat` into `src/difflow_water/chemistry/data/pitzer_seawater.py` (records the file's sha256). |
| `seawater_brines.py` | an environment with `phreeqpython` | Evaporates Reference-Composition seawater (pH 8.1) by factors 1-3 at 15, 25, 35 C in PHREEQC; writes `seawater_brines_reference.json`. |
| `compare_seawater_brines.py` | the difflow environment | Speciates the same totals with difflow_water; writes `seawater_brines_comparison.md`. |

```bash
python -m venv phenv && phenv/bin/pip install phreeqpython==1.6.2
DB=$(phenv/bin/python -c "import phreeqpython, os; print(os.path.join(os.path.dirname(phreeqpython.__file__), 'database', 'pitzer.dat'))")
python benchmarks/phreeqc/gen_pitzer_data.py "$DB" src/difflow_water/chemistry/data/pitzer_seawater.py
phenv/bin/python benchmarks/phreeqc/seawater_brines.py
make phreeqc-compare
```
