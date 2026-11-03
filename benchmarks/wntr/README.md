# EPANET 2.2 cross-checks for difflow_water networks

EPANET 2.2 (run through WNTR) is the reference for `difflow_water.network`.
Run by hand, not by the test suite.

| File | Runs in | Does |
|---|---|---|
| `export_networks.py` | an environment with WNTR | Reads EPANET example networks (Net1, Net3, ky10, and Net1 with Darcy-Weisbach), solves one period in EPANET 2.2, writes `networks/*.json` (network in `WaterNetwork.from_dict` form + EPANET's heads, pressures, demands, flows). |
| `compare_networks.py` | the difflow environment | Solves the same networks in difflow_water (both valve formulations); writes `networks_comparison.md`. |

```bash
python -m venv wntrenv && wntrenv/bin/pip install wntr==1.5.0 six
wntrenv/bin/python benchmarks/wntr/export_networks.py
make wntr-compare
```

`src/difflow_water/network/data/Net1.json` and `Net3.json` (shipped for
`load_example`) are copies of the exported networks.
