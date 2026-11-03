# WaterTAP comparisons for difflow_water

WaterTAP is the benchmark `difflow_water` must be comparable to. These
scripts build the same cases in WaterTAP and in difflow_water and report the
differences. They are run by hand, not by the test suite (WaterTAP is not a
difflow dependency, and the tests do not compare against reference data).

| File | Runs in | Does |
|---|---|---|
| `swro_px.py` | a WaterTAP environment | Solves the seawater RO + pressure exchanger plant (WaterTAP's `RO_with_energy_recovery` layout, seawater property package, RO1D with 160 elements) for each case in `CASES`, and the RO stage alone; writes `swro_px_reference.json` with the versions used. |
| `compare_swro_px.py` | the difflow environment | Runs `difflow_water` on the same inputs; prints and writes `swro_px_comparison.md`. |
| `mvc.py` | a WaterTAP environment | Runs WaterTAP's single-stage MVC flowsheet (`flowsheets/mvc/mvc_single_stage.py`: its LCOW optimum, then simulations at fixed designs); writes `mvc_reference.json`. |
| `gac.py` | a WaterTAP environment | Runs WaterTAP's example GAC flowsheet (`flowsheets/gac/gac.py`, fixed `kf`, `ds`) at four inputs; writes `gac_reference.json`. |
| `compare_gac.py` | the difflow environment | Runs `GACAdsorber` on the same inputs; writes `gac_comparison.md`. |
| `compare_mvc.py` | the difflow environment | Runs `difflow_water.thermal.mvc_performance` on the same designs and repeats the LCOW optimization with SLSQP and `jax.grad`; writes `mvc_comparison.md`. |

```bash
python -m venv wtenv && wtenv/bin/pip install watertap==1.8.0 && wtenv/bin/idaes get-extensions
wtenv/bin/python benchmarks/watertap/swro_px.py
wtenv/bin/python benchmarks/watertap/mvc.py
wtenv/bin/python benchmarks/watertap/gac.py
make watertap-compare
```

When a `difflow_water` unit changes, rerun `make watertap-compare` and update
the comparison section of `docs/unit-operations-water.md`. When WaterTAP is
upgraded, regenerate the reference and record the new versions there.
