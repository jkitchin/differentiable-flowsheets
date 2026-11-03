# BSM1 open-loop steady state (published reference)

`report_tables.json` holds Tables 6 and 7 of the BSM1 technical report
(Alex et al. 2008, LTH-IEA-7229, Lund University): the open-loop steady
state of the five reactors and the ten settler layers, transcribed by hand
at the report's printed precision (3-4 significant digits).
`benchmarks/watertap/compare_bsm1.py` compares difflow_water's
`activated_sludge_plant` (Takacs settler) with it, and with WaterTAP's
BSM1 example. A hand-run cross-check, not a test.
