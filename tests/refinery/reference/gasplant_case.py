"""The gas plant's IDAES cross-check cases (#312), shared by the generator
and the tests.

Two columns, both with the reflux and boilup ratios fixed (the two
specifications IDAES's ``TrayColumn`` takes directly), equilibrium trays, no
pressure drop, total condenser at the bubble point:

* ``debutanizer`` -- C3 to nC5 at 10 bar, 10 trays;
* ``c3c4_splitter`` -- C2 to nC4 with propylene, at 17 bar, 20 trays.

The component constants come from difflow
(:func:`difflow_refinery.gasplant.gas_components`) and are frozen into the
JSON, so both sides compute on the same Tc, Pc, omega and ideal-gas Cp, and
a later change to them is caught as staleness rather than read as
disagreement. ``kij`` is zero on both sides (none is tabulated for these
pairs).
"""

from __future__ import annotations

CASES: dict[str, dict] = {
    "debutanizer": {
        "names": ["propane", "isobutane", "n_butane", "isopentane", "n_pentane"],
        "z": [0.20, 0.15, 0.25, 0.20, 0.20],
        "F": 100.0, "T": 360.0, "P": 10e5,
        "n_trays": 10, "feed_tray": 5,
        "reflux_ratio": 2.0, "boilup_ratio": 2.0,
    },
    "c3c4_splitter": {
        "names": ["ethane", "propylene", "propane", "isobutane", "n_butane"],
        "z": [0.02, 0.18, 0.35, 0.20, 0.25],
        "F": 100.0, "T": 320.0, "P": 17e5,
        "n_trays": 20, "feed_tray": 10,
        "reflux_ratio": 5.0, "boilup_ratio": 3.0,
        # IDAES's own TrayColumn.initialize ends locally infeasible on this
        # column; the generator starts it from a shortcut profile instead
        # (gasplant_generate.profile_initialize)
        "idaes_start": "profile",
    },
}


def component_data(names) -> dict:
    """difflow's constants for ``names``, as plain lists."""
    from difflow_refinery.gasplant import gas_components

    c = gas_components(names)
    return {"names": list(c.names), "MW": [float(v) for v in c.MW],
            "Tc": [float(v) for v in c.Tc], "Pc": [float(v) for v in c.Pc],
            "omega": [float(v) for v in c.omega],
            "cp_ig": [[float(v) for v in row] for row in c.cp]}


def difflow_column(case: dict):
    """The same column in difflow: a :class:`GasPlantColumn` and its feed."""
    from difflow_refinery.gasplant import GasPlantColumn, GasPlantColumnParams, gas_components
    from difflow_refinery.vacuum.column import StageSpec

    c = gas_components(case["names"])
    p = GasPlantColumnParams(
        components=c, n_trays=case["n_trays"], feed_trays={"feed": case["feed_tray"]},
        condenser="total", reboiler=True, top_P=case["P"], tray_dP=0.0,
        specs=(StageSpec("reflux_ratio", case["reflux_ratio"], replaces="distillate.rate"),
               StageSpec("boilup_ratio", case["boilup_ratio"], replaces="reboiler.duty")),
        tray_efficiency=1.0, overhead_keys=tuple(case["names"][:3]),
        max_iter=100)
    feed = {f"F_{n}": case["F"] * z for n, z in zip(case["names"], case["z"])}
    feed.update(T=case["T"], P=case["P"])
    return GasPlantColumn(p), feed
