"""The DWSIM gas-plant cases: inputs shared by ``dwsim_gasplant_generate.py``
and ``tests/refinery/test_dwsim_gasplant.py``.

* **Columns** (``COLUMNS``): the two cases of the IDAES reference
  (:data:`.gasplant_case.CASES`: ``debutanizer``, ``c3c4_splitter``, used as
  they are) and a ``naphtha_splitter`` on real C4-C6 components and four
  naphtha pseudo-components (Twu constants, the smoke cases' cuts). Reflux
  and boilup ratios fixed, equilibrium trays (``tray_efficiency=1.0``), a
  total condenser at the bubble point, a kettle reboiler, no pressure drop.
  ``solver`` is the DWSIM column solver that converges the case (see the
  generator).
* **Compressor** (``COMPRESSOR``): a wet gas (H2, H2S, C1-nC6) through an
  inlet knock-out and two stages of compression, intercooling and knock-out
  -- :class:`~difflow_refinery.gasplant.GasCompressor`'s construction.
* **Vapour pressure** (``RVP``): two liquids, a debutanized C3-C5 bottoms and
  a stabilized naphtha with pseudo-components, for difflow's D323
  construction and the true vapour pressure at 100 F.

Comparison (a) puts difflow's constants (and difflow's kij, zero for every
pair here but H2S with the light paraffins) into DWSIM as hypothetical
compounds; (b) uses DWSIM's database compounds and DWSIM's own kij. The
constants are frozen into the JSON; a change to them is a staleness
failure, not a disagreement.
"""

from __future__ import annotations

from . import gasplant_case as gc

#: Naphtha pseudo-components (name, NBP K, SG 60/60 F), as in the smoke cases.
NAPHTHA_CUTS = [("NBP_375", 375.0, 0.715), ("NBP_405", 405.0, 0.745),
                ("NBP_435", 435.0, 0.770), ("NBP_465", 465.0, 0.790)]

COLUMNS: dict[str, dict] = {
    "debutanizer": {**gc.CASES["debutanizer"], "pseudo": [], "solver": "naphtali-sandholm"},
    "c3c4_splitter": {**{k: v for k, v in gc.CASES["c3c4_splitter"].items() if k != "reference"},
                      "pseudo": [], "solver": "wang-henke"},
    "naphtha_splitter": {
        "names": ["n_butane", "isopentane", "n_pentane", "n_hexane"], "pseudo": NAPHTHA_CUTS,
        "z": [0.04, 0.10, 0.14, 0.16, 0.20, 0.16, 0.12, 0.08],
        "F": 100.0, "T": 390.0, "P": 2.5e5,
        "n_trays": 20, "feed_tray": 10,
        "reflux_ratio": 1.5, "boilup_ratio": 1.2,
        "solver": "wang-henke",
    },
}

COMPRESSOR: dict = {
    "names": ["hydrogen", "hydrogen_sulfide", "methane", "ethane", "propane", "isobutane",
              "n_butane", "isopentane", "n_pentane", "n_hexane"],
    "z": [0.08, 0.04, 0.22, 0.14, 0.16, 0.07, 0.11, 0.06, 0.07, 0.05],
    "F": 100.0, "T": 313.15, "P": 1.6e5,
    "outlet_P": 14e5, "n_stages": 2, "efficiency": 0.75, "cooler_T": 313.15, "cooler_dP": 0.0,
}

RVP: dict[str, dict] = {
    "debutanizer_bottoms": {
        "names": ["propane", "isobutane", "n_butane", "isopentane", "n_pentane"], "pseudo": [],
        "z": [0.001, 0.03, 0.15, 0.40, 0.419],
    },
    "stabilized_naphtha": {
        "names": ["n_butane", "isopentane", "n_pentane", "n_hexane"], "pseudo": NAPHTHA_CUTS[:3],
        "z": [0.03, 0.12, 0.15, 0.20, 0.20, 0.15, 0.15],
    },
}


def component_data(names, pseudo=(), kij_from_gas_components: bool = True) -> dict:
    """difflow's constants: ``names``, ``MW``, ``Tc``, ``Pc``, ``omega``,
    ``cp_ig`` (J/mol/K cubic), ``Tb``/``SG`` (``None`` for real components)
    and the ``kij`` matrix :func:`~difflow_refinery.gasplant.gas_components`
    gives the real components (zero for the pseudo-components)."""
    import numpy as np

    from . import dwsim_smoke_case as sm

    out = sm.component_data({"light": list(names), "pseudo": list(pseudo)})
    n = len(out["names"])
    K = np.zeros((n, n))
    if kij_from_gas_components:
        from difflow_refinery.gasplant import gas_components

        k = np.asarray(gas_components(list(names)).kij)
        K[:len(names), :len(names)] = k
    out["kij"] = K.tolist()
    return out


def gas_components_of(comp: dict):
    """A :class:`~difflow_refinery.gasplant.GasComponents` on frozen constants."""
    import jax.numpy as jnp
    import numpy as np

    from difflow_refinery.gasplant import GasComponents

    n = len(comp["names"])
    return GasComponents(MW=jnp.asarray(comp["MW"]), Tc=jnp.asarray(comp["Tc"]),
                         Pc=jnp.asarray(comp["Pc"]), omega=jnp.asarray(comp["omega"]),
                         cp=jnp.asarray(comp["cp_ig"]), kij=jnp.asarray(comp["kij"]),
                         lhv=jnp.zeros(n), pseudo=jnp.asarray(np.array(
                             [0.0 if t is None else 1.0 for t in comp["Tb"]])),
                         names=tuple(comp["names"]))


def difflow_column(case: dict, comp: dict):
    """The case as a :class:`~difflow_refinery.gasplant.GasPlantColumn` and its feed."""
    from difflow_refinery.gasplant import GasPlantColumn, GasPlantColumnParams
    from difflow_refinery.vacuum.column import StageSpec

    c = gas_components_of(comp)
    names = comp["names"]
    p = GasPlantColumnParams(
        components=c, n_trays=case["n_trays"], feed_trays={"feed": case["feed_tray"]},
        condenser="total", reboiler=True, top_P=case["P"], tray_dP=0.0,
        specs=(StageSpec("reflux_ratio", case["reflux_ratio"], replaces="distillate.rate"),
               StageSpec("boilup_ratio", case["boilup_ratio"], replaces="reboiler.duty")),
        tray_efficiency=1.0, overhead_keys=tuple(names[:3]), max_iter=100)
    feed = {f"F_{n}": case["F"] * z for n, z in zip(names, case["z"])}
    feed.update(T=case["T"], P=case["P"])
    return GasPlantColumn(p), feed


def difflow_compressor(comp: dict):
    """:class:`~difflow_refinery.gasplant.GasCompressor` on the frozen constants and its inlet."""
    from difflow_refinery.gasplant import GasCompressor, GasCompressorParams

    c = COMPRESSOR
    unit = GasCompressor(GasCompressorParams(
        components=gas_components_of(comp), outlet_P=c["outlet_P"], n_stages=c["n_stages"],
        efficiency=c["efficiency"], cooler_T=c["cooler_T"], cooler_dP=c["cooler_dP"]))
    feed = {f"F_{n}": c["F"] * z for n, z in zip(comp["names"], c["z"])}
    feed.update(T=c["T"], P=c["P"])
    return unit, feed
