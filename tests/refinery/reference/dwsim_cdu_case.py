"""The DWSIM crude-side cases: inputs shared by ``dwsim_cdu_generate.py`` and
``tests/refinery/test_dwsim_cdu.py``.

Four comparisons of ``difflow_refinery``'s crude side with DWSIM 9.0.5, each
labelled by what it tests (see the docs section "Validation against DWSIM:
characterization and crude unit"):

1. **Characterization.** Two assays through difflow's :func:`characterize`
   and DWSIM's distillation-curve characterization, on the SAME cut
   temperatures: the test crude of the CDU validation (``case.py``, 95 000
   bbl/d, volume basis, three light ends) and a heavy crude whose TBP curve
   stops at 60 wt % (difflow's :class:`HeavyEnd` extends it; DWSIM cannot).
   DWSIM's defaults, its closest options to difflow's, and DWSIM's
   correlations evaluated at difflow's own (Tb, SG) -- which separates the
   correlations from the rest of the pipeline.
2. **Thermodynamics on the crude.** (a) difflow's ``ColumnThermo``
   (Raoult, Lee-Kesler Psat, ideal-gas Cp, Watson latent heat) against
   DWSIM's Raoult's Law package on difflow's constants -- the same model,
   an implementation check; (b) DWSIM's PR, Grayson-Streed and
   Lee-Kesler-Plocker on DWSIM's own characterization of the same crude --
   a model check. Bubble and dew points, the flash zone, the heating path
   and the furnace duty.
3. **The atmospheric column.** DWSIM's rigorous column has no side strippers,
   no pumparounds and no free-water phase (``dwsim_columns``), so the CDU
   case of ``case.py`` cannot be built in it. :data:`COLUMN_A` is the
   largest configuration both can build exactly: the main column with liquid
   side draws, a total condenser, no steam -- solved by both on the same
   model and constants. The full case's differences are reported, not
   simulated.
4. **The vacuum feed.** The CDU's atmospheric residue flashed at vacuum
   flash-zone conditions.

Everything here is plain numbers except :func:`difflow_crude` and friends,
which call difflow: the constants are an input to both sides and are frozen
into the reference.
"""

from __future__ import annotations

from . import case

#: The test crude of the CDU validation (``case.py`` / ``test_column.py``).
TEST_CRUDE = {"pct": [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
              "T_C": [-10, 60, 95, 150, 205, 260, 315, 370, 430, 500, 600, 680, 850],
              "sg": 0.86, "basis": "volume",
              "light_ends": {"propane": 0.5, "n_butane": 1.0, "n_pentane": 1.5}}

#: The heavy crude (:func:`difflow_refinery.vacuum.assay.heavy_crude`): mass
#: basis, API 20, the curve stops at 60 wt % at 565 C; a :class:`HeavyEnd`.
HEAVY_CRUDE = {"pct": [1, 4, 8, 13, 18, 24, 31, 38, 45, 52, 58, 60],
               "T_C": [36, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 565],
               "api": 20.0, "basis": "mass", "light_ends": {}}

#: DWSIM's characterization options, by run. ``default`` is the UI's;
#: ``rd`` is the closest DWSIM has to one of difflow's methods (difflow
#: ``riazi_daubert_1987`` = DWSIM "Riazi-Daubert (1985)" Tc and Pc, the same
#: equations; MW Riazi (1986) is the same equation as difflow's RD 1987 MW
#: but with one coefficient truncated, see the docs); no omega fit.
DWSIM_CHAR_RUNS = {
    "default": {},
    "rd": {"Tc_corr": "Riazi-Daubert (1985)", "Pc_corr": "Riazi-Daubert (1985)",
           "mw_corr": "Riazi (1986)", "adjust_omega": False},
}

#: Temperatures (K) at which ideal-gas Cp is tabulated per cut.
T_CP = (300.0, 600.0, 800.0)

# -- thermo on the crude ---------------------------------------------------

#: Pressures (Pa) of the crude's bubble and dew points.
BUBBLE_DEW_P = (1.0e5, 2.0e5)
#: Furnace inlet (``case.py``) and the coil outlet / flash zone of the CDU
#: reference solution (``cdu_reference.json``: coil outlet 586.30 K; flash
#: zone pressure = the feed stage's, 27 of 30 between 1.5 and 1.9 bar).
T_IN, P_IN = case.T_FURNACE_IN, case.P_FURNACE_IN
T_COT = 586.3028
P_FZ = 1.5e5 + (1.9e5 - 1.5e5) * 26 / 29
#: Points along the heating path, (T K, P Pa): furnace inlet to coil outlet.
HEAT_PATH = [(T_IN + (T_COT - T_IN) * f, P_IN + (P_FZ - P_IN) * f) for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
#: DWSIM packages of the model check, on DWSIM's own fractions.
MODEL_PACKAGES = ("PR", "GS", "LKP", "RAOULT")

# -- the column ----------------------------------------------------------

#: The largest configuration both simulators can build exactly: the main
#: column of ``case.py`` without strippers, pumparounds or steam; the crude
#: enters the bottom stage (no stripping section without steam), at a
#: temperature above the full case's coil outlet so the flash-zone vapour
#: carries the products; liquid side draws at kero/diesel/AGO's draw stages
#: of ``case.py``, at molar rates.
COLUMN_A = {
    "n_stages": 26, "feed_stage": 26, "T_feed": 600.0, "P_feed": 1.86e5,
    "P_top": 1.5e5, "P_bottom": 1.86e5, "P_condenser": 1.3e5,
    "distillate": 250.0,
    "side_draws": [["kero", 9, 100.0], ["diesel", 16, 110.0], ["ago", 22, 25.0]],
}

#: What DWSIM 9.0.5's rigorous column did with :data:`COLUMN_A` (measured
#: by hand, recorded here because the runs take tens of minutes each and
#: none produced an answer; the generator does not repeat them).
COLUMN_A_ATTEMPTS = [
    {"configuration": "refluxed absorber (no reboiler), total condenser, Naphtali-Sandholm",
     "start": "DWSIM's own estimates, then a linear 380-590 K profile",
     "outcome": "NaN on the first function evaluation ('Error evaluating error functions'), "
                "470 s / 370 s; reproduced on a 5-component column in 1 s"},
    {"configuration": "refluxed absorber, Wang-Henke (bubble point)",
     "start": "linear 380-590 K profile",
     "outcome": "exception in Solve_Internal ('Sequence contains no elements'), 7 s"},
    {"configuration": "distillation column, reboiler duty spec 0, Naphtali-Sandholm / Wang-Henke",
     "start": "DWSIM's own estimates, and a consistent hand profile",
     "outcome": "NaN at once (NS: the duty-spec row is 0/0); WH 'convergence error' "
                "(5-component column)"},
    {"configuration": "distillation column, bottom-stage temperature spec (the secant "
                      "construction of dwsim_columns.DWSIMColumn), Naphtali-Sandholm",
     "start": "linear 380-590 K profile, 1000 / 500 mol/s",
     "outcome": "iteration cap after 245 s, error stuck at 1.2e17 for 1500 Broyden steps"},
    {"configuration": "the same", "start": "difflow's converged T, V, L profile",
     "outcome": "iteration cap after 870 s, error stuck at 1.3e17"},
    {"configuration": "the same, Naphtali-Sandholm and Wang-Henke",
     "start": "difflow's converged T, V, L AND stage compositions",
     "outcome": "no answer after 42 / 31 CPU-minutes; stopped"},
]

#: A column DWSIM's solver does converge: five of the crude's cuts
#: (pc03-pc11 by twos, Tb 363-563 K), 10 stages, the feed (20 mol/s of each,
#: 60 mol % vaporized at 1.6 bar -- 471.7 K) on the bottom stage, one liquid
#: side draw, a total condenser, no reboiler; the same model and constants on
#: both sides. The bottom-stage secant starts 2 and 4 K below the feed.
COLUMN_SMALL = {
    "components": ["pc03", "pc05", "pc07", "pc09", "pc11"], "feed_each": 20.0,
    "n_stages": 10, "feed_stage": 10, "T_feed": 471.7135, "P_feed": 1.6e5,
    "P_top": 1.5e5, "P_bottom": 1.6e5, "P_condenser": 1.3e5,
    "distillate": 30.0, "side_draws": [["sd", 5, 15.0]],
}

# -- the vacuum feed -------------------------------------------------------

#: Vacuum flash-zone conditions for the atmospheric residue: (T K, P Pa).
VACUUM_FLASH = [(673.15, 50 * 133.322368), (673.15, 100 * 133.322368),
                (693.15, 75 * 133.322368)]


# ---------------------------------------------------------------------------
# difflow's side (inputs to both)
# ---------------------------------------------------------------------------

def assay_of(spec: dict):
    """difflow's :class:`Assay` for one of the assay dicts above."""
    import difflow_refinery as dr

    T_K = [t + 273.15 for t in spec["T_C"]]
    kw = dict(basis=spec["basis"], light_ends=dict(spec["light_ends"]))
    if "sg" in spec:
        kw["sg"] = spec["sg"]
    else:
        kw["api"] = spec["api"]
    if spec is HEAVY_CRUDE or spec.get("heavy_end"):
        kw["heavy_end"] = dr.HeavyEnd()
    return dr.Assay(spec["pct"], T_K, **kw)


def characterization_table(char, method: str) -> dict:
    """difflow's per-component table of a characterization, as lists."""
    import numpy as np

    from difflow_refinery import correlations as corr

    def lst(a):
        return [float(v) for v in np.asarray(a).reshape(-1)]

    cp = np.asarray(char.cp_ig_coeffs)
    return {
        "method": method,
        "pseudo_names": list(char.pseudo_names), "light_names": list(char.light_names),
        "cut_edges": lst(char.cut_edges),
        "Tb": lst(char.Tb), "SG": lst(char.SG), "MW": lst(char.MW), "Tc": lst(char.Tc),
        "Pc": lst(char.Pc), "omega": lst(char.omega), "omega_vp": lst(char.omega_vp),
        "Kw": lst(char.Kw), "hvap_nb": lst(char.hvap_nb),
        "cp_ig_coeffs": [lst(r) for r in cp],
        "cp_ig": [[float(r[0] + r[1] * T + r[2] * T * T + r[3] * T ** 3) for T in T_CP] for r in cp],
        "volume_fraction": lst(char.volume_fraction), "mass_fraction": lst(char.mass_fraction),
        "mole_fraction": lst(char.mole_fraction),
        "residue_lump": bool(char.residue_lump),
        "bulk_sg": float(char.bulk_sg),
        "kw_used": float(np.asarray(corr.watson_k(char.Tb, char.SG))[0]),
    }


def difflow_characterizations() -> dict:
    """difflow's characterizations of both assays (default ``twu`` and
    ``riazi_daubert_1987``), as tables."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import difflow_refinery as dr

    out = {}
    for key, spec in (("test_crude", TEST_CRUDE), ("heavy_crude", HEAVY_CRUDE)):
        a = assay_of(spec)
        out[key] = {m: characterization_table(dr.characterize(a, method=m), m)
                    for m in ("twu", "riazi_daubert_1987")}
    return out


def small_thermo(comp: dict):
    """difflow's ``ColumnThermo`` on the :data:`COLUMN_SMALL` components,
    from a component table (``case.component_data`` plus ``hvap_A``)."""
    import jax.numpy as jnp

    from difflow_refinery.thermo import ColumnThermo

    idx = [comp["names"].index(n) for n in COLUMN_SMALL["components"]]

    def a(k):
        return jnp.asarray([comp[k][i] for i in idx], dtype=float)

    return ColumnThermo(names=tuple(COLUMN_SMALL["components"]), MW=a("MW"), SG=a("SG"),
                        Tb=a("Tb"), Tc=a("Tc"), Pc=a("Pc"), omega_vp=a("omega_vp"),
                        hvap_A=a("hvap_A"), cp_ig=a("cp_ig"))


def difflow_small_column(comp: dict):
    """difflow's :class:`CrudeColumn` solve of :data:`COLUMN_SMALL`."""
    from difflow_refinery import column as cc

    cfg = COLUMN_SMALL
    th = small_thermo(comp)
    feed = {f"F_{n}": cfg["feed_each"] for n in th.names}
    feed.update(T=cfg["T_feed"], P=cfg["P_feed"])
    specs = (cc.product_rate("naphtha", cfg["distillate"], "mole"),) + tuple(
        cc.product_rate(n, r, "mole") for n, _, r in cfg["side_draws"])
    p = cc.CrudeColumnParams(
        n_stages=cfg["n_stages"], feed_stage=cfg["feed_stage"], specs=specs,
        P_top=cfg["P_top"], P_bottom=cfg["P_bottom"], P_condenser=cfg["P_condenser"],
        side_products=tuple(cc.SideProduct(n, s, 0) for n, s, _ in cfg["side_draws"]))
    return cc.CrudeColumn(p, th).solve(feed), th


def difflow_crude():
    """The CDU case's crude, thermo and feed (``case.py``); feed molar flows
    (mol/s) per component, in ``thermo.names`` order."""
    unit, crude, thermo, Vf = case.difflow_case()
    flows = case.feed_flows(unit, thermo)
    return unit, crude, thermo, Vf, flows
