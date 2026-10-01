"""The preheat-train cases: the validation column behind an eight-exchanger train.

The column is :mod:`.case`'s 30-stage atmospheric column, with one change:
each pumparound is specified by its circulation and its return temperature
instead of its duty and delta-T, because the train cools it and so the
return temperature becomes the train's answer (the spec's value is only the
loop's starting guess).

The train is a textbook layout -- not any real unit's. Cold train: kerosene,
residue (cold end) and the top pumparound heat the crude to the desalter;
hot train: diesel, AGO, then the preflash drum, then the bottom pumparound
and the residue twice more, hottest last.

Two crudes: the validation case's own (``"base"``, 0.86 SG) and a heavier
one (``"heavy"``, every TBP point 25 C higher above 10 %, 0.885 SG) -- an
invented assay, used only to show the coupled unit converges from the same
default start on a different crude.

:func:`small_case` is a cheap stand-in for the per-commit tests: an 8-stage
column with no pumparounds and a three-exchanger train.
"""

from __future__ import annotations

from dataclasses import replace

from ..test_column import LIGHT, PCT, T_C
from . import case

#: Exchangers: name -> (U W/m^2/K, area m^2). One clean U for all; the
#: areas set the duties.
EXCHANGERS = {
    "E1": (350.0, 300.0), "E2": (350.0, 1200.0), "E3": (350.0, 1500.0),
    "E4": (350.0, 600.0), "E5": (350.0, 300.0), "E6": (350.0, 1500.0),
    "E7": (350.0, 1500.0), "E8": (350.0, 1200.0),
}
#: Hot streams, hottest exchanger first.
HOT_STREAMS = {
    "residue": ("E8", "E7", "E2"), "pa2": ("E6",), "ago": ("E5",),
    "diesel": ("E4",), "pa1": ("E3",), "kero": ("E1",),
}
CRUDE_PATH = ("E1", "E2", "E3", "desalter", "E4", "E5", "preflash", "E6", "E7", "E8")
#: Pumparound circulation (fraction of the crude's standard volume) and the
#: return temperature the loop starts from (K).
PA_RATE = {"pa1": 0.5, "pa2": 0.6}
PA_RETURN_GUESS = {"pa1": 400.0, "pa2": 470.0}
DRUM_P = 3.0e5
T_TANK = 300.0

#: The two crudes: (TBP in C, SG). "heavy" is invented (see the module docstring).
ASSAYS = {
    "base": (T_C, 0.86),
    "heavy": ([t + (25.0 if p > 10 else 0.0) for p, t in zip(PCT, T_C)], 0.885),
}


def assay(name: str = "base"):
    from difflow_refinery import Assay

    tbp, sg = ASSAYS[name]
    return Assay(PCT, [t + 273.15 for t in tbp], sg=sg, light_ends=LIGHT)


def column_params(Vf):
    """:func:`.case.column_params` with the pumparounds on rate and return temperature."""
    from difflow_refinery import column as cc

    base = case.column_params(Vf)
    specs = tuple(s for s in base.specs if not s.kind.startswith("pa_"))
    for pa in ("pa1", "pa2"):
        specs += (cc.pumparound_rate(pa, PA_RATE[pa] * Vf),
                  cc.pumparound_return_temperature(pa, PA_RETURN_GUESS[pa]))
    return replace(base, specs=specs)


def train_params(Rf=None):
    """The eight-exchanger train; ``Rf`` a ``{name: m^2 K/W}`` dict (default clean)."""
    import difflow_refinery as dr

    Rf = Rf or {}
    return dr.PreheatTrainParams(
        tuple(dr.PreheatExchanger(n, U, A, Rf=Rf.get(n, 0.0)) for n, (U, A) in EXCHANGERS.items()),
        tuple(dr.HotStream(s, e) for s, e in HOT_STREAMS.items()),
        CRUDE_PATH, desalter=dr.DesalterParams(), drum=dr.PreflashDrumParams(P=DRUM_P))


def preheated_unit(name: str = "base", Rf=None):
    """``(unit, Vf)``: the :class:`PreheatedCrudeUnit` for a crude, and the
    standard volume flow (m^3/s) of :data:`.case.BPD` of it, which the
    yield specs are fractions of."""
    import jax.numpy as jnp

    import difflow_refinery as dr
    from difflow_refinery import column as cc

    a = assay(name)
    crude = dr.characterize(a)
    th = dr.ColumnThermo.from_characterization(crude)
    kg_s = case.BPD * cc.BARREL / 86400.0 * float(crude.bulk_sg) * 999.016
    s = crude.stream(kg_s, T=300.0, P=1e5, basis="mass")
    Vf = float(th.std_volume(jnp.stack([jnp.asarray(s[f"F_{n}"]) for n in th.names])))
    return dr.PreheatedCrudeUnit(a, column_params(Vf), train_params(Rf)), Vf


# -----------------------------------------------------------------------------
# The cheap case
# -----------------------------------------------------------------------------

SMALL_ASSAY = ([0, 30, 70, 100], [320.0, 480.0, 640.0, 900.0], 0.85)
SMALL_RATE = 100.0  # mol/s


def small_case(drum: bool = True, desalter: bool = True, vapor_fraction=None):
    """``(assay, column params, train params)`` for an 8-stage column behind a
    three-exchanger train, hot streams the residue and the naphtha."""
    import difflow_refinery as dr
    from difflow_refinery import column as cc

    a = dr.Assay(SMALL_ASSAY[0], SMALL_ASSAY[1], sg=SMALL_ASSAY[2])
    col = cc.CrudeColumnParams(
        n_stages=8, feed_stage=7, bottom_steam=20.0, P_top=1.4e5, P_bottom=1.6e5,
        specs=(cc.product_rate("naphtha", 25.0, basis="mole"), cc.coil_outlet_temperature(620.0)))
    ex = (dr.PreheatExchanger("E1", 300.0, 40.0), dr.PreheatExchanger("E2", 300.0, 80.0, Rf=3e-4),
          dr.PreheatExchanger("E3", 300.0, 80.0, Rf=5e-4))
    path = ["E1"] + (["desalter"] if desalter else []) + ["E2"] + (["preflash"] if drum else []) + ["E3"]
    tp = dr.PreheatTrainParams(
        ex, (dr.HotStream("residue", ("E3", "E2")), dr.HotStream("naphtha", ("E1",))), tuple(path),
        desalter=dr.DesalterParams() if desalter else None,
        drum=dr.PreflashDrumParams(P=2.0e5, vapor_fraction=vapor_fraction) if drum else None)
    return a, col, tp
