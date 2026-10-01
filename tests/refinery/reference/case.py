"""The validation case: the test crude in the 30-stage atmospheric column.

The same assay, layout and specs as ``tests/refinery/test_unit.py``: 95 000
bbl/d at 240 C and 6 bar into the furnace, a 30-stage column fed on stage 27,
kerosene, diesel and AGO side strippers, two pumparounds, bottom and stripper
steam, a total condenser, and a 5 vol % overflash closing the furnace.

Everything the reference models need is here as plain numbers. The only call
into difflow is :func:`difflow_case`, which characterises the crude: the
pseudo-component constants are an *input* to both models, so the column
comparison (layer 3) tests the column, not the correlations (layer 1 does
that, against published values).
"""

from __future__ import annotations

from ..test_column import LIGHT, PCT, STEAM, T_C

BPD = 95_000.0
T_FURNACE_IN = 273.15 + 240.0
P_FURNACE_IN = 6.0e5
FURNACE_EFFICIENCY = 0.85

#: Column layout, stage numbers 1 (top) to N. A stripper's vapour returns to
#: the stage above its draw.
LAYOUT = {
    "n_stages": 30,
    "feed_stage": 27,
    "P_top": 1.5e5,
    "P_bottom": 1.9e5,
    "P_condenser": 1.3e5,
    "steam_T": 273.15 + 260.0,
    "bottom_steam": STEAM["bottom"],
    # name, draw stage, stripper stages, steam (mol/s)
    "side_products": [
        ["kero", 9, 4, STEAM["kero"]],
        ["diesel", 16, 4, STEAM["diesel"]],
        ["ago", 22, 3, STEAM["ago"]],
    ],
    # name, draw stage, return stage
    "pumparounds": [["pa1", 12, 10], ["pa2", 19, 17]],
    "distillate": "naphtha",
}

#: Specs: product rates as fractions of the crude's standard volume,
#: pumparound duties (W) and draw-minus-return temperature (K), overflash
#: (liquid off the stage above the flash zone, standard volume over crude).
SPECS = {
    "volume_yield": {"naphtha": 0.20, "kero": 0.11, "diesel": 0.17, "ago": 0.05},
    "pa_duty": {"pa1": 15e6, "pa2": 20e6},
    "pa_delta_T": {"pa1": 60.0, "pa2": 60.0},
    "overflash": 0.05,
}


def column_params(Vf, volume_yield=None, overflash=None, pa_duty=None):
    """difflow's :class:`CrudeColumnParams` for the case.

    Args:
        Vf: Crude standard volume flow (m^3/s) the yield specs are fractions of.
        volume_yield, overflash, pa_duty: Replace entries of :data:`SPECS`.
            Values may be JAX tracers (layer 4 differentiates through them).
    """
    from difflow_refinery import column as cc

    vy = dict(SPECS["volume_yield"], **(volume_yield or {}))
    pad = dict(SPECS["pa_duty"], **(pa_duty or {}))
    of = SPECS["overflash"] if overflash is None else overflash
    lay = LAYOUT
    return cc.CrudeColumnParams(
        n_stages=lay["n_stages"], feed_stage=lay["feed_stage"], P_top=lay["P_top"],
        P_bottom=lay["P_bottom"], P_condenser=lay["P_condenser"],
        bottom_steam=lay["bottom_steam"], steam_T=lay["steam_T"],
        side_products=tuple(cc.SideProduct(n, d, s, steam=st)
                            for n, d, s, st in lay["side_products"]),
        pumparounds=tuple(cc.Pumparound(n, d, r) for n, d, r in lay["pumparounds"]),
        specs=tuple(cc.product_rate(n, f * Vf) for n, f in vy.items())
        + tuple(s for pa in ("pa1", "pa2")
                for s in (cc.pumparound_duty(pa, pad[pa]),
                          cc.pumparound_delta_t(pa, SPECS["pa_delta_T"][pa])))
        + (cc.overflash(of),),
        furnace=cc.Furnace(efficiency=FURNACE_EFFICIENCY),
    )


def difflow_case():
    """difflow's characterisation and :class:`CrudeUnit` for the case.

    The feed volume the yield specs are fractions of is the standard volume
    of the characterised feed, the basis :mod:`difflow_refinery.products`
    reports yields on, so a 20 vol % spec reads back as 20 vol %.

    Returns:
        ``(unit, crude, thermo, Vf)``; ``Vf`` in m^3/s.
    """
    import jax.numpy as jnp

    import difflow_refinery as dr
    from difflow_refinery import Assay, characterize
    from difflow_refinery import column as cc
    from difflow_refinery.thermo import ColumnThermo

    assay = Assay(PCT, [t + 273.15 for t in T_C], sg=0.86, light_ends=LIGHT)
    crude = characterize(assay)
    thermo = ColumnThermo.from_characterization(crude)
    kg_s = BPD * cc.BARREL / 86400.0 * float(crude.bulk_sg) * 999.016
    feed = crude.stream(kg_s, T=T_FURNACE_IN, P=P_FURNACE_IN, basis="mass")
    Vf = float(thermo.std_volume(jnp.stack([jnp.asarray(feed[f"F_{n}"]) for n in thermo.names])))
    unit = dr.CrudeUnit(assay, column_params(Vf))
    return unit, crude, thermo, Vf


def solve_difflow(unit, Vf, **spec_overrides):
    """Run the unit at the case's feed; ``spec_overrides`` go to :func:`column_params`."""
    params = column_params(Vf, **spec_overrides) if spec_overrides else None
    return unit.solve(BPD, T=T_FURNACE_IN, P=P_FURNACE_IN, params=params)


def feed_flows(unit, thermo) -> list:
    """The crude's molar flows (mol/s) at the case's rate, per component."""
    f = unit.feed(BPD, T=T_FURNACE_IN, P=P_FURNACE_IN)
    return [float(f[f"F_{n}"]) for n in thermo.names]


def component_data(crude, thermo) -> dict:
    """The pseudo-component constants both models are built on, as lists.

    ``hvap_nb`` is the heat of vaporisation at ``Tb`` (Riedel for the cuts,
    the tabulated value for the light ends); the reference derives the Watson
    constant from it itself rather than taking difflow's.
    """
    import numpy as np

    n_le = len(crude.light_names)
    from difflow_refinery.thermo import LIGHT_ENDS

    hvap = [LIGHT_ENDS[n][5] for n in crude.light_names] + [float(v) for v in crude.hvap_nb]
    omega_eos = [LIGHT_ENDS[n][4] for n in crude.light_names] + [float(v) for v in crude.omega]

    def lst(a):
        return [float(v) for v in np.asarray(a).reshape(-1)]

    return {
        "names": list(thermo.names),
        "n_light_ends": n_le,
        "MW": lst(thermo.MW),
        "SG": lst(thermo.SG),
        "Tb": lst(thermo.Tb),
        "Tc": lst(thermo.Tc),
        "Pc": lst(thermo.Pc),
        "omega_vp": lst(thermo.omega_vp),
        "omega_eos": omega_eos,
        "hvap_nb": hvap,
        "cp_ig": [lst(r) for r in np.asarray(thermo.cp_ig)],
        "volume_fraction": lst(crude.volume_fraction),
        "mole_fraction": lst(crude.mole_fraction),
    }
