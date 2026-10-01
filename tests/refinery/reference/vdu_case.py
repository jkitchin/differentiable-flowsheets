"""The vacuum-column validation case: heavy crude's atmospheric residue in the VDU.

The same feed and column as ``tests/refinery/test_vacuum.py``'s ``heavy``
fixture and the vacuum table in ``docs/unit-operations-refinery.md``: the
synthetic heavy sour crude, cut on the default 300-800 C grid, its
atmospheric residue (370 C cut, 100 kg/s of crude) at 360 C and 200 kPa,
into the default nine-stage column (two-stage LVGO, HVGO and wash beds, a
flash zone, two stripping stages) with the default specs: 70 C top, 3 %
overflash, LVGO T95 of 450 C.

The defaults that :class:`~difflow_refinery.vacuum.VacuumColumnParams`
derives from the feed (stripping steam 0.5 % of feed, pumparounds 0.3 and
0.8 x feed) are written out here as kg/s of the *frozen* feed, so both
models are handed numbers rather than rules.

The only calls into difflow are :func:`characterize` (the component table
and the feed, an input to both models) and :func:`column_params` /
:func:`solve_difflow` (the model under test). The component table is frozen
into ``vdu_reference.json`` and the comparisons rebuild difflow's
:class:`PseudoComponents` from it, so a change to the characterisation
cannot move the column comparison -- it fails
``TestReferenceIsCurrent`` instead, which says to regenerate.
"""

from __future__ import annotations

MMHG = 133.322368
C_TO_K = 273.15

CRUDE = {"assay": "heavy_crude", "crude_rate_kg_s": 100.0, "cut_point_C": 370.0,
         "sharpness_C": 12.0, "feed_T": 360.0 + C_TO_K, "feed_P": 200e3}

#: Section sizes and the operating knobs, in SI. Stage 0 is the top.
LAYOUT = {
    "n_lvgo": 2, "n_hvgo": 2, "n_wash": 2, "n_strip": 2,
    "furnace_T": 400.0 + C_TO_K,
    "transfer_line_dP": 2000.0,
    "flash_zone_P": 30.0 * MMHG,
    "top_P": 10.0 * MMHG,
    "stripping_dP": 300.0,
    "steam_T": 300.0 + C_TO_K,
    "entrainment": 0.01,
    "deentrainment": 0.9,
    #: multiples of the feed's mass rate (what the params default to)
    "steam_per_feed": 0.005,
    "lvgo_pa_per_feed": 0.3,
    "hvgo_pa_per_feed": 0.8,
}

#: Murphree vapour efficiency per bed; the default case is equilibrium
#: stages, the second case exercises the efficiency path.
EFFICIENCY = {"lvgo": 1.0, "hvgo": 1.0, "wash": 1.0, "strip": 1.0}
EFFICIENCY_CASE = {"lvgo": 0.8, "hvgo": 0.7, "wash": 0.5, "strip": 0.4}

#: The default spec set: ``output -> target`` (K, mass fraction, K), each in
#: place of the knob ``VacuumColumn`` frees for it.
SPECS = {"top.T": 70.0 + C_TO_K, "overflash": 0.03, "lvgo.T95": 450.0 + C_TO_K}

#: The specs for the Murphree case. Beds this poor pass enough heavy vapour
#: to the LVGO section that a 450 C LVGO end point is unreachable at any
#: pumparound duty -- difflow and the reference both stop finding a column
#: once the efficiencies are halfway from 1 to EFFICIENCY_CASE (and
#: ``test_vacuum.py`` already pins that a 70 % HVGO bed cannot make it) -- so
#: the end point is relaxed to 520 C, as that test does.
EFFICIENCY_SPECS = dict(SPECS, **{"lvgo.T95": 520.0 + C_TO_K})

COMPONENT_FIELDS = ("Tb", "SG", "MW", "Kw", "Tc", "Pc", "omega", "T_lo", "T_hi", "sulfur",
                    "nitrogen", "ccr", "nickel_vanadium", "asphaltenes")


def characterize() -> tuple[dict, list]:
    """difflow's characterisation of the case's crude and its residue.

    Returns:
        ``(components, feed_mol_s)``: the full :class:`PseudoComponents`
        table as lists (``names`` included), and the residue's molar flows.
    """
    import numpy as np

    from difflow_refinery import vacuum as v

    char = v.characterize(getattr(v, CRUDE["assay"])())
    feed = v.atmospheric_residue(char, crude_rate_kg_s=CRUDE["crude_rate_kg_s"],
                                 cut_point_C=CRUDE["cut_point_C"],
                                 sharpness_C=CRUDE["sharpness_C"],
                                 T_C=CRUDE["feed_T"] - C_TO_K, P=CRUDE["feed_P"])
    c = char.components
    comp = {k: [float(x) for x in np.asarray(getattr(c, k))] for k in COMPONENT_FIELDS}
    comp["names"] = list(c.names)
    return comp, [float(feed[f"F_{n}"]) for n in c.names]


def feed_mass(comp: dict, feed_mol_s) -> float:
    """Feed mass rate (kg/s)."""
    return sum(f * mw for f, mw in zip(feed_mol_s, comp["MW"])) / 1000.0


def components(comp: dict):
    """difflow's :class:`PseudoComponents` rebuilt from the frozen table."""
    import jax.numpy as jnp

    from difflow_refinery.vacuum.assay import PseudoComponents

    return PseudoComponents(names=tuple(comp["names"]),
                            **{k: jnp.asarray(comp[k], dtype=float) for k in COMPONENT_FIELDS})


def feed_stream(comp: dict, feed_mol_s, **override):
    """The frozen feed as a difflow stream; ``override`` replaces ``F_<name>``."""
    import jax.numpy as jnp

    s = {f"F_{n}": jnp.asarray(f, dtype=float) for n, f in zip(comp["names"], feed_mol_s)}
    s.update(override)
    s["T"] = jnp.asarray(CRUDE["feed_T"])
    s["P"] = jnp.asarray(CRUDE["feed_P"])
    return s


def column_params(comp: dict, feed_mol_s, efficiency=None, specs=None, **knobs):
    """difflow's :class:`VacuumColumnParams` for the case.

    ``knobs`` replace :data:`LAYOUT` entries (``furnace_T``, ``flash_zone_P``,
    ...) and may be JAX tracers; ``efficiency`` replaces :data:`EFFICIENCY`
    and ``specs`` :data:`SPECS` (entry by entry).
    """
    from difflow_refinery import vacuum as v

    lay = dict(LAYOUT, **knobs)
    eff = dict(EFFICIENCY, **(efficiency or {}))
    F = feed_mass(comp, feed_mol_s)
    sp = dict(SPECS, **(specs or {}))
    specs = (v.StageSpec("top.T", sp["top.T"], replaces="lvgo_pa.duty"),
             v.StageSpec("overflash", sp["overflash"], replaces="hvgo.rate"),
             v.StageSpec("lvgo.T95", sp["lvgo.T95"], replaces="hvgo_pa.duty"))
    return v.VacuumColumnParams(
        components=components(comp),
        furnace_T=lay["furnace_T"], transfer_line_dP=lay["transfer_line_dP"],
        flash_zone_P=lay["flash_zone_P"], top_P=lay["top_P"],
        stripping_dP=lay["stripping_dP"], steam_rate=lay["steam_per_feed"] * F,
        steam_T=lay["steam_T"], coil_steam_rate=0.0,
        lvgo_pa_rate=lay["lvgo_pa_per_feed"] * F, hvgo_pa_rate=lay["hvgo_pa_per_feed"] * F,
        entrainment=lay["entrainment"], deentrainment=lay["deentrainment"],
        n_lvgo=lay["n_lvgo"], n_hvgo=lay["n_hvgo"], n_wash=lay["n_wash"],
        n_strip=lay["n_strip"],
        lvgo_efficiency=eff["lvgo"], hvgo_efficiency=eff["hvgo"],
        wash_efficiency=eff["wash"], strip_efficiency=eff["strip"],
        specs=specs)


def solve_difflow(comp: dict, feed_mol_s, efficiency=None, specs=None, **knobs):
    """Run difflow's column on the frozen case; returns ``(streams..., info)``."""
    from difflow_refinery import vacuum as v

    col = v.VacuumColumn(column_params(comp, feed_mol_s, efficiency, specs, **knobs))
    return col(feed_stream(comp, feed_mol_s))
