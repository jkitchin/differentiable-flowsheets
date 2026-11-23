"""difflow units mapped onto benchmark process units (#533).

Each mapping does three things:

* it builds the unit's feed from the benchmark inlet's properties;
* it lumps the unit's products onto the benchmark's outlet streams;
* it names the calibration knobs, with their bounds.

:func:`~.calibration.calibrate` fits the knobs to the benchmark yields, and
:func:`~.calibration.substitute` writes the calibrated unit back.

**FCC** (``Upf3``, ``Upf4``). The outlets were identified from the stream
routing and the published properties (#533):

| Benchmark outlet | Evidence | Lump |
|---|---|---|
| s137 / s138 | published sulfur 90 wt%, sent to sulfur recovery (`Upf16`) | sour gas |
| s223 / s224 | sent to fuel-gas processing (`Upf15`) | dry gas |
| s29 / s39 | SG 0.58, sent to gas fractionation | LPG |
| s32 / s40 | RON 92-94, olefins 27.5-38 vol% | gasoline |
| s24 / s34 | cetane index 26-27 | LCO |
| s33 / s41 | SG 1.07-1.08, CCR 6 | slurry |
| s28 / s38 | sent to fuel | coke |
| s61 / s62 | a 0.1 % product | loss (not modelled) |

The sour gas is not H2S alone. ``Upf3`` sends 1.8 % of a 0.24 wt% sulfur
feed there, but that feed can make at most 0.26 % H2S. Its sulfur-free
recracking batch ``m10`` sends 24 %. So sour gas and dry gas are calibrated
as one lump, and the benchmark's split between them is not reproduced.

The knobs are catalyst activity; the riser outlet temperature (490-560 C);
the fractionator gasoline/LCO and LCO/slurry cut points (180-240 C and
320-400 C); multipliers on the dry-gas and coke rate constants (0.2-5x); and
the feed preheat temperature (150-350 C), which sets the coke in heat
balance. A soft target keeps the regenerator near 700 C, on the physical
branch of the heat balance. The kinetic constants are ILLUSTRATIVE (``fcc/CLAUDE.md``), and
calibrating them to the benchmark's perturbed yields does not validate them.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np

#: FCC outlet streams per benchmark unit, by lump (see the module docstring).
FCC_OUTLETS = {
    "Upf3": {"acid_gas": "s137", "dry_gas": "s223", "lpg": "s29", "gasoline": "s32",
             "lco": "s24", "slurry": "s33", "coke": "s28", "loss": "s61"},
    "Upf4": {"acid_gas": "s138", "dry_gas": "s224", "lpg": "s39", "gasoline": "s40",
             "lco": "s34", "slurry": "s41", "coke": "s38", "loss": "s62"},
}

#: Calibrated lumps, in order: (name, FCC output keys summed).
FCC_LUMPS = (("dry_gas+acid_gas", ("yield.dry_gas", "yield.h2s")),
             ("lpg", ("yield.lpg",)), ("gasoline", ("yield.gasoline",)),
             ("lco", ("yield.lco",)), ("slurry", ("yield.slurry",)),
             ("coke", ("yield.coke",)))

#: Knobs: (name, lower, upper, start).
FCC_KNOBS = (("activity", 0.2, 5.0, 1.0),
             ("riser_outlet_T", 763.15, 833.15, 793.15),
             ("gasoline_cut", 453.15, 513.15, 494.15),
             ("lco_cut", 593.15, 673.15, 616.15),
             ("dry_gas_rate", 0.2, 5.0, 1.0),
             ("coke_rate", 0.2, 5.0, 1.0),
             ("feed_T", 423.15, 623.15, 500.0))


def fcc_feed(base, *, sg, sulfur_wt, ccr_wt=0.0, nickel_vanadium_wppm=0.0):
    """An :class:`~difflow_refinery.fcc.FCCFeed` with the benchmark's
    properties. ``base`` supplies the boiling point, MW and the rest. Watson K
    follows the new SG, so the feed stays self-consistent."""
    sg = jnp.asarray(sg, float)
    return dataclasses.replace(
        base, SG=sg, sulfur=jnp.asarray(sulfur_wt, float) / 100.0,
        ccr=jnp.asarray(ccr_wt, float) / 100.0,
        nickel_vanadium=jnp.asarray(nickel_vanadium_wppm, float) * 1e-6,
        Kw=(1.8 * base.Tb) ** (1.0 / 3.0) / sg)


#: Regenerator temperature the calibration steers toward (K), and the scale
#: that makes a 30 K miss count like a 0.3 wt% yield miss. Without it the fit
#: can land on the heat balance's hot branch (regenerator ~1290 K, near a
#: turning point where a 1e-4 change in feed SG loses the solution).
FCC_REGEN_T_TARGET = 973.15
FCC_REGEN_T_SCALE = 1.0e4


def fcc_predict(unit, feed, theta, *, with_regenerator: bool = False):
    """FCC yields on feed, lumped as :data:`FCC_LUMPS`, at knobs ``theta``.

    With ``with_regenerator=True`` a last entry
    ``(T_regenerator - FCC_REGEN_T_TARGET) / FCC_REGEN_T_SCALE`` is appended
    (target 0), for :func:`~.calibration.calibrate`. Under ``jax.jacfwd``,
    ``theta`` and the feed's fields are traceable.
    """
    a, rot, gcut, lcut, mdg, mck, tf = (theta[i] for i in range(7))
    k = unit.theta(feed)["params"]["k_ref"]
    react = unit.scheme.reactions if hasattr(unit.scheme, "reactions") else None
    mult = _rate_multipliers(unit, mdg, mck, k.shape[0], react)
    o = unit.solve(feed, activity=a, riser_outlet_T=rot, gasoline_cut=gcut, lco_cut=lcut,
                   feed_T=tf, k_ref=k * mult)["outputs"]
    y = jnp.stack([sum(o[key] for key in keys) for _, keys in FCC_LUMPS])
    if with_regenerator:
        y = jnp.concatenate([y, ((o["regenerator_T"] - FCC_REGEN_T_TARGET)
                                 / FCC_REGEN_T_SCALE)[None]])
    return y


def _rate_multipliers(unit, mdg, mck, n, reactions):
    """Multipliers on the rate constants of the reactions to dry gas and coke."""
    ones = jnp.ones(n)
    if reactions is None:  # pragma: no cover - every scheme lists its reactions
        return ones
    out = []
    for r in reactions:
        prod = getattr(r, "product", "")
        prods = prod if isinstance(prod, (tuple, list)) else (prod,)
        out.append(mdg if "dry_gas" in prods else mck if "coke" in prods else 1.0)
    return jnp.stack([jnp.asarray(v, float) for v in out])


def fcc_targets(base) -> np.ndarray:
    """A benchmark FCC batch's yields lumped as :data:`FCC_LUMPS` (``base``
    from :func:`~.calibration.unit_base`); an outlet the batch lacks is 0."""
    y = dict(zip(base.outlets, base.yields))
    o = {k: y.get(v, 0.0) for k, v in FCC_OUTLETS[base.unit].items()}
    # the FCC's product slate sums to 1, so the 0.1 % loss rides in gasoline
    return np.array([o["dry_gas"] + o["acid_gas"], o["lpg"], o["gasoline"] + o["loss"],
                     o["lco"], o["slurry"], o["coke"]])


def is_cracking_batch(base) -> bool:
    """Whether an FCC batch has the gas-oil product slate (LCO, slurry, coke)
    the VGO kinetics describe. ``Upf3``'s ``m10`` -- a light, sulfur-free feed
    with only gas, LPG and gasoline outlets -- is a recracking batch the
    model does not cover."""
    o = FCC_OUTLETS[base.unit]
    return all(o[k] in base.outlets for k in ("lco", "slurry", "coke"))


def fcc_outlet_yields(base, lumped) -> np.ndarray:
    """Lumped FCC yields back onto the benchmark outlets (``base.outlets``
    order). The dry-gas+acid-gas lump keeps the benchmark's split, and the
    loss stream keeps its yield, taken from the gasoline."""
    y = dict(zip(base.outlets, base.yields))
    o = FCC_OUTLETS[base.unit]
    dg = lumped[0]
    a, d = y.get(o["acid_gas"], 0.0), y.get(o["dry_gas"], 0.0)
    share = a / (a + d) if a + d > 0 else 0.0
    loss = y.get(o["loss"], 0.0)
    out = {o["acid_gas"]: dg * share, o["dry_gas"]: dg * (1 - share), o["lpg"]: lumped[1],
           o["gasoline"]: lumped[2] - loss, o["lco"]: lumped[3],
           o["slurry"]: lumped[4], o["coke"]: lumped[5], o["loss"]: loss}
    return np.array([out[s] for s in base.outlets])
