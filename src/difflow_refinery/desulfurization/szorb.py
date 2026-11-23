"""FCC gasoline desulfurization by reactive adsorption, S Zorb type (#535).

FCC gasoline and hydrogen pass through a fluidized bed of Ni/ZnO sorbent.
Nickel hydrogenolyses the C-S bonds, the sulfur is captured as ZnS
(ZnO + H2S -> ZnS + H2O), and the sorbent circulates to a regenerator that
burns the ZnS back to ZnO and sends the sulfur out as SO2. Olefin
hydrogenation is the side reaction, and it costs octane. (Process: Babich &
Moulijn, *Fuel* 82 (2003) 607, section 4.4; the ConocoPhillips S Zorb
literature it cites.)

**Model** (lumped, pure JAX):

* **Sulfur removal.** Per sulfur class of #305
  (:data:`~difflow_refinery.composition.SULFUR_CLASSES`: sulfides incl.
  mercaptans, thiophenes, benzothiophenes, dibenzothiophenes, hindered
  DBTs), first order in the gas, plug flow through the bed:
  ``S_i,out = S_i,in exp(-k_i(T) theta / WHSV)``.
* **Sorbent activity.** ``theta = 1 - q / q_max`` is the free capacity at
  the reactor's sorbent loading ``q``. At steady state the circulating
  sorbent carries the captured sulfur:
  ``q = q_regen + S captured / circulation``. Removal depends on ``theta``
  and ``theta`` on removal, so ``q`` is the root of one scalar equation
  (Newton, differentiated implicitly).
* **Olefin saturation.** First order, with its own rate constant and
  activation energy, scaled by the H2/oil ratio relative to the base.
  ``RON`` and ``MON`` fall in proportion to the olefins saturated (vol%).
* **Hydrogen.** Per mole of S removed, by class (hydrogenolysis), plus one
  per mole of olefin saturated. One H2 per S leaves in the H2S that the ZnO
  turns to water, and the rest goes into the hydrocarbon product.
* **Balances.** S: feed = product + captured, exactly. Mass: feed + H2 =
  product + light ends + captured S + the H2 that leaves as water, exactly
  by construction.

**Every rate constant, activation energy, the sorbent capacity and loading,
the H2 stoichiometry per class, the octane-loss slopes and the light-ends
yield are ILLUSTRATIVE.** They are chosen to give the published *ranges*
for S Zorb: about 500 wppm feed sulfur to below 10 wppm, a RON loss of
about 1, and a few tenths of a wt% of H2. They are not fitted to any plant
data set, because none was at hand (#535 asks for one). Do not quote the
outputs as predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow_refinery.composition import SULFUR_CLASSES

R = 8.314
MW_S, MW_H2, MW_H2S = 32.06, 2.016, 34.08
#: Nm3 of H2 per kg (0 C, 1 atm).
NM3_PER_KG_H2 = 22.414 / MW_H2
#: An FCC gasoline sulfur class split, ILLUSTRATIVE (Babich & Moulijn 2003
#: report thiophenes and benzothiophene as the bulk of FCC naphtha sulfur,
#: mercaptans and sulfides in the light end).
FCC_GASOLINE_SULFUR_SPLIT = (0.25, 0.55, 0.20, 0.0, 0.0)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SZorbFeed:
    """FCC gasoline to the unit.

    Attributes:
        mass: Mass rate (kg/s).
        SG: Specific gravity.
        sulfur: Sulfur, mass fraction.
        sulfur_split: ``(5,)`` shares of the sulfur by
            :data:`~difflow_refinery.composition.SULFUR_CLASSES` (sum 1).
        olefins_vol, aromatics_vol: vol%.
        RON, MON: Octane numbers.
    """

    mass: Array
    SG: Array
    sulfur: Array
    sulfur_split: Array
    olefins_vol: Array
    aromatics_vol: Array
    RON: Array
    MON: Array

    @classmethod
    def from_properties(cls, mass, *, sg, sulfur_wppm, olefins_vol, aromatics_vol, ron, mon,
                        sulfur_split=FCC_GASOLINE_SULFUR_SPLIT) -> "SZorbFeed":
        """From the usual units (S in wppm)."""
        f = lambda v: jnp.asarray(v, dtype=float)  # noqa: E731
        return cls(f(mass), f(sg), f(sulfur_wppm) * 1e-6, f(sulfur_split), f(olefins_vol),
                   f(aromatics_vol), f(ron), f(mon))


@dataclass
class SZorbParams(ParamsMixin):
    """S Zorb parameters (all rate and sorbent constants ILLUSTRATIVE).

    Attributes:
        T: Reactor temperature (K); base 693.15 (420 C).
        T_ref: Temperature of the reference rate constants (K).
        whsv: Weight hourly space velocity (kg feed / kg sorbent in bed / h).
        h2_oil: H2/oil molar ratio; olefin saturation scales with it.
        h2_oil_ref: Its base.
        k_s: ``(5,)`` sulfur-class rate constants at ``T_ref`` (1/h, per unit
            WHSV), ordered as ``SULFUR_CLASSES``.
        Ea_s: Activation energy of sulfur removal (J/mol).
        k_olefin: Olefin-saturation rate constant at ``T_ref`` (1/h).
        Ea_olefin: Its activation energy (J/mol); higher than sulfur
            removal, so a hotter bed costs octane faster than it removes
            sulfur.
        q_max: Sorbent sulfur capacity (kg S / kg sorbent).
        q_regen: Sulfur loading of regenerated sorbent (kg S / kg sorbent).
        circulation: Sorbent circulation (kg sorbent / kg feed).
        h2_per_s: ``(5,)`` mol H2 per mol S removed, by class.
        olefin_mw: Mean olefin molar mass (g/mol), for H2 consumption.
        ron_per_olefin, mon_per_olefin: Octane loss per vol% olefins
            saturated.
        light_ends: Mass fraction of feed cracked to C4- per unit olefin
            saturation fraction.
    """

    T: float = 693.15
    T_ref: float = 693.15
    whsv: float = 4.0
    h2_oil: float = 0.3
    h2_oil_ref: float = 0.3
    k_s: tuple = (60.0, 30.0, 20.0, 10.0, 5.0)
    Ea_s: float = 60.0e3
    k_olefin: float = 0.35
    Ea_olefin: float = 90.0e3
    q_max: float = 0.10
    q_regen: float = 0.02
    circulation: float = 0.025
    h2_per_s: tuple = (1.5, 3.0, 3.0, 2.0, 2.0)
    olefin_mw: float = 84.0
    ron_per_olefin: float = 0.35
    mon_per_olefin: float = 0.15
    light_ends: float = 0.03


def _arr(k, Ea, T, Tr):
    return k * jnp.exp(-Ea / R * (1.0 / T - 1.0 / Tr))


class SZorbRangeWarning(UserWarning):
    """The sorbent is near saturation (activity below 5 %)."""


class SZorbUnit:
    """S Zorb reactive-adsorption desulfurizer.

    Args:
        params: :class:`SZorbParams`.

    Example:
        >>> feed = SZorbFeed.from_properties(50.0, sg=0.73, sulfur_wppm=500.0,
        ...                                  olefins_vol=30.0, aromatics_vol=22.0,
        ...                                  ron=92.0, mon=80.0)
        >>> out = SZorbUnit().solve(feed)
        >>> float(out["product.S_wppm"]) < 10.0
        True
    """

    def __init__(self, params: SZorbParams | None = None):
        self.params = SZorbParams() if params is None else params

    def _check(self, feed: SZorbFeed, p: SZorbParams):
        if len(p.k_s) != len(SULFUR_CLASSES) or len(p.h2_per_s) != len(SULFUR_CLASSES):
            raise ValueError(f"k_s and h2_per_s need {len(SULFUR_CLASSES)} entries")
        split = feed.sulfur_split
        try:
            split = np.asarray(split, float)
        except (TypeError, jax.errors.TracerArrayConversionError):
            return
        if split.shape != (len(SULFUR_CLASSES),) or abs(split.sum() - 1.0) > 1e-9:
            raise ValueError("sulfur_split must have one share per sulfur class, summing to 1")
        for name in ("mass", "SG"):
            try:
                if float(getattr(feed, name)) <= 0.0:
                    raise ValueError(f"feed.{name} must be positive")
            except (TypeError, jax.errors.ConcretizationTypeError):
                pass

    def solve(self, feed: SZorbFeed, **overrides) -> dict:
        """Solve the reactor and sorbent loop.

        Args:
            feed: :class:`SZorbFeed`.
            **overrides: Numeric :class:`SZorbParams` fields to replace
                (traceable), e.g. ``T=jnp.asarray(700.0)``.

        Returns:
            A flat dict: ``product.S_wppm``, ``product.RON``,
            ``product.MON``, ``product.olefins_vol``,
            ``product.aromatics_vol``, ``product.yield``,
            ``light_ends.yield``, ``sulfur_captured.yield`` (S),
            ``acid_gas.yield`` (the captured S as H2S), ``h2.consumption_wt``
            (% of feed), ``h2.consumption_nm3_m3``, ``hds.conversion``,
            ``olefin.saturation``, ``ron.loss``, ``sorbent.loading``,
            ``sorbent.activity``, ``residual`` and ``balance.mass`` /
            ``.S`` / ``.H2``.
        """
        p = replace(self.params, **overrides) if overrides else self.params
        self._check(feed, p)
        g = lambda k: jnp.asarray(getattr(p, k), dtype=float)  # noqa: E731
        T, whsv = g("T"), g("whsv")
        k_s = _arr(jnp.asarray(p.k_s, float), g("Ea_s"), T, g("T_ref"))
        k_o = _arr(g("k_olefin"), g("Ea_olefin"), T, g("T_ref")) * g("h2_oil") / g("h2_oil_ref")
        S_in = jnp.asarray(feed.sulfur, float) * jnp.asarray(feed.sulfur_split, float)  # (5,)
        q_max, q_reg, circ = g("q_max"), g("q_regen"), g("circulation")

        def activity(q):
            # free capacity, smoothly floored: never negative under a trial q
            return jax.nn.softplus(50.0 * (1.0 - q / q_max)) / 50.0

        def captured(q):
            return jnp.sum(S_in * (1.0 - jnp.exp(-k_s * activity(q) / whsv)))

        def resid(q, _):
            return q - q_reg - captured(q) / circ

        import optimistix as optx
        q0 = q_reg + 0.5 * jnp.sum(S_in) / circ
        sol = optx.root_find(resid, optx.Newton(rtol=1e-12, atol=1e-14), q0, max_steps=50,
                             throw=False)
        q = sol.value
        theta = activity(q)
        S_out_cls = S_in * jnp.exp(-k_s * theta / whsv)
        S_out, S_cap = jnp.sum(S_out_cls), jnp.sum(S_in - S_out_cls)

        sat = 1.0 - jnp.exp(-k_o / whsv)                       # olefin fraction saturated
        d_ole = jnp.asarray(feed.olefins_vol, float) * sat      # vol%
        # olefins assumed at the gasoline's density, so vol% ~ wt%
        mol_ole = d_ole / 100.0 / (g("olefin_mw") * 1e-3)       # mol per kg feed
        mol_s = (S_in - S_out_cls) / (MW_S * 1e-3)              # (5,) mol per kg feed
        h2_mol = jnp.sum(mol_s * jnp.asarray(p.h2_per_s, float)) + mol_ole
        h2_to_water = jnp.sum(mol_s) * MW_H2 * 1e-3             # kg per kg feed
        h2_total = h2_mol * MW_H2 * 1e-3
        light = g("light_ends") * sat
        product = 1.0 + h2_total - light - S_cap - h2_to_water

        out = {
            "product.S_wppm": 1e6 * S_out / product,
            "product.RON": jnp.asarray(feed.RON, float) - g("ron_per_olefin") * d_ole,
            "product.MON": jnp.asarray(feed.MON, float) - g("mon_per_olefin") * d_ole,
            "product.olefins_vol": jnp.asarray(feed.olefins_vol, float) - d_ole,
            "product.aromatics_vol": jnp.asarray(feed.aromatics_vol, float),
            "product.yield": product,
            "light_ends.yield": light,
            "sulfur_captured.yield": S_cap,
            "acid_gas.yield": S_cap * MW_H2S / MW_S,
            "h2.consumption_wt": 100.0 * h2_total,
            "h2.consumption_nm3_m3": h2_total * NM3_PER_KG_H2 * 1000.0 * jnp.asarray(feed.SG, float),
            "hds.conversion": S_cap / jnp.sum(S_in),
            "olefin.saturation": sat,
            "ron.loss": g("ron_per_olefin") * d_ole,
            "sorbent.loading": q,
            "sorbent.activity": theta,
            "residual": jnp.abs(resid(q, None)),
            "balance.mass": (1.0 + h2_total) - (product + light + S_cap + h2_to_water),
            "balance.S": (S_out + S_cap - jnp.sum(S_in)) / jnp.sum(S_in),
            "balance.H2": h2_total - (h2_to_water + (h2_total - h2_to_water)),
        }
        out["product.rate"] = product * jnp.asarray(feed.mass, float)
        try:
            if float(theta) < 0.05:
                import warnings
                warnings.warn(f"sorbent activity {float(theta):.3f}: the sorbent is near "
                              "saturation; raise circulation", SZorbRangeWarning)
        except (TypeError, jax.errors.ConcretizationTypeError):
            pass
        return out
