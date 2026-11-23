"""Delayed coking unit (#534): correlation yields, feed pooling, product properties.

The yields are the delayed-coking correlations of Gary, Handwerk & Kaiser,
*Petroleum Refining: Technology and Economics*, 5th ed. (2007), ch. 5. They
are functions of the feed's Conradson carbon residue (CCR, wt%), and the
volume yields are functions of its API gravity:

* coke, wt%     = 1.6 x CCR
* gas (C4-), wt% = 7.8 + 0.144 x CCR
* gasoline, wt% = 11.29 + 0.343 x CCR
* gas oil, wt%  = 100 - coke - gas - gasoline
* gasoline, vol% = 186.5 / (131.5 + API) x gasoline wt%
* gas oil, vol%  = 155.5 / (131.5 + API) x gas oil wt%

The constants are transcribed as the book's table gives them. They are
**not** re-checked against the book in this repository (no copy at hand),
so treat them as UNVERIFIED until checked. Two independent pieces of data
are consistent with them: the delta-base coke vector of the Du et al.
(2025) Case 2 cokers (+0.0156 per wt% CCR, against 1.6/100), and the coke
trend in Castiglioni (1983).

The volume formulas fix the products' gravities. A product with mass yield
``w`` and volume yield ``v`` has ``SG = SG_feed * w / v``. So the gasoline is
at ``SG_feed (131.5 + API) / 186.5`` and the gas oil at
``SG_feed (131.5 + API) / 155.5``.

Everything else is ILLUSTRATIVE (typical values, not fitted to any one
source):

* the split of gas into dry gas and LPG;
* the split of gas oil into coker diesel (light coker gas oil) and heavy
  coker gas oil, and the gravity spread between them, which keeps the gas
  oil's volume;
* the sulfur and nitrogen distributions among the products;
* the olefin contents of naphtha and diesel;
* the operating-lever sensitivities (drum pressure, coil outlet temperature,
  recycle ratio).

All feed nickel and vanadium report to the coke. Sulfur in the gas leaves
as H2S, and nitrogen in the gas as NH3. Both are carved out of the C4-
total, so the overall mass balance closes exactly.

A coker takes several feeds: atmospheric residue, vacuum residue,
deasphalter pitch, FCC slurry. :meth:`CokerFeed.blend` pools them first,
SG by volume and the rest by mass, as the Du et al. coker mixer does
(their Fig. 5), and the yield model runs on the pooled feed.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Sequence

import jax
import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin

MW_S, MW_H2S, MW_N, MW_NH3 = 32.06, 34.08, 14.007, 17.031

#: Product names, in the order :meth:`DelayedCoker.yields` reports them.
PRODUCTS = ("dry_gas", "lpg", "h2s", "nh3", "naphtha", "diesel", "gas_oil", "coke")


def api_from_sg(sg):
    return 141.5 / sg - 131.5


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CokerFeed:
    """A coker feed: one residue stream or a pooled blend.

    Attributes:
        mass: Mass rate (kg/s, or any consistent unit).
        SG: Specific gravity (60/60 F).
        sulfur: Sulfur, mass fraction.
        nitrogen: Nitrogen, mass fraction.
        ccr: Conradson carbon residue, mass fraction.
        nickel_vanadium: Ni + V, mass fraction.
    """

    mass: Array
    SG: Array
    sulfur: Array
    nitrogen: Array = 0.0
    ccr: Array = 0.0
    nickel_vanadium: Array = 0.0

    @classmethod
    def from_properties(cls, mass, *, sg, sulfur_wt, ccr_wt, nitrogen_wppm=0.0,
                        nickel_vanadium_wppm=0.0) -> "CokerFeed":
        """From the usual units: wt% sulfur and CCR, wppm N and Ni+V."""
        f = lambda v: jnp.asarray(v, dtype=float)  # noqa: E731
        return cls(f(mass), f(sg), f(sulfur_wt) / 100.0, f(nitrogen_wppm) * 1e-6,
                   f(ccr_wt) / 100.0, f(nickel_vanadium_wppm) * 1e-6)

    @classmethod
    def blend(cls, feeds: Sequence["CokerFeed"]) -> "CokerFeed":
        """Pool feeds: SG by volume, every other property by mass."""
        if not feeds:
            raise ValueError("blend needs at least one feed")
        m = jnp.stack([jnp.asarray(f.mass, float) for f in feeds])
        total = jnp.sum(m)
        vol = jnp.sum(m / jnp.stack([jnp.asarray(f.SG, float) for f in feeds]))
        avg = lambda k: jnp.sum(m * jnp.stack([jnp.asarray(getattr(f, k), float)  # noqa: E731
                                               for f in feeds])) / total
        return cls(total, total / vol, avg("sulfur"), avg("nitrogen"), avg("ccr"),
                   avg("nickel_vanadium"))


@dataclass
class CokerParams(ParamsMixin):
    """Delayed coker parameters.

    Attributes:
        coke_per_ccr: Coke wt% per wt% CCR (Gary et al., 1.6).
        gas_intercept, gas_per_ccr: Gas (C4-) wt% = a + b CCR (7.8, 0.144).
        gasoline_intercept, gasoline_per_ccr: Gasoline wt% = a + b CCR
            (11.29, 0.343).
        gasoline_vol_k: Gasoline vol% = k / (131.5 + API) x wt% (186.5).
        gas_oil_vol_k: Gas oil vol% = k / (131.5 + API) x wt% (155.5).
        dry_gas_share: ILLUSTRATIVE. Dry gas (C2-) share of the hydrocarbon
            gas (0.65).
        diesel_share: ILLUSTRATIVE. Coker diesel (light coker gas oil)
            share of the gas oil (0.5).
        diesel_sg_offset: ILLUSTRATIVE. Diesel SG = gas oil SG - offset; the
            heavy gas oil takes the volume left (0.05).
        sulfur_split: ILLUSTRATIVE. Shares of feed S to (gas as H2S,
            naphtha, diesel, gas oil, coke); sum 1 (0.30, 0.05, 0.12, 0.23,
            0.30).
        nitrogen_split: ILLUSTRATIVE. Shares of feed N to (gas as NH3,
            naphtha, diesel, gas oil, coke); sum 1 (0.01, 0.02, 0.07, 0.15,
            0.75).
        naphtha_olefins_vol, diesel_olefins_vol: ILLUSTRATIVE. Olefins, vol%
            (40, 25).
        drum_P, drum_P_ref: Drum pressure and its base, bar gauge (1.03).
        coil_outlet_T, coil_outlet_T_ref: Heater outlet temperature and its
            base, K (768.15, 495 C).
        recycle, recycle_ref: Recycle ratio (recycle / fresh feed) and its
            base (0.05).
        coke_per_bar, coke_per_K, coke_per_recycle: ILLUSTRATIVE. Relative
            change of the coke yield per bar of drum pressure (+0.04), per K
            of coil outlet (-0.002) and per unit of recycle ratio (+0.6).
            Higher pressure or recycle makes more coke and gas; a hotter coil
            less coke (Gary et al., qualitatively).
        gas_per_bar, gas_per_K, gas_per_recycle: ILLUSTRATIVE. The same for
            the gas yield (+0.03, +0.001, +0.4).
    """

    coke_per_ccr: float = 1.6
    gas_intercept: float = 7.8
    gas_per_ccr: float = 0.144
    gasoline_intercept: float = 11.29
    gasoline_per_ccr: float = 0.343
    gasoline_vol_k: float = 186.5
    gas_oil_vol_k: float = 155.5
    dry_gas_share: float = 0.65
    diesel_share: float = 0.5
    diesel_sg_offset: float = 0.05
    sulfur_split: tuple = (0.30, 0.05, 0.12, 0.23, 0.30)
    nitrogen_split: tuple = (0.01, 0.02, 0.07, 0.15, 0.75)
    naphtha_olefins_vol: float = 40.0
    diesel_olefins_vol: float = 25.0
    drum_P: float = 1.03
    drum_P_ref: float = 1.03
    coil_outlet_T: float = 768.15
    coil_outlet_T_ref: float = 768.15
    recycle: float = 0.05
    recycle_ref: float = 0.05
    coke_per_bar: float = 0.04
    coke_per_K: float = -0.002
    coke_per_recycle: float = 0.6
    gas_per_bar: float = 0.03
    gas_per_K: float = 0.001
    gas_per_recycle: float = 0.4


#: Range of feed CCR (wt%) the correlation is meant for; outside it the
#: unit warns, and never clips.
CCR_RANGE = (4.0, 30.0)


class DelayedCoker:
    """A delayed coking unit.

    Args:
        params: :class:`CokerParams`.

    Example:
        >>> feed = CokerFeed.from_properties(10.0, sg=1.01, sulfur_wt=3.0, ccr_wt=19.5)
        >>> out = DelayedCoker(CokerParams()).solve(feed)
        >>> round(float(out["yield.coke"]), 3)
        0.312
    """

    def __init__(self, params: CokerParams | None = None):
        self.params = CokerParams() if params is None else params

    def _check(self, feed: CokerFeed):
        for name in ("mass", "SG"):
            v = getattr(feed, name)
            try:
                if float(v) <= 0.0:
                    raise ValueError(f"feed.{name} must be positive, got {float(v)}")
            except (TypeError, jax.errors.ConcretizationTypeError):
                pass
        for name in ("sulfur", "nitrogen", "ccr", "nickel_vanadium"):
            try:
                if float(getattr(feed, name)) < 0.0:
                    raise ValueError(f"feed.{name} must be nonnegative")
            except (TypeError, jax.errors.ConcretizationTypeError):
                pass
        p = self.params
        for name, split in (("sulfur_split", p.sulfur_split), ("nitrogen_split", p.nitrogen_split)):
            if len(split) != 5 or abs(sum(split) - 1.0) > 1e-9:
                raise ValueError(f"{name} must have 5 shares summing to 1")
        try:
            ccr = 100.0 * float(feed.ccr)
        except (TypeError, jax.errors.ConcretizationTypeError):
            return
        if not CCR_RANGE[0] <= ccr <= CCR_RANGE[1]:
            import warnings
            warnings.warn(f"feed CCR {ccr:.1f} wt% is outside the correlation's range "
                          f"{CCR_RANGE}; yields are extrapolated", CokerRangeWarning)

    def solve(self, feed: CokerFeed | Sequence[CokerFeed], **overrides) -> dict:
        """Yields, product rates and properties, and balances.

        Args:
            feed: One :class:`CokerFeed`, or several (pooled first).
            **overrides: Numeric :class:`CokerParams` fields to replace
                (traceable), e.g. ``drum_P=jnp.asarray(2.0)``.

        Returns:
            A flat dict:

            * ``yield.<product>``: mass fraction of feed, for each of
              :data:`PRODUCTS`;
            * ``<product>.rate``;
            * ``naphtha.SG``, ``diesel.SG``, ``gas_oil.SG``;
            * ``<liquid>.S_wt``, ``<liquid>.N_wppm``, ``coke.S_wt``,
              ``coke.NiV_wppm``;
            * ``naphtha.olefins_vol``, ``diesel.olefins_vol``;
            * ``naphtha.volume_yield``, ``gas_oil.volume_yield``;
            * ``feed.*`` of the pooled feed;
            * ``balance.mass`` / ``.S`` / ``.N`` / ``.NiV`` (relative closure).
        """
        if not isinstance(feed, CokerFeed):
            feed = CokerFeed.blend(list(feed))
        self._check(feed)
        p = replace(self.params, **overrides) if overrides else self.params
        g = lambda k: jnp.asarray(getattr(p, k), dtype=float)  # noqa: E731
        ccr = 100.0 * jnp.asarray(feed.ccr, float)
        sg = jnp.asarray(feed.SG, float)
        api = api_from_sg(sg)

        dP = g("drum_P") - g("drum_P_ref")
        dT = g("coil_outlet_T") - g("coil_outlet_T_ref")
        dR = g("recycle") - g("recycle_ref")
        coke = g("coke_per_ccr") * ccr * (1.0 + g("coke_per_bar") * dP + g("coke_per_K") * dT
                                          + g("coke_per_recycle") * dR) / 100.0
        gas = (g("gas_intercept") + g("gas_per_ccr") * ccr) * (
            1.0 + g("gas_per_bar") * dP + g("gas_per_K") * dT + g("gas_per_recycle") * dR) / 100.0
        naphtha = (g("gasoline_intercept") + g("gasoline_per_ccr") * ccr) / 100.0
        gas_oil = 1.0 - coke - gas - naphtha

        S = jnp.asarray(feed.sulfur, float)
        N = jnp.asarray(feed.nitrogen, float)
        ss = jnp.asarray(p.sulfur_split, float)
        ns = jnp.asarray(p.nitrogen_split, float)
        h2s = ss[0] * S * MW_H2S / MW_S
        nh3 = ns[0] * N * MW_NH3 / MW_N
        hc_gas = gas - h2s - nh3
        dry = g("dry_gas_share") * hc_gas
        lpg = hc_gas - dry
        diesel = g("diesel_share") * gas_oil
        hgo = gas_oil - diesel
        y = dict(zip(PRODUCTS, (dry, lpg, h2s, nh3, naphtha, diesel, hgo, coke)))

        sg_naphtha = sg * (131.5 + api) / g("gasoline_vol_k")
        sg_go = sg * (131.5 + api) / g("gas_oil_vol_k")
        sg_diesel = sg_go - g("diesel_sg_offset")
        # the heavy gas oil takes the volume the gas oil has left
        sg_hgo = hgo / (gas_oil / sg_go - diesel / sg_diesel)

        F = jnp.asarray(feed.mass, float)
        out = {f"yield.{k}": v for k, v in y.items()}
        out.update({f"{k}.rate": v * F for k, v in y.items()})
        out.update({"naphtha.SG": sg_naphtha, "diesel.SG": sg_diesel, "gas_oil.SG": sg_hgo,
                    "naphtha.volume_yield": naphtha * sg / sg_naphtha,
                    "gas_oil.volume_yield": gas_oil * sg / sg_go,
                    "naphtha.olefins_vol": g("naphtha_olefins_vol"),
                    "diesel.olefins_vol": g("diesel_olefins_vol")})
        for i, k in enumerate(("naphtha", "diesel", "gas_oil", "coke")):
            out[f"{k}.S_wt"] = 100.0 * ss[i + 1] * S / y[k]
            out[f"{k}.N_wppm"] = 1e6 * ns[i + 1] * N / y[k]
        out["coke.NiV_wppm"] = 1e6 * jnp.asarray(feed.nickel_vanadium, float) / coke
        for f in fields(feed):
            out[f"feed.{f.name}"] = jnp.asarray(getattr(feed, f.name), float)
        # balances: products against feed, relative
        tot = sum(y.values())
        s_out = (h2s * MW_S / MW_H2S + sum(ss[i + 1] * S for i in range(4)))
        n_out = (nh3 * MW_N / MW_NH3 + sum(ns[i + 1] * N for i in range(4)))
        out["balance.mass"] = tot - 1.0
        out["balance.S"] = jnp.where(S > 0, (s_out - S) / jnp.where(S > 0, S, 1.0), 0.0)
        out["balance.N"] = jnp.where(N > 0, (n_out - N) / jnp.where(N > 0, N, 1.0), 0.0)
        out["balance.NiV"] = jnp.asarray(0.0)      # all of it is in the coke by construction
        return out

    def yields(self, feed, **overrides) -> Array:
        """``(len(PRODUCTS),)`` mass yields, in :data:`PRODUCTS` order."""
        o = self.solve(feed, **overrides)
        return jnp.stack([o[f"yield.{k}"] for k in PRODUCTS])


class CokerRangeWarning(UserWarning):
    """Feed CCR outside the correlation's range (extrapolated, not clipped)."""
