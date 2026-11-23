"""Jet and wax-oil (VGO) hydrotreating services (#536).

Both run the existing :class:`~difflow_refinery.hydrotreating.Hydrotreater`,
with its reactor, HP separator, recycle loop and stripper unchanged. This
module adds what each service needs around it.

**Jet / kerosene.**

* :func:`jet_params` -- a mild, low-pressure operating point: 30 bar,
  LHSV 3/h, 300 C inlet, about 100 Nm3/m3 H2/oil.
* :data:`JET_SPECS` -- the product specification set of Jet A-1 / Chinese
  No. 3 jet fuel.
* :func:`jet_properties` -- smoke point, flash point, aromatics and sulfur
  from the unit's product outputs, through the shared estimators of
  :mod:`difflow_refinery.properties`.

  The freeze point needs per-component n-paraffins, which the outputs do
  not carry; it is reported as NaN and named in ``missing``. Mercaptan
  sulfur is not tracked separately, so the total sulfur is reported as its
  upper bound.

**Wax oil (VGO).**

* :func:`vgo_params` -- 100 bar, LHSV 1/h, 380 C inlet, 500 Nm3/m3, with
  the hydrocracker's VGO pretreat kinetics
  (:data:`~difflow_refinery.hydrocracking.feed.VGO_PRETREAT_PARAMS`). Those
  kinetics already carry basic-nitrogen inhibition of HDS and the catalyst
  activity multiplier, which is the deactivation hook.
* :func:`heavy_ends_removal` -- Conradson carbon and Ni+V removal (HDCCR,
  HDM), first order in Arrhenius form at the WABT and LHSV. The reactor
  does not track CCR or metals, so this is a post-reactor correlation.
* :func:`fcc_feed_from_hydrotreater` -- the treated VGO as an
  :class:`~difflow_refinery.fcc.FCCFeed` (SG, Tb, Kw, MW, H, S, N, CCR,
  Ni+V), so the product feeds the FCC with no unit conversion left to the
  caller.

**ILLUSTRATIVE:** the HDCCR and HDM rate constants and activation energies.
They are of the form Ancheyta (2011, ch. on hydrotreating) uses, with
magnitudes typical of VGO service, and no published data set is reproduced.
The operating points are typical values. The kinetics are the unit's own,
and are marked illustrative where they are defined.
"""

from __future__ import annotations

import dataclasses
from typing import Mapping

import jax.numpy as jnp
from jax import Array

from difflow_refinery import correlations as corr
from difflow_refinery import properties as props
from difflow_refinery.hydrotreating.unit import HydrotreaterParams

R = 8.314

#: Jet fuel specifications, (property, op, limit): Jet A-1 (DEF STAN 91-091 /
#: ASTM D1655) and Chinese No. 3 jet (GB 6537) agree on these.
JET_SPECS = (("smoke_point_mm", ">=", 25.0),
             ("flash_point_C", ">=", 38.0),
             ("freeze_point_C", "<=", -47.0),
             ("aromatics_vol", "<=", 25.0),
             ("S_wppm", "<=", 3000.0),
             ("mercaptan_S_wppm", "<=", 30.0))


def jet_params(**changes) -> HydrotreaterParams:
    """A mild, low-pressure kerosene hydrotreating point (typical values)."""
    base = HydrotreaterParams(T_in=(573.15,), quench=None, bed_fractions=(1.0,), P=30e5,
                              lhsv=3.0, h2_oil=100.0)
    return dataclasses.replace(base, **changes)


def vgo_params(**changes) -> HydrotreaterParams:
    """A wax-oil hydrotreating point with the VGO pretreat kinetics."""
    from difflow_refinery.hydrocracking.feed import VGO_PRETREAT_PARAMS
    base = HydrotreaterParams(T_in=(653.15,), P=100e5, lhsv=1.0, h2_oil=500.0,
                              kinetics=VGO_PRETREAT_PARAMS)
    return dataclasses.replace(base, **changes)


def jet_properties(outputs: Mapping[str, Array]) -> dict[str, Array]:
    """Jet specification properties of a hydrotreated kerosene.

    Args:
        outputs: A :class:`~difflow_refinery.hydrotreating.HydrotreaterResult`'s
            ``outputs`` (``product.T10``, ``T50``, ``sg``, ``aromatics_vol``,
            ``S_wppm``).

    Returns:
        ``{smoke_point_mm, flash_point_C, aromatics_vol, S_wppm,
        mercaptan_S_wppm, freeze_point_C}``. The flash point takes the TBP
        10 % point for the D86 one, which is about 10 K low for a narrow
        kerosene, so the flash point is conservative. The mercaptan sulfur
        is the total sulfur, an upper bound. The freeze point is NaN, since
        it needs per-component n-paraffins.
    """
    sg = jnp.asarray(outputs["product.sg"], float)
    T50 = jnp.asarray(outputs["product.T50"], float)
    T10 = jnp.asarray(outputs["product.T10"], float)
    S = jnp.asarray(outputs["product.S_wppm"], float)
    return {"smoke_point_mm": props.smoke_point(T50, sg),
            "flash_point_C": props.flash_point(T10) - 273.15,
            "aromatics_vol": jnp.asarray(outputs["product.aromatics_vol"], float),
            "S_wppm": S,
            "mercaptan_S_wppm": S,
            "freeze_point_C": jnp.asarray(jnp.nan)}


def jet_margins(jet: Mapping[str, Array], specs=JET_SPECS) -> dict[str, Array]:
    """Signed margins (positive = on spec) of :func:`jet_properties`.

    A property that is NaN (the freeze point) gives a NaN margin. It is
    never counted as met.
    """
    out = {}
    for name, op, limit in specs:
        v = jnp.asarray(jet[name], float)
        out[name] = (v - limit) if op == ">=" else (limit - v)
    return out


@dataclasses.dataclass
class HeavyEndsParams:
    """HDCCR and HDM constants (ILLUSTRATIVE).

    Attributes:
        k_ccr, Ea_ccr: CCR removal at ``T_ref`` (1/h) and activation energy.
        k_metals, Ea_metals: Ni+V removal (HDM).
        T_ref: Reference temperature (K).
    """

    k_ccr: float = 0.35
    Ea_ccr: float = 80.0e3
    k_metals: float = 1.2
    Ea_metals: float = 100.0e3
    T_ref: float = 653.15


def heavy_ends_removal(ccr_wt, niv_wppm, wabt, lhsv,
                       p: HeavyEndsParams | None = None) -> dict[str, Array]:
    """Product CCR and Ni+V after VGO hydrotreating (first order, plug flow).

    ``x_out = x_in exp(-k(T) / LHSV)``, with ``k`` Arrhenius in the WABT.

    Args:
        ccr_wt: Feed CCR (wt%).
        niv_wppm: Feed Ni+V (wppm).
        wabt: Weighted average bed temperature (K), the unit's ``wabt``.
        lhsv: Liquid hourly space velocity (1/h).
        p: :class:`HeavyEndsParams`.

    Returns:
        ``{ccr_wt, niv_wppm, hdccr, hdm}``: product values and conversions.
    """
    p = HeavyEndsParams() if p is None else p
    T = jnp.asarray(wabt, float)
    lhsv = jnp.asarray(lhsv, float)
    k_c = p.k_ccr * jnp.exp(-p.Ea_ccr / R * (1.0 / T - 1.0 / p.T_ref))
    k_m = p.k_metals * jnp.exp(-p.Ea_metals / R * (1.0 / T - 1.0 / p.T_ref))
    fc, fm = jnp.exp(-k_c / lhsv), jnp.exp(-k_m / lhsv)
    return {"ccr_wt": jnp.asarray(ccr_wt, float) * fc,
            "niv_wppm": jnp.asarray(niv_wppm, float) * fm,
            "hdccr": 1.0 - fc, "hdm": 1.0 - fm}


def fcc_feed_from_hydrotreater(outputs: Mapping[str, Array], *, ccr_wt, niv_wppm):
    """The hydrotreated VGO as an :class:`~difflow_refinery.fcc.FCCFeed`.

    Args:
        outputs: The hydrotreater's ``outputs`` (``product.rate`` kg/s,
            ``sg``, ``T50``, ``H_wt``, ``S_wppm``, ``N_wppm``).
        ccr_wt, niv_wppm: Product CCR and Ni+V, e.g. from
            :func:`heavy_ends_removal`.

    Returns:
        An ``FCCFeed``. ``Tb`` is the product's TBP 50 % point, standing in
        for the mass-average boiling point. ``MW`` is from Riazi-Daubert
        (1987), and ``Kw`` from ``Tb`` and ``SG``.
    """
    from difflow_refinery.fcc import FCCFeed
    sg = jnp.asarray(outputs["product.sg"], float)
    Tb = jnp.asarray(outputs["product.T50"], float)
    mw, _, _ = corr.riazi_daubert_1987(Tb, sg)
    return FCCFeed(mass=jnp.asarray(outputs["product.rate"], float), MW=mw, SG=sg, Tb=Tb,
                   Kw=corr.watson_k(Tb, sg),
                   hydrogen=jnp.asarray(outputs["product.H_wt"], float) / 100.0,
                   sulfur=jnp.asarray(outputs["product.S_wppm"], float) * 1e-6,
                   nitrogen=jnp.asarray(outputs["product.N_wppm"], float) * 1e-6,
                   ccr=jnp.asarray(ccr_wt, float) / 100.0,
                   nickel_vanadium=jnp.asarray(niv_wppm, float) * 1e-6)
