"""Crude assays and their characterization into pseudocomponents.

An :class:`Assay` is what a lab reports: a TBP curve (cumulative wt%
distilled against temperature) and bulk properties. :func:`characterize`
cuts it into pseudocomponents on a fixed temperature grid and gives each
one a boiling point, gravity, molecular weight, critical properties and a
sulfur / nitrogen / CCR / metals content, plus a *residue lump* for
everything past the last cut.

Everything is a ``jax.numpy`` function of the assay's numbers, so a
derivative with respect to one TBP point -- or the bulk gravity, or the
bulk sulfur -- reaches every pseudocomponent property and from there every
column result.

The heavy end, which is what a vacuum column is about:

- A TBP curve stops at about 565 C (the end of D5236); a VDU needs
  pseudocomponents up to 750-800 C. The curve is extended on a *probability
  scale* -- ``z = Phi^-1(x)`` is close to linear in T for crude oils (a
  log-normal-like distribution) -- by a least-squares line through the last
  three points. That is smooth and monotone where a polynomial or a spline
  in ``x`` is neither. Inside the data the curve is a monotone cubic in
  ``z``, C1 at every point, so derivatives with respect to a TBP point are
  single-valued even where a cut boundary sits on it.
- Twu's correlations (see :mod:`~difflow_refinery.correlations`) are used
  for every cut and hold up to about 820 C TBP; past that the n-alkane
  reference has no root. The residue lump is therefore *not* run through
  them: its boiling point, gravity and molecular weight are set directly
  (``residue_Tb``, ``residue_sg``, ``residue_mw``), as Riazi recommends for
  the non-distillable end.
- Gravity follows a constant Watson K fitted so the whole crude matches its
  bulk SG -- the classical assumption, and exact for the bulk number.
- Contaminants follow default boiling-point shapes (logistic in T, rising
  with boiling point: sulfur gradually, nitrogen later, CCR and metals
  almost only in the residue), scaled so each matches the bulk assay value.
  Any of them can be replaced by a measured curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr, ndtri

from difflow_refinery import correlations as corr

C_TO_K = 273.15


@dataclass
class Assay:
    """A crude assay: TBP curve plus bulk properties.

    Attributes:
        name: Label for the crude.
        tbp_C: Temperatures of the TBP curve (C), increasing.
        tbp_wt: Cumulative wt% distilled at ``tbp_C``, increasing, in (0, 100).
        sg: Bulk specific gravity (60/60 F). Give this or ``api``.
        api: Bulk API gravity, if ``sg`` is not given.
        sulfur_wt: Bulk sulfur (wt%).
        nitrogen_wppm: Bulk total nitrogen (wppm).
        ccr_wt: Bulk Conradson carbon residue (wt%).
        nickel_vanadium_wppm: Bulk Ni + V (wppm).
        asphaltenes_wt: Bulk C7 asphaltenes (wt%).
        sulfur_curve: Optional measured sulfur curve ``(T_C, wt%)`` pairs for
            cut midpoints; replaces the default shape (no rescaling).
        nitrogen_curve: As ``sulfur_curve``, for nitrogen in wppm.
        ccr_curve: As ``sulfur_curve``, for CCR in wt%.
    """

    name: str
    tbp_C: Sequence[float]
    tbp_wt: Sequence[float]
    sg: Optional[float] = None
    api: Optional[float] = None
    sulfur_wt: float = 1.0
    nitrogen_wppm: float = 1000.0
    ccr_wt: float = 4.0
    nickel_vanadium_wppm: float = 30.0
    asphaltenes_wt: float = 2.0
    sulfur_curve: Optional[Sequence[Sequence[float]]] = None
    nitrogen_curve: Optional[Sequence[Sequence[float]]] = None
    ccr_curve: Optional[Sequence[Sequence[float]]] = None

    def bulk_sg(self):
        if self.sg is not None:
            return jnp.asarray(self.sg, dtype=float)
        if self.api is None:
            raise ValueError(f"assay {self.name!r}: give sg or api")
        return corr.sg_from_api(jnp.asarray(self.api, dtype=float))

    def with_tbp_point(self, index: int, temperature_C):
        """A copy with one TBP temperature replaced (for sensitivities)."""
        T = jnp.asarray(self.tbp_C, dtype=float).at[index].set(temperature_C)
        return _replace(self, tbp_C=T)


jax.tree_util.register_dataclass(
    Assay,
    data_fields=["tbp_C", "tbp_wt", "sg", "api", "sulfur_wt", "nitrogen_wppm",
                 "ccr_wt", "nickel_vanadium_wppm", "asphaltenes_wt",
                 "sulfur_curve", "nitrogen_curve", "ccr_curve"],
    meta_fields=["name"],
)


def _replace(obj, **kw):
    import dataclasses
    return dataclasses.replace(obj, **kw)


def light_crude() -> Assay:
    """A representative light sweet crude (about 38 API, 15% vacuum residue).

    Synthetic numbers of the right shape, not a named crude's assay.
    """
    return Assay(
        name="light sweet (synthetic)",
        tbp_C=[36, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 565],
        tbp_wt=[2, 10, 19, 28, 37, 46, 55, 63, 70, 77, 83, 84.5],
        api=38.0,
        sulfur_wt=0.6, nitrogen_wppm=700.0, ccr_wt=2.5,
        nickel_vanadium_wppm=8.0, asphaltenes_wt=0.8,
    )


def heavy_crude() -> Assay:
    """A representative heavy sour crude (about 20 API, 40% vacuum residue).

    Synthetic numbers of the right shape, not a named crude's assay.
    """
    return Assay(
        name="heavy sour (synthetic)",
        tbp_C=[36, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 565],
        tbp_wt=[1, 4, 8, 13, 18, 24, 31, 38, 45, 52, 58, 60],
        api=20.0,
        sulfur_wt=3.5, nitrogen_wppm=3500.0, ccr_wt=12.0,
        nickel_vanadium_wppm=200.0, asphaltenes_wt=9.0,
    )


# ---------------------------------------------------------------------------
# The TBP curve, extended
# ---------------------------------------------------------------------------


def _tbp_spline(assay: Assay, n_tail: int = 3):
    """Knots, values and slopes of the TBP curve on the probability scale.

    ``z = Phi^-1(x)`` against T, as a monotone cubic Hermite (Fritsch-Butland
    weighted-harmonic slopes) inside the data and straight lines outside,
    whose slopes are least-squares fits to the first / last ``n_tail``
    points and are also the Hermite end slopes -- so the curve is C1
    everywhere, including at every data point.

    C1 matters here: the default cut grid puts a cut boundary *on* every
    assay point (both are multiples of 50 C), and a piecewise-linear curve
    has a kink exactly there -- a derivative with respect to that TBP point
    then has two values, and central differences average them.
    """
    Td = jnp.asarray(assay.tbp_C, dtype=float)
    zd = ndtri(jnp.asarray(assay.tbp_wt, dtype=float) / 100.0)

    def ls_slope(t, z):
        tm, zm = jnp.mean(t), jnp.mean(z)
        return jnp.sum((t - tm) * (z - zm)) / jnp.sum((t - tm) ** 2)

    h = jnp.diff(Td)
    delta = jnp.diff(zd) / h
    w1 = 2.0 * h[1:] + h[:-1]
    w2 = h[1:] + 2.0 * h[:-1]
    d_int = (w1 + w2) / (w1 / delta[:-1] + w2 / delta[1:])
    # end slopes, capped at Fritsch-Carlson's 3x so the end intervals stay monotone
    s_lo = jnp.minimum(ls_slope(Td[:n_tail], zd[:n_tail]), 3.0 * delta[0])
    s_hi = jnp.minimum(ls_slope(Td[-n_tail:], zd[-n_tail:]), 3.0 * delta[-1])
    d = jnp.concatenate([s_lo[None], d_int, s_hi[None]])
    return Td, zd, d


def _hermite(T, Td, zd, d):
    n = Td.shape[0]
    k = jnp.clip(jnp.searchsorted(Td, T, side="right") - 1, 0, n - 2)
    h = Td[k + 1] - Td[k]
    t = (T - Td[k]) / h
    t2, t3 = t * t, t * t * t
    inside = ((2 * t3 - 3 * t2 + 1) * zd[k] + (t3 - 2 * t2 + t) * h * d[k]
              + (-2 * t3 + 3 * t2) * zd[k + 1] + (t3 - t2) * h * d[k + 1])
    up = zd[-1] + d[-1] * (T - Td[-1])
    dn = zd[0] + d[0] * (T - Td[0])
    return jnp.where(T > Td[-1], up, jnp.where(T < Td[0], dn, inside))


def tbp_fraction(assay: Assay, T_C, n_tail: int = 3):
    """Cumulative mass fraction distilled at ``T_C`` (C), any temperature.

    See :func:`_tbp_spline`: a C1 monotone curve in ``Phi^-1(x)`` through the
    data, extended by straight lines on the probability scale.
    """
    Td, zd, d = _tbp_spline(assay, n_tail)
    return ndtr(_hermite(jnp.asarray(T_C, dtype=float), Td, zd, d))


def tbp_temperature(assay: Assay, fraction, n_tail: int = 3):
    """Inverse of :func:`tbp_fraction`: TBP temperature (C) at a mass fraction.

    Newton on the Hermite curve from the piecewise-linear inverse (unrolled,
    so it differentiates like the forward curve).
    """
    Td, zd, d = _tbp_spline(assay, n_tail)
    z = ndtri(jnp.asarray(fraction, dtype=float))
    T = jnp.where(z > zd[-1], Td[-1] + (z - zd[-1]) / d[-1],
                  jnp.where(z < zd[0], Td[0] + (z - zd[0]) / d[0],
                            jnp.interp(z, zd, Td)))
    for _ in range(8):
        val, slope = jax.jvp(lambda t: _hermite(t, Td, zd, d), (T,), (jnp.ones_like(T),))
        T = T - (val - z) / slope
    return T


# ---------------------------------------------------------------------------
# Pseudocomponents
# ---------------------------------------------------------------------------


@jax.tree_util.register_dataclass
@dataclass
class PseudoComponents:
    """Property table of a pseudocomponent set (a JAX pytree).

    Arrays are per component, ordered by boiling point; the last entry is
    the residue lump. ``names`` is static metadata.

    Attributes:
        Tb: Normal (atmospheric equivalent) boiling point (K).
        SG: Specific gravity (60/60 F).
        MW: Molecular weight (g/mol).
        Kw: Watson characterization factor.
        Tc: Critical temperature (K).
        Pc: Critical pressure (Pa).
        omega: Acentric factor.
        T_lo: Lower TBP bound of the cut (K); used to rebuild product curves.
        T_hi: Upper TBP bound of the cut (K).
        sulfur: Sulfur mass fraction.
        nitrogen: Nitrogen mass fraction.
        ccr: Conradson carbon mass fraction.
        nickel_vanadium: Ni + V mass fraction.
        asphaltenes: C7 asphaltene mass fraction.
        names: Component names (static).
    """

    Tb: jax.Array
    SG: jax.Array
    MW: jax.Array
    Kw: jax.Array
    Tc: jax.Array
    Pc: jax.Array
    omega: jax.Array
    T_lo: jax.Array
    T_hi: jax.Array
    sulfur: jax.Array
    nitrogen: jax.Array
    ccr: jax.Array
    nickel_vanadium: jax.Array
    asphaltenes: jax.Array
    names: tuple = field(metadata=dict(static=True), default=())

    @property
    def n(self) -> int:
        return len(self.names)


@jax.tree_util.register_dataclass
@dataclass
class Characterization:
    """Result of :func:`characterize`.

    Attributes:
        components: The pseudocomponent property table.
        yields: Mass fraction of the whole crude in each pseudocomponent.
        light_ends: Mass fraction of the crude boiling below the first cut
            (not represented by any pseudocomponent here).
        Kw: The crude's Watson K (fitted to the bulk gravity).
    """

    components: PseudoComponents
    yields: jax.Array
    light_ends: jax.Array
    Kw: jax.Array


def vacuum_cuts(start_C: float = 300.0, stop_C: float = 800.0,
                step_C: float = 25.0) -> np.ndarray:
    """Default cut grid for a vacuum column feed: 300-800 C in 25 C steps."""
    return np.arange(start_C, stop_C + 0.5 * step_C, step_C)


def _logistic(T_C, center, width):
    return 1.0 / (1.0 + jnp.exp(-(T_C - center) / width))


# default contaminant shapes: (center C, width C) of a logistic in TBP
_SHAPES = {
    "sulfur": (400.0, 80.0),
    "nitrogen": (500.0, 70.0),
    "ccr": (650.0, 50.0),
    "nickel_vanadium": (760.0, 30.0),
    "asphaltenes": (740.0, 35.0),
}


def characterize(
    assay: Assay,
    cuts_C=None,
    residue_Tb_C: float = 950.0,
    residue_sg=None,
    residue_mw: float = 1500.0,
    fine_step_C: float = 10.0,
    fine_start_C: float = 0.0,
) -> Characterization:
    """Cut an assay into pseudocomponents plus a residue lump.

    Args:
        assay: The crude assay.
        cuts_C: Cut boundaries (C); ``len(cuts_C) - 1`` pseudocomponents.
            Default :func:`vacuum_cuts`.
        residue_Tb_C: Equivalent boiling point given to the residue lump.
        residue_sg: Gravity of the lump; default from the crude's Watson K.
        residue_mw: Molecular weight of the lump (set directly; Twu and
            Riazi-Daubert have no root or no validity there).
        fine_step_C: Grid spacing for fitting the bulk-property balances
            (Watson K, contaminant scales) over the *whole* crude.
        fine_start_C: Bottom of that grid; the crude below it is one lump.

    Returns:
        :class:`Characterization`.
    """
    cuts_np = np.asarray(vacuum_cuts() if cuts_C is None else cuts_C, dtype=float)
    return _characterize(assay, tuple(float(c) for c in cuts_np), residue_Tb_C,
                         residue_sg, residue_mw, float(fine_step_C),
                         float(fine_start_C))


@partial(jax.jit, static_argnums=(1, 5, 6))
def _characterize(assay, cuts_key, residue_Tb_C, residue_sg, residue_mw,
                  fine_step_C, fine_start_C):
    cuts_np = np.asarray(cuts_key)
    cuts = jnp.asarray(cuts_np)
    lo, hi = cuts[:-1], cuts[1:]
    names = tuple(f"PC{int(round(a))}_{int(round(b))}"
                  for a, b in zip(cuts_np[:-1], cuts_np[1:])) + ("RESID",)

    def cut_set(lo, hi):
        x_lo, x_hi = tbp_fraction(assay, lo), tbp_fraction(assay, hi)
        w = x_hi - x_lo
        Tb_C = tbp_temperature(assay, 0.5 * (x_lo + x_hi))  # mid-percent
        return w, Tb_C

    w, Tb_C = cut_set(lo, hi)
    x_end = tbp_fraction(assay, cuts[-1])
    w_res = 1.0 - x_end
    light = tbp_fraction(assay, cuts[0])

    # Watson K from the bulk gravity, over the whole crude on a fine grid
    # (a static grid: the TBP temperatures may be traced)
    nfine = int(np.ceil((cuts_np[-1] - fine_start_C) / fine_step_C))
    fine = jnp.linspace(fine_start_C, cuts_np[-1], nfine + 1)
    wf, Tbf_C = cut_set(fine[:-1], fine[1:])
    wf0 = tbp_fraction(assay, fine[0])          # below the fine grid
    Tb0_C = tbp_temperature(assay, 0.5 * wf0)
    wf = jnp.concatenate([wf0[None], wf, w_res[None]])
    Tbf_K = jnp.concatenate([Tb0_C[None], Tbf_C,
                             jnp.asarray([residue_Tb_C], dtype=float)]) + C_TO_K
    cbrt = (1.8 * Tbf_K) ** (1.0 / 3.0)
    # 1/SG_bulk = sum w_i / SG_i = Kw^-1 ... no: SG_i = cbrt_i / Kw
    #   => 1/SG_bulk = Kw * sum(w_i / cbrt_i)
    Kw = 1.0 / (assay.bulk_sg() * jnp.sum(wf / cbrt))
    Tbf_C_all = Tbf_K - C_TO_K

    Tb = Tb_C + C_TO_K
    SG = corr.sg_from_watson_k(Tb, Kw)
    twu = corr.twu_critical_properties(Tb, SG)
    om = corr.lee_kesler_acentric(Tb, twu["Tc"], twu["Pc"], Kw)

    res_Tb = jnp.asarray(residue_Tb_C, dtype=float) + C_TO_K
    res_SG = (corr.sg_from_watson_k(res_Tb, Kw) if residue_sg is None
              else jnp.asarray(residue_sg, dtype=float))
    # lump criticals: Twu at the last cut's boundary (as far as it goes),
    # carried to the lump's Tb at constant Tb/Tc; Pc and omega stay at the
    # edge values. Placeholders, reported for completeness -- the lump's VLE
    # and enthalpy use only Tb, SG, Kw and MW.
    T_edge = cuts[-1] + C_TO_K
    edge = corr.twu_critical_properties(T_edge, corr.sg_from_watson_k(T_edge, Kw))
    res_om = corr.lee_kesler_acentric(T_edge, edge["Tc"], edge["Pc"], Kw)
    res_Tc = edge["Tc"] * res_Tb / T_edge

    # contaminants: shape(T) scaled to the bulk number over the whole crude
    Tb_all_C = jnp.concatenate([Tb_C, jnp.asarray([residue_Tb_C], dtype=float)])
    bulk = {
        "sulfur": assay.sulfur_wt / 100.0,
        "nitrogen": assay.nitrogen_wppm * 1e-6,
        "ccr": assay.ccr_wt / 100.0,
        "nickel_vanadium": assay.nickel_vanadium_wppm * 1e-6,
        "asphaltenes": assay.asphaltenes_wt / 100.0,
    }
    measured = {"sulfur": (assay.sulfur_curve, 1e-2),
                "nitrogen": (assay.nitrogen_curve, 1e-6),
                "ccr": (assay.ccr_curve, 1e-2)}
    contaminants = {}
    for key, (center, width) in _SHAPES.items():
        curve = measured.get(key, (None, None))
        if curve[0] is not None:
            arr = jnp.asarray(curve[0], dtype=float)
            contaminants[key] = jnp.interp(Tb_all_C, arr[:, 0], arr[:, 1]) * curve[1]
            continue
        shape_f = _logistic(Tbf_C_all, center, width)
        scale = bulk[key] / jnp.sum(wf * shape_f)
        contaminants[key] = jnp.minimum(_logistic(Tb_all_C, center, width) * scale,
                                        0.9)

    def cat(a, b):
        return jnp.concatenate([a, jnp.atleast_1d(b)])

    span = cuts[-1] - cuts[-2]
    comps = PseudoComponents(
        Tb=cat(Tb, res_Tb),
        SG=cat(SG, res_SG),
        MW=cat(twu["MW"], jnp.asarray(residue_mw, dtype=float)),
        Kw=jnp.full(len(names), Kw),
        Tc=cat(twu["Tc"], res_Tc),
        Pc=cat(twu["Pc"], edge["Pc"]),
        omega=cat(om, res_om),
        T_lo=cat(lo, cuts[-1]) + C_TO_K,
        T_hi=cat(hi, cuts[-1] + 4.0 * span) + C_TO_K,
        names=names,
        **contaminants,
    )
    return Characterization(components=comps, yields=cat(w, w_res),
                            light_ends=light, Kw=Kw)


# ---------------------------------------------------------------------------
# Product properties
# ---------------------------------------------------------------------------


def product_properties(components: PseudoComponents, mass_flows):
    """Bulk properties of a product from its pseudocomponent mass flows.

    Args:
        components: The pseudocomponent set.
        mass_flows: Mass flow (any consistent unit) per pseudocomponent.

    Returns:
        dict: ``rate`` (sum of ``mass_flows``), ``sg`` (volume-additive),
        ``api``, ``sulfur_wt``, ``nitrogen_wppm``, ``ccr_wt``,
        ``nickel_vanadium_wppm``, ``asphaltenes_wt``, ``mw`` (number
        average), and TBP ``T05``, ``T10``, ``T50``, ``T90``, ``T95`` (K),
        read off the
        product's curve with each pseudocomponent spread uniformly over its
        cut.
    """
    m = jnp.asarray(mass_flows)
    total = jnp.sum(m)
    w = m / total
    sg = 1.0 / jnp.sum(w / components.SG)
    out = {
        "rate": total,
        "sg": sg,
        "api": corr.api_from_sg(sg),
        "sulfur_wt": 100.0 * jnp.sum(w * components.sulfur),
        "nitrogen_wppm": 1e6 * jnp.sum(w * components.nitrogen),
        "ccr_wt": 100.0 * jnp.sum(w * components.ccr),
        "nickel_vanadium_wppm": 1e6 * jnp.sum(w * components.nickel_vanadium),
        "asphaltenes_wt": 100.0 * jnp.sum(w * components.asphaltenes),
        "mw": 1.0 / jnp.sum(w / components.MW),
    }
    for p in (5, 10, 50, 90, 95):
        out[f"T{p:02d}"] = tbp_point(components, w, p / 100.0)
    return out


def tbp_point(components: PseudoComponents, mass_fractions, fraction):
    """TBP temperature (K) at which ``fraction`` of a product has distilled."""
    cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(mass_fractions)])
    Tk = jnp.concatenate([components.T_lo[:1], components.T_hi])
    return jnp.interp(fraction, cum, Tk)


# ---------------------------------------------------------------------------
# Feed for the vacuum column
# ---------------------------------------------------------------------------


def atmospheric_residue(
    char: Characterization,
    crude_rate_kg_s,
    cut_point_C=370.0,
    sharpness_C=12.0,
    T_C=360.0,
    P=200e3,
):
    """Atmospheric residue as a difflow stream, from an idealized CDU cut.

    A stand-in for the bottoms of a crude column: each pseudocomponent goes
    to the residue in proportion ``1 / (1 + exp(-(Tb - cut) / sharpness))``,
    which gives the overlap a real CDU stripping section leaves (light gas
    oil in the residue, residue in the gas oil). When ``CrudeColumn`` lands,
    its bottoms stream replaces this function and nothing downstream
    changes: the vacuum column only sees a stream.

    Args:
        char: The crude's characterization.
        crude_rate_kg_s: Crude charge (kg/s).
        cut_point_C: TBP cut point between AGO and residue (C).
        sharpness_C: Width of the overlap (C).
        T_C: Temperature of the residue as it reaches the vacuum furnace (C).
        P: Its pressure (Pa).

    Returns:
        A difflow stream ``{"F_<name>": mol/s, ..., "T": K, "P": Pa}``.
    """
    comps = char.components
    Tb_C = comps.Tb - C_TO_K
    frac = 1.0 / (1.0 + jnp.exp(-(Tb_C - cut_point_C) / sharpness_C))
    mass = crude_rate_kg_s * char.yields * frac          # kg/s
    mol = mass / comps.MW * 1000.0                         # mol/s
    stream = {f"F_{n}": mol[i] for i, n in enumerate(comps.names)}
    stream["T"] = jnp.asarray(T_C, dtype=float) + C_TO_K
    stream["P"] = jnp.asarray(P, dtype=float)
    return stream
