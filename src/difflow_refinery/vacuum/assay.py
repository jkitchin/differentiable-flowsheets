"""The vacuum column's view of a crude: compatibility layer over one assay.

The vacuum column used to have its own assay and characterization. Since
#301 there is one: :class:`difflow_refinery.assay.Assay` with a
:class:`~difflow_refinery.assay.HeavyEnd` (curve extended on a probability
scale, a residue lump whose properties are set directly) and contaminants per
cut, characterised by :func:`difflow_refinery.assay.characterize`, which the
crude unit, this column and the blending pool all read. What remains here:

- :class:`PseudoComponents`, the column's property table (now built by
  :meth:`difflow_refinery.assay.Characterization.pseudo_components`), with
  :func:`product_properties`, :func:`tbp_point` and
  :func:`atmospheric_residue` over it.
- :class:`Assay` -- a thin input adapter in the vacuum column's units (TBP in
  C against cumulative wt%, contaminants with the old defaults), kept so code
  written against it runs. It holds data only; :meth:`Assay.to_assay` is the
  one place it turns into the shared assay, and nothing here characterises.
  It was kept rather than removed because its differences are units and
  defaults, not modelling, and every vacuum example, test and notebook used
  it; a shim costs nothing to keep and breaks nobody.
- :func:`characterize` -- the shared characterization, sliced to the cuts at
  and above ``cuts_C[0]`` (the old interface, where everything lighter was
  reported only as a ``light_ends`` fraction).

What changed numerically with the merge, so that old numbers can be compared:
each cut's Tb is the mean TBP temperature over the cut (it was the
mid-percent temperature); the crude's Watson K is fitted over the
characterization's own cuts and lump in closed form (it was fitted over a
separate 10 C grid); and contaminants are scaled over the cuts themselves.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from difflow_refinery import assay as _assay
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
        return dataclasses.replace(self, tbp_C=T)

    def to_assay(self, heavy_end: "_assay.HeavyEnd | None" = None) -> "_assay.Assay":
        """The same data as a :class:`difflow_refinery.assay.Assay`.

        Mass basis, kelvin, a :class:`~difflow_refinery.assay.HeavyEnd`
        (default settings unless one is given) and the contaminants. Curves
        given as ``(T_C, value)`` rows become ``(T_K, value)`` columns.
        """
        def curve(rows):
            if rows is None:
                return None
            arr = jnp.asarray(rows, dtype=float)
            return (arr[:, 0] + C_TO_K, arr[:, 1])

        return _assay.Assay(
            tbp_percent=self.tbp_wt,
            tbp_T=jnp.asarray(self.tbp_C, dtype=float) + C_TO_K,
            basis="mass",
            sg=self.sg,
            api=self.api if self.sg is None else None,
            name=self.name,
            heavy_end=_assay.HeavyEnd() if heavy_end is None else heavy_end,
            sulfur_wt=self.sulfur_wt,
            nitrogen_wppm=self.nitrogen_wppm,
            ccr_wt=self.ccr_wt,
            nickel_vanadium_wppm=self.nickel_vanadium_wppm,
            asphaltenes_wt=self.asphaltenes_wt,
            sulfur_curve=curve(self.sulfur_curve),
            nitrogen_curve=curve(self.nitrogen_curve),
            ccr_curve=curve(self.ccr_curve),
        )


jax.tree_util.register_dataclass(
    Assay,
    data_fields=["tbp_C", "tbp_wt", "sg", "api", "sulfur_wt", "nitrogen_wppm",
                 "ccr_wt", "nickel_vanadium_wppm", "asphaltenes_wt",
                 "sulfur_curve", "nitrogen_curve", "ccr_curve"],
    meta_fields=["name"],
)


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


def _as_assay(assay) -> "_assay.Assay":
    if isinstance(assay, Assay):
        return assay.to_assay()
    if assay.heavy_end is None:
        raise ValueError("the vacuum column needs an assay with a heavy_end")
    return assay


def tbp_fraction(assay, T_C, n_tail: int = 3):
    """Cumulative mass fraction distilled at ``T_C`` (C), any temperature.

    The shared assay's probability-scale curve
    (:func:`difflow_refinery.assay.tbp_curve`); ``n_tail`` is kept for
    compatibility and read from the assay's heavy end.
    """
    return _assay.tbp_curve(_as_assay(assay))(jnp.asarray(T_C, dtype=float) + C_TO_K)


def tbp_temperature(assay, fraction, n_tail: int = 3):
    """Inverse of :func:`tbp_fraction`: TBP temperature (C) at a mass fraction."""
    return _assay.tbp_curve(_as_assay(assay)).invert(fraction) - C_TO_K


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
    """Result of :func:`characterize`: the cuts at and above ``cuts_C[0]``.

    Attributes:
        components: The pseudocomponent property table (residue lump last).
        yields: Mass fraction of the whole crude in each pseudocomponent.
        light_ends: Mass fraction of the crude below ``cuts_C[0]`` (in the
            shared characterization, but not in this view).
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


def _take(components: PseudoComponents, start: int) -> PseudoComponents:
    arrays = {f.name: getattr(components, f.name)[start:]
              for f in dataclasses.fields(components) if f.name != "names"}
    return PseudoComponents(names=components.names[start:], **arrays)


def characterize(
    assay,
    cuts_C=None,
    residue_Tb_C: float = 950.0,
    residue_sg=None,
    residue_mw: float = 1500.0,
    fine_step_C: float = 10.0,
    fine_start_C: float = 0.0,
) -> Characterization:
    """The shared characterization on a vacuum cut grid (compatibility).

    Cuts the assay with :func:`difflow_refinery.assay.characterize` at
    ``cuts_C`` -- the last boundary is the heavy end's ``T_max`` and the
    lump follows -- and returns the cuts from ``cuts_C[0]`` up. Material
    below ``cuts_C[0]`` is in the characterization as one more cut (or
    more, where ``cuts_C`` starts below the first TBP point) and is reported
    here as ``light_ends``.

    Args:
        assay: A :class:`Assay` or a :class:`difflow_refinery.assay.Assay`
            with a heavy end (whose ``T_max`` and residue settings are then
            replaced by these arguments).
        cuts_C: Cut boundaries (C). Default :func:`vacuum_cuts`.
        residue_Tb_C: Equivalent boiling point of the residue lump (C).
        residue_sg: Gravity of the lump; default from the crude's Watson K.
        residue_mw: Molecular weight of the lump.
        fine_step_C, fine_start_C: Ignored; kept for signature compatibility
            (the Watson K is now fitted over the cuts themselves).
    """
    ua = _as_assay(assay)
    cuts = np.asarray(vacuum_cuts() if cuts_C is None else cuts_C, dtype=float) + C_TO_K
    pts = cuts[:-1]
    if not _assay._traced((ua.tbp_T, ua.tbp_percent)):
        x_le = sum(float(v) for v in ua.light_ends.values()) / 100.0
        T_start = float(_assay._heavy_start(ua, _assay.tbp_curve(ua), x_le))
        pts = pts[pts > T_start + 1e-6]
    i0 = int(np.sum(pts <= cuts[0]))
    he = dataclasses.replace(
        ua.heavy_end, T_max=float(cuts[-1]),
        residue_Tb=jnp.asarray(residue_Tb_C, dtype=float) + C_TO_K,
        residue_mw=residue_mw, residue_sg=residue_sg,
        lump_span=4.0 * (cuts[-1] - cuts[-2]) / (cuts[-1] - (pts[-1] if pts.size else cuts[-2])))
    full = _characterize_cached(dataclasses.replace(ua, heavy_end=he), tuple(float(p) for p in pts))
    k = len(full.light_names)
    yields = full.mass_fraction[k + i0:]
    comps = _take(full.pseudo_components(), i0)
    return Characterization(components=comps, yields=yields,
                            light_ends=1.0 - jnp.sum(yields), Kw=full.Kw[i0])


def _characterize_cached(ua, pts):
    return _assay.characterize(ua, cut_points=pts, method="twu_1984")


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
        char: The crude's characterization: the shared
            :class:`difflow_refinery.assay.Characterization` (light ends are
            left out -- they leave with the crude column's overhead) or the
            vacuum view from :func:`characterize`.
        crude_rate_kg_s: Crude charge (kg/s).
        cut_point_C: TBP cut point between AGO and residue (C).
        sharpness_C: Width of the overlap (C).
        T_C: Temperature of the residue as it reaches the vacuum furnace (C).
        P: Its pressure (Pa).

    Returns:
        A difflow stream ``{"F_<name>": mol/s, ..., "T": K, "P": Pa}``.
    """
    if isinstance(char, _assay.Characterization):
        comps = char.pseudo_components()
        yields = char.mass_fraction[len(char.light_names):]
    else:
        comps, yields = char.components, char.yields
    Tb_C = comps.Tb - C_TO_K
    frac = 1.0 / (1.0 + jnp.exp(-(Tb_C - cut_point_C) / sharpness_C))
    mass = crude_rate_kg_s * yields * frac          # kg/s
    mol = mass / comps.MW * 1000.0                         # mol/s
    stream = {f"F_{n}": mol[i] for i, n in enumerate(comps.names)}
    stream["T"] = jnp.asarray(T_C, dtype=float) + C_TO_K
    stream["P"] = jnp.asarray(P, dtype=float)
    return stream
