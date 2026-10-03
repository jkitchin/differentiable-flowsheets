"""Product properties estimated from a stream's composition (#330).

The blend pool (:mod:`difflow_refinery.blending`) blends flash, freeze and
smoke points, viscosity and octane, but a component built from a stream had
to be *given* them.  This module estimates them from what a pseudocomponent
stream carries -- boiling points, gravities, molar masses and, where the
characterization has one, the #305 hydrocarbon-type composition -- and
:meth:`BlendComponent.from_stream <difflow_refinery.blending.BlendComponent.from_stream>`
uses them whenever the caller gives no measured value.

=====================  ===============================================  ============
property               estimate                                         status
=====================  ===============================================  ============
``flash_C``            Riazi-Daubert (1987) / API TDB 2B7.1, from the   (unverified)
                       ASTM D86 10 % point (:func:`flash_point`)
``freeze_C``           ideal solubility of the n-paraffins, Won (1986)  melting points
                       melting points and heats of fusion               checked (see
                       (:func:`freeze_point`)                           there)
``smoke_mm``           Riazi, MNL50 Ch. 3, from Watson K and Tb         (unverified)
                       (:func:`smoke_point`)
``viscosity_cSt``      Abbott, Kaufmann & Domash (1971) / API TDB       (unverified)
                       11A4.2 at 100 and 210 F, ASTM D341 (Walther) to
                       any temperature (:func:`abbott_viscosity`,
                       :func:`walther_viscosity`)
``RON``, ``MON``       straight-run naphtha octane from the P/N/A/O     (unverified)
                       composition: pure-compound octanes of the
                       reformer's model compounds, blended by Ethyl
                       RT-70 (:func:`straight_run_octane`)
=====================  ===============================================  ============

"(unverified)" means what it says: no primary source could be opened from the
environment this was written in (publishers, ASTM and the API Technical Data
Book were unreachable), so each correlation's form and constants are as
recalled by this project and NOT checked against the source, and no
published worked example is reproduced.  The tests check what can be checked
without one: limits, monotonicity in the direction the physics requires,
plausible values on typical products, and differentiability.  A measured
value always wins: pass it to ``from_stream`` as an override.

The freeze-point estimate is the exception, in part: Won's n-paraffin
melting-point correlation is checked against the CRC/OpenNotebook melting
points of n-C10 to n-C24 (as tabulated in the ``chemicals`` package, version
1.5.2) in ``tests/refinery/test_properties.py``; its heat of fusion is
compared with the CRC heats of fusion there too, and the comparison is
recorded rather than hidden (see :func:`won_heat_of_fusion`).  The freeze
point built from them -- the temperature at which the last n-paraffin
crystal dissolves -- is a physical model, not a fitted correlation, and it is
not validated against a measured jet freeze point.

Every function is pure JAX and differentiable in its inputs, so a property
computed through :func:`estimate_properties` reaches back to the stream and,
through the characterization, to the assay.
"""

from __future__ import annotations

import warnings
from typing import Iterable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow_refinery.correlations import watson_k

jax.config.update("jax_enable_x64", True)

_C0 = 273.15
_R = 8.314462618
_CAL = 4.184

#: 100 F and 210 F, the temperatures of the Abbott correlations (K).
T_100F = 310.92777777777775
T_210F = 371.8833333333333

#: Properties :func:`estimate_properties` can return.
ESTIMATED_PROPERTIES: tuple[str, ...] = (
    "flash_C", "freeze_C", "smoke_mm", "viscosity_cSt", "RON", "MON")


class PropertyRangeWarning(UserWarning):
    """A property correlation is used outside the range it is meaningful in."""


def _concrete(x) -> np.ndarray | None:
    try:
        return np.asarray(x)
    except (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError):
        return None


# ---------------------------------------------------------------------------
# Flash point.
# ---------------------------------------------------------------------------

def flash_point(T10_d86_K: Array) -> Array:
    """Flash point (K) of a petroleum fraction from its ASTM D86 10 % point (K).

    ``1/T_F = -0.024209 + 2.84947/T10 + 3.4254e-3 ln T10``, temperatures in
    kelvin.  Riazi & Daubert (1987), adopted as API Technical Data Book
    procedure 2B7.1; restated in Riazi, *Characterization and Properties of
    Petroleum Fractions*, ASTM MNL50 (2005), Ch. 3 (the flash point section).
    Riazi's reasoning for the 10 % point: the flash point is set by the
    lightest material in the fraction.  Closed-cup basis.

    (unverified) The form and the constants are as recalled; neither the
    API TDB nor MNL50 could be opened, and no worked example is reproduced.
    On typical products it gives ~ -45 C for a gasoline (D86 T10 ~ 50 C),
    55-65 C for a kerosene (T10 ~ 180-190 C) and ~ 95 C for a diesel
    (T10 ~ 245 C): the right ordering and roughly the right values, with
    kerosenes on the high side of what is measured.
    """
    T = jnp.asarray(T10_d86_K)
    return 1.0 / (-0.024209 + 2.84947 / T + 3.4254e-3 * jnp.log(T))


# ---------------------------------------------------------------------------
# Freeze point.
# ---------------------------------------------------------------------------

def won_melting_point(MW: Array) -> Array:
    """Melting point (K) of the n-paraffin of molar mass ``MW`` (g/mol).

    ``T_f = 374.5 + 0.02617 MW - 20172/MW``; Won, *Fluid Phase Equilibria*
    30 (1986) 265-279 (wax precipitation).  The constants are as recalled,
    and checked: against the tabulated melting points of n-C10 to n-C24
    (CRC / OpenNotebook, as shipped with ``chemicals`` 1.5.2) the
    correlation is within 7 K at n-C10 and within 2.5 K from n-C12 up
    (``tests/refinery/test_properties.py``).  Below about n-C7 it is
    meaningless (it goes negative near MW 56); those paraffins do not set a
    freeze point.
    """
    MW = jnp.asarray(MW)
    return 374.5 + 0.02617 * MW - 20172.0 / MW


def won_heat_of_fusion(MW: Array, Tm: Array | None = None) -> Array:
    """Heat of fusion (J/mol) of the n-paraffin of molar mass ``MW``.

    ``dH_f = 0.1426 MW T_f`` cal/mol; Won (1986), with ``T_f`` from
    :func:`won_melting_point`.  The constant is as recalled.  Against the
    CRC heats of fusion (``chemicals`` 1.5.2) it is 4-7 % high for the
    odd-carbon n-paraffins n-C11 to n-C19 and 25-32 % low for the even
    ones n-C10 to n-C20, which melt from a different crystal form -- the
    odd/even alternation a single smooth correlation cannot follow (both
    pinned in ``tests/refinery/test_properties.py``).  A low
    heat of fusion puts the freeze-point estimate a few kelvin LOW, i.e.
    on the optimistic side of a maximum-freeze-point spec.
    """
    MW = jnp.asarray(MW)
    Tm = won_melting_point(MW) if Tm is None else Tm
    return 0.1426 * MW * Tm * _CAL


def saturation_temperature(x: Array, MW: Array) -> Array:
    """Temperature (K) at which an n-paraffin at mole fraction ``x`` saturates.

    Ideal solubility (no heat-capacity term, an ideal liquid solution and a
    pure solid): ``ln x = -(dH_f/R)(1/T - 1/T_f)``, so
    ``1/T = 1/T_f - R ln x / dH_f``, with Won's ``T_f`` and ``dH_f``.
    """
    MW = jnp.asarray(MW)
    Tm = won_melting_point(MW)
    dH = won_heat_of_fusion(MW, Tm)
    return 1.0 / (1.0 / Tm - _R * jnp.log(x) / dH)


def freeze_point(x_nP: Array, MW: Array, width: float = 0.5,
                 min_MW: float = 100.0) -> Array:
    """Freeze point (K): the highest n-paraffin saturation temperature.

    ASTM D2386 measures the temperature at which the last hydrocarbon
    crystal disappears on warming, and in a kerosene those crystals are
    n-paraffins.  So each pseudocomponent's n-paraffins (mole fraction
    ``x_nP_j`` of the whole liquid, molar mass ``MW_j``) get a saturation
    temperature (:func:`saturation_temperature`), and the freeze point is
    the largest of them -- a smooth maximum, ``width log sum exp(T_j /
    width)`` over the components that have any, so the estimate is
    differentiable.

    Assumptions (a model, not a correlation; not validated against a
    measured freeze point): each cut's n-paraffins are one n-alkane of the
    cut's molar mass, so a coarse cut grid hides the heaviest members of a
    cut and puts the estimate LOW; the n-paraffins crystallize pure, from an
    ideal solution; iso-paraffins, naphthenes and aromatics do not
    crystallize first.  Components lighter than ``min_MW`` (default 100,
    about n-C7, which melts at 182 K) are left out: Won's correlation is not
    meant for them and they never set a jet freeze point.

    Args:
        x_nP: Mole fraction of each component's n-paraffins in the liquid.
        MW: Molar mass of each component (g/mol).
        width: Smoothing of the maximum (K).
        min_MW: Lightest component considered (g/mol).
    """
    x_nP = jnp.asarray(x_nP)
    MW = jnp.asarray(MW)
    take = (x_nP > 1e-12) & (MW >= min_MW)
    x_s = jnp.where(take, x_nP, 0.5)
    MW_s = jnp.where(take, MW, 200.0)
    T = saturation_temperature(x_s, MW_s)
    T = jnp.where(take, T, 0.0)
    # A floor term, so a liquid with no n-paraffin heavy enough to matter (a
    # light naphtha) gets FREEZE_FLOOR_K rather than -inf.
    T = jnp.concatenate([T, jnp.asarray([FREEZE_FLOOR_K])])
    b = jnp.concatenate([take.astype(T.dtype), jnp.ones(1)])
    return width * jax.nn.logsumexp(T / width, b=b)


#: What :func:`freeze_point` returns for a liquid with no n-paraffins heavier
#: than ``min_MW`` (K; -123 C): below any freeze-point specification, and a
#: floor of the smooth maximum, never a value.
FREEZE_FLOOR_K = 150.0


# ---------------------------------------------------------------------------
# Smoke point.
# ---------------------------------------------------------------------------

def smoke_point(Tb: Array, SG: Array) -> Array:
    """Smoke point (mm) of a kerosene-range fraction from ``Tb`` (K) and ``SG``.

    ``SP = exp(-1.028 + 0.474 Kw - 0.00168 Tb')``, ``Kw`` the Watson
    characterization factor; Riazi, ASTM MNL50 (2005), Ch. 3 (smoke point;
    after the API Technical Data Book's method on aniline point / gravity).
    The smoke point rises with paraffinicity (Kw) and falls with boiling
    point.

    (unverified) The constants are as recalled, and so is the unit of the
    boiling point, which is the doubtful part: read with ``Tb'`` in kelvin
    the equation puts an ordinary kerosene (Tb 473 K, SG 0.80) at 44 mm,
    far above the 20-30 mm such kerosenes measure; read with ``Tb'`` in
    degrees Rankine (``1.8 Tb``) it puts it at 23 mm, and a naphthenic one
    (Kw 11.2) at 15 mm.  The Rankine reading is used here because it is the
    one that gives measured magnitudes; that is a judgement, not a citation.
    Meaningful for kerosene-range fractions (Tb roughly 420-560 K).
    """
    Tb = jnp.asarray(Tb)
    Kw = watson_k(Tb, SG)
    return jnp.exp(-1.028 + 0.474 * Kw - 0.00168 * 1.8 * Tb)


# ---------------------------------------------------------------------------
# Viscosity.
# ---------------------------------------------------------------------------

def api_gravity(SG: Array) -> Array:
    """API gravity of a liquid of specific gravity ``SG`` (60 F/60 F)."""
    return 141.5 / jnp.asarray(SG) - 131.5


def _abbott_denominators(Kw, API):
    return API + 50.3642 - 4.78231 * Kw, API + 26.786 - 2.6296 * Kw


def abbott_viscosity(Tb: Array, SG: Array) -> tuple[Array, Array]:
    """Kinematic viscosities (cSt) at 100 F and 210 F from ``Tb`` (K) and ``SG``.

    Abbott, Kaufmann & Domash, *Can. J. Chem. Eng.* 49 (1971) 379-384,
    adopted as API Technical Data Book procedure 11A4.2 and restated in
    Riazi, ASTM MNL50 (2005), Ch. 2::

        log10 nu100 = 4.39371 - 1.94733 K + 0.12769 K^2 + 3.2629e-4 A^2
                      - 1.18246e-2 K A + (0.171617 K^2 + 10.9943 A
                      + 9.50663e-2 A^2 - 0.860218 K A)
                      / (A + 50.3642 - 4.78231 K)
        log10 nu210 = -0.463634 - 0.166532 A + 5.13447e-4 A^2
                      - 8.48995e-3 K A + (8.0325e-2 K + 1.24899 A
                      + 0.19768 A^2) / (A + 26.786 - 2.6296 K)

    with ``K`` the Watson factor and ``A`` the API gravity.  Each has a
    pole where its denominator vanishes (low API at high K); a denominator
    below 5 raises :class:`PropertyRangeWarning` (concrete inputs only).

    (unverified) Constants as recalled; no worked example reproduced.  On
    typical products it gives 1.4 cSt at 100 F for a kerosene (SG 0.80,
    Tb 473 K) and 3.8 cSt for a diesel (SG 0.85, Tb 560 K) -- the right
    magnitudes.  Riazi gives its range as roughly Kw 10-13 and API 0-80;
    a residue is near the edge of it.
    """
    Tb, SG = jnp.asarray(Tb), jnp.asarray(SG)
    K = watson_k(Tb, SG)
    A = api_gravity(SG)
    d100, d210 = _abbott_denominators(K, A)
    for d in (d100, d210):
        dc = _concrete(d)
        if dc is not None and np.any(dc < 5.0):
            warnings.warn(
                "Abbott viscosity near its pole (low API gravity at high Watson K); "
                "the estimate is not meaningful here -- give a measured viscosity",
                PropertyRangeWarning, stacklevel=2)
            break
    log100 = (4.39371 - 1.94733 * K + 0.12769 * K ** 2 + 3.2629e-4 * A ** 2
              - 1.18246e-2 * K * A
              + (0.171617 * K ** 2 + 10.9943 * A + 9.50663e-2 * A ** 2
                 - 0.860218 * K * A) / d100)
    log210 = (-0.463634 - 0.166532 * A + 5.13447e-4 * A ** 2 - 8.48995e-3 * K * A
              + (8.0325e-2 * K + 1.24899 * A + 0.19768 * A ** 2) / d210)
    return 10.0 ** log100, 10.0 ** log210


def walther_viscosity(nu1: Array, T1: Array, nu2: Array, T2: Array,
                      T: Array) -> Array:
    """Kinematic viscosity (cSt) at ``T`` from two points, ASTM D341 (Walther).

    ``log10 log10(nu + 0.7) = A - B log10 T`` (T in K), the straight line of
    the ASTM D341 viscosity-temperature chart, through ``(T1, nu1)`` and
    ``(T2, nu2)``.  D341 adds small correction terms below about 2 cSt;
    they are left out, so a light distillate's viscosity is approximate
    there.  Exact on its two points by construction (tested).
    """
    def ll(nu):
        return jnp.log10(jnp.log10(jnp.asarray(nu) + 0.7))

    x1, x2, x = jnp.log10(T1), jnp.log10(T2), jnp.log10(T)
    y = ll(nu1) + (ll(nu2) - ll(nu1)) * (x - x1) / (x2 - x1)
    return 10.0 ** (10.0 ** y) - 0.7


def viscosity(Tb: Array, SG: Array, T: Array) -> Array:
    """Kinematic viscosity (cSt) at ``T`` (K): Abbott at 100/210 F, Walther between/beyond."""
    nu100, nu210 = abbott_viscosity(Tb, SG)
    return walther_viscosity(nu100, T_100F, nu210, T_210F, T)


# ---------------------------------------------------------------------------
# Straight-run naphtha octane.
# ---------------------------------------------------------------------------

#: Model-compound series per hydrocarbon type for the octane estimate:
#: species keys of :data:`difflow_refinery.reforming.species.SPECIES`, in
#: order of boiling point.
OCTANE_SERIES: dict[str, tuple[str, ...]] = {
    "nP": ("nC4", "nC5", "nP6", "nP7", "nP8", "nP9", "nP10"),
    "iP": ("iC4", "iC5", "iP6", "iP7", "iP8", "iP9", "iP10"),
    "N": ("N5_6", "N6", "N7", "N8", "N9", "N10"),
    "A": ("A6", "A7", "A8", "A9", "A10"),
}

#: Light ends of :data:`difflow_refinery.thermo.LIGHT_ENDS` that are one
#: compound with an octane number, mapped to the reformer species that is
#: that compound.  Methane, ethane and propane carry none and are left out
#: of the octane average.
LIGHT_END_OCTANE: dict[str, str] = {
    "isobutane": "iC4", "n_butane": "nC4", "isopentane": "iC5",
    "n_pentane": "nC5", "n_hexane": "nP6",
}


def _series_tables():
    from difflow_refinery.reforming import species as sp

    out = {}
    for kind, keys in OCTANE_SERIES.items():
        tb = np.array([sp.SPECIES[k].Tb for k in keys])
        order = np.argsort(tb)
        out[kind] = tuple(np.array([getattr(sp.SPECIES[keys[i]], a) for i in order])
                          for a in ("Tb", "RON", "MON"))
    return out


def group_octane(kind: str, Tb: Array) -> tuple[Array, Array]:
    """Pure-compound (RON, MON) of hydrocarbon type ``kind`` boiling at ``Tb`` (K).

    Linear in ``Tb`` between the model compounds of :data:`OCTANE_SERIES`
    (the reformer's, :mod:`difflow_refinery.reforming.species`: n-alkanes,
    2-methylalkanes, the cyclohexane series with methylcyclopentane, the
    n-alkylbenzenes), held at the end values outside them.  The octanes are
    those of :mod:`~difflow_refinery.reforming.species` -- API Research
    Project 45 values as recalled there, (unverified), with the C9-C10
    paraffins extrapolated.
    """
    tb, ron, mon = _series_tables()[kind]
    Tb = jnp.asarray(Tb)
    return jnp.interp(Tb, tb, ron), jnp.interp(Tb, tb, mon)


def straight_run_octane(phi: Array, Tb: Array, types_vol: Mapping[str, Array],
                        names: Sequence[str] | None = None,
                        n_paraffin_share: float = 0.5) -> tuple[Array, Array]:
    """RON and MON of a straight-run (or hydrotreated) naphtha from its composition.

    Each pseudocomponent ``j`` (volume fraction ``phi_j``, boiling point
    ``Tb_j``) is split by its #305 hydrocarbon types into sub-components:
    n-paraffins (``n_paraffin_share`` of the paraffins), iso-paraffins (the
    rest), naphthenes, aromatics and olefins, each with the pure-compound
    octane of its type at ``Tb_j`` (:func:`group_octane`).  The sub-components
    are blended with the Ethyl RT-70 rule (aromatics 100 vol% for an aromatic,
    olefins 100 vol% for an olefin) -- exactly as
    :func:`difflow_refinery.reforming.products.octane` blends the reformate,
    so a straight-run naphtha and a reformate in one gasoline pool are on
    one octane basis.  Light ends that are one compound (:data:`LIGHT_END_OCTANE`)
    take that compound's octane; methane, ethane and propane are left out.

    (unverified) Method of this project, not a published correlation.  Its
    inputs are pure-compound octanes as recalled (see :func:`group_octane`),
    one model compound per type and boiling point (a straight-run C8
    iso-paraffin is many isomers, most of them higher in octane than
    2-methylheptane, so heavy naphthas come out LOW), an ILLUSTRATIVE
    n-/iso-paraffin split, and olefins given the iso-paraffin octane as a
    placeholder (straight-run naphtha has none).  Typical straight-run
    light naphthas (C5-C6) measure RON 60-75 and heavy naphthas 40-60; the
    estimate lands in those ranges on the tests' compositions, which is all
    that is claimed.  Pure-compound octanes below zero and above 100 are
    outside the RT-70 fit; it extrapolates there.  Meaningful for naphthas
    boiling below about 460 K (n-decane); the series are flat beyond.

    Args:
        phi: Liquid volume fraction of each pseudocomponent.
        Tb: Boiling point of each pseudocomponent (K).
        types_vol: ``{"paraffins_vol", "naphthenes_vol", "aromatics_vol",
            "olefins_vol"}`` per pseudocomponent, vol% (``olefins_vol`` may
            be missing: zero).
        names: Pseudocomponent names, to recognise single-compound light ends.
        n_paraffin_share: Normal share of each cut's paraffins. ILLUSTRATIVE.
    """
    from difflow_refinery.blending import ethyl_rt70
    from difflow_refinery.reforming import species as sp

    phi, Tb = jnp.asarray(phi), jnp.asarray(Tb)
    n = phi.shape[0]
    frac = {k: jnp.asarray(types_vol[f"{k}_vol"]) / 100.0 if f"{k}_vol" in types_vol
            else jnp.zeros(n) for k in ("paraffins", "naphthenes", "aromatics", "olefins")}
    names = list(names) if names is not None else [""] * n
    pure = np.array([nm in LIGHT_END_OCTANE for nm in names])
    gas = np.array([nm in ("methane", "ethane", "propane", "hydrogen") for nm in names])
    cut = ~(pure | gas)
    pure_ron = np.array([sp.SPECIES[LIGHT_END_OCTANE[nm]].RON if nm in LIGHT_END_OCTANE else 0.0
                         for nm in names])
    pure_mon = np.array([sp.SPECIES[LIGHT_END_OCTANE[nm]].MON if nm in LIGHT_END_OCTANE else 0.0
                         for nm in names])
    s = jnp.asarray(n_paraffin_share)
    oct_ = {k: group_octane(k, Tb) for k in OCTANE_SERIES}
    m = jnp.asarray(cut, dtype=phi.dtype)
    # (volume, RON, MON, aromatics, olefins) of each sub-component.
    subs = [
        (phi * m * frac["paraffins"] * s, *oct_["nP"], 0.0, 0.0),
        (phi * m * frac["paraffins"] * (1.0 - s), *oct_["iP"], 0.0, 0.0),
        (phi * m * frac["naphthenes"], *oct_["N"], 0.0, 0.0),
        (phi * m * frac["aromatics"], *oct_["A"], 100.0, 0.0),
        (phi * m * frac["olefins"], *oct_["iP"], 0.0, 100.0),
        (phi * jnp.asarray(pure, dtype=phi.dtype), jnp.asarray(pure_ron),
         jnp.asarray(pure_mon), 0.0, 0.0),
    ]
    v = jnp.concatenate([x[0] for x in subs])
    ron = jnp.concatenate([jnp.broadcast_to(x[1], (n,)) for x in subs])
    mon = jnp.concatenate([jnp.broadcast_to(x[2], (n,)) for x in subs])
    aro = jnp.concatenate([jnp.full((n,), x[3]) for x in subs])
    ole = jnp.concatenate([jnp.full((n,), x[4]) for x in subs])
    return ethyl_rt70(v, ron, mon, ole, aro)


# ---------------------------------------------------------------------------
# What a stream gives.
# ---------------------------------------------------------------------------

def estimable(char) -> tuple[str, ...]:
    """The properties :func:`estimate_properties` can give for ``char``.

    ``flash_C``, ``smoke_mm`` and ``viscosity_cSt`` need only boiling points
    and gravities; ``freeze_C`` also needs ``paraffins_vol``; ``RON`` and
    ``MON`` need ``paraffins_vol``, ``naphthenes_vol`` and ``aromatics_vol``
    (a characterization built with a #305 composition has all of them).
    """
    q = char.qualities
    has = {"flash_C", "smoke_mm", "viscosity_cSt"}
    if "paraffins_vol" in q:
        has.add("freeze_C")
    if all(k in q for k in ("paraffins_vol", "naphthenes_vol", "aromatics_vol")):
        has |= {"RON", "MON"}
    return tuple(p for p in ESTIMATED_PROPERTIES if p in has)


def estimate_properties(char, moles: Array, which: Iterable[str] | None = None, *,
                        viscosity_T_C: float = 50.0, n_paraffin_share: float = 0.5,
                        distillation_width: float = 5.0) -> dict[str, Array]:
    """Estimated product properties of a mixture of ``char``'s pseudocomponents.

    Args:
        char: A :class:`~difflow_refinery.characterization.BlendCharacterization`.
        moles: Pseudocomponent molar amounts (any scale), in ``char.names`` order.
        which: Properties to estimate (default: every one in
            :data:`ESTIMATED_PROPERTIES` the characterization has the inputs
            for). ``freeze_C`` needs ``paraffins_vol`` among the
            characterization's qualities; ``RON``/``MON`` need
            ``paraffins_vol``, ``naphthenes_vol`` and ``aromatics_vol``.
            Asking for one whose inputs are missing raises ``KeyError``.
        viscosity_T_C: Temperature of ``viscosity_cSt`` (degC). 50 C is the
            residual fuel oil reference (ISO 8217); every component of a pool
            must be at one temperature for the Refutas rule.
        n_paraffin_share: Normal share of the paraffins (freeze point and
            octane). ILLUSTRATIVE.
        distillation_width: Logistic smoothing (K) of the TBP staircase, as in
            :class:`~difflow_refinery.blending.BlendPool`.

    Characteristic boiling point: the TBP 50 % point of the smoothed curve
    (smoke point, viscosity); the flash point takes the ASTM D86 10 % point
    (Riazi-Daubert TBP -> D86).  The bulk ``SG`` is the volume average.

    Returns:
        ``{property: value}``, temperatures in degC, ``smoke_mm`` in mm,
        ``viscosity_cSt`` in cSt at ``viscosity_T_C``.
    """
    from difflow_refinery.blending import tbp_temperature, tbp_to_d86

    q = char.qualities
    available = set(estimable(char))
    if which is None:
        want = [p for p in ESTIMATED_PROPERTIES if p in available]
    else:
        want = list(which)
        unknown = sorted(set(want) - set(ESTIMATED_PROPERTIES))
        if unknown:
            raise ValueError(f"cannot estimate {unknown}; estimable: {ESTIMATED_PROPERTIES}")
        lacking = [p for p in want if p not in available]
        if lacking:
            raise KeyError(f"estimating {lacking} needs the characterization's hydrocarbon "
                           "types (paraffins_vol, naphthenes_vol, aromatics_vol): build it "
                           "with a composition")
    moles = jnp.asarray(moles)
    vol = moles * char.molar_volume
    phi = vol / jnp.sum(vol)
    sg = jnp.sum(phi * char.SG)
    out: dict[str, Array] = {}
    if any(p in want for p in ("smoke_mm", "viscosity_cSt")):
        tb50 = tbp_temperature(0.5, char.Tb, phi, distillation_width)
    if "flash_C" in want:
        t10 = tbp_to_d86(tbp_temperature(0.1, char.Tb, phi, distillation_width), 10)
        out["flash_C"] = flash_point(t10) - _C0
    if "freeze_C" in want:
        z = moles / jnp.sum(moles)
        x_nP = z * q["paraffins_vol"] / 100.0 * n_paraffin_share
        out["freeze_C"] = freeze_point(x_nP, char.mw) - _C0
    if "smoke_mm" in want:
        out["smoke_mm"] = smoke_point(tb50, sg)
    if "viscosity_cSt" in want:
        out["viscosity_cSt"] = viscosity(tb50, sg, viscosity_T_C + _C0)
    if "RON" in want or "MON" in want:
        ron, mon = straight_run_octane(phi, char.Tb, q, names=char.names,
                                       n_paraffin_share=n_paraffin_share)
        if "RON" in want:
            out["RON"] = ron
        if "MON" in want:
            out["MON"] = mon
    return out
