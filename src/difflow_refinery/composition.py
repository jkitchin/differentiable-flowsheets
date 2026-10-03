"""Hydrocarbon type, hydrogen and heteroatom classes per pseudo-component (#305).

A boiling point and a gravity say how a cut distils; they do not say what it
is made of. Every conversion unit downstream of the crude unit needs that: a
hydrotreater's hydrogen consumption follows the aromatics it saturates and the
sulfur *classes* it has to reach (a thiol goes at any severity, a
4,6-dimethyldibenzothiophene only at the last ppm), a reformer converts
naphthenes, an FCC cracks paraffins and condenses aromatics. This module puts
those numbers on each pseudo-component of a
:class:`~difflow_refinery.assay.Characterization`, as arrays that ride with
the component flows exactly as sulfur, nitrogen and CCR already do, so the
composition of any stream of those components is a weighted average, and
differentiable in the assay.

Per component (light ends first, then the cuts -- :attr:`Characterization.names`
order), a :class:`Composition` holds:

* ``hc_type`` -- ``(n, 4)`` volume fractions of :data:`HC_TYPES`
  (paraffins, naphthenes, aromatics, olefins); every row sums to one. Olefins
  are zero in a straight-run crude: cracking units set them.
* ``hydrogen`` -- hydrogen mass fraction.
* ``sulfur``, ``nitrogen`` -- the characterization's heteroatom mass
  fractions, and ``sulfur_split`` ``(n, 5)`` / ``nitrogen_split`` ``(n, 2)``,
  the share of each in :data:`SULFUR_CLASSES` / :data:`NITROGEN_CLASSES`.

Where the numbers come from (each selectable, and each overridable by data):

* **Refractive index** ``n20``: measured, or Riazi-Daubert's Huang index
  ``I = (n^2 - 1)/(n^2 + 2)`` from ``(Tb, SG)`` -- the Riazi-Daubert (1987)
  six-constant form below 620 K and Riazi's (2005) heavy-fraction form
  above, blended.
* **Hydrogen**: Goossens (1997) from ``(n20, d20, MW)`` (default), or the
  Riazi-Daubert (1987) C/H weight ratio from ``(Tb, SG)``; or measured.
* **Hydrocarbon types**: the API procedure 2B4.1 form of Riazi-Daubert
  (1986) from the refractivity intercept ``Ri = n20 - d20/2``, the C/H ratio
  and ``m = MW (n20 - 1.475)`` (``"riazi_daubert"``, the default), or the
  n-d-M carbon-type analysis of ASTM D3238 (``"ndm"``). Measured PIONA or
  SARA, per cut or as a curve over boiling point, overrides the estimate cut
  by cut.
* **Sulfur and nitrogen classes**: an *illustrative* default split by boiling
  point (:data:`DEFAULT_SULFUR_SPLIT`, :data:`DEFAULT_NITROGEN_SPLIT`) -- the
  shape every crude shows, no particular crude's numbers -- that the caller
  replaces with their own.

The C/H ratio that feeds the type correlation is computed from the hydrogen
content, ``C/H = (1 - H - S - N)/H``, so hydrogen and hydrocarbon type are
estimated from one set of numbers and move together: a measured hydrogen
content moves the estimated aromatics with it.

Correlations used outside the data they were fitted to -- heavy vacuum gas
oil, and always the residue lump -- raise :class:`CompositionRangeWarning`
(when the inputs are concrete). They still return a number, which is what a
differentiable flowsheet needs; the warning is the caller's notice that the
number is an extrapolation.

References (what was checked, and how, is in the docs' references table;
"(unverified)" marks a detail not checked against the source itself):
    Riazi, M.R. and Daubert, T.E., "Prediction of molecular-type analysis of
        petroleum fractions and coal liquids", Ind. Eng. Chem. Process Des.
        Dev. 25(4), 1009-1015 (1986), doi:10.1021/i200035a027 -- the
        molecular-type correlation; API Technical Data Book procedure 2B4.1.
        Coefficients as coded by pychemqt (lib/petro.py, PNA_Riazi); not
        checked against the paper or the TDB (unverified).
    Riazi, M.R. and Daubert, T.E., "Characterization parameters for petroleum
        fractions", Ind. Eng. Chem. Res. 26(4), 755-759 (1987),
        doi:10.1021/ie00064a023 (DOI from pychemqt's reference list,
        unverified) -- the Huang-index and C/H correlations in (Tb, SG),
        Tables X and XI as numbered by pychemqt (unverified).
    Riazi, M.R., Characterization and Properties of Petroleum Fractions, ASTM
        MNL50, ASTM International (2005), doi:10.1520/MNL50-EB -- the
        heavy-fraction Huang index, Eq. 2.46 / Table 2.9 as cited by pychemqt
        (unverified), and the d20-from-SG conversion (equation number
        unverified).
    Goossens, A.G., "Prediction of the hydrogen content of petroleum
        fractions", Ind. Eng. Chem. Res. 36(6), 2500-2504 (1997),
        doi:10.1021/ie960772x (volume, pages and DOI from pychemqt's
        reference list, unverified) -- Eq. 3 as numbered by pychemqt.
    van Nes, K. and van Westen, H.A., Aspects of the Constitution of Mineral
        Oils, Elsevier (1951); ASTM D3238, "Standard Test Method for
        Calculation of Carbon Distribution and Structural Group Analysis of
        Petroleum Oils by the n-d-M Method" (edition unverified) -- the
        n-d-M equations.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow_refinery.thermo import RHO_WATER_60F as _RHO_WATER_60F

jax.config.update("jax_enable_x64", True)

#: Hydrocarbon types, in the column order of every ``hc_type`` array.
HC_TYPES: tuple[str, ...] = ("paraffins", "naphthenes", "aromatics", "olefins")

#: Sulfur classes, in the column order of every ``sulfur_split`` array:
#: thiols and sulfides (aliphatic sulfur, the easiest to remove), thiophenes,
#: benzothiophenes, dibenzothiophenes without beta substitution, and the
#: sterically hindered 4- and 4,6-alkyl dibenzothiophenes (the last ppm of
#: ULSD). Heavier condensed thiophenes (benzonaphthothiophenes) count with
#: the dibenzothiophenes.
SULFUR_CLASSES: tuple[str, ...] = (
    "sulfides", "thiophenes", "benzothiophenes", "dibenzothiophenes", "hindered_dbts")

#: Nitrogen classes, in the column order of every ``nitrogen_split`` array:
#: basic (pyridines, quinolines, acridines -- titratable with perchloric
#: acid, and the FCC and hydrocracker catalyst poisons) and non-basic
#: (pyrroles, indoles, carbazoles).
NITROGEN_CLASSES: tuple[str, ...] = ("basic", "non_basic")

#: Hydrocarbon-type estimation methods.
PNA_METHODS: tuple[str, ...] = ("riazi_daubert", "ndm")

#: Hydrogen-content estimation methods.
HYDROGEN_METHODS: tuple[str, ...] = ("goossens", "riazi_daubert")

#: Keys a measured hydrocarbon-type analysis may give: the four types, or
#: SARA's groups -- ``saturates`` (split into paraffins and naphthenes in the
#: estimate's proportion) and ``resins``/``asphaltenes`` (counted as
#: aromatics, which their cores are).
MEASURED_TYPE_KEYS: tuple[str, ...] = HC_TYPES + ("saturates", "resins", "asphaltenes")

#: Illustrative default sulfur-class split: ``(T_K, shares)``, each row of
#: ``shares`` the fraction of a cut's sulfur in each of
#: :data:`SULFUR_CLASSES` at that boiling point; linear in between, flat
#: beyond. Thiols, sulfides and thiophenes in naphtha; benzothiophene
#: (boils at 221 C) from kerosene; dibenzothiophene (332 C) and the
#: hindered alkyl-DBTs (4,6-DMDBT about 365 C) from the diesel range on, and
#: aliphatic sulfides again in the vacuum range. The SHAPE is the one the
#: hydrodesulfurization literature describes; the numbers are not any crude's
#: and come from no source -- ILLUSTRATIVE, chosen for this module. The
#: boiling points quoted (benzothiophene 221 C, dibenzothiophene 332 C,
#: 4,6-DMDBT about 365 C) are handbook values (unverified here). Pass
#: ``sulfur_split=`` for a real one.
DEFAULT_SULFUR_SPLIT: tuple[tuple[float, ...], tuple[tuple[float, ...], ...]] = (
    (323.15, 423.15, 503.15, 573.15, 623.15, 723.15, 873.15),
    (
        (0.80, 0.20, 0.00, 0.00, 0.00),
        (0.55, 0.45, 0.00, 0.00, 0.00),
        (0.40, 0.25, 0.35, 0.00, 0.00),
        (0.30, 0.05, 0.45, 0.15, 0.05),
        (0.25, 0.00, 0.25, 0.30, 0.20),
        (0.30, 0.00, 0.15, 0.30, 0.25),
        (0.35, 0.00, 0.10, 0.30, 0.25),
    ),
)

#: Illustrative default basic-nitrogen share: ``(T_K, basic_share)``.
#: About a quarter to a third of crude-oil nitrogen titrates as basic, a
#: little more in the heavy end; the rest is pyrrolic. That fraction is a
#: commonly quoted rule of thumb, not taken from a cited source here
#: (unverified); the numbers are ILLUSTRATIVE, like the sulfur split. Pass
#: ``nitrogen_split=`` for a real one.
DEFAULT_NITROGEN_SPLIT: tuple[tuple[float, ...], tuple[float, ...]] = (
    (473.15, 623.15, 823.15),
    (0.25, 0.30, 0.35),
)

# Hydrogen atoms in each light end the characterization may carry (all are
# paraffins; water is not a hydrocarbon and has no type).
_LIGHT_END_H: dict[str, int] = {
    "methane": 4, "ethane": 6, "propane": 8, "isobutane": 10, "n_butane": 10,
    "isopentane": 12, "n_pentane": 12, "n_hexane": 14, "water": 2,
}
_H_MASS = 1.00794

# Ranges outside which a correlation warns (boiling point K, or MW). The
# Riazi-Daubert (1987) (Tb, SG) forms were fitted over about 80-650 F
# (the bounds pychemqt's coding of the same correlation enforces, 300-620 K);
# Riazi's heavy-fraction form is stated for C20-C50 (n-C50 boils at about
# 850 K); procedure 2B4.1's light and heavy branches split at MW 200, and
# above about 600 g/mol it is extrapolated here. n-d-M was developed on heavy
# (lubricating-oil range) fractions and warns below MW 200 here.
_RD1987_TB = (300.0, 620.0)
_RI_TB = (300.0, 850.0)
_PNA_MW = (70.0, 600.0)
_NDM_MW_MIN = 200.0

# Where the two Huang-index correlations, and procedure 2B4.1's two
# branches, are joined: a logistic blend so that a gradient through a cut
# crossing the join stays continuous.
_I_JOIN_TB, _I_JOIN_WIDTH = 620.0, 20.0
_PNA_JOIN_MW, _PNA_JOIN_WIDTH = 200.0, 10.0

# Smooth positive part (t * logaddexp(x/t, 0)) applied before renormalizing
# a correlation's fractions, which can stray below zero at the edges of its
# range. Differs from max(x, 0) by t*ln2 = 7e-4 at x = 0, by < 1e-40 at x > 0.1.
_SIMPLEX_T = 1e-3


class CompositionRangeWarning(UserWarning):
    """A composition correlation was used outside the data it was fitted to."""


# =============================================================================
# The correlations
# =============================================================================


def _rd(a, b, c, d, e, f, T, SG):
    """Riazi-Daubert form ``a exp(b T + c SG + d T SG) T^e SG^f``."""
    return a * jnp.exp(b * T + c * SG + d * T * SG) * T**e * SG**f


def density_20c(SG: Array) -> Array:
    """Liquid density at 20 C (g/cm^3) from SG 60/60 F: ``SG - 4.5e-3 (2.34 - 1.9 SG)``.

    The API Technical Data Book conversion, given in Riazi MNL50 (2005) Ch. 2
    (equation number unverified); the same expression pychemqt uses in
    ``PNA_Riazi``. n-decane: 0.7342 -> 0.72995 against a measured 0.7300.
    """
    SG = jnp.asarray(SG, dtype=float)
    return SG - 4.5e-3 * (2.34 - 1.9 * SG)


def huang_index_rd1987(Tb: Array, SG: Array) -> Array:
    """Huang index ``I = (n^2 - 1)/(n^2 + 2)`` at 20 C, Riazi-Daubert (1987).

    ``I = 2.2657e-2 exp(3.9052e-4 Tb + 2.468316 SG - 5.70425e-4 Tb SG)
    Tb^0.057209 SG^-0.719895`` with Tb in Rankine. Source: Riazi and Daubert,
    Ind. Eng. Chem. Res. 26, 755 (1987), Table X (table number and
    constants as coded by pychemqt, ``prop_Riazi_Daubert`` "I"/"Tb-SG";
    unverified against the paper). Checked here against measured n20 of
    n-decane, benzene and toluene (within 0.004). Fitted for about 300-620 K.
    """
    T = 1.8 * jnp.asarray(Tb, dtype=float)
    return _rd(2.2657e-2, 3.9052e-4, 2.468316, -5.70425e-4, 5.7209e-2, -0.719895,
               T, jnp.asarray(SG, dtype=float))


def huang_index_heavy(Tb: Array, SG: Array) -> Array:
    """Huang index at 20 C for heavy fractions, Riazi (2005), Tb in K.

    ``I = 3.2709e-3 exp(8.4377e-4 Tb + 4.59487 SG - 1.0617e-3 Tb SG)
    Tb^0.03201 SG^-2.34887``. Source: Riazi, ASTM MNL50 (2005), Eq. 2.46 /
    Table 2.9, stated for C20-C50 (equation, table and constants as coded
    by pychemqt, ``prop_Riazi``; unverified against the book). pychemqt
    converts Tb to Rankine for it; the constants only give a physical n20
    with Tb in K (n-C36: 1.458), which is what is coded here -- a
    discrepancy between the two codings, resolved by that check.
    """
    return _rd(3.2709e-3, 8.4377e-4, 4.59487, -1.0617e-3, 0.03201, -2.34887,
               jnp.asarray(Tb, dtype=float), jnp.asarray(SG, dtype=float))


def huang_index(Tb: Array, SG: Array) -> Array:
    """Huang index at 20 C: the 1987 form below 620 K, the heavy one above.

    Joined by a logistic in Tb (scale 20 K, so the ramp runs from about 530
    to 710 K) so that ``n20`` -- and everything estimated from it -- is
    smooth in the boiling point. The two forms do not agree at the join: the
    heavy form is about 0.01 higher in n20 at 620 K on typical gas-oil
    gravities (0.3 wt% in Goossens hydrogen), and a narrow join would put
    that step into one cut.
    """
    w = jax.nn.sigmoid((jnp.asarray(Tb, dtype=float) - _I_JOIN_TB) / _I_JOIN_WIDTH)
    return (1.0 - w) * huang_index_rd1987(Tb, SG) + w * huang_index_heavy(Tb, SG)


def refractive_index_from_huang(I: Array) -> Array:
    """``n = sqrt((1 + 2 I)/(1 - I))``, the inverse of the Huang index."""
    return jnp.sqrt((1.0 + 2.0 * I) / (1.0 - I))


def estimate_refractive_index(Tb: Array, SG: Array) -> Array:
    """Refractive index at 20 C (sodium D line) of a cut from ``(Tb, SG)``."""
    return refractive_index_from_huang(huang_index(Tb, SG))


def refractivity_intercept(n20: Array, d20: Array) -> Array:
    """Kurtz-Ward refractivity intercept ``Ri = n20 - d20/2`` (d20 in g/cm^3)."""
    return n20 - d20 / 2.0


def ch_ratio_rd1987(Tb: Array, SG: Array) -> Array:
    """Carbon-to-hydrogen WEIGHT ratio, Riazi-Daubert (1987), Tb in Rankine.

    ``CH = 17.22022 exp(8.24983e-3 Tb + 16.9402 SG - 6.93931e-3 Tb SG)
    Tb^-2.72522 SG^-6.79769``. Source: Riazi and Daubert, Ind. Eng. Chem.
    Res. 26, 755 (1987), Table XI (as coded by pychemqt, ``prop_Riazi_Daubert``
    "CH"/"Tb-SG"; unverified against the paper). Checked against n-decane's
    formula (0.2 %) and benzene's (6 %). Fitted for about 300-620 K.
    """
    T = 1.8 * jnp.asarray(Tb, dtype=float)
    return _rd(17.22022, 8.24983e-3, 16.9402, -6.93931e-3, -2.72522, -6.79769,
               T, jnp.asarray(SG, dtype=float))


def hydrogen_goossens(n20: Array, d20: Array, MW: Array) -> Array:
    """Hydrogen content (MASS FRACTION) of a fraction, Goossens (1997).

    ``H wt% = 30.346 + (82.952 - 65.341 n20)/d20 - 306/MW``, d20 in g/cm^3.
    Source: Goossens, Ind. Eng. Chem. Res. 36, 2500 (1997), Eq. 3 (as coded
    and numbered by pychemqt, ``H_Goossens``; unverified against the paper).
    Reproduces pychemqt's doctest for n-decane from the paper's Table 1:
    15.45 wt%.
    """
    return (30.346 + (82.952 - 65.341 * n20) / d20 - 306.0 / MW) / 100.0


def riazi_daubert_pna(MW: Array, SG: Array, n20: Array, CH: Array,
                      d20: Array | None = None, blend: bool = True
                      ) -> tuple[Array, Array, Array]:
    """Paraffin, naphthene and aromatic fractions, API procedure 2B4.1 (Ri, CH form).

    Riazi-Daubert (1986), doi:10.1021/i200035a027, as API Technical Data
    Book procedure 2B4.1 gives it when a viscosity is not available
    (constants as coded by pychemqt, ``PNA_Riazi`` with ``CH``; unverified
    against the paper or the TDB):

    * ``MW <= 200``: ``x_P = 2.57 - 2.877 SG + 0.02876 CH``,
      ``x_N = 0.52641 - 0.7494 x_P - 0.021811 m``;
    * ``MW > 200``: ``x_P = 1.9842 - 0.27722 Ri - 0.15643 CH``,
      ``x_N = 0.5977 - 0.761745 Ri + 0.068048 CH``;

    and ``x_A = 1 - x_P - x_N``, with ``Ri = n20 - d20/2``,
    ``m = MW (n20 - 1.475)`` and ``CH`` the carbon/hydrogen weight ratio.
    The published result is a fraction of molecules; for a narrow cut the
    mole, mass and volume fractions nearly coincide, and the module uses it
    as a volume fraction.

    Args:
        MW, SG, n20: Molecular weight, specific gravity and refractive index
            at 20 C.
        CH: Carbon-to-hydrogen weight ratio.
        d20: Density at 20 C (g/cm^3); default :func:`density_20c`.
        blend: Join the branches by a logistic in MW (scale 10 g/mol)
            instead of the published hard switch at 200, so the result is
            smooth in MW; each branch is reproduced to within 1 % of the
            difference between them beyond 200 +/- 46. The branches do not
            agree at 200 -- on typical straight-run cuts the heavy branch
            gives 0.1-0.2 less aromatics -- so the join is a real ramp, not
            a cosmetic one. ``False`` gives the published piecewise form.

    Returns:
        ``(x_P, x_N, x_A)`` as published -- they sum to one by construction
        and may stray outside [0, 1] at the edges of the range (the module
        projects them back; this function does not).
    """
    MW, SG, n20, CH = (jnp.asarray(v, dtype=float) for v in (MW, SG, n20, CH))
    d20 = density_20c(SG) if d20 is None else jnp.asarray(d20, dtype=float)
    Ri = refractivity_intercept(n20, d20)
    m = MW * (n20 - 1.475)
    p_lo = 2.57 - 2.877 * SG + 0.02876 * CH
    n_lo = 0.52641 - 0.7494 * p_lo - 0.021811 * m
    p_hi = 1.9842 - 0.27722 * Ri - 0.15643 * CH
    n_hi = 0.5977 - 0.761745 * Ri + 0.068048 * CH
    if blend:
        w = jax.nn.sigmoid((MW - _PNA_JOIN_MW) / _PNA_JOIN_WIDTH)
    else:
        w = (MW > _PNA_JOIN_MW).astype(float)
    xp = (1.0 - w) * p_lo + w * p_hi
    xn = (1.0 - w) * n_lo + w * n_hi
    return xp, xn, 1.0 - xp - xn


def riazi_daubert_pna_vgc(MW: Array, n20: Array, VGC: Array, d20: Array,
                          blend: bool = True) -> tuple[Array, Array, Array]:
    """PNA fractions from ``Ri`` and the viscosity-gravity constant (2B4.1).

    Riazi-Daubert (1986) / API TDB 2B4.1; constants as coded by pychemqt
    (``PNA_Riazi`` with ``VGC``; unverified against the source), whose
    doctest (M 378 -> 0.606/0.275/0.118) is reproduced in the tests.
    ``x = a + b Ri + c VGC``: for ``MW <= 200`` ``x_P = -13.359 + 14.4591 Ri
    - 1.41344 VGC``, ``x_N = 23.9825 - 23.333 Ri + 0.81517 VGC``; above,
    ``x_P = 2.5737 + 1.0133 Ri - 3.573 VGC``, ``x_N = 2.464 - 3.6701 Ri +
    1.96312 VGC``; ``x_A = 1 - x_P - x_N``. For a caller who has a measured
    viscosity (VGC needs one, which a TBP assay does not give, so the
    module's own estimate uses :func:`riazi_daubert_pna`).
    """
    MW, n20, VGC, d20 = (jnp.asarray(v, dtype=float) for v in (MW, n20, VGC, d20))
    Ri = refractivity_intercept(n20, d20)
    p_lo = -13.359 + 14.4591 * Ri - 1.41344 * VGC
    n_lo = 23.9825 - 23.333 * Ri + 0.81517 * VGC
    p_hi = 2.5737 + 1.0133 * Ri - 3.573 * VGC
    n_hi = 2.464 - 3.6701 * Ri + 1.96312 * VGC
    w = (jax.nn.sigmoid((MW - _PNA_JOIN_MW) / _PNA_JOIN_WIDTH) if blend
         else (MW > _PNA_JOIN_MW).astype(float))
    xp = (1.0 - w) * p_lo + w * p_hi
    xn = (1.0 - w) * n_lo + w * n_hi
    return xp, xn, 1.0 - xp - xn


def ndm(n20: Array, d20: Array, MW: Array, S_wt: Array = 0.0) -> dict[str, Array]:
    """n-d-M carbon-type and ring analysis, ASTM D3238 (van Nes and van Westen, 1951).

    The carbon-type equations agree with pychemqt's independent coding
    (``PNA_van_Nes``); the ring-count equations (``R_A``, ``R_T``) are
    recalled, not checked against D3238 (unverified).

    ``v = 2.51 (n - 1.4750) - (d - 0.8510)``, ``w = (d - 0.8510) - 1.11 (n -
    1.4750)`` at 20 C; ``%C_A = 430 v + 3660/M`` (``v > 0``) or ``670 v +
    3660/M``; ``%C_R = 820 w - 3 S + 10000/M`` (``w > 0``) or ``1440 w - 3 S
    + 10600/M``; ``%C_N = %C_R - %C_A``, ``%C_P = 100 - %C_R``; rings
    ``R_A = 0.44 + 0.055 M v`` (``0.080`` for ``v < 0``), ``R_T = 1.33 +
    0.146 M (w - 0.005 S)`` (``0.180`` for ``w < 0``).

    These are CARBON-type fractions -- the share of carbon atoms in aromatic
    rings, naphthenic rings and paraffinic chains -- not molecule types: an
    alkylbenzene is one aromatic molecule but mostly paraffinic carbon. The
    branches meet continuously at ``v = 0`` and ``w = 0`` with a kink in
    slope there, as published.

    Args:
        n20, d20, MW: Refractive index and density (g/cm^3) at 20 C, MW.
        S_wt: Sulfur, wt%.

    Returns:
        dict ``C_P``, ``C_N``, ``C_A`` (fractions, summing to one), ``R_A``,
        ``R_N``, ``R_T`` (rings per mean molecule).
    """
    n20, d20, MW, S = (jnp.asarray(v, dtype=float) for v in (n20, d20, MW, S_wt))
    v = 2.51 * (n20 - 1.4750) - (d20 - 0.8510)
    w = (d20 - 0.8510) - 1.11 * (n20 - 1.4750)
    CA = jnp.where(v > 0, 430.0, 670.0) * v + 3660.0 / MW
    CR = jnp.where(w > 0, 820.0 * w - 3.0 * S + 10000.0 / MW,
                   1440.0 * w - 3.0 * S + 10600.0 / MW)
    RA = 0.44 + jnp.where(v > 0, 0.055, 0.080) * MW * v
    RT = 1.33 + jnp.where(w > 0, 0.146, 0.180) * MW * (w - 0.005 * S)
    return {"C_P": (100.0 - CR) / 100.0, "C_N": (CR - CA) / 100.0, "C_A": CA / 100.0,
            "R_A": RA, "R_N": RT - RA, "R_T": RT}


def _to_simplex(x: Array) -> Array:
    """Rows onto the probability simplex: smooth positive part, then normalize."""
    pos = _SIMPLEX_T * jnp.logaddexp(x / _SIMPLEX_T, 0.0)
    return pos / jnp.sum(pos, axis=-1, keepdims=True)


# =============================================================================
# Measured data
# =============================================================================


@dataclass(frozen=True)
class CutData:
    """Measured data, per pseudo-component or as a curve over boiling point.

    Attributes:
        values: An array (one property: refractive index, hydrogen wt%) or a
            mapping ``{key: array}`` (hydrocarbon types, keys from
            :data:`MEASURED_TYPE_KEYS`, as fractions 0-1).
        T: ``None``: ``values`` are per pseudo-component (``n_cuts``
            entries, residue lump included), with NaN where nothing was
            measured. Otherwise the boiling points (K) the values were
            measured at -- cut mid-points, strictly increasing -- and each
            cut takes the linear interpolation at its ``Tb``.
        extrapolate: For a curve, what a cut boiling outside ``[T[0],
            T[-1]]`` gets: ``"estimate"`` (the correlation; the default -- a
            naphtha PIONA says nothing about the gas oil) or ``"flat"`` (the
            end value, as the contaminant curves do).
    """

    values: Any
    T: Sequence[float] | Array | None = None
    extrapolate: Literal["estimate", "flat"] = "estimate"

    def __post_init__(self):
        if self.extrapolate not in ("estimate", "flat"):
            raise ValueError(f"extrapolate must be 'estimate' or 'flat', not {self.extrapolate!r}")
        if isinstance(self.values, Mapping):
            bad = sorted(set(self.values) - set(MEASURED_TYPE_KEYS))
            if bad:
                raise ValueError(f"unknown measured types {bad}; known: {', '.join(MEASURED_TYPE_KEYS)}")

    def evaluate(self, Tb: Array, values: Any = None) -> tuple[Array, Array]:
        """``(value, measured)`` at each cut: the value (0 where not
        measured) and a 0/1 float mask. Differentiable in the values and in
        ``Tb``."""
        v = jnp.asarray(self.values if values is None else values, dtype=float)
        if self.T is None:
            if v.shape != Tb.shape:
                raise ValueError(
                    f"per-cut data needs one entry per pseudo-component ({Tb.shape[0]}), "
                    f"got shape {v.shape}; use NaN where a cut was not measured")
            mask = jnp.isfinite(v)
            return jnp.where(mask, v, 0.0), mask.astype(float)
        T = jnp.asarray(self.T, dtype=float)
        if v.shape != T.shape:
            raise ValueError(f"a measured curve needs one value per T; got {v.shape} and {T.shape}")
        val = jnp.interp(Tb, T, v)
        if self.extrapolate == "flat":
            return val, jnp.ones_like(Tb)
        return val, ((Tb >= T[0]) & (Tb <= T[-1])).astype(float)


@dataclass(frozen=True)
class CompositionData:
    """How to estimate a composition, and the measurements that override it.

    Attributes:
        pna_method: :data:`PNA_METHODS`: ``"riazi_daubert"`` (API 2B4.1, the
            default) or ``"ndm"`` (ASTM D3238 carbon types, used as the
            types -- see :func:`ndm`).
        hydrogen_method: :data:`HYDROGEN_METHODS`: ``"goossens"`` (from n20,
            d20 and MW; the default, and the one meant for heavy cuts) or
            ``"riazi_daubert"`` (the 1987 C/H ratio from Tb and SG).
        refractive_index: Measured ``n20`` (:class:`CutData`, a bare array
            per cut, or ``None`` to estimate every cut).
        hydrogen_wt: Measured hydrogen, wt% (:class:`CutData` or per-cut
            array).
        hc_types: Measured PIONA/SARA (:class:`CutData` with a mapping, or a
            bare mapping of per-cut arrays). A cut takes the types measured
            for it and fills the remainder with the unmeasured types in the
            estimate's proportions; then each row is renormalized.
        sulfur_split: ``(T_K, shares)`` with ``shares`` ``(k, 5)`` over
            :data:`SULFUR_CLASSES` (linear in Tb, flat beyond), or an
            ``(n_cuts, 5)`` array. Default :data:`DEFAULT_SULFUR_SPLIT`.
        nitrogen_split: ``(T_K, basic_share)`` with ``(k,)`` shares, or an
            ``(n_cuts,)`` basic share per cut. Default
            :data:`DEFAULT_NITROGEN_SPLIT`.
        warn: Raise :class:`CompositionRangeWarning` for out-of-range use.
    """

    pna_method: str = "riazi_daubert"
    hydrogen_method: str = "goossens"
    refractive_index: Any = None
    hydrogen_wt: Any = None
    hc_types: Any = None
    sulfur_split: Any = None
    nitrogen_split: Any = None
    warn: bool = True

    def __post_init__(self):
        if self.pna_method not in PNA_METHODS:
            raise ValueError(f"pna_method must be one of {PNA_METHODS}, not {self.pna_method!r}")
        if self.hydrogen_method not in HYDROGEN_METHODS:
            raise ValueError(
                f"hydrogen_method must be one of {HYDROGEN_METHODS}, not {self.hydrogen_method!r}")


def _as_cutdata(x) -> CutData | None:
    if x is None or isinstance(x, CutData):
        return x
    return CutData(values=x)


def _split_table(spec, Tb: Array, default, n_classes: int) -> Array:
    """Per-cut class shares ``(n, n_classes)`` from a table or per-cut array."""
    if spec is None:
        spec = default
    if isinstance(spec, tuple) and len(spec) == 2 and np.ndim(spec[0]) == 1:
        T = jnp.asarray(spec[0], dtype=float)
        shares = jnp.asarray(spec[1], dtype=float)
        if n_classes == 2 and shares.ndim == 1:
            shares = jnp.stack([shares, 1.0 - shares], axis=-1)
        if shares.shape != (T.shape[0], n_classes):
            raise ValueError(f"split table needs ({T.shape[0]}, {n_classes}) shares, got {shares.shape}")
        out = jnp.stack([jnp.interp(Tb, T, shares[:, k]) for k in range(n_classes)], axis=-1)
    else:
        out = jnp.asarray(spec, dtype=float)
        if n_classes == 2 and out.ndim == 1:
            out = jnp.stack([out, 1.0 - out], axis=-1)
        if out.shape != (Tb.shape[0], n_classes):
            raise ValueError(
                f"per-cut split needs shape ({Tb.shape[0]}, {n_classes}), got {out.shape}")
    return out / jnp.sum(out, axis=-1, keepdims=True)


def _apply_measured_types(est: Array, Tb: Array, data: CutData) -> Array:
    """Override the estimated types with the measured ones, cut by cut."""
    vals = data.values
    meas = [jnp.zeros_like(Tb) for _ in HC_TYPES]
    mask = [jnp.zeros_like(Tb) for _ in HC_TYPES]
    P, N, A = 0, 1, 2

    def put(k, v, m):
        meas[k] = meas[k] + v * m
        mask[k] = jnp.maximum(mask[k], m)

    for key, arr in vals.items():
        v, m = data.evaluate(Tb, arr)
        if key in HC_TYPES:
            put(HC_TYPES.index(key), v, m)
        elif key == "saturates":
            pn = est[:, P] + est[:, N]
            share = jnp.where(pn > 0, est[:, P] / jnp.where(pn > 0, pn, 1.0), 0.5)
            put(P, v * share, m)
            put(N, v * (1.0 - share), m)
        else:  # resins, asphaltenes: aromatic cores
            put(A, v, m)
    meas = jnp.stack(meas, axis=-1)
    mask = jnp.stack(mask, axis=-1)
    measured_sum = jnp.sum(meas * mask, axis=-1, keepdims=True)
    free = (1.0 - mask) * est
    free_sum = jnp.sum(free, axis=-1, keepdims=True)
    room = jnp.maximum(1.0 - measured_sum, 0.0)
    has_free = free_sum > 1e-300
    scale = jnp.where(has_free, room / jnp.where(has_free, free_sum, 1.0), 0.0)
    x = mask * meas + free * scale
    return x / jnp.sum(x, axis=-1, keepdims=True)


# =============================================================================
# The composition of every component
# =============================================================================


@dataclass(frozen=True)
class Composition:
    """Hydrocarbon type, hydrogen and heteroatom classes of every component.

    Every array has one row per component in :attr:`names` order -- the
    characterization's light ends, then its cuts -- so it lines up with
    ``Characterization.component_MW``, a :class:`ColumnThermo`'s arrays and a
    stream's ``F_<name>`` flows. A registered pytree; differentiable with
    respect to the assay and to any measurement it was given. A conversion
    unit that changes a component's composition builds its own with
    :meth:`replace`.

    Attributes:
        names: The components (static).
        n_light: How many of them are light ends (static).
        pna_method, hydrogen_method: How the estimate was made (static).
        hc_type: ``(n, 4)`` volume fractions of :data:`HC_TYPES`; rows sum
            to one (zero for water).
        hydrogen: ``(n,)`` hydrogen mass fraction.
        sulfur, nitrogen: ``(n,)`` mass fractions (the characterization's).
        sulfur_split: ``(n, 5)`` share of each component's sulfur in each of
            :data:`SULFUR_CLASSES`; rows sum to one.
        nitrogen_split: ``(n, 2)`` share of its nitrogen in
            :data:`NITROGEN_CLASSES`; rows sum to one.
        refractive_index: ``(n,)`` n20 used (estimated or measured; for the
            light ends only the correlation's extrapolation, unused).
        d20: ``(n,)`` density at 20 C, g/cm^3, from SG.
        MW, SG: ``(n,)`` molecular weight and SG 60/60 F -- the averaging
            weights.
    """

    names: tuple[str, ...]
    n_light: int
    pna_method: str
    hydrogen_method: str
    hc_type: Array
    hydrogen: Array
    sulfur: Array
    nitrogen: Array
    sulfur_split: Array
    nitrogen_split: Array
    refractive_index: Array
    d20: Array
    MW: Array
    SG: Array

    # ----- per-component views --------------------------------------------

    @property
    def n(self) -> int:
        return len(self.names)

    @property
    def paraffins(self) -> Array:
        return self.hc_type[:, 0]

    @property
    def naphthenes(self) -> Array:
        return self.hc_type[:, 1]

    @property
    def aromatics(self) -> Array:
        return self.hc_type[:, 2]

    @property
    def olefins(self) -> Array:
        return self.hc_type[:, 3]

    @property
    def sulfur_classes(self) -> Array:
        """``(n, 5)`` mass fraction of each component that is sulfur of each class."""
        return self.sulfur[:, None] * self.sulfur_split

    @property
    def nitrogen_classes(self) -> Array:
        """``(n, 2)`` mass fraction of each component that is basic / non-basic N."""
        return self.nitrogen[:, None] * self.nitrogen_split

    @property
    def carbon(self) -> Array:
        """``(n,)`` carbon mass fraction, ``1 - H - S - N`` (O and metals neglected)."""
        return 1.0 - self.hydrogen - self.sulfur - self.nitrogen

    @property
    def ch_ratio(self) -> Array:
        """``(n,)`` carbon-to-hydrogen weight ratio."""
        return self.carbon / self.hydrogen

    def replace(self, **changes) -> "Composition":
        """A copy with some arrays replaced (``dataclasses.replace``)."""
        return dataclasses.replace(self, **changes)

    # ----- a stream of these components -----------------------------------

    def of_flows(self, flows: Array, basis: Literal["mole", "mass"] = "mole") -> "StreamComposition":
        """The composition of a mixture of these components.

        Types are averaged by standard liquid volume (``m_i / SG_i``, ideal
        mixing), hydrogen, sulfur and nitrogen and their classes by mass.

        Args:
            flows: ``(n,)`` component flows in :attr:`names` order: mol/s
                (``basis="mole"``) or kg/s (``basis="mass"``).
        """
        if basis not in ("mole", "mass"):
            raise ValueError(f"basis must be 'mole' or 'mass', not {basis!r}")
        flows = jnp.asarray(flows, dtype=float)
        mass = flows * self.MW / 1000.0 if basis == "mole" else flows
        volume = mass / (self.SG * _RHO_WATER_60F)
        M, V = jnp.sum(mass), jnp.sum(volume)
        w, phi = mass / M, volume / V
        return StreamComposition(
            mass=M, volume=V,
            hc_type=phi @ self.hc_type,
            hydrogen_wt=100.0 * jnp.sum(w * self.hydrogen),
            sulfur_wt=100.0 * jnp.sum(w * self.sulfur),
            nitrogen_wppm=1e6 * jnp.sum(w * self.nitrogen),
            sulfur_classes_wt=100.0 * (w @ self.sulfur_classes),
            nitrogen_classes_wppm=1e6 * (w @ self.nitrogen_classes),
        )

    def flows_of(self, stream: Mapping[str, Any]) -> Array:
        """``(n,)`` molar flows of :attr:`names` in a difflow stream.

        Missing components count as zero; water (``F_water``, ``F_H2O``)
        and any other species not in :attr:`names` are ignored -- they carry
        no hydrocarbon composition.
        """
        zero = jnp.asarray(0.0)
        return jnp.stack([jnp.asarray(stream.get(f"F_{n}", zero), dtype=float) for n in self.names])

    def of_stream(self, stream: Mapping[str, Any]) -> "StreamComposition":
        """The composition of a difflow stream of these components (see :meth:`of_flows`)."""
        return self.of_flows(self.flows_of(stream))

    def blend_qualities(self) -> dict[str, Array]:
        """The types as :class:`BlendCharacterization` quality vectors, vol%."""
        return {f"{t}_vol": 100.0 * self.hc_type[:, k] for k, t in enumerate(HC_TYPES)}

    def table(self) -> str:
        """Per-component composition as fixed-width text."""
        head = (f"{'name':<10} {'P':>6} {'N':>6} {'A':>6} {'O':>6} {'H wt%':>6} "
                f"{'n20':>6} {'S wt%':>6} " + " ".join(f"{c[:6]:>6}" for c in SULFUR_CLASSES)
                + f" {'N ppm':>7} {'basic':>6}")
        rows = [head]
        a = {k: np.asarray(getattr(self, k)) for k in (
            "hc_type", "hydrogen", "refractive_index", "sulfur", "sulfur_split",
            "nitrogen", "nitrogen_split")}
        for i, name in enumerate(self.names):
            rows.append(
                f"{name:<10} " + " ".join(f"{v:6.3f}" for v in a["hc_type"][i])
                + f" {100 * a['hydrogen'][i]:6.2f} {a['refractive_index'][i]:6.4f}"
                f" {100 * a['sulfur'][i]:6.3f} " + " ".join(f"{v:6.3f}" for v in a["sulfur_split"][i])
                + f" {1e6 * a['nitrogen'][i]:7.0f} {a['nitrogen_split'][i][0]:6.3f}")
        return "\n".join(rows)


jax.tree_util.register_dataclass(
    Composition,
    data_fields=["hc_type", "hydrogen", "sulfur", "nitrogen", "sulfur_split", "nitrogen_split",
                 "refractive_index", "d20", "MW", "SG"],
    meta_fields=["names", "n_light", "pna_method", "hydrogen_method"],
)

@dataclass(frozen=True)
class StreamComposition:
    """The composition of one stream or product.

    Attributes:
        mass: Hydrocarbon mass flow (kg/s for flows in mol/s).
        volume: Standard liquid volume flow (m^3/s at 60 F).
        hc_type: ``(4,)`` volume fractions of :data:`HC_TYPES`, summing to one.
        hydrogen_wt, sulfur_wt: wt%.
        nitrogen_wppm: wppm.
        sulfur_classes_wt: ``(5,)`` wt% of the stream that is sulfur of each
            of :data:`SULFUR_CLASSES`; sums to ``sulfur_wt``.
        nitrogen_classes_wppm: ``(2,)`` basic and non-basic nitrogen, wppm;
            sums to ``nitrogen_wppm``.
    """

    mass: Array
    volume: Array
    hc_type: Array
    hydrogen_wt: Array
    sulfur_wt: Array
    nitrogen_wppm: Array
    sulfur_classes_wt: Array
    nitrogen_classes_wppm: Array

    @property
    def paraffins(self) -> Array:
        return self.hc_type[0]

    @property
    def naphthenes(self) -> Array:
        return self.hc_type[1]

    @property
    def aromatics(self) -> Array:
        return self.hc_type[2]

    @property
    def olefins(self) -> Array:
        return self.hc_type[3]

    def as_dict(self) -> dict[str, Array]:
        """Flat ``{name: value}``: ``paraffins_vol`` ... (vol%), ``hydrogen_wt``,
        ``sulfur_wt``, ``S_<class>_wt``, ``nitrogen_wppm``, ``N_<class>_wppm``."""
        out = {f"{t}_vol": 100.0 * self.hc_type[k] for k, t in enumerate(HC_TYPES)}
        out.update(hydrogen_wt=self.hydrogen_wt, sulfur_wt=self.sulfur_wt,
                   nitrogen_wppm=self.nitrogen_wppm)
        out.update({f"S_{c}_wt": self.sulfur_classes_wt[k] for k, c in enumerate(SULFUR_CLASSES)})
        out.update({f"N_{c}_wppm": self.nitrogen_classes_wppm[k]
                    for k, c in enumerate(NITROGEN_CLASSES)})
        return out


jax.tree_util.register_dataclass(
    StreamComposition,
    data_fields=["mass", "volume", "hc_type", "hydrogen_wt", "sulfur_wt", "nitrogen_wppm",
                 "sulfur_classes_wt", "nitrogen_classes_wppm"],
    meta_fields=[],
)


# =============================================================================
# Estimation
# =============================================================================


def _concrete(x) -> np.ndarray | None:
    try:
        return np.asarray(x)
    except (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError):
        return None


def _check_ranges(char, Tb, MW, n_measured, h_measured, t_measured, data: CompositionData):
    """Warn (once, listing cuts) where a correlation is extrapolated."""
    vals = [_concrete(a) for a in (Tb, MW, n_measured, h_measured, t_measured)]
    if any(v is None for v in vals):
        return
    Tb, MW, n_m, h_m, t_m = vals
    names = char.pseudo_names
    problems = []

    def note(what, bad):
        if np.any(bad):
            problems.append(f"{what}: {', '.join(n for n, b in zip(names, bad) if b)}")

    est_n = n_m < 0.5
    note(f"refractive index (Riazi-Daubert Huang index, fitted ~{_RI_TB[0]:.0f}-{_RI_TB[1]:.0f} K)",
         est_n & ((Tb < _RI_TB[0]) | (Tb > _RI_TB[1])))
    if data.hydrogen_method == "riazi_daubert":
        note(f"C/H ratio (Riazi-Daubert 1987, fitted ~{_RD1987_TB[0]:.0f}-{_RD1987_TB[1]:.0f} K)",
             (h_m < 0.5) & ((Tb < _RD1987_TB[0]) | (Tb > _RD1987_TB[1])))
    est_t = t_m < 0.5
    if data.pna_method == "riazi_daubert":
        note(f"hydrocarbon types (API 2B4.1, MW ~{_PNA_MW[0]:.0f}-{_PNA_MW[1]:.0f})",
             est_t & ((MW < _PNA_MW[0]) | (MW > _PNA_MW[1])))
    else:
        note(f"hydrocarbon types (n-d-M, heavy fractions, MW > {_NDM_MW_MIN:.0f})",
             est_t & (MW < _NDM_MW_MIN))
    if char.residue_lump and (est_n[-1] or est_t[-1] or h_m[-1] < 0.5):
        problems.append(f"{names[-1]} is the residue lump: its Tb and MW are set, not "
                        "correlated, and every composition correlation is extrapolated there")
    if problems:
        warnings.warn("composition correlations used outside their fitted range -- "
                      + "; ".join(problems) + ". Measured data (CutData) overrides them.",
                      CompositionRangeWarning, stacklevel=4)


def estimate_composition(char, data: CompositionData | None = None, **kwargs) -> Composition:
    """The :class:`Composition` of every component of a characterization.

    Light ends are what they are: paraffins with their exact hydrogen
    content, no sulfur or nitrogen. Each cut is estimated (see the module
    docstring for the chain: n20, then hydrogen, then C/H, then types) and
    then overridden by whatever was measured for it.

    Args:
        char: A :class:`~difflow_refinery.assay.Characterization`.
        data: A :class:`CompositionData`; or give its fields as keyword
            arguments.

    Differentiable with respect to the assay behind ``char`` and to every
    number in ``data``; traceable (the range warnings need concrete inputs
    and are skipped under a trace).
    """
    if data is None:
        data = CompositionData(**kwargs)
    elif kwargs:
        data = dataclasses.replace(data, **kwargs)
    k = len(char.light_names)
    Tb, SG, MW = char.Tb, char.SG, char.MW
    S_cut = jnp.asarray(char.sulfur)[k:]
    N_cut = jnp.asarray(char.nitrogen)[k:]
    d20 = density_20c(SG)

    # refractive index
    n_est = estimate_refractive_index(Tb, SG)
    ri = _as_cutdata(data.refractive_index)
    if ri is not None:
        v, m_n = ri.evaluate(Tb)
        n20 = jnp.where(m_n > 0, v, n_est)
    else:
        n20, m_n = n_est, jnp.zeros_like(Tb)

    # hydrogen
    if data.hydrogen_method == "goossens":
        H_est = hydrogen_goossens(n20, d20, MW)
    else:
        H_est = (1.0 - S_cut - N_cut) / (1.0 + ch_ratio_rd1987(Tb, SG))
    hd = _as_cutdata(data.hydrogen_wt)
    if hd is not None:
        v, m_h = hd.evaluate(Tb)
        H = jnp.where(m_h > 0, v / 100.0, H_est)
    else:
        H, m_h = H_est, jnp.zeros_like(Tb)
    CH = (1.0 - H - S_cut - N_cut) / H

    # hydrocarbon types
    if data.pna_method == "riazi_daubert":
        xp, xn, xa = riazi_daubert_pna(MW, SG, n20, CH, d20)
    else:
        c = ndm(n20, d20, MW, 100.0 * S_cut)
        xp, xn, xa = c["C_P"], c["C_N"], c["C_A"]
    # straight run: no olefins (exactly), the PNA projected onto the simplex
    est = jnp.concatenate([_to_simplex(jnp.stack([xp, xn, xa], axis=-1)),
                           jnp.zeros_like(xp)[:, None]], axis=-1)
    ht = data.hc_types
    if ht is not None:
        ht = ht if isinstance(ht, CutData) else CutData(values=ht)
        if not isinstance(ht.values, Mapping):
            raise ValueError("hc_types needs a mapping {type: values}")
        types = _apply_measured_types(est, Tb, ht)
        m_t = jnp.max(jnp.stack([ht.evaluate(Tb, a)[1] for a in ht.values.values()]), axis=0)
    else:
        types, m_t = est, jnp.zeros_like(Tb)

    if data.warn:
        _check_ranges(char, Tb, MW, m_n, m_h, m_t, data)

    s_split = _split_table(data.sulfur_split, Tb, DEFAULT_SULFUR_SPLIT, len(SULFUR_CLASSES))
    n_split = _split_table(data.nitrogen_split, Tb, DEFAULT_NITROGEN_SPLIT, len(NITROGEN_CLASSES))

    # light ends: paraffins, exact hydrogen, no heteroatoms
    le_mw = jnp.asarray(char.component_MW)[:k]
    le_h = jnp.asarray([_LIGHT_END_H[n] * _H_MASS for n in char.light_names], dtype=float).reshape(-1)
    le_types = jnp.asarray([[0.0, 0.0, 0.0, 0.0] if n == "water" else [1.0, 0.0, 0.0, 0.0]
                            for n in char.light_names], dtype=float).reshape(-1, 4)
    le_sg = jnp.asarray(char.component_SG)[:k]
    le_s = jnp.broadcast_to(jnp.asarray(DEFAULT_SULFUR_SPLIT[1][0], dtype=float), (k, 5))
    le_n = jnp.broadcast_to(jnp.asarray([0.0, 1.0]), (k, 2))
    le_n20 = estimate_refractive_index(jnp.full((k,), 300.0), le_sg)

    def cat(a, b):
        return jnp.concatenate([a, b], axis=0)

    return Composition(
        names=tuple(char.names),
        n_light=k,
        pna_method=data.pna_method,
        hydrogen_method=data.hydrogen_method,
        hc_type=cat(le_types, types),
        hydrogen=cat(le_h / le_mw, H),
        sulfur=jnp.asarray(char.sulfur, dtype=float),
        nitrogen=jnp.asarray(char.nitrogen, dtype=float),
        sulfur_split=cat(le_s, s_split),
        nitrogen_split=cat(le_n, n_split),
        refractive_index=cat(le_n20, n20),
        d20=cat(density_20c(le_sg), d20),
        MW=jnp.asarray(char.component_MW, dtype=float),
        SG=jnp.asarray(char.component_SG, dtype=float),
    )


__all__ = [
    "HC_TYPES", "SULFUR_CLASSES", "NITROGEN_CLASSES", "PNA_METHODS", "HYDROGEN_METHODS",
    "MEASURED_TYPE_KEYS", "DEFAULT_SULFUR_SPLIT", "DEFAULT_NITROGEN_SPLIT",
    "CompositionRangeWarning", "CutData", "CompositionData", "Composition",
    "StreamComposition", "estimate_composition",
    "density_20c", "huang_index", "huang_index_rd1987", "huang_index_heavy",
    "refractive_index_from_huang", "estimate_refractive_index", "refractivity_intercept",
    "ch_ratio_rd1987", "hydrogen_goossens", "riazi_daubert_pna", "riazi_daubert_pna_vgc", "ndm",
]
