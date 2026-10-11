"""Binary interaction parameters for NRTL and UNIQUAC, with sources.

A small, curated database of activity-coefficient parameters for common
binaries, and lookups that assemble :class:`~difflow.units.lle.NRTLParams`
or :class:`~difflow.units.lle.UNIQUACParams` for any list of species.

Conventions (identical to the parameter classes, so nothing is converted
at use):

- **NRTL**: ``tau_ij = a_ij + b_ij / T`` (``b`` in K), ``G_ij =
  exp(-alpha_ij tau_ij)``, ``alpha_ij = alpha_ji``. The DECHEMA tables print
  ``A_ij = g_ij - g_jj`` in cal/mol, so ``a_ij = 0`` and ``b_ij = A_ij / R``
  with ``R = 1.98720 cal/(mol K)``. The published ``A_ij`` is what is
  stored; :func:`get_nrtl_params` divides by ``R``.
- **UNIQUAC**: ``tau_ij = exp(-(a_ij + b_ij / T))``, ``a_ij = 0``, ``b_ij =
  (u_ij - u_jj) / R`` in K; ``r``, ``q`` per species; coordination number
  ``z = 10``. The ``b`` values are stored in K.

Index order: ``tau_ij`` is the parameter of species ``i`` *in* the
expression ``ln gamma_i = ... tau_ji G_ji ...``, the usual Renon-Prausnitz /
Abrams-Prausnitz convention that :func:`~difflow.units.lle.nrtl_activity_coefficients`
implements; a pair stored as ``(i, j)`` holds ``(A_ij, A_ji)``.

Sources
-------
NRTL: Gmehling, Onken et al., *Vapor-Liquid Equilibrium Data Collection*,
DECHEMA Chemistry Data Series Vol. I (Frankfurt, 1977 onward); the part
and page of each fit is given per pair (``"I/2c p. 78"`` = Vol. I,
Part 2c, page 78). The numbers were read from the transcription in the
ChemSep interaction-parameter file ``nrtl.ipd`` ("DECHEMA NRTL data @
1 atm", Kooijman & Taylor, Artistic License 2.0), as distributed with the
``thermo`` package (C. Bell, v0.6.1).

UNIQUAC: the ChemSep UNIQUAC interaction parameters (DECHEMA-regressed fits
at 1 atm, Kooijman & Taylor), as distributed with ``thermo`` v0.6.1
(``ChemSep/uniquac.json``, which stores ``-b``); the ChemSep file gives no
DECHEMA page numbers. ``r`` and ``q`` are the Abrams-Prausnitz structural
parameters, Bondi van der Waals volume and area over the standard segment
(``r = V_w / 15.17 cm^3/mol``, ``q = A_w / 2.5e9 cm^2/mol``), as printed
with the DECHEMA UNIQUAC fits (Abrams & Prausnitz, AIChE J. 21, 116
(1975); Anderson & Prausnitz, Ind. Eng. Chem. Process Des. Dev. 17, 552
(1978)).

Verification
------------
A parameter set is listed here only if, with the database's own Antoine
constants, it reproduces a physical fact that does not depend on it (the
tests in ``tests/test_binary_params.py``): the 1-atm azeotrope composition
and temperature from the CRC Handbook azeotrope tables, the absence of an
azeotrope where there is none, the maximum-boiling azeotrope of
acetone-chloroform, near-unity activity coefficients for benzene-toluene.
Sets that failed were dropped: the DECHEMA NRTL acetone-water set predicts
a spurious azeotrope at 98 mol% acetone, and the ChemSep UNIQUAC
acetic acid-water set (vapour dimerization ignored) one at 86 mol% water.
No set was found for isopropanol-water in NRTL form (UNIQUAC has one).

Missing pairs
-------------
:func:`get_nrtl_params` and :func:`get_uniquac_params` raise ``KeyError``
for a pair not in the database (``missing="raise"``, the default). With
``missing="ideal"`` the pair's interaction parameters are zero (NRTL:
``tau = 0``, ``gamma = 1`` for that pair; UNIQUAC: ``tau = 1``, so only the
combinatorial size term remains) and a :class:`MissingBinaryParametersWarning`
names the pairs, so an ideal stand-in is never silent.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple, Sequence

import jax.numpy as jnp

from difflow.units.lle import NRTLParams, UNIQUACParams

#: Gas constant in cal/(mol K): the DECHEMA NRTL A_ij are in cal/mol.
R_CAL = 1.98720425864083


class MissingBinaryParametersWarning(UserWarning):
    """A binary pair had no parameters and was set ideal (``missing="ideal"``)."""


class NRTLBinary(NamedTuple):
    """One published NRTL binary fit (``i``, ``j`` as the source orders them).

    Attributes:
        A_ij: ``g_ij - g_jj`` (cal/mol); ``tau_ij = A_ij / (R T)``.
        A_ji: ``g_ji - g_ii`` (cal/mol).
        alpha: Non-randomness ``alpha_ij = alpha_ji``.
        source: Citation (DECHEMA volume/part/page).
    """
    A_ij: float
    A_ji: float
    alpha: float
    source: str


class UNIQUACBinary(NamedTuple):
    """One UNIQUAC binary fit, already in K.

    Attributes:
        b_ij: ``(u_ij - u_jj) / R`` (K); ``tau_ij = exp(-b_ij / T)``.
        b_ji: ``(u_ji - u_ii) / R`` (K).
        source: Citation.
    """
    b_ij: float
    b_ji: float
    source: str


_DECHEMA = "Gmehling & Onken, DECHEMA Chemistry Data Series Vol. "
_VIA_CHEMSEP = " (via ChemSep nrtl.ipd, as in thermo 0.6.1)"

#: NRTL binaries, key ``(i, j)`` in database species names.
NRTL_BINARY_PARAMETERS: dict[tuple[str, str], NRTLBinary] = {
    ("ethanol", "water"): NRTLBinary(
        -57.9601, 1241.7396, 0.2937,
        _DECHEMA + "I (aqueous-organic systems), DECHEMA NRTL fit at 1 atm; "
        "page not recorded in the ChemSep transcription" + _VIA_CHEMSEP),
    ("methanol", "water"): NRTLBinary(
        -189.0469, 792.8020, 0.2999, _DECHEMA + "I/1a p. 61" + _VIA_CHEMSEP),
    ("benzene", "toluene"): NRTLBinary(
        60.1980, -51.0865, 0.3019, _DECHEMA + "I/7 p. 282" + _VIA_CHEMSEP),
    ("ethyl_acetate", "ethanol"): NRTLBinary(
        305.6041, 330.5105, 0.2988, _DECHEMA + "I/2c p. 338" + _VIA_CHEMSEP),
    ("water", "1_butanol"): NRTLBinary(
        2633.6951, 504.0381, 0.4447, _DECHEMA + "I/1a p. 336" + _VIA_CHEMSEP),
    ("acetone", "chloroform"): NRTLBinary(
        -651.1909, 301.8389, 0.3054, _DECHEMA + "I/3+4 p. 92" + _VIA_CHEMSEP),
    ("acetone", "methanol"): NRTLBinary(
        184.2662, 226.5580, 0.3009, _DECHEMA + "I/2c p. 78" + _VIA_CHEMSEP),
    ("ethyl_acetate", "water"): NRTLBinary(
        1285.9880, 1606.0820, 0.4393, _DECHEMA + "I/1a p. 304" + _VIA_CHEMSEP),
    ("ethanol", "benzene"): NRTLBinary(
        516.1410, 1065.9086, 0.4774, _DECHEMA + "I/2c p. 406" + _VIA_CHEMSEP),
    ("methanol", "benzene"): NRTLBinary(
        721.6136, 1158.5131, 0.4694, _DECHEMA + "I/2c p. 191" + _VIA_CHEMSEP),
    ("ethanol", "cyclohexane"): NRTLBinary(
        876.7933, 1390.4162, 0.4485, _DECHEMA + "I/2c p. 419" + _VIA_CHEMSEP),
    ("acetone", "benzene"): NRTLBinary(
        -396.4935, 886.5703, 0.2971, _DECHEMA + "I/3+4 p. 195" + _VIA_CHEMSEP),
    ("methanol", "ethanol"): NRTLBinary(
        -327.9991, 376.2667, 0.3057, _DECHEMA + "I/2c p. 60" + _VIA_CHEMSEP),
    ("methanol", "1_butanol"): NRTLBinary(
        793.8173, -486.3299, 0.2483, _DECHEMA + "I/2c p. 127" + _VIA_CHEMSEP),
    ("ethanol", "2_propanol"): NRTLBinary(
        690.1392, -529.3472, 0.3125, _DECHEMA + "I/2c p. 317" + _VIA_CHEMSEP),
}

_CHEMSEP_UQ = ("ChemSep UNIQUAC parameters (DECHEMA-regressed, 1 atm; "
               "Kooijman & Taylor), as in thermo 0.6.1 ChemSep/uniquac.json")

#: UNIQUAC binaries, key ``(i, j)``; ``b`` in K, difflow's sign convention.
UNIQUAC_BINARY_PARAMETERS: dict[tuple[str, str], UNIQUACBinary] = {
    ("water", "ethanol"): UNIQUACBinary(55.2881, 87.4601, _CHEMSEP_UQ),
    ("water", "methanol"): UNIQUACBinary(276.4164, -169.6503, _CHEMSEP_UQ),
    ("water", "2_propanol"): UNIQUACBinary(32.4279, 164.7757, _CHEMSEP_UQ),
    ("water", "1_propanol"): UNIQUACBinary(146.2124, 95.9110, _CHEMSEP_UQ),
    ("water", "acetone"): UNIQUACBinary(-55.5468, 351.6493, _CHEMSEP_UQ),
    ("methanol", "acetone"): UNIQUACBinary(-42.3894, 203.2264, _CHEMSEP_UQ),
    ("methanol", "benzene"): UNIQUACBinary(-38.5578, 587.2208, _CHEMSEP_UQ),
    ("ethanol", "benzene"): UNIQUACBinary(-53.0037, 385.6958, _CHEMSEP_UQ),
    ("methanol", "toluene"): UNIQUACBinary(-32.8470, 604.5238, _CHEMSEP_UQ),
    ("ethanol", "toluene"): UNIQUACBinary(-74.0288, 441.8839, _CHEMSEP_UQ),
    ("methanol", "chloroform"): UNIQUACBinary(-136.6045, 651.8848, _CHEMSEP_UQ),
    ("ethanol", "chloroform"): UNIQUACBinary(-168.8052, 494.0437, _CHEMSEP_UQ),
    ("benzene", "toluene"): UNIQUACBinary(35.5889, -29.3386, _CHEMSEP_UQ),
    ("acetone", "benzene"): UNIQUACBinary(-180.6169, 304.0855, _CHEMSEP_UQ),
    ("ethanol", "n_hexane"): UNIQUACBinary(-68.2106, 531.8516, _CHEMSEP_UQ),
    ("methanol", "ethanol"): UNIQUACBinary(-101.7196, 130.1793, _CHEMSEP_UQ),
}

#: UNIQUAC structural parameters ``(r, q)`` (see the module docstring).
UNIQUAC_RQ: dict[str, tuple[float, float]] = {
    "water": (0.9200, 1.4000),
    "methanol": (1.4311, 1.4320),
    "ethanol": (2.1055, 1.9720),
    "1_propanol": (2.7799, 2.5120),
    "2_propanol": (2.7791, 2.5080),
    "acetone": (2.5735, 2.3360),
    "benzene": (3.1878, 2.4000),
    "toluene": (3.9228, 2.9680),
    "chloroform": (2.8700, 2.4100),
    "n_hexane": (4.4998, 3.8560),
}

#: alpha used for a pair set ideal by ``missing="ideal"`` (it multiplies a
#: zero tau, so its value does not matter; 0.3 keeps the matrix ordinary).
_IDEAL_ALPHA = 0.3


def _canonical(name: str) -> str:
    from difflow.database import resolve_alias
    return resolve_alias(name)


def _check_missing(missing: str) -> None:
    if missing not in ("raise", "ideal"):
        raise ValueError(f"missing must be 'raise' or 'ideal', not {missing!r}")


def _lookup(table: dict, a: str, b: str):
    """``(entry, flipped)`` for the pair, or ``(None, False)``."""
    if (a, b) in table:
        return table[(a, b)], False
    if (b, a) in table:
        return table[(b, a)], True
    return None, False


def _report_missing(model: str, absent: list, missing: str) -> None:
    if not absent:
        return
    pairs = ", ".join(f"{a}/{b}" for a, b in absent)
    if missing == "raise":
        raise KeyError(
            f"no {model} binary parameters for {pairs}; pass missing='ideal' "
            f"to treat them as ideal, or build the parameters by hand")
    warnings.warn(f"no {model} binary parameters for {pairs}: set ideal "
                  "(zero interaction)", MissingBinaryParametersWarning,
                  stacklevel=3)


def get_nrtl_params(species: Sequence[str], missing: str = "raise") -> NRTLParams:
    """Assemble :class:`NRTLParams` for ``species`` from the database.

    Args:
        species: Species names (database names or aliases), in the order of
            the mole-fraction vector the model will be called with.
        missing: ``"raise"`` (default): ``KeyError`` naming every pair with
            no parameters. ``"ideal"``: such pairs get ``tau = 0`` and a
            :class:`MissingBinaryParametersWarning`.

    Returns:
        ``NRTLParams`` with ``a = 0``, ``b_ij = A_ij / R`` (K) and the
        published ``alpha``; ``species`` as given.

    Raises:
        KeyError: A pair is missing and ``missing="raise"``.
        ValueError: Fewer than two species, a repeated species, or an
            unknown ``missing``.

    Example:
        >>> import jax.numpy as jnp
        >>> p = get_nrtl_params(["ethanol", "water"])
        >>> g = p.gamma(jnp.array([0.5, 0.5]), 350.0)
        >>> bool(g[0] > 1.0 and g[1] > 1.0)
        True
    """
    _check_missing(missing)
    names = tuple(species)
    keys = [_canonical(s) for s in names]
    if len(keys) < 2 or len(set(keys)) != len(keys):
        raise ValueError(f"need two or more distinct species, got {names}")
    n = len(keys)
    b = [[0.0] * n for _ in range(n)]
    alpha = [[0.0] * n for _ in range(n)]
    absent = []
    for i in range(n):
        for j in range(i + 1, n):
            e, flipped = _lookup(NRTL_BINARY_PARAMETERS, keys[i], keys[j])
            if e is None:
                absent.append((keys[i], keys[j]))
                alpha[i][j] = alpha[j][i] = _IDEAL_ALPHA
                continue
            A_ij, A_ji = (e.A_ji, e.A_ij) if flipped else (e.A_ij, e.A_ji)
            b[i][j], b[j][i] = A_ij / R_CAL, A_ji / R_CAL
            alpha[i][j] = alpha[j][i] = e.alpha
    _report_missing("NRTL", absent, missing)
    return NRTLParams(species=names, a=jnp.zeros((n, n)), b=jnp.array(b),
                      alpha=jnp.array(alpha))


def get_uniquac_params(species: Sequence[str], missing: str = "raise",
                       z: float = 10.0) -> UNIQUACParams:
    """Assemble :class:`UNIQUACParams` for ``species`` from the database.

    Every species needs ``r`` and ``q`` (:data:`UNIQUAC_RQ`); a species
    without them raises ``KeyError`` whatever ``missing`` says, as there is
    no ideal stand-in for a molecule's size.

    Args:
        species: Species names, in mole-fraction order.
        missing: ``"raise"`` (default) or ``"ideal"`` for absent pairs, as
            in :func:`get_nrtl_params` (ideal here: ``b = 0``, ``tau = 1``).
        z: Coordination number (10, the value the parameters were fitted
            with).

    Returns:
        ``UNIQUACParams`` with ``a = 0`` and ``b`` in K.

    Raises:
        KeyError: Missing ``r``/``q``, or a missing pair with
            ``missing="raise"``.
        ValueError: As for :func:`get_nrtl_params`.

    Example:
        >>> import jax.numpy as jnp
        >>> p = get_uniquac_params(["water", "2_propanol"])
        >>> bool(p.gamma(jnp.array([0.5, 0.5]), 353.0)[1] > 1.0)
        True
    """
    _check_missing(missing)
    names = tuple(species)
    keys = [_canonical(s) for s in names]
    if len(keys) < 2 or len(set(keys)) != len(keys):
        raise ValueError(f"need two or more distinct species, got {names}")
    no_rq = [k for k in keys if k not in UNIQUAC_RQ]
    if no_rq:
        raise KeyError(f"no UNIQUAC r, q for {', '.join(no_rq)}; available: "
                       f"{', '.join(sorted(UNIQUAC_RQ))}")
    n = len(keys)
    b = [[0.0] * n for _ in range(n)]
    absent = []
    for i in range(n):
        for j in range(i + 1, n):
            e, flipped = _lookup(UNIQUAC_BINARY_PARAMETERS, keys[i], keys[j])
            if e is None:
                absent.append((keys[i], keys[j]))
                continue
            b[i][j], b[j][i] = (e.b_ji, e.b_ij) if flipped else (e.b_ij, e.b_ji)
    _report_missing("UNIQUAC", absent, missing)
    r = jnp.array([UNIQUAC_RQ[k][0] for k in keys])
    q = jnp.array([UNIQUAC_RQ[k][1] for k in keys])
    return UNIQUACParams(species=names, r=r, q=q, a=jnp.zeros((n, n)),
                         b=jnp.array(b), z=z)


def list_binary_pairs(model: str = "nrtl") -> list[tuple[str, str]]:
    """The pairs with parameters for ``model`` (``"nrtl"`` or ``"uniquac"``).

    Example:
        >>> ("ethanol", "water") in list_binary_pairs("nrtl")
        True
    """
    tables = {"nrtl": NRTL_BINARY_PARAMETERS,
              "uniquac": UNIQUAC_BINARY_PARAMETERS}
    if model not in tables:
        raise ValueError(f"model must be 'nrtl' or 'uniquac', not {model!r}")
    return sorted(tables[model])


def binary_parameter_source(model: str, i: str, j: str) -> str:
    """Citation for the ``(i, j)`` pair's parameters under ``model``.

    Example:
        >>> "DECHEMA" in binary_parameter_source("nrtl", "water", "methanol")
        True
    """
    tables = {"nrtl": NRTL_BINARY_PARAMETERS,
              "uniquac": UNIQUAC_BINARY_PARAMETERS}
    if model not in tables:
        raise ValueError(f"model must be 'nrtl' or 'uniquac', not {model!r}")
    e, _ = _lookup(tables[model], _canonical(i), _canonical(j))
    if e is None:
        raise KeyError(f"no {model} parameters for {i}/{j}")
    return e.source
