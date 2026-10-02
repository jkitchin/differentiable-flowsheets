"""Vectorised thermodynamics for hydroprocessing: Peng-Robinson VLE, enthalpies, liquid volume.

Why not :class:`difflow.thermo.CubicThermo`: it is keyed by species name and
built from Python floats (``Characterization.thermo("pr")`` needs concrete
values), so a gradient with respect to the assay cannot reach the
pseudo-components' critical constants through it. Here every constant is an
array, the cut constants are the characterization's own (traceable) arrays,
and the equations are the same ones.

What is in it, and where each piece comes from:

* **Peng-Robinson** (Peng & Robinson 1976, Eqs. 3, 4, 9-12, 17) with the van
  der Waals one-fluid mixing rule and a ``k_ij`` matrix. ``kappa`` is the 1976
  quadratic in ``omega`` up to ``omega = 0.491`` and the Robinson-Peng (1978)
  cubic above it, which is what heavy pseudo-components need. Fugacity
  coefficients are the standard PR closed form (Eq. 17 of the 1976 paper).
  Cubic roots by Cardano/trigonometric formulas, each polished by two
  Newton steps on the live coefficients, so their first and second
  derivatives are the implicit-function ones.
* **Enthalpy**: the ideal-gas path, as the crude unit's
  :class:`~difflow_refinery.thermo.ColumnThermo` uses it -- ideal-gas Cp
  integrated from 298.15 K, a liquid below it by Watson's heat of
  vaporisation through ``dHvap(Tb)``, smoothly floored above ``Tc`` (the same
  ``_watson`` function). Reference: ideal gas at 298.15 K for every species;
  formation enthalpies are omitted (reaction heats are supplied per reaction
  by the kinetic model).
* **Liquid molar volume** of the cuts at temperature: the Rackett equation
  in the Spencer-Danner form scaled through each cut's own 15 C density
  (``V(T) = V_ref Z_RA^[(1-Tr)^(2/7) - (1-Tr_ref)^(2/7)]``) with the
  Yamada-Gunn ``Z_RA = 0.29056 - 0.08775 omega``. ``Tr`` is smoothly capped
  at 0.95 (a light cut above its critical temperature has no liquid volume
  of its own; the cap keeps the volume finite for the fugacity-equivalent
  concentrations of :mod:`.reactor`). Dissolved gases are given no volume.

Gas-species constants (:data:`GAS_PROPERTIES`): ``Tc``, ``Pc``, ``omega`` from
:func:`difflow.database.get_critical_props` for H2, H2S and NH3 and from the
crude unit's light-end table (Poling, Prausnitz & O'Connell 5th ed., App. A)
for the hydrocarbons; ideal-gas Cp polynomials from Reid, Prausnitz & Poling,
4th ed., App. A (for H2, H2S, NH3 as recalled, and cross-checked here against
the Poling-Prausnitz-O'Connell 5th ed. polynomials as tabulated in the
``chemicals`` package: within 0.5 % from 300 to 700 K -- pinned in the tests);
heats of vaporisation at the normal boiling point from the CRC Handbook as
tabulated in ``chemicals`` (H2 0.90, H2S 18.67, NH3 23.33 kJ/mol).

References:
    Peng, D.-Y. and Robinson, D.B., "A new two-constant equation of state",
        Ind. Eng. Chem. Fundam. 15(1), 59-64 (1976), doi:10.1021/i160057a011.
    Robinson, D.B. and Peng, D.-Y., "The characterization of the heptanes
        and heavier fractions for the GPA Peng-Robinson programs", GPA
        Research Report RR-28 (1978) -- the omega > 0.49 kappa (unverified
        against the report itself; the form is the one in common use).
    Rackett, H.G., "Equation of state for saturated liquids", J. Chem. Eng.
        Data 15(4), 514-517 (1970), doi:10.1021/je60047a012.
    Spencer, C.F. and Danner, R.P., "Improved equation for prediction of
        saturated liquid density", J. Chem. Eng. Data 17(2), 236-241 (1972),
        doi:10.1021/je60053a012.
    Yamada, T. and Gunn, R.D., "Saturated liquid molar volumes. The Rackett
        equation", J. Chem. Eng. Data 18(2), 234-236 (1973),
        doi:10.1021/je60057a006.
    (Volumes, pages and DOIs of these four as recalled -- unverified.)
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.database import get_critical_props
from difflow_refinery import thermo as rthermo
from difflow_refinery.hydroprocessing.layout import Layout, gas_mw

jax.config.update("jax_enable_x64", True)

R_GAS = 8.314462618
T_REF = rthermo.T_REF
_SQRT2 = float(np.sqrt(2.0))

#: Ideal-gas Cp (J/mol/K, cubic in T/K) of H2, H2S, NH3: Reid, Prausnitz &
#: Poling, 4th ed., App. A (see module docstring for how they were checked).
_CP_IG_GASES = {
    "hydrogen": (27.14, 9.274e-3, -1.381e-5, 7.645e-9),
    "hydrogen_sulfide": (31.94, 1.436e-3, 2.432e-5, -1.176e-8),
    "ammonia": (27.31, 2.383e-2, 1.707e-5, -1.185e-8),
}
#: Normal boiling point (K) and heat of vaporisation there (J/mol): CRC
#: Handbook values as tabulated in the ``chemicals`` package.
_TB_HVAP_GASES = {
    "hydrogen": (20.39, 900.0),
    "hydrogen_sulfide": (213.6, 18670.0),
    "ammonia": (239.82, 23330.0),
}


def _gas_row(name: str) -> tuple:
    """(MW, Tb, Tc, Pc, omega, dHvap_nb, cp) of one gas species."""
    if name in rthermo.LIGHT_ENDS:
        MW, Tb, Tc, Pc, w, hv, cp = rthermo.LIGHT_ENDS[name]
        return gas_mw(name), Tb, Tc, Pc, w, hv, tuple(cp)
    if name == "water":
        return (gas_mw(name), rthermo._WATER_TB, rthermo._WATER_TC, rthermo._WATER_PC, 0.3443,
                rthermo._WATER_HVAP_NB, tuple(rthermo._WATER_CP_IG))
    cp = get_critical_props(name)
    Tb, hv = _TB_HVAP_GASES[name]
    return gas_mw(name), Tb, cp.Tc, cp.Pc, cp.omega, hv, _CP_IG_GASES[name]


#: Every gas species' constants, ``name -> (MW, Tb, Tc, Pc, omega, dHvap_nb, cp_ig)``.
GAS_PROPERTIES: dict[str, tuple] = {
    n: _gas_row(n) for n in ("hydrogen", "hydrogen_sulfide", "ammonia", "water", "methane", "ethane",
                             "propane", "isobutane", "n_butane", "isopentane", "n_pentane", "n_hexane")
}

#: Default binary interaction parameters for the PR EOS. Light-gas pairs from
#: the ChemSep PR table as distributed with the ``thermo`` package (Bell et
#: al.); H2S with every cut 0.0333, ChemSep's H2S/n-decane value, the heaviest
#: paraffin it lists. Every other pair -- notably H2 with the cuts, for which
#: no transferable value exists -- is zero, the plain PR. ILLUSTRATIVE
#: choices: pass ``kij=`` with your own.
DEFAULT_KIJ: dict[tuple[str, str], float] = {
    ("hydrogen", "methane"): -0.0044, ("hydrogen", "ethane"): -0.0781,
    ("hydrogen", "propane"): -0.1311,
    ("hydrogen_sulfide", "ethane"): 0.0952, ("hydrogen_sulfide", "propane"): 0.0878,
    ("hydrogen_sulfide", "isobutane"): 0.0474, ("hydrogen_sulfide", "n_pentane"): 0.063,
    ("methane", "ethane"): -0.0059, ("methane", "propane"): 0.0119,
    ("methane", "isobutane"): 0.0256, ("methane", "n_butane"): 0.0185,
    ("methane", "n_pentane"): 0.023, ("methane", "n_hexane"): 0.04,
    ("ethane", "propane"): 0.0011, ("ethane", "isobutane"): -0.0067,
    ("ethane", "n_butane"): 0.0089, ("ethane", "n_pentane"): 0.0078,
    ("propane", "isobutane"): -0.0078, ("propane", "n_butane"): 0.0033,
    ("propane", "n_pentane"): 0.0267, ("isobutane", "n_butane"): -0.0004,
    ("n_butane", "n_pentane"): 0.0174, ("n_butane", "n_hexane"): -0.0056,
}
#: H2S-cut default k_ij (see :data:`DEFAULT_KIJ`).
DEFAULT_KIJ_H2S_CUT = 0.0333


# =============================================================================
# Component property table (gases that flash + cuts), a pytree
# =============================================================================


@dataclass(frozen=True)
class Components:
    """Constants of the flashing components: the layout's gases (water
    excluded) followed by its cuts, as arrays.

    Attributes:
        names: Component names, in array order (static).
        n_gas: How many of them are gases (static).
        Tb, Tc, Pc, omega: K, K, Pa, -.
        hvap_nb: Heat of vaporisation at ``Tb`` (J/mol).
        cp_ig: ``(n, 4)`` ideal-gas Cp coefficients.
        SG: Standard liquid gravity (1.0 placeholder for the gases).
        MW: g/mol (the feed's for the cuts; mass is computed from atoms).
        kij: ``(n, n)`` PR binary interaction parameters.
    """

    names: tuple[str, ...]
    n_gas: int
    Tb: Array
    Tc: Array
    Pc: Array
    omega: Array
    hvap_nb: Array
    cp_ig: Array
    SG: Array
    MW: Array
    kij: Array

    @property
    def n(self) -> int:
        return len(self.names)

    @property
    def hvap_A(self) -> Array:
        """Watson ``A`` so that ``A (1 - Tb/Tc)^0.38 = dHvap(Tb)``."""
        return self.hvap_nb / rthermo._watson_base(self.Tb, self.Tc) ** rthermo._WATSON_N

    def h_vapor(self, T) -> Array:
        """Ideal-gas molar enthalpy (J/mol) of every component at ``T``."""
        return rthermo._cp_integral(self.cp_ig, jnp.asarray(T, dtype=float))

    def h_liquid(self, T) -> Array:
        """Liquid molar enthalpy (J/mol): ideal gas less Watson's dHvap."""
        T = jnp.asarray(T, dtype=float)
        return self.h_vapor(T) - rthermo._watson(T, self.hvap_A, self.Tc)

    def cut_liquid_volume(self, T) -> Array:
        """``(n_cut,)`` liquid molar volume (m^3/mol) of the cuts at ``T`` (Rackett)."""
        k = self.n_gas
        Tc, w, SG, MW = self.Tc[k:], self.omega[k:], self.SG[k:], self.MW[k:]
        return rackett_volume(jnp.asarray(T, dtype=float), Tc, w, SG, MW)

    @staticmethod
    def build(layout: Layout, Tb, SG, MW, Tc, Pc, omega, hvap_nb, cp_ig,
              kij: dict | Array | None = None) -> "Components":
        """The table for ``layout`` from per-cut arrays (traceable).

        Gases come from :data:`GAS_PROPERTIES` (water is left out: it is
        decanted, not flashed). ``kij`` is a dict of name pairs (merged over
        :data:`DEFAULT_KIJ` and the H2S-cut default), a full matrix, or
        ``None`` for the defaults.
        """
        gases = tuple(g for g in layout.gases if g != "water")
        rows = [GAS_PROPERTIES[g] for g in gases]

        def col(i):
            return jnp.asarray([r[i] for r in rows], dtype=float).reshape(-1)

        names = gases + tuple(layout.cuts)
        n = len(names)
        if kij is None or isinstance(kij, dict):
            table = dict(DEFAULT_KIJ)
            for c in layout.cuts:
                table[("hydrogen_sulfide", c)] = DEFAULT_KIJ_H2S_CUT
            if isinstance(kij, dict):
                table.update(kij)
            K = np.zeros((n, n))
            idx = {nm: i for i, nm in enumerate(names)}
            for (a, b), v in table.items():
                if a in idx and b in idx:
                    K[idx[a], idx[b]] = K[idx[b], idx[a]] = v
            K = jnp.asarray(K)
        else:
            K = jnp.asarray(kij, dtype=float)
        return Components(
            names=names, n_gas=len(gases),
            Tb=jnp.concatenate([col(1), jnp.asarray(Tb, dtype=float)]),
            Tc=jnp.concatenate([col(2), jnp.asarray(Tc, dtype=float)]),
            Pc=jnp.concatenate([col(3), jnp.asarray(Pc, dtype=float)]),
            omega=jnp.concatenate([col(4), jnp.asarray(omega, dtype=float)]),
            hvap_nb=jnp.concatenate([col(5), jnp.asarray(hvap_nb, dtype=float)]),
            cp_ig=jnp.concatenate([jnp.asarray([r[6] for r in rows], dtype=float).reshape(-1, 4),
                                   jnp.asarray(cp_ig, dtype=float)]),
            SG=jnp.concatenate([jnp.ones(len(gases)), jnp.asarray(SG, dtype=float)]),
            MW=jnp.concatenate([col(0), jnp.asarray(MW, dtype=float)]),
            kij=K,
        )


jax.tree_util.register_dataclass(
    Components,
    data_fields=["Tb", "Tc", "Pc", "omega", "hvap_nb", "cp_ig", "SG", "MW", "kij"],
    meta_fields=["names", "n_gas"],
)


# =============================================================================
# Liquid volume
# =============================================================================

_RACKETT_TR_CAP = 0.95


def rackett_volume(T, Tc, omega, SG, MW, T_ref: float = 288.71):
    """Liquid molar volume (m^3/mol) at ``T`` scaled from the 60 F density.

    Spencer-Danner (1972) form of the Rackett (1970) equation with the
    Yamada-Gunn (1973) ``Z_RA = 0.29056 - 0.08775 omega``, anchored to the
    measured density at 60 F (``SG``)::

        V(T) = V(T_ref) * Z_RA ** ((1 - Tr)^(2/7) - (1 - Tr_ref)^(2/7))

    ``Tr`` is capped smoothly at 0.95.
    """
    Z = 0.29056 - 0.08775 * omega
    V_ref = MW / (1000.0 * SG * rthermo.RHO_WATER_60F)
    return V_ref * Z ** (_tau_capped(T / Tc) - _tau_capped(T_ref / Tc))


def _tau_capped(Tr):
    """``(1 - min(Tr, 0.95))^(2/7)`` with the min smoothed (width 0.01)."""
    w = 0.01
    Tr_c = _RACKETT_TR_CAP - w * jax.nn.softplus((_RACKETT_TR_CAP - Tr) / w)
    return (1.0 - Tr_c) ** (2.0 / 7.0)


# =============================================================================
# Peng-Robinson
# =============================================================================


def pr_kappa(omega: Array) -> Array:
    """PR ``kappa``: 1976 quadratic for omega <= 0.491, Robinson-Peng (1978) cubic above."""
    k76 = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    k78 = 0.379642 + 1.48503 * omega - 0.164423 * omega**2 + 0.016666 * omega**3
    return jnp.where(omega <= 0.491, k76, k78)


def pr_ab(T, Tc, Pc, omega):
    """Pure-component PR ``a(T)`` (Pa m^6/mol^2) and ``b`` (m^3/mol); Peng-Robinson (1976) Eqs. 9-12."""
    Tr = T / Tc
    alpha = (1.0 + pr_kappa(omega) * (1.0 - jnp.sqrt(Tr))) ** 2
    a = 0.45724 * R_GAS**2 * Tc**2 / Pc * alpha
    b = 0.07780 * R_GAS * Tc / Pc
    return a, b


def _cubic_roots(c2, c1, c0):
    """Smallest and largest real roots of ``Z^3 + c2 Z^2 + c1 Z + c0``.

    Cardano for one real root, the trigonometric form for three; both
    branches are computed with safe arguments and selected by the
    discriminant.
    """
    p = c1 - c2**2 / 3.0
    q = 2.0 * c2**3 / 27.0 - c2 * c1 / 3.0 + c0
    disc = (q / 2.0) ** 2 + (p / 3.0) ** 3
    # one real root
    sd = jnp.sqrt(jnp.maximum(disc, 0.0))
    t1 = jnp.cbrt(-q / 2.0 + sd) + jnp.cbrt(-q / 2.0 - sd)
    one = t1 - c2 / 3.0
    # three real roots
    m = jnp.sqrt(jnp.maximum(-p / 3.0, 1e-300))
    arg = jnp.clip(-q / 2.0 / jnp.maximum(m**3, 1e-300), -1.0, 1.0)
    th = jnp.arccos(arg) / 3.0
    r1 = 2.0 * m * jnp.cos(th) - c2 / 3.0
    r3 = 2.0 * m * jnp.cos(th + 2.0 * jnp.pi / 3.0) - c2 / 3.0
    three = disc < 0.0
    zmax = jnp.where(three, r1, one)
    zmin = jnp.where(three, r3, one)
    return zmin, zmax


def pr_lnphi(T, P, x, comps: Components, phase: str):
    """ln fugacity coefficients of a mixture ``x`` (mole fractions) in phase ``phase``.

    Peng & Robinson (1976) Eq. 17 with the van der Waals mixing rules
    ``a = sum x_i x_j sqrt(a_i a_j)(1 - k_ij)``, ``b = sum x_i b_i``. The
    liquid takes the smallest real root of the cubic above ``B``, the vapour
    the largest. Returns ``(lnphi, Z)``.
    """
    a_i, b_i = pr_ab(T, comps.Tc, comps.Pc, comps.omega)
    sa = jnp.sqrt(a_i)
    a_ij = jnp.outer(sa, sa) * (1.0 - comps.kij)
    xa = a_ij @ x
    am = x @ xa
    bm = x @ b_i
    RT = R_GAS * T
    A = am * P / RT**2
    B = bm * P / RT
    c2, c1, c0 = -(1.0 - B), A - 3.0 * B**2 - 2.0 * B, -(A * B - B**2 - B**3)
    zmin, zmax = _cubic_roots(*jax.lax.stop_gradient((c2, c1, c0)))
    Z0 = zmin if phase == "liquid" else zmax
    Z0 = jnp.maximum(Z0, jax.lax.stop_gradient(B) * (1.0 + 1e-9))
    # two Newton steps on the live coefficients: same value; the first makes
    # the first derivatives the implicit-function ones, the second makes the
    # second derivatives exact as well (the reactor differentiates d ln K/dT)
    Z = Z0
    for _ in range(2):
        f = Z**3 + c2 * Z**2 + c1 * Z + c0
        fp = 3.0 * Z**2 + 2.0 * c2 * Z + c1
        Z = Z - f / jnp.where(jnp.abs(fp) < 1e-300, 1e-300, fp)
    lnphi = (b_i / bm * (Z - 1.0) - jnp.log(Z - B)
             - A / (2.0 * _SQRT2 * B) * (2.0 * xa / am - b_i / bm)
             * jnp.log((Z + (1.0 + _SQRT2) * B) / (Z + (1.0 - _SQRT2) * B)))
    return lnphi, Z


def wilson_lnK(T, P, comps: Components) -> Array:
    """Wilson's K-value estimate, ``ln K = ln(Pc/P) + 5.373 (1 + omega)(1 - Tc/T)``."""
    return jnp.log(comps.Pc / P) + 5.373 * (1.0 + comps.omega) * (1.0 - comps.Tc / T)


__all__ = ["R_GAS", "GAS_PROPERTIES", "DEFAULT_KIJ", "DEFAULT_KIJ_H2S_CUT", "Components",
           "rackett_volume", "pr_kappa", "pr_ab", "pr_lnphi", "wilson_lnK"]
