"""Thermodynamics of a pseudocomponent column at vacuum.

:class:`ColumnThermo` is deliberately simple, and each simplification is one
a vacuum column makes true:

- **K-values** ``K_i = psat_i(T) / P``: ideal gas and an ideal liquid
  solution. At 1-10 kPa the vapor is ideal to far better than the
  correlations are accurate, and a mixture of neighbouring petroleum cuts is
  close to an ideal solution. Vapor pressure is Maxwell-Bonnell (see
  :mod:`~difflow_refinery.correlations`), which is fitted down to a fraction
  of a mmHg.
- **Steam** is a vapor-only component. Water's vapor pressure at the
  column's coldest point (60-80 C, 20-50 kPa) is ten times the column's
  pressure, so it cannot condense, and its solubility in the oil is
  negligible. It lowers the hydrocarbons' partial pressure, which is what
  stripping steam is for, and carries enthalpy.
- **Enthalpy**: liquid from Kesler-Lee's Cp; vapor = liquid + heat of
  vaporization, and the heat of vaporization is ``R T^2 dln(psat)/dT`` --
  the Clausius-Clapeyron slope of the very curve that sets the K-values. So
  the energy balance and the phase equilibrium cannot disagree about how
  volatile a cut is. Datum: liquid at 25 C for hydrocarbons, ideal-gas
  water at 25 C for steam.

All methods take a temperature and broadcast over components (last axis).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from difflow_refinery import correlations as corr
from difflow_refinery.assay import PseudoComponents

MW_WATER = 18.01528
R_GAS = corr.R_GAS

# NIST Shomate constants for water vapor (500-1700 K); used down to 330 K,
# where only enthalpy *differences* of superheated steam are needed and the
# error is a fraction of a percent of cp.
_SHOMATE_WATER = (30.09200, 6.832514, 6.793435, -2.534480, 0.082139)


def steam_enthalpy(T):
    """Ideal-gas enthalpy of water, J/mol, relative to 298.15 K."""
    A, B, C, D, E = _SHOMATE_WATER

    def H(T):
        t = T / 1000.0
        return 1000.0 * (A * t + B * t ** 2 / 2 + C * t ** 3 / 3
                         + D * t ** 4 / 4 - E / t)

    return H(T) - H(298.15)


class ColumnThermo:
    """K-values and enthalpies of a pseudocomponent set (see module doc).

    Args:
        components: The :class:`~difflow_refinery.assay.PseudoComponents`.
        correct_kw: Apply Maxwell-Bonnell's Watson-K correction.
    """

    def __init__(self, components: PseudoComponents, correct_kw: bool = True):
        self.c = components
        self.correct_kw = correct_kw

    def log_psat(self, T):
        """ln of the vapor pressure (Pa); ``T`` broadcast against components."""
        T = jnp.asarray(T)[..., None]
        return jnp.log(corr.maxwell_bonnell_psat(
            T, self.c.Tb, self.c.Kw, correct_kw=self.correct_kw))

    def log_K(self, T, P):
        """ln K = ln psat - ln P."""
        return self.log_psat(T) - jnp.log(jnp.asarray(P))[..., None]

    def h_liquid(self, T):
        """Liquid enthalpy, J/mol."""
        T = jnp.asarray(T)[..., None]
        return (corr.kesler_lee_liquid_enthalpy(T, self.c.SG, self.c.Kw)
                * self.c.MW / 1000.0)

    def dh_vap(self, T):
        """Heat of vaporization, J/mol, from Clausius-Clapeyron on psat."""
        T = jnp.asarray(T, dtype=float)
        _, dlnp = jax.jvp(self.log_psat, (T,), (jnp.ones_like(T),))
        return R_GAS * T[..., None] ** 2 * dlnp

    def h_vapor(self, T):
        """Vapor enthalpy, J/mol."""
        return self.h_liquid(T) + self.dh_vap(T)

    @staticmethod
    def h_steam(T):
        """Steam enthalpy, J/mol."""
        return steam_enthalpy(jnp.asarray(T))
