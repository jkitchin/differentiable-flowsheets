"""Vectorised thermodynamics for a crude column.

difflow's :class:`~difflow.thermo.IdealThermo` is keyed by species name and
holds Python floats -- right for a flowsheet of a handful of named species,
wrong for a column of thirty pseudo-components on forty stages, where every
property is wanted as one array operation over ``(stage, component)`` and the
constants themselves are outputs of an assay that is being differentiated.
:class:`ColumnThermo` is that: arrays in, arrays out, a registered pytree.

The model is the one refinery simulators use for an atmospheric column at a
bar or two:

* **Vapour pressure** -- Lee-Kesler from each component's ``Tc``, ``Pc`` and
  the acentric factor that boils it at its own ``Tb``. Raoult's law
  ``K = Psat/P``. Applied to light ends above their critical temperature it
  extrapolates smoothly to a large K, which is what they should have.
* **Enthalpy** -- ideal-gas path. ``H_vap(T)`` is the ideal-gas Cp integrated
  from 298.15 K; the liquid is ``H_vap(T) - dHvap(T)`` with Watson's
  ``dHvap = A (1 - T/Tc)^0.38`` through the heat of vaporisation at ``Tb``.
  A light end dissolved in liquid above its ``Tc`` has ``dHvap = 0``. The
  reference state is the ideal gas at 298.15 K for every species; heats of
  formation are omitted, which is exact for a separation (nothing reacts).
* **Water** is not a component of the hydrocarbon liquid. It travels as vapour
  (the stripping steam) and condenses only as a free-water phase in the
  overhead drum. Its vapour pressure is Wagner-Pruss (IAPWS 1993), not
  Lee-Kesler, which is 28 % low for water at 300 K -- a polar molecule outside
  the correlation's basis.

Light-end constants are Poling, Prausnitz & O'Connell, *The Properties of
Gases and Liquids*, 5th ed., Appendix A; their ideal-gas Cp polynomials are
Reid, Prausnitz & Poling, 4th ed. (the same source as the database's).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from difflow_refinery import correlations as corr

T_REF = 298.15

#: name: (MW g/mol, Tb K, Tc K, Pc Pa, omega, dHvap at Tb J/mol,
#:        ideal-gas Cp a, b, c, d in J/mol/K with T in K)
LIGHT_ENDS: dict[str, tuple] = {
    "methane": (16.043, 111.66, 190.56, 45.99e5, 0.011, 8190.0, (19.25, 5.213e-2, 1.197e-5, -1.132e-8)),
    "ethane": (30.070, 184.55, 305.32, 48.72e5, 0.099, 14700.0, (5.409, 1.781e-1, -6.938e-5, 8.713e-9)),
    "propane": (44.097, 231.02, 369.83, 42.48e5, 0.152, 19040.0, (-4.224, 3.063e-1, -1.586e-4, 3.215e-8)),
    "isobutane": (58.123, 261.34, 407.85, 36.40e5, 0.186, 21300.0, (-1.390, 3.847e-1, -1.846e-4, 2.895e-8)),
    "n_butane": (58.123, 272.66, 425.12, 37.96e5, 0.200, 22440.0, (9.487, 3.313e-1, -1.108e-4, -2.822e-9)),
    "isopentane": (72.150, 300.99, 460.39, 33.81e5, 0.229, 24690.0, (-9.525, 5.066e-1, -2.729e-4, 5.723e-8)),
    "n_pentane": (72.150, 309.22, 469.70, 33.70e5, 0.252, 25790.0, (-3.626, 4.873e-1, -2.580e-4, 5.305e-8)),
    "n_hexane": (86.177, 341.88, 507.60, 30.25e5, 0.300, 28850.0, (-4.413, 5.820e-1, -3.119e-4, 6.494e-8)),
}

WATER_MW = 18.015
_WATER_TC = 647.096
_WATER_PC = 22.064e6
_WATER_HVAP_NB = 40660.0
_WATER_TB = 373.124
_WATER_CP_IG = (32.24, 1.924e-3, 1.055e-5, -3.596e-9)
_WAGNER = ((-7.85951783, 1.0), (1.84408259, 1.5), (-11.7866497, 3.0),
           (22.6807411, 3.5), (-15.9618719, 4.0), (1.80122502, 7.5))
_WATSON_N = 0.38


def _cp_integral(coeffs: Array, T: Array) -> Array:
    """``integral(T_REF -> T) Cp dT`` for cubic coefficients ``(..., 4)``."""
    a, b, c, d = (coeffs[..., k] for k in range(4))

    def antideriv(t):
        return a * t + b * t**2 / 2 + c * t**3 / 3 + d * t**4 / 4

    return antideriv(T) - antideriv(T_REF)


_WATSON_EPS = 0.01


def _watson_base(T: Array, Tc: Array) -> Array:
    """``1 - T/Tc``, smoothly floored at zero.

    A light end dissolved in a liquid above its own critical temperature has
    no latent heat, so ``1 - Tr`` must be held at zero there -- but not with
    ``max``: ``x**0.38`` has an infinite slope at zero, and propane, butane
    and pentane go critical at 97-197 C, the middle of a crude column, where
    a kink in every stage's energy balance stalls Newton. The smooth floor
    ``(x + sqrt(x^2 + eps^2))/2`` differs from ``x`` by ``eps^2/(4x)``: 0.01 %
    of the latent heat at ``Tr = 0.7``. The price is a tail above ``Tc``,
    decaying only as ``(eps^2/4|x|)^0.38``: propane 20 K supercritical keeps
    8 % of its latent heat at ``Tb`` (1.5 kJ/mol), 100 K above it 4 %.
    """
    x = 1.0 - T / Tc
    return 0.5 * (x + jnp.sqrt(x * x + _WATSON_EPS**2))


def _watson(T: Array, A: Array, Tc: Array) -> Array:
    return A * _watson_base(T, Tc) ** _WATSON_N


def water_vapor_pressure(T: Array) -> Array:
    """Saturation pressure of water (Pa), Wagner-Pruss (IAPWS 1993)."""
    tau = 1.0 - jnp.asarray(T) / _WATER_TC
    s = sum(a * jnp.maximum(tau, 1e-12) ** e for a, e in _WAGNER)
    return _WATER_PC * jnp.exp(_WATER_TC / T * s)


@dataclass(frozen=True)
class ColumnThermo:
    """Raoult K-values and ideal-gas-path enthalpies over arrays.

    Attributes:
        names: The hydrocarbon components, in array order. Water is separate.
        MW, SG: Molecular weight and standard liquid gravity, per component.
        Tb, Tc, Pc: Normal boiling point, critical temperature (K) and
            pressure (Pa).
        omega_vp: Acentric factor for the Lee-Kesler vapour pressure.
        hvap_A: Watson ``A`` (J/mol): ``dHvap = A (1 - T/Tc)^0.38``.
        cp_ig: ``(nc, 4)`` ideal-gas Cp coefficients.
    """

    names: tuple[str, ...]
    MW: Array
    SG: Array
    Tb: Array
    Tc: Array
    Pc: Array
    omega_vp: Array
    hvap_A: Array
    cp_ig: Array

    @property
    def n_components(self) -> int:
        return len(self.names)

    @classmethod
    def from_characterization(cls, crude) -> "ColumnThermo":
        """The thermo of a :class:`~difflow_refinery.assay.Characterization`.

        Differentiable: the arrays are the characterisation's, so a gradient
        taken through a column built on this reaches back to the assay.
        """
        light = crude.light_names
        unknown = [n for n in light if n not in LIGHT_ENDS]
        if unknown:
            raise ValueError(f"no column constants for light ends {unknown}; known: {', '.join(LIGHT_ENDS)}")
        rows = [LIGHT_ENDS[n] for n in light]
        le = {
            "MW": jnp.asarray([r[0] for r in rows], dtype=float).reshape(-1),
            "Tb": jnp.asarray([r[1] for r in rows], dtype=float).reshape(-1),
            "Tc": jnp.asarray([r[2] for r in rows], dtype=float).reshape(-1),
            "Pc": jnp.asarray([r[3] for r in rows], dtype=float).reshape(-1),
            "omega": jnp.asarray([r[4] for r in rows], dtype=float).reshape(-1),
            "hvap": jnp.asarray([r[5] for r in rows], dtype=float).reshape(-1),
            "cp": jnp.asarray([r[6] for r in rows], dtype=float).reshape(-1, 4),
        }
        Tb = jnp.concatenate([le["Tb"], crude.Tb])
        Tc = jnp.concatenate([le["Tc"], crude.Tc])
        hvap_nb = jnp.concatenate([le["hvap"], crude.hvap_nb])
        return cls(
            names=crude.names,
            MW=crude.component_MW,
            SG=crude.component_SG,
            Tb=Tb,
            Tc=Tc,
            Pc=jnp.concatenate([le["Pc"], crude.Pc]),
            omega_vp=jnp.concatenate([le["omega"], crude.omega_vp]),
            hvap_A=hvap_nb / _watson_base(Tb, Tc) ** _WATSON_N,
            cp_ig=jnp.concatenate([le["cp"], crude.cp_ig_coeffs]),
        )

    # ------------------------------------------------------------------
    # Properties. ``T`` broadcasts against the component axis: a scalar, or
    # ``(..., 1)`` for one temperature per stage.
    # ------------------------------------------------------------------

    def psat(self, T: Array) -> Array:
        """Vapour pressure (Pa), ``(..., nc)``."""
        return corr.vapor_pressure(T, self.Tc, self.Pc, self.omega_vp)

    def K(self, T: Array, P: Array) -> Array:
        """Raoult K-values ``Psat/P``, ``(..., nc)``."""
        return self.psat(T) / P

    def h_vapor(self, T: Array) -> Array:
        """Ideal-gas molar enthalpy (J/mol), ``(..., nc)``."""
        return _cp_integral(self.cp_ig, T)

    def dhvap(self, T: Array) -> Array:
        """Heat of vaporisation (J/mol), ``(..., nc)``."""
        return _watson(T, self.hvap_A, self.Tc)

    def h_liquid(self, T: Array) -> Array:
        """Liquid molar enthalpy (J/mol), ``(..., nc)``."""
        return self.h_vapor(T) - self.dhvap(T)

    @staticmethod
    def water_psat(T: Array) -> Array:
        return water_vapor_pressure(T)

    @staticmethod
    def water_h_vapor(T: Array) -> Array:
        return _cp_integral(jnp.asarray(_WATER_CP_IG), T)

    @staticmethod
    def water_h_liquid(T: Array) -> Array:
        A = _WATER_HVAP_NB / _watson_base(_WATER_TB, _WATER_TC) ** _WATSON_N
        return ColumnThermo.water_h_vapor(T) - _watson(T, A, _WATER_TC)

    def std_volume(self, flows: Array) -> Array:
        """Standard (60 F) liquid volume flow, m^3/s, of molar flows ``(..., nc)``."""
        return jnp.sum(flows * self.MW / (1000.0 * self.SG * RHO_WATER_60F), axis=-1)

    def mass(self, flows: Array) -> Array:
        """Mass flow, kg/s, of molar flows ``(..., nc)``."""
        return jnp.sum(flows * self.MW, axis=-1) / 1000.0


#: Density of water at 60 F (kg/m^3), the reference for specific gravity.
RHO_WATER_60F = 999.016

jax.tree_util.register_dataclass(
    ColumnThermo,
    data_fields=["MW", "SG", "Tb", "Tc", "Pc", "omega_vp", "hvap_A", "cp_ig"],
    meta_fields=["names"],
)

__all__ = ["ColumnThermo", "LIGHT_ENDS", "RHO_WATER_60F", "WATER_MW", "water_vapor_pressure", "T_REF"]
