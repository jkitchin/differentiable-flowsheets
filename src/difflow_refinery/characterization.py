"""Pseudocomponent characterization for refinery streams.

A refinery stream is carried as molar flows of *pseudocomponents*: narrow
boiling cuts, each described by a normal boiling point ``Tb`` and a specific
gravity ``SG`` at 15 degC, plus whatever composition bookkeeping the
upstream units track (sulfur, nitrogen, PNA, olefins, ...).  This module
holds that grid and the correlations that turn ``(Tb, SG)`` into the
quantities blending needs:

* molecular weight, critical temperature and pressure -- Riazi & Daubert
  (1980), the simple two-parameter form adopted by the API Technical Data
  Book (Riazi, *Characterization and Properties of Petroleum Fractions*,
  ASTM MNL50, 2005, Ch. 2);
* acentric factor -- Edmister (1958), Riazi MNL50 Ch. 2;
* vapor pressure -- Lee & Kesler (1975), Riazi MNL50 Ch. 7;
* liquid density at 15 degC -- ``SG * rho_water(15 degC)``.

Defined light components (butanes, pentanes) are poorly served by a
boiling-point correlation, so every derived property can be overridden per
pseudocomponent (``MW=``, ``Tc=``, ``Pc=``, ``omega=``).

This is deliberately a small characterization -- what a product blending
pool needs to compute properties from composition.  It is not an assay
model: there is no TBP-curve fitting or heavy-end extrapolation here.

Everything is JAX, so properties are differentiable in the grid's ``Tb``,
``SG`` and composition vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow.streams import get_flow_array

jax.config.update("jax_enable_x64", True)

#: Density of water at 15 degC (kg/m^3).  ``SG`` in this package is
#: ``rho(15 degC) / rho_water(15 degC)``, which is the basis on which product
#: volumes are reported (the issue's "volume basis uses SG at 15 degC").
RHO_WATER_15C = 999.10

#: Gas constant (J/mol/K).
R_GAS = 8.314462618

#: One standard atmosphere (Pa) and one psi (Pa).
ATM = 101325.0
PSI = 6894.757293168

_K_TO_R = 1.8


def riazi_daubert_mw(Tb: Array, SG: Array) -> Array:
    """Molecular weight (g/mol) from Riazi & Daubert (1980).

    ``MW = 4.5673e-5 Tb^2.1962 SG^-1.0164`` with ``Tb`` in degR (Riazi
    MNL50 Ch. 2); intended for MW ~70-300.

    Args:
        Tb: Normal boiling point (K).
        SG: Specific gravity at 15 degC.
    """
    Tb_R = Tb * _K_TO_R
    return 4.5673e-5 * Tb_R ** 2.1962 * SG ** -1.0164


def riazi_daubert_tc(Tb: Array, SG: Array) -> Array:
    """Critical temperature (K) from Riazi & Daubert (1980).

    ``Tc = 24.2787 Tb^0.58848 SG^0.3596``, both temperatures in degR.
    """
    Tb_R = Tb * _K_TO_R
    return 24.2787 * Tb_R ** 0.58848 * SG ** 0.3596 / _K_TO_R


def riazi_daubert_pc(Tb: Array, SG: Array) -> Array:
    """Critical pressure (Pa) from Riazi & Daubert (1980).

    ``Pc = 3.12281e9 Tb^-2.3125 SG^2.3201`` with ``Tb`` in degR and ``Pc``
    in psia.
    """
    Tb_R = Tb * _K_TO_R
    return 3.12281e9 * Tb_R ** -2.3125 * SG ** 2.3201 * PSI


def edmister_omega(Tb: Array, Tc: Array, Pc: Array) -> Array:
    """Acentric factor from Edmister (1958).

    ``omega = 3/7 * log10(Pc / 1 atm) / (Tc/Tb - 1) - 1``.
    """
    return 3.0 / 7.0 * jnp.log10(Pc / ATM) / (Tc / Tb - 1.0) - 1.0


def lee_kesler_psat(T: Array, Tc: Array, Pc: Array, omega: Array) -> Array:
    """Vapor pressure (Pa) from the Lee-Kesler (1975) correlation.

    ``ln Pr = f0(Tr) + omega f1(Tr)`` with

    * ``f0 = 5.92714 - 6.09648/Tr - 1.28862 ln Tr + 0.169347 Tr^6``
    * ``f1 = 15.2518 - 15.6875/Tr - 13.4721 ln Tr + 0.43577 Tr^6``

    Heavy pseudocomponents at gasoline-test
    temperatures sit at ``Tr ~ 0.4`` where the value is tiny; that is the
    correct answer for an RVP, not a numerical problem.
    """
    Tr = T / Tc
    lnTr = jnp.log(Tr)
    f0 = 5.92714 - 6.09648 / Tr - 1.28862 * lnTr + 0.169347 * Tr ** 6
    f1 = 15.2518 - 15.6875 / Tr - 13.4721 * lnTr + 0.43577 * Tr ** 6
    return Pc * jnp.exp(f0 + omega * f1)


@dataclass
class BlendCharacterization(ParamsMixin):
    """A pseudocomponent grid and its per-component property vectors.

    Attributes:
        names: Pseudocomponent names; streams carry them as ``F_<name>``.
        Tb: Normal boiling points (K), one per pseudocomponent.
        SG: Specific gravities at 15 degC.
        MW: Molecular weights (g/mol).  ``None`` -> Riazi-Daubert; a vector
            with NaN entries overrides only the finite ones.
        Tc: Critical temperatures (K); ``None``/NaN -> Riazi-Daubert.
        Pc: Critical pressures (Pa); ``None``/NaN -> Riazi-Daubert.
        omega: Acentric factors; ``None``/NaN -> Edmister.
        qualities: Per-pseudocomponent composition vectors, keyed by the
            property they feed.  Recognised keys and the basis each is
            averaged on: ``S_ppm`` and ``N_ppm`` (mass), ``CCR_wt`` (mass),
            ``aromatics_vol``, ``olefins_vol``, ``naphthenes_vol``,
            ``paraffins_vol`` and ``benzene_vol`` (volume).
    """

    names: Sequence[str]
    Tb: Any
    SG: Any
    MW: Any = None
    Tc: Any = None
    Pc: Any = None
    omega: Any = None
    qualities: Mapping[str, Any] = field(default_factory=dict)

    #: Averaging basis of each recognised quality vector.
    QUALITY_BASIS = {
        "S_ppm": "mass", "N_ppm": "mass", "CCR_wt": "mass",
        "aromatics_vol": "volume", "olefins_vol": "volume",
        "naphthenes_vol": "volume", "paraffins_vol": "volume",
        "benzene_vol": "volume",
    }

    def __post_init__(self):
        self.names = list(self.names)
        n = len(self.names)
        self.Tb = jnp.asarray(self.Tb, dtype=jnp.float64)
        self.SG = jnp.asarray(self.SG, dtype=jnp.float64)
        if self.Tb.shape != (n,) or self.SG.shape != (n,):
            raise ValueError(
                f"Tb and SG need one entry per pseudocomponent ({n}); got "
                f"{self.Tb.shape} and {self.SG.shape}")
        for key, vec in self.qualities.items():
            if key not in self.QUALITY_BASIS:
                raise ValueError(
                    f"unknown quality {key!r}; recognised: "
                    f"{sorted(self.QUALITY_BASIS)}")
            if np.shape(vec) != (n,):
                raise ValueError(f"quality {key!r} needs {n} entries")
        self.qualities = {k: jnp.asarray(v, dtype=jnp.float64)
                          for k, v in self.qualities.items()}

    @property
    def n(self) -> int:
        """Number of pseudocomponents."""
        return len(self.names)

    def _override(self, given, default: Array) -> Array:
        if given is None:
            return default
        given = jnp.asarray(given, dtype=jnp.float64)
        return jnp.where(jnp.isnan(given), default, given)

    @property
    def mw(self) -> Array:
        """Molecular weights (g/mol)."""
        return self._override(self.MW, riazi_daubert_mw(self.Tb, self.SG))

    @property
    def tc(self) -> Array:
        """Critical temperatures (K)."""
        return self._override(self.Tc, riazi_daubert_tc(self.Tb, self.SG))

    @property
    def pc(self) -> Array:
        """Critical pressures (Pa)."""
        return self._override(self.Pc, riazi_daubert_pc(self.Tb, self.SG))

    @property
    def acentric(self) -> Array:
        """Acentric factors."""
        return self._override(
            self.omega, edmister_omega(self.Tb, self.tc, self.pc))

    @property
    def density(self) -> Array:
        """Liquid density at 15 degC (kg/m^3)."""
        return self.SG * RHO_WATER_15C

    @property
    def molar_volume(self) -> Array:
        """Liquid molar volume at 15 degC (m^3/mol)."""
        return self.mw * 1e-3 / self.density

    def psat(self, T: Array) -> Array:
        """Lee-Kesler vapor pressures (Pa) at temperature ``T`` (K)."""
        return lee_kesler_psat(T, self.tc, self.pc, self.acentric)

    # -- composition conversions -----------------------------------------

    def flows(self, stream: Mapping[str, Any]) -> Array:
        """Molar flows (mol/s) of the grid's pseudocomponents in ``stream``.

        A pseudocomponent missing from the stream counts as zero flow.
        """
        zero = jnp.asarray(0.0, dtype=jnp.float64)
        present = [n for n in self.names if f"F_{n}" in stream]
        extra = [k for k in stream if k.startswith("F_")
                 and k[2:] not in self.names]
        if extra:
            raise ValueError(
                f"stream carries species outside the characterization: "
                f"{[k[2:] for k in extra]}")
        if len(present) == self.n:
            return get_flow_array(stream, self.names)
        return jnp.stack([jnp.asarray(stream.get(f"F_{n}", zero),
                                      dtype=jnp.float64)
                          for n in self.names])

    def volume_fractions(self, moles: Array) -> Array:
        """Volume fractions at 15 degC from molar amounts (ideal mixing)."""
        vol = moles * self.molar_volume
        return vol / jnp.sum(vol)

    def mass_fractions(self, moles: Array) -> Array:
        """Mass fractions from molar amounts."""
        m = moles * self.mw
        return m / jnp.sum(m)
