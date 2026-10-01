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
from difflow_refinery import correlations as _corr
from difflow_refinery.correlations import edmister_omega, lee_kesler_psat  # noqa: F401

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

# The correlations themselves live in difflow_refinery.correlations (#301);
# these are the names the blending pool has always exported.


def riazi_daubert_mw(Tb: Array, SG: Array) -> Array:
    """Molecular weight (g/mol), Riazi & Daubert (1980): ``4.5673e-5 Tb^2.1962 SG^-1.0164``, Tb in degR."""
    return _corr.riazi_daubert_1980(Tb, SG)[0]


def riazi_daubert_tc(Tb: Array, SG: Array) -> Array:
    """Critical temperature (K), Riazi & Daubert (1980): ``24.2787 Tb^0.58848 SG^0.3596``, degR."""
    return _corr.riazi_daubert_1980(Tb, SG)[1]


def riazi_daubert_pc(Tb: Array, SG: Array) -> Array:
    """Critical pressure (Pa), Riazi & Daubert (1980): ``3.12281e9 Tb^-2.3125 SG^2.3201`` psia, degR."""
    return _corr.riazi_daubert_1980(Tb, SG)[2]


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

    #: Species a refinery stream may carry that are not blend components:
    #: the crude unit's decanted/stripping water (``F_water``) and the
    #: vacuum column's steam (``F_H2O``). :meth:`flows` ignores them --
    #: water is drained from a product tank, not blended.
    WATER_SPECIES = ("water", "H2O")

    @classmethod
    def from_characterization(cls, char, contaminants: bool | None = None,
                              composition=None) -> "BlendCharacterization":
        """The blend grid of a crude-unit :class:`~difflow_refinery.assay.Characterization`.

        So the crude and vacuum units' product streams -- whose species are
        ``char.names`` -- go straight into :meth:`BlendComponent.from_stream`
        with the properties those units ran on, rather than re-estimated
        from Riazi-Daubert on a grid of their own.

        * ``Tb``, ``SG`` and ``MW`` are the characterization's, light ends
          included (their constants from the column's light-end table).
        * ``Tc``, ``Pc`` and the acentric factor are the ones the columns'
          Lee-Kesler vapour pressure uses: for a cut that is ``omega_vp``,
          which puts ``psat(Tb)`` at one atmosphere, so the pool's Raoult
          RVP sees the same volatility the column did.
        * The per-component contaminants become the pool's quality vectors:
          ``S_ppm`` and ``N_ppm`` (wppm) and ``CCR_wt`` (wt%), all
          mass-averaged. Ni+V and asphaltenes have no blending rule in the
          pool and are left out.

        Args:
            char: The characterization.
            contaminants: Include the quality vectors. ``None`` (default)
                includes each one the assay actually gave (a vector that is
                not all zero); ``True`` includes all three; ``False`` none.
            composition: Hydrocarbon types as the ``paraffins_vol``,
                ``naphthenes_vol``, ``aromatics_vol`` and ``olefins_vol``
                qualities (vol%, volume-averaged): a
                :class:`~difflow_refinery.composition.Composition` over
                ``char.names``. ``None`` (default) takes
                ``char.composition`` if the characterization carries one;
                ``False`` leaves them out.

        Differentiable: every array is the characterization's, so a blend
        property computed through this reaches back to the assay.
        """
        from difflow_refinery.thermo import LIGHT_ENDS

        rows = [LIGHT_ENDS[n] for n in char.light_names]
        light = {key: jnp.asarray([r[i] for r in rows], dtype=jnp.float64).reshape(-1)
                 for i, key in ((1, "Tb"), (2, "Tc"), (3, "Pc"), (4, "omega"))}
        qualities = {}
        for key, field_name, scale in (("S_ppm", "sulfur", 1e6), ("N_ppm", "nitrogen", 1e6),
                                       ("CCR_wt", "ccr", 100.0)):
            vec = jnp.asarray(getattr(char, field_name), dtype=jnp.float64) * scale
            if contaminants is None:
                try:
                    keep = bool(np.any(np.asarray(vec) != 0.0))
                except (jax.errors.ConcretizationTypeError,
                        jax.errors.TracerArrayConversionError):
                    keep = True
            else:
                keep = bool(contaminants)
            if keep:
                qualities[key] = vec
        if composition is None:
            composition = getattr(char, "composition", None)
        if composition is not None and composition is not False:
            if tuple(composition.names) != tuple(char.names):
                raise ValueError("composition must cover char.names, in order")
            qualities.update(composition.blend_qualities())
        return cls(
            names=list(char.names),
            Tb=jnp.concatenate([light["Tb"], jnp.asarray(char.Tb)]),
            SG=jnp.asarray(char.component_SG),
            MW=jnp.asarray(char.component_MW),
            Tc=jnp.concatenate([light["Tc"], jnp.asarray(char.Tc)]),
            Pc=jnp.concatenate([light["Pc"], jnp.asarray(char.Pc)]),
            omega=jnp.concatenate([light["omega"], jnp.asarray(char.omega_vp)]),
            qualities=qualities,
        )

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

        A pseudocomponent missing from the stream counts as zero flow;
        water (:attr:`WATER_SPECIES`) is ignored, any other species is an
        error.
        """
        zero = jnp.asarray(0.0, dtype=jnp.float64)
        present = [n for n in self.names if f"F_{n}" in stream]
        extra = [k for k in stream if k.startswith("F_")
                 and k[2:] not in self.names
                 and k[2:] not in self.WATER_SPECIES]
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
