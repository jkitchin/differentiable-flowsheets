"""Reformate properties from composition: octane, aromatics, benzene, RVP.

The reformate is a mixture of the reformer's species, so its properties are
computed from what is in it rather than correlated from a severity:

* **Octane** -- each species' pure-compound RON and MON
  (:data:`~difflow_refinery.reforming.species.SPECIES`, mostly unverified,
  see there) blended with the Ethyl RT-70 rule that the blend pool uses
  (:func:`difflow_refinery.blending.ethyl_rt70`, Healy, Maassen & Peterson
  1959), with each species' aromatics content 100 vol% if it is an aromatic
  and 0 otherwise, and no olefins. The plain volume average is reported
  alongside (``RON_linear``); the two differ because the aromatics and
  sensitivity spreads of a reformate are large. Pure-compound octanes of
  the low-octane paraffins are below the 0-120 range the RT-70 fit was made
  on; the rule extrapolates there.
* **Aromatics and benzene** -- standard liquid volume fractions (ideal mixing
  at 60 F).
* **RVP** -- :func:`difflow_refinery.blending.raoult_rvp`: Raoult's law at
  100 F with the D323 four-to-one vapour space, Lee-Kesler vapour pressures
  from each species' critical constants. A check, not a measured RVP.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow_refinery.blending import T_RVP, ethyl_rt70, raoult_rvp
from difflow_refinery.characterization import PSI
from difflow_refinery.correlations import vapor_pressure
from difflow_refinery.reforming import species as sp
from difflow_refinery.thermo import RHO_WATER_60F

_LIQ = np.isfinite(sp.RHO60)
#: Standard liquid molar volume (m^3/mol at 60 F); 0 for H2 and methane.
MOLAR_VOLUME = np.where(_LIQ, sp.MW / 1000.0 / np.where(_LIQ, sp.RHO60, 1.0), 0.0)
_AROM = np.array([sp.SPECIES[k].kind == "A" for k in sp.NAMES])
# Octane of species with no octane number (H2, C1-C3) is irrelevant: they are
# not in a stabilized reformate. Zero-volume species carry no weight anyway.
_RON = np.nan_to_num(sp.RON, nan=0.0)
_MON = np.nan_to_num(sp.MON, nan=0.0)


def volume_fractions(F: Array) -> Array:
    """``(N,)`` standard liquid volume fractions of flows ``F`` (mol/s)."""
    v = jnp.asarray(F) * jnp.asarray(MOLAR_VOLUME)
    return v / jnp.sum(v)


def octane(F: Array) -> dict[str, Array]:
    """RON and MON of a liquid of flows ``F`` (RT-70), and the linear averages."""
    phi = volume_fractions(F)
    arom = jnp.asarray(np.where(_AROM, 100.0, 0.0))
    ron, mon = ethyl_rt70(phi, jnp.asarray(_RON), jnp.asarray(_MON), jnp.zeros_like(phi), arom)
    return {"RON": ron, "MON": mon, "RON_linear": phi @ jnp.asarray(_RON),
            "MON_linear": phi @ jnp.asarray(_MON)}


def rvp(F: Array) -> Array:
    """Reid vapour pressure (Pa) of a liquid of flows ``F`` (Raoult, D323 geometry)."""
    F = jnp.asarray(F)
    liq = jnp.asarray(_LIQ)
    Fl = jnp.where(liq, F, 0.0)
    psat = vapor_pressure(T_RVP, jnp.asarray(sp.TC), jnp.asarray(sp.PC), jnp.asarray(sp.OMEGA))
    return raoult_rvp(Fl, jnp.where(liq, psat, 0.0), jnp.asarray(MOLAR_VOLUME))


def reformate_properties(F: Array) -> dict[str, Array]:
    """Everything a blender asks of a reformate of flows ``F`` (mol/s).

    Keys: ``RON``, ``MON``, ``RON_linear``, ``MON_linear``, ``aromatics_vol``,
    ``benzene_vol``, ``paraffins_vol``, ``naphthenes_vol``, ``RVP_psi``,
    ``SG``.
    """
    phi = volume_fractions(F)
    out = octane(F)
    kinds = [sp.SPECIES[k].kind for k in sp.NAMES]
    for name, ks in (("aromatics_vol", ("A",)), ("naphthenes_vol", ("N",)),
                     ("paraffins_vol", ("nP", "iP", "light"))):
        m = jnp.asarray([k in ks for k in kinds])
        out[name] = 100.0 * jnp.sum(jnp.where(m, phi, 0.0))
    out["benzene_vol"] = 100.0 * phi[sp.INDEX["A6"]]
    out["RVP_psi"] = rvp(F) / PSI
    rho = jnp.asarray(np.where(_LIQ, sp.RHO60, 0.0))
    out["SG"] = phi @ rho / RHO_WATER_60F
    return out


def blend_component(name: str, F: Array, **overrides):
    """The reformate as a :class:`~difflow_refinery.blending.BlendComponent` (property mode)."""
    from difflow_refinery.blending import BlendComponent

    p = reformate_properties(F)
    props = {k: p[k] for k in ("SG", "RON", "MON", "aromatics_vol", "benzene_vol", "RVP_psi",
                               "naphthenes_vol", "paraffins_vol")}
    props["olefins_vol"] = jnp.asarray(0.0)
    props.update(overrides)
    return BlendComponent.from_properties(name, **props)
