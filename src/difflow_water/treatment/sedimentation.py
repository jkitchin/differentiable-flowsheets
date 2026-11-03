"""Ideal sedimentation basin (discrete settling): Hazen's overflow-rate theory.

Hazen (1904) [D3] and Camp (1946) [D4] (citations verified, papers not
read here: the result is the classical ideal-basin one and is derived
below) treat a basin as plug flow with uniform velocity and particles that
settle at their own constant velocity ``v_s`` without interacting. A
particle entering at any height is removed if it reaches the floor before
the outlet, which gives the fraction removed

    r = min(1, v_s / v_o),     v_o = Q / A   (overflow rate),

independent of depth. Settling velocities come from Stokes' law,
``v_s = (2/9) a_p^2 (rho_p - rho_w) g / mu`` (as stated by Messina et al.
2015 [F2], read), or are given directly. ``min`` is applied smoothly
(``softplus``-free: ``r = v_s/v_o - relu(v_s/v_o - 1)``; exact, with a kink
at ``v_s = v_o`` where the derivative is taken from the left).

Each particle class ``j`` is a stream component ``F_<name_j>`` (``1 mol =
1 g`` of solids, :data:`~difflow_water.treatment.common.MW_PARTICLE`). The
removed solids leave in the ``sludge`` outlet with enough water to make a
sludge of solids mass fraction ``sludge_solids`` (ILLUSTRATIVE default).
Flocculent and hindered settling (Camp's type II and III) are not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream
from difflow_water.chemistry import seawater as sw
from difflow_water.chemistry.seawater import MW_WATER
from difflow_water.treatment.common import MW_PARTICLE, RHO_DEFAULT, volumetric_flow

G = 9.80665  # m/s^2


def stokes_velocity(d_p, rho_p, T, rho_w: float | Array = RHO_DEFAULT) -> Array:
    """Stokes settling velocity (m/s) of a sphere of diameter ``d_p`` (m) in water at T.

    ``(2/9) a_p^2 (rho_p - rho_w) g / mu`` with ``a_p = d_p/2`` and the
    water viscosity of [S1] Eq. 23.
    """
    a = 0.5 * jnp.asarray(d_p, dtype=jnp.float64)
    return 2.0 / 9.0 * a**2 * (rho_p - rho_w) * G / sw.viscosity_water(T)


@dataclass(repr=False)
class SedimentationParams(ParamsMixin):
    """Ideal settling basin (defaults ILLUSTRATIVE).

    Attributes:
        area: Basin surface area (m^2).
        classes: Particle class names (stream keys ``F_<name>``).
        diameters: Particle diameter per class (m), used with Stokes' law.
        particle_density: Particle density per class (kg/m^3).
        settling_velocity: Settling velocity per class (m/s); overrides the
            Stokes velocity when given.
        sludge_solids: Solids mass fraction of the sludge (kg/kg).
        rho: Liquid density (kg/m^3).
    """

    area: float | Array = 2000.0
    classes: tuple = ("TSS_fine", "TSS_coarse")
    diameters: tuple | Array = (10e-6, 50e-6)
    particle_density: tuple | Array = (1050.0, 1200.0)
    settling_velocity: tuple | Array | None = None
    sludge_solids: float | Array = 0.02
    rho: float | Array = RHO_DEFAULT


class Sedimentation:
    """Ideal rectangular settling basin (discrete particles, overflow-rate removal).

    Inlet: water with particle classes. Outlets: ``clarified`` and ``sludge``.

    Example:
        >>> from difflow import make_stream
        >>> feed = make_stream({"H2O": 5.55e4, "TSS_fine": 20.0, "TSS_coarse": 30.0},
        ...                    T=288.15, P=101325.0)
        >>> clear, sludge, info = Sedimentation(SedimentationParams())(feed)
        >>> 0.0 < float(info["removal_total"]) < 1.0
        True
    """

    symbol = "SED"
    equations = [
        r"r_j = \min(1, v_{s,j} / v_o),\qquad v_o = Q / A",
        r"v_{s} = \tfrac{2}{9}\, a_p^2 (\rho_p - \rho_w)\, g / \mu\qquad \text{(Stokes)}",
    ]
    assumptions = [
        "Ideal basin: plug flow, uniform velocity, discrete non-interacting particles (Hazen, Camp type I).",
        "Stokes settling (particle Reynolds number < ~1) unless settling velocities are given.",
        "Sludge water set by a solids mass fraction.",
    ]
    references = [
        "Hazen, Trans. ASCE 53 (1904) 45-71, doi:10.1061/TACEAT.0001655.",
        "Camp, Trans. ASCE 111 (1946) 895-936, doi:10.1061/TACEAT.0005912.",
        "Messina, Marchisio & Sethi, J. Colloid Interface Sci. 446 (2015) 185-193, doi:10.1016/j.jcis.2015.01.024.",
    ]
    parameter_symbols = {"area": "A", "settling_velocity": "v_s", "diameters": "d_p",
                         "particle_density": "rho_p"}
    parameter_units = {"area": "m^2", "classes": "-", "diameters": "m",
                       "particle_density": "kg/m^3", "settling_velocity": "m/s",
                       "sludge_solids": "kg/kg", "rho": "kg/m^3"}
    outlet_roles = ["clarified", "sludge"]

    def __init__(self, params: SedimentationParams):
        """Initialize.

        Args:
            params: :class:`SedimentationParams`.
        """
        n = len(params.classes)
        if n == 0:
            raise ValueError("classes must name at least one particle class")
        sizes = (params.diameters, params.particle_density) if params.settling_velocity is None \
            else (params.settling_velocity,)
        for arr in sizes:
            if jnp.shape(jnp.asarray(arr)) != (n,):
                raise ValueError(f"one value per class expected ({n}), got shape {jnp.shape(jnp.asarray(arr))}")
        self.params = params

    def __call__(self, feed: Stream) -> tuple[Stream, Stream, dict]:
        """Settle a feed.

        Returns:
            ``(clarified, sludge, info)``; ``info`` has ``overflow_rate``
            (m/s), ``settling_velocity`` and ``removal`` per class,
            ``removal_total`` (mass basis) and ``sludge_flow_vol`` (m^3/s).
        """
        p = self.params
        names = list(p.classes)
        for nm in names:
            if f"F_{nm}" not in feed:
                raise ValueError(f"the feed has no F_{nm}")
        Q = volumetric_flow(feed, dict.fromkeys(names, MW_PARTICLE), p.rho)
        v_o = Q / p.area
        if p.settling_velocity is None:
            v_s = stokes_velocity(jnp.asarray(p.diameters), jnp.asarray(p.particle_density),
                                  feed["T"], p.rho)
        else:
            v_s = jnp.asarray(p.settling_velocity, dtype=jnp.float64)
        x = v_s / v_o
        r = x - jax.nn.relu(x - 1.0)
        F_in = jnp.stack([feed[f"F_{nm}"] for nm in names])
        F_rem = r * F_in
        m_solids = jnp.sum(F_rem) * MW_PARTICLE
        m_water = m_solids * (1.0 - p.sludge_solids) / p.sludge_solids
        clear, sludge = dict(feed), {k: (jnp.zeros_like(v) if k.startswith("F_") else v)
                                     for k, v in feed.items()}
        for j, nm in enumerate(names):
            clear[f"F_{nm}"] = F_in[j] - F_rem[j]
            sludge[f"F_{nm}"] = F_rem[j]
        clear["F_H2O"] = feed["F_H2O"] - m_water / MW_WATER
        sludge["F_H2O"] = m_water / MW_WATER
        info = {
            "overflow_rate": v_o,
            "settling_velocity": v_s,
            "removal": r,
            "removal_total": jnp.sum(F_rem) / jnp.sum(F_in),
            "sludge_flow_vol": (m_solids + m_water) / p.rho,
        }
        return clear, sludge, info
