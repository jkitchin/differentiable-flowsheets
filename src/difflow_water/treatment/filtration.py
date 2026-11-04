"""Rapid granular (depth) filter: colloid filtration theory and clean-bed head loss.

Particle removal (clean bed). Yao, Habibian & O'Melia (1971) [F1]: a
filter is a packed bed of spherical collectors of diameter ``d_c``; each
collector removes the particles in its projected area with the single
collector efficiency ``eta`` times an attachment efficiency ``alpha``. A
mass balance over a slice ``dz`` of a bed of porosity ``eps`` (collectors
per volume ``(1 - eps) / (pi d_c^3 / 6)``, each sweeping
``alpha eta (pi d_c^2 / 4) U C``) gives

    ln(C / C0) = -(3/2) (1 - eps) alpha eta L / d_c.

Single collector efficiency: Yao's sum of diffusion, gravity and
interception, as Messina et al. (2015) [F2] restate it (read; [F1] itself
not read: cross-checked, secondary),

    eta = 4.04 N_Pe^(-2/3) + N_G + (3/2) N_R^2,
    N_Pe = d_c U / D,  D = k_B T / (3 pi mu d_p),
    N_G = v_s / U,  v_s = (2/9) a_p^2 (rho_p - rho_w) g / mu,  N_R = d_p / d_c.

(Messina et al. use radii: ``N_Pe = 2 a_c U / D`` and ``N_R = a_p / a_c``,
the same numbers.) Yao's correlation can exceed 1 for very small or very
large particles or very slow flow; ``info["eta"]`` reports it and nothing
caps it. Later correlations (Rajagopalan-Tien, Tufenkji-Elimelech, Messina
et al.) are not implemented.

Clean-bed head loss: Carman-Kozeny [F3] (citation verified; Carman's
paper not read, the constant 180 = 5 x 36 is the standard Kozeny constant 5
with a sphere's specific surface 6/d; (unverified)),

    h_L / L = 180 mu (1 - eps)^2 U / (rho g eps^3 (phi d_c)^2).

Clogging, ripening and backwash are not modelled (a clean-bed snapshot).
Particle classes are stream components with diameters and densities, as in
:mod:`~difflow_water.treatment.sedimentation`. Defaults ILLUSTRATIVE.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream
from difflow_water.chemistry import seawater as sw
from difflow_water.treatment.common import MW_PARTICLE, RHO_DEFAULT, volumetric_flow
from difflow_water.treatment.sedimentation import G, stokes_velocity

K_B = 1.380649e-23  # J/K (exact, SI 2019)


def collector_efficiency(d_p, rho_p, d_c, U, T, rho_w: float | Array = RHO_DEFAULT) -> dict:
    """Yao's single collector efficiency and its three parts.

    Args:
        d_p: Particle diameter (m).
        rho_p: Particle density (kg/m^3).
        d_c: Collector (media grain) diameter (m).
        U: Approach (filtration) velocity (m/s).
        T: Temperature (K).
        rho_w: Water density (kg/m^3).

    Returns:
        dict with ``eta``, ``eta_D``, ``eta_G``, ``eta_I``, ``N_Pe``, ``N_G``, ``N_R``.
    """
    mu = sw.viscosity_water(T)
    D = K_B * T / (3.0 * jnp.pi * mu * d_p)
    N_Pe = d_c * U / D
    N_G = stokes_velocity(d_p, rho_p, T, rho_w) / U
    N_R = d_p / d_c
    eta_D = 4.04 * N_Pe ** (-2.0 / 3.0)
    eta_I = 1.5 * N_R**2
    return {"eta": eta_D + N_G + eta_I, "eta_D": eta_D, "eta_G": N_G, "eta_I": eta_I,
            "N_Pe": N_Pe, "N_G": N_G, "N_R": N_R}


@dataclass(repr=False)
class FiltrationParams(ParamsMixin):
    """Rapid granular filter, clean bed (defaults ILLUSTRATIVE).

    Attributes:
        area: Filter area (m^2).
        depth: Media depth L (m).
        media_diameter: Collector (grain) diameter d_c (m).
        porosity: Bed porosity (-).
        sphericity: Grain sphericity phi (-), for the head loss.
        alpha: Attachment efficiency (-).
        classes: Particle class names (``F_<name>``).
        diameters: Particle diameter per class (m).
        particle_density: Particle density per class (kg/m^3).
        rho: Liquid density (kg/m^3).
    """

    area: float | Array = 300.0
    depth: float | Array = 0.75
    media_diameter: float | Array = 0.5e-3
    porosity: float | Array = 0.42
    sphericity: float | Array = 0.85
    alpha: float | Array = 0.5
    classes: tuple = ("TSS_fine", "TSS_coarse")
    diameters: tuple | Array = (10e-6, 50e-6)
    particle_density: tuple | Array = (1050.0, 1200.0)
    rho: float | Array = RHO_DEFAULT


class Filtration:
    """Rapid granular filter: Yao colloid filtration and Carman-Kozeny head loss (clean bed).

    Inlet: water with particle classes. Outlets: ``filtrate`` and
    ``retained`` (the particles caught, no water: backwash is not modelled).

    Example:
        >>> from difflow import make_stream
        >>> feed = make_stream({"H2O": 5.55e4, "TSS_fine": 2.0, "TSS_coarse": 0.5},
        ...                    T=288.15, P=101325.0)
        >>> filt, kept, info = Filtration(FiltrationParams())(feed)
        >>> float(info["head_loss"]) > 0
        True
    """

    symbol = "RGF"
    equations = [
        r"\ln\frac{C}{C_0} = -\frac{3}{2}\,\frac{(1-\varepsilon)\,\alpha\,\eta\,L}{d_c}",
        r"\eta = 4.04\,N_{Pe}^{-2/3} + N_G + \tfrac{3}{2} N_R^2",
        r"\frac{h_L}{L} = \frac{180\,\mu\,(1-\varepsilon)^2\,U}{\rho g\,\varepsilon^3 (\phi d_c)^2}",
    ]
    assumptions = [
        "Clean bed: no clogging, ripening or breakthrough; no backwash.",
        "Yao's additive single-collector efficiency (diffusion, gravity, interception); not capped at 1.",
        "Carman-Kozeny laminar head loss with Kozeny constant 5.",
        "Attachment efficiency alpha is an input.",
    ]
    references = [
        "Yao, Habibian & O'Melia, Environ. Sci. Technol. 5 (1971) 1105-1112, doi:10.1021/es60058a005.",
        "Messina, Marchisio & Sethi, J. Colloid Interface Sci. 446 (2015) 185-193, doi:10.1016/j.jcis.2015.01.024.",
        "Carman, Trans. IChemE 15 (1937) 150; reprinted Chem. Eng. Res. Des. 75 (1997) S32-S48, doi:10.1016/S0263-8762(97)80003-2.",
    ]
    parameter_symbols = {"depth": "L", "media_diameter": "d_c", "porosity": "eps",
                         "alpha": "alpha", "sphericity": "phi", "area": "A"}
    parameter_units = {"area": "m^2", "depth": "m", "media_diameter": "m", "porosity": "-",
                       "sphericity": "-", "alpha": "-", "classes": "-", "diameters": "m",
                       "particle_density": "kg/m^3", "rho": "kg/m^3"}
    outlet_roles = ["filtrate", "retained"]

    def __init__(self, params: FiltrationParams):
        """Initialize.

        Args:
            params: :class:`FiltrationParams`.
        """
        n = len(params.classes)
        if n == 0:
            raise ValueError("classes must name at least one particle class")
        for arr in (params.diameters, params.particle_density):
            if jnp.shape(jnp.asarray(arr)) != (n,):
                raise ValueError(f"one value per class expected ({n}), got shape {jnp.shape(jnp.asarray(arr))}")
        self.params = params

    def __call__(self, feed: Stream) -> tuple[Stream, Stream, dict]:
        """Filter a feed.

        Returns:
            ``(filtrate, retained, info)``; ``info`` has ``filtration_rate``
            (m/s), ``eta`` and ``removal`` per class, ``removal_total``,
            ``head_loss`` (m of water).
        """
        p = self.params
        names = list(p.classes)
        for nm in names:
            if f"F_{nm}" not in feed:
                raise ValueError(f"the feed has no F_{nm}")
        Q = volumetric_flow(feed, dict.fromkeys(names, MW_PARTICLE), p.rho)
        U = Q / p.area
        T = feed["T"]
        ce = collector_efficiency(jnp.asarray(p.diameters, dtype=jnp.float64),
                                  jnp.asarray(p.particle_density, dtype=jnp.float64),
                                  p.media_diameter, U, T, p.rho)
        eps = p.porosity
        frac = jnp.exp(-1.5 * (1.0 - eps) * p.alpha * ce["eta"] * p.depth / p.media_diameter)
        F_in = jnp.stack([feed[f"F_{nm}"] for nm in names])
        filt = dict(feed)
        kept = {k: (jnp.zeros_like(v) if k.startswith("F_") else v) for k, v in feed.items()}
        for j, nm in enumerate(names):
            filt[f"F_{nm}"] = F_in[j] * frac[j]
            kept[f"F_{nm}"] = F_in[j] * (1.0 - frac[j])
        mu = sw.viscosity_water(T)
        h_L = (180.0 * mu * (1.0 - eps) ** 2 * U * p.depth
               / (p.rho * G * eps**3 * (p.sphericity * p.media_diameter) ** 2))
        info = {
            "filtration_rate": U,
            "eta": ce["eta"],
            "eta_D": ce["eta_D"], "eta_G": ce["eta_G"], "eta_I": ce["eta_I"],
            "removal": 1.0 - frac,
            "removal_total": jnp.sum(F_in * (1.0 - frac)) / jnp.sum(F_in),
            "head_loss": h_L,
        }
        return filt, kept, info
