"""Granular activated carbon (GAC) adsorber: the constant-pattern HSDM.

The equations are WaterTAP 1.8.0's ``GAC`` unit model (``cphsdm_calculation_
method="input"``, fixed film-transfer and surface-diffusion coefficients,
the trapezoidal steady-state effluent approximation, its default),
transcribed from the code (read directly) so the two compare
(``benchmarks/watertap/gac.py``). They are the empirical
constant-pattern homogeneous surface diffusion model (CPHSDM) of Hand,
Crittenden & Thacker (1984) [A1]; that paper was not read here, so the
model form and the regression constants are (unverified) against it.

- Freundlich equilibrium ``q_e = K C0^(1/n)``; solute distribution parameter
  ``D_g = rho_a q_e (1 - eps) / (eps C0)``; Biot number
  ``Bi = k_f (d_p / 2) (1 - eps) / (D_s D_g eps)``;
- minimum Stanton number for a constant pattern ``St_min = a0 Bi + a1``;
  minimum EBCT ``EBCT_min = St_min (d_p / 2) / ((1 - eps) k_f)``;
- throughput at a breakthrough ratio ``c``:
  ``T(c) = b0 + b1 c^b2 + b3 / (1.01 - c^b4)``;
- time to reach ``c``: ``t(c) = eps EBCT_min (D_g + 1) T(c) + eps (EBCT - EBCT_min) (D_g + 1)``;
- bed: ``eps = 1 - rho_b / rho_a``, ``L = v EBCT``, ``A = Q / v``,
  ``V = L A``, carbon mass ``rho_b V``; usage rate ``rho_b V / t(c_replace)``;
- steady-state removal: the effluent concentration averaged over a bed's
  life by the trapezoidal rule on ``n_elements`` breakthrough points from
  0.01 to ``c_replace`` (with the point (t = 0, c = 0)); the outlet carries
  ``c_avg`` of the target, the rest leaves through the ``removed`` outlet.

The regression constants ``a0, a1, b0..b4`` depend on Bi and 1/n (tables in
[A1]); the defaults are the values WaterTAP's GAC example fixes, as are the
other defaults (ILLUSTRATIVE). Valid only when ``EBCT >= EBCT_min``
(``info["constant_pattern"]``).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream
from difflow_water.treatment.common import (
    RHO_DEFAULT,
    only,
    scale_component,
    volumetric_flow,
)


@dataclass(repr=False)
class GACParams(ParamsMixin):
    """GAC adsorber design and CPHSDM inputs (defaults: WaterTAP's example; ILLUSTRATIVE).

    Attributes:
        target: Name of the adsorbed component (stream key ``F_<target>``).
        mw_target: Molar mass of the target (kg/mol).
        freund_k: Freundlich K in SI base units, ``(m^3/kg)^(1/n)``.
        freund_ninv: Freundlich exponent 1/n (-).
        kf: Liquid film transfer coefficient (m/s).
        ds: Surface diffusion coefficient (m^2/s).
        particle_dens_app: Apparent GAC particle density (kg/m^3).
        particle_dia: GAC particle diameter (m).
        bed_voidage: Bed void fraction (-); bulk density follows.
        ebct: Empty bed contact time (s).
        velocity_sup: Superficial velocity (m/s).
        conc_ratio_replace: Effluent/feed ratio at which the bed is replaced (-).
        a0: Minimum-Stanton regression slope (-).
        a1: Minimum-Stanton regression intercept (-).
        b0: Throughput regression parameter (-).
        b1: Throughput regression parameter (-).
        b2: Throughput regression parameter (-).
        b3: Throughput regression parameter (-).
        b4: Throughput regression parameter (-).
        n_elements: Breakthrough points of the steady-state average (-).
        conc_ratio_start: First breakthrough point (-).
        rho: Liquid density for the volumetric flow (kg/m^3).
    """

    target: str = "solute"
    mw_target: float | Array = 0.08
    freund_k: float | Array = 10.0
    freund_ninv: float | Array = 0.9
    kf: float | Array = 5e-5
    ds: float | Array = 2e-13
    particle_dens_app: float | Array = 750.0
    particle_dia: float | Array = 0.001
    bed_voidage: float | Array = 0.4
    ebct: float | Array = 600.0
    velocity_sup: float | Array = 0.01
    conc_ratio_replace: float | Array = 0.5
    a0: float | Array = 3.68421
    a1: float | Array = 13.1579
    b0: float | Array = 0.784576
    b1: float | Array = 0.239663
    b2: float | Array = 0.484422
    b3: float | Array = 0.003206
    b4: float | Array = 0.134987
    n_elements: int = 5
    conc_ratio_start: float | Array = 0.01
    rho: float | Array = RHO_DEFAULT


def gac_performance(feed: Stream, p: GACParams) -> dict:
    """Solve the CPHSDM for a feed (explicit; differentiable).

    Returns:
        dict of the model's quantities (``equil_conc``, ``dg``, ``N_Bi``,
        ``min_N_St``, ``min_ebct``, ``throughput``, ``operational_time`` (s),
        ``bed_volumes_treated``, ``bed_length``/``_area``/``_diameter``/
        ``_volume``, ``bed_mass_gac`` (kg), ``gac_usage_rate`` (kg/s),
        ``conc_ratio_avg``, ``mass_adsorbed`` (kg), ``removal_fraction``,
        ``constant_pattern`` (bool)).
    """
    t = p.target
    Q = volumetric_flow(feed, {t: p.mw_target}, p.rho)
    C0 = feed[f"F_{t}"] * p.mw_target / Q
    eps = p.bed_voidage
    q_e = p.freund_k * C0 ** p.freund_ninv
    dg = p.particle_dens_app * q_e * (1.0 - eps) / (eps * C0)
    Bi = p.kf * (p.particle_dia / 2.0) * (1.0 - eps) / (p.ds * dg * eps)
    St_min = p.a0 * Bi + p.a1
    ebct_min = St_min * (p.particle_dia / 2.0) / ((1.0 - eps) * p.kf)
    tau, tau_min = eps * p.ebct, eps * ebct_min

    def throughput(c):
        return p.b0 + p.b1 * c ** p.b2 + p.b3 / (1.01 - c ** p.b4)

    def op_time(c):
        return tau_min * (dg + 1.0) * throughput(c) + (tau - tau_min) * (dg + 1.0)

    n = int(p.n_elements)
    if n < 2:
        raise ValueError(f"n_elements must be at least 2, got {n}")
    c_pts = p.conc_ratio_start + jnp.arange(n) * (p.conc_ratio_replace - p.conc_ratio_start) / (n - 1)
    t_pts = op_time(c_pts)
    c_all = jnp.concatenate([jnp.zeros(1), c_pts])
    t_all = jnp.concatenate([jnp.zeros(1), t_pts])
    c_avg = jnp.sum((t_all[1:] - t_all[:-1]) / t_all[-1] * 0.5 * (c_all[1:] + c_all[:-1]))
    t_op = op_time(p.conc_ratio_replace)
    L = p.velocity_sup * p.ebct
    A = Q / p.velocity_sup
    V = L * A
    rho_b = p.particle_dens_app * (1.0 - eps)
    m_gac = rho_b * V
    removed = (1.0 - c_avg) * feed[f"F_{t}"] * p.mw_target   # kg/s
    return {
        "flow_vol": Q, "conc_in": C0, "equil_conc": q_e, "dg": dg, "N_Bi": Bi,
        "min_N_St": St_min, "min_ebct": ebct_min, "throughput": throughput(p.conc_ratio_replace),
        "min_residence_time": tau_min, "residence_time": tau,
        "min_operational_time": tau_min * (dg + 1.0) * throughput(p.conc_ratio_replace),
        "operational_time": t_op, "bed_volumes_treated": t_op * eps / tau,
        "bed_length": L, "bed_area": A, "bed_diameter": jnp.sqrt(4.0 * A / jnp.pi),
        "bed_volume": V, "bed_mass_gac": m_gac, "gac_usage_rate": m_gac / t_op,
        "conc_ratio_avg": c_avg, "mass_adsorbed": removed * t_op,
        "removal_fraction": 1.0 - c_avg, "constant_pattern": p.ebct >= ebct_min,
    }


class GACAdsorber:
    """Granular activated carbon contactor (WaterTAP's CPHSDM).

    Inlet: water with the target component. Outlets: ``treated`` (the
    target at the bed-life-averaged effluent ratio) and ``removed`` (the
    adsorbed target, same T and P).

    Example:
        >>> from difflow import make_stream
        >>> feed = make_stream({"H2O": 2433.8, "solute": 0.0548}, T=298.15, P=101325.0)
        >>> treated, removed, info = GACAdsorber(GACParams())(feed)
        >>> 0.0 < float(info["removal_fraction"]) < 1.0
        True
    """

    symbol = "GAC"
    equations = [
        r"q_e = K C_0^{1/n},\quad D_g = \frac{\rho_a q_e (1-\varepsilon)}{\varepsilon C_0},\quad "
        r"Bi = \frac{k_f (d_p/2)(1-\varepsilon)}{D_s D_g \varepsilon}",
        r"St_{min} = a_0 Bi + a_1,\quad EBCT_{min} = \frac{St_{min}\, d_p/2}{(1-\varepsilon) k_f}",
        r"T(c) = b_0 + b_1 c^{b_2} + \frac{b_3}{1.01 - c^{b_4}},\quad "
        r"t(c) = \varepsilon (D_g + 1)\,[EBCT_{min} T(c) + EBCT - EBCT_{min}]",
        r"\bar c = \textstyle\sum \frac{\Delta t}{t_{replace}} \frac{c_{k} + c_{k-1}}{2}",
    ]
    assumptions = [
        "Constant-pattern HSDM (valid for EBCT >= EBCT_min); single adsorbing solute; Freundlich isotherm.",
        "Fixed film-transfer and surface-diffusion coefficients; regression constants are inputs.",
        "Steady-state removal = 1 - bed-life-averaged effluent ratio (trapezoidal rule), as WaterTAP.",
        "Equations as coded in WaterTAP 1.8.0; unverified against Hand et al. (1984).",
    ]
    references = [
        "Hand, Crittenden & Thacker, J. Environ. Eng. 110 (1984) 440-456, doi:10.1061/(ASCE)0733-9372(1984)110:2(440).",
        "WaterTAP 1.8.0 unit_models/gac.py (equations as coded).",
    ]
    parameter_symbols = {"freund_k": "K", "freund_ninv": "1/n", "kf": "k_f", "ds": "D_s",
                         "bed_voidage": "eps", "ebct": "EBCT", "conc_ratio_replace": "c_r"}
    parameter_units = {"target": "-", "mw_target": "kg/mol", "freund_k": "(m^3/kg)^(1/n)",
                       "freund_ninv": "-", "kf": "m/s", "ds": "m^2/s",
                       "particle_dens_app": "kg/m^3", "particle_dia": "m", "bed_voidage": "-",
                       "ebct": "s", "velocity_sup": "m/s", "conc_ratio_replace": "-",
                       "a0": "-", "a1": "-", "b0": "-", "b1": "-", "b2": "-", "b3": "-", "b4": "-",
                       "n_elements": "-", "conc_ratio_start": "-", "rho": "kg/m^3"}
    outlet_roles = ["treated", "removed"]

    def __init__(self, params: GACParams):
        """Initialize.

        Args:
            params: :class:`GACParams`.
        """
        if int(params.n_elements) < 2:
            raise ValueError(f"n_elements must be at least 2, got {params.n_elements}")
        self.params = params

    def __call__(self, feed: Stream) -> tuple[Stream, Stream, dict]:
        """Adsorb the target from a feed.

        Returns:
            ``(treated, removed, info)``; ``info`` from :func:`gac_performance`.
        """
        t = self.params.target
        if f"F_{t}" not in feed:
            raise ValueError(f"the feed has no F_{t} (GACParams.target = {t!r})")
        info = gac_performance(feed, self.params)
        treated = scale_component(feed, t, info["conc_ratio_avg"])
        removed = scale_component(only(feed, [t]), t, info["removal_fraction"])
        return treated, removed, info
