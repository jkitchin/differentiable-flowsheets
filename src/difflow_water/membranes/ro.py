"""Seawater reverse-osmosis stage (plug flow, solution-diffusion, film theory).

The physics and where each piece comes from (IDs are the reference table in
``docs/unit-operations-water.md``):

- Water and salt fluxes, solution-diffusion [M1]::

      J_w = A rho_w (P_f - P_p - (pi(S_m) - pi(S_p)))     kg/(m^2 s)
      J_s = B (C_m - C_p)                                 kg/(m^2 s)

  with ``C = S rho(S)`` the salt mass concentration, ``S_m`` at the membrane
  wall, ``S_p = J_s/(J_s + J_w)`` the local permeate. This is the form
  WaterTAP's RO models use (``A_comp`` m/(s Pa), ``B_comp`` m/s), so the
  same ``A`` and ``B`` mean the same thing in both codes.
- Concentration polarization, film theory::

      (C_m - C_p) / (C_b - C_p) = exp(J_v / k),   J_v = J_w / rho_w

- Seawater osmotic pressure and density from Sharqawy et al. 2010 [S1]
  (:mod:`difflow_water.chemistry.seawater`).
- Optional spiral-wound feed channel (``k=None`` and/or
  ``pressure_drop="calculated"``): hydraulic diameter of a spacer-filled
  channel, Schock & Miquel 1987 [M4] Eq. 17; Darcy friction factor
  ``f = 0.42 + 189.3/Re`` for a flat-sheet module (WaterTAP's default;
  WaterTAP cites Eq. S27 of the SI of Bartholomew et al. 2018 [M8]) or
  ``f = 6.23 Re^-0.3`` for a spiral-wound module ([M4] Eq. 24); Sherwood number
  ``Sh = 0.46 (Re Sc)^0.36``; diffusivity of NaCl as a quartic in mass
  fraction (Bartholomew & Mauter 2019 [M7] Eq. 6, at 25 C). All four are
  transcribed from WaterTAP 1.8.0 (``membrane_channel_base.py``,
  ``seawater_prop_pack.py``) and NOT read against the papers
  (unverified); WaterTAP does not name a source for the Sherwood
  correlation, and none is claimed here.

The stage is integrated along its membrane area with fixed-step RK4 on
``(water mass flow, salt mass flow, feed pressure)``; each step solves the
local flux for ``J_w`` by bracketed bisection plus a value-neutral Newton
step, which gives implicit-function gradients. Isothermal.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, make_stream
from difflow.units.membranes import _bisect_increasing, _implicit_step, _rk4
from difflow_water.chemistry import seawater as sw
from difflow_water.chemistry.seawater import MW_SEA_SALT, MW_WATER
from difflow_water.streams import SALT, WATER

# Bartholomew & Mauter (2019) Eq. 6 as coded in WaterTAP 1.8.0 (unverified):
# D = d0 + d1 w + d2 w^2 + d3 w^3 + d4 w^4, w = salt mass fraction, 25 C.
_DIFFUS = (1.51e-9, -2.00e-9, 3.01e-8, -1.22e-7, 1.53e-7)


def diffusivity_nacl(S) -> Array:
    """Salt diffusivity (m^2/s) as WaterTAP codes it, [M7] Eq. 6 (unverified).

    Fitted for NaCl at 25 C; WaterTAP's seawater package uses it for TDS.

    Args:
        S: Salt mass fraction (kg/kg).

    Returns:
        Diffusivity (m^2/s).
    """
    S = jnp.asarray(S, dtype=jnp.float64)
    d = _DIFFUS
    return d[0] + d[1] * S + d[2] * S**2 + d[3] * S**3 + d[4] * S**4


def spacer_hydraulic_diameter(channel_height, spacer_porosity) -> Array:
    """Hydraulic diameter of a spacer-filled channel, [M4] Eq. 17 (unverified).

    ``d_h = 4 eps / (2/h + (1 - eps) 8/h)``, as coded in WaterTAP.
    """
    h = jnp.asarray(channel_height, dtype=jnp.float64)
    eps = jnp.asarray(spacer_porosity, dtype=jnp.float64)
    return 4.0 * eps / (2.0 / h + (1.0 - eps) * 8.0 / h)


def local_flux(dP, S_b, T, A_w, B, k, n_iter: int = 80) -> dict:
    """Local water and salt fluxes at one point of a seawater RO membrane.

    Solves the solution-diffusion + film-theory equations in the module
    docstring for ``J_w`` on ``[0, A rho_w dP]`` (the equation is
    increasing in ``J_w``). The permeate side is taken at pure-water
    density for the volume flux (``S_p`` is of order 1e-4).

    Args:
        dP: Transmembrane hydraulic pressure difference ``P_f - P_p`` (Pa).
        S_b: Bulk feed-side salinity (kg/kg).
        T: Temperature (K).
        A_w: Water permeability coefficient (m/(s Pa)).
        B: Salt permeability coefficient (m/s).
        k: Feed-side mass-transfer coefficient (m/s).
        n_iter: Bisection halvings.

    Returns:
        dict with ``J_w``, ``J_s`` (kg/(m^2 s)), ``S_m`` (wall), ``S_p``
        (local permeate) salinities, ``cp_modulus`` ``C_m/C_b`` and
        ``delta_pi`` (Pa).
    """
    dP = jnp.maximum(jnp.asarray(dP, dtype=jnp.float64), 0.0)
    rho_w = sw.density_water(T)
    C_b = S_b * sw.density(T, S_b)
    Jmax = A_w * rho_w * dP

    def parts(Jw):
        E = jnp.exp(Jw / rho_w / k)
        BE = B * E
        # J_s solves J_s = B E (C_b - C_p) with C_p = rho_w J_s / (J_s + J_w):
        # J_s^2 + J_s (J_w + BE (rho_w - C_b)) - BE C_b J_w = 0, positive root
        b = Jw + BE * (rho_w - C_b)
        c = BE * C_b * Jw
        Js = 2.0 * c / (b + jnp.sqrt(b * b + 4.0 * c))
        S_p = jnp.where(Jw > 0, Js / (Js + jnp.where(Jw > 0, Jw, 1.0)), C_b / rho_w)
        C_p = S_p * rho_w
        C_m = C_p + (C_b - C_p) * E
        S_m = sw.salinity_from_mass_concentration(C_m, T)
        return Js, S_p, S_m, C_m

    def g(s):
        Jw = s * Jmax
        _, S_p, S_m, _ = parts(Jw)
        dpi = sw.osmotic_pressure(T, S_m) - sw.osmotic_pressure(T, S_p)
        return (Jw - A_w * rho_w * (dP - dpi)) / jnp.where(Jmax > 0, Jmax, 1.0)

    s0 = _bisect_increasing(g, 0.0, 1.0, n_iter)
    s = _implicit_step(g, s0)
    Jw = s * Jmax
    Js, S_p, S_m, C_m = parts(Jw)
    return {"J_w": Jw, "J_s": Js, "S_m": S_m, "S_p": S_p,
            "cp_modulus": C_m / C_b,
            "delta_pi": sw.osmotic_pressure(T, S_m) - sw.osmotic_pressure(T, S_p)}


@dataclass(repr=False)
class ROStageParams(ParamsMixin):
    """Parameters of a seawater reverse-osmosis stage.

    Attributes:
        A_w: Water permeability coefficient (m/(s Pa)); WaterTAP's ``A_comp``.
        B: Salt permeability coefficient (m/s); WaterTAP's ``B_comp``.
        area: Membrane area of the stage (m^2).
        width: Feed-channel width (m); the channel length is ``area/width``.
            Used only by the channel correlations.
        permeate_pressure: Permeate pressure (Pa).
        k: Feed-side mass-transfer coefficient (m/s); None to compute it from
            the spacer-filled channel correlations.
        pressure_drop: "none", "calculated" (channel friction), or a fixed
            total feed-side pressure drop (Pa) spread evenly over the area.
        module_type: "flat_sheet" (WaterTAP's default) or "spiral_wound";
            selects the friction-factor correlation.
        channel_height: Feed-channel height (m).
        spacer_porosity: Spacer porosity (-).
        n_steps: RK4 steps along the membrane area.
        scaling: Report mineral saturation indices of the concentrate and of
            the membrane wall at the outlet (Pitzer speciation; adds a few
            seconds of compile time).
        feed_totals: Element totals of the feed (mol/kg water, see
            :mod:`difflow_water.chemistry.speciation`) for ``scaling``; None
            for Reference-Composition seawater at the feed salinity and
            ``feed_pH``.
        feed_pH: Feed pH (MacInnes scale) used when ``feed_totals`` is None
            (ILLUSTRATIVE default 8.1).
    """

    A_w: float | Array
    B: float | Array
    area: float | Array
    width: float | Array = 5.0
    permeate_pressure: float | Array = 101325.0
    k: float | Array | None = None
    pressure_drop: str | float | Array = "calculated"
    module_type: str = "flat_sheet"
    channel_height: float | Array = 1e-3
    spacer_porosity: float | Array = 0.85
    n_steps: int = 40
    scaling: bool = False
    feed_totals: dict | None = None
    feed_pH: float | Array = 8.1


class ROStage:
    """Seawater reverse-osmosis stage: plug flow along the membrane area.

    Feed is a seawater stream (``H2O`` and ``TDS``, see
    :mod:`difflow_water.streams`) at the stage inlet pressure. Returns the
    concentrate (retentate) and permeate.

    Example:
        >>> from difflow_water.streams import make_seawater_stream
        >>> feed = make_seawater_stream(flow_vol=1e-3, salinity=0.035, P=60e5)
        >>> ro = ROStage(ROStageParams(A_w=4.2e-12, B=3.5e-8, area=50.0))
        >>> conc, perm, info = ro(feed)
        >>> 0 < float(info["recovery_mass_water"]) < 1
        True
    """

    symbol = "RO"
    equations = [
        r"J_w = A\,\rho_w\,(P_f - P_p - (\pi_m - \pi_p)),\qquad J_s = B\,(C_m - C_p)",
        r"\frac{C_m - C_p}{C_b - C_p} = e^{J_v/k}\qquad \text{(film theory)}",
        r"\pi = \phi(T,S)\, m\, \rho_w R T\qquad \text{(Sharqawy et al. 2010, Eq. 49)}",
        r"\frac{dP}{dx} = -\frac{f}{2 d_h}\rho u^2,\quad f = 0.42 + 189.3/Re\ \text{or}\ 6.23\,Re^{-0.3}",
    ]
    assumptions = [
        "Solution-diffusion with constant A and B; isothermal.",
        "Plug flow along the membrane area; permeate fully mixed at permeate_pressure.",
        "Seawater as one lumped salt (TDS) at Reference Composition.",
        "Channel correlations (k, friction) as coded in WaterTAP 1.8.0; unverified against their papers.",
    ]
    references = [
        "Wijmans & Baker, J. Membr. Sci. 107 (1995) 1-21, doi:10.1016/0376-7388(95)00102-I.",
        "Sharqawy, Lienhard & Zubair, Desalin. Water Treat. 16 (2010) 354-380, doi:10.5004/dwt.2010.1079.",
        "Schock & Miquel, Desalination 64 (1987) 339-352, doi:10.1016/0011-9164(87)90107-X.",
    ]
    parameter_symbols = {"A_w": "A", "B": "B", "area": "A_m", "k": "k",
                         "permeate_pressure": "P_p"}
    parameter_units = {"A_w": "m/(s Pa)", "B": "m/s", "area": "m^2", "width": "m",
                       "permeate_pressure": "Pa", "k": "m/s", "pressure_drop": "Pa",
                       "module_type": "-", "scaling": "-", "feed_totals": "mol/kg",
                       "feed_pH": "-", "channel_height": "m", "spacer_porosity": "-", "n_steps": "-"}
    numerical_method = ("RK4 along the area; local flux by bisection with a "
                        "value-neutral Newton step (implicit-function gradients).")
    outlet_roles = ["concentrate", "permeate"]

    def __init__(self, params: ROStageParams):
        """Initialize.

        Args:
            params: :class:`ROStageParams`.
        """
        if isinstance(params.pressure_drop, str) and params.pressure_drop not in ("none", "calculated"):
            raise ValueError("pressure_drop must be 'none', 'calculated' or a number (Pa)")
        if params.module_type not in ("flat_sheet", "spiral_wound"):
            raise ValueError("module_type must be 'flat_sheet' or 'spiral_wound'")
        self.params = params

    def __call__(self, feed: Stream) -> tuple[Stream, Stream, dict]:
        """Desalinate a seawater feed.

        Args:
            feed: Seawater stream (``F_H2O``, ``F_TDS``) at the feed pressure.

        Returns:
            ``(concentrate, permeate, info)``. ``info`` has
            ``recovery_mass_water`` (WaterTAP's water recovery),
            ``recovery_vol``, ``salt_rejection`` (``1 - S_p/S_f``),
            ``salinity_feed``/``_concentrate``/``_permeate``, ``deltaP``
            (feed-side pressure change, Pa, negative), ``flux_mass_water_avg``
            (kg/(m^2 s)), ``max_cp_modulus``, ``Re_in``/``Re_out`` (when the
            channel is modelled), ``min_driving_pressure`` (``P - P_p -
            pi_m``, Pa, at the outlet) and ``profile``. With
            ``scaling=True`` also ``SI_concentrate`` and ``SI_wall_outlet``
            (dict of saturation indices per mineral) and ``pH_concentrate``,
            ``pH_wall_outlet``: every solute is taken as rejected alike
            (totals scale with the sea-salt molality) and CO2 stays in.
        """
        p = self.params
        if f"F_{WATER}" not in feed or f"F_{SALT}" not in feed:
            raise ValueError(f"the RO feed needs 'F_{WATER}' and 'F_{SALT}'")
        f64 = lambda v: jnp.asarray(v, dtype=jnp.float64)  # noqa: E731
        calc_k = p.k is None
        calc_dp = isinstance(p.pressure_drop, str) and p.pressure_drop == "calculated"
        fixed_dp = not isinstance(p.pressure_drop, str)
        num = {
            "A_w": f64(p.A_w), "B": f64(p.B), "area": f64(p.area), "width": f64(p.width),
            "Pp": f64(p.permeate_pressure), "k": f64(0.0 if calc_k else p.k),
            "dp": f64(p.pressure_drop if fixed_dp else 0.0),
            "h": f64(p.channel_height), "eps": f64(p.spacer_porosity), "feed_pH": f64(p.feed_pH),
        }
        x = (feed[f"F_{WATER}"] * MW_WATER, feed[f"F_{SALT}"] * MW_SEA_SALT,
             f64(feed["P"]), f64(feed["T"]))
        static = (calc_k, calc_dp, fixed_dp, p.module_type, int(p.n_steps), bool(p.scaling))
        totals = None if p.feed_totals is None else {k: f64(v) for k, v in p.feed_totals.items()}
        out = _run_stage(x, num, static)
        T = x[3]
        if p.scaling:
            _scaling(out["info"], totals, num["feed_pH"], T)
        conc = make_stream({WATER: out["mwc"] / MW_WATER, SALT: out["msc"] / MW_SEA_SALT}, T, out["Pc"])
        perm = make_stream({WATER: out["mwp"] / MW_WATER, SALT: out["msp"] / MW_SEA_SALT}, T, num["Pp"])
        return conc, perm, out["info"]


def _channel(m_tot, S, T, num, module_type):
    """Mass-transfer coefficient, dP/dx (Pa/m) and Re of the feed channel."""
    rho = sw.density(T, S)
    mu = sw.viscosity(T, S)
    dh = spacer_hydraulic_diameter(num["h"], num["eps"])
    A_x = num["h"] * num["width"] * num["eps"]
    Re = m_tot * dh / (A_x * mu)
    D = diffusivity_nacl(S)
    Sc = mu / (rho * D)
    k = 0.46 * (Re * Sc) ** 0.36 * D / dh
    u = m_tot / rho / A_x
    if module_type == "spiral_wound":
        f = 6.23 * Re ** -0.3
    else:
        f = 0.42 + 189.3 / Re
    return k, -0.5 * f * rho * u**2 / dh, Re


@partial(jax.jit, static_argnums=(2,))
def _run_stage(x, num, static):
    """The stage as one jitted function (static: the configuration flags)."""
    calc_k, calc_dp, fixed_dp, module_type, n_steps, _ = static
    mw0, ms0, P0, T = x
    Pp, area = num["Pp"], num["area"]

    def local(y):
        mw, ms, P = y[0], y[1], y[2]
        mt = mw + ms
        S = ms / mt
        if calc_k or calc_dp:
            k_c, dpdx, Re = _channel(mt, S, T, num, module_type)
        else:
            k_c, dpdx, Re = None, jnp.asarray(0.0), jnp.asarray(jnp.nan)
        k = k_c if calc_k else num["k"]
        f = local_flux(P - Pp, S, T, num["A_w"], num["B"], k)
        if calc_dp:
            dPda = dpdx / num["width"]
        elif fixed_dp:
            dPda = -num["dp"] / area
        else:
            dPda = jnp.asarray(0.0)
        return f, dPda, k, Re, S

    def rhs(y):
        f, dPda, _, _, _ = local(y)
        return jnp.stack([-f["J_w"], -f["J_s"], dPda * jnp.ones_like(f["J_w"])])

    h = area / n_steps
    ys = _rk4(rhs, jnp.stack([mw0, ms0, P0]), h, n_steps)
    mwc, msc, Pc = ys[-1, 0], ys[-1, 1], ys[-1, 2]
    mwp, msp = mw0 - mwc, ms0 - msc
    flux, _, kk, Re, S_prof = jax.vmap(local)(ys)
    S_f = ms0 / (mw0 + ms0)
    S_p = msp / (mwp + msp)
    S_c = msc / (mwc + msc)
    vol_f = (mw0 + ms0) / sw.density(T, S_f)
    vol_p = (mwp + msp) / sw.density(T, S_p)
    info = {
        "recovery_mass_water": mwp / mw0,
        "recovery_vol": vol_p / vol_f,
        "salt_rejection": 1.0 - S_p / S_f,
        "salinity_feed": S_f,
        "salinity_concentrate": S_c,
        "salinity_permeate": S_p,
        "deltaP": Pc - P0,
        "flux_mass_water_avg": mwp / area,
        "max_cp_modulus": jnp.max(flux["cp_modulus"]),
        "min_driving_pressure": Pc - Pp - sw.osmotic_pressure(T, flux["S_m"][-1]),
        "permeate_flow_vol": vol_p,
        "profile": {
            "area": jnp.linspace(0.0, 1.0, n_steps + 1) * area,
            "flow_mass_water": ys[:, 0], "flow_mass_salt": ys[:, 1],
            "pressure": ys[:, 2], "salinity": S_prof,
            "J_w": flux["J_w"], "J_s": flux["J_s"],
            "salinity_wall": flux["S_m"], "cp_modulus": flux["cp_modulus"],
            "k": kk,
        },
    }
    if calc_k or calc_dp:
        info["Re_in"], info["Re_out"] = Re[0], Re[-1]
    return {"mwc": mwc, "msc": msc, "Pc": Pc, "mwp": mwp, "msp": msp, "info": info}


def _scaling(info, feed_totals, feed_pH, T):
    """Saturation indices of the concentrate and the outlet wall.

    Outside the stage's jit on purpose: the speciation solve is jitted on
    its own, so it compiles once per process and is reused by every stage
    and every solve, rather than being inlined (three times) into each
    stage's compile.
    """
    from difflow_water.chemistry import speciation as spc
    S_f = info["salinity_feed"]
    tot = feed_totals if feed_totals is not None else spc.seawater_feed(S_f, feed_pH, T)
    m_f = sw.molality(S_f)
    for key, S_x in (("SI_concentrate", info["salinity_concentrate"]),
                     ("SI_wall_outlet", info["profile"]["salinity_wall"][-1])):
        r = spc.speciate(spc.concentrate(tot, sw.molality(S_x) / m_f), T)
        info[key] = r["SI"]
        info[key.replace("SI", "pH")] = r["pH_macinnes"]
