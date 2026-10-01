"""The FCC riser: 1-D adiabatic plug flow, integrated in height with diffrax.

State along the height ``z`` (m): the lump mass fractions ``y`` (feed basis)
and the catalyst residence time ``t_c`` (s). Equations::

    dt_c/dz = s / u_g                      (catalyst velocity u_g / s)
    dy/dz   = (C/O) * (dt_c/dz) * f_feed * a * r(y, T, phi)
    u_g     = n_vap R T / (P A)            (ideal-gas superficial velocity)
    T       = T_mix - F_o dH_c (1 - y_go) / Cp_tot     (adiabatic)

``r`` is the scheme's rate vector (:mod:`difflow_refinery.fcc.kinetics`),
``s`` the slip factor (catalyst moves ``s`` times slower than the gas), ``a``
the equilibrium-catalyst activity, ``f_feed`` the feed multiplier (Watson K
crackability times basic-nitrogen poisoning), ``phi`` the deactivation.
The ``dy/dz`` form is the catalyst-holdup formulation: ``dW = F_c dt_c`` of
catalyst meets ``F_o`` of oil, so ``F_o dy = r F_c dt_c``.

Energy: every hydrocarbon lump has the same constant vapour heat capacity
``cp_vapor``; each kg of gas oil converted -- to any product, coke included
-- absorbs the heat of cracking ``dH_c`` (J/kg, endothermic). With these
two assumptions the riser's enthalpy is invariant along ``z`` and the
temperature is the algebraic expression above, so it is not an ODE state.
``T_mix`` is the adiabatic mixing temperature of hot regenerated catalyst,
liquid feed (which vaporises, absorbing ``latent_heat`` at the preheat
temperature) and riser steam, before any cracking. Assumptions stated in
the docs: instantaneous vaporisation and mixing at the riser base; no
pressure drop (``P`` constant); the slip factor constant; the catalyst
and gas at the same temperature; no heat loss from the riser.

Integration: Tsit5 at a constant step (``n_steps`` steps over the height)
with ``diffrax.DirectAdjoint``, so the solution is differentiable in both
forward mode (the heat-balance Newton's Jacobian) and reverse mode (the
implicit-function VJP). A constant step makes the discrete solution a
smooth function of every parameter; at the default 120 steps the 5th-order
scheme's error in the outlet yields is below 1e-10 (checked against 480
steps in the tests).
"""

from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp
from jax import Array

from difflow_refinery.fcc import species as sp
from difflow_refinery.fcc.kinetics import GROUPS, R_GAS, LumpScheme, deactivation

T_REF = sp.T_REF


def mixing_temperature(p: dict, feed, CTO: Array, T_rg: Array) -> tuple[Array, Array]:
    """Riser-base mixing temperature (K) and the riser's total ``Cp`` (W/K)."""
    Fo = feed.mass
    Fc = CTO * Fo
    Fs = p["steam_ratio"] * Fo
    cp_tot = Fo * p["cp_vapor"] + Fc * p["cp_catalyst"] + Fs * p["cp_steam"]
    h_in = (Fo * (p["cp_vapor"] * (p["feed_T"] - T_REF) - p["latent_heat"])
            + Fs * p["cp_steam"] * (p["steam_T"] - T_REF)
            + Fc * p["cp_catalyst"] * (T_rg - T_REF))
    return T_REF + h_in / cp_tot, cp_tot


def inlet_yields(p: dict, feed) -> dict[str, Array]:
    """Feed-property yields formed at the riser base (mass fraction of feed).

    * additive (Conradson) coke ``ccr_to_coke * CCR``;
    * contaminant coke ``metals_coke * (Ni+V)`` and contaminant hydrogen
      ``metals_h2 * (Ni+V)`` (Ni+V as a mass fraction).

    All illustrative rules of thumb (see docs); taken out of the gas-oil
    lump before it cracks.
    """
    return {
        "coke_ccr": p["ccr_to_coke"] * feed.ccr,
        "coke_metals": p["metals_coke"] * feed.nickel_vanadium,
        "h2_metals": p["metals_h2"] * feed.nickel_vanadium,
    }


def feed_multiplier(p: dict, feed) -> Array:
    """Rate multiplier from the feed: crackability and basic-N poisoning.

    ``exp(kw_sensitivity (Kw - kw_ref)) / (1 + basic_N / nitrogen_poisoning)``
    with ``basic_N = basic_nitrogen_fraction * N``. Both factors are
    illustrative forms (paraffinic feeds crack faster; basic nitrogen
    titrates acid sites), not a published correlation.
    """
    crack = jnp.exp(p["kw_sensitivity"] * (feed.Kw - p["kw_ref"]))
    basic_n = p["basic_nitrogen_fraction"] * feed.nitrogen
    return crack / (1.0 + basic_n / p["nitrogen_poisoning"])


def lump_inverse_mw(scheme: LumpScheme, M: Array, group_mw: dict) -> Array:
    """mol/kg of each lump in the vapour (coke contributes nothing)."""
    inv = jnp.stack([1000.0 / group_mw[g] if g != "coke" else jnp.asarray(0.0)
                     for g in GROUPS])
    return M @ inv


def integrate(p: dict, feed, scheme: LumpScheme, M: Array, inv_mw: Array,
              CTO: Array, T_rg: Array, n_steps: int, deact: str,
              ts: Array | None = None) -> dict[str, Array]:
    """Integrate the riser. Returns outlet (or profile at ``ts``) state."""
    Fo = feed.mass
    Fc = CTO * Fo
    Fs = p["steam_ratio"] * Fo
    T_mix, cp_tot = mixing_temperature(p, feed, CTO, T_rg)
    inl = inlet_yields(p, feed)
    a_in = inl["coke_ccr"] + inl["coke_metals"] + inl["h2_metals"]
    area = jnp.pi * p["riser_diameter"] ** 2 / 4.0
    mult = p["activity"] * feed_multiplier(p, feed)
    coke_col = M[:, GROUPS.index("coke")]
    n_fixed = Fo * inl["h2_metals"] * 1000.0 / sp.MW["hydrogen"] + Fs * 1000.0 / sp.MW["water"]
    k_ref, Ea = p["k_ref"], p["Ea"]

    def temperature(y):
        return T_mix - Fo * p["heat_of_cracking"] * (1.0 - y[0]) / cp_tot

    def rhs(z, s, args):
        y, t_c = s[:-1], s[-1]
        T = temperature(y)
        n_vap = Fo * jnp.dot(y, inv_mw) + n_fixed
        u_g = n_vap * R_GAS * T / (p["riser_P"] * area)
        dtdz = p["slip"] / u_g
        coke_on_cat = (jnp.dot(y, coke_col) + inl["coke_ccr"] + inl["coke_metals"]) * Fo / Fc
        phi = deactivation(deact, t_c, coke_on_cat, p["deactivation_alpha"])
        dy = CTO * dtdz * scheme.rates(y, T, phi, k_ref, Ea, mult)
        return jnp.concatenate([dy, dtdz[None]])

    y0 = jnp.zeros(scheme.n).at[0].set(1.0 - a_in)
    s0 = jnp.concatenate([y0, jnp.zeros(1)])
    H = p["riser_height"]
    saveat = diffrax.SaveAt(t1=True) if ts is None else diffrax.SaveAt(ts=ts)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs), diffrax.Tsit5(), t0=0.0, t1=H, dt0=H / n_steps,
        y0=s0, saveat=saveat, stepsize_controller=diffrax.ConstantStepSize(),
        adjoint=diffrax.DirectAdjoint(), max_steps=n_steps + 1)
    S = sol.ys
    if ts is None:
        S = S[-1]
        y, t_c = S[:-1], S[-1]
        T = temperature(y)
        coke_on_cat = (jnp.dot(y, coke_col) + inl["coke_ccr"] + inl["coke_metals"]) * Fo / Fc
        return {"y": y, "t_c": t_c, "T_out": T, "T_mix": T_mix, "cp_tot": cp_tot,
                "phi_out": deactivation(deact, t_c, coke_on_cat, p["deactivation_alpha"]),
                "coke_on_cat": coke_on_cat, "multiplier": mult, **inl}
    y, t_c = S[:, :-1], S[:, -1]
    return {"z": ts, "y": y, "t_c": t_c, "T": jax.vmap(temperature)(y)}
