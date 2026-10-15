"""Amine regenerator: an equilibrium-stage steam stripper with a reboiler.

Why not ``StageColumn``
-----------------------
The refinery's ``StageColumn`` (vacuum/gas plant) is built on
pseudocomponent ``ColumnThermo``/``CubicThermo``: Raoult or a cubic EOS,
log component flows, a furnace or condenser spec. A loaded amine solution
is a reactive electrolyte: the acid gases' volatility is the
Kent-Eisenberg back-pressure, which depends on BOTH loadings, the amine
molality and T at once, and the liquid's "components" are loadings, not
mole fractions. So the regenerator is its own small column on the same
idea (MESH equations, all stages solved together by Newton).

Model
-----
Stages ``j = 1..N`` from the top; stage ``N`` is the reboiler (duty ``Q``).
The rich solvent (and the condenser's water reflux) enters stage 1; vapor
leaving stage 1 is cooled in the overhead condenser to ``T_condenser``,
where the water condenses down to saturation and returns as reflux; the
acid gas (CO2, H2S, saturated with water) goes to the SRU. Every stage is
an EQUILIBRIUM stage: ``n_stages`` is a number of theoretical stages, as
in the usual design practice for regenerators (Kohl & Nielsen 1997, ch. 2).

Per stage the unknowns are ``ln alpha_CO2``, ``ln alpha_H2S``, ``T``,
``ln V`` (vapor) and ``ln W`` (liquid water), and the equations are the
CO2, H2S and water balances, the energy balance and the bubble point

    ``x_w Psat_w(T) + p_CO2(alpha, T, m) + p_H2S(alpha, T, m) = P``

with ``x_w = W/(W + L)`` (Raoult for water over the acid-gas-free
solvent; ``Psat`` IAPWS-IF97, :mod:`difflow.water`) and the acid-gas
pressures from :func:`difflow_cc.equilibrium.acid_gas.ke_speciate` at the
stage's own amine molality ``m = L/(W M_w)``. The amine is involatile.

Enthalpies (J/s; any reference works because every species keeps its own):

* water: steam-table liquid and saturated-vapor enthalpies
  (:mod:`difflow.water`);
* amine: ``L cp_amine (T - T_ref)``;
* CO2, H2S: ideal-gas sensible enthalpy from
  :mod:`difflow_refinery.thermochemistry`, minus the heat of absorption
  ``dH_abs`` (a positive number, released on absorption) when dissolved.

The heats of absorption are constants (:mod:`.properties`), checked in
the tests against the Kent-Eisenberg model's own van 't Hoff heat.

The solve is the damped Newton of :mod:`difflow_cc.equilibrium.acid_gas`
on stop-gradient inputs, then one implicit step, so derivatives with
respect to the duty, the feed and every parameter are exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax

from difflow.water import (water_liquid_enthalpy, water_saturation_pressure,
                           water_saturation_temperature, water_vapor_enthalpy)
from difflow_cc.equilibrium import acid_gas as ag
from difflow_refinery import thermochemistry as tc
from difflow_refinery.treating.properties import amine_properties

M_W = ag.M_WATER  # kg/mol
#: reference temperature of the amine sensible enthalpy (K); cancels in every balance
T_REF = 298.15


def gas_h(T):
    """Ideal-gas sensible enthalpies (J/mol) of CO2 and H2S at ``T``."""
    return tc.sensible_enthalpy("carbon_dioxide", T), tc.sensible_enthalpy("hydrogen_sulfide", T)


def liquid_enthalpy(am, L, W, nC, nS, T):
    """Enthalpy flow (W) of an amine solution: amine L, water W, CO2 nC, H2S nS (mol/s)."""
    pr = amine_properties(am.name)
    hC, hS = gas_h(T)
    return (W * M_W * water_liquid_enthalpy(T) + L * pr.cp_amine * (T - T_REF)
            + nC * (hC - pr.dH_CO2) + nS * (hS - pr.dH_H2S))


def vapor_enthalpy(vW, vC, vS, T):
    """Enthalpy flow (W) of a water/CO2/H2S vapor."""
    hC, hS = gas_h(T)
    return vW * M_W * water_vapor_enthalpy(T) + vC * hC + vS * hS


def stage_pressures(am, L, W, aC, aS, T):
    """``(p_w, p_CO2, p_H2S)`` (Pa) over a solution (vectorised over stages)."""
    m = L / (W * M_W)
    sp = jax.vmap(lambda T_, m_, c_, s_: ag.ke_speciate(am, T_, m_, c_, s_))(T, m, aC, aS)
    xw = W / (W + L)
    return xw * water_saturation_pressure(T), sp["p_CO2"], sp["p_H2S"], sp["converged"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RegeneratorResult:
    """What :func:`regenerate` returns (all arrays stage-ordered, top first)."""

    lean: dict        # L, W, CO2, H2S (mol/s), T (K)
    acid_gas: dict    # CO2, H2S, water (mol/s), T, P
    T: jax.Array      # stage temperatures (K)
    alpha_CO2: jax.Array
    alpha_H2S: jax.Array
    V: jax.Array      # vapor leaving each stage (mol/s)
    W: jax.Array      # liquid water leaving each stage (mol/s)
    reflux: jax.Array  # water reflux (mol/s)
    condenser_duty: jax.Array  # W, heat removed (positive)
    converged: jax.Array
    residual: jax.Array


def regenerate(amine, rich: dict, Q, P, n_stages: int = 10, T_condenser=318.15,
               max_iter: int = 100) -> RegeneratorResult:
    """Strip a rich amine solution.

    Args:
        amine: ``"MDEA"``/``"DEA"`` or a ``KentEisenbergAmine``.
        rich: ``{"L", "W", "CO2", "H2S", "T"}``: amine, water, CO2, H2S
            (mol/s) and temperature (K) of the rich feed (after the
            lean/rich exchanger).
        Q: Reboiler duty (W).
        P: Column pressure (Pa), uniform.
        n_stages: Theoretical stages including the reboiler.
        T_condenser: Overhead condenser / reflux drum temperature (K).

    Returns:
        :class:`RegeneratorResult`.

    Example:
        The J.M. Campbell MDEA case (50 wt%, 214 kPa, 1.1 lb steam/gal):

        >>> rich = dict(L=100.0, W=661.4, CO2=20.0, H2S=20.0, T=372.05)
        >>> r = regenerate("MDEA", rich, Q=6.37e6, P=2.14e5, n_stages=10, T_condenser=322.05)
        >>> bool(r.converged), round(float(r.T[-1]) - 273.15), float(r.alpha_H2S[-1]) < 0.005
        (True, 127, True)
    """
    am = ag.ke_amine(amine)
    N = int(n_stages)
    theta = dict(L=rich["L"], W=rich["W"], C=rich["CO2"], S=rich["H2S"], Tf=rich["T"],
                 Q=Q, P=P, Tc=T_condenser)
    theta = {k: jnp.asarray(v, dtype=float) for k, v in theta.items()}

    def unpack(z):
        z = z.reshape(N, 5)
        return jnp.exp(z[:, 0]), jnp.exp(z[:, 1]), z[:, 2] * 100.0, jnp.exp(z[:, 3]), jnp.exp(z[:, 4])

    def condenser(vW1, vC1, vS1, th):
        ps = water_saturation_pressure(th["Tc"])
        out_w = (vC1 + vS1) * ps / (th["P"] - ps)
        return vW1 - out_w, out_w

    def evaluate(z, th):
        aC, aS, T, V, W = unpack(z)
        L = th["L"]
        pw, pC, pS, ok = stage_pressures(am, L, W, aC, aS, T)
        P = th["P"]
        vW, vC, vS = V * pw / P, V * pC / P, V * pS / P
        R, out_w = condenser(vW[0], vC[0], vS[0], th)
        return dict(aC=aC, aS=aS, T=T, V=V, W=W, pw=pw, pC=pC, pS=pS, vW=vW, vC=vC, vS=vS,
                    R=R, out_w=out_w, ok=ok)

    def res(z, th):
        s = evaluate(z, th)
        L = th["L"]
        zero = jnp.zeros(1)
        # liquid entering each stage (stage 1: rich feed + reflux)
        nC_in = jnp.concatenate([th["C"][None], L * s["aC"][:-1]])
        nS_in = jnp.concatenate([th["S"][None], L * s["aS"][:-1]])
        W_in = jnp.concatenate([(th["W"] + s["R"])[None], s["W"][:-1]])
        H_in = jnp.concatenate([
            (liquid_enthalpy(am, L, th["W"], th["C"], th["S"], th["Tf"])
             + s["R"] * M_W * water_liquid_enthalpy(th["Tc"]))[None],
            liquid_enthalpy(am, L, s["W"][:-1], L * s["aC"][:-1], L * s["aS"][:-1], s["T"][:-1])])
        # vapor entering each stage (from below; none into the reboiler)
        up = lambda x: jnp.concatenate([x[1:], zero])  # noqa: E731
        Hv = vapor_enthalpy(s["vW"], s["vC"], s["vS"], s["T"])
        H_out = liquid_enthalpy(am, L, s["W"], L * s["aC"], L * s["aS"], s["T"]) + Hv
        Qj = jnp.zeros(N).at[-1].set(th["Q"])
        # each acid gas on its own feed (plus a sliver of the total), so a
        # trace one (an absent CO2 held at a floor) is still resolved
        acid = th["C"] + th["S"] + 1e-3 * L
        rC = (nC_in + up(s["vC"]) - L * s["aC"] - s["vC"]) / (th["C"] + 1e-6 * acid)
        rS = (nS_in + up(s["vS"]) - L * s["aS"] - s["vS"]) / (th["S"] + 1e-6 * acid)
        rW = (W_in + up(s["vW"]) - s["W"] - s["vW"]) / th["W"]
        rE = (H_in + up(Hv) + Qj - H_out) / (th["W"] * M_W * 2.3e5)
        rB = jnp.log((s["pw"] + s["pC"] + s["pS"]) / th["P"])
        # small-scale balances on the logs of loadings would be better
        # conditioned, but the mass balances must close to round-off
        return jnp.stack([rC, rS, rW, rE, rB], axis=1).ravel()

    th0 = jax.tree_util.tree_map(lax.stop_gradient, theta)
    z0 = initial_guess(am, th0, N)
    z_star, k, err = ag._newton(lambda z: res(z, th0), z0, cap=2.0, max_iter=max_iter, tol=1e-11)
    z = ag._implicit(res, z_star, theta)
    s = evaluate(z, theta)
    L = theta["L"]
    lean = dict(L=L, W=s["W"][-1], CO2=L * s["aC"][-1], H2S=L * s["aS"][-1], T=s["T"][-1])
    acid_gas = dict(CO2=s["vC"][0], H2S=s["vS"][0], water=s["out_w"], T=theta["Tc"], P=theta["P"])
    Hv1 = vapor_enthalpy(s["vW"][0], s["vC"][0], s["vS"][0], s["T"][0])
    Qc = Hv1 - vapor_enthalpy(s["out_w"], s["vC"][0], s["vS"][0], theta["Tc"]) \
        - s["R"] * M_W * water_liquid_enthalpy(theta["Tc"])
    return RegeneratorResult(lean=lean, acid_gas=acid_gas, T=s["T"], alpha_CO2=s["aC"],
                             alpha_H2S=s["aS"], V=s["V"], W=s["W"], reflux=s["R"],
                             condenser_duty=Qc,
                             converged=(err < 1e-8) & jnp.all(s["ok"]) & (s["R"] > 0),
                             residual=err)


def initial_guess(am, th, N):
    """Temperatures from the feed to the water boiling point; vapor from the
    duty over the latent heat; loadings falling geometrically down the column."""
    from difflow.water import water_latent_heat

    Tb = water_saturation_temperature(th["P"])
    T = jnp.linspace(jnp.minimum(th["Tf"] + 5.0, Tb - 2.0), Tb + 2.0, N)
    V = jnp.full(N, th["Q"] / (M_W * water_latent_heat(Tb)))
    a_rich = jnp.maximum((th["C"] + th["S"]) / th["L"], 1e-3)
    fC = jnp.maximum(th["C"], 1e-6 * th["L"]) / (th["C"] + th["S"] + 2e-6 * th["L"])
    a = a_rich * jnp.geomspace(0.5, 0.02, N)
    W = jnp.full(N, th["W"])
    return jnp.stack([jnp.log(a * fC), jnp.log(a * (1.0 - fC)), T / 100.0, jnp.log(V),
                      jnp.log(W)], axis=1).ravel()
