"""The refinery amine unit: contactors, rich flash, lean/rich exchanger, regenerator.

::

    sour gas k --> [absorber k] --> sweet gas k
                       ^   | rich k
    lean amine --------+   v
        ^           (mix) --> [rich flash] --> flash gas (to fuel)
        |                        |
    [trim cooler]                v
        ^              [lean/rich exchanger] (rich heated to T_lean_hot - approach)
        |                        |
    [makeup water]               v
        ^             [regenerator] --> acid gas (to the SRU)
        +----- lean ------------+

The lean loadings and the reboiler temperature are the tear, closed by
Newton (three unknowns: ``ln alpha_CO2``, ``ln alpha_H2S`` of the lean
amine, and the lean hot-end temperature the exchanger approaches); one
implicit step then gives exact derivatives with respect to the
circulation, the reboiler duty, the feeds and every parameter. Makeup
water restores the lean solvent's design amine strength (a negative
makeup is a water purge); the amine itself is conserved (no losses).

A library, not a palette operation (the gas plant's ``AmineTreater`` stays
the fixed-fraction shortcut).
"""

from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax import lax

from difflow_cc.equilibrium import acid_gas as ag
from difflow_refinery.treating.absorber import CO2, H2S, WATER, TrayParams, absorb
from difflow_refinery.treating.regenerator import (M_W, RegeneratorResult, liquid_enthalpy,
                                                   regenerate)
from difflow.water import water_saturation_pressure


@dataclass(frozen=True)
class Contactor:
    """One absorber on the lean header.

    Attributes:
        name: Contactor name (the key of its sour gas).
        circulation: Amine flow to it (mol amine/s).
        n_stages: Actual trays.
        tray: :class:`TrayParams`.
        T: Column temperature (K); default the lean temperature.
    """

    name: str
    circulation: float
    n_stages: int
    tray: TrayParams
    T: float | None = None


@dataclass(frozen=True)
class AmineUnitParams:
    """Parameters of :func:`solve_amine_unit`.

    Attributes:
        amine: ``"MDEA"`` or ``"DEA"``.
        w_amine: Design amine mass fraction of the lean solvent (acid-gas free).
        contactors: The absorbers.
        Q_reboiler: Reboiler duty (W).
        P_regenerator: Regenerator pressure (Pa).
        n_regenerator: Theoretical stages of the regenerator (incl. reboiler).
        T_condenser: Regenerator overhead condenser temperature (K).
        P_flash: Rich flash drum pressure (Pa).
        approach: Lean/rich exchanger hot-end approach (K).
        T_lean: Lean amine temperature after the trim cooler (K).
    """

    amine: str
    w_amine: float
    contactors: tuple = field(default_factory=tuple)
    Q_reboiler: float = 5.0e6
    P_regenerator: float = 1.9e5
    n_regenerator: int = 10
    T_condenser: float = 318.15
    P_flash: float = 5.0e5
    approach: float = 10.0
    T_lean: float = 313.15


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AmineUnitResult:
    """What :func:`solve_amine_unit` returns."""

    sweet: dict            # contactor name -> sweet gas stream
    acid_gas: dict         # F_ CO2/H2S/water (mol/s), T, P: the SRU feed
    flash_gas: dict        # F_ flows (mol/s), T, P
    lean: dict             # lean amine at the contactors (L, W, CO2, H2S, T)
    rich: dict             # mixed rich amine before the flash
    makeup_water: jax.Array
    duties: dict           # reboiler, condenser, exchanger, trim cooler (W)
    absorbers: dict        # name -> AbsorberResult
    regenerator: RegeneratorResult
    converged: jax.Array
    residual: jax.Array


def _newton_seq(res, z0, cap, max_iter=40, tol=1e-10):
    """Damped Newton with a SEQUENTIAL backtracking search (halving up to 12
    times). The acid-gas Newton evaluates its 16 trial steps at once with
    ``vmap``; around a whole loop of nested solves that batches every inner
    ``while_loop`` 16 ways and the compile grows with it."""

    def norm(z):
        r = res(z)
        return jnp.where(jnp.all(jnp.isfinite(r)), jnp.max(jnp.abs(r)), jnp.inf)

    def body(state):
        z, k, e = state
        r, J = res(z), jax.jacfwd(res)(z)
        dz = -jnp.linalg.solve(J, r)
        dz = jnp.where(jnp.all(jnp.isfinite(dz)), dz, 0.0)
        dz = dz / jnp.maximum(1.0, jnp.max(jnp.abs(dz)) / cap)

        def ls_cond(c):
            a, en, i = c
            return (en >= e) & (i < 12)

        def ls_body(c):
            a, _, i = c
            a = 0.5 * a
            return a, norm(z + a * dz), i + 1

        a, en, _ = lax.while_loop(ls_cond, ls_body, (1.0, norm(z + dz), 0))
        return z + a * dz, k + 1, en

    def cond(state):
        _, k, e = state
        return (k < max_iter) & (e > tol)

    return lax.while_loop(cond, body, (z0, jnp.asarray(0), norm(z0)))


def lean_water(amine, L, w_amine):
    """Water (mol/s) with ``L`` mol/s of amine at mass fraction ``w_amine``."""
    MW = ag.ke_amine(amine).MW * 1e-3
    return L * MW * (1.0 - w_amine) / w_amine / M_W


def rich_flash(amine, rich: dict, dissolved: dict, P, max_iter: int = 60):
    """Isothermal flash of a rich solution with dissolved light gases at ``P``.

    The dissolved non-acid gases leave entirely (they are far above their
    solubility at the flash pressure); the acid gases and water leave in
    equilibrium with the flashed liquid. Unknowns ``ln V``, ``ln alpha_CO2``,
    ``ln alpha_H2S``; equations the two acid-gas balances and
    ``sum p = P``. Returns ``(liquid, vapor, info)``.
    """
    am = ag.ke_amine(amine)
    L, W, T = rich["L"], rich["W"], rich["T"]
    n_in = sum(dissolved.values()) if dissolved else jnp.asarray(0.0)
    theta = {k: jnp.asarray(v, dtype=float) for k, v in
             dict(C=rich["CO2"], S=rich["H2S"], n=n_in, P=P, L=L, W=W, T=T).items()}
    # an absent acid gas is held at a trace (log loadings); the trace added
    # is taken back out of the liquid below, so the balances stay exact
    trace = {k: jnp.maximum(1e-12 * theta["L"] - theta[k], 0.0) for k in ("C", "S")}
    theta["C"] = theta["C"] + trace["C"]
    theta["S"] = theta["S"] + trace["S"]

    def pressures(z, th):
        aC, aS = jnp.exp(z[1]), jnp.exp(z[2])
        sp = ag.ke_speciate(am, th["T"], th["L"] / (th["W"] * M_W), aC, aS)
        pw = th["W"] / (th["W"] + th["L"]) * water_saturation_pressure(th["T"])
        return aC, aS, sp["p_CO2"], sp["p_H2S"], pw, sp["converged"]

    def res(z, th):
        V = jnp.exp(z[0])
        aC, aS, pC, pS, pw, _ = pressures(z, th)
        acid = th["C"] + th["S"]
        return jnp.stack([(th["C"] - th["L"] * aC - V * pC / th["P"]) / acid,
                          (th["S"] - th["L"] * aS - V * pS / th["P"]) / acid,
                          (th["n"] + V * (pC + pS + pw) / th["P"] - V) / (V + th["n"] + 1e-6 * acid)])

    th0 = jax.tree_util.tree_map(lax.stop_gradient, theta)
    z0 = jnp.stack([jnp.log(th0["n"] + 1e-3 * (th0["C"] + th0["S"])),
                    jnp.log(0.99 * th0["C"] / th0["L"]), jnp.log(0.99 * th0["S"] / th0["L"])])
    z_star, k, err = ag._newton(lambda z: res(z, th0), z0, cap=2.0, max_iter=max_iter, tol=1e-12)
    z = ag._implicit(res, z_star, theta)
    V = jnp.exp(z[0])
    aC, aS, pC, pS, pw, ok = pressures(z, theta)
    P = theta["P"]
    vapor = {f"F_{n}": v for n, v in dissolved.items()}
    vapor.update({f"F_{CO2}": V * pC / P, f"F_{H2S}": V * pS / P, f"F_{WATER}": V * pw / P,
                  "T": theta["T"], "P": P})
    liquid = dict(L=L, W=theta["W"] - V * pw / P, CO2=L * aC - trace["C"], H2S=L * aS - trace["S"],
                  T=theta["T"])
    return liquid, vapor, dict(converged=(err < 1e-9) & ok, residual=err)


def _heat_to(am, s, T_target, H_target, iters=4):
    """Temperature at which solution ``s`` has enthalpy ``H_target`` (Newton from ``T_target``)."""
    T = T_target
    f = lambda T: liquid_enthalpy(am, s["L"], s["W"], s["CO2"], s["H2S"], T)  # noqa: E731
    for _ in range(iters):
        T = T - (f(T) - H_target) / jax.grad(f)(T)
    return T


def solve_amine_unit(params: AmineUnitParams, gases: dict, circulation=None, Q_reboiler=None,
                     max_iter: int = 40) -> AmineUnitResult:
    """Solve the closed amine loop for the sour gases ``{contactor name: stream}``.

    Args:
        params: :class:`AmineUnitParams`.
        gases: ``{contactor name: sour gas stream}`` (``F_<name>`` mol/s, ``T``, ``P``).
        circulation: Optional amine flows (mol/s), one per contactor,
            overriding ``Contactor.circulation`` (pass these, not new
            params, to differentiate with respect to them).
        Q_reboiler: Optional reboiler duty (W) overriding the params'.

    Returns:
        :class:`AmineUnitResult`. The solve is jitted, cached by the unit's
        structure (amine, contactors' stages and trays, pressures, ...),
        so repeated calls with new flows, circulations or duties do not
        recompile.

    Example:
        ``docs/unit-operations-refinery.md`` ("Amine treating unit") and
        ``tests/refinery/test_treating.py``.
    """
    p = params
    names = [c.name for c in p.contactors]
    missing = set(names) - set(gases)
    if missing:
        raise ValueError(f"no sour gas for contactor(s) {sorted(missing)}")
    if circulation is None:
        circulation = tuple(c.circulation for c in p.contactors)
    if len(circulation) != len(names):
        raise ValueError(f"{len(circulation)} circulations for {len(names)} contactors")
    theta = dict(Q=p.Q_reboiler if Q_reboiler is None else Q_reboiler,
                 Ls=tuple(circulation), gases={n: dict(gases[n]) for n in names})
    theta = jax.tree_util.tree_map(lambda v: jnp.asarray(v, dtype=float), theta)
    key = dataclasses.replace(p, Q_reboiler=0.0, contactors=tuple(
        dataclasses.replace(c, circulation=0.0) for c in p.contactors))
    gas_keys = tuple((n, tuple(sorted(gases[n]))) for n in names)
    return _compiled(key, gas_keys, max_iter)(theta)


LEAN_FLOOR = 1e-12  # mol/mol amine: offset in the lean-loading tear's log; numerical, not physical


@functools.lru_cache(maxsize=32)
def _compiled(p: AmineUnitParams, gas_keys, max_iter):
    return jax.jit(functools.partial(_solve, p, max_iter=max_iter))


def _solve(p: AmineUnitParams, theta, max_iter: int = 40) -> AmineUnitResult:
    am = ag.ke_amine(p.amine)

    def run(lnx, th):
        # tear on ln(alpha + floor): finite when an acid gas is absent, and its
        # fixed point is the regenerated loading exactly (a floor held by a
        # max() would circulate, and so create, floor * L every pass)
        aC0 = jnp.maximum(jnp.exp(lnx[0]) - LEAN_FLOOR, 0.0)
        aS0 = jnp.maximum(jnp.exp(lnx[1]) - LEAN_FLOOR, 0.0)
        Ls = th["Ls"]
        Ltot = sum(Ls)
        abs_res = {}
        rich = dict(L=0.0, W=0.0, CO2=0.0, H2S=0.0)
        H_rich = 0.0
        dissolved = {}
        for c, L in zip(p.contactors, Ls):
            lean = dict(L=L, W=lean_water(am, L, p.w_amine), CO2=L * aC0, H2S=L * aS0, T=p.T_lean)
            r = absorb(am, th["gases"][c.name], lean, c.n_stages, c.tray, T=c.T)
            abs_res[c.name] = r
            for k in ("L", "W", "CO2", "H2S"):
                rich[k] = rich[k] + r.rich[k]
            H_rich = H_rich + liquid_enthalpy(am, r.rich["L"], r.rich["W"], r.rich["CO2"],
                                              r.rich["H2S"], r.rich["T"])
            for n, v in r.dissolved.items():
                dissolved[n] = dissolved.get(n, 0.0) + v
        rich["T"] = _heat_to(am, rich, jnp.asarray(p.T_lean, dtype=float), H_rich)
        flashed, flash_gas, finfo = rich_flash(am, rich, dissolved, p.P_flash)
        # lean/rich exchanger: rich heated to (lean hot - approach); the lean
        # hot end is the reboiler temperature, the third tear variable
        T_hot = lnx[2] * 100.0
        feed = dict(flashed, T=T_hot - p.approach)
        reg = regenerate(am, feed, th["Q"], p.P_regenerator, p.n_regenerator, p.T_condenser)
        lean_hot = reg.lean
        Qx = (liquid_enthalpy(am, feed["L"], feed["W"], feed["CO2"], feed["H2S"], feed["T"])
              - liquid_enthalpy(am, flashed["L"], flashed["W"], flashed["CO2"], flashed["H2S"],
                                flashed["T"]))
        out = dict(abs=abs_res, rich=rich, flashed=flashed, flash_gas=flash_gas, finfo=finfo,
                   reg=reg, Qx=Qx, feed=feed, L=Ltot)
        return jnp.stack([jnp.log(lean_hot["CO2"] / Ltot + LEAN_FLOOR),
                          jnp.log(lean_hot["H2S"] / Ltot + LEAN_FLOOR),
                          lean_hot["T"] / 100.0]), out

    def res(lnx, th):
        return run(lnx, th)[0] - lnx

    th0 = jax.tree_util.tree_map(lax.stop_gradient, theta)
    from difflow.water import water_saturation_temperature
    # start each lean loading at its feed's scale (capped at the usual 1e-3 /
    # 5e-3): with no source the loop map of a log loading is a pure shift
    # (the stripping factor), flat in the residual, so an absent acid gas
    # must start at zero, where its fixed point is
    Lt = sum(th0["Ls"])
    fed = {k: sum(g.get(f"F_{k}", 0.0) for g in th0["gases"].values()) for k in (CO2, H2S)}
    x0 = jnp.array([jnp.log(jnp.minimum(fed[CO2] / Lt, 1e-3) + LEAN_FLOOR),
                    jnp.log(jnp.minimum(fed[H2S] / Lt, 5e-3) + LEAN_FLOOR),
                    (water_saturation_temperature(p.P_regenerator) + 3.0) / 100.0])
    x_star, k, err = _newton_seq(lambda x: res(x, th0), x0, cap=2.0, max_iter=max_iter, tol=1e-10)
    x = ag._implicit(res, x_star, theta)
    _, o = run(x, theta)
    reg = o["reg"]
    lean_hot = reg.lean
    W_design = lean_water(am, sum(theta["Ls"]), p.w_amine)
    makeup = W_design - lean_hot["W"]
    lean = dict(L=lean_hot["L"], W=W_design, CO2=lean_hot["CO2"], H2S=lean_hot["H2S"], T=p.T_lean)
    # duties: the lean side gives up the exchanger duty, the trim cooler the rest
    H_hot = liquid_enthalpy(am, lean_hot["L"], lean_hot["W"], lean_hot["CO2"], lean_hot["H2S"],
                            lean_hot["T"]) + makeup * M_W * _water_h(p.T_lean)
    H_lean = liquid_enthalpy(am, lean["L"], lean["W"], lean["CO2"], lean["H2S"], lean["T"])
    trim = H_hot - o["Qx"] - H_lean
    a = reg.acid_gas
    acid_gas = {f"F_{CO2}": a["CO2"], f"F_{H2S}": a["H2S"], f"F_{WATER}": a["water"],
                "T": a["T"], "P": a["P"]}
    ok = (err < 1e-8) & reg.converged & o["finfo"]["converged"]
    for r in o["abs"].values():
        ok = ok & r.converged
    return AmineUnitResult(
        sweet={n: r.sweet for n, r in o["abs"].items()}, acid_gas=acid_gas,
        flash_gas=o["flash_gas"], lean=lean, rich=o["rich"], makeup_water=makeup,
        duties=dict(reboiler=theta["Q"], condenser=reg.condenser_duty, exchanger=o["Qx"],
                    trim_cooler=trim),
        absorbers=o["abs"], regenerator=reg, converged=ok, residual=err)


def _water_h(T):
    from difflow.water import water_liquid_enthalpy
    return water_liquid_enthalpy(T)
