"""Steady-state hydraulics of a water distribution network (JAX, differentiable).

One equation set, solved equation-oriented for junction heads and link flows
together (as EPANET's gradient method does [N8]; no tears):

- link: ``H_start - H_end = h_loss(Q) - h_gain(Q) + t - u``
- junction: ``sum(Q in) - sum(Q out) - demand(p) - emitter(p) = 0``

with ``t >= 0`` an extra (throttling) loss on control valves and ``u >= 0`` a
blocking head on links that cannot run backwards (check valves, pumps, PRVs,
PSVs). Their switching is written as complementarity:

- check valve / pump / PRV / PSV: ``u >= 0  perp  Q >= 0``
- PRV: ``t >= 0  perp  H_set - H_end >= 0`` (throttles only to hold the
  downstream pressure at its setting)
- PSV: ``t >= 0  perp  H_start - H_set >= 0``
- FCV: ``t >= 0  perp  Q_set - Q >= 0``

each pair through the smoothed Fischer-Burmeister function
``phi(a, b) = a + b - sqrt(a^2 + b^2 + 2 mu^2)``. Two formulations
(DESIGN.md Section 3a):

- ``"complementarity"``: ``mu`` is continued down to 1e-9 (in scaled units),
  so the valves switch sharply, as EPANET's status logic does; gradients are
  valid away from a switching point.
- ``"smooth"``: ``mu`` stays at ``smoothing`` (default 1e-3), so the
  equations are differentiable everywhere, at the price of a small rounding
  of each switch.

Physics, from the EPANET 2.2 manual [N8] (content verified, Section 13.1
and Table 3.1; its US-unit constants converted to SI here):

- Hazen-Williams: ``h = 10.667 C^-1.852 d^-4.871 L |Q|^0.852 Q`` (4.727 in ft
  and cfs).
- Darcy-Weisbach: ``h = 0.0827 f L d^-5 |Q| Q`` (0.0252 in ft and cfs), f by
  Hagen-Poiseuille (Re < 2000), Swamee-Jain (Re > 4000) [N7] and EPANET's
  cubic interpolation (Dunlop 1991) between. Swamee-Jain is coded with
  log10; the manual's rendering with ln is a typo (see ``darcy_friction``).
- minor loss ``m Q|Q|``, ``m = 0.0826 K / d^4`` (0.02517 in ft units).
- open valves: a smooth pipe (f = 0.02) twice the valve diameter long, plus
  the valve's minor loss; a TCV's setting is its loss coefficient.
- emitters: ``q = C p^gamma``; pressure-dependent demand: Wagner et al.
  [N3] in EPANET's form ``d = D ((p - P0)/(Pf - P0))^e`` clipped to [0, D].
- pumps: ``h_gain = w^2 (A - B (Q/w)^C)`` for speed ratio w; a
  constant-power pump [N8] ``h_gain = P / (rho g Q)``, rho g = 9802 N/m^3
  (62.4 lbf/ft^3, the unit weight EPANET's US-unit constants imply; not
  stated in the manual, so unverified; it brought ky10's median head
  difference from 1.3e-3 to 2.6e-4 m). Pumps with no flow path (see
  ``WaterNetwork.compile``) are closed, as EPANET's status check does.

Departures from EPANET, each smooth where EPANET switches: ``|Q|^(n-1)`` is
``(Q^2 + delta^2)^((n-1)/2)`` (EPANET instead linearizes below a flow
threshold); ``max(p, 0)`` in emitters and PDD is a smooth maximum; the D-W
friction factor is piecewise in Re with EPANET's own pieces.

The Newton solve runs on stop-gradient values (fixed iterations, line search
over step fractions) and ends with a value-neutral Newton step, so first
derivatives with respect to every parameter are the implicit-function ones.
"""

from __future__ import annotations

import warnings
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow_water.network.model import NetworkArrays, WaterNetwork

FT = 0.3048
CFS = FT**3
#: EPANET Table 3.1 / Section 13.1 constants (US units), converted to SI
K_HW = 4.727 * FT ** (4.871 - 3.0 * 1.852)        # = 10.667: h[m], d[m], L[m], Q[m^3/s]
K_DW = 0.0252 / FT                                # = 0.08268
K_MINOR = 0.02517 / FT                            # = 0.08258
F_VALVE = 0.02                                    # open-valve friction factor (EPANET)
R_CLOSED = 1e8 * FT / CFS                         # closed link h = R Q (EPANET 1e8 ft/cfs)
RHO_G = 9802.0                                    # N/m^3: 62.4 lbf/ft^3, EPANET's unit weight (unverified)

Q_SCALE = 1e-3   # m^3/s, scale of flows in the residual
H_SCALE = 1.0    # m, scale of heads
DELTA = 1e-7     # m^3/s, smoothing of |Q| near zero flow
P_SMOOTH = 1e-3  # m, smoothing of max(p, 0)

class NetworkConvergenceWarning(UserWarning):
    """The network Newton solve stopped above its tolerance."""


CODE_PIPE, CODE_PUMP, CODE_PRV, CODE_PSV, CODE_FCV, CODE_TCV = range(6)


def _smax0(x, eps):
    return 0.5 * (x + jnp.sqrt(x * x + eps * eps))


def _fb(a, b, mu):
    return a + b - jnp.sqrt(a * a + b * b + 2.0 * mu * mu)


def darcy_friction(Q, d, eps, nu) -> Array:
    """Darcy friction factor as EPANET computes it [N8] (laminar, transition, Swamee-Jain)."""
    Re = 4.0 * jnp.sqrt(Q * Q + DELTA**2) / (jnp.pi * d * nu)
    rel = eps / (3.7 * d)
    f_lam = 64.0 / jnp.maximum(Re, 1e-12)
    # Swamee-Jain with log10: the EPANET 2.2 manual prints "ln", which gives an
    # f (ln 10)^2 = 5.3 times too small and does not reproduce EPANET's results
    # (benchmarks/wntr); its own transition constant 0.86859 = 2/ln(10) is the
    # log10 form written with ln.
    f_turb = 0.25 / jnp.log10(rel + 5.74 / jnp.maximum(Re, 1.0) ** 0.9) ** 2
    # EPANET's cubic interpolation for 2000 < Re < 4000 (Dunlop 1991)
    Y2 = rel + 5.74 / jnp.maximum(Re, 1.0) ** 0.9
    Y3 = -0.86859 * jnp.log(rel + 5.74 / 4000.0**0.9)
    FA = Y3 ** -2.0
    FB = FA * (2.0 - 0.00514215 / (Y2 * Y3))
    R = Re / 2000.0
    X1 = 7.0 * FA - FB
    X2 = 0.128 - 17.0 * FA + 2.5 * FB
    X3 = -0.128 + 13.0 * FA - 2.0 * FB
    X4 = R * (0.032 - 3.0 * FA + 0.5 * FB)
    f_tr = X1 + R * (X2 + R * (X3 + X4))
    return jnp.where(Re < 2000.0, f_lam, jnp.where(Re > 4000.0, f_turb, f_tr))


def _setup(a: NetworkArrays):
    """Static index sets (numpy) for the residual."""
    kind = a.kind
    # every link that can switch carries its t/u, closed or not: a link's
    # status is a parameter (extended-period controls change it)
    t_links = np.where(np.isin(kind, (CODE_PRV, CODE_PSV, CODE_FCV)))[0]
    u_links = np.where(a.check)[0]
    return t_links, u_links


def default_parameters(a: NetworkArrays) -> dict:
    """The differentiable parameters of a network, as jnp arrays."""
    f = lambda v: jnp.asarray(v, dtype=jnp.float64)  # noqa: E731
    return {"demand": f(a.demand), "emitter": f(a.emitter), "roughness": f(a.roughness),
            "diameter": f(a.diameter), "fixed_head": f(a.fixed_head), "speed": f(a.speed),
            "setting": f(a.setting), "minor_loss": f(a.minor_loss),
            "closed": f(a.closed)}


def _residual(x, prm, mu, a: NetworkArrays, t_links, u_links, demand_model, pdd, viscosity):
    nJ, nL = a.n_junctions, a.n_links
    H = x[:nJ] * H_SCALE
    Q = x[nJ:nJ + nL] * Q_SCALE
    t = jnp.zeros(nL).at[t_links].set(x[nJ + nL:nJ + nL + len(t_links)] * H_SCALE)
    u = jnp.zeros(nL).at[u_links].set(x[nJ + nL + len(t_links):] * H_SCALE)
    H_all = jnp.concatenate([H, prm["fixed_head"]])
    elev_all = jnp.concatenate([jnp.asarray(a.elevation), jnp.zeros(len(a.fixed))])
    kind = a.kind
    d = prm["diameter"]
    L = jnp.asarray(a.length)
    Qs = Q * jnp.sqrt(Q * Q + DELTA**2)  # Q|Q|, smooth

    # friction
    is_pipe = kind == CODE_PIPE
    # pumps and valves carry no roughness: a safe stand-in keeps 0 ** -1.852
    # (inf, masked out below) from turning their gradients into NaN
    rough = jnp.where(is_pipe, prm["roughness"], 100.0)
    if a.headloss == "H-W":
        r_pipe = K_HW * rough ** -1.852 * d ** -4.871 * L
        h_pipe = r_pipe * Q * (Q * Q + DELTA**2) ** (0.852 / 2.0)
    else:
        f = darcy_friction(Q, d, jnp.where(is_pipe, prm["roughness"], 1e-4), viscosity)
        h_pipe = K_DW * f * L * d ** -5.0 * Qs
    h_valve = K_DW * F_VALVE * L * d ** -5.0 * Qs
    h_minor = K_MINOR * prm["minor_loss"] * d ** -4.0 * Qs
    is_pump = kind == CODE_PUMP
    hl = jnp.where(is_pipe, h_pipe, jnp.where(is_pump, 0.0, h_valve)) + jnp.where(is_pump, 0.0, h_minor)
    w = prm["speed"]
    Qp = jnp.sqrt(Q * Q + DELTA**2)
    curve_gain = (w**2 * jnp.asarray(a.pump_A)
                  - jnp.asarray(a.pump_B) * w ** (2.0 - jnp.asarray(a.pump_C))
                  * Qp ** jnp.asarray(a.pump_C))
    power = jnp.asarray(a.pump_power)
    power_gain = power / (RHO_G * Qp)
    gain = jnp.where(is_pump, jnp.where(power > 0, power_gain, curve_gain), 0.0)
    r_link = (H_all[a.start] - H_all[a.end] - (hl - gain + t - u)) / H_SCALE
    # closed links: EPANET's linear h = R Q with a large R [N8] (1e8 in ft/cfs),
    # which keeps the head of a node reached only through closed links defined
    r_closed = (H_all[a.start] - H_all[a.end] - R_CLOSED * Q) / H_SCALE
    closed = prm["closed"] > 0.5
    r_link = jnp.where(closed, r_closed, r_link)

    # junction continuity
    p = H - jnp.asarray(a.elevation)
    if demand_model == "PDD":
        P0, Pf, e = pdd
        z = (p - P0) / (Pf - P0)
        frac = _smax0(z, P_SMOOTH) - _smax0(z - 1.0, P_SMOOTH)
        dem = prm["demand"] * frac ** e
    else:
        dem = prm["demand"]
    emit = prm["emitter"] * _smax0(p, P_SMOOTH) ** jnp.asarray(a.emitter_exponent)
    inflow = jnp.zeros(nJ + len(a.fixed)).at[a.end].add(Q).at[a.start].add(-Q)[:nJ]
    r_node = (inflow - dem - emit) / Q_SCALE

    # complementarity
    r_t = []
    if len(t_links):
        sel = t_links
        kd = kind[sel]
        set_ = prm["setting"][sel]
        g_prv = (set_ + elev_all[a.end[sel]] - H_all[a.end[sel]]) / H_SCALE
        g_psv = (H_all[a.start[sel]] - set_ - elev_all[a.start[sel]]) / H_SCALE
        g_fcv = (set_ - Q[sel]) / Q_SCALE
        g = jnp.where(kd == CODE_PRV, g_prv, jnp.where(kd == CODE_PSV, g_psv, g_fcv))
        r_t = [jnp.where(closed[sel], t[sel] / H_SCALE, _fb(t[sel] / H_SCALE, g, mu))]
    r_u = ([jnp.where(closed[u_links], u[u_links] / H_SCALE,
                      _fb(u[u_links] / H_SCALE, Q[u_links] / Q_SCALE, mu))]
           if len(u_links) else [])
    return jnp.concatenate([r_node, r_link, *r_t, *r_u]), (dem, emit, hl, gain, t, u)


class NetworkModel:
    """A compiled network with a differentiable steady-state solve.

    Args:
        network: :class:`~difflow_water.network.model.WaterNetwork`.
        valve_formulation: "complementarity" (sharp switching) or "smooth"
            (differentiable everywhere; switches rounded over ``smoothing``).
        smoothing: Fischer-Burmeister smoothing for "smooth" (scaled units:
            1 m of head, 1 L/s of flow).
        demand_model: "DD" (demands met) or "PDD" (Wagner, EPANET form).
        pdd: ``(P0, Pf, exponent)`` for PDD (m, m, -).
        n_iter: Most Newton iterations per continuation stage.
        tol: Residual (scaled max norm) at which a stage stops.

    Example:
        >>> from difflow_water.network.model import WaterNetwork
        >>> net = WaterNetwork()
        >>> net.add_reservoir("R", head=100.0)
        >>> net.add_junction("J", elevation=20.0, demand=0.01)
        >>> net.add_pipe("P", "R", "J", length=1000.0, diameter=0.2, roughness=120.0)
        >>> res = NetworkModel(net).solve()
        >>> round(float(res["pressure"][0]), 2)
        79.24
    """

    def __init__(self, network: WaterNetwork, valve_formulation: str = "complementarity",
                 smoothing: float = 1e-3, demand_model: str = "DD",
                 pdd: tuple = (0.0, 20.0, 0.5), n_iter: int = 200, tol: float = 1e-10):
        if valve_formulation not in ("complementarity", "smooth"):
            raise ValueError("valve_formulation must be 'complementarity' or 'smooth'")
        if demand_model not in ("DD", "PDD"):
            raise ValueError("demand_model must be 'DD' or 'PDD'")
        self.network = network
        self.arrays = network.compile()
        self.t_links, self.u_links = _setup(self.arrays)
        self.valve_formulation = valve_formulation
        path = (1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-9)
        self.mus = (path if valve_formulation == "complementarity"
                    else tuple(m for m in path if m > smoothing) + (float(smoothing),))
        self.demand_model = demand_model
        self.pdd = tuple(float(v) for v in pdd)
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self._solve = jax.jit(partial(_solve, self))

    def parameters(self) -> dict:
        """Default (as-built) parameters: ``demand``, ``emitter``, ``roughness``,
        ``diameter``, ``fixed_head``, ``speed``, ``setting``, ``minor_loss``,
        ``closed`` (1.0 for a closed link; a status, not differentiable)."""
        return default_parameters(self.arrays)

    def _f(self, x, prm, mu):
        a = self.arrays
        return _residual(x, prm, mu, a, self.t_links, self.u_links, self.demand_model,
                         self.pdd, self.network.viscosity)

    def x0(self, prm) -> Array:
        a = self.arrays
        H0 = jnp.full(a.n_junctions, jnp.max(prm["fixed_head"]))
        area = jnp.pi * prm["diameter"] ** 2 / 4.0
        Q0 = 0.3 * area  # 0.3 m/s, as EPANET starts at 1 ft/s
        return jnp.concatenate([H0 / H_SCALE, Q0 / Q_SCALE, jnp.ones(len(self.t_links)),
                                jnp.ones(len(self.u_links))])

    def solve(self, params: dict | None = None) -> dict:
        """Solve the network.

        Args:
            params: Overrides for :meth:`parameters` (any subset).

        Returns:
            dict with ``head``, ``pressure``, ``demand`` (delivered),
            ``emitter_flow`` (per junction), ``flow``, ``headloss``,
            ``head_gain``, ``throttle`` (t), ``block`` (u) per link,
            ``velocity``, ``residual`` (max abs scaled), ``converged``
            (residual < 1e-6; absent under jit/grad), ``junctions``,
            ``links``, ``auto_closed`` (pumps closed for lack of a flow path).
        """
        prm = self.parameters()
        if params:
            unknown = set(params) - set(prm)
            if unknown:
                raise ValueError(f"unknown network parameters {sorted(unknown)}")
            for k, v in params.items():
                v = jnp.asarray(v, dtype=jnp.float64)
                if v.shape != prm[k].shape:
                    # an out-of-range gather is clamped forward and its gradient
                    # silently dropped: refuse rather than return a zero derivative
                    raise ValueError(f"parameter {k!r} has shape {v.shape}, expected {prm[k].shape}")
                prm[k] = v
        out = self._solve(prm)
        out["junctions"] = self.arrays.junctions
        out["links"] = self.arrays.links
        out["auto_closed"] = list(self.arrays.auto_closed)
        try:
            res = float(out["residual"])
        except jax.errors.ConcretizationTypeError:  # under jit/grad: no check
            return out
        out["converged"] = res < 1e-6
        if not out["converged"]:
            warnings.warn(f"network solve did not converge (residual {res:.2e}); raise "
                          "n_iter, or try valve_formulation='complementarity'",
                          NetworkConvergenceWarning, stacklevel=2)
        return out


def _solve(model: NetworkModel, prm):
    a = model.arrays
    nJ, nL = a.n_junctions, a.n_links
    alphas = jnp.asarray([1.0, 0.5, 0.25, 0.125, 1 / 16, 1 / 32, 1 / 64, 1 / 128])

    x = jax.lax.stop_gradient(model.x0(prm))
    sprm = jax.lax.stop_gradient(prm)
    for mu in model.mus:
        x = jax.lax.stop_gradient(_newton_stage(model, x, sprm, mu, alphas))
    mu = model.mus[-1]
    f = lambda z: model._f(z, prm, mu)[0]  # noqa: E731
    Js = jax.lax.stop_gradient(jax.jacfwd(f)(x))
    r = f(x)
    x = x - jnp.linalg.solve(Js, r - jax.lax.stop_gradient(r))
    res, (dem, emit, hl, gain, t, u) = model._f(x, prm, mu)
    H = x[:nJ] * H_SCALE
    Q = x[nJ:nJ + nL] * Q_SCALE
    area = jnp.pi * prm["diameter"] ** 2 / 4.0
    return {
        "head": H, "pressure": H - jnp.asarray(a.elevation), "demand": dem,
        "emitter_flow": emit, "flow": Q, "headloss": hl, "head_gain": gain,
        "throttle": t, "block": u,
        "velocity": jnp.where(a.kind == CODE_PUMP, 0.0, Q / area),
        "residual": jnp.max(jnp.abs(res)),
    }


def _newton_stage(model, x, prm, mu, alphas):
    f = lambda z: model._f(z, prm, mu)[0]  # noqa: E731

    def body(state):
        k, z, _ = state
        r = f(z)
        J = jax.jacfwd(f)(z)
        dz = jnp.linalg.solve(J, -r)
        norms = jax.vmap(lambda al: jnp.max(jnp.abs(f(z + al * dz))))(alphas)
        i = jnp.argmin(norms)
        return k + 1, z + alphas[i] * dz, norms[i]

    def cond(state):
        k, _, res = state
        return (k < model.n_iter) & (res > model.tol)

    _, x, _ = jax.lax.while_loop(cond, body, (0, x, jnp.asarray(jnp.inf)))
    return x
