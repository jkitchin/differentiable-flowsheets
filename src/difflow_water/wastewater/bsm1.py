"""Activated sludge plant: ASM1 reactors in series + secondary settler, steady state.

The plant of Benchmark Simulation Model no. 1 (Alex et al. 2008 [W5],
Section 2.3; content verified): ``N`` CSTRs with ASM1 (:mod:`.asm1`), an
internal recycle ``Q_a`` from the last reactor to the first, and a
secondary settler whose underflow is split into the external recycle
``Q_r`` and the wastage ``Q_w``.

Reactor ``k`` (mass balance, [W5] 2.3.2), at steady state::

    0 = Q_in Z_in - Q Z_k + r(Z_k) V_k  (+ KLa_k V_k (S_O* - S_O,k) for oxygen)

A reactor may instead hold its dissolved oxygen at a setpoint (the oxygen
balance replaced by ``S_O,k = setpoint``; the aeration it needs is
reported), which is how WaterTAP's BSM1 example runs its last reactor.

Settler, two options:

- ``"takacs"`` (BSM1): the 10-layer, non-reactive, one-dimensional settler
  with the Takacs double-exponential settling velocity
  ``v_s = max(0, min(v0', v0 (e^{-r_h (X - X_min)} - e^{-r_p (X - X_min)})))``,
  ``X_min = f_ns X_f``, the clarification-layer flux limited by the
  threshold ``X_t`` ([W5] 2.3.3; Takacs et al. 1991 [W1]). Solubles pass
  unchanged; particulate fractions are those of the settler feed. The feed
  layer's balance is written with ``(v_up + v_dn) X`` leaving it (the
  mass-conservative form; the sign in the extracted report text is
  ambiguous);
- ``"split"`` (WaterTAP's BSM1 example): a separator that sends a fixed
  fraction of each particulate to the effluent; solubles follow the water.

The whole plant is one equation set. It is integrated in time from a
generic activated-sludge state for ``t_settle`` days (stiff, adaptive:
diffrax Kvaerno5), as the BSM1 stabilization procedure prescribes, then
polished by Newton (``optimistix``), so gradients are implicit and exact at
the steady state. A plain Newton (or a large-step continuation) from the
initial guess can land on the autotroph-washout steady state, which also
solves the equations; following the dynamics avoids it. Units: m^3/d, g/m^3 (S_ALK mol/m^3), days.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream
from difflow_water.wastewater.asm1 import (
    IDX,
    PART_MASK,
    S_O_SAT,
    STATES,
    ASM1Params,
    conversion_rates,
    state_to_stream,
    stream_to_state,
    suspended_solids,
)

NS = len(STATES)


@dataclass(repr=False)
class TakacsParams(ParamsMixin):
    """Takacs settler, BSM1 values ([W5] 2.3.3, Table 4).

    Attributes:
        area: Settler surface area (m^2).
        height: Settler height (m).
        n_layers: Number of layers (-).
        feed_layer: Feed layer, counted from the bottom (1-based).
        v0_max: Maximum settling velocity v0' (m/d).
        v0: Maximum Vesilind settling velocity (m/d).
        r_h: Hindered zone settling parameter (m^3/g SS).
        r_p: Flocculant zone settling parameter (m^3/g SS).
        f_ns: Non-settleable fraction (-).
        X_t: Threshold concentration for the clarification flux (g SS/m^3).
    """

    area: float | Array = 1500.0
    height: float | Array = 4.0
    n_layers: int = 10
    feed_layer: int = 6
    v0_max: float | Array = 250.0
    v0: float | Array = 474.0
    r_h: float | Array = 0.000576
    r_p: float | Array = 0.00286
    f_ns: float | Array = 0.00228
    X_t: float | Array = 3000.0


@dataclass(repr=False)
class ActivatedSludgeParams(ParamsMixin):
    """An activated sludge plant (defaults: BSM1 open loop, [W5] Sections 2.3 and 4).

    Attributes:
        volumes: Reactor volumes (m^3).
        KLa: Oxygen transfer coefficient per reactor (1/d); BSM1 open loop
            is 0, 0, 240, 240, 84 (10 h^-1 and 3.5 h^-1).
        DO_setpoint: Per-reactor dissolved-oxygen setpoint (g/m^3), or
            None for reactors on KLa. A setpoint replaces the oxygen balance.
        Q_a: Internal recycle, last reactor to first (m^3/d).
        Q_r: External (sludge) recycle (m^3/d).
        Q_w: Wastage (m^3/d).
        settler: ``"takacs"`` or ``"split"``.
        split_particulate: Fraction of each particulate to the effluent
            (``"split"`` settler only).
        takacs: :class:`TakacsParams`.
        asm: :class:`~difflow_water.wastewater.asm1.ASM1Params`.
        t_settle: Days of constant-influent dynamics before the Newton
            polish (-); BSM1 stabilizes for 100 days.
    """

    volumes: tuple = (1000.0, 1000.0, 1333.0, 1333.0, 1333.0)
    KLa: tuple = (0.0, 0.0, 240.0, 240.0, 84.0)
    DO_setpoint: tuple = (None, None, None, None, None)
    Q_a: float | Array = 55338.0
    Q_r: float | Array = 18446.0
    Q_w: float | Array = 385.0
    settler: str = "takacs"
    split_particulate: float | Array = 0.00187
    takacs: TakacsParams = field(default_factory=TakacsParams)
    asm: ASM1Params = field(default_factory=ASM1Params)
    t_settle: float = 200.0


def settling_velocity(X, Xf, t: TakacsParams):
    """Takacs double-exponential settling velocity (m/d) [W1, W5]."""
    Xmin = t.f_ns * Xf
    v = t.v0 * (jnp.exp(-t.r_h * (X - Xmin)) - jnp.exp(-t.r_p * (X - Xmin)))
    return jnp.maximum(0.0, jnp.minimum(t.v0_max, v))


def takacs_residual(X, Xf, Qf, Qu, Qe, t: TakacsParams):
    """dX/dt of the settler's solids layers (bottom first), g SS/m^3/d."""
    n, fl = int(t.n_layers), int(t.feed_layer) - 1
    z = t.height / n
    v_dn, v_up = Qu / t.area, Qe / t.area
    Js = settling_velocity(X, Xf, t) * X
    out = []
    for m in range(n):
        if m == 0:
            d = v_dn * (X[1] - X[0]) + jnp.minimum(Js[1], Js[0])
        elif m < fl:
            d = v_dn * (X[m + 1] - X[m]) + jnp.minimum(Js[m], Js[m + 1]) - jnp.minimum(Js[m], Js[m - 1])
        else:
            def jclar(j):   # clarification flux into layer j-1 from layer j (j above the feed)
                return jnp.where(X[j - 1] > t.X_t, jnp.minimum(Js[j], Js[j - 1]), Js[j])
            if m == fl:
                up = jclar(m + 1) if m + 1 < n else 0.0
                d = Qf * Xf / t.area + up - (v_up + v_dn) * X[m] - jnp.minimum(Js[m], Js[m - 1])
            elif m < n - 1:
                d = v_up * (X[m - 1] - X[m]) + jclar(m + 1) - jclar(m)
            else:
                d = v_up * (X[m - 1] - X[m]) - jclar(m)
        out.append(d / z)
    return jnp.stack(out)


def _flows(p, Q0):
    Q1 = Q0 + p.Q_a + p.Q_r
    Qf = Q1 - p.Q_a
    Qu = p.Q_r + p.Q_w
    return Q1, Qf, Qu, Qf - Qu


def _settler_out(Z5, Xl, Qf, Qu, Qe, p):
    """(Z_underflow, Z_effluent) from the settler feed Z5 and its state."""
    if p.settler == "takacs":
        Xf = suspended_solids(Z5)
        fu, fe = Xl[0] / Xf, Xl[-1] / Xf
    else:
        fu = Qf * (1.0 - p.split_particulate) / Qu
        fe = Qf * p.split_particulate / Qe
    Zu = jnp.where(PART_MASK, Z5 * fu, Z5)
    Ze = jnp.where(PART_MASK, Z5 * fe, Z5)
    return Zu, Ze


def _residual(x, Q0, Z0, p: ActivatedSludgeParams):
    nr = len(p.volumes)
    Z = x[: nr * NS].reshape(nr, NS)
    Xl = x[nr * NS:]
    Q1, Qf, Qu, Qe = _flows(p, Q0)
    Zu, _ = _settler_out(Z[-1], Xl, Qf, Qu, Qe, p)
    Zin = (Q0 * Z0 + p.Q_a * Z[-1] + p.Q_r * Zu) / Q1
    res = []
    for k in range(nr):
        Zi = Zin if k == 0 else Z[k - 1]
        V = p.volumes[k]
        dz = Q1 * (Zi - Z[k]) / V + conversion_rates(Z[k], p.asm)
        dz = dz.at[IDX["S_O"]].add(p.KLa[k] * (S_O_SAT - Z[k, IDX["S_O"]]))
        if p.DO_setpoint[k] is not None:
            # a fast relaxation onto the setpoint (stable in the dynamics; same root)
            dz = dz.at[IDX["S_O"]].set(-1e3 * (Z[k, IDX["S_O"]] - p.DO_setpoint[k]))
        res.append(dz)
    r = jnp.concatenate(res)
    if p.settler == "takacs":
        r = jnp.concatenate([r, takacs_residual(Xl, suspended_solids(Z[-1]), Qf, Qu, Qe,
                                                p.takacs)])
    return r


def _initial(Q0, Z0, p):
    nr = len(p.volumes)
    # a generic activated-sludge state: inerts and biomass accumulated, substrates low
    z = jnp.array([Z0[0], 2.0, 1000.0, 80.0, 2500.0, 150.0, 450.0, 1.0, 5.0, 5.0, 1.0, 5.0, 5.0])
    x = jnp.tile(z, nr)
    if p.settler == "takacs":
        n = int(p.takacs.n_layers)
        Xl = jnp.where(jnp.arange(n) < int(p.takacs.feed_layer) - 1, 3000.0, 50.0)
        Xl = Xl.at[0].set(6000.0)
        x = jnp.concatenate([x, Xl])
    return x


def _sg(obj):
    """A copy of a (nested) params dataclass with every number stop-gradiented."""
    if dataclasses.is_dataclass(obj):
        return dataclasses.replace(obj, **{f.name: _sg(getattr(obj, f.name))
                                           for f in dataclasses.fields(obj)})
    if isinstance(obj, tuple):
        return tuple(_sg(v) for v in obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):   # constants: no tangent
        return obj
    return jax.lax.stop_gradient(obj)


def _ptc(x0, Q0, Z0, p):
    """Integrate the plant dynamics to near steady state (stiff, adaptive).

    diffrax Kvaerno5 with a PID step-size controller over ``t_settle`` days
    of constant influent (the BSM1 stabilization procedure, [W5] Section 3,
    uses 100 days); Newton then finishes the steady state.
    """
    import diffrax

    term = diffrax.ODETerm(lambda t, x, a: _residual(x, Q0, Z0, p))
    sol = diffrax.diffeqsolve(term, diffrax.Kvaerno5(), t0=0.0, t1=float(p.t_settle), dt0=1e-4,
                              y0=x0, stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-6),
                              max_steps=20000, throw=False, adjoint=diffrax.ForwardMode())
    return sol.ys[-1]


def activated_sludge_plant(influent: Stream, params: ActivatedSludgeParams | None = None) -> dict:
    """Steady state of an activated sludge plant (BSM1 layout).

    Args:
        influent: Influent stream (``F_H2O`` and ASM1 ``F_<state>``, see
            :mod:`.asm1`); e.g. :func:`~difflow_water.wastewater.asm1.bsm1_influent`.
        params: :class:`ActivatedSludgeParams` (BSM1 open loop when None).

    Returns:
        dict with ``reactors`` (n x 13, g/m^3), ``settler_solids`` (layers,
        bottom first, g SS/m^3; ``"takacs"`` only), ``effluent``,
        ``underflow``, ``wastage`` streams and their concentration vectors
        ``Z_effluent``/``Z_underflow``, flows ``Q`` (dict, m^3/d),
        ``aeration`` (g O2/d supplied per reactor), ``sludge_age`` (d, on
        biomass, [W5] 2.3.3), ``TSS_effluent``, ``converged``, ``residual``.

    Example:
        >>> from difflow_water.wastewater.asm1 import bsm1_influent
        >>> r = activated_sludge_plant(bsm1_influent())
        >>> bool(r["converged"])
        True
    """
    p = params or ActivatedSludgeParams()
    if p.settler not in ("takacs", "split"):
        raise ValueError("settler must be 'takacs' or 'split'")
    nr = len(p.volumes)
    if not (len(p.KLa) == nr and len(p.DO_setpoint) == nr):
        raise ValueError("volumes, KLa and DO_setpoint need one entry per reactor")
    Q0, Z0 = stream_to_state(influent)
    x0 = _initial(Q0, Z0, p)
    # the settle phase only finds the starting point: nothing in it is differentiated
    # (stop-gradient every input, the params included, so forward mode works too)
    x_ptc = jax.lax.stop_gradient(_ptc(jax.lax.stop_gradient(x0), jax.lax.stop_gradient(Q0),
                                       jax.lax.stop_gradient(Z0), _sg(p)))
    sol = optx.root_find(lambda x, a: _residual(x, a[0], a[1], p),
                         optx.Newton(rtol=1e-10, atol=1e-8), x_ptc, args=(Q0, Z0),
                         max_steps=20, throw=False)
    x = sol.value
    res = _residual(x, Q0, Z0, p)
    Z = x[: nr * NS].reshape(nr, NS)
    Xl = x[nr * NS:]
    Q1, Qf, Qu, Qe = _flows(p, Q0)
    Zu, Ze = _settler_out(Z[-1], Xl, Qf, Qu, Qe, p)
    # aeration: KLa term, or (setpoint) the oxygen the reactor consumes net of flow
    aer = []
    for k in range(nr):
        V = p.volumes[k]
        Zi = (Q0 * Z0 + p.Q_a * Z[-1] + p.Q_r * Zu) / Q1 if k == 0 else Z[k - 1]
        if p.DO_setpoint[k] is not None:
            o = IDX["S_O"]
            aer.append(-(Q1 * (Zi[o] - Z[k, o]) + conversion_rates(Z[k], p.asm)[o] * V))
        else:
            aer.append(p.KLa[k] * V * (S_O_SAT - Z[k, IDX["S_O"]]))
    bio = IDX["X_BH"], IDX["X_BA"]
    TXa = sum((Z[k, bio[0]] + Z[k, bio[1]]) * p.volumes[k] for k in range(nr))
    if p.settler == "takacs":
        Xf = suspended_solids(Z[-1])
        zA = p.takacs.height / int(p.takacs.n_layers) * p.takacs.area
        TXs = jnp.sum(Xl / Xf) * (Z[-1, bio[0]] + Z[-1, bio[1]]) * zA
    else:
        TXs = 0.0
    loss = (Ze[bio[0]] + Ze[bio[1]]) * Qe + (Zu[bio[0]] + Zu[bio[1]]) * p.Q_w
    return {
        "reactors": Z,
        "settler_solids": Xl,
        "Z_effluent": Ze,
        "Z_underflow": Zu,
        "effluent": state_to_stream(Qe, Ze, influent["T"], influent["P"]),
        "underflow": state_to_stream(Qu, Zu, influent["T"], influent["P"]),
        "wastage": state_to_stream(p.Q_w, Zu, influent["T"], influent["P"]),
        "Q": {"influent": Q0, "reactors": Q1, "settler_feed": Qf, "underflow": Qu,
              "effluent": Qe, "internal_recycle": p.Q_a, "external_recycle": p.Q_r,
              "wastage": p.Q_w},
        "aeration": jnp.stack(aer),
        "sludge_age": (TXa + TXs) / loss,
        "TSS_effluent": suspended_solids(Ze),
        "converged": (sol.result == optx.RESULTS.successful) & (jnp.max(jnp.abs(res)) < 1e-6),
        "residual": jnp.max(jnp.abs(res)),
    }


#: BSM1 effluent limits ([W5] Table 9), g/m^3
BSM1_LIMITS = {"N_tot": 18.0, "COD": 100.0, "S_NH": 4.0, "TSS": 30.0, "BOD5": 10.0}
#: EQ weighting factors ([W5] Table 10), g pollution unit / g
_EQ_B = {"SS": 2.0, "COD": 1.0, "NKj": 30.0, "NO": 10.0, "BOD5": 2.0}


def bsm1_indices(result: dict, params: ActivatedSludgeParams | None = None) -> dict:
    """BSM1 performance indices at steady state ([W5] Section 6).

    The report's definitions, with its time averages over constant
    (steady-state) values:

    - effluent ``COD = S_S + S_I + X_S + X_I + X_BH + X_BA + X_P``,
      ``BOD5 = 0.25 (S_S + X_S + (1 - f_P)(X_BH + X_BA))``,
      ``TSS = 0.75 (X_S + X_I + X_BH + X_BA + X_P)``,
      ``N_Kj = S_NH + S_ND + X_ND + i_XB (X_BH + X_BA) + i_XP (X_P + X_I)``,
      ``N_tot = S_NO + N_Kj`` (g/m^3);
    - effluent quality ``EQ = (2 TSS + COD + 30 N_Kj + 10 S_NO + 2 BOD5) Q_e / 1000``
      (kg pollution units/d, Table 10 weights);
    - aeration energy ``AE = S_O,sat / (1.8 * 1000) sum V_i KLa_i`` (kWh/d,
      KLa in 1/d; reactors on a DO setpoint are not counted);
    - pumping energy ``PE = 0.004 Q_a + 0.008 Q_r + 0.05 Q_w`` (kWh/d).

    Args:
        result: The output of :func:`activated_sludge_plant`.
        params: The :class:`ActivatedSludgeParams` it was run with.

    Returns:
        dict of the effluent concentrations, ``EQ``, ``AE``, ``PE`` and
        ``violations`` (each concentration minus its Table 9 limit; positive
        is a violation).
    """
    p = params or ActivatedSludgeParams()
    a = p.asm
    Z = result["Z_effluent"]
    g = {s: Z[IDX[s]] for s in STATES}
    cod = g["S_S"] + g["S_I"] + g["X_S"] + g["X_I"] + g["X_BH"] + g["X_BA"] + g["X_P"]
    bod = 0.25 * (g["S_S"] + g["X_S"] + (1.0 - a.f_P) * (g["X_BH"] + g["X_BA"]))
    tss = 0.75 * (g["X_S"] + g["X_I"] + g["X_BH"] + g["X_BA"] + g["X_P"])
    nkj = (g["S_NH"] + g["S_ND"] + g["X_ND"] + a.i_XB * (g["X_BH"] + g["X_BA"])
           + a.i_XP * (g["X_P"] + g["X_I"]))
    ntot = g["S_NO"] + nkj
    Qe = result["Q"]["effluent"]
    eq = (_EQ_B["SS"] * tss + _EQ_B["COD"] * cod + _EQ_B["NKj"] * nkj + _EQ_B["NO"] * g["S_NO"]
          + _EQ_B["BOD5"] * bod) * Qe / 1000.0
    ae = S_O_SAT / (1.8 * 1000.0) * sum(
        p.volumes[k] * p.KLa[k] for k in range(len(p.volumes)) if p.DO_setpoint[k] is None)
    pe = 0.004 * p.Q_a + 0.008 * p.Q_r + 0.05 * p.Q_w
    conc = {"COD": cod, "BOD5": bod, "TSS": tss, "N_Kj": nkj, "N_tot": ntot,
            "S_NH": g["S_NH"], "S_NO": g["S_NO"]}
    return {**conc, "EQ": eq, "AE": ae, "PE": pe,
            "violations": {k: conc[k] - v for k, v in BSM1_LIMITS.items()}}
