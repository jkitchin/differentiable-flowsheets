"""Extended-period simulation (EPS): a sequence of steady states, as EPANET does.

Between hydraulic solves the tank levels advance with their net inflow
(``dh/dt = Q_net / A``, cylindrical tanks), demands and reservoir heads follow
their patterns, pump speeds their speed patterns, and simple controls switch
links. The time step follows EPANET 2.2's rule [N8] (Section 13.1): the next
time is the earliest of the next hydraulic step, the next pattern period, the
next timed control, the time for a tank to reach a control level, and the
time for a tank to fill or empty.

Each step is a :class:`~difflow_water.network.hydraulics.NetworkModel`
solve with the step's demands, heads, speeds and statuses as parameters.
Gradients: a plain run records a ``plan`` (the link statuses of every step
and what set its length); ``simulate(..., plan=plan, params=traced)``
replays it in JAX, holding the switching sequence fixed but recomputing, in
JAX, the length of every step that was cut to land a tank on a control or
limit level, so the derivative includes how the switch *times* move. This
is the derivative of the trajectory with the same sequence of switches, the
usual one for hybrid systems.

Departures from EPANET: tanks are cylinders (no volume curves); a full
(empty) tank closes the links carrying flow into (out of) it for the step,
which is EPANET's rule, but checked only at the start of the step; junction
pressure controls use the pressure of the previous solve.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from difflow_water.network.hydraulics import NetworkModel
from difflow_water.network.inp import Schedule
from difflow_water.network.model import WaterNetwork


def dead_end_pumps(arrays, closed: np.ndarray, demand: np.ndarray, emitter: np.ndarray) -> np.ndarray:
    """Mask of open pumps whose outlet side holds no demand, emitter, reservoir or tank.

    The same rule as :meth:`WaterNetwork.compile`, for the statuses of one step.
    """
    nJ = arrays.n_junctions
    n = nJ + len(arrays.fixed)
    sink = np.zeros(n, bool)
    sink[nJ:] = True
    sink[:nJ] = (demand != 0.0) | (emitter != 0.0)
    out = np.zeros(arrays.n_links, bool)
    for k in np.where((arrays.kind == 1) & ~closed)[0]:
        parent = np.arange(n)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for j in np.where(~closed)[0]:
            if j != k:
                parent[find(arrays.start[j])] = find(arrays.end[j])
        root = find(arrays.end[k])
        out[k] = not any(sink[i] for i in range(n) if find(i) == root)
    return out


def simulate(network: WaterNetwork, schedule: Schedule, duration: float | None = None,
             model: NetworkModel | None = None, params: dict | None = None,
             report_step: float | None = None, plan: list | None = None) -> dict:
    """Run an extended-period simulation.

    Args:
        network: The network at its initial state (e.g. from :func:`read_inp`).
        schedule: Patterns, controls and times (:class:`Schedule`).
        duration: Simulated time (s); the schedule's when None.
        model: A :class:`NetworkModel` of ``network`` to reuse (its compiled
            solve is cached); built with defaults when None.
        params: Overrides for the model's static parameters (``roughness``,
            ``emitter``, ...), applied at every step; differentiable.
        report_step: Reporting interval (s); the schedule's when None.
        plan: The ``plan`` of an earlier run of the same simulation, to replay
            it with (possibly traced) ``params``: statuses and step rules are
            taken from it instead of being decided on the fly.

    Returns:
        dict with ``time`` (s, the report times), and per report time
        ``head``, ``pressure`` (junctions), ``flow`` (links), ``tank_level``
        (tanks, m), ``status`` (links, 1 open / 0 closed), ``demand``; plus
        ``junctions``, ``links``, ``tanks`` names, ``steps`` (all solve
        times, including those between reports), ``plan`` and ``history``
        (per hydraulic step: start time, length, flows, tank levels, demands;
        the input of :func:`difflow_water.network.quality.water_quality`).
    """
    model = model or NetworkModel(network)
    a = model.arrays
    duration = schedule.duration if duration is None else duration
    report = schedule.report_step if report_step is None else report_step
    tanks = list(network.tanks)
    res_names = list(network.reservoirs)
    links = a.links
    link_idx = {n: i for i, n in enumerate(links)}
    node_idx = {n: i for i, n in enumerate(a.junctions)}
    area = np.array([np.pi * network.tanks[t].diameter ** 2 / 4.0 for t in tanks])
    levels = jnp.asarray([network.tanks[t].level for t in tanks], dtype=jnp.float64)
    lo = np.array([network.tanks[t].min_level for t in tanks])
    hi = np.array([network.tanks[t].max_level for t in tanks])
    tank_elev = np.array([network.tanks[t].elevation for t in tanks])
    nJ = a.n_junctions
    tank_node = {t: nJ + len(res_names) + i for i, t in enumerate(tanks)}
    base = model.parameters()
    if params:
        base.update({k: jnp.asarray(v, dtype=jnp.float64) for k, v in params.items()})
    user_closed = np.asarray(a.closed, bool).copy()
    # auto-closed pumps at compile time are re-decided every step
    for name in a.auto_closed:
        user_closed[link_idx[name]] = bool(_initial_status_closed(network, name))
    speed0 = base["speed"]           # a parameter: differentiable, patterns multiply it
    settings = np.asarray(base["setting"], float).copy()
    done_time = set()
    out = {k: [] for k in ("time", "head", "pressure", "flow", "tank_level", "status", "demand")}
    steps = []
    t = 0.0
    last_pressure = None
    next_report = 0.0
    replay = plan is not None
    new_plan = []
    history = []
    k_step = 0
    while t <= duration + 1e-6:
        if replay:
            if k_step >= len(plan):
                break
            closed_p, rule = plan[k_step]
        clock = (schedule.start_clocktime + t) % 86400.0
        # timed controls
        for k, c in enumerate([] if replay else schedule.controls):
            fire = ((c.kind == "time" and abs(t - c.value) < 1e-6 and k not in done_time)
                    or (c.kind == "clocktime" and abs(clock - c.value) < 1e-6))
            if fire:
                done_time.add(k)
                _apply(c, user_closed, settings, link_idx, network)
        # level / pressure controls on the current state
        for c in ([] if replay else schedule.controls):
            if c.kind not in ("above", "below"):
                continue
            if c.node in network.tanks:
                v = float(levels[tanks.index(c.node)])
            elif c.node in node_idx and last_pressure is not None:
                v = float(last_pressure[node_idx[c.node]])
            else:
                continue
            # the step was cut to land a tank exactly on a control level: compare
            # with a tolerance so a level reached is a level crossed
            if (c.kind == "above" and v > c.value - 1e-6) or (c.kind == "below" and v < c.value + 1e-6):
                _apply(c, user_closed, settings, link_idx, network)
        demand = np.array([schedule.demand(j, t) for j in a.junctions])
        res_heads = np.array([network.reservoirs[r].head
                              * schedule.multiplier(schedule.reservoir_patterns.get(r), t)
                              / schedule.multiplier(schedule.reservoir_patterns.get(r), 0.0)
                              for r in res_names])
        heads = jnp.concatenate([jnp.asarray(res_heads, dtype=jnp.float64),
                                 jnp.asarray(tank_elev) + levels])
        mult = np.ones(a.n_links)
        for pmp, pat in schedule.pump_patterns.items():
            mult[link_idx[pmp]] = schedule.multiplier(pat, t)
        speed = speed0 * jnp.asarray(mult)
        if replay:
            closed = np.asarray(closed_p, bool)
        else:
            closed = user_closed.copy()
            # full / empty tanks: close the links that would overfill / drain them (EPANET)
            closed |= _tank_limit_closures(a, tanks, tank_node, levels, lo, hi, last_flow=None)
            closed |= dead_end_pumps(a, closed, demand, np.asarray(base["emitter"]))
        prm = {**base, "demand": jnp.asarray(demand), "fixed_head": jnp.asarray(heads),
               "speed": speed, "closed": jnp.asarray(closed, dtype=jnp.float64),
               "setting": jnp.asarray(settings)}
        sol = model._solve(prm)
        q = sol["flow"]
        if not replay:
            last_pressure = np.asarray(sol["pressure"])
        steps.append(t)
        if t >= next_report - 1e-6:
            out["time"].append(t)
            out["head"].append(sol["head"])
            out["pressure"].append(sol["pressure"])
            out["flow"].append(q)
            out["tank_level"].append(levels)
            out["status"].append(1.0 - closed.astype(float))
            out["demand"].append(sol["demand"])
            next_report += report
        if t >= duration - 1e-6:
            if not replay:
                new_plan.append((closed.copy(), ("end", 0.0)))
            break
        # net inflow to each tank
        qn = jnp.stack([jnp.sum(jnp.where(jnp.asarray(a.end) == tank_node[tk], q, 0.0))
                        - jnp.sum(jnp.where(jnp.asarray(a.start) == tank_node[tk], q, 0.0))
                        for tk in tanks]) if tanks else jnp.zeros(0)
        if replay:
            dt_c = rule[1]
            if rule[0] == "tank":   # recompute in JAX: the switch time moves with the parameters
                i, target = rule[2], rule[3]
                dt = (target - levels[i]) * area[i] / qn[i]
            else:
                dt = dt_c
        else:
            dt_c, rule = _next_step(t, schedule, duration, next_report, levels, qn, area, lo, hi,
                                    tanks, schedule.controls)
            dt = dt_c
            new_plan.append((closed.copy(), rule))
        if not replay:
            history.append({"t": t, "dt": float(dt_c), "flow": np.asarray(q),
                            "closed": closed.copy(),
                            "tank_level": np.asarray(levels), "demand": np.asarray(sol["demand"]),
                            "emitter_flow": np.asarray(sol["emitter_flow"])})
        levels = levels + qn * dt / jnp.asarray(area) if tanks else levels
        t += dt_c
        k_step += 1
    return {**{k: (jnp.stack(v) if k != "time" else np.asarray(v)) for k, v in out.items()},
            "junctions": a.junctions, "links": links, "tanks": tanks, "steps": np.asarray(steps),
            "plan": plan if replay else new_plan, "history": history}


def _initial_status_closed(network, name) -> bool:
    for d in (network.pipes, network.pumps, network.valves):
        if name in d:
            return d[name].status == "closed"
    return False


def _apply(c, closed, settings, link_idx, network):
    i = link_idx[c.link]
    if c.status is not None:
        closed[i] = c.status == "closed"
    elif c.link in network.pumps:
        closed[i] = c.setting == 0.0     # a speed setting of 0 closes a pump
    else:
        settings[i] = c.setting
        closed[i] = False


def _tank_limit_closures(a, tanks, tank_node, levels, lo, hi, last_flow):
    """Links into a full tank / out of an empty one, closed for the step (EPANET rule).

    Without a flow direction yet, a link whose other end is a junction is
    closed only if the tank is at a limit: into a full tank the link is
    blocked from delivering (check-valve semantics are left to the solve).
    """
    closed = np.zeros(a.n_links, bool)
    for i, tk in enumerate(tanks):
        lev = float(levels[i])
        if lev >= hi[i] - 1e-9 or lev <= lo[i] + 1e-9:
            node = tank_node[tk]
            touching = (a.start == node) | (a.end == node)
            closed |= touching & (a.kind == 1)  # pumps filling/draining a tank at its limit
    return closed


def _next_step(t, schedule, duration, next_report, levels, qn, area, lo, hi, tanks, controls):
    """EPANET's time step: earliest of hydraulic step, pattern period, timed
    control, tank reaching a control level, tank filling/emptying, report.

    Returns ``(dt, rule)``; ``rule`` is ``("tank", dt, tank index, level)``
    when a tank reaching a level set the step, else ``("fixed", dt)``."""
    cand = [(schedule.hydraulic_step - (t % schedule.hydraulic_step), None),
            (schedule.pattern_step - ((t + schedule.pattern_start) % schedule.pattern_step), None),
            (duration - t, None), (next_report - t, None)]
    clock = (schedule.start_clocktime + t) % 86400.0
    for c in controls:
        if c.kind == "time" and c.value > t + 1e-6:
            cand.append((c.value - t, None))
        elif c.kind == "clocktime":
            d = (c.value - clock) % 86400.0
            if d > 1e-6:
                cand.append((d, None))
    lv = np.asarray(levels)
    q = np.asarray(qn)
    for i, tk in enumerate(tanks):
        rate = q[i] / area[i]
        if abs(rate) < 1e-12:
            continue
        targets = [hi[i], lo[i]] + [c.value for c in controls
                                    if c.node == tk and c.kind in ("above", "below")]
        for target in targets:
            d = (target - lv[i]) / rate
            if d > 1e-6:
                cand.append((d, (i, float(target))))
    dt, tank = min((c for c in cand if c[0] > 1e-6), key=lambda c: c[0])
    rule = ("tank", float(dt), tank[0], tank[1]) if tank is not None else ("fixed", float(dt))
    return float(dt), rule
