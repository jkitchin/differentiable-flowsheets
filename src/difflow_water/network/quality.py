"""Water quality: EPANET's Lagrangian transport with bulk and wall reactions.

Implements Section 13.2 of the EPANET 2.2 manual [N8] (content verified):

- advection without dispersion: each pipe holds a train of water parcels
  ("segments"); every water-quality step the segments react, then nodes are
  visited upstream to downstream (topological order of the current flows);
  a node blends the water its inflow links release (``Q dt`` from the
  leading segments of each), a junction's quality is the flow-weighted
  mean (Eq. 13.9), a tank mixes it completely with its contents
  (Eq. 13.10), and a new segment of the node's quality enters every
  outflow link (merged into the last one when within ``tolerance``);
- a link reversing its flow reverses its segment order;
- initially every pipe is one segment of its upstream node's quality;
- reactions: bulk ``r = K_b C^n`` (order n, here 0 or 1), first-order wall
  ``r = 2 k_w k_f C / (R (|k_w| + k_f))`` with the mass-transfer coefficient
  ``k_f = Sh D / d``, ``Sh = 3.65 + 0.0668 (d/L) Re Sc / (1 + 0.04 ((d/L) Re Sc)^(2/3))``
  (laminar, Edwards et al. 1976) or ``0.0149 Re^0.88 Sc^(1/3)`` (turbulent,
  Notter & Sleicher 1971); tanks react with the tank bulk coefficient;
- water age: zero-order growth at 1 hour per hour, reservoirs at age 0;
  trace: the traced node at 100 (%), every other source at 0.

Not stated in the manual, taken as EPANET's (unverified): the chlorine
diffusivity 1.3e-8 ft^2/s (scaled by the DIFFUSIVITY option), the laminar /
turbulent switch at Re = 2300, ``|k_w|`` in the wall-rate denominator, and
first-order reaction over a step as ``C exp(k dt)``.

This module is NumPy, not JAX: the segment bookkeeping (creation, merging,
destruction) is discrete, and the result is not differentiable yet.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from difflow_water.network.inp import Schedule
from difflow_water.network.model import WaterNetwork

D_CHLORINE = 1.3e-8 * 0.3048**2   # m^2/s at 20 C (EPANET's value; unverified)
RE_LAMINAR = 2300.0
Q_ZERO = 1e-6 * 0.3048**3         # m^3/s (EPANET's QZERO, 1e-6 cfs; unverified)


def _topological(n_nodes, start, end, flow, tiny=1e-12):
    """Nodes ordered upstream to downstream along the current flows (Kahn)."""
    succ = [[] for _ in range(n_nodes)]
    indeg = np.zeros(n_nodes, int)
    for k in range(len(flow)):
        if abs(flow[k]) <= tiny:
            continue
        u, v = (start[k], end[k]) if flow[k] > 0 else (end[k], start[k])
        succ[u].append(v)
        indeg[v] += 1
    order = [i for i in range(n_nodes) if indeg[i] == 0]
    out, seen = [], np.zeros(n_nodes, bool)
    while order:
        u = order.pop()
        out.append(u)
        seen[u] = True
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                order.append(v)
    out += [i for i in range(n_nodes) if not seen[i]]   # cycles of ~zero flow
    return out


def water_quality(network: WaterNetwork, schedule: Schedule, eps: dict,
                  quality_step: float | None = None, report_step: float | None = None) -> dict:
    """Run water quality over an extended-period simulation.

    Args:
        network: The network (as given to :func:`~difflow_water.network.eps.simulate`).
        schedule: Its schedule (quality mode, initial quality, reactions).
        eps: The result of ``simulate(network, schedule)`` (its ``history``).
        quality_step: Water-quality time step (s); the schedule's when None.
        report_step: Reporting interval (s); the schedule's when None.

    Returns:
        dict with ``time`` (s), ``quality`` (report times x nodes: junctions,
        then reservoirs, then tanks; mg/L for CHEMICAL, hours for AGE, % for
        TRACE), ``nodes`` (names) and ``mode``.
    """
    opt = schedule.quality_options
    mode = opt.get("mode", "NONE")
    if mode == "NONE":
        raise ValueError("the schedule has no water-quality analysis (QUALITY NONE)")
    a = network.compile()
    J, F = a.junctions, a.fixed
    nodes = J + F
    nJ, n = len(J), len(J) + len(F)
    tanks = list(network.tanks)
    res_names = list(network.reservoirs)
    tank_idx = {t: nJ + len(res_names) + i for i, t in enumerate(tanks)}
    res_idx = {r: nJ + i for i, r in enumerate(res_names)}
    dq = schedule.quality_step if quality_step is None else quality_step
    rep = schedule.report_step if report_step is None else report_step
    tol = opt.get("tolerance", 0.01)
    R = schedule.reactions or {}
    D = D_CHLORINE * opt.get("diffusivity", 1.0)
    nu = network.viscosity

    # link geometry and reaction constants
    L = np.asarray(a.length)
    d = np.asarray(a.diameter)
    is_pipe = a.kind == 0
    vol = np.where(is_pipe, np.pi * d**2 / 4.0 * L, 0.0)
    kb = np.array([R.get("bulk", {}).get(nm, R.get("global_bulk", 0.0)) for nm in a.links])
    kw = np.array([R.get("wall", {}).get(nm, R.get("global_wall", 0.0)) for nm in a.links])
    order_b, order_w = R.get("order_bulk", 1.0), R.get("order_wall", 1.0)
    if mode == "CHEMICAL" and (order_b not in (0.0, 1.0) or order_w != 1.0 and np.any(kw != 0)):
        raise ValueError("only bulk orders 0 and 1 and first-order wall reactions are supported")
    kt = {t: R.get("tank", {}).get(t, R.get("global_bulk", 0.0)) for t in tanks}

    # initial quality
    c_node = np.zeros(n)
    if mode == "CHEMICAL":
        for nm, v in schedule.quality.items():
            if nm in nodes:
                c_node[nodes.index(nm)] = v
    elif mode == "TRACE":
        c_node[nodes.index(opt["trace_node"])] = 100.0
    source = np.zeros(n, bool)
    for r in res_names:
        source[res_idx[r]] = True
    if mode == "TRACE":
        source[nodes.index(opt["trace_node"])] = True
    c_source = c_node.copy()            # reservoirs (and the traced node) hold their quality

    hist = eps["history"]
    flow0 = np.where(hist[0].get("closed", np.zeros(a.n_links, bool)), 0.0, hist[0]["flow"])
    segs = []
    for k in range(a.n_links):
        up = a.start[k] if flow0[k] >= 0 else a.end[k]
        segs.append(deque([[vol[k], c_node[up]]]) if vol[k] > 0 else deque())
    direction = np.sign(np.where(flow0 == 0, 1.0, flow0))
    tank_vol = {t: np.pi * network.tanks[t].diameter**2 / 4.0 * hist[0]["tank_level"][i]
                for i, t in enumerate(tanks)}

    def kf_of(k, q):
        if not is_pipe[k] or q == 0.0:
            return 0.0
        u = abs(q) / (np.pi * d[k] ** 2 / 4.0)
        Re = u * d[k] / nu
        Sc = nu / D
        if Re < RE_LAMINAR:
            x = d[k] / L[k] * Re * Sc
            Sh = 3.65 + 0.0668 * x / (1.0 + 0.04 * x ** (2.0 / 3.0))
        else:
            Sh = 0.0149 * Re**0.88 * Sc ** (1.0 / 3.0)
        return Sh * D / d[k]

    def react(c, k, q, dt):
        if mode == "AGE":
            return c + dt / 3600.0
        if mode == "TRACE":
            return c
        if order_b == 0.0:
            c = c + kb[k] * dt
        else:
            rate = kb[k]
            if kw[k] != 0.0:
                kf = kf_of(k, q)
                rate += 2.0 * kw[k] * kf / ((d[k] / 2.0) * (abs(kw[k]) + kf)) if kf > 0 else 0.0
            c = c * np.exp(rate * dt)
        return max(c, 0.0)

    out_t, out_c = [], []
    next_report = 0.0
    t_end = eps["steps"][-1]
    for hi, h in enumerate(hist):
        # closed links carry no water for quality (their modelled h = R Q trickle
        # would otherwise set a dead-end node's quality); neither do flows below
        # EPANET's zero-flow threshold (1e-6 cfs, unverified)
        q = np.where(h.get("closed", np.zeros(a.n_links, bool)) | (np.abs(h["flow"]) < Q_ZERO),
                     0.0, h["flow"])
        lv1 = hist[hi + 1]["tank_level"] if hi + 1 < len(hist) else h["tank_level"]
        # flow reversals: reverse the segment train
        new_dir = np.sign(np.where(q == 0, direction, q))
        for k in np.where(new_dir != direction)[0]:
            segs[k].reverse()
        direction = new_dir
        order = _topological(n, a.start, a.end, q)
        lv0 = h["tank_level"]
        n_sub = max(1, int(round(h["dt"] / dq)))
        dts = h["dt"] / n_sub
        for s in range(n_sub):
            t = h["t"] + s * dts
            if t >= next_report - 1e-6:
                out_t.append(t)
                out_c.append(c_node.copy())
                next_report += rep
            # 1. react
            for k in range(a.n_links):
                for seg in segs[k]:
                    seg[1] = react(seg[1], k, q[k], dts)
            for ti, tk in enumerate(tanks):
                i = tank_idx[tk]
                if mode == "AGE":
                    c_node[i] += dts / 3600.0
                elif mode == "CHEMICAL":
                    c_node[i] = (c_node[i] * np.exp(kt[tk] * dts) if R.get("order_tank", 1.0) == 1.0
                                 else max(c_node[i] + kt[tk] * dts, 0.0))
            # 2. nodes upstream to downstream
            for i in order:
                vin = 0.0
                min_ = 0.0
                for k in np.where(((a.end == i) & (q > 0)) | ((a.start == i) & (q < 0)))[0]:
                    v = abs(q[k]) * dts
                    if vol[k] == 0.0:          # pump / valve: passes its upstream quality
                        up = a.start[k] if q[k] > 0 else a.end[k]
                        min_ += v * c_node[up]
                        vin += v
                        continue
                    need = v
                    tr = segs[k]
                    while need > 1e-15 and tr:
                        seg = tr[0]
                        take = min(need, seg[0])
                        min_ += take * seg[1]
                        need -= take
                        seg[0] -= take
                        if seg[0] <= 1e-15:
                            tr.popleft()
                    vin += v - need
                if source[i]:
                    c_node[i] = c_source[i]
                elif nodes[i] in network.tanks:
                    tk = nodes[i]
                    V = tank_vol[tk]
                    if V + vin > 0:
                        c_node[i] = (c_node[i] * V + min_) / (V + vin)
                elif vin > 0:
                    c_node[i] = min_ / vin
                # 3. new segments in the outflow links
                for k in np.where(((a.start == i) & (q > 0)) | ((a.end == i) & (q < 0)))[0]:
                    if vol[k] == 0.0:
                        continue
                    v = abs(q[k]) * dts
                    tr = segs[k]
                    if tr and abs(tr[-1][1] - c_node[i]) <= tol:
                        tr[-1][1] = (tr[-1][1] * tr[-1][0] + c_node[i] * v) / (tr[-1][0] + v)
                        tr[-1][0] += v
                    else:
                        tr.append([v, c_node[i]])
            # tank volumes follow the hydraulics
            for ti, tk in enumerate(tanks):
                A_ = np.pi * network.tanks[tk].diameter ** 2 / 4.0
                tank_vol[tk] = A_ * (lv0[ti] + (s + 1) / n_sub * (lv1[ti] - lv0[ti]))
    if t_end >= next_report - 1e-6:
        out_t.append(t_end)
        out_c.append(c_node.copy())
    return {"time": np.asarray(out_t), "quality": np.asarray(out_c), "nodes": nodes, "mode": mode}
