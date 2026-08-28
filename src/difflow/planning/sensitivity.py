"""Sensitivity of the *plan*, not just the plan.

A conventional planning system returns an optimal ``u``.  Because the blocks
here are differentiable, the converged plan can also be differentiated: how
does the optimum move when a price changes, when a design parameter changes,
when a model parameter is re-estimated?  That is what k_aug and sIPOPT provide
for a single NLP, and it is what makes capital planning tractable without
enumerating scenarios.

The machinery is the implicit function theorem applied to the KKT conditions
at the converged plan.  With free decisions ``u_F`` (those not at a bound),
active constraints ``h_A`` and multipliers ``nu``::

    L(u_F, nu, theta) = phi(u_F, theta) + nu . h_A(u_F, theta)

    [ d2L/du2   A^T ] [ du_F/dtheta ]   [ -d2L/du dtheta ]
    [ A         0   ] [ dnu/dtheta  ] = [ -dh_A/dtheta   ]

and the envelope theorem gives the objective sensitivity directly,

    dphi*/dtheta = dL/dtheta = dphi/dtheta + nu . dh_A/dtheta

which is exact even when the plan sits at a vertex and ``du/dtheta`` is zero.

Every derivative in that system comes from AD on the caller's own blocks, so
nothing here is a finite difference.

Bang-bang caveat
----------------
When a lever is at a bound, ``du/dtheta`` for that lever is genuinely zero:
the corner does not move under an infinitesimal price change, it *switches* at
a finite one.  The derivative is the right answer to the wrong question there,
so :func:`price_switch_point` finds the finite price at which the plan
actually changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class PlanSensitivity:
    """Derivatives of a converged plan with respect to parameters.

    Attributes:
        wrt: What was differentiated against, ``"prices"`` or ``"theta"``.
        param_names: Parameter names, in column order.
        decision_names: Decision names, in row order.
        d_plan: ``d(u*)/d(theta)``, shape ``(n_decisions, n_params)``.  Rows
            for decisions pinned at a bound are exactly zero.
        d_objective: ``d(objective*)/d(theta)`` by the envelope theorem.
        multipliers: Multiplier per active constraint.
        active_constraints: Names of the active specs.
        fixed_decisions: Decisions pinned at a bound.
        stationarity_residual: Norm of the KKT stationarity residual.  A large
            value means the plan is not a stationary point and the
            sensitivities below describe a point the planner did not reach.
        degenerate: The KKT matrix was rank-deficient and a least-squares
            solution was used.
        note: Plain-language caveat for this particular result.
    """

    wrt: str
    param_names: list[str]
    decision_names: list[str]
    d_plan: np.ndarray
    d_objective: np.ndarray
    multipliers: dict[str, float]
    active_constraints: list[str] = field(default_factory=list)
    fixed_decisions: list[str] = field(default_factory=list)
    stationarity_residual: float = 0.0
    degenerate: bool = False
    note: str = ""

    def as_dict(self) -> dict[str, dict[str, float]]:
        """``{decision: {parameter: derivative}}``."""
        return {d: {p: float(self.d_plan[i, j])
                    for j, p in enumerate(self.param_names)}
                for i, d in enumerate(self.decision_names)}

    def objective_sensitivity(self) -> dict[str, float]:
        """``{parameter: d(objective)/d(parameter)}``."""
        return {p: float(self.d_objective[j])
                for j, p in enumerate(self.param_names)}

    def summary(self, atol: float = 1e-9) -> str:
        """A human-readable report."""
        lines = [f"Plan sensitivity w.r.t. {self.wrt}"]
        if self.note:
            lines.append(f"  note: {self.note}")
        if self.fixed_decisions:
            lines.append("  at a bound (d(plan)/d(param) = 0): "
                         + ", ".join(self.fixed_decisions))
        if self.active_constraints:
            lines.append("  active constraints: "
                         + ", ".join(self.active_constraints))
        lines.append("  d(objective)/d(param):")
        for p, v in self.objective_sensitivity().items():
            lines.append(f"    {p:<32s} {v:14.6g}")
        moving = [(i, d) for i, d in enumerate(self.decision_names)
                  if np.max(np.abs(self.d_plan[i])) > atol]
        if moving:
            lines.append("  d(plan)/d(param):")
            head = " " * 36 + "".join(f"{p[:13]:>14s}"
                                      for p in self.param_names)
            lines.append(head)
            for i, d in moving:
                cells = "".join(f"{self.d_plan[i, j]:14.4g}"
                                for j in range(len(self.param_names)))
                lines.append(f"    {d:<32s}{cells}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"PlanSensitivity(wrt={self.wrt!r}, "
                f"shape={self.d_plan.shape}, degenerate={self.degenerate})")


def _relu(x):
    return jnp.maximum(x, 0.0)


def _spec_value(spec, values):
    return sum(c * values[k] for k, c in spec.coeffs.items())


def _spec_violation(spec, values):
    v = _spec_value(spec, values)
    if spec.op == "<=":
        return _relu(v - spec.rhs)
    if spec.op == ">=":
        return _relu(spec.rhs - v)
    return jnp.abs(v - spec.rhs)


def _flatten_theta(theta: Mapping[str, Mapping[str, Any]]
                   ) -> tuple[np.ndarray, list[str], Callable]:
    """Flatten ``{block: {param: value}}`` to a vector plus an unflattener."""
    names: list[str] = []
    values: list[float] = []
    layout: list[tuple[str, str]] = []
    for bname in sorted(theta):
        for pname in sorted(theta[bname]):
            v = theta[bname][pname]
            arr = np.atleast_1d(np.asarray(v, dtype=float))
            if arr.size != 1:
                # Only scalar parameters take part; array parameters stay
                # fixed rather than being silently flattened into unlabelled
                # columns.
                continue
            names.append(f"{bname}.{pname}")
            values.append(float(arr[0]))
            layout.append((bname, pname))

    base = {b: dict(d) for b, d in theta.items()}

    def unflatten(vec):
        out = {b: dict(d) for b, d in base.items()}
        for i, (b, p) in enumerate(layout):
            out[b][p] = vec[i]
        return out

    return np.asarray(values, dtype=float), names, unflatten


def plan_sensitivity(result, wrt: str = "prices",
                     params: Sequence[str] | None = None,
                     active_tol: float = 1e-6,
                     bound_tol: float = 1e-7) -> PlanSensitivity:
    """Differentiate a converged plan with respect to prices or parameters.

    Args:
        result: A :class:`~difflow.planning.planner.PlanResult`.
        wrt: ``"prices"`` differentiates against the price vector;
            ``"theta"`` against the blocks' scalar ``theta`` parameters.
        params: Restrict to these parameter names (price variable names, or
            ``"<block>.<param>"``).  Defaults to all of them.
        active_tol: Tolerance for calling a spec active.
        bound_tol: Tolerance for calling a decision pinned at a bound.

    Returns:
        A :class:`PlanSensitivity`.

    Example:
        >>> s = res.plan_sensitivity(wrt="prices")
        >>> s.objective_sensitivity()["power.Power"]   # = the power sold
        1832.4...
    """
    planner = result.planner
    network = result.network
    sign = planner._merit_sign
    u_star = jnp.asarray(result.decisions, dtype=float)
    names = list(network.decision_names)
    n = len(names)

    if wrt == "prices":
        keys = list(planner.prices) if params is None else list(params)
        unknown = [k for k in keys if k not in planner.prices]
        if unknown:
            raise KeyError(f"no price is set for {unknown}")
        theta0 = jnp.asarray([float(planner.prices[k]) for k in keys])

        def price_map(tvec):
            d = dict(planner.prices)
            for i, k in enumerate(keys):
                d[k] = tvec[i]
            return d

        def evaluate(u, tvec):
            return (planner.evaluation_network.evaluate(
                u, planner.theta).values, price_map(tvec))

    elif wrt == "theta":
        theta = planner.theta
        if theta is None:
            theta = {b.name: b.theta for b in network.blocks
                     if b.theta is not None}
        if not theta:
            raise ValueError(
                "no block declares scalar `theta` parameters, so there is "
                "nothing to differentiate against. Give the blocks a `theta` "
                "dict and an `fn(u, theta)` signature, or use wrt='prices'.")
        vec0, all_names, unflatten = _flatten_theta(theta)
        if params is not None:
            unknown = [p for p in params if p not in all_names]
            if unknown:
                raise KeyError(
                    f"unknown parameter(s) {unknown}. Scalar parameters "
                    f"available for differentiation: {all_names}")
            keep = [all_names.index(p) for p in params]
            keys = list(params)
        else:
            keep = list(range(len(all_names)))
            keys = all_names
        if not keys:
            raise ValueError("no scalar theta parameters to differentiate")
        theta0 = jnp.asarray(vec0[keep])
        full0 = jnp.asarray(vec0)

        def evaluate(u, tvec):
            full = full0.at[jnp.asarray(keep)].set(tvec)
            return (planner.evaluation_network.evaluate(
                u, unflatten(full)).values, planner.prices)

    else:
        raise ValueError(
            f"wrt must be 'prices' or 'theta', got {wrt!r}")

    def objective(u, tvec):
        values, prices = evaluate(u, tvec)
        return sum(prices[v] * values[v] for v in prices)

    def phi(u, tvec):
        values, prices = evaluate(u, tvec)
        obj = sum(prices[v] * values[v] for v in prices)
        pen = 0.0
        for s in planner.specs:
            p = planner.penalty if s.penalty is None else s.penalty
            pen = pen + abs(float(p)) * _spec_violation(s, values)
        return sign * obj - pen

    # -- active set ------------------------------------------------------
    values_star = result.state.values
    active = [s for s in planner.specs
              if abs(float(_spec_value(s, values_star)) - s.effective_rhs)
              <= active_tol * max(1.0, abs(s.effective_rhs))]
    active_names = [s.name for s in active]

    lo, hi = network.decision_bounds()
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    u_np = np.asarray(result.decisions, dtype=float)
    scale = np.where(np.isfinite(hi - lo), np.maximum(hi - lo, 1.0), 1.0)
    pinned = ((np.isfinite(lo) & (np.abs(u_np - lo) <= bound_tol * scale))
              | (np.isfinite(hi) & (np.abs(u_np - hi) <= bound_tol * scale)))
    free = np.flatnonzero(~pinned)
    fixed_names = [names[i] for i in np.flatnonzero(pinned)]

    def h(u, tvec):
        if not active:
            return jnp.zeros((0,))
        values, _ = evaluate(u, tvec)
        return jnp.stack([_spec_value(s, values) - s.effective_rhs
                          for s in active])

    n_p = int(theta0.shape[0])
    n_h = len(active)
    d_plan = np.zeros((n, n_p))

    # -- multipliers from stationarity on the free variables -------------
    grad_phi_u = np.asarray(jax.grad(phi, argnums=0)(u_star, theta0))
    A_full = (np.asarray(jax.jacobian(h, argnums=0)(u_star, theta0))
              if n_h else np.zeros((0, n)))
    A = A_full[:, free] if n_h else np.zeros((0, len(free)))
    g = grad_phi_u[free]

    degenerate = False
    if n_h and len(free):
        nu, *_ = np.linalg.lstsq(A.T, -g, rcond=None)
        residual = float(np.linalg.norm(A.T @ nu + g))
    elif n_h:
        nu = np.zeros(n_h)
        residual = 0.0
    else:
        nu = np.zeros(0)
        residual = float(np.linalg.norm(g)) if len(free) else 0.0

    # -- the KKT sensitivity system --------------------------------------
    note = ""
    if len(free) == 0:
        note = ("every decision is at a bound, so the plan is locally "
                "insensitive: it switches at a finite parameter change, not "
                "an infinitesimal one. Use price_switch_point() to find where.")
    else:
        def lagrangian(u, tvec):
            return phi(u, tvec) + (jnp.dot(jnp.asarray(nu), h(u, tvec))
                                   if n_h else 0.0)

        H_full = np.asarray(jax.hessian(lagrangian, argnums=0)(u_star, theta0))
        W = H_full[np.ix_(free, free)]
        M_ut = np.asarray(
            jax.jacobian(jax.grad(lagrangian, argnums=0), argnums=1)(
                u_star, theta0))[free, :]
        H_theta = (np.asarray(jax.jacobian(h, argnums=1)(u_star, theta0))
                   if n_h else np.zeros((0, n_p)))

        nf = len(free)
        K = np.zeros((nf + n_h, nf + n_h))
        K[:nf, :nf] = W
        if n_h:
            K[:nf, nf:] = A.T
            K[nf:, :nf] = A
        rhs = np.zeros((nf + n_h, n_p))
        rhs[:nf, :] = -M_ut
        if n_h:
            rhs[nf:, :] = -H_theta

        try:
            cond = np.linalg.cond(K)
        except np.linalg.LinAlgError:  # pragma: no cover
            cond = np.inf
        if not np.isfinite(cond) or cond > 1e12:
            degenerate = True
            sol, *_ = np.linalg.lstsq(K, rhs, rcond=None)
            note = (note or
                    "the KKT matrix is rank-deficient (a degenerate active "
                    "set or a locally flat objective); a least-squares "
                    "solution is reported and d(plan)/d(param) is not unique.")
        else:
            sol = np.linalg.solve(K, rhs)
        d_plan[free, :] = sol[:nf, :]

    # -- envelope theorem for the objective ------------------------------
    d_obj_direct = np.asarray(jax.grad(objective, argnums=1)(u_star, theta0))
    grad_obj_u = np.asarray(jax.grad(objective, argnums=0)(u_star, theta0))
    d_objective = d_obj_direct + grad_obj_u @ d_plan

    if residual > 1e-4 * max(1.0, float(np.linalg.norm(g))):
        note = (note + " " if note else "") + (
            f"the KKT stationarity residual is {residual:.3g}: this plan is "
            "not a stationary point (the loop may have stopped on "
            f"'{result.reason}'), so these derivatives describe a point the "
            "planner did not converge to.")

    return PlanSensitivity(
        wrt=wrt, param_names=list(keys), decision_names=names,
        d_plan=d_plan, d_objective=d_objective,
        multipliers={n: float(v) for n, v in zip(active_names, nu)},
        active_constraints=active_names, fixed_decisions=fixed_names,
        stationarity_residual=residual, degenerate=degenerate, note=note)


def price_switch_point(planner, variable: str, lo: float, hi: float,
                       decision: str | None = None,
                       tol: float = 1e-4, max_iter: int = 40,
                       **solve_kwargs) -> dict[str, Any]:
    """Bisect for the price at which the plan switches corner.

    A bang-bang lever does not respond to an infinitesimal price change; it
    jumps at a finite one.  This finds that price by re-solving the plan and
    bisecting on the decision's value.

    Args:
        planner: A :class:`~difflow.planning.planner.DeltaBasePlanner`.
        variable: The priced variable whose price is varied.
        lo: Price at which one corner is optimal.
        hi: Price at which the other is.
        decision: Which decision to watch.  Defaults to whichever moves most
            between the two endpoints.
        tol: Absolute price tolerance for the bisection.
        max_iter: Bisection iterations.
        **solve_kwargs: Passed to ``planner.solve``.

    Returns:
        Dict with ``price``, ``decision``, ``plan_low``, ``plan_high``,
        ``bracket`` and ``n_iter``.  ``price`` is ``None`` when the plan does
        not switch anywhere in the bracket.

    Example:
        >>> price_switch_point(planner, "power.Power", 10.0, 80.0)["price"]
        43.6...
    """
    if variable not in planner.prices:
        raise KeyError(f"no price is set for {variable!r}")
    original = dict(planner.prices)

    def solve_at(p: float):
        planner.prices = dict(original)
        planner.prices[variable] = float(p)
        try:
            return planner.solve(**solve_kwargs)
        finally:
            planner.prices = dict(original)

    res_lo = solve_at(lo)
    res_hi = solve_at(hi)
    plan_lo, plan_hi = res_lo.plan, res_hi.plan

    if decision is None:
        diffs = {k: abs(plan_hi[k] - plan_lo[k]) for k in plan_lo}
        decision = max(diffs, key=diffs.get)
    if decision not in plan_lo:
        raise KeyError(f"{decision!r} is not a decision of this network")

    v_lo, v_hi = plan_lo[decision], plan_hi[decision]
    span = abs(v_hi - v_lo)
    if span <= tol:
        return {"price": None, "decision": decision, "plan_low": plan_lo,
                "plan_high": plan_hi, "bracket": (lo, hi), "n_iter": 0,
                "note": (f"{decision!r} does not switch between prices "
                         f"{lo:g} and {hi:g}")}

    mid_value = 0.5 * (v_lo + v_hi)
    a, b = float(lo), float(hi)
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        m = 0.5 * (a + b)
        v = solve_at(m).plan[decision]
        near_lo = abs(v - v_lo) < abs(v - v_hi)
        if near_lo:
            a = m
        else:
            b = m
        if abs(b - a) <= tol:
            break

    return {"price": 0.5 * (a + b), "decision": decision, "plan_low": plan_lo,
            "plan_high": plan_hi, "bracket": (a, b), "n_iter": n_iter,
            "threshold_value": mid_value}
