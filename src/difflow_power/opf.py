"""Optimal power flow: AC and DC, both as one interior-point NLP.

A power flow spends the network's degrees of freedom on setpoints. An
optimal power flow spends them on cost: it chooses every generator's
real and reactive output and every bus voltage to minimise the cost of
serving the load, subject to the same balance equations plus the limits
a power flow ignores.

.. math::

    \\min_{V, \\theta, P_g, Q_g} \\; \\sum_k c_k(P_{g,k})
    \\quad \\text{subject to}

    \\quad S^{sched} - V \\overline{Y V} = 0
    \\quad\\text{(the power flow equations)}

    \\quad V^{min} \\le |V| \\le V^{max}, \\;
    P^{min}_g \\le P_g \\le P^{max}_g, \\;
    Q^{min}_g \\le Q_g \\le Q^{max}_g

    \\quad |S_{f}|^2 \\le \\bar S^2, \\; |S_{t}|^2 \\le \\bar S^2
    \\quad\\text{(thermal ratings, both ends)}

The equalities come from :mod:`difflow_power.residuals` unchanged ---
this module adds no physics --- and the whole thing is handed to
:func:`difflow_power.ipm.solve_nlp`.

Why the squared thermal limit
-----------------------------

``|S| <= rate`` and ``|S|^2 <= rate^2`` describe the same set for a
non-negative rating, but ``|S|`` has unbounded curvature at the origin
and a lightly loaded branch is exactly where an interior-point method
puts an early iterate. The squared form is a smooth quadratic
everywhere. Both ends are limited because a lossy branch carries more
at its sending end than its receiving end, and the rating applies to
the conductor at both.

Prices
------

The multiplier on a bus's real-power balance IS its locational marginal
price. With the balance written as ``(P_g - P_d) - P_{inj}(V) = 0``,
adding a MW of load at bus *i* perturbs row *i* by ``-1``, so

    ``LMP_i = -lambda_i / base_mva``   in $/MWh.

At an uncongested solution every LMP equals the marginal cost of the
marginal unit plus a small loss component. Where a rating binds they
separate, and the spread is the congestion rent --- which is the whole
reason to run an AC-OPF instead of an economic dispatch.
:meth:`ACOPFResult.lmp_mw` reports them, and
:meth:`ACOPFResult.check_prices` verifies them against
``jax.grad`` of the optimal cost with respect to load, which is an
independent computation of the same number.

DC-OPF
------

:func:`difflow_power.dc.solve_dcopf` is the standard linearisation --- lossless, flat
voltages, ``sin(theta) ~ theta`` --- which turns the problem into a
convex QP. It is what markets clear on, it always has a solution when
the AC problem does, and it is a good warm start. It runs through the
SAME interior-point solver: a QP is just an NLP whose Hessian happens
to be constant, and having one solver means one set of conventions for
the multipliers, so a DC LMP and an AC LMP are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import jax
import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin

from difflow_power.ipm import IPMResult, NLP, differentiable_solution, solve_nlp
from difflow_power.network import PowerNetwork
from difflow_power.physics import apparent_power_squared, polynomial_cost
from difflow_power.powerflow import flat_start, solve_power_flow
from difflow_power.residuals import (
    PowerStateLayout,
    branch_flows,
    power_flow_residuals,
    power_state_layout,
)

#: a bound pair closer than this is treated as an equality rather than
#: two opposing inequalities, whose multipliers would be non-unique
FIXED_BOUND_TOL = 1e-9

#: angle-difference limits at or beyond this are dropped as vacuous
#: (case files write +-360 degrees to mean "no limit")
VACUOUS_ANGLE_RAD = 3.0


# =============================================================================
# Problem assembly
# =============================================================================


@dataclass
class OPFStructure(ParamsMixin):
    """Which rows the assembled OPF carries, and what each one means.

    Built by :func:`acopf_structure`. Everything here is static Python
    decided before the solve: which bounds are finite, which branches
    are rated, which bound pairs collapsed to equalities. Keeping it
    out of the traced functions is what lets the constraint set be
    reported and audited, rather than being an anonymous vector of
    numbers.

    Attributes:
        lower_indices: state positions with a finite lower bound.
        lower_values: those bounds.
        upper_indices: state positions with a finite upper bound.
        upper_values: those bounds.
        fixed_indices: state positions whose bounds coincide; carried as
            equalities.
        fixed_values: those values.
        rated_branches: positions of branches with a finite rating.
        rate_squared: their squared ratings (pu^2).
        angle_branches: positions of branches with a real angle limit.
        angle_min, angle_max: those limits (rad).
        bound_names: label of each bound inequality row, in order.
        equality_names: label of each equality row, in order.
    """

    lower_indices: tuple[int, ...] = ()
    lower_values: tuple[float, ...] = ()
    upper_indices: tuple[int, ...] = ()
    upper_values: tuple[float, ...] = ()
    fixed_indices: tuple[int, ...] = ()
    fixed_values: tuple[float, ...] = ()
    rated_branches: tuple[int, ...] = ()
    rate_squared: tuple[float, ...] = ()
    angle_branches: tuple[int, ...] = ()
    angle_min: tuple[float, ...] = ()
    angle_max: tuple[float, ...] = ()
    bound_names: tuple[str, ...] = ()
    equality_names: tuple[str, ...] = ()

    @property
    def n_thermal(self) -> int:
        """Thermal rows: two per rated branch, from and to end."""
        return 2 * len(self.rated_branches)

    @property
    def n_angle(self) -> int:
        return 2 * len(self.angle_branches)

    @property
    def n_inequality(self) -> int:
        return len(self.bound_names) + self.n_thermal + self.n_angle

    @property
    def inequality_names(self) -> list[str]:
        """Label of every inequality row, in the order ``h`` returns them."""
        return list(self.bound_names) + [
            f"{end}_{i}"
            for i in self.rated_branches
            for end in ("rate_from", "rate_to")
        ] + [
            f"{side}_{i}"
            for i in self.angle_branches
            for side in ("angle_max", "angle_min")
        ]


def acopf_structure(
    network: PowerNetwork,
    layout: PowerStateLayout,
    *,
    enforce_ratings: bool = True,
    enforce_voltage: bool = True,
    enforce_angles: bool = True,
) -> OPFStructure:
    """Decide the OPF's constraint set from the network's limits.

    Only rows that can actually bind are built. An unrated branch, an
    infinite bound, a +-360 degree angle limit: each is dropped rather
    than carried as a constraint that is always slack, because every
    dropped row is one fewer slack, multiplier and column in the KKT
    factorisation.

    Args:
        network: the network.
        layout: the state packing.
        enforce_ratings: include branch thermal limits.
        enforce_voltage: include bus voltage magnitude limits. Turning
            this off gives the "P-only" dispatch that is sometimes
            wanted for comparison against a DC solution.
        enforce_angles: include angle-difference limits.

    Returns:
        An :class:`OPFStructure`.
    """
    names = layout.names
    where = {nm: i for i, nm in enumerate(names)}
    lo_i, lo_v, up_i, up_v, fx_i, fx_v, bound_names = [], [], [], [], [], [], []

    def add_box(name: str, lower: float | None, upper: float | None):
        idx = where[name]
        if (
            lower is not None
            and upper is not None
            and abs(upper - lower) < FIXED_BOUND_TOL
        ):
            fx_i.append(idx)
            fx_v.append(float(lower))
            return
        if lower is not None and jnp.isfinite(lower):
            lo_i.append(idx)
            lo_v.append(float(lower))
            bound_names.append(f"{name}_min")
        if upper is not None and jnp.isfinite(upper):
            up_i.append(idx)
            up_v.append(float(upper))
            bound_names.append(f"{name}_max")

    if enforce_voltage:
        for bid in layout.buses:
            bus = network.buses[bid]
            add_box(f"vm_{bid}", bus.vm_min, bus.vm_max)
    base = network.base_mva
    for gid in layout.generators:
        gen = network.generators[gid]
        add_box(f"pg_{gid}", gen.p_min_mw / base, gen.p_max_mw / base)
        add_box(f"qg_{gid}", gen.q_min_mvar / base, gen.q_max_mvar / base)
    for aid in layout.tap_branches:
        add_box(f"tap_{aid}", 0.9, 1.1)
    for aid in layout.shift_branches:
        add_box(f"shift_{aid}", -0.5236, 0.5236)     # +-30 degrees

    rated, rate_sq = [], []
    if enforce_ratings:
        for i, aid in enumerate(network.branch_ids):
            rate = network.branches[aid].rate_mva
            if rate is not None:
                rated.append(i)
                rate_sq.append((rate / base) ** 2)

    ang_i, ang_lo, ang_hi = [], [], []
    if enforce_angles:
        for i, aid in enumerate(network.branch_ids):
            br = network.branches[aid]
            lo = br.angle_min if br.angle_min is not None else -jnp.inf
            hi = br.angle_max if br.angle_max is not None else jnp.inf
            if abs(float(lo)) < VACUOUS_ANGLE_RAD or abs(float(hi)) < VACUOUS_ANGLE_RAD:
                ang_i.append(i)
                ang_lo.append(float(max(lo, -VACUOUS_ANGLE_RAD)))
                ang_hi.append(float(min(hi, VACUOUS_ANGLE_RAD)))

    from difflow_power.residuals import residual_names

    return OPFStructure(
        lower_indices=tuple(lo_i),
        lower_values=tuple(lo_v),
        upper_indices=tuple(up_i),
        upper_values=tuple(up_v),
        fixed_indices=tuple(fx_i),
        fixed_values=tuple(fx_v),
        rated_branches=tuple(rated),
        rate_squared=tuple(rate_sq),
        angle_branches=tuple(ang_i),
        angle_min=tuple(ang_lo),
        angle_max=tuple(ang_hi),
        bound_names=tuple(bound_names),
        equality_names=tuple(
            residual_names(network, layout)
            + [f"fixed_{names[i]}" for i in fx_i]
        ),
    )


def generation_cost(
    x: Array, network: PowerNetwork, layout: PowerStateLayout
) -> Array:
    """Total generation cost ($/h) of a packed state.

    Each unit's polynomial is evaluated on its output in MW, because
    that is the unit cost curves are quoted in; the state is per unit,
    so the conversion happens here and nowhere else.
    """
    pg_mw = x[layout.slice_pg] * network.base_mva
    total = jnp.asarray(0.0, dtype=jnp.float64)
    for i, gid in enumerate(layout.generators):
        total = total + polynomial_cost(
            pg_mw[i], network.generators[gid].cost
        )
    return total


def acopf_problem(
    network: PowerNetwork,
    layout: PowerStateLayout,
    structure: OPFStructure,
) -> NLP:
    """Assemble the AC-OPF as an :class:`~difflow_power.ipm.NLP`.

    ``params`` is a ``(pd, qd)`` pair of per-bus demand arrays (pu), so
    that differentiating the solution with respect to ``params``
    answers "what does another MW at this bus cost?" --- which is what
    :meth:`ACOPFResult.price_sensitivity` uses. Any other quantity can
    be exposed the same way by closing over it here.

    Returns:
        The NLP. Its equalities are the network physics plus any
        collapsed fixed bounds; its inequalities are the finite box
        bounds, the thermal ratings and the angle limits, in the order
        :meth:`OPFStructure.inequality_names` gives.
    """
    n = layout.size
    lo_i = jnp.asarray(structure.lower_indices, dtype=int)
    lo_v = jnp.asarray(structure.lower_values, dtype=jnp.float64)
    up_i = jnp.asarray(structure.upper_indices, dtype=int)
    up_v = jnp.asarray(structure.upper_values, dtype=jnp.float64)
    fx_i = jnp.asarray(structure.fixed_indices, dtype=int)
    fx_v = jnp.asarray(structure.fixed_values, dtype=jnp.float64)
    rated = jnp.asarray(structure.rated_branches, dtype=int)
    rate_sq = jnp.asarray(structure.rate_squared, dtype=jnp.float64)
    ang_i = jnp.asarray(structure.angle_branches, dtype=int)
    ang_lo = jnp.asarray(structure.angle_min, dtype=jnp.float64)
    ang_hi = jnp.asarray(structure.angle_max, dtype=jnp.float64)

    f_idx, t_idx = network.branch_index_arrays()
    n_bus = layout.n_bus
    # The bound rows evaluate as one block of lowers then one of uppers,
    # but are NAMED variable by variable so a report reads in order.
    # Compute the permutation between the two once, here, rather than on
    # every trace of the constraint function.
    bound_order = jnp.asarray(_bound_permutation(structure), dtype=int)

    def objective(x, params):
        return generation_cost(x, network, layout)

    def equalities(x, params):
        rows = [power_flow_residuals(x, network, layout, demand=params)]
        if fx_i.shape[0]:
            rows.append(x[fx_i] - fx_v)
        return jnp.concatenate(rows)

    def inequalities(x, params):
        rows = []
        if lo_i.shape[0]:
            rows.append(lo_v - x[lo_i])
        if up_i.shape[0]:
            rows.append(x[up_i] - up_v)
        if rows:
            rows = [jnp.concatenate(rows)[bound_order]]
        if rated.shape[0]:
            s_from, s_to = branch_flows(x, network, layout)
            rows.append(
                jnp.stack(
                    [
                        apparent_power_squared(s_from[rated]) - rate_sq,
                        apparent_power_squared(s_to[rated]) - rate_sq,
                    ],
                    axis=1,
                ).reshape(-1)
            )
        if ang_i.shape[0]:
            va = x[layout.slice_va]
            diff = va[f_idx[ang_i]] - va[t_idx[ang_i]]
            rows.append(
                jnp.stack([diff - ang_hi, ang_lo - diff], axis=1).reshape(-1)
            )
        if not rows:
            return jnp.zeros((0,), dtype=jnp.float64)
        return jnp.concatenate(rows)

    return NLP(
        objective=objective,
        n=n,
        m_eq=2 * n_bus + 1 + len(structure.fixed_indices),
        m_in=structure.n_inequality,
        equalities=equalities,
        inequalities=inequalities if structure.n_inequality else None,
    )


def _bound_permutation(structure: OPFStructure) -> list[int]:
    """Map the ``[all lowers, all uppers]`` block onto ``bound_names`` order.

    ``bound_names`` interleaves each variable's ``_min`` and ``_max``
    rows so a report reads variable by variable; the vectorised
    evaluation naturally produces all the lowers then all the uppers.
    This is the permutation between them, computed once.
    """
    n_lo = len(structure.lower_indices)
    lo_seen, up_seen = 0, 0
    order = []
    for name in structure.bound_names:
        if name.endswith("_min"):
            order.append(lo_seen)
            lo_seen += 1
        else:
            order.append(n_lo + up_seen)
            up_seen += 1
    return order


# =============================================================================
# Results
# =============================================================================


@dataclass
class ACOPFResult(ParamsMixin):
    """A solved AC optimal power flow, in engineering units.

    Attributes:
        network: the network solved.
        layout: the state packing.
        structure: which constraints were carried.
        nlp: the assembled problem, kept so sensitivities can re-solve
            its KKT system.
        ipm: the raw interior-point result, including the multipliers.
        demand: the ``(pd, qd)`` the solve used (pu).
    """

    network: PowerNetwork
    layout: PowerStateLayout
    structure: OPFStructure
    nlp: NLP
    ipm: IPMResult
    demand: tuple[Array, Array]

    @property
    def x(self) -> Array:
        return self.ipm.x

    @property
    def converged(self) -> bool:
        return self.ipm.converged

    @property
    def cost(self) -> float:
        """Optimal generation cost ($/h)."""
        return self.ipm.objective

    # -- the solution -----------------------------------------------------

    @property
    def pg_mw(self) -> dict[str, float]:
        """Optimal real dispatch (MW)."""
        pg = self.x[self.layout.slice_pg] * self.network.base_mva
        return {g: float(pg[i]) for i, g in enumerate(self.layout.generators)}

    @property
    def qg_mvar(self) -> dict[str, float]:
        """Optimal reactive dispatch (MVAr)."""
        qg = self.x[self.layout.slice_qg] * self.network.base_mva
        return {g: float(qg[i]) for i, g in enumerate(self.layout.generators)}

    @property
    def vm(self) -> dict[str, float]:
        """Optimal bus voltage magnitudes (pu)."""
        vm = self.x[self.layout.slice_vm]
        return {b: float(vm[i]) for i, b in enumerate(self.layout.buses)}

    @property
    def va_degrees(self) -> dict[str, float]:
        va = self.x[self.layout.slice_va]
        return {
            b: float(jnp.degrees(va[i]))
            for i, b in enumerate(self.layout.buses)
        }

    @property
    def branch_mva(self) -> dict[str, float]:
        """Apparent power on each branch (MVA), the larger of its two ends."""
        s_f, s_t = branch_flows(self.x, self.network, self.layout)
        base = self.network.base_mva
        return {
            a: float(jnp.maximum(jnp.abs(s_f[i]), jnp.abs(s_t[i]))) * base
            for i, a in enumerate(self.network.branch_ids)
        }

    @property
    def losses_mw(self) -> float:
        s_f, s_t = branch_flows(self.x, self.network, self.layout)
        return float(jnp.sum(jnp.real(s_f + s_t))) * self.network.base_mva

    # -- prices -----------------------------------------------------------

    @property
    def lmp_mw(self) -> dict[str, float]:
        """Locational marginal price of real power ($/MWh) per bus.

        The multiplier on the bus's real-power balance, sign-corrected
        and put on an MW basis. Equal everywhere but for losses when
        nothing is congested; separated when something is.
        """
        lam = self.ipm.lam[: self.layout.n_bus]
        return {
            b: -float(lam[i]) / self.network.base_mva
            for i, b in enumerate(self.layout.buses)
        }

    @property
    def lmp_mvar(self) -> dict[str, float]:
        """Marginal price of REACTIVE power ($/MVArh) per bus.

        Usually small and usually ignored by markets, but not zero: vars
        cost real power to move, and near a binding voltage limit they
        can cost a great deal.
        """
        lam = self.ipm.lam[self.layout.n_bus: 2 * self.layout.n_bus]
        return {
            b: -float(lam[i]) / self.network.base_mva
            for i, b in enumerate(self.layout.buses)
        }

    def binding(self, tol: float = 1e-6) -> dict[str, float]:
        """Binding inequality constraints and their shadow prices.

        The value is the multiplier: what the objective would improve by
        per unit of relaxation. A branch rating's shadow price is the
        congestion rent that makes LMPs separate; a voltage limit's says
        what a capacitor bank at that bus would be worth.
        """
        names = self.structure.inequality_names
        active = self.ipm.active(tol)
        return {
            names[i]: float(self.ipm.z[i])
            for i in range(len(names))
            if bool(active[i])
        }

    # -- sensitivities ----------------------------------------------------

    def solution_sensitivity(self) -> Array:
        """``d(state) / d(pd)``: how the optimum moves with real demand.

        Implicit differentiation of the KKT system at the solution ---
        see :func:`difflow_power.ipm.differentiable_solution`. Shape
        ``(layout.size, n_bus)``.
        """
        pd, qd = self.demand
        n = self.layout.size

        def solve(p):
            return differentiable_solution(
                self.nlp, self.ipm.kkt, (p, qd), self.ipm.mu
            )[:n]

        return jax.jacobian(solve)(pd)

    def price_sensitivity(self) -> Array:
        """``d(cost) / d(pd)`` in $/h per pu, an independent LMP check.

        Differentiates the OPTIMAL COST with respect to demand rather
        than reading a multiplier. Envelope theorem says the two agree;
        :meth:`check_prices` asserts that they do, which is the test
        that catches a sign or scaling error in either path.
        """
        pd, qd = self.demand
        n = self.layout.size

        def cost(p):
            w = differentiable_solution(
                self.nlp, self.ipm.kkt, (p, qd), self.ipm.mu
            )
            return generation_cost(w[:n], self.network, self.layout)

        return jax.grad(cost)(pd)

    def check_prices(self) -> dict[str, float]:
        """Largest disagreement between the two ways of getting an LMP.

        Returns ``{bus: |multiplier LMP - gradient LMP|}`` in $/MWh.
        Everything should be at solver noise; anything else means the
        multipliers are not what they are claimed to be.
        """
        grad_lmp = self.price_sensitivity() / self.network.base_mva
        lmp = self.lmp_mw
        return {
            b: abs(lmp[b] - float(grad_lmp[i]))
            for i, b in enumerate(self.layout.buses)
        }

    # -- reporting --------------------------------------------------------

    def summary(self) -> str:
        state = "converged" if self.converged else "DID NOT CONVERGE"
        lmp = self.lmp_mw
        lo, hi = min(lmp.values()), max(lmp.values())
        bind = self.binding()
        return (
            f"{self.network.name or 'AC-OPF'}: {state} in "
            f"{self.ipm.iterations} interior-point iterations, cost "
            f"${self.cost:.2f}/h for {self.network.total_load_mw:.1f} MW "
            f"load ({self.losses_mw:.2f} MW loss); LMP {lo:.2f} to "
            f"{hi:.2f} $/MWh; {len(bind)} binding constraints"
            + (f" ({', '.join(sorted(bind)[:4])})" if bind else "")
        )

    def __repr__(self) -> str:
        return f"ACOPFResult({self.summary()})"


# =============================================================================
# Solving
# =============================================================================


def solve_acopf(
    network: PowerNetwork,
    *,
    layout: PowerStateLayout | None = None,
    structure: OPFStructure | None = None,
    x0: Array | None = None,
    demand: tuple[Array, Array] | None = None,
    warm_start: bool = True,
    enforce_ratings: bool = True,
    enforce_voltage: bool = True,
    enforce_angles: bool = True,
    tap_branches: Sequence[str] = (),
    shift_branches: Sequence[str] = (),
    **ipm_options: Any,
) -> ACOPFResult:
    """Solve the AC optimal power flow.

    Args:
        network: the network to optimise.
        layout: state packing; built from the network by default,
            extended with any ``tap_branches`` / ``shift_branches``.
        structure: the constraint set; built by
            :func:`acopf_structure` by default.
        x0: starting point; see ``warm_start``.
        demand: ``(pd, qd)`` per-bus arrays (pu) replacing the
            network's loads. This is the differentiable handle on the
            problem.
        warm_start: start from a converged power flow rather than a
            flat start. Costs one Newton solve and typically halves the
            interior-point iterations, because the equality constraints
            begin satisfied and the method only has to work on
            optimality. Turn it off for a case whose power flow does not
            converge --- the OPF often still does, since it can move
            voltages the power flow had pinned.
        enforce_ratings, enforce_voltage, enforce_angles: which limit
            families to carry; see :func:`acopf_structure`.
        tap_branches: transformers whose tap ratio is a decision
            variable, bounded to +-10%.
        shift_branches: phase shifters whose angle is a decision
            variable, bounded to +-30 degrees.
        **ipm_options: forwarded to
            :func:`difflow_power.ipm.solve_nlp` (``max_iterations``,
            tolerances, ``verbose``).

    Returns:
        An :class:`ACOPFResult`.

    Example:
        >>> import difflow_power as dp
        >>> res = dp.solve_acopf(dp.cases.case9())
        >>> round(res.cost, 2)
        5296.69
    """
    layout = layout or power_state_layout(
        network, tap_branches=tap_branches, shift_branches=shift_branches
    )
    structure = structure or acopf_structure(
        network,
        layout,
        enforce_ratings=enforce_ratings,
        enforce_voltage=enforce_voltage,
        enforce_angles=enforce_angles,
    )
    nlp = acopf_problem(network, layout, structure)

    if demand is None:
        demand = network.load_arrays_pu()

    if x0 is None:
        if warm_start:
            pf = solve_power_flow(network, layout=layout, demand=demand)
            x0 = pf.x if pf.converged else flat_start(network, layout)
        else:
            x0 = flat_start(network, layout)
    x0 = _nudge_interior(jnp.asarray(x0, dtype=jnp.float64), structure)

    ipm = solve_nlp(nlp, x0, demand, **ipm_options)
    return ACOPFResult(
        network=network,
        layout=layout,
        structure=structure,
        nlp=nlp,
        ipm=ipm,
        demand=demand,
    )


def _nudge_interior(x: Array, structure: OPFStructure, margin: float = 1e-4):
    """Pull a starting point strictly inside its bounds.

    A warm start from a power flow routinely sits ON a limit --- a PV
    bus at exactly 1.0 pu against a 1.0 pu ceiling, a unit at exactly
    its schedule --- and the log barrier is ``-inf`` there. Moving it
    inside by a hair costs nothing and avoids an immediate line-search
    failure.
    """
    for idx, val in zip(structure.lower_indices, structure.lower_values):
        x = x.at[idx].set(jnp.maximum(x[idx], val + margin))
    for idx, val in zip(structure.upper_indices, structure.upper_values):
        x = x.at[idx].set(jnp.minimum(x[idx], val - margin))
    for idx, val in zip(structure.fixed_indices, structure.fixed_values):
        x = x.at[idx].set(val)
    return x
