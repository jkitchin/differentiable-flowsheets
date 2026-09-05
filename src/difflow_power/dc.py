"""The DC approximation: linearised power flow, DC-OPF, PTDF and LODF.

Three assumptions turn the AC equations into a linear model: branch
resistance is negligible beside reactance, every voltage magnitude is
1.0 pu, and angle differences are small enough that
``sin(theta) ~ theta``. What survives is

.. math::

    P_{inj} = B_{bus} \\theta + P_{shift}, \\qquad
    P_{f} = B_{f} \\theta + P_{f,shift}

--- real power only, no losses, no voltages, and linear. That is a
severe approximation and an indispensable one: it is what every
wholesale electricity market clears on, what security-constrained unit
commitment is built from, and what makes contingency screening over
thousands of outages tractable.

What it is good for, and what it is not
---------------------------------------

DC flows on a well-behaved transmission network land within a few
percent of the AC answer, and DC-OPF is convex, so it always has a
solution and always finds the global one. It knows nothing about
reactive power, voltage limits or losses, so it cannot tell you that a
dispatch collapses the voltage at a load pocket, and its cost is
systematically optimistic --- losses are real power somebody has to
generate. Use it to screen, to warm-start, and to price; confirm with
:func:`difflow_power.opf.solve_acopf`.

PTDF and LODF
-------------

Because the model is linear, the map from injections to flows is a
constant matrix. :func:`ptdf` is that matrix: ``PTDF[l, b]`` is the MW
that appears on branch ``l`` per MW injected at bus ``b`` and withdrawn
at the slack. :func:`lodf` is its outage counterpart: ``LODF[l, k]`` is
the fraction of branch ``k``'s pre-outage flow that lands on branch
``l`` when ``k`` trips. Together they answer "what breaks if this line
trips" for every line at the cost of one matrix multiply, which is why
contingency analysis is a DC calculation even at utilities that dispatch
on AC.

Both are built from ``jnp`` operations on the same ``Bbus`` the DC power
flow uses, so they differentiate with respect to line reactances like
everything else here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin

from difflow_power.ipm import IPMResult, NLP, solve_nlp
from difflow_power.network import PowerNetwork
from difflow_power.physics import polynomial_cost


@dataclass
class DCMatrices(ParamsMixin):
    """The linear model's matrices; see :func:`dc_matrices`.

    Attributes:
        b_bus: ``(n_bus, n_bus)`` nodal susceptance matrix.
        b_f: ``(n_branch, n_bus)`` map from angles to branch flows.
        p_bus_shift: ``(n_bus,)`` nodal injections from phase shifters.
        p_f_shift: ``(n_branch,)`` flow offsets from phase shifters.
        susceptance: ``(n_branch,)`` each branch's ``1 / (x tau)``.
    """

    b_bus: Array
    b_f: Array
    p_bus_shift: Array
    p_f_shift: Array
    susceptance: Array

    def __iter__(self):
        """Unpack as ``(b_bus, b_f, p_bus_shift, p_f_shift)``."""
        return iter(
            (self.b_bus, self.b_f, self.p_bus_shift, self.p_f_shift)
        )


def dc_matrices(
    network: PowerNetwork, branch_params: dict[str, Array] | None = None
) -> DCMatrices:
    """Build the DC model's matrices.

    Follows MATPOWER's ``makeBdc``: each branch contributes a
    susceptance ``b = 1 / (x tau)``, and a phase shifter contributes a
    fixed injection ``-b theta_shift`` at its from end --- which is
    exactly how a phase shifter controls flow, by injecting an angle
    rather than power.

    Args:
        network: the network.
        branch_params: optional overrides for ``x``, ``tap``, ``shift``,
            as arrays in branch order. Traced values are fine.

    Returns:
        A :class:`DCMatrices` with

        - ``b_bus``: ``(n_bus, n_bus)`` nodal susceptance matrix,
        - ``b_f``: ``(n_branch, n_bus)`` mapping angles to branch flows,
        - ``p_bus_shift``: ``(n_bus,)`` injections from phase shifters,
        - ``p_f_shift``: ``(n_branch,)`` flow offsets from phase shifters,
        - ``susceptance``: ``(n_branch,)`` the ``b`` of each branch.
    """
    params = network.branch_param_arrays()
    if branch_params:
        params = {**params, **branch_params}
    b = 1.0 / (params["x"] * params["tap"])
    p_f_shift = -b * params["shift"]

    f_idx, t_idx = network.branch_index_arrays()
    n_bus, n_branch = network.n_bus, network.n_branch
    rows = jnp.arange(n_branch)
    b_f = (
        jnp.zeros((n_branch, n_bus), dtype=jnp.float64)
        .at[rows, f_idx].add(b)
        .at[rows, t_idx].add(-b)
    )
    # Cft^T Bf, written as scatter-adds so it stays traceable.
    b_bus = (
        jnp.zeros((n_bus, n_bus), dtype=jnp.float64)
        .at[f_idx].add(b_f)
        .at[t_idx].add(-b_f)
    )
    p_bus_shift = (
        jnp.zeros(n_bus, dtype=jnp.float64)
        .at[f_idx].add(p_f_shift)
        .at[t_idx].add(-p_f_shift)
    )
    return DCMatrices(b_bus, b_f, p_bus_shift, p_f_shift, b)


def _shunt_draw_pu(network: PowerNetwork) -> Array:
    """Real power drawn by bus shunts at 1.0 pu voltage.

    The DC model holds every magnitude at 1.0, so a shunt conductance
    draws exactly ``Gs / base_mva`` --- a constant load, not a
    voltage-dependent one.
    """
    return jnp.real(network.shunt_array_pu())


# =============================================================================
# DC power flow
# =============================================================================


@dataclass
class DCPowerFlowResult(ParamsMixin):
    """A solved DC power flow.

    Attributes:
        network: the network solved.
        va: bus voltage angles (rad).
        p_from: real power leaving each branch's from bus (pu). In the
            DC model this is the whole flow --- there are no losses, so
            the to-end flow is its negative.
        p_slack: the real power the slack bus had to supply (pu). Kept
            as an array, not a float: this result is produced inside
            traced code --- differentiating a DC flow with respect to a
            line reactance, for one --- and converting to a Python float
            there would break the trace.
    """

    network: PowerNetwork
    va: Array
    p_from: Array
    p_slack: Array

    @property
    def va_degrees(self) -> dict[str, float]:
        return {
            b: float(jnp.degrees(self.va[i]))
            for i, b in enumerate(self.network.bus_ids)
        }

    @property
    def branch_mw(self) -> dict[str, float]:
        base = self.network.base_mva
        return {
            a: float(self.p_from[i]) * base
            for i, a in enumerate(self.network.branch_ids)
        }

    def summary(self) -> str:
        mw = self.branch_mw
        worst = max(mw, key=lambda a: abs(mw[a]))
        return (
            f"{self.network.name or 'DC power flow'}: slack supplies "
            f"{float(self.p_slack) * self.network.base_mva:.1f} MW; heaviest "
            f"branch {worst} at {mw[worst]:.1f} MW"
        )

    def __repr__(self) -> str:
        return f"DCPowerFlowResult({self.summary()})"


def solve_dc_power_flow(
    network: PowerNetwork,
    *,
    injections: Array | None = None,
    branch_params: dict[str, Array] | None = None,
) -> DCPowerFlowResult:
    """Solve the linear power flow: one linear system, no iteration.

    The slack bus's angle is pinned and its row dropped, leaving a
    nonsingular system in the remaining angles. That is the same rank
    argument the AC formulation makes with its reference row, in its
    simplest possible form.

    Args:
        network: the network.
        injections: net real injection per bus (pu), generation minus
            load. Defaults to the network's own schedule, with the slack
            bus's own entry ignored since it is solved for.
        branch_params: overrides for ``x``, ``tap``, ``shift``.

    Returns:
        A :class:`DCPowerFlowResult`.
    """
    m = dc_matrices(network, branch_params)
    if injections is None:
        pd, _ = network.load_arrays_pu()
        gen_idx = network.generator_bus_indices()
        pg = jnp.asarray(
            [
                g.p_mw / network.base_mva
                for g in network.generators.values()
            ],
            dtype=jnp.float64,
        )
        injections = (
            jnp.zeros(network.n_bus).at[gen_idx].add(pg) - pd
        )
    rhs = injections - m.p_bus_shift - _shunt_draw_pu(network)

    slack = network.bus_index[network.slack_bus]
    keep = jnp.asarray(
        [i for i in range(network.n_bus) if i != slack], dtype=int
    )
    va_reduced = jnp.linalg.solve(
        m.b_bus[jnp.ix_(keep, keep)], rhs[keep]
    )
    va = (
        jnp.zeros(network.n_bus, dtype=jnp.float64)
        .at[keep].set(va_reduced)
        .at[slack].set(network.buses[network.slack_bus].va_reference)
    )
    p_from = m.b_f @ va + m.p_f_shift
    p_slack = (m.b_bus @ va + m.p_bus_shift)[slack]
    return DCPowerFlowResult(network, va, p_from, p_slack)


# =============================================================================
# PTDF and LODF
# =============================================================================


def ptdf(
    network: PowerNetwork,
    *,
    slack: str | None = None,
    branch_params: dict[str, Array] | None = None,
) -> Array:
    """Power transfer distribution factors, shape ``(n_branch, n_bus)``.

    ``PTDF[l, b]`` is the MW appearing on branch ``l`` per MW injected
    at bus ``b`` and withdrawn at the reference. The reference column is
    identically zero by construction: an injection there goes nowhere.

    Args:
        network: the network.
        slack: reference bus; defaults to the network's slack. The
            choice does not change any physical transfer --- only the
            reference the columns are measured against --- so a PTDF
            difference between two columns is slack-independent.
        branch_params: overrides for ``x``, ``tap``, ``shift``.

    Returns:
        The PTDF matrix, dimensionless.
    """
    m = dc_matrices(network, branch_params)
    ref = network.bus_index[slack or network.slack_bus]
    keep = jnp.asarray(
        [i for i in range(network.n_bus) if i != ref], dtype=int
    )
    inverse = jnp.linalg.inv(m.b_bus[jnp.ix_(keep, keep)])
    return (
        jnp.zeros((network.n_branch, network.n_bus), dtype=jnp.float64)
        .at[:, keep].set(m.b_f[:, keep] @ inverse)
    )


def lodf(
    network: PowerNetwork,
    *,
    slack: str | None = None,
    branch_params: dict[str, Array] | None = None,
) -> Array:
    """Line outage distribution factors, shape ``(n_branch, n_branch)``.

    ``LODF[l, k]`` is the fraction of branch ``k``'s pre-outage flow
    that appears on branch ``l`` after ``k`` trips, so the post-outage
    flow is ``P_l + LODF[l, k] P_k``. Contingency analysis over every
    single-branch outage is then one matrix product rather than
    ``n_branch`` power flows.

    The diagonal is ``-1``: an outaged branch loses all of its own flow.

    A branch whose outage would ISLAND the network has
    ``Phi[k, k] = 1``, and its column is undefined --- there is no
    post-outage flow to redistribute because there is no post-outage
    network. Those columns come back as ``nan`` rather than as a large
    finite number, so a screening loop cannot mistake a disconnection
    for a manageable overload.
    """
    factors = ptdf(network, slack=slack, branch_params=branch_params)
    f_idx, t_idx = network.branch_index_arrays()
    # Phi[l, k]: flow change on l per unit injected at k's from bus and
    # withdrawn at its to bus -- i.e. per unit of flow rerouted around k.
    phi = factors[:, f_idx] - factors[:, t_idx]
    denominator = 1.0 - jnp.diag(phi)
    islanding = jnp.abs(denominator) < 1e-9
    safe = jnp.where(islanding, 1.0, denominator)
    out = phi / safe[None, :]
    out = jnp.where(islanding[None, :], jnp.nan, out)
    n = network.n_branch
    return out.at[jnp.arange(n), jnp.arange(n)].set(
        jnp.where(islanding, jnp.nan, -1.0)
    )


def contingency_flows(
    network: PowerNetwork,
    base_flows: Array,
    *,
    slack: str | None = None,
) -> Array:
    """Post-outage branch flows for every single-branch contingency.

    Args:
        network: the network.
        base_flows: pre-outage branch flows (pu), e.g. from
            :func:`solve_dc_power_flow`.
        slack: reference bus for the underlying PTDF.

    Returns:
        ``(n_branch, n_branch)``: entry ``[l, k]`` is the flow on branch
        ``l`` after branch ``k`` trips. Column ``k`` is ``nan`` where
        the outage islands the network, and its ``[k, k]`` entry is 0
        (the outaged branch carries nothing).
    """
    factors = lodf(network, slack=slack)
    return base_flows[:, None] + factors * base_flows[None, :]


# =============================================================================
# DC-OPF
# =============================================================================


@dataclass
class DCOPFResult(ParamsMixin):
    """A solved DC optimal power flow.

    Attributes:
        network: the network optimised.
        nlp: the assembled convex QP.
        ipm: the raw interior-point result.
        injections: the fixed demand (pu) the solve used.
        constraint_names: label of each inequality row.
    """

    network: PowerNetwork
    nlp: NLP
    ipm: IPMResult
    injections: Array
    constraint_names: tuple[str, ...]

    @property
    def n_bus(self) -> int:
        return self.network.n_bus

    @property
    def converged(self) -> bool:
        return self.ipm.converged

    @property
    def cost(self) -> float:
        """Optimal generation cost ($/h). Optimistic: the DC model has
        no losses, so nobody pays to generate them."""
        return self.ipm.objective

    @property
    def va(self) -> Array:
        return self.ipm.x[: self.n_bus]

    @property
    def pg_mw(self) -> dict[str, float]:
        pg = self.ipm.x[self.n_bus:] * self.network.base_mva
        return {
            g: float(pg[i]) for i, g in enumerate(self.network.generator_ids)
        }

    @property
    def branch_mw(self) -> dict[str, float]:
        m = dc_matrices(self.network)
        flows = (m.b_f @ self.va + m.p_f_shift) * self.network.base_mva
        return {
            a: float(flows[i])
            for i, a in enumerate(self.network.branch_ids)
        }

    @property
    def lmp_mw(self) -> dict[str, float]:
        """Locational marginal prices ($/MWh).

        Same convention as the AC result: the multiplier on the bus's
        balance row, negated and put on an MW basis. Without congestion
        every DC LMP is IDENTICAL --- the model has no losses, so there
        is nothing else to separate them --- which makes it very easy to
        see what a binding constraint is doing.
        """
        lam = self.ipm.lam[: self.n_bus]
        return {
            b: -float(lam[i]) / self.network.base_mva
            for i, b in enumerate(self.network.bus_ids)
        }

    def binding(self, tol: float = 1e-6) -> dict[str, float]:
        """Binding constraints and their shadow prices."""
        active = self.ipm.active(tol)
        return {
            self.constraint_names[i]: float(self.ipm.z[i])
            for i in range(len(self.constraint_names))
            if bool(active[i])
        }

    def summary(self) -> str:
        state = "converged" if self.converged else "DID NOT CONVERGE"
        lmp = self.lmp_mw
        bind = self.binding()
        return (
            f"{self.network.name or 'DC-OPF'}: {state} in "
            f"{self.ipm.iterations} iterations, cost ${self.cost:.2f}/h; "
            f"LMP {min(lmp.values()):.2f} to {max(lmp.values()):.2f} "
            f"$/MWh; {len(bind)} binding constraints"
        )

    def __repr__(self) -> str:
        return f"DCOPFResult({self.summary()})"


def solve_dcopf(
    network: PowerNetwork,
    *,
    demand: Array | None = None,
    enforce_ratings: bool = True,
    **ipm_options: Any,
) -> DCOPFResult:
    """Solve the DC optimal power flow: a convex QP.

    The decision vector is ``(theta, P_g)``; the equalities are the
    linear nodal balance plus the reference angle; the inequalities are
    the generator boxes and the branch ratings, applied to the signed
    flow in both directions.

    It runs through the SAME interior-point solver as the AC problem
    --- a QP is an NLP with a constant Hessian --- which is deliberate:
    one set of multiplier conventions means a DC LMP and an AC LMP are
    directly comparable, and their difference is a clean measure of what
    the linearisation costs.

    Args:
        network: the network.
        demand: per-bus real demand (pu). Defaults to the network's
            loads. This is the differentiable handle.
        enforce_ratings: include branch flow limits.
        **ipm_options: forwarded to
            :func:`difflow_power.ipm.solve_nlp`.

    Returns:
        A :class:`DCOPFResult`.

    Example:
        >>> import difflow_power as dp
        >>> dc = dp.solve_dcopf(dp.cases.case5())
        >>> dc.converged
        True
    """
    n_bus, n_gen = network.n_bus, network.n_gen
    base = network.base_mva
    m = dc_matrices(network)
    gen_idx = network.generator_bus_indices()
    slack = network.bus_index[network.slack_bus]
    shunt = _shunt_draw_pu(network)

    if demand is None:
        demand, _ = network.load_arrays_pu()

    # The names are built in exactly the order the constraint function
    # emits its rows -- generator bounds interleaved min/max per unit,
    # then the two flow directions per rated branch -- so a shadow price
    # can be read straight off by name.
    rated, rate_pu, names = [], [], []
    for gid in network.generator_ids:
        names += [f"pg_{gid}_min", f"pg_{gid}_max"]
    if enforce_ratings:
        for i, aid in enumerate(network.branch_ids):
            rate = network.branches[aid].rate_mva
            if rate is not None:
                rated.append(i)
                rate_pu.append(rate / base)
                names += [f"rate_{aid}_forward", f"rate_{aid}_reverse"]
    rated_idx = jnp.asarray(rated, dtype=int)
    rate_arr = jnp.asarray(rate_pu, dtype=jnp.float64)
    p_lo = jnp.asarray(
        [network.generators[g].p_min_mw / base for g in network.generator_ids],
        dtype=jnp.float64,
    )
    p_hi = jnp.asarray(
        [network.generators[g].p_max_mw / base for g in network.generator_ids],
        dtype=jnp.float64,
    )
    costs = [network.generators[g].cost for g in network.generator_ids]

    def objective(x, pd):
        pg_mw = x[n_bus:] * base
        total = jnp.asarray(0.0, dtype=jnp.float64)
        for i, coefficients in enumerate(costs):
            total = total + polynomial_cost(pg_mw[i], coefficients)
        return total

    def equalities(x, pd):
        va, pg = x[:n_bus], x[n_bus:]
        injection = jnp.zeros(n_bus, dtype=jnp.float64).at[gen_idx].add(pg)
        balance = (
            injection - pd - shunt - (m.b_bus @ va + m.p_bus_shift)
        )
        reference = va[slack] - network.buses[network.slack_bus].va_reference
        return jnp.concatenate([balance, jnp.atleast_1d(reference)])

    def inequalities(x, pd):
        va, pg = x[:n_bus], x[n_bus:]
        rows = [
            jnp.stack([p_lo - pg, pg - p_hi], axis=1).reshape(-1)
        ]
        if rated_idx.shape[0]:
            flow = (m.b_f @ va + m.p_f_shift)[rated_idx]
            rows.append(
                jnp.stack(
                    [flow - rate_arr, -flow - rate_arr], axis=1
                ).reshape(-1)
            )
        return jnp.concatenate(rows)

    nlp = NLP(
        objective=objective,
        n=n_bus + n_gen,
        m_eq=n_bus + 1,
        m_in=len(names),
        equalities=equalities,
        inequalities=inequalities,
    )
    x0 = jnp.concatenate(
        [jnp.zeros(n_bus, dtype=jnp.float64), 0.5 * (p_lo + p_hi)]
    )
    ipm = solve_nlp(nlp, x0, demand, **ipm_options)
    return DCOPFResult(
        network=network,
        nlp=nlp,
        ipm=ipm,
        injections=demand,
        constraint_names=tuple(names),
    )
