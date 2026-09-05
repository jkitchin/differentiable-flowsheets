"""Sensitivities of a solved network, by automatic differentiation.

Power systems have a long tradition of hand-derived sensitivity
factors: generation shift factors, loss factors, dP/dV matrices, the
reduced Jacobian of voltage stability analysis. Each is a derivative of
a solved state with respect to something, each was historically
derived, coded and validated by hand, and each is one ``jax.jacobian``
call here.

That is the point of building the model differentiably. The factors
below are not reimplementations of the classical formulae; they are the
derivatives themselves, taken through the implicit function theorem at
the converged point, so they are exact by construction and cannot drift
out of step with the model the way a separately-maintained sensitivity
routine does.

What each one answers
---------------------

===============================  ==================================
:func:`demand_sensitivity`       how the whole state moves as load
                                 changes: the AC generalisation of
                                 a shift factor, including voltages
:func:`branch_flow_sensitivity`  AC injection shift factors --- what
                                 :func:`difflow_power.dc.ptdf`
                                 approximates linearly
:func:`loss_sensitivity`         marginal loss factors: the MW of
                                 extra loss caused by a MW at a bus
:func:`parameter_sensitivity`    how the state moves with a line
                                 parameter --- for a rating study, or
                                 for fitting a model to measurements
:func:`voltage_stability_margin` the reduced Jacobian's smallest
                                 singular value, which goes to zero at
                                 the loadability limit
===============================  ==================================

All of them differentiate a POWER FLOW, in which the setpoints are held
and the slack absorbs the change. The corresponding OPF sensitivities
--- where the dispatch re-optimises instead --- are on
:class:`difflow_power.opf.ACOPFResult`, and they are different numbers
answering a different question.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from difflow_power.network import PowerNetwork
from difflow_power.powerflow import (
    Specification,
    flat_start,
    power_flow_system,
    solve_state,
    specification_from_network,
)
from difflow_power.residuals import (
    PowerStateLayout,
    branch_flows,
    power_state_layout,
    total_losses,
)


def _posed(network, layout, spec, x0):
    """Fill in the layout, specification and start point defaults."""
    layout = layout or power_state_layout(network)
    spec = spec or specification_from_network(network)
    if x0 is None:
        x0 = flat_start(network, layout, spec)
    return layout, spec, x0


def demand_sensitivity(
    network: PowerNetwork,
    *,
    layout: PowerStateLayout | None = None,
    spec: Specification | None = None,
    x0: Array | None = None,
    reactive: bool = False,
) -> Array:
    """``d(state) / d(demand)`` at the solved power flow.

    Args:
        network: the network.
        layout: state packing; the plain one by default.
        spec: the setpoints closing the power flow.
        x0: initial guess for the solve.
        reactive: differentiate with respect to REACTIVE demand instead
            of real.

    Returns:
        ``(layout.size, n_bus)``. Row ``layout.index("vm_5")`` and
        column 3, for example, is how much bus 5's voltage moves per pu
        of load added at bus 3 --- the number a voltage-support study
        wants and the DC model cannot produce at all.
    """
    layout, spec, x0 = _posed(network, layout, spec, x0)
    pd, qd = network.load_arrays_pu()

    def solve(demand_block):
        demand = (pd, demand_block) if reactive else (demand_block, qd)
        return solve_state(network, layout, spec, x0, demand=demand)[0]

    return jax.jacobian(solve)(qd if reactive else pd)


def branch_flow_sensitivity(
    network: PowerNetwork,
    *,
    layout: PowerStateLayout | None = None,
    spec: Specification | None = None,
    x0: Array | None = None,
) -> Array:
    """AC injection shift factors: ``d(branch MW) / d(bus demand)``.

    The exact, voltage-aware counterpart of
    :func:`difflow_power.dc.ptdf`. Comparing the two is the honest way
    to decide whether a DC screening study is good enough for a given
    network: where they agree the linearisation is safe, and where they
    do not it is the AC answer that is right.

    Returns:
        ``(n_branch, n_bus)``, in pu per pu. The sign convention is the
        branch's own from -> to direction, and the entry is per unit of
        DEMAND added, so a positive value means more load at that bus
        pushes more power along the branch's reference direction.
    """
    layout, spec, x0 = _posed(network, layout, spec, x0)
    pd, qd = network.load_arrays_pu()

    def flows(demand_block):
        x = solve_state(
            network, layout, spec, x0, demand=(demand_block, qd)
        )[0]
        s_from, _ = branch_flows(x, network, layout)
        return jnp.real(s_from)

    return jax.jacobian(flows)(pd)


def loss_sensitivity(
    network: PowerNetwork,
    *,
    layout: PowerStateLayout | None = None,
    spec: Specification | None = None,
    x0: Array | None = None,
) -> Array:
    """Marginal loss factors: ``d(total loss) / d(demand)``, per bus.

    A MW of load at an electrically distant bus costs more than a MW
    next to a generator, and this says how much more. Markets that
    settle on marginal losses compute exactly this quantity; it is also
    the loss component of a locational marginal price, which is why an
    uncongested AC-OPF has LMPs that vary by a percent or two rather
    than being identical.

    Returns:
        ``(n_bus,)``, dimensionless (pu of loss per pu of load). The
        slack bus's entry is essentially zero: load added there is
        served there.
    """
    layout, spec, x0 = _posed(network, layout, spec, x0)
    pd, qd = network.load_arrays_pu()

    def losses(demand_block):
        x = solve_state(
            network, layout, spec, x0, demand=(demand_block, qd)
        )[0]
        return total_losses(x, network, layout)

    return jax.grad(losses)(pd)


def parameter_sensitivity(
    network: PowerNetwork,
    parameter: str,
    *,
    layout: PowerStateLayout | None = None,
    spec: Specification | None = None,
    x0: Array | None = None,
) -> Array:
    """``d(state) / d(branch parameter)`` at the solved power flow.

    Args:
        network: the network.
        parameter: one of ``"r"``, ``"x"``, ``"b"``, ``"g"``, ``"tap"``,
            ``"shift"``.
        layout: state packing.
        spec: the setpoints.
        x0: initial guess.

    Returns:
        ``(layout.size, n_branch)``.

    Example:
        >>> # what does re-tapping transformer 3 do to every voltage?
        >>> d = parameter_sensitivity(net, "tap")      # doctest: +SKIP
    """
    layout, spec, x0 = _posed(network, layout, spec, x0)
    values = network.branch_param_arrays()
    if parameter not in values:
        raise KeyError(
            f"unknown branch parameter {parameter!r}; expected one of "
            f"{sorted(values)}"
        )

    def solve(value):
        return solve_state(
            network, layout, spec, x0, branch_params={parameter: value}
        )[0]

    return jax.jacobian(solve)(values[parameter])


def voltage_stability_margin(
    x: Array,
    network: PowerNetwork,
    layout: PowerStateLayout | None = None,
    *,
    spec: Specification | None = None,
) -> float:
    """Smallest singular value of the power flow Jacobian at a state.

    The classical proximity-to-collapse index. As a network is loaded
    towards its limit the Jacobian approaches singularity --- that is
    what the nose of a P-V curve IS, the point where the solution
    ceases to exist rather than merely becoming poor --- and this goes
    to zero there.

    Read it as a trend, not an absolute: its magnitude depends on how
    the state is scaled, so a value is meaningful compared against the
    same network at a different loading, not against a different
    network.

    Example:
        >>> for k in (1.0, 1.2, 1.4):                  # doctest: +SKIP
        ...     net = base.scaled_load(k)
        ...     res = solve_power_flow(net)
        ...     print(k, voltage_stability_margin(res.x, net))
    """
    layout = layout or power_state_layout(network)
    spec = spec or specification_from_network(network)
    residual = power_flow_system(network, layout, spec)
    jacobian = jax.jacobian(residual)(x)
    return float(jnp.min(jnp.linalg.svd(jacobian, compute_uv=False)))
