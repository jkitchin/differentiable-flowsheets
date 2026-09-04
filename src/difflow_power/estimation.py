"""State estimation: data reconciliation for an electrical network.

Power system state estimation and chemical-process data reconciliation
are the same computation. Both minimise a weighted least-squares
distance from noisy measurements subject to a model's equations; both
distinguish measured variables from ones inferred through the model;
both detect bad data by looking for a measurement whose residual is too
large to be noise. The vocabulary differs --- "state estimator" versus
"reconciliation", "bad data detection" versus "gross error detection"
--- and the mathematics does not.

So this module is a thin layer over :mod:`difflow.reconciliation`
rather than a reimplementation. It builds the residual closure from a
network and a layout, supplies plausible meter accuracies, and converts
a reconciled state back into the dicts that
:mod:`difflow_power.verify` consumes, so an estimate can be checked
with the same verifier as any other solution of the same network.

What is measured, and what is not
---------------------------------

A real control centre sees a fraction of the network: voltage
magnitudes where there are transducers, injections where there is
metering, and nothing at all at most buses. Angles are essentially
never measured outside a PMU deployment. The default
:func:`measurement_sigma` reflects that --- angles unmeasured,
magnitudes good, injections poor --- and everything unmeasured is
ESTIMATED through the equations rather than assumed.

Demand belongs in the state, not the network
--------------------------------------------

Load is measured badly and is exactly what an estimator corrects, so
:func:`estimate_state` wants a layout built with
``demand_buses``. Keeping it out of the
:class:`~difflow_power.network.PowerNetwork` also avoids a real
awkwardness: measured loads never balance generation, and a network
built from them would be describing a state that cannot exist. Making
them balance is precisely what the estimate does.

Observability
-------------

An unobservable network --- too few measurements to pin down the state
through the equations --- gives a singular normal-equation system.
:func:`difflow.reconciliation.reconcile` checks the structure first and
raises rather than returning a plausible-looking NaN-free answer, so
placing one more meter is a diagnosable fix rather than a guess.
"""

from __future__ import annotations

from typing import Any, Sequence

import jax
import jax.numpy as jnp
from jax import Array

from difflow.reconciliation import (
    MonitorResult,
    MultiReconcileResult,
    ReconcileResult,
    monitor,
    reconcile,
    reconcile_multi,
)

from difflow_power.network import PowerNetwork
from difflow_power.residuals import PowerStateLayout, power_flow_residuals


def network_residual_fn(
    network: PowerNetwork,
    layout: PowerStateLayout,
    **kwargs: Any,
):
    """Close a network and layout into an ``F(x, params) -> Array``.

    ``params``, when given, is merged into the ``branch_params``
    argument of
    :func:`~difflow_power.residuals.power_flow_residuals`, so
    ``jax.grad`` with respect to it answers "how would the estimated
    state move if this line's reactance were different?" --- a
    different question from estimating the reactance, which is done by
    putting it in the layout instead.
    """
    # Bound once, not per call: the closure is evaluated many times (a
    # monitoring campaign, a Gauss-Newton iteration, every column of a
    # Jacobian), so consuming the argument inside it would silently drop
    # the parameters after the first call.
    base_params = dict(kwargs.pop("branch_params", None) or {})

    def residual_fn(x, params=None):
        merged = dict(base_params)
        if params:
            merged.update(params)
        return power_flow_residuals(
            x, network, layout, branch_params=merged or None, **kwargs
        )

    return residual_fn


def measurement_sigma(
    layout: PowerStateLayout,
    *,
    sigma_vm: float = 0.004,
    sigma_va: float = float("inf"),
    sigma_pg: float = 0.02,
    sigma_qg: float = 0.03,
    sigma_pd: float = 0.05,
    sigma_qd: float = 0.08,
    sigma_tap: float = float("inf"),
    sigma_shift: float = float("inf"),
    sigma_shunt: float = float("inf"),
    unmeasured: Sequence[str] = (),
    overrides: dict[str, float] | None = None,
) -> Array:
    """Per-variable measurement accuracies for a network state, in pu.

    Defaults reflect control-centre practice on a 100 MVA base:
    voltage transducers are good to a few tenths of a percent;
    generator output is metered well; load is inferred from feeder
    measurements and is the least reliable number in the system.
    Angles default to ``inf`` --- unmeasured --- because without PMUs
    they are, and the estimator's job is to infer them. Taps, phase
    shifts and switched shunts likewise default to estimated.

    Args:
        layout: the state layout.
        sigma_vm: voltage magnitude accuracy (pu).
        sigma_va: voltage angle accuracy (rad); ``inf`` unless the bus
            has a PMU, in which case override it per bus.
        sigma_pg, sigma_qg: generator output accuracy (pu).
        sigma_pd, sigma_qd: demand accuracy (pu), for layouts carrying
            demand as state.
        sigma_tap, sigma_shift, sigma_shunt: accuracy of the control
            variables; ``inf`` estimates them, a finite value acts as a
            prior on the set-point readback.
        unmeasured: variable names to force to ``inf``.
        overrides: variable name -> sigma, applied last. This is where
            a PMU goes: ``{"va_7": 0.001}``.

    Returns:
        Standard deviations, shape ``(layout.size,)``.

    Raises:
        KeyError: if a name in ``unmeasured`` or ``overrides`` is not a
            variable of this layout.
    """
    values = (
        [sigma_vm] * layout.n_bus
        + [sigma_va] * layout.n_bus
        + [sigma_pg] * layout.n_gen
        + [sigma_qg] * layout.n_gen
        + [sigma_pd] * layout.n_demand
        + [sigma_qd] * layout.n_demand
        + [sigma_tap] * len(layout.tap_branches)
        + [sigma_shift] * len(layout.shift_branches)
        + [sigma_shunt] * len(layout.shunt_buses)
    )
    sigma = dict(zip(layout.names, values))
    for name in unmeasured:
        if name not in sigma:
            raise KeyError(f"{name!r} is not a variable of this layout")
        sigma[name] = float("inf")
    for name, value in (overrides or {}).items():
        if name not in sigma:
            raise KeyError(f"{name!r} is not a variable of this layout")
        sigma[name] = value
    return jnp.asarray(
        [sigma[n] for n in layout.names], dtype=jnp.float64
    )


def perturb(
    x_true: Array,
    sigma: Array,
    key,
    *,
    layout: PowerStateLayout | None = None,
    bad_data: dict[str, float] | None = None,
) -> Array:
    """Add measurement noise, and optionally bad data, to a true state.

    Unmeasured entries (infinite sigma) are left untouched, so the
    result can be handed straight to :func:`estimate_state`.

    Args:
        x_true: the true state.
        sigma: standard deviations; ``inf`` entries get no noise.
        key: a ``jax.random`` key.
        layout: needed only when ``bad_data`` is given.
        bad_data: variable name -> bias **in multiples of that
            variable's sigma**. A meter reading three sigma off is
            what a bad-data test is meant to catch.

    Returns:
        The simulated measurement vector.
    """
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    noise = jax.random.normal(key, x_true.shape)
    finite = jnp.isfinite(sigma)
    y = x_true + jnp.where(finite, jnp.where(finite, sigma, 0.0) * noise, 0.0)
    for name, n_sigma in (bad_data or {}).items():
        if layout is None:
            raise ValueError("a layout is required to place bad data")
        i = layout.index(name)
        y = y.at[i].add(n_sigma * sigma[i])
    return y


def _posed(network, layout, kwargs):
    """The residual closure and the layout-derived defaults.

    The names and the scales of the unmeasured entries both come from
    the layout, and every entry point into
    :mod:`difflow.reconciliation` wants them, so they are filled in
    once here rather than at each call site.
    """
    kwargs.setdefault("names", layout.names)
    kwargs.setdefault("unmeasured_scale", layout.default_scale)
    return network_residual_fn(network, layout), kwargs


def estimate_state(
    network: PowerNetwork,
    y: Array,
    sigma: Array,
    layout: PowerStateLayout,
    **kwargs: Any,
) -> ReconcileResult:
    """Weighted least-squares state estimate from noisy measurements.

    Args:
        network: the network the measurements belong to.
        y: measurement vector, packed by ``layout``. Entries with
            infinite sigma are ignored and may be ``nan``.
        sigma: standard deviations; ``inf`` marks an unmeasured
            variable to be estimated through the equations.
        layout: the state layout. Build it with ``demand_buses`` so the
            estimator can correct load, which is the point.
        **kwargs: forwarded to
            :func:`difflow.reconciliation.reconcile`.

    Returns:
        A :class:`~difflow.reconciliation.ReconcileResult` whose
        ``names`` are ``layout.names``.

    Example:
        >>> layout = power_state_layout(net, demand_buses=net.bus_ids)
        >>> sigma = measurement_sigma(layout)
        >>> y = perturb(x_true, sigma, jax.random.PRNGKey(0))
        >>> est = estimate_state(net, y, sigma, layout)   # doctest: +SKIP
    """
    residual_fn, kwargs = _posed(network, layout, kwargs)
    return reconcile(residual_fn, y, sigma, **kwargs)


def monitor_network(
    network: PowerNetwork,
    measurements: Sequence[Array],
    sigma: Array,
    layout: PowerStateLayout,
    **kwargs: Any,
) -> MonitorResult:
    """Run a campaign of estimates against one fixed network model.

    What a control centre actually does: an estimate every few seconds,
    each one a fresh reconciliation against the SAME model. The value is
    in the trend --- a residual that walks steadily away from zero is a
    meter drifting or a line heating, and neither is visible in any
    single snapshot.

    Args:
        network: the network.
        measurements: one measurement vector per scan.
        sigma: standard deviations, shared across scans.
        layout: the state layout.
        **kwargs: forwarded to
            :func:`difflow.reconciliation.monitor`.

    Returns:
        A :class:`~difflow.reconciliation.MonitorResult`.
    """
    residual_fn, kwargs = _posed(network, layout, kwargs)
    return monitor(residual_fn, measurements, sigma, **kwargs)


def estimate_state_multi(
    network: PowerNetwork,
    measurements: Sequence[Array],
    sigma: Array,
    layout: PowerStateLayout,
    shared: Sequence[str],
    **kwargs: Any,
) -> MultiReconcileResult:
    """Pool several scans that share a parameter.

    A tap position or a line's impedance does not change between scans,
    but the state does. Estimating the shared quantity from one scan is
    usually weakly identified; pooling several, each with its own state
    but a COMMON parameter, is what makes it observable.

    Args:
        network: the network.
        measurements: one measurement vector per scan.
        sigma: standard deviations, shared across scans.
        layout: the state layout, carrying the shared quantity.
        shared: names of the variables held common across scans, e.g.
            ``["tap_br8"]``.
        **kwargs: forwarded to
            :func:`difflow.reconciliation.reconcile_multi`.

    Returns:
        A :class:`~difflow.reconciliation.MultiReconcileResult`.
    """
    residual_fn, kwargs = _posed(network, layout, kwargs)
    return reconcile_multi(
        residual_fn, measurements, sigma, shared=shared, **kwargs
    )


def estimated_values(
    result: ReconcileResult, layout: PowerStateLayout, network: PowerNetwork
) -> dict[str, dict[str, float]]:
    """Unpack an estimate into readable dicts in engineering units.

    Returns:
        ``{"vm": {bus: pu}, "va_degrees": {bus: deg},
        "pg_mw": {gen: MW}, "qg_mvar": {gen: MVAr}}``.
    """
    x = jnp.asarray(result.x)
    base = network.base_mva
    return {
        "vm": {
            b: float(x[layout.slice_vm][i])
            for i, b in enumerate(layout.buses)
        },
        "va_degrees": {
            b: float(jnp.degrees(x[layout.slice_va][i]))
            for i, b in enumerate(layout.buses)
        },
        "pg_mw": {
            g: float(x[layout.slice_pg][i]) * base
            for i, g in enumerate(layout.generators)
        },
        "qg_mvar": {
            g: float(x[layout.slice_qg][i]) * base
            for i, g in enumerate(layout.generators)
        },
    }
