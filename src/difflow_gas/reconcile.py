"""Data reconciliation of a gas transmission network.

A thin layer over :mod:`difflow.reconciliation`: it builds the residual
closure from a network and a layout, supplies plausible meter
accuracies, and converts a reconciled state back into the ``p_bar`` /
``q_kg_s`` dicts that :mod:`difflow_gas.verify` consumes --- so a
reconciliation can be checked with the same verifier as any other
solution of the same network.

Note that measured boundary flows are deliberately *not* stored in a
:class:`~difflow_gas.network.GasNetwork`: real nominations do not sum
to zero, and the network object rejects supplies that do not. They live
in the measurement vector, and making them balance is precisely what
the reconciliation does.
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

from difflow_gas.network import GasNetwork
from difflow_gas.residuals import GasStateLayout, network_residuals


def network_residual_fn(
    network: GasNetwork,
    layout: GasStateLayout,
    *,
    ratios: dict[str, float] | None = None,
    cv_drops_bar: dict[str, float] | None = None,
    **kwargs: Any,
):
    """Close a network and layout into an ``F(x, params) -> Array``.

    ``params``, when given, is merged into the ``efficiencies``
    argument of :func:`~difflow_gas.residuals.network_residuals`, so
    ``jax.grad`` with respect to it answers "how would the reconciled
    state move if this pipe were dirtier?" --- a different question
    from estimating the efficiency, which is done by putting it in the
    layout instead.
    """
    # Bound once, not per call: the closure is evaluated many times (a
    # monitoring campaign, a Gauss-Newton iteration, every column of a
    # Jacobian), so consuming the argument inside it would silently drop
    # the efficiencies after the first call.
    base_efficiencies = dict(kwargs.pop("efficiencies", None) or {})

    def residual_fn(x, params=None):
        eff = dict(base_efficiencies)
        if params:
            eff.update(params)
        return network_residuals(
            x, network, layout, ratios=ratios,
            cv_drops_bar=cv_drops_bar, efficiencies=eff or None, **kwargs,
        )

    return residual_fn


def measurement_sigma(
    layout: GasStateLayout,
    *,
    sigma_p_bar: float = 0.3,
    sigma_q_kg_s: float = 1.0,
    sigma_supply_kg_s: float = 1.5,
    sigma_ratio: float = 0.005,
    sigma_eta: float = float("inf"),
    sigma_dp_bar: float = float("inf"),
    unmeasured: Sequence[str] = (),
    overrides: dict[str, float] | None = None,
) -> Array:
    """Per-variable measurement accuracies for a gas network state.

    Defaults reflect transmission practice: pressure transmitters are
    good, flow meters less so, and nominated boundary flows are the
    least reliable numbers in the system. Efficiencies and valve drops
    default to ``inf``, i.e. estimated rather than measured.

    Args:
        layout: the state layout.
        sigma_p_bar: node pressure accuracy (bar).
        sigma_q_kg_s: arc flow meter accuracy (kg/s).
        sigma_supply_kg_s: boundary flow / nomination accuracy (kg/s).
        sigma_ratio: compressor ratio set-point readback accuracy.
        sigma_eta: pipe efficiency accuracy; ``inf`` estimates it, a
            finite value acts as a prior.
        sigma_dp_bar: control valve drop accuracy.
        unmeasured: variable names to force to ``inf``.
        overrides: variable name -> sigma, applied last.

    Returns:
        Standard deviations, shape ``(layout.size,)``.
    """
    inf = float("inf")
    values = (
        [sigma_p_bar] * layout.n_p
        + [sigma_q_kg_s] * layout.n_q
        + [sigma_supply_kg_s] * layout.n_s
        + [sigma_eta] * len(layout.efficiency_arcs)
        + [sigma_ratio] * len(layout.ratio_arcs)
        + [sigma_dp_bar] * len(layout.cv_arcs)
    )
    sigma = dict(zip(layout.names, values))
    for name in unmeasured:
        if name not in sigma:
            raise KeyError(f"{name!r} is not a variable of this layout")
        sigma[name] = inf
    for name, val in (overrides or {}).items():
        if name not in sigma:
            raise KeyError(f"{name!r} is not a variable of this layout")
        sigma[name] = val
    return jnp.array([sigma[n] for n in layout.names], dtype=jnp.float64)


def perturb(
    x_true: Array,
    sigma: Array,
    key,
    *,
    layout: GasStateLayout | None = None,
    gross_errors: dict[str, float] | None = None,
) -> Array:
    """Add measurement noise, and optionally a gross error, to a state.

    Unmeasured entries (infinite sigma) are left untouched, so the
    result can be handed straight to :func:`reconcile_network`.

    Args:
        x_true: the true state.
        sigma: standard deviations; ``inf`` entries get no noise.
        key: a ``jax.random`` key.
        layout: needed only when ``gross_errors`` is given.
        gross_errors: variable name -> bias **in multiples of that
            variable's sigma**.

    Returns:
        The simulated measurement vector.
    """
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    noise = jax.random.normal(key, x_true.shape)
    finite = jnp.isfinite(sigma)
    y = x_true + jnp.where(finite, jnp.where(finite, sigma, 0.0) * noise, 0.0)
    for name, n_sigma in (gross_errors or {}).items():
        if layout is None:
            raise ValueError("a layout is required to place gross errors")
        i = layout.index(name)
        y = y.at[i].add(n_sigma * sigma[i])
    return y


def reconcile_network(
    network: GasNetwork,
    y: Array,
    sigma: Array,
    layout: GasStateLayout,
    *,
    ratios: dict[str, float] | None = None,
    cv_drops_bar: dict[str, float] | None = None,
    **kwargs: Any,
) -> ReconcileResult:
    """Reconcile measured pressures, flows and nominations of a network.

    Args:
        network: the network the measurements belong to.
        y: measurement vector, packed by ``layout``.
        sigma: standard deviations; ``inf`` = unmeasured.
        layout: the state layout.
        ratios: compressor ratios not carried in the state.
        cv_drops_bar: control valve drops not carried in the state.
        **kwargs: forwarded to :func:`difflow.reconciliation.reconcile`.

    Returns:
        A :class:`~difflow.reconciliation.ReconcileResult` whose
        ``names`` are ``layout.names``.

    Example:
        >>> layout = gas_state_layout(net)
        >>> sigma = measurement_sigma(layout)
        >>> y = perturb(x_true, sigma, jax.random.PRNGKey(0))
        >>> res = reconcile_network(net, y, sigma, layout, ratios={"cs1": 1.2})
    """
    residual_fn, kwargs = _posed(network, layout, ratios, cv_drops_bar, kwargs)
    return reconcile(residual_fn, y, sigma, **kwargs)


def _posed(
    network: GasNetwork,
    layout: GasStateLayout,
    ratios: dict[str, float] | None,
    cv_drops_bar: dict[str, float] | None,
    kwargs: dict[str, Any],
):
    """The residual closure and the layout-derived reconciliation defaults.

    The names and the scales of the unmeasured entries both come from
    the layout, and every entry point into
    :mod:`difflow.reconciliation` wants them, so they are filled in
    once here rather than at each call site.
    """
    kwargs.setdefault("names", layout.names)
    kwargs.setdefault("unmeasured_scale", layout.default_scale)
    residual_fn = network_residual_fn(
        network, layout, ratios=ratios, cv_drops_bar=cv_drops_bar
    )
    return residual_fn, kwargs


def monitor_network(
    network: GasNetwork,
    measurements: Sequence[Array],
    sigma: Array,
    layout: GasStateLayout,
    *,
    ratios: dict[str, float] | None = None,
    cv_drops_bar: dict[str, float] | None = None,
    **kwargs: Any,
) -> MonitorResult:
    """Reconcile a run of measurements against one fixed network.

    The network, the layout and the sigmas are held fixed while the
    data change, which is what makes the resulting statistic series
    readable: every movement in it comes from the data.
    :meth:`~difflow.reconciliation.MonitorResult.diagnose` then says
    whether a meter or the model is at fault.

    Args:
        network: the network the measurements belong to, taken as
            correct for the duration of the campaign.
        measurements: one measurement vector per period, each packed by
            ``layout``.
        sigma: standard deviations; ``inf`` = unmeasured.
        layout: the state layout.
        ratios: compressor ratios not carried in the state.
        cv_drops_bar: control valve drops not carried in the state.
        **kwargs: forwarded to :func:`difflow.reconciliation.monitor`.

    Returns:
        A :class:`~difflow.reconciliation.MonitorResult` whose suspects
        are named by ``layout.names``.

    Example:
        >>> mon = monitor_network(net, daily, sigma, layout,
        ...                       ratios={"cs1": 1.2})     # doctest: +SKIP
        >>> mon.diagnose(window=15)                        # doctest: +SKIP
        model drift: 93% of the last 15 steps reject, blame concentration 40%
    """
    residual_fn, kwargs = _posed(network, layout, ratios, cv_drops_bar, kwargs)
    return monitor(residual_fn, measurements, sigma, **kwargs)


def reconcile_network_multi(
    network: GasNetwork,
    measurements: Sequence[Array],
    sigma: Array,
    layout: GasStateLayout,
    *,
    shared: Sequence[str | int],
    ratios: dict[str, float] | None = None,
    cv_drops_bar: dict[str, float] | None = None,
    **kwargs: Any,
) -> MultiReconcileResult:
    """Pool several periods that share a parameter into one solve.

    Each period gets its own pressures and flows; the variables in
    ``shared`` --- a fouling factor believed constant over the window,
    an unmetered offtake --- appear once and are estimated from every
    period at the same time. That is not the same as reconciling each
    period separately and averaging: the standard error it reports is
    that of the pooled estimate, and the degrees of redundancy count
    the parameter once instead of once per period.

    The parameter must be in the layout to be shared, which means
    ``layout`` carries it (``efficiency_arcs``, ``ratio_arcs``,
    ``cv_arcs``) while ``network`` does not --- the estimate is what
    the network is missing. :meth:`GasStateLayout.embed
    <difflow_gas.residuals.GasStateLayout.embed>` re-packs measurements
    taken against a plainer layout.

    Args:
        network: the network the measurements belong to.
        measurements: one measurement vector per period, each packed by
            ``layout``.
        sigma: standard deviations; ``inf`` = unmeasured. A finite
            sigma on a shared variable is a prior, applied once.
        layout: the state layout, carrying the shared parameters.
        shared: names or indices of the variables held common.
        ratios: compressor ratios not carried in the state.
        cv_drops_bar: control valve drops not carried in the state.
        **kwargs: forwarded to
            :func:`difflow.reconciliation.reconcile_multi`.

    Returns:
        A :class:`~difflow.reconciliation.MultiReconcileResult`.

    Example:
        >>> layout = gas_state_layout(net, efficiency_arcs=["p3"])
        >>> res = reconcile_network_multi(          # doctest: +SKIP
        ...     net, window, sigma, layout, shared=["eta_p3"],
        ...     ratios={"cs1": 1.2},
        ... )
        >>> res.shared["eta_p3"], res.shared_std["eta_p3"]
    """
    residual_fn, kwargs = _posed(network, layout, ratios, cv_drops_bar, kwargs)
    return reconcile_multi(
        residual_fn, measurements, sigma, shared=shared, **kwargs
    )


def reconciled_values(
    result: ReconcileResult, layout: GasStateLayout
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Split a reconciled state into ``(p_bar, q_kg_s, supply_kg_s)``.

    The dicts are plain floats, ready for
    :func:`difflow_gas.verify.residuals_from_values` --- which is how a
    reconciliation gets checked by the same verifier as a sequential
    solve.
    """
    x = result.x
    p = {n: float(x[layout.index(f"p_{n}")]) for n in layout.nodes}
    q = {a: float(x[layout.index(f"q_{a}")]) for a in layout.arcs}
    s = {n: float(x[layout.index(f"s_{n}")]) for n in layout.supply_nodes}
    return p, q, s
