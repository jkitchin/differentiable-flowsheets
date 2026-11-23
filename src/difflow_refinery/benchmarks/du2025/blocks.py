"""Generic planning blocks of the Du et al. (2025) benchmark, as pure JAX.

Each block is the paper's equation solved forward: flows (t, mass) and
properties in, outlet flows and properties out. They are what a later
nonlinear unit replaces (#531, #533-#537), and what
:mod:`~difflow_refinery.benchmarks.du2025.model` writes as residuals; at the
published solution both agree.

Property rules: specific gravity blends by volume (``V = F / SPG``); a
weight-based property blends by mass (``sum F q / sum F``); a volume-based
one by volume. Case 1 has no volume-based property other than SPG.

All functions take and return ``jnp`` arrays and are safe under ``jax.grad``
at a zero total flow (the property of an empty mix is 0, with a finite
gradient).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

_TINY = 1e-30


def _safe_div(num: Array, den: Array) -> Array:
    """``num / den`` with 0 for ``den == 0`` and a finite gradient there."""
    ok = jnp.abs(den) > _TINY
    return jnp.where(ok, num / jnp.where(ok, den, 1.0), 0.0)


def fixed_yield(feeds: Array, gamma_in: Array, gamma_out: Array) -> Array:
    """Eq. 8: outlet flows of a fixed-yield unit.

    ``F_out = gamma_out * sum(F_in) / sum(gamma_in)``.

    Args:
        feeds: ``(n_in,)`` inlet mass flows of one batch.
        gamma_in: ``(n_in,)`` base yield coefficients of the inlets.
        gamma_out: ``(n_out,)`` base yield coefficients of the outlets.

    Returns:
        ``(n_out,)`` outlet mass flows.

    Example:
        >>> fixed_yield(jnp.array([2.0]), jnp.array([1.0]), jnp.array([0.7, 0.3]))
        Array([1.4, 0.6], dtype=float64)
    """
    return gamma_out * jnp.sum(feeds) / jnp.sum(gamma_in)


def delta_base_coefficients(gamma: Array, props: Array, base: Array,
                            delta: Array, step: Array) -> Array:
    """Eq. 9: yield coefficients shifted by the feed properties.

    ``Gamma = gamma + sum_q (FQ_q - B_q) * delta[:, q] / Del_q``.

    Args:
        gamma: ``(n,)`` base yield coefficients (inlets and outlets).
        props: ``(n_q,)`` feed properties ``FQ`` that move the yields.
        base: ``(n_q,)`` base feed properties ``B``.
        delta: ``(n, n_q)`` yield change per step of each property.
        step: ``(n_q,)`` property steps ``Del``.

    Returns:
        ``(n,)`` adjusted coefficients ``Gamma``.
    """
    return gamma + delta @ ((props - base) / step)


def delta_base(feeds: Array, gamma_in: Array, gamma_out: Array, props: Array,
               base: Array, delta_in: Array, delta_out: Array,
               step: Array) -> tuple[Array, Array]:
    """Eqs. 9-10: outlet flows of a delta-base unit.

    Args:
        feeds: ``(n_in,)`` inlet mass flows.
        gamma_in, gamma_out: Base coefficients of inlets and outlets.
        props: ``(n_q,)`` feed properties named by ``DBSQ``.
        base: ``(n_q,)`` base properties ``B``.
        delta_in, delta_out: ``(n_in, n_q)``, ``(n_out, n_q)`` delta vectors.
        step: ``(n_q,)`` steps ``Del``.

    Returns:
        ``(outlet flows, Gamma_out)``.

    Example:
        >>> out, g = delta_base(jnp.array([1.0]), jnp.array([1.0]),
        ...                     jnp.array([0.6, 0.4]), jnp.array([0.9]),
        ...                     jnp.array([0.9]), jnp.zeros((1, 1)),
        ...                     jnp.array([[0.1], [-0.1]]), jnp.array([1.0]))
        >>> out
        Array([0.6, 0.4], dtype=float64)
    """
    g_in = delta_base_coefficients(gamma_in, props, base, delta_in, step)
    g_out = delta_base_coefficients(gamma_out, props, base, delta_out, step)
    return g_out * jnp.sum(feeds) / jnp.sum(g_in), g_out


def blend(flows: Array, spg: Array, props: Array | None = None) -> tuple[Array, Array, Array]:
    """Eqs. 12-13 (mixer) and 18 (blender): pool a set of streams.

    Args:
        flows: ``(n,)`` mass flows.
        spg: ``(n,)`` specific gravities.
        props: ``(n, n_q)`` weight-based properties, or ``None``.

    Returns:
        ``(total mass, SPG by volume, weight-based properties by mass)``;
        the last is ``(n_q,)`` (empty when ``props`` is ``None``).

    Example:
        >>> F, s, q = blend(jnp.array([1.0, 1.0]), jnp.array([0.8, 1.0]),
        ...                 jnp.array([[1.0], [3.0]]))
        >>> float(q[0])
        2.0
    """
    total = jnp.sum(flows)
    volume = jnp.sum(_safe_div(flows, spg))
    spg_mix = _safe_div(total, volume)
    if props is None:
        return total, spg_mix, jnp.zeros((0,))
    q = _safe_div(flows @ props, total)
    return total, spg_mix, q


def split(flow: Array, fractions: Array, props: Array) -> tuple[Array, Array]:
    """Eqs. 3, 17: a splitter; every outlet carries the inlet's properties.

    Args:
        flow: Inlet mass flow.
        fractions: ``(n_out,)`` split fractions (sum to 1).
        props: ``(n_q,)`` inlet properties.

    Returns:
        ``(outlet flows (n_out,), outlet properties (n_out, n_q))``.
    """
    return flow * fractions, jnp.broadcast_to(props, (fractions.shape[0],) + props.shape)


def transfer(props_in: Array, alpha: Array) -> Array:
    """Eq. 11: ``FQ_out = alpha * FQ_in`` for the transferred properties."""
    return alpha * props_in


def blend_margins(flows: Array, spg: Array, props: Array, q_min: Array,
                  q_max: Array) -> tuple[Array, Array]:
    """Eq. 19c, signed: ``(sum F q - qmin sum F, qmax sum F - sum F q)``.

    Mass-weighted margins (nonnegative when the spec holds), the form that
    stays defined at an empty blend. ``spg`` is unused for weight-based
    properties and kept for a uniform signature.
    """
    del spg
    total = jnp.sum(flows)
    fq = flows @ props
    return fq - q_min * total, q_max * total - fq


def swing_cut(crude_flows: Array, y: Array, merge: Array) -> Array:
    """Eq. 4: CDU product flows from cut yields and swing-cut merge ratios.

    Args:
        crude_flows: ``(n_crude,)`` crude mass flows of one batch.
        y: ``(n_cut, n_crude)`` cut yields (heart and swing cuts).
        merge: ``(n_prod, n_cut)`` share of each cut that ends in each
            product: 1 for a product's own heart cut, ``phi`` for a swing cut
            merged into it, and ``1 - sum(phi)`` for a swing cut's own
            outlet.

    Returns:
        ``(n_prod,)`` product mass flows.
    """
    return merge @ (y @ crude_flows)


def cut_properties(crude_flows: Array, y: Array, merge: Array, spg_cut: Array,
                   props_cut: Array) -> tuple[Array, Array]:
    """Eq. 5: CDU product SPG (by volume) and weight-based properties.

    Args:
        crude_flows, y, merge: As :func:`swing_cut`.
        spg_cut: ``(n_cut, n_crude)`` SPG of each cut from each crude
            (``FQcut[.., SPG]``).
        props_cut: ``(n_cut, n_crude, n_q)`` weight-based cut properties.

    Returns:
        ``(spg (n_prod,), props (n_prod, n_q))``.
    """
    mass = y * crude_flows                       # (n_cut, n_crude)
    prod_mass = merge @ jnp.sum(mass, axis=1)
    prod_vol = merge @ jnp.sum(_safe_div(mass, spg_cut), axis=1)
    prod_q = merge @ jnp.einsum("kc,kcq->kq", mass, props_cut)
    return _safe_div(prod_mass, prod_vol), _safe_div(prod_q, prod_mass[:, None])
