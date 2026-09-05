"""Thin convenience layer over ``pounce`` for difflow flowsheets.

Everything here exists to make one guarantee: **pounce is never allowed to
discover sparsity by probing.**

``pounce.jax.from_jax`` and ``pounce.jax.JaxProblem`` detect the Jacobian and
Lagrangian-Hessian structure by evaluating derivatives at random
:math:`\\mathcal{N}(0, 1)` points unless a pattern is supplied. Those points
have nothing to do with ``x0`` or the bounds: a difflow model gets asked for
its derivatives at ``T ~ -1.3 K`` and ``P ~ 0.4 Pa``, where the Arrhenius
terms overflow and the reactor linear solve is singular. It is not a
tuning problem, it is a category error, and every difflow model hits it.

So :func:`solve_with_pounce` always passes ``jac_pattern`` and
``hess_pattern``, taken from :class:`~difflow.solvers.nlp.Bounds`, and
raises if neither those nor an explicit override supply them. There is no
code path in this module that reaches pounce with a pattern unset, and none
that substitutes a dense one on your behalf: a dense Hessian pattern costs
``n`` colors per evaluation, which on a real flowsheet is the difference
between a solve and a stall. If that is what you want, ask for it --
``as_nlp(..., sparsity="dense")``.

The other half of the contract is the caller's: a supplied pattern must be a
**superset** of the true structure. pounce does not check, and a missing
entry is silently wrong -- dropped on the dense path, and aliased into a
same-colored neighbour under ``sparse=True``. ``as_nlp`` derives its
patterns from the computation graph (or, failing that, from the flowsheet
topology), so they are supersets everywhere, and verifies them at ``x0``;
see :mod:`difflow.solvers.sparsity`.

Note on ``pounce.jax.solve``
----------------------------
``pounce.jax.solve(p, f=..., g=...)`` -- the ``custom_vjp`` wrapper that
makes a solve differentiable with respect to ``p`` -- builds its internal
problem *without* a pattern argument and therefore always probes. It cannot
be used on a difflow model as it stands. The differentiable path that does
work is ``pounce.jax.JaxProblem``, which accepts ``jac_pattern`` /
``hess_pattern`` and exposes the same implicit-function backward through the
KKT system. :func:`differentiable_problem` builds one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.solvers._lazy import require
from difflow.solvers.nlp import Bounds, as_nlp
from difflow.solvers.sparsity import SparsityDetectionError
from difflow.streams import Stream

__all__ = [
    "FlowsheetOptimum",
    "solve_with_pounce",
    "optimize_flowsheet",
    "differentiable_problem",
    "bound_sensitivities",
]


def _patterns(bounds: Bounds, jac_pattern, hess_pattern):
    """Resolve the patterns, guaranteeing neither is ``None``.

    Raises rather than substituting a dense pattern. A dense fallback here
    would be silent and would cost ``n`` colors per Hessian evaluation, which
    is exactly the outcome this module exists to prevent; ``as_nlp`` always
    fills both fields in, so reaching this means a hand-built
    :class:`~difflow.solvers.nlp.Bounds` left one out.
    """
    jac = jac_pattern if jac_pattern is not None else bounds.jac_pattern
    hess = hess_pattern if hess_pattern is not None else bounds.hess_pattern
    missing = [nm for nm, v in (("jac_pattern", jac), ("hess_pattern", hess))
               if v is None]
    if missing:
        raise SparsityDetectionError(
            f"{' and '.join(missing)} is None, and pounce would then discover "
            "the structure by probing at random N(0, 1) points -- where this "
            "model is undefined. Build the problem with as_nlp(), or derive "
            "the patterns with difflow.solvers.detect_patterns(f, g, x0, m). "
            "difflow.solvers.dense_jacobian_pattern / dense_hessian_pattern "
            "are the deliberate way to go dense."
        )
    return jac, hess


def solve_with_pounce(
    f: Callable,
    g: Callable,
    bounds: Bounds,
    *,
    x0: Array | None = None,
    options: dict | None = None,
    sparse: bool = False,
    jac_pattern=None,
    hess_pattern=None,
) -> tuple[np.ndarray, dict]:
    """Solve an :func:`~difflow.solvers.nlp.as_nlp` problem with pounce.

    Args:
        f: Objective ``f(x)``.
        g: Constraint body ``g(x)``.
        bounds: The :class:`~difflow.solvers.nlp.Bounds` from ``as_nlp``.
        x0: Starting point; defaults to ``bounds.x0``.
        options: pounce options, e.g. ``{"tol": 1e-8, "print_level": 0}``.
            ``print_level`` defaults to 0.
        sparse: Use CPR-style colored AD for the per-iteration derivatives.
            Only worth it when the pattern is genuinely sparse -- and only
            safe when it is a true superset, because under compression a
            missing entry corrupts its whole color rather than merely
            vanishing.
        jac_pattern: Override for ``bounds.jac_pattern``.
        hess_pattern: Override for ``bounds.hess_pattern``.

    Returns:
        ``(x, info)`` from ``pounce.Problem.solve``. ``info["mult_g"]`` holds
        the constraint multipliers; see :func:`bound_sensitivities` for the
        sign convention.

    Raises:
        ImportError: If pounce is not installed (PyPI name ``pounce-solver``).

    Example:
        >>> f, g, bd = as_nlp(fs, decisions, specs, objective=profit)  # doctest: +SKIP
        >>> x, info = solve_with_pounce(f, g, bd, options={"tol": 1e-9})  # doctest: +SKIP
    """
    pj = require("pounce.jax")
    jac, hess = _patterns(bounds, jac_pattern, hess_pattern)
    problem = pj.from_jax(
        f,
        g,
        n=bounds.n,
        m=bounds.m,
        lb=np.asarray(bounds.lb),
        ub=np.asarray(bounds.ub),
        cl=np.asarray(bounds.cl),
        cu=np.asarray(bounds.cu),
        sparse=sparse,
        jac_pattern=jac,
        hess_pattern=hess,
    )
    opts = {"print_level": 0}
    opts.update(options or {})
    for key, value in opts.items():
        problem.add_option(key, value)
    start = np.asarray(bounds.x0 if x0 is None else x0, dtype=float)
    return problem.solve(x0=start)


def differentiable_problem(
    f: Callable,
    g: Callable,
    bounds: Bounds,
    *,
    options: dict | None = None,
    sparse: bool = False,
    jac_pattern=None,
    hess_pattern=None,
    **kwargs: Any,
):
    """Build a ``pounce.jax.JaxProblem``: build once, solve many, differentiable.

    ``jp.solve(p, x0)`` returns ``x*`` and is differentiable with respect to
    ``p`` (the parameter vector declared via ``as_nlp(parameters=...)``) by
    the implicit-function rule on the KKT system, so a whole design problem
    becomes one node in an outer JAX computation.

    This is the differentiable entry point rather than ``pounce.jax.solve``
    because ``solve`` has no pattern arguments and would probe.

    Args:
        f: Objective ``f(x, p)``.
        g: Constraint body ``g(x, p)``.
        bounds: From ``as_nlp``; supplies the boxes, ``p0`` and the patterns.
        options: pounce options applied once at build time.
        sparse: Colored AD for the forward derivatives.
        jac_pattern: Override for ``bounds.jac_pattern``.
        hess_pattern: Override for ``bounds.hess_pattern``.
        **kwargs: Passed through to ``JaxProblem`` (``factor_reuse``, ...).

    Returns:
        A ``pounce.jax.JaxProblem``.

    Raises:
        ImportError: If pounce is not installed.

    Example:
        >>> jp = differentiable_problem(f, g, bd)              # doctest: +SKIP
        >>> loss = lambda p: f(jp.solve(p, bd.x0), p)          # doctest: +SKIP
        >>> jax.grad(loss)(bd.p0)                              # doctest: +SKIP
    """
    pj = require("pounce.jax")
    jac, hess = _patterns(bounds, jac_pattern, hess_pattern)
    opts = {"print_level": 0}
    opts.update(options or {})
    return pj.JaxProblem(
        f=f,
        g=g,
        n=bounds.n,
        m=bounds.m,
        p_example=np.asarray(bounds.p0),
        lb=np.asarray(bounds.lb),
        ub=np.asarray(bounds.ub),
        cl=np.asarray(bounds.cl),
        cu=np.asarray(bounds.cu),
        options=opts,
        sparse=sparse,
        jac_pattern=jac,
        hess_pattern=hess,
        **kwargs,
    )


def bound_sensitivities(info: dict, bounds: Bounds) -> dict[str, float]:
    """``d(objective) / d(constraint bound)`` for every constraint.

    pounce forms its Lagrangian as ``L = sigma f + lambda^T g``, so the value
    function of the perturbed problem has

    .. math:: \\frac{d f^*}{d b} = -\\lambda

    where ``b`` is whichever of ``cl`` / ``cu`` is active. The sign matters:
    ``info["mult_g"]`` is ``lambda``, and the sensitivity a process engineer
    wants -- what one more unit of conversion costs -- is its negative. On an
    inactive constraint the multiplier is zero and so is the sensitivity.

    **The sign was measured, not assumed.** On the CSTR design problem
    (minimize reactor volume subject to a product-flow spec) a central
    difference of the converged objective with respect to the bound agrees with
    ``-mult_g`` to seven digits, and it does so in *both* directions: for
    ``F_B >= 8`` the FD is ``+0.72494`` against ``mult_g = -0.72494``, and for
    ``F_A <= 2`` the FD is ``-0.72494`` against ``mult_g = +0.72494``. The
    convention is uniform, so there is no per-constraint sign bookkeeping to
    do. ``tests/test_solvers_bridge.py`` pins both directions.

    Args:
        info: The ``info`` dict from a pounce solve.
        bounds: The problem's :class:`~difflow.solvers.nlp.Bounds`, for names.

    Returns:
        ``{constraint name: d objective / d bound}``.

    Example:
        >>> x, info = solve_with_pounce(f, g, bd)      # doctest: +SKIP
        >>> bound_sensitivities(info, bd)["conv >= 0.97"]  # doctest: +SKIP
        -3.41
    """
    mult = np.asarray(info["mult_g"], dtype=float)
    return {nm: float(-mult[i]) for i, nm in enumerate(bounds.con_names)}


@dataclass
class FlowsheetOptimum:
    """Result of :func:`optimize_flowsheet`.

    Attributes:
        x: Optimal variable vector.
        info: Raw pounce info dict.
        decisions: Optimal decision values by name.
        streams: Solved streams, feeds included.
        objective: Objective value at the optimum.
        bounds: The :class:`~difflow.solvers.nlp.Bounds` used.
        success: True if pounce reported ``Solve_Succeeded``.
    """

    x: np.ndarray
    info: dict
    decisions: dict[str, float]
    streams: dict[str, Stream]
    objective: float
    bounds: Bounds
    success: bool

    def sensitivities(self) -> dict[str, float]:
        """``d(objective)/d(bound)`` per constraint. See :func:`bound_sensitivities`."""
        return bound_sensitivities(self.info, self.bounds)


def optimize_flowsheet(
    flowsheet,
    decisions: Sequence,
    specs: Sequence = (),
    *,
    objective: Callable | None = None,
    options: dict | None = None,
    sparse: bool = False,
    **nlp_kwargs: Any,
) -> FlowsheetOptimum:
    """``as_nlp`` + ``solve_with_pounce`` in one call.

    Args:
        flowsheet: See :func:`~difflow.solvers.nlp.as_nlp`.
        decisions: See :func:`~difflow.solvers.nlp.as_nlp`.
        specs: See :func:`~difflow.solvers.nlp.as_nlp`.
        objective: ``objective(streams, decisions) -> scalar``, *minimized*.
            Negate a profit.
        options: pounce options.
        sparse: Colored AD for the per-iteration derivatives.
        **nlp_kwargs: Forwarded to :func:`~difflow.solvers.nlp.as_nlp`.

    Returns:
        A :class:`FlowsheetOptimum`.

    Raises:
        ImportError: If pounce is not installed.

    Example:
        >>> res = optimize_flowsheet(          # doctest: +SKIP
        ...     fs,
        ...     [Decision("unit:reactor.params.V", 0.01, 5.0, 0.5)],
        ...     [("product.F_B", ">=", 8.0)],
        ...     objective=lambda s, d: d["unit:reactor.params.V"],
        ... )
        >>> res.decisions                       # doctest: +SKIP
        {'unit:reactor.params.V': 0.83...}
    """
    f, g, bounds = as_nlp(
        flowsheet, decisions, specs, objective=objective, **nlp_kwargs
    )
    x, info = solve_with_pounce(f, g, bounds, options=options, sparse=sparse)
    dvals, streams = bounds.unpack(jnp.asarray(x))
    return FlowsheetOptimum(
        x=np.asarray(x),
        info=info,
        decisions={k: float(v) for k, v in dvals.items()},
        streams=streams,
        objective=float(f(jnp.asarray(x))),
        bounds=bounds,
        success=info.get("status_msg") == "Solve_Succeeded",
    )
