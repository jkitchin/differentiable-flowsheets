"""The user-facing reconciliation entry point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.params_mixin import ParamsMixin

from difflow.reconciliation.core import (
    Scaling,
    auto_scaling,
    identity_scaling,
    measured_mask,
    reconciled_covariance,
    solve_reconciliation,
)
from difflow.reconciliation.structure import StructureReport, classify


@dataclass
class ReconcileResult(ParamsMixin):
    """Outcome of a data reconciliation.

    Attributes:
        x: reconciled state, shape ``(n,)``.
        x_named: the same, as ``{name: value}``.
        adjustment: ``x - y``, ``nan`` where unmeasured.
        multipliers: Lagrange multipliers of the constraints.
        objective: the weighted sum of squared adjustments. This is
            also the global-test statistic; see
            :func:`difflow.reconciliation.global_test`.
        residual: ``F(x)`` at the solution.
        residual_norm: its infinity norm.
        covariance: covariance of the reconciled estimates.
        std: their standard deviations, as ``{name: value}``.
        converged: whether the constraints are satisfied to ``tol``.
        names: variable names.
        sigma: the standard deviations used.
        structure: the observability/redundancy report.
        scaling: the scaling actually used.
    """

    x: Array
    x_named: dict[str, float]
    adjustment: Array
    multipliers: Array
    objective: float
    residual: Array
    residual_norm: float
    covariance: Array
    std: dict[str, float]
    converged: bool
    names: list[str]
    sigma: Array
    structure: StructureReport
    scaling: Scaling
    n_steps: int = 0
    y: Array | None = None

    def std_of(self, name: str) -> float:
        """Standard deviation of one reconciled estimate."""
        return self.std[name]

    def summary(self) -> str:
        """Table of measurements, reconciled values and adjustments."""
        sig = np.asarray(self.sigma, dtype=float)
        y = np.asarray(self.y, dtype=float) if self.y is not None else None
        x = np.asarray(self.x, dtype=float)
        lines = [
            f"objective {self.objective:.4g} on "
            f"{self.structure.degree_of_redundancy} degrees of redundancy, "
            f"|F| = {self.residual_norm:.3g}, converged = {self.converged}",
            "",
            f"{'variable':<20} {'measured':>12} {'reconciled':>12} "
            f"{'adjust':>10} {'sigma':>9} {'sd_hat':>9}",
            "-" * 76,
        ]
        for i, nm in enumerate(self.names):
            meas = "-" if y is None or not np.isfinite(sig[i]) else f"{y[i]:12.4f}"
            adj = (
                "-"
                if y is None or not np.isfinite(sig[i])
                else f"{x[i] - y[i]:10.4f}"
            )
            sg = "-" if not np.isfinite(sig[i]) else f"{sig[i]:9.4f}"
            lines.append(
                f"{nm:<20} {meas:>12} {x[i]:12.4f} {adj:>10} {sg:>9} "
                f"{self.std[nm]:9.4f}"
            )
        return "\n".join(lines)


def reconcile(
    residual_fn: Callable,
    y: Array,
    sigma: Array,
    *,
    params: Any = None,
    names: Sequence[str] | None = None,
    x0: Array | None = None,
    unmeasured_init: Array | float | None = None,
    unmeasured_scale: Array | float | None = None,
    scaling: Scaling | bool = True,
    max_steps: int = 20,
    tol: float = 1e-9,
    method: str = "gauss_newton",
    check_structure: bool = True,
    rank_tol: float | None = None,
) -> ReconcileResult:
    """Reconcile measurements against a model's constraint equations.

    Solves :math:`\\min (x-y)^T W (x-y)` subject to
    :math:`F(x, \\theta) = 0`, where entries of ``sigma`` that are
    infinite mark variables to be *estimated* rather than reconciled.
    A finite sigma on an unmetered variable acts as a prior, which is
    the graceful way to handle a weakly identified parameter.

    Args:
        residual_fn: ``F(x, params) -> (m,)``, JAX-traceable.
        y: measurements, shape ``(n,)``. Entries with infinite sigma
            are ignored and may be ``nan``.
        sigma: standard deviations, shape ``(n,)``; ``inf`` or ``nan``
            marks an unmeasured variable.
        params: extra argument threaded to ``residual_fn``; gradients
            with respect to it are exact, which answers a different
            question from estimating a variable (how the reconciled
            state would move if a *fixed* model parameter changed).
        names: variable names for reporting, defaulting to ``x0...``.
        x0: full initial state. Defaults to ``y`` on measured entries
            and ``unmeasured_init`` elsewhere.
        unmeasured_init: starting value for unmeasured entries
            (default 1.0), used only when ``x0`` is not given.
        unmeasured_scale: typical magnitude of unmeasured entries, for
            scaling; defaults to their initial magnitude.
        scaling: ``True`` for automatic scaling (recommended and the
            default), a :class:`Scaling` to force one, or ``False`` to
            solve in raw units.
        max_steps: Gauss-Newton iterations.
        tol: constraint tolerance deciding ``converged``.
        method: ``"gauss_newton"`` or ``"newton"``.
        check_structure: run the observability check first and raise
            :class:`~difflow.reconciliation.ReconciliationStructureError`
            rather than returning a NaN state. Turn it off only inside
            a ``jit``/``vmap`` sweep whose structure you have already
            validated.
        rank_tol: override the rank threshold of the structure check.

    Returns:
        A :class:`ReconcileResult`.

    Raises:
        ReconciliationStructureError: if unmeasured variables cannot be
            determined from the constraints.

    Example:
        >>> res = reconcile(F, y, sigma, names=layout.names)
        >>> res.objective, res.std_of("q_p3")
    """
    y = jnp.asarray(y, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    n = y.shape[0]
    if sigma.shape != y.shape:
        raise ValueError(
            f"sigma has shape {sigma.shape} but y has shape {y.shape}"
        )
    names = list(names) if names is not None else [f"x{i}" for i in range(n)]
    if len(names) != n:
        raise ValueError(f"got {len(names)} names for {n} variables")

    mask = measured_mask(sigma)
    if x0 is None:
        init = 1.0 if unmeasured_init is None else unmeasured_init
        init = jnp.broadcast_to(jnp.asarray(init, dtype=jnp.float64), y.shape)
        x0 = jnp.where(mask, jnp.where(mask, y, 0.0), init)
    x0 = jnp.asarray(x0, dtype=jnp.float64)

    m = int(residual_fn(x0, params).shape[0])
    if scaling is True:
        sc = auto_scaling(
            residual_fn, x0, sigma, params=params,
            unmeasured_scale=unmeasured_scale,
        )
    elif scaling is False:
        sc = identity_scaling(n, m)
    else:
        sc = scaling

    if check_structure:
        classify(
            residual_fn, x0, sigma, scaling=sc, params=params,
            names=names, rank_tol=rank_tol,
        ).raise_if_unsolvable()

    x_hat, multipliers = solve_reconciliation(
        residual_fn, y, sigma, x0=x0, scaling=sc, params=params,
        n_steps=max_steps, method=method,
    )

    residual = residual_fn(x_hat, params)
    residual_norm = float(jnp.max(jnp.abs(residual)))
    adjustment = jnp.where(mask, x_hat - jnp.where(mask, y, 0.0), jnp.nan)
    objective = float(
        jnp.sum(jnp.where(mask, (x_hat - jnp.where(mask, y, 0.0)) ** 2
                          / jnp.where(mask, sigma, 1.0) ** 2, 0.0))
    )

    covariance = reconciled_covariance(
        residual_fn, x_hat, sigma, scaling=sc, params=params
    )
    std_arr = jnp.sqrt(jnp.clip(jnp.diag(covariance), 0.0, jnp.inf))
    std = {nm: float(std_arr[i]) for i, nm in enumerate(names)}

    structure = classify(
        residual_fn, x_hat, sigma, scaling=sc, params=params,
        names=names, rank_tol=rank_tol, covariance=covariance,
    )

    return ReconcileResult(
        x=x_hat,
        x_named={nm: float(x_hat[i]) for i, nm in enumerate(names)},
        adjustment=adjustment,
        multipliers=multipliers,
        objective=objective,
        residual=residual,
        residual_norm=residual_norm,
        covariance=covariance,
        std=std,
        converged=bool(residual_norm < tol),
        names=names,
        sigma=sigma,
        structure=structure,
        scaling=sc,
        n_steps=max_steps,
        y=y,
    )
