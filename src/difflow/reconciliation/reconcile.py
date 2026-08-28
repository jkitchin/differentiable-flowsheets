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


@dataclass
class MultiReconcileResult(ParamsMixin):
    """Outcome of a reconciliation across several data sets.

    Attributes:
        stacked: the underlying single reconciliation of the stacked
            problem. Its ``names`` suffix each per-data-set variable
            with ``[k]``, while shared variables keep their bare name.
        shared_names: the variables held common across data sets.
        shared: their estimates, as ``{name: value}``.
        shared_std: their standard errors, as ``{name: value}``.
        states: one ``{name: value}`` per data set, over the full
            per-data-set variable list.
        n_datasets: how many data sets were combined.
        names: the per-data-set variable names.
    """

    stacked: ReconcileResult
    shared_names: list[str]
    shared: dict[str, float]
    shared_std: dict[str, float]
    states: list[dict[str, float]]
    n_datasets: int
    names: list[str]

    @property
    def objective(self) -> float:
        """The stacked objective, i.e. the global-test statistic."""
        return self.stacked.objective

    @property
    def structure(self) -> StructureReport:
        """Structure of the stacked problem.

        Together with :attr:`objective` this is all
        :func:`difflow.reconciliation.global_test` reads, so it accepts
        this result directly. The per-sensor
        :func:`~difflow.reconciliation.measurement_test` needs the
        stacked variable names and so takes :attr:`stacked`.
        """
        return self.stacked.structure

    @property
    def converged(self) -> bool:
        return self.stacked.converged

    def summary(self) -> str:
        """The shared estimates, with the pooling that produced them."""
        lines = [
            f"{self.n_datasets} data sets, "
            f"{self.structure.degree_of_redundancy} degrees of redundancy, "
            f"objective {self.objective:.4g}, converged = {self.converged}",
            "",
            f"{'shared parameter':<24} {'estimate':>12} {'std error':>12}",
            "-" * 50,
        ]
        for nm in self.shared_names:
            lines.append(
                f"{nm:<24} {self.shared[nm]:12.5f} {self.shared_std[nm]:12.5f}"
            )
        return "\n".join(lines)


def reconcile_multi(
    residual_fn: Callable,
    measurements: Sequence[Any],
    sigma: Any,
    *,
    shared: Sequence[str | int],
    names: Sequence[str] | None = None,
    params: Any = None,
    **kwargs: Any,
) -> MultiReconcileResult:
    """Reconcile several data sets that share a set of parameters.

    Each data set gets its own copy of the plant state, but the
    variables in ``shared`` --- a fouling factor, an activity, an
    unmetered offtake believed constant over the campaign --- appear
    once and are estimated from all of them at the same time. That is
    the statistically correct way to pool a campaign, and it differs
    from reconciling each data set separately and averaging the
    estimates in two ways that matter:

    * the standard error it reports is the standard error *of the
      pooled estimate*, roughly :math:`\\sqrt{K}` tighter than a single
      data set's, whereas averaging point estimates leaves you holding
      one data set's error bar for a quantity K of them informed;
    * every data set constrains the parameter through its own
      equations, so a parameter too weakly identified to be recovered
      from one data set can still be observable from several. The
      structure check runs on the stacked problem and reports this.

    Averaging is not merely looser --- it estimates the wrong thing if
    the parameter moves during the campaign, since the mean of the
    per-period estimates tracks the *mean* of the truth, not its
    current value. Pool over a window short enough that the parameter
    is genuinely constant across it.

    Args:
        residual_fn: ``F(x, params) -> (m,)``, JAX-traceable, for a
            single data set.
        measurements: one measurement vector per data set, each shape
            ``(n,)``.
        sigma: standard deviations, shape ``(n,)`` shared by every data
            set, or one ``(n,)`` array per data set. ``inf`` marks a
            variable to be estimated. On a shared variable a finite
            sigma is a prior, and it is applied *once*, taken (with the
            corresponding entry of ``measurements``) from the first
            data set --- K copies of one prior would count it K times.
        shared: names (needs ``names``) or indices of the variables
            held common across data sets.
        names: per-data-set variable names, defaulting to ``x0...``.
        params: extra argument threaded to ``residual_fn``: one value
            used for every data set, or a sequence of one per data set
            when they differ (a changed set point, a different
            compressor ratio).
        **kwargs: forwarded to :func:`reconcile`. An ``x0`` or
            ``unmeasured_scale`` given per data set, shape ``(n,)``, is
            expanded to the stacked problem automatically.

    Returns:
        A :class:`MultiReconcileResult`.

    Raises:
        ValueError: on a shape mismatch, an unknown shared name, or
            sigmas that disagree across data sets on a shared variable.
        ReconciliationStructureError: if the stacked problem cannot
            determine its unmeasured variables.

    Example:
        >>> res = reconcile_multi(                      # doctest: +SKIP
        ...     F, week_of_data, sigma,
        ...     shared=["eta_p3"], names=layout.names,
        ... )
        >>> res.shared["eta_p3"], res.shared_std["eta_p3"]
    """
    ys = [jnp.asarray(y, dtype=jnp.float64) for y in measurements]
    if not ys:
        raise ValueError("reconcile_multi needs at least one data set")
    k_sets = len(ys)
    n = ys[0].shape[0]
    for i, y in enumerate(ys):
        if y.shape != (n,):
            raise ValueError(
                f"data set {i} has shape {y.shape}, expected {(n,)}"
            )

    names = list(names) if names is not None else [f"x{i}" for i in range(n)]
    if len(names) != n:
        raise ValueError(f"got {len(names)} names for {n} variables")

    sigmas = _per_set_sigma(sigma, n, k_sets)
    shared_idx = _shared_indices(shared, names)
    private_idx = [i for i in range(n) if i not in set(shared_idx)]
    n_priv, n_sh = len(private_idx), len(shared_idx)

    for j in shared_idx:
        col = jnp.array([s[j] for s in sigmas])
        if not bool(jnp.all(_same_sigma(col, col[0]))):
            raise ValueError(
                f"shared variable {names[j]!r} has different sigmas across "
                "data sets; a shared variable carries one sigma, since its "
                "prior is applied once"
            )

    params_per_set = _per_set_params(params, k_sets)
    priv = jnp.asarray(private_idx, dtype=int)
    sh = jnp.asarray(shared_idx, dtype=int)

    def expand(z: Array, k: int) -> Array:
        """Rebuild data set ``k``'s full state from the stacked vector."""
        block = z[k * n_priv:(k + 1) * n_priv]
        x = jnp.zeros(n, dtype=z.dtype)
        return x.at[priv].set(block).at[sh].set(z[k_sets * n_priv:])

    def stacked_residual(z: Array, ps: Any) -> Array:
        return jnp.concatenate(
            [residual_fn(expand(z, k), ps[k]) for k in range(k_sets)]
        )

    def stack(per_set: Sequence[Array], shared_row: Array) -> Array:
        return jnp.concatenate(
            [y[priv] for y in per_set] + [shared_row]
        )

    y_stacked = stack(ys, ys[0][sh])
    sigma_stacked = stack(sigmas, sigmas[0][sh])
    names_stacked = [
        f"{names[i]}[{k}]" for k in range(k_sets) for i in private_idx
    ] + [names[i] for i in shared_idx]

    for key in ("x0", "unmeasured_scale"):
        value = kwargs.get(key)
        if value is None:
            continue
        arr = jnp.asarray(value, dtype=jnp.float64)
        if arr.shape == (n,):
            kwargs[key] = stack([arr] * k_sets, arr[sh])

    stacked = reconcile(
        stacked_residual, y_stacked, sigma_stacked,
        params=tuple(params_per_set), names=names_stacked, **kwargs,
    )

    x = np.asarray(stacked.x, dtype=float)
    std = np.sqrt(
        np.clip(np.diag(np.asarray(stacked.covariance, dtype=float)), 0.0, None)
    )
    shared_names = [names[i] for i in shared_idx]
    offset = k_sets * n_priv

    states: list[dict[str, float]] = []
    for k in range(k_sets):
        state = {
            names[i]: float(x[k * n_priv + j])
            for j, i in enumerate(private_idx)
        }
        state.update(
            {nm: float(x[offset + j]) for j, nm in enumerate(shared_names)}
        )
        states.append({nm: state[nm] for nm in names})

    return MultiReconcileResult(
        stacked=stacked,
        shared_names=shared_names,
        shared={nm: float(x[offset + j]) for j, nm in enumerate(shared_names)},
        shared_std={
            nm: float(std[offset + j]) for j, nm in enumerate(shared_names)
        },
        states=states,
        n_datasets=k_sets,
        names=names,
    )


def _same_sigma(a: Array, b: Array) -> Array:
    """Elementwise equality that counts two infinities as equal."""
    return (a == b) | (jnp.isinf(a) & jnp.isinf(b)) | (
        jnp.isnan(a) & jnp.isnan(b)
    )


def _per_set_sigma(sigma: Any, n: int, k_sets: int) -> list[Array]:
    """Normalize ``sigma`` to one ``(n,)`` array per data set."""
    arr = jnp.asarray(sigma, dtype=jnp.float64)
    if arr.shape == (n,):
        return [arr] * k_sets
    if arr.shape == (k_sets, n):
        return [arr[k] for k in range(k_sets)]
    raise ValueError(
        f"sigma has shape {arr.shape}; expected {(n,)} or {(k_sets, n)}"
    )


def _per_set_params(params: Any, k_sets: int) -> list[Any]:
    """One params object per data set, broadcasting a single one."""
    if isinstance(params, (list, tuple)) and len(params) == k_sets:
        return list(params)
    return [params] * k_sets


def _shared_indices(
    shared: Sequence[str | int], names: Sequence[str]
) -> list[int]:
    """Resolve shared variable names or indices, in packed order."""
    idx: list[int] = []
    for item in shared:
        if isinstance(item, str):
            try:
                idx.append(list(names).index(item))
            except ValueError:
                raise KeyError(
                    f"{item!r} is not a variable of this problem; "
                    f"expected one of {list(names)}"
                ) from None
        else:
            i = int(item)
            if not 0 <= i < len(names):
                raise ValueError(
                    f"shared index {i} is out of range for "
                    f"{len(names)} variables"
                )
            idx.append(i)
    if len(set(idx)) != len(idx):
        raise ValueError("shared contains a duplicate variable")
    return sorted(idx)
