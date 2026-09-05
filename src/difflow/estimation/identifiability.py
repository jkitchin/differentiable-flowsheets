"""Structural identifiability: can these parameters be told apart at all?

This module answers the question that comes *before* fitting and before
experiment design. Both the estimator and the Fisher-information design
criteria are built on the sensitivity matrix

.. math::

    S_{kj} = \\frac{\\partial y_k}{\\partial \\theta_j}

evaluated at the current parameter estimate. If :math:`S` is rank
deficient -- if some direction :math:`v` in parameter space satisfies
:math:`S v = 0` -- then moving the parameters along :math:`v` changes no
prediction, the Fisher information :math:`S^T \\Sigma^{-1} S` is singular,
the covariance is infinite along :math:`v`, and every design criterion is
degenerate. The two classic cases are parameters that enter only as a
product (:math:`k = A B`) or only as a sum.

**No amount of data fixes this.** More experiments add rows to :math:`S`,
but every row is orthogonal to :math:`v` by construction, so the null
space survives. The fixes are structural: measure something else,
reparameterize on the identifiable combination, or fix one parameter.
That is why :func:`check_identifiability` runs first in every workflow in
this package, and why :func:`difflow.estimation.design.design_experiments`
and :func:`difflow.estimation.design.predicted_covariance` refuse to run
until it passes.

The rank machinery is the observability analysis of
:mod:`difflow.reconciliation.structure`, reused as-is: the two questions
are the same linear-algebra question asked of different Jacobians
(constraints vs. parameters). Following that module, ranks are discrete
and not differentiable, so the SVD is done in NumPy while the Jacobian
itself comes from JAX.

Example:
    >>> import jax.numpy as jnp
    >>> from difflow.estimation import Experiment, check_identifiability
    >>> def model(theta, exp):            # A and B only ever appear as A*B
    ...     return {'y': theta['A'] * theta['B'] * exp.inputs['x']}
    >>> exps = [Experiment.candidate({'x': x}, ['y']) for x in (1.0, 2.0)]
    >>> report = check_identifiability(model, {'A': 2.0, 'B': 3.0}, exps)
    >>> report.identifiable
    False
    >>> sorted(report.unidentifiable)
    ['A', 'B']
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin

# Reused rather than reimplemented: the SVD-based numerical rank used for
# observability analysis in data reconciliation is exactly the test needed
# here, including its "SVD, never eigenvalues of A^T A" reasoning.
from difflow.reconciliation.structure import _rank_and_spectrum

__all__ = [
    "IdentifiabilityError",
    "IdentifiabilityReport",
    "check_identifiability",
    "sensitivity_matrix",
]


class IdentifiabilityError(ValueError):
    """The parameters cannot be estimated separately from these measurements.

    Raised when the sensitivity matrix is rank deficient. The message names
    the parameters implicated in the null space. The remedy is a structural
    change -- an added measurement, a reparameterization, or fixing a
    parameter -- not more data.
    """


@dataclass
class IdentifiabilityReport(ParamsMixin):
    """Structural identifiability of a parameter set from a measurement set.

    Attributes:
        identifiable: True when the sensitivity matrix has full column rank.
        rank: Numerical rank of the (scaled) sensitivity matrix.
        n_params: Number of parameters.
        n_obs: Number of scalar measurements contributing rows.
        singular_values: Full spectrum of the scaled sensitivity matrix, so
            a marginal case can be inspected instead of guessed at.
        rank_tol: Singular-value threshold used for the rank decision.
        rank_gap: Ratio of the smallest retained to the largest discarded
            singular value; small means the answer depends on ``rank_tol``.
        condition_number: ``s_max / s_min``; large means practically (even
            if not structurally) unidentifiable.
        unidentifiable: Parameters implicated in a null-space direction.
        null_space: Columns spanning the null space in scaled coordinates,
            shape ``(n_params, n_params - rank)``. Each column is a
            combination of parameters that no measurement responds to.
        combinations: Human-readable rendering of ``null_space`` columns.
        param_names: Parameter names, in column order.
        scale: The column scaling applied before the rank test.
        reason: Empty when identifiable, else a short diagnosis.
    """

    identifiable: bool
    rank: int
    n_params: int
    n_obs: int
    singular_values: np.ndarray
    rank_tol: float
    rank_gap: float
    condition_number: float
    unidentifiable: list[str] = field(default_factory=list)
    null_space: np.ndarray | None = None
    combinations: list[str] = field(default_factory=list)
    param_names: list[str] = field(default_factory=list)
    scale: np.ndarray | None = None
    reason: str = ""

    def raise_if_unidentifiable(self) -> None:
        """Raise :class:`IdentifiabilityError` when the rank test fails."""
        if self.identifiable:
            return
        detail = ""
        if self.unidentifiable:
            detail = f" Implicated parameters: {', '.join(self.unidentifiable)}."
        combos = ""
        if self.combinations:
            combos = " Unresolved direction(s): " + "; ".join(self.combinations) + "."
        raise IdentifiabilityError(
            f"parameters are not structurally identifiable: {self.reason}."
            f"{detail}{combos} rank(S)={self.rank} < {self.n_params} parameters "
            f"from {self.n_obs} measurement(s). More experiments cannot fix "
            "this: add a measurement that responds to the direction above, "
            "reparameterize on the identifiable combination, or fix one of "
            "the parameters."
        )

    def summary(self) -> str:
        """Human-readable summary of the rank test."""
        lines = [
            f"identifiable      : {self.identifiable}"
            + (f"  ({self.reason})" if self.reason else ""),
            f"rank / parameters : {self.rank} / {self.n_params}",
            f"measurements      : {self.n_obs}",
            f"condition number  : {self.condition_number:.4g}",
            f"rank gap          : {self.rank_gap:.4g}",
            f"rank tolerance    : {self.rank_tol:.4g}",
            "",
            "singular values   : "
            + ", ".join(f"{s:.4g}" for s in self.singular_values),
        ]
        if self.unidentifiable:
            lines += ["", "unidentifiable    : " + ", ".join(self.unidentifiable)]
        for combo in self.combinations:
            lines.append(f"  null direction  : {combo}")
        return "\n".join(lines)


def _as_theta_array(
    theta: dict[str, float] | Sequence[float] | Array,
    param_names: Sequence[str] | None,
) -> tuple[Array, list[str]]:
    """Normalize ``theta`` to an array plus an ordered list of names.

    Args:
        theta: Parameter dict, or an array with ``param_names`` given.
        param_names: Names fixing the column order. Required when ``theta``
            is not a dict; defaults to the dict insertion order otherwise.

    Returns:
        ``(theta_array, param_names)``.
    """
    if isinstance(theta, dict):
        names = list(param_names) if param_names is not None else list(theta)
        missing = [n for n in names if n not in theta]
        if missing:
            raise ValueError(f"theta is missing parameter(s): {missing}")
        return jnp.array([theta[n] for n in names], dtype=float), names
    arr = jnp.asarray(theta, dtype=float)
    if param_names is None:
        raise ValueError("param_names is required when theta is not a dict")
    names = list(param_names)
    if arr.shape != (len(names),):
        raise ValueError(
            f"theta has shape {arr.shape} but {len(names)} parameter names"
        )
    return arr, names


def _predictions(model_fn, theta_arr, exp, names, out_names) -> Array:
    """Model predictions for one experiment, flattened over ``out_names``."""
    theta_dict = {n: theta_arr[i] for i, n in enumerate(names)}
    preds = model_fn(theta_dict, exp)
    missing = [k for k in out_names if k not in preds]
    if missing:
        raise KeyError(
            f"model_fn did not return output(s) {missing} for experiment "
            f"{exp.name or '(unnamed)'}; it returned {sorted(preds)}"
        )
    return jnp.concatenate(
        [jnp.atleast_1d(jnp.asarray(preds[k], dtype=float).ravel()) for k in out_names]
    )


def _row_sigma(model_fn, theta_arr, exp, names, out_names) -> Array:
    """One sigma per *row* of an experiment's Jacobian block.

    An output may be vector valued -- a composition vector, a time series --
    in which case its single declared uncertainty applies to each of its
    entries. The per-output lengths come from :func:`jax.eval_shape`, which
    traces the model abstractly and so costs nothing: guessing them from
    ``n_rows // n_outputs`` silently misaligns the sigmas as soon as the
    outputs have *different* lengths.
    """
    sigma = exp.sigma_array
    spec = jax.eval_shape(
        lambda t: model_fn({n: t[i] for i, n in enumerate(names)}, exp), theta_arr
    )
    sizes = []
    for k in out_names:
        shape = jnp.shape(spec[k])
        sizes.append(int(np.prod(shape)) if shape else 1)
    return jnp.concatenate(
        [jnp.full(size, sigma[i]) for i, size in enumerate(sizes)]
    )


def sensitivity_matrix(
    model_fn: Callable,
    theta: dict[str, float] | Sequence[float] | Array,
    experiments: Sequence[Any],
    param_names: Sequence[str] | None = None,
    *,
    weighted: bool = True,
    scale: str | Sequence[float] | Array | None = None,
) -> Array:
    """Sensitivity of every measurement to every parameter.

    Row ``k`` is one scalar measurement (an experiment/output pair, in the
    order the experiments and their ``measured_names`` are given); column
    ``j`` is a parameter. One ``jax.jacobian`` call per experiment -- exact,
    no finite differences.

    Args:
        model_fn: ``model_fn(theta_dict, experiment) -> dict[str, value]``.
        theta: Parameter values (dict, or array with ``param_names``).
        experiments: Experiments or candidates. Rows come from each
            experiment's ``measured_names``.
        param_names: Column order; defaults to the ``theta`` dict order.
        weighted: Divide each row by its 1-sigma uncertainty, so the matrix
            is in units of standard deviations and ``S.T @ S`` is the Fisher
            information. Uncertainties default to 1.0 when not given.
        scale: Column scaling applied *after* weighting. ``None`` for none,
            ``'theta'`` for ``|theta_j|`` (relative sensitivity, which makes
            a rank test independent of parameter units), or an explicit
            vector of per-parameter scales.

    Returns:
        Array of shape ``(n_obs, n_params)``.

    Example:
        >>> from difflow.estimation import Experiment, sensitivity_matrix
        >>> def model(theta, exp):
        ...     return {'y': theta['a'] * exp.inputs['x'] + theta['b']}
        >>> exps = [Experiment.candidate({'x': 1.0}, ['y']),
        ...         Experiment.candidate({'x': 3.0}, ['y'])]
        >>> sensitivity_matrix(model, {'a': 1.0, 'b': 0.0}, exps)
        Array([[1., 1.],
               [3., 1.]], dtype=float64)
    """
    theta_arr, names = _as_theta_array(theta, param_names)
    if len(experiments) == 0:
        return jnp.zeros((0, len(names)))

    blocks = []
    for exp in experiments:
        out_names = exp.measured_names
        if not out_names:
            raise ValueError(
                f"experiment {exp.name or '(unnamed)'} measures nothing; give "
                "it observed values or build it with Experiment.candidate"
            )
        jac = jax.jacobian(
            lambda t, e=exp, o=tuple(out_names): _predictions(
                model_fn, t, e, names, o
            )
        )(theta_arr)
        jac = jnp.atleast_2d(jac)
        if weighted:
            # One sigma per named output, broadcast across the entries of a
            # vector-valued output.
            sigma = _row_sigma(model_fn, theta_arr, exp, names, tuple(out_names))
            if sigma.shape[0] != jac.shape[0]:  # pragma: no cover - defensive
                raise ValueError(
                    f"experiment {exp.name or '(unnamed)'}: model_fn returned "
                    f"{jac.shape[0]} scalar prediction(s) but "
                    f"{sigma.shape[0]} uncertainty slot(s)"
                )
            jac = jac / sigma[:, None]
        blocks.append(jac)

    s = jnp.concatenate(blocks, axis=0)

    if scale is None:
        return s
    if isinstance(scale, str):
        if scale != "theta":
            raise ValueError(f"unknown scale {scale!r}; use 'theta' or a vector")
        d = jnp.abs(theta_arr)
        d = jnp.where(d > 0, d, 1.0)
    else:
        d = jnp.asarray(scale, dtype=float)
        if d.shape != (len(names),):
            raise ValueError(f"scale has shape {d.shape}, expected {(len(names),)}")
    return s * d[None, :]


def check_identifiability(
    model_fn: Callable,
    theta: dict[str, float] | Sequence[float] | Array,
    experiments: Sequence[Any],
    param_names: Sequence[str] | None = None,
    *,
    rank_tol: float | None = None,
    scale: str | Sequence[float] | Array | None = "theta",
    weighted: bool = True,
) -> IdentifiabilityReport:
    """Test whether the parameters are separately identifiable.

    Run this **first**, before fitting and before designing experiments. It
    linearizes the model at ``theta`` and asks whether the sensitivity
    matrix has full column rank. A rank deficiency means some combination of
    parameters leaves every prediction unchanged, so no estimator and no
    experiment design can recover them individually.

    Columns are scaled by ``|theta_j|`` by default, so the test is about
    *relative* sensitivity and does not depend on the units a parameter
    happens to be expressed in -- the same equilibration
    :func:`difflow.reconciliation.structure.classify` applies before its
    rank test.

    Args:
        model_fn: ``model_fn(theta_dict, experiment) -> dict[str, value]``.
        theta: Parameter values to linearize about (dict or array).
        experiments: Experiments or candidates providing the measurements.
            Pass the full candidate pool when screening a design: if the
            whole pool cannot identify the parameters, no subset can.
        param_names: Column order; defaults to the ``theta`` dict order.
        rank_tol: Override the singular-value threshold.
        scale: Column scaling before the rank test (see
            :func:`sensitivity_matrix`); ``'theta'`` by default.
        weighted: Weight rows by ``1/sigma`` (default True).

    Returns:
        An :class:`IdentifiabilityReport`. It does *not* raise; call
        :meth:`IdentifiabilityReport.raise_if_unidentifiable` for that.

    Example:
        >>> from difflow.estimation import Experiment, check_identifiability
        >>> def model(theta, exp):
        ...     return {'y': (theta['a'] + theta['b']) * exp.inputs['x']}
        >>> exps = [Experiment.candidate({'x': x}, ['y']) for x in (1.0, 2.0)]
        >>> check_identifiability(model, {'a': 1.0, 'b': 1.0}, exps).identifiable
        False
    """
    theta_arr, names = _as_theta_array(theta, param_names)
    n_p = len(names)

    s_jax = sensitivity_matrix(
        model_fn, theta_arr, experiments, names, weighted=weighted, scale=scale
    )
    s = np.asarray(s_jax, dtype=float)
    n_obs = int(s.shape[0])

    if not np.all(np.isfinite(s)):
        bad = [names[j] for j in range(n_p) if not np.all(np.isfinite(s[:, j]))]
        raise ValueError(
            "sensitivity matrix contains non-finite entries for parameter(s) "
            f"{bad}; the model is not differentiable at this theta"
        )

    rank, sv, tol, gap = _rank_and_spectrum(s, rank_tol)

    if sv.size and sv[-1] > 0:
        cond = float(sv[0] / sv[-1])
    else:
        cond = float("inf")

    unidentifiable: list[str] = []
    null_space = None
    combinations: list[str] = []
    if rank < n_p:
        # Right singular vectors below the threshold span the null space; a
        # direction implicates a *set* of parameters, not a single one --
        # the same reading structure.py gives its unobservable variables.
        _, _, vh = np.linalg.svd(s, full_matrices=True)
        null_space = vh[rank:].T
        implicated: set[str] = set()
        for k in range(rank, n_p):
            vec = vh[k]
            mag = np.abs(vec)
            if mag.max() <= 0:
                continue
            idx = np.where(mag > 0.1 * mag.max())[0]
            for j in idx:
                implicated.add(names[j])
            terms = " ".join(
                f"{'-' if vec[j] < 0 else '+'} {abs(vec[j]):.3g}*{names[j]}"
                for j in idx
            )
            combinations.append(terms.lstrip("+ ") + " ~ 0")
        unidentifiable = [n for n in names if n in implicated]

    if rank < n_p:
        reason = (
            f"{n_p - rank} parameter combination(s) leave every prediction "
            "unchanged"
        )
    elif gap < 1e3:
        reason = (
            f"no clean rank gap (ratio {gap:.3g}); the classification is "
            "sensitive to rank_tol"
        )
    elif cond > 1e8:
        reason = (
            f"full rank but severely ill-conditioned (condition number "
            f"{cond:.3g}); the parameters are only weakly distinguishable"
        )
    else:
        reason = ""

    scale_vec = None
    if scale is not None:
        if isinstance(scale, str):
            d = np.abs(np.asarray(theta_arr, dtype=float))
            scale_vec = np.where(d > 0, d, 1.0)
        else:
            scale_vec = np.asarray(scale, dtype=float)

    return IdentifiabilityReport(
        identifiable=bool(rank == n_p),
        rank=int(rank),
        n_params=n_p,
        n_obs=n_obs,
        singular_values=sv,
        rank_tol=float(tol),
        rank_gap=float(gap),
        condition_number=cond,
        unidentifiable=unidentifiable,
        null_space=null_space,
        combinations=combinations,
        param_names=names,
        scale=scale_vec,
        reason=reason,
    )
