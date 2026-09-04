"""Fisher-information model-based design of experiments.

Given a model, the current parameter estimate and a pool of candidate
experimental conditions, this module answers *which experiments to run
next*. The Fisher information of a set of experiments is

.. math::

    \\mathrm{FIM}(\\theta) = \\sum_i S_i^T \\Sigma_i^{-1} S_i,
    \\qquad S_i = \\frac{\\partial y_i}{\\partial \\theta},

and its inverse is the asymptotic covariance of the estimate the campaign
would produce. So a campaign can be scored, and compared, *before* it is
run. Because difflow models are differentiable, :math:`S` is one
``jax.jacobian`` call -- exact and cheap. Computing it by finite
differences is what makes model-based design expensive elsewhere.

Workflow, in this order:

1. :func:`difflow.estimation.check_identifiability` -- can these
   parameters be told apart at all? A rank-deficient sensitivity matrix
   makes the FIM singular, and no design can repair it. Both functions
   here run this check first and refuse to proceed when it fails.
2. :func:`design_experiments` -- pick the ``n`` conditions that shrink the
   parameter uncertainty most, under a chosen criterion.
3. :func:`predicted_covariance` -- the confidence intervals that campaign
   would buy, so the cost of running it can be weighed before committing.

Criteria (see :func:`design_criterion`):

===========  ==========================  =====================================
name         value                       geometry of the confidence ellipsoid
===========  ==========================  =====================================
``'D'``      ``log det FIM`` (max)       shrink its volume
``'A'``      ``trace(FIM^-1)`` (min)     shrink the average axis length
``'E'``      ``lambda_min(FIM)`` (max)   shrink its longest axis
``'ME'``     ``lambda_max/lambda_min``   round it out (conditioning)
             (min)
===========  ==========================  =====================================

D-optimality is the usual default: it is invariant to rescaling the
parameters, which A- and E-optimality are not, so it does not silently
chase whichever parameter happens to be expressed in small units.

Numerics, in three parts:

*The log-determinant* comes from a Cholesky factor, never from ``det``,
whose over/underflow for even modestly sized FIMs is severe. The matrix is
equilibrated by its own diagonal first (see :func:`log_det`), so the
factorization is well scaled even when the parameters differ by ten orders
of magnitude -- the normal case for, say, a pre-exponential factor and an
activation energy.

*Singularity is decided on the spectrum*, not on whether the Cholesky
succeeded and not on the relative size of its pivots: a singular FIM
frequently factors anyway, and a badly scaled one has pivots spanning the
scale of its diagonal, which hides the small one. An eigenvalue below
``n*eps*lambda_max`` is zero. That is the same threshold
:func:`~difflow.estimation.check_identifiability` applies to the singular
values of ``S`` (``sqrt(eps)``, squared), so the design machinery and the
structural rank test agree on which problems are degenerate. When the FIM
is singular, ``log det`` is ``-inf``, ``trace(FIM^-1)`` is ``+inf``,
``lambda_min`` is 0 and the condition number ``+inf`` -- the correct
limits, not errors.

*Selection under a singular FIM.* Early in a greedy selection fewer runs
have been chosen than there are parameters, so the FIM is always singular
and the criterion is ``-inf`` for every candidate. Selection therefore
ranks on the pair ``(rank, pseudo-value)``, where the pseudo-value applies
the criterion to the nonzero eigenvalues only: information is first added
in as many independent directions as possible, and only then optimized.
Once the FIM is nonsingular the pseudo-value equals the criterion exactly.

The API follows Pyomo.DoE (Wang & Dowling, AIChE J. 68 (2022) e17813,
doi:10.1002/aic.17813), which does the same thing for Pyomo models; see
Franceschini & Macchietto, Chem. Eng. Sci. 63 (2008) 4846,
doi:10.1016/j.ces.2007.11.034 for the wider method.

**Scope, and where the rest of DoE lives.** This module selects runs from a
*candidate list*, for any model JAX can differentiate. The ``discopt-doe``
plugin (a separate distribution, importable as ``discopt.doe``) is the much
larger DoE package: it optimizes continuously over a design box
(``optimal_experiment``, ``batch_optimal_experiment``) and adds profile
likelihood, model discrimination and selection, estimability ranking,
classical and screening designs, ANOVA and Bayesian optimization. What it
cannot do is take a difflow flowsheet: ``discopt`` models are symbolic
``discopt.modeling`` expression DAGs with no black-box hook, so JAX is a
backend it lowers to rather than an entry point. That is the whole reason
this module exists. See ``docs/experiment-design.md`` for the full
comparison and guidance on which to reach for.

Example:
    >>> from difflow.estimation import Experiment, design_experiments
    >>> def model(theta, exp):
    ...     return {'y': theta['a'] * exp.inputs['x'] + theta['b']}
    >>> pool = [Experiment.candidate({'x': float(x)}, ['y'], {'y': 1.0})
    ...         for x in range(11)]
    >>> res = design_experiments(model, {'a': 1.0, 'b': 0.0}, pool, n=4)
    >>> sorted(e.inputs['x'] for e in res.selected)   # the two extremes
    [0.0, 0.0, 10.0, 10.0]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow.estimation.confidence import (
    ConfidenceResult,
    confidence_result_from_covariance,
)
from difflow.estimation.identifiability import (
    IdentifiabilityReport,
    _as_theta_array,
    check_identifiability,
    sensitivity_matrix,
)

__all__ = [
    "CRITERIA",
    "DesignResult",
    "design_criterion",
    "design_experiments",
    "fisher_information",
    "log_det",
    "predicted_covariance",
]

#: Supported design criteria.
CRITERIA = ("D", "A", "E", "ME")

#: Criteria whose conventional value is better when smaller.
_MINIMIZE = ("A", "ME")

#: Machine epsilon, the relative threshold below which a FIM *eigenvalue*
#: counts as zero.
#:
#: Note the exponent. :mod:`difflow.reconciliation.structure` -- and
#: :func:`~difflow.estimation.check_identifiability` with it -- calls a
#: *singular value* of the sensitivity matrix zero below ``sqrt(eps)``
#: times the largest, because ``S`` is evaluated at an approximate theta and
#: a structurally zero direction is only zero to about that. The FIM's
#: eigenvalues are the squares of those singular values, so the matching
#: threshold here is ``eps``, and the two tests then agree on exactly the
#: same set of degenerate problems. Using ``sqrt(eps)`` on the FIM instead
#: would be a ``eps**0.25`` test on ``S``: a genuinely informative but
#: ill-conditioned design (an Arrhenius pair, say, where the pre-exponential
#: and the activation energy are correlated to one part in 1e5) would be
#: declared singular and the selection would walk away from it.
_EPS = float(np.finfo(float).eps)


def _fim_rank_tol(w) -> Any:
    """Eigenvalue floor for a FIM, given its ascending spectrum ``w``.

    ``n * eps * lambda_max``, the tolerance :func:`numpy.linalg.matrix_rank`
    uses on singular values, applied to the eigenvalues of a symmetric
    positive semidefinite matrix (where they coincide).
    """
    n = w.shape[0]
    return n * _EPS * jnp.maximum(w[-1], 0.0)


def _check_criterion(criterion: str) -> str:
    c = str(criterion).upper()
    if c not in CRITERIA:
        raise ValueError(f"unknown criterion {criterion!r}; use one of {CRITERIA}")
    return c


def _is_nonsingular(fim: Array) -> Array:
    """Whether the FIM is numerically positive definite (traced-safe bool).

    An eigenvalue test, not a Cholesky test. Cholesky success is *not* a
    rank test: a FIM that is singular in exact arithmetic frequently factors
    anyway with a tiny pivot, and the resulting enormous-but-finite standard
    errors read as a merely poor design rather than an impossible one.
    Neither is a *relative* pivot test, which is what this function replaced:
    the pivots of ``L`` inherit the scale spread of the FIM's diagonal, so
    for a badly scaled FIM (parameters in wildly different units, which is
    the normal case) the smallest pivot can sit far above ``eps`` times the
    largest while the matrix is exactly rank deficient.

    The spectrum is taken under :func:`jax.lax.stop_gradient`: it feeds only
    a boolean, and eigenvalue derivatives blow up for the degenerate spectra
    this function exists to detect.
    """
    w = jnp.linalg.eigvalsh(jax.lax.stop_gradient(jnp.asarray(fim, dtype=float)))
    return jnp.all(jnp.isfinite(w)) & (w[0] > _fim_rank_tol(w))


def log_det(fim: Array) -> Array:
    """Log-determinant of a symmetric positive definite matrix.

    Computed from a Cholesky factor, never from ``det``: for an ``n x n``
    FIM the determinant scales like the ``n``-th power of the information
    and overflows or underflows long before its logarithm does. The matrix
    is first equilibrated by its own diagonal, ``M = D M~ D`` with
    ``D = diag(sqrt(FIM_ii))``, and

    ``log det M = 2 sum(log d) + 2 sum(log diag(chol(M~)))``,

    which is exact and keeps every logarithm on a sane scale even when the
    parameters differ by ten orders of magnitude.

    Args:
        fim: Symmetric matrix, shape ``(n, n)``.

    Returns:
        Scalar log-determinant, or ``-inf`` when the matrix is singular,
        indefinite, or numerically indistinguishable from singular (see
        :func:`_is_nonsingular`). ``-inf`` is the right answer, not an
        error: a singular FIM means infinite variance in some direction.

    Example:
        >>> import jax.numpy as jnp
        >>> round(float(log_det(jnp.diag(jnp.array([2.0, 8.0])))), 6)  # log 16
        2.772589
        >>> float(log_det(jnp.array([[1.0, 1.0], [1.0, 1.0]])))  # rank 1
        -inf
    """
    fim = jnp.asarray(fim, dtype=float)
    d = jnp.sqrt(jnp.diagonal(fim))
    d_good = jnp.isfinite(d) & (d > 0)
    d_safe = jnp.where(d_good, d, 1.0)
    chol = jnp.linalg.cholesky(fim / jnp.outer(d_safe, d_safe))
    piv = jnp.diagonal(chol)
    piv_good = jnp.isfinite(piv) & (piv > 0)
    piv_safe = jnp.where(piv_good, piv, 1.0)
    value = 2.0 * (jnp.sum(jnp.log(d_safe)) + jnp.sum(jnp.log(piv_safe)))
    ok = jnp.all(d_good) & jnp.all(piv_good) & _is_nonsingular(fim)
    return jnp.where(ok, value, -jnp.inf)


def fisher_information(
    model_fn: Callable,
    theta: dict[str, float] | Sequence[float] | Array,
    experiments: Sequence[Any],
    param_names: Sequence[str] | None = None,
    *,
    prior_fim: Array | None = None,
) -> Array:
    """Fisher information matrix of a set of experiments.

    ``FIM = sum_i S_i.T @ inv(Sigma_i) @ S_i`` with ``Sigma_i`` the diagonal
    measurement-error covariance built from each experiment's
    ``uncertainties`` (1.0 where not given). Equivalently ``S_w.T @ S_w``
    for the sigma-weighted sensitivity matrix, which is how it is formed
    here -- squaring the weighted matrix once rather than accumulating
    per-experiment products keeps the numerics identical to the SVD used by
    the rank test.

    The result is a function of ``theta`` alone (through the sensitivities)
    and of the experimental conditions -- never of the measured values, so
    it is defined for experiments that have not been run.

    Args:
        model_fn: ``model_fn(theta_dict, experiment) -> dict[str, value]``.
        theta: Parameter values (dict, or array with ``param_names``).
        experiments: Experiments or candidates.
        param_names: Row/column order; defaults to the ``theta`` dict order.
        prior_fim: Optional information already in hand -- from previously
            run experiments, or a Bayesian prior precision -- added to the
            sum. Shape ``(n_params, n_params)``.

    Returns:
        Symmetric array of shape ``(n_params, n_params)``.

    Example:
        >>> from difflow.estimation import Experiment, fisher_information
        >>> def model(theta, exp):
        ...     return {'y': theta['a'] * exp.inputs['x']}
        >>> e = Experiment.candidate({'x': 2.0}, ['y'], {'y': 0.5})
        >>> float(fisher_information(model, {'a': 1.0}, [e])[0, 0])  # (2/0.5)^2
        16.0
    """
    theta_arr, names = _as_theta_array(theta, param_names)
    s = sensitivity_matrix(
        model_fn, theta_arr, experiments, names, weighted=True, scale=None
    )
    fim = s.T @ s
    if prior_fim is not None:
        prior = jnp.asarray(prior_fim, dtype=float)
        if prior.shape != fim.shape:
            raise ValueError(
                f"prior_fim has shape {prior.shape}, expected {fim.shape}"
            )
        fim = fim + prior
    # Symmetrize: S.T @ S is symmetric in exact arithmetic, and forcing it
    # keeps eigvalsh and cholesky from seeing rounding asymmetry.
    return 0.5 * (fim + fim.T)


def design_criterion(fim: Array, criterion: str = "D") -> Array:
    """Value of a design criterion for a Fisher information matrix.

    Conventional (not sign-flipped) values, so ``'D'`` and ``'E'`` are
    better when larger and ``'A'`` and ``'ME'`` better when smaller:

    - ``'D'``: ``log det FIM``, via Cholesky. Maximize. Proportional to
      minus the log volume of the confidence ellipsoid, and invariant to
      rescaling the parameters.
    - ``'A'``: ``trace(FIM^-1)``, the sum of the parameter variances.
      Minimize.
    - ``'E'``: ``lambda_min(FIM)``, the information along the worst
      direction. Maximize.
    - ``'ME'``: modified E, ``lambda_max/lambda_min`` -- the condition
      number, i.e. the aspect ratio of the ellipsoid. Minimize.

    A singular FIM gives ``-inf``, ``+inf``, ``0`` and ``+inf``
    respectively, which are the correct limits, not errors.

    Args:
        fim: Fisher information matrix, shape ``(n, n)``.
        criterion: One of ``'D'``, ``'A'``, ``'E'``, ``'ME'``.

    Returns:
        Scalar criterion value.

    Example:
        >>> import jax.numpy as jnp
        >>> fim = jnp.diag(jnp.array([4.0, 1.0]))
        >>> float(design_criterion(fim, 'A'))       # 1/4 + 1/1
        1.25
        >>> float(design_criterion(fim, 'ME'))      # 4 / 1
        4.0
    """
    c = _check_criterion(criterion)
    fim = jnp.asarray(fim, dtype=float)
    if c == "D":
        return log_det(fim)

    w = jnp.linalg.eigvalsh(fim)
    w_max = w[-1]
    w_min = w[0]
    singular = w_min <= _fim_rank_tol(w)

    if c == "E":
        return jnp.where(singular, 0.0, w_min)
    if c == "ME":
        return jnp.where(singular, jnp.inf, w_max / jnp.where(singular, 1.0, w_min))
    # A-optimality: sum of 1/lambda, which is trace(FIM^-1) without forming
    # the inverse.
    safe = jnp.where(singular, 1.0, w)
    return jnp.where(singular, jnp.inf, jnp.sum(1.0 / safe))


def _pseudo_score(fim: np.ndarray, criterion: str) -> tuple[int, float]:
    """Rank-then-criterion score used for selection; larger is better.

    Returns ``(rank, value)`` compared lexicographically. ``value`` applies
    the criterion to the nonzero eigenvalues only and is sign-flipped so
    that larger is always better. For a nonsingular FIM the rank is the
    same for every candidate and the value is the criterion itself, so this
    reduces to plain criterion optimization; while the FIM is still
    singular it prefers whichever candidate opens a new information
    direction, which is what actually distinguishes the early picks.
    """
    w = np.linalg.eigvalsh(np.asarray(fim, dtype=float))
    if not w.size:
        return 0, -np.inf
    pos = w[w > w.size * _EPS * max(float(w[-1]), 0.0)]
    rank = int(pos.size)
    if rank == 0:
        return 0, -np.inf
    if criterion == "D":
        value = float(np.sum(np.log(pos)))
    elif criterion == "A":
        value = -float(np.sum(1.0 / pos))
    elif criterion == "E":
        value = float(pos.min())
    else:  # ME
        value = -float(pos.max() / pos.min())
    return rank, value


@dataclass
class DesignResult(ParamsMixin):
    """Outcome of :func:`design_experiments`.

    Attributes:
        selected: The chosen experiments, in the order they were added.
        indices: Their positions in the candidate pool (repeats appear more
            than once when ``replace=True``).
        criterion: The criterion optimized.
        criterion_value: Its conventional value for the final design.
        fim: Fisher information of the final design (including ``prior_fim``
            and any ``existing`` experiments).
        covariance: ``inv(fim)``, the asymptotic parameter covariance the
            campaign would buy; ``None`` when the FIM is singular.
        std_errors: Predicted standard errors per parameter.
        criterion_history: Criterion value after each addition, so the point
            of diminishing returns is visible.
        identifiability: Report from the pre-flight rank test.
        param_names: Parameter names, in FIM order.
        theta: The parameter values the design was computed at.
        method: ``'greedy'`` or ``'exchange'``.
        n_candidates: Size of the candidate pool.
        n_exchanges: Number of accepted swaps (exchange method only).
    """

    selected: list[Any]
    indices: list[int]
    criterion: str
    criterion_value: float
    fim: Array
    covariance: Array | None
    std_errors: dict[str, float]
    criterion_history: list[float] = field(default_factory=list)
    identifiability: IdentifiabilityReport | None = None
    param_names: list[str] = field(default_factory=list)
    theta: dict[str, float] = field(default_factory=dict)
    method: str = "greedy"
    n_candidates: int = 0
    n_exchanges: int = 0

    def summary(self) -> str:
        """Human-readable table of the design and what it buys."""
        better = "minimize" if self.criterion in _MINIMIZE else "maximize"
        lines = [
            "=" * 62,
            f"{self.criterion}-optimal design ({self.method}, {better})",
            "=" * 62,
            f"selected {len(self.selected)} of {self.n_candidates} candidates",
            f"criterion value: {self.criterion_value:.6g}",
        ]
        if self.n_exchanges:
            lines.append(f"accepted exchanges: {self.n_exchanges}")
        lines += ["", f"  {'#':>3} {'candidate':>9}  inputs", "  " + "-" * 56]
        for k, (i, exp) in enumerate(zip(self.indices, self.selected)):
            label = exp.name or f"[{i}]"
            inputs = ", ".join(f"{k2}={_fmt(v)}" for k2, v in exp.inputs.items())
            lines.append(f"  {k + 1:>3} {label:>9}  {inputs}")
        lines += ["", "Predicted standard errors:", ""]
        lines.append(f"  {'parameter':<15} {'value':>12} {'std err':>12} {'rel':>10}")
        lines.append("  " + "-" * 51)
        for name in self.param_names:
            val = self.theta.get(name, float("nan"))
            se = self.std_errors.get(name, float("nan"))
            rel = abs(se / val) if val else float("nan")
            lines.append(f"  {name:<15} {val:>12.6g} {se:>12.6g} {rel:>10.3g}")
        lines.append("=" * 62)
        return "\n".join(lines)


def _fmt(v: Any) -> str:
    try:
        return f"{float(v):.4g}"
    except (TypeError, ValueError):
        return str(v)


def _covariance_and_errors(fim: Array, names: Sequence[str]):
    """``inv(FIM)`` by Cholesky solve, plus per-parameter standard errors.

    Returns ``(None, {name: inf})`` when the FIM is numerically singular --
    see :func:`_is_nonsingular` for why the test is on the spectrum and not
    on whether the factorization succeeded.
    """
    fim_np = np.asarray(fim, dtype=float)
    w = np.linalg.eigvalsh(fim_np)
    singular = not np.all(np.isfinite(w)) or w[0] <= w.size * _EPS * max(
        float(w[-1]), 0.0
    )
    d = np.sqrt(np.diagonal(fim_np))
    if singular or not np.all(np.isfinite(d)) or np.any(d <= 0):
        return None, {n: float("inf") for n in names}
    # Equilibrate before factoring, for the same reason log_det does.
    scaled = fim_np / np.outer(d, d)
    try:
        chol = np.linalg.cholesky(scaled)
    except np.linalg.LinAlgError:  # pragma: no cover - excluded by the test above
        return None, {n: float("inf") for n in names}
    ident = np.eye(fim_np.shape[0])
    # Two triangular solves rather than a general inverse.
    y = np.linalg.solve(chol, ident)
    cov = np.linalg.solve(chol.T, y) / np.outer(d, d)
    cov = 0.5 * (cov + cov.T)
    var = np.diag(cov)
    errs = {
        n: float(np.sqrt(var[i])) if var[i] >= 0 else float("nan")
        for i, n in enumerate(names)
    }
    return jnp.asarray(cov), errs


def predicted_covariance(
    model_fn: Callable,
    theta: dict[str, float] | Sequence[float] | Array,
    experiments: Sequence[Any],
    param_names: Sequence[str] | None = None,
    *,
    alpha: float = 0.05,
    prior_fim: Array | None = None,
    require_identifiable: bool = True,
    rank_tol: float | None = None,
) -> ConfidenceResult:
    """Confidence intervals a proposed campaign would buy, before running it.

    The asymptotic covariance of the maximum-likelihood estimate is
    ``inv(FIM)``, and the FIM depends only on the conditions and the
    measurement uncertainties -- not on the values that will be measured.
    So the intervals can be quoted in the proposal.

    This differs from :func:`difflow.estimation.fisher_confidence_intervals`
    in exactly one respect: that function estimates the residual variance
    from data that exist, whereas here the ``uncertainties`` declared on
    each experiment *are* the assumed error model. The result type is the
    same :class:`~difflow.estimation.confidence.ConfidenceResult`, so the
    predicted and achieved intervals can be compared field by field.

    The intervals are centered on ``theta`` and use a Student-t quantile
    with ``n_obs - n_params`` degrees of freedom, matching
    ``fisher_confidence_intervals``. They are a linearization about
    ``theta``: for a nonlinear model they are exact only to the extent the
    model is locally linear over the interval, and they are only as good as
    the ``theta`` used to compute them.

    Args:
        model_fn: ``model_fn(theta_dict, experiment) -> dict[str, value]``.
        theta: Parameter values the design is evaluated at.
        experiments: The proposed campaign (candidates and/or already-run
            experiments).
        param_names: Parameter order; defaults to the ``theta`` dict order.
        alpha: Significance level; 0.05 gives 95% intervals.
        prior_fim: Information already in hand, added to the FIM.
        require_identifiable: Run the structural rank test first and raise
            :class:`~difflow.estimation.identifiability.IdentifiabilityError`
            when it fails. Set False only to inspect a degenerate case
            deliberately, in which case the standard errors come back
            infinite.
        rank_tol: Override for the rank test threshold.

    Returns:
        A :class:`~difflow.estimation.confidence.ConfidenceResult` holding
        the predicted covariance, standard errors, intervals and parameter
        correlation matrix.

    Example:
        >>> from difflow.estimation import Experiment, predicted_covariance
        >>> def model(theta, exp):
        ...     return {'y': theta['a'] * exp.inputs['x'] + theta['b']}
        >>> exps = [Experiment.candidate({'x': x}, ['y'], {'y': 1.0})
        ...         for x in (0.0, 0.0, 1.0, 1.0)]
        >>> ci = predicted_covariance(model, {'a': 2.0, 'b': 1.0}, exps)
        >>> round(ci.std_errors['a'], 6)     # inv([[2, 2], [2, 4]])[0, 0] = 1
        1.0
    """
    theta_arr, names = _as_theta_array(theta, param_names)

    report = None
    if require_identifiable:
        # Ordering is enforced, not merely recommended: a rank-deficient
        # sensitivity matrix makes everything below meaningless.
        report = check_identifiability(
            model_fn, theta_arr, experiments, names, rank_tol=rank_tol
        )
        report.raise_if_unidentifiable()

    fim = fisher_information(
        model_fn, theta_arr, experiments, names, prior_fim=prior_fim
    )
    cov, _ = _covariance_and_errors(fim, names)

    n_obs = sum(len(exp.measured_names) for exp in experiments)
    # Same assembler as fisher_confidence_intervals, so the predicted and
    # the achieved intervals are directly comparable.
    return confidence_result_from_covariance(
        cov, theta_arr, names, alpha=alpha, dof=n_obs - len(names)
    )


def design_experiments(
    model_fn: Callable,
    theta: dict[str, float] | Sequence[float] | Array,
    candidates: Sequence[Any],
    n: int,
    criterion: str = "D",
    param_names: Sequence[str] | None = None,
    *,
    method: str = "greedy",
    replace: bool = True,
    existing: Sequence[Any] | None = None,
    prior_fim: Array | None = None,
    require_identifiable: bool = True,
    rank_tol: float | None = None,
    max_exchange_sweeps: int = 20,
) -> DesignResult:
    """Choose the ``n`` candidate experiments that most reduce uncertainty.

    The pre-flight structural check comes first (see
    :func:`~difflow.estimation.identifiability.check_identifiability`): if
    the *whole* candidate pool cannot identify the parameters, no subset of
    it can, and the answer is a new kind of measurement rather than a
    better-chosen run list. That check raises rather than returning a
    design, because a design computed on a singular FIM is meaningless.

    Selection is greedy by default: repeatedly add the candidate that most
    improves the criterion, given everything already selected (plus
    ``existing`` and ``prior_fim``). Greedy is the standard construction
    and, for D-optimality, it is submodular-flavored and usually within a
    few percent of the optimum. ``method='exchange'`` then runs Fedorov-style
    swaps -- try replacing each selected run with each unselected one, keep
    the best improvement, repeat -- which escapes the greedy path at the
    cost of ``n * n_candidates`` evaluations per sweep.

    Args:
        model_fn: ``model_fn(theta_dict, experiment) -> dict[str, value]``.
        theta: Parameter values the design is computed at. Design is local:
            a different ``theta`` can give a different design, which is why
            designing, running and refitting is an iterative loop.
        candidates: Pool of conditions to choose from, typically built with
            :meth:`difflow.estimation.Experiment.candidate`.
        n: Number of runs to select.
        criterion: ``'D'`` (default), ``'A'``, ``'E'`` or ``'ME'``.
        param_names: Parameter order; defaults to the ``theta`` dict order.
        method: ``'greedy'`` or ``'exchange'``.
        replace: Allow a candidate to be chosen more than once, i.e.
            replicates. True by default, since replicating an informative
            condition is often genuinely optimal. With False, ``n`` must not
            exceed the pool size.
        existing: Experiments already run, whose information is included but
            which are not part of the returned design.
        prior_fim: Additional prior information matrix.
        require_identifiable: Run the rank test on the full pool first
            (default True). Disabling it is only for deliberate inspection
            of a degenerate problem.
        rank_tol: Override for the rank test threshold.
        max_exchange_sweeps: Cap on exchange sweeps.

    Returns:
        A :class:`DesignResult` with the selected experiments, the final
        FIM, the predicted covariance and standard errors, and the
        criterion trajectory.

    Raises:
        IdentifiabilityError: When the pool cannot identify the parameters.
        ValueError: For an empty pool, ``n <= 0``, or ``n`` larger than the
            pool with ``replace=False``.

    Example:
        >>> from difflow.estimation import Experiment, design_experiments
        >>> def model(theta, exp):        # straight line: extremes win
        ...     return {'y': theta['a'] * exp.inputs['x'] + theta['b']}
        >>> pool = [Experiment.candidate({'x': float(x)}, ['y'], {'y': 1.0})
        ...         for x in range(5)]
        >>> res = design_experiments(model, {'a': 1.0, 'b': 0.0}, pool, n=2)
        >>> sorted(e.inputs['x'] for e in res.selected)
        [0.0, 4.0]
    """
    c = _check_criterion(criterion)
    if method not in ("greedy", "exchange"):
        raise ValueError(f"unknown method {method!r}; use 'greedy' or 'exchange'")
    candidates = list(candidates)
    if not candidates:
        raise ValueError("candidate pool is empty")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not replace and n > len(candidates):
        raise ValueError(
            f"asked for {n} runs from {len(candidates)} candidates without "
            "replacement; pass replace=True to allow replicates"
        )

    theta_arr, names = _as_theta_array(theta, param_names)
    n_p = len(names)

    report = None
    if require_identifiable:
        report = check_identifiability(
            model_fn, theta_arr, candidates, names, rank_tol=rank_tol
        )
        report.raise_if_unidentifiable()

    # Per-candidate information contributions, computed once. Each is
    # S_i.T @ inv(Sigma_i) @ S_i, so the FIM of any subset is just their
    # sum -- which is what makes greedy and exchange cheap.
    contributions = [
        np.asarray(
            fisher_information(model_fn, theta_arr, [exp], names), dtype=float
        )
        for exp in candidates
    ]

    base = np.zeros((n_p, n_p))
    if prior_fim is not None:
        prior = np.asarray(prior_fim, dtype=float)
        if prior.shape != base.shape:
            raise ValueError(f"prior_fim has shape {prior.shape}, expected {base.shape}")
        base = base + prior
    if existing:
        base = base + np.asarray(
            fisher_information(model_fn, theta_arr, list(existing), names), dtype=float
        )

    chosen: list[int] = []
    history: list[float] = []
    current = base.copy()

    for _ in range(n):
        best_idx = None
        best_score = (-1, -np.inf)
        for i, contrib in enumerate(contributions):
            if not replace and i in chosen:
                continue
            score = _pseudo_score(current + contrib, c)
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx is None:  # pragma: no cover - guarded by the n check
            break
        chosen.append(best_idx)
        current = current + contributions[best_idx]
        history.append(float(design_criterion(jnp.asarray(current), c)))

    n_exchanges = 0
    if method == "exchange":
        for _ in range(max_exchange_sweeps):
            best_swap = None
            best_score = _pseudo_score(current, c)
            for pos, out_idx in enumerate(chosen):
                reduced = current - contributions[out_idx]
                for i, contrib in enumerate(contributions):
                    if i == out_idx:
                        continue
                    if not replace and i in chosen:
                        continue
                    score = _pseudo_score(reduced + contrib, c)
                    if score > best_score:
                        best_score = score
                        best_swap = (pos, out_idx, i)
            if best_swap is None:
                break
            pos, out_idx, in_idx = best_swap
            current = current - contributions[out_idx] + contributions[in_idx]
            chosen[pos] = in_idx
            n_exchanges += 1
            history.append(float(design_criterion(jnp.asarray(current), c)))

    fim = jnp.asarray(0.5 * (current + current.T))
    cov, std_errors = _covariance_and_errors(fim, names)

    return DesignResult(
        selected=[candidates[i] for i in chosen],
        indices=chosen,
        criterion=c,
        criterion_value=float(design_criterion(fim, c)),
        fim=fim,
        covariance=cov,
        std_errors=std_errors,
        criterion_history=history,
        identifiability=report,
        param_names=names,
        theta={nm: float(theta_arr[i]) for i, nm in enumerate(names)},
        method=method,
        n_candidates=len(candidates),
        n_exchanges=n_exchanges,
    )
