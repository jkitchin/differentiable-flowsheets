"""Model-discrimination design: runs that tell rival models apart.

:func:`~difflow.estimation.design_experiments` chooses runs to measure the
parameters of *one* model. Those designs typically use as many distinct
conditions as there are parameters, and a model can fit that many
conditions whatever its form, so they cannot test it. When two or more
mechanisms are in contention -- first- or second-order kinetics, one site
or two -- the next runs should go where the fitted models disagree most.

Each rival model is fitted to the data in hand first; this module takes
the fits (``thetas``) and scores every candidate run:

``'HR'`` -- Hunter and Reiner (*Technometrics* 7 (1965) 307)
    .. math::

        D(x) = \\sum_{j<k} \\sum_m
               \\frac{(y_{j,m}(x) - y_{k,m}(x))^2}{\\sigma_m^2},

    the squared difference of the predictions, summed over model pairs
    and measured outputs ``m``, in units of the measurement variance. For
    one measurement it is the expected increase of the wrong model's
    weighted sum of squares (``chi^2``) per run, if the other model is
    right.

``'BF'`` -- Buzzi-Ferraris and Forzatti (*Chem. Eng. Sci.* 39 (1984) 81)
    .. math::

        D(x) = \\sum_{j<k} \\Delta^T (2\\Sigma + V_j + V_k)^{-1} \\Delta,
        \\qquad V_j = S_j C_j S_j^T,

    the same difference measured against everything that blurs it: the
    measurement variance of both observations and each model's prediction
    variance, from its parameter covariance ``C_j`` and sensitivities
    ``S_j``. A difference the models' own uncertainty could explain counts
    for little. With ``C_j = 0`` it is half the Hunter--Reiner value.

The Hunter--Reiner value of a set of runs is the sum over the runs, so the
best ``n`` runs are the best single run replicated (or, with
``replace=False``, the ``n`` best runs). For Buzzi-Ferraris the runs share
the parameter uncertainty, so the set is chosen greedily on its joint
value, and replicating a run is worth less than the first time.

Predictions come from one compiled ``jax.vmap`` of each model over the
candidate pool when the candidates can be stacked (see
:mod:`difflow.estimation._batch`), and from a loop otherwise.

Example:
    >>> from difflow.estimation import Experiment, design_discrimination
    >>> def line(theta, exp):
    ...     return {'y': theta['a'] * exp.inputs['x']}
    >>> def square(theta, exp):
    ...     return {'y': theta['b'] * exp.inputs['x'] ** 2}
    >>> pool = [Experiment.candidate({'x': x}, ['y'], {'y': 0.1})
    ...         for x in (0.0, 0.5, 1.0, 2.0)]
    >>> res = design_discrimination([line, square], [{'a': 1.0}, {'b': 1.0}],
    ...                             pool, n=1)
    >>> res.selected[0].inputs['x'], round(res.criterion_value, 6)   # (4-2)^2/0.01
    (2.0, 400.0)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from difflow.estimation._batch import NotStackable, batched_model, stack_inputs
from difflow.estimation.design import _covariance_and_errors, fisher_information
from difflow.estimation.identifiability import (
    _as_theta_array,
    _predictions,
    _row_sigma,
    _sensitivity_blocks,
)
from difflow.params_mixin import ParamsMixin

__all__ = [
    "DISCRIMINATION_CRITERIA",
    "DiscriminationResult",
    "design_discrimination",
]

#: Supported model-discrimination criteria.
DISCRIMINATION_CRITERIA = ("HR", "BF")

_ALIASES = {
    "HR": "HR", "HUNTER_REINER": "HR", "HUNTER-REINER": "HR",
    "BF": "BF", "BUZZI_FERRARIS": "BF", "BUZZI-FERRARIS": "BF",
}


@dataclass
class DiscriminationResult(ParamsMixin):
    """Outcome of :func:`design_discrimination`.

    Attributes:
        selected: The chosen experiments, in the order they were added.
        indices: Their positions in the candidate pool (repeats appear more
            than once when ``replace=True``).
        criterion: ``'HR'`` (Hunter--Reiner) or ``'BF'`` (Buzzi-Ferraris).
        criterion_value: The criterion for the selected runs together
            (larger separates the models better).
        scores: The criterion of each candidate run on its own, shape
            ``(n_candidates,)``; plot it over the conditions to see where the
            models disagree.
        pair_scores: The same per model pair, ``{(name_j, name_k): array}``.
        predictions: Each model's predictions of the measured outputs at
            every candidate, ``{name: array (n_candidates, n_rows)}`` (a list
            of per-candidate arrays when the candidates measure different
            outputs).
        model_names: The models' names.
        criterion_history: ``criterion_value`` after each addition.
        n_candidates: Size of the candidate pool.
    """

    selected: list[Any]
    indices: list[int]
    criterion: str
    criterion_value: float
    scores: np.ndarray
    pair_scores: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    predictions: dict[str, Any] = field(default_factory=dict)
    model_names: list[str] = field(default_factory=list)
    criterion_history: list[float] = field(default_factory=list)
    n_candidates: int = 0

    def summary(self) -> str:
        """Human-readable table of the chosen runs and the models' predictions."""
        name = {"HR": "Hunter-Reiner", "BF": "Buzzi-Ferraris"}[self.criterion]
        lines = [
            "=" * 62,
            f"{name} model-discrimination design: {', '.join(self.model_names)}",
            "=" * 62,
            f"selected {len(self.selected)} of {self.n_candidates} candidates",
            f"criterion value: {self.criterion_value:.6g}",
            "",
            f"  {'#':>3} {'candidate':>9} {'score':>10}  inputs",
            "  " + "-" * 56,
        ]
        for k, (i, exp) in enumerate(zip(self.indices, self.selected)):
            label = exp.name or f"[{i}]"
            inputs = ", ".join(f"{k2}={_fmt(v)}" for k2, v in exp.inputs.items())
            lines.append(f"  {k + 1:>3} {label:>9} {self.scores[i]:>10.4g}  {inputs}")
        lines.append("=" * 62)
        return "\n".join(lines)


def _fmt(v: Any) -> str:
    try:
        return f"{float(v):.4g}"
    except (TypeError, ValueError):
        return str(v)


def _check_criterion(criterion: str) -> str:
    c = _ALIASES.get(str(criterion).upper())
    if c is None:
        raise ValueError(
            f"unknown discrimination criterion {criterion!r}; use one of "
            f"{DISCRIMINATION_CRITERIA} ('hunter_reiner', 'buzzi_ferraris')"
        )
    return c


def _rows(model_fn, theta_arr, names, candidates):
    """Per-candidate predictions and 1-sigma values, flattened over the outputs.

    One compiled ``vmap`` over the pool when the candidates stack, one call
    per candidate otherwise. Returns two lists of 1-D numpy arrays.
    """
    try:
        out_names = tuple(candidates[0].measured_names)
        if any(tuple(e.measured_names) != out_names for e in candidates[1:]):
            raise NotStackable("candidates measure different outputs")
        stacked, static = stack_inputs(candidates)
        one = batched_model(model_fn, names, candidates, static)

        def row(inp):
            preds = one(theta_arr, inp)
            return tuple(jnp.ravel(jnp.asarray(preds[k], dtype=float))
                         for k in out_names)

        parts = jax.jit(jax.vmap(row))(stacked)          # per output: (N, size_k)
        sigma = np.stack([np.asarray(e.sigma_array) for e in candidates])
        pred = np.concatenate([np.asarray(p) for p in parts], axis=1)
        sig = np.concatenate(
            [np.repeat(sigma[:, i:i + 1], p.shape[1], axis=1)
             for i, p in enumerate(parts)], axis=1)
        return list(pred), list(sig)
    except Exception:  # noqa: BLE001, S110 - the loop below reports real errors
        pass
    preds, sigs = [], []
    for exp in candidates:
        out = tuple(exp.measured_names)
        preds.append(np.asarray(_predictions(model_fn, theta_arr, exp, names, out)))
        sigs.append(np.asarray(_row_sigma(model_fn, theta_arr, exp, names, out)))
    return preds, sigs


def _as_named(models, thetas, model_names):
    if isinstance(models, Mapping):
        names = list(models)
        fns = [models[k] for k in names]
        if isinstance(thetas, Mapping):
            missing = [k for k in names if k not in thetas]
            if missing:
                raise ValueError(f"thetas has no entry for model(s) {missing}")
            thetas = [thetas[k] for k in names]
    else:
        fns = list(models)
        names = None
    if isinstance(thetas, Mapping):
        raise TypeError("thetas is a mapping but models is not; pass both as "
                        "sequences or both as mappings")
    thetas = list(thetas)
    if model_names is not None:
        names = list(model_names)
    if names is None:
        names = [getattr(f, "__name__", None) or f"model {i + 1}"
                 for i, f in enumerate(fns)]
        if len(set(names)) < len(names):
            names = [f"model {i + 1}" for i in range(len(fns))]
    if len(fns) < 2:
        raise ValueError(f"model discrimination needs at least two models, got {len(fns)}")
    if len(thetas) != len(fns):
        raise ValueError(f"{len(fns)} models but {len(thetas)} parameter sets")
    if len(names) != len(fns) or len(set(names)) < len(names):
        raise ValueError(f"model_names must be {len(fns)} distinct names, got {names}")
    if any(not callable(f) for f in fns):
        raise TypeError("every model must be a callable model_fn(theta, exp)")
    return names, fns, thetas


def design_discrimination(
    models: Sequence[Callable] | Mapping[str, Callable],
    thetas: Sequence[dict[str, float]] | Mapping[str, dict[str, float]],
    candidates: Sequence[Any],
    n: int = 1,
    criterion: str = "HR",
    *,
    replace: bool = True,
    covariances: Sequence[Any] | None = None,
    existing: Sequence[Any] | None = None,
    model_names: Sequence[str] | None = None,
) -> DiscriminationResult:
    """Choose the ``n`` candidate runs that best tell rival models apart.

    Fit every rival model to the data in hand, then pass the models and
    their fitted parameters. The candidates are scored on how far apart
    the models' predictions are, in units of what blurs the difference
    (see the module docstring for the two criteria); the runs with the
    largest score are the ones whose outcome most changes which model
    is believed. Run them, refit every model, and compare the fits (AIC,
    the sum of squares); repeat if the answer is still unclear.

    Args:
        models: The rival models, each ``model_fn(theta_dict, experiment)
            -> dict``, as a sequence or as a ``{name: model_fn}`` mapping.
            Their parameters may differ in number and name.
        thetas: Each model's fitted parameters (a dict per model), in the
            same order as ``models`` or as a mapping with the same keys.
        candidates: Pool of conditions, typically built with
            :meth:`difflow.estimation.Experiment.candidate`; their
            ``uncertainties`` are the measurement sigma (1.0 where not
            given), and every model must predict each measured output.
        n: Number of runs to select.
        criterion: ``'HR'`` (Hunter--Reiner, default; also
            ``'hunter_reiner'``) or ``'BF'`` (Buzzi-Ferraris; also
            ``'buzzi_ferraris'``).
        replace: Allow a candidate to be chosen more than once (default
            True). With False, ``n`` must not exceed the pool size.
        covariances: For ``'BF'``: each model's parameter covariance, an
            ``(n_params, n_params)`` array in its ``thetas`` order (for
            example ``EstimationResult``'s or ``inv(fisher_information)``).
        existing: For ``'BF'`` without ``covariances``: the runs already
            made; each model's covariance is then ``inv(FIM)`` of these
            runs at its ``thetas``, with the declared uncertainties.
        model_names: Names for the models (defaults to the mapping keys,
            or the functions' ``__name__``).

    Returns:
        A :class:`DiscriminationResult` with the chosen runs, the
        criterion for them together and for every candidate alone, and
        each model's predictions over the pool.

    Raises:
        ValueError: Fewer than two models, mismatched ``thetas``, an
            unknown criterion, an empty pool, ``n <= 0``, ``n`` larger
            than the pool with ``replace=False``, or ``'BF'`` with neither
            ``covariances`` nor ``existing`` (or a covariance of the wrong
            shape, or a singular ``existing`` information matrix).
        TypeError: A model that is not callable, or ``thetas`` given as a
            mapping when ``models`` is a sequence.

    Example:
        >>> from difflow.estimation import Experiment, design_discrimination
        >>> def first(theta, exp):         # X of a first-order CSTR
        ...     u = theta['k'] * exp.inputs['tau']
        ...     return {'X': u / (1 + u)}
        >>> def second(theta, exp):        # ... and of a second-order one
        ...     u = theta['k'] * exp.inputs['tau']
        ...     return {'X': (2 * u + 1 - (4 * u + 1) ** 0.5) / (2 * u)}
        >>> pool = [Experiment.candidate({'tau': t}, ['X'], {'X': 0.01})
        ...         for t in (0.5, 1.0, 2.0, 8.0)]
        >>> res = design_discrimination({'first': first, 'second': second},
        ...                             {'first': {'k': 1.0}, 'second': {'k': 2.0}},
        ...                             pool, n=2)
        >>> [e.inputs['tau'] for e in res.selected]   # high conversion
        [8.0, 8.0]
    """
    crit = _check_criterion(criterion)
    names, fns, thetas = _as_named(models, thetas, model_names)
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
    for exp in candidates:
        if not exp.measured_names:
            raise ValueError(
                f"candidate {exp.name or '(unnamed)'} measures nothing; build it "
                "with Experiment.candidate(inputs, measures, uncertainties)"
            )

    arrs = [_as_theta_array(th, None) for th in thetas]
    preds, sigma = {}, None
    for name, fn, (t_arr, p_names) in zip(names, fns, arrs):
        p, s = _rows(fn, t_arr, p_names, candidates)
        preds[name] = p
        sigma = s

    if crit == "BF":
        covs = _model_covariances(names, fns, arrs, covariances, existing)
        sens = {
            name: [np.asarray(b, dtype=float)
                   for b in _sensitivity_blocks(fn, t_arr, candidates, p_names, False)]
            for name, fn, (t_arr, p_names) in zip(names, fns, arrs)
        }

    pairs = list(combinations(names, 2))
    n_c = len(candidates)

    def set_value(idx: list[int]) -> tuple[float, dict]:
        """Criterion for the runs ``idx`` together, and its value per pair."""
        var = np.concatenate([sigma[i] for i in idx]) ** 2
        per_pair = {}
        for a, b in pairs:
            delta = np.concatenate([preds[a][i] - preds[b][i] for i in idx])
            if crit == "HR":
                per_pair[(a, b)] = float(np.sum(delta ** 2 / var))
            else:
                v = 2.0 * np.diag(var)
                for m in (a, b):
                    s_m = np.concatenate([sens[m][i] for i in idx], axis=0)
                    v = v + s_m @ covs[m] @ s_m.T
                per_pair[(a, b)] = float(delta @ np.linalg.solve(v, delta))
        return sum(per_pair.values()), per_pair

    singles = [set_value([i]) for i in range(n_c)]
    scores = np.array([v for v, _ in singles])
    pair_scores = {pr: np.array([pp[pr] for _, pp in singles]) for pr in pairs}

    chosen: list[int] = []
    history: list[float] = []
    if crit == "HR":
        # Additive over runs: the best set is the best run(s).
        order = np.argsort(-scores, kind="stable")
        chosen = [int(order[0])] * n if replace else [int(i) for i in order[:n]]
        total = 0.0
        for i in chosen:
            total += float(scores[i])
            history.append(total)
    else:
        for _ in range(n):
            best, best_val = None, -np.inf
            for i in range(n_c):
                if not replace and i in chosen:
                    continue
                val = set_value(chosen + [i])[0]
                if val > best_val:
                    best, best_val = i, val
            chosen.append(best)
            history.append(best_val)

    return DiscriminationResult(
        selected=[candidates[i] for i in chosen],
        indices=chosen,
        criterion=crit,
        criterion_value=history[-1],
        scores=scores,
        pair_scores=pair_scores,
        predictions={k: _maybe_stack(v) for k, v in preds.items()},
        model_names=names,
        criterion_history=history,
        n_candidates=n_c,
    )


def _maybe_stack(rows):
    try:
        return np.stack(rows)
    except ValueError:
        return rows


def _model_covariances(names, fns, arrs, covariances, existing):
    """Each model's parameter covariance for the Buzzi-Ferraris criterion."""
    covs = {}
    if covariances is not None:
        covariances = list(covariances)
        if len(covariances) != len(names):
            raise ValueError(f"{len(names)} models but {len(covariances)} covariances")
        for name, cov, (_, p_names) in zip(names, covariances, arrs):
            cov = np.asarray(cov, dtype=float)
            if cov.shape != (len(p_names), len(p_names)):
                raise ValueError(
                    f"covariance of {name!r} has shape {cov.shape}, expected "
                    f"{(len(p_names), len(p_names))}"
                )
            covs[name] = cov
        return covs
    if not existing:
        raise ValueError(
            "criterion 'BF' needs each model's parameter covariance: pass "
            "covariances=[...] or the runs already made as existing=[...]"
        )
    for name, fn, (t_arr, p_names) in zip(names, fns, arrs):
        fim = fisher_information(fn, t_arr, list(existing), p_names)
        cov, _ = _covariance_and_errors(fim, p_names)
        if cov is None:
            raise ValueError(
                f"the existing runs cannot identify the parameters of {name!r} "
                "(singular information matrix); pass covariances= instead"
            )
        covs[name] = np.asarray(cov)
    return covs
