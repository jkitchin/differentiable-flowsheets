"""One trace of the model for a whole list of experiments.

The estimation objectives are sums over experiments, and the obvious way
to write them -- a Python loop calling ``model_fn(theta, exp)`` once per
``Experiment`` -- makes ``jit`` trace and compile one copy of the model per
experiment. For a closed-form model that is harmless; for a difflow unit
(a ``CSTR`` with its internal Newton solve) it is the whole cost of a fit.

:class:`Batch` stacks the experiments instead. When they share their
structure -- the same input keys, numeric inputs of equal shape (inputs
that are not numeric must be the same object or equal in every
experiment), the same observed outputs -- the model is ``jax.vmap``-ed
over the stacked inputs and traced once, whatever the number of
experiments. The model still receives an ``Experiment``: a
:class:`BatchedExperiment` whose ``inputs`` hold one experiment's (traced)
values. Anything the model reads that differs between experiments and
cannot be stacked (``name``, ``metadata``, ``observed``) makes the batch
fall back to the loop, as does any error while tracing the vectorized
model (Python control flow on an input value, say), so a model that works
one experiment at a time keeps working unchanged.

The observations, uncertainties and a per-experiment *count* are
arguments of the objective rather than constants baked into it
(:meth:`Batch.objective` returns ``f(theta, data)``). That is what lets the
bootstrap and cross-validation compile once and refit many times: a
resample with replacement is a vector of counts, a residual resample is a
new ``observed``, and neither changes a shape.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from difflow.estimation.experiment import Experiment

#: The per-entry terms of the three objectives, ``r`` the residual and
#: ``s`` the 1-sigma uncertainty (1 where none is given).
_TERMS: dict[str, Callable] = {
    'sse': lambda r, s: r ** 2,
    'wsse': lambda r, s: (r / s) ** 2,
    'nll': lambda r, s: 0.5 * ((r / s) ** 2 + jnp.log(2 * jnp.pi * s ** 2)),
}


class NotStackable(Exception):
    """The experiments cannot share one vectorized trace of the model."""


def _is_numeric(v: Any) -> bool:
    return isinstance(v, (bool, int, float, complex, np.number, np.ndarray, jax.Array))


def _same(values: Sequence[Any]) -> bool:
    """Whether every value is the first one (identical, or equal as a bool)."""
    v0 = values[0]
    for v in values[1:]:
        if v is v0:
            continue
        try:
            eq = v == v0
            if not isinstance(eq, bool) or not eq:
                return False
        except Exception:  # noqa: BLE001 - an uncomparable value is a difference
            return False
    return True


def _common(experiments, attr):
    values = [getattr(e, attr) for e in experiments]
    if not _same(values):
        raise NotStackable(f"experiments differ in {attr!r}")
    return values[0]


def _stack(values: Sequence[Any], what: str):
    arrs = [jnp.asarray(v) for v in values]
    shape = arrs[0].shape
    if any(a.shape != shape for a in arrs):
        raise NotStackable(f"{what} has different shapes across experiments")
    return jnp.stack(arrs)


class BatchedExperiment(Experiment):
    """An :class:`Experiment` standing for every experiment of a batch.

    ``inputs`` holds one experiment's values (traced under ``vmap``). The
    other attributes are those the experiments share; reading one they do
    not share raises :class:`NotStackable`, which sends the batch back to
    the one-experiment-at-a-time loop.
    """

    def __init__(self, inputs: dict, experiments: Sequence[Experiment]):
        self.__dict__['inputs'] = inputs
        self.__dict__['_experiments'] = experiments

    def __repr__(self) -> str:
        return f"BatchedExperiment(inputs={sorted(self.inputs)})"

    name = property(lambda self: _common(self._experiments, 'name'))
    metadata = property(lambda self: _common(self._experiments, 'metadata'))
    observed = property(lambda self: _common(self._experiments, 'observed'))
    uncertainties = property(
        lambda self: _common(self._experiments, 'uncertainties'))
    measured = property(lambda self: _common(self._experiments, 'measured'))
    output_names = property(
        lambda self: _common(self._experiments, 'output_names'))
    measured_names = property(
        lambda self: _common(self._experiments, 'measured_names'))


def stack_inputs(experiments: Sequence[Experiment]) -> tuple[dict, dict]:
    """Split the experiments' inputs into stacked arrays and shared values.

    Returns:
        ``(stacked, static)``: ``stacked[k]`` has a leading axis of length
        ``len(experiments)``; ``static[k]`` is the value every experiment
        shares for a non-numeric input.

    Raises:
        NotStackable: the input keys differ, a numeric input changes shape,
            or a non-numeric input changes value.
    """
    if not experiments:
        raise NotStackable("no experiments")
    keys = list(experiments[0].inputs)
    if any(set(e.inputs) != set(keys) for e in experiments[1:]):
        raise NotStackable("experiments have different input keys")
    stacked, static = {}, {}
    for k in keys:
        values = [e.inputs[k] for e in experiments]
        if all(_is_numeric(v) for v in values):
            stacked[k] = _stack(values, f"input {k!r}")
        elif _same(values):
            static[k] = values[0]
        else:
            raise NotStackable(f"input {k!r} is not numeric and differs")
    return stacked, static


def batched_model(model_fn, param_names, experiments, static):
    """``one(theta_array, inputs) -> preds`` for a single (vmapped) row."""
    names = list(param_names)

    def one(theta, inputs):
        theta_dict = {n: theta[i] for i, n in enumerate(names)}
        exp = BatchedExperiment({**static, **inputs}, experiments)
        return model_fn(theta_dict, exp)

    return one


class Batch:
    """Experiments prepared for repeated evaluation of an objective.

    Args:
        model_fn: ``model_fn(theta_dict, experiment) -> dict``.
        experiments: the experiments.
        param_names: parameter names, the order of ``theta``.
        vectorize: try the vectorized path (default). ``False`` forces the
            loop.

    Attributes:
        vectorized: whether the model is evaluated once, ``vmap``-ed over
            the stacked experiments, rather than once per experiment.
        data: the default data pytree for :meth:`objective`.
    """

    def __init__(self, model_fn, experiments, param_names, vectorize=True):
        self.model_fn = model_fn
        self.experiments = list(experiments)
        self.param_names = list(param_names)
        self.n = len(self.experiments)
        self.vectorized = False
        if vectorize and self.n > 0:
            try:
                self._prepare_vectorized()
                self.vectorized = True
            except Exception:  # noqa: BLE001 - any failure means: use the loop
                self.vectorized = False
        self.data = self.data_for()

    # -- set-up -----------------------------------------------------------

    def _prepare_vectorized(self):
        exps = self.experiments
        out_names = exps[0].output_names
        if any(e.output_names != out_names for e in exps[1:]):
            raise NotStackable("experiments observe different outputs")
        self.out_names = tuple(out_names)
        self.inputs, static = stack_inputs(exps)
        one = batched_model(self.model_fn, self.param_names, exps, static)
        keys = self.out_names

        def predict(theta, inputs):
            def row(inp):
                preds = one(theta, inp)
                return {k: jnp.asarray(preds[k]) for k in keys}
            return jax.vmap(row)(inputs)

        self._predict = predict
        # Stack what the objective reads, so a mismatch is found here
        # rather than inside a trace.
        self._stack_observed([e.observed for e in exps])
        self._stack_sigma()
        # One abstract trace decides whether the model vectorizes.
        theta = jax.ShapeDtypeStruct((len(self.param_names),), jnp.float64)
        jax.eval_shape(predict, theta, self.inputs)

    def _stack_observed(self, observed):
        return {k: _stack([o[k] for o in observed], f"observed {k!r}")
                for k in self.out_names}

    def _stack_sigma(self):
        return {k: _stack([_sigma(e, k) for e in self.experiments],
                          f"uncertainty of {k!r}")
                for k in self.out_names}

    # -- data ---------------------------------------------------------------

    def data_for(self, observed=None, count=None):
        """The data pytree for :meth:`objective`.

        Args:
            observed: per-experiment ``{name: value}`` dicts replacing the
                experiments' own observations (a residual resample).
            count: per-experiment multiplicity (a resample with
                replacement, or 0/1 for a cross-validation fold).
        """
        if observed is None:
            observed = [e.observed for e in self.experiments]
        count = (jnp.ones(self.n) if count is None
                 else jnp.asarray(count, dtype=jnp.float64))
        if self.vectorized:
            return {'inputs': self.inputs,
                    'obs': self._stack_observed(observed),
                    'sigma': self._stack_sigma(),
                    'count': count}
        return {
            'obs': [{k: jnp.asarray(o[k]) for k in e.output_names}
                    for e, o in zip(self.experiments, observed)],
            'sigma': [{k: jnp.asarray(_sigma(e, k)) for k in e.output_names}
                      for e in self.experiments],
            'count': count,
        }

    # -- evaluation ---------------------------------------------------------

    def objective(self, kind: str = 'sse') -> Callable:
        """``f(theta, data=self.data) -> scalar`` for 'sse', 'wsse' or 'nll'."""
        term = _TERMS[kind]
        default = self.data

        if self.vectorized:
            predict = self._predict

            def f(theta, data=None):
                data = default if data is None else data
                preds = predict(theta, data['inputs'])
                total = jnp.zeros(())
                for k in self.out_names:
                    p = preds[k]
                    y = _expand(data['obs'][k], p.ndim)
                    s = _expand(data['sigma'][k], p.ndim)
                    t = term(p - y, s)
                    per = jnp.sum(t.reshape(t.shape[0], -1), axis=1)
                    total = total + jnp.dot(data['count'], per)
                return total
            return f

        model_fn, names, exps = self.model_fn, self.param_names, self.experiments

        def f(theta, data=None):
            data = default if data is None else data
            theta_dict = {n: theta[i] for i, n in enumerate(names)}
            total = jnp.zeros(())
            for i, exp in enumerate(exps):
                if not exp.output_names:
                    continue
                preds = model_fn(theta_dict, exp)
                part = 0.0
                for k in exp.output_names:
                    r = preds[k] - data['obs'][i][k]
                    part = part + jnp.sum(term(r, data['sigma'][i][k]))
                total = total + data['count'][i] * part
            return total
        return f

    def predictions(self, theta) -> list[dict[str, np.ndarray]]:
        """Per-experiment predictions of the observed outputs at ``theta``."""
        theta = jnp.asarray(theta, dtype=jnp.float64)
        if self.vectorized:
            stacked = jax.jit(self._predict)(theta, self.inputs)
            return [{k: np.asarray(stacked[k][i]) for k in self.out_names}
                    for i in range(self.n)]
        theta_dict = {n: theta[i] for i, n in enumerate(self.param_names)}
        out = []
        for exp in self.experiments:
            preds = self.model_fn(theta_dict, exp)
            out.append({k: np.asarray(preds[k]) for k in exp.output_names})
        return out


def _sigma(exp, key):
    if exp.uncertainties is not None and key in exp.uncertainties:
        return exp.uncertainties[key]
    return 1.0


def _expand(a, ndim):
    """Append unit axes so a per-experiment value broadcasts against ``ndim``."""
    a = jnp.asarray(a)
    return a.reshape(a.shape + (1,) * max(ndim - a.ndim, 0))
