"""The sample: a finite set of realizations standing in for a distribution.

Sample average approximation replaces an expectation nobody can evaluate,

.. math:: \\min_x\\ \\mathbb{E}_\\theta\\,[\\,F(x, \\theta)\\,]

with an average over ``S`` draws that anybody can,

.. math:: \\min_x\\ \\sum_{s} w_s\\, F(x, \\theta_s) .

Everything downstream is then an ordinary deterministic optimization, which is
the entire trick.  What it costs is stated honestly here rather than hidden:
the answer depends on the draw, so a :class:`ScenarioSet` carries the seed
that produced it and the machinery to redraw it.

Common random numbers, and why the seed is part of the object
-------------------------------------------------------------
An SAA objective evaluated on a *fresh* sample at every outer iteration is not
a function --- it is a function plus noise of the same order as the
improvements the optimizer is chasing, and a descent method on it stalls at a
distance from the solution set by the noise, not by the tolerance.  Fixing the
sample makes the SAA objective a genuine deterministic function of ``x``, with
an exact gradient, and that is the only form the solvers here will accept.
So a :class:`ScenarioSet` is drawn once, held, and reused; there is no
resample-per-iteration option, because there is no correct way to use one.

Redrawing is for *assessing* the answer, not for computing it --- see
:func:`~difflow.stochastic.diagnostics.optimality_gap`, which redraws
deliberately and independently, and reports a confidence bound rather than a
better point.

Where the numbers come from
---------------------------
The constructors are ordered by how much you actually know:

``ScenarioSet.from_samples``
    You have the draws already --- a posterior from an MCMC run, historical
    campaign data, a vendor's lot-to-lot record.  Nothing is assumed.
``ScenarioSet.from_covariance``
    You have a fit.  :func:`difflow.estimation.predicted_covariance` and
    :func:`difflow.reconciliation.reconciled_covariance` both hand back
    exactly the ``Sigma`` this wants, so the distribution the plant is designed
    against is the one the data actually supports, correlations included.
``ScenarioSet.normal`` / ``.lognormal`` / ``.uniform``
    You have a nominal value and a spread, and the parameters are independent.
``ScenarioSet.from_uncertainty_set``
    You have a :class:`~difflow.flexibility.sets.UncertaintySet` written for a
    worst-case study and want the stochastic counterpart of the same envelope.

Correlation is not a refinement here.  Two distribution coefficients fitted to
the same isotherm are strongly correlated, and the difference between their
logarithms --- which is the separation factor, and the only thing the plant
cares about --- has a variance that a diagonal ``Sigma`` gets wrong by a
factor of a few in whichever direction is least convenient.  Prefer
:meth:`from_covariance` whenever a fit exists.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.params_mixin import ParamsMixin

#: Distribution families the keyword constructors understand.
DISTRIBUTIONS = ("normal", "lognormal", "uniform")


def _as_key(key) -> Array:
    """A PRNG key from a key or an integer seed."""
    return key if hasattr(key, "dtype") and hasattr(key, "shape") else \
        jax.random.PRNGKey(int(key))


def _split_spec(spec: Mapping[str, tuple]) -> tuple[tuple[str, ...],
                                                    np.ndarray, np.ndarray]:
    """``{name: (center, spread)}`` -> names, centers, spreads."""
    if not spec:
        raise ValueError("An empty parameter spec has nothing to sample.")
    names, centers, spreads = [], [], []
    for name, pair in spec.items():
        try:
            center, spread = pair
        except (TypeError, ValueError):
            raise ValueError(
                f"Parameter {name!r} must be given as (center, spread), got "
                f"{pair!r}."
            ) from None
        names.append(str(name))
        centers.append(float(center))
        spreads.append(float(spread))
    s = np.asarray(spreads, dtype=float)
    if np.any(s < 0):
        bad = [n for n, v in zip(names, s) if v < 0]
        raise ValueError(f"Spreads must be non-negative; {bad} are negative.")
    return tuple(names), np.asarray(centers, dtype=float), s


@dataclass
class ScenarioSet(ParamsMixin):
    """A finite, weighted sample of the uncertain parameters.

    Attributes:
        draws: ``(n_scenarios, n_parameters)`` realizations.
        names: Parameter names, in column order.
        weights: Scenario probabilities, summing to one.  Equal by default;
            unequal weights carry a reduced or quadrature sample.
        seed: The integer seed the sample was drawn with, or ``None`` for a
            sample supplied directly.  Recorded so a run is reproducible and
            so :func:`~difflow.stochastic.diagnostics.optimality_gap` can draw
            an *independent* one.
        distribution: The family the sample came from, for reports.

    Example:
        >>> scen = ScenarioSet.normal({"logD_Nd": (1.2, 0.15)}, n=64, seed=0)
        >>> scen.n_scenarios, scen.n_parameters
        (64, 1)
        >>> scen.mean_scenario().draws.shape
        (1, 1)
    """

    draws: Array
    names: tuple[str, ...]
    weights: Array | None = None
    seed: int | None = None
    distribution: str = "custom"

    def __post_init__(self):
        v = jnp.atleast_2d(jnp.asarray(self.draws, dtype=float))
        if v.ndim != 2:
            raise ValueError(
                f"draws must be (n_scenarios, n_parameters), got shape "
                f"{v.shape}."
            )
        self.draws = v
        self.names = tuple(str(n) for n in self.names)
        if len(self.names) != v.shape[1]:
            raise ValueError(
                f"{len(self.names)} names for {v.shape[1]} parameter columns."
            )
        if self.weights is None:
            self.weights = jnp.full((v.shape[0],), 1.0 / v.shape[0])
        else:
            w = jnp.asarray(self.weights, dtype=float).reshape(-1)
            if w.shape[0] != v.shape[0]:
                raise ValueError(
                    f"{w.shape[0]} weights for {v.shape[0]} scenarios."
                )
            if bool(jnp.any(w < 0)):
                raise ValueError("Scenario weights must be non-negative.")
            total = float(jnp.sum(w))
            if total <= 0:
                raise ValueError("Scenario weights sum to zero.")
            self.weights = w / total

    # -- shape ----------------------------------------------------------

    @property
    def n_scenarios(self) -> int:
        """How many realizations."""
        return int(self.draws.shape[0])

    @property
    def n_parameters(self) -> int:
        """How many uncertain parameters."""
        return int(self.draws.shape[1])

    def index(self, name: str) -> int:
        """Column index of a parameter.

        Args:
            name: Parameter name.

        Returns:
            Its column index.

        Raises:
            KeyError: If the name is not one of :attr:`names`.
        """
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(
                f"No parameter {name!r}; this set carries {list(self.names)}."
            ) from None

    def column(self, name: str) -> Array:
        """All realizations of one parameter.

        Args:
            name: Parameter name.

        Returns:
            A ``(n_scenarios,)`` array.
        """
        return self.draws[:, self.index(name)]

    def as_dict(self, theta: Array) -> dict[str, Array]:
        """One realization as ``{name: value}``.

        Args:
            theta: A ``(n_parameters,)`` row, typically the argument a model
                function is handed.

        Returns:
            The row keyed by parameter name.  Traceable: the values are
            whatever was passed in, tracers included.
        """
        t = jnp.atleast_1d(theta)
        return {n: t[i] for i, n in enumerate(self.names)}

    # -- derived sets ---------------------------------------------------

    def mean_scenario(self) -> "ScenarioSet":
        """The one-scenario set at the sample mean.

        This is the *expected value problem* of stochastic programming: solve
        it, and you have the deterministic design that ignoring uncertainty
        would have produced.  It is the reference the value of the stochastic
        solution is measured against, not a shortcut --- see
        :func:`~difflow.stochastic.diagnostics.value_of_stochastic_solution`.

        Returns:
            A :class:`ScenarioSet` with a single row.
        """
        mean = jnp.sum(self.weights[:, None] * self.draws, axis=0)
        return replace(self, draws=mean[None, :],
                       weights=jnp.ones((1,)), distribution="mean")

    def subset(self, idx: Sequence[int]) -> "ScenarioSet":
        """A re-weighted sub-sample.

        Args:
            idx: Scenario indices to keep.

        Returns:
            A :class:`ScenarioSet` over those rows, weights renormalized.
        """
        i = np.asarray(idx, dtype=int)
        return replace(self, draws=self.draws[i], weights=self.weights[i])

    def redraw(self, seed: int) -> "ScenarioSet":
        """An independent sample from the same distribution.

        Only meaningful for a set built by one of the keyword constructors,
        which recorded what they drew from.  Use it to *assess* a solution,
        never to compute one; see the module docstring.

        Args:
            seed: A different integer seed.

        Returns:
            A fresh :class:`ScenarioSet`.

        Raises:
            ValueError: If this set did not come from a known family.
        """
        spec = getattr(self, "_spec", None)
        if spec is None:
            raise ValueError(
                "This ScenarioSet was supplied directly (or reduced), so it "
                "carries no distribution to redraw from. Rebuild it with "
                "ScenarioSet.normal / .lognormal / .uniform / "
                ".from_covariance, or draw the replication yourself."
            )
        kind, payload = spec
        out = getattr(ScenarioSet, kind)(*payload["args"], seed=seed,
                                         **payload["kwargs"])
        return out

    # -- constructors ---------------------------------------------------

    @classmethod
    def from_samples(cls, draws, names: Sequence[str], *,
                     weights=None) -> "ScenarioSet":
        """Wrap draws you already have.

        Args:
            draws: ``(n_scenarios, n_parameters)`` realizations.
            names: Parameter names, in column order.
            weights: Optional scenario probabilities.

        Returns:
            A :class:`ScenarioSet`.

        Example:
            >>> import numpy as np
            >>> posterior = np.array([[1.1, 0.4], [1.3, 0.5], [1.2, 0.45]])
            >>> ScenarioSet.from_samples(posterior, ["a", "b"]).n_scenarios
            3
        """
        return cls(draws=jnp.asarray(draws, dtype=float),
                   names=tuple(names), weights=weights,
                   distribution="empirical")

    @classmethod
    def from_covariance(cls, names: Sequence[str], mean, covariance, *,
                        n: int = 256, seed: int = 0,
                        log: bool = False) -> "ScenarioSet":
        """Draw from a multivariate normal --- the constructor to prefer.

        ``mean`` and ``covariance`` are what a parameter fit hands back, so
        this is the path from data to design with nothing invented in
        between: :func:`difflow.estimation.predicted_covariance` and
        :func:`difflow.reconciliation.reconciled_covariance` produce exactly
        this pair.

        Args:
            names: Parameter names, in the order of ``mean``.
            mean: ``(n_parameters,)`` central values.
            covariance: ``(n_parameters, n_parameters)`` covariance.  Need
                only be positive *semi*-definite; the factorization is an
                eigendecomposition with negative eigenvalues clipped, so a
                rank-deficient fit (two parameters the data cannot separate)
                samples along the directions it does constrain instead of
                raising.
            n: Number of scenarios.
            seed: PRNG seed.
            log: Treat the drawn values as logarithms and exponentiate, i.e.
                sample lognormally with ``mean``/``covariance`` describing the
                logarithm.  A distribution coefficient is positive and its
                *logarithm* is what correlations are fitted in, so this is
                usually the right flag for one.

        Returns:
            A :class:`ScenarioSet`.
        """
        mu = np.atleast_1d(np.asarray(mean, dtype=float))
        cov = np.atleast_2d(np.asarray(covariance, dtype=float))
        k = mu.size
        if cov.shape != (k, k):
            raise ValueError(
                f"covariance is {cov.shape}, expected ({k}, {k}) to match "
                f"mean."
            )
        if len(names) != k:
            raise ValueError(f"{len(names)} names for {k} parameters.")
        sym = 0.5 * (cov + cov.T)
        w, V = np.linalg.eigh(sym)
        L = V @ np.diag(np.sqrt(np.clip(w, 0.0, None)))
        z = np.asarray(jax.random.normal(_as_key(seed), (int(n), k)))
        draws = mu[None, :] + z @ L.T
        if log:
            draws = np.exp(draws)
        out = cls(draws=jnp.asarray(draws), names=tuple(names), seed=int(seed),
                  distribution="lognormal" if log else "normal")
        out._spec = ("from_covariance",
                     {"args": (tuple(names), mu, cov),
                      "kwargs": {"n": int(n), "log": bool(log)}})
        return out

    @classmethod
    def normal(cls, spec: Mapping[str, tuple], *, n: int = 256,
               seed: int = 0) -> "ScenarioSet":
        """Independent normals from ``{name: (mean, sigma)}``.

        Args:
            spec: Per-parameter mean and standard deviation.
            n: Number of scenarios.
            seed: PRNG seed.

        Returns:
            A :class:`ScenarioSet`.
        """
        names, mu, sd = _split_spec(spec)
        z = np.asarray(jax.random.normal(_as_key(seed), (int(n), mu.size)))
        out = cls(draws=jnp.asarray(mu[None, :] + sd[None, :] * z),
                  names=names, seed=int(seed), distribution="normal")
        out._spec = ("normal", {"args": (dict(spec),), "kwargs": {"n": int(n)}})
        return out

    @classmethod
    def lognormal(cls, spec: Mapping[str, tuple], *, n: int = 256,
                  seed: int = 0) -> "ScenarioSet":
        """Independent lognormals from ``{name: (median, sigma_log)}``.

        The spread is the standard deviation of the *logarithm*, so
        ``sigma_log = 0.1`` is roughly a 10% relative spread and the draws stay
        positive --- the right shape for a rate constant, a mass-transfer
        coefficient or a distribution ratio.

        Args:
            spec: Per-parameter median and log standard deviation.
            n: Number of scenarios.
            seed: PRNG seed.

        Returns:
            A :class:`ScenarioSet`.

        Raises:
            ValueError: If a median is not positive.
        """
        names, median, sd = _split_spec(spec)
        if np.any(median <= 0):
            bad = [n_ for n_, v in zip(names, median) if v <= 0]
            raise ValueError(
                f"A lognormal median must be positive; {bad} are not. To "
                f"sample a quantity that can be negative use "
                f"ScenarioSet.normal."
            )
        z = np.asarray(jax.random.normal(_as_key(seed), (int(n), median.size)))
        draws = median[None, :] * np.exp(sd[None, :] * z)
        out = cls(draws=jnp.asarray(draws), names=names, seed=int(seed),
                  distribution="lognormal")
        out._spec = ("lognormal",
                     {"args": (dict(spec),), "kwargs": {"n": int(n)}})
        return out

    @classmethod
    def uniform(cls, spec: Mapping[str, tuple], *, n: int = 256,
                seed: int = 0) -> "ScenarioSet":
        """Independent uniforms from ``{name: (low, high)}``.

        Args:
            spec: Per-parameter interval.  Given as ``(low, high)``, not as a
                centre and a half-width.
            n: Number of scenarios.
            seed: PRNG seed.

        Returns:
            A :class:`ScenarioSet`.

        Raises:
            ValueError: If any ``high < low``.
        """
        names, lo, hi = _split_spec({k: (v[0], v[1]) for k, v in spec.items()})
        if np.any(hi < lo):
            bad = [n_ for n_, a, b in zip(names, lo, hi) if b < a]
            raise ValueError(f"uniform needs (low, high); {bad} are reversed.")
        u = np.asarray(jax.random.uniform(_as_key(seed), (int(n), lo.size)))
        out = cls(draws=jnp.asarray(lo[None, :] + (hi - lo)[None, :] * u),
                  names=names, seed=int(seed), distribution="uniform")
        out._spec = ("uniform",
                     {"args": (dict(spec),), "kwargs": {"n": int(n)}})
        return out

    @classmethod
    def from_uncertainty_set(cls, uncertainty_set, *, n: int = 256,
                             seed: int = 0, distribution: str = "uniform",
                             scale: float = 1.0) -> "ScenarioSet":
        """The stochastic counterpart of a flexibility envelope.

        Bridges :mod:`difflow.flexibility`, so a box written for a worst-case
        study can be reused here without restating it.  The sampling is
        :func:`difflow.flexibility.stochastic.sample_set`, including its
        two-piece normal, which keeps an asymmetric envelope asymmetric.

        Args:
            uncertainty_set: An
                :class:`~difflow.flexibility.sets.UncertaintySet` or a
                ``{name: (nominal, pm)}`` mapping.
            n: Number of scenarios.
            seed: PRNG seed.
            distribution: ``"uniform"`` or ``"normal"``.
            scale: Set scaling to sample at.

        Returns:
            A :class:`ScenarioSet`.
        """
        from difflow.flexibility.sets import as_uncertainty_set
        from difflow.flexibility.stochastic import sample_set

        T = as_uncertainty_set(uncertainty_set)
        draws = sample_set(T, int(n), seed, distribution=distribution,
                           scale=scale)
        names = T.names or tuple(f"theta{i}" for i in range(T.n))
        out = cls(draws=draws, names=tuple(names), seed=int(seed),
                  distribution=distribution)
        out._spec = ("from_uncertainty_set",
                     {"args": (T,),
                      "kwargs": {"n": int(n), "distribution": distribution,
                                 "scale": float(scale)}})
        return out

    # -- reporting ------------------------------------------------------

    def summary(self) -> str:
        """A per-parameter table of the sample's own statistics."""
        v = np.asarray(self.draws)
        w = np.asarray(self.weights)
        mean = w @ v
        sd = np.sqrt(np.clip(w @ (v - mean) ** 2, 0.0, None))
        lines = [
            f"ScenarioSet: {self.n_scenarios} scenarios x "
            f"{self.n_parameters} parameters ({self.distribution}"
            + (f", seed {self.seed}" if self.seed is not None else "") + ")",
            f"  {'parameter':<22s}{'mean':>12s}{'sd':>12s}"
            f"{'min':>12s}{'max':>12s}",
        ]
        for i, name in enumerate(self.names):
            lines.append(f"  {name:<22s}{mean[i]:12.5g}{sd[i]:12.5g}"
                         f"{v[:, i].min():12.5g}{v[:, i].max():12.5g}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"ScenarioSet({self.n_scenarios} x {self.n_parameters}, "
                f"names={list(self.names)}, {self.distribution})")
