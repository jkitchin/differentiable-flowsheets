"""When the hard worst case is too conservative to design against.

``psi(d) <= 0`` is a guarantee over every point of the set, including the
corner where every parameter is simultaneously at its extreme in the worst
direction.  That corner is often absurdly improbable: for ten independent
parameters it is one part in a thousand *if* each extreme were a coin flip,
and far less if the extremes are tails of real distributions.  Designing for
it buys a guarantee nobody asked for at a capital cost somebody has to pay.

The stochastic counterpart replaces the guarantee with a probability.  Sample
the parameters, re-optimize the controls for each sample exactly as
flexibility analysis does, and read off

* the **probability of feasibility**, ``P(psi <= 0)``;
* the **chance-constrained margin** --- the ``alpha``-quantile of the
  per-sample value, which is ``<= 0`` exactly when the design meets
  ``P(feasible) >= alpha``;
* the **per-constraint violation frequencies**, which say *which* constraint
  is doing the failing, and how often.

The last is the practical payoff and the reason not to collapse this to a
single number.  A design that fails 4% of the time on one purity spec is a
different problem from one that fails 4% of the time spread over six
constraints.

The sampling is a plain ``vmap`` over the same inner solve the deterministic
path uses, so nothing here is a second modelling stack: it is the same
``min_u max_j f_j`` evaluated at sampled ``theta`` instead of at vertices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.flexibility.feasibility import (
    ModelFn, _n_constraints, _names, inner_value,
)
from difflow.flexibility.inner import DEFAULT_OPTIONS, SolverOptions
from difflow.flexibility.sets import (
    UncertaintySet, as_control_spec, as_uncertainty_set,
)

DISTRIBUTIONS = ("uniform", "normal")


@dataclass
class StochasticFeasibilityResult:
    """Sampled feasibility: a probability instead of a guarantee.

    Attributes:
        probability: Fraction of samples with ``psi_sample <= 0``.
        mean: Mean of the per-sample inner value.
        worst: Largest per-sample value seen.
        values: The per-sample inner values.
        samples: The sampled realizations, ``(n_samples, n_theta)``.
        controls: Re-optimized controls per sample.
        constraint_values: ``(n_samples, n_constraints)`` at those controls.
        violation_rate: Per-constraint fraction of samples with ``f_j > 0``.
        blame: Per-constraint fraction of *infeasible* samples in which that
            constraint was the worst one.
        distribution: ``"uniform"`` or ``"normal"``.
        scale: Set scaling sampled at.
        set: The uncertainty set.
        constraint_names: Names of the constraints.
        control_names: Names of the recourse variables.

    Example:
        >>> res.probability                              # doctest: +SKIP
        0.96
    """

    probability: float
    mean: float
    worst: float
    values: np.ndarray
    samples: np.ndarray
    controls: np.ndarray
    constraint_values: np.ndarray
    violation_rate: np.ndarray
    blame: np.ndarray
    distribution: str
    scale: float
    set: UncertaintySet
    constraint_names: tuple[str, ...] = ()
    control_names: tuple[str, ...] = ()

    @property
    def n_samples(self) -> int:
        """How many realizations were drawn."""
        return int(self.values.size)

    @property
    def standard_error(self) -> float:
        """Standard error of ``probability``, from the binomial variance.

        Reported because the difference between a 0.96 and a 0.98 estimated
        from 200 samples is noise, and a chance constraint asserted on noise
        is worse than no chance constraint.
        """
        p = float(self.probability)
        return float(np.sqrt(max(p * (1.0 - p), 0.0) / max(self.n_samples, 1)))

    def quantile(self, q: float) -> float:
        """The ``q``-quantile of the per-sample value.

        Args:
            q: Probability level in ``[0, 1]``.

        Returns:
            The quantile.
        """
        return float(np.quantile(self.values, q))

    def chance_margin(self, alpha: float = 0.95) -> float:
        """Margin for the chance constraint ``P(feasible) >= alpha``.

        Args:
            alpha: Required reliability.

        Returns:
            The ``alpha``-quantile of the per-sample value.  The chance
            constraint holds exactly when this is ``<= 0``, and its magnitude
            is how much constraint units the design is over or under by.
        """
        return self.quantile(alpha)

    def satisfies(self, alpha: float = 0.95) -> bool:
        """Whether ``P(psi <= 0) >= alpha`` on this sample.

        Args:
            alpha: Required reliability.

        Returns:
            True if the chance constraint holds.
        """
        return bool(self.chance_margin(alpha) <= 0.0)

    def worst_sample(self) -> dict[str, float]:
        """``{parameter: value}`` at the worst realization drawn."""
        i = int(np.argmax(self.values))
        return {n: float(v) for n, v in zip(self.set.names, self.samples[i])}

    def summary(self) -> str:
        """Probability, quantiles, and the per-constraint blame table."""
        lines = [
            f"P(feasible) = {self.probability:.4f} "
            f"+/- {self.standard_error:.4f} "
            f"({self.n_samples} {self.distribution} samples, "
            f"scale {self.scale:g})",
            f"  mean value {self.mean:.5g}, worst {self.worst:.5g}",
            f"  90% margin {self.quantile(0.90):.5g}, "
            f"95% margin {self.quantile(0.95):.5g}, "
            f"99% margin {self.quantile(0.99):.5g}",
            f"  {'constraint':<24s}{'P(violated)':>14s}{'blame':>10s}",
        ]
        for i, nm in enumerate(self.constraint_names):
            lines.append(f"  {nm:<24s}{self.violation_rate[i]:14.4f}"
                         f"{self.blame[i]:10.3f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"StochasticFeasibilityResult(P={self.probability:.4f}, "
                f"n={self.n_samples}, distribution={self.distribution!r})")


def sample_set(uncertainty_set, n_samples: int, key=0, *,
               distribution: str = "uniform", scale: float = 1.0) -> Array:
    """Draw realizations from a scaled uncertainty set.

    Args:
        uncertainty_set: The set.
        n_samples: How many to draw.
        key: A ``jax.random`` key or an integer seed.
        distribution: ``"uniform"`` over the box, or ``"normal"``, a two-piece
            normal whose one-sigma half-widths are the box deviations, so an
            asymmetric envelope stays asymmetric and roughly 68% of draws land
            inside the ``scale = 1`` box.
        scale: Set scaling.

    Returns:
        A ``(n_samples, n_theta)`` array.

    Raises:
        ValueError: For an unknown ``distribution``.
    """
    T = as_uncertainty_set(uncertainty_set)
    if distribution not in DISTRIBUTIONS:
        raise ValueError(f"distribution must be one of {DISTRIBUTIONS}, "
                         f"got {distribution!r}")
    k = key if hasattr(key, "shape") else jax.random.PRNGKey(int(key))
    shape = (int(n_samples), T.n)
    s = jnp.asarray(scale, dtype=float)
    if distribution == "uniform":
        lo, hi = T.bounds(s)
        return jax.random.uniform(k, shape, minval=lo, maxval=hi)
    z = jax.random.normal(k, shape)
    dev = jnp.where(z >= 0, T.upper[None, :], T.lower[None, :])
    return T.nominal[None, :] + s * z * dev


def expected_feasibility(model_fn: ModelFn, d, uncertainty_set, controls=None,
                         *, n_samples: int = 256, key=0,
                         distribution: str = "uniform", scale: float = 1.0,
                         options: SolverOptions = DEFAULT_OPTIONS,
                         constraint_names: Sequence[str] | None = None,
                         ) -> StochasticFeasibilityResult:
    """Sampled feasibility with recourse: the chance-constrained counterpart.

    Each sampled realization gets its own re-optimized controls, exactly as in
    :func:`~difflow.flexibility.feasibility.feasibility_function`; only the
    outer ``max`` over the set is replaced by an expectation and a quantile.

    Args:
        model_fn: ``f(d, u, theta) -> array``, feasible where every entry is
            ``<= 0``.
        d: The design being tested.
        uncertainty_set: An :class:`~difflow.flexibility.sets.UncertaintySet`
            or a ``{name: (nominal, pm)}`` mapping.
        controls: The recourse variables, or ``None`` for none.
        n_samples: Number of realizations.
        key: A ``jax.random`` key or an integer seed.
        distribution: ``"uniform"`` or ``"normal"``; see :func:`sample_set`.
        scale: Set scaling to sample at.
        options: Search settings.
        constraint_names: Names for the rows of ``f``.

    Returns:
        A :class:`StochasticFeasibilityResult`.

    Note:
        This is a Monte Carlo estimate, so it is exposed with its standard
        error rather than as a bare probability, and it says nothing about the
        tail beyond the samples drawn.  Use it to relax an over-conservative
        worst case, not to certify a rare event.

    Example:
        >>> import jax.numpy as jnp
        >>> from difflow.flexibility import expected_feasibility
        >>> f = lambda d, u, th: jnp.array([u[0] - d[0], th[0] - u[0]])
        >>> res = expected_feasibility(
        ...     f, [1.0], {"feed": (1.0, 0.5)}, {"u": (-10.0, 10.0)},
        ...     n_samples=64)
        >>> 0.3 < res.probability < 0.7      # nominal sits on the boundary
        True
    """
    T = as_uncertainty_set(uncertainty_set)
    cs = as_control_spec(controls)
    d = jnp.asarray(d, dtype=float)
    n_f = _n_constraints(model_fn, d, cs, T)
    c_names = _names(constraint_names, n_f, "f")

    thetas = sample_set(T, n_samples, key, distribution=distribution,
                        scale=scale)

    def one(theta):
        val, u = inner_value(model_fn, d, theta, cs, options)
        return val, u, jnp.atleast_1d(model_fn(d, u, theta))

    vals, us, fs = jax.vmap(one)(thetas)
    vals = np.asarray(vals, dtype=float)
    fs = np.asarray(fs, dtype=float)
    bad = vals > 0.0
    blame = np.zeros(n_f)
    if bad.any():
        worst_j = np.argmax(fs[bad], axis=1)
        for j in worst_j:
            blame[j] += 1.0
        blame /= float(bad.sum())
    return StochasticFeasibilityResult(
        probability=float(np.mean(~bad)), mean=float(np.mean(vals)),
        worst=float(np.max(vals)), values=vals,
        samples=np.asarray(thetas, dtype=float),
        controls=np.asarray(us, dtype=float), constraint_values=fs,
        violation_rate=np.mean(fs > 0.0, axis=0), blame=blame,
        distribution=distribution, scale=float(scale), set=T,
        constraint_names=c_names, control_names=tuple(cs.names))
