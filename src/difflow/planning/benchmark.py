"""Timing utilities for the AD-versus-perturbation claim.

The argument for generating delta vectors by AD rather than by one-at-a-time
perturbation is asymptotic, so it has to be measured rather than asserted.
The quantity that matters is the *cost of a gradient in units of model
evaluations*:

* reverse-mode AD costs a small constant number of evaluations, independent of
  the number of decisions ``n``;
* central differences cost exactly ``2n``.

So the ratio between them grows linearly in ``n``, and planning — horizon x
units x decisions — is where ``n`` gets large.

:func:`gradient_cost_ratio` measures both on the same callable, and
:func:`scaling_study` sweeps ``n``.  Both JIT-compile and warm up before
timing, so what is reported is steady-state execution rather than compilation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


def _block_until_ready(x):
    """Force completion of JAX's asynchronous dispatch."""
    return jax.block_until_ready(x)


def _calibrate(sample: Callable[[int], float], min_time: float,
               max_inner: int) -> tuple[int, float]:
    """Find how many calls fill ``min_time``, returning ``(inner, elapsed)``."""
    inner = 1
    elapsed = sample(inner)
    while elapsed < min_time and inner < max_inner:
        inner = min(max_inner,
                    max(inner * 2,
                        int(inner * min_time / max(elapsed, 1e-9)) + 1))
        elapsed = sample(inner)
    return inner, elapsed


def _sampler(fn: Callable[..., Any], args: tuple) -> Callable[[int], float]:
    """A timing sampler that runs ``fn(*args)`` ``inner`` times."""

    def sample(inner: int) -> float:
        t0 = time.perf_counter()
        for _ in range(inner):
            _block_until_ready(fn(*args))
        return time.perf_counter() - t0

    return sample


def paired_time(first: Callable[..., Any], second: Callable[..., Any],
                args: tuple, repeats: int = 5, warmup: int = 2,
                min_time: float = 0.05,
                max_inner: int = 1 << 20) -> tuple[float, float]:
    """Time two callables with their samples interleaved.

    The reported quantity here is a *ratio* of two timings, so anything that
    slows the machine during one measurement and not the other corrupts it
    directly. Alternating the samples means a noisy interval lands on both,
    and taking the minimum of each then cancels most of it.

    Args:
        first: First callable (the model evaluation).
        second: Second callable (the gradient).
        args: Positional arguments for both.
        repeats: Timed samples of each.
        warmup: Untimed calls first.
        min_time: Target duration of one timed sample.
        max_inner: Cap on calls per sample.

    Returns:
        ``(seconds per call of first, seconds per call of second)``.
    """
    for fn in (first, second):
        for _ in range(max(1, warmup)):
            _block_until_ready(fn(*args))

    s1, s2 = _sampler(first, args), _sampler(second, args)
    n1, e1 = _calibrate(s1, min_time, max_inner)
    n2, e2 = _calibrate(s2, min_time, max_inner)
    best1, best2 = e1 / n1, e2 / n2
    for _ in range(max(1, repeats) - 1):
        best1 = min(best1, s1(n1) / n1)
        best2 = min(best2, s2(n2) / n2)
    return best1, best2


def time_callable(fn: Callable[..., Any], args: tuple, repeats: int = 5,
                  warmup: int = 2, min_time: float = 0.05,
                  max_inner: int = 1 << 20) -> float:
    """Best-of-``repeats`` wall time for one call, after warmup.

    Each timed sample runs the call however many times it takes to fill
    ``min_time``, then divides. A JIT-compiled flowsheet can evaluate in a
    couple of hundred microseconds, and timing a single such call measures
    clock resolution and dispatch jitter as much as it measures the model —
    which is fatal here, because the reported number is a *ratio* of two such
    measurements. The minimum over repeats is used rather than the mean: it is
    the estimate least polluted by other load on the machine.

    Args:
        fn: Callable to time.
        args: Positional arguments.
        repeats: Timed samples.
        warmup: Untimed calls first (JIT compilation happens here).
        min_time: Target duration of one timed sample, in seconds.
        max_inner: Cap on calls per sample.

    Returns:
        Seconds for one call.
    """
    for _ in range(max(1, warmup)):
        _block_until_ready(fn(*args))
    sample = _sampler(fn, args)
    inner, elapsed = _calibrate(sample, min_time, max_inner)
    best = elapsed / inner
    for _ in range(max(1, repeats) - 1):
        best = min(best, sample(inner) / inner)
    return best


def central_difference_gradient(fn: Callable[[Array], Array], x: Array,
                                step: float = 1e-6) -> Array:
    """Gradient of a scalar ``fn`` by central differences (``2n`` evaluations).

    Provided for the comparison, not for use: prefer ``jax.grad``.
    """
    x = jnp.asarray(x, dtype=float)
    n = int(x.shape[0])
    out = np.empty(n)
    for i in range(n):
        e = jnp.zeros(n).at[i].set(step)
        out[i] = float((fn(x + e) - fn(x - e)) / (2 * step))
    return jnp.asarray(out)


@dataclass
class CostRatio:
    """Measured cost of a gradient relative to one model evaluation.

    Attributes:
        n: Number of decision variables.
        eval_seconds: One model evaluation.
        ad_seconds: One AD gradient.
        fd_seconds: One central-difference gradient.
        ad_ratio: ``ad_seconds / eval_seconds`` — expected to be O(1). It can
            dip below 1 on a compiled model: XLA prunes parts of the forward
            program that the gradient does not need, so the two are not
            strictly nested.
        fd_ratio: ``fd_seconds / eval_seconds`` — expected to track ``2n``.
        speedup: ``fd_seconds / ad_seconds``.
        mode: AD mode used.
        max_abs_error: Largest disagreement between the two gradients, when
            ``check`` was requested.
    """

    n: int
    eval_seconds: float
    ad_seconds: float
    fd_seconds: float
    ad_ratio: float
    fd_ratio: float
    speedup: float
    mode: str = "rev"
    max_abs_error: float | None = None

    def __repr__(self) -> str:
        return (f"CostRatio(n={self.n}, ad/eval={self.ad_ratio:.2f}x, "
                f"fd/eval={self.fd_ratio:.1f}x, speedup={self.speedup:.1f}x)")


def gradient_cost_ratio(fn: Callable[[Array], Array], x0: Array,
                        mode: str = "rev", repeats: int = 5,
                        warmup: int = 2, jit: bool = True,
                        check: bool = False, step: float = 1e-6) -> CostRatio:
    """Measure AD and finite-difference gradient cost against one evaluation.

    Args:
        fn: Scalar-valued callable of a 1-D array.
        x0: Point to measure at; its length is ``n``.
        mode: ``"rev"`` (the mode that delivers the scaling) or ``"fwd"``.
        repeats: Timed repetitions per measurement.
        warmup: Untimed calls first.
        jit: JIT-compile the evaluation and the AD gradient.  Finite
            differences are timed against the same compiled evaluation, so the
            comparison is like-for-like.
        check: Also compare the two gradients numerically.
        step: Finite-difference step.

    Returns:
        A :class:`CostRatio`.

    Example:
        >>> r = gradient_cost_ratio(objective, jnp.zeros(80))
        >>> r.ad_ratio < 3.0
        True
    """
    x0 = jnp.atleast_1d(jnp.asarray(x0, dtype=float))
    n = int(x0.shape[0])
    if mode not in ("rev", "fwd"):
        raise ValueError(f"mode must be 'rev' or 'fwd', got {mode!r}")

    f = jax.jit(fn) if jit else fn
    grad_raw = jax.grad(fn) if mode == "rev" else jax.jacfwd(fn)
    g = jax.jit(grad_raw) if jit else grad_raw

    # Interleaved, because the headline number is the ratio of these two.
    eval_s, ad_s = paired_time(f, g, (x0,), repeats=repeats, warmup=warmup)
    fd_s = time_callable(
        lambda x: central_difference_gradient(f, x, step), (x0,),
        max(1, repeats // 2), warmup=1)


    err = None
    if check:
        err = float(jnp.max(jnp.abs(
            g(x0) - central_difference_gradient(f, x0, step))))

    return CostRatio(
        n=n, eval_seconds=eval_s, ad_seconds=ad_s, fd_seconds=fd_s,
        ad_ratio=ad_s / eval_s, fd_ratio=fd_s / eval_s,
        speedup=fd_s / ad_s, mode=mode, max_abs_error=err)


def scaling_study(make_problem: Callable[[int], tuple[Callable, Array]],
                  sizes: Sequence[int], **kwargs) -> list[CostRatio]:
    """Sweep problem size and measure the gradient cost ratio at each.

    Args:
        make_problem: ``n -> (fn, x0)`` building a scalar objective with ``n``
            decisions.
        sizes: The ``n`` values to measure.
        **kwargs: Passed to :func:`gradient_cost_ratio`.

    Returns:
        One :class:`CostRatio` per size.
    """
    rows = []
    for n in sizes:
        fn, x0 = make_problem(n)
        rows.append(gradient_cost_ratio(fn, x0, **kwargs))
    return rows


def format_scaling_table(rows: Sequence[CostRatio]) -> str:
    """Render a scaling study as a Markdown table."""
    out = ["| n | one eval | AD gradient | FD gradient | AD / eval | "
           "FD / eval | speedup |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(
            f"| {r.n} | {r.eval_seconds:.3g} s | {r.ad_seconds:.3g} s | "
            f"{r.fd_seconds:.3g} s | {r.ad_ratio:.1f}x | {r.fd_ratio:.0f}x | "
            f"{r.speedup:.0f}x |")
    return "\n".join(out)


def planner_objective(planner) -> Callable[[Array], Array]:
    """The planner's priced objective as a scalar callable of the decisions.

    This is the function whose gradient a planner needs, and whose cost the
    scaling study measures.  It runs every block — including their internal
    flash, recycle and unit solves — so its AD gradient is the reduced
    sensitivity of the whole chain.

    Args:
        planner: A :class:`~difflow.planning.planner.DeltaBasePlanner`.

    Returns:
        ``u -> objective``, JAX-traceable.
    """
    net = planner.evaluation_network
    prices = dict(planner.prices)
    theta = planner.theta

    def objective(u):
        values = net.evaluate(u, theta).values
        total = 0.0
        for v, p in prices.items():
            total = total + float(p) * values[v]
        return total

    return objective
