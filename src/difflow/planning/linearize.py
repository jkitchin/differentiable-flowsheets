"""Delta-vector generation: AD Jacobians as first-order unit submodels.

A delta-base planning model represents each unit as ``y ~= y0 + J (u - u0)``.
Commercial systems build ``J`` by perturbing a rigorous simulator once per
decision variable, which costs ``O(n)`` model evaluations and is why the
vectors are refreshed on the order of annually.  Reverse-mode AD returns the
same matrix for a cost independent of ``n``.

Two things follow, and both are encoded here:

* :func:`choose_ad_mode` picks reverse or forward mode from the block's shape.
  Reverse mode costs ``O(n_y)`` passes and forward mode ``O(n_u)``, so reverse
  wins exactly when outputs are fewer than inputs — the planning regime.
* :func:`check_delta_vectors` verifies the AD Jacobian against central
  differences, because a delta vector that is silently wrong produces an LP
  that is confidently wrong.

The linearisation also records which *phase regime* each block sits in, so the
planner can warn when a proposal moves a block across a phase boundary.  A
delta vector computed across phase appearance or disappearance is meaningless
rather than merely inaccurate: the underlying function is not differentiable
there, and the Taylor model extrapolates a branch that has ceased to exist.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.planning.block import Block


class PhaseBoundaryWarning(UserWarning):
    """A linearisation or proposal straddles a phase boundary.

    The delta vectors either side of a phase appearance/disappearance describe
    different functions, so the LP built from them is not an approximation of
    anything.  Shrink the trust region, re-centre the linearisation inside one
    regime, or split the block into per-regime blocks.
    """


@dataclass
class Linearization:
    """First-order model of one block: the delta vectors and their base.

    Attributes:
        block: Block name.
        u0: Linearisation point (input array).
        y0: Block outputs at ``u0``.
        J: Delta vectors, shape ``(n_y, n_u)`` — ``J[i, j] = dy_i/du_j``.
        mode: AD mode actually used, ``"rev"`` or ``"fwd"``.
        phase: Phase indicator values at ``u0``, or ``None``.
        phase_code: Integer regime code per indicator, or ``None``.
        n_evals: Model-evaluation equivalents charged for this linearisation
            (1 for the base point; the AD pass itself is counted separately by
            the caller when benchmarking).
    """

    block: str
    u0: Array
    y0: Array
    J: Array
    mode: str
    phase: Array | None = None
    phase_code: Array | None = None
    n_evals: int = 1

    def predict(self, u: Array) -> Array:
        """Evaluate the Taylor model at ``u``."""
        u = jnp.atleast_1d(jnp.asarray(u, dtype=float))
        return self.y0 + self.J @ (u - self.u0)

    def as_table(self, u_names: list[str], y_names: list[str]) -> str:
        """Render the delta vectors as a fixed-width table."""
        J = np.asarray(self.J)
        width = max(12, max((len(n) for n in u_names), default=12) + 2)
        head = "".ljust(18) + "".join(n.rjust(width) for n in u_names)
        rows = [f"delta vectors for block {self.block!r}", head]
        for i, yn in enumerate(y_names):
            cells = "".join(f"{J[i, j]:{width}.4g}" for j in range(J.shape[1]))
            rows.append(yn[:17].ljust(18) + cells)
        return "\n".join(rows)


def choose_ad_mode(n_u: int, n_y: int, mode: str = "auto") -> str:
    """Choose reverse or forward mode from the block's shape.

    Reverse mode builds the Jacobian one *output* at a time and forward mode
    one *input* at a time, so the cheaper mode is the one with fewer of them.

    Args:
        n_u: Number of inputs.
        n_y: Number of outputs.
        mode: ``"auto"`` to decide by shape, or ``"rev"``/``"fwd"`` to force.

    Returns:
        ``"rev"`` or ``"fwd"``.

    Example:
        >>> choose_ad_mode(80, 1)      # many decisions, scalar objective
        'rev'
        >>> choose_ad_mode(1, 40)      # one lever, many reported outputs
        'fwd'
    """
    if mode in ("rev", "fwd"):
        return mode
    if mode != "auto":
        raise ValueError(f"unknown AD mode {mode!r}")
    return "rev" if n_y <= n_u else "fwd"


def jacobian_fn(fn: Callable[[Array], Array], n_u: int, n_y: int,
                mode: str = "auto") -> tuple[Callable[[Array], Array], str]:
    """Return a Jacobian callable for ``fn`` and the mode it uses.

    Args:
        fn: Callable ``u -> y``.
        n_u: Length of ``u``.
        n_y: Length of ``y``.
        mode: See :func:`choose_ad_mode`.

    Returns:
        ``(jac_fn, mode)`` where ``jac_fn(u)`` has shape ``(n_y, n_u)``.
    """
    chosen = choose_ad_mode(n_u, n_y, mode)
    return (jax.jacrev(fn) if chosen == "rev" else jax.jacfwd(fn)), chosen


def linearize_block(block: Block, u0: Array | None = None,
                    theta: Mapping[str, Any] | None = None,
                    mode: str | None = None) -> Linearization:
    """Compute the delta vectors for one block at ``u0``.

    Args:
        block: The block to linearise.
        u0: Linearisation point.  Defaults to ``block.u0``.
        theta: Parameter override passed through to the block.
        mode: AD mode override; defaults to ``block.ad_mode``.

    Returns:
        A :class:`Linearization`.

    Example:
        >>> lin = linearize_block(ngl_block)
        >>> lin.J.shape
        (5, 4)
    """
    u0 = block.u0 if u0 is None else jnp.atleast_1d(jnp.asarray(u0, dtype=float))
    chosen = choose_ad_mode(block.n_u, block.n_y,
                            block.ad_mode if mode is None else mode)
    y0 = block.evaluate(u0, theta)
    J = jnp.atleast_2d(block.jacobian(chosen)(u0, theta))
    if J.shape != (block.n_y, block.n_u):
        raise ValueError(
            f"Block {block.name!r} Jacobian has shape {tuple(J.shape)}, "
            f"expected {(block.n_y, block.n_u)}")

    phase = block.evaluate_phases(u0, theta)
    code = None if phase is None else classify_phase(phase, block.phase_bounds)
    return Linearization(block=block.name, u0=u0, y0=y0, J=J, mode=chosen,
                         phase=phase, phase_code=code)


def classify_phase(values: Array, bounds: tuple[float, ...]) -> Array:
    """Bin phase indicators into integer regime codes.

    Args:
        values: Indicator values, e.g. vapour fractions.
        bounds: Interior thresholds separating regimes, ascending.  The
            default block setting ``(0.0, 1.0)`` gives code 0 for a
            subcooled liquid, 1 for two phases and 2 for a superheated
            vapour.

    Returns:
        Integer array of the same shape as ``values``.
    """
    v = np.asarray(values, dtype=float)
    edges = np.asarray(bounds, dtype=float)
    # A value sitting exactly on a threshold is on the boundary itself; treat
    # it as belonging to the lower regime so the code is deterministic. That
    # is the useful convention here: a flash reports V_frac == 0 exactly on
    # its clamped single-phase branch, which is the regime a delta vector
    # must not be extrapolated out of.
    return np.searchsorted(edges, v, side="left").astype(int)


def check_phase_transition(block: Block, lin: Linearization, u_new: Array,
                           theta: Mapping[str, Any] | None = None,
                           warn: bool = True) -> list[str]:
    """Report phase regimes crossed between the linearisation point and ``u_new``.

    Args:
        block: The block.
        lin: Its current linearisation.
        u_new: Proposed operating point.
        theta: Parameter override.
        warn: Emit a :class:`PhaseBoundaryWarning` for each crossing.

    Returns:
        A list of human-readable descriptions, empty when no boundary is
        crossed or the block declares no ``phase_fn``.
    """
    if block.phase_fn is None or lin.phase_code is None:
        return []
    new_phase = block.evaluate_phases(u_new, theta)
    if new_phase is None:
        return []
    new_code = classify_phase(new_phase, block.phase_bounds)
    old = np.atleast_1d(lin.phase_code)
    new = np.atleast_1d(new_code)
    old_v = np.atleast_1d(np.asarray(lin.phase, dtype=float))
    new_v = np.atleast_1d(np.asarray(new_phase, dtype=float))

    names = (list(block.phase_names)
             if len(block.phase_names) == len(new)
             else [f"phase[{i}]" for i in range(len(new))])

    messages = []
    for i, (a, b) in enumerate(zip(old, new)):
        if a == b:
            continue
        msg = (f"block {block.name!r}: indicator {names[i]!r} crossed a phase "
               f"boundary between the linearisation point and the proposal "
               f"({old_v[i]:.4g} -> {new_v[i]:.4g}, regime {int(a)} -> "
               f"{int(b)}). The delta vectors either side describe different "
               f"functions, so this linearisation is not valid at the "
               f"proposal — shrink the trust region or re-centre inside one "
               f"regime.")
        messages.append(msg)
        if warn:
            warnings.warn(msg, PhaseBoundaryWarning, stacklevel=2)
    return messages


def check_delta_vectors(block: Block, u0: Array | None = None,
                        theta: Mapping[str, Any] | None = None,
                        step: float | None = None,
                        rtol: float = 1e-4,
                        raise_on_fail: bool = False) -> dict[str, Any]:
    """Verify AD delta vectors against central differences.

    Args:
        block: The block to check.
        u0: Point at which to check.  Defaults to ``block.u0``.
        theta: Parameter override.
        step: Absolute perturbation.  Defaults to a per-variable step scaled
            by the bound range (or the variable magnitude when unbounded).
        rtol: Relative tolerance on the largest entry of ``J``.
        raise_on_fail: Raise ``AssertionError`` instead of returning
            ``passed=False``.

    Returns:
        Dict with ``J_ad``, ``J_fd``, ``max_abs_error``, ``max_rel_error``,
        ``passed`` and ``step``.

    Note:
        Central differences cost ``2 n_u`` model evaluations; the AD Jacobian
        costs ``O(1)``.  This function is a correctness check, not a
        recommendation — see the benchmark in
        :func:`difflow.planning.benchmark.gradient_cost_ratio`.
    """
    u0 = block.u0 if u0 is None else jnp.atleast_1d(jnp.asarray(u0, dtype=float))
    lin = linearize_block(block, u0, theta)

    span = np.asarray(block.range, dtype=float)
    base = np.abs(np.asarray(u0, dtype=float))
    if step is None:
        h = np.where(np.isfinite(span) & (span > 0), 1e-5 * span,
                     1e-6 * np.maximum(base, 1.0))
    else:
        h = np.full(block.n_u, float(step))

    cols = []
    for j in range(block.n_u):
        e = np.zeros(block.n_u)
        e[j] = h[j]
        y_plus = block.evaluate(jnp.asarray(u0 + e), theta)
        y_minus = block.evaluate(jnp.asarray(u0 - e), theta)
        cols.append((np.asarray(y_plus) - np.asarray(y_minus)) / (2 * h[j]))
    J_fd = np.stack(cols, axis=1)
    J_ad = np.asarray(lin.J)

    err = np.abs(J_ad - J_fd)
    scale = max(float(np.max(np.abs(J_ad))), 1e-12)
    max_rel = float(np.max(err)) / scale
    passed = bool(max_rel <= rtol)

    result = {
        "block": block.name,
        "J_ad": J_ad,
        "J_fd": J_fd,
        "max_abs_error": float(np.max(err)),
        "max_rel_error": max_rel,
        "rtol": rtol,
        "passed": passed,
        "step": h,
        "mode": lin.mode,
    }
    if raise_on_fail and not passed:
        raise AssertionError(
            f"delta vectors for block {block.name!r} disagree with central "
            f"differences: max relative error {max_rel:.3e} > {rtol:.3e}")
    return result
