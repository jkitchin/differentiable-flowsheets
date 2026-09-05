"""Sparsity patterns for the flat NLP view: detected globally, never probed.

An interior-point solver needs the structure of the constraint Jacobian and of
the Lagrangian Hessian before it evaluates either one. There are three ways to
get it, and only one of them scales:

1. **Probing.** Evaluate the derivative at random points and keep whatever came
   out nonzero. This is what ``pounce.jax.from_jax`` does when no pattern is
   supplied, and it is wrong for every process model: the probe points are
   :math:`\\mathcal{N}(0, 1)`, so the model is evaluated at ``T = -1.3 K``,
   Arrhenius terms overflow, and ``nan > eps`` is ``False`` -- so a whole
   column of real derivatives is recorded as structurally zero and silently
   dropped. Nothing here ever takes this path.
2. **Topology.** A unit's residual rows can only touch the stream variables of
   its own inlets and outlets plus the decisions written into that unit. That
   argument makes :func:`difflow.solvers.nlp.as_nlp`'s ``sparsity="structural"``
   pattern a superset *by construction*, at any point, with no evaluation at
   all. It is a fine safety net, but it is coarse: it knows nothing about the
   objective or about a user-supplied spec body, so both come out dense, and a
   dense objective block makes the Lagrangian Hessian dense whatever the
   constraint rows do.
3. **Global graph analysis** -- :mod:`asdex`, which walks the jaxpr and
   propagates index sets through it. The result is valid for all inputs,
   because no derivative is ever evaluated, and it is tight: on difflow's
   equation-oriented residuals it returns exactly the entries that are nonzero
   at a feasible point. This is the default.

The difference is not cosmetic. On a chain of ``N`` reactor + heater stages:

======  ==========  ===========  ============  ===========  ==========
``N``   ``n``       dense Jac    topology Jac  dense Hess   asdex Hess
======  ==========  ===========  ============  ===========  ==========
1       9           81           53            45           7
4       36          1 188        257           666          28
16      144         18 576       1 073         10 440       112
======  ==========  ===========  ============  ===========  ==========

The topology Hessian column is missing from that table because it *is* the
dense column -- 45, 666, 10 440 -- and it grows as :math:`n^2` while the true
structure grows as :math:`n`. A dense Hessian pattern costs ``n`` colors, so
every Hessian evaluation costs ``n`` reverse-mode passes through the whole
flowsheet, and the solver's linear algebra loses the block structure that made
the equation-oriented form worth writing. That is the failure this module
exists to prevent, which is why **dense is never a default**: it is what you
get from ``sparsity="dense"``, chosen deliberately, and nothing falls back to
it on its own.

The contract on any pattern is that it must be a **superset** of the true
structure. An extra entry merely reports a zero and may cost one more color; a
missing entry is silently wrong -- dropped on the dense path, and aliased into
a same-colored entry under sparse AD, corrupting that one too. pounce never
checks. So :func:`validate_patterns` is the only check there is, and it runs by
default: exactly against a dense AD derivative when the problem is small,
column-by-column against JVPs and Hessian-vector products when it is not.

One numerical subtlety in that check, which is easy to get wrong: the
Lagrangian Hessian must be checked at *random* multipliers, never at
``lambda = 1``. A unit's material balances share one reaction term whose
stoichiometric coefficients sum to zero over the species, so weighting the rows
equally cancels the nonlinearity exactly and gives ``H = 0``, at which point any
pattern passes -- including an empty one. Graph analysis is immune to this
(reachability does not cancel), which is another reason to prefer it; the
numerical check has to work around it.
"""

from __future__ import annotations

from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.solvers._lazy import require

__all__ = [
    "SparsityPatternError",
    "SparsityDetectionError",
    "detect_jacobian_pattern",
    "detect_hessian_pattern",
    "detect_patterns",
    "dense_jacobian_pattern",
    "dense_hessian_pattern",
    "lower_triangle",
    "pattern_density",
    "validate_patterns",
]

#: Above this many entries a dense ``(m, n)`` boolean mask is not built and the
#: validation switches to sampled columns. 40 000 entries is 40 kB of mask and
#: a few hundred milliseconds of dense AD; a hundred times that is neither.
DENSE_CHECK_LIMIT = 40_000


class SparsityPatternError(RuntimeError):
    """A supplied sparsity pattern is not a superset of the true structure.

    Raised by :func:`validate_patterns`. This is the failure that pounce
    itself will *not* report: it accepts any pattern and silently returns
    wrong derivatives for the entries the pattern omits.
    """


class SparsityDetectionError(RuntimeError):
    """No sparsity pattern could be derived for the model.

    Raised instead of quietly returning a dense pattern. Dense is a valid
    superset but a false economy -- see the module docstring -- so choosing
    it has to be the caller's decision, made in the open with
    ``sparsity="dense"``.
    """


# ---------------------------------------------------------------------------
# Pattern arithmetic
# ---------------------------------------------------------------------------


def dense_jacobian_pattern(m: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Every entry of an ``(m, n)`` matrix, in cyipopt ``(rows, cols)`` form.

    Always a valid superset, and always the most expensive one. Reach for it
    only to bisect a suspected pattern bug.
    """
    rows, cols = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")
    return rows.ravel().astype(np.int64), cols.ravel().astype(np.int64)


def dense_hessian_pattern(n: int) -> tuple[np.ndarray, np.ndarray]:
    """The full lower triangle of an ``(n, n)`` matrix."""
    rows, cols = np.tril_indices(n)
    return rows.astype(np.int64), cols.astype(np.int64)


def lower_triangle(rows, cols) -> tuple[np.ndarray, np.ndarray]:
    """Fold a symmetric pattern onto its lower triangle, de-duplicated.

    Args:
        rows: Row indices.
        cols: Column indices.

    Returns:
        ``(rows, cols)`` with ``rows >= cols``, sorted and unique.

    Example:
        >>> r, c = lower_triangle([0, 1, 0], [1, 0, 0])
        >>> r.tolist(), c.tolist()
        ([0, 1], [0, 0])
    """
    r = np.asarray(rows, dtype=np.int64)
    c = np.asarray(cols, dtype=np.int64)
    lo = np.minimum(r, c)
    hi = np.maximum(r, c)
    pairs = np.unique(np.stack([hi, lo], axis=1), axis=0) if len(r) else np.zeros(
        (0, 2), dtype=np.int64
    )
    return pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64)


def pattern_density(pattern, m: int, n: int) -> float:
    """Fraction of the ``(m, n)`` matrix the pattern occupies.

    Args:
        pattern: ``(rows, cols)``.
        m: Number of rows.
        n: Number of columns.

    Returns:
        ``nnz / (m * n)``, or ``0.0`` for an empty matrix.
    """
    total = m * n
    if total == 0:
        return 0.0
    return float(len(np.asarray(pattern[0]))) / total


def _row_to_cols(pattern, m: int) -> list[set[int]]:
    """Per-row column sets, without materializing an ``(m, n)`` mask."""
    out: list[set[int]] = [set() for _ in range(m)]
    r = np.asarray(pattern[0], dtype=np.int64)
    c = np.asarray(pattern[1], dtype=np.int64)
    for i, j in zip(r.tolist(), c.tolist()):
        out[i].add(j)
    return out


def _col_to_rows(pattern, n: int) -> list[set[int]]:
    """Per-column row sets, without materializing the mask."""
    out: list[set[int]] = [set() for _ in range(n)]
    r = np.asarray(pattern[0], dtype=np.int64)
    c = np.asarray(pattern[1], dtype=np.int64)
    for i, j in zip(r.tolist(), c.tolist()):
        out[j].add(i)
    return out


# ---------------------------------------------------------------------------
# Global detection (asdex)
# ---------------------------------------------------------------------------


def _asdex():
    """The asdex module, or an ImportError naming the extra that installs it."""
    try:
        return require("asdex")
    except ImportError as exc:
        raise ImportError(
            f"{exc} asdex is where difflow's default sparsity pattern comes "
            "from. Without it, sparsity='structural' falls back to the "
            "topology-derived pattern (valid, but dense in the Hessian) and "
            "sparsity='dense' accepts no pattern at all."
        ) from exc


def _as_pattern(sp) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(sp.rows, dtype=np.int64),
        np.asarray(sp.cols, dtype=np.int64),
    )


def detect_jacobian_pattern(
    g: Callable, x0: Array, *, args: Sequence = ()
) -> tuple[np.ndarray, np.ndarray]:
    """Global sparsity of ``dg/dx``, from the computation graph.

    No derivative is evaluated and no value is used: :mod:`asdex` walks the
    jaxpr of ``g`` and propagates index sets, so the answer holds at every
    point, not only at ``x0``. ``x0`` supplies shape and dtype.

    Args:
        g: Constraint body, ``g(x, *args) -> (m,)``, JAX-traceable.
        x0: Any point of the right shape and dtype.
        args: Extra positional arguments of ``g``, held fixed. They are not
            differentiated, so their values are irrelevant to the result.

    Returns:
        ``(rows, cols)`` in cyipopt convention, a superset of the true
        structure everywhere.

    Raises:
        SparsityDetectionError: If the graph contains a primitive asdex
            cannot interpret -- most often an inner solve. Sequential-modular
            unit calls close their balance with ``optimistix.Newton``, which
            emits ``linear_solve``; use the equation-oriented residual form,
            which is both traceable and genuinely sparser.
        ImportError: If asdex is not installed.

    Example:
        >>> rows, cols = detect_jacobian_pattern(g, bd.x0)   # doctest: +SKIP
    """
    asdex = _asdex()
    try:
        sp = asdex.jacobian_sparsity(g, jnp.asarray(x0), *args)
    except Exception as exc:  # pragma: no cover - depends on the model
        raise SparsityDetectionError(
            "asdex could not derive the Jacobian structure of this model "
            f"({type(exc).__name__}: {exc}). This is usually an inner solve in "
            "the graph -- a sequential-modular unit call, or a `lax.while_loop` "
            "-- which the equation-oriented form does not have. Build the model "
            "through as_nlp(), or pass sparsity='structural' for the "
            "topology-derived pattern."
        ) from exc
    return _as_pattern(sp)


def detect_hessian_pattern(
    f: Callable,
    g: Callable | None,
    x0: Array,
    m: int,
    *,
    args: Sequence = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Global sparsity of the Lagrangian Hessian's lower triangle.

    The Lagrangian is ``L(x, lam) = f(x) + lam . g(x)``, and the multipliers
    are passed as a **traced argument**, never baked in as constants. That is
    what makes the result the union over all multiplier values: every
    ``grad^2 g_i`` reaches the sum through the graph regardless of what
    ``lam`` happens to hold. (Contrast the numerical check in
    :func:`validate_patterns`, where ``lam = 1`` cancels a real term to zero.)

    Args:
        f: Objective, ``f(x, *args) -> scalar``.
        g: Constraint body, or ``None`` for an unconstrained problem.
        x0: Any point of the right shape and dtype.
        m: Number of constraints.
        args: Extra positional arguments of ``f`` and ``g``, held fixed.

    Returns:
        ``(rows, cols)`` for the lower triangle, a superset everywhere.

    Raises:
        SparsityDetectionError: As :func:`detect_jacobian_pattern`.
        ImportError: If asdex is not installed.
    """
    asdex = _asdex()
    x0 = jnp.asarray(x0)
    try:
        if g is None or m == 0:
            sp = asdex.hessian_sparsity(f, x0, *args)
        else:

            def lagrangian(x, lam, *rest):
                return jnp.reshape(jnp.asarray(f(x, *rest)), ()) + jnp.dot(
                    lam, jnp.asarray(g(x, *rest)).ravel()
                )

            sp = asdex.hessian_sparsity(
                lagrangian, x0, jnp.ones(m, dtype=x0.dtype), *args
            )
    except Exception as exc:  # pragma: no cover - depends on the model
        raise SparsityDetectionError(
            "asdex could not derive the Lagrangian Hessian structure of this "
            f"model ({type(exc).__name__}: {exc}). See "
            "detect_jacobian_pattern for the usual cause."
        ) from exc
    return lower_triangle(*_as_pattern(sp))


def detect_patterns(
    f: Callable,
    g: Callable,
    x0: Array,
    m: int,
    *,
    args: Sequence = (),
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Both patterns for a hand-built ``(f, g)`` pair.

    The convenience entry point for a model written directly against
    ``pounce``, rather than through
    :func:`~difflow.solvers.nlp.as_nlp`, which calls this itself.

    Args:
        f: Objective.
        g: Constraint body.
        x0: Any point of the right shape and dtype.
        m: Number of constraints.
        args: Extra positional arguments of both, held fixed.

    Returns:
        ``(jac_pattern, hess_pattern)``, ready for
        ``pounce.jax.JaxProblem(jac_pattern=..., hess_pattern=...)``.

    Example:
        >>> jac, hess = detect_patterns(f, g, x0, m=3)        # doctest: +SKIP
        >>> problem = pj.JaxProblem(f, g, n=x0.size, m=3,     # doctest: +SKIP
        ...                         jac_pattern=jac, hess_pattern=hess)
    """
    return (
        detect_jacobian_pattern(g, x0, args=args),
        detect_hessian_pattern(f, g, x0, m, args=args),
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _multiplier_draws(m: int, n_multipliers: int, seed: int) -> list[np.ndarray]:
    """Random multipliers plus a zero draw, so the objective is covered too."""
    rng = np.random.default_rng(seed)
    return [np.zeros(m)] + [
        rng.standard_normal(m) for _ in range(max(1, int(n_multipliers)))
    ]


def _check_dense(g, x, jac_pattern, m, n, f, hess_pattern, n_multipliers, seed):
    """Exact check against dense AD derivatives. Costs ``O(n)`` memory."""
    mask = np.zeros((m, n), dtype=bool)
    r, c = jac_pattern
    mask[np.asarray(r), np.asarray(c)] = True
    J = np.asarray(jax.jacobian(g)(x))
    missing = np.argwhere((np.abs(J) > 0) & ~mask)
    if missing.size:
        raise SparsityPatternError(
            f"jac_pattern misses {len(missing)} structurally nonzero entries "
            f"at x0, e.g. {missing[:5].tolist()}. A missing entry is silently "
            "wrong in pounce -- the derivative is dropped, or aliased into a "
            "same-colored entry under sparse AD."
        )
    if f is None or hess_pattern is None:
        return
    hmask = np.zeros((n, n), dtype=bool)
    hr, hc = np.asarray(hess_pattern[0]), np.asarray(hess_pattern[1])
    hmask[hr, hc] = True
    hmask[hc, hr] = True  # pounce folds the upper triangle onto its mirror

    seen = np.zeros((n, n), dtype=bool)
    for lam_np in _multiplier_draws(m, n_multipliers, seed):
        lam = jnp.asarray(lam_np)

        def lagrangian(z, lam=lam):
            return f(z) + jnp.dot(lam, g(z))

        seen |= np.abs(np.asarray(jax.hessian(lagrangian)(x))) > 0
    missing = np.argwhere(seen & ~hmask)
    if missing.size:
        raise SparsityPatternError(
            f"hess_pattern misses {len(missing)} structurally nonzero entries "
            f"at x0, e.g. {missing[:5].tolist()}."
        )


def _check_sampled(
    g, x, jac_pattern, m, n, f, hess_pattern, n_multipliers, seed, n_samples
):
    """Exact check of a random sample of *columns*, at ``O(n)`` memory.

    A JVP in the direction ``e_j`` returns column ``j`` of the Jacobian
    exactly, so each sampled column is checked in full: this samples columns,
    it does not approximate them. The Hessian columns come the same way, from
    a Hessian-vector product at each multiplier draw.

    Every JVP is taken under one ``vmap``, so the flowsheet is traced once per
    derivative rather than once per column. Tracing dominates the cost here --
    unbatched, this check ran 80x slower than the pattern detection it is
    checking.
    """
    rng = np.random.default_rng(seed)
    k = min(int(n_samples), n)
    if k == 0:
        return
    cols = np.sort(rng.choice(n, size=k, replace=False))
    tangents = (
        jnp.zeros((k, n), dtype=x.dtype)
        .at[jnp.arange(k), jnp.asarray(cols)]
        .set(1.0)
    )

    col_rows = _col_to_rows(jac_pattern, n)
    J = np.asarray(jax.vmap(lambda e: jax.jvp(g, (x,), (e,))[1])(tangents))
    for slot, j in enumerate(cols.tolist()):
        extra = [int(i) for i in np.flatnonzero(np.abs(J[slot]) > 0)
                 if int(i) not in col_rows[j]]
        if extra:
            raise SparsityPatternError(
                f"jac_pattern misses {len(extra)} structurally nonzero entries "
                f"in column {j}, e.g. rows {extra[:5]}. A missing entry is "
                "silently wrong in pounce -- the derivative is dropped, or "
                "aliased into a same-colored entry under sparse AD."
            )
    if f is None or hess_pattern is None:
        return

    def lagrangian(z, lam):
        return jnp.reshape(jnp.asarray(f(z)), ()) + jnp.dot(
            lam, jnp.asarray(g(z)).ravel()
        )

    grad_L = jax.grad(lagrangian, argnums=0)

    def hessian_column(lam, e):
        return jax.jvp(lambda z: grad_L(z, lam), (x,), (e,))[1]

    lams = jnp.asarray(
        np.stack(_multiplier_draws(m, n_multipliers, seed)), dtype=x.dtype
    )
    H = np.asarray(
        jax.vmap(jax.vmap(hessian_column, in_axes=(None, 0)), in_axes=(0, None))(
            lams, tangents
        )
    )  # (draws, k, n)

    hcols = _col_to_rows(hess_pattern, n)
    hrows = _row_to_cols(hess_pattern, n)
    seen = np.abs(H).max(axis=0) > 0  # (k, n), unioned over the draws
    for slot, j in enumerate(cols.tolist()):
        allowed = hcols[j] | hrows[j]  # the pattern is a triangle; mirror it
        extra = [int(i) for i in np.flatnonzero(seen[slot])
                 if int(i) not in allowed]
        if extra:
            raise SparsityPatternError(
                f"hess_pattern misses {len(extra)} structurally nonzero "
                f"entries in column {j}, e.g. rows {extra[:5]}."
            )


def validate_patterns(
    g: Callable,
    x: Array,
    jac_pattern,
    m: int,
    n: int,
    *,
    f: Callable | None = None,
    hess_pattern=None,
    n_multipliers: int = 4,
    seed: int = 0,
    mode: str = "auto",
    n_samples: int = 16,
    dense_limit: int = DENSE_CHECK_LIMIT,
) -> None:
    """Check that the patterns cover the AD structure at ``x``.

    pounce accepts any pattern without looking at the model, and a missing
    entry is silently wrong, so this is the only check there is. It is a
    *point* check: passing means the pattern covers the structure at ``x``,
    not everywhere. What makes a pattern valid everywhere is its derivation --
    graph analysis in :func:`detect_jacobian_pattern`, the by-construction
    argument in :func:`~difflow.solvers.nlp.as_nlp`'s ``"structural"`` mode --
    and this catches mistakes in that derivation.

    Large problems are checked by *sampling columns*, not by skipping. A JVP
    in the direction ``e_j`` is column ``j`` of the Jacobian exactly, so each
    sampled column is verified in full at ``O(n)`` memory instead of the
    ``O(mn)`` a dense Jacobian would need.

    The Lagrangian Hessian is checked at **several random multiplier
    vectors**, not at ``lambda = 1``. The obvious all-ones choice is
    numerically vacuous on exactly the models this package targets: a unit's
    material balances share one reaction term with stoichiometric coefficients
    that sum to zero over the species, so summing the rows with equal weight
    cancels the nonlinearity and gives ``H = 0`` identically. The check then
    passes for *any* pattern, including an empty one. Random multipliers break
    the cancellation; a ``lambda = 0`` draw is also included so the objective's
    own Hessian is covered.

    Args:
        g: Constraint body.
        x: Point at which to evaluate the derivative.
        jac_pattern: ``(rows, cols)`` for the ``(m, n)`` Jacobian.
        m: Number of constraints.
        n: Number of variables.
        f: Objective; when given, the Lagrangian Hessian is checked too.
        hess_pattern: ``(rows, cols)`` for the Hessian's lower triangle.
        n_multipliers: Number of random multiplier draws to union.
        seed: Seed for those draws, so the check is reproducible.
        mode: ``"auto"`` picks dense under ``dense_limit`` entries and sampled
            above it; ``"dense"`` and ``"sampled"`` force one.
        n_samples: Columns to sample in ``"sampled"`` mode.
        dense_limit: ``m * n`` at which ``"auto"`` switches to sampling.

    Raises:
        SparsityPatternError: If any structurally nonzero entry at ``x`` is
            missing from the pattern.
        ValueError: On an unknown ``mode``.

    Example:
        >>> validate_patterns(g, x0, jac_pattern, m, n)   # doctest: +SKIP
    """
    x = jnp.asarray(x)
    if mode == "auto":
        mode = "dense" if m * n <= dense_limit else "sampled"
    if mode == "dense":
        _check_dense(g, x, jac_pattern, m, n, f, hess_pattern, n_multipliers, seed)
    elif mode == "sampled":
        _check_sampled(
            g, x, jac_pattern, m, n, f, hess_pattern, n_multipliers, seed, n_samples
        )
    else:
        raise ValueError(
            f"mode must be 'auto', 'dense' or 'sampled', got {mode!r}"
        )
