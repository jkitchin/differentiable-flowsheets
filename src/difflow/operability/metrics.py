"""The measures themselves: RGA, singular values, disturbance directionality.

Every function here is a pure ``jnp`` expression on matrices, so all of them
work under ``jit``, ``vmap`` and ``grad``.  That is deliberate: the point of
computing these from AD rather than from a separately fitted linear model is
that they become cheap enough to sit *inside* a design objective, which they
can only do if they are traceable.

Two properties are worth keeping in mind while reading:

* The **RGA is invariant** to diagonal input and output scaling.  It is the
  one measure in this package that means the same thing scaled or not, which
  is exactly why it is so widely quoted — and why it is not sufficient on its
  own.
* Everything built on **singular values is not** invariant.  ``sigma_min``,
  the condition number and the disturbance gains all change with the units
  you happen to have used, so they are only interpretable after
  :class:`~difflow.operability.scaling.Scaling`.

References:
    Bristol, E.H. "On a new measure of interaction for multivariable process
    control", IEEE Trans. Automat. Contr. 11(1), 133, 1966
    (doi:10.1109/TAC.1966.1098266) — the RGA.
    Chang, J.-W. and Yu, C.-C. "The relative gain for non-square multivariable
    systems", Chem. Eng. Sci. 45(5), 1309, 1990
    (doi:10.1016/0009-2509(90)87123-a) — the pseudo-inverse RGA used here.
    Morari, M. "Design of resilient processing plants — III", Chem. Eng. Sci.
    38(11), 1881, 1983 (doi:10.1016/0009-2509(83)85044-1).
    Skogestad, S. and Postlethwaite, I. *Multivariable Feedback Control*,
    2nd ed., Wiley 2005, chapters 3, 6 and 10.
"""

from __future__ import annotations

import warnings
from typing import Any, Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.operability.scaling import OperabilityWarning, Scaling

__all__ = [
    "singular_values", "min_singular_value", "max_singular_value",
    "condition_number", "rga", "rga_number", "negative_pairings",
    "suggest_pairing", "effective_rank", "pinv",
    "disturbance_condition_number", "required_input_move",
]

#: Relative singular-value cutoff below which a direction is treated as null.
RCOND = float(np.finfo(float).eps) * 8.0


def _prepare(G: Any, scaling: Scaling | None, assume_scaled: bool,
             what: str, disturbance: bool = False) -> Array:
    """Apply scaling if given, warn loudly if the caller skipped it."""
    G = jnp.atleast_2d(jnp.asarray(G, dtype=float))
    if scaling is not None:
        G = (scaling.scale_disturbance(G) if disturbance
             else scaling.scale_gain(G))
        scaling.warn_if_unscaled(stacklevel=4)
    elif not assume_scaled:
        warnings.warn(
            f"{what} was computed from a gain matrix with no Scaling. In "
            "mixed engineering units this number measures your unit system, "
            "not the plant: it changes if you report a temperature in "
            "Celsius instead of Kelvin. Pass scaling=Scaling(...), or "
            "assume_scaled=True if the matrix is already dimensionless.",
            OperabilityWarning, stacklevel=3)
    return G


def singular_values(G: Any, scaling: Scaling | None = None,
                    assume_scaled: bool = False) -> Array:
    """Singular values of the gain matrix, descending.

    Args:
        G: Gain matrix ``(n_y, n_u)``.
        scaling: Applied to ``G`` first when given.
        assume_scaled: Suppress the unscaled warning because ``G`` is already
            dimensionless.

    Returns:
        Array of length ``min(n_y, n_u)``, descending.

    Note:
        Singular values are differentiable only where they are *distinct*,
        and the two families of measures in this module fail differently
        there.  This function and everything built on it use
        ``compute_uv=False``, whose gradient stays finite: at a crossing,
        ``sigma_min`` has a kink and AD silently returns one arm's slope, so
        an optimiser can stall rather than diverge.  Anything that needs the
        singular *vectors* — :func:`pinv`, and therefore :func:`rga`,
        :func:`required_input_move` and
        :func:`disturbance_condition_number` — differentiates through a
        ``1 / (s_i**2 - s_j**2)`` term and returns ``nan`` at an exactly
        repeated singular value.  A plant with two exactly equal gain
        directions is contrived, but a symmetric test case reaches it.
    """
    G = _prepare(G, scaling, assume_scaled, "singular_values")
    return jnp.linalg.svd(G, compute_uv=False)


def min_singular_value(G: Any, scaling: Scaling | None = None,
                       assume_scaled: bool = False) -> Array:
    """Minimum singular value of the *scaled* gain matrix.

    This is the standard measure of how hard a plant is to control: it is the
    smallest output move the inputs can produce per unit input move, taken
    over all directions, so it describes the plant's *worst* direction rather
    than its typical one.

    With the scaling convention of
    :class:`~difflow.operability.scaling.Scaling`, the threshold is 1:
    ``sigma_min >= 1`` says that in every direction the available inputs can
    move the outputs over the full range that has to be covered.  Below 1
    there is a direction in which they cannot, and no controller design
    recovers what the steady-state gains do not have.

    Args:
        G: Gain matrix ``(n_y, n_u)``.
        scaling: Applied to ``G`` first when given.  Omitting it makes the
            result unit-dependent and raises an
            :class:`~difflow.operability.scaling.OperabilityWarning`.
        assume_scaled: Suppress that warning because ``G`` is already
            dimensionless.

    Returns:
        Scalar array.

    Example:
        >>> import jax.numpy as jnp
        >>> from difflow.operability import Scaling, min_singular_value
        >>> G = jnp.array([[1.0, 0.0], [0.0, 0.1]])
        >>> float(min_singular_value(G, Scaling(u_span=[1.0, 10.0],
        ...                                     y_span=[1.0, 1.0])))
        1.0
    """
    G = _prepare(G, scaling, assume_scaled, "min_singular_value")
    return jnp.linalg.svd(G, compute_uv=False)[-1]


def max_singular_value(G: Any, scaling: Scaling | None = None,
                       assume_scaled: bool = False) -> Array:
    """Maximum singular value — the gain in the plant's easiest direction."""
    G = _prepare(G, scaling, assume_scaled, "max_singular_value")
    return jnp.linalg.svd(G, compute_uv=False)[0]


def condition_number(G: Any, scaling: Scaling | None = None,
                     assume_scaled: bool = False) -> Array:
    """Ratio of largest to smallest singular value of the scaled gain.

    A large condition number means the plant is *directional*: it responds
    strongly to some combinations of inputs and weakly to others.  Such a
    plant is not necessarily hard to control, but it is hard to control with
    decentralised loops, and it is sensitive to input uncertainty in its weak
    direction.  Above roughly 10 the directionality is worth taking seriously.

    Args:
        G: Gain matrix ``(n_y, n_u)``.
        scaling: Applied first when given.
        assume_scaled: Suppress the unscaled warning.

    Returns:
        Scalar array; ``inf`` when the smallest singular value is exactly
        zero, which is the honest answer for a singular plant.

    Note:
        The condition number is *not* scaling-invariant, and it is easy to
        make it small by choosing convenient units.  That is a reason to
        state the scaling, not a reason to distrust the measure.
    """
    G = _prepare(G, scaling, assume_scaled, "condition_number")
    s = jnp.linalg.svd(G, compute_uv=False)
    return jnp.where(s[-1] > 0, s[0] / jnp.where(s[-1] > 0, s[-1], 1.0),
                     jnp.inf)


def pinv(G: Any, rcond: float = RCOND) -> Array:
    """Moore-Penrose pseudo-inverse via SVD with an explicit cutoff.

    Args:
        G: Matrix ``(n_y, n_u)``.
        rcond: Singular values below ``rcond * s_max`` are treated as zero.

    Returns:
        ``(n_u, n_y)`` pseudo-inverse.

    Note:
        Written out rather than delegated so the cutoff is explicit and the
        expression stays free of ``0/0`` under ``grad``.  It still uses the
        singular *vectors*, so its gradient is ``nan`` at an exactly repeated
        singular value — see :func:`singular_values`.  Prefer
        :func:`min_singular_value` for a quantity to differentiate inside an
        objective; it does not form ``U`` or ``V``.
    """
    G = jnp.atleast_2d(jnp.asarray(G, dtype=float))
    U, s, Vt = jnp.linalg.svd(G, full_matrices=False)
    keep = s > rcond * s[0]
    s_inv = jnp.where(keep, 1.0 / jnp.where(keep, s, 1.0), 0.0)
    return (Vt.T * s_inv[None, :]) @ U.T


def effective_rank(G: Any, rcond: float = RCOND) -> Array:
    """Number of singular values above ``rcond * s_max``.

    Args:
        G: Matrix ``(n_y, n_u)``.
        rcond: Relative cutoff.

    Returns:
        Scalar array (an integer count as a float, so it stays traceable).
    """
    s = jnp.linalg.svd(jnp.atleast_2d(jnp.asarray(G, dtype=float)),
                       compute_uv=False)
    return jnp.sum(s > rcond * s[0]).astype(float)


def rga(G: Any, rcond: float = RCOND) -> Array:
    """Relative gain array, ``RGA = G * pinv(G).T`` elementwise.

    Entry ``[i, j]`` is the ratio of the open-loop gain from input ``j`` to
    output ``i`` to the gain that remains when every *other* loop is perfectly
    controlled.  So:

    * ``1`` — the pairing is unaffected by the other loops.
    * ``0`` — input ``j`` has no effect on output ``i`` once the other loops
      are closed; a useless pairing.
    * large positive — the loops fight each other and the pairing is very
      sensitive to model error.
    * **negative** — the gain of this pairing *changes sign* when the other
      loops are closed.  Pairing on a negative RGA element gives a system
      that is unstable either with all loops closed, or with that loop alone,
      or when a loop saturates; under integral control that is not a tuning
      problem, it is a structural one.  See
      :func:`negative_pairings`.

    The RGA is invariant to diagonal scaling of inputs and outputs, so unlike
    the singular-value measures it may be read directly off a raw AD Jacobian.

    Args:
        G: Gain matrix ``(n_y, n_u)``.  Need not be square: the pseudo-inverse
            gives the non-square RGA of Chang and Yu, for which only *one* of
            the two sum rules survives.  Row sums are ``diag(G G+)`` and
            column sums ``diag(G+ G)``, so with more inputs than outputs
            (full row rank) the **rows** sum to 1 and the columns do not,
            and with more outputs than inputs (full column rank) the
            **columns** sum to 1 and the rows do not.  Both hold for a square
            nonsingular ``G``.
        rcond: Relative singular-value cutoff used by the pseudo-inverse.  A
            singular ``G`` does not raise here; it produces an RGA whose rows
            do *not* sum to 1, which is itself the diagnosis.

    Returns:
        Array of shape ``(n_y, n_u)``.

    Example:
        >>> import jax.numpy as jnp
        >>> from difflow.operability import rga
        >>> rga(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
        Array([[-2.,  3.],
               [ 3., -2.]], dtype=float64)
    """
    G = jnp.atleast_2d(jnp.asarray(G, dtype=float))
    return G * pinv(G, rcond).T


def rga_number(G: Any, pairing: Sequence[int] | None = None,
               rcond: float = RCOND) -> Array:
    """Distance of the RGA from the identity under a chosen pairing.

    ``||RGA - P||`` summed elementwise, where ``P`` is the permutation matrix
    of the proposed pairing.  Zero means the pairing decouples the plant at
    steady state; large means decentralised control of that pairing will
    interact badly.

    Args:
        G: Gain matrix.
        pairing: ``pairing[i]`` is the index of the input paired with output
            ``i``.  Defaults to the diagonal pairing ``0-0, 1-1, ...``.
        rcond: Relative cutoff for the pseudo-inverse.

    Returns:
        Scalar array.

    Example:
        >>> import jax.numpy as jnp
        >>> from difflow.operability import rga_number
        >>> float(rga_number(jnp.eye(2)))
        0.0
    """
    R = rga(G, rcond)
    n_y, n_u = R.shape
    if pairing is None:
        P = jnp.eye(n_y, n_u)
    else:
        idx = np.atleast_1d(np.asarray(pairing, dtype=int))
        if idx.ndim != 1 or idx.size > n_y or (idx >= n_u).any():
            raise ValueError(
                f"pairing must give at most one input index per output "
                f"(n_y={n_y}, n_u={n_u}), got {list(idx)}")
        P = jnp.zeros((n_y, n_u)).at[jnp.arange(idx.size),
                                     jnp.asarray(idx)].set(1.0)
    return jnp.sum(jnp.abs(R - P))


def negative_pairings(G_or_rga: Any, pairing: Sequence[int] | None = None,
                      rcond: float = RCOND, is_rga: bool = False
                      ) -> list[tuple[int, int]]:
    """Pairings whose relative gain is negative.

    Args:
        G_or_rga: A gain matrix, or an RGA when ``is_rga`` is True.
        pairing: ``pairing[i]`` is the input paired with output ``i``.
            Defaults to the diagonal pairing.
        rcond: Relative cutoff for the pseudo-inverse.
        is_rga: Treat the first argument as an already-computed RGA.

    Returns:
        List of ``(output_index, input_index)`` pairs with a negative relative
        gain.  Empty is the good answer.

    Note:
        This is a concrete (non-traceable) helper: it returns Python data, so
        it cannot be called on tracers inside ``jit``.  The traceable form is
        the RGA diagonal itself, carried on
        :class:`~difflow.operability.screen.OperabilityReport`.
    """
    R = np.asarray(G_or_rga if is_rga else rga(G_or_rga, rcond), dtype=float)
    n_y = R.shape[0]
    idx = (np.arange(min(n_y, R.shape[1])) if pairing is None
           else np.asarray(pairing, dtype=int))
    return [(int(i), int(j)) for i, j in enumerate(idx) if R[i, j] < 0.0]


def suggest_pairing(G_or_rga: Any, rcond: float = RCOND,
                    is_rga: bool = False) -> list[int]:
    """Greedy input-output pairing from the RGA.

    Repeatedly takes the unused ``(output, input)`` pair whose relative gain
    is closest to 1 among the positive entries, which is the usual rule of
    thumb: pair on positive relative gains near 1 and never on negative ones.

    Args:
        G_or_rga: A gain matrix, or an RGA when ``is_rga`` is True.
        rcond: Relative cutoff for the pseudo-inverse.
        is_rga: Treat the first argument as an already-computed RGA.

    Returns:
        ``pairing`` where ``pairing[i]`` is the input index suggested for
        output ``i``, or ``-1`` where no positive relative gain was left.

    Note:
        A suggestion, not a design.  It ignores dynamics entirely, which is
        the standing limitation of every steady-state pairing rule.
    """
    R = np.asarray(G_or_rga if is_rga else rga(G_or_rga, rcond), dtype=float)
    n_y, n_u = R.shape
    score = np.where(R > 0, np.abs(R - 1.0), np.inf)
    pairing = [-1] * n_y
    used_u: set[int] = set()
    for _ in range(min(n_y, n_u)):
        masked = score.copy()
        for i, p in enumerate(pairing):
            if p != -1:
                masked[i, :] = np.inf
        for j in used_u:
            masked[:, j] = np.inf
        if not np.isfinite(masked).any():
            break
        i, j = np.unravel_index(int(np.argmin(masked)), masked.shape)
        pairing[int(i)] = int(j)
        used_u.add(int(j))
    return pairing


def required_input_move(G: Any, Gd: Any, scaling: Scaling | None = None,
                        assume_scaled: bool = False,
                        rcond: float = RCOND) -> Array:
    """Input move needed to reject each disturbance, ``pinv(G) @ Gd``.

    In scaled variables this is the answer to the question the whole screen
    exists for: *do the available inputs span the directions the disturbances
    actually push?*  Entry ``[j, k]`` is the fraction of input ``j``'s
    available range required to cancel a full-size excursion of disturbance
    ``k`` at steady state.  A column whose largest magnitude exceeds 1 is a
    disturbance the plant cannot reject with the inputs it has, no matter
    what controller is fitted.

    Args:
        G: Gain matrix ``(n_y, n_u)``.
        Gd: Disturbance gain ``(n_y, n_d)``.
        scaling: Applied to both when given.
        assume_scaled: Suppress the unscaled warning.
        rcond: Relative cutoff for the pseudo-inverse.

    Returns:
        Array of shape ``(n_u, n_d)``.

    Note:
        When ``G`` is rank deficient the pseudo-inverse returns the
        least-squares move, which cancels only the part of the disturbance
        lying in the plant's range space.  The residual is real and is not
        visible in this matrix; :func:`min_singular_value` and
        :func:`effective_rank` are what catch it.
    """
    Gs = _prepare(G, scaling, assume_scaled, "required_input_move")
    Gds = _prepare(Gd, scaling, True, "required_input_move", disturbance=True)
    return pinv(Gs, rcond) @ Gds


def disturbance_condition_number(G: Any, Gd: Any,
                                 scaling: Scaling | None = None,
                                 assume_scaled: bool = False,
                                 rcond: float = RCOND) -> Array:
    """Disturbance condition number, one value per disturbance.

    ``gamma_d = sigma_max(G) * ||pinv(G) y_d||`` with ``y_d`` the unit vector
    along the disturbance's output direction.  It measures *alignment*: how
    much harder this particular disturbance is to reject than one pushing
    along the plant's strongest direction.

    * ``gamma_d = 1`` — the disturbance pushes exactly where the plant is
      strongest; as easy as it gets.
    * ``gamma_d = cond(G)`` — it pushes exactly where the plant is weakest,
      the worst case.

    A large ``gamma_d`` with a small disturbance gain is harmless; a large
    ``gamma_d`` on a disturbance whose scaled gain exceeds 1 is the
    combination that makes a design uncontrollable, and it is invisible to
    either measure alone.

    Args:
        G: Gain matrix ``(n_y, n_u)``.
        Gd: Disturbance gain ``(n_y, n_d)``.
        scaling: Applied to both when given.
        assume_scaled: Suppress the unscaled warning.
        rcond: Relative cutoff for the pseudo-inverse.

    Returns:
        Array of length ``n_d``.  A disturbance with an exactly zero output
        direction is reported as 1 — it needs no rejection.

    Example:
        >>> import jax.numpy as jnp
        >>> from difflow.operability import disturbance_condition_number
        >>> G = jnp.array([[1.0, 0.0], [0.0, 0.01]])
        >>> gd = jnp.array([[0.0], [1.0]])          # pushes the weak direction
        >>> float(disturbance_condition_number(G, gd, assume_scaled=True)[0])
        100.0
    """
    Gs = _prepare(G, scaling, assume_scaled, "disturbance_condition_number")
    Gds = _prepare(Gd, scaling, True, "disturbance_condition_number",
                   disturbance=True)
    norms = jnp.linalg.norm(Gds, axis=0)
    safe = jnp.where(norms > 0, norms, 1.0)
    directions = Gds / safe[None, :]
    s_max = jnp.linalg.svd(Gs, compute_uv=False)[0]
    moves = jnp.linalg.norm(pinv(Gs, rcond) @ directions, axis=0)
    gamma = s_max * moves
    return jnp.where(norms > 0, gamma, 1.0)
