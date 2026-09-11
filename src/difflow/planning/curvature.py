"""Second-order models: curvature by Hessian-vector product.

A delta vector is a first-order model, ``y ~= y0 + J (u - u0)``.  AD supplies
the second order for about what the first order costs: a Hessian-vector
product is one :func:`jax.jvp` through :func:`jax.grad`, so the exact Hessian
of one scalar output costs ``O(n_u)`` HVPs — roughly what a *central-difference
Jacobian alone* costs the systems that build delta vectors by perturbation.
The second-order model is therefore available at the price the incumbent
already pays for its first-order one.

Where it helps, it helps a great deal.  On a reduced AC power-flow model,
stepping the whole way from an incumbent generator schedule to the optimal
one, the linear model is off by 125.7 in cost while the exact quadratic is off
by 0.13 — three orders of magnitude, over a step no trust region would even
allow in one cycle.

Whether to *use* it is a different question, and this module answers it rather
than assuming.  Every guarantee in :mod:`difflow.planning` chains off the
trust-region subproblem being an LP: solved to global optimality, duals
readable as prices, Eason-Biegler filter convergence.  A quadratic objective
preserves those only while its Hessian is definite in the direction of
optimisation.  An indefinite one makes the subproblem a nonconvex QP and voids
all three *while still returning a number* — the failure mode this package is
organised against.

And definiteness is a property of **where you are**, not of the model.  On the
same nine-bus network the reduced cost Hessian is positive definite at the
incumbent operating point and strongly indefinite at heavily loaded ones, and
the Hessians of the voltage and thermal limits are indefinite at nearly every
point sampled.  So it has to be measured at the linearisation point each
cycle, which is what :func:`block_curvature` is for, and what
:func:`check_model_order` folds into a recommendation.

Nothing here changes the planner.  These are diagnostics on a block, in the
same spirit as :func:`difflow.planning.linearize.check_delta_vectors` and
:func:`difflow.planning.health.check_delta_health`: they tell you what a
second-order subproblem would buy and what it would cost in guarantees, so the
decision to build one is taken on evidence.

Example:
    >>> import jax.numpy as jnp
    >>> from difflow.planning import Block
    >>> from difflow.planning.curvature import check_model_order
    >>> blk = Block(name="r", fn=lambda u: jnp.array([u[0] ** 2]),
    ...             u_names=["x"], y_names=["y"], lb=[0.0], ub=[2.0])
    >>> rep = check_model_order(blk, "y")
    >>> rep.recommended
    'quadratic'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.planning.block import Block
from difflow.planning.health import _u_scale

__all__ = [
    "Curvature",
    "ModelOrderReport",
    "block_curvature",
    "block_hvp",
    "check_model_order",
    "classify_definiteness",
    "hessian_of",
    "hvp",
    "scalar_output",
]

#: Relative tolerance on an eigenvalue counted as zero.
EIG_TOL = 1e-8

#: Error-reduction factor at which the quadratic model is worth its cost.
GAIN_TOL = 4.0


def hvp(fn: Callable[[Array], Array], u: Array, v: Array) -> Array:
    """Hessian-vector product ``H(u) @ v`` for a scalar-valued ``fn``.

    One forward-over-reverse pass: :func:`jax.jvp` through :func:`jax.grad`.
    It costs a small constant multiple of one evaluation of ``fn`` and never
    forms ``H``, so ``n_u`` of these build the full Hessian in ``O(n_u)``
    evaluations rather than ``O(n_u**2)``.

    Args:
        fn: Scalar-valued callable of one array.
        u: Point at which to take the curvature.
        v: Direction.

    Returns:
        ``H(u) @ v``, shaped like ``u``.

    Example:
        >>> import jax.numpy as jnp
        >>> f = lambda x: jnp.sum(x ** 3)
        >>> hvp(f, jnp.array([1.0, 2.0]), jnp.array([1.0, 0.0]))
        Array([6., 0.], dtype=float64)
    """
    u = jnp.atleast_1d(jnp.asarray(u, dtype=float))
    v = jnp.atleast_1d(jnp.asarray(v, dtype=float))
    return jax.jvp(jax.grad(fn), (u,), (v,))[1]


def scalar_output(block: Block, output: str | int | Mapping[str, float],
                  theta: Mapping[str, Any] | None = None,
                  ) -> Callable[[Array], Array]:
    """Reduce a block to a scalar callable of ``u``.

    Args:
        block: The block.
        output: One output, by bare or qualified name or by index; or a
            mapping ``{output name: weight}`` giving a linear combination.
            The mapping form is how you take the curvature of the *priced*
            objective rather than of a single output.
        theta: Parameter override.

    Returns:
        A traceable callable ``u -> scalar``.

    Raises:
        KeyError: If an output name is not one of the block's.
    """
    if isinstance(output, Mapping):
        if not output:
            raise ValueError("output weighting is empty")
        w = np.zeros(block.n_y)
        for name, weight in output.items():
            w[block.y_index(name)] = float(weight)
        weights = jnp.asarray(w)

        def f(u: Array) -> Array:
            return jnp.dot(weights, jnp.ravel(block.evaluate(u, theta)))
        return f

    index = output if isinstance(output, (int, np.integer)) \
        else block.y_index(output)
    index = int(index)
    if not 0 <= index < block.n_y:
        raise IndexError(
            f"output index {index} out of range for block {block.name!r} "
            f"with {block.n_y} outputs")

    def f(u: Array) -> Array:
        return jnp.ravel(block.evaluate(u, theta))[index]
    return f


def block_hvp(block: Block, output: str | int | Mapping[str, float],
              v: Array, u0: Array | None = None,
              theta: Mapping[str, Any] | None = None) -> Array:
    """Curvature of one block output in one direction.

    Args:
        block: The block.
        output: Output selector; see :func:`scalar_output`.
        v: Direction, length ``block.n_u``.
        u0: Point.  Defaults to ``block.u0``.
        theta: Parameter override.

    Returns:
        ``H @ v``, length ``block.n_u``.
    """
    u0 = block.u0 if u0 is None else jnp.atleast_1d(jnp.asarray(u0, float))
    return hvp(scalar_output(block, output, theta), u0, v)


def hessian_of(block: Block, output: str | int | Mapping[str, float],
               u0: Array | None = None,
               theta: Mapping[str, Any] | None = None) -> Array:
    """Exact Hessian of one block output, shape ``(n_u, n_u)``.

    Costs ``O(n_u)`` model evaluations — the same order as the central
    differences that would be needed for the *first* derivative alone.

    Args:
        block: The block.
        output: Output selector; see :func:`scalar_output`.
        u0: Point.  Defaults to ``block.u0``.
        theta: Parameter override.

    Returns:
        The symmetric Hessian at ``u0``.
    """
    u0 = block.u0 if u0 is None else jnp.atleast_1d(jnp.asarray(u0, float))
    return jax.hessian(scalar_output(block, output, theta))(u0)


def classify_definiteness(eigenvalues: np.ndarray,
                          tol: float = EIG_TOL) -> str:
    """Classify a symmetric matrix from its eigenvalues.

    Args:
        eigenvalues: Real eigenvalues, ascending or not.
        tol: Relative tolerance; an eigenvalue within ``tol * max|lambda|``
            of zero counts as zero.  A weakly curved model has genuine zero
            eigenvalues, so this cannot be an absolute test.

    Returns:
        ``"psd"``, ``"nsd"``, ``"indefinite"`` or ``"zero"``.
    """
    eig = np.asarray(eigenvalues, dtype=float)
    if eig.size == 0:
        return "zero"
    scale = float(np.max(np.abs(eig)))
    cut = max(tol * scale, 1e-300)
    if scale <= cut:
        return "zero"
    if float(np.min(eig)) >= -cut:
        return "psd"
    if float(np.max(eig)) <= cut:
        return "nsd"
    return "indefinite"


@dataclass
class Curvature:
    """Second-order information about one block output at one point.

    Attributes:
        block: Block name.
        output: The output selector this was taken for, rendered as text.
        u0: Point the curvature was taken at.
        H: Exact Hessian, shape ``(n_u, n_u)``.
        eigenvalues: Ascending eigenvalues of ``H``.
        definiteness: ``"psd"``, ``"nsd"``, ``"indefinite"`` or ``"zero"``
            from :func:`classify_definiteness`.
    """

    block: str
    output: str
    u0: np.ndarray
    H: np.ndarray
    eigenvalues: np.ndarray
    definiteness: str

    @property
    def min_eigenvalue(self) -> float:
        return float(np.min(self.eigenvalues)) if self.eigenvalues.size else 0.0

    @property
    def max_eigenvalue(self) -> float:
        return float(np.max(self.eigenvalues)) if self.eigenvalues.size else 0.0

    def convex_for(self, sense: str) -> bool:
        """Would a quadratic subproblem stay convex under this ``sense``?

        Args:
            sense: ``"min"`` or ``"max"``, as passed to the planner.

        Returns:
            True when minimising a PSD model or maximising an NSD one — the
            two cases in which the subproblem still solves to global
            optimality and its duals still read as prices.  A ``"zero"``
            Hessian is convex either way: the model is simply linear.
        """
        if sense not in ("min", "max"):
            raise ValueError(f"sense must be 'min' or 'max', got {sense!r}")
        if self.definiteness == "zero":
            return True
        return self.definiteness == ("psd" if sense == "min" else "nsd")

    def summary(self) -> str:
        """Render the curvature as text."""
        return (f"curvature of {self.block}.{self.output}: {self.definiteness} "
                f"(eigenvalues {self.min_eigenvalue:.3e} .. "
                f"{self.max_eigenvalue:.3e})")

    def __repr__(self) -> str:
        return (f"Curvature(block={self.block!r}, output={self.output!r}, "
                f"definiteness={self.definiteness!r}, "
                f"min_eig={self.min_eigenvalue:.3e})")


def block_curvature(block: Block, output: str | int | Mapping[str, float],
                    u0: Array | None = None,
                    theta: Mapping[str, Any] | None = None,
                    tol: float = EIG_TOL) -> Curvature:
    """Exact Hessian of one block output, with its definiteness verdict.

    Args:
        block: The block.
        output: Output selector; see :func:`scalar_output`.
        u0: Point.  Defaults to ``block.u0``.
        theta: Parameter override.
        tol: Relative eigenvalue tolerance.

    Returns:
        A :class:`Curvature`.

    Example:
        >>> import jax.numpy as jnp
        >>> from difflow.planning import Block
        >>> blk = Block(name="b", fn=lambda u: jnp.array([u[0] ** 2]),
        ...             u_names=["x"], y_names=["y"], lb=[0.0], ub=[1.0])
        >>> block_curvature(blk, "y").definiteness
        'psd'
    """
    u0 = block.u0 if u0 is None else jnp.atleast_1d(jnp.asarray(u0, float))
    H = np.asarray(hessian_of(block, output, u0, theta), dtype=float)
    H = 0.5 * (H + H.T)          # symmetrise: AD returns it symmetric to
                                 # rounding, and eigvalsh reads only one
                                 # triangle, so make the object honest.
    eig = np.linalg.eigvalsh(H)
    label = output if isinstance(output, str) else (
        "+".join(sorted(output)) if isinstance(output, Mapping)
        else block.y_names[int(output)])
    return Curvature(block=block.name, output=str(label),
                     u0=np.asarray(u0, dtype=float), H=H, eigenvalues=eig,
                     definiteness=classify_definiteness(eig, tol))


@dataclass
class ModelOrderReport:
    """Linear against quadratic, measured on the block's own response.

    Attributes:
        block: Block name.
        output: Output selector, rendered as text.
        u0: Expansion point.
        direction: The step the models were compared along.
        radius: Trust-region fraction the direction was scaled to.
        t: Fractions of ``direction`` sampled.
        true: The block's nonlinear response at each ``t``.
        linear: First-order prediction at each ``t``.
        quadratic: Second-order prediction at each ``t``.
        curvature: The :class:`Curvature` behind the quadratic model.
        sense: Optimisation sense the convexity verdict was taken for.
        gain_tol: Error-reduction factor required to recommend quadratic.
    """

    block: str
    output: str
    u0: np.ndarray
    direction: np.ndarray
    radius: float
    t: np.ndarray
    true: np.ndarray
    linear: np.ndarray
    quadratic: np.ndarray
    curvature: Curvature
    sense: str = "min"
    gain_tol: float = GAIN_TOL

    @property
    def linear_error(self) -> np.ndarray:
        return self.linear - self.true

    @property
    def quadratic_error(self) -> np.ndarray:
        return self.quadratic - self.true

    @property
    def max_linear_error(self) -> float:
        return float(np.max(np.abs(self.linear_error)))

    @property
    def max_quadratic_error(self) -> float:
        return float(np.max(np.abs(self.quadratic_error)))

    @property
    def scale(self) -> float:
        """Magnitude of the response, for judging an error as negligible."""
        return max(float(np.max(np.abs(self.true))), 1e-300)

    @property
    def quadratic_is_exact(self) -> bool:
        """Is the quadratic error at the level of floating-point noise?

        A genuinely quadratic response reproduces exactly, and the residual
        error is then rounding.  Dividing by it would report an improvement
        of ``1e15``, which is a statement about float64 rather than about the
        model, so it is reported as exact instead.
        """
        return self.max_quadratic_error <= 1e-12 * self.scale

    @property
    def improvement(self) -> float:
        """Factor by which the quadratic model reduces the worst error.

        ``inf`` when the quadratic model is exact to rounding, ``1.0`` when
        the two are equally wrong, below ``1.0`` when the quadratic is worse —
        which does happen, on a response whose third derivative dominates
        over the step.
        """
        if self.quadratic_is_exact:
            return float("inf") if not self._linear_is_exact else 1.0
        return self.max_linear_error / self.max_quadratic_error

    @property
    def _linear_is_exact(self) -> bool:
        return self.max_linear_error <= 1e-12 * self.scale

    @property
    def convex(self) -> bool:
        """Would the quadratic subproblem stay convex under ``sense``?"""
        return self.curvature.convex_for(self.sense)

    @property
    def recommended(self) -> str:
        """``"linear"`` or ``"quadratic"``: which model earns its cost here.

        Quadratic is recommended only when it is *both* materially more
        accurate over the step (by at least ``gain_tol``) and definite in the
        direction of optimisation.  An indefinite Hessian is recommended
        against however well it fits, because a nonconvex QP subproblem
        forfeits the global optimality every downstream guarantee rests on.
        See :attr:`caveat` for what to do about that.
        """
        if self.improvement >= self.gain_tol and self.convex:
            return "quadratic"
        return "linear"

    @property
    def caveat(self) -> str | None:
        """Why a well-fitting quadratic model was still not recommended."""
        if self.recommended == "quadratic" or self.improvement < self.gain_tol:
            return None
        how = ("exact to rounding where the linear one is not"
               if self.quadratic_is_exact
               else f"{self.improvement:.0f}x more accurate than the linear "
                    f"one")
        return (
            f"the quadratic model is {how} over this step, but its Hessian is "
            f"{self.curvature.definiteness} and the sense is {self.sense!r}, "
            f"so the subproblem would be a nonconvex QP: solved locally "
            f"rather than globally, with duals that no longer read as prices. "
            f"Convexify it (Gauss-Newton, modified Cholesky, or a damped BFGS "
            f"update, which needs no second derivatives at all) before using "
            f"the curvature in a subproblem.")

    @property
    def _improvement_text(self) -> str:
        return ("exact (to rounding)" if self.quadratic_is_exact
                else f"{self.improvement:.1f}x")

    def summary(self) -> str:
        """Render the comparison as a table."""
        head = (f"model order for {self.block}.{self.output} "
                f"(radius {self.radius:g}, sense {self.sense!r})")
        rows = [f"{'t':>6} {'true':>14} {'linear err':>14} {'quad err':>14}"]
        for i, t in enumerate(self.t):
            rows.append(f"{t:6.2f} {self.true[i]:14.6g} "
                        f"{self.linear_error[i]:14.6g} "
                        f"{self.quadratic_error[i]:14.6g}")
        tail = [
            f"  worst linear error    {self.max_linear_error:.6g}",
            f"  worst quadratic error {self.max_quadratic_error:.6g}",
            f"  improvement           {self._improvement_text}",
            f"  {self.curvature.summary()}",
            f"  recommended: {self.recommended}",
        ]
        if self.caveat:
            tail.append(f"  caveat: {self.caveat}")
        return "\n".join([head] + rows + tail)

    def __repr__(self) -> str:
        return (f"ModelOrderReport(block={self.block!r}, "
                f"output={self.output!r}, "
                f"improvement={self._improvement_text}, "
                f"recommended={self.recommended!r})")


def _default_direction(block: Block, grad: np.ndarray, radius: float,
                       sense: str) -> np.ndarray:
    """The trust-region vertex an LP would step to for this output alone.

    A linear objective over a box is optimised at a vertex, so the step worth
    testing the model along is the full-radius move with each component taking
    the sign that improves the output.  That is the step the planner will
    actually propose, which is the only one whose model error matters.
    """
    scale = _u_scale(block)
    sign = -np.sign(grad) if sense == "min" else np.sign(grad)
    sign = np.where(sign == 0.0, 1.0, sign)
    return sign * radius * scale


def check_model_order(block: Block,
                      output: str | int | Mapping[str, float],
                      u0: Array | None = None,
                      direction: Array | None = None,
                      radius: float = 0.3,
                      n_points: int = 5,
                      sense: str = "min",
                      theta: Mapping[str, Any] | None = None,
                      gain_tol: float = GAIN_TOL) -> ModelOrderReport:
    """Compare the linear and quadratic models against the block itself.

    Both models are built at ``u0`` by AD and evaluated along a step; the
    block's own nonlinear response is evaluated at the same points.  The
    report says which model is worth its cost *here*, and refuses to
    recommend a quadratic whose Hessian would make the subproblem nonconvex.

    Args:
        block: The block.
        output: Output selector; see :func:`scalar_output`.  Pass a mapping
            to test the priced objective rather than a single output.
        u0: Expansion point.  Defaults to ``block.u0``.
        direction: Step to compare along.  Defaults to the trust-region
            vertex an LP would propose for this output — the step that
            actually gets taken, and so the one whose error matters.
        radius: Trust-region fraction used to scale the default direction.
        n_points: Number of fractions of the step to sample, excluding zero.
        sense: ``"min"`` or ``"max"``; decides which definiteness keeps the
            subproblem convex.
        theta: Parameter override.
        gain_tol: Error-reduction factor at which quadratic is recommended.

    Returns:
        A :class:`ModelOrderReport`.

    Raises:
        ValueError: If ``n_points`` is below 1, or ``direction`` has the
            wrong length.

    Example:
        >>> import jax.numpy as jnp
        >>> from difflow.planning import Block
        >>> blk = Block(name="b", fn=lambda u: jnp.array([u[0] ** 2]),
        ...             u_names=["x"], y_names=["y"], lb=[0.0], ub=[2.0])
        >>> rep = check_model_order(blk, "y")
        >>> rep.max_quadratic_error < 1e-9      # exact on a quadratic
        True
    """
    if n_points < 1:
        raise ValueError(f"n_points must be at least 1, got {n_points}")
    if sense not in ("min", "max"):
        raise ValueError(f"sense must be 'min' or 'max', got {sense!r}")

    u0 = block.u0 if u0 is None else jnp.atleast_1d(jnp.asarray(u0, float))
    f = scalar_output(block, output, theta)

    f0 = float(f(u0))
    g = np.asarray(jax.grad(f)(u0), dtype=float)
    curv = block_curvature(block, output, u0, theta)

    if direction is None:
        d = _default_direction(block, g, float(radius), sense)
    else:
        d = np.atleast_1d(np.asarray(direction, dtype=float))
        if d.shape != (block.n_u,):
            raise ValueError(
                f"direction has length {d.size}, expected {block.n_u} for "
                f"block {block.name!r}")

    ts = np.linspace(1.0 / n_points, 1.0, n_points)
    true, lin, quad = [], [], []
    for t in ts:
        s = t * d
        true.append(float(f(jnp.asarray(np.asarray(u0, float) + s))))
        lin.append(f0 + float(g @ s))
        quad.append(f0 + float(g @ s) + 0.5 * float(s @ (curv.H @ s)))

    return ModelOrderReport(
        block=block.name, output=curv.output, u0=np.asarray(u0, dtype=float),
        direction=d, radius=float(radius), t=ts, true=np.asarray(true),
        linear=np.asarray(lin), quadratic=np.asarray(quad), curvature=curv,
        sense=sense, gain_tol=float(gain_tol))
