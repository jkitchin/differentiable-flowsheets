"""The discrete-time model an estimator sees, and how to build one.

Both estimators in this package --- the extended Kalman filter and the
moving-horizon estimator --- read the plant through one object,

.. math::

    x_{k+1} = f(x_k, u_k, w_k, \\theta), \\qquad
    y_k     = h(x_k, u_k, \\theta),

so that anything expressible as a pair of JAX-traceable callables can be
estimated. A dynamic flowsheet is turned into that pair by
:meth:`StateSpaceModel.from_ode`, which discretises the right-hand side
with :func:`difflow.dynamic.integrate` over one sampling interval rather
than re-implementing an integrator here.

The second constructor, :func:`augment_parameters`, is what makes this
useful for *monitoring* rather than only for control: it appends slowly
drifting parameters to the state as a random walk, so estimating an
extractant's stage efficiency and estimating a stage inventory become
the same computation. The augmented parameters come back out of
:attr:`~difflow.mhe.estimator.MHEResult.parameters` as a plain
``{name: value}`` mapping, which is the shape
:class:`difflow.planning.Block` takes for ``theta`` --- an estimator
feeding a real-time optimisation layer needs no adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.params_mixin import ParamsMixin


@dataclass
class StateSpaceModel(ParamsMixin):
    """A discrete-time model with additive-in-``w`` process noise.

    Attributes:
        f: transition ``f(x, u, w, theta) -> (n_x,)``. It must be
            JAX-traceable; ``w`` enters however the model says, not
            necessarily additively.
        h: measurement ``h(x, u, theta) -> (n_y,)``. Every channel of
            ``y`` appears here; a channel that is not sampled at a
            given time is switched off by an infinite sigma, never by
            changing ``h``, so the model stays shape-stable under
            ``jit``.
        n_x: number of states.
        n_y: number of measurement channels.
        n_u: number of inputs (0 for an autonomous model).
        n_w: number of process-noise entries, defaulting to ``n_x``.
        dt: sampling interval, in the time units of the measurement
            timestamps. The grid is assumed uniform.
        x_names, y_names, u_names: names for reporting.
        lb, ub: state bounds, shape ``(n_x,)``, with ``-inf``/``inf``
            where a state is unbounded. These are what
            :func:`~difflow.mhe.estimator.solve_mhe` enforces so an
            estimated concentration cannot come back negative.
        n_params: how many *trailing* states are augmented parameters
            rather than physical states; set by
            :func:`augment_parameters`.
        param_names: their names.

    Example:
        >>> import jax.numpy as jnp
        >>> model = StateSpaceModel(
        ...     f=lambda x, u, w, th: 0.9 * x + w,
        ...     h=lambda x, u, th: x,
        ...     n_x=1, n_y=1,
        ... )
        >>> model.step(jnp.array([1.0]), jnp.zeros(0), jnp.zeros(1))
        Array([0.9], dtype=float64)
    """

    f: Callable[..., Array]
    h: Callable[..., Array]
    n_x: int
    n_y: int
    n_u: int = 0
    n_w: int | None = None
    dt: float = 1.0
    x_names: list[str] = field(default_factory=list)
    y_names: list[str] = field(default_factory=list)
    u_names: list[str] = field(default_factory=list)
    lb: Array | None = None
    ub: Array | None = None
    n_params: int = 0
    param_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.n_w is None:
            self.n_w = int(self.n_x)
        self.n_x = int(self.n_x)
        self.n_y = int(self.n_y)
        self.n_u = int(self.n_u)
        self.n_w = int(self.n_w)
        self.x_names = list(self.x_names) or [
            f"x{i}" for i in range(self.n_x)
        ]
        self.y_names = list(self.y_names) or [
            f"y{i}" for i in range(self.n_y)
        ]
        self.u_names = list(self.u_names) or [
            f"u{i}" for i in range(self.n_u)
        ]
        for nm, want in (("x_names", self.n_x), ("y_names", self.n_y),
                         ("u_names", self.n_u)):
            if len(getattr(self, nm)) != want:
                raise ValueError(
                    f"{nm} has {len(getattr(self, nm))} entries but the "
                    f"model has {want}"
                )
        if self.lb is not None:
            self.lb = jnp.broadcast_to(
                jnp.asarray(self.lb, dtype=jnp.float64), (self.n_x,)
            )
        if self.ub is not None:
            self.ub = jnp.broadcast_to(
                jnp.asarray(self.ub, dtype=jnp.float64), (self.n_x,)
            )
        self.param_names = list(self.param_names)
        if self.n_params and len(self.param_names) != self.n_params:
            raise ValueError(
                f"got {len(self.param_names)} param_names for "
                f"{self.n_params} augmented parameters"
            )
        # Resolve the bounds once, in NumPy. Whether a state is bounded
        # decides which *code path* runs, so it has to be a Python bool
        # settled before tracing, not an array read inside a scan.
        lb = (np.full(self.n_x, -np.inf) if self.lb is None
              else np.asarray(self.lb, dtype=float))
        ub = (np.full(self.n_x, np.inf) if self.ub is None
              else np.asarray(self.ub, dtype=float))
        if np.any(lb > ub):
            bad = [self.x_names[i] for i in np.where(lb > ub)[0]]
            raise ValueError(f"lower bound above upper bound on {bad}")
        self._lb = jnp.asarray(lb, dtype=jnp.float64)
        self._ub = jnp.asarray(ub, dtype=jnp.float64)
        self._bounded = bool(np.any(np.isfinite(lb)) or np.any(np.isfinite(ub)))

    # -- bounds ---------------------------------------------------------

    @property
    def bounds(self) -> tuple[Array, Array]:
        """``(lb, ub)`` with infinities filled in where unbounded."""
        return self._lb, self._ub

    @property
    def is_bounded(self) -> bool:
        """Whether any state carries a finite bound."""
        return self._bounded

    # -- evaluation -----------------------------------------------------

    def step(self, x: Array, u: Array, w: Array, theta: Any = None) -> Array:
        """One transition, ``f(x, u, w, theta)``."""
        return jnp.asarray(self.f(x, u, w, theta), dtype=jnp.float64)

    def observe(self, x: Array, u: Array, theta: Any = None) -> Array:
        """The measurement map, ``h(x, u, theta)``."""
        return jnp.asarray(self.h(x, u, theta), dtype=jnp.float64)

    def rollout(
        self, x0: Array, u: Array, w: Array, theta: Any = None
    ) -> Array:
        """States over a window, shape ``(K + 1, n_x)``.

        Uses ``lax.scan``, so the horizon length is a traced loop rather
        than an unrolled Python one and the cost of ``jit`` does not
        grow with the horizon.

        Args:
            x0: initial state, shape ``(n_x,)``.
            u: inputs, shape ``(K, n_u)``.
            w: process noise, shape ``(K, n_w)``.
            theta: model parameters, threaded to ``f`` unchanged.

        Returns:
            States ``x_0 ... x_K``, shape ``(K + 1, n_x)``.
        """
        x0 = jnp.asarray(x0, dtype=jnp.float64)

        def body(x, uw):
            uk, wk = uw
            x_next = self.step(x, uk, wk, theta)
            return x_next, x_next

        _, xs = jax.lax.scan(body, x0, (u, w))
        return jnp.concatenate([x0[None, :], xs], axis=0)

    def jacobians(
        self, x: Array, u: Array, w: Array, theta: Any = None
    ) -> tuple[Array, Array]:
        """``(F, G) = (df/dx, df/dw)`` at a point, by autodiff."""
        f_x = jax.jacobian(lambda xx: self.step(xx, u, w, theta))(x)
        f_w = jax.jacobian(lambda ww: self.step(x, u, ww, theta))(w)
        return f_x, f_w

    def observation_jacobian(
        self, x: Array, u: Array, theta: Any = None
    ) -> Array:
        """``H = dh/dx`` at a point, shape ``(n_y, n_x)``."""
        return jax.jacobian(lambda xx: self.observe(xx, u, theta))(x)

    # -- construction ---------------------------------------------------

    @classmethod
    def from_ode(
        cls,
        rhs: Callable[..., Array],
        h: Callable[..., Array],
        *,
        n_x: int,
        n_y: int,
        dt: float,
        n_u: int = 0,
        n_w: int | None = None,
        method: str = "RK4",
        n_substeps: int = 4,
        noise: Callable[[Array, Array], Array] | Array | None = None,
        x_names: Sequence[str] = (),
        y_names: Sequence[str] = (),
        u_names: Sequence[str] = (),
        lb: Array | Sequence[float] | float | None = None,
        ub: Array | Sequence[float] | float | None = None,
        **kwargs: Any,
    ) -> "StateSpaceModel":
        """Discretise a continuous model over one sampling interval.

        The flow map is obtained from :func:`difflow.dynamic.integrate`,
        so every solver that module offers --- including the diffrax
        backends for stiff systems --- is available to an estimator
        without this package owning an integrator of its own.

        Args:
            rhs: ``rhs(t, x, u, theta) -> dx/dt``.
            h: measurement map ``h(x, u, theta) -> (n_y,)``.
            n_x, n_y, n_u: dimensions.
            dt: sampling interval.
            n_w: process-noise dimension, default ``n_x``.
            method: any ``method`` string accepted by
                :func:`difflow.dynamic.integrate`, e.g. ``"RK4"`` or
                ``"diffrax:kvaerno5"``.
            n_substeps: integration steps per sampling interval, for the
                fixed-step methods.
            noise: how ``w`` enters. ``None`` (default) adds it to the
                state at the end of the interval; an array ``G`` of
                shape ``(n_x, n_w)`` adds ``G @ w``; a callable
                ``(x, w) -> x`` does whatever you say.
            x_names, y_names, u_names, lb, ub: passed to the model.
            **kwargs: forwarded to :func:`difflow.dynamic.integrate`.

        Returns:
            A :class:`StateSpaceModel`.

        Example:
            >>> model = StateSpaceModel.from_ode(       # doctest: +SKIP
            ...     lambda t, x, u, th: -th["k"] * x,
            ...     lambda x, u, th: x,
            ...     n_x=1, n_y=1, dt=0.5,
            ... )
        """
        from difflow.dynamic import integrate

        n_w_ = n_x if n_w is None else int(n_w)

        if noise is None:
            def add_noise(x, w):
                return x + w
        elif callable(noise):
            add_noise = noise
        else:
            g = jnp.asarray(noise, dtype=jnp.float64)

            def add_noise(x, w):
                return x + g @ w

        if method in ("RK4", "Euler"):
            kwargs.setdefault("n_steps", n_substeps)

        def f(x, u, w, theta):
            result = integrate(
                lambda t, y: jnp.asarray(rhs(t, y, u, theta),
                                         dtype=jnp.float64),
                jnp.asarray(x, dtype=jnp.float64),
                (0.0, float(dt)),
                method=method,
                **kwargs,
            )
            return add_noise(result.y_final, w)

        return cls(f=f, h=h, n_x=n_x, n_y=n_y, n_u=n_u, n_w=n_w_, dt=dt,
                   x_names=list(x_names), y_names=list(y_names),
                   u_names=list(u_names), lb=lb, ub=ub)


def augment_parameters(
    model: StateSpaceModel,
    names: Sequence[str],
    *,
    inject: Callable[[Any, Array], Any] | None = None,
    lb: Array | Sequence[float] | float | None = None,
    ub: Array | Sequence[float] | float | None = None,
) -> StateSpaceModel:
    """Append drifting parameters to the state as a random walk.

    Joint state and parameter estimation is the standard way to see
    degradation: the parameter is given no dynamics of its own beyond

    .. math:: p_{k+1} = p_k + w^p_k,

    and the estimator moves it only as far as the data insist, with the
    process-noise standard deviation on ``w^p`` setting how fast it is
    allowed to drift. That number is a modelling choice with real
    consequences --- too large and the parameter absorbs sensor noise
    and the state estimate stops correcting; too small and a genuine
    drift is rejected as noise --- so it is an explicit argument to the
    estimator rather than a default.

    Whether a parameter can be recovered at all from the window is a
    separate question, answered by
    :func:`difflow.mhe.check_observability` before the fact rather than
    by inspecting a NaN afterwards.

    Args:
        model: the base model, whose ``f`` and ``h`` take ``theta``.
        names: names of the parameters to estimate, in order. They
            become the trailing entries of the augmented state.
        inject: ``inject(theta, p) -> theta'`` builds the parameter
            object the base model expects from the fixed part and the
            estimated vector. The default handles the two common cases:
            ``theta=None`` passes ``p`` straight through, and a mapping
            ``theta`` is copied with ``names`` overwritten by ``p``.
        lb, ub: bounds on the parameters, scalar or per parameter. The
            base model's own state bounds are kept.

    Returns:
        A new :class:`StateSpaceModel` with ``n_x + len(names)`` states
        and ``n_w + len(names)`` noise entries.

    Example:
        >>> aug = augment_parameters(model, ["k"], lb=0.0)  # doctest: +SKIP
        >>> aug.n_x == model.n_x + 1                   # doctest: +SKIP
        True
    """
    names = list(names)
    n_p = len(names)
    if n_p == 0:
        raise ValueError("augment_parameters needs at least one parameter")
    if model.n_params:
        raise ValueError(
            "model already carries augmented parameters "
            f"{model.param_names}; augment once, naming them all"
        )
    n_x, n_w = model.n_x, int(model.n_w)

    if inject is None:
        def inject_fn(theta, p):
            if theta is None:
                return p
            if isinstance(theta, Mapping):
                merged = dict(theta)
                merged.update({nm: p[i] for i, nm in enumerate(names)})
                return merged
            raise TypeError(
                "the default parameter injection handles theta=None and "
                f"mapping theta, but got {type(theta).__name__}; pass an "
                "explicit inject=..."
            )
    else:
        inject_fn = inject

    def f_aug(z, u, w, theta):
        x, p = z[:n_x], z[n_x:]
        wx, wp = w[:n_w], w[n_w:]
        x_next = model.step(x, u, wx, inject_fn(theta, p))
        return jnp.concatenate([x_next, p + wp])

    def h_aug(z, u, theta):
        x, p = z[:n_x], z[n_x:]
        return model.observe(x, u, inject_fn(theta, p))

    base_lb, base_ub = model.bounds
    p_lb = (jnp.full((n_p,), -jnp.inf) if lb is None else
            jnp.broadcast_to(jnp.asarray(lb, dtype=jnp.float64), (n_p,)))
    p_ub = (jnp.full((n_p,), jnp.inf) if ub is None else
            jnp.broadcast_to(jnp.asarray(ub, dtype=jnp.float64), (n_p,)))

    return StateSpaceModel(
        f=f_aug,
        h=h_aug,
        n_x=n_x + n_p,
        n_y=model.n_y,
        n_u=model.n_u,
        n_w=n_w + n_p,
        dt=model.dt,
        x_names=list(model.x_names) + names,
        y_names=list(model.y_names),
        u_names=list(model.u_names),
        lb=jnp.concatenate([base_lb, p_lb]),
        ub=jnp.concatenate([base_ub, p_ub]),
        n_params=n_p,
        param_names=names,
    )


def linear_model(
    a: Array,
    c: Array,
    *,
    b: Array | None = None,
    g: Array | None = None,
    **kwargs: Any,
) -> StateSpaceModel:
    """A linear model ``x+ = A x + B u + G w``, ``y = C x``.

    Provided because the linear-Gaussian case is where the estimators
    have an exact answer to be checked against: with no constraints and
    a full-information horizon, moving-horizon estimation and the
    Kalman filter are the same estimator, and this is how that is
    demonstrated.

    Args:
        a: transition matrix, shape ``(n_x, n_x)``.
        c: observation matrix, shape ``(n_y, n_x)``.
        b: input matrix, shape ``(n_x, n_u)``; ``None`` for none.
        g: noise matrix, shape ``(n_x, n_w)``; ``None`` for the
            identity.
        **kwargs: forwarded to :class:`StateSpaceModel` (names, bounds).

    Returns:
        A :class:`StateSpaceModel`.
    """
    a = jnp.asarray(a, dtype=jnp.float64)
    c = jnp.asarray(c, dtype=jnp.float64)
    n_x = a.shape[0]
    n_y = c.shape[0]
    n_u = 0 if b is None else jnp.asarray(b).shape[1]
    n_w = n_x if g is None else jnp.asarray(g).shape[1]

    def f(x, u, w, theta):
        out = a @ x
        if b is not None:
            out = out + jnp.asarray(b, dtype=jnp.float64) @ u
        return out + (w if g is None
                      else jnp.asarray(g, dtype=jnp.float64) @ w)

    def h(x, u, theta):
        return c @ x

    return StateSpaceModel(f=f, h=h, n_x=n_x, n_y=n_y, n_u=int(n_u),
                           n_w=int(n_w), **kwargs)
