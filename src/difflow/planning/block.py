"""Planning blocks: the unit of linearisation in a delta-base plan.

A :class:`Block` wraps *any* pure JAX callable ``u -> y``.  In practice that
callable is a difflow flowsheet: because a flowsheet embeds its own flash,
recycle and unit solves, ``jax.jacobian`` of the callable returns the
*reduced* input-output sensitivity, already implicitly differentiated through
those inner solves.  That reduced Jacobian is precisely the "delta vector"
that refinery planning systems (Aspen PIMS, Haverly GRTMPS, Honeywell RPMS,
AVEVA Spiral Plan) obtain by perturbing a rigorous simulator once per
decision variable.

The distinction matters asymptotically.  One-at-a-time perturbation costs
``O(n)`` model evaluations for ``n`` decisions; reverse-mode AD costs ``O(1)``.
See :mod:`difflow.planning.linearize`.

Example:
    >>> import jax.numpy as jnp
    >>> from difflow.planning import Block
    >>> def outputs(u):
    ...     recovery, split = u
    ...     return jnp.array([recovery * split, 1.0 - recovery * split])
    >>> blk = Block(name="sep", fn=outputs,
    ...             u_names=["recovery", "split"],
    ...             y_names=["product", "residue"],
    ...             lb=[0.0, 0.0], ub=[1.0, 1.0])
    >>> blk.evaluate(jnp.array([0.8, 0.5]))
    Array([0.4, 0.6], dtype=float64)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin


def _as_array(values: Sequence[float] | Array | None, n: int, default: float,
              what: str) -> Array:
    """Coerce a bound-like argument to a float array of length ``n``."""
    if values is None:
        return jnp.full((n,), default)
    arr = jnp.atleast_1d(jnp.asarray(values, dtype=float))
    if arr.shape == (1,) and n != 1:
        arr = jnp.full((n,), arr[0])
    if arr.shape != (n,):
        raise ValueError(
            f"{what} has shape {tuple(arr.shape)}, expected ({n},)")
    return arr


@dataclass
class Block(ParamsMixin):
    """One linearisable submodel in a planning network.

    Attributes:
        name: Block name.  Variables are addressed globally as
            ``"<block>.<variable>"``.
        fn: Pure JAX callable.  Either ``fn(u) -> y`` or, when ``theta`` is
            given, ``fn(u, theta) -> y``.  ``u`` is an array ordered like
            ``u_names``; the return is an array ordered like ``y_names``
            (a dict keyed by ``y_names`` is also accepted).
        u_names: Names of the block inputs (decisions and linked inlets).
        y_names: Names of the block outputs.
        lb: Lower bounds on ``u``, length ``len(u_names)``.
        ub: Upper bounds on ``u``.
        u0: Nominal operating point.  Defaults to the midpoint of the bounds
            and is used as the cold start for the trust-region loop.
        theta: Optional dict of model/design parameters.  When present ``fn``
            is called as ``fn(u, theta)`` and the parameters become available
            to :meth:`~difflow.planning.planner.PlanResult.plan_sensitivity`.
        phase_fn: Optional callable ``u -> Array`` (or ``(u, theta) -> Array``)
            returning phase indicators, typically vapour fractions.  Used to
            detect linearisations that straddle a phase boundary.
        phase_names: Names of the indicators returned by ``phase_fn``.
        phase_bounds: Interior thresholds that separate phase regimes.  The
            default ``(0.0, 1.0)`` bins a vapour fraction into
            subcooled / two-phase / superheated.
        ad_mode: ``"auto"`` (choose by shape), ``"rev"`` or ``"fwd"``.
        jit: JIT-compile ``fn`` and ``phase_fn``.  Worth it whenever the block
            is a flowsheet with an inner solve, which is the usual case — the
            planner calls it once per trust-region cycle and once per AD pass.
            Requires ``fn`` to be traceable, which it must be for AD anyway.

    Note:
        ``fn`` must be a *pure* function of ``u`` (and ``theta``).  Any inner
        iteration must be JAX-traceable — ``optimistix`` root finds and
        ``diffrax`` integrations are, and difflow's units are built on them.
    """

    name: str
    fn: Callable[..., Array | Mapping[str, Array]]
    u_names: list[str]
    y_names: list[str]
    lb: Any = None
    ub: Any = None
    u0: Any = None
    theta: dict[str, Any] | None = None
    phase_fn: Callable[..., Array] | None = None
    phase_names: tuple[str, ...] = ()
    phase_bounds: tuple[float, ...] = (0.0, 1.0)
    ad_mode: str = "auto"
    jit: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.u_names:
            raise ValueError(f"Block {self.name!r} has no inputs")
        if not self.y_names:
            raise ValueError(f"Block {self.name!r} has no outputs")
        if len(set(self.u_names)) != len(self.u_names):
            raise ValueError(f"Block {self.name!r} has duplicate u_names")
        if len(set(self.y_names)) != len(self.y_names):
            raise ValueError(f"Block {self.name!r} has duplicate y_names")
        overlap = sorted(set(self.u_names) & set(self.y_names))
        if overlap:
            raise ValueError(
                f"Block {self.name!r} uses {overlap} as both an input and an "
                "output name. Qualified names must be unique within a block "
                "because they become LP column names; rename one side "
                "(e.g. 'T_in' / 'T_out').")
        if "." in self.name:
            raise ValueError(
                f"Block name {self.name!r} may not contain '.' — it is the "
                "separator for qualified variable names")
        if self.ad_mode not in ("auto", "rev", "fwd"):
            raise ValueError(
                f"ad_mode must be 'auto', 'rev' or 'fwd', got {self.ad_mode!r}")

        n = self.n_u
        self.lb = _as_array(self.lb, n, -jnp.inf, f"Block {self.name!r} lb")
        self.ub = _as_array(self.ub, n, jnp.inf, f"Block {self.name!r} ub")
        if bool(jnp.any(self.ub < self.lb)):
            raise ValueError(f"Block {self.name!r} has ub < lb")

        if self.u0 is None:
            finite = jnp.isfinite(self.lb) & jnp.isfinite(self.ub)
            mid = jnp.where(finite, 0.5 * (self.lb + self.ub), 0.0)
            self.u0 = mid
        else:
            self.u0 = _as_array(self.u0, n, 0.0, f"Block {self.name!r} u0")

        if self.phase_fn is not None and not self.phase_names:
            # Names are only used for diagnostics; synthesise if not given.
            self.phase_names = ("phase",)

        self._fn = jax.jit(self.fn) if self.jit else self.fn
        self._phase_fn = (jax.jit(self.phase_fn)
                          if self.jit and self.phase_fn is not None
                          else self.phase_fn)
        self._jac_cache: dict[str, Any] = {}

    # -- shape helpers ---------------------------------------------------

    @property
    def n_u(self) -> int:
        """Number of block inputs."""
        return len(self.u_names)

    @property
    def n_y(self) -> int:
        """Number of block outputs."""
        return len(self.y_names)

    @property
    def range(self) -> Array:
        """Width of each input's bound interval (``inf`` where unbounded)."""
        return self.ub - self.lb

    def qualified_u(self) -> list[str]:
        """Globally qualified input names, ``"<block>.<u>"``."""
        return [f"{self.name}.{u}" for u in self.u_names]

    def qualified_y(self) -> list[str]:
        """Globally qualified output names, ``"<block>.<y>"``."""
        return [f"{self.name}.{y}" for y in self.y_names]

    def u_index(self, name: str) -> int:
        """Index of an input by bare or qualified name."""
        bare = name.split(".", 1)[1] if name.startswith(f"{self.name}.") else name
        try:
            return self.u_names.index(bare)
        except ValueError:
            raise KeyError(f"{name!r} is not an input of block {self.name!r}")

    def y_index(self, name: str) -> int:
        """Index of an output by bare or qualified name."""
        bare = name.split(".", 1)[1] if name.startswith(f"{self.name}.") else name
        try:
            return self.y_names.index(bare)
        except ValueError:
            raise KeyError(f"{name!r} is not an output of block {self.name!r}")

    # -- evaluation ------------------------------------------------------

    def evaluate(self, u: Array, theta: Mapping[str, Any] | None = None) -> Array:
        """Evaluate the block's nonlinear model.

        Args:
            u: Input array ordered like ``u_names``.
            theta: Override for ``self.theta``.  Ignored when the block
                declares no parameters.

        Returns:
            Output array ordered like ``y_names``.
        """
        u = jnp.atleast_1d(jnp.asarray(u, dtype=float))
        raw = self._call(self._fn, u, theta)
        return self._pack(raw, self.y_names, "outputs")

    def evaluate_phases(self, u: Array,
                        theta: Mapping[str, Any] | None = None) -> Array | None:
        """Evaluate the block's phase indicators, or ``None`` if undeclared."""
        if self.phase_fn is None:
            return None
        u = jnp.atleast_1d(jnp.asarray(u, dtype=float))
        raw = self._call(self._phase_fn, u, theta)
        if isinstance(raw, Mapping):
            return self._pack(raw, list(self.phase_names), "phase indicators")
        return jnp.atleast_1d(jnp.asarray(raw, dtype=float))

    def _call(self, fn: Callable, u: Array, theta: Mapping | None):
        """Dispatch on whether the block carries parameters."""
        params = self.theta if theta is None else theta
        if params is None:
            return fn(u)
        return fn(u, params)

    def _pack(self, raw, names: list[str], what: str) -> Array:
        """Coerce a dict-or-array return value into a positional array."""
        if isinstance(raw, Mapping):
            missing = [k for k in names if k not in raw]
            if missing:
                raise KeyError(
                    f"Block {self.name!r} {what} missing keys {missing}")
            return jnp.stack([jnp.asarray(raw[k], dtype=float) for k in names])
        arr = jnp.atleast_1d(jnp.asarray(raw, dtype=float))
        if what == "outputs" and arr.shape != (len(names),):
            raise ValueError(
                f"Block {self.name!r} returned shape {tuple(arr.shape)}, "
                f"expected ({len(names)},) to match y_names")
        return arr

    def clip(self, u: Array) -> Array:
        """Clip an input vector to the block's bounds."""
        return jnp.clip(jnp.asarray(u, dtype=float), self.lb, self.ub)

    def jacobian(self, mode: str) -> Callable:
        """A cached (and, when ``jit``, compiled) Jacobian callable.

        The cache matters because the planner rebuilds the delta vectors on
        every trust-region cycle; recompiling each time would charge the
        compilation cost once per iteration instead of once per run.
        """
        key = f"{mode}:{self.jit}"
        cached = self._jac_cache.get(key)
        if cached is None:
            def f(u, theta=None):
                return self.evaluate(u, theta)

            raw = jax.jacrev(f) if mode == "rev" else jax.jacfwd(f)
            cached = jax.jit(raw) if self.jit else raw
            self._jac_cache[key] = cached
        return cached

    def __repr__(self) -> str:
        return (f"Block(name={self.name!r}, n_u={self.n_u}, n_y={self.n_y}, "
                f"ad_mode={self.ad_mode!r}, jit={self.jit})")
