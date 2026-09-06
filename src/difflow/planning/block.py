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


_QUANTITY_UNITS = {"total_flow": "mol/s", "T": "K", "P": "Pa"}


def _sanitize(key: str) -> str:
    """Turn a flowsheet key into a bare variable name.

    ``"feed:F1.total_flow"`` -> ``"feed_F1_total_flow"``.  Qualified names
    become LP column names, so the separators that difflow uses to address
    units, feeds and streams have to go.
    """
    out = []
    for ch in key:
        out.append(ch if (ch.isalnum() or ch == "_") else "_")
    name = "".join(out).strip("_")
    return name or "var"


def _quantity_units(quantity: str) -> str | None:
    """Physical unit of a stream quantity, or ``None`` if unknown."""
    if quantity in _QUANTITY_UNITS:
        return _QUANTITY_UNITS[quantity]
    if quantity.startswith("F_"):
        return "mol/s"
    if quantity.startswith("x_"):
        return "mol/mol"
    return None


def _stream_quantity(stream, quantity: str):
    """Read one scalar quantity out of a solved stream.

    Recognised: ``total_flow``, ``T``, ``P``, ``F_<species>``,
    ``x_<species>``.  Pure ``jnp``, so it differentiates.
    """
    if quantity == "total_flow":
        return sum(v for k, v in stream.items() if k.startswith("F_"))
    if quantity in ("T", "P"):
        return jnp.asarray(stream[quantity], dtype=float)
    if quantity.startswith("x_"):
        key = f"F_{quantity[2:]}"
        if key not in stream:
            raise KeyError(f"stream has no species {quantity[2:]!r}")
        total = sum(v for k, v in stream.items() if k.startswith("F_"))
        return jnp.asarray(stream[key], dtype=float) / total
    if quantity in stream:
        return jnp.asarray(stream[quantity], dtype=float)
    raise KeyError(
        f"Unknown stream quantity {quantity!r}. Use 'total_flow', 'T', 'P', "
        f"'F_<species>' or 'x_<species>'"
    )


def _flowsheet_param_units(flowsheet, key: str) -> str | None:
    """Units of a ``"<unit>.<param>"`` lever, from the dataclass metadata.

    Reads the same ``field(metadata={"units": ...})`` that
    :class:`~difflow.catalog.ParameterSpec` serves to the GUI, so a delta-vector
    export becomes self-describing as the metadata rollout lands.  Returns
    ``None`` when the field carries no metadata, which is the state of most of
    the catalog today.
    """
    import dataclasses

    unit_name, _, param_name = key.partition(".")
    for unit in flowsheet.units:
        if unit.name != unit_name:
            continue
        params = getattr(unit.operation, "params", None)
        if params is None or not dataclasses.is_dataclass(params):
            return None
        for f in dataclasses.fields(params):
            if f.name == param_name:
                return f.metadata.get("units")
    return None


def _current_value(flowsheet, key: str) -> float:
    """The flowsheet's present value for a lever key, for use as ``u0``."""
    from difflow.flowsheet import FEED_PREFIX

    if key.startswith(FEED_PREFIX):
        feed_name, _, field_name = key[len(FEED_PREFIX):].partition(".")
        if feed_name not in flowsheet.feeds:
            raise KeyError(
                f"No feed stream named {feed_name!r} in flowsheet. "
                f"Available feeds: {list(flowsheet.feeds.keys())}")
        return float(_stream_quantity(flowsheet.feeds[feed_name], field_name))

    unit_name, _, param_name = key.partition(".")
    for unit in flowsheet.units:
        if unit.name == unit_name:
            params = getattr(unit.operation, "params", None)
            if params is None or not hasattr(params, param_name):
                raise KeyError(
                    f"Unit {unit_name!r} has no parameter {param_name!r}")
            return float(jnp.asarray(getattr(params, param_name)))
    raise KeyError(
        f"No unit named {unit_name!r} in flowsheet. "
        f"Available units: {[u.name for u in flowsheet.units]}")


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

    # -- construction from a flowsheet ------------------------------------

    @classmethod
    def from_flowsheet(cls, flowsheet, u: Sequence[str], y: Sequence[str],
                       name: str = "flowsheet",
                       lb: Any = None, ub: Any = None, u0: Any = None,
                       u_names: Sequence[str] | None = None,
                       y_names: Sequence[str] | None = None,
                       solve_kwargs: Mapping[str, Any] | None = None,
                       **kwargs: Any) -> "Block":
        """Build a planning block from a difflow flowsheet.

        This is the bridge that was missing between simulation and planning.
        The returned block's ``fn`` applies the chosen levers to a *copy* of
        the flowsheet, solves it, and reads back the chosen outputs.  Because
        the whole path is pure JAX, ``jax.jacobian`` of that callable is the
        reduced input-output sensitivity of the flowsheet — implicitly
        differentiated through the recycle tear solve and every inner unit
        solve.  That matrix is the delta vector an LP planning model wants.

        Args:
            flowsheet: A :class:`~difflow.flowsheet.Flowsheet`.  Not modified;
                each evaluation works on a copy.
            u: Lever keys in :meth:`~difflow.flowsheet.Flowsheet._apply_params`
                notation: ``"<unit>.<param>"`` or ``"feed:<stream>.<field>"``.
            y: Output keys, ``"<stream>.<quantity>"`` where quantity is
                ``total_flow``, ``T``, ``P``, ``F_<species>`` or
                ``x_<species>``.
            name: Block name.
            lb: Lower bounds on the levers.  Defaults to unbounded, but a
                planning LP wants real ones.
            ub: Upper bounds on the levers.
            u0: Linearisation point.  Defaults to the flowsheet's *current*
                values, which is almost always the base case you want.
            u_names: Override the derived bare names for the levers.
            y_names: Override the derived bare names for the outputs.
            solve_kwargs: Passed through to :meth:`Flowsheet.solve`.
            **kwargs: Forwarded to :class:`Block` (``jit``, ``ad_mode``,
                ``phase_fn``, ``metadata``, ...).

        Returns:
            A :class:`Block` whose ``metadata`` records the original keys and
            the units of every variable, so the delta-vector export is
            self-describing.

        Raises:
            KeyError: If a lever or output key does not resolve.

        Example:
            >>> blk = Block.from_flowsheet(       # doctest: +SKIP
            ...     fs,
            ...     u=["reactor.V", "feed:feed.total_flow"],
            ...     y=["product.F_B", "product.total_flow"],
            ...     lb=[0.5, 5.0], ub=[5.0, 20.0], name="plant")
        """
        u_keys = list(u)
        y_keys = list(y)
        if not u_keys:
            raise ValueError("from_flowsheet needs at least one lever in u")
        if not y_keys:
            raise ValueError("from_flowsheet needs at least one output in y")

        solve_kw = dict(solve_kwargs or {})
        parsed_y = []
        for key in y_keys:
            stream_name, sep, quantity = key.rpartition(".")
            if not sep:
                raise ValueError(
                    f"Output key {key!r} must use '<stream>.<quantity>' "
                    "notation")
            parsed_y.append((stream_name, quantity))

        # Resolve u0 from the flowsheet's current state before building fn,
        # so a failure here is reported at construction, not at solve time.
        if u0 is None:
            u0 = [_current_value(flowsheet, k) for k in u_keys]

        def fn(u_arr):
            params = {key: u_arr[i] for i, key in enumerate(u_keys)}
            streams = flowsheet._apply_params(params).solve(**solve_kw)
            missing = [s for s, _ in parsed_y if s not in streams]
            if missing:
                raise KeyError(
                    f"Solved flowsheet has no stream(s) {sorted(set(missing))}. "
                    f"Available: {sorted(streams)}")
            return jnp.stack([
                jnp.asarray(_stream_quantity(streams[s], q), dtype=float)
                for s, q in parsed_y
            ])

        derived_u = list(u_names) if u_names else [_sanitize(k) for k in u_keys]
        derived_y = list(y_names) if y_names else [_sanitize(k) for k in y_keys]

        metadata = dict(kwargs.pop("metadata", {}) or {})
        metadata.setdefault("source", "flowsheet")
        metadata.setdefault("u_keys", u_keys)
        metadata.setdefault("y_keys", y_keys)
        metadata.setdefault("u_units", [
            _quantity_units(k.rpartition(".")[2])
            if k.startswith("feed:")
            else _flowsheet_param_units(flowsheet, k)
            for k in u_keys
        ])
        metadata.setdefault("y_units", [_quantity_units(q) for _, q in parsed_y])

        return cls(name=name, fn=fn, u_names=derived_u, y_names=derived_y,
                   lb=lb, ub=ub, u0=u0, metadata=metadata, **kwargs)

    def __repr__(self) -> str:
        return (f"Block(name={self.name!r}, n_u={self.n_u}, n_y={self.n_y}, "
                f"ad_mode={self.ad_mode!r}, jit={self.jit})")
