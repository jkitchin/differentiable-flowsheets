"""Flat NLP view of a difflow flowsheet.

:func:`as_nlp` turns a :class:`~difflow.flowsheet.Flowsheet` plus a set of
decision variables and specifications into the three objects an NLP solver
wants::

    f, g, bounds = as_nlp(flowsheet, decisions, specs, objective=profit)

``f(x)`` is the objective, ``g(x)`` the constraint body, both JAX-traceable
in float64 -- exactly the input contract of ``pounce.jax.from_jax``.

The formulation is **equation-oriented**. The variable vector is

.. code-block:: text

    x = [ decisions | stream variables ]

where the stream block is the state vector of
:class:`~difflow.eo_solver.EOStateLayout` -- every non-feed stream
contributes ``[F_s1, ..., F_sn, T, P]``. The constraint body is

.. code-block:: text

    g(x) = [ unit residuals (= 0) | specification bodies (in [lo, hi]) ]

and the unit residuals come from :meth:`difflow.eo_solver.EOSolver.
_build_residual_fn`, i.e. from each unit's ``eo_residuals``. Nothing is
re-derived here: the equations solved by ``flowsheet.solve_eo()`` are the
equality constraints of the NLP, so a feasible point of the NLP is a
converged flowsheet by construction.

Why equation-oriented and not sequential-modular
------------------------------------------------
A sequential-modular call (``cstr(inlet, T_spec=...)``) closes its material
balance with an inner ``optimistix.Newton``. That is perfectly
differentiable, but it is opaque to *structural* analysis: the inner solve
emits a ``linear_solve`` primitive, and it couples every variable in its
block, so the sequential form is both harder to trace and *denser* than the
residual form. Promoting the stream variables to decision variables makes
every equation explicit and the Jacobian block-banded by topology, which is
what the sparsity detection below exploits -- and ``asdex`` has no handler
for the ``linear_solve`` an inner Newton emits, so the sequential form
cannot be analyzed at all.

One trap when writing the residual form by hand rather than through
:func:`as_nlp`: ``eo_residuals`` returns ``n_species`` material balances
plus a ``T`` row and a ``P`` row. Here those are real equations, because
``T`` and ``P`` are stream *variables*. Promote ``T`` to a decision variable
directly and hold ``P`` fixed -- as the hand-written reactor train in
``examples/27_pounce_optimization.ipynb`` does -- and the last two rows
collapse to ``0 = 0``; they must be sliced off or the constraint Jacobian
gets exactly-zero rows.

Sparsity is derived, never probed
---------------------------------
The single most important thing this module does is hand the solver a
sparsity pattern instead of letting the solver find one by probing.

``pounce.jax.from_jax`` (and ``JaxProblem``) detect sparsity by evaluating
derivatives at random :math:`\\mathcal{N}(0, 1)` points, which have nothing
to do with ``x0`` or the bounds. For a process model that means evaluating
at, say, ``T = -1.3 K``: Arrhenius terms overflow, reactor linear solves go
singular, and ``nan > eps`` is ``False``, so a whole column of real
derivatives is recorded as structurally zero. **Every difflow model hits
this.** So the adapters here always supply ``jac_pattern`` and
``hess_pattern`` and never fall through to probing.

By default the pattern comes from **global graph analysis** (:mod:`asdex`):
index sets propagated through the jaxpr of ``g``, and of the Lagrangian for
the Hessian, with no derivative evaluated anywhere, so the answer holds at
every point. On difflow's equation-oriented residuals it is also tight --
exactly the entries that are nonzero at a feasible point.

The topology derivation is the fallback (``sparsity="structural"``): a
unit's residual rows can only touch the stream variables of its own inlets
and outlets, plus the decisions that reach that unit. That is a superset by
construction, but it is blind to the objective and to callable spec bodies,
which makes the Lagrangian Hessian dense -- ``n`` colors per evaluation, and
:math:`n^2` growth where the true structure grows like :math:`n`. Dense is
never a default anywhere in this package; ``sparsity="dense"`` is how you
ask for it. :mod:`difflow.solvers.sparsity` has the numbers.

The contract on any pattern is that it must be a **superset** of the true
structure. Extra entries merely report a zero and may cost an extra color; a
*missing* entry is silently wrong -- on the dense path the derivative is
dropped, and under ``sparse=True`` it aliases into a same-colored entry and
corrupts that one too. pounce never checks, so
:func:`~difflow.solvers.sparsity.validate_patterns` does, and it runs by
default: exactly against dense AD for a small problem, column by column
against JVPs for a large one. Point verification is necessary, not
sufficient; the derivation is what makes a pattern valid everywhere.

One numerical subtlety in that verification, which is easy to get wrong:
the Lagrangian Hessian must be checked at *random* multipliers, not at
``lambda = 1``. A unit's material balances share one reaction term whose
stoichiometric coefficients sum to zero over the species, so weighting the
rows equally cancels the nonlinearity exactly and gives ``H = 0``, at which
point any pattern passes -- including an empty one. Graph analysis is immune
(reachability does not cancel); the numerical check works around it.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.eo_solver import EOSolver, EOStateLayout
from difflow.flowsheet import Flowsheet
from difflow.solvers.sparsity import (
    SparsityDetectionError,
    SparsityPatternError,
    dense_hessian_pattern,
    dense_jacobian_pattern,
    detect_hessian_pattern,
    detect_jacobian_pattern,
    pattern_density,
    validate_patterns,
)
from difflow.streams import Stream

__all__ = [
    "Decision",
    "Parameter",
    "Spec",
    "Bounds",
    "SparsityDetectionError",
    "SparsityPatternError",
    "as_nlp",
    "dense_jacobian_pattern",
    "dense_hessian_pattern",
    "detect_jacobian_pattern",
    "detect_hessian_pattern",
    "require_eo_residuals",
    "validate_patterns",
]

#: Stand-in for an infinite bound. Interior-point codes treat anything past
#: ~1e19 as "no bound"; a finite sentinel keeps every array float64-clean.
BIG = 1.0e20

#: Default box for stream variables, by kind. Temperature is deliberately
#: *not* unbounded: an optimizer that wanders to T < 0 overflows every
#: Arrhenius term in the model long before it discovers that the point is
#: bad, which is the same failure mode as random sparsity probing.
DEFAULT_FLOW_BOUNDS = (0.0, BIG)
DEFAULT_T_BOUNDS = (200.0, 1000.0)
DEFAULT_P_BOUNDS = (1.0, 1.0e9)


# ---------------------------------------------------------------------------
# Problem description
# ---------------------------------------------------------------------------


@dataclass
class Decision:
    """One decision variable of the design problem.

    Attributes:
        name: Label; also the default ``address``.
        lb: Lower bound.
        ub: Upper bound.
        x0: Starting value.
        address: Where the value is written into the flowsheet. See
            :func:`as_nlp` for the address grammar. Ignored when
            ``flowsheet`` is a builder callable, which receives the value
            under ``name``.

    Example:
        >>> Decision("unit:reactor.params.V", lb=0.01, ub=5.0, x0=0.5)
        Decision(name='unit:reactor.params.V', ...)
    """

    name: str
    lb: float = -BIG
    ub: float = BIG
    x0: float = 0.0
    address: str | None = None

    def __post_init__(self):
        if self.address is None:
            self.address = self.name
        if self.lb > self.ub:
            raise ValueError(f"decision {self.name!r}: lb {self.lb} > ub {self.ub}")


@dataclass
class Parameter:
    """A flowsheet quantity held fixed in the NLP but differentiable outside it.

    Parameters become the ``p`` argument of ``f(x, p)`` / ``g(x, p)``, which
    is what ``pounce.jax.JaxProblem`` differentiates the solve with respect
    to. They are *not* columns of ``x``.

    Attributes:
        name: Label; also the default ``address``.
        value: Nominal value, used for the ``p0`` in :class:`Bounds` and for
            the single-argument calls ``f(x)`` / ``g(x)``.
        address: Where the value is written into the flowsheet.
    """

    name: str
    value: float = 0.0
    address: str | None = None

    def __post_init__(self):
        if self.address is None:
            self.address = self.name


@dataclass
class Spec:
    """A specification: one row of ``g`` with bounds ``[lo, hi]``.

    Attributes:
        name: Label, used in :attr:`Bounds.con_names` and in reports.
        fn: ``fn(streams, decisions) -> scalar``, JAX-traceable. ``streams``
            maps stream name to :class:`~difflow.streams.Stream`; ``decisions``
            maps decision name to its scalar value.
        lo: Lower bound on the body (``-BIG`` for none).
        hi: Upper bound on the body (``BIG`` for none). ``lo == hi`` makes it
            an equality.
        variables: Optional list of variable names the body touches. Supplying
            it tightens the Jacobian row (and hence the Hessian); leaving it
            ``None`` makes the row dense, which is always a valid superset.

    Example:
        >>> spec = Spec.parse(("product.F_B", ">=", 8.0))
        >>> spec.lo, spec.hi
        (8.0, 1e+20)
    """

    name: str
    fn: Callable[[dict[str, Stream], dict[str, Array]], Array]
    lo: float = -BIG
    hi: float = BIG
    variables: list[str] | None = None

    @classmethod
    def parse(cls, obj: Any) -> "Spec":
        """Coerce a shorthand into a :class:`Spec`.

        Accepts a :class:`Spec` unchanged, or a 3-tuple
        ``(target, op, value)`` where ``op`` is ``"<="``, ``">="`` or
        ``"=="`` and ``target`` is either a callable ``fn(streams,
        decisions)`` or a stream address ``"<stream>.<key>"`` (``key`` is
        ``T``, ``P`` or ``F_<species>``).

        Args:
            obj: The spec or shorthand.

        Returns:
            A :class:`Spec`.

        Raises:
            ValueError: On an unknown operator or malformed tuple.
        """
        if isinstance(obj, Spec):
            return obj
        if not (isinstance(obj, tuple) and len(obj) == 3):
            raise ValueError(
                "a spec must be a Spec or a (target, op, value) 3-tuple; "
                f"got {obj!r}"
            )
        target, op, value = obj
        value = float(value)
        if callable(target):
            name = getattr(target, "__name__", "spec")
            fn, variables = target, None
        else:
            name = str(target)
            fn = _stream_getter(name)
            variables = [name]
        if op == "<=":
            lo, hi = -BIG, value
        elif op == ">=":
            lo, hi = value, BIG
        elif op in ("==", "="):
            lo, hi = value, value
        else:
            raise ValueError(f"unknown spec operator {op!r}; use <=, >= or ==")
        return cls(name=f"{name} {op} {value:g}", fn=fn, lo=lo, hi=hi,
                   variables=variables)


def _stream_getter(address: str) -> Callable:
    """Build ``fn(streams, decisions)`` reading ``"<stream>.<key>"``."""
    if "." not in address:
        raise ValueError(
            f"stream address {address!r} must look like '<stream>.<key>', "
            "e.g. 'product.F_B' or 'reactor_out.T'"
        )
    stream_name, key = address.rsplit(".", 1)

    def getter(streams, decisions):
        try:
            stream = streams[stream_name]
        except KeyError:
            raise KeyError(
                f"spec refers to stream {stream_name!r}, which is not in the "
                f"flowsheet. Known streams: {sorted(streams)}"
            ) from None
        return jnp.reshape(jnp.asarray(stream[key]), ())

    getter.__name__ = f"get_{stream_name}_{key}"
    return getter


@dataclass
class Bounds:
    """Everything a solver needs beyond ``f`` and ``g``.

    Named ``Bounds`` because bounds are its core content, but it also
    carries the two sparsity patterns: they are part of the same handoff,
    and keeping them here is what lets the pounce wrapper guarantee it never
    probes.

    Attributes:
        lb, ub: Variable bounds, shape ``(n,)``.
        cl, cu: Constraint bounds, shape ``(m,)``. Residual rows are
            ``0 <= g_i <= 0``.
        x0: Starting point, shape ``(n,)``, clipped into ``[lb, ub]``.
        p0: Nominal parameter vector, shape ``(n_parameters,)``.
        var_names: Names of the ``n`` variables, decisions first.
        con_names: Names of the ``m`` constraints, residuals first.
        jac_pattern: ``(rows, cols)`` int arrays, cyipopt convention, for the
            ``(m, n)`` constraint Jacobian. A superset of the true structure.
        hess_pattern: ``(rows, cols)`` for the lower triangle of the
            ``(n, n)`` Lagrangian Hessian. pounce folds upper-triangle
            entries onto their mirror, so a full symmetric pattern is also
            accepted.
        n_decisions: Length of the decision block at the head of ``x``.
        layout: The :class:`~difflow.eo_solver.EOStateLayout` for the stream
            block, so ``x[n_decisions:]`` can be unpacked into streams.
        feeds: The reference flowsheet's feed streams.
        feed_decisions: ``(index, feed, key)`` for each ``feed:``-addressed
            decision, used by :meth:`unpack` to report moved feeds correctly.
    """

    lb: Array
    ub: Array
    cl: Array
    cu: Array
    x0: Array
    p0: Array
    var_names: list[str]
    con_names: list[str]
    jac_pattern: tuple[np.ndarray, np.ndarray]
    hess_pattern: tuple[np.ndarray, np.ndarray]
    n_decisions: int
    layout: EOStateLayout
    feeds: dict[str, Stream] = field(default_factory=dict)
    # (decision index, feed name, stream key) for every feed-addressed decision,
    # so unpack can overlay the optimizer's values onto the reference feeds
    # instead of reporting them at their starting values (#207 review).
    feed_decisions: tuple[tuple[int, str, str], ...] = ()
    #: Where the patterns came from: "global" (asdex graph analysis),
    #: "structural" (flowsheet topology) or "dense". Reported by __repr__,
    #: because a model that quietly fell back to dense is a model whose
    #: Hessian costs n colors.
    sparsity_source: str = "global"

    @property
    def n(self) -> int:
        """Number of variables."""
        return len(self.var_names)

    @property
    def m(self) -> int:
        """Number of constraints."""
        return len(self.con_names)

    def unpack(self, x: Array) -> tuple[dict[str, Array], dict[str, Stream]]:
        """Split ``x`` into decision values and the full streams dict.

        Args:
            x: Variable vector, shape ``(n,)``.

        Returns:
            ``(decisions, streams)`` where ``decisions`` maps decision name to
            scalar and ``streams`` includes the feeds.
        """
        x = jnp.asarray(x)
        dvals = {nm: x[i] for i, nm in enumerate(self.var_names[: self.n_decisions])}
        # `feeds` is frozen at the reference flowsheet, but a decision may be
        # addressed as `feed:<stream>.<key>`. Overlay those here, or a feed the
        # optimizer moved is reported at its starting value while the objective
        # and residuals -- which see the rebuilt flowsheet -- used the new one
        # (#207 review).
        streams = {name: dict(stream) for name, stream in self.feeds.items()}
        for i, owner, attr in self.feed_decisions:
            if owner in streams:
                streams[owner][attr] = x[i]
        streams.update(self.layout.unpack(x[self.n_decisions:]))
        return dvals, streams

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        jac_nnz = len(self.jac_pattern[0])
        hess_nnz = len(self.hess_pattern[0])
        return (
            f"Bounds(n={self.n}, m={self.m}, n_decisions={self.n_decisions}, "
            f"jac_nnz={jac_nnz} ({100 * self.jac_density:.2g}% of dense), "
            f"hess_nnz={hess_nnz}, sparsity={self.sparsity_source!r})"
        )

    @property
    def jac_density(self) -> float:
        """Fraction of the dense ``(m, n)`` Jacobian the pattern occupies."""
        return pattern_density(self.jac_pattern, self.m, self.n)

    @property
    def hess_density(self) -> float:
        """Fraction of the dense lower triangle the Hessian pattern occupies."""
        tri = self.n * (self.n + 1) // 2
        return 0.0 if tri == 0 else len(self.hess_pattern[0]) / tri


# ---------------------------------------------------------------------------
# Address grammar
# ---------------------------------------------------------------------------


def _split_address(address: str) -> tuple[str, str, str]:
    """Parse a decision/parameter address.

    Grammar::

        unit:<unit>.<kwarg>            -> flowsheet Unit.params[kwarg]
        unit:<unit>.params.<field>     -> the operation's Params dataclass field
        feed:<stream>.<key>            -> a feed stream entry (T, P, F_<species>)
        <unit>.<rest>                  -> sugar for unit:<unit>.<rest>

    Returns:
        ``(kind, owner, attr)`` with ``kind`` in
        ``{"unit_kwarg", "unit_param", "feed"}``.

    Raises:
        ValueError: If the address does not parse.
    """
    if address.startswith("feed:"):
        rest = address[len("feed:"):]
        if "." not in rest:
            raise ValueError(
                f"feed address {address!r} must be 'feed:<stream>.<key>'"
            )
        stream, key = rest.rsplit(".", 1)
        return "feed", stream, key
    rest = address[len("unit:"):] if address.startswith("unit:") else address
    if "." not in rest:
        raise ValueError(
            f"address {address!r} must be 'unit:<unit>.<kwarg>', "
            "'unit:<unit>.params.<field>' or 'feed:<stream>.<key>'"
        )
    head, tail = rest.split(".", 1)
    if tail.startswith("params."):
        return "unit_param", head, tail[len("params."):]
    if "." in tail:
        raise ValueError(f"address {address!r} has too many dots")
    return "unit_kwarg", head, tail


def _apply(fs: Flowsheet, address: str, value) -> None:
    """Write ``value`` into a *copy* of the flowsheet, in place."""
    kind, owner, attr = _split_address(address)
    if kind == "feed":
        if owner not in fs.feeds:
            raise KeyError(
                f"address {address!r}: no feed named {owner!r}; "
                f"feeds are {sorted(fs.feeds)}"
            )
        stream = dict(fs.feeds[owner])
        if attr not in stream:
            raise KeyError(
                f"address {address!r}: feed {owner!r} has no key {attr!r}; "
                f"keys are {sorted(stream)}"
            )
        stream[attr] = jnp.asarray(value)
        fs.feeds[owner] = stream
        return

    for i, unit in enumerate(fs.units):
        if unit.name != owner:
            continue
        if kind == "unit_kwarg":
            params = dict(unit.params)
            params[attr] = jnp.asarray(value)
            fs.units[i] = replace(unit, params=params)
        else:
            op = unit.operation
            if not hasattr(op, "params"):
                raise AttributeError(
                    f"address {address!r}: unit {owner!r} operation "
                    f"{type(op).__name__} has no .params dataclass"
                )
            new_op = copy.copy(op)
            new_op.params = op.params.update(**{attr: jnp.asarray(value)})
            fs.units[i] = replace(unit, operation=new_op)
        return
    raise KeyError(
        f"address {address!r}: no unit named {owner!r}; "
        f"units are {[u.name for u in fs.units]}"
    )


def require_eo_residuals(fs: Flowsheet) -> None:
    """Refuse a flowsheet whose units have no equation-oriented form.

    Both views in this package are built on
    :meth:`difflow.eo_solver.EOSolver._build_residual_fn`, which dispatches on
    ``hasattr(op, "eo_residuals")``. Units that have it contribute explicit
    residuals; units that do not take a fallback branch that calls the unit
    forward and differences the result.

    That fallback branch is **broken in difflow as of this writing**: its body
    reads a bare name ``feed_names`` that is a local of ``EOSolver.__init__``
    and is neither a closure cell nor a module global, so it raises
    ``NameError: name 'feed_names' is not defined`` the moment it executes.
    A user who hits it from here would see that NameError from deep inside a
    JAX trace with no indication of which unit caused it, so the check is done
    up front instead.

    The fallback is also the wrong formulation for this package even when
    fixed: a forward unit call closes its balance with an inner
    ``optimistix.Newton``, which is opaque to structural analysis and couples
    every variable in its block. See the module docstring.

    Args:
        fs: The flowsheet about to be turned into an NLP or a residual.

    Raises:
        TypeError: If any unit's operation lacks ``eo_residuals``, naming the
            offending units.

    Example:
        >>> require_eo_residuals(fs)   # doctest: +SKIP
    """
    bad = [u.name for u in fs.units if not hasattr(u.operation, "eo_residuals")]
    if bad:
        raise TypeError(
            f"units {bad} have no eo_residuals method, so the flowsheet has no "
            "equation-oriented form and cannot be turned into a flat NLP or a "
            "residual. Units with an EO interface today: CSTR, Flash, Mixer, "
            "Splitter, and the heat exchangers. (EOSolver's fallback branch for "
            "units without eo_residuals raises NameError on a bare 'feed_names' "
            "and is unusable; this check exists to report that up front rather "
            "than from inside a JAX trace.)"
        )


def _copy_flowsheet(fs: Flowsheet) -> Flowsheet:
    """Shallow functional copy: new unit list, new feed dicts, same operations."""
    new = Flowsheet(
        species_order=list(fs.species_order),
        default_flow=fs.default_flow,
        default_T=fs.default_T,
        default_P=fs.default_P,
    )
    new.feeds = {k: dict(v) for k, v in fs.feeds.items()}
    new.units = [replace(u, params=dict(u.params)) for u in fs.units]
    new.recycles = dict(fs.recycles)
    return new


def _make_builder(flowsheet, decisions, parameters):
    """Return ``build(dvals, pvals) -> Flowsheet``.

    ``flowsheet`` is either a :class:`Flowsheet` (addresses resolved against
    it) or a callable taking a single dict of *all* named values.
    """
    if callable(flowsheet) and not isinstance(flowsheet, Flowsheet):
        def build(dvals, pvals):
            fs = flowsheet({**dvals, **pvals})
            if not isinstance(fs, Flowsheet):
                raise TypeError(
                    "the flowsheet builder must return a difflow Flowsheet, "
                    f"got {type(fs).__name__}"
                )
            return fs
        return build, True

    def build(dvals, pvals):
        fs = _copy_flowsheet(flowsheet)
        for d in decisions:
            _apply(fs, d.address, dvals[d.name])
        for p in parameters:
            _apply(fs, p.address, pvals[p.name])
        return fs

    return build, False


# ---------------------------------------------------------------------------
# Topology-derived sparsity -- the fallback when the graph cannot be analyzed
# ---------------------------------------------------------------------------


def _pattern_from_sets(row_cols: list[set[int]], m: int, n: int):
    """``(rows, cols)`` from a per-row column set."""
    rows: list[int] = []
    cols: list[int] = []
    for i, cs in enumerate(row_cols):
        for j in sorted(cs):
            rows.append(i)
            cols.append(j)
    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


def _hessian_from_rows(row_cols: list[set[int]], obj_cols: set[int], n: int):
    """Lower-triangle Hessian pattern implied by the row column-sets.

    ``H = sigma * grad^2 f + sum_i lambda_i grad^2 g_i``. Term ``i`` can only
    have nonzeros inside ``cols(g_i) x cols(g_i)``, so the union of those
    outer products (plus the objective's) is a superset.
    """
    entries: set[tuple[int, int]] = set()
    for cs in list(row_cols) + [obj_cols]:
        cl = sorted(cs)
        for a in cl:
            for b in cl:
                if a >= b:
                    entries.add((a, b))
    if not entries:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    rows, cols = zip(*sorted(entries))
    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


def _topology_patterns(
    ref_fs,
    layout,
    decisions,
    specs,
    x0,
    *,
    n_d: int,
    n: int,
    m: int,
    n_res: int,
    var_names: list[str],
    objective_vars,
    addressed: bool,
):
    """Jacobian and Hessian patterns from the flowsheet's topology alone.

    The fallback for a model whose graph cannot be analyzed. A unit's
    residual rows can only touch the stream variables of its own inlets and
    outlets plus the decisions written into that unit -- everything
    downstream travels through stream variables, which are columns in their
    own right -- so the pattern is a superset at every point, with nothing
    evaluated.

    What it cannot see is the *objective* and any spec body given as a
    callable. Both come out dense unless ``objective_vars`` and
    ``Spec.variables`` say otherwise, and a dense objective block makes the
    whole Lagrangian Hessian dense. That is the reason this is the fallback
    and not the default.

    Args:
        ref_fs: The reference flowsheet.
        layout: Its :class:`~difflow.eo_solver.EOStateLayout`.
        decisions: Decisions, in variable order.
        specs: Parsed specs, in constraint order.
        x0: Starting point, used only to rebuild the reference streams.
        n_d: Size of the decision block.
        n: Number of variables.
        m: Number of constraints.
        n_res: Number of residual rows (constraints before the specs).
        var_names: Variable names, in order.
        objective_vars: Variables the objective touches, or ``None``.
        addressed: False for a builder callable, where a decision cannot be
            attributed to a unit and so touches all of them.

    Returns:
        ``(jac_pattern, hess_pattern)``, or ``None`` if the per-unit row
        counts do not add up to ``n_res`` -- in which case the rows cannot be
        attributed to units and the derivation is not a superset. A wrong
        guess degrades to ``None`` rather than to a corrupt pattern.
    """
    ref_streams = dict(ref_fs.feeds)
    ref_streams.update(layout.unpack(x0[n_d:]))
    sizes = _unit_row_blocks(ref_fs, layout, ref_streams)
    if sizes is None or sum(sizes) != n_res:
        return None

    index = {nm: i for i, nm in enumerate(var_names)}
    dec_map = _decision_rows(ref_fs, decisions, addressed)
    row_cols: list[set[int]] = []
    for k, unit in enumerate(ref_fs.units):
        cols: set[int] = set(dec_map[k])
        for nm in set(unit.inlet_names) | set(unit.outlet_names):
            if nm in layout.stream_names:
                sl = layout.stream_slice(nm)
                cols.update(range(n_d + sl.start, n_d + sl.stop))
        row_cols.extend([set(cols)] * sizes[k])

    dense_rows = []
    for sp in specs:
        if sp.variables is None:
            dense_rows.append(sp.name)
            row_cols.append(set(range(n)))
            continue
        missing = [nm for nm in sp.variables if nm not in index]
        if missing:
            raise KeyError(
                f"spec {sp.name!r} declares unknown variables {missing}"
            )
        row_cols.append({index[nm] for nm in sp.variables})

    if objective_vars is None:
        obj_cols = set(range(n))
        dense_rows.append("the objective")
    else:
        obj_cols = {index[nm] for nm in objective_vars}

    if dense_rows:
        warnings.warn(
            "the topology-derived pattern cannot see inside "
            + ", ".join(dense_rows)
            + f", so {'their' if len(dense_rows) > 1 else 'its'} row is dense "
            "and the Lagrangian Hessian is dense with it. Pass "
            "objective_vars=... and Spec(variables=...), or use the default "
            "sparsity='auto', which reads the structure off the graph.",
            RuntimeWarning,
            stacklevel=3,
        )

    return (
        _pattern_from_sets(row_cols, m, n),
        _hessian_from_rows(row_cols, obj_cols, n),
    )


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


def _coerce_decisions(decisions: Iterable) -> list[Decision]:
    out = []
    for d in decisions:
        if isinstance(d, Decision):
            out.append(d)
        elif isinstance(d, (tuple, list)) and len(d) == 4:
            out.append(Decision(str(d[0]), float(d[1]), float(d[2]), float(d[3])))
        elif isinstance(d, dict):
            out.append(Decision(**d))
        else:
            raise ValueError(
                "each decision must be a Decision, a dict, or a "
                f"(address, lb, ub, x0) 4-tuple; got {d!r}"
            )
    if len({d.name for d in out}) != len(out):
        raise ValueError("decision names must be unique")
    return out


def _feed_decisions(decisions) -> tuple[tuple[int, str, str], ...]:
    """Locate the ``feed:``-addressed decisions within the decision block.

    Args:
        decisions: The :class:`Decision` sequence, in variable order.

    Returns:
        ``(index, feed_name, stream_key)`` for each feed-addressed decision.
    """
    out = []
    for i, d in enumerate(decisions):
        address = getattr(d, "address", None) or d.name
        if not address.startswith("feed:"):
            continue
        kind, owner, attr = _split_address(address)
        if kind == "feed":
            out.append((i, owner, attr))
    return tuple(out)


def _var_names(decisions, layout) -> list[str]:
    names = [d.name for d in decisions]
    for s in layout.stream_names:
        names.extend([f"{s}.F_{sp}" for sp in layout.species_order])
        names.append(f"{s}.T")
        names.append(f"{s}.P")
    return names


def _stream_var_bounds(layout, flow_bounds, T_bounds, P_bounds):
    lb, ub = [], []
    for _ in layout.stream_names:
        for _sp in layout.species_order:
            lb.append(flow_bounds[0])
            ub.append(flow_bounds[1])
        lb.append(T_bounds[0])
        ub.append(T_bounds[1])
        lb.append(P_bounds[0])
        ub.append(P_bounds[1])
    return lb, ub


def _unit_row_blocks(fs: Flowsheet, layout, streams) -> list[int] | None:
    """Row count contributed by each unit, or ``None`` if it cannot be known.

    Mirrors the dispatch in :meth:`EOSolver._build_residual_fn` -- units with
    ``eo_residuals`` contribute whatever that returns; the fallback path
    contributes ``n_per_stream`` rows per non-feed outlet. The caller checks
    the total against the real residual length and goes dense on a mismatch,
    so a wrong guess here degrades the pattern rather than corrupting it.
    """
    sizes = []
    for unit in fs.units:
        op = unit.operation
        try:
            if hasattr(op, "eo_residuals"):
                inlets = [streams[nm] for nm in unit.inlet_names]
                outlets = [streams[nm] for nm in unit.outlet_names]
                r = op.eo_residuals(inlets, outlets, **unit.params)
                sizes.append(int(jnp.asarray(r).size))
            else:
                n_out = sum(
                    1 for nm in unit.outlet_names if nm in layout.stream_names
                )
                sizes.append(n_out * layout.n_per_stream)
        except Exception:
            return None
    return sizes


def _decision_rows(fs: Flowsheet, decisions, addressed: bool) -> list[set[int] | None]:
    """For each unit index, the decision columns that can touch its rows.

    A decision written into unit ``k`` can only change unit ``k``'s
    residuals; every downstream effect travels through stream variables,
    which are already columns of their own. A feed decision touches the
    units that consume that feed. When the flowsheet is a builder callable
    we cannot know, so every decision touches every unit.
    """
    per_unit: list[set[int]] = [set() for _ in fs.units]
    for j, d in enumerate(decisions):
        if not addressed:
            for s in per_unit:
                s.add(j)
            continue
        try:
            kind, owner, _attr = _split_address(d.address)
        except ValueError:
            for s in per_unit:
                s.add(j)
            continue
        for k, unit in enumerate(fs.units):
            if kind == "feed":
                if owner in unit.inlet_names or owner in unit.outlet_names:
                    per_unit[k].add(j)
            elif unit.name == owner:
                per_unit[k].add(j)
    return per_unit


def as_nlp(
    flowsheet,
    decisions: Sequence,
    specs: Sequence = (),
    *,
    objective: Callable | None = None,
    objective_vars: Sequence[str] | None = None,
    parameters: Sequence[Parameter] = (),
    flow_bounds: tuple[float, float] = DEFAULT_FLOW_BOUNDS,
    T_bounds: tuple[float, float] = DEFAULT_T_BOUNDS,
    P_bounds: tuple[float, float] = DEFAULT_P_BOUNDS,
    var_bounds: dict[str, tuple[float, float]] | None = None,
    x0_streams: dict[str, Stream] | None = None,
    sparsity: str = "auto",
    validate: bool | str = "auto",
) -> tuple[Callable, Callable, Bounds]:
    """Build a flat NLP view of a flowsheet.

    Args:
        flowsheet: A :class:`~difflow.flowsheet.Flowsheet`, or a callable
            ``build(values) -> Flowsheet`` taking a dict of decision and
            parameter values keyed by name. The callable form is the escape
            hatch for anything the address grammar cannot reach.
        decisions: :class:`Decision` objects (or ``(address, lb, ub, x0)``
            tuples). Addresses:

            * ``"unit:<unit>.<kwarg>"`` -- an entry of ``Unit.params``, i.e. a
              keyword passed to the unit call and to ``eo_residuals``
              (``T_spec``, ``volumetric_flow``, ...).
            * ``"unit:<unit>.params.<field>"`` -- a field of the operation's
              ``Params`` dataclass (``V``, ``UA``, ...), applied with
              ``ParamsMixin.update``.
            * ``"feed:<stream>.<key>"`` -- ``T``, ``P`` or ``F_<species>`` of a
              feed stream.

            The ``unit:`` prefix may be dropped.
        specs: :class:`Spec` objects or ``(target, op, value)`` shorthands.
        objective: ``objective(streams, decisions) -> scalar``. Defaults to a
            constant 0, which turns the NLP into a feasibility problem.
        objective_vars: Variable names the objective touches. Only the
            ``sparsity="structural"`` path needs them -- graph analysis reads
            them off the objective itself. Omitting them on that path makes
            the Hessian dense, and says so.
        parameters: :class:`Parameter` objects. These become the ``p``
            argument of ``f(x, p)`` / ``g(x, p)``, so an outer JAX
            computation can differentiate through the solve with
            ``pounce.jax.JaxProblem``. ``f(x)`` and ``g(x)`` (one argument)
            evaluate at the nominal values.
        flow_bounds: ``(lb, ub)`` applied to every species flow variable.
        T_bounds: ``(lb, ub)`` applied to every stream temperature.
        P_bounds: ``(lb, ub)`` applied to every stream pressure.
        var_bounds: Per-variable overrides keyed by name, e.g.
            ``{"product.T": (300.0, 400.0)}``.
        x0_streams: Explicit starting streams. Default: the sequential-modular
            solution at the decisions' ``x0``, falling back to feed
            propagation.
        sparsity: Where the sparsity patterns come from. None of these
            probes; they differ in how tight they are and in what has to hold
            for them to be a superset. See :mod:`difflow.solvers.sparsity`.

            * ``"auto"`` (default) -- global graph analysis with :mod:`asdex`:
              index sets propagated through the jaxpr, so the result is valid
              at every point and is tight enough that the Lagrangian Hessian
              grows like ``n`` rather than ``n^2``. Falls back to
              ``"structural"``, with a :class:`RuntimeWarning`, if the graph
              cannot be analyzed; a missing ``asdex`` install raises instead,
              since that has a fix.
            * ``"global"`` -- the same, but raising instead of falling back.
            * ``"structural"`` -- from the flowsheet topology alone: a unit's
              rows touch its own streams and its own decisions. Valid
              everywhere, but blind to the objective and to a callable spec
              body, both of which then come out dense.
            * ``"dense"`` -- no pattern. A valid superset that costs ``n``
              colors per Hessian evaluation; nothing falls back to it on its
              own, because on a real flowsheet that is the difference between
              a solve and a stall.
        validate: Check the pattern against AD at ``x0``. ``"auto"``
            (default, and the same as ``True``) checks exactly against dense
            derivatives when the problem is small and against a random sample
            of columns when it is large -- it is never skipped, since a
            missing entry is silently wrong. ``"dense"`` or ``"sampled"``
            force one; ``False`` skips. See
            :func:`~difflow.solvers.sparsity.validate_patterns`.

    Returns:
        ``(f, g, bounds)``. ``f(x, p=p0)`` returns a scalar; ``g(x, p=p0)``
        returns an array of length ``bounds.m``; ``bounds`` is a
        :class:`Bounds` carrying the boxes, the starting point and both
        sparsity patterns.

    Raises:
        TypeError: If any unit lacks ``eo_residuals``; see
            :func:`require_eo_residuals`.
        ValueError: If the flowsheet has a recycle whose source and
            destination stream names differ (the EO formulation identifies a
            recycle by *naming*, so ``source != dest`` would leave the system
            underdetermined), or if a spec is malformed.
        SparsityPatternError: If validation is on and the derived pattern is
            not a superset at ``x0``.
        SparsityDetectionError: If no pattern could be derived. Dense is a
            valid superset, but falling back to it has to be the caller's
            decision -- ``sparsity="dense"`` -- not a silent one.

    Example:
        >>> f, g, bd = as_nlp(                       # doctest: +SKIP
        ...     fs,
        ...     [Decision("unit:reactor.params.V", 0.01, 5.0, 0.5),
        ...      Decision("unit:reactor.T_spec", 320.0, 420.0, 360.0)],
        ...     [("product.F_B", ">=", 8.0)],
        ...     objective=lambda s, d: -2.0 * s["product"]["F_B"],
        ... )
        >>> x, info = solve_with_pounce(f, g, bd)    # doctest: +SKIP
    """
    decisions = _coerce_decisions(decisions)
    specs = [Spec.parse(s) for s in specs]
    parameters = list(parameters)
    if objective is None:
        def objective(streams, dvals):  # noqa: ARG001 - feasibility problem
            return jnp.asarray(0.0)

    build, is_builder = _make_builder(flowsheet, decisions, parameters)
    addressed = not is_builder

    d0 = {d.name: jnp.asarray(float(d.x0)) for d in decisions}
    p0_map = {p.name: jnp.asarray(float(p.value)) for p in parameters}
    p0 = jnp.asarray([float(p.value) for p in parameters])

    ref_fs = build(d0, p0_map)
    require_eo_residuals(ref_fs)
    for src, dst in ref_fs.recycles.items():
        if src != dst:
            raise ValueError(
                f"recycle {src!r} -> {dst!r}: the equation-oriented form "
                "identifies a recycle by naming the destination and the source "
                "the same stream (fs.add_recycle('recycle', 'recycle')). With "
                "different names there is no equation tying them together and "
                "the NLP is underdetermined."
            )

    ref_solver = EOSolver(ref_fs)
    layout = ref_solver.layout
    n_d = len(decisions)
    n_stream = layout.total_vars
    n = n_d + n_stream

    # --- functions -------------------------------------------------------
    def _context(x, p):
        x = jnp.asarray(x)
        dvals = {d.name: x[i] for i, d in enumerate(decisions)}
        pvals = {pp.name: jnp.asarray(p)[i] for i, pp in enumerate(parameters)}
        fs = build(dvals, pvals)
        streams = dict(fs.feeds)
        streams.update(layout.unpack(x[n_d:]))
        return fs, dvals, streams

    def f(x, p=p0):
        _fs, dvals, streams = _context(x, p)
        return jnp.reshape(jnp.asarray(objective(streams, dvals)), ())

    def g(x, p=p0):
        fs, dvals, streams = _context(x, p)
        solver = EOSolver(fs)
        if solver.layout.stream_names != layout.stream_names:
            raise ValueError(
                "the flowsheet builder changed the set of streams; the NLP "
                "layout must be fixed. Keep the topology constant and vary "
                "only parameter values."
            )
        res = solver._build_residual_fn()(x[n_d:], dict(fs.feeds))
        rows = [jnp.asarray(res).ravel()]
        for sp in specs:
            rows.append(jnp.reshape(jnp.asarray(sp.fn(streams, dvals)), (1,)))
        return jnp.concatenate(rows)

    # --- starting point --------------------------------------------------
    if x0_streams is not None:
        guess = {nm: x0_streams[nm] for nm in layout.stream_names}
    else:
        try:
            guess = ref_solver._sm_init()
            layout.pack(guess)  # raises if a name is missing
        except Exception:
            guess = ref_solver._feed_propagation_init()
    x0_s = layout.pack(guess)
    x0 = jnp.concatenate([jnp.asarray([float(d.x0) for d in decisions]), x0_s])

    # --- bounds ----------------------------------------------------------
    lb = [d.lb for d in decisions]
    ub = [d.ub for d in decisions]
    s_lb, s_ub = _stream_var_bounds(layout, flow_bounds, T_bounds, P_bounds)
    lb.extend(s_lb)
    ub.extend(s_ub)
    var_names = _var_names(decisions, layout)
    if var_bounds:
        index = {nm: i for i, nm in enumerate(var_names)}
        for nm, (a, b) in var_bounds.items():
            if nm not in index:
                raise KeyError(
                    f"var_bounds refers to unknown variable {nm!r}; "
                    f"known names include {var_names[:6]} ..."
                )
            lb[index[nm]], ub[index[nm]] = float(a), float(b)
    lb = jnp.asarray(lb, dtype=jnp.float64)
    ub = jnp.asarray(ub, dtype=jnp.float64)
    x0 = jnp.clip(x0, lb, ub)

    # --- constraint bounds -----------------------------------------------
    n_res = int(jnp.asarray(g(x0)).size) - len(specs)
    cl = jnp.concatenate(
        [jnp.zeros(n_res), jnp.asarray([s.lo for s in specs], dtype=jnp.float64)]
    ) if specs else jnp.zeros(n_res)
    cu = jnp.concatenate(
        [jnp.zeros(n_res), jnp.asarray([s.hi for s in specs], dtype=jnp.float64)]
    ) if specs else jnp.zeros(n_res)
    con_names = [f"residual[{i}]" for i in range(n_res)] + [s.name for s in specs]
    m = n_res + len(specs)

    # --- sparsity --------------------------------------------------------
    # Order matters. Global graph analysis first, because it is the only one
    # of the three that is both valid everywhere and tight; the topology
    # derivation as the fallback, because it is valid everywhere but coarse;
    # dense only when explicitly asked for. See difflow.solvers.sparsity for
    # what each costs.
    if sparsity not in ("auto", "global", "structural", "dense"):
        raise ValueError(
            "sparsity must be 'auto', 'global', 'structural' or 'dense', "
            f"got {sparsity!r}"
        )

    def _topology():
        return _topology_patterns(
            ref_fs,
            layout,
            decisions,
            specs,
            x0,
            n_d=n_d,
            n=n,
            m=m,
            n_res=n_res,
            var_names=var_names,
            objective_vars=objective_vars,
            addressed=addressed,
        )

    sparsity_source = sparsity
    if sparsity == "dense":
        jac_pattern = dense_jacobian_pattern(m, n)
        hess_pattern = dense_hessian_pattern(n)
    elif sparsity in ("auto", "global"):
        try:
            jac_pattern = detect_jacobian_pattern(g, x0)
            hess_pattern = detect_hessian_pattern(f, g, x0, m)
            sparsity_source = "global"
        except ImportError:
            # An install problem, with an obvious fix and an actionable
            # message. Falling back would hide it behind a pattern that is
            # dense in the Hessian on nearly every model.
            raise
        except SparsityDetectionError as exc:
            if sparsity == "global":
                raise
            fallback = _topology()
            if fallback is None:
                raise SparsityDetectionError(
                    "no sparsity pattern could be derived for this flowsheet: "
                    f"graph analysis failed ({exc}) and the topology "
                    "derivation could not account for the residual rows. "
                    "sparsity='dense' will solve it, at n colors per Hessian."
                ) from exc
            warnings.warn(
                f"falling back to the topology-derived sparsity pattern: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            jac_pattern, hess_pattern = fallback
            sparsity_source = "structural"
    else:  # "structural"
        fallback = _topology()
        if fallback is None:
            raise SparsityDetectionError(
                "the topology derivation could not account for the residual "
                "rows of this flowsheet, so it cannot be trusted to be a "
                "superset. sparsity='auto' derives the pattern from the graph "
                "instead; sparsity='dense' accepts no pattern at all."
            )
        jac_pattern, hess_pattern = fallback

    if validate is True:
        validate = "auto"
    if validate:
        validate_patterns(
            g,
            x0,
            jac_pattern,
            m,
            n,
            f=f,
            hess_pattern=hess_pattern,
            mode="auto" if validate == "auto" else str(validate),
        )

    bounds = Bounds(
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
        x0=x0,
        p0=p0,
        var_names=var_names,
        con_names=con_names,
        jac_pattern=jac_pattern,
        hess_pattern=hess_pattern,
        sparsity_source=sparsity_source,
        n_decisions=n_d,
        layout=layout,
        feeds=dict(ref_fs.feeds),
        feed_decisions=_feed_decisions(decisions),
    )
    return f, g, bounds
