"""Residual view of a unit or a flowsheet section: ``g(u, v) = 0``.

:func:`as_residual` exposes the equations difflow already writes -- a unit's
``eo_residuals``, or the whole system assembled by
:meth:`difflow.eo_solver.EOSolver._build_residual_fn` -- in the shape an
implicit-function node wants: inputs and parameters in ``u``, internal
states in ``v``, and a square residual that vanishes at the solution.

Nothing is re-derived. For a flowsheet the returned callable *is* the EO
solver's residual function with the feed streams unpacked from ``u``; for a
unit it is that unit's own ``eo_residuals`` with the streams unpacked from
``u`` and ``v``. If the flowsheet solves, the residual view is consistent
with it by construction.

:func:`residual_from_system` is the third constructor, for models that
already *are* a residual in difflow's section-scope ``r(z; args)``
convention (see :func:`difflow.eo_solver.solve_residual_system`) and never
had a :class:`~difflow.flowsheet.Flowsheet` built around them --
``difflow_ree.equilibrium.mass_action.make_section_residual``'s
counter-current cascade, for instance.

The canonical consumer is ``discopt.modeling.implicit``, which compiles
``g(u, v) = 0`` into a differentiable inner Newton solve whose derivatives
come from ``jax.lax.custom_root`` -- see
:mod:`difflow.solvers.discopt_bridge`, and read its warning about what a
model containing such a node may and may not contain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree

from difflow.eo_solver import EOSolver, EOStateLayout
from difflow.flowsheet import Flowsheet, Unit
from difflow.solvers.nlp import require_eo_residuals
from difflow.streams import Stream, make_stream

__all__ = ["ResidualView", "as_residual", "residual_from_system"]


def _pack_stream(stream: Stream, species: Sequence[str]) -> list:
    return [jnp.asarray(stream[f"F_{s}"]) for s in species] + [
        jnp.asarray(stream["T"]),
        jnp.asarray(stream["P"]),
    ]


def _unpack_streams(vec: Array, names: Sequence[str], species: Sequence[str]):
    """Split a flat vector into ``[F..., T, P]`` streams, in ``names`` order."""
    per = len(species) + 2
    out = {}
    for i, nm in enumerate(names):
        b = i * per
        flows = {s: vec[b + j] for j, s in enumerate(species)}
        out[nm] = make_stream(flows, vec[b + len(species)], vec[b + len(species) + 1])
    return out


@dataclass
class ResidualView:
    """A callable ``g(u, v) -> array`` plus the metadata to use it.

    Instances are callable, so a :class:`ResidualView` is a drop-in
    ``residual`` argument for ``discopt.modeling.implicit``.

    Attributes:
        fn: The residual, ``fn(u, v) -> (n_unknowns,)``.
        u_names: Names of the ``u`` entries, in order.
        v_names: Names of the ``v`` entries, in order.
        u0: Nominal input vector (the values the view was built from).
        v0: A starting guess for ``v``, suitable as ``implicit(x0=...)``.
        species_order: Species order used to flatten the streams.
        stream_names: Names of the streams making up ``v``, in order.
        source: The object the view was built from.

    Example:
        >>> view = as_residual(flowsheet)             # doctest: +SKIP
        >>> float(abs(view(view.u0, view.v0)).max())  # doctest: +SKIP
        1.2e-09
    """

    fn: Callable[[Array, Array], Array]
    u_names: list[str]
    v_names: list[str]
    u0: Array
    v0: Array
    species_order: list[str]
    stream_names: list[str]
    source: Any = None

    def __call__(self, u, v):
        return self.fn(jnp.asarray(u), jnp.asarray(v))

    @property
    def n_unknowns(self) -> int:
        """Length of ``v`` -- the ``n_unknowns`` argument of ``dm.implicit``."""
        return len(self.v_names)

    @property
    def n_inputs(self) -> int:
        """Length of ``u``."""
        return len(self.u_names)

    def unpack_v(self, v: Array) -> dict[str, Stream]:
        """Turn a solved ``v`` back into named streams.

        Raises:
            TypeError: For a view built by :func:`residual_from_system`, whose
                ``v`` is an arbitrary state vector rather than a set of
                streams.
        """
        if not self.stream_names:
            raise TypeError(
                "this residual view has no stream structure (it was built from "
                "a bare residual function, not a flowsheet or unit), so v "
                "cannot be unpacked into streams. Read v directly; v_names "
                "labels its entries."
            )
        return _unpack_streams(jnp.asarray(v), self.stream_names, self.species_order)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ResidualView(n_inputs={self.n_inputs}, "
            f"n_unknowns={self.n_unknowns}, "
            f"source={type(self.source).__name__})"
        )


def _species_of(op, species_order):
    if species_order is not None:
        return list(species_order)
    params = getattr(op, "params", None)
    if params is not None and getattr(params, "species_order", None):
        return list(params.species_order)
    if getattr(op, "species_order", None):
        return list(op.species_order)
    raise ValueError(
        f"cannot infer species_order from {type(op).__name__}; pass "
        "species_order= explicitly"
    )


def _leaf_names(pytree, prefix: str = "u") -> list[str]:
    """Per-scalar names for ``ravel_pytree(pytree)``, in the same order.

    ``ravel_pytree`` flattens with ``jax.tree_util``, which sorts dict keys, so
    the names have to come from the same traversal rather than from the order
    the caller happened to write the dict in.
    """
    leaves, _ = jax.tree_util.tree_flatten_with_path(pytree)
    names: list[str] = []
    for path, leaf in leaves:
        base = prefix + jax.tree_util.keystr(path)
        arr = jnp.asarray(leaf)
        if arr.ndim == 0:
            names.append(base)
        else:
            names.extend(f"{base}[{i}]" for i in range(arr.size))
    return names


def residual_from_system(
    residual_fn: Callable,
    z0: Array,
    args: Any = None,
    *,
    u_keys: Sequence[str] | None = None,
    v_names: Sequence[str] | None = None,
    name: str = "section",
) -> ResidualView:
    """Wrap a difflow ``r(z; args) = 0`` section residual as ``g(u, v)``.

    This is the second constructor of the residual view, for models that are
    already a residual and never had a :class:`~difflow.flowsheet.Flowsheet`
    built around them -- the shape
    :func:`difflow.eo_solver.solve_residual_system` takes, and what
    ``difflow_ree.equilibrium.mass_action.make_section_residual`` returns for a
    counter-current mass-action cascade.

    The only work done here is re-ordering the two arguments and flattening
    ``args``: ``v`` is ``z``, and ``u`` is the raveled parameter pytree, so an
    outer model can drive the section through ``u`` and differentiate the
    solved ``v`` with respect to it.

    Args:
        residual_fn: ``residual_fn(z, args) -> r`` with ``r`` the same length
            as ``z``. JAX-traceable.
        z0: Starting guess for ``z``; becomes :attr:`ResidualView.v0`.
        args: The parameter pytree passed through to ``residual_fn``.
        u_keys: When ``args`` is a dict, the subset of keys to expose in ``u``.
            Everything else is closed over at its nominal value. ``None``
            exposes the whole pytree. Naming a subset is usually what you want:
            an implicit block should depend on the handful of quantities the
            outer model actually varies, not on every static table in ``args``.

            **The order of ``u`` is not the order of ``u_keys``.**
            ``ravel_pytree`` flattens with ``jax.tree_util``, which sorts dict
            keys, so ``u_keys=["feed", "K"]`` yields ``u = [K..., feed...]``.
            ``dm.implicit`` concatenates its ``u_inputs`` in argument order, so
            read :attr:`ResidualView.u_names` and pass the model expressions in
            *that* order, never in the order you wrote ``u_keys``.
        v_names: Names for the ``v`` entries; defaults to ``v[0] ...``.
        name: Label used in the default ``v`` names and in the repr.

    Returns:
        A :class:`ResidualView` whose ``fn(u, v)`` calls ``residual_fn`` with
        the reconstructed ``args``.

    Raises:
        KeyError: If a name in ``u_keys`` is not a key of ``args``.
        TypeError: If ``u_keys`` is given but ``args`` is not a dict.
        ValueError: If ``residual_fn(z0, args)`` is not the same length as
            ``z0`` (``dm.implicit`` and Newton both need a square system).

    Example:
        >>> res_fn, _ = make_section_residual(net, n_stages=4)   # doctest: +SKIP
        >>> view = residual_from_system(                          # doctest: +SKIP
        ...     res_fn, z0=u_guess, args=args, u_keys=["feed_totals"])
        >>> node = dm.implicit(view, [F_expr], view.n_unknowns, x0=view.v0)  # doctest: +SKIP
    """
    z0 = jnp.asarray(z0)
    if u_keys is None:
        exposed, fixed = args, None
    else:
        if not isinstance(args, dict):
            raise TypeError(
                f"u_keys= only applies when args is a dict; got "
                f"{type(args).__name__}"
            )
        missing = [k for k in u_keys if k not in args]
        if missing:
            raise KeyError(
                f"u_keys {missing} are not keys of args; args has "
                f"{sorted(args)}"
            )
        exposed = {k: args[k] for k in u_keys}
        fixed = {k: v for k, v in args.items() if k not in set(u_keys)}

    u0, unravel = ravel_pytree(exposed)
    u_names = _leaf_names(exposed, prefix="u")

    def fn(u, v):
        rebuilt = unravel(jnp.asarray(u))
        full = rebuilt if fixed is None else {**fixed, **rebuilt}
        return jnp.asarray(residual_fn(jnp.asarray(v), full)).ravel()

    n_res = int(jnp.asarray(fn(u0, z0)).size)
    if n_res != int(z0.size):
        raise ValueError(
            f"residual_fn returns {n_res} equations for {int(z0.size)} unknowns. "
            "A residual view must be square: dm.implicit and "
            "solve_residual_system both solve it with Newton."
        )

    return ResidualView(
        fn=fn,
        u_names=u_names,
        v_names=list(v_names) if v_names is not None
        else [f"{name}.v[{i}]" for i in range(int(z0.size))],
        u0=u0,
        v0=z0,
        species_order=[],
        stream_names=[],
        source=residual_fn,
    )


def _flowsheet_view(fs: Flowsheet) -> ResidualView:
    require_eo_residuals(fs)
    solver = EOSolver(fs)
    layout: EOStateLayout = solver.layout
    species = list(layout.species_order)
    feed_names = sorted(fs.feeds)
    residual_fn = solver._build_residual_fn()

    def fn(u, v):
        feeds = _unpack_streams(u, feed_names, species)
        return residual_fn(v, feeds)

    u_names = []
    for nm in feed_names:
        u_names.extend([f"{nm}.F_{s}" for s in species] + [f"{nm}.T", f"{nm}.P"])
    v_names = []
    for nm in layout.stream_names:
        v_names.extend([f"{nm}.F_{s}" for s in species] + [f"{nm}.T", f"{nm}.P"])

    u0 = jnp.concatenate(
        [jnp.stack(_pack_stream(fs.feeds[nm], species)) for nm in feed_names]
    ) if feed_names else jnp.zeros(0)
    try:
        guess = solver._sm_init()
        v0 = layout.pack(guess)
    except Exception:
        v0 = layout.pack(solver._feed_propagation_init())

    return ResidualView(
        fn=fn,
        u_names=u_names,
        v_names=v_names,
        u0=u0,
        v0=v0,
        species_order=species,
        stream_names=list(layout.stream_names),
        source=fs,
    )


def _unit_view(
    op,
    inlets: Sequence[Stream],
    outlets: Sequence[Stream],
    params: dict,
    species: list[str],
    inlet_names: Sequence[str],
    outlet_names: Sequence[str],
    source,
) -> ResidualView:
    if not hasattr(op, "eo_residuals"):
        raise TypeError(
            f"{type(op).__name__} has no eo_residuals, so it has no residual "
            "form. Only units with an equation-oriented interface (CSTR, "
            "Flash, heat exchangers, ...) can be exposed this way; a "
            "sequential-modular call closes its balance with an inner Newton "
            "solve and is not a residual."
        )

    def fn(u, v):
        ins = _unpack_streams(u, list(inlet_names), species)
        outs = _unpack_streams(v, list(outlet_names), species)
        r = op.eo_residuals(
            [ins[nm] for nm in inlet_names],
            [outs[nm] for nm in outlet_names],
            **params,
        )
        return jnp.asarray(r).ravel()

    u_names, v_names = [], []
    for nm in inlet_names:
        u_names.extend([f"{nm}.F_{s}" for s in species] + [f"{nm}.T", f"{nm}.P"])
    for nm in outlet_names:
        v_names.extend([f"{nm}.F_{s}" for s in species] + [f"{nm}.T", f"{nm}.P"])

    u0 = jnp.concatenate([jnp.stack(_pack_stream(s, species)) for s in inlets])
    v0 = jnp.concatenate([jnp.stack(_pack_stream(s, species)) for s in outlets])

    n_res = int(jnp.asarray(fn(u0, v0)).size)
    if n_res != len(v_names):
        raise ValueError(
            f"{type(op).__name__}.eo_residuals returns {n_res} equations for "
            f"{len(v_names)} unknowns. dm.implicit needs a square system; "
            "supply the outlets whose variables the unit actually determines, "
            "or use the flowsheet form of as_residual."
        )

    return ResidualView(
        fn=fn,
        u_names=u_names,
        v_names=v_names,
        u0=u0,
        v0=v0,
        species_order=species,
        stream_names=list(outlet_names),
        source=source,
    )


def as_residual(
    unit_or_section,
    *,
    inlets: Sequence[Stream] | None = None,
    outlets: Sequence[Stream] | None = None,
    params: dict | None = None,
    species_order: Sequence[str] | None = None,
    z0: Array | None = None,
    args: Any = None,
    u_keys: Sequence[str] | None = None,
) -> ResidualView:
    """Expose a unit, a flowsheet section, or a raw residual as ``g(u, v) = 0``.

    Args:
        unit_or_section: One of

            * a :class:`~difflow.flowsheet.Flowsheet` -- ``u`` is the flat
              vector of every feed stream's ``[F..., T, P]`` (feeds sorted by
              name) and ``v`` is the EO state vector of every non-feed
              stream. The residual is the EO solver's own.
            * a :class:`~difflow.flowsheet.Unit` -- ``u`` is its inlet
              streams, ``v`` its outlets, and the residual is the
              operation's ``eo_residuals``. ``inlets`` and ``outlets`` give
              the nominal values / starting guess.
            * a bare unit operation with an ``eo_residuals`` method -- same,
              but ``inlets`` and ``outlets`` are required.
            * a plain callable ``residual_fn(z, args)`` -- difflow's
              section-scope convention (see
              :func:`difflow.eo_solver.solve_residual_system`). ``z0`` is
              required; the call is forwarded to
              :func:`residual_from_system`.

        inlets: Nominal inlet streams (unit forms only).
        outlets: Starting-guess outlet streams (unit forms only).
        params: Keyword arguments for ``eo_residuals``. Defaults to the
            :class:`~difflow.flowsheet.Unit`'s ``params`` where available.
        species_order: Species order; inferred from the unit's ``Params``
            when omitted.
        z0: Starting guess (raw-residual form only).
        args: Parameter pytree (raw-residual form only).
        u_keys: Subset of ``args`` keys to expose in ``u`` (raw-residual form
            only). See :func:`residual_from_system`.

    Returns:
        A :class:`ResidualView`: callable as ``g(u, v)``, carrying
        ``n_unknowns``, ``u0`` and ``v0``.

    Raises:
        TypeError: If the object has no equation-oriented form (a flowsheet
            with a unit that lacks ``eo_residuals``, or an object that is
            neither a flowsheet, a unit, nor a callable).
        ValueError: If the unit's residual count does not match the number of
            outlet unknowns (``dm.implicit`` requires a square system), if the
            species order cannot be inferred, or if the raw-residual form is
            used without ``z0``.

    Example:
        >>> view = as_residual(cstr, inlets=[feed], outlets=[guess])  # doctest: +SKIP
        >>> import discopt.modeling as dm                            # doctest: +SKIP
        >>> node = dm.implicit(view, [T_var], view.n_unknowns, x0=view.v0)  # doctest: +SKIP
    """
    if isinstance(unit_or_section, Flowsheet):
        if inlets is not None or outlets is not None:
            raise ValueError(
                "inlets/outlets do not apply to a Flowsheet; its feeds are u "
                "and its non-feed streams are v"
            )
        return _flowsheet_view(unit_or_section)

    if isinstance(unit_or_section, Unit):
        unit = unit_or_section
        op = unit.operation
        species = _species_of(op, species_order)
        if inlets is None or outlets is None:
            raise ValueError(
                "a Unit needs inlets= and outlets= to fix the nominal input "
                "vector and the starting guess for v"
            )
        return _unit_view(
            op,
            inlets,
            outlets,
            dict(unit.params) if params is None else dict(params),
            species,
            unit.inlet_names,
            unit.outlet_names,
            unit,
        )

    op = unit_or_section
    if not hasattr(op, "eo_residuals"):
        # A callable with no eo_residuals is the raw r(z; args) form. Anything
        # else has no residual view at all.
        if not callable(op):
            raise TypeError(
                f"{type(op).__name__} is not a Flowsheet, a Unit, a unit "
                "operation with eo_residuals, or a residual function"
            )
        if z0 is None:
            raise ValueError(
                "a bare residual function needs z0= (the starting guess, which "
                "also fixes the number of unknowns). Call signature is "
                "residual_fn(z, args), difflow's section-scope convention."
            )
        return residual_from_system(op, z0, args, u_keys=u_keys)

    species = _species_of(op, species_order)
    if inlets is None or outlets is None:
        raise ValueError(
            "a bare unit operation needs inlets= and outlets= (there are no "
            "stream names to look them up by)"
        )
    inlet_names = [f"in{i}" for i in range(len(inlets))]
    outlet_names = [f"out{i}" for i in range(len(outlets))]
    return _unit_view(
        op,
        inlets,
        outlets,
        dict(params or {}),
        species,
        inlet_names,
        outlet_names,
        op,
    )
