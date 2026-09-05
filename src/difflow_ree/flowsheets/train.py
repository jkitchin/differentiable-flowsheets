"""A REE separation train as a graph, with the organic loop closed (#202).

``FullSeparationTrain`` is a fixed sequence: cerium removal, then group
separation, then individual separations, by direct calls in a set order.
The topology is decided at import time, and the barren organic that every
circuit returns "for recycle" is never actually recycled --- which
silently assumes perfect stripping and flatters every design the package
produces.

:class:`SeparationTrain` replaces the sequence with **module instances
plus a connectivity map**. An outlet may feed any type-compatible inlet,
including one upstream of itself, so the organic loop is closed by adding
an edge rather than by writing a solver:

    >>> train = SeparationTrain("nd_circuit")           # doctest: +SKIP
    >>> train.add_module(sep)                           # doctest: +SKIP
    >>> train.add_feed("leach", feed, "sep.feed")       # doctest: +SKIP
    >>> train.connect("sep.barren_organic", "sep.solvent")   # doctest: +SKIP
    >>> result = train.solve()                          # doctest: +SKIP

What is reused rather than reinvented (#202)
--------------------------------------------

* :class:`difflow.flowsheet.Flowsheet` is the solver. :meth:`to_flowsheet`
  emits a real ``Flowsheet`` --- units in topological order, feeds,
  ``add_recycle`` for each back edge --- and :meth:`solve` calls its
  Anderson-accelerated tear solver. Nothing here iterates.
* :mod:`difflow.serialize` supplies the JSON layer. :meth:`to_dict`
  writes the *train* --- one level above a flowsheet, so a topology can
  be enumerated and diffed without instantiating anything --- but the
  stream and parameter encoders, the ``SerializationError`` and the
  ``FORMAT_VERSION`` are difflow's, not a second set of rules that would
  drift from them. The emitted ``Flowsheet`` is not yet writable by
  ``difflow.serialize.to_dict`` itself, because its unit operations are
  module adapters rather than registered classes; see :meth:`to_dict`.
* :mod:`difflow.catalog` supplies the parameter schema through each
  module's :meth:`~difflow_ree.flowsheets.modules.REEModule.describe`.

The one thing not delegated is the traceable solve.
:meth:`solve_differentiable` runs the *same* ``Flowsheet`` graph through
``optimistix.fixed_point``, because ``Flowsheet.solve`` records its
convergence diagnostics as Python floats and so cannot be traced. It is
the same units, the same order and the same tear set; only the outer
bookkeeping differs.

Out of scope, deliberately (#202)
---------------------------------

There is no discrete search here, and there should not be: difflow avoids
Pyomo and MINLP solvers. What this module provides is the graph
representation, the closed recycle, the constraint handles
(:meth:`constraints`) and the cheap screening bound
(:mod:`difflow_ree.flowsheets.screening`), so an external discrete layer
can drive it. Route out through :mod:`difflow.solvers` (#203), whose
``as_nlp`` / ``as_residual`` wrap a train's continuous subproblem.
Note that discopt's ``CustomCall`` cannot carry binaries *around* a
wrapped flowsheet: the wrapped call is opaque to the MILP, so a discrete
layer must decompose --- enumerate or branch on the topology outside, and
call in for each fixed topology --- rather than hoping the solver will
see the connectivity variables through the wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, get_flows
from difflow_ree.flowsheets.constraints import ConstraintSet
from difflow_ree.flowsheets.modules import REEModule, module_from_dict, pad_stream
from difflow_ree.flowsheets.ports import Port, check_connection


class TopologyError(ValueError):
    """A train's connectivity map is not a solvable flowsheet.

    Raised for references to modules or ports that do not exist,
    duplicate module names, inlets with no source, and outlets driven by
    more than one source.
    """


@dataclass(repr=False)
class Connection(ParamsMixin):
    """One directed edge of the connectivity map.

    Attributes:
        source: ``"<module>.<outlet port>"``.
        dest: ``"<module>.<inlet port>"``.
        tear: Whether this edge is torn, i.e. is a recycle whose
            destination is evaluated before its source. Set by
            :meth:`SeparationTrain.validate`; a caller may force it.
        allow_species_loss: Permit species the destination port does not
            declare; see
            :func:`difflow_ree.flowsheets.ports.check_connection`.
    """

    source: str
    dest: str
    tear: bool = False
    allow_species_loss: bool = False

    @property
    def source_module(self) -> str:
        """Module half of :attr:`source`."""
        return self.source.split(".", 1)[0]

    @property
    def source_port(self) -> str:
        """Port half of :attr:`source`."""
        return self.source.split(".", 1)[1]

    @property
    def dest_module(self) -> str:
        """Module half of :attr:`dest`."""
        return self.dest.split(".", 1)[0]

    @property
    def dest_port(self) -> str:
        """Port half of :attr:`dest`."""
        return self.dest.split(".", 1)[1]

    def to_dict(self) -> dict:
        """A plain JSON-ready dictionary."""
        return {
            "source": self.source,
            "dest": self.dest,
            "tear": self.tear,
            "allow_species_loss": self.allow_species_loss,
        }


@dataclass(repr=False)
class Feed(ParamsMixin):
    """An external stream entering the train.

    Attributes:
        name: Feed name, used as the flowsheet stream name.
        stream: The stream itself.
        dest: ``"<module>.<inlet port>"`` it feeds.
    """

    name: str
    stream: Stream
    dest: str

    @property
    def dest_module(self) -> str:
        """Module half of :attr:`dest`."""
        return self.dest.split(".", 1)[0]

    @property
    def dest_port(self) -> str:
        """Port half of :attr:`dest`."""
        return self.dest.split(".", 1)[1]


@dataclass(repr=False)
class TrainResult(ParamsMixin):
    """The converged state of a train.

    Attributes:
        streams: Every stream, keyed by ``"<module>.<port>"`` for module
            outlets, by feed name for feeds, and by
            ``"<module>.<port>.tear"`` for torn inlets.
        info: Each module's ``info`` dict, keyed by module name.
        converged: Whether the tear solve met its tolerance. ``True`` for
            a train with no recycles.
        iterations: Tear iterations used.
        residual: Final tear residual (max absolute change).
        tear_streams: Names of the torn streams.
    """

    streams: dict[str, Stream]
    info: dict[str, dict] = field(default_factory=dict)
    converged: bool = True
    iterations: int = 0
    residual: float = 0.0
    tear_streams: tuple[str, ...] = ()

    def stream(self, ref: str) -> Stream:
        """One stream by reference.

        Args:
            ref: ``"<module>.<port>"`` or a feed name.

        Returns:
            The stream.

        Raises:
            KeyError: If no such stream exists.
        """
        try:
            return self.streams[ref]
        except KeyError:
            raise KeyError(
                f"no stream {ref!r}; have {sorted(self.streams)}"
            ) from None

    def flows(self, ref: str, elements: tuple[str, ...]) -> dict[str, Array]:
        """Molar flows of selected species in one stream.

        Args:
            ref: Stream reference.
            elements: Species to pull out.

        Returns:
            ``{species: flow}``, zero where absent.
        """
        f = get_flows(self.stream(ref))
        return {
            e: jnp.asarray(f.get(e, 0.0), dtype=jnp.float64) for e in elements
        }

    def purity(self, ref: str, elements: tuple[str, ...]) -> dict[str, Array]:
        """Mole fractions of ``elements`` within their own total.

        Args:
            ref: Stream reference.
            elements: The basis; fractions sum to 1 over this set.

        Returns:
            ``{species: mole fraction}``.
        """
        flows = self.flows(ref, elements)
        total = sum(flows.values(), jnp.asarray(0.0, dtype=jnp.float64))
        total = jnp.where(total > 0.0, total, 1.0)
        return {e: v / total for e, v in flows.items()}


class SeparationTrain:
    """A REE separation train: module instances plus a connectivity map.

    Attributes:
        name: Train name.
        T: Default temperature (K) passed to every module.
        modules: ``{name: module}``, in insertion order.
        connections: The edges.
        feeds: External streams.

    Example:
        >>> train = SeparationTrain("demo")
        >>> train.name
        'demo'
        >>> train.modules
        {}
    """

    def __init__(self, name: str = "train", T: float = 298.15):
        """Create an empty train.

        Args:
            name: Train name.
            T: Default temperature (K).
        """
        self.name = name
        self.T = T
        self.modules: dict[str, REEModule] = {}
        self.connections: list[Connection] = []
        self.feeds: list[Feed] = []
        self._order: list[str] = []

    # -- building --------------------------------------------------------

    def add_module(self, module: REEModule) -> REEModule:
        """Add a module instance.

        Args:
            module: The module.

        Returns:
            The module, so calls can be chained or the result bound.

        Raises:
            TopologyError: If the name is already taken.
        """
        if module.name in self.modules:
            raise TopologyError(
                f"module {module.name!r} is already in train {self.name!r}."
            )
        self.modules[module.name] = module
        return module

    def add_feed(self, name: str, stream: Stream, dest: str) -> None:
        """Add an external feed to a module inlet.

        Args:
            name: Feed name.
            stream: The stream.
            dest: ``"<module>.<inlet port>"``.

        Raises:
            TopologyError: If the destination does not exist or is
                already driven.
        """
        port = self._inlet(dest)
        if self._source_of(dest) is not None:
            raise TopologyError(
                f"inlet {dest!r} already has a source; an inlet takes "
                f"exactly one stream (put a mixer module in front if you "
                f"need two)."
            )
        if port.phase == "solid":
            raise TopologyError(f"cannot feed the solid port {dest!r}.")
        self.feeds.append(Feed(name, stream, dest))

    def connect(
        self,
        source: str,
        dest: str,
        *,
        allow_species_loss: bool = False,
    ) -> Connection:
        """Connect an outlet to an inlet.

        The phases and species vocabularies are checked immediately, so
        an organic outlet plugged into an aqueous inlet raises here
        rather than producing a plausible converged answer.

        Args:
            source: ``"<module>.<outlet port>"``.
            dest: ``"<module>.<inlet port>"``.
            allow_species_loss: Permit species the destination does not
                declare.

        Returns:
            The connection.

        Raises:
            TopologyError: If either reference is unknown, or the
                destination is already driven.
            difflow_ree.flowsheets.ports.PortMismatchError: If the ports
                are not compatible.
        """
        src_port = self._outlet(source)
        dst_port = self._inlet(dest)
        if self._source_of(dest) is not None:
            raise TopologyError(
                f"inlet {dest!r} already has a source "
                f"({self._source_of(dest)!r}); an inlet takes exactly one "
                f"stream."
            )
        check_connection(
            src_port, dst_port,
            source_ref=source, dest_ref=dest,
            allow_species_loss=allow_species_loss,
        )
        conn = Connection(source, dest, allow_species_loss=allow_species_loss)
        self.connections.append(conn)
        return conn

    # -- lookups ---------------------------------------------------------

    def _module(self, ref: str) -> REEModule:
        """The module named by the first half of a dotted reference.

        Args:
            ref: ``"<module>.<port>"``.

        Returns:
            The module.

        Raises:
            TopologyError: If the reference is malformed or unknown.
        """
        if "." not in ref:
            raise TopologyError(
                f"{ref!r} is not a port reference; use '<module>.<port>'."
            )
        name = ref.split(".", 1)[0]
        if name not in self.modules:
            raise TopologyError(
                f"no module {name!r} in train {self.name!r}; have "
                f"{sorted(self.modules)}."
            )
        return self.modules[name]

    def _inlet(self, ref: str) -> Port:
        """The inlet port a reference names.

        Args:
            ref: ``"<module>.<inlet port>"``.

        Returns:
            The port.

        Raises:
            TopologyError: If the port does not exist.
        """
        module = self._module(ref)
        try:
            return module.ports.inlet(ref.split(".", 1)[1])
        except KeyError as exc:
            raise TopologyError(str(exc)) from None

    def _outlet(self, ref: str) -> Port:
        """The outlet port a reference names.

        Args:
            ref: ``"<module>.<outlet port>"``.

        Returns:
            The port.

        Raises:
            TopologyError: If the port does not exist.
        """
        module = self._module(ref)
        try:
            return module.ports.outlet(ref.split(".", 1)[1])
        except KeyError as exc:
            raise TopologyError(str(exc)) from None

    def _source_of(self, dest: str) -> str | None:
        """What drives an inlet, or None.

        Args:
            dest: ``"<module>.<inlet port>"``.

        Returns:
            The source reference or feed name.
        """
        for conn in self.connections:
            if conn.dest == dest:
                return conn.source
        for feed in self.feeds:
            if feed.dest == dest:
                return feed.name
        return None

    @property
    def species_order(self) -> list[str]:
        """The union of every port's declared species, in a stable order.

        This is the vocabulary the flowsheet's tear packing works in, so
        every module's outlets are padded to it; see
        :func:`difflow_ree.flowsheets.modules.pad_stream`.
        """
        seen: list[str] = []
        for module in self.modules.values():
            for s in module.species:
                if s not in seen:
                    seen.append(s)
        for feed in self.feeds:
            for s in get_flows(feed.stream):
                if s not in seen:
                    seen.append(s)
        return seen

    # -- validation and ordering -----------------------------------------

    def validate(self) -> list[str]:
        """Check the graph and compute the evaluation order.

        Marks back edges as torn: a depth-first pass over the modules
        finds edges pointing at a module already on the stack, and those
        become the recycles handed to
        :meth:`difflow.flowsheet.Flowsheet.add_recycle`.

        Returns:
            Module names in evaluation order.

        Raises:
            TopologyError: If a module inlet has no source.
        """
        for name, module in self.modules.items():
            for port in module.ports.inlets:
                ref = f"{name}.{port.name}"
                if self._source_of(ref) is None:
                    raise TopologyError(
                        f"inlet {ref!r} has no source. Every module inlet "
                        f"needs a feed or a connection; an unconnected "
                        f"organic inlet is exactly the open loop #202 is "
                        f"about."
                    )

        # Adjacency over module names, ignoring self-edges for the search
        # (a module feeding itself is always a tear).
        succ: dict[str, list[Connection]] = {n: [] for n in self.modules}
        for conn in self.connections:
            succ[conn.source_module].append(conn)

        state: dict[str, int] = {n: 0 for n in self.modules}  # 0 new 1 open 2 done
        order: list[str] = []
        for conn in self.connections:
            conn.tear = False

        def visit(node: str) -> None:
            state[node] = 1
            for conn in succ[node]:
                target = conn.dest_module
                if target == node or state[target] == 1:
                    conn.tear = True
                elif state[target] == 0:
                    visit(target)
            state[node] = 2
            order.append(node)

        for name in self.modules:
            if state[name] == 0:
                visit(name)

        self._order = list(reversed(order))
        return list(self._order)

    @property
    def tear_connections(self) -> tuple[Connection, ...]:
        """The torn edges, after :meth:`validate`."""
        return tuple(c for c in self.connections if c.tear)

    # -- flowsheet -------------------------------------------------------

    def to_flowsheet(self, tear_initial: dict[str, Stream] | None = None):
        """Build a :class:`difflow.flowsheet.Flowsheet` for this train.

        Every module becomes one :class:`difflow.flowsheet.Unit` whose
        operation is a thin adapter that pads the module's outlets to the
        train's species vocabulary and stashes its ``info``. Torn edges
        become ``add_recycle`` calls, which is what closes the organic
        loop on the existing tear solver.

        Args:
            tear_initial: Unused here; kept so callers can see that the
                initial guess belongs to the solve, not the topology.

        Returns:
            The flowsheet.

        Raises:
            TopologyError: From :meth:`validate`.
        """
        from difflow.flowsheet import Flowsheet, Unit

        order = self.validate()
        species = self.species_order
        fs = Flowsheet(species_order=species)

        for feed in self.feeds:
            fs.add_feed(feed.name, pad_stream(feed.stream, tuple(species)))

        self._adapters: dict[str, _ModuleAdapter] = {}
        for name in order:
            module = self.modules[name]
            adapter = _ModuleAdapter(module, tuple(species), self.T)
            self._adapters[name] = adapter
            inlet_names = []
            for port in module.ports.inlets:
                ref = f"{name}.{port.name}"
                conn = next(
                    (c for c in self.connections if c.dest == ref), None
                )
                if conn is None:
                    inlet_names.append(self._source_of(ref))  # a feed name
                elif conn.tear:
                    inlet_names.append(_tear_name(ref))
                else:
                    inlet_names.append(conn.source)
            fs.add_unit(Unit(
                name=name,
                operation=adapter,
                inlet_names=inlet_names,
                outlet_names=[f"{name}.{p.name}" for p in module.ports.outlets],
            ))

        # Flowsheet.add_recycle keys its recycles by source, so two tears
        # sharing one outlet would collapse to one: the second call overwrites
        # the first, the dropped tear keeps whatever default_tear_streams
        # seeded it with, and solve() still reports convergence because the
        # equation it should have closed is not in the residual at all.
        # Silently converging on an unsolved tear is exactly what this module
        # exists to prevent, so refuse it rather than let it pass (#202).
        seen: dict[str, str] = {}
        for conn in self.tear_connections:
            previous = seen.get(conn.source)
            if previous is not None:
                raise TopologyError(
                    f"tear source {conn.source!r} feeds both {previous!r} and "
                    f"{conn.dest!r}. Flowsheet.add_recycle keys recycles by "
                    "source, so one of the two tears would be dropped and its "
                    "destination never solved, while the train still reported "
                    "convergence. Split the outlet before tearing it."
                )
            seen[conn.source] = conn.dest
            fs.add_recycle(conn.source, _tear_name(conn.dest))
        return fs

    def default_tear_streams(self) -> dict[str, Stream]:
        """Initial guesses for the torn streams: the open-loop assumption.

        A torn organic inlet is initialised with the fresh, REE-free
        solvent the open-loop circuit would have synthesised, which is
        both the physically sensible starting point and, literally, the
        assumption the closed loop is there to test (#202).

        Returns:
            ``{tear stream name: stream}``, ready for
            :meth:`difflow.flowsheet.Flowsheet.solve`'s ``tear_initial``.
        """
        species = tuple(self.species_order)
        guesses: dict[str, Stream] = {}
        for conn in self.tear_connections:
            module = self.modules[conn.dest_module]
            fresh = getattr(module, "fresh_solvent", None)
            if fresh is None:
                continue
            feed = self._upstream_feed(conn.dest_module)
            if feed is None:
                continue
            guesses[_tear_name(conn.dest)] = pad_stream(
                fresh(feed, self.T), species
            )
        return guesses

    def _upstream_feed(self, module_name: str) -> Stream | None:
        """An external feed reaching a module, for sizing its solvent.

        Walks backwards over the untorn edges, so a module in the middle
        of a chain still gets an initial guess of roughly the right
        magnitude rather than the flowsheet's generic 0.01 mol/s default.

        Args:
            module_name: The module.

        Returns:
            The stream, or None if no feed reaches the module.
        """
        seen = {module_name}
        queue = [module_name]
        while queue:
            node = queue.pop(0)
            for feed in self.feeds:
                if feed.dest_module == node:
                    return feed.stream
            for conn in self.connections:
                if conn.tear or conn.dest_module != node:
                    continue
                if conn.source_module not in seen:
                    seen.add(conn.source_module)
                    queue.append(conn.source_module)
        return None

    # -- solving ---------------------------------------------------------

    def solve(
        self,
        tear_initial: dict[str, Stream] | None = None,
        tol: float = 1e-10,
        max_iter: int = 200,
        acceleration: str = "anderson",
    ) -> TrainResult:
        """Solve the train, converging every torn loop.

        Delegates to :meth:`difflow.flowsheet.Flowsheet.solve`; nothing
        in this class iterates.

        Args:
            tear_initial: Initial guesses for torn streams. None uses
                :meth:`default_tear_streams`.
            tol: Tear tolerance.
            max_iter: Maximum tear iterations.
            acceleration: ``"anderson"``, ``"wegstein"`` or ``"none"``.

        Returns:
            The :class:`TrainResult`.
        """
        fs = self.to_flowsheet()
        guesses = (
            self.default_tear_streams() if tear_initial is None
            else {
                (k if k.endswith(".tear") else _tear_name(k)): v
                for k, v in tear_initial.items()
            }
        )
        streams = fs.solve(
            tear_initial=guesses or None,
            tol=tol,
            max_iter=max_iter,
            acceleration=acceleration,
        )
        return TrainResult(
            streams=streams,
            info={n: a.last_info for n, a in self._adapters.items()},
            converged=bool(fs.last_solve_converged),
            iterations=int(fs.last_solve_iterations or 0),
            residual=float(fs.last_solve_residual or 0.0),
            tear_streams=tuple(fs.last_solve_tear_streams),
        )

    def solve_differentiable(
        self,
        tear_initial: dict[str, Stream] | None = None,
        tol: float = 1e-10,
        max_iter: int = 200,
    ) -> dict[str, Stream]:
        """Solve the train inside a ``jit``/``grad`` trace.

        Same graph, same units, same tear set as :meth:`solve` --- the
        flowsheet is built by :meth:`to_flowsheet` either way. Only the
        outer loop differs: ``Flowsheet.solve`` records ``float``
        diagnostics after each step, which cannot be traced, so this runs
        the flowsheet's own tear map through ``optimistix.fixed_point``
        and gets implicit differentiation through the converged solution
        for free.

        Args:
            tear_initial: Initial guesses for torn streams. None uses
                :meth:`default_tear_streams`.
            tol: Fixed-point tolerance.
            max_iter: Maximum fixed-point steps.

        Returns:
            Every stream, keyed as in :attr:`TrainResult.streams`.
        """
        import lineax as lx
        import optimistix as optx

        fs = self.to_flowsheet()
        if not fs.recycles:
            return fs.solve()

        guesses = (
            self.default_tear_streams() if tear_initial is None
            else {
                (k if k.endswith(".tear") else _tear_name(k)): v
                for k, v in tear_initial.items()
            }
        )
        tear_names = list(fs.recycles.values())
        initial = {
            name: guesses.get(name) or fs._make_zero_stream()
            for name in tear_names
        }
        x0 = fs._streams_to_array(initial)

        def tear_map(x, args):
            tear = fs._array_to_streams(x, tear_names)
            streams = dict(fs.feeds)
            streams.update(tear)
            streams = fs._run_units(streams)
            return fs._streams_to_array(
                {dest: streams[src] for src, dest in fs.recycles.items()}
            )

        # A closed organic loop is structurally singular in its carrier
        # coordinates: the extractant and diluent flows come out of the
        # loop exactly as they went in, so those rows of ``I - dg/dx`` are
        # zero and *any* solvent inventory is a fixed point. The inventory
        # is a design degree of freedom (what the make-up pumps hold),
        # not something the loop determines. A well-posed linear solve in
        # the implicit-differentiation adjoint therefore fails outright;
        # the least-squares solver takes the minimum-norm solution, which
        # is the one that holds the inventory fixed --- the physically
        # intended choice (#202).
        sol = optx.fixed_point(
            tear_map,
            optx.FixedPointIteration(rtol=tol, atol=tol),
            x0,
            max_steps=max_iter,
            throw=False,
            adjoint=optx.ImplicitAdjoint(
                linear_solver=lx.AutoLinearSolver(well_posed=False)
            ),
        )
        final = fs._array_to_streams(sol.value, tear_names)
        streams = dict(fs.feeds)
        streams.update(final)
        return fs._run_units(streams)

    # -- constraints -----------------------------------------------------

    def constraints(self, result: TrainResult | dict) -> ConstraintSet:
        """Every module's operating-boundary constraints, pooled.

        Args:
            result: A :class:`TrainResult`, or the ``{module: info}``
                dict directly.

        Returns:
            The pooled :class:`~difflow_ree.flowsheets.constraints.ConstraintSet`,
            feasible when every margin is ``>= 0``.
        """
        infos = result.info if isinstance(result, TrainResult) else result
        out = ConstraintSet()
        for name, module in self.modules.items():
            info = infos.get(name)
            if info:
                out = out + module.constraints(info)
        return out

    # -- reporting and serialization --------------------------------------

    def describe(self) -> dict:
        """A machine-readable description of the train.

        Returns:
            Dict with ``name``, ``modules`` (each module's
            :meth:`~difflow_ree.flowsheets.modules.REEModule.describe`,
            which embeds the core :mod:`difflow.catalog` schemas),
            ``connections``, ``feeds`` and the evaluation ``order``.
        """
        return {
            "name": self.name,
            "T": self.T,
            "modules": [m.describe() for m in self.modules.values()],
            "connections": [c.to_dict() for c in self.connections],
            "feeds": [{"name": f.name, "dest": f.dest} for f in self.feeds],
            "order": list(self._order) or list(self.modules),
        }

    def to_dict(self) -> dict:
        """Serialize the train --- modules, feeds and connectivity.

        One level above :func:`difflow.serialize.to_dict`, which writes
        the *flowsheet* :meth:`to_flowsheet` produces. Both are useful:
        the flowsheet form is what a solver or GUI reads, and this one is
        what an external topology search enumerates, because it holds the
        connectivity map explicitly.

        The parts that overlap are delegated rather than duplicated:
        streams and parameter values go through
        :mod:`difflow.serialize`'s own encoders, and the file carries its
        :data:`difflow.serialize.FORMAT_VERSION`.

        Note that the ``Flowsheet`` from :meth:`to_flowsheet` is not
        itself writable by :func:`difflow.serialize.to_dict` today: its
        unit operations are module adapters rather than classes in the
        operation registry, and ``to_dict`` refuses (correctly) to write
        an operation nothing could rebuild. Registering the module
        classes would close that gap; until then this is the train's
        file format.

        Returns:
            JSON-ready dict.
        """
        from difflow.serialize import FORMAT_VERSION, _encode_stream

        return {
            "format_version": FORMAT_VERSION,
            "name": self.name,
            "T": self.T,
            "modules": [m.to_dict() for m in self.modules.values()],
            "connections": [c.to_dict() for c in self.connections],
            "feeds": [
                {
                    "name": f.name,
                    "dest": f.dest,
                    "stream": _encode_stream(f.stream, f.name),
                }
                for f in self.feeds
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SeparationTrain":
        """Rebuild a train from :meth:`to_dict` output.

        Args:
            data: The dictionary to read.

        Returns:
            The train, validated.

        Raises:
            difflow.serialize.SerializationError: On an unsupported
                format version.
        """
        from difflow.serialize import (
            FORMAT_VERSION,
            SerializationError,
            _decode_stream,
        )

        version = data.get("format_version")
        if version is not None and version != FORMAT_VERSION:
            raise SerializationError(
                f"format version {version!r} is not supported; this "
                f"difflow reads version {FORMAT_VERSION}."
            )
        train = cls(data.get("name", "train"), data.get("T", 298.15))
        for spec in data.get("modules", []):
            train.add_module(module_from_dict(spec))
        for spec in data.get("feeds", []):
            train.add_feed(
                spec["name"],
                _decode_stream(spec["stream"]),
                spec["dest"],
            )
        for spec in data.get("connections", []):
            train.connect(
                spec["source"], spec["dest"],
                allow_species_loss=spec.get("allow_species_loss", False),
            )
        train.validate()
        return train

    def to_json(self, indent: int = 2) -> str:
        """:meth:`to_dict` as JSON text.

        Args:
            indent: JSON indentation.

        Returns:
            The JSON string.
        """
        import json

        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "SeparationTrain":
        """Rebuild a train from :meth:`to_json` output.

        Args:
            text: The JSON string.

        Returns:
            The train.
        """
        import json

        return cls.from_dict(json.loads(text))

    def __repr__(self) -> str:
        return (
            f"SeparationTrain(name={self.name!r}, "
            f"modules={list(self.modules)}, "
            f"connections={len(self.connections)}, "
            f"tears={len(self.tear_connections)})"
        )


def _tear_name(dest: str) -> str:
    """Flowsheet stream name for a torn inlet.

    Args:
        dest: ``"<module>.<inlet port>"``.

    Returns:
        The tear stream name.
    """
    return f"{dest}.tear"


class _ModuleAdapter:
    """Make a :class:`REEModule` callable as a flowsheet unit operation.

    Pads every outlet to the train's species vocabulary (the tear packing
    in :class:`difflow.flowsheet.Flowsheet` requires it) and keeps the
    last ``info`` so the constraints can be read after the solve.

    Attributes:
        module: The wrapped module.
        species: The train's species vocabulary.
        T: Temperature (K).
        last_info: The most recent run's ``info``. Valid after the
            flowsheet's final pass, which is the converged one.
    """

    def __init__(self, module: REEModule, species: tuple[str, ...], T: float):
        """Bind a module to a species vocabulary.

        Args:
            module: The module to wrap.
            species: The train's species vocabulary.
            T: Temperature (K).
        """
        self.module = module
        self.species = species
        self.T = T
        self.last_info: dict = {}
        # difflow.serialize reaches for ``.params`` when writing a unit.
        self.params = getattr(module, "params", None)

    def __call__(self, *inlets: Stream):
        """Run the module and pad its outlets.

        Args:
            *inlets: Inlet streams, in inlet-port order.

        Returns:
            Padded outlet streams followed by the module's ``info``.
        """
        result = self.module(*inlets, T=self.T)
        *streams, info = result
        self.last_info = info
        return (*[pad_stream(s, self.species) for s in streams], info)

    def __hash__(self):
        return hash((self.module.name, id(self.module)))
