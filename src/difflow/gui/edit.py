"""Small edits to a live flowsheet: one unit, one wire, one position.

The editor used to send the whole document on every change and rebuild
every unit from it. That is correct and it is what :meth:`replace` still
does, but it makes a keystroke cost a full reconstruction, and it makes
the *live* constructor objects --- a ``thermo``, an ``eos`` --- take a
round trip through JSON that some of them cannot survive.

So the operations here work on the flowsheet in place, and rebuild only
what the edit touched. A patched unit is rebuilt through
:func:`difflow.serialize._build_operation`, the same path a file load
takes, so what the editor accepts is exactly what will reload --- but
with the constructor objects passed in *live*, by identity, rather than
encoded and decoded.

Everything here raises :class:`EditError` on a bad edit. The session
turns that into ``{"ok": False, "error": ...}``; nothing here writes to a
socket or knows there is one.
"""

from __future__ import annotations

from difflow.serialize import (
    SerializationError,
    _build_operation,
    _encode_value,
    constructor_extras,
)


class EditError(ValueError):
    """A rejected edit. Carries the message the editor should show."""


# ---------------------------------------------------------------------
# Reading the graph
# ---------------------------------------------------------------------

def unit(flowsheet, name: str):
    """The unit called ``name``, or an :class:`EditError` naming the others."""
    for u in flowsheet.units:
        if u.name == name:
            return u
    known = ", ".join(u.name for u in flowsheet.units) or "none"
    raise EditError(f"no unit called {name!r} (have: {known})")


def producers(flowsheet) -> dict[str, str]:
    """Stream name -> the unit that makes it."""
    return {s: u.name for u in flowsheet.units for s in u.outlet_names}


def unfed(flowsheet) -> list[str]:
    """Inlets with nothing on the other end, in the order they are read.

    An inlet is fed by exactly one of three things: a declared feed, a
    unit's outlet, or a recycle arc. An inlet fed by none of them is what
    the canvas draws as a feed node with nothing behind it, and what
    ``Flowsheet.solve`` reports as a bare ``KeyError`` on the stream name
    -- a message that names the symptom and nothing else.

    Kept here rather than in the session because the canvas asks the same
    question (`graph.js:feedStreams`) and the two answers have to agree:
    a node drawn as an unfed inlet that the solver is happy with, or the
    other way round, is a picture that lies.
    """
    made = producers(flowsheet)
    recycled = set(flowsheet.recycles.values())
    out = []
    for u in flowsheet.units:
        for inlet in u.inlet_names:
            if (inlet not in flowsheet.feeds and inlet not in made
                    and inlet not in recycled and inlet not in out):
                out.append(inlet)
    return out


def stream_names(flowsheet) -> set[str]:
    """Every stream name in use, feeds and recycle destinations included."""
    names = set(flowsheet.feeds) | set(flowsheet.recycles.values())
    for u in flowsheet.units:
        names.update(u.inlet_names)
        names.update(u.outlet_names)
    return names


def unique(base: str, taken) -> str:
    """``base``, or ``base2``, ``base3``... --- the first one free."""
    if base not in taken:
        return base
    n = 2
    while f"{base}{n}" in taken:
        n += 1
    return f"{base}{n}"


def reaches(flowsheet, start: str, goal: str) -> bool:
    """Whether ``goal`` is downstream of ``start`` along non-recycle arcs.

    This is what decides whether a new wire is an ordinary connection or
    a recycle: a wire back into something that already feeds the source
    closes a loop, and difflow spells a loop as a tear
    (:meth:`~difflow.flowsheet.Flowsheet.add_recycle`), never as an arc.
    Recycle arcs are excluded from the walk --- counting them would make
    every unit in an existing loop reach every other, and a second,
    genuinely new loop would then be mistaken for one of them.
    """
    downstream: dict[str, set[str]] = {}
    for u in flowsheet.units:
        arcs = {s for s in u.outlet_names if s not in flowsheet.recycles}
        if not arcs:
            continue
        downstream[u.name] = {
            other.name for other in flowsheet.units
            if arcs.intersection(other.inlet_names)
        }

    seen = {start}
    stack = [start]
    while stack:
        for nxt in downstream.get(stack.pop(), ()):
            if nxt == goal:
                return True
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


# ---------------------------------------------------------------------
# Rebuilding one unit
# ---------------------------------------------------------------------

def live_extras(operation) -> dict:
    """An operation's non-``Params`` constructor arguments, as objects.

    Read off the instance by attribute --- the convention
    :func:`~difflow.serialize.constructor_extras` documents --- and
    handed back to the constructor unchanged. A ``thermo`` survives an
    edit by identity rather than by round-tripping through JSON, which
    is both faster and the only thing that works for the kinds
    :mod:`difflow.serialize` cannot write.
    """
    out = {}
    for arg in constructor_extras(type(operation)):
        value = getattr(operation, arg, None)
        if value is not None:
            out[arg] = value
    return out


def species_order(flowsheet, bindings: dict | None = None) -> list[str]:
    """The species list to build a new unit with.

    The flowsheet's own order first --- it is what its streams are
    indexed by, and a unit built against anything else would disagree
    with every stream it touches. A code context's list is the fallback,
    for the flowsheet that has none yet, and ``SPECIES`` counts as well
    as ``species_order``: that is the name a difflow script uses and the
    name the editor's starter snippet writes, and looking only for the
    other one left the snippet unable to unblock the palette.
    """
    from difflow.gui.session import species_from

    order = list(getattr(flowsheet, "species_order", None) or [])
    return order or (species_from(bindings or {}) or [])


def known_extras(flowsheet, cls: type, bindings: dict | None = None) -> dict:
    """Constructor arguments a new unit can take without being told.

    A unit dropped from the palette has no file to read ``extras=`` out
    of, so anything required has to come from somewhere. Two places:

    * ``species_order`` from the flowsheet, which already holds it and
      which is what blocks the most operations --- ``Mixer``,
      ``Splitter``, and the rest that take it instead of a ``Params``;
    * a ``thermo`` or an ``eos`` from the code context, matched by name
      first and then by type. The name match is the predictable one, and
      the type match is what makes ``thermo = IdealThermo(...)`` --- the
      line every difflow script opens with --- enough to drop a Flash.

    An ambiguous type match is left out rather than guessed at, so the
    refusal names what is missing instead of building the wrong unit.
    """
    out = {}
    order = species_order(flowsheet, bindings)
    for arg in constructor_extras(cls):
        if arg == "species_order" and order:
            out[arg] = order
        elif bindings and arg in bindings:
            out[arg] = bindings[arg]
        elif bindings:
            fits = [v for v in bindings.values()
                    if arg in type(v).__name__.lower()]
            if len(fits) == 1:
                out[arg] = fits[0]
    return out


#: What a numeric parameter with no default is set to when a unit is
#: dropped on the canvas. Not a physical claim --- it is a placeholder,
#: reported as one, and the first thing the inspector asks about.
PLACEHOLDER = 1.0


def _is_number(annotation) -> bool:
    """Whether a dataclass field annotation admits a plain number.

    Read off the text of the annotation rather than by comparing types:
    the interesting ones are unions (``float | jax.Array``) and strings
    (under ``from __future__ import annotations``), and both defeat an
    identity test while reading perfectly well.
    """
    text = str(annotation)
    if any(bad in text for bad in ("Callable", "dict", "list", "tuple")):
        return False
    return "float" in text or "int" in text


def placeholder_extras(cls: type, have: dict) -> tuple[dict, list[str]]:
    """Constructor arguments that are plain numbers, given a placeholder.

    The same bargain :func:`known_params` already strikes for a required
    ``Params`` field, applied to the other half of the constructor. A
    unit that builds its own ``Params`` from plain arguments ---
    ``GasPipe(beta)``, ``CompressorBoost(ratio, direction)`` --- was
    reported unmet for a plain ``float``, while an identical field one
    line away in a ``Params`` got :data:`PLACEHOLDER` and dropped fine.
    Nothing chose that; the placeholder path simply only ran on one of
    the two.

    ``have`` is everything already supplied, and it must include what
    :func:`known_params` answered as well as the caller's own extras.
    Where :func:`difflow.catalog._params_class` does find the ``Params``
    --- ``Compressor`` and ``CompressorParams`` are named for each other,
    ``GasPipe`` and ``PipeParams`` are not --- the number is answered
    there and is not this function's to invent a second time.

    Only annotations :func:`_is_number` admits, so a ``thermo`` or a
    ``tuple[float, ...]`` is still named rather than invented. An
    ``int``-only argument gets an ``int``, since a discrete one --- a
    ``direction`` of 1 --- is the whole value of the annotation.

    Returns ``(values, placeholders)``, and like ``known_params`` the
    second list is what has to be shown to the user rather than trusted.
    """
    import inspect

    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return {}, []
    values, placeholders = {}, []
    for arg in constructor_extras(cls):
        if arg in have:
            continue
        annotation = sig.parameters[arg].annotation
        if not _is_number(annotation):
            continue
        text = str(annotation)
        values[arg] = (int(PLACEHOLDER) if "int" in text and "float" not in text
                       else PLACEHOLDER)
        placeholders.append(arg)
    return values, placeholders


def known_params(flowsheet, params_cls, bindings: dict | None = None):
    """Values for the ``Params`` fields that have no default.

    A palette drop has nothing to say about parameters, but a ``Params``
    class with a required field cannot be constructed from nothing ---
    which is why half the catalog reads as unplaceable. Four sources, in
    order:

    * ``species_order`` from the flowsheet;
    * a binding of the same name from the code context;
    * a binding that offers ``params_kwargs()`` and has the field in it.
      This is the declarative route:
      :func:`~difflow.kinetics.mass_action_kinetics` returns exactly
      such an object, and one ``kin = mass_action_kinetics(...)`` in the
      code context is what makes a reactor droppable --- ``rate_fn``,
      ``stoich`` and ``rate_params`` all arrive together and consistent,
      which is the point of building a rate law from data;
    * for a plain number and nothing else, :data:`PLACEHOLDER`.

    Returns:
        ``(values, placeholders, missing)``. ``placeholders`` names the
        fields that got a made-up number and so must be shown to the
        user rather than trusted; ``missing`` names the required fields
        that none of the four sources could supply, which is what makes
        the unit undroppable until the code context grows a binding.
    """
    import dataclasses

    if params_cls is None or not dataclasses.is_dataclass(params_cls):
        return {}, [], []
    order = species_order(flowsheet, bindings)
    kwargs = {}
    for value in (bindings or {}).values():
        supplier = getattr(value, "params_kwargs", None)
        if callable(supplier):
            try:
                kwargs.update(supplier())
            except Exception:            # not the kind of object we hoped
                pass

    values, placeholders, missing = {}, [], []
    for f in dataclasses.fields(params_cls):
        if (f.default is not dataclasses.MISSING
                or f.default_factory is not dataclasses.MISSING):
            continue
        if f.name == "species_order" and order:
            values[f.name] = order
        elif bindings and f.name in bindings:
            values[f.name] = bindings[f.name]
        elif f.name in kwargs:
            values[f.name] = kwargs[f.name]
        elif _is_number(f.type):
            values[f.name] = PLACEHOLDER
            placeholders.append(f.name)
        else:
            # A rate law, a stoichiometry, a solvent name: required, and
            # not a number, so there is nothing honest to invent. Naming
            # it is the whole answer -- it is what the palette flags and
            # what the refusal quotes.
            missing.append(f.name)
    return values, placeholders, missing


def unmet(flowsheet, cls: type, bindings: dict | None = None) -> list[str]:
    """What a palette drop of ``cls`` cannot supply for itself.

    The one definition of "droppable", used by both the served catalog
    and :meth:`~difflow.gui.session.FlowsheetSession.add_unit`, because
    a palette that promises a unit the adder then refuses is worse than
    either being wrong alone. It reports constructor objects and
    required ``Params`` fields in one list, since from the canvas they
    are the same problem: something has to exist before this unit can.

    It is answered against the *current* bindings, so it changes as the
    code context grows. One ``thermo = IdealThermo(...)`` empties this
    list for every unit that was waiting on a ``thermo``.
    """
    from difflow.catalog import _params_class

    have = known_extras(flowsheet, cls, bindings)
    values, _, missing = known_params(flowsheet, _params_class(cls), bindings)
    # A unit that builds its own `Params` names the same number twice:
    # once as a constructor argument and once as the field it goes into.
    # `known_params` has already answered for the field, placeholder and
    # all, so asking again under the argument's name reports a need that
    # is met. What it cannot reach --- a `PipeParams` behind a `GasPipe`,
    # which is not found by name -- is what the placeholder is for.
    supplied = {**values, **have}
    guessed, _ = placeholder_extras(cls, supplied)
    needs = [a for a in constructor_extras(cls)
             if a not in supplied and a not in guessed]
    return needs + missing


def encoded_params(operation, unit_name: str) -> dict:
    """One operation's ``Params`` as :func:`difflow.serialize.to_dict` writes it.

    Going through the encoding rather than the live dataclass is
    deliberate: it means a parameter the editor accepts is a parameter
    the file can carry, so an edit cannot produce a flowsheet that
    refuses to save.
    """
    import dataclasses

    params = getattr(operation, "params", None)
    if params is None or not dataclasses.is_dataclass(params):
        return {}
    return {
        f.name: _encode_value(getattr(params, f.name),
                              f"unit {unit_name!r} field {f.name!r}")
        for f in dataclasses.fields(params)
    }


def rebuild(operation, unit_name: str, updates: dict):
    """A copy of ``operation`` with ``updates`` applied to its parameters."""
    import dataclasses

    params = getattr(operation, "params", None)
    if params is None or not dataclasses.is_dataclass(params):
        raise EditError(
            f"{type(operation).__name__} takes no parameters, so there is "
            f"nothing on {unit_name!r} to set."
        )
    fields = {f.name for f in dataclasses.fields(params)}
    unknown = sorted(set(updates) - fields)
    if unknown:
        raise EditError(
            f"{type(params).__name__} has no field "
            f"{', '.join(repr(u) for u in unknown)}; it has "
            f"{', '.join(sorted(fields))}."
        )
    try:
        encoded = encoded_params(operation, unit_name)
    except SerializationError as exc:
        # A hand-written callable parameter. The same edit through
        # `POST /api/flowsheet` fails identically, and for the same
        # reason -- say so rather than half-applying it.
        raise EditError(
            f"unit {unit_name!r} cannot be edited field by field: {exc}"
        ) from exc
    encoded.update(updates)
    try:
        return _build_operation(type(operation), encoded, unit_name,
                                override=live_extras(operation))
    except SerializationError as exc:
        raise EditError(str(exc)) from exc


# ---------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------

def rename_stream(flowsheet, old: str, new: str) -> None:
    """Rename a stream everywhere it appears.

    A stream name *is* the wiring in difflow, so renaming one is how a
    connection is made and broken. Every place the name occurs has to
    move together --- feeds, both ends of a recycle, every unit that
    reads or writes it --- or the flowsheet silently splits into two
    graphs that each look fine on their own.
    """
    for u in flowsheet.units:
        u.inlet_names = [new if s == old else s for s in u.inlet_names]
        u.outlet_names = [new if s == old else s for s in u.outlet_names]
    if old in flowsheet.feeds:
        flowsheet.feeds[new] = flowsheet.feeds.pop(old)
    flowsheet.recycles = {
        (new if src == old else src): (new if dst == old else dst)
        for src, dst in flowsheet.recycles.items()
    }
    nodes = (flowsheet.view or {}).get("nodes")
    if isinstance(nodes, dict):
        for prefix in ("feed:", "product:"):
            if prefix + old in nodes:
                nodes[prefix + new] = nodes.pop(prefix + old)


def rename(flowsheet, old: str, new: str) -> dict:
    """Rename one stream, refusing the renames that would rewire.

    :func:`rename_stream` is the mechanism and does no checking, because
    :func:`connect` uses it precisely *to* rewire --- wiring an outlet to
    an inlet in difflow is renaming the inlet to match. Asked for by a
    user, though, the same call has two ways to do something other than
    what was asked:

    * a name already in use merges two streams into one, which is a
      connection and should be drawn as one;
    * a name that is not a Python identifier survives here and fails
      later, in ``codegen`` or on reload, a long way from the typing.

    So this is the checked door and ``rename_stream`` is the unchecked
    one, and both stay.
    """
    old, new = str(old or ""), str(new or "").strip()
    names = stream_names(flowsheet)
    if old not in names:
        raise EditError(f"no stream called {old!r}")
    if not new:
        raise EditError("a stream needs a name")
    if new == old:
        return {"kind": "stream", "stream": old}
    if not new.isidentifier():
        raise EditError(
            f"{new!r} cannot be a stream name: it has to be a Python "
            "identifier, because that is what the exported script calls it."
        )
    if new in names:
        raise EditError(
            f"{new!r} is already a stream. Renaming onto it would join the "
            "two into one, which is a connection --- draw it as one."
        )
    rename_stream(flowsheet, old, new)
    return {"kind": "stream", "stream": new}


def variadic(ports: dict) -> bool:
    """Whether this operation takes as many inlets as it is given."""
    return bool(ports.get("variadic"))


def add_inlet(flowsheet, name: str, ports: dict) -> dict:
    """Give a variadic unit one more inlet, dangling.

    A ``Mixer`` mixes however many streams it is handed, so its inlet
    count is a property of the flowsheet and not of the class --- which
    means it is something the canvas has to be able to change, and until
    now the only way to get a third inlet was to write the JSON by hand.

    The new port arrives unwired, as a dangling stream, which is what a
    unit dropped from the palette does too: an inlet with nothing behind
    it draws as a feed node waiting to be filled in.
    """
    u = unit(flowsheet, name)
    if not variadic(ports):
        n = len(u.inlet_names)
        raise EditError(
            f"{name!r} takes exactly {n} inlet{'s' if n != 1 else ''}; "
            "only a mixer-like unit can take more."
        )
    # Count up to the first free ordinal rather than letting `unique`
    # bolt a digit onto a taken base: with `mixer_in3` in use that gives
    # `mixer_in32`, which reads as the thirty-second inlet.
    taken = stream_names(flowsheet)
    n = len(u.inlet_names) + 1
    while f"{name}_in{n}" in taken:
        n += 1
    fresh = f"{name}_in{n}"
    u.inlet_names = [*u.inlet_names, fresh]
    return {"kind": "inlet", "stream": fresh}


def remove_inlet(flowsheet, name: str, stream: str, ports: dict) -> dict:
    """Take one inlet off a variadic unit, if nothing is on it.

    Refused rather than cascaded when the port is wired or fed. Deleting
    a port and silently deleting the feed behind it are two edits, and
    the second one is not recoverable by undoing the first --- the feed's
    composition, temperature and pressure are gone. So the refusal names
    what is in the way and the user removes that first.
    """
    u = unit(flowsheet, name)
    if not variadic(ports):
        raise EditError(
            f"{name!r} has a fixed set of inlets; {stream!r} cannot be removed."
        )
    if stream not in u.inlet_names:
        have = ", ".join(u.inlet_names) or "none"
        raise EditError(f"{name!r} has no inlet {stream!r} (has: {have})")
    if len(u.inlet_names) <= 1:
        raise EditError(
            f"{name!r} is down to its last inlet. Delete the unit instead."
        )
    if stream in flowsheet.feeds:
        raise EditError(
            f"{stream!r} is a feed. Remove the feed first, so its conditions "
            "are not thrown away with the port."
        )
    made = producers(flowsheet)
    if stream in made:
        raise EditError(
            f"{stream!r} comes from {made[stream]!r}. Disconnect it first."
        )
    if stream in flowsheet.recycles.values():
        source = next(s for s, d in flowsheet.recycles.items() if d == stream)
        raise EditError(
            f"{stream!r} is the destination of the recycle from {source!r}. "
            "Remove the recycle first."
        )
    u.inlet_names = [s for s in u.inlet_names if s != stream]
    return {"kind": "inlet", "stream": stream}


def connect(flowsheet, source: str, outlet: str, target: str, inlet: str) -> dict:
    """Wire one unit's outlet to another's inlet.

    Returns what it did: ``{"kind": "arc"|"recycle", "stream": ...}``.
    """
    src = unit(flowsheet, source)
    dst = unit(flowsheet, target)
    if outlet not in src.outlet_names:
        raise EditError(f"{source!r} has no outlet {outlet!r} "
                        f"(has: {', '.join(src.outlet_names)})")
    if inlet not in dst.inlet_names:
        raise EditError(f"{target!r} has no inlet {inlet!r} "
                        f"(has: {', '.join(dst.inlet_names)})")
    if inlet in flowsheet.feeds:
        raise EditError(f"{target!r}'s inlet {inlet!r} is a feed; remove the "
                        "feed first, or wire into a different port.")
    made = producers(flowsheet)
    if inlet in made:
        raise EditError(f"{target!r}'s inlet {inlet!r} already comes from "
                        f"{made[inlet]!r}; disconnect that first.")
    if outlet in flowsheet.recycles:
        raise EditError(f"{outlet!r} is already recycled to "
                        f"{flowsheet.recycles[outlet]!r}.")

    if source == target or reaches(flowsheet, target, source):
        # The wire closes a loop, so it is a tear, and the two ends keep
        # their own names: that is exactly what add_recycle expresses.
        flowsheet.add_recycle(outlet, inlet)
        return {"kind": "recycle", "source": outlet, "dest": inlet}

    rename_stream(flowsheet, inlet, outlet)
    return {"kind": "arc", "stream": outlet}


def disconnect(flowsheet, source: str, outlet: str, target: str, inlet: str) -> dict:
    """Undo :func:`connect`. The freed inlet gets a fresh dangling name."""
    dst = unit(flowsheet, target)
    if flowsheet.recycles.get(outlet) == inlet:
        del flowsheet.recycles[outlet]
        return {"kind": "recycle", "source": outlet, "dest": inlet}

    if inlet not in dst.inlet_names:
        raise EditError(f"{target!r} has no inlet {inlet!r}.")
    if producers(flowsheet).get(inlet) != source:
        raise EditError(f"{target!r}'s inlet {inlet!r} does not come from "
                        f"{source!r}, so there is nothing to disconnect.")
    # Only this unit's port is freed. Another consumer of the same stream
    # is a different arc and stays wired.
    fresh = unique(f"{target}_in", stream_names(flowsheet))
    dst.inlet_names = [fresh if s == inlet else s for s in dst.inlet_names]
    return {"kind": "arc", "stream": fresh}


def default_ports(name: str, ports: dict, taken) -> tuple[list[str], list[str]]:
    """Inlet and outlet names for a unit just dropped on the canvas.

    A variadic or unannotated port count reports ``None``. A *variadic*
    unit gets two inlets, because one is what it means to not be there:
    a ``Mixer`` mixing one stream is a piece of pipe. An unannotated one
    gets a single port, since nothing says it wants more.

    Either way the count is only a starting point --- :func:`add_inlet`
    and :func:`remove_inlet` are how it changes afterwards.
    """
    inlets, outlets = [], []
    seen = set(taken)
    default_inlets = 2 if variadic(ports) else 1
    for i in range(max(1, ports.get("n_inlets") or default_inlets)):
        s = unique(f"{name}_in" if i == 0 else f"{name}_in{i + 1}", seen)
        seen.add(s)
        inlets.append(s)
    for i in range(max(1, ports.get("n_outlets") or 1)):
        s = unique(f"{name}_out" if i == 0 else f"{name}_out{i + 1}", seen)
        seen.add(s)
        outlets.append(s)
    return inlets, outlets
