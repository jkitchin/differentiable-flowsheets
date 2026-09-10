"""Read and write flowsheets as JSON.

A flowsheet is currently only expressible as Python: the code that
builds it *is* the model. That is fine while you stay in a script, but
it means a flowsheet cannot be saved, diffed, sent to a service, or
handed to anything that did not import the module that built it.

This module gives it a file format::

    from difflow import serialize

    serialize.save(fs, "plant.json")
    fs2 = serialize.load("plant.json")

Round-tripping relies on the operation registry: a unit is written as
the name it is registered under, and read back by looking that name up
(see :mod:`difflow.catalog`). A unit that is not registered cannot be
written, because nothing would know how to rebuild it.

What can and cannot be written
------------------------------

Parameters are written when they are *data* --- numbers, strings,
arrays, lists, dicts and nested ``Params`` dataclasses. A parameter
holding a **callable** cannot be, and raises rather than being dropped:
a file that silently lost a reactor's rate law would reload into a
different model that still looked plausible.

That restriction bites exactly where :mod:`difflow.kinetics` helps. Of
the package's ``Params`` dataclasses only the reactors hold callables,
and always for the rate law, so a reactor built through
``mass_action_kinetics`` is the declarative case this format is for.

Stability
---------

``FORMAT_VERSION`` is written into every file and checked on read
against ``SUPPORTED_VERSIONS``: difflow writes the current version and
reads every version it still understands, so a file written by an older
release keeps opening.

Version 2 added the optional top-level ``view`` block, which carries
presentation state --- canvas positions, the editor's code context ---
and is read by nothing numeric. A version 1 file simply has none, which
is indistinguishable from a version 2 file whose flowsheet was never
opened in an editor.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import json
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import jax.numpy as jnp

#: bumped when the on-disk layout changes; written into every file
FORMAT_VERSION = 2

#: every version :func:`from_dict` still reads. Version 1 files predate the
#: ``view`` block and are otherwise identical, so reading them costs nothing
#: and refusing them would strand every file written before the editor.
SUPPORTED_VERSIONS = (1, 2)

#: keys used to tag values that JSON has no native type for
ARRAY_TAG = "$array"
DATACLASS_TAG = "$dataclass"
NAMEDTUPLE_TAG = "$namedtuple"
THERMO_TAG = "$thermo"
CALLABLE_TAG = "$callable"
#: A value that lives in the file's *code context* rather than in the
#: file. ``{"$ref": "thermo_ideal"}`` means "whatever the name
#: ``thermo_ideal`` is bound to", and resolving it needs that namespace
#: to be supplied --- see :func:`ref_namespace`.
REF_TAG = "$ref"


#: The namespace ``$ref`` tags resolve against, for the duration of one
#: read or write. A ContextVar rather than an argument threaded through
#: the recursion: every ``_encode_value`` / ``_decode_value`` call site
#: would otherwise have to carry it, and the GUI server is threaded, so
#: a module global would be wrong.
_REFS: ContextVar[dict] = ContextVar("difflow_serialize_refs", default={})


@contextlib.contextmanager
def ref_namespace(namespace: dict | None):
    """Make ``namespace`` the bindings that ``$ref`` tags resolve against.

    Some things a flowsheet needs are objects rather than data --- a
    ``thermo`` of a kind this format cannot write, a hand-written rate
    law. The escape hatch is to build them in Python that travels with
    the flowsheet (``view["code_context"]``, which the editor evaluates)
    and to refer to them by name::

        with serialize.ref_namespace({"thermo": my_thermo}):
            data = serialize.to_dict(fs)      # writes {"$ref": "thermo"}
            fs2 = serialize.from_dict(data)   # reads it back

    Both directions are the same namespace, and outside one nothing
    changes: a flowsheet with no such objects writes exactly the bytes
    it always did.
    """
    token = _REFS.set({
        name: obj for name, obj in (namespace or {}).items()
        if not name.startswith("_")
    })
    try:
        yield
    finally:
        _REFS.reset(token)


def _with_refs(fn):
    """Give ``fn`` a ``refs=`` keyword that sets :func:`ref_namespace`."""
    @functools.wraps(fn)
    def wrapper(*args, refs: dict | None = None, **kwargs):
        with ref_namespace(refs):
            return fn(*args, **kwargs)
    return wrapper


def _ref_for(value: Any) -> dict | None:
    """A ``$ref`` tag for ``value``, if it is bound in the namespace.

    Identity, not equality: the point is to write *this object*, and two
    equal dicts are not interchangeable when one of them is what a unit
    was constructed with.
    """
    for name, obj in _REFS.get().items():
        if obj is value:
            return {REF_TAG: name}
    return None


def _resolve_ref(name: str) -> Any:
    """The object a ``$ref`` names, or an error that says where to get it."""
    namespace = _REFS.get()
    if name not in namespace:
        raise SerializationError(
            f"this flowsheet refers to {name!r}, which is defined in its code "
            "context rather than in the file. Supply it with "
            "refs={'" + name + "': ...}, or open the flowsheet in "
            "difflow.gui, which evaluates the code context for you."
        )
    return namespace[name]


class SerializationError(ValueError):
    """A flowsheet or parameter cannot be written or read.

    Raised for unregistered operations, parameters holding code rather
    than data, and files whose format version is not understood. The
    message names the unit and field responsible.
    """


# ---------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------


def _encode_value(value: Any, where: str) -> Any:
    """Convert one parameter value to something JSON can hold."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    # Checked before every other kind, and after the primitives: a name
    # in the code context is the *identity* of the object a unit holds,
    # so writing it out as data would silently make a copy. Primitives
    # are excluded because small integers and short strings are interned
    # -- `x = 1` in the context would otherwise capture every `1`.
    ref = _ref_for(value)
    if ref is not None:
        return ref
    if callable(value):
        spec = getattr(value, "__difflow_spec__", None)
        if spec is not None:
            # A callable built from data. The closure cannot be written,
            # but the specification that produced it can, and rebuilding
            # from that gives back the identical function.
            return {
                CALLABLE_TAG: {
                    "factory": spec["factory"],
                    "attr": spec.get("attr"),
                    "kwargs": _encode_value(spec["kwargs"], f"{where} spec"),
                }
            }
        raise SerializationError(
            f"{where} holds a callable ({getattr(value, '__name__', type(value).__name__)!r}), "
            "which cannot be written to a file. Build it from data instead "
            "-- see difflow.kinetics.mass_action_kinetics for rate laws -- "
            "or drop the field before saving."
        )
    if hasattr(value, "_fields"):
        # a NamedTuple -- SpeciesData and friends. This must precede the
        # tuple branch below, which would erase the class.
        return {
            NAMEDTUPLE_TAG: type(value).__name__,
            "fields": {
                name: _encode_value(getattr(value, name), f"{where}.{name}")
                for name in value._fields
            },
        }
    if hasattr(value, "shape") or hasattr(value, "tolist"):
        arr = jnp.asarray(value)
        return {ARRAY_TAG: arr.tolist()}
    if dataclasses.is_dataclass(value):
        return {
            DATACLASS_TAG: type(value).__name__,
            "fields": {
                f.name: _encode_value(getattr(value, f.name), f"{where}.{f.name}")
                for f in dataclasses.fields(value)
            },
        }
    if isinstance(value, (list, tuple)):
        return [_encode_value(v, f"{where}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, dict):
        return {str(k): _encode_value(v, f"{where}[{k!r}]") for k, v in value.items()}
    raise SerializationError(
        f"{where} holds {type(value).__name__}, which has no JSON form. "
        "Only numbers, strings, arrays, lists, dicts and nested Params "
        "dataclasses can be written."
    )


def _decode_value(value: Any) -> Any:
    """Invert :func:`_encode_value`."""
    if isinstance(value, dict):
        if REF_TAG in value:
            return _resolve_ref(value[REF_TAG])
        if ARRAY_TAG in value:
            return jnp.asarray(value[ARRAY_TAG], dtype=jnp.float64)
        if DATACLASS_TAG in value:
            cls = _lookup_dataclass(value[DATACLASS_TAG])
            fields = {k: _decode_value(v) for k, v in value["fields"].items()}
            return cls(**fields)
        if NAMEDTUPLE_TAG in value:
            cls = _lookup_type(value[NAMEDTUPLE_TAG])
            fields = {k: _decode_value(v) for k, v in value["fields"].items()}
            return cls(**fields)
        if THERMO_TAG in value:
            return _decode_thermo(value)
        if CALLABLE_TAG in value:
            return _decode_callable(value[CALLABLE_TAG])
        return {k: _decode_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_value(v) for v in value]
    return value


#: packages searched when rebuilding a type by name
_PACKAGES = ("difflow", "difflow_bio", "difflow_ree", "difflow_cc",
             "difflow_gas", "difflow_power")


def _lookup_type(name: str, predicate=None) -> type:
    """Find an exported type by name, across difflow and its plugins."""
    import importlib

    for package in _PACKAGES:
        try:
            module = importlib.import_module(package)
        except ImportError:
            continue
        found = getattr(module, name, None)
        if found is not None and (predicate is None or predicate(found)):
            return found
    raise SerializationError(
        f"no type named {name!r} is exported by difflow or its plugins, so a "
        "value written with it cannot be rebuilt. If it comes from a plugin, "
        "install the plugin."
    )


def _lookup_dataclass(name: str) -> type:
    """Find a ``Params`` dataclass by name."""
    return _lookup_type(name, dataclasses.is_dataclass)


# ---------------------------------------------------------------------
# Thermodynamics
# ---------------------------------------------------------------------

#: How to take a thermo object apart and put it back together. Roughly
#: half the core units need one in their constructor, and it is an
#: object rather than data, so each supported kind needs an explicit
#: handler. Anything else must be supplied on load via ``extras=``.
def _encode_thermo(obj: Any, where: str) -> dict:
    """Write a thermo object as data, if its kind is known."""
    kind = type(obj).__name__
    if kind == "IdealThermo":
        return {
            THERMO_TAG: kind,
            "species": {
                name: _encode_value(data, f"{where}[{name!r}]")
                for name, data in obj.species.items()
            },
        }
    raise SerializationError(
        f"{where} holds a {kind}, which this format cannot write. Only "
        "IdealThermo is supported so far. Save the flowsheet without it, "
        "and pass it back on load with "
        "load(path, extras={'<unit>': {'thermo': my_thermo}})."
    )


def _decode_callable(spec: dict):
    """Rebuild a callable from the specification that produced it.

    The factory is looked up by name among difflow's exports, called
    with the stored arguments, and the named attribute taken off the
    result --- so ``mass_action_kinetics(...).rate_fn`` comes back as
    the function it was.
    """
    factory = _lookup_type(spec["factory"], callable)
    built = factory(**{k: _decode_value(v) for k, v in spec["kwargs"].items()})
    attr = spec.get("attr")
    return getattr(built, attr) if attr else built


def _decode_thermo(data: dict):
    """Rebuild a thermo object written by :func:`_encode_thermo`."""
    kind = data[THERMO_TAG]
    cls = _lookup_type(kind)
    if kind == "IdealThermo":
        return cls({k: _decode_value(v) for k, v in data["species"].items()})
    raise SerializationError(f"cannot rebuild thermo of kind {kind!r}")


def _encode_stream(stream: dict, name: str) -> dict:
    """A stream is a dict of arrays; write it as plain numbers."""
    out = {}
    for key, value in stream.items():
        if isinstance(value, str):          # e.g. "phase"
            out[key] = value
        else:
            out[key] = float(jnp.asarray(value))
    return out


def _decode_stream(data: dict) -> dict:
    return {
        k: (v if isinstance(v, str) else jnp.asarray(v, dtype=jnp.float64))
        for k, v in data.items()
    }


# ---------------------------------------------------------------------
# Flowsheets
# ---------------------------------------------------------------------


def _takes_params_first(sig) -> bool:
    """Whether a constructor's leading argument is its ``Params`` object.

    Most units are built as ``Unit(Params(...), thermo)``, but a
    substantial minority --- ``Mixer``, ``Splitter``, ``GasPipe``,
    ``Compressor`` and the rest of the gas plugin --- take a plain
    argument there instead. Assuming the first argument is always the
    ``Params`` silently drops a *required* one, and the unit then fails
    to rebuild.
    """
    import dataclasses

    for name, param in sig.parameters.items():
        if name == "self":
            continue
        annotation = param.annotation
        return (
            name == "params"
            or dataclasses.is_dataclass(annotation)
            or (isinstance(annotation, str) and annotation.endswith("Params"))
        )
    return False


def constructor_extras(cls: type) -> list[str]:
    """Constructor arguments a class requires besides its ``Params``.

    About half the core units need a ``thermo`` or ``eos`` here, which
    is an object rather than data. They are read off the instance by
    attribute of the same name, and either written (when the kind is
    supported) or supplied on load via ``extras=``.

    Units that take no ``Params`` at all --- ``Mixer(species_order)``,
    ``GasPipe(beta)`` --- report every required argument, since all of
    them have to be written for the unit to be rebuilt.

    Only *required* arguments are reported, so an optional constructor
    object left at its default --- ``Mixer``'s ``thermo``, ``Flash``'s
    ``eos`` --- is not carried by either format. Pass it back on load
    with ``extras=`` when it matters.
    """
    import inspect

    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return []
    required = [
        name for name, p in sig.parameters.items()
        if name != "self"
        and p.default is inspect.Parameter.empty
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    if required and _takes_params_first(sig):
        return required[1:]
    return required


def _encode_extras(operation: Any, unit_name: str) -> dict:
    """Write the non-Params constructor arguments of one operation."""
    out = {}
    for arg in constructor_extras(type(operation)):
        value = getattr(operation, arg, None)
        if value is None:
            continue
        where = f"unit {unit_name!r} constructor argument {arg!r}"
        ref = _ref_for(value)
        if ref is not None:
            out[arg] = ref
        elif type(value).__name__.endswith(("Thermo", "thermo")):
            out[arg] = _encode_thermo(value, where)
        else:
            out[arg] = _encode_value(value, where)
    return out


def _registry_name(cls: type, registry=None) -> str:
    """The name an operation class is registered under."""
    from difflow.catalog import _default_registry

    registry = registry or _default_registry()
    for name, info in registry.list_operations().items():
        if info.cls is cls:
            return name
    raise SerializationError(
        f"{cls.__name__} is not in the operation registry, so it cannot be "
        "written: nothing would know how to rebuild it on read. Register it "
        "with difflow.plugins.registry.register()."
    )


@_with_refs
def to_dict(flowsheet, registry=None) -> dict:
    """Represent a flowsheet as plain, JSON-ready data.

    Args:
        flowsheet: the :class:`~difflow.flowsheet.Flowsheet` to write.
        registry: operation registry for the name lookup; defaults to
            the global one.
        refs: bindings that objects may be written as ``$ref`` tags
            against, keyword-only --- see :func:`ref_namespace`.

    Returns:
        A dictionary with ``format_version``, ``species_order``,
        ``defaults``, ``feeds``, ``units``, ``recycles``, and ``view``
        if the flowsheet carries any presentation state.

    Raises:
        SerializationError: if an operation is unregistered or a
            parameter holds a callable.
    """
    from difflow import __version__

    units = []
    for unit in flowsheet.units:
        operation = unit.operation
        name = _registry_name(type(operation), registry)
        params = getattr(operation, "params", None)
        encoded = {}
        if params is not None and dataclasses.is_dataclass(params):
            for f in dataclasses.fields(params):
                encoded[f.name] = _encode_value(
                    getattr(params, f.name), f"unit {unit.name!r} field {f.name!r}"
                )
        units.append({
            "name": unit.name,
            "operation": name,
            "params": encoded,
            "constructor": _encode_extras(operation, unit.name),
            "extra_params": _encode_value(
                unit.params, f"unit {unit.name!r} extra params"
            ),
            "inlets": list(unit.inlet_names),
            "outlets": list(unit.outlet_names),
        })

    return {
        "format_version": FORMAT_VERSION,
        "difflow_version": __version__,
        "species_order": list(flowsheet.species_order),
        "defaults": {
            "flow": float(flowsheet.default_flow),
            "T": float(flowsheet.default_T),
            "P": float(flowsheet.default_P),
        },
        "feeds": {
            name: _encode_stream(stream, name)
            for name, stream in flowsheet.feeds.items()
        },
        "units": units,
        "recycles": dict(flowsheet.recycles),
        # Omitted rather than written empty, so a flowsheet that has never
        # been opened in an editor produces the same bytes it always did.
        **({"view": _encode_value(flowsheet.view, "view")}
           if getattr(flowsheet, "view", None) else {}),
    }


@_with_refs
def from_dict(data: dict, registry=None, extras: dict | None = None):
    """Rebuild a flowsheet from :func:`to_dict` output.

    Args:
        data: the dictionary to read.
        registry: operation registry for the name lookup.
        extras: constructor objects per unit, ``{unit: {arg: obj}}``.
        refs: bindings that ``$ref`` tags resolve against, keyword-only
            --- see :func:`ref_namespace`.

    Returns:
        A :class:`~difflow.flowsheet.Flowsheet`.

    Raises:
        SerializationError: on an unsupported format version or an
            operation name that is not registered.
    """
    from difflow.catalog import _default_registry
    from difflow.flowsheet import Flowsheet, Unit

    version = data.get("format_version")
    if version not in SUPPORTED_VERSIONS:
        raise SerializationError(
            f"format version {version!r} is not supported; this difflow "
            f"reads {', '.join(str(v) for v in SUPPORTED_VERSIONS)} and "
            f"writes {FORMAT_VERSION}."
        )
    registry = registry or _default_registry()
    operations = registry.list_operations()

    defaults = data.get("defaults", {})
    flowsheet = Flowsheet(
        species_order=list(data["species_order"]),
        default_flow=defaults.get("flow", 0.01),
        default_T=defaults.get("T", 300.0),
        default_P=defaults.get("P", 101325.0),
    )

    for name, stream in data.get("feeds", {}).items():
        flowsheet.add_feed(name, _decode_stream(stream))

    for spec in data.get("units", []):
        op_name = spec["operation"]
        info = operations.get(op_name)
        if info is None:
            raise SerializationError(
                f"unit {spec['name']!r} needs operation {op_name!r}, which is "
                "not registered. If it comes from a plugin, install the "
                "plugin; otherwise register the class first."
            )
        operation = _build_operation(
            info.cls, spec.get("params", {}), spec["name"],
            stored=spec.get("constructor", {}),
            override=(extras or {}).get(spec["name"], {}),
        )
        flowsheet.add_unit(Unit(
            name=spec["name"],
            operation=operation,
            inlet_names=list(spec["inlets"]),
            outlet_names=list(spec["outlets"]),
            params=_decode_value(spec.get("extra_params", {})) or {},
        ))

    for source, dest in data.get("recycles", {}).items():
        flowsheet.add_recycle(source, dest)
    flowsheet.view = _decode_value(data.get("view", {})) or {}
    return flowsheet


def _build_operation(cls: type, encoded_params: dict, unit_name: str,
                     stored: dict | None = None, override: dict | None = None):
    """Instantiate an operation from its encoded parameters and extras."""
    import inspect

    from difflow.catalog import _params_class

    extras = {k: _decode_value(v) for k, v in (stored or {}).items()}
    extras.update(override or {})

    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        # Same guard `constructor_extras` keeps: a C-implemented `__init__`
        # has no signature to read, and the Params path below is the older
        # and safer answer when nothing is known.
        sig = None

    if sig is not None and not _takes_params_first(sig):
        # This unit keeps its numbers as plain constructor arguments rather
        # than in a Params object it is handed --- ``Compressor(ratio)``,
        # ``FlowSplit(w)``, ``GasPipe(beta)``, most of the gas plugin. They
        # build the Params themselves, so there is nothing to hand over;
        # the values go back in the way they came out.
        #
        # ``to_dict`` writes them under "params" regardless, because it
        # reads whatever the instance calls ``.params``. Wrapping those in
        # a fresh Params and passing it positionally puts a dataclass where
        # a float belongs, and demanding them under "extras" instead
        # rejects a file that does carry the value --- one field away.
        # Neither is what the file means.
        kwargs = {k: _decode_value(v) for k, v in encoded_params.items()}
        # extras last: a caller's ``extras=`` outranks the file, and an
        # object that never went to JSON at all --- a thermo, a species
        # order --- only ever arrives that way.
        kwargs.update(extras)
        missing = [a for a in constructor_extras(cls) if a not in kwargs]
        if missing:
            raise SerializationError(
                f"unit {unit_name!r}: {cls.__name__} requires "
                f"{', '.join(missing)}, which the file does not carry. Supply "
                f"it on load, e.g. extras={{{unit_name!r}: "
                f"{{{missing[0]!r}: ...}}}}."
            )
        try:
            return cls(**kwargs)
        except TypeError as exc:
            raise SerializationError(
                f"unit {unit_name!r}: {cls.__name__} rejected the stored "
                f"parameters ({exc}). The file may have been written by a "
                "different version of difflow."
            ) from exc

    missing = [a for a in constructor_extras(cls) if a not in extras]
    if missing:
        raise SerializationError(
            f"unit {unit_name!r}: {cls.__name__} requires "
            f"{', '.join(missing)}, which the file does not carry. Supply it "
            f"on load, e.g. extras={{{unit_name!r}: {{{missing[0]!r}: ...}}}}."
        )

    params_cls = _params_class(cls)

    if not encoded_params:
        try:
            # units that take no Params at all: Mixer, GasPipe
            return cls(**extras)
        except TypeError as exc:
            # A Params class whose fields all carry defaults is still
            # constructible from nothing, and "no parameters written"
            # means exactly that -- which is the state a unit is in the
            # moment difflow.gui adds it to a flowsheet.
            if params_cls is not None:
                try:
                    return cls(params_cls(), **extras)
                except TypeError:
                    pass
            raise SerializationError(
                f"unit {unit_name!r}: {cls.__name__} needs parameters, but "
                f"none were written ({exc})."
            ) from exc

    if params_cls is None:
        raise SerializationError(
            f"unit {unit_name!r}: cannot find the Params class for "
            f"{cls.__name__}, so its parameters cannot be rebuilt."
        )
    kwargs = {k: _decode_value(v) for k, v in encoded_params.items()}
    try:
        return cls(params_cls(**kwargs), **extras)
    except TypeError as exc:
        raise SerializationError(
            f"unit {unit_name!r}: {params_cls.__name__} rejected the stored "
            f"parameters ({exc}). The file may have been written by a "
            "different version of difflow."
        ) from exc


# ---------------------------------------------------------------------
# Text and files
# ---------------------------------------------------------------------


def to_json(flowsheet, indent: int = 2, registry=None, *,
            refs: dict | None = None) -> str:
    """Serialize a flowsheet to a JSON string."""
    return json.dumps(to_dict(flowsheet, registry, refs=refs), indent=indent)


def from_json(text: str, registry=None, extras: dict | None = None, *,
              refs: dict | None = None):
    """Rebuild a flowsheet from a JSON string."""
    return from_dict(json.loads(text), registry, extras, refs=refs)


def save(flowsheet, path: str | Path, indent: int = 2, registry=None, *,
         refs: dict | None = None) -> Path:
    """Write a flowsheet to a file.

    Returns:
        The path written.
    """
    path = Path(path)
    path.write_text(to_json(flowsheet, indent=indent, registry=registry,
                            refs=refs))
    return path


def load(path: str | Path, registry=None, extras: dict | None = None, *,
         refs: dict | None = None):
    """Read a flowsheet from a file written by :func:`save`."""
    return from_json(Path(path).read_text(), registry, extras, refs=refs)
