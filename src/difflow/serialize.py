"""Read and write difflow models as JSON.

A flowsheet is otherwise only expressible as Python: the code that
builds it *is* the model. That is fine while you stay in a script, but
it means a flowsheet cannot be saved, diffed, sent to a service, or
handed to anything that did not import the module that built it.

This module gives it a file format::

    from difflow import serialize

    serialize.save(fs, "plant.json")
    fs2 = serialize.load("plant.json")

``save``/``load`` take a steady-state ``Flowsheet`` (and its subclasses),
a ``DynamicFlowsheet``, or any other model this format can hold --- a
``PowerNetwork``, a ``RadialFeederFlowsheet``. ``save_object`` /
``load_object`` do the same for one object, and ``to_value`` /
``from_value`` give the JSON-ready data.

Round-tripping relies on the operation registry: a unit is written as
the name it is registered under, and read back by looking that name up
(see :mod:`difflow.catalog`). A unit class that is not registered is
written by its import path instead.

What can and cannot be written
------------------------------

Values are written when they are *data*: numbers, strings, arrays,
lists, dicts, dataclasses and NamedTuples; thermo and equations of
state, as the arguments that build them; and any object whose class
keeps its constructor arguments under their own names, or records its
constructor call (:mod:`difflow.init_args`).

Functions are written as what makes them: a rate law built by
``mass_action_kinetics`` as that call, a module-level function by its
import path. A lambda or a closure has neither, and raises rather than
being dropped: a file that silently lost a reactor's rate law would
reload into a different model that still looked plausible. Such an
object travels by name with ``refs=``.

A function or class from outside difflow is imported on load only with
``trust=True``, because importing a module runs its code.

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

Version 3 added values an older reader would misread rather than refuse:
an equation of state (``$eos``), a ``CubicThermo`` or ``ActivityThermo``,
a difflow function by name, and a flowsheet subclass. Read by a version 2
reader, an ``$eos`` is just a dict, and the unit would be built with one.
So a file is written at the *lowest* version that can hold it
(:func:`version_for`): a flowsheet using none of these still says 2 and
still opens in an older release; one that does says 3, which an older
release refuses by name.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import json
import math
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import jax.numpy as jnp

#: the newest layout this difflow writes; a file says the lowest version
#: that can hold it (:func:`version_for`), which is at most this
FORMAT_VERSION = 3

#: every version :func:`from_dict` still reads. Version 1 files predate the
#: ``view`` block and are otherwise identical, so reading them costs nothing
#: and refusing them would strand every file written before the editor.
SUPPORTED_VERSIONS = (1, 2, 3)

#: keys used to tag values that JSON has no native type for
ARRAY_TAG = "$array"
DATACLASS_TAG = "$dataclass"
NAMEDTUPLE_TAG = "$namedtuple"
THERMO_TAG = "$thermo"
#: a cubic equation of state, written as what builds it: the critical
#: properties of each species and the binary interaction parameters
EOS_TAG = "$eos"
CALLABLE_TAG = "$callable"
#: JSON has no literal for the non-finite floats, and Python's ``json``
#: writes them as the bare tokens ``Infinity`` / ``NaN``, which are not
#: JSON: the browser's ``JSON.parse``, ``jq`` and every strict reader
#: reject the file. That is the *common* case, not an exotic one ---
#: :func:`~difflow.kinetics.mass_action_kinetics` puts ``inf`` in
#: ``K_eq`` for every irreversible reaction --- so they are tagged.
#: Files written before this encoding hold the bare tokens, which
#: Python's reader still accepts, so they keep loading unchanged.
NONFINITE_TAG = "$float"

#: the tagged spellings, and what they decode back to
_NONFINITE = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}
#: A value that lives in the file's *code context* rather than in the
#: file. ``{"$ref": "thermo_ideal"}`` means "whatever the name
#: ``thermo_ideal`` is bound to", and resolving it needs that namespace
#: to be supplied --- see :func:`ref_namespace`.
REF_TAG = "$ref"
#: an object rebuilt by calling its class with recorded arguments
#: (:mod:`difflow.init_args`) --- ``RadialFeederFlowsheet``, a dynamic unit
INSTANCE_TAG = "$instance"


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


#: Whether this read may import modules outside difflow. A file can name
#: a function or class by its import path (``"mylab.kinetics:rate"``);
#: importing a module runs its top-level code, so a file from someone
#: else must not be able to do that unasked. difflow's own modules are
#: always allowed --- they are already imported.
_TRUST: ContextVar[bool] = ContextVar("difflow_serialize_trust", default=False)


def _with_refs(fn):
    """Give ``fn`` the ``refs=`` and ``trust=`` keywords.

    ``refs`` sets :func:`ref_namespace`; ``trust`` lets a read import the
    modules outside difflow that the file names (see :data:`_TRUST`).
    """
    @functools.wraps(fn)
    def wrapper(*args, refs: dict | None = None, trust: bool = False, **kwargs):
        token = _TRUST.set(bool(trust))
        try:
            with ref_namespace(refs):
                return fn(*args, **kwargs)
        finally:
            _TRUST.reset(token)
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


class SerializationWarning(UserWarning):
    """Something a unit was built with could not be written to the file."""


# ---------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------


def _is_code(value: Any) -> bool:
    """A function or class, as opposed to an object that happens to be callable.

    ``EquilibriumCurve`` has a ``__call__`` and is data all the same; it
    is written from its constructor arguments, not refused as code.
    """
    import types

    return (isinstance(value, (types.FunctionType, types.BuiltinFunctionType,
                               types.MethodType, functools.partial, type))
            or hasattr(value, "__difflow_spec__"))


#: set while :func:`_instance_from_attributes` re-encodes what it rebuilt,
#: so the check does not check itself
_CHECKING: ContextVar[bool] = ContextVar("difflow_serialize_checking",
                                         default=False)


def _instance_from_attributes(value: Any, where: str) -> dict | None:
    """``$instance`` for an object rebuilt from same-named attributes, or None.

    The last resort for an object of no known kind: if every constructor
    argument is an attribute of the same name (``EquilibriumCurve(kind,
    data, efficiency)``), the class can be called again with them. That
    is a convention, not a guarantee --- an attribute may hold a value
    derived from the argument --- so the object is rebuilt here and
    re-encoded, and written only if that gives back the same thing. A
    class whose attributes do not say how it was built is refused rather
    than written wrong.
    """
    try:
        encoded = {INSTANCE_TAG: _encode_instance(value, where)}
    except SerializationError:
        return None
    if _CHECKING.get():
        return encoded
    checking, trusting = _CHECKING.set(True), _TRUST.set(True)
    try:
        # trust: this is the writer's own object, its module already loaded
        rebuilt = _decode_value(encoded)
        same = _encode_value(rebuilt, where) == encoded
    except Exception:
        same = False
    finally:
        _CHECKING.reset(checking)
        _TRUST.reset(trusting)
    return encoded if same else None


def _encode_nonfinite(x: float) -> Any:
    """Tag a non-finite float; pass every other float through."""
    if x != x:
        return {NONFINITE_TAG: "nan"}
    if x == math.inf:
        return {NONFINITE_TAG: "inf"}
    if x == -math.inf:
        return {NONFINITE_TAG: "-inf"}
    return x


def _encode_array_payload(values: Any) -> Any:
    """Tag the non-finite entries of a nested ``tolist()`` result."""
    if isinstance(values, list):
        return [_encode_array_payload(v) for v in values]
    if isinstance(values, float):
        return _encode_nonfinite(values)
    return values


def _decode_array_payload(values: Any) -> Any:
    """Invert :func:`_encode_array_payload`."""
    if isinstance(values, list):
        return [_decode_array_payload(v) for v in values]
    if isinstance(values, dict) and NONFINITE_TAG in values:
        return _NONFINITE[values[NONFINITE_TAG]]
    return values


def _encode_value(value: Any, where: str) -> Any:
    """Convert one parameter value to something JSON can hold."""
    if isinstance(value, float):
        return _encode_nonfinite(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    # Checked before every other kind, and after the primitives: a name
    # in the code context is the *identity* of the object a unit holds,
    # so writing it out as data would silently make a copy. Primitives
    # are excluded because small integers and short strings are interned
    # -- `x = 1` in the context would otherwise capture every `1`.
    ref = _ref_for(value)
    if ref is not None:
        return ref
    if _is_eos(value):
        return _encode_eos(value, where)
    if type(value) in _thermo_classes().values():
        return _encode_thermo(value, where)
    if _is_code(value):
        spec = getattr(value, "__difflow_spec__", None)
        function = _library_function(value) or (
            _import_path(value) if isinstance(value, type) else None)
        if function is not None:
            # A module-level function (difflow's `monod_kinetics`, or one
            # of yours): written as where it lives and found there again on
            # load --- yours only with trust=True, since that imports it.
            return {CALLABLE_TAG: {"function": function}}
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
            "which cannot be written to a file: a lambda, a closure or a "
            "function defined in a script or notebook has no import path to "
            "be found again by. Define it at the top level of a module, "
            "build it from data (difflow.kinetics.mass_action_kinetics for "
            "rate laws), or pass it by name with refs={'name': fn} on both "
            "save and load."
        )
    if hasattr(value, "_fields"):
        # a NamedTuple -- SpeciesData and friends. This must precede the
        # tuple branch below, which would erase the class.
        return {
            NAMEDTUPLE_TAG: _type_tag(type(value), where),
            "fields": {
                name: _encode_value(getattr(value, name), f"{where}.{name}")
                for name in value._fields
            },
        }
    if hasattr(value, "shape") or hasattr(value, "tolist"):
        import jax

        if isinstance(value, jax.core.Tracer):
            raise SerializationError(
                f"{where} holds a traced value: the object was built inside "
                "jax.grad, jit or vmap, where its numbers have no value yet. "
                "Save it from ordinary code, outside the transformation."
            )
        arr = jnp.asarray(value)
        return {ARRAY_TAG: _encode_array_payload(arr.tolist())}
    if dataclasses.is_dataclass(value):
        return {
            DATACLASS_TAG: _type_tag(type(value), where),
            "fields": {
                f.name: _encode_value(getattr(value, f.name), f"{where}.{f.name}")
                for f in dataclasses.fields(value)
            },
        }
    if isinstance(value, (list, tuple)):
        return [_encode_value(v, f"{where}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, dict):
        return {str(k): _encode_value(v, f"{where}[{k!r}]") for k, v in value.items()}
    from difflow.init_args import init_args

    if init_args(value) is not None:
        return {INSTANCE_TAG: _encode_instance(value, where)}
    instance = _instance_from_attributes(value, where)
    if instance is not None:
        return instance
    if type(value).__name__.endswith(("Thermo", "thermo")):
        raise SerializationError(
            f"{where} holds a {type(value).__name__}, which this format cannot "
            f"write; it writes {', '.join(_thermo_classes())}, and any thermo "
            "that is a dataclass. Pass it back on load with "
            "load(path, extras={'<unit>': {'thermo': my_thermo}})."
        )
    raise SerializationError(
        f"{where} holds {type(value).__name__}, which has no JSON form. "
        "What can be written: numbers, strings, arrays, lists, dicts, "
        "dataclasses and NamedTuples, and an object of a class that either "
        "keeps every constructor argument under its own name or records its "
        "constructor call (difflow.init_args.record_init_args) --- defined "
        "in an importable module. Otherwise pass it with "
        "refs={'<name>': obj} on both save and load."
    )


def _decode_value(value: Any) -> Any:
    """Invert :func:`_encode_value`."""
    if isinstance(value, dict):
        if REF_TAG in value:
            return _resolve_ref(value[REF_TAG])
        if ARRAY_TAG in value:
            return jnp.asarray(_decode_array_payload(value[ARRAY_TAG]),
                               dtype=jnp.float64)
        if NONFINITE_TAG in value:
            return _NONFINITE[value[NONFINITE_TAG]]
        if DATACLASS_TAG in value:
            cls = _type_from_tag(value[DATACLASS_TAG], dataclasses.is_dataclass)
            fields = {k: _decode_value(v) for k, v in value["fields"].items()}
            return cls(**_restore_tuples(cls, fields))
        if NAMEDTUPLE_TAG in value:
            cls = _type_from_tag(value[NAMEDTUPLE_TAG],
                                 lambda c: hasattr(c, "_fields"))
            fields = {k: _decode_value(v) for k, v in value["fields"].items()}
            return cls(**_restore_tuples(cls, fields))
        if THERMO_TAG in value:
            return _decode_thermo(value)
        if EOS_TAG in value:
            return _decode_eos(value)
        if INSTANCE_TAG in value:
            return _decode_instance(value[INSTANCE_TAG])
        if CALLABLE_TAG in value:
            return _decode_callable(value[CALLABLE_TAG])
        return {k: _decode_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_value(v) for v in value]
    return value


def _restore_tuples(cls: type, fields: dict) -> dict:
    """Turn back into tuples the fields ``cls`` declares as tuples.

    JSON has one sequence type, so a tuple comes back a list. That is
    harmless to arithmetic and not to identity: ``SpeciesData`` holds its
    coefficients as tuples, so a reloaded ``IdealThermo`` compared unequal
    to the one it was written from, and every object keyed on it --- the
    JAX cache included --- treated it as new. The annotation says which
    fields were tuples; nothing in the file has to.
    """
    import typing

    try:
        hints = typing.get_type_hints(cls)
    except Exception:
        hints = getattr(cls, "__annotations__", {})

    def as_tuple(value: Any, text: str) -> Any:
        if not isinstance(value, list):
            return value
        inner = text.split("[", 1)[1] if "[" in text else ""
        nested = inner.lstrip().startswith(("tuple", "Tuple"))
        return tuple(as_tuple(v, inner) if nested else v for v in value)

    out = dict(fields)
    for name, value in fields.items():
        text = str(hints.get(name, ""))
        text = text.replace("typing.", "").strip()
        if text.startswith(("tuple", "Tuple")):
            out[name] = as_tuple(value, text)
    return out


#: packages searched when rebuilding a type by name
_PACKAGES = ("difflow", "difflow_bio", "difflow_ree", "difflow_cc",
             "difflow_gas", "difflow_power", "difflow_refinery")


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



# ---------------------------------------------------------------------
# Thermodynamics
# ---------------------------------------------------------------------

#: How to take a thermo object apart and put it back together. Roughly
#: half the core units need one in their constructor, and it is an
#: object rather than data, so each supported kind needs an explicit
#: handler. Anything else must be supplied on load via ``extras=``.
def _thermo_classes() -> dict[str, type]:
    """The thermo kinds with their own handler, by name.

    A thermo that is a dataclass (``IdealGasThermo``, ``GibbsThermo``)
    needs none: it is written field by field like any other.
    """
    from difflow.thermo import ActivityThermo, CubicThermo, IdealThermo

    return {"IdealThermo": IdealThermo, "CubicThermo": CubicThermo,
            "ActivityThermo": ActivityThermo}


def _encode_thermo(obj: Any, where: str) -> dict:
    """Write a thermo object as the arguments that build it.

    Matched on the exact class, as for an EOS: a subclass rebuilt as its
    base would silently lose whatever it adds.
    """
    kind = type(obj).__name__
    if _thermo_classes().get(kind) is not type(obj):
        return _encode_value(obj, where)            # refused there, by name
    if kind == "IdealThermo":
        return {
            THERMO_TAG: kind,
            "species": {
                name: _encode_value(data, f"{where}[{name!r}]")
                for name, data in obj.species.items()
            },
        }
    # CubicThermo(ideal, eos) and ActivityThermo(ideal, activity_model):
    # an IdealThermo plus the one object that corrects it.
    other = "eos" if kind == "CubicThermo" else "activity_model"
    return {
        THERMO_TAG: kind,
        "ideal": _encode_thermo(obj.ideal, f"{where}.ideal"),
        other: _encode_value(getattr(obj, other), f"{where}.{other}"),
    }


def _eos_classes() -> dict[str, type]:
    """The equations of state this format writes, by name."""
    from difflow.eos import SRK, PengRobinson

    return {"PengRobinson": PengRobinson, "SRK": SRK}


def _is_eos(value: Any) -> bool:
    """Exactly a PengRobinson or SRK: a subclass would come back as its base."""
    return _eos_classes().get(type(value).__name__) is type(value)


def _encode_eos(eos: Any, where: str) -> dict:
    """Write an EOS as the arguments that build it.

    Everything else it holds (``a_c``, ``b``, ``kappa``) is computed from
    these in its constructor, so they are all a reload needs. ``k_ij`` is
    kept in the form it was given: a ``{(i, j): value}`` dict, which JSON
    cannot key by a tuple, is written as ``[i, j, value]`` triples.
    """
    k_ij = eos._kij_input
    if isinstance(k_ij, dict):
        encoded_kij = {"pairs": [
            [str(i), str(j), _encode_value(v, f"{where}.k_ij[{(i, j)!r}]")]
            for (i, j), v in k_ij.items()
        ]}
    else:
        encoded_kij = _encode_value(k_ij, f"{where}.k_ij")
    return {
        EOS_TAG: type(eos).__name__,
        "species": {
            name: _encode_value(props, f"{where}.species[{name!r}]")
            for name, props in eos.species.items()
        },
        "k_ij": encoded_kij,
    }


def _decode_eos(data: dict):
    """Rebuild an EOS written by :func:`_encode_eos`."""
    cls = _eos_classes().get(data[EOS_TAG])
    if cls is None:
        raise SerializationError(
            f"cannot rebuild an equation of state of kind {data[EOS_TAG]!r}; "
            f"this difflow reads {', '.join(_eos_classes())}."
        )
    k_ij = data.get("k_ij")
    if isinstance(k_ij, dict) and "pairs" in k_ij:
        k_ij = {(i, j): _decode_value(v) for i, j, v in k_ij["pairs"]}
    else:
        k_ij = _decode_value(k_ij)
    species = {k: _decode_value(v) for k, v in data["species"].items()}
    return cls(species, k_ij=k_ij)


def _difflow_module(module: str) -> bool:
    """Whether ``module`` belongs to difflow or one of its plugins."""
    return module.split(".")[0] in _PACKAGES


def _import_path(obj: Any) -> str | None:
    """``"module:qualname"`` for a function or class import finds again.

    Only a module-level object that the same path leads back to: a
    lambda, a closure, a bound method or a nested class has no such path.
    Nor has anything defined in ``__main__`` --- a script or a notebook ---
    since on load ``__main__`` is whatever is running then.
    """
    import importlib
    import types

    if not isinstance(obj, (types.FunctionType, type)):
        return None
    module, name = getattr(obj, "__module__", None), obj.__qualname__
    if not module or module == "__main__" or "<" in name or "." in name:
        return None
    try:
        found = getattr(importlib.import_module(module), name, None)
    except ImportError:
        return None
    return f"{module}:{name}" if found is obj else None


def _library_function(fn: Any) -> str | None:
    """:func:`_import_path` for a function, whoever defined it.

    difflow's own (``monod_kinetics``) load freely; anyone else's needs
    ``trust=True`` on load (:func:`_resolve_function`).
    """
    import types

    return _import_path(fn) if isinstance(fn, types.FunctionType) else None


def _resolve_function(path: str):
    """The function or class an import path names, within the trust rules.

    Modules of difflow and its plugins are always imported; any other
    module only on a read with ``trust=True``, because importing it runs
    its code.
    """
    import importlib

    module, _, name = path.partition(":")
    if not module or not name:
        raise SerializationError(f"{path!r} is not a 'module:name' path")
    if not _difflow_module(module) and not _TRUST.get():
        raise SerializationError(
            f"the file names {path!r}, from the module {module!r}, which is "
            "not part of difflow. Loading it would import that module and "
            "run its code, so it is only done when you say so: pass "
            "trust=True if you trust this file. Or supply the object with "
            "refs={'<name>': obj} instead."
        )
    try:
        found = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as exc:
        raise SerializationError(
            f"the file names {path!r}, which cannot be imported here. It may "
            "have been written by another version, or need a module or "
            "plugin that is not installed."
        ) from exc
    if not callable(found):
        raise SerializationError(f"{path!r} is not a function or class")
    return found


def untrusted_modules(data: Any) -> list[str]:
    """Modules outside difflow that reading ``data`` would import.

    What ``trust=True`` would let a read do, found without doing it: the
    module of every function or class the document names by import path,
    minus difflow's own. An editor shows this list before asking whether
    to trust a file; an empty one means the file imports nothing new.
    """
    found: set[str] = set()

    def note(path: Any) -> None:
        if isinstance(path, str) and ":" in path:
            module = path.partition(":")[0]
            if module and not _difflow_module(module):
                found.add(module)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key in (DATACLASS_TAG, NAMEDTUPLE_TAG, "operation",
                        "flowsheet_class", "params_class", "class"):
                note(node.get(key))
            if isinstance(node.get(CALLABLE_TAG), dict):
                note(node[CALLABLE_TAG].get("function"))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(data)
    return sorted(found)


def _type_tag(cls: type, where: str) -> str:
    """How a dataclass or NamedTuple's class is named in the file.

    Its bare name when that finds this very class among difflow's
    exports --- what every file so far has said --- and its import path
    otherwise. A bare name alone rebuilt the wrong class whenever two
    packages export the same one (core and gas both have a
    ``PipeParams``), and could not name a class of your own at all.
    """
    try:
        if _lookup_type(cls.__name__) is cls:
            return cls.__name__
    except SerializationError:
        pass
    path = _import_path(cls)
    if path is None:
        raise SerializationError(
            f"{where} holds a {cls.__name__}, a class defined in "
            f"{cls.__module__!r} that cannot be imported back by name (it is "
            "local to a function, or lives in a script or notebook). Move it "
            "into a module, or pass the object with refs={'<name>': obj}."
        )
    return path


def _type_from_tag(tag: str, predicate) -> type:
    """Invert :func:`_type_tag`."""
    if ":" in tag:
        cls = _resolve_function(tag)
        if not predicate(cls):
            raise SerializationError(f"{tag!r} is not the kind of class expected")
        return cls
    return _lookup_type(tag, predicate)


def _decode_callable(spec: dict):
    """Rebuild a callable from the specification that produced it.

    The factory is looked up by name among difflow's exports, called
    with the stored arguments, and the named attribute taken off the
    result --- so ``mass_action_kinetics(...).rate_fn`` comes back as
    the function it was. A library function (``{"function": path}``)
    is imported from where it lives.
    """
    if "function" in spec:
        return _resolve_function(spec["function"])
    factory = _lookup_type(spec["factory"], callable)
    built = factory(**{k: _decode_value(v) for k, v in spec["kwargs"].items()})
    attr = spec.get("attr")
    return getattr(built, attr) if attr else built


def _decode_thermo(data: dict):
    """Rebuild a thermo object written by :func:`_encode_thermo`."""
    kind = data[THERMO_TAG]
    cls = _thermo_classes().get(kind)
    if cls is None:
        raise SerializationError(
            f"cannot rebuild thermo of kind {kind!r}; this difflow reads "
            f"{', '.join(_thermo_classes())}."
        )
    if kind == "IdealThermo":
        return cls({k: _decode_value(v) for k, v in data["species"].items()})
    other = "eos" if kind == "CubicThermo" else "activity_model"
    return cls(_decode_value(data["ideal"]), _decode_value(data[other]))


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


def _takes_params_first_cls(cls: type) -> bool:
    """:func:`_takes_params_first` for a class, False with no signature."""
    import inspect

    try:
        return _takes_params_first(inspect.signature(cls.__init__))
    except (TypeError, ValueError):
        return False


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

    Only *required* arguments are reported here. An optional one that
    was set away from its default --- ``Mixer``'s ``thermo``, ``CSTR``'s
    ``mode`` --- is found by :func:`optional_extras` and written too.
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


def is_number_annotation(annotation) -> bool:
    """Whether an annotation admits a plain number.

    Read off the text of the annotation rather than by comparing types:
    the interesting ones are unions (``float | jax.Array``) and strings
    (under ``from __future__ import annotations``), and both defeat an
    identity test while reading perfectly well.
    """
    text = str(annotation)
    if any(bad in text for bad in ("Callable", "dict", "list", "tuple")):
        return False
    return "float" in text or "int" in text


def numeric_extras(cls: type) -> list[str]:
    """The :func:`constructor_extras` that are plain numbers.

    ``GasPipe(beta)`` and ``Compressor(ratio)`` keep their numbers as
    constructor arguments rather than in a ``Params`` they are handed.
    A number is data whichever side of that line it sits on, so a form
    can supply it just as it supplies a ``Params`` field; a ``thermo``
    or a ``species_order`` it cannot.
    """
    import inspect

    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return []
    return [a for a in constructor_extras(cls)
            if is_number_annotation(sig.parameters[a].annotation)]


def _is_default(value: Any, default: Any) -> bool:
    """Whether ``value`` is what an optional argument defaults to.

    Identity first, then equality for the plain values a default can be
    (``"liquid"``, ``0.0``, a tuple of names). Anything else --- an
    array, a thermo --- set to a non-default object is not the default.
    """
    if value is default:
        return True
    if isinstance(default, (bool, int, float, str, tuple)):
        try:
            return bool(value == default)
        except Exception:
            return False
    return False


def optional_extras(operation: Any) -> dict:
    """Optional constructor arguments ``operation`` was built away from.

    The counterpart of :func:`constructor_extras` for arguments with a
    default. Left out, ``Mixer(species, thermo)`` reloads as
    ``Mixer(species)``: it then mixes temperatures without an enthalpy
    balance, and the reloaded flowsheet solves to a different answer
    that still looks plausible. An argument still at its default is not
    reported, so a flowsheet that never set one writes the same bytes it
    always did.

    Returns:
        ``{argument: value}``, read off the instance by attribute of the
        same name.
    """
    import inspect

    cls = type(operation)
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return {}
    out = {}
    for name, p in sig.parameters.items():
        if (name == "self" or p.default is inspect.Parameter.empty
                or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)):
            continue
        if not hasattr(operation, name):
            continue
        value = getattr(operation, name)
        if value is None or _is_default(value, p.default):
            continue
        out[name] = value
    return out


def _encode_constructor_value(value: Any, where: str) -> Any:
    """Write one constructor argument: a ref, a thermo, or plain data."""
    return _encode_value(value, where)


def _encode_extras(operation: Any, unit_name: str) -> dict:
    """Write the non-Params constructor arguments of one operation."""
    out = {}
    for arg in constructor_extras(type(operation)):
        value = getattr(operation, arg, None)
        if value is None:
            continue
        where = f"unit {unit_name!r} constructor argument {arg!r}"
        out[arg] = _encode_constructor_value(value, where)
    for arg, value in optional_extras(operation).items():
        if arg in out:
            continue
        where = f"unit {unit_name!r} constructor argument {arg!r}"
        try:
            encoded = _encode_constructor_value(value, where)
            # A dataclass or named tuple only comes back if its type is
            # exported; find out now, not when someone opens the file.
            _decode_value(encoded)
        except SerializationError as exc:
            # Optional arguments were never written before, and refusing
            # the whole file now would break saving a unit built with a
            # thermo of a kind this format cannot write. Say so instead of
            # dropping it silently.
            import warnings

            warnings.warn(
                f"{exc} It is not written, so the unit reloads with "
                f"{arg}'s default unless you pass it back with "
                f"extras={{{unit_name!r}: {{{arg!r}: ...}}}}.",
                SerializationWarning, stacklevel=4,
            )
            continue
        out[arg] = encoded
    return out


def _registry_name(cls: type, registry=None) -> str:
    """The name an operation class is registered under."""
    from difflow.catalog import _default_registry

    registry = registry or _default_registry()
    for name, info in registry.list_operations().items():
        if info.cls is cls:
            return name
    # Not registered, but importable: written as where it lives, which a
    # read finds again (with trust=True when it is not difflow's).
    path = _import_path(cls)
    if path is not None:
        return path
    raise SerializationError(
        f"{cls.__name__} is neither in the operation registry nor importable "
        "by name (it is defined in a script, a notebook or a function), so "
        "nothing would know how to rebuild it on read. Move it into a "
        "module, or register it with difflow.plugins.registry.register()."
    )


@_with_refs
def to_dict(flowsheet, registry=None, results: dict | None = None) -> dict:
    """Represent a flowsheet as plain, JSON-ready data.

    Args:
        flowsheet: the :class:`~difflow.flowsheet.Flowsheet` to write.
        registry: operation registry for the name lookup; defaults to
            the global one.
        results: the streams a solve returned, to keep with the model
            (``to_dict(fs, results=fs.solve())``). They are read back as
            ``saved_results`` and can warm-start the next solve; see
            :func:`warm_start`.
        refs: bindings that objects may be written as ``$ref`` tags
            against, keyword-only --- see :func:`ref_namespace`.

    Returns:
        A dictionary with ``format_version``, ``species_order``,
        ``defaults``, ``feeds``, ``units``, ``recycles``, ``view`` if
        the flowsheet carries any presentation state, and ``results``
        if given.

    Raises:
        SerializationError: if an operation is unregistered or a
            parameter holds a callable.
    """
    from difflow import __version__

    from difflow.incomplete import Incomplete

    kind = _document_kind(flowsheet)
    if kind is not None:
        return kind[0](flowsheet)

    units = []
    for unit in flowsheet.units:
        operation = unit.operation
        if isinstance(operation, Incomplete):
            # On the flowsheet, wired, but without the thing it is
            # waiting for. Written as what it is going to be, so the
            # file keeps the ports and the wiring rather than losing a
            # half-built flowsheet at save time.
            units.append({
                "name": unit.name,
                "operation": operation.operation,
                "params": {},
                "constructor": {},
                "extra_params": {},
                "incomplete": {"needs": list(operation.needs),
                               "hint": operation.hint},
                "inlets": list(unit.inlet_names),
                "outlets": list(unit.outlet_names),
            })
            continue
        name = _registry_name(type(operation), registry)
        params = getattr(operation, "params", None)
        encoded = {}
        if params is not None and dataclasses.is_dataclass(params):
            for f in dataclasses.fields(params):
                encoded[f.name] = _encode_value(
                    getattr(params, f.name), f"unit {unit.name!r} field {f.name!r}"
                )
        from difflow.catalog import _params_class

        # The catalog finds the Params class from the constructor's
        # annotation or its name; say which when that would find another.
        params_tag = (
            {"params_class": _type_tag(type(params),
                                       f"unit {unit.name!r} params")}
            if encoded and _params_class(type(operation)) is not type(params)
            and _takes_params_first_cls(type(operation)) else {}
        )
        units.append({
            "name": unit.name,
            "operation": name,
            **params_tag,
            "params": encoded,
            "constructor": _encode_extras(operation, unit.name),
            "extra_params": _encode_value(
                unit.params, f"unit {unit.name!r} extra params"
            ),
            "inlets": list(unit.inlet_names),
            "outlets": list(unit.outlet_names),
        })

    from difflow.flowsheet import Flowsheet

    data = {
        "format_version": FORMAT_VERSION,
        "difflow_version": __version__,
        **({} if type(flowsheet) is Flowsheet
           else {"flowsheet_class": _flowsheet_class_path(type(flowsheet))}),
        "species_order": list(flowsheet.species_order),
        # T/P are written only when pinned: an absent key reads back as
        # None, which keeps the tear guess inferred (Flowsheet.__init__).
        "defaults": {
            "flow": float(flowsheet.default_flow),
            **({} if getattr(flowsheet, "infer_tear_T", False)
               else {"T": float(flowsheet.default_T)}),
            **({} if getattr(flowsheet, "infer_tear_P", False)
               else {"P": float(flowsheet.default_P)}),
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
        **({"results": _encode_results(flowsheet, results)}
           if results is not None else {}),
    }
    data["format_version"] = version_for(data)
    return data


def _encode_results(flowsheet, streams: dict) -> dict:
    """Solved streams, and what the solve that produced them reported.

    An older reader ignores the block rather than misreading it, so it
    does not raise the format version.
    """
    from difflow.flowsheet import _concrete

    def scalar_or_array(value: Any, where: str) -> Any:
        if isinstance(value, str):
            return value
        if getattr(value, "ndim", 0) == 0:
            number = _concrete(value)
            if number is None:
                raise SerializationError(
                    f"{where} is a traced value (the solve ran under "
                    "jax.grad or jit), which has no number to write. Save "
                    "the results of an ordinary solve."
                )
            return _encode_nonfinite(number)
        return _encode_value(value, where)

    out: dict = {"streams": {
        name: {k: scalar_or_array(v, f"result stream {name!r} {k!r}")
               for k, v in stream.items()}
        for name, stream in streams.items()
    }}
    info = {
        "converged": getattr(flowsheet, "last_solve_converged", None),
        "iterations": getattr(flowsheet, "last_solve_iterations", None),
        "residual": _concrete(getattr(flowsheet, "last_solve_residual", None)),
        "method": getattr(flowsheet, "last_solve_method", None),
        "tear_streams": list(getattr(flowsheet, "last_solve_tear_streams", [])),
    }
    info = {k: v for k, v in info.items() if v is not None}
    if isinstance(info.get("converged"), bool) or info.get("tear_streams"):
        out["solve"] = _encode_value(
            {k: (bool(v) if k == "converged" else v) for k, v in info.items()},
            "solve info",
        )
    return out


def warm_start(flowsheet) -> dict:
    """Saved tear-stream values, ready for ``solve(tear_initial=...)``.

    A flowsheet read with saved results can start its next solve where
    the saved one finished, which on a stiff recycle is the difference
    between a few iterations and many::

        fs = serialize.load("plant.json")
        streams = fs.solve(tear_initial=serialize.warm_start(fs))

    Returns the saved stream for every recycle destination that has one,
    and an empty dict when nothing was saved.
    """
    saved = getattr(flowsheet, "saved_results", None) or {}
    return {dest: saved[dest] for dest in flowsheet.recycles.values()
            if dest in saved}


# ---------------------------------------------------------------------
# Objects rebuilt from their constructor call, and other model kinds
# ---------------------------------------------------------------------


def _constructor_args(obj: Any, where: str) -> dict:
    """The keyword arguments that rebuild ``obj`` as ``type(obj)(**args)``.

    The call its class recorded (:mod:`difflow.init_args`) when there is
    one; otherwise every constructor argument read off the attribute of
    the same name, the convention the steady-state units follow. Refused,
    naming the argument, when neither says what the object was built with.
    """
    import inspect

    from difflow.init_args import init_args

    recorded = init_args(obj)
    if recorded is not None:
        return dict(recorded)
    try:
        sig = inspect.signature(type(obj).__init__)
    except (TypeError, ValueError) as exc:
        raise SerializationError(f"{where}: cannot read how it was built") from exc
    args, missing = {}, []
    for name, p in sig.parameters.items():
        if name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if hasattr(obj, name):
            value = getattr(obj, name)
            if p.default is p.empty or not _is_default(value, p.default):
                args[name] = value
        elif p.default is p.empty:
            missing.append(name)
    if missing:
        raise SerializationError(
            f"{where} is a {type(obj).__name__}, which keeps "
            f"{', '.join(missing)} under another name, so it cannot be "
            "rebuilt from the file. Give it an attribute of the same name, "
            "or record its constructor call with "
            "difflow.init_args.record_init_args."
        )
    return args


def _encode_instance(obj: Any, where: str) -> dict:
    """``{"class": ..., "args": {...}}`` for an object built by a call."""
    return {
        "class": _type_tag(type(obj), where),
        "args": {k: _encode_value(v, f"{where}.{k}")
                 for k, v in _constructor_args(obj, where).items()},
    }


def _decode_instance(data: dict) -> Any:
    """Invert :func:`_encode_instance`."""
    cls = _type_from_tag(data["class"], callable)
    args = {k: _decode_value(v) for k, v in data.get("args", {}).items()}
    try:
        return cls(**args)
    except TypeError as exc:
        raise SerializationError(
            f"{data['class']} rejected the stored arguments ({exc}). The file "
            "may have been written by a different version of difflow."
        ) from exc


#: ``kind`` of a dynamic flowsheet document
DYNAMIC_KIND = "dynamic"


def _dynamic_to_dict(flowsheet) -> dict:
    """Write a :class:`~difflow.dynamic.DynamicFlowsheet`.

    Each unit is its class and constructor call; feeds are streams or,
    for a time-dependent feed, a function written by import path (or a
    ``$ref``). The wiring is kept port by port, as ``add_unit`` took it.
    """
    from difflow import __version__

    feeds = {}
    for name, feed in flowsheet._feeds.items():
        feeds[name] = (_encode_value(feed, f"feed {name!r}") if callable(feed)
                       else {"stream": _encode_stream(feed, name)})
    units = []
    for entry in flowsheet.units:
        units.append({
            "name": entry.name,
            "unit": _encode_instance(entry.unit, f"unit {entry.name!r}"),
            "inlets": dict(entry.inlet_ports),
            "outlets": (dict(entry.outlet_ports) if entry.outlet_ports is not None
                        else list(entry.outlet_names)),
        })
    body = {
        "kind": DYNAMIC_KIND,
        "difflow_version": __version__,
        "species_order": list(flowsheet.species_order),
        "feeds": feeds,
        "units": units,
        "connections": [[c.source, c.dest] for c in flowsheet._connections],
    }
    return {"format_version": version_for(body), **body}


def _dynamic_from_dict(data: dict):
    """Invert :func:`_dynamic_to_dict`."""
    from difflow.dynamic import DynamicFlowsheet

    flowsheet = DynamicFlowsheet(species_order=list(data["species_order"]))
    for name, feed in data.get("feeds", {}).items():
        if isinstance(feed, dict) and "stream" in feed:
            flowsheet.add_feed(name, _decode_stream(feed["stream"]))
        else:
            flowsheet.add_feed(name, _decode_value(feed))
    for spec in data.get("units", []):
        outlets = spec.get("outlets")
        flowsheet.add_unit(
            _decode_instance(spec["unit"]),
            inlets=dict(spec.get("inlets", {})),
            **({"outlets": dict(outlets)} if isinstance(outlets, dict)
               else {"outlet_names": list(outlets or [])}),
            name=spec["name"],
        )
    for source, dest in data.get("connections", []):
        flowsheet.connect(source, dest)
    return flowsheet


def _object_to_dict(model) -> dict:
    """Any other model --- a ``PowerNetwork``, a ``RadialFeederFlowsheet``."""
    from difflow import __version__

    body = {"kind": OBJECT_KIND, "difflow_version": __version__,
            "value": _encode_value(model, type(model).__name__)}
    return {"format_version": version_for(body), **body}


def _object_from_dict(data: dict):
    return _decode_value(data["value"])


def _document_kind(model):
    """``(writer, reader)`` for a model that is not a steady-state Flowsheet."""
    from difflow.dynamic import DynamicFlowsheet
    from difflow.flowsheet import Flowsheet

    if isinstance(model, Flowsheet):
        return None
    if isinstance(model, DynamicFlowsheet):
        return _DOCUMENT_KINDS[DYNAMIC_KIND]
    if isinstance(model, (dict, list, tuple, str, int, float)) or model is None:
        raise SerializationError(
            f"save/to_dict take a model, not a {type(model).__name__}; use "
            "dumps_object/save_object for a plain value."
        )
    return _DOCUMENT_KINDS[OBJECT_KIND]


#: what tells an object file from a flowsheet file
OBJECT_KIND = "object"

#: document kinds other than the steady-state flowsheet, by ``kind``
_DOCUMENT_KINDS = {DYNAMIC_KIND: (_dynamic_to_dict, _dynamic_from_dict),
                   OBJECT_KIND: (_object_to_dict, _object_from_dict)}


def _check_version(version: Any) -> None:
    """Refuse a format version this difflow does not read."""
    if version not in SUPPORTED_VERSIONS:
        raise SerializationError(
            f"format version {version!r} is not supported; this difflow "
            f"reads {', '.join(str(v) for v in SUPPORTED_VERSIONS)} and "
            f"writes up to {FORMAT_VERSION}."
        )


def _flowsheet_class_path(cls: type) -> str:
    """``"module:name"`` for a Flowsheet subclass.

    ``GasNetworkFlowsheet`` solves its own way; rebuilt as a plain
    ``Flowsheet`` it would converge differently, or not at all. One that
    import cannot find again is refused rather than quietly downcast, and
    one from outside difflow needs ``trust=True`` to load.
    """
    path = _import_path(cls)
    if path is not None:
        return path
    raise SerializationError(
        f"the flowsheet is a {cls.__name__}, a Flowsheet subclass defined in "
        f"{cls.__module__!r} that cannot be imported back by name (a script, "
        "a notebook or a function). It would reload as a plain Flowsheet, so "
        "it is refused; move the class into a module."
    )


#: what makes a file version 3: tags and keys a version 2 reader would
#: accept and misread
_V3_THERMO = ("CubicThermo", "ActivityThermo")


def version_for(data: Any) -> int:
    """The lowest format version that can hold ``data``.

    A file using nothing newer than version 2 is written as 2, so the
    releases that read 2 keep opening it; see the module docstring.
    """
    def newer(node: Any) -> bool:
        if isinstance(node, dict):
            if any(k in node for k in (EOS_TAG, INSTANCE_TAG, "flowsheet_class",
                                       "params_class")):
                return True
            if node.get("kind") in _DOCUMENT_KINDS:
                return True
            # a class or operation named by import path
            for key in (DATACLASS_TAG, NAMEDTUPLE_TAG, "operation"):
                if ":" in str(node.get(key, "")):
                    return True
            if node.get(THERMO_TAG) in _V3_THERMO:
                return True
            if isinstance(node.get(CALLABLE_TAG), dict) \
                    and "function" in node[CALLABLE_TAG]:
                return True
            return any(newer(v) for v in node.values())
        if isinstance(node, list):
            return any(newer(v) for v in node)
        return False

    return 3 if newer(data) else 2


@_with_refs
def from_dict(data: dict, registry=None, extras: dict | None = None):
    """Rebuild a flowsheet from :func:`to_dict` output.

    Args:
        data: the dictionary to read.
        registry: operation registry for the name lookup.
        extras: constructor objects per unit, ``{unit: {arg: obj}}``.
        refs: bindings that ``$ref`` tags resolve against, keyword-only
            --- see :func:`ref_namespace`.
        trust: allow importing modules outside difflow that the file
            names, keyword-only --- see :func:`load`.

    Returns:
        A :class:`~difflow.flowsheet.Flowsheet`.

    Raises:
        SerializationError: on an unsupported format version or an
            operation name that is not registered.
    """
    from difflow.catalog import _default_registry
    from difflow.flowsheet import Flowsheet, Unit

    _check_version(data.get("format_version"))
    for kind, (_, reader) in _DOCUMENT_KINDS.items():
        if data.get("kind") == kind:
            return reader(data)
    registry = registry or _default_registry()
    operations = registry.list_operations()

    defaults = data.get("defaults", {})
    cls = Flowsheet
    if "flowsheet_class" in data:
        cls = _resolve_function(data["flowsheet_class"])
        if not (isinstance(cls, type) and issubclass(cls, Flowsheet)):
            raise SerializationError(
                f"{data['flowsheet_class']!r} is not a Flowsheet subclass"
            )
    flowsheet = cls(
        species_order=list(data["species_order"]),
        default_flow=defaults.get("flow", 0.01),
        default_T=defaults.get("T"),
        default_P=defaults.get("P"),
    )

    for name, stream in data.get("feeds", {}).items():
        flowsheet.add_feed(name, _decode_stream(stream))

    for spec in data.get("units", []):
        op_name = spec["operation"]
        if spec.get("incomplete"):
            # Read back as unfinished rather than built with guesses:
            # the file recorded that nobody had chosen these yet, and
            # inventing them on the way in would hide that.
            from difflow.incomplete import Incomplete

            flowsheet.add_unit(Unit(
                name=spec["name"],
                operation=Incomplete(
                    op_name,
                    list(spec["incomplete"].get("needs", [])),
                    spec["incomplete"].get("hint", ""),
                ),
                inlet_names=list(spec["inlets"]),
                outlet_names=list(spec["outlets"]),
            ))
            continue
        info = operations.get(op_name)
        if info is None and ":" in op_name:
            op_cls = _resolve_function(op_name)
        elif info is None:
            raise SerializationError(
                f"unit {spec['name']!r} needs operation {op_name!r}, which is "
                "not registered. If it comes from a plugin, install the "
                "plugin; otherwise register the class first."
            )
        else:
            op_cls = info.cls
        operation = _build_operation(
            op_cls, spec.get("params", {}), spec["name"],
            stored=spec.get("constructor", {}),
            override=(extras or {}).get(spec["name"], {}),
            params_class=(_type_from_tag(spec["params_class"],
                                         dataclasses.is_dataclass)
                          if "params_class" in spec else None),
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
    results = data.get("results") or {}
    flowsheet.saved_results = {
        name: _decode_stream({k: _decode_value(v) for k, v in stream.items()})
        for name, stream in results.get("streams", {}).items()
    }
    flowsheet.saved_solve_info = _decode_value(results.get("solve", {})) or {}
    return flowsheet


def _build_operation(cls: type, encoded_params: dict, unit_name: str,
                     stored: dict | None = None, override: dict | None = None,
                     params_class: type | None = None):
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

    params_cls = params_class or _params_class(cls)

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


@_with_refs
def to_value(obj: Any) -> Any:
    """Represent any one object this format can hold as JSON-ready data.

    The same encoding a flowsheet's parameters go through, for an object
    on its own: a ``Params``, a thermo, an equation of state, a
    ``ReactionSet``, an array, or a dict or list of them.

    Args:
        obj: what to write.
        refs: bindings it may be written against, keyword-only --- see
            :func:`ref_namespace`.

    Raises:
        SerializationError: for anything with no data form.
    """
    return _encode_value(obj, type(obj).__name__)


@_with_refs
def from_value(data: Any) -> Any:
    """Invert :func:`to_value`; takes ``refs=`` and ``trust=`` like :func:`load`."""
    return _decode_value(data)


def dumps_object(obj: Any, indent: int = 2, *,
                 refs: dict | None = None) -> str:
    """Write one object as a JSON document with a format version.

    Example:
        >>> text = serialize.dumps_object(CSTRParams(V=2.0, ...))  # doctest: +SKIP
        >>> params = serialize.loads_object(text)                   # doctest: +SKIP
    """
    from difflow import __version__

    body = {"kind": OBJECT_KIND, "difflow_version": __version__,
            "value": to_value(obj, refs=refs)}
    return json.dumps({"format_version": version_for(body), **body},
                      indent=indent)


def loads_object(text: str, *, refs: dict | None = None,
                 trust: bool = False) -> Any:
    """Read an object written by :func:`dumps_object`."""
    data = json.loads(text)
    if data.get("kind") != OBJECT_KIND:
        raise SerializationError(
            "this is not a difflow object document"
            + (" (it looks like a flowsheet: use load/from_json)"
               if "units" in data else "")
        )
    _check_version(data.get("format_version"))
    return from_value(data["value"], refs=refs, trust=trust)


def save_object(obj: Any, path: str | Path, indent: int = 2, *,
                refs: dict | None = None) -> Path:
    """Write one object to a file; see :func:`dumps_object`."""
    path = Path(path)
    path.write_text(dumps_object(obj, indent=indent, refs=refs))
    return path


def load_object(path: str | Path, *, refs: dict | None = None,
                trust: bool = False) -> Any:
    """Read an object from a file written by :func:`save_object`."""
    return loads_object(Path(path).read_text(), refs=refs, trust=trust)


def to_json(flowsheet, indent: int = 2, registry=None, *,
            refs: dict | None = None, results: dict | None = None) -> str:
    """Serialize a flowsheet to a JSON string; see :func:`to_dict`."""
    return json.dumps(to_dict(flowsheet, registry, results, refs=refs),
                      indent=indent)


def from_json(text: str, registry=None, extras: dict | None = None, *,
              refs: dict | None = None, trust: bool = False):
    """Rebuild a flowsheet from a JSON string; see :func:`load`."""
    return from_dict(json.loads(text), registry, extras, refs=refs, trust=trust)


def save(flowsheet, path: str | Path, indent: int = 2, registry=None, *,
         refs: dict | None = None, results: dict | None = None) -> Path:
    """Write a flowsheet to a file.

    Args:
        results: the streams a solve returned, kept with the model; see
            :func:`to_dict` and :func:`warm_start`.

    Returns:
        The path written.
    """
    path = Path(path)
    path.write_text(to_json(flowsheet, indent=indent, registry=registry,
                            refs=refs, results=results))
    return path


def load(path: str | Path, registry=None, extras: dict | None = None, *,
         refs: dict | None = None, trust: bool = False):
    """Read a flowsheet from a file written by :func:`save`.

    Args:
        path: the file.
        registry: operation registry for the name lookup.
        extras: constructor objects per unit, ``{unit: {arg: obj}}``.
        refs: bindings that ``$ref`` tags resolve against.
        trust: let the file import the modules outside difflow it names
            --- your own functions and classes, written by import path.
            Importing a module runs its code, so leave this off for a
            file you did not write.
    """
    return from_json(Path(path).read_text(), registry, extras, refs=refs,
                     trust=trust)
