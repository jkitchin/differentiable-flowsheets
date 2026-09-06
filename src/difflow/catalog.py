"""A machine-readable catalog of unit operations.

The registry knows a name, a class, a category and a description. That
is enough to *list* operations, but not enough to build anything
against them: a form needs to know what parameters a unit takes and
which are required, and a canvas needs to know how many streams go in
and come out before anything is instantiated.

This module derives that from the code itself rather than from a second
hand-maintained table:

* **parameters** come from ``dataclasses.fields`` of the unit's
  ``Params`` class --- name, type, default, whether it is required, and
  whether it is a callable (which is what a declarative front end
  cannot author; see :mod:`difflow.kinetics`).
* **ports** come from the ``__call__`` signature: parameters annotated
  as :data:`~difflow.streams.Stream` are inlets, and the leading
  ``Stream`` entries of the return tuple are outlets.
* **equations**, along with the unit's symbol, assumptions, references,
  numerical method and per-parameter units and symbols, come from the
  metadata contract in :mod:`difflow.report.metadata` --- the class
  attributes the unit operations already carry for the report writer.

Deriving rather than declaring means the catalog cannot drift from the
code, and that an operation whose signature is unannotated is reported
as *unknown* rather than guessed at.

    >>> from difflow.catalog import describe_operation
    >>> spec = describe_operation("Flash")
    >>> spec.ports.n_outlets
    2
    >>> [p.name for p in spec.parameters if p.required]
    ['species_order']

Every schema is JSON-serializable through :meth:`OperationSchema.to_dict`,
which is what a GUI, a CLI or a code generator would consume.
"""

from __future__ import annotations

import dataclasses
import inspect
import typing
from dataclasses import dataclass, field
from typing import Any

from difflow.params_mixin import ParamsMixin

#: module name -> category, for the core unit operations
CORE_CATEGORIES = {
    "cstr": "reactors",
    "pfr": "reactors",
    "fed_batch": "reactors",
    "flash": "separations",
    "distillation": "distillation",
    "heat_exchanger": "heat_transfer",
    "lle": "extraction",
    "eos_units": "pressure_change",
    "gas_turbine": "power",
}

#: classes in ``difflow.units`` that are not themselves operations
NOT_OPERATIONS = frozenset({"UnitBase", "ReactorBase", "DistributionCoeffs"})

#: registry name -> class name, where the two deliberately differ.
#: ``difflow_gas`` also registers a "Compressor", so the EOS-consistent
#: one takes a qualified name rather than silently overwriting it (or
#: being overwritten, since plugins load after the core).
CORE_NAME_OVERRIDES = {"Compressor": "EOSCompressor"}


@dataclass
class PortSpec(ParamsMixin):
    """Stream connectivity of an operation.

    Attributes:
        inlets: names of the stream-typed arguments, in call order.
        n_outlets: number of outlet streams, or ``None`` when the return
            annotation does not say.
        variadic: whether the operation takes any number of inlets
            (``*inlets``), as a mixer does.
    """

    inlets: list[str] = field(default_factory=list)
    n_outlets: int | None = None
    variadic: bool = False

    @property
    def n_inlets(self) -> int | None:
        """Number of inlets, or ``None`` if variadic."""
        return None if self.variadic else len(self.inlets)


@dataclass
class ParameterSpec(ParamsMixin):
    """One field of an operation's ``Params`` class.

    Attributes:
        name: field name.
        type: the annotation, as written.
        default: the default, rendered as a string; ``None`` if required.
        required: whether the field has no default.
        is_callable: whether the field holds a function or an arbitrary
            object rather than data. These are the fields a declarative
            front end cannot fill in.
        units: physical units. Read from the field's own dataclass
            metadata when it declares them, and otherwise from the unit
            class's ``parameter_units`` --- the field is the more
            specific of the two, so it wins.
        symbol: LaTeX symbol for the field, from the unit class's
            ``parameter_symbols``.
        description: help text, when the field declares it.
    """

    name: str
    type: str
    default: str | None = None
    required: bool = False
    is_callable: bool = False
    units: str | None = None
    symbol: str | None = None
    description: str | None = None


@dataclass
class OperationSchema(ParamsMixin):
    """Everything known about one unit operation, as data.

    Attributes:
        name: the name it is registered under.
        class_name: the Python class, which may differ from ``name``.
        module: where the class lives.
        category: registry category, for grouping in a palette.
        description: one-line summary.
        plugin: the package that registered it.
        symbol: short identifier the unit uses in report headings.
        doc: the whole class docstring, cleaned of its indentation.
            ``description`` is its first line; this is the rest, which
            an inspector renders.
        equations: LaTeX governing equations, if the unit declares any.
        assumptions: what the model takes for granted.
        references: where the model comes from.
        numerical_method: how it is solved, in a sentence.
        ports: stream connectivity.
        parameters: the ``Params`` fields.
        params_class: name of the ``Params`` dataclass, if one was found.
        constructor_extras: constructor arguments the class requires
            besides its ``Params`` --- a ``thermo``, an ``eos``. These
            are objects, so a front end cannot supply them.
    """

    name: str
    class_name: str
    module: str
    category: str
    description: str
    plugin: str
    symbol: str = ""
    doc: str = ""
    equations: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    numerical_method: str | None = None
    ports: PortSpec = field(default_factory=PortSpec)
    parameters: list[ParameterSpec] = field(default_factory=list)
    params_class: str | None = None
    constructor_extras: list[str] = field(default_factory=list)

    @property
    def is_declarative(self) -> bool:
        """Whether every parameter is data a form could supply."""
        return not any(p.is_callable for p in self.parameters)

    @property
    def is_buildable(self) -> bool:
        """Whether a form could construct this operation unaided.

        Weaker than :attr:`is_declarative`, and deliberately so: an
        *optional* callable left at its default is no obstacle to
        building the unit, only to filling that one field. What blocks
        a form is a required callable, or a constructor argument that
        is an object rather than data.
        """
        return not self.constructor_extras and not any(
            p.required and p.is_callable for p in self.parameters
        )

    @property
    def callable_parameters(self) -> list[str]:
        """Fields that hold code rather than data."""
        return [p.name for p in self.parameters if p.is_callable]

    def required_parameters(self) -> list[str]:
        return [p.name for p in self.parameters if p.required]

    def to_dict(self) -> dict:
        """A plain, JSON-serializable dictionary."""
        return {
            "name": self.name,
            "class_name": self.class_name,
            "module": self.module,
            "category": self.category,
            "description": self.description,
            "plugin": self.plugin,
            "symbol": self.symbol,
            "doc": self.doc,
            "equations": list(self.equations),
            "assumptions": list(self.assumptions),
            "references": list(self.references),
            "numerical_method": self.numerical_method,
            "params_class": self.params_class,
            "declarative": self.is_declarative,
            "constructor_extras": list(self.constructor_extras),
            "buildable": self.is_buildable,
            "ports": {
                "inlets": list(self.ports.inlets),
                "n_inlets": self.ports.n_inlets,
                "n_outlets": self.ports.n_outlets,
                "variadic": self.ports.variadic,
            },
            "parameters": [dataclasses.asdict(p) for p in self.parameters],
        }


# ---------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------


def _stream_repr() -> str:
    from difflow.streams import Stream

    return str(Stream)


def _is_stream(annotation: Any) -> bool:
    """Whether an annotation denotes a :data:`~difflow.streams.Stream`.

    Two forms have to be recognised. Where the alias has been evaluated
    it renders as ``dict[str, Array | float]``, which must be matched
    *exactly*: the info payload every unit returns alongside its outlets
    is ``dict[str, Array]``, and a loose match counts it as a stream.
    Where a module uses ``from __future__ import annotations`` the
    annotation survives as the literal string ``"Stream"``.
    """
    if annotation is inspect.Parameter.empty:
        return False
    text = str(annotation).strip().strip("'\"")
    return text == _stream_repr() or text == "Stream"


def _split_top_level(text: str) -> list[str]:
    """Split ``a, b[c, d], e`` on its top-level commas."""
    parts, depth, current = [], 0, ""
    for ch in text:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current)
    return [p.strip() for p in parts]


def _outlet_count(ret: Any) -> int | None:
    """Number of leading ``Stream`` entries in a return annotation."""
    if ret is inspect.Signature.empty:
        return None
    args = typing.get_args(ret)
    if args:
        return sum(1 for a in args if _is_stream(a))
    # a lone stream, as the gas units return: one outlet, no info payload
    if _is_stream(ret):
        return 1
    # a string annotation, e.g. "tuple[Stream, dict]"
    text = str(ret).strip().strip("'\"")
    if text.startswith("tuple[") and text.endswith("]"):
        return sum(1 for p in _split_top_level(text[6:-1]) if _is_stream(p))
    return None


def _ports(cls: type) -> PortSpec:
    """Derive stream connectivity from the ``__call__`` signature.

    A class that defines no ``__call__`` of its own has no ports at all.
    Asking for one anyway reaches the metaclass slot, whose signature is
    ``(*args, **kwargs)`` -- which reads back as a unit accepting any
    number of inlets, and would draw a mixer where the catalog holds a
    model object with named methods.
    """
    if not inspect.isfunction(getattr(cls, "__call__", None)):
        return PortSpec()
    try:
        sig = inspect.signature(cls.__call__)
    except (TypeError, ValueError):
        return PortSpec()

    inlets, variadic = [], False
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            # *inlets on a mixer: any number of streams
            variadic = True
            inlets.append(name)
            continue
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        if _is_stream(param.annotation):
            inlets.append(name)

    # outlets are the Stream entries of the return tuple; the trailing
    # dict is the info payload every unit returns alongside them
    return PortSpec(
        inlets=inlets,
        n_outlets=_outlet_count(sig.return_annotation),
        variadic=variadic,
    )


def _params_class(cls: type) -> type | None:
    """The ``Params`` dataclass a unit is constructed from.

    Taken from the annotation of ``__init__``'s first argument, falling
    back to the ``<ClassName>Params`` naming convention.
    """
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return None
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        ann = param.annotation
        if dataclasses.is_dataclass(ann):
            return ann
        if isinstance(ann, str) and ann.endswith("Params"):
            found = getattr(inspect.getmodule(cls), ann, None)
            if dataclasses.is_dataclass(found):
                return found
        break
    guess = getattr(inspect.getmodule(cls), f"{cls.__name__}Params", None)
    return guess if dataclasses.is_dataclass(guess) else None


def _is_code(annotation: Any) -> bool:
    """Whether a field holds a function or an arbitrary object."""
    text = str(annotation)
    return "Callable" in text or text.strip() in ("typing.Any", "Any")


def _parameters(
    params_cls: type | None, metadata: Any = None
) -> list[ParameterSpec]:
    """Describe every field of a ``Params`` dataclass.

    Args:
        params_cls: the dataclass, or ``None`` if none was found.
        metadata: the owning unit's
            :class:`~difflow.report.metadata.UnitMetadata`, whose
            ``parameter_units`` and ``parameter_symbols`` fill in fields
            that do not carry their own dataclass metadata.
    """
    if params_cls is None:
        return []
    units = dict(getattr(metadata, "parameter_units", None) or {})
    symbols = dict(getattr(metadata, "parameter_symbols", None) or {})
    specs = []
    for f in dataclasses.fields(params_cls):
        has_default = (
            f.default is not dataclasses.MISSING
            or f.default_factory is not dataclasses.MISSING  # type: ignore[misc]
        )
        default = None
        if f.default is not dataclasses.MISSING:
            default = repr(f.default)
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            default = f"{f.default_factory.__name__}()"  # type: ignore[misc]
        specs.append(ParameterSpec(
            name=f.name,
            type=str(f.type),
            default=default,
            required=not has_default,
            is_callable=_is_code(f.type),
            units=f.metadata.get("units") or units.get(f.name),
            symbol=symbols.get(f.name),
            description=f.metadata.get("description"),
        ))
    return specs


def describe_class(
    cls: type,
    *,
    name: str | None = None,
    category: str = "general",
    description: str = "",
    plugin: str = "core",
) -> OperationSchema:
    """Build a schema for a unit operation class.

    Works on any class, registered or not, which is what makes it
    usable on a plugin's units before they are wired in.
    """
    from difflow.report.metadata import get_metadata
    from difflow.serialize import constructor_extras

    params_cls = _params_class(cls)
    doc = (description or cls.__doc__ or "").strip()
    meta = get_metadata(cls)
    return OperationSchema(
        name=name or cls.__name__,
        class_name=cls.__name__,
        module=cls.__module__,
        category=category,
        description=doc.splitlines()[0] if doc else "",
        plugin=plugin,
        symbol=meta.symbol,
        doc=inspect.cleandoc(cls.__doc__) if cls.__doc__ else "",
        equations=list(meta.equations),
        assumptions=list(meta.assumptions),
        references=list(meta.references),
        numerical_method=meta.numerical_method,
        ports=_ports(cls),
        parameters=_parameters(params_cls, meta),
        params_class=params_cls.__name__ if params_cls else None,
        constructor_extras=constructor_extras(cls),
    )


def describe_operation(name: str, registry=None) -> OperationSchema:
    """Schema for a registered operation, by name.

    Args:
        name: the registered name.
        registry: the registry to look in; defaults to the global one
            with plugins loaded.

    Raises:
        KeyError: if nothing is registered under that name.
    """
    registry = registry or _default_registry()
    info = registry.list_operations().get(name)
    if info is None:
        raise KeyError(
            f"no operation registered as {name!r}; "
            f"{len(registry.list_operations())} are available"
        )
    return describe_class(
        info.cls, name=info.name, category=info.category,
        description=info.description, plugin=info.plugin,
    )


def catalog(category: str | None = None, registry=None) -> dict[str, OperationSchema]:
    """Schemas for every registered operation.

    Args:
        category: restrict to one category.
        registry: the registry to read; defaults to the global one.

    Returns:
        ``{name: OperationSchema}``, sorted by name.
    """
    registry = registry or _default_registry()
    out = {}
    for name, info in sorted(registry.list_operations().items()):
        if category is not None and info.category != category:
            continue
        out[name] = describe_class(
            info.cls, name=info.name, category=info.category,
            description=info.description, plugin=info.plugin,
        )
    return out


def _default_registry():
    from difflow.plugins import load_plugins, registry

    load_plugins()
    return registry


# ---------------------------------------------------------------------
# Core registration
# ---------------------------------------------------------------------


def core_operations() -> dict[str, type]:
    """The core unit operation classes, by the name to register them as.

    Discovered from ``difflow.units`` rather than listed by hand, so a
    new unit joins the catalog as soon as it is exported.
    """
    import difflow

    found: dict[str, type] = {}
    for attr in getattr(difflow, "__all__", []):
        obj = getattr(difflow, attr, None)
        if not inspect.isclass(obj):
            continue
        if not obj.__module__.startswith("difflow.units"):
            continue
        if attr.endswith("Params") or attr in NOT_OPERATIONS:
            continue
        found[CORE_NAME_OVERRIDES.get(attr, attr)] = obj
    return found


def register_core_operations(registry=None) -> int:
    """Register the core unit operations.

    The registry was previously populated only by plugins, so the
    catalog held bio, carbon-capture, gas and REE units but none of the
    reactors, separators, columns or exchangers that most flowsheets are
    built from. This closes that gap.

    Args:
        registry: registry to populate; defaults to the global one.

    Returns:
        How many operations were registered.
    """
    if registry is None:
        from difflow.plugins import registry as registry  # noqa: PLW0127

    count = 0
    for name, cls in core_operations().items():
        module = cls.__module__.rsplit(".", 1)[-1]
        doc = (cls.__doc__ or "").strip()
        registry.register(
            name, cls,
            category=CORE_CATEGORIES.get(module, "general"),
            description=doc.splitlines()[0] if doc else "",
            plugin="core",
        )
        count += 1
    return count
