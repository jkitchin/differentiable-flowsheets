"""Emit a flowsheet as runnable Python.

:mod:`difflow.serialize` gives a flowsheet a file format. This gives it
a *source* form --- the same model written as the code someone would
have typed:

    >>> print(codegen.to_python(fs))
    fs = Flowsheet(species_order=["A", "B"])
    fs.add_feed("feed", make_stream({"A": 1.0}, T=350.0, P=101325.0))
    ...

The two together close the loop a researcher needs. A graphical editor
that can only be entered is worse than none: the moment you want
something the palette does not offer, you have to be able to drop into
Python and keep going. Build in a GUI, export a script, edit it, and
read the result back through :mod:`difflow.serialize`.

The generated script is meant to be *read and edited*, not just run,
so it is laid out the way the docs write flowsheets: imports, then
thermodynamics, then kinetics, then the flowsheet, then the solve.

The same restriction as the JSON format applies, and for the same
reason: a parameter holding a callable can only be emitted when it
carries the specification that built it (see
:mod:`difflow.kinetics`). A generated script that quietly dropped a
rate law would still run, and would be a different model.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import jax.numpy as jnp

from difflow.serialize import (
    REF_TAG,
    SerializationError,
    SerializationWarning,
    _ref_for,
    _takes_params_first,
    _with_refs,
    constructor_extras,
    optional_extras,
)

#: rendered when a value is a JAX array
ARRAY_CALL = "jnp.array"


class CodegenError(SerializationError):
    """A flowsheet cannot be written as Python.

    Shares a base with :class:`~difflow.serialize.SerializationError`
    because the two formats refuse the same things: unregistered
    operations, and callables that do not carry the specification that
    produced them.
    """


def _render(value: Any, where: str, imports: set[str]) -> str:
    """Render one value as a Python expression."""
    if isinstance(value, float):
        return _render_nested(value)            # inf/nan are not names
    if value is None or isinstance(value, (bool, int, str)):
        return repr(value)
    # An object the code context defines is written as the *name* it is
    # bound to there, because the code context is emitted above as the
    # script's preamble. Same rule as the JSON format, and same reason:
    # rendering it again would make a second, equal object where the
    # model has one.
    ref = _ref_for(value)
    if ref is not None:
        return ref[REF_TAG]
    if callable(value):
        spec = getattr(value, "__difflow_spec__", None)
        if spec is None:
            raise CodegenError(
                f"{where} holds a callable "
                f"({getattr(value, '__name__', type(value).__name__)!r}) that "
                "does not record how it was built, so it cannot be written as "
                "source. Build it from data instead -- see "
                "difflow.kinetics.mass_action_kinetics."
            )
        imports.add(spec["factory"])
        args = ", ".join(
            f"{k}={_render(v, f'{where}.{k}', imports)}"
            for k, v in spec["kwargs"].items()
        )
        call = f"{spec['factory']}({args})"
        return f"{call}.{spec['attr']}" if spec.get("attr") else call
    if hasattr(value, "_fields"):                       # a NamedTuple
        _note(imports, type(value))
        inner = ", ".join(
            f"{name}={_render(getattr(value, name), f'{where}.{name}', imports)}"
            for name in value._fields
        )
        return f"{type(value).__name__}({inner})"
    if hasattr(value, "shape") or hasattr(value, "tolist"):
        imports.add("jnp")
        return f"{ARRAY_CALL}({_render_nested(jnp.asarray(value).tolist())})"
    if dataclasses.is_dataclass(value):
        _note(imports, type(value))
        inner = ", ".join(
            f"{f.name}={_render(getattr(value, f.name), f'{where}.{f.name}', imports)}"
            for f in dataclasses.fields(value)
        )
        return f"{type(value).__name__}({inner})"
    if isinstance(value, (list, tuple)):
        inner = ", ".join(
            _render(v, f"{where}[{i}]", imports) for i, v in enumerate(value)
        )
        return f"[{inner}]"
    if isinstance(value, dict):
        inner = ", ".join(
            f"{k!r}: {_render(v, f'{where}[{k!r}]', imports)}"
            for k, v in value.items()
        )
        return "{" + inner + "}"
    raise CodegenError(
        f"{where} holds {type(value).__name__}, which has no source form."
    )


def _render_nested(value: Any) -> str:
    """Render a nested list of numbers.

    ``repr`` is not usable here: it writes infinity as ``inf``, which is
    not a name in the generated module and would fail on import.
    """
    if isinstance(value, list):
        return "[" + ", ".join(_render_nested(v) for v in value) + "]"
    if isinstance(value, float):
        if value != value:
            return "float('nan')"
        if value == float("inf"):
            return "float('inf')"
        if value == float("-inf"):
            return "-float('inf')"
    return repr(value)


def _render_thermo(obj: Any, where: str, imports: set[str]) -> str:
    """Render a thermo object, preferring the species database.

    A thermo built from the database reads far better as the expression
    that built it than as a wall of inlined constants, so that form is
    used whenever every species is in the database.
    """
    from difflow import list_species

    kind = type(obj).__name__
    if kind != "IdealThermo":
        raise CodegenError(
            f"{where} holds a {kind}, which cannot be written as source. "
            "Only IdealThermo is supported so far."
        )
    imports.add("IdealThermo")
    names = list(obj.species)
    if all(n in list_species() for n in names):
        imports.add("get_species_data")
        return (
            f"IdealThermo({{s: get_species_data(s) for s in {names!r}}})"
        )
    inner = ", ".join(
        f"{n!r}: {_render(data, f'{where}[{n!r}]', imports)}"
        for n, data in obj.species.items()
    )
    return "IdealThermo({" + inner + "})"


def _render_stream(stream: dict, imports: set[str]) -> str:
    """Render a stream as a ``make_stream`` call."""
    imports.add("make_stream")
    flows, temperature, pressure = {}, None, None
    for key, value in stream.items():
        if key == "T":
            temperature = float(value)
        elif key == "P":
            pressure = float(value)
        elif key.startswith("F_"):
            flows[key[2:]] = float(value)
    body = ", ".join(f"{k!r}: {v!r}" for k, v in flows.items())
    return (
        "make_stream({" + body + "}, "
        f"T={temperature!r}, P={pressure!r})"
    )


@_with_refs
def to_python(
    flowsheet,
    *,
    include_solve: bool = True,
    registry=None,
    code_context: str | None = None,
) -> str:
    """Write a flowsheet as a runnable Python script.

    Args:
        flowsheet: the flowsheet to emit.
        include_solve: append a ``__main__`` block that solves it.
        registry: operation registry for the name lookup; defaults to
            the global one.
        code_context: Python emitted above the flowsheet, defining the
            objects it refers to by name. Defaults to the flowsheet's
            own ``view["code_context"]``, which is what the editor
            writes there.
        refs: the bindings that code context produces, keyword-only. A
            value that is one of them is emitted as its name rather than
            rendered again --- see
            :func:`difflow.serialize.ref_namespace`.

    Returns:
        Python source. Running it rebuilds the same flowsheet --- code
        context included, so a script exported from the editor is what
        actually ran there.

    Raises:
        CodegenError: if an operation is unregistered, or a parameter
            holds a callable that does not record how it was built.
    """
    from difflow import __version__
    from difflow.serialize import _registry_name

    imports: set[str] = _Imports({"Flowsheet", "Unit"})
    body: list[str] = []

    # The code context goes first: everything below may refer to it by
    # name, and it carries its own imports.
    if code_context is None:
        code_context = (getattr(flowsheet, "view", None) or {}).get("code_context")
    preamble: list[str] = []
    if code_context and code_context.strip():
        preamble += [
            "# --- code context: the objects this flowsheet refers to by name",
            *code_context.strip().splitlines(),
            "# --- end code context",
            "",
        ]

    body.append(
        f"fs = Flowsheet(species_order={list(flowsheet.species_order)!r}, "
        f"default_flow={flowsheet.default_flow!r}, "
        f"default_T={None if flowsheet.infer_tear_T else flowsheet.default_T!r}, "
        f"default_P={None if flowsheet.infer_tear_P else flowsheet.default_P!r})"
    )
    body.append("")

    for name, stream in flowsheet.feeds.items():
        body.append(f"fs.add_feed({name!r}, {_render_stream(stream, imports)})")
    if flowsheet.feeds:
        body.append("")

    for unit in flowsheet.units:
        operation = unit.operation
        cls = type(operation)
        try:
            # refuse unregistered units early, as source or JSON alike
            _registry_name(cls, registry)
        except SerializationError as exc:
            raise CodegenError(str(exc)) from exc
        _note(imports, cls)

        where = f"unit {unit.name!r}"
        args = []
        # keyword arguments already emitted, so none is given twice
        emitted: set[str] = set()
        params = getattr(operation, "params", None)
        sig = _signature(cls)
        if (params is not None and dataclasses.is_dataclass(params)
                and sig is not None and not _takes_params_first(sig)):
            # `GasPipe(beta)`, `ControlValveDrop(dp_pa)`: the unit builds its
            # own Params from plain arguments, so its fields go back in as
            # keywords --- what serialize._build_operation does on load.
            # Wrapping them in a Params put a dataclass where a float
            # belongs, and the name lookup could even find another class
            # (core `PipeParams` for the gas plugin's `GasPipe`).
            for f in dataclasses.fields(params):
                value = _render(getattr(params, f.name), f"{where}.{f.name}", imports)
                args.append(f"{f.name}={value}")
                emitted.add(f.name)
        elif params is not None and dataclasses.is_dataclass(params):
            # the instance's own class: a lookup by name can find another
            # (gas `AmineAbsorber`'s params resolved to `StripperParams`)
            params_cls = type(params)
            _note(imports, params_cls)

            # A rate law built from data is hoisted to its own statement
            # and splatted back in. Inlining it would bury the reactor in
            # a single unreadable line, and would repeat the arrays the
            # factory derives anyway.
            supplied, prelude = _hoist_built_params(
                params, unit.name, where, imports
            )
            preamble.extend(prelude)

            rendered = [
                f"{f.name}={_render(getattr(params, f.name), f'{where}.{f.name}', imports)}"
                for f in dataclasses.fields(params)
                if f.name not in supplied
            ]
            fields = ", ".join(rendered)
            if supplied:
                splat = f"**{supplied['__variable__']}.params_kwargs()"
                fields = f"{splat}, {fields}" if fields else splat
            args.append(f"{params_cls.__name__}({fields})")

        for extra in constructor_extras(cls):
            value = getattr(operation, extra, None)
            if value is None or extra in emitted:
                continue
            ref = _ref_for(value)
            if ref is not None:
                rendered = ref[REF_TAG]
            elif type(value).__name__.endswith(("Thermo", "thermo")):
                rendered = f"thermo_{unit.name}"
                preamble.append(
                    f"{rendered} = {_render_thermo(value, where, imports)}"
                )
            else:
                rendered = _render(value, f"{where}.{extra}", imports)
            # positional after a keyword is a SyntaxError
            args.append(f"{extra}={rendered}" if emitted else rendered)

        # Optional arguments set away from their default, as keywords: a
        # `Mixer(species, thermo)` emitted as `Mixer(species)` runs, and
        # solves to a different answer (see serialize.optional_extras).
        for extra, value in optional_extras(operation).items():
            if extra in constructor_extras(cls) or extra in emitted:
                continue
            ref = _ref_for(value)
            try:
                if ref is not None:
                    rendered = ref[REF_TAG]
                elif type(value).__name__.endswith(("Thermo", "thermo")):
                    rendered = f"{extra}_{unit.name}"
                    preamble.append(
                        f"{rendered} = {_render_thermo(value, where, imports)}"
                    )
                else:
                    rendered = _render(value, f"{where}.{extra}", imports)
            except SerializationError as exc:
                import warnings

                warnings.warn(
                    f"{exc} It is left out of the script, so the unit is "
                    f"built with {extra}'s default.",
                    SerializationWarning, stacklevel=3,
                )
                continue
            args.append(f"{extra}={rendered}")

        # `Unit.params` are passed to the operation on every call (a
        # Splitter's `split_frac`). Leaving them out gave a script that
        # built and then raised "missing 1 required positional argument".
        call = ""
        if unit.params:
            items = ", ".join(
                f"{key!r}: {_render(value, f'{where} call parameter {key!r}', imports)}"
                for key, value in unit.params.items()
            )
            call = f", params={{{items}}}"
        body.append(
            f"fs.add_unit(Unit({unit.name!r}, {cls.__name__}({', '.join(args)}), "
            f"{list(unit.inlet_names)!r}, {list(unit.outlet_names)!r}{call}))"
        )

    if flowsheet.recycles:
        body.append("")
        for source, dest in flowsheet.recycles.items():
            body.append(f"fs.add_recycle({source!r}, {dest!r})")

    difflow_names = sorted(n for n in imports if n != "jnp")
    lines = [
        f'"""Flowsheet generated by difflow {__version__}.',
        "",
        "Edit freely --- this is ordinary difflow code.",
        '"""',
        "",
    ]
    if "jnp" in imports:
        lines.append("import jax.numpy as jnp")
        lines.append("")
    for package, names in _packages_for(difflow_names, imports.objects).items():
        lines.append(_import_block(names, package=package))
    lines.append("")
    if preamble:
        lines.extend(preamble)
        lines.append("")
    lines.extend(body)
    if include_solve:
        lines += [
            "",
            "",
            'if __name__ == "__main__":',
            "    streams = fs.solve()",
            "    for name, stream in streams.items():",
            "        print(name, stream)",
        ]
    return "\n".join(lines) + "\n"


def _hoist_built_params(params, unit_name: str, where: str, imports: set[str]):
    """Lift a data-built rate law out of the parameter list.

    ``mass_action_kinetics`` supplies four fields at once through
    ``params_kwargs()``. When a unit's ``rate_fn`` records that it came
    from there, and the other three still match, the whole group is
    emitted as one call and splatted back --- which is how the docs
    write it, and how someone editing the script would want it.

    Returns ``(supplied, statements)``, where ``supplied`` names the
    fields covered (plus the variable holding them) and is empty when
    nothing could be hoisted.
    """
    rate_fn = getattr(params, "rate_fn", None)
    spec = getattr(rate_fn, "__difflow_spec__", None)
    if spec is None:
        return {}, []

    group = ("rate_fn", "stoich", "rate_params", "species_order")
    if not all(hasattr(params, name) for name in group):
        return {}, []

    imports.add(spec["factory"])
    variable = f"kinetics_{unit_name}"
    args = ", ".join(
        f"{k}={_render(v, f'{where}.{k}', imports)}"
        for k, v in spec["kwargs"].items()
    )
    statement = f"{variable} = {spec['factory']}({args})"
    supplied = {name: True for name in group}
    supplied["__variable__"] = variable
    return supplied, [statement]


def _signature(cls: type):
    """``cls.__init__``'s signature, or None where there is none to read."""
    import inspect

    try:
        return inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return None


class _Imports(set):
    """The names a script imports, and the class behind each one noted."""

    def __init__(self, *args):
        super().__init__(*args)
        self.objects: dict[str, type] = {}


def _note(imports: set[str], cls: type) -> None:
    """Import ``cls`` by name, remembering which class the name means."""
    imports.add(cls.__name__)
    objects = getattr(imports, "objects", None)
    if objects is None:
        return
    if objects.setdefault(cls.__name__, cls) is not cls:
        raise CodegenError(
            f"the flowsheet uses two different classes named "
            f"{cls.__name__!r} ({objects[cls.__name__].__module__} and "
            f"{cls.__module__}), which one script cannot import side by side."
        )


def _packages_for(names: list[str], objects: dict | None = None
                  ) -> dict[str, list[str]]:
    """Group names by the package that exports them, difflow first.

    A plugin unit is exported by its plugin, not by ``difflow``, so a
    script that imported every name from ``difflow`` failed on its first
    line for ``GasPipe`` or ``TFF``. A class in ``objects`` is matched by
    identity, since names collide: the gas plugin's ``Compressor`` is not
    ``difflow.Compressor``. Any other name (a factory such as
    ``mass_action_kinetics``) is looked up the way :mod:`difflow.serialize`
    rebuilds a type on load.
    """
    import importlib

    from difflow.serialize import _PACKAGES

    objects = objects or {}
    grouped: dict[str, list[str]] = {}
    for name in names:
        for package in _PACKAGES:
            try:
                module = importlib.import_module(package)
            except ImportError:
                continue
            found = getattr(module, name, None)
            if found is not None and (name not in objects
                                      or found is objects[name]):
                grouped.setdefault(package, []).append(name)
                break
        else:
            raise CodegenError(
                f"no package exports {name!r}, so the generated script "
                "could not import it. Export it from difflow or its plugin."
            )
    return {p: grouped[p] for p in _PACKAGES if p in grouped}


def _import_block(names: list[str], width: int = 79,
                  package: str = "difflow") -> str:
    """One import line if it fits, otherwise a parenthesised block."""
    single = f"from {package} import {', '.join(names)}"
    if len(single) <= width:
        return single
    inner = "\n".join(f"    {n}," for n in names)
    return f"from {package} import (\n{inner}\n)"


def save_script(flowsheet, path: str | Path, **kwargs) -> Path:
    """Write a flowsheet to a ``.py`` file.

    Returns:
        The path written.
    """
    path = Path(path)
    path.write_text(to_python(flowsheet, **kwargs))
    return path
