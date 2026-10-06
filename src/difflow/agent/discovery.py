"""What difflow can do, read from the code that does it.

Nothing here is a list someone maintains. Operations come from the
registry and :mod:`difflow.catalog` (the ``Params`` dataclass, the
``__call__`` signature, the docstrings); libraries that are not palette
operations (``difflow.planning``, ``difflow_refinery.reforming``, ...)
come from each package's ``__all__`` and :mod:`inspect`; documentation
comes from the index the editor's assistant already searches. A plugin
installed tomorrow is described by the same calls, with no change here.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import pkgutil
from typing import Any

#: how much of a docstring :func:`describe_api` returns
DOC_LIMIT = 6000


def _summary(doc: str | None) -> str:
    """The first paragraph of a docstring, on one line."""
    text = inspect.cleandoc(doc or "")
    return " ".join(text.split("\n\n", 1)[0].split())


def _signature(obj) -> str | None:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return None


def _kind(obj) -> str:
    if inspect.ismodule(obj):
        return "module"
    if inspect.isclass(obj):
        return "dataclass" if dataclasses.is_dataclass(obj) else "class"
    if callable(obj):
        return "function"
    return type(obj).__name__


# -- plugins and operations -------------------------------------------


def plugin_status() -> dict:
    """Installed plugins, the operations each registered, and any failures."""
    from difflow.plugins import plugin_status as status

    return {"ok": True, **status()}


def packages() -> list[str]:
    """``difflow`` and the top-level package of every installed plugin."""
    from difflow.plugins import discover_plugins  # noqa: F401 -- loads metadata
    import importlib.metadata

    found = {"difflow"}
    for ep in importlib.metadata.entry_points(group="difflow.plugins"):
        found.add(ep.value.split(":", 1)[0].split(".", 1)[0])
    return sorted(found)


def list_operations(query: str = "", category: str = "", plugin: str = "") -> dict:
    """Registered unit operations, filtered, one compact line each."""
    from difflow.catalog import catalog

    words = query.lower().split()
    found = []
    for name, schema in sorted(catalog().items()):
        if category and schema.category != category:
            continue
        if plugin and plugin not in (schema.plugin, schema.plugin.removeprefix("difflow_")):
            continue
        summary = _summary(schema.description)
        haystack = f"{name} {schema.category} {schema.plugin} {summary}".lower()
        if any(w not in haystack for w in words):
            continue
        found.append({
            "name": name,
            "category": schema.category,
            "plugin": schema.plugin,
            "summary": summary[:200],
            "inlets": schema.ports.n_inlets,
            "outlets": schema.ports.n_outlets,
            "declarative": schema.is_declarative,
        })
    return {"ok": True, "count": len(found), "operations": found}


# -- libraries ----------------------------------------------------------


def _allowed(name: str) -> bool:
    """Only difflow's own packages: importing a module runs it."""
    top = name.split(".", 1)[0]
    return top == "difflow" or top.startswith("difflow_")


def _resolve(name: str):
    """The object a dotted name refers to, importing the longest module."""
    parts = name.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        for attr in parts[cut:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(f"cannot import {name!r}")


def list_api(package: str = "difflow", query: str = "") -> dict:
    """The public names of a difflow package, with signatures and summaries.

    Reads ``__all__`` (or the public module attributes when there is none)
    and lists the subpackages and modules beneath, so the library can be
    walked one level at a time.
    """
    if not _allowed(package):
        return {"ok": False, "error": f"{package!r} is not a difflow package "
                f"(the packages are {', '.join(packages())})"}
    try:
        module = importlib.import_module(package)
    except ImportError as exc:
        return {"ok": False, "error": f"cannot import {package!r}: {exc}"}
    names = getattr(module, "__all__", None) or [
        n for n in vars(module) if not n.startswith("_")]
    words = query.lower().split()
    entries = []
    for name in sorted(names):
        obj = getattr(module, name, None)
        if obj is None:
            continue
        summary = _summary(getattr(obj, "__doc__", None)) \
            if not isinstance(obj, (int, float, str, tuple, list, dict)) else ""
        if any(w not in f"{name} {summary}".lower() for w in words):
            continue
        entry = {"name": name, "kind": _kind(obj), "summary": summary[:200]}
        if entry["kind"] in ("function", "class", "dataclass"):
            entry["signature"] = _signature(obj)
        entries.append(entry)
    children = []
    if hasattr(module, "__path__"):
        children = [{"name": f"{package}.{m.name}",
                     "kind": "package" if m.ispkg else "module"}
                    for m in pkgutil.iter_modules(module.__path__)
                    if not m.name.startswith("_")]
    return {"ok": True, "package": package,
            "summary": _summary(module.__doc__), "count": len(entries),
            "names": entries, "submodules": children}


def describe_api(name: str) -> dict:
    """The full signature and docstring of one difflow name.

    A class also lists its dataclass fields and public methods; a module
    is the same as :func:`list_api`.
    """
    if not _allowed(name):
        return {"ok": False, "error": f"{name!r} is not in a difflow package"}
    try:
        obj = _resolve(name)
    except (ImportError, AttributeError) as exc:
        return {"ok": False, "error": f"cannot find {name!r}: {exc}"}
    if inspect.ismodule(obj):
        return list_api(name)
    doc = inspect.cleandoc(getattr(obj, "__doc__", None) or "")
    out: dict[str, Any] = {
        "ok": True, "name": name, "kind": _kind(obj),
        "module": getattr(obj, "__module__", None),
        "signature": _signature(obj) if callable(obj) else None,
        "doc": doc[:DOC_LIMIT] + ("\n[...]" if len(doc) > DOC_LIMIT else ""),
    }
    if inspect.isclass(obj):
        out["bases"] = [b.__qualname__ for b in obj.__mro__[1:-1]]
        if dataclasses.is_dataclass(obj):
            out["fields"] = [
                {"name": f.name, "type": str(f.type),
                 "default": (repr(f.default) if f.default is not dataclasses.MISSING
                             else "<factory>" if f.default_factory is not dataclasses.MISSING
                             else None)}
                for f in dataclasses.fields(obj)]
        out["methods"] = [
            {"name": m, "signature": _signature(getattr(obj, m)),
             "summary": _summary(getattr(obj, m).__doc__)[:200]}
            for m in sorted(vars(obj)) if not m.startswith("_")
            and callable(getattr(obj, m, None))]
    return out


# -- species, documentation, examples ------------------------------------


def search_species(query: str = "", limit: int = 25) -> dict:
    """Species in the property database, with what data each one has."""
    from difflow import database

    words = query.lower().replace("-", "_").split()
    names = database.list_species()
    if words:
        try:
            alias = database.resolve_alias(query)
        except Exception:  # noqa: BLE001 -- not an alias
            alias = None
        names = [n for n in names if all(w in n for w in words) or n == alias]
    found = []
    for name in names[:max(int(limit), 1)]:
        info = database.get_species_info(name)
        props = info.get("critical") or info.get("ideal_thermo") or {}
        found.append({"name": name, "MW": props.get("MW"),
                      "critical": bool(info.get("critical")),
                      "ideal_thermo": bool(info.get("ideal_thermo"))})
    return {"ok": True, "count": len(names), "species": found,
            "note": "critical: has Tc/Pc/omega for a cubic EOS; "
                    "ideal_thermo: has Cp, Hvap and Antoine for IdealThermo"}


def search_docs(query: str, limit: int = 5) -> dict:
    """Sections of the difflow book that best match a question."""
    from difflow.gui import docs_index
    from difflow.gui.doclinks import BASE_URL

    index = docs_index.load()
    if index is None:
        return {"ok": False, "error": "the documentation index is not installed"}
    hits = docs_index.search(index, query, limit=max(int(limit), 1))
    return {"ok": True, "results": [
        {"heading": h["path"], "score": h["score"],
         "url": f"{BASE_URL}{h['source'].removesuffix('.md')}.html#{h['anchor']}",
         "text": h["text"]}
        for h in hits]}


def list_examples() -> dict:
    """The example flowsheets that open in a session."""
    from difflow.gui import examples

    return {"ok": True, "examples": examples.listing()}
