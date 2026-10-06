"""What each plugin adds to the agent tools, discovered like its units.

A plugin registers unit operations through the ``difflow.plugins`` entry
point; it registers agent support through ``difflow.agent``, whose target
is a function returning an :class:`AgentSupport`. Core keeps no list of
plugins and no knowledge of them: a plugin installed tomorrow brings its
own guidance, symptom cards and tools.

    [project.entry-points."difflow.agent"]
    power = "difflow_power.agent:support"
"""

from __future__ import annotations

import importlib.metadata
import inspect
from dataclasses import dataclass, field
from typing import Callable

from difflow.diagnostics import Symptom

#: the entry-point group
GROUP = "difflow.agent"


@dataclass(frozen=True)
class PluginTool:
    """One extra tool a plugin provides.

    Attributes:
        function: ``function(workbench, **arguments) -> dict``; its signature
            after ``workbench`` (with type hints) is the tool's input schema
            and its docstring the description.
        kind: ``"read"``, ``"edit"`` or ``"exec"``, as in
            :data:`difflow.agent.TOOLS`.
    """

    function: Callable
    kind: str = "read"


@dataclass(frozen=True)
class AgentSupport:
    """A plugin's contribution to the agent tools.

    Attributes:
        plugin: The plugin's short name (its ``difflow.plugins`` entry point).
        summary: What it models and how to approach building with it.
        solver_notes: Its own solvers, the settings they need, and how they
            differ from the core recycle solve.
        symptoms: Failure patterns specific to it, matched by ``diagnose``
            alongside the core cards.
        tools: Extra tools, served as ``<plugin>_<name>``.
    """

    plugin: str
    summary: str = ""
    solver_notes: str = ""
    symptoms: tuple[Symptom, ...] = ()
    tools: dict[str, PluginTool] = field(default_factory=dict)


_cache: tuple[dict[str, AgentSupport], dict[str, str]] | None = None


def load(force: bool = False) -> tuple[dict[str, AgentSupport], dict[str, str]]:
    """Every installed plugin's support, and the plugins whose support failed.

    Returns:
        ``(support, errors)``: plugin name to :class:`AgentSupport`, and
        plugin name to the reason it could not be loaded.
    """
    global _cache
    if _cache is not None and not force:
        return _cache
    support, errors = {}, {}
    for ep in importlib.metadata.entry_points(group=GROUP):
        try:
            found = ep.load()()
            if not isinstance(found, AgentSupport):
                raise TypeError(f"{ep.value} returned {type(found).__name__}, "
                                "not AgentSupport")
            for name, tool in found.tools.items():
                params = list(inspect.signature(tool.function).parameters)
                if not params or params[0] != "workbench":
                    raise TypeError(f"tool {name!r} must take workbench first")
                if tool.kind not in ("read", "edit", "exec"):
                    raise TypeError(f"tool {name!r} has kind {tool.kind!r}")
            support[ep.name] = found
        except Exception as exc:  # noqa: BLE001 -- reported, not raised
            errors[ep.name] = f"{type(exc).__name__}: {exc}"
    _cache = (support, errors)
    return _cache


def symptoms() -> list[Symptom]:
    """Every plugin's symptom cards."""
    return [s for sup in load()[0].values() for s in sup.symptoms]


def tools() -> dict[str, tuple[str, PluginTool]]:
    """``<plugin>_<name>`` -> (plugin, tool) for every plugin tool."""
    return {f"{plugin}_{name}": (plugin, tool)
            for plugin, sup in load()[0].items() for name, tool in sup.tools.items()}
