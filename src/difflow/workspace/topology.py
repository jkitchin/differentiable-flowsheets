"""A flowsheet described for looking at: topology, parameters, catalog.

Not :func:`difflow.serialize.to_dict`. That is a document meant to be
loaded back, and it refuses a flowsheet whose units hold objects JSON
cannot carry. This is a description for a reader --- the assistant's
brief, or a page that draws the flowsheet --- with every value already
reduced to something printable.
"""

from __future__ import annotations

import math

import jax.numpy as jnp


# ---------------------------------------------------------------------
# What the page says about the flowsheet
# ---------------------------------------------------------------------

#: Arrays longer than this are named rather than printed. A stoichiometry
#: vector is worth seeing; a 500-point property table is not.
_MAX_INLINE = 12


def _plain(value, depth: int = 0):
    """One parameter value as something JSON and a reader can both take.

    Whatever cannot be shown as data is named by its type instead of
    dropped, because a parameter that vanishes from a published page
    reads as a parameter the unit does not have. The non-finite floats
    become strings for the same reason `difflow.workspace` sends them as
    strings: ``JSON.parse`` rejects the ``Infinity`` Python writes, and
    ``mass_action_kinetics`` puts one in ``K_eq`` for every irreversible
    reaction.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)):
        f = float(value)
        return f if math.isfinite(f) else str(f).replace("inf", "Infinity")
    if isinstance(value, dict):
        if depth >= 2:
            return "{...}"
        return {str(k): _plain(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_INLINE:
            return f"<{len(value)} values>"
        return [_plain(v, depth + 1) for v in value]
    shape = getattr(value, "shape", None)
    if shape is not None:
        array = jnp.asarray(value)
        if array.size > _MAX_INLINE:
            return f"<array {tuple(array.shape)}>"
        return _plain(array.tolist(), depth + 1)
    return f"<{type(value).__name__}>"


def _display_params(operation) -> dict:
    """A unit's parameters, as far as they can be shown."""
    params = getattr(operation, "params", None)
    if params is None:
        return {}
    try:
        return {name: _plain(params[name]) for name in params.keys()}
    except Exception:                      # a params object that is not one
        return {}


def _topology(flowsheet) -> dict | None:
    """The flowsheet in the shape the canvas reads.

    Deliberately not :func:`difflow.serialize.to_dict`. That is a
    document meant to be loaded back, and it refuses a flowsheet whose
    units hold objects JSON cannot carry --- which is most interesting
    flowsheets, and no reason to publish a page without a picture. This
    is a description for looking at: the same node keys the editor uses,
    and every value already reduced to something printable.

    Returns ``None`` if there is nothing to draw.
    """
    from difflow.workspace.layout import auto_layout

    units = list(getattr(flowsheet, "units", None) or [])
    if not units:
        return None
    view = dict(getattr(flowsheet, "view", None) or {})
    return {
        "units": [
            {
                "name": u.name,
                "operation": type(u.operation).__name__,
                "inlets": list(u.inlet_names),
                "outlets": list(u.outlet_names),
                "params": _display_params(u.operation),
            }
            for u in units
        ],
        "feeds": {
            name: {k: _plain(v) for k, v in stream.items()}
            for name, stream in (getattr(flowsheet, "feeds", None) or {}).items()
        },
        "recycles": dict(getattr(flowsheet, "recycles", None) or {}),
        # Stored positions on top of the automatic ones, exactly as the
        # editor serves them, so a page published from a flowsheet someone
        # arranged in the GUI opens arranged that way.
        "view": {"nodes": {**auto_layout(flowsheet),
                           **(view.get("nodes") or {})}},
    }


def _catalog_for(flowsheet) -> dict:
    """Catalog entries for the operations this flowsheet actually uses.

    Trimmed to what the panel shows. The whole ``OperationSchema`` would
    carry parameter *defaults*, which are arbitrary objects and not
    JSON, and 87 entries where the page needs three.
    """
    from difflow.catalog import describe_class

    out: dict[str, dict] = {}
    for unit in getattr(flowsheet, "units", None) or []:
        cls = type(unit.operation)
        if cls.__name__ in out:
            continue
        try:
            spec = describe_class(cls)
        except Exception:                  # a class the catalog cannot read
            continue
        out[cls.__name__] = {
            "description": spec.description,
            "equations": list(spec.equations),
            "assumptions": list(spec.assumptions),
            "references": list(spec.references),
            "parameters": [
                {"name": p.name, "units": p.units, "description": p.description}
                for p in spec.parameters
            ],
        }
    return out
