"""Derivatives of a solved flowsheet, in the two directions AD offers.

difflow's difference from every other flowsheet editor is that the model
is differentiable, and a results table that only reported numbers would
hide it. So the results panel can ask two questions, and they are not
the same question asked twice:

**One lever, every stream.** ``d(everything) / d(V)`` --- how the whole
flowsheet moves when one parameter does. One *forward* pass, because
forward mode costs one pass per input regardless of how many outputs
there are. This is the mode that colours the canvas: every edge gets a
number.

**One output, every lever.** ``d(product flow) / d(everything)`` ---
which knobs the thing you care about actually responds to. One *reverse*
pass, because reverse mode costs one pass per output regardless of how
many inputs. This is the mode that ranks the levers.

Choosing the mode by which end is pinned is not an optimization detail
here; it is the reason both questions are cheap. Asking either one by
finite differences would cost one solve per lever.

Everything a lever can be is what :meth:`difflow.Flowsheet._apply_params`
accepts --- ``"<unit>.<param>"`` and ``"feed:<stream>.<field>"`` --- so
the picker offers exactly the keys the planning module's ``u`` list takes,
and nothing that will fail when it is used.
"""

from __future__ import annotations

from typing import Any

#: Feed-stream fields offered as levers, with their units. ``x_<species>``
#: is a lever too (see ``_update_feed_stream``) but is deliberately left
#: out of the picker: it rescales the *other* species to hold the total,
#: so its derivative is a composition swap rather than one knob, and a
#: reader would take it for the flow of that species.
FEED_FIELDS = {"total_flow": "mol/s", "T": "K", "P": "Pa"}


def _scalar(value: Any) -> float | None:
    """``value`` as a real scalar, or ``None`` if it is not one.

    The test is what ``jax.grad`` can actually take a derivative with
    respect to, not what the annotation claims: a field annotated
    ``float`` that currently holds an array is not a scalar lever, and a
    bool is a flag rather than a knob even though Python calls it an int.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape == () and dtype is not None and dtype.kind == "f":
        return float(value)
    return None


def _units_for(operation) -> dict[str, str | None]:
    """Parameter units for one operation, from the catalog."""
    from difflow.catalog import describe_class

    try:
        spec = describe_class(type(operation))
    except Exception:                       # a class the catalog cannot read
        return {}
    return {p.name: p.units for p in spec.parameters}


def levers(flowsheet) -> list[dict]:
    """Every parameter of this flowsheet a derivative can be taken against.

    A field qualifies when its *current value* is a real scalar. That
    rules out the rate functions, the thermo objects, the species lists
    and the arrays --- very nearly the set the inspector greys out,
    arrived at from the other direction --- and it rules them out by the
    same test the solver would apply rather than by a list kept
    somewhere else.
    """
    found: list[dict] = []
    for unit in getattr(flowsheet, "units", []):
        params = getattr(unit.operation, "params", None)
        if params is None:
            continue
        units = _units_for(unit.operation)
        for name in params.keys():
            value = _scalar(params[name])
            if value is None:
                continue
            found.append({
                "key": f"{unit.name}.{name}",
                "owner": unit.name,
                "field": name,
                "value": value,
                "units": units.get(name),
                "kind": "unit",
            })
    for feed, stream in (getattr(flowsheet, "feeds", None) or {}).items():
        for field, unit in FEED_FIELDS.items():
            value = (
                sum(float(v) for k, v in stream.items() if k.startswith("F_"))
                if field == "total_flow"
                else _scalar(stream.get(field))
            )
            if value is None:
                continue
            found.append({
                "key": f"feed:{feed}.{field}",
                "owner": feed,
                "field": field,
                "value": float(value),
                "units": unit,
                "kind": "feed",
            })
    return found


def outputs(streams: dict) -> list[dict]:
    """Every quantity of a solved flowsheet a derivative can be taken of.

    Keys are ``"<stream>.<quantity>"`` where the quantity is ``T``, ``P``,
    ``total_flow`` or ``F_<species>`` --- the same vocabulary
    :meth:`difflow.planning.Block.from_flowsheet` uses for its ``y``
    list, so a lever ranking done here names the outputs a planning
    export will name.
    """
    found: list[dict] = []
    for name, stream in streams.items():
        flows = [k for k in stream if k.startswith("F_")]
        total = sum(float(stream[k]) for k in flows)
        found.append({"key": f"{name}.total_flow", "stream": name,
                      "quantity": "total_flow", "value": total,
                      "units": "mol/s"})
        for k in ("T", "P"):
            if k in stream:
                found.append({"key": f"{name}.{k}", "stream": name,
                              "quantity": k, "value": float(stream[k]),
                              "units": "K" if k == "T" else "Pa"})
        for k in flows:
            found.append({"key": f"{name}.{k}", "stream": name, "quantity": k,
                          "value": float(stream[k]), "units": "mol/s"})
    return found


def quantity(stream: dict, name: str):
    """One quantity out of one stream, as a traceable expression.

    ``total_flow`` is a sum rather than a lookup, so it differentiates
    through every species at once.
    """
    if name == "total_flow":
        flows = [v for k, v in stream.items() if k.startswith("F_")]
        if not flows:
            raise KeyError("stream carries no species flows")
        total = flows[0]
        for value in flows[1:]:
            total = total + value
        return total
    if name not in stream:
        raise KeyError(f"stream has no {name!r}")
    return stream[name]


def _split(key: str) -> tuple[str, str]:
    """``"<stream>.<quantity>"``, minding that species keys have no dot."""
    stream, _, name = key.rpartition(".")
    if not stream:
        raise ValueError(f"output key {key!r} must be '<stream>.<quantity>'")
    return stream, name


def _numeric(streams: dict) -> dict:
    """The solved streams with the string entries dropped.

    A stream may carry a ``phase`` label, which is a string and not
    something a Jacobian has a column for.
    """
    return {
        name: {k: v for k, v in stream.items() if not isinstance(v, str)}
        for name, stream in streams.items()
    }


def _relative(derivative: float, u0: float, y0: float) -> float | None:
    """``d ln y / d ln u``, when both ends are nonzero.

    The dimensionless form is the comparable one --- a derivative in
    mol/s per m^3 cannot be ranked against one in mol/s per K --- and it
    is ``None`` rather than infinity when the base value is zero, because
    a percentage of nothing is not a large number, it is not a number.
    """
    if not u0 or not y0:
        return None
    return float(derivative) * u0 / y0


def forward(flowsheet, lever: str, **solve_kw) -> dict:
    """``d(every stream quantity) / d(one lever)``, in one forward pass.

    Args:
        flowsheet: the flowsheet to differentiate. Not modified.
        lever: an ``_apply_params`` key.
        **solve_kw: passed through to :meth:`Flowsheet.solve`.

    Returns:
        ``{"mode": "forward", "lever", "u0", "streams": {name: {quantity:
        {"value", "d", "rel"}}}}``.
    """
    import jax

    known = {item["key"]: item["value"] for item in levers(flowsheet)}
    if lever not in known:
        raise KeyError(f"{lever!r} is not a lever of this flowsheet")
    u0 = known[lever]

    def run(u):
        return _numeric(flowsheet._apply_params({lever: u}).solve(**solve_kw))

    base, tangent = jax.jvp(run, (u0,), (1.0,))
    streams = {}
    for name, stream in base.items():
        row = {}
        for k, value in stream.items():
            row[k] = {"value": float(value), "d": float(tangent[name][k]),
                      "rel": _relative(tangent[name][k], u0, float(value))}
        total = sum(float(v) for k, v in stream.items() if k.startswith("F_"))
        d_total = sum(float(v) for k, v in tangent[name].items()
                      if k.startswith("F_"))
        row["total_flow"] = {"value": total, "d": d_total,
                             "rel": _relative(d_total, u0, total)}
        streams[name] = row
    return {"mode": "forward", "lever": lever, "u0": u0, "streams": streams}


def reverse(flowsheet, target: str, keys: list[str] | None = None,
            **solve_kw) -> dict:
    """``d(one output) / d(every lever)``, in one reverse pass.

    Args:
        flowsheet: the flowsheet to differentiate. Not modified.
        target: ``"<stream>.<quantity>"``.
        keys: levers to rank; every scalar parameter by default.
        **solve_kw: passed through to :meth:`Flowsheet.solve`.

    Returns:
        ``{"mode": "reverse", "target", "y0", "levers": [{"key", "u0",
        "d", "rel", "units"}]}``, ranked by ``|rel|`` and then ``|d|``.
    """
    import jax

    stream_name, name = _split(target)
    found = levers(flowsheet)
    chosen = ([item for item in found if item["key"] in set(keys)]
              if keys is not None else found)
    if not chosen:
        raise ValueError("no levers to differentiate against")

    def pick(streams):
        if stream_name not in streams:
            raise KeyError(f"no stream named {stream_name!r}")
        return quantity(streams[stream_name], name)

    # `make_objective_fn` in all but name; spelled out because it takes
    # no solver keywords, and the panel needs to pass them (a recycle
    # that wants `clip_negative_flows=False` is not an exotic case).
    def objective(params):
        return pick(flowsheet._apply_params(params).solve(**solve_kw))

    u0 = {item["key"]: item["value"] for item in chosen}
    y0, gradient = jax.value_and_grad(objective)(u0)
    y0 = float(y0)
    ranked = [
        {"key": item["key"], "owner": item["owner"], "field": item["field"],
         "units": item["units"], "kind": item["kind"], "u0": item["value"],
         "d": float(gradient[item["key"]]),
         "rel": _relative(float(gradient[item["key"]]), item["value"], y0)}
        for item in chosen
    ]
    ranked.sort(key=lambda r: (abs(r["rel"]) if r["rel"] is not None else -1.0,
                               abs(r["d"])), reverse=True)
    return {"mode": "reverse", "target": target, "y0": y0, "levers": ranked}
