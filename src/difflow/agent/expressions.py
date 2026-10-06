"""Objectives without callables: a small expression language over a flowsheet.

Analysis in difflow takes functions (an objective of the solved streams, a
model of the parameters), and a function cannot cross an MCP boundary. A
string can. This compiles strings such as::

    vapor.F_ethyl_acetate / (vapor.F_ethanol + vapor.F_ethyl_acetate)
    -(12.0 * product.F_B - 1.5 * feed.total_flow - 0.02 * reactor.V)

into JAX functions of a solved flowsheet, so they differentiate like any
other difflow function. The names an expression may use:

``<stream>.<quantity>``
    ``T``, ``P``, ``total_flow`` or ``F_<species>`` of a solved stream.
``<unit>.<param>``
    A unit's parameter (a ``Params`` field or a call parameter), as it is
    at the point being evaluated, so ``reactor.V`` follows a lever.
``<name>``
    A named quantity (see :class:`Quantities`), itself an expression.
``econ.<function>``
    A function of :mod:`difflow.economics`, found by introspection.

and the operations ``+ - * / **``, unary minus, numbers, and the functions
``exp log log10 sqrt abs min max`` (``min``/``max`` are not smooth; prefer
them only where the kink is the point).

Parsing is with :mod:`ast` against a whitelist: nothing is ``eval``-ed, and
an attribute or call outside the list above is refused with its position.
"""

from __future__ import annotations

import ast
from functools import reduce

#: functions an expression may call by bare name
_FUNCTIONS = {
    "exp": "exp", "log": "log", "log10": "log10", "sqrt": "sqrt",
    "abs": "abs", "min": "minimum", "max": "maximum",
}
_BINOPS = {
    ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
}
#: deepest chain of named quantities referring to one another
MAX_DEPTH = 32


class ExpressionError(ValueError):
    """An expression that cannot be parsed or refers to nothing known."""


def _economics() -> dict:
    """The callables :mod:`difflow.economics` exports, by name."""
    import difflow.economics as econ

    names = getattr(econ, "__all__", None) or [n for n in vars(econ)
                                               if not n.startswith("_")]
    return {n: getattr(econ, n) for n in names if callable(getattr(econ, n, None))
            and not isinstance(getattr(econ, n), type)}


def parse(text: str) -> ast.expr:
    """``text`` as a checked expression tree, or :class:`ExpressionError`."""
    try:
        tree = ast.parse(text.strip(), mode="eval").body
    except SyntaxError as exc:
        raise ExpressionError(f"cannot parse {text!r}: {exc.msg}") from None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.Load, ast.operator, ast.unaryop,
                             ast.keyword)):
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            continue
        if isinstance(node, (ast.BinOp,)) and type(node.op) in _BINOPS:
            continue
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            continue
        if isinstance(node, ast.Name):
            continue
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            continue
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in _FUNCTIONS:
                continue
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) \
                    and f.value.id == "econ":
                continue
        raise ExpressionError(
            f"{type(node).__name__} is not allowed in an expression "
            f"(column {getattr(node, 'col_offset', '?')} of {text!r})")
    return tree


def references(text: str) -> set[str]:
    """Every ``a.b`` and bare name an expression mentions (not functions)."""
    tree = parse(text)
    called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    found = set()
    for node in ast.walk(tree):
        if id(node) in called:
            continue
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            found.add(f"{node.value.id}.{node.attr}")
        elif isinstance(node, ast.Name):
            found.add(node.id)
    # a Name under an Attribute is part of it, not a quantity of its own
    inner = {n.value.id for n in ast.walk(tree) if isinstance(n, ast.Attribute)
             and isinstance(n.value, ast.Name)}
    return {r for r in found if "." in r or r not in inner}


class Context:
    """What names resolve against while one expression is evaluated."""

    def __init__(self, streams: dict, flowsheet, quantities: dict[str, str]):
        self.streams = streams
        self.flowsheet = flowsheet
        self.quantities = quantities
        self._units = {u.name: u for u in flowsheet.units}
        self._econ = None

    def attribute(self, owner: str, name: str):
        from difflow.gui.sensitivity import quantity

        if owner in self.streams:
            try:
                return quantity(self.streams[owner], name)
            except KeyError as exc:
                raise ExpressionError(f"{owner}.{name}: {exc.args[0]}") from None
        unit = self._units.get(owner)
        if unit is not None:
            params = getattr(unit.operation, "params", None)
            if params is not None and hasattr(params, name):
                return getattr(params, name)
            if name in (unit.params or {}):
                return unit.params[name]
            raise ExpressionError(f"unit {owner!r} has no parameter {name!r}")
        raise ExpressionError(
            f"{owner!r} is neither a stream nor a unit (streams: "
            f"{', '.join(sorted(self.streams))}; units: {', '.join(self._units)})")

    def econ(self, name: str):
        if self._econ is None:
            self._econ = _economics()
        if name not in self._econ:
            raise ExpressionError(f"difflow.economics has no function {name!r}")
        return self._econ[name]


def evaluate(text: str, context: Context, _depth: int = 0):
    """The value of ``text`` in ``context``; traceable when the streams are."""
    import jax.numpy as jnp

    if _depth > MAX_DEPTH:
        raise ExpressionError("named quantities refer to each other in a cycle")

    def run(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return _BINOPS[type(node.op)](run(node.left), run(node.right))
        if isinstance(node, ast.UnaryOp):
            value = run(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.Attribute):
            return context.attribute(node.value.id, node.attr)
        if isinstance(node, ast.Name):
            if node.id in context.quantities:
                return evaluate(context.quantities[node.id], context, _depth + 1)
            raise ExpressionError(
                f"unknown name {node.id!r}: use <stream>.<quantity>, "
                f"<unit>.<param>, or a named quantity "
                f"({', '.join(context.quantities) or 'none defined'})")
        if isinstance(node, ast.Call):
            args = [run(a) for a in node.args]
            kwargs = {k.arg: run(k.value) for k in node.keywords}
            f = node.func
            if isinstance(f, ast.Name):
                fn = getattr(jnp, _FUNCTIONS[f.id])
                if f.id in ("min", "max"):
                    if kwargs or not args:
                        raise ExpressionError(f"{f.id} takes values, not keywords")
                    return reduce(fn, args)
                return fn(*args, **kwargs)
            return context.econ(f.attr)(*args, **kwargs)
        raise ExpressionError(f"cannot evaluate {ast.dump(node)}")

    return run(parse(text))


class Quantities:
    """Named expressions kept on a flowsheet (``view["quantities"]``).

    Saved with the file, so a ``capex`` or ``revenue`` defined once is
    there the next time the flowsheet is opened, and used by name in any
    other expression.
    """

    KEY = "quantities"

    def __init__(self, flowsheet):
        self.flowsheet = flowsheet

    def all(self) -> dict[str, str]:
        return dict(self.flowsheet.view.get(self.KEY) or {})

    def define(self, name: str, expression: str) -> None:
        if not name.isidentifier() or name == "econ":
            raise ExpressionError(f"{name!r} is not a usable quantity name")
        if name in {u.name for u in self.flowsheet.units}:
            raise ExpressionError(f"{name!r} is already a unit name")
        parse(expression)
        if name in references(expression):
            raise ExpressionError(f"{name!r} refers to itself")
        stored = self.all()
        stored[name] = expression
        self.flowsheet.view[self.KEY] = stored

    def remove(self, name: str) -> bool:
        stored = self.all()
        found = stored.pop(name, None) is not None
        if stored:
            self.flowsheet.view[self.KEY] = stored
        else:
            self.flowsheet.view.pop(self.KEY, None)
        return found
