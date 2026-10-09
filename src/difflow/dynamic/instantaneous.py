"""Stateless units in a dynamic flowsheet.

A heater or cooler held at a setpoint, a mixer or a splitter has no holdup
worth a state: its outlet follows its inlet instantly. :class:`InstantaneousUnit`
puts such a unit in a :class:`~difflow.dynamic.flowsheet.DynamicFlowsheet`
with zero states, so the flowsheet evaluates it inside every right-hand-side
call and integrates nothing for it (#390).

It wraps either a steady-state unit operation (anything callable as
``op(*inlets, **call_params)`` returning outlet streams, optionally followed
by an info dict, as every difflow unit does) or a plain function
``fn(inputs, params) -> {port: Stream}`` for logic no single unit covers --
a heater that holds a setpoint up to a duty limit, for instance.

Time and disturbances reach the unit by arity, the way ``add_feed`` tells
``(t,)`` from ``(t, args)``:

===============  =============================  =====================================
callable         positional parameters          called as
===============  =============================  =====================================
``fn``           2 (legacy)                     ``fn(inputs, params)``
``fn``           3                              ``fn(t, inputs, params)``
``fn``           4                              ``fn(t, inputs, params, args)``
``call_params``  1 (legacy)                     ``v(params)``
``call_params``  2                              ``v(t, params)``
``call_params``  3                              ``v(t, params, args)``
===============  =============================  =====================================

``args`` is the pytree handed to ``DynamicFlowsheet.simulate(args=...)``, the
same one a ``(t, args)`` feed reads. That is the intended channel for
time-varying disturbances (a feed composition step *and* a setpoint ramp read
one ``args``); ``params`` is for design parameters.
"""

from typing import Any, Callable, Sequence

import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream
from difflow.dynamic.state import StateSpec


def _n_positional(fn: Callable) -> int | None:
    """Positional parameter count of ``fn``, or None when it can't be told.

    A ``*args`` callable or one with no inspectable signature is None, which
    the callers treat as the legacy (shortest) form.
    """
    import inspect

    try:
        prms = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return None
    n = 0
    for prm in prms:
        if prm.kind is prm.VAR_POSITIONAL:
            return None
        if prm.kind in (prm.POSITIONAL_ONLY, prm.POSITIONAL_OR_KEYWORD):
            n += 1
    return n


class InstantaneousUnit:
    """Zero-state adapter for a steady-state unit or an outlet function.

    Ports: the inlets are read under ``inlet_ports`` and the outputs are
    returned under ``output_ports``; wire them to flowsheet streams with
    ``DynamicFlowsheet.add_unit(..., inlets=..., outlets=...)``.

    Args:
        op: A steady-state unit operation, called as
            ``op(*[inputs[p] for p in inlet_ports], **call_params)``. Its
            outlet streams (the leading streams of its result) are matched to
            ``output_ports`` in order. Mutually exclusive with ``fn``.
        fn: ``fn(inputs, params) -> {port: Stream}``, with ``inputs`` keyed by
            inlet port and ``params`` the flowsheet's (traced) params. Take
            ``(t, inputs, params)`` to see the time, or
            ``(t, inputs, params, args)`` to see the integrator ``args`` too.
        inlet_ports: Inlet port names. Default ``("inlet",)``.
        output_ports: Outlet port names. Default ``("outlet",)``.
        call_params: Fixed keyword arguments for ``op``. A value that is a
            callable ``params -> value`` is evaluated on every call with the
            flowsheet params, which is how a setpoint becomes a traced input.
            ``(t, params)`` and ``(t, params, args)`` callables also get the
            time, which is how a setpoint ramps.
        name: Unit name.

    Example:
        >>> heater = InstantaneousUnit(
        ...     Heater(HeaterParams(T_out=460.0), thermo), name="trim")  # doctest: +SKIP
        >>> fs.add_unit(heater, inlets={"inlet": "preheated"},
        ...             outlets={"outlet": "rx_in"})                      # doctest: +SKIP
    """

    #: Tells the flowsheet to pass its integrator ``args`` to ``outputs``.
    wants_args = True

    def __init__(
        self,
        op: Callable | None = None,
        *,
        fn: Callable[[dict[str, Stream], Any], dict[str, Stream]] | None = None,
        inlet_ports: Sequence[str] = ("inlet",),
        output_ports: Sequence[str] = ("outlet",),
        call_params: dict[str, Any] | None = None,
        name: str = "instantaneous",
    ):
        if (op is None) == (fn is None):
            raise ValueError("give exactly one of op or fn")
        self.op = op
        self.fn = fn
        self.inlet_ports = tuple(inlet_ports)
        self.output_ports = tuple(output_ports)
        self.call_params = dict(call_params or {})
        self.name = name

    def state_spec(self) -> StateSpec:
        return StateSpec([])

    def initial_state(self, inputs: dict[str, Stream], params=None) -> Array:
        return jnp.zeros(0)

    def derivatives(self, t: Array, state: Array, inputs: dict[str, Stream], params=None) -> Array:
        return jnp.zeros(0)

    def outputs(
        self, t: Array, state: Array, inputs: dict[str, Stream], params=None, args=None
    ) -> dict[str, Stream]:
        if self.fn is not None:
            n = _n_positional(self.fn)
            if n == 3:
                return self.fn(t, inputs, params)
            if n == 4:
                return self.fn(t, inputs, params, args)
            return self.fn(inputs, params)
        from difflow.flowsheet import _split_result

        def _eval(v):
            if not callable(v):
                return v
            n = _n_positional(v)
            if n == 2:
                return v(t, params)
            if n == 3:
                return v(t, params, args)
            return v(params)

        kwargs = {k: _eval(v) for k, v in self.call_params.items()}
        result = self.op(*[inputs[p] for p in self.inlet_ports], **kwargs)
        streams, _ = _split_result(result, list(self.output_ports))
        return dict(zip(self.output_ports, streams))

    def __repr__(self) -> str:
        what = type(self.op).__name__ if self.op is not None else getattr(self.fn, "__name__", "fn")
        return f"InstantaneousUnit(name='{self.name}', {what})"
