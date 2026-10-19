"""A steady unit behind outlet holdups: the dynamic stand-in for units whose insides are fast.

Some units settle in seconds to a minute inside while the plant around them
moves over many minutes: an amine contactor, a sulfur plant's furnace and
converters, a steam reformer's tubes, a column's trays relative to its
reboiler and drums. For those a full holdup model buys little, and a steady
unit with the right *inventories* on its outlets captures what the rest of
the plant sees: a delay and a smoothing of every product.

:class:`QuasiSteadyHoldup` does that. At every time it runs the steady
operation on its current inlets, ``F_ss = op(inlets)``, and drains each
outlet's well-mixed holdup ``n = tau F_out`` toward it::

    tau dF_out/dt = F_ss - F_out          (every F_ key)
    tau_T dT_out/dt = T_ss - T_out

so the holdup's accumulation is ``F_ss - F_out``: material and elements are
conserved exactly whenever the steady unit conserves them, and the steady
state is the steady unit's. ``tau`` is a physical holdup over a design flow
(a sump volume over its draw, a drum inventory over its throughput); the
temperature lag ``tau_T`` defaults to ``tau``. Pressure follows the steady
unit instantly.

This is an APPROXIMATION, and the docs say where it is one: the unit's own
internal dynamics (a column's composition profile moving tray by tray, an
amine loop's lean-loading inventory) are represented only by the outlet
lags.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import jax.numpy as jnp
from jax import Array

from difflow.dynamic.state import StateSpec, StateVar


class QuasiSteadyHoldup:
    """A steady unit operation with a first-order holdup on each outlet.

    Args:
        op: The steady operation, ``op(*inlet_streams) -> (outlet_streams..., [info])``
            (every difflow unit's calling convention; a trailing dict is
            dropped).
        inlet_ports: Port names of the inlets, in ``op``'s argument order.
        outlet_ports: Port names of the outlets, in ``op``'s return order.
        keys: The ``F_`` keys an outlet carries (missing ones are zero).
        tau: Holdup time constant (s), one for all outlets or ``{port: tau}``.
        tau_T: Temperature lag (s), default ``tau``.
        y0: Initial outlets ``{port: stream}`` (e.g. the steady unit's
            answer at the start, :meth:`at_steady`). Required: the outlets
            depend on the state alone, so a flowsheet evaluates this unit
            before its inlets exist.
        name: Unit name.
    """

    def __init__(self, op: Callable, inlet_ports: Sequence[str], outlet_ports: Sequence[str],
                 keys: Sequence[str], tau: float | Mapping[str, float], y0: Mapping[str, Mapping],
                 tau_T: float | Mapping[str, float] | None = None, name: str = "unit"):
        self.op = op
        self.inlet_ports = tuple(inlet_ports)
        self.output_ports = tuple(outlet_ports)
        self.output_dependencies = {p: () for p in self.output_ports}
        self.keys = tuple(keys)
        tau_T = tau if tau_T is None else tau_T
        self.tau = jnp.asarray([tau[p] if isinstance(tau, Mapping) else tau for p in self.output_ports],
                               dtype=float)
        self.tau_T = jnp.asarray([tau_T[p] if isinstance(tau_T, Mapping) else tau_T for p in self.output_ports],
                                 dtype=float)
        self.name = name
        self._P = {p: jnp.asarray(y0[p].get("P", 101325.0), dtype=float) for p in self.output_ports}
        self._scale = jnp.concatenate([self._vec_scale(y0[p]) for p in self.output_ports])
        self._y0 = jnp.concatenate([self._vec(y0[p]) for p in self.output_ports]) / self._scale

    @classmethod
    def at_steady(cls, op: Callable, inlets: Mapping[str, Mapping], inlet_ports: Sequence[str],
                  outlet_ports: Sequence[str], keys: Sequence[str], tau, **kw) -> "QuasiSteadyHoldup":
        """Build with the initial outlets the steady unit gives on ``inlets`` (a start at rest)."""
        outs = cls._call(op, [inlets[p] for p in inlet_ports], len(outlet_ports))
        return cls(op, inlet_ports, outlet_ports, keys, tau, y0=dict(zip(outlet_ports, outs)), **kw)

    @staticmethod
    def _call(op, inlets, n_out):
        out = op(*inlets)
        if isinstance(out, Mapping):
            out = (out,)
        out = tuple(out)
        if len(out) > n_out and isinstance(out[-1], Mapping) and not any(k.startswith("F_") for k in out[-1]):
            out = out[:-1]                                  # the info dict
        if len(out) != n_out:
            raise ValueError(f"the operation returned {len(out)} outlets, expected {n_out}")
        return out

    # ----- state ---------------------------------------------------------------------

    @property
    def n_per_port(self) -> int:
        return len(self.keys) + 1

    def _vec(self, s: Mapping) -> Array:
        z = jnp.asarray(0.0)
        return jnp.stack([jnp.asarray(s.get(k, z), dtype=float) for k in self.keys]
                         + [jnp.asarray(s["T"], dtype=float)])

    def _vec_scale(self, s: Mapping) -> Array:
        v = jnp.abs(self._vec(s))
        F = v[:-1]
        floor = 1e-10 * jnp.max(F) + 1e-300 if F.size else 1.0
        return jnp.concatenate([jnp.maximum(F, floor), jnp.asarray([100.0])])

    def state_spec(self) -> StateSpec:
        return StateSpec([StateVar(f"{p}.{k}", "generic", "-", "") for p in self.output_ports
                          for k in self.keys + ("T",)])

    def initial_state(self, inputs=None, params=None) -> Array:
        return self._y0

    def _blocks(self, state):
        x = (state * self._scale).reshape(len(self.output_ports), self.n_per_port)
        return x

    # ----- DynamicUnit ---------------------------------------------------------------

    def derivatives(self, t, state, inputs, params=None):
        x = self._blocks(state)
        ss = self._call(self.op, [inputs[p] for p in self.inlet_ports], len(self.output_ports))
        target = jnp.stack([self._vec(s) for s in ss])
        tau = jnp.concatenate([jnp.broadcast_to(self.tau[:, None], (len(self.output_ports), len(self.keys))),
                               self.tau_T[:, None]], axis=1)
        return ((target - x) / tau).reshape(-1) / self._scale

    def _outlets(self, state):
        x = self._blocks(state)
        out = {}
        for i, p in enumerate(self.output_ports):
            s = {k: x[i, j] for j, k in enumerate(self.keys)}
            s["T"] = x[i, -1]
            s["P"] = self._P[p]
            out[p] = s
        return out

    def outputs(self, t, state, inputs, params=None):
        return self._outlets(state)

    def partial_outputs(self, t, state, inputs, params=None):
        return self._outlets(state)

    def inventory(self, state) -> dict:
        """``{port: {F_key: mol}}``: the material each outlet holdup holds (``tau F_out``)."""
        x = self._blocks(state)
        return {p: {k: self.tau[i] * x[i, j] for j, k in enumerate(self.keys)}
                for i, p in enumerate(self.output_ports)}


__all__ = ["QuasiSteadyHoldup"]
