"""A PI(D) feedback controller as a dynamic unit.

:class:`PIDController` closes a loop inside a
:class:`~difflow.dynamic.flowsheet.DynamicFlowsheet`: it reads a measured
stream, compares one of its values with a setpoint and writes the controller
output as a *signal*, a stream with no flows whose temperature ``T`` carries
the output ``u`` (in whatever units ``u`` has). A port that reads a
temperature reads it directly -- ``DynamicCSTR(mode="jacket", Tc="coolant")``
takes the coolant temperature from it, and ``duty="heat"`` or ``UA="ua"`` the
duty or the conductance -- and an
:class:`~difflow.dynamic.instantaneous.InstantaneousUnit` reads it as
``inputs[port]["T"]`` (a valve that turns it into a feed flow, say).

The control law, in the parallel ("ISA") form with derivative on the
measurement and back-calculation anti-windup (Astrom and Hagglund, *Advanced
PID Control*, ISA 2006, Sec. 3.5)::

    e  = s (r - y)                 s = +1 reverse acting, -1 direct acting
    v  = b + Kc e + I + D          unclipped output
    u  = clip(v, u_min, u_max)     the output the plant sees
    dI/dt = (Kc / tau_I) e + (u - v) / tau_T
    D  = -s Kc tau_D dy_f/dt,      dy_f/dt = (y - y_f) / (alpha tau_D)

``I`` (in output units) is a state, and so is the filtered measurement
``y_f`` when there is derivative action. While the output is inside its
limits ``u = v`` and the tracking term vanishes; while it is pinned the term
pulls ``I`` back toward the value the limit implies, with time constant
``tau_T``, so the integral does not wind up. ``tau_T = inf`` switches that
off (a plain clipped controller, which winds up).

Everything is JAX: the gains are read per call from the unit's ``params``,
so ``fs.simulate(params={"Kc": kc, "tau_I": ti})`` makes the closed-loop
trajectory a differentiable function of the tuning, and ``jax.grad`` of an
integrated squared error through ``simulate`` tunes the loop.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Literal

import jax.numpy as jnp
from jax import Array

from difflow.dynamic.base import DynamicUnitBase, _value_at
from difflow.dynamic.state import StateSpec, StateVar, StateVector
from difflow.streams import Stream, make_stream

_TUNING = ("Kc", "tau_I", "tau_D", "alpha", "tau_T", "bias", "u_min", "u_max",
           "setpoint")


def _is_number(v: Any) -> bool:
    """A concrete scalar we can validate now (not a callable, port or tracer)."""
    if v is None or callable(v) or isinstance(v, str):
        return False
    try:
        float(v)
    except (TypeError, ValueError):  # a tracer, an array of several values, ...
        return False
    return True


class PIDController(DynamicUnitBase):
    """PI(D) controller with output limits and back-calculation anti-windup.

    Ports: the measured stream comes in on port ``"measurement"`` (or on the
    unit's only inlet that is not a setpoint port); the output leaves on
    ``"output"`` as a signal stream whose ``T`` is the controller output.

    States: ``I``, the integral term in output units (when ``tau_I`` is
    given) and ``y_f``, the filtered measurement (when ``tau_D`` is given).

    Every tuning value (``Kc``, ``tau_I``, ``tau_D``, ``alpha``, ``tau_T``,
    ``bias``, ``u_min``, ``u_max``, ``setpoint``) may be a constant, ``f(t)``
    or ``f(t, args)`` reading the ``args`` of ``DynamicFlowsheet.simulate``
    (a setpoint ramp, gain scheduling), and is stored under the same key in
    ``params``, so ``simulate(params={"Kc": kc})`` overrides it,
    differentiably. ``simulate(params=...)`` reaches every unit of the
    flowsheet, though: two controllers in one flowsheet take distinct traced
    gains through ``args`` (``Kc=lambda t, a: a["Kc_level"]``), not through
    ``params``. ``setpoint`` may also name an inlet port, whose stream ``T``
    is then the setpoint (a cascade: the master's output is the slave's
    setpoint).

    Attributes:
        Kc: Controller gain (output units per measurement unit), positive
            for the usual negative feedback with the chosen ``action``.
        tau_I: Integral time (s); None for no integral action.
        tau_D: Derivative time (s); None for no derivative action.
        alpha: Derivative filter, as a fraction of ``tau_D`` (0.1 is usual).
        tau_T: Anti-windup tracking time constant (s); ``inf`` for none.
        bias: Output at zero error and zero integral (the manual output).
        u_min: Lower output limit (None: unbounded).
        u_max: Upper output limit (None: unbounded).
        setpoint: Setpoint, in the units of the measured value.
    """

    symbol = "PID controller"
    equations = [
        r"e = s\,(r - y),\quad v = b + K_c e + I + D,\quad u = \min(\max(v, u_\mathrm{min}), u_\mathrm{max})",
        r"\frac{dI}{dt} = \frac{K_c}{\tau_I}\,e + \frac{u - v}{\tau_T}",
        r"D = -s K_c \tau_D \frac{dy_f}{dt},\quad \frac{dy_f}{dt} = \frac{y - y_f}{\alpha\,\tau_D}",
    ]
    assumptions = [
        "Continuous-time (analog) control law; no sampling or dead time.",
        "Derivative acts on the filtered measurement, not on the error (no setpoint kick).",
        "Back-calculation anti-windup with tracking time constant tau_T.",
        "The output travels as the temperature T of a flowless signal stream.",
    ]
    references = [
        "Astrom, K.J., Hagglund, T. Advanced PID Control, ISA, 2006, Ch. 3.",
        "Seborg, Edgar, Mellichamp, Doyle. Process Dynamics and Control, 4e, Ch. 8.",
    ]
    parameter_symbols = {"Kc": "K_c", "tau_I": r"\tau_I", "tau_D": r"\tau_D",
                         "alpha": r"\alpha", "tau_T": r"\tau_T", "bias": "b",
                         "u_min": r"u_\mathrm{min}", "u_max": r"u_\mathrm{max}",
                         "setpoint": "r"}
    parameter_units = {"tau_I": "s", "tau_D": "s", "tau_T": "s"}
    numerical_method = "ODE states (integral, derivative filter) integrated with the flowsheet."

    #: Output port names.
    output_ports = ("output",)

    def __init__(
        self,
        Kc: float | Array | Callable,
        tau_I: float | Array | Callable | None = None,
        tau_D: float | Array | Callable | None = None,
        *,
        setpoint: float | Array | Callable | str,
        bias: float | Array | Callable = 0.0,
        u_min: float | Array | Callable | None = None,
        u_max: float | Array | Callable | None = None,
        tau_T: float | Array | Callable | None = None,
        alpha: float | Array | Callable = 0.1,
        action: Literal["reverse", "direct"] = "reverse",
        measure: str | Callable[[Stream], Array] = "T",
        I0: float | Array = 0.0,
        signal_P: float = 101325.0,
        name: str = "controller",
    ):
        """Build a controller.

        Args:
            Kc: Controller gain (output units per measured unit).
            tau_I: Integral time (s). None gives no integral state (P or PD).
            tau_D: Derivative time (s). None (or 0) gives no derivative state.
            setpoint: The setpoint ``r``: a constant, ``f(t)``,
                ``f(t, args)`` or an inlet port name (its stream ``T``).
            bias: Output at zero error, ``b`` (the steady output the loop
                starts from).
            u_min, u_max: Output limits; None for unbounded.
            tau_T: Anti-windup tracking time constant (s). Default
                ``sqrt(tau_I * tau_D)`` with derivative action, else
                ``tau_I`` (Astrom and Hagglund's rule of thumb); ``inf``
                disables anti-windup. Only acts while the output is limited.
            alpha: Derivative filter time as a fraction of ``tau_D``.
            action: ``"reverse"`` (output falls as the measurement rises:
                ``e = r - y``; a coolant on a reactor temperature) or
                ``"direct"`` (``e = y - r``; a heater duty on a cooling
                load reads as reverse, a cooling duty as direct).
            measure: The measured value: a key of the measured stream
                (``"T"``, ``"P"``, ``"F_A"``) or ``f(stream) -> value``.
            I0: Initial integral term (output units).
            signal_P: Pressure written on the output signal stream (unused
                by readers of ``T``; kept finite for stream utilities).
            name: Unit name.

        Raises:
            ValueError: unknown ``action``; non-positive ``tau_I``,
                ``tau_T`` or ``alpha``; negative ``tau_D``; ``u_min`` not
                below ``u_max``.
            TypeError: ``measure`` neither a key nor a callable.

        Example:
            >>> pi = PIDController(Kc=3.0, tau_I=900.0, setpoint=350.0,
            ...                    bias=340.0, u_min=300.0, u_max=351.6)
            >>> pi.state_spec().names
            ['I']
        """
        if action not in ("reverse", "direct"):
            raise ValueError(f"action must be 'reverse' or 'direct', got {action!r}")
        if not (isinstance(measure, str) or callable(measure)):
            raise TypeError("measure must be a stream key such as 'T' or a callable stream -> value")
        if _is_number(tau_I) and not float(tau_I) > 0:
            raise ValueError(f"tau_I must be positive (None for no integral action), got {tau_I}")
        if _is_number(tau_D) and float(tau_D) < 0:
            raise ValueError(f"tau_D must be >= 0 (None for no derivative action), got {tau_D}")
        if _is_number(alpha) and not float(alpha) > 0:
            raise ValueError(f"alpha (derivative filter / tau_D) must be positive, got {alpha}")
        if _is_number(tau_T) and not float(tau_T) > 0:
            raise ValueError(f"tau_T must be positive (inf for no anti-windup), got {tau_T}")
        if _is_number(u_min) and _is_number(u_max) and not float(u_min) < float(u_max):
            raise ValueError(f"u_min ({u_min}) must be below u_max ({u_max})")

        has_D = tau_D is not None and not (_is_number(tau_D) and float(tau_D) == 0.0)
        if tau_T is None and tau_I is not None:
            if has_D and _is_number(tau_I) and _is_number(tau_D):
                tau_T = math.sqrt(float(tau_I) * float(tau_D))
            else:
                tau_T = tau_I
        if tau_T is None:
            tau_T = math.inf

        def as_input(v):
            return v if v is None or callable(v) or isinstance(v, str) else jnp.asarray(v, dtype=jnp.float64)

        params = {key: as_input(val) for key, val in (
            ("Kc", Kc), ("tau_I", tau_I), ("tau_D", tau_D if has_D else None),
            ("alpha", alpha), ("tau_T", tau_T), ("bias", bias), ("u_min", u_min),
            ("u_max", u_max), ("setpoint", setpoint))}
        params.update(action=action, measure=measure, I0=jnp.asarray(I0, dtype=jnp.float64),
                      signal_P=signal_P, has_I=tau_I is not None, has_D=has_D)
        super().__init__(params, name)

    # -- structure --------------------------------------------------------

    def _build_state_spec(self) -> StateSpec:
        vars_ = []
        if self.params["has_I"]:
            vars_.append(StateVar("I", "generic", "", "Integral term (output units)"))
        if self.params["has_D"]:
            vars_.append(StateVar("y_f", "generic", "", "Filtered measurement"))
        return StateSpec(vars_)

    def _signal_ports(self) -> set[str]:
        sp = self.params.get("setpoint")
        return {sp} if isinstance(sp, str) else set()

    def _measured_stream(self, inputs: dict[str, Stream]) -> Stream:
        if "measurement" in inputs:
            return inputs["measurement"]
        candidates = [k for k in inputs if k not in self._signal_ports()]
        if len(candidates) != 1:
            raise ValueError(
                f"controller {self.name!r}: wire the measured stream to port "
                f"'measurement' (inlets seen: {sorted(inputs)})")
        return inputs[candidates[0]]

    def _measured(self, inputs: dict[str, Stream]) -> Array:
        m = self.params["measure"]
        stream = self._measured_stream(inputs)
        if callable(m):
            return jnp.asarray(m(stream), dtype=jnp.float64)
        if m not in stream:
            raise KeyError(f"controller {self.name!r}: the measured stream has no {m!r} "
                           f"(keys: {sorted(stream)})")
        return jnp.asarray(stream[m], dtype=jnp.float64)

    def _get(self, key: str, t: Array, inputs: dict[str, Stream]) -> Any:
        v = self.params.get(key)
        return None if v is None else _value_at(v, t, self._args, inputs)

    # -- the control law --------------------------------------------------

    def _terms(self, t: Array, state: StateVector, inputs: dict[str, Stream]):
        """(e, y, v, u) at time ``t``."""
        p = self.params
        s = 1.0 if p["action"] == "reverse" else -1.0
        y = self._measured(inputs)
        e = s * (self._get("setpoint", t, inputs) - y)
        Kc = self._get("Kc", t, inputs)
        v = self._get("bias", t, inputs) + Kc * e
        if p["has_I"]:
            v = v + state["I"]
        if p["has_D"]:
            tau_f = self._get("alpha", t, inputs) * self._get("tau_D", t, inputs)
            v = v - s * Kc * self._get("tau_D", t, inputs) * (y - state["y_f"]) / tau_f
        u_min, u_max = self._get("u_min", t, inputs), self._get("u_max", t, inputs)
        u = v
        if u_min is not None:
            u = jnp.maximum(u, u_min)
        if u_max is not None:
            u = jnp.minimum(u, u_max)
        return e, y, v, u

    def control_output(self, t: Array, state: Array, inputs: dict[str, Stream],
                       params: dict | None = None, args: Any = None) -> Array:
        """The (limited) controller output ``u`` for a state and inputs."""
        with self._call_context(params, args):
            return self._terms(t, StateVector(state, self.state_spec()), inputs)[3]

    def _derivatives(self, t: Array, state: StateVector, inputs: dict[str, Stream]) -> Array:
        p = self.params
        e, y, v, u = self._terms(t, state, inputs)
        parts = [jnp.zeros(0)]
        if p["has_I"]:
            dI = self._get("Kc", t, inputs) / self._get("tau_I", t, inputs) * e
            dI = dI + (u - v) / self._get("tau_T", t, inputs)
            parts.append(jnp.atleast_1d(dI))
        if p["has_D"]:
            tau_f = self._get("alpha", t, inputs) * self._get("tau_D", t, inputs)
            parts.append(jnp.atleast_1d((y - state["y_f"]) / tau_f))
        return jnp.concatenate(parts)

    def _outputs(self, t: Array, state: StateVector, inputs: dict[str, Stream]) -> dict[str, Stream]:
        u = self._terms(t, state, inputs)[3]
        return {"output": make_stream({}, T=u, P=self.params["signal_P"])}

    def initial_state(self, inputs: dict[str, Stream], params: dict | None = None) -> Array:
        """``I = I0`` and ``y_f`` = the measurement now (no derivative kick)."""
        with self._call_context(params, None):
            parts = [jnp.zeros(0)]
            if self.params["has_I"]:
                parts.append(jnp.atleast_1d(self.params["I0"]))
            if self.params["has_D"]:
                parts.append(jnp.atleast_1d(self._measured(inputs)))
            return jnp.concatenate(parts)

    def __repr__(self) -> str:
        return f"PIDController(name='{self.name}', states={self.state_spec().names})"
