"""Operating signals of a dynamic refinery unit: setpoints and disturbances as functions of time.

A dynamic flowsheet's disturbances are its integrator ``args``
(:meth:`difflow.dynamic.DynamicFlowsheet.simulate`): one compiled right-hand
side serves every scenario, and ``jax.grad`` with respect to a step size or a
step time does not retrace. This module builds those ``args`` and reads them
back.

A **signal** is ``base + sum_k delta_k s((t - t_k) / ramp)``: a base value
and any number of steps, each smoothed over ``ramp`` seconds by the
smoothstep ``s(u) = 3u^2 - 2u^3`` on ``[0, 1]`` (0 before, 1 after). The ramp
keeps the right-hand side continuous, so an adaptive integrator does not have
to find the discontinuity, and it makes a step's time and size
differentiable. A step is ``(t_k, delta_k)``; a pulse is two steps.

The signals are handed to the units as a pseudo-stream (a feed of the
flowsheet named ``"signals"``): a plain dict ``{name: value}``. A unit reads
the names it knows (``"T_heater"``, ``"compressor"``, ...) and ignores the
rest.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import jax.numpy as jnp
from jax import Array

#: Default ramp of a step, s.
DEFAULT_RAMP = 60.0


def smoothstep(u: Array) -> Array:
    """``3u^2 - 2u^3`` on ``[0, 1]``, 0 below and 1 above (C1-continuous)."""
    u = jnp.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def signal_args(base: Mapping[str, float], steps: Mapping[str, Sequence[tuple[float, float]]] | None = None,
                ramp: float | Mapping[str, float] = DEFAULT_RAMP) -> dict:
    """The integrator ``args`` for a set of signals.

    Args:
        base: ``{name: value}`` at ``t = 0`` (every signal needs one).
        steps: ``{name: [(t_k, delta_k), ...]}``; a name may have none.
        ramp: Ramp time of every step (s), or ``{name: ramp}``.

    Returns:
        ``{name: {"base", "t", "delta", "ramp"}}`` of JAX arrays: differentiate
        with respect to any of them.
    """
    steps = steps or {}
    unknown = set(steps) - set(base)
    if unknown:
        raise ValueError(f"steps given for signals with no base value: {sorted(unknown)}")
    out = {}
    for name, b in base.items():
        st = list(steps.get(name, ()))
        r = ramp[name] if isinstance(ramp, Mapping) else ramp
        out[name] = {
            "base": jnp.asarray(b, dtype=float),
            "t": jnp.asarray([s[0] for s in st], dtype=float).reshape(-1),
            "delta": jnp.asarray([s[1] for s in st], dtype=float).reshape(-1),
            "ramp": jnp.asarray(r, dtype=float),
        }
    return out


def evaluate(t: Array, args: Mapping) -> dict:
    """``{name: value at t}`` of every signal in ``args`` (from :func:`signal_args`)."""
    out = {}
    for name, s in args.items():
        out[name] = s["base"] + jnp.sum(s["delta"] * smoothstep((t - s["t"]) / s["ramp"]))
    return out


def ramp_for_rate(delta: float, max_rate: float) -> float:
    """The ramp time (s) of a smoothstep move of size ``delta`` whose fastest rate is ``max_rate``.

    The smoothstep's steepest slope is 1.5 times its average, so a move of
    ``delta`` over ``r`` seconds peaks at ``1.5 delta / r``: an operator
    move limited to ``max_rate`` (units of ``delta`` per second) takes
    ``r = 1.5 |delta| / max_rate``.
    """
    return 1.5 * abs(float(delta)) / float(max_rate)


__all__ = ["DEFAULT_RAMP", "smoothstep", "signal_args", "evaluate", "ramp_for_rate"]
