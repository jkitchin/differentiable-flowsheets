"""What the debugger can see at a pause.

A :class:`DebugContext` is built at a fire-site out of locals the solve
already has, so building one costs nothing the solve was not paying for.
Everything the commands print comes from here; nothing here changes the
solve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .checkpoints import Checkpoint

if TYPE_CHECKING:
    from difflow.flowsheet import Flowsheet, Unit


def _float(value: Any) -> float | None:
    """A plain float for display, or None for anything that is not one."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _finite(value: Any) -> bool:
    out = _float(value)
    return out is None or math.isfinite(out)


def stream_is_finite(stream: dict) -> bool:
    """True when every numeric entry of a stream is finite."""
    return all(_finite(v) for v in stream.values() if not isinstance(v, str))


def plain_stream(stream: dict) -> dict:
    """A stream with JAX scalars turned into floats (strings left alone)."""
    return {k: (v if isinstance(v, str) else _float(v)) for k, v in stream.items()}


@dataclass
class DebugContext:
    """The state at one checkpoint.

    Attributes:
        checkpoint: Where the solve is.
        flowsheet: The flowsheet being solved.
        phase: ``"init"`` (the pass from the feeds that guesses the
            tears), ``"solve"`` (the tear iteration), ``"sequential"`` (no
            recycle), ``"probe"`` (the error probe after convergence) or
            ``"final"`` (the last pass from the converged tears).
        iteration: Tear iteration (pass) number, when there is one.
        unit: The unit at a unit checkpoint.
        streams: Every stream known at this point (feeds, tears, outlets so far).
        inlets: The unit's inlet streams, at a unit checkpoint.
        outlets: The unit's outlet streams, after it ran.
        info: The info dict the unit returned, if any.
        error: The exception, at ``unit_error`` and a raising ``terminated``.
        x: Packed tear iterate (sorted tear names, ``[F..., T, P]`` each).
        g: The tear map at ``x``.
        x_next: The accelerated next guess, at ``after_accel``.
        residual: ``max |g - x|``.
        history: Residuals so far.
        tear_names: Tear streams, in packing order.
        method: The acceleration in use.
        event: The event this checkpoint raised, if any.
        extra: Free-form notes (the pause ``reason`` lives here).
    """

    checkpoint: Checkpoint
    flowsheet: "Flowsheet"
    phase: str = "solve"
    iteration: int | None = None
    unit: "Unit | None" = None
    streams: dict | None = None
    inlets: dict | None = None
    outlets: dict | None = None
    info: dict | None = None
    error: BaseException | None = None
    x: Any = None
    g: Any = None
    x_next: Any = None
    residual: float | None = None
    history: list[float] = field(default_factory=list)
    tear_names: list[str] = field(default_factory=list)
    method: str | None = None
    event: str | None = None
    extra: dict = field(default_factory=dict)

    # -- scalars for `break if` ------------------------------------------

    def metrics(self) -> dict[str, float]:
        out: dict[str, float] = {}
        if self.iteration is not None:
            out["iter"] = float(self.iteration)
        if self.residual is not None:
            out["residual"] = self.residual
            if self.residual > 0:
                out["log_residual"] = math.log10(self.residual)
            if len(self.history) >= 2 and self.history[-2] > 0:
                out["ratio"] = self.history[-1] / self.history[-2]
        return out

    # -- lookups ---------------------------------------------------------

    def stream(self, name: str) -> dict | None:
        """A stream by name: this unit's ports first, then everything known."""
        for pool in (self.outlets, self.inlets, self.streams, self.flowsheet.feeds):
            if pool and name in pool:
                return pool[name]
        return None

    def known_streams(self) -> list[str]:
        names = set(self.flowsheet.feeds)
        for pool in (self.streams, self.inlets, self.outlets):
            names.update(pool or {})
        return sorted(names)

    def tear_breakdown(self) -> list[dict]:
        """Every packed tear entry with x, g(x) and |g - x|, largest first.

        Says *which* stream and which field the residual is made of, which
        the scalar residual hides.
        """
        if self.x is None or self.g is None or not self.tear_names:
            return []
        fs = self.flowsheet
        fields = [f"F_{s}" for s in fs.species_order] + ["T", "P"]
        rows = []
        xs, gs = list(map(_float, self.x)), list(map(_float, self.g))
        for i, name in enumerate(sorted(self.tear_names)):
            for j, fname in enumerate(fields):
                k = i * len(fields) + j
                if k >= len(xs):
                    break
                x, g = xs[k], gs[k]
                delta = abs(g - x) if x is not None and g is not None else None
                rows.append({"stream": name, "field": fname, "x": x, "g": g,
                             "delta": delta})
        rows.sort(key=lambda r: -(r["delta"] if r["delta"] is not None
                                  and math.isfinite(r["delta"]) else math.inf))
        return rows

    # -- the wire form ---------------------------------------------------

    def to_event(self) -> dict:
        """A JSON-safe summary: what a ``pause`` event carries."""
        event: dict[str, Any] = {
            "checkpoint": self.checkpoint.value,
            "phase": self.phase,
            "iteration": self.iteration,
            "unit": self.unit.name if self.unit is not None else None,
            "residual": self.residual,
            "method": self.method,
            "trigger": self.event,
            "reason": self.extra.get("reason"),
        }
        event.update({k: v for k, v in self.metrics().items()
                      if k not in event and k != "iter"})
        if self.error is not None:
            event["error"] = f"{type(self.error).__name__}: {self.error}"
        if self.checkpoint is Checkpoint.TERMINATED:
            fs = self.flowsheet
            event["converged"] = getattr(fs, "last_solve_converged", None)
            event["iterations"] = getattr(fs, "last_solve_iterations", None)
        return event
