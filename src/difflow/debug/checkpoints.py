"""Where a flowsheet solve can pause, and the events a breakpoint can wait for.

Every checkpoint here has a fire-site in :mod:`difflow.flowsheet`, so the
JSON ``hello`` (which advertises all of them) never promises a pause that
cannot happen. A new checkpoint is added together with its fire-site.
"""

from __future__ import annotations

from enum import Enum


class Checkpoint(str, Enum):
    """A pause point in a sequential-modular solve, in the order they fire."""

    SOLVE_START = "solve_start"   # tears guessed, before the first pass
    BEFORE_UNIT = "before_unit"   # a unit's inlets are known, it has not run
    AFTER_UNIT = "after_unit"     # a unit ran; outlets and info are fresh
    UNIT_ERROR = "unit_error"     # a unit raised; the solve re-raises after
    ITERATION = "iteration"       # tear map evaluated, residual |g(x) - x| measured
    AFTER_ACCEL = "after_accel"   # the accelerated next tear guess is computed
    TERMINATED = "terminated"     # the solve is about to return (or raise)

    def __str__(self) -> str:
        return self.value


#: Friendly names accepted by ``stop-at``.
ALIASES: dict[str, Checkpoint] = {
    "start": Checkpoint.SOLVE_START,
    "before": Checkpoint.BEFORE_UNIT,
    "unit": Checkpoint.AFTER_UNIT,
    "after": Checkpoint.AFTER_UNIT,
    "error": Checkpoint.UNIT_ERROR,
    "iter": Checkpoint.ITERATION,
    "accel": Checkpoint.AFTER_ACCEL,
    "end": Checkpoint.TERMINATED,
    "done": Checkpoint.TERMINATED,
}


def resolve_checkpoint(name: str) -> Checkpoint:
    """A checkpoint by name or alias; ``ValueError`` if it is neither."""
    key = name.strip().lower()
    return ALIASES[key] if key in ALIASES else Checkpoint(key)


#: Events a ``break on`` can wait for, and what raises each one.
EVENTS: dict[str, str] = {
    "nonfinite": "a unit turned finite inlets into a NaN/inf outlet, "
                 "or the tear residual is not finite",
    "inner_unconverged": "a unit's own solve reported converged=False",
    "residual_rise": "the tear residual went up from one iteration to the next",
    "unit_error": "a unit raised",
}

#: Scalars a ``break if`` condition can test.
METRICS: dict[str, str] = {
    "iter": "tear iteration (pass) number",
    "residual": "max |g(x) - x| over the tear streams",
    "ratio": "residual / previous residual (the observed contraction)",
    "log_residual": "log10 of the residual",
}
