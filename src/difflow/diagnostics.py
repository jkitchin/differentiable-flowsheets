"""Why a flowsheet solve failed, or why an answer that "worked" is suspect.

What difflow knows about failing solves used to be prose: the checklist in
``docs/convergence.md`` and the symptom cards the editor's assistant
quotes. Here it is data, so the editor, the agent tools and a script draw
on one source:

* :data:`SYMPTOMS` --- error and diagnostic patterns, each with the
  explanation and the remedies that address it;
* :data:`REMEDIES` --- solver settings to try, each marked ``numerics``
  (changes how the fixed point is reached, not which one) or ``problem``
  (can change the answer, so it is proposed rather than applied);
* :func:`classify_history` --- what the tear residual's history says about
  the loop: converged, creeping, oscillating, diverging, stalled;
* :func:`solve_findings` --- findings from a flowsheet's last solve: the
  verdict, the gap between step and error, clipping, inner unit solves that
  failed, and tear-set problems.

Example:
    >>> fs.solve(on_nonconvergence="ignore")
    >>> for finding in solve_findings(fs):
    ...     print(finding.severity, finding.detail)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

#: severities, most serious first
SEVERITIES = ("error", "warning", "info")


@dataclass(frozen=True)
class Finding:
    """One thing a diagnosis found.

    Attributes:
        kind: A short slug, e.g. ``"not-converged"`` or ``"inner-solve"``.
        severity: ``"error"`` (the answer is wrong or missing),
            ``"warning"`` (it may be) or ``"info"``.
        detail: One or two sentences saying what and why.
        unit: The unit concerned, if any.
        value: The number behind the finding, if any.
        remedies: Names in :data:`REMEDIES` that address it.
    """

    kind: str
    severity: str
    detail: str
    unit: str | None = None
    value: float | None = None
    remedies: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        out = asdict(self)
        out["remedies"] = list(self.remedies)
        return out


@dataclass(frozen=True)
class Remedy:
    """A solver setting to try.

    Attributes:
        name: Key in :data:`REMEDIES`.
        options: Keyword arguments for :meth:`Flowsheet.solve`.
        kind: ``"numerics"`` changes how the fixed point is reached and not
            which one, so applying it cannot change a correct answer;
            ``"problem"`` changes the map being iterated or the starting
            point, and on a flowsheet with more than one steady state can
            land on a different one.
        why: When it helps.
    """

    name: str
    options: dict
    kind: str
    why: str


#: The remedy ladder, in the order a search tries it: cheap changes of
#: method first, more iterations (the costliest) last among the numerics.
REMEDIES: dict[str, Remedy] = {r.name: r for r in (
    Remedy("anderson", {"acceleration": "anderson"}, "numerics",
           "Anderson mixing; the default, and the best general choice."),
    Remedy("wegstein", {"acceleration": "wegstein"}, "numerics",
           "Wegstein extrapolates each tear variable separately; it handles "
           "phase-coupled flash loops that Anderson misses."),
    Remedy("damped", {"acceleration": "none", "damping": 0.3, "max_iter": 2000},
           "numerics",
           "Damped substitution. A residual that oscillates or grows means a "
           "loop gain at or below -1, which damping turns into a contraction "
           "and more iterations never fix."),
    Remedy("error_basis", {"tol_basis": "error"}, "numerics",
           "Test the measured error rather than the step: at gain g the error "
           "is about 1/(1-g) times the step, so a high-gain loop that "
           "'converged' can still be far from its fixed point."),
    Remedy("more_iterations", {"max_iter": 500}, "numerics",
           "A loop that is converging but slowly (gain near 1) only needs "
           "more iterations."),
    Remedy("signed_tears", {"clip_negative_flows": False}, "problem",
           "Stop clipping negative tear flows. Needed when a tear flow is "
           "genuinely negative (signed flows); it changes the iterated map."),
    Remedy("cold_start", {"use_initialization": False}, "problem",
           "Start the tears from the default guess instead of a pass from the "
           "feeds; a different start can reach a different steady state."),
)}

#: The settings a search starts from when the flowsheet has none of its own.
NUMERICS = tuple(n for n, r in REMEDIES.items() if r.kind == "numerics")


@dataclass(frozen=True)
class Symptom:
    """An error or diagnostic pattern and what it means.

    A symptom matches when any of its ``triggers`` occurs in the text it is
    checked against (an error message, a warning, a finding).
    """

    title: str
    triggers: tuple[str, ...]
    text: str
    remedies: tuple[str, ...] = ()

    def matches(self, text: str) -> bool:
        return any(t in text for t in self.triggers)


SYMPTOMS: tuple[Symptom, ...] = (
    Symptom(
        "The recycle did not converge",
        ("converged", "max_iter", "residual"),
        "A sequential-modular solve tears the recycle streams and "
        "iterates. `converged=False` means it stopped at `max_iter` "
        "with the residual still above `tol`; the numbers it returns "
        "look like an answer and are not one. What helps, in order: "
        "raise `max_iter`; switch `acceleration` to 'anderson' or "
        "'wegstein' (direct substitution converges linearly and a loop "
        "gain near 1 makes that arbitrarily slow); damp the tear map "
        "with `acceleration='none', damping=0.3`; give the tear stream a "
        "better initial guess with `tear_initial`. A residual that "
        "*rises* is a loop gain above 1, which damping fixes and "
        "iterations do not.",
        ("more_iterations", "anderson", "wegstein", "damped"),
    ),
    Symptom(
        "TracerArrayConversionError",
        ("TracerArrayConversionError", "TracerBoolConversionError"),
        "A JAX tracer reached Python control flow -- `if x > 0:`, "
        "`float(x)`, `int(x)`, or a numpy call on a traced array. Use "
        "`jnp.where`, `jax.lax.cond`, `jax.lax.switch` or "
        "`jax.lax.fori_loop` instead. Inside a flowsheet this is almost "
        "always a `rate_fn` or a property correlation branching on a "
        "value rather than selecting with `jnp.where`.",
    ),
    Symptom(
        "ConcretizationError",
        ("ConcretizationError",),
        "Something needed a concrete value from an abstract one -- a "
        "shape, a loop bound, or a Python `bool`. Shapes must be static: "
        "they cannot depend on traced values. If the value is only a "
        "diagnostic, read it outside the trace.",
    ),
    Symptom(
        "NaN in the result or the gradient",
        ("nan", "NaN"),
        "The usual sources are `log(0)`, `sqrt` of a negative, `0/0`, "
        "and `x**y` at `x=0`. A forward pass can be finite while the "
        "gradient is not -- `sqrt(x)` at `x=0` is 0 with an infinite "
        "derivative. Add an epsilon, clip the argument, or use a safe "
        "form. `jax.config.update('jax_debug_nans', True)` stops at the "
        "first one.",
    ),
    Symptom(
        "Singular or ill-conditioned solve",
        ("singular", "LinAlgError", "condition", "rank"),
        "An equation-oriented or Newton solve hit a Jacobian it cannot "
        "invert. Usually a variable that nothing determines (an unset "
        "spec) or two rows saying the same thing (a redundant spec). "
        "Count equations against unknowns before reaching for a "
        "different solver.",
    ),
)


def matching_symptoms(text: str) -> list[Symptom]:
    """The symptoms whose triggers occur in ``text``."""
    return [s for s in SYMPTOMS if s.matches(text)]


# ---------------------------------------------------------------------
# Reading the iteration
# ---------------------------------------------------------------------


def classify_history(history: list[float] | None, converged: bool | None,
                     gain: float | None = None) -> str:
    """What a tear residual history says about the loop.

    Returns one of ``"converged"``, ``"no-history"`` (the fixed-point path
    keeps none), ``"nan"``, ``"diverging"`` (the residual grew overall),
    ``"oscillating"`` (it alternates up and down), ``"creeping"`` (it falls,
    but too slowly to reach ``tol`` in time: a gain near 1), or
    ``"stalled"`` (it stopped falling).
    """
    if converged:
        return "converged"
    if not history:
        return "no-history"
    if any(not math.isfinite(r) for r in history):
        return "nan"
    tail = history[-min(len(history), 20):]
    if len(tail) < 3:
        return "stalled"
    if tail[-1] > 2 * tail[0]:
        return "diverging"
    steps = [b - a for a, b in zip(tail, tail[1:])]
    flips = sum(1 for a, b in zip(steps, steps[1:]) if a * b < 0)
    if flips > len(steps) // 2:
        return "oscillating"
    if tail[-1] < 0.9 * tail[0]:
        return "creeping"
    return "stalled"


#: what each regime suggests, as remedy names
REGIME_REMEDIES = {
    "diverging": ("damped", "wegstein"),
    "oscillating": ("damped", "wegstein"),
    "creeping": ("more_iterations", "anderson", "wegstein"),
    "stalled": ("wegstein", "damped", "cold_start", "signed_tears"),
    "nan": ("damped", "cold_start"),
    "no-history": ("anderson", "wegstein", "more_iterations"),
}


def solve_findings(fs) -> list[Finding]:
    """Findings from a flowsheet's last solve, most serious first.

    Reads the ``last_solve_*`` record, the per-unit info and the tear
    analysis; it does not solve. Call it after
    ``fs.solve(on_nonconvergence="ignore")``.
    """
    found: list[Finding] = []
    converged = fs.last_solve_converged
    history = getattr(fs, "last_solve_history", None)
    gain = fs.last_solve_gain
    if converged is False:
        regime = classify_history(history, converged, gain)
        last = fs.last_solve_residual
        found.append(Finding(
            "not-converged", "error",
            f"the recycle did not converge in {fs.last_solve_iterations} "
            f"iterations ({fs.last_solve_method}); the residual is "
            f"{last:.3g} against tol {fs.last_solve_tol:g}, and the history "
            f"reads as {regime}." if last is not None else
            f"the recycle did not converge ({fs.last_solve_method}).",
            value=last, remedies=REGIME_REMEDIES.get(regime, ())))
    error = fs.last_solve_error_estimate
    tol = fs.last_solve_tol
    if converged and error is not None and tol and error > tol:
        found.append(Finding(
            "error-above-tol", "warning",
            f"converged on the step test, but at loop gain "
            f"{gain if gain is None else round(gain, 4)} the estimated error "
            f"is {error:.3g}, above tol {tol:g}.",
            value=error, remedies=("error_basis",)))
    if gain is not None and abs(gain) >= 0.95:
        found.append(Finding(
            "high-gain", "info",
            f"the loop gain is {gain:.4f}: errors are amplified about "
            f"{1 / max(1 - gain, 1e-12):.0f}x relative to the step.",
            value=gain, remedies=("error_basis", "more_iterations")))
    if getattr(fs, "last_solve_clip_active", 0):
        found.append(Finding(
            "clipping", "warning",
            f"clip_negative_flows moved the iterate on "
            f"{fs.last_solve_clip_active} iterations. If a tear flow is "
            "genuinely negative (signed flows) the clip is fighting the "
            "answer.", value=float(fs.last_solve_clip_active),
            remedies=("signed_tears",)))
    for name, info in (getattr(fs, "last_solve_unit_info", None) or {}).items():
        ok = info.get("converged") if isinstance(info, dict) else None
        if ok is not None and not bool(ok):
            residual = info.get("balance_residual")
            found.append(Finding(
                "inner-solve", "error",
                f"{name}'s own solve did not close its balance"
                + (f" (relative residual {float(residual):.3g})"
                   if residual is not None else "")
                + "; its outlet is not a solution of the unit.",
                unit=name, value=None if residual is None else float(residual)))
    try:
        tears = fs.tear_analysis()
    except Exception:  # noqa: BLE001 -- a graph it cannot read is not a finding
        tears = None
    if tears is not None:
        if tears.uncovered:
            found.append(Finding(
                "uncovered-cycle", "error",
                f"{len(tears.uncovered)} cycle(s) are not broken by any "
                "declared tear, so the calculation order cannot be met.",
                remedies=()))
        if tears.missing_inputs or tears.out_of_order:
            found.append(Finding(
                "calculation-order", "warning",
                "units run before their inputs exist: "
                + ", ".join(map(str, list(tears.missing_inputs)
                                + list(tears.out_of_order)))[:300]))
    order = {s: i for i, s in enumerate(SEVERITIES)}
    return sorted(found, key=lambda f: order[f.severity])


@dataclass
class Trial:
    """One remedy tried by a search: what was set and what came of it."""

    remedy: str
    options: dict
    kind: str
    converged: bool | None = None
    correct: bool | None = None
    iterations: int | None = None
    residual: float | None = None
    gain: float | None = None
    seconds: float | None = None
    error: str | None = None
    #: what makes the answer wrong (NaN, mass not conserved, a unit's own
    #: solve not closed, an error well above tol)
    problems: list[str] = field(default_factory=list)
    #: what qualifies a correct answer (negative flows, unread feeds)
    caveats: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.converged and self.correct)
