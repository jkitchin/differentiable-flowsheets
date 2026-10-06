"""Diagnosing a flowsheet and searching for settings that converge it.

Two calls, one that explains and one that acts:

* :func:`diagnose` checks what can be checked before solving (pending
  units, unfed inlets, the tear set), solves with the flowsheet's own
  settings, and turns the record of that solve into
  :class:`~difflow.diagnostics.Finding` objects with the remedies that
  address each.
* :func:`converge` tries the remedies on the live flowsheet and judges each
  attempt "converged AND correct": converged, a clean audit (mass balance,
  negative flows, NaN) and every unit's inner solve closed. Remedies only
  change solver settings, never the model, so nothing needs copying; the
  stored settings are untouched unless ``apply`` is set, and then only a
  ``numerics`` remedy is applied (one that changes how the fixed point is
  reached, not which one). When two passing attempts disagree on the
  products, that is reported as more than one steady state.
"""

from __future__ import annotations

import time
import warnings as _warnings

from difflow.diagnostics import (
    NUMERICS,
    REMEDIES,
    Finding,
    Remedy,
    Trial,
    classify_history,
    matching_symptoms,
    solve_findings,
)

#: relative difference in a product flow above which two passing attempts
#: are reported as different steady states
STEADY_STATE_RTOL = 1e-4


def _preflight(session) -> list[Finding]:
    found = []
    for name, entry in sorted(session.pending.items()):
        found.append(Finding(
            "pending-unit", "error",
            f"{name} cannot be built yet: it needs "
            f"{', '.join(entry['needs'])}. {entry.get('hint') or ''}".strip(),
            unit=name))
    unfed = session.feeds().get("unfed") or []
    if unfed:
        found.append(Finding(
            "unfed-inlet", "error",
            f"nothing feeds {', '.join(unfed)}: give each a feed or connect "
            "a unit's outlet to it."))
    return found


def _symptom_findings(text: str) -> list[Finding]:
    return [Finding("symptom", "info", f"{s.title}: {s.text}",
                    remedies=s.remedies) for s in matching_symptoms(text)]


def diagnose(session) -> dict:
    """Findings about a session's flowsheet, most serious first."""
    if session.flowsheet is None:
        return {"ok": False, "error": "no flowsheet loaded"}
    found = _preflight(session)
    if any(f.severity == "error" for f in found):
        return {"ok": True, "solved": False, "findings": [f.to_dict() for f in found],
                "remedies": {}}
    answer = session.solve()
    fs = session.flowsheet
    if not answer.get("ok"):
        error = answer.get("error") or ""
        found.append(Finding("solve-raised", "error", error))
        found += _symptom_findings(error)
    else:
        found += solve_findings(fs)
        for text in answer.get("audit", {}).get("warnings", []):
            if "own solve" in text:
                continue          # already a finding from solve_findings
            found.append(Finding("audit", "error" if "mass" in text or "non-finite"
                                 in text else "warning", text))
        for w in answer.get("warnings", []):
            found.append(Finding("warning", "warning",
                                 f"{w['category']}: {w['message']}"))
    names = sorted({r for f in found for r in f.remedies})
    return {
        "ok": True,
        "solved": bool(answer.get("ok")),
        "converged": answer.get("converged"),
        "regime": classify_history(getattr(fs, "last_solve_history", None),
                                   answer.get("converged"), answer.get("gain")),
        "findings": [f.to_dict() for f in found],
        "remedies": {n: {"options": REMEDIES[n].options, "kind": REMEDIES[n].kind,
                         "why": REMEDIES[n].why} for n in names},
    }


def _products(fs, streams) -> dict:
    read = {n for u in fs.units for n in u.inlet_names}
    made = {n for u in fs.units for n in u.outlet_names}
    return {n: {k: float(v) for k, v in streams[n].items()
                if k.startswith("F_")}
            for n in sorted(made - read - set(fs.recycles)) if n in streams}


def _differ(a: dict, b: dict) -> float:
    worst = 0.0
    for name, flows in a.items():
        for key, value in flows.items():
            other = b.get(name, {}).get(key, value)
            scale = max(abs(value), abs(other), 1e-12)
            worst = max(worst, abs(value - other) / scale)
    return worst


def _soft(text: str) -> bool:
    """An audit warning that qualifies an answer rather than refuting it."""
    return text.startswith("negative flows") or "no unit reads" in text


def _attempt(session, remedy, base: dict) -> tuple[Trial, dict | None]:
    options = {**base, **remedy.options}
    trial = Trial(remedy.name, remedy.options, remedy.kind)
    fs = session.flowsheet
    start = time.perf_counter()
    try:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            streams = fs.solve(on_nonconvergence="ignore", **options)
    except Exception as exc:  # noqa: BLE001 -- a raise is an outcome
        trial.seconds = time.perf_counter() - start
        trial.error = f"{type(exc).__name__}: {exc}"
        return trial, None
    trial.seconds = round(time.perf_counter() - start, 3)
    trial.converged = fs.last_solve_converged
    trial.iterations = fs.last_solve_iterations
    trial.residual = fs.last_solve_residual
    trial.gain = fs.last_solve_gain
    audit = session._audit(streams)
    # Negative flows and unread feeds are caveats, not failures: a signed
    # tear (a gas network's flows) is negative in the right answer. What
    # makes an answer wrong is NaN, material not conserved, a unit whose
    # own solve did not close, or an error well above tol.
    for text in audit["warnings"]:
        (trial.caveats if _soft(text) else trial.problems).append(text)
    error = fs.last_solve_error_estimate
    if trial.converged and error is not None and fs.last_solve_tol \
            and error > 10 * fs.last_solve_tol:
        trial.problems.append(f"estimated error {error:.3g} is above tol")
    trial.correct = not trial.problems
    return trial, (_products(fs, streams) if trial.passed else None)


def converge(session, *, apply: bool = False, budget: int = 8) -> dict:
    """Try the remedy ladder; report what passed and, with ``apply``,
    keep the first numerics remedy that did."""
    if session.flowsheet is None:
        return {"ok": False, "error": "no flowsheet loaded"}
    blocking = _preflight(session)
    if blocking:
        return {"ok": False, "error": "fix these first: "
                + "; ".join(f.detail for f in blocking)}
    from difflow.gui.session import SOLVER_DEFAULTS

    base = session._solve_kw()
    effective = {**SOLVER_DEFAULTS, **base}
    # The flowsheet's own settings first, then the ladder in order; a
    # remedy that would change nothing (its options are already in
    # effect) is not run a second time.
    ladder = [n for n in REMEDIES
              if any(effective.get(k) != v for k, v in REMEDIES[n].options.items())]
    trials, passed = [], []
    with session._lock:
        current = Remedy("current", {}, "numerics", "the flowsheet's own settings")
        for remedy in [current] + [REMEDIES[n] for n in ladder][:max(budget - 1, 0)]:
            trial, products = _attempt(session, remedy, base)
            trials.append(trial)
            if products is not None:
                passed.append((trial, products))
            if len(passed) >= 2:
                break
        session.streams = None        # the trials' streams are not the model's
    findings = []
    if len(passed) >= 2:
        spread = _differ(passed[0][1], passed[1][1])
        if spread > STEADY_STATE_RTOL:
            findings.append(Finding(
                "multiple-steady-states", "warning",
                f"'{passed[0][0].remedy}' and '{passed[1][0].remedy}' both "
                f"converged correctly but their products differ by up to "
                f"{spread:.2%}: the flowsheet has more than one steady state, "
                "and which one a solve reaches depends on how it gets there.",
                value=spread).to_dict())
    best = passed[0][0] if passed else None
    applied = None
    if best is not None and apply and best.remedy != "current":
        if best.kind == "numerics":
            result = session.set_solver_options(dict(best.options))
            if result.get("ok"):
                applied = best.remedy
        else:
            findings.append(Finding(
                "not-applied", "info",
                f"'{best.remedy}' changes the problem, not just the numerics, "
                "so it is proposed rather than applied: set "
                f"{best.options} with set_solver_options if it is right.").to_dict())
    return {
        "ok": True,
        "passed": best is not None,
        "best": None if best is None else {
            "remedy": best.remedy, "options": best.options, "kind": best.kind},
        "applied": applied,
        # Undo is recorded only for a flowsheet the session can serialize;
        # one built in Python from unregistered pieces cannot be undone.
        "undoable": bool(applied) and session.history()["undo"],
        "trials": [{**t.__dict__, "passed": t.passed} for t in trials],
        "findings": findings,
        "note": "a trial passes only when it converged AND its audit is clean "
                "(mass balance, negative flows, NaN, inner unit solves)",
    }


def tear_analysis(session) -> dict:
    """The flowsheet's cycles and tear sets, declared and suggested."""
    if session.flowsheet is None:
        return {"ok": False, "error": "no flowsheet loaded"}
    analysis = session.flowsheet.tear_analysis()
    return {
        "ok": True,
        "cycles": [list(c) for c in analysis.cycles],
        "declared": list(analysis.declared),
        "heuristic": list(analysis.heuristic),
        "minimum": list(analysis.minimum),
        "uncovered": [list(c) for c in analysis.uncovered],
        "missing_inputs": [str(m) for m in analysis.missing_inputs],
        "out_of_order": [str(m) for m in analysis.out_of_order],
        "summary": analysis.summary(),
    }


def trace_solve(session) -> dict:
    """Solve with the stored settings and return the iteration history."""
    answer = session.solve()
    fs = session.flowsheet
    if not answer.get("ok"):
        return answer
    history = getattr(fs, "last_solve_history", None)
    return {
        "ok": True,
        "converged": answer.get("converged"),
        "method": answer.get("method"),
        "iterations": answer.get("iterations"),
        "history": history,
        "regime": classify_history(history, answer.get("converged"),
                                   answer.get("gain")),
        "gain": answer.get("gain"),
        "note": None if history is not None else
            "acceleration 'none' runs optimistix's fixed point, which keeps "
            "no history; set acceleration to 'anderson' or 'wegstein' to trace",
    }


def unit_info(session, name: str | None = None) -> dict:
    """What each unit reported about itself on the last solve."""
    fs = session.flowsheet
    if fs is None:
        return {"ok": False, "error": "no flowsheet loaded"}
    info = getattr(fs, "last_solve_unit_info", None) or {}
    if session.streams is None:
        return {"ok": False, "error": "not solved since the last change; call solve"}
    if name is not None:
        if name not in info:
            return {"ok": False, "error": f"{name!r} reported nothing on the last "
                    f"solve (units that did: {', '.join(info) or 'none'})"}
        return {"ok": True, "units": {name: info[name]}}
    return {"ok": True, "units": info}


__all__ = ["converge", "diagnose", "tear_analysis", "trace_solve", "unit_info",
           "NUMERICS"]
