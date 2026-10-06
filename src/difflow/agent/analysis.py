"""Analysis of a solved flowsheet, driven by expressions.

Every function here takes a :class:`~difflow.gui.session.FlowsheetSession`
and expressions in the language of :mod:`difflow.agent.expressions`, and
solves with the session's own solver settings (not ``Flowsheet.solve``'s
defaults, which ``make_objective_fn`` would use). Levers are the keys
``Flowsheet._apply_params`` takes: ``"<unit>.<param>"`` and
``"feed:<stream>.<field>"``; :func:`levers` lists them.

Derivatives come from JAX through the converged solve (implicit
differentiation), not from finite differences, and every result carries
the convergence verdict of the point it was taken at: a derivative of an
unconverged solve is a derivative of the wrong function.
"""

from __future__ import annotations

import math
import time
import warnings as _warnings

from difflow.agent.expressions import Context, ExpressionError, Quantities, evaluate


def _kw(session) -> dict:
    return {**session._solve_kw(), "on_nonconvergence": "ignore"}


def _numeric(streams: dict) -> dict:
    return {n: {k: v for k, v in s.items() if not isinstance(v, str)}
            for n, s in streams.items()}


def _function(session, text: str):
    """``f(params) -> value`` for one expression, through a solve."""
    fs = session.flowsheet
    quantities = Quantities(fs).all()
    kw = _kw(session)

    def f(params):
        updated = fs._apply_params(params) if params else fs
        streams = _numeric(updated.solve(**kw))
        return evaluate(text, Context(streams, updated, quantities))

    return f


def _known_levers(session) -> dict[str, dict]:
    from difflow.gui.sensitivity import levers

    return {item["key"]: item for item in levers(session.flowsheet)}


def _check_levers(session, keys) -> dict[str, dict]:
    known = _known_levers(session)
    unknown = [k for k in keys if k not in known]
    if unknown:
        raise ExpressionError(
            f"not levers of this flowsheet: {', '.join(unknown)} (list them "
            "with levers; a lever is '<unit>.<param>' or "
            "'feed:<stream>.<field>')")
    return {k: known[k] for k in keys}


def _verdict(session) -> dict:
    """Convergence of one concrete solve at the current point."""
    fs = session.flowsheet
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        streams = fs.solve(**_kw(session))
    session.streams = streams
    return {"converged": fs.last_solve_converged,
            "residual": fs.last_solve_residual}


def _float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


# ---------------------------------------------------------------------


def levers(session) -> dict:
    """Every scalar parameter a derivative or an optimizer can move."""
    from difflow.gui.sensitivity import levers as find

    return {"ok": True, "levers": find(session.flowsheet)}


def define_quantity(session, name: str, expression: str) -> dict:
    Quantities(session.flowsheet).define(name, expression)
    out = {"ok": True, "name": name, "expression": expression}
    if session.streams is not None:
        try:
            out["value"] = _float(evaluate_expressions(session, [expression])
                                  ["values"][expression])
        except ExpressionError as exc:
            out["warning"] = str(exc)
    return out


def remove_quantity(session, name: str) -> dict:
    if not Quantities(session.flowsheet).remove(name):
        return {"ok": False, "error": f"no quantity {name!r}"}
    return {"ok": True, "quantities": Quantities(session.flowsheet).all()}


def list_quantities(session) -> dict:
    return {"ok": True, "quantities": Quantities(session.flowsheet).all()}


def evaluate_expressions(session, expressions: list[str]) -> dict:
    """Values of expressions at the last solve (solving first if needed)."""
    verdict = None
    if session.streams is None:
        verdict = _verdict(session)
    context = Context(_numeric(session.streams), session.flowsheet,
                      Quantities(session.flowsheet).all())
    values = {e: _float(evaluate(e, context)) for e in expressions}
    return {"ok": True, "values": values,
            **({} if verdict is None else verdict)}


def sensitivity(session, of: str, wrt: list[str] | None = None) -> dict:
    """d(expression)/d(lever) for every lever, in one reverse pass."""
    import jax

    chosen = _check_levers(session, wrt) if wrt else _known_levers(session)
    if not chosen:
        return {"ok": False, "error": "this flowsheet has no scalar levers"}
    u0 = {k: item["value"] for k, item in chosen.items()}
    y0, grad = jax.value_and_grad(_function(session, of))(u0)
    y0 = float(y0)
    rows = []
    for key, item in chosen.items():
        d = float(grad[key])
        rel = d * item["value"] / y0 if item["value"] and y0 else None
        rows.append({"lever": key, "value": item["value"], "units": item["units"],
                     "derivative": d, "elasticity": rel})
    rows.sort(key=lambda r: abs(r["elasticity"]) if r["elasticity"] is not None
              else -1.0, reverse=True)
    return {"ok": True, "of": of, "value": y0, "levers": rows,
            "note": "elasticity is d ln y / d ln u, comparable across units",
            **_verdict(session)}


def sweep(session, lever: str, outputs: list[str], values: list[float] | None = None,
          lo: float | None = None, hi: float | None = None, n: int = 11) -> dict:
    """Outputs at each value of one lever, with the verdict at each point."""
    _check_levers(session, [lever])
    if values is None:
        if lo is None or hi is None or n < 2:
            return {"ok": False, "error": "give values, or lo, hi and n >= 2"}
        values = [lo + (hi - lo) * i / (n - 1) for i in range(int(n))]
    fs = session.flowsheet
    quantities = Quantities(fs).all()
    kw = _kw(session)
    rows = []
    for value in values:
        row = {"value": float(value)}
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                updated = fs._apply_params({lever: float(value)})
                streams = _numeric(updated.solve(**kw))
            context = Context(streams, updated, quantities)
            row["converged"] = updated.last_solve_converged
            for e in outputs:
                row[e] = _float(evaluate(e, context))
        except ExpressionError:
            raise
        except Exception as exc:  # noqa: BLE001 -- a point that raises is an outcome
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return {"ok": True, "lever": lever, "outputs": outputs, "points": rows}


def optimize(session, objective: str, levers: dict[str, list[float]],
             constraints: list[dict] | None = None, maximize: bool = False,
             max_iter: int = 50, apply: bool = False) -> dict:
    """Minimize (or maximize) an expression over bounded levers.

    SLSQP from scipy with exact gradients from JAX through the solve.
    Constraints are ``{"expression": ..., "lb": ..., "ub": ...}``.
    """
    import jax
    import numpy as np
    from scipy.optimize import minimize

    chosen = _check_levers(session, list(levers))
    keys = list(chosen)
    bounds = []
    for k in keys:
        lo, hi = (list(levers[k]) + [None, None])[:2]
        bounds.append((lo, hi))
    sign = -1.0 if maximize else 1.0
    f = _function(session, objective)
    value_grad = jax.value_and_grad(lambda p: sign * f(p))
    calls = {"n": 0}

    def as_params(x):
        return {k: float(v) for k, v in zip(keys, x)}

    def fun(x):
        calls["n"] += 1
        v, g = value_grad(as_params(x))
        return float(v), np.array([float(g[k]) for k in keys])

    cons = []
    for c in constraints or []:
        g = _function(session, c["expression"])
        gg = jax.value_and_grad(g)
        for bound, s in (("lb", 1.0), ("ub", -1.0)):
            if c.get(bound) is None:
                continue
            b = float(c[bound])

            def cf(x, gg=gg, b=b, s=s):
                v, _ = gg(as_params(x))
                return s * (float(v) - b)

            def cj(x, gg=gg, s=s):
                _, d = gg(as_params(x))
                return s * np.array([float(d[k]) for k in keys])

            cons.append({"type": "ineq", "fun": cf, "jac": cj})
    x0 = np.array([chosen[k]["value"] for k in keys], dtype=float)
    start = time.perf_counter()
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        result = minimize(fun, x0, jac=True, bounds=bounds, constraints=cons,
                          method="SLSQP", options={"maxiter": int(max_iter)})
    best = as_params(result.x)
    out = {
        "ok": True, "success": bool(result.success), "message": str(result.message),
        "objective": objective, "maximize": maximize,
        "value": sign * float(result.fun), "start": as_params(x0), "levers": best,
        "iterations": int(getattr(result, "nit", 0)), "evaluations": calls["n"],
        "seconds": round(time.perf_counter() - start, 2),
        "constraints": [
            {**c, "value": _float(_function(session, c["expression"])(best))}
            for c in constraints or []],
    }
    if apply and result.success:
        out["applied"] = _apply_levers(session, best)
    return out


def _apply_levers(session, values: dict[str, float]) -> list[str]:
    """Write lever values into the model through the session's edits."""
    done = []
    for key, value in values.items():
        if key.startswith("feed:"):
            stream, _, field = key[len("feed:"):].rpartition(".")
            if field in ("T", "P"):
                answer = session.set_feed(stream, {field: value})
            else:
                continue          # total_flow rescales flows; left to the caller
        else:
            unit, _, field = key.rpartition(".")
            u = next(u for u in session.flowsheet.units if u.name == unit)
            target = "call_params" if field in (u.params or {}) else "params"
            answer = session.patch_unit(unit, {target: {field: value}})
        if answer.get("ok"):
            done.append(key)
    return done


def uncertainty(session, output: str, uncertain: dict[str, float],
                samples: int = 0, seed: int = 0) -> dict:
    """Spread of an output from independent normal uncertainty in levers.

    First-order (linear) propagation from the gradient always; Monte Carlo
    with ``samples`` solves when asked, as a check on the linearization.
    """
    import jax
    import numpy as np

    chosen = _check_levers(session, list(uncertain))
    f = _function(session, output)
    u0 = {k: item["value"] for k, item in chosen.items()}
    y0, grad = jax.value_and_grad(f)(u0)
    contributions = {k: (float(grad[k]) * float(uncertain[k])) ** 2 for k in chosen}
    variance = sum(contributions.values())
    out = {
        "ok": True, "output": output, "value": float(y0),
        "std_linear": math.sqrt(variance),
        "contributions": {k: (v / variance if variance else 0.0)
                          for k, v in contributions.items()},
        **_verdict(session),
    }
    if samples:
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(int(samples)):
            point = {k: u0[k] + float(uncertain[k]) * rng.standard_normal()
                     for k in chosen}
            try:
                draws.append(float(f(point)))
            except Exception:  # noqa: BLE001 -- counted, not raised
                continue
        if draws:
            arr = np.array(draws)
            out["monte_carlo"] = {"samples": len(draws), "mean": float(arr.mean()),
                                  "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                                  "p05": float(np.quantile(arr, 0.05)),
                                  "p95": float(np.quantile(arr, 0.95))}
    return out


def report(session, format: str = "markdown") -> dict:
    """The flowsheet's self-documenting report (topology, units, results)."""
    from difflow.report import build_report

    if session.streams is None:
        _verdict(session)
    built = build_report(session.flowsheet, session.streams, include_git=False)
    if format == "json":
        from difflow.report import to_json

        return {"ok": True, "format": "json", "report": to_json(built)}
    from difflow.report import to_markdown

    return {"ok": True, "format": "markdown", "report": to_markdown(built)}
