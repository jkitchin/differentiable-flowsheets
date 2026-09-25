"""Which delta vectors are wrong, estimated from routine plant data.

:mod:`difflow.planning.modifiers` corrects a block with a value term and a
gradient term,

    y_plant(u) ~ y_model(u) + eps + lam (u - u_ad),

and :func:`~difflow.planning.modifiers.update_modifiers` takes the plant
gradient from a callable.  A running plant is not a callable: what exists is
a history of the block's inputs and the measured outputs.  This module
estimates ``eps`` and ``lam`` from that history and, more importantly, says
which of them the history can support at all.

For each observed output the residual ``r = g(y_meas) - g(block(u))`` is
regressed, by weighted least squares, on

    [ 1,  (t - t_now),  (u_1 - u_ref_1)/move_1,  ...,  (u_m - u_ref_m)/move_m ]

where ``g`` is the identity or ``log(y + floor)``.  The level and trend are
always estimated; the level is the value correction at ``(t_now, u_ref)``
and each slope is the gradient error per characteristic move of one input.

What the data can and cannot say
--------------------------------
Plant data are not a designed experiment.  An input that the operators held
still carries no information about its own gradient, and inputs that moved
together (one controller driving both) are only estimable as a combination.
Expect *most* slopes to be inestimable from routine closed-loop operation,
and the level to be reliable.  So the fit is organised around four checks,
each reported rather than hidden:

1. **Estimability is decided from the design, before the fit.**  The slope
   columns, scaled by the row standard deviation and with the level and trend
   projected out, go through a column-pivoted QR.  A slope is kept while
   ``|R_kk| * materiality >= estimable`` -- the data could resolve an error of
   size ``materiality`` per move at about two standard errors.  The rest are
   reported as ``"not estimable"`` and left out.  The decision never looks at
   ``y``, so it cannot be tuned by the answer.
2. **Aliasing is reported.**  Each held-out column is regressed on the kept
   ones; a kept coefficient absorbs ``A * beta_held`` of any held-out slope.
   A significant estimate with ``|A| > alias_tol`` is reported as a
   ``"combination"``, not as a finding about that one input.
3. **Standard errors are inflated for what WLS assumes away.**  Plant
   residuals are autocorrelated and the stated noise is usually optimistic,
   so standard errors are multiplied by
   ``sqrt(phi * (1 + rho) / (1 - rho))`` with ``phi = max(1, chi2/dof)`` and
   ``rho`` the lag-1 autocorrelation of the weighted residual, clipped to
   ``[0, 0.9]``.  Without it a slow drift produces a stream of false flags.
4. **A Picard check separates "wrong delta" from "wrong form".**  The weighted
   residual is projected onto the left singular vectors of the full scaled
   design.  A large coefficient along a direction the design barely resolves
   would need an implausibly large slope to explain; it is the signature of a
   block whose *form* is wrong over the data, which no affine modifier fixes.
   Such outputs are marked ``structural``.

A coefficient is flagged only when it is estimable, ``|z| > z_flag`` after
inflation, and larger than ``materiality``.  Input measurement noise is
carried into the row weights through the block Jacobian (errors in
variables), so a noisy input does not look like a gradient error.

The outputs of the block that were not measured are listed in
:attr:`AttributionResult.unobserved`: nothing here says anything about them.

Example:
    >>> res = attribute_deltas(block, u_hist, {"conv": conv_hist},
    ...                        sigma_y={"conv": 0.01}, t=t_hist)
    >>> print(res.table())
    >>> planner.modifiers[block.name] = res.to_modifiers()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg

from difflow.planning.block import Block
from difflow.planning.modifiers import Modifiers

#: ``|R_kk| * materiality`` must reach this for a slope to be estimated.
ESTIMABLE = 1.9
#: ``|alias|`` above this makes a significant estimate a combination.
ALIAS_TOL = 0.3
#: ``|z|`` above this (after inflation) is significant.
Z_FLAG = 3.0


@dataclass
class Coefficient:
    """One estimated (or held-out) correction term for one output.

    Attributes:
        output: Output name.
        term: ``"level"``, ``"trend"`` or an input name.
        estimate: Level in (transformed) output units, trend per unit time,
            slope per unit input.  ``nan`` when not estimable.
        se: Inflated standard error, same units as ``estimate``.
        per_move: The error over the characteristic move (slopes), over the
            window (trend) or the level itself -- the size judged against
            ``materiality``.
        z: ``estimate / se``.
        estimable: Whether the design supports this term.
        status: ``"flag"``, ``"combination"``, ``"ok"`` or ``"not estimable"``.
        aliases: ``{held-out term: alias coefficient}`` above ``alias_tol``.
    """

    output: str
    term: str
    estimate: float
    se: float
    per_move: float
    z: float
    estimable: bool
    status: str
    aliases: dict[str, float] = field(default_factory=dict)


@dataclass
class OutputAttribution:
    """The fit for one observed output.

    Attributes:
        output: Output name.
        n: Rows used.
        coefficients: Level, trend and one entry per input.
        phi: ``max(1, chi2/dof)`` of the estimable fit.
        rho: Lag-1 autocorrelation of the weighted residual, clipped.
        inflation: Factor applied to the WLS standard errors.
        picard: ``(singular value, residual coefficient)`` pairs of the full
            scaled design, largest singular value first.
        structural: The residual has weight along directions the design
            cannot resolve (see the module docstring).
        log_floor: ``floor`` when the output is fitted as ``log(y + floor)``.
    """

    output: str
    n: int
    coefficients: list[Coefficient]
    phi: float
    rho: float
    inflation: float
    picard: list[tuple[float, float]]
    structural: bool
    log_floor: float | None = None

    def coefficient(self, term: str) -> Coefficient:
        """The coefficient for ``term``."""
        for c in self.coefficients:
            if c.term == term:
                return c
        raise KeyError(f"no term {term!r} for output {self.output!r}")

    @property
    def not_estimable(self) -> list[str]:
        """Inputs whose slope the data cannot resolve."""
        return [c.term for c in self.coefficients if not c.estimable]


@dataclass
class AttributionResult:
    """Delta attribution for one block.

    Attributes:
        block: The block that was checked.
        u_ref: Input point the level refers to (the modifiers' ``u_ad``).
        t_now: Time the level and trend refer to.
        move: Characteristic move per input.
        outputs: One :class:`OutputAttribution` per observed output.
        unobserved: Block outputs with no measurements.
    """

    block: Block
    u_ref: np.ndarray
    t_now: float
    move: np.ndarray
    outputs: dict[str, OutputAttribution]
    unobserved: list[str]
    theta: Mapping[str, Any] | None = None

    @property
    def flagged(self) -> list[Coefficient]:
        """Every coefficient with status ``"flag"``."""
        return [c for o in self.outputs.values() for c in o.coefficients
                if c.status == "flag"]

    def rows(self) -> list[dict[str, Any]]:
        """All coefficients as plain dicts, e.g. for a DataFrame."""
        out = []
        for o in self.outputs.values():
            for c in o.coefficients:
                out.append({"output": c.output, "term": c.term,
                            "estimate": c.estimate, "se": c.se,
                            "per_move": c.per_move, "z": c.z,
                            "estimable": c.estimable, "status": c.status,
                            "aliases": dict(c.aliases),
                            "structural": o.structural})
        return out

    def table(self) -> str:
        """A fixed-width report of every coefficient."""
        lines = [f"Delta attribution for block {self.block.name!r}",
                 f"  {'output':<14s} {'term':<14s} {'estimate':>11s} "
                 f"{'se':>10s} {'per move':>10s} {'z':>7s}  status"]
        for o in self.outputs.values():
            for c in o.coefficients:
                est = "-" if not c.estimable else f"{c.estimate:11.4g}"
                se = "-" if not c.estimable else f"{c.se:10.3g}"
                pm = "-" if not c.estimable else f"{c.per_move:10.3g}"
                z = "-" if not c.estimable else f"{c.z:7.2f}"
                note = c.status
                if c.aliases:
                    note += " (with " + ", ".join(
                        f"{k} {v:+.2f}" for k, v in c.aliases.items()) + ")"
                lines.append(f"  {o.output:<14s} {c.term:<14s} {est:>11s} "
                             f"{se:>10s} {pm:>10s} {z:>7s}  {note}")
            tail = (f"  {'':<14s} n={o.n}, phi={o.phi:.2f}, rho={o.rho:.2f}, "
                    f"inflation={o.inflation:.2f}")
            if o.structural:
                tail += ", STRUCTURAL: residual the affine correction cannot fit"
            lines.append(tail)
        if self.unobserved:
            lines.append(f"  unobserved (no evidence either way): "
                         f"{', '.join(self.unobserved)}")
        return "\n".join(lines)

    def to_modifiers(self, flagged_only: bool = True,
                     statuses: Sequence[str] = ("flag",)) -> Modifiers:
        """Modifiers built from the estimates, at ``u_ad = u_ref``.

        The level becomes ``eps`` and each slope, per unit input, a row of
        ``lam``.  For a log-fitted output the correction is mapped back to
        output units exactly at ``u_ref`` by the chain rule, so the
        corrected model reproduces the fitted value and gradient there.  The
        trend is not carried: the level already refers to ``t_now``.

        Args:
            flagged_only: Use only coefficients whose status is in
                ``statuses``.  ``False`` uses every estimable coefficient.
            statuses: The statuses admitted when ``flagged_only``.

        Returns:
            :class:`~difflow.planning.modifiers.Modifiers` for the block;
            unobserved outputs and excluded terms get zero correction.
        """
        blk = self.block
        u_ad = jnp.asarray(self.u_ref, dtype=float)
        y_m = np.asarray(blk.evaluate(u_ad, self.theta), dtype=float)
        J = np.asarray(blk.jacobian("fwd")(u_ad, self.theta), dtype=float)
        eps = np.zeros(blk.n_y)
        lam = np.zeros((blk.n_y, blk.n_u))

        def keep(c: Coefficient) -> bool:
            if not c.estimable:
                return False
            return c.status in statuses if flagged_only else True

        for name, o in self.outputs.items():
            k = blk.y_index(name)
            level = o.coefficient("level")
            a = level.estimate if keep(level) else 0.0
            s = np.array([o.coefficient(n).estimate
                          if keep(o.coefficient(n)) else 0.0
                          for n in blk.u_names])
            if o.log_floor is None:
                eps[k] = a
                lam[k] = s
            else:
                f = o.log_floor
                y_p = (y_m[k] + f) * np.exp(a) - f
                grad_p = (y_p + f) * (J[k] / (y_m[k] + f) + s)
                eps[k] = y_p - y_m[k]
                lam[k] = grad_p - J[k]
        return Modifiers(eps=jnp.asarray(eps), lam=jnp.asarray(lam),
                         u_ad=u_ad)

    def exposure(self, plan_result) -> list[dict[str, Any]]:
        """What each observed level error is worth at a plan.

        The planner's model rows are named ``model[<block>.<output>]`` and
        their right-hand side moves one-for-one with a value correction, so
        the row's shadow price times the level error (in output units) is
        the first-order change in the LP objective were the correction
        applied.  It is a ranking device, valid near the plan the duals came
        from; it says nothing about the slopes.

        Args:
            plan_result: A :class:`~difflow.planning.planner.PlanResult` for
                a network containing this block.

        Returns:
            One dict per observed output: ``output``, ``dual``, ``eps``,
            ``eps_se`` (output units) and ``exposure``, ``exposure_se``
            (objective units, magnitudes), largest exposure first.  Empty if
            the solver reported no marginals.
        """
        eq = (plan_result.duals or {}).get("eq", {})
        if not eq:
            return []
        blk = self.block
        y_m = np.asarray(blk.evaluate(jnp.asarray(self.u_ref), self.theta))
        out = []
        for name, o in self.outputs.items():
            key = f"model[{blk.name}.{name}]"
            if key not in eq:
                continue
            lev = o.coefficient("level")
            a, se = lev.estimate, lev.se
            if o.log_floor is not None:
                # d y / d g = y + floor at the model value.
                scale = float(y_m[blk.y_index(name)]) + o.log_floor
                a, se = scale * np.expm1(a), scale * se
            d = float(eq[key])
            out.append({"output": name, "dual": d, "eps": float(a),
                        "eps_se": float(se), "exposure": abs(d * a),
                        "exposure_se": abs(d) * float(se)})
        return sorted(out, key=lambda r: -r["exposure"])

    def __repr__(self) -> str:
        return (f"AttributionResult(block={self.block.name!r}, "
                f"outputs={list(self.outputs)}, flagged={len(self.flagged)})")


def _per_input(value, names: list[str], what: str,
               default: float | None = None) -> np.ndarray:
    """Coerce a scalar, array or ``{input: value}`` mapping to an array."""
    if value is None:
        if default is None:
            raise ValueError(f"{what} is required")
        return np.full(len(names), float(default))
    if isinstance(value, Mapping):
        unknown = sorted(set(value) - set(names))
        if unknown:
            raise KeyError(f"{what} names unknown input(s) {unknown}")
        fill = 0.0 if default is None else default
        return np.array([float(value.get(n, fill)) for n in names])
    arr = np.broadcast_to(np.asarray(value, dtype=float), (len(names),))
    return np.array(arr, dtype=float)


def _per_output(value, outputs: list[str], what: str,
                default: Mapping[str, float] | None = None) -> dict[str, float]:
    if value is None:
        if default is None:
            raise ValueError(f"{what} is required")
        return dict(default)
    if isinstance(value, Mapping):
        missing = [o for o in outputs if o not in value]
        if missing:
            raise KeyError(f"{what} missing output(s) {missing}")
        return {o: float(value[o]) for o in outputs}
    return {o: float(value) for o in outputs}


def _model_values(block: Block, u: np.ndarray, theta, need_jac: bool):
    """Block outputs (and Jacobians) at every row, batched where possible."""
    uj = jnp.asarray(u, dtype=float)
    ev = lambda x: block.evaluate(x, theta)  # noqa: E731
    jac = block.jacobian("fwd")
    try:
        y = np.asarray(jax.vmap(ev)(uj))
        J = np.asarray(jax.vmap(lambda x: jac(x, theta))(uj)) if need_jac else None
    except Exception:
        # Python control flow in the block: fall back to a loop.
        y = np.stack([np.asarray(ev(x)) for x in uj])
        J = (np.stack([np.asarray(jac(x, theta)) for x in uj])
             if need_jac else None)
    return y, J


def _lag1(e: np.ndarray) -> float:
    if e.size < 3:
        return 0.0
    d = e - e.mean()
    den = float(d @ d)
    if den <= 0.0:
        return 0.0
    return float(np.clip((d[1:] @ d[:-1]) / den, 0.0, 0.9))


def attribute_deltas(block: Block, u: Any, y: Mapping[str, Any],
                     sigma_y: Mapping[str, float] | float,
                     t: Any = None, *,
                     sigma_u: Mapping[str, float] | Any = None,
                     move: Mapping[str, float] | Any = None,
                     u_ref: Any = None,
                     materiality: Mapping[str, float] | float | None = None,
                     log_outputs: Mapping[str, float] | None = None,
                     theta: Mapping[str, Any] | None = None,
                     estimable: float = ESTIMABLE,
                     z_flag: float = Z_FLAG,
                     alias_tol: float = ALIAS_TOL) -> AttributionResult:
    """Estimate which of a block's value and gradient corrections the data support.

    Args:
        block: The model block.
        u: Measured block inputs, shape ``(n, n_u)``, ordered like
            ``block.u_names``.
        y: ``{output name: measurements, shape (n,)}`` for the outputs that
            were measured.  ``nan`` marks a missing value.
        sigma_y: Measurement standard deviation per output, in the space
            the output is fitted in (relative error for a log output).
        t: Sample times, shape ``(n,)``; defaults to ``0 .. n-1``.  The level
            refers to the latest time.
        sigma_u: Input measurement standard deviation, per input; carried
            into the row weights through the block Jacobian.  Default zero.
        move: Characteristic move per input -- the size of step a planner
            would take.  Slopes are judged per move.  Defaults to a tenth of
            the bound range; required for unbounded inputs.
        u_ref: Input point the level refers to and the modifiers'
            ``u_ad``.  Defaults to the mean input, where the level is not
            aliased with any held-out slope.
        materiality: Smallest error worth acting on, per output, in fitted
            units (a level, or a slope times its move).  Defaults to
            ``sigma_y``.
        log_outputs: ``{output: floor}`` for outputs fitted as
            ``log(y + floor)`` -- concentrations and other positive
            quantities whose error is relative.
        theta: Parameter override for the block.
        estimable: Threshold on ``|R_kk| * materiality``.
        z_flag: Significance threshold on the inflated ``|z|``.
        alias_tol: Alias coefficient above which an estimate is a combination.

    Returns:
        An :class:`AttributionResult`.

    Example:
        >>> res = attribute_deltas(blk, U, {"yield": Y}, sigma_y=0.02, t=days)
        >>> [c.term for c in res.flagged]
        ['level', 'feed_rate']
    """
    U = np.atleast_2d(np.asarray(u, dtype=float))
    n, n_u = U.shape
    if n_u != block.n_u:
        raise ValueError(f"u has {n_u} columns, block {block.name!r} has "
                         f"{block.n_u} inputs")
    unknown = sorted(set(y) - set(block.y_names))
    if unknown:
        raise KeyError(f"{unknown} are not outputs of block {block.name!r}")
    outputs = [o for o in block.y_names if o in y]
    if not outputs:
        raise ValueError("no measured outputs")
    log_outputs = dict(log_outputs or {})

    T = np.arange(n, dtype=float) if t is None else np.asarray(t, dtype=float)
    if T.shape != (n,):
        raise ValueError(f"t has shape {T.shape}, expected ({n},)")
    order = np.argsort(T, kind="stable")
    U, T = U[order], T[order]
    Y = {o: np.asarray(y[o], dtype=float).reshape(-1)[order] for o in outputs}

    sy = _per_output(sigma_y, outputs, "sigma_y")
    mat = _per_output(materiality, outputs, "materiality", default=sy)
    su = _per_input(sigma_u, block.u_names, "sigma_u", default=0.0)
    if move is None:
        rng = np.asarray(block.range, dtype=float)
        if not np.all(np.isfinite(rng)) or np.any(rng <= 0):
            raise ValueError("move is required for unbounded inputs")
        mv = 0.1 * rng
    else:
        mv = _per_input(move, block.u_names, "move")
    if np.any(mv <= 0):
        raise ValueError("move must be positive")

    u_ok = np.all(np.isfinite(U), axis=1)
    ref = (U[u_ok].mean(axis=0) if u_ref is None
           else _per_input(u_ref, block.u_names, "u_ref"))
    t_now = float(T[u_ok].max())
    span = float(np.ptp(T[u_ok])) or 1.0

    need_jac = bool(np.any(su > 0))
    Ufill = np.where(np.isfinite(U), U, ref)
    Ym, Jm = _model_values(block, Ufill, theta, need_jac)

    fits: dict[str, OutputAttribution] = {}
    for o in outputs:
        k = block.y_index(o)
        floor = log_outputs.get(o)
        ok = u_ok & np.isfinite(Y[o])
        ym = Ym[ok, k]
        yo = Y[o][ok]
        if floor is None:
            r = yo - ym
            gprime = np.ones_like(ym)
        else:
            r = np.log(yo + floor) - np.log(ym + floor)
            gprime = 1.0 / (ym + floor)
        var = np.full(r.shape, sy[o] ** 2)
        if need_jac:
            dg = gprime[:, None] * Jm[ok, k, :]
            var = var + np.sum((dg * su) ** 2, axis=1)
        sd = np.sqrt(var)

        tt = (T[ok] - t_now) / span
        base = np.column_stack([np.ones(ok.sum()), tt])
        has_trend = np.ptp(tt) > 0
        if not has_trend:
            base = base[:, :1]
        S = (U[ok] - ref) / mv
        Wb, Ws, rw = base / sd[:, None], S / sd[:, None], r / sd

        # 1. Estimability from the design alone.
        Qb, _ = np.linalg.qr(Wb)
        Z = Ws - Qb @ (Qb.T @ Ws)
        keep: list[int] = []
        if Z.shape[0] > 0 and n_u:
            _, R, piv = scipy.linalg.qr(Z, mode="economic", pivoting=True)
            diag = np.abs(np.diag(R))
            for j, dj in zip(piv, diag):
                if dj * mat[o] < estimable:
                    break
                keep.append(int(j))
        held = [j for j in range(n_u) if j not in keep]
        X = np.column_stack([Wb, Ws[:, keep]])
        dof = X.shape[0] - X.shape[1]
        if dof < 1:
            raise ValueError(f"output {o!r}: {X.shape[0]} rows cannot "
                             f"support {X.shape[1]} terms")

        beta, *_ = np.linalg.lstsq(X, rw, rcond=None)
        e = rw - X @ beta
        cov = np.linalg.pinv(X.T @ X)
        phi = max(1.0, float(e @ e) / dof)
        rho = _lag1(e)
        infl = float(np.sqrt(phi * (1 + rho) / (1 - rho)))
        se = np.sqrt(np.maximum(np.diag(cov), 0.0)) * infl

        # 2. Aliases of the held-out columns onto the kept terms.
        A = (np.linalg.lstsq(X, Ws[:, held], rcond=None)[0] if held
             else np.zeros((X.shape[1], 0)))
        terms = ["level"] + (["trend"] if has_trend else []) + \
            [block.u_names[j] for j in keep]
        held_names = [block.u_names[j] for j in held]

        # 4. Picard: residual weight along barely-resolved directions.
        Uf, s, _ = np.linalg.svd(np.column_stack([Wb, Ws]), full_matrices=False)
        c = Uf.T @ rw
        # Judged against the autocorrelation factor only: phi is itself
        # inflated by misspecification and would hide what this looks for.
        ar = float(np.sqrt((1 + rho) / (1 - rho)))
        weak = s * mat[o] < estimable
        structural = bool(np.any(weak & (np.abs(c) > z_flag * ar)))

        coefs: list[Coefficient] = []
        for i, term in enumerate(terms):
            b, sei = float(beta[i]), float(se[i])
            if term == "level":
                est, est_se, pm = b, sei, b
            elif term == "trend":
                est, est_se, pm = b / span, sei / span, b
            else:
                j = block.u_index(term)
                est, est_se, pm = b / mv[j], sei / mv[j], b
            z = b / sei if sei > 0 else np.inf * np.sign(b)
            al = {h: float(A[i, m]) for m, h in enumerate(held_names)
                  if abs(A[i, m]) > alias_tol}
            if abs(z) > z_flag and abs(pm) > mat[o]:
                status = "combination" if al else "flag"
            else:
                status = "ok"
            coefs.append(Coefficient(o, term, est, est_se, pm, float(z),
                                     True, status, al))
        if not has_trend:
            coefs.insert(1, Coefficient(o, "trend", np.nan, np.nan, np.nan,
                                        np.nan, False, "not estimable"))
        for h in held_names:
            coefs.append(Coefficient(o, h, np.nan, np.nan, np.nan, np.nan,
                                     False, "not estimable"))
        # Report inputs in block order after level and trend.
        head, tail = coefs[:2], {c.term: c for c in coefs[2:]}
        coefs = head + [tail[nm] for nm in block.u_names]

        fits[o] = OutputAttribution(
            output=o, n=int(ok.sum()), coefficients=coefs, phi=phi, rho=rho,
            inflation=infl,
            picard=[(float(a), float(b)) for a, b in zip(s, c)],
            structural=structural, log_floor=floor)

    return AttributionResult(
        block=block, u_ref=np.asarray(ref, dtype=float), t_now=t_now,
        move=mv, outputs=fits,
        unobserved=[o for o in block.y_names if o not in y], theta=theta)
