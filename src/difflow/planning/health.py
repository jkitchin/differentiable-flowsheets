"""Delta-vector health: the diagnostics a large planning model needs.

A delta-base LP is only as good as the Jacobians underneath it, and the ways
those go wrong at scale are not the ways a small example goes wrong.  Nothing
here raises an exception during a solve; the point is to make three specific
failure modes *visible* before they are silently planned around.

The three, in the order they bite:

**Dead levers.**  Every ``jnp.clip``, ``jnp.minimum`` and ``jnp.where`` sitting
on an active bound contributes an *exactly* zero column to ``J``.  The LP then
correctly concludes the lever does nothing and never moves it — not because it
does not matter, but because the linearisation cannot see that it does.  This
is the one genuine analogue of "lost information", and it scales with the
number of active quality specs, which is to say it scales with the size of the
model.

**Amplification, not attenuation.**  A tear solve differentiated implicitly
returns ``(I - A)^-1``, so a recycle of loop gain ``g`` multiplies
sensitivities by ``1/(1 - g)``.  Recycle-to-extinction loops push ``g`` toward
one and the delta vectors blow up; the trust region then has to shrink to stay
honest.  Large flowsheets fail by *exploding* sensitivities far more often
than by vanishing ones.

**Scale spread.**  Mixed engineering units — ppm against kbbl/d against
$/bbl — put entries spanning many orders of magnitude in the same constraint
matrix.  This is the mundane failure that actually stops a large model, and it
is a units problem, not a gradient problem.

What is deliberately *not* flagged is a small composed sensitivity.  A lever
twenty units upstream of a product genuinely has little absolute leverage on
it, and in ``float64`` (which :mod:`difflow` enables at import) the *relative*
sensitivity survives that intact — a chain deep enough to drive outputs to
``1e-61`` still reports its log-derivative to full precision.  Absolute
smallness is physics.  Only a *relative* sensitivity that has gone to zero
means information was lost, and that is what :func:`composed_sensitivity`
measures.

Example:
    >>> report = check_delta_health(net)
    >>> report.ok
    False
    >>> print(report.summary())
    delta-vector health: 2 findings (1 error, 1 warning)
    ...
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from difflow.planning.block import Block
from difflow.planning.linearize import Linearization, jacobian_fn, linearize_block
from difflow.planning.network import Network, NetworkState

#: A scaled sensitivity at or below this is treated as structurally zero.
DEAD_TOL = 1e-9

#: Predicted relative change over one trust-region step above which the
#: first-order model is not worth trusting.
AMPLIFY_TOL = 10.0

#: Condition number above which ``float64`` has lost half its digits.
COND_TOL = 1e8

#: Ratio of largest to smallest nonzero ``|A|`` entry that an LP solver can be
#: expected to handle without scaling.
SPREAD_TOL = 1e8


class DeltaHealthWarning(UserWarning):
    """A delta vector is structurally suspect.

    Raised by :meth:`HealthReport.warn` rather than during a solve: the LP
    built from such a Jacobian is still solvable, and still wrong in a way the
    objective value will not reveal.
    """


@dataclass(frozen=True)
class Finding:
    """One diagnostic result.

    Attributes:
        kind: One of ``"non_finite"``, ``"dead_lever"``, ``"dead_output"``,
            ``"amplifying"``, ``"ill_conditioned"``, ``"no_influence"``,
            ``"scale_spread"``.
        severity: ``"error"`` (the LP is built on something meaningless) or
            ``"warning"`` (the LP is buildable but the model is strained).
        block: Block name, or ``None`` for network- and LP-level findings.
        variable: Qualified variable name, where one is implicated.
        value: The measured quantity that triggered the finding.
        detail: Human-readable explanation, including the remedy.
    """

    kind: str
    severity: str
    detail: str
    block: str | None = None
    variable: str | None = None
    value: float = float("nan")

    def __str__(self) -> str:
        where = self.variable or self.block or "network"
        return f"[{self.severity}] {self.kind}: {where}: {self.detail}"


@dataclass
class HealthReport:
    """The findings from a health check, plus the scaled Jacobians behind them.

    Attributes:
        findings: Every :class:`Finding`, errors first.
        scaled: ``{block: dimensionless Jacobian}``.  Entry ``[i, j]`` is the
            fractional change in output ``i`` per full-bound-range move of
            input ``j``.
        thresholds: The tolerances actually used, for the record.
    """

    findings: list[Finding] = field(default_factory=list)
    scaled: dict[str, np.ndarray] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when nothing was flagged."""
        return not self.findings

    @property
    def errors(self) -> list[Finding]:
        """Findings that make the LP coefficients meaningless."""
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        """Findings that strain the model without invalidating it."""
        return [f for f in self.findings if f.severity == "warning"]

    def of_kind(self, kind: str) -> list[Finding]:
        """Findings of one kind."""
        return [f for f in self.findings if f.kind == kind]

    def dead_levers(self) -> list[str]:
        """Qualified names of inputs with a structurally zero delta column."""
        return [f.variable for f in self.of_kind("dead_lever") if f.variable]

    def warn(self, stacklevel: int = 2) -> "HealthReport":
        """Emit each finding as a :class:`DeltaHealthWarning`.  Returns self."""
        for f in self.findings:
            warnings.warn(str(f), DeltaHealthWarning, stacklevel=stacklevel)
        return self

    def raise_on_error(self) -> "HealthReport":
        """Raise ``ValueError`` if any finding is an error.  Returns self."""
        errs = self.errors
        if errs:
            raise ValueError(
                f"delta-vector health check found {len(errs)} error(s):\n"
                + "\n".join(f"  {e}" for e in errs))
        return self

    def summary(self) -> str:
        """Render the findings as text."""
        if not self.findings:
            return "delta-vector health: no findings"
        head = (f"delta-vector health: {len(self.findings)} findings "
                f"({len(self.errors)} error, {len(self.warnings)} warning)")
        return "\n".join([head] + [f"  {f}" for f in self.findings])

    def __repr__(self) -> str:
        return (f"HealthReport(findings={len(self.findings)}, "
                f"errors={len(self.errors)}, blocks={sorted(self.scaled)})")


def _u_scale(block: Block) -> np.ndarray:
    """Per-input scale: the bound range, or ``max(|u0|, 1)`` where unbounded.

    This mirrors :func:`difflow.planning.assemble.trust_region_bounds`, so a
    scaled sensitivity of 1 means "a full-range move changes the output by
    100% of its own value" in the same currency the trust region steps in.
    """
    span = np.asarray(block.range, dtype=float)
    u0 = np.abs(np.asarray(block.u0, dtype=float))
    finite = np.isfinite(span) & (span > 0)
    return np.where(finite, np.where(finite, span, 0.0), np.maximum(u0, 1.0))


def _y_scale(y0: np.ndarray) -> np.ndarray:
    """Per-output scale: ``|y0|``, falling back to 1 where the output is zero."""
    a = np.abs(np.asarray(y0, dtype=float))
    return np.where(a > 0, a, 1.0)


def scaled_jacobian(block: Block, lin: Linearization,
                    u_scale: np.ndarray | None = None) -> np.ndarray:
    """Make a block's delta vectors dimensionless.

    Args:
        block: The block, for its bounds.
        lin: Its :class:`~difflow.planning.linearize.Linearization`.
        u_scale: Per-input scale to use instead of the bound range.  A
            *linked* input needs this: the trust region never steps it — the
            LP's link row ties it to the upstream output — so its declared
            bounds are not the move the diagnostic should reason about.

    Returns:
        ``Js`` with ``Js[i, j] = J[i, j] * u_scale[j] / y_scale[i]`` — the
        fractional change in output ``i`` per full-range move of input ``j``.

    Note:
        Scaling is what separates "this lever does nothing" from "this lever is
        measured in pascals and that one in mole fractions".  Comparing raw
        ``J`` entries across a model in mixed units compares unit conversions.
    """
    J = np.asarray(lin.J, dtype=float)
    us = _u_scale(block) if u_scale is None else np.asarray(u_scale, float)
    return J * us[None, :] / _y_scale(np.asarray(lin.y0))[:, None]


def check_block_health(block: Block, lin: Linearization | None = None,
                       u0: Any = None,
                       theta: Mapping[str, Any] | None = None,
                       radius: float = 0.3,
                       dead_tol: float = DEAD_TOL,
                       amplify_tol: float = AMPLIFY_TOL,
                       cond_tol: float = COND_TOL,
                       linked: Sequence[str] = ()) -> HealthReport:
    """Check one block's delta vectors.

    Args:
        block: The block to check.
        lin: A precomputed linearisation.  Computed at ``u0`` when omitted.
        u0: Point at which to linearise.  Defaults to ``block.u0``.
        theta: Parameter override.
        radius: Trust-region radius the report should reason about, as a
            fraction of the bound range.  Amplification is judged by the
            relative change a step of this size is predicted to cause.
        dead_tol: Scaled sensitivity at or below which a column or row counts
            as structurally zero.
        amplify_tol: Predicted relative change over one trust-region step
            above which the first-order model is flagged.  Outputs whose value
            is zero at the linearisation point are excluded: "a fraction of
            its own value" is undefined there, and a bang-bang lever at a
            corner routinely produces such an output.
        cond_tol: Condition number of the scaled Jacobian above which the
            block is flagged.
        linked: Bare names of inputs driven by a link rather than by a free
            decision.  These are scaled by their value at the operating point
            instead of by their bound range, because the trust region does not
            step them — see :func:`scaled_jacobian`.

    Returns:
        A :class:`HealthReport` covering this block only.

    Example:
        >>> report = check_block_health(blend_block)
        >>> report.dead_levers()
        ['blend.reformate_rate']
    """
    if lin is None:
        lin = linearize_block(block, u0, theta)
    u_at = np.asarray(lin.u0, dtype=float)
    us = _u_scale(block)
    linked_set = {n.split(".", 1)[-1] for n in linked}
    is_linked = np.array([n in linked_set for n in block.u_names], dtype=bool)
    if is_linked.any():
        us = np.where(is_linked, np.maximum(np.abs(u_at), 1e-30), us)
    Js = scaled_jacobian(block, lin, u_scale=us)
    found: list[Finding] = []

    bad = ~np.isfinite(Js)
    if bad.any():
        rows, cols = np.nonzero(bad)
        names = sorted({block.u_names[j] for j in cols})
        found.append(Finding(
            kind="non_finite", severity="error", block=block.name,
            variable=f"{block.name}.{names[0]}" if len(names) == 1 else None,
            value=float(bad.sum()),
            detail=(f"{int(bad.sum())} non-finite delta entries, implicating "
                    f"input(s) {names}. The LP coefficients are meaningless. "
                    "Usual causes are log(0), sqrt(negative) or a 0/0 inside "
                    "the block at this operating point; re-centre the "
                    "linearisation or guard the offending expression.")))
        Js = np.where(bad, 0.0, Js)

    # -- dead columns: the LP cannot see this lever at all ------------------
    lb = np.asarray(block.lb, dtype=float)
    ub = np.asarray(block.ub, dtype=float)
    u0v = u_at
    span = np.where(np.isfinite(ub - lb) & (ub - lb > 0), ub - lb, 1.0)
    at_lb = np.abs(u0v - lb) <= 1e-9 * span
    at_ub = np.abs(u0v - ub) <= 1e-9 * span

    col_max = np.abs(Js).max(axis=0) if Js.size else np.zeros(block.n_u)
    for j in np.nonzero(col_max <= dead_tol)[0]:
        pinned = ("at its lower bound" if at_lb[j] else
                  "at its upper bound" if at_ub[j] else None)
        where = (f" The lever itself sits {pinned}."
                 if pinned else
                 " The lever is interior, so the zero comes from a saturated "
                 "expression downstream of it — a clip, minimum or where on "
                 "an active spec — rather than from the bound.")
        found.append(Finding(
            kind="dead_lever", severity="warning", block=block.name,
            variable=f"{block.name}.{block.u_names[j]}",
            value=float(col_max[j]),
            detail=(f"delta column is structurally zero (max scaled "
                    f"sensitivity {col_max[j]:.3e}); the LP will never move "
                    f"this lever.{where} Re-centre inside the smooth region, "
                    "or model the saturation explicitly with a piecewise "
                    "block rather than letting the clip hide it.")))

    # -- dead rows: a priced or constrained output that responds to nothing --
    row_max = np.abs(Js).max(axis=1) if Js.size else np.zeros(block.n_y)
    for i in np.nonzero(row_max <= dead_tol)[0]:
        found.append(Finding(
            kind="dead_output", severity="warning", block=block.name,
            variable=f"{block.name}.{block.y_names[i]}",
            value=float(row_max[i]),
            detail=(f"delta row is structurally zero (max scaled sensitivity "
                    f"{row_max[i]:.3e}); this output is a constant to the LP. "
                    "If it carries a price or appears in a spec, the plan is "
                    "being made against a number that cannot move.")))

    # -- amplification: the recycle signature -------------------------------
    # Amplification asks what fraction of its own value an output moves, which
    # is undefined for an output whose value is zero -- and vertex seeding
    # guarantees bang-bang levers land on exactly those corners (a gas_sold
    # that is zero because the allocation lever burns everything). Judging
    # those rows would fire on almost every solved plan.
    y_abs = np.abs(np.asarray(lin.y0, dtype=float))
    live = (y_abs > 1e-12 * y_abs.max()) if y_abs.size and y_abs.max() > 0 \
        else np.zeros(y_abs.shape, dtype=bool)
    if Js.size and live.any():
        Js_live = Js[live]
        peak = float(np.abs(Js_live).max())
        predicted = radius * peak
        if predicted > amplify_tol:
            i_live, j = np.unravel_index(int(np.argmax(np.abs(Js_live))),
                                         Js_live.shape)
            i = int(np.nonzero(live)[0][i_live])
            # Distinguish the two causes rather than guessing: a bound range
            # far wider than the operating point inflates the scaled step all
            # by itself, and calling that a recycle would send the reader to
            # the wrong place.
            wide = (not is_linked[j] and np.isfinite(us[j])
                    and us[j] > 20.0 * max(abs(float(u_at[j])), 1e-30))
            if wide:
                cause = (f"The bounds on this lever span {us[j]:.3g} while it "
                         f"operates at {float(u_at[j]):.3g}, so a full-range "
                         "step is far outside the region the linearisation "
                         "describes. Tighten the bounds to the range the "
                         "lever is actually planned over.")
            else:
                cause = ("Where the block contains a recycle this is the "
                         "signature of a loop gain near one — implicit "
                         "differentiation of the tear returns (I - A)^-1 — "
                         "and the remedy is a smaller radius. Otherwise the "
                         "block is simply strongly nonlinear here; consider a "
                         "piecewise model.")
            found.append(Finding(
                kind="amplifying", severity="warning", block=block.name,
                variable=f"{block.name}.{block.u_names[j]}", value=predicted,
                detail=(f"a trust-region step of radius {radius:g} is "
                        f"predicted to change {block.y_names[i]!r} by "
                        f"{predicted:.3g}x its own value; a first-order model "
                        f"does not survive that. {cause}")))

    if Js.size and min(Js.shape) > 1:
        cond = float(np.linalg.cond(Js))
        if np.isfinite(cond) and cond > cond_tol:
            found.append(Finding(
                kind="ill_conditioned", severity="warning",
                block=block.name, value=cond,
                detail=(f"scaled delta vectors have condition number "
                        f"{cond:.3e}; distinct levers are close to "
                        "linearly dependent, so the LP's choice among "
                        "them is decided by rounding. Drop a redundant "
                        "lever or re-scale the bounds.")))

    return HealthReport(
        findings=sorted(found, key=lambda f: f.severity != "error"),
        scaled={block.name: Js},
        thresholds={"dead_tol": dead_tol, "amplify_tol": amplify_tol,
                    "cond_tol": cond_tol, "radius": radius})


def composed_sensitivity(network: Network, decisions: Any = None,
                         outputs: Sequence[str] | None = None,
                         theta: Mapping[str, Mapping[str, Any]] | None = None,
                         ) -> tuple[np.ndarray, list[str], list[str]]:
    """Relative sensitivity of network outputs to the free decisions.

    This is the quantity that answers "has depth destroyed the signal?".  It
    is obtained by one AD pass over the composed network, *not* by multiplying
    per-block Jacobians together, so it does not itself suffer the rounding it
    is measuring.

    Args:
        network: The network.
        decisions: Operating point.  Defaults to ``network.decision_start()``.
        outputs: Qualified output names to track.  Defaults to all of them.
        theta: Parameter override.

    Returns:
        ``(S, output_names, decision_names)`` where ``S[i, j]`` is the
        fractional change in output ``i`` per full-bound-range move of
        decision ``j``.

    Note:
        A *small* entry here is usually physics, not a numerical loss: a lever
        far upstream of a product legitimately has little leverage on it.  An
        *exactly* zero entry is the meaningful one — it says no path from that
        decision to that output survives the linearisation.
    """
    import jax.numpy as jnp

    d0 = (network.decision_start() if decisions is None
          else network.decision_array(decisions))
    names_y = list(outputs) if outputs else network.output_names
    names_d = network.decision_names

    def f(d):
        st = network.evaluate(d, theta)
        return jnp.stack([jnp.asarray(st[n], dtype=float) for n in names_y])

    y0 = np.asarray(f(d0), dtype=float)
    jac, _ = jacobian_fn(f, n_u=len(names_d), n_y=len(names_y))
    J = np.atleast_2d(np.asarray(jac(d0), dtype=float))

    lb, ub = network.decision_bounds()
    span = np.asarray(ub, dtype=float) - np.asarray(lb, dtype=float)
    d0v = np.abs(np.asarray(d0, dtype=float))
    finite = np.isfinite(span) & (span > 0)
    u_scale = np.where(finite, np.where(finite, span, 0.0),
                       np.maximum(d0v, 1.0))
    return J * u_scale[None, :] / _y_scale(y0)[:, None], names_y, names_d


def check_network_health(network: Network, decisions: Any = None,
                         theta: Mapping[str, Mapping[str, Any]] | None = None,
                         radius: float = 0.3,
                         dead_tol: float = DEAD_TOL,
                         amplify_tol: float = AMPLIFY_TOL,
                         cond_tol: float = COND_TOL,
                         composed: bool = True) -> HealthReport:
    """Check every block in a network, at the network's own operating point.

    Each block is linearised where the *network* puts it, which is the point
    that matters: a block's own ``u0`` says nothing about the inlet a link
    actually delivers.

    Args:
        network: The network to check.
        decisions: Operating point.  Defaults to ``network.decision_start()``.
        theta: ``{block: parameter dict}`` override.
        radius: Trust-region radius to reason about.
        dead_tol: See :func:`check_block_health`.
        amplify_tol: See :func:`check_block_health`.
        cond_tol: See :func:`check_block_health`.
        composed: Also measure end-to-end relative sensitivity and flag
            decisions that reach no output at all.

    Returns:
        A combined :class:`HealthReport`.

    Example:
        >>> print(check_network_health(net).summary())
        delta-vector health: 1 findings (0 error, 1 warning)
          [warning] dead_lever: fcc.riser_T: delta column is structurally zero ...
    """
    state: NetworkState = network.evaluate(
        network.decision_start() if decisions is None else decisions, theta)

    findings: list[Finding] = []
    scaled: dict[str, np.ndarray] = {}
    for b in network.blocks:
        th = None if theta is None else theta.get(b.name)
        linked = [n for n in b.qualified_u() if network.is_linked(n)]
        rep = check_block_health(b, u0=state.u[b.name], theta=th,
                                 radius=radius, dead_tol=dead_tol,
                                 amplify_tol=amplify_tol, cond_tol=cond_tol,
                                 linked=linked)
        findings.extend(rep.findings)
        scaled.update(rep.scaled)

    if composed:
        S, names_y, names_d = composed_sensitivity(network, decisions, None,
                                                   theta)
        reach = np.abs(S).max(axis=0)
        for j in np.nonzero(reach <= dead_tol)[0]:
            findings.append(Finding(
                kind="no_influence", severity="warning", variable=names_d[j],
                value=float(reach[j]),
                detail=(f"this decision has no first-order influence on any "
                        f"network output (max scaled sensitivity "
                        f"{reach[j]:.3e}). Every path from it to an output is "
                        "blocked by a saturated expression, so the LP is free "
                        "to place it anywhere in its bounds.")))
        scaled["<composed>"] = S

    return HealthReport(
        findings=sorted(findings, key=lambda f: f.severity != "error"),
        scaled=scaled,
        thresholds={"dead_tol": dead_tol, "amplify_tol": amplify_tol,
                    "cond_tol": cond_tol, "radius": radius})


def check_lp_scaling(lp, spread_tol: float = SPREAD_TOL) -> HealthReport:
    """Check the assembled program for the units problem.

    Args:
        lp: An :class:`~difflow.planning.lp.LPModel`.
        spread_tol: Ratio of largest to smallest nonzero ``|A|`` entry above
            which the matrix is flagged.

    Returns:
        A :class:`HealthReport` with ``"scale_spread"`` findings.

    Note:
        This is the failure that actually stops a large planning model, and it
        is a *units* problem: ppm against kbbl/d against $/bbl in one matrix.
        Nondimensionalise the levers before blaming the gradients.
    """
    A = np.vstack([np.asarray(lp.A_eq, dtype=float).reshape(-1, lp.n_cols),
                   np.asarray(lp.A_ub, dtype=float).reshape(-1, lp.n_cols)])
    found: list[Finding] = []
    nz = np.abs(A[A != 0])
    if nz.size:
        spread = float(nz.max() / nz.min())
        if spread > spread_tol:
            found.append(Finding(
                kind="scale_spread", severity="warning", value=spread,
                detail=(f"constraint entries span {nz.min():.3e} to "
                        f"{nz.max():.3e} (ratio {spread:.3e}); the solver's "
                        "pivot tolerances are being asked to separate signal "
                        "from unit conversion. Re-scale the offending "
                        "variables to comparable magnitudes.")))

    names = list(lp.eq_names) + list(lp.ub_names)
    for r in range(A.shape[0]):
        row = np.abs(A[r][A[r] != 0])
        if row.size > 1 and row.max() / row.min() > spread_tol:
            found.append(Finding(
                kind="scale_spread", severity="warning",
                variable=names[r] if r < len(names) else f"row[{r}]",
                value=float(row.max() / row.min()),
                detail=(f"a single row spans {row.min():.3e} to "
                        f"{row.max():.3e}; its small coefficients are below "
                        "the solver's effective precision relative to its "
                        "large ones and will be treated as zero.")))

    return HealthReport(findings=found, thresholds={"spread_tol": spread_tol})


def check_delta_health(target: Block | Network, **kwargs: Any) -> HealthReport:
    """Check a :class:`Block` or a :class:`Network`, whichever is given.

    Args:
        target: The block or network to check.
        **kwargs: Forwarded to :func:`check_block_health` or
            :func:`check_network_health`.

    Returns:
        A :class:`HealthReport`.

    Example:
        >>> check_delta_health(net, radius=0.1).raise_on_error()
    """
    if isinstance(target, Network):
        return check_network_health(target, **kwargs)
    if isinstance(target, Block):
        kwargs.pop("composed", None)
        return check_block_health(target, **kwargs)
    raise TypeError(
        f"check_delta_health expects a Block or a Network, got "
        f"{type(target).__name__}")
