"""``screen`` — the whole controllability check in one traceable call.

The interesting claim in issue #199 is not that these metrics are new.  It is
that AD makes them exact and cheap enough to evaluate *inside* a design loop
rather than after it, so a candidate flowsheet can be asked whether it is
controllable at the same moment it is asked whether it is profitable.  That
only works if the computation is a pure function of the design variables, so
:func:`screen` is built as one: two ``jax.jacobian`` calls and a handful of
SVDs, with no Python branching on the numbers.

The consequence is a slightly unusual object.
:class:`OperabilityReport` is a registered pytree whose numeric fields may be
tracers, and whose *interpretation* — the findings, the summary text — is
computed lazily from those fields when they are concrete.  So::

    jax.jit(lambda u: screen(plant, u, scaling=sc).msv)(u0)   # works
    jax.vmap(lambda u: screen(plant, u, scaling=sc).msv)(U)   # works
    print(screen(plant, u0, scaling=sc).summary())            # also works

The reporting style deliberately follows
:func:`difflow.planning.health.check_delta_health`: a list of
:class:`~difflow.planning.health.Finding` objects, each naming the measured
value *and* the remedy, and nothing raising during a solve.

Example:
    >>> import jax.numpy as jnp
    >>> from difflow.operability import Scaling, screen
    >>> plant = lambda u: jnp.array([u[0] + 0.9 * u[1], 0.9 * u[0] + u[1]])
    >>> rep = screen(plant, jnp.zeros(2),
    ...              scaling=Scaling(u_span=[1.0, 1.0], y_span=[1.0, 1.0]))
    >>> float(rep.msv) < 0.2
    True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.planning.health import Finding
from difflow.operability.gains import disturbance_gain, gain_matrix
from difflow.operability.metrics import (
    RCOND, condition_number, disturbance_condition_number, negative_pairings,
    required_input_move, rga, rga_number, singular_values, suggest_pairing,
)
from difflow.operability.scaling import Scaling

__all__ = ["OperabilityReport", "screen", "MSV_TOL", "COND_TOL", "RGA_TOL",
           "GD_TOL"]

#: Scaled ``sigma_min`` below which the inputs cannot cover the required
#: output range in the plant's worst direction.
MSV_TOL = 1.0

#: Condition number above which the plant is treated as strongly directional.
COND_TOL = 10.0

#: ``|RGA|`` entry above which decentralised loops interact strongly.
RGA_TOL = 5.0

#: Scaled disturbance gain above which a disturbance must be rejected by
#: control rather than absorbed by the process.
GD_TOL = 1.0


def _names(given: Sequence[str] | None, n: int, prefix: str) -> tuple[str, ...]:
    if given is None:
        return tuple(f"{prefix}{i}" for i in range(n))
    if len(given) != n:
        raise ValueError(
            f"expected {n} {prefix} names, got {len(given)}: {list(given)}")
    return tuple(given)


def _concrete(x: Any) -> bool:
    return not isinstance(x, jax.core.Tracer)


@dataclass
class OperabilityReport:
    """Every steady-state operability measure for one operating point.

    All numeric attributes are JAX arrays and the class is a registered
    pytree, so a report may be produced inside ``jit``/``vmap``/``grad``.  The
    interpretive members (:attr:`findings`, :meth:`summary`, :attr:`ok`) read
    those arrays concretely and therefore only work outside a trace.

    Attributes:
        G: Scaled gain matrix ``dy/du``, shape ``(n_y, n_u)``.
        Gd: Scaled disturbance gain ``dy/dd``, shape ``(n_y, n_d)``, or
            ``None`` when no disturbances were given.
        RGA: Relative gain array, shape ``(n_y, n_u)``.
        svals: Singular values of ``G``, descending.
        msv: Minimum singular value of ``G``.
        cond: Condition number of ``G``.
        rga_num: ``||RGA - P||`` for the pairing under consideration.
        rga_pairs: Relative gain of each proposed pairing, length
            ``min(n_y, n_u)``.  Negative entries are the alarming ones.
        rank: Effective rank of ``G``.
        dist_cond: Disturbance condition number per disturbance, or ``None``.
        u_required: ``pinv(G) @ Gd``, the fraction of each input's available
            range needed to reject each disturbance, or ``None``.
        u_names: Manipulated variable names.
        y_names: Controlled variable names.
        d_names: Disturbance names.
        pairing: ``pairing[i]`` is the input index paired with output ``i``.
        scaled: False when the report was built on unit spans, which makes
            every magnitude below unit-dependent.
        scaling_note: Provenance of the spans used.
    """

    G: Array
    RGA: Array
    svals: Array
    msv: Array
    cond: Array
    rga_num: Array
    rga_pairs: Array
    rank: Array
    Gd: Array | None = None
    dist_cond: Array | None = None
    u_required: Array | None = None
    u_names: tuple[str, ...] = ()
    y_names: tuple[str, ...] = ()
    d_names: tuple[str, ...] = ()
    pairing: tuple[int, ...] = ()
    scaled: bool = True
    scaling_note: str = ""

    # -- shape --------------------------------------------------------------
    @property
    def n_u(self) -> int:
        """Number of manipulated variables."""
        return len(self.u_names)

    @property
    def n_y(self) -> int:
        """Number of controlled variables."""
        return len(self.y_names)

    @property
    def n_d(self) -> int:
        """Number of disturbances."""
        return len(self.d_names)

    @property
    def square(self) -> bool:
        """True when there are as many inputs as outputs."""
        return self.n_u == self.n_y

    @property
    def singular(self) -> bool:
        """True when ``G`` has fewer independent directions than outputs."""
        return bool(float(self.rank) < min(self.n_y, self.n_u))

    def suggested_pairing(self) -> list[tuple[str, str]]:
        """Greedy RGA pairing as ``(output, input)`` name pairs."""
        idx = suggest_pairing(np.asarray(self.RGA), is_rga=True)
        return [(self.y_names[i], self.u_names[j] if j >= 0 else "-")
                for i, j in enumerate(idx)]

    # -- interpretation -----------------------------------------------------
    @property
    def findings(self) -> list[Finding]:
        """Diagnoses read off the numbers, errors first.

        Returns:
            A list of :class:`~difflow.planning.health.Finding`.  Empty means
            nothing in this screen argues against the design.

        Raises:
            ValueError: If the report holds tracers, i.e. it was produced
                inside ``jit`` or ``vmap``.  Pull the numeric fields out of
                the trace first.
        """
        if not _concrete(self.msv):
            raise ValueError(
                "this OperabilityReport holds tracers, so its findings cannot "
                "be evaluated: interpreting them needs Python branching on "
                "the values. Inside jit/vmap use the numeric fields (msv, "
                "cond, RGA, ...) and interpret them after the trace.")
        return _diagnose(self)

    @property
    def ok(self) -> bool:
        """True when no finding was raised."""
        return not self.findings

    @property
    def errors(self) -> list[Finding]:
        """Findings that say the design cannot be controlled as posed."""
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        """Findings that make control harder without making it impossible."""
        return [f for f in self.findings if f.severity == "warning"]

    def summary(self) -> str:
        """Render the screen as text.

        Returns:
            A multi-line report: the caveat banner if unscaled, the headline
            numbers, the RGA, the disturbance table, then the findings.
        """
        found = self.findings
        lines: list[str] = []
        if not self.scaled:
            lines.append("!! UNSCALED: " + (
                self.scaling_note or "unit spans were used"))
            lines.append("!! Magnitudes below describe the unit system as "
                         "much as the plant and may not be compared to 1.")
        lines.append(
            f"operability screen: {self.n_y} outputs, {self.n_u} inputs, "
            f"{self.n_d} disturbances")
        if self.scaling_note and self.scaled:
            lines.append(f"  scaling: {self.scaling_note}")
        lines.append(f"  sigma_min(G) = {float(self.msv):.4g}"
                     f"   cond(G) = {float(self.cond):.4g}"
                     f"   rank(G) = {int(float(self.rank))}"
                     f"/{min(self.n_y, self.n_u)}")
        lines.append(f"  RGA number  = {float(self.rga_num):.4g}"
                     f"   (pairing {self._pairing_text()})")
        lines.append("  RGA:")
        lines.extend("    " + row for row in self._matrix_rows(
            np.asarray(self.RGA), self.u_names, self.y_names))
        if self.Gd is not None:
            lines.append("  scaled disturbance gain (|.| > 1 needs control):")
            lines.extend("    " + row for row in self._matrix_rows(
                np.asarray(self.Gd), self.d_names, self.y_names))
            dc = np.atleast_1d(np.asarray(self.dist_cond))
            ur = np.abs(np.asarray(self.u_required))
            lines.append("  per disturbance: "
                         + ", ".join(
                             f"{n}: gamma_d={dc[k]:.3g}, "
                             f"max input move={ur[:, k].max():.3g}"
                             for k, n in enumerate(self.d_names)))
        if found:
            lines.append(f"  {len(found)} findings "
                         f"({len(self.errors)} error, "
                         f"{len(self.warnings)} warning)")
            lines.extend(f"    {f}" for f in found)
        else:
            lines.append("  no findings")
        return "\n".join(lines)

    def _pairing_text(self) -> str:
        return ", ".join(f"{self.y_names[i]}-{self.u_names[j]}"
                         for i, j in enumerate(self.pairing))

    @staticmethod
    def _matrix_rows(M: np.ndarray, col_names: Sequence[str],
                     row_names: Sequence[str]) -> list[str]:
        width = max(11, max((len(c) for c in col_names), default=11) + 2)
        rows = ["".ljust(16) + "".join(c.rjust(width) for c in col_names)]
        for i, rn in enumerate(row_names):
            rows.append(rn[:15].ljust(16)
                        + "".join(f"{M[i, j]:{width}.4g}"
                                  for j in range(M.shape[1])))
        return rows

    def warn(self, stacklevel: int = 2) -> "OperabilityReport":
        """Emit each finding as an ``OperabilityWarning``.  Returns self."""
        import warnings as _w

        from difflow.operability.scaling import OperabilityWarning
        for f in self.findings:
            _w.warn(str(f), OperabilityWarning, stacklevel=stacklevel)
        return self

    def __repr__(self) -> str:
        if not _concrete(self.msv):
            return (f"OperabilityReport(traced, n_y={self.n_y}, "
                    f"n_u={self.n_u}, n_d={self.n_d})")
        return (f"OperabilityReport(msv={float(self.msv):.4g}, "
                f"cond={float(self.cond):.4g}, "
                f"rga_num={float(self.rga_num):.4g}, "
                f"n_d={self.n_d}, scaled={self.scaled})")


def _diagnose(rep: "OperabilityReport") -> list[Finding]:
    """Turn a concrete report into findings.  See :attr:`findings`."""
    found: list[Finding] = []
    G = np.asarray(rep.G, dtype=float)
    R = np.asarray(rep.RGA, dtype=float)

    if not rep.scaled:
        found.append(Finding(
            kind="unscaled", severity="warning", block="plant",
            value=float("nan"),
            detail=("this screen used unit spans, so sigma_min, the condition "
                    "number and the disturbance gains below are in mixed "
                    "engineering units and cannot be compared against 1. "
                    "Only the RGA is scaling-invariant and safe to read as "
                    "it stands. Declare Scaling(u_span=..., y_span=..., "
                    "d_span=...) with the available input moves and the "
                    "largest acceptable control errors.")))

    if not np.all(np.isfinite(G)):
        n_bad = int((~np.isfinite(G)).sum())
        found.append(Finding(
            kind="non_finite", severity="error", block="plant",
            value=float(n_bad),
            detail=(f"{n_bad} non-finite entries in the gain matrix; nothing "
                    "below means anything. The usual cause is a log(0), "
                    "sqrt(negative) or 0/0 inside the flowsheet at this "
                    "operating point, or a unit that has not converged. "
                    "Re-centre the screen at a feasible point.")))
        return found

    rank = int(float(rep.rank))
    full = min(rep.n_y, rep.n_u)
    if rank < full:
        found.append(Finding(
            kind="singular", severity="error", block="plant",
            value=float(rank),
            detail=(f"the gain matrix has rank {rank} of {full}: the inputs "
                    f"move the outputs in only {rank} independent "
                    "directions, so no controller — however designed — can "
                    "hold all of these outputs at independent setpoints. Two "
                    "of the chosen inputs act through the same path, or an "
                    "output combination is uncontrollable at steady state. "
                    "Change the structure: add a manipulated variable, or "
                    "give up one setpoint. The RGA below was formed with a "
                    "pseudo-inverse and its rows do not sum to 1.")))
    elif rep.n_u < rep.n_y:
        found.append(Finding(
            kind="underactuated", severity="error", block="plant",
            value=float(rep.n_u),
            detail=(f"{rep.n_y} controlled variables and only {rep.n_u} "
                    "manipulated variables. At steady state at most "
                    f"{rep.n_u} of them can be held independently; the RGA is "
                    "the non-square one, whose columns sum to 1 while its "
                    "rows do not — a row summing to well under 1 names an "
                    "output that no input combination really controls.")))
    elif rep.n_u > rep.n_y:
        found.append(Finding(
            kind="overactuated", severity="warning", block="plant",
            value=float(rep.n_u),
            detail=(f"{rep.n_u} manipulated variables for {rep.n_y} "
                    "controlled variables, so there is steady-state freedom "
                    "left over. The non-square RGA rows sum to 1 but the "
                    "columns do not, so a small column entry means that input "
                    "carries little of the job, not that it is a bad pairing; "
                    "use the spare inputs for an economic objective rather "
                    "than reading the extra columns as pairings.")))

    msv = float(rep.msv)
    cond = float(rep.cond)
    if rep.scaled and np.isfinite(msv) and msv < MSV_TOL:
        found.append(Finding(
            kind="weak_direction", severity="warning", block="plant",
            value=msv,
            detail=(f"sigma_min of the scaled gain is {msv:.3g} < "
                    f"{MSV_TOL:g}: there is a direction in output space that "
                    f"the inputs can only move {msv:.3g} of the way across "
                    "its acceptable range, using their full travel. Control "
                    "cannot create gain that the steady state does not have "
                    "— widen an input range, add an input, or relax the "
                    "output tolerance that is driving this.")))

    if np.isfinite(cond) and cond > COND_TOL:
        found.append(Finding(
            kind="directional", severity="warning", block="plant",
            value=cond,
            detail=(f"condition number {cond:.3g} > {COND_TOL:g}: the plant "
                    "responds strongly to some input combinations and weakly "
                    "to others. Decentralised loops will fight, and the weak "
                    "direction is the one most sensitive to model error. "
                    "This is a reason to prefer a multivariable controller, "
                    "or to change the design so the directions are more "
                    "even.")))

    neg = negative_pairings(R, pairing=rep.pairing, is_rga=True)
    for i, j in neg:
        found.append(Finding(
            kind="rga_negative", severity="error",
            variable=f"{rep.y_names[i]}-{rep.u_names[j]}",
            value=float(R[i, j]),
            detail=(f"relative gain {R[i, j]:.3g} is negative: the gain from "
                    f"{rep.u_names[j]!r} to {rep.y_names[i]!r} changes sign "
                    "when the other loops close. Pairing here gives a system "
                    "that is unstable with all loops closed, with this loop "
                    "alone, or whenever another loop saturates — under "
                    "integral control that is structural, not a tuning "
                    "problem. Re-pair; see report.suggested_pairing().")))

    if rep.pairing:
        peak = max((abs(float(R[i, j])) for i, j in enumerate(rep.pairing)),
                   default=0.0)
        if peak > RGA_TOL and not neg:
            found.append(Finding(
                kind="rga_interaction", severity="warning", block="plant",
                value=peak,
                detail=(f"largest paired relative gain is {peak:.3g} > "
                        f"{RGA_TOL:g}: the loops interact strongly and the "
                        "pairing is very sensitive to gain error — a small "
                        "relative error in one gain can change the sign of "
                        "the effective one. Consider a different pairing, "
                        "decoupling, or a multivariable controller.")))

    if rep.Gd is not None:
        Gd = np.asarray(rep.Gd, dtype=float)
        dc = np.atleast_1d(np.asarray(rep.dist_cond, dtype=float))
        ur = np.abs(np.asarray(rep.u_required, dtype=float))
        for k, name in enumerate(rep.d_names):
            gk = float(np.max(np.abs(Gd[:, k]))) if Gd.size else 0.0
            need = float(ur[:, k].max()) if ur.size else 0.0
            if rep.scaled and need > 1.0:
                worst = rep.u_names[int(np.argmax(ur[:, k]))]
                found.append(Finding(
                    kind="disturbance_infeasible", severity="error",
                    variable=name, value=need,
                    detail=(f"rejecting a full-size excursion of {name!r} "
                            f"needs {need:.3g} times the available range of "
                            f"{worst!r} at steady state. The inputs do not "
                            "span the direction this disturbance pushes, so "
                            "the output goes off-spec whatever the "
                            f"controller does (gamma_d = {dc[k]:.3g}, "
                            f"scaled gain {gk:.3g}). Add or widen an input, "
                            "or reduce the disturbance at source.")))
            elif rep.scaled and gk > GD_TOL:
                found.append(Finding(
                    kind="disturbance_gain", severity="warning",
                    variable=name, value=gk,
                    detail=(f"scaled disturbance gain {gk:.3g} > {GD_TOL:g}: "
                            f"{name!r} moves an output past its acceptable "
                            "error, so it has to be rejected by control "
                            "rather than absorbed. Feasible here — it costs "
                            f"{need:.3g} of an input's range — but it sets a "
                            "bandwidth requirement the dynamics must meet "
                            f"(gamma_d = {dc[k]:.3g}).")))
            if rep.scaled and gk > GD_TOL and dc[k] > COND_TOL:
                found.append(Finding(
                    kind="disturbance_direction", severity="warning",
                    variable=name, value=float(dc[k]),
                    detail=(f"disturbance condition number {dc[k]:.3g} > "
                            f"{COND_TOL:g}: {name!r} pushes along the plant's "
                            "weak direction, which is the expensive one to "
                            "reject. Rejection uses the input combination "
                            "the plant is least able to deliver, so it is "
                            "unusually sensitive to model error.")))

    return sorted(found, key=lambda f: f.severity != "error")


def screen(model: Callable[..., Array] | Array,
           u0: Any = None, d0: Any = None, *,
           scaling: Scaling,
           Gd: Any = None,
           u_names: Sequence[str] | None = None,
           y_names: Sequence[str] | None = None,
           d_names: Sequence[str] | None = None,
           pairing: Sequence[int] | None = None,
           mode: str = "auto",
           rcond: float = RCOND) -> OperabilityReport:
    """Screen a design for steady-state controllability.

    Computes ``G = dy/du`` and, when disturbances are given, ``G_d = dy/dd``
    by AD, scales both, and derives every measure in this package from them.
    The whole call is two Jacobians and a few SVDs, is pure, and is traceable,
    so it can be evaluated inside a design objective:

    ``objective(design) = -profit(design) + w * relu(1 - screen(...).msv)``

    Args:
        model: A pure JAX callable ``fn(u) -> y``, or ``fn(u, d) -> y`` when
            ``d0`` is given.  A difflow flowsheet function qualifies: its
            flash, recycle and unit solves are implicitly differentiated, so
            the Jacobian returned is the *converged* steady-state gain.  A
            precomputed raw gain matrix is also accepted, with ``Gd`` for the
            disturbance gain.
        u0: Operating point of the manipulated variables.
        d0: Nominal disturbance values.  Their presence is what makes the
            screen compute disturbance measures; the *magnitudes* that matter
            are ``scaling.d_span``, not these values.
        scaling: Required.  :class:`~difflow.operability.scaling.Scaling`
            declaring the available input moves, the acceptable control
            errors and the expected disturbance sizes.  There is no default,
            because a default would be a guess at the one thing that decides
            whether any of these numbers mean anything;
            ``Scaling.unscaled(n_u, n_y)`` is the deliberate escape hatch and
            marks the report accordingly.
        Gd: Precomputed raw disturbance gain, used only when ``model`` is a
            matrix rather than a callable.
        u_names: Names of the manipulated variables.
        y_names: Names of the controlled variables.
        d_names: Names of the disturbances.
        pairing: ``pairing[i]`` is the input paired with output ``i``.
            Defaults to the diagonal pairing, which is what the RGA number
            and the negative-relative-gain check are then reported against.
        mode: AD mode, ``"auto"``, ``"rev"`` or ``"fwd"``.
        rcond: Relative singular-value cutoff for the pseudo-inverse.

    Returns:
        An :class:`OperabilityReport`.

    Example:
        >>> import jax.numpy as jnp
        >>> from difflow.operability import Scaling, screen
        >>> def plant(u, d):
        ...     return jnp.array([u[0] - 0.7 * u[1] + 2.0 * d[0],
        ...                       0.6 * u[0] + u[1]])
        >>> sc = Scaling(u_span=[1.0, 1.0], y_span=[1.0, 1.0], d_span=[0.1])
        >>> rep = screen(plant, jnp.zeros(2), jnp.zeros(1), scaling=sc,
        ...              u_names=["reflux", "boilup"],
        ...              y_names=["x_D", "x_B"], d_names=["feed"])
        >>> rep.ok
        True

    Note:
        Steady state only.  A design that passes this screen can still be
        undone by right-half-plane zeros, dead time or actuator dynamics; a
        design that fails it cannot be rescued by any controller, which is
        what makes the screen worth running first and worth running early.
    """
    if not isinstance(scaling, Scaling):
        raise TypeError(
            "screen requires a Scaling. Every measure it returns except the "
            "RGA is a magnitude, and a magnitude in mixed engineering units "
            "is meaningless: build Scaling(u_span=..., y_span=..., "
            "d_span=...), or Scaling.unscaled(n_u, n_y) to say explicitly "
            "that you are not scaling.")

    scaling.warn_if_unscaled(stacklevel=3)
    if Gd is not None and callable(model):
        raise ValueError(
            "Gd is for the matrix form of screen. With a callable model the "
            "disturbance gain is obtained by AD — pass fn(u, d) and d0.")
    if callable(model):
        G = gain_matrix(model, u0, d0, scaling=scaling, mode=mode)
        Gd_s = (disturbance_gain(model, u0, d0, scaling=scaling, mode=mode)
                if d0 is not None else None)
    else:
        G = scaling.scale_gain(model)
        Gd_s = None if Gd is None else scaling.scale_disturbance(Gd)

    n_y, n_u = int(G.shape[0]), int(G.shape[1])
    u_names = _names(u_names, n_u, "u")
    y_names = _names(y_names, n_y, "y")
    n_d = 0 if Gd_s is None else int(Gd_s.shape[1])
    d_names = _names(d_names, n_d, "d")

    if pairing is None:
        pair = tuple(range(min(n_y, n_u)))
    else:
        pair = tuple(int(p) for p in pairing)
        if len(pair) > n_y or any(p < 0 or p >= n_u for p in pair):
            raise ValueError(
                f"pairing must give at most one input index per output "
                f"(n_y={n_y}, n_u={n_u}), got {list(pair)}. An underactuated "
                f"plant can pair only {min(n_y, n_u)} loops.")

    R = rga(G, rcond)
    s = singular_values(G, assume_scaled=True)
    idx_y = jnp.arange(len(pair))
    rga_pairs = R[idx_y, jnp.asarray(pair, dtype=int)] if pair else s[:0]

    dist_cond = u_required = None
    if Gd_s is not None:
        dist_cond = disturbance_condition_number(G, Gd_s, assume_scaled=True,
                                                 rcond=rcond)
        u_required = required_input_move(G, Gd_s, assume_scaled=True,
                                         rcond=rcond)

    return OperabilityReport(
        G=G,
        RGA=R,
        svals=s,
        msv=s[-1],
        cond=condition_number(G, assume_scaled=True),
        rga_num=rga_number(G, pairing=pair, rcond=rcond),
        rga_pairs=rga_pairs,
        rank=jnp.sum(s > rcond * s[0]).astype(float),
        Gd=Gd_s,
        dist_cond=dist_cond,
        u_required=u_required,
        u_names=u_names,
        y_names=y_names,
        d_names=d_names,
        pairing=pair,
        scaled=bool(scaling.explicit),
        scaling_note=scaling.note,
    )


_REPORT_CHILDREN = ("G", "RGA", "svals", "msv", "cond", "rga_num",
                    "rga_pairs", "rank", "Gd", "dist_cond", "u_required")
_REPORT_AUX = ("u_names", "y_names", "d_names", "pairing", "scaled",
               "scaling_note")


def _report_flatten(rep: OperabilityReport):
    children = tuple(getattr(rep, name) for name in _REPORT_CHILDREN)
    aux = tuple(getattr(rep, name) for name in _REPORT_AUX)
    return children, aux


def _report_unflatten(aux, children):
    kwargs = dict(zip(_REPORT_CHILDREN, children))
    kwargs.update(dict(zip(_REPORT_AUX, aux)))
    return OperabilityReport(**kwargs)


jax.tree_util.register_pytree_node(OperabilityReport, _report_flatten,
                                   _report_unflatten)
