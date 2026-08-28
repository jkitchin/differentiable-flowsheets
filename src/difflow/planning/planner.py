"""The delta-base planner: an LP inside a trust region.

The loop is the classical trust-region model-management scheme, specialised to
a first-order model built by AD:

1. Linearise every block at the current point (the delta vectors).
2. Solve the LP inside a trust region to get a proposal.
3. **Evaluate the caller's own nonlinear blocks at that proposal.**
4. Accept only if the realised merit improved as much as the LP promised;
   otherwise shrink the region and retry.

Step 3 is the whole point and is not optional.  Without it — the "recursion"
heuristic that commercial planning systems use — repeated re-linearisation
walks the iterate far outside the region where any Taylor model is valid and
the LP confidently returns a plan that the real model does not support.  Pass
``accept_test=False`` to reproduce that failure mode deliberately; it exists so
the difference can be measured, not so it can be used.

The convergence theory is Eason and Biegler, "A trust region filter method for
glass box/black box optimization", AIChE J 62(9), 2016
(doi:10.1002/aic.15325).  Their first-order consistency requirement — that the
surrogate match the true model's value *and gradient* at the trust-region
centre — is satisfied exactly, not approximately, by an AD Taylor model.  A
delta vector obtained by one-at-a-time finite differencing satisfies it only to
truncation error, which is why "recursion" has no comparable guarantee.

Two further traps are encoded rather than left to the caller:

* **Realised violation, not predicted violation.**  The LP's elastic slacks
  keep it feasible, but the merit function scores the violation computed from
  the *nonlinear* blocks.  Score against the LP's own prediction and a planner
  with stale coefficients wins by running off-spec on paper.
* **Vertex seeding.**  Levers like ethane recovery versus rejection sit at a
  bound and switch discretely with prices.  A single interior start converges
  to whichever corner it happens to face, so the planner seeds the loop from
  bound vertices of the most price-sensitive decisions as well.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.planning.assemble import build_lp
from difflow.planning.linearize import (
    Linearization, check_phase_transition, linearize_block,
)
from difflow.planning.lp import LPModel, LPSolution, as_spec
from difflow.planning.network import Network, NetworkState
if TYPE_CHECKING:  # pragma: no cover - annotation only
    from difflow.planning.health import HealthReport

from difflow.planning.piecewise import (
    PiecewiseData, PiecewiseSpec, sample_piecewise,
)


@dataclass
class TrustRegionOptions:
    """Trust-region loop settings.

    Attributes:
        radius: Initial radius, as a fraction of each variable's bound range.
        radius_min: Stop when the radius falls below this.
        radius_max: Cap on the radius.
        eta_accept: Accept a step when the merit ratio ``rho`` reaches this.
        eta_expand: Expand the region when ``rho`` reaches this and the step
            hit the trust-region boundary.
        shrink: Multiplier applied on rejection.
        expand: Multiplier applied on a very good step.
        max_iter: Maximum linearise/solve/evaluate cycles.
        tol: Relative predicted-improvement threshold for convergence.
    """

    radius: float = 0.3
    radius_min: float = 1e-7
    radius_max: float = 1.0
    eta_accept: float = 0.1
    eta_expand: float = 0.75
    shrink: float = 0.5
    expand: float = 2.0
    max_iter: int = 100
    tol: float = 1e-8


@dataclass
class Iteration:
    """One trust-region cycle, recorded for inspection.

    Attributes:
        index: Iteration number.
        radius: Radius used for the subproblem.
        merit: Realised merit at the incumbent before the step.
        predicted: Merit the LP promised at the proposal.
        realised: Merit the nonlinear blocks delivered at the proposal.
        rho: ``(realised - merit) / (predicted - merit)``.
        accepted: Whether the step was taken.
        lp_status: Solver message.
        decisions: The proposal.
        phase_warnings: Phase-boundary messages raised for this proposal.
    """

    index: int
    radius: float
    merit: float
    predicted: float
    realised: float
    rho: float
    accepted: bool
    lp_status: str
    decisions: np.ndarray
    phase_warnings: list[str] = field(default_factory=list)


class DeltaBasePlanner:
    """Plan a block network with AD-generated delta vectors.

    Args:
        network: The :class:`~difflow.planning.network.Network` to plan.
        prices: ``{qualified variable: price}``.  Linear in the variables,
            which is what keeps the model an LP.  Negative prices charge for
            a lever or a utility.
        specs: Constraints as :class:`~difflow.planning.lp.Spec` objects or
            ``(variable, op, rhs)`` tuples.
        radius: Initial trust-region radius, a fraction of each variable's
            bound range.
        penalty: Default cost per unit of spec violation, used both for the
            LP's elastic slacks and for scoring realised violations.
        sense: ``"max"`` (default) or ``"min"``.
        accept_test: Judge each proposal against the nonlinear blocks.  Leave
            this on.  Setting it to ``False`` reproduces the unguarded
            re-linearisation heuristic and is provided for comparison only.
        vertex_seeding: Also start the loop from bound vertices of the most
            price-sensitive decisions, so bang-bang levers are not decided by
            the starting point.
        max_vertices: Cap on seeded vertices.
        modifiers: Optional ``{block: Modifiers}`` plant-model corrections;
            see :mod:`difflow.planning.modifiers`.
        theta: Optional ``{block: parameter dict}`` override.
        piecewise: :class:`~difflow.planning.piecewise.PiecewiseSpec` entries
            requesting SOS2 piecewise-linear models instead of single delta
            vectors.  This makes the subproblem a MILP; see
            :mod:`difflow.planning.piecewise`.
        options: A :class:`TrustRegionOptions` for the finer settings.
        warn_phase: Emit :class:`~difflow.planning.linearize.PhaseBoundaryWarning`
            when a proposal moves a block across a phase boundary.

    Example:
        >>> planner = DeltaBasePlanner(
        ...     net, prices={"ngl.NGL_C2": 12.0, "power.Power": 40.0},
        ...     specs=[("ngl.T_colfeed", "<=", 213.0)], radius=0.3)
        >>> res = planner.solve()
        >>> res.plan["ngl.ethane_recovery"]
        0.92...
    """

    def __init__(self,
                 network: Network,
                 prices: Mapping[str, float],
                 specs: Sequence[Any] = (),
                 radius: float = 0.3,
                 penalty: float = 1e4,
                 sense: str = "max",
                 accept_test: bool = True,
                 vertex_seeding: bool = True,
                 max_vertices: int = 8,
                 modifiers: Mapping[str, Any] | None = None,
                 theta: Mapping[str, Mapping[str, Any]] | None = None,
                 piecewise: Sequence[PiecewiseSpec] | None = None,
                 options: TrustRegionOptions | None = None,
                 warn_phase: bool = True):
        if sense not in ("max", "min"):
            raise ValueError(f"sense must be 'max' or 'min', got {sense!r}")
        self.network = network
        self.prices = dict(prices)
        self.specs = [as_spec(s, penalty) for s in specs]
        self.penalty = float(penalty)
        self.sense = sense
        self.accept_test = bool(accept_test)
        self.vertex_seeding = bool(vertex_seeding)
        self.max_vertices = int(max_vertices)
        self.modifiers = dict(modifiers) if modifiers else {}
        self.theta = {k: dict(v) for k, v in theta.items()} if theta else None
        self.piecewise = list(piecewise or [])
        self.warn_phase = bool(warn_phase)
        self.options = options or TrustRegionOptions()
        if options is None:
            self.options.radius = float(radius)

        known = set(network.input_names) | set(network.output_names)
        unknown = sorted(set(self.prices) - known)
        if unknown:
            raise KeyError(
                f"price(s) given for unknown variable(s) {unknown}. Variables "
                "are qualified as '<block>.<name>'.")
        for s in self.specs:
            bad = sorted(set(s.coeffs) - known)
            if bad:
                raise KeyError(f"spec {s!r} references unknown variable(s) {bad}")
        for pw in self.piecewise:
            network.block(pw.block).u_index(pw.variable)

        self._merit_sign = 1.0 if sense == "max" else -1.0
        self._eval_net: Network | None = None
        self._eval_key: Any = None

    # -- the planner's own model -----------------------------------------

    @property
    def evaluation_network(self) -> Network:
        """The network as the planner's own model sees it.

        When modifiers are in force the planner's model is the *corrected*
        model, so that is what the acceptance test must be judged against —
        not the uncorrected blocks, and never a "plant". Without modifiers
        this is the caller's network unchanged.
        """
        key = tuple(sorted((k, id(v)) for k, v in self.modifiers.items()))
        if self._eval_net is None or self._eval_key != key:
            self._eval_net = self._build_corrected_network()
            self._eval_key = key
        return self._eval_net

    def _build_corrected_network(self) -> Network:
        """Substitute modifier-corrected callables into a copy of the network."""
        if not self.modifiers:
            return self.network
        blocks = []
        for b in self.network.blocks:
            mod = self.modifiers.get(b.name)
            blocks.append(b if mod is None
                          else replace(b, fn=_corrected_fn(b, mod)))
        return Network(blocks,
                       [(l.source, l.target) for l in self.network.links])

    # -- scoring ---------------------------------------------------------

    def objective_value(self, values: Mapping[str, Any]) -> float:
        """The priced linear objective, from a variable dict."""
        return float(sum(float(p) * float(values[v])
                         for v, p in self.prices.items()))

    def violations(self, values: Mapping[str, Any]) -> dict[str, float]:
        """Realised violation of each spec, from a variable dict."""
        return {s.name: float(s.violation(values)) for s in self.specs}

    def penalty_cost(self, violations: Mapping[str, float]) -> float:
        """Total charge for the given violations."""
        total = 0.0
        for s in self.specs:
            pen = self.penalty if s.penalty is None else s.penalty
            total += abs(float(pen)) * float(violations.get(s.name, 0.0))
        return total

    def score(self, decisions) -> dict[str, Any]:
        """Score a plan against the *nonlinear* blocks.

        This is the honest scoring rule: constraint violation is measured by
        evaluating the caller's model at the proposed decisions, never by
        reading the LP's own slack variables.  A planner whose coefficients
        have gone stale will predict feasibility it does not achieve; charging
        for predicted violation would reward exactly that.

        Args:
            decisions: Free-decision array or dict.

        Returns:
            Dict with ``objective``, ``violations``, ``penalty``, ``merit``
            and the full ``state``.
        """
        state = self.evaluation_network.evaluate(decisions, self.theta)
        return self._score_state(state)

    def _score_state(self, state: NetworkState) -> dict[str, Any]:
        obj = self.objective_value(state.values)
        viol = self.violations(state.values)
        pen = self.penalty_cost(viol)
        return {
            "objective": obj,
            "violations": viol,
            "total_violation": float(sum(viol.values())),
            "penalty": pen,
            "merit": self._merit_sign * obj - pen,
            "state": state,
        }

    def describe(self, u0=None, width: int = 78) -> str:
        """State the planning problem in words and symbols.

        Three questions come before any result: what is being planned,
        what may be changed to get it, and what may not be violated on
        the way.  This answers them from the planner itself, so the
        statement cannot drift away from the model that is solved.

        Args:
            u0: Operating point to report as the starting decisions;
                the network's nominal start when omitted.
            width: Wrap width for the objective.

        Returns:
            The problem statement as text.

        Example:
            >>> print(planner.describe())            # doctest: +SKIP
        """
        import textwrap

        net = self.network
        names = net.decision_names
        lb, ub = (np.asarray(a, dtype=float) for a in net.decision_bounds())
        start = np.asarray(net.decision_array(
            net.decision_start() if u0 is None else u0), dtype=float)
        radius = self.options.radius

        terms = []
        for var, price in self.prices.items():
            sign = "-" if price < 0 else "+"
            terms.append(f"{sign} {abs(float(price)):g} {var}")
        head = terms[0][2:] if terms and terms[0].startswith("+ ") else ""
        body = " ".join([head] + terms[1:]) if terms else "0"

        sense = "maximise" if self.sense == "max" else "minimise"
        lines = [f"Planning problem: {sense} the priced objective", ""]
        lines += ["    " + row for row in
                  textwrap.wrap(body, width=width - 4) or ["0"]]
        lines += ["",
                  f"  by choosing {len(names)} decision"
                  f"{'' if len(names) == 1 else 's'} "
                  f"(a trust region lets each move {radius:g} of its bound "
                  "range per cycle):"]
        for i, name in enumerate(names):
            step = radius * (ub[i] - lb[i])
            window = f"[{lb[i]:g}, {ub[i]:g}]"
            lines.append(f"    {name:<26s} start {start[i]:>10.4g}   "
                         f"in {window:<20s} step +/- {step:.3g}")

        linked = [l for l in net.links]
        outputs = net.output_names
        lines += ["",
                  f"  everything else follows from the blocks "
                  f"({len(outputs)} outputs, "
                  f"{len(linked)} link{'' if len(linked) == 1 else 's'}):"]
        for link in linked:
            lines.append(f"    {link.target:<26s} = {link.source}"
                         "   (not a free decision)")
        constrained = {k for s in self.specs for k in s.coeffs}
        unpriced = [y + (" (constrained)" if y in constrained else "")
                    for y in outputs
                    if y not in self.prices
                    and not any(l.source == y for l in linked)]
        if unpriced:
            lines.append(f"    not priced: {', '.join(unpriced)}")

        lines += ["", "  subject to:"]
        if not self.specs:
            lines.append("    no specs; only the decision bounds above")
        for spec in self.specs:
            lhs = (next(iter(spec.coeffs)) if len(spec.coeffs) == 1
                   else " + ".join(f"{c:g} {k}" for k, c in spec.coeffs.items()))
            pen = self.penalty if spec.penalty is None else spec.penalty
            how = (f"elastic, {abs(float(pen)):g} per unit of violation"
                   if spec.elastic else "hard")
            back = (f", back-off {spec.backoff:g}" if spec.backoff else "")
            lines.append(f"    {spec.name:<16s} {lhs} {spec.op} "
                         f"{spec.rhs:g}   ({how}{back})")
        lines += [
            "    the decision bounds above, in every cycle",
            "",
            "  Violation is scored by evaluating the nonlinear blocks at the "
            "proposal,",
            "  never by reading the LP's own slacks, and a proposal is "
            "accepted only",
            "  when those blocks confirm the improvement the LP predicted."
            if self.accept_test else
            "  WARNING: accept_test=False -- proposals are taken on the LP's "
            "word alone.",
        ]
        return "\n".join(lines)

    # -- linearisation ---------------------------------------------------

    def linearize(self, state: NetworkState) -> dict[str, Linearization]:
        """Linearise every block at the given network state.

        Delta vectors come back from one AD pass per block, whose cost is
        independent of the number of decisions.  Any modifier corrections are
        folded in here so the LP sees a corrected first-order model.
        """
        lins = {}
        for name in self.network.order:
            block = self.network.block(name)
            th = None if self.theta is None else self.theta.get(name)
            lin = linearize_block(block, state.u[name], th)
            mod = self.modifiers.get(name)
            if mod is not None:
                lin = mod.apply(lin)
            lins[name] = lin
        return lins

    def sample_piecewise(self, state: NetworkState
                         ) -> dict[str, PiecewiseData]:
        """Sample every requested piecewise model at the current state.

        Each spec costs two batched ``vmap`` dispatches — one for the values
        and one for the Jacobians — regardless of how many breakpoints it
        asks for.
        """
        out: dict[str, PiecewiseData] = {}
        for spec in self.piecewise:
            block = self.evaluation_network.block(spec.block)
            th = None if self.theta is None else self.theta.get(spec.block)
            out[spec.block] = sample_piecewise(
                block, spec, center=state.u[spec.block], theta=th)
        return out

    def build_lp(self, lins: Mapping[str, Linearization],
                 state: NetworkState, radius: float,
                 piecewise: Mapping[str, PiecewiseData] | None = None
                 ) -> LPModel:
        """Assemble the LP (or MILP) for one trust-region subproblem."""
        if piecewise is None and self.piecewise:
            piecewise = self.sample_piecewise(state)
        return build_lp(self.network, lins, self.prices, self.specs,
                        centers=state.u, radius=radius,
                        penalty=self.penalty, sense=self.sense,
                        piecewise=piecewise)

    def check_health(self, decisions: Any = None,
                     radius: float | None = None,
                     include_lp: bool = True) -> "HealthReport":
        """Diagnose the delta vectors and the assembled program.

        Run this before trusting a plan from a large model.  It reports dead
        levers, recycle amplification, and the constraint-matrix scale spread
        — the three ways a delta-base model degrades with size.  See
        :mod:`difflow.planning.health`.

        Args:
            decisions: Operating point to diagnose.  Defaults to the
                network's nominal start, which is where ``solve`` begins.
            radius: Trust-region radius to reason about.  Defaults to the
                planner's own initial radius.
            include_lp: Also check the assembled LP's coefficient scaling.

        Returns:
            A :class:`~difflow.planning.health.HealthReport`.

        Example:
            >>> planner.check_health().warn()
            >>> print(planner.check_health().summary())
        """
        from difflow.planning.health import check_lp_scaling, check_network_health

        r = float(self.options.radius if radius is None else radius)
        net = self.evaluation_network
        start = (net.decision_start() if decisions is None
                 else net.decision_array(decisions))
        report = check_network_health(net, decisions=start, theta=self.theta,
                                      radius=r)
        if include_lp:
            state = net.evaluate(start, self.theta)
            lp = self.build_lp(self.linearize(state), state, r)
            lp_report = check_lp_scaling(lp)
            report.findings.extend(lp_report.findings)
            report.findings.sort(key=lambda f: f.severity != "error")
            report.thresholds.update(lp_report.thresholds)
        return report

    def _decisions_from_lp(self, sol: LPSolution) -> np.ndarray:
        """Read the free decisions out of an LP solution."""
        return np.array([sol[name] for name in self.network.decision_names])

    # -- seeding ---------------------------------------------------------

    def vertex_seeds(self, base: Array | None = None,
                     max_vertices: int | None = None) -> list[np.ndarray]:
        """Bound-vertex starting points for the most price-sensitive decisions.

        Levers whose optimum is bang-bang switch corner discretely with prices.
        Ranking by ``|d objective / d u| * range`` — one reverse-mode AD pass,
        regardless of how many decisions there are — identifies which levers
        those are, and the corners of just those are enumerated.

        Args:
            base: Point at which to rank, and the value used for the decisions
                not being cornered.  Defaults to the network's nominal start.
            max_vertices: Cap on the number of vertices returned.

        Returns:
            A list of decision arrays, excluding ``base`` itself.
        """
        base = (self.network.decision_start() if base is None
                else jnp.asarray(base, dtype=float))
        lo, hi = self.network.decision_bounds()
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        n = self.network.n_decisions
        cap = self.max_vertices if max_vertices is None else int(max_vertices)
        if cap < 2 or n == 0:
            return []

        bounded = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
        if not bounded.any():
            return []

        def smooth_objective(d):
            st = self.evaluation_network.evaluate(d, self.theta)
            return sum(float(p) * st.values[v] for v, p in self.prices.items())

        try:
            g = np.asarray(jax.grad(smooth_objective)(base), dtype=float)
        except Exception:  # pragma: no cover - non-differentiable price map
            g = np.ones(n)
        influence = np.abs(g) * np.where(bounded, hi - lo, 0.0)
        influence = np.where(bounded, influence, -np.inf)

        k = max(1, int(np.floor(np.log2(cap))))
        k = min(k, int(bounded.sum()))
        idx = list(np.argsort(-influence)[:k])

        base_np = np.asarray(base, dtype=float)
        seeds: list[np.ndarray] = []
        for combo in itertools.product(*[(lo[i], hi[i]) for i in idx]):
            cand = base_np.copy()
            for i, v in zip(idx, combo):
                cand[i] = v
            if np.allclose(cand, base_np):
                continue
            if not any(np.allclose(cand, s) for s in seeds):
                seeds.append(cand)
        return seeds[:cap]

    # -- the loop --------------------------------------------------------

    def solve(self, u0: Array | Mapping[str, float] | None = None,
              seeds: Sequence[Any] | None = None) -> "PlanResult":
        """Run the trust-region loop and return the plan.

        Args:
            u0: Starting decisions.  Defaults to the network's nominal point.
            seeds: Extra starting points.  Defaults to
                :meth:`vertex_seeds` when ``vertex_seeding`` is on.

        Returns:
            A :class:`PlanResult` for the best start.
        """
        start = (self.network.decision_start() if u0 is None
                 else self.network.decision_array(u0))
        start = np.asarray(self.network.clip(start), dtype=float)

        starts: list[np.ndarray] = [start]
        if seeds is not None:
            starts.extend(np.asarray(self.network.clip(s), dtype=float)
                          for s in seeds)
        elif self.vertex_seeding:
            starts.extend(self.vertex_seeds(base=jnp.asarray(start)))

        best: PlanResult | None = None
        attempts: list[PlanResult] = []
        for s in starts:
            res = self._solve_one(s)
            attempts.append(res)
            if best is None or res.merit > best.merit:
                best = res
        assert best is not None
        best.attempts = attempts
        best.n_starts = len(starts)
        return best

    def _solve_one(self, start: np.ndarray) -> "PlanResult":
        """One trust-region run from a single starting point."""
        opts = self.options
        radius = float(opts.radius)
        net = self.evaluation_network
        state = net.evaluate(jnp.asarray(start), self.theta)
        scored = self._score_state(state)
        merit = scored["merit"]

        history: list[Iteration] = []
        phase_messages: list[str] = []
        converged = False
        reason = "max_iter"

        for it in range(opts.max_iter):
            lins = self.linearize(state)
            pw = self.sample_piecewise(state) if self.piecewise else None
            lp = self.build_lp(lins, state, radius, pw)
            sol = lp.solve()

            if not sol.success:
                history.append(Iteration(
                    index=it, radius=radius, merit=merit, predicted=float("nan"),
                    realised=float("nan"), rho=float("nan"), accepted=False,
                    lp_status=sol.message,
                    decisions=np.asarray(state.decisions)))
                radius *= opts.shrink
                if radius < opts.radius_min:
                    reason = "lp_infeasible"
                    break
                continue

            predicted = (sol.objective if self.sense == "max"
                         else -sol.objective)
            gain = predicted - merit
            chi_tol = opts.tol * (1.0 + abs(merit))
            if gain / max(radius, opts.radius_min) <= chi_tol:
                # Predicted gain per unit radius is the first-order
                # criticality measure. It can be depressed at a large radius
                # by physical bounds clipping the box, so confirm it on a
                # probe radius before declaring stationarity. The probe reuses
                # this linearisation, so it costs one LP solve and no model
                # evaluations.
                chi0 = self._criticality(lins, state, merit, opts.radius_min,
                                         pw)
                if chi0 <= chi_tol:
                    history.append(Iteration(
                        index=it, radius=radius, merit=merit,
                        predicted=predicted, realised=merit, rho=0.0,
                        accepted=False,
                        lp_status="first-order stationary",
                        decisions=np.asarray(state.decisions)))
                    converged = True
                    reason = "stationary"
                    break
                radius = max(opts.radius_min, radius * opts.shrink)
                continue

            trial = self._decisions_from_lp(sol)
            msgs = self._check_phases(lins, sol)
            phase_messages.extend(msgs)

            # Evaluate the caller's OWN nonlinear model at the proposal. The
            # acceptance test is judged here, never against a "plant".
            trial_state = net.evaluate(jnp.asarray(trial), self.theta)
            trial_scored = self._score_state(trial_state)
            realised = trial_scored["merit"]
            rho = (realised - merit) / gain

            accepted = (True if not self.accept_test
                        else rho >= opts.eta_accept)
            history.append(Iteration(
                index=it, radius=radius, merit=merit, predicted=predicted,
                realised=realised, rho=float(rho), accepted=accepted,
                lp_status=sol.message, decisions=trial.copy(),
                phase_warnings=msgs))

            if accepted:
                at_boundary = self._hit_trust_region(state, trial, radius)
                state, scored, merit = trial_state, trial_scored, realised
                if self.accept_test and rho >= opts.eta_expand and at_boundary:
                    radius = min(opts.radius_max, radius * opts.expand)
            else:
                radius *= opts.shrink
                if radius < opts.radius_min:
                    converged = True
                    reason = "radius_min"
                    break
        else:
            reason = "max_iter"

        lins = self.linearize(state)
        lp = self.build_lp(lins, state, radius)
        return PlanResult(
            planner=self, network=self.network, state=state,
            decisions=np.asarray(state.decisions, dtype=float),
            objective=scored["objective"], merit=merit,
            violations=scored["violations"], penalty=scored["penalty"],
            linearizations=lins, lp_model=lp, history=history,
            converged=converged, reason=reason, radius=radius,
            phase_warnings=phase_messages, start=start)

    def _criticality(self, lins: Mapping[str, Linearization],
                     state: NetworkState, merit: float, radius: float,
                     piecewise: Mapping[str, PiecewiseData] | None = None
                     ) -> float:
        """Predicted merit gain per unit radius on a small probe region.

        This is the first-order criticality measure of the linearised
        problem. It reuses the current delta vectors, so it costs one LP
        solve and no evaluations of the caller's blocks.
        """
        sol = self.build_lp(lins, state, radius, piecewise).solve()
        if not sol.success:
            return float("inf")
        predicted = sol.objective if self.sense == "max" else -sol.objective
        return max(0.0, predicted - merit) / max(radius, 1e-300)

    def _check_phases(self, lins: Mapping[str, Linearization],
                      sol: LPSolution) -> list[str]:
        """Warn about any block whose proposal crosses a phase boundary."""
        msgs: list[str] = []
        for name in self.network.order:
            block = self.network.block(name)
            if block.phase_fn is None:
                continue
            u_new = np.array([sol[q] for q in block.qualified_u()])
            th = None if self.theta is None else self.theta.get(name)
            msgs.extend(check_phase_transition(
                block, lins[name], jnp.asarray(u_new), th,
                warn=self.warn_phase))
        return msgs

    def _hit_trust_region(self, state: NetworkState, trial: np.ndarray,
                          radius: float, rtol: float = 1e-6) -> bool:
        """Whether the step reached the trust region rather than a real bound.

        Expanding the region is only warranted when the region — not a
        physical limit — is what stopped the step.
        """
        lo, hi = self.network.decision_bounds()
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        center = np.asarray(state.decisions, dtype=float)
        span = hi - lo
        scale = np.where(np.isfinite(span), span,
                         np.maximum(np.abs(center), 1.0))
        step = radius * scale
        moved = np.abs(np.asarray(trial, dtype=float) - center)
        interior = step < (span - 1e-12)  # TR strictly inside the bounds
        return bool(np.any(interior & (moved >= step * (1.0 - rtol))
                           & (step > 0)))

    def __repr__(self) -> str:
        return (f"DeltaBasePlanner(blocks={len(self.network.blocks)}, "
                f"n_decisions={self.network.n_decisions}, "
                f"specs={len(self.specs)}, accept_test={self.accept_test})")


@dataclass
class PlanResult:
    """A plan, the model it came from, and the sensitivity of the plan itself.

    Attributes:
        planner: The planner that produced this.
        network: The block network.
        state: Nonlinear network state at the plan.
        decisions: Free-decision array.
        objective: Priced objective at the plan, from the nonlinear model.
        merit: Objective less the charge for realised violations.
        violations: Realised violation per spec.
        penalty: Total charge for those violations.
        linearizations: Delta vectors at the plan.
        lp_model: The final :class:`~difflow.planning.lp.LPModel`.
        history: The trust-region cycles.
        converged: Whether the loop reached a stationary point.
        reason: Why the loop stopped.
        radius: Final trust-region radius.
        phase_warnings: Phase-boundary diagnostics raised during the solve.
        start: The starting point this run used.
        attempts: One :class:`PlanResult` per seeded start (best-run only).
        n_starts: Number of starts tried.
    """

    planner: DeltaBasePlanner
    network: Network
    state: NetworkState
    decisions: np.ndarray
    objective: float
    merit: float
    violations: dict[str, float]
    penalty: float
    linearizations: dict[str, Linearization]
    lp_model: LPModel
    history: list[Iteration]
    converged: bool
    reason: str
    radius: float
    phase_warnings: list[str] = field(default_factory=list)
    start: np.ndarray | None = None
    attempts: list["PlanResult"] = field(default_factory=list)
    n_starts: int = 1

    @property
    def plan(self) -> dict[str, float]:
        """The optimal decisions, ``{qualified name: value}``."""
        return {n: float(v) for n, v
                in zip(self.network.decision_names, self.decisions)}

    @property
    def values(self) -> dict[str, float]:
        """Every network variable at the plan."""
        return self.state.as_dict()

    @property
    def delta_vectors(self) -> dict[str, np.ndarray]:
        """The ``J`` blocks actually used, ``{block: (n_y, n_u) array}``."""
        return {k: np.asarray(v.J) for k, v in self.linearizations.items()}

    @property
    def pyomo_model(self):
        """The emitted Pyomo model, for reuse or inspection.

        Requires Pyomo; see :meth:`difflow.planning.lp.LPModel.to_pyomo`.
        """
        if getattr(self, "_pyomo", None) is None:
            self._pyomo = self.lp_model.to_pyomo()
        return self._pyomo

    @property
    def total_violation(self) -> float:
        """Sum of realised spec violations."""
        return float(sum(self.violations.values()))

    @property
    def n_iterations(self) -> int:
        """Number of trust-region cycles in the winning run."""
        return len(self.history)

    @property
    def n_accepted(self) -> int:
        """Number of accepted steps."""
        return sum(1 for h in self.history if h.accepted)

    def delta_table(self, block: str | None = None) -> str:
        """Render the delta vectors as text."""
        names = [block] if block else list(self.linearizations)
        out = []
        for n in names:
            b = self.network.block(n)
            out.append(self.linearizations[n].as_table(b.u_names, b.y_names))
        return "\n\n".join(out)

    def plan_sensitivity(self, wrt: str = "prices", **kwargs):
        """Sensitivity of the *plan* — not just the plan.

        See :func:`difflow.planning.sensitivity.plan_sensitivity`.

        Args:
            wrt: ``"prices"`` for ``d(plan)/d(price)``, or ``"theta"`` for
                ``d(plan)/d(model or design parameter)``.
            **kwargs: Passed through to the underlying routine.

        Returns:
            A :class:`~difflow.planning.sensitivity.PlanSensitivity`.
        """
        from difflow.planning.sensitivity import plan_sensitivity
        return plan_sensitivity(self, wrt=wrt, **kwargs)

    def summary(self) -> str:
        """A human-readable report of the plan."""
        lines = [
            f"Plan ({'converged' if self.converged else 'NOT converged'}, "
            f"{self.reason}, {self.n_iterations} iterations, "
            f"{self.n_starts} start(s))",
            f"  objective      {self.objective:14.6g}",
            f"  merit          {self.merit:14.6g}"
            f"   (objective less realised-violation charge)",
            f"  violation      {self.total_violation:14.6g}"
            f"   (from the nonlinear model, not the LP)",
            f"  final radius   {self.radius:14.6g}",
            "  decisions:",
        ]
        for name, value in self.plan.items():
            lines.append(f"    {name:<32s} {value:14.6g}")
        if any(v > 0 for v in self.violations.values()):
            lines.append("  realised violations:")
            for k, v in self.violations.items():
                if v > 0:
                    lines.append(f"    {k:<32s} {v:14.6g}")
        if self.phase_warnings:
            lines.append(f"  phase-boundary warnings: {len(self.phase_warnings)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"PlanResult(objective={self.objective:.6g}, "
                f"merit={self.merit:.6g}, converged={self.converged}, "
                f"iterations={self.n_iterations})")


def _corrected_fn(block: "Any", modifiers: "Any") -> Callable:
    """Wrap a block's callable with its modifier corrections."""

    def corrected(u, *args):
        y = block.evaluate(u, args[0] if args else None)
        return y + jnp.asarray(modifiers.eps) + jnp.asarray(modifiers.lam) @ (
            jnp.atleast_1d(jnp.asarray(u, dtype=float))
            - jnp.asarray(modifiers.u_ad))

    corrected.__name__ = f"{block.name}_corrected"
    return corrected
