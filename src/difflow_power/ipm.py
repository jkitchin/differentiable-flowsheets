"""A primal-dual interior-point NLP solver, written in JAX.

AC-OPF is a nonconvex nonlinear program, and the field solves it with a
primal-dual interior-point method: IPOPT, MATPOWER's MIPS, KNITRO. None
of those is a JAX program, and calling out to one would end the
differentiability that is the whole point of this framework --- the
gradient of a dispatch with respect to a load, a line rating or a fuel
price is what makes an OPF useful inside a larger model. So the method
is implemented here, in JAX, following MIPS closely enough that its
convergence behaviour carries over.

The problem
-----------

.. math::

    \\min_x f(x) \\quad \\text{subject to} \\quad g(x) = 0, \\quad h(x) \\le 0

Simple bounds are not special-cased: put them in ``h``. For a network of
a few hundred buses the bound rows dominate the count but not the cost,
because the linear algebra is dense either way.

The method
----------

Slacks turn the inequalities into ``h(x) + s = 0`` with ``s > 0``, and a
log barrier keeps them positive. The perturbed KKT conditions are

.. math::

    \\nabla f + J_g^T \\lambda + J_h^T z = 0, \\quad
    g(x) = 0, \\quad h(x) + s = 0, \\quad S Z e = \\mu e

and each iteration is one Newton step on them, condensed by eliminating
``ds`` and ``dz`` into

.. math::

    \\begin{bmatrix} W + J_h^T \\Sigma J_h & J_g^T \\\\ J_g & 0
    \\end{bmatrix}
    \\begin{bmatrix} dx \\\\ d\\lambda \\end{bmatrix} = \\dots,
    \\qquad \\Sigma = S^{-1} Z

with ``W`` the exact Hessian of the Lagrangian from ``jax.hessian``.
Three things then make it actually converge on a nonconvex problem:

**Inertia correction.** At a minimum the condensed KKT matrix has
exactly ``m_eq`` negative eigenvalues. Where it does not, the Newton
direction points at a saddle rather than a minimum, and a multiple of
the identity is added to the ``(1,1)`` block until the inertia is
right. This is not optional for AC-OPF: the power flow equations are
genuinely nonconvex and an uncorrected step will happily converge to a
high-voltage solution that costs more.

**Fraction to boundary.** The step is capped so ``s`` and ``z`` keep a
fraction ``tau`` of their distance to zero, which is what keeps the
iterate interior without any explicit check.

**An l1 merit line search.** Backtracking on
``f - mu sum log s + nu (||g||_1 + ||h + s||_1)`` with the penalty above
the multiplier norm. Without it the full Newton step overshoots badly
from a flat start on a congested case.

``mu`` is reduced once the current barrier subproblem is solved, on
IPOPT's monotone schedule. Tying it to the complementarity ``s . z``
instead --- the other common choice --- deadlocks on a degenerate
problem: a step the line search rejects leaves ``s`` and ``z``
unchanged, so ``mu`` stops moving and the iteration spins in place
having in fact already converged.

Differentiating the solution
----------------------------

The iteration is a Python loop over jitted steps, not a
``lax.while_loop``, and is not itself differentiable --- deliberately.
Differentiating an iterative solver is both expensive and wrong-headed
when the answer is characterised by an equation: at the solution the
KKT system holds, so :func:`differentiable_solution` re-solves it with
``optimistix``'s Newton root find, which converges in one step from the
converged point and carries implicit-function-theorem gradients. The
cost of a gradient is then one linear solve, whatever the forward pass
took, and the answer does not depend on the iteration path.

The gradient is of the BARRIER solution at the final ``mu``, not of the
exact NLP solution. They differ by ``O(mu)``, which at the default
convergence tolerance is far below any modelling error --- but it is
also why driving ``mu`` down matters for a sensitivity even when the
primal answer already looks converged.

Scaling
-------

The linear algebra is dense: an ``(n + m_eq)`` symmetric solve plus an
eigenvalue decomposition for the inertia, each iteration. That is the
right choice up to a few hundred buses and the wrong one past a few
thousand, where a sparse factorisation with an inertia-revealing
``LDL^T`` is what production solvers use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array

from difflow.params_mixin import ParamsMixin

#: fraction-to-boundary parameter: how much of the distance to s = 0 a
#: step may consume. 0.995 is the standard aggressive value.
DEFAULT_TAU = 0.995


#: smallest slack a starting point is given, so that log(s) is finite
#: even at a point sitting exactly on a constraint
GAMMA_SLACK = 1e-2

#: inertia-correction constants, following IPOPT's Algorithm IC
DELTA_MIN = 1e-20        # floor on the primal shift
DELTA_0 = 1e-4           # first shift tried when none was needed before
KAPPA_DELTA_UP = 8.0     # escalation once a shift is known to be needed
KAPPA_DELTA_UP_FIRST = 100.0   # escalation on the very first failure
MAX_INERTIA_TRIES = 40


def _empty(n: int) -> Array:
    return jnp.zeros((n,), dtype=jnp.float64)


@dataclass(frozen=True)
class NLP:
    """A smooth nonlinear program ``min f(x) s.t. g(x) = 0, h(x) <= 0``.

    Every callable takes ``(x, params)`` and must be JAX-traceable;
    ``params`` is an arbitrary pytree threaded through unchanged, and is
    what :func:`differentiable_solution` differentiates with respect to.

    Attributes:
        objective: ``f(x, params) -> scalar``.
        equalities: ``g(x, params) -> (m_eq,)``, or ``None``.
        inequalities: ``h(x, params) -> (m_in,)``, or ``None``. The
            convention is ``h <= 0``; a lower bound ``x >= lo`` is the
            row ``lo - x``.
        n: length of ``x``.
        m_eq: number of equality rows.
        m_in: number of inequality rows.
    """

    objective: Callable[[Array, Any], Array]
    n: int
    m_eq: int = 0
    m_in: int = 0
    equalities: Callable[[Array, Any], Array] | None = None
    inequalities: Callable[[Array, Any], Array] | None = None

    def g(self, x: Array, params: Any) -> Array:
        """Equality residuals, or an empty array."""
        if self.equalities is None:
            return _empty(0)
        return self.equalities(x, params)

    def h(self, x: Array, params: Any) -> Array:
        """Inequality residuals (``<= 0`` is feasible), or an empty array."""
        if self.inequalities is None:
            return _empty(0)
        return self.inequalities(x, params)

    def lagrangian(self, x: Array, lam: Array, z: Array, params: Any) -> Array:
        """``f + lambda . g + z . h``, whose Hessian is the ``W`` block."""
        total = self.objective(x, params)
        if self.m_eq:
            total = total + jnp.dot(lam, self.g(x, params))
        if self.m_in:
            total = total + jnp.dot(z, self.h(x, params))
        return total


@dataclass
class IPMResult(ParamsMixin):
    """The outcome of an interior-point solve.

    Attributes:
        x: the primal solution.
        lam: equality multipliers. For an OPF these are the locational
            marginal prices, up to the sign and scaling the caller
            applies.
        z: inequality multipliers, non-negative. A ``z_i`` well above
            zero marks a binding constraint and prices it.
        s: slacks, ``s = -h(x)`` at convergence.
        objective: ``f(x)`` at the solution.
        converged: whether all four MIPS conditions were met.
        iterations: Newton steps taken.
        mu: the final barrier parameter. The reported solution is the
            solution of the barrier problem at this ``mu``, which
            differs from the true optimum by ``O(mu)``.
        feasibility: scaled primal infeasibility at the solution.
        stationarity: scaled dual infeasibility.
        complementarity: scaled ``s . z``.
        history: per-iteration ``(objective, feasibility, stationarity,
            complementarity, mu, step)``, for diagnosing a solve that
            stalled.
    """

    x: Array
    lam: Array
    z: Array
    s: Array
    objective: float
    converged: bool
    iterations: int
    mu: float
    feasibility: float
    stationarity: float
    complementarity: float
    history: list[dict[str, float]] = field(default_factory=list)

    @property
    def kkt(self) -> Array:
        """The stacked primal-dual vector ``(x, lam, z, s)``."""
        return jnp.concatenate([self.x, self.lam, self.z, self.s])

    def active(self, tol: float = 1e-6) -> Array:
        """Boolean mask of the inequality rows that are binding.

        A row counts as binding when its slack has collapsed, which is
        the numerically meaningful test at an interior-point solution:
        the multiplier is the price, but the slack is what says whether
        the constraint is holding the solution back.
        """
        return self.s < tol * (1.0 + jnp.max(jnp.abs(self.s)))

    def summary(self) -> str:
        state = "converged" if self.converged else "DID NOT CONVERGE"
        return (
            f"{state} in {self.iterations} iterations: f = "
            f"{self.objective:.6g}, feas {self.feasibility:.2e}, "
            f"stat {self.stationarity:.2e}, comp "
            f"{self.complementarity:.2e}, mu {self.mu:.2e}, "
            f"{int(jnp.sum(self.active()))}/{self.s.shape[0]} active"
        )

    def __repr__(self) -> str:
        return f"IPMResult({self.summary()})"


# =============================================================================
# The Newton step
# =============================================================================


def _condensed_kkt(w_hess, jac_g, jac_h, sigma_diag, delta, delta_c):
    """Assemble the regularised condensed KKT matrix."""
    n = w_hess.shape[0]
    top_left = w_hess + delta * jnp.eye(n)
    if jac_h.shape[0]:
        top_left = top_left + jac_h.T @ (sigma_diag[:, None] * jac_h)
    m_eq = jac_g.shape[0]
    if m_eq == 0:
        return top_left
    return jnp.block(
        [
            [top_left, jac_g.T],
            [jac_g, -delta_c * jnp.eye(m_eq)],
        ]
    )


def _inertia_ok(
    kkt: Array, m_eq: int, rtol: float = 1e-12, sweeps: int = 3
) -> bool:
    """True if the KKT matrix has exactly ``m_eq`` negative eigenvalues.

    At a minimum the condensed KKT matrix has inertia ``(n, m_eq, 0)``;
    anywhere else the Newton direction points at a saddle rather than a
    descent direction, and the ``(1,1)`` block has to be shifted until
    it does not. Getting this test right is the difference between a
    solver that converges on a nonconvex AC-OPF and one that stalls.

    Two things make a naive sign count fail, and both are fixed here.

    **The matrix is ill-conditioned by construction.** ``Sigma = z / s``
    diverges on the active rows as the barrier tightens, so near the
    solution the eigenvalues span twelve decades or more, and any
    tolerance scaled to the largest of them swallows the ones that carry
    the answer. Diagonal equilibration first --- a few Ruiz sweeps ---
    brings the spectrum back to O(1). This is exact, not a heuristic:
    inertia is invariant under congruence ``D K D`` for positive
    diagonal ``D`` (Sylvester's law), so the scaled matrix has the same
    inertia as the original and a spectrum a sign test can read.

    **Zero eigenvalues are real.** A weakly convex problem --- linear
    generation costs, say --- has genuinely zero curvature in some
    directions at the optimum, and roundoff puts those on an arbitrary
    side of zero. The test is therefore a BAND: the count of
    clearly-negative eigenvalues must not exceed ``m_eq``, and the count
    of negative-or-numerically-zero ones must not fall short of it. That
    accepts a degenerate optimum, and still rejects a saddle, whose
    extra negative eigenvalue is unambiguous.

    Args:
        kkt: the symmetric condensed KKT matrix.
        m_eq: number of equality constraints, hence the number of
            negative eigenvalues a minimum must have.
        rtol: tolerance on the EQUILIBRATED spectrum, where the largest
            eigenvalue is O(1).
        sweeps: Ruiz equilibration sweeps. Three is plenty; the
            conditioning improves geometrically and the cost is one
            row-max per sweep.
    """
    scaled = kkt
    for _ in range(sweeps):
        row_max = jnp.sqrt(
            jnp.maximum(jnp.max(jnp.abs(scaled), axis=1), 1e-300)
        )
        scaled = scaled / row_max[:, None] / row_max[None, :]
    eigs = jnp.linalg.eigvalsh(scaled)
    tol = rtol * jnp.max(jnp.abs(eigs))
    n_negative = int(jnp.sum(eigs < -tol))
    n_nonpositive = int(jnp.sum(eigs < tol))
    return n_negative <= m_eq <= n_nonpositive


def _fraction_to_boundary(v: Array, dv: Array, tau: float) -> Array:
    """Largest ``alpha <= 1`` with ``v + alpha dv >= (1 - tau) v``."""
    if v.shape[0] == 0:
        return jnp.asarray(1.0)
    ratio = jnp.where(dv < 0.0, -tau * v / dv, jnp.inf)
    return jnp.minimum(1.0, jnp.min(ratio))


def _merit(nlp, x, s, params, mu, nu):
    """``f - mu sum log s + nu (||g||_1 + ||h + s||_1)``."""
    val = nlp.objective(x, params)
    if nlp.m_in:
        val = val - mu * jnp.sum(jnp.log(s))
        val = val + nu * jnp.sum(jnp.abs(nlp.h(x, params) + s))
    if nlp.m_eq:
        val = val + nu * jnp.sum(jnp.abs(nlp.g(x, params)))
    return val


class _Evaluation(NamedTuple):
    """Everything one iteration needs, from one traced evaluation."""

    objective: Array
    grad_f: Array
    jac_g: Array
    jac_h: Array
    g_val: Array
    h_val: Array
    hessian: Array


def _compile(nlp: NLP):
    """Trace and compile the derivative graph ONCE per solve.

    Every iteration needs the objective, its gradient, both constraint
    Jacobians and the Lagrangian Hessian at the current iterate.
    Building them one call at a time re-traces the whole model on every
    iteration, which for an AC-OPF means re-tracing the admittance
    assembly a hundred times over. Fusing them into a single jitted
    function makes the iteration cost one XLA call.
    """
    n = nlp.n

    def evaluate(x, lam, z, params) -> _Evaluation:
        return _Evaluation(
            objective=nlp.objective(x, params),
            grad_f=jax.grad(nlp.objective)(x, params),
            jac_g=(
                jax.jacobian(nlp.g)(x, params)
                if nlp.m_eq else jnp.zeros((0, n))
            ),
            jac_h=(
                jax.jacobian(nlp.h)(x, params)
                if nlp.m_in else jnp.zeros((0, n))
            ),
            g_val=nlp.g(x, params),
            h_val=nlp.h(x, params),
            hessian=jax.hessian(nlp.lagrangian, argnums=0)(x, lam, z, params),
        )

    return jax.jit(evaluate)


def _conditions(nlp, ev: _Evaluation, x, lam, z, s):
    """The three scaled KKT errors, plus the dual residual itself.

    Feasibility, stationarity and complementarity, each scaled by the
    magnitude of what it is measured against, so the tolerances mean the
    same thing on a problem whose variables are O(1) and one whose are
    O(1000). The unscaled dual residual is returned too because the
    Newton step needs it as a right-hand side and it is already formed.
    """
    dual = ev.grad_f + ev.jac_g.T @ lam + ev.jac_h.T @ z
    scale_x = 1.0 + jnp.max(jnp.abs(x))
    scale_mult = 1.0 + jnp.maximum(
        jnp.max(jnp.abs(lam), initial=0.0), jnp.max(jnp.abs(z), initial=0.0)
    )
    feas = jnp.maximum(
        jnp.max(jnp.abs(ev.g_val), initial=0.0),
        jnp.max(ev.h_val, initial=0.0),
    ) / scale_x
    stat = jnp.max(jnp.abs(dual)) / scale_mult
    comp = jnp.dot(s, z) / scale_x if nlp.m_in else jnp.asarray(0.0)
    return feas, stat, comp, dual


def solve_nlp(
    nlp: NLP,
    x0: Array,
    params: Any = None,
    *,
    max_iterations: int = 200,
    tol_feasibility: float = 1e-8,
    tol_stationarity: float = 1e-6,
    tol_complementarity: float = 1e-8,
    tau: float = DEFAULT_TAU,
    mu0: float = 1.0,
    kappa_epsilon: float = 10.0,
    kappa_mu: float = 0.2,
    theta_mu: float = 1.5,
    max_backtracks: int = 25,
    verbose: bool = False,
) -> IPMResult:
    """Solve an NLP by the primal-dual interior-point method.

    Args:
        nlp: the problem.
        x0: starting point. It need NOT be feasible --- the method
            drives ``g`` and ``h`` to feasibility along with optimality
            --- but a point near the feasible region converges far
            faster, which is why the OPF starts from a power flow.
        params: pytree threaded to every callable.
        max_iterations: cap on Newton steps.
        tol_feasibility: on scaled ``max(|g|, max(h))``.
        tol_stationarity: on the scaled Lagrangian gradient.
        tol_complementarity: on scaled ``s . z``. These three ARE the
            KKT conditions, and meeting them is what convergence means
            here; there is deliberately no "the objective stopped
            moving" test, which a degenerate problem can satisfy at a
            point that is not a solution.
        tau: minimum fraction-to-boundary parameter. The value actually
            used is ``max(tau, 1 - mu)``, so the method becomes more
            aggressive as the barrier tightens, which is what lets the
            last few iterations take full steps.
        mu0: initial barrier parameter.
        kappa_epsilon: the barrier subproblem is declared solved, and
            ``mu`` reduced, once the KKT error falls below
            ``kappa_epsilon * mu``.
        kappa_mu, theta_mu: the reduction
            ``mu <- max(mu_min, min(kappa_mu mu, mu ** theta_mu))``, a
            linear rate far from the solution and a superlinear one
            near it.
        max_backtracks: line-search halvings before the shortest step is
            taken anyway.
        verbose: print the iteration log.

    Returns:
        An :class:`IPMResult`. Non-convergence is REPORTED, not raised:
        an OPF that cannot be solved usually means the case is
        infeasible --- not enough capacity behind a constraint --- and
        the iterate says where it got stuck.

    Example:
        >>> nlp = NLP(objective=lambda x, p: jnp.sum(x ** 2), n=2,
        ...           m_in=1,
        ...           inequalities=lambda x, p: jnp.array([1.0 - x[0]]))
        >>> res = solve_nlp(nlp, jnp.array([2.0, 2.0]))
        >>> bool(abs(res.x[0] - 1.0) < 1e-6 and abs(res.x[1]) < 1e-6)
        True
    """
    x = jnp.asarray(x0, dtype=jnp.float64)
    if x.shape != (nlp.n,):
        raise ValueError(f"x0 has shape {x.shape}, expected {(nlp.n,)}")

    evaluate = _compile(nlp)
    merit = jax.jit(lambda xx, ss, pp, mm, nn: _merit(nlp, xx, ss, pp, mm, nn))

    h0 = nlp.h(x, params)
    s = jnp.maximum(-h0, GAMMA_SLACK) if nlp.m_in else _empty(0)
    z = (mu0 / s) if nlp.m_in else _empty(0)
    lam = _empty(nlp.m_eq)
    mu = float(mu0)
    # The floor on mu has to make the complementarity tolerance
    # REACHABLE. At the barrier solution every pair satisfies
    # s_i z_i = mu, so the measure the tolerance is applied to is
    # m_in * mu / (1 + max|x|). A floor of mu = tol would leave that
    # measure a factor m_in too large, and the solve would spin at the
    # floor forever having in fact already converged.
    mu_min = tol_complementarity / (10.0 * max(nlp.m_in, 1))

    history: list[dict[str, float]] = []
    converged = False
    delta_last = 0.0
    steps = 0

    for _iteration in range(max_iterations):
        ev = evaluate(x, lam, z, params)
        feas, stat, comp, r_dual = _conditions(nlp, ev, x, lam, z, s)
        grad_f, jac_g, jac_h = ev.grad_f, ev.jac_g, ev.jac_h
        g_val, h_val = ev.g_val, ev.h_val

        if verbose:
            print(
                f"  it {steps:3d}  f {float(ev.objective): .8e}  "
                f"feas {float(feas):.2e}  stat {float(stat):.2e}  "
                f"comp {float(comp):.2e}  mu {mu:.2e}  delta {delta_last:.1e}"
            )
        if (
            feas < tol_feasibility
            and stat < tol_stationarity
            and comp < tol_complementarity
        ):
            converged = True
            break

        # -- barrier schedule ---------------------------------------------
        # Reduce mu once the CURRENT barrier subproblem is solved, not
        # from the complementarity itself. Tying mu to s.z deadlocks on a
        # degenerate problem: a rejected step leaves s and z unchanged,
        # so mu stops moving and the iteration spins. Judging the
        # subproblem instead guarantees mu -> 0 whatever the steps do.
        for _ in range(64):
            e_mu = jnp.maximum(
                jnp.max(jnp.abs(r_dual)),
                jnp.maximum(
                    jnp.max(jnp.abs(g_val), initial=0.0),
                    jnp.max(jnp.abs(s * z - mu), initial=0.0),
                ),
            )
            if float(e_mu) > kappa_epsilon * mu or mu <= mu_min:
                break
            mu = max(mu_min, min(kappa_mu * mu, mu ** theta_mu))

        r_slack = h_val + s if nlp.m_in else _empty(0)
        r_comp = (s * z - mu) if nlp.m_in else _empty(0)
        r_pri = g_val

        # -- Newton system ------------------------------------------------
        w_hess = ev.hessian
        sigma_diag = (z / s) if nlp.m_in else _empty(0)

        rhs_x = -r_dual
        if nlp.m_in:
            rhs_x = rhs_x - jac_h.T @ ((z * r_slack - r_comp) / s)
        rhs = jnp.concatenate([rhs_x, -r_pri]) if nlp.m_eq else rhs_x

        # Inertia correction, IPOPT's Algorithm IC. The starting delta
        # is a THIRD of the last successful one rather than the floor:
        # a problem that needed regularisation once usually needs it
        # again, and re-climbing from 1e-8 by decades every iteration
        # both costs eigendecompositions and lands on a far larger
        # delta than necessary, which corrupts the Newton direction.
        delta = 0.0 if delta_last == 0.0 else max(DELTA_MIN, delta_last / 3.0)
        delta_c = 0.0
        step = None
        for _try in range(MAX_INERTIA_TRIES):
            kkt = _condensed_kkt(
                w_hess, jac_g, jac_h, sigma_diag, delta, delta_c
            )
            if _inertia_ok(kkt, nlp.m_eq):
                step = jnp.linalg.solve(kkt, rhs)
                if bool(jnp.all(jnp.isfinite(step))):
                    break
                step = None
            if delta == 0.0:
                delta = DELTA_0 if delta_last == 0.0 else DELTA_MIN
            else:
                delta *= KAPPA_DELTA_UP if delta_last else KAPPA_DELTA_UP_FIRST
            if nlp.m_eq and delta_c == 0.0:
                delta_c = 1e-8 * mu ** 0.25
        if step is None:
            break
        delta_last = delta

        dx = step[: nlp.n]
        dlam = step[nlp.n:] if nlp.m_eq else _empty(0)
        if nlp.m_in:
            ds = -r_slack - jac_h @ dx
            dz = (-r_comp - z * ds) / s
        else:
            ds = dz = _empty(0)

        # -- step length ---------------------------------------------------
        tau_eff = max(tau, 1.0 - mu)
        alpha_p = _fraction_to_boundary(s, ds, tau_eff)
        alpha_d = _fraction_to_boundary(z, dz, tau_eff)

        nu = 1.1 * jnp.maximum(
            jnp.max(jnp.abs(lam + alpha_d * dlam), initial=0.0),
            jnp.max(jnp.abs(z + alpha_d * dz), initial=0.0),
        ) + 1.0
        merit0 = merit(x, s, params, mu, nu)
        alpha = float(alpha_p)
        accepted = False
        for _ in range(max_backtracks):
            trial_x = x + alpha * dx
            trial_s = s + alpha * ds
            if not nlp.m_in or bool(jnp.all(trial_s > 0.0)):
                trial_merit = merit(trial_x, trial_s, params, mu, nu)
                if bool(jnp.isfinite(trial_merit)) and trial_merit < merit0:
                    accepted = True
                    break
            alpha *= 0.5
        if not accepted:
            # No decrease anywhere along the direction. Take the short
            # step to keep moving, and force more regularisation next
            # time so the direction itself changes; if that never helps,
            # the iteration cap reports non-convergence rather than
            # pretending.
            delta_last = max(DELTA_0, delta_last * KAPPA_DELTA_UP)

        x = x + alpha * dx
        if nlp.m_in:
            s = s + alpha * ds
            z = z + alpha_d * dz
        if nlp.m_eq:
            lam = lam + alpha_d * dlam

        steps += 1
        history.append(
            {
                "objective": float(ev.objective),
                "feasibility": float(feas),
                "stationarity": float(stat),
                "complementarity": float(comp),
                "mu": mu,
                "step": alpha,
                "regularization": delta,
            }
        )

    ev = evaluate(x, lam, z, params)
    feas, stat, comp, _ = _conditions(nlp, ev, x, lam, z, s)
    return IPMResult(
        x=x,
        lam=lam,
        z=z,
        s=s,
        objective=float(ev.objective),
        converged=converged,
        iterations=steps,
        mu=mu,
        feasibility=float(feas),
        stationarity=float(stat),
        complementarity=float(comp),
        history=history,
    )


# =============================================================================
# Differentiating the solution
# =============================================================================


def kkt_residuals(
    w: Array, nlp: NLP, params: Any, mu: float
) -> Array:
    """The perturbed KKT system as one residual vector.

    ``w = (x, lam, z, s)``. Its root is the solution of the barrier
    problem at ``mu``; :func:`solve_nlp` finds it by Newton with
    safeguards, and :func:`differentiable_solution` re-solves it here to
    attach gradients.
    """
    n, m_eq, m_in = nlp.n, nlp.m_eq, nlp.m_in
    x = w[:n]
    lam = w[n:n + m_eq]
    z = w[n + m_eq:n + m_eq + m_in]
    s = w[n + m_eq + m_in:]

    grad_f = jax.grad(nlp.objective)(x, params)
    blocks = [grad_f]
    if m_eq:
        jac_g = jax.jacobian(nlp.g)(x, params)
        blocks[0] = blocks[0] + jac_g.T @ lam
    if m_in:
        jac_h = jax.jacobian(nlp.h)(x, params)
        blocks[0] = blocks[0] + jac_h.T @ z
    if m_eq:
        blocks.append(nlp.g(x, params))
    if m_in:
        blocks.append(nlp.h(x, params) + s)
        blocks.append(s * z - mu)
    return jnp.concatenate(blocks)


def differentiable_solution(
    nlp: NLP,
    w0: Array,
    params: Any,
    mu: float,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-10,
    max_steps: int = 20,
) -> Array:
    """Re-solve the KKT system from a converged point, differentiably.

    ``optimistix``'s Newton converges in a step or two from ``w0`` and
    attaches implicit-function-theorem gradients, so
    ``jax.jacobian(...)(params)`` gives the exact sensitivity of the
    solution to the parameters at one linear solve's cost.

    Args:
        nlp: the problem.
        w0: converged ``(x, lam, z, s)``, e.g. :attr:`IPMResult.kkt`.
        params: the parameters to differentiate with respect to.
        mu: the barrier parameter ``w0`` was converged at. The
            sensitivity is of the barrier solution, which differs from
            the true one by ``O(mu)``.
        rtol, atol, max_steps: passed to the Newton solve.

    Returns:
        The refined ``(x, lam, z, s)``, differentiable in ``params``.

    Example:
        >>> res = solve_nlp(nlp, x0, params)
        >>> jac = jax.jacobian(
        ...     lambda p: differentiable_solution(
        ...         nlp, res.kkt, p, res.mu)[:nlp.n]
        ... )(params)                                 # doctest: +SKIP
    """
    solver = optx.Newton(rtol=rtol, atol=atol)
    sol = optx.root_find(
        lambda w, args: kkt_residuals(w, nlp, args, mu),
        solver,
        w0,
        args=params,
        max_steps=max_steps,
        throw=False,
    )
    return sol.value
