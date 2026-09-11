"""Feasibility restoration: what to do when the subproblem is infeasible.

An elastic spec cannot make an LP infeasible — that is what the slack is for.
An *inelastic* one can, and inelastic is the right setting for anything
physical: a mass balance violated at a price is not a plan, it is fiction.

So the two go together, and the second needs this module. When the LP comes
back infeasible the trust-region loop's only instinct is to shrink the radius,
and shrinking a region that already excludes the feasible set can only exclude
it harder. The run then grinds down to ``radius_min`` and reports
``lp_infeasible`` with every decision still sitting on its starting value —
a correct report of a dead end that no amount of patience escapes.

The standard remedy is a phase-one problem: stop optimising, and minimise
infeasibility instead.

    minimize    sum(a)
    subject to  A_eq x  = b_eq
                A_ub x - a <= b_ub
                lb <= x <= ub,  a >= 0

Two choices in that statement are deliberate.

**The equality rows are not relaxed.** They are the delta-vector model rows
(``y = y0 + J (u - u0)``) and the link rows, and both are *definitional*: given
``u`` inside its bounds there is always a ``y`` that satisfies them. An
equality-infeasible subproblem therefore means the model itself is broken —
contradictory links, a degenerate block — and relaxing it would bury that
under an artificial variable instead of reporting it. Only the spec rows,
which are the caller's requirements rather than the model's structure, get
artificials.

**The restored point is judged on the nonlinear model.** The phase-one LP
minimises *predicted* infeasibility over a Taylor model; whether the point it
returns is really less infeasible is a question for the caller's own blocks,
and :meth:`~difflow.planning.planner.DeltaBasePlanner.solve` asks them. This is
the same rule the module states for the acceptance test and for scoring
violations, applied to restoration: never believe the LP about the plant.

Restoration widens the trust region rather than shrinking it. Infeasibility
usually means the feasible set is *outside* the current box — in the worked
case that produced this module, a tank's opening level was pinned to zero by a
spec while the region was clipped around the midpoint of its bounds — so more
room is the remedy and less room is the disease.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from difflow.planning.lp import LPModel, LPSolution

__all__ = ["restoration_model", "restoration_violation"]


def restoration_model(lp: LPModel) -> LPModel:
    """Build the phase-one model that minimises predicted infeasibility.

    Every *spec* row gains a non-negative artificial column that lets it be
    violated, and the objective becomes the sum of those artificials. The
    original objective is dropped entirely: restoration is not a trade-off
    against profit, it is a search for any feasible point at all.

    Structural inequality rows -- the SOS2 adjacency of a piecewise block --
    are left alone for the same reason the equalities are. They define the
    model rather than constrain it on the caller's behalf.

    Rows that already carry an elastic slack get an artificial too. Their
    slack is priced in the *original* objective, which restoration discards,
    so without one they would be free to absorb violation that the phase-one
    objective never sees.

    Args:
        lp: The infeasible model. Not modified.

    Returns:
        A new :class:`~difflow.planning.lp.LPModel` whose optimal objective is
        the smallest total inequality violation achievable inside the bounds
        and equalities. It is feasible whenever ``lb <= ub`` and the equality
        rows are consistent, so an infeasible *restoration* model is a report
        that the model structure, not the specs, is at fault.

    Example:
        >>> from difflow.planning.restoration import restoration_model
        >>> phase1 = restoration_model(lp)          # doctest: +SKIP
        >>> sol = phase1.solve()                    # doctest: +SKIP
        >>> sol.lp_objective                        # doctest: +SKIP
        0.0
    """
    n_rows = int(lp.A_ub.shape[0]) if lp.A_ub.size else 0
    n_cols = lp.n_cols

    # Not every inequality row is the caller's to relax. The SOS2 adjacency
    # rows of a piecewise block say "lambda_k may only be nonzero on the
    # chosen interval", which is the definition of the encoding and not a
    # requirement anyone asked for -- exactly the argument that keeps the
    # model and link equalities out of phase one. Handing one an artificial
    # would let restoration buy a lower predicted violation by breaking the
    # piecewise model, and report the discount as progress.
    relaxable = [i for i, name in enumerate(lp.ub_names)
                 if name.startswith("spec[")]

    if n_rows == 0 or not relaxable:
        # Nothing to relax: infeasibility is in the bounds, the equalities or
        # the structural rows, and an artificial column would not touch any
        # of them.
        return replace(lp, c=np.zeros(n_cols), objective_offset=0.0)

    n_art = len(relaxable)
    names = [f"artificial[{lp.ub_names[i]}]" for i in relaxable]
    columns = list(lp.columns) + names

    c = np.concatenate([np.zeros(n_cols), np.ones(n_art)])
    lb = np.concatenate([lp.lb, np.zeros(n_art)])
    ub = np.concatenate([lp.ub, np.full(n_art, np.inf)])

    # A_ub x - a <= b_ub, with a column only for the rows phase one may relax.
    relax_block = np.zeros((n_rows, n_art))
    relax_block[relaxable, np.arange(n_art)] = -1.0
    A_ub = np.hstack([lp.A_ub, relax_block])
    A_eq = (np.hstack([lp.A_eq, np.zeros((lp.A_eq.shape[0], n_art))])
            if lp.A_eq.size else lp.A_eq)

    return replace(
        lp, columns=columns, c=c, A_ub=A_ub, A_eq=A_eq,
        lb=lb, ub=ub, sense=1.0, objective_offset=0.0,
        # The artificials are continuous even in a MILP: they measure
        # violation, they are not a decision anybody makes.
        integer_cols=list(lp.integer_cols),
    )


def restoration_violation(solution: LPSolution) -> float:
    """Total predicted infeasibility at a restoration solution.

    Args:
        solution: The solution of a :func:`restoration_model`.

    Returns:
        The sum of the artificial variables, or ``inf`` when the phase-one
        problem itself could not be solved — which indicates a structurally
        infeasible model rather than an over-constrained one.
    """
    if not solution.success or solution.x is None:
        return float("inf")
    artificials = [i for i, name in enumerate(solution.model.columns)
                   if name.startswith("artificial[")]
    if not artificials:
        return 0.0
    return float(np.sum(np.maximum(solution.x[artificials], 0.0)))
