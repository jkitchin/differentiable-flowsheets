"""Flexibility analysis: does the design still work when the feed does not?

An optimizer hands back a design that sits exactly on its binding
specification, because that is where the money is.  A design placed exactly on
a purity floor spends roughly half its campaigns on the wrong side of it.
This module measures that, and separates the two reasons it happens.

The measurements
----------------
The feasibility function of Halemane and Grossmann,

.. math:: \\psi(d) = \\max_{\\theta \\in T}\\ \\min_u\\ \\max_j\\
          f_j(d, u, \\theta)

is the worst constraint residual left over after the controls have been
re-optimized against whatever realization ``theta`` the set ``T`` produced.
The design is feasible over the whole set exactly when ``psi(d) <= 0``.

The flexibility index of Swaney and Grossmann is the largest scaling of that
set the design survives::

    F = max { delta : psi(d, delta) <= 0 }

``F >= 1`` covers the envelope you were given; ``F = 0.6`` covers 60% of it.
The index alone rarely changes a decision, so
:func:`flexibility_index` also reports **which vertex** runs out first, **which
constraint** binds there, and **how much room every other vertex has** --- a
single low direction names a specification to renegotiate or a control to add.

Feed uncertainty and parameter uncertainty are different bills
--------------------------------------------------------------
This is the distinction the module is built to make measurable.

*Feed uncertainty is answerable.*  You learn today's feed before you have to
run it, so the controls can be re-optimized against it.  Its cost is only what
survives that re-optimization --- which is exactly what ``psi`` computes, and
the part that recourse removes is reported as the recourse credit.

*Parameter uncertainty is not answerable.*  An equilibrium constant is never
revealed; you cannot schedule a control move against a constant you do not
know.  The whole propagated swing lands on the constraint and must be bought
as margin in advance.  That is back-off, sized as ``kappa * sigma`` from
:func:`difflow.uncertainty.propagate_covariance`, exactly as
:mod:`difflow.planning.backoff` sizes it for a plan.

:func:`uncertainty_penalties` charges both and prints them side by side,
because they have different remedies: feed penalty is bought down with
controls and instruments, parameter back-off is bought down with experiments.

When the worst case is too conservative
---------------------------------------
The corner where every parameter is simultaneously extreme may be absurdly
improbable.  :func:`expected_feasibility` replaces the guarantee with a
probability and a chance-constrained margin, using the *same* inner solve at
sampled realizations instead of at vertices, and reports which constraint is
doing the failing and how often.

Quick start
-----------
::

    import jax.numpy as jnp
    from difflow.flexibility import feasibility_function, flexibility_index

    # constraints, written so that f_j <= 0 means "satisfied"
    def f(d, u, theta):
        return jnp.array([u[0] - d[0],        # duty cannot exceed the design
                          theta[0] - u[0]])   # duty must cover the feed

    res = feasibility_function(f, [2.0], {"feed": (1.0, 0.5)},
                               {"duty": (0.0, 10.0)})
    res.psi, res.feasible, res.binding_constraint

    idx = flexibility_index(f, [2.0], {"feed": (1.0, 0.5)},
                            {"duty": (0.0, 10.0)})
    print(idx.summary())

What is here, and what is not
-----------------------------
Vertex enumeration is the default and is exact whenever the critical
realization is a vertex --- the usual case for constraints monotone in the
parameters.  ``method="continuous"`` adds a projected ascent over ``theta``,
seeded at the best vertex, for critical points interior to the set; it
converges to a KKT point of the outer maximization and is therefore a local
guarantee, not a global one.

Deliberately absent: a design *optimization* under flexibility constraints
(the two-stage stochastic program), and mixed-integer formulations of ``psi``.
Both are large problems in their own right; this module measures a given
design, and :mod:`difflow.planning` is where optimization lives.

References:
    Halemane and Grossmann, AIChE J. 29 (1983) 425, doi:10.1002/aic.690290312.
    Swaney and Grossmann, AIChE J. 31 (1985) 621, 631,
    doi:10.1002/aic.690310412, doi:10.1002/aic.690310413.
    Grossmann, Calfa and Garcia-Herreros, Comput. Chem. Eng. 70 (2014) 22,
    doi:10.1016/j.compchemeng.2013.12.013.
"""

from difflow.flexibility.diagram import (
    draw_flexibility_region, draw_penalty_split,
)
from difflow.flexibility.feasibility import (
    METHODS, FeasibilityResult, feasibility_function, feasibility_value,
    inner_value, vertex_values,
)
from difflow.flexibility.index import (
    FlexibilityResult, flexibility_index, vertex_limits,
)
from difflow.flexibility.inner import (
    DEFAULT_OPTIONS, SolverOptions, box_adam, minimax_controls, minimax_value,
    smooth_max,
)
from difflow.flexibility.penalties import PenaltyReport, uncertainty_penalties
from difflow.flexibility.sets import (
    MAX_VERTICES, NO_CONTROLS, ControlSpec, UncertaintySet, as_control_spec,
    as_uncertainty_set,
)
from difflow.flexibility.stochastic import (
    DISTRIBUTIONS, StochasticFeasibilityResult, expected_feasibility,
    sample_set,
)

__all__ = [
    # Sets and recourse
    "UncertaintySet",
    "ControlSpec",
    "as_uncertainty_set",
    "as_control_spec",
    "NO_CONTROLS",
    "MAX_VERTICES",
    # Feasibility function
    "feasibility_function",
    "feasibility_value",
    "FeasibilityResult",
    "inner_value",
    "vertex_values",
    "METHODS",
    # Flexibility index
    "flexibility_index",
    "vertex_limits",
    "FlexibilityResult",
    # Stochastic counterpart
    "expected_feasibility",
    "sample_set",
    "StochasticFeasibilityResult",
    "DISTRIBUTIONS",
    # Feed vs parameter penalties
    "uncertainty_penalties",
    "PenaltyReport",
    # Inner solver
    "minimax_value",
    "minimax_controls",
    "smooth_max",
    "box_adam",
    "SolverOptions",
    "DEFAULT_OPTIONS",
    # Drawings
    "draw_flexibility_region",
    "draw_penalty_split",
]
