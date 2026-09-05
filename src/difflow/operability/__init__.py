"""Steady-state operability and controllability screening from AD gains.

A flowsheet can be optimal and uncontrollable.  The question this module asks
is whether the manipulated variables have enough *independent* influence on
the controlled variables to hold them against the disturbances the plant will
actually see — a steady-state, gradient question, and therefore one a
differentiable flowsheet answers almost for free.

Everything follows from two Jacobians of the converged flowsheet:

.. math::

    G = \\frac{\\partial y}{\\partial u}, \\qquad
    G_d = \\frac{\\partial y}{\\partial d}

Computed by AD these are exact and cost a constant multiple of one model
evaluation.  Obtained the usual way — finite differences through a
sequential-modular simulator — they cost ``2n`` re-solves and carry the
simulator's convergence tolerance as noise, which is why controllability
screening is normally done on a linear model fitted *separately* from the
design model, after the design is fixed.  Here it can be done on the design
model itself, inside the optimisation loop, which is the point of issue #199.

What you get from them
----------------------
:func:`rga`
    The relative gain array, ``G * pinv(G).T``, for input-output pairing and
    as a warning of interaction.  Scaling-invariant.  A **negative** relative
    gain on a proposed pairing is the headline result: that loop is unstable
    under integral control in some configuration, structurally.
:func:`min_singular_value`, :func:`condition_number`
    How hard the plant is to control in its *worst* direction, and how
    directional it is.  Both are magnitudes and both are meaningless unscaled.
:func:`disturbance_gain`, :func:`disturbance_condition_number`,
:func:`required_input_move`
    Whether the inputs span the directions the disturbances push, and what
    fraction of their available travel rejecting each disturbance costs.
:func:`screen`
    All of the above in one traceable call, returning an
    :class:`OperabilityReport` that reads its own numbers back as findings.

Scaling is not optional
-----------------------
The measures above are magnitudes, and a magnitude in mixed engineering units
measures the unit system.  :class:`Scaling` therefore takes three engineering
judgements — the available move in each input, the largest *acceptable
control error* in each output, and the expected size of each disturbance — and
makes every dimensionless entry mean the same thing, so that the threshold in
every rule of thumb is the number 1.  :func:`screen` has no default scaling
and will not run without one; ``Scaling.unscaled(n_u, n_y)`` is the explicit
opt-out and stamps the resulting report with a caveat.

Example:
    >>> import jax.numpy as jnp
    >>> from difflow.operability import Scaling, screen
    >>> def column(u, d):
    ...     # toy two-point model of a distillation column: reflux and boilup
    ...     # against the two product compositions, with a feed disturbance.
    ...     L, V = u
    ...     return jnp.array([0.878 * L - 0.864 * V + 0.394 * d[0],
    ...                       1.082 * L - 1.096 * V + 0.586 * d[0]])
    >>> sc = Scaling(u_span=[1.0, 1.0], y_span=[0.01, 0.01], d_span=[0.2])
    >>> rep = screen(column, jnp.zeros(2), jnp.zeros(1), scaling=sc,
    ...              u_names=["L", "V"], y_names=["x_D", "x_B"],
    ...              d_names=["F"])
    >>> float(rep.cond) > 100          # famously ill-conditioned
    True

See also:
    :mod:`difflow.planning.health` — the same reporting pattern applied to
    the delta vectors of a planning LP.  Its ``check_delta_health`` and this
    module's ``screen`` are asking related questions of the same Jacobians.
    ``docs/operability.md`` — the narrative version, including how to read
    each finding and where the scaling usually goes wrong.
"""

from difflow.operability.gains import disturbance_gain, gain_matrix
from difflow.operability.metrics import (
    RCOND, condition_number, disturbance_condition_number, effective_rank,
    max_singular_value, min_singular_value, negative_pairings, pinv,
    required_input_move, rga, rga_number, singular_values, suggest_pairing,
)
from difflow.operability.scaling import OperabilityWarning, Scaling
from difflow.operability.screen import (
    COND_TOL, GD_TOL, MSV_TOL, RGA_TOL, OperabilityReport, screen,
)

__all__ = [
    # Scaling
    "Scaling",
    "OperabilityWarning",
    # Gains
    "gain_matrix",
    "disturbance_gain",
    # Metrics
    "rga",
    "rga_number",
    "negative_pairings",
    "suggest_pairing",
    "min_singular_value",
    "max_singular_value",
    "singular_values",
    "condition_number",
    "effective_rank",
    "pinv",
    "required_input_move",
    "disturbance_condition_number",
    # Screening
    "screen",
    "OperabilityReport",
    # Thresholds
    "MSV_TOL",
    "COND_TOL",
    "RGA_TOL",
    "GD_TOL",
    "RCOND",
]
