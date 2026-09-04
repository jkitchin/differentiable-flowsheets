"""Bridges from difflow flowsheets to external optimization solvers.

Two views of the same model:

* **Flat NLP view** -- :func:`as_nlp` gives ``f(x)``, ``g(x)`` and a
  :class:`Bounds`, which is exactly ``pounce.jax.from_jax``'s input
  contract. :func:`solve_with_pounce` and :func:`optimize_flowsheet` run it;
  :func:`differentiable_problem` makes the whole design problem one
  differentiable node in an outer JAX computation.
* **Residual view** -- :func:`as_residual` gives ``g(u, v) = 0`` with ``u``
  the inputs and parameters and ``v`` the internal states, ready for
  ``discopt.modeling.implicit``. :func:`residual_from_system` does the same
  for a model that already *is* a residual in difflow's ``r(z; args)``
  section-scope convention (see
  :func:`difflow.eo_solver.solve_residual_system`, and
  ``difflow_ree.equilibrium.mass_action.make_section_residual``).
  :func:`as_implicit` wires either into a discopt model and refuses the one
  combination discopt cannot solve.

Two things to read before using either:

1. **Sparsity is never probed.** pounce detects sparsity at random
   ``N(0, 1)`` points unless a pattern is supplied; a process model
   evaluated at ``T = -1.3 K`` overflows or goes singular. Every adapter
   here supplies a pattern that is a superset by construction. See
   :mod:`difflow.solvers.nlp`.
2. **A difflow block in discopt is local-NLP-only, and integer or binary
   variables make the solve raise.** See
   :data:`~difflow.solvers.discopt_bridge.CUSTOMCALL_RESTRICTION`.

``pounce`` and ``discopt`` are optional; they are imported lazily inside the
functions that need them. pounce's PyPI distribution is ``pounce-solver``.
"""

from difflow.solvers.nlp import (
    Bounds,
    Decision,
    Parameter,
    Spec,
    SparsityPatternError,
    as_nlp,
    dense_hessian_pattern,
    dense_jacobian_pattern,
    require_eo_residuals,
    validate_patterns,
)
from difflow.solvers.pounce_bridge import (
    FlowsheetOptimum,
    bound_sensitivities,
    differentiable_problem,
    optimize_flowsheet,
    solve_with_pounce,
)
from difflow.solvers.residual import (
    ResidualView,
    as_residual,
    residual_from_system,
)
from difflow.solvers.discopt_bridge import (
    CUSTOMCALL_RESTRICTION,
    DiscoptIntegralityError,
    as_implicit,
    check_no_integrality,
    integer_variables,
)

__all__ = [
    # NLP view
    "as_nlp",
    "Decision",
    "Parameter",
    "Spec",
    "Bounds",
    "SparsityPatternError",
    "validate_patterns",
    "dense_jacobian_pattern",
    "dense_hessian_pattern",
    "require_eo_residuals",
    # pounce
    "solve_with_pounce",
    "optimize_flowsheet",
    "differentiable_problem",
    "bound_sensitivities",
    "FlowsheetOptimum",
    # residual view / discopt
    "as_residual",
    "residual_from_system",
    "ResidualView",
    "as_implicit",
    "check_no_integrality",
    "integer_variables",
    "DiscoptIntegralityError",
    "CUSTOMCALL_RESTRICTION",
]
