"""Benchmarking utilities for comparing SM and EO solvers.

Provides tools to compare sequential modular (SM) and equation-oriented
(EO) solution approaches on the same flowsheet problem.
"""

import time
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from difflow.flowsheet import Flowsheet
from difflow.eo_solver import EOSolver
from difflow.streams import get_flows, get_species


@dataclass
class SolverComparison:
    """Results from comparing SM and EO solvers.

    Attributes:
        sm_time: SM solve wall time (seconds)
        eo_time: EO solve wall time (seconds)
        sm_iterations: SM tear iteration count, from
            ``Flowsheet.last_solve_iterations``.  ``None`` for a
            recycle-free flowsheet, which does not iterate at all.
        eo_iterations: EO Newton iteration count
        max_stream_difference: Maximum absolute difference in any stream variable
        gradient_difference: Maximum difference in gradients (if computed)
        sm_residual: Final SM tear residual.
        sm_converged: Whether the SM solve met its tolerance, from
            ``Flowsheet.last_solve_converged``.
        eo_converged: Whether EO solver converged
    """
    sm_time: float
    eo_time: float
    sm_iterations: int | None
    eo_iterations: int
    max_stream_difference: float
    gradient_difference: float | None = None
    sm_residual: float | None = None
    sm_converged: bool = True
    eo_converged: bool = True


def compare_solvers(
    flowsheet: Flowsheet,
    tol: float = 1e-8,
    max_iter: int = 100,
    sm_acceleration: str = "anderson",
) -> SolverComparison:
    """Compare SM and EO solvers on a flowsheet.

    Runs both solvers on the same problem and compares results.

    ``max_stream_difference`` is only meaningful when both solvers
    converged: comparing a converged answer against a last iterate measures
    how far one of them got, not how far apart the methods are.  Check
    ``sm_converged`` and ``eo_converged`` before reading it.

    A non-converged SM solve warns as it normally would; the verdict is also
    returned, so a caller can test ``sm_converged`` instead of catching.

    Args:
        flowsheet: The flowsheet to solve
        tol: Convergence tolerance for both solvers
        max_iter: Maximum iterations for both solvers
        sm_acceleration: Acceleration method for SM solver

    Returns:
        SolverComparison with timing and accuracy metrics
    """
    # Run SM solver.  The non-convergence warning is left to fire: #249
    # made a failed recycle solve say so, and a comparison is no reason to
    # take that back.  sm_converged carries the same verdict for a caller
    # that would rather test it than catch it.
    t0 = time.time()
    sm_streams = flowsheet.solve(
        tol=tol,
        max_iter=max_iter,
        acceleration=sm_acceleration,
    )
    sm_time = time.time() - t0

    # Run EO solver
    eo_solver = EOSolver(flowsheet)
    eo_result = eo_solver.solve(
        use_sm_init=False,
        tol=tol,
        max_steps=max_iter,
    )
    eo_time = eo_result.wall_time

    # Compare stream values
    max_diff = 0.0
    for name in sm_streams:
        if name in eo_result.streams:
            sm_s = sm_streams[name]
            eo_s = eo_result.streams[name]
            for key in sm_s:
                diff = float(jnp.abs(sm_s[key] - eo_s[key]))
                max_diff = max(max_diff, diff)

    # The flowsheet records its own verdict (#249); before that this
    # function reported sm_converged=True unconditionally and sm_iterations
    # as the iteration cap, so a diverged SM solve looked like a converged
    # one that happened to disagree with EO.
    return SolverComparison(
        sm_time=sm_time,
        eo_time=eo_time,
        sm_iterations=flowsheet.last_solve_iterations,
        eo_iterations=eo_result.n_iterations,
        max_stream_difference=max_diff,
        sm_residual=flowsheet.last_solve_residual,
        sm_converged=bool(flowsheet.last_solve_converged),
        eo_converged=eo_result.converged,
    )
