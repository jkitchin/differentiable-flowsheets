"""Moving-horizon estimation for dynamic flowsheets.

:mod:`difflow.dynamic` integrates a flowsheet forward and
:mod:`difflow.reconciliation` reconciles a steady-state data set. Between
them sits the question a running plant actually asks: *given a stream of
noisy, sparse and late measurements, where is the process now?* This
module answers it, by solving at each sampling time

.. math::

    \\min_{x_0, w} \\; \\|x_0 - \\bar x\\|^2_{P^{-1}}
        + \\sum_k \\|w_k\\|^2_{Q^{-1}}
        + \\sum_k \\|y_k - h(x_k)\\|^2_{R^{-1}}
    \\quad \\text{s.t.} \\quad x_{k+1} = f(x_k, u_k, w_k),
    \\; x_k \\in [\\ell, u],

over the last few intervals, with an *arrival cost* summarising
everything before them. Reference: Rao, Rawlings and Mayne, *IEEE Trans.
Automat. Contr.* 48 (2003) 246, doi:10.1109/TAC.2002.808470.

Why an optimisation rather than a filter? Because the case that
motivates this is slow, high-order, sparsely measured and delayed --- a
solvent-extraction train whose organic inventory turns over in a day,
whose feed disturbance takes many residence times to cross thirty stages
with recycle, and whose revealing assay comes back hours later on a
shift cadence. An extended Kalman filter linearises once at the current
mean and commits to it, cannot express "this concentration is not
negative", and has no way to accept a measurement drawn six hours ago. A
horizon can do all three. The filter is here too --- :func:`run_ekf` ---
as the baseline the horizon has to beat and as the source of the arrival
cost's covariance.

Two things are built in rather than bolted on:

* **Multi-rate and delayed measurements.** A :class:`Measurement`
  carries both the time its sample was *taken* and the time it was
  *reported*, and :func:`build_window` places it against the state at
  the former. A channel not sampled at a given time is a reading of
  infinite variance, which keeps every array shape fixed under ``jit``
  and reuses reconciliation's convention that ``sigma = inf`` means
  "estimate this, do not fit it".
* **Joint state and parameter estimation.**
  :func:`augment_parameters` appends drifting parameters to the state as
  a random walk, which is how degradation is detected rather than merely
  suffered. The estimates come back as a plain ``{name: value}``
  mapping --- the shape :class:`difflow.planning.Block` takes for
  ``theta`` --- so feeding a real-time optimisation layer through
  :mod:`difflow.planning.modifiers` needs no adapter.

What is reused rather than duplicated: reconciliation's
:func:`~difflow.reconciliation.measured_mask` and its ``sigma = inf``
convention, its :class:`~difflow.reconciliation.Scaling` type, its rank
convention in :func:`check_observability`, and its
:class:`~difflow.reconciliation.GlobalTestResult` for
:func:`mhe_global_test`, so a dynamic consistency check reads the same
as a steady-state one. Integration comes from
:func:`difflow.dynamic.integrate` via
:meth:`StateSpaceModel.from_ode`; this module owns no integrator.

Example:
    >>> import jax.numpy as jnp
    >>> from difflow.mhe import (ArrivalCost, MHEProblem, build_window,
    ...                          linear_model, solve_mhe, Measurement)
    >>> model = linear_model(jnp.array([[0.9]]), jnp.array([[1.0]]))
    >>> records = [Measurement(time=float(k), values=[1.0],
    ...                        sigma=[0.1]) for k in range(5)]
    >>> window, _ = build_window(range(5), records, n_y=1)
    >>> problem = MHEProblem(model, ArrivalCost.diagonal(jnp.zeros(1), 1.0),
    ...                      process_std=jnp.array([0.05]))
    >>> res = solve_mhe(problem, window)
    >>> bool(0.7 < float(res.x_final[0]) < 1.3)
    True
"""

from difflow.mhe.arrival import (
    CHOLESKY_JITTER,
    ArrivalCost,
    advance_arrival_cost,
)
from difflow.mhe.ekf import (
    EKFRunResult,
    EKFState,
    ekf_predict,
    ekf_update,
    run_ekf,
)
from difflow.mhe.estimator import (
    CONSTRAINT_WEIGHT,
    MHEProblem,
    MHEResult,
    MHERunResult,
    estimate,
    mhe_global_test,
    run_mhe,
    solve_mhe,
)
from difflow.mhe.measurements import (
    Measurement,
    MeasurementWindow,
    build_window,
    slice_window,
)
from difflow.mhe.model import (
    StateSpaceModel,
    augment_parameters,
    linear_model,
)
from difflow.mhe.observability import (
    ObservabilityReport,
    check_observability,
)

__all__ = [
    # model
    "StateSpaceModel",
    "augment_parameters",
    "linear_model",
    # measurements
    "Measurement",
    "MeasurementWindow",
    "build_window",
    "slice_window",
    # extended Kalman filter
    "EKFState",
    "EKFRunResult",
    "ekf_predict",
    "ekf_update",
    "run_ekf",
    # arrival cost
    "ArrivalCost",
    "advance_arrival_cost",
    "CHOLESKY_JITTER",
    # moving-horizon estimation
    "MHEProblem",
    "MHEResult",
    "MHERunResult",
    "solve_mhe",
    "estimate",
    "run_mhe",
    "mhe_global_test",
    "CONSTRAINT_WEIGHT",
    # observability
    "ObservabilityReport",
    "check_observability",
]
