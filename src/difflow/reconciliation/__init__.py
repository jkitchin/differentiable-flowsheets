"""Data reconciliation for differentiable flowsheets.

Plant measurements are noisy and, taken at face value, contradict the
model: nominations do not close, balances do not balance. Data
reconciliation finds the smallest statistically weighted adjustment
that makes them consistent,

.. math::

    \\min_x (x - y)^T W (x - y) \\quad \\text{s.t.} \\quad F(x, \\theta) = 0,

and in doing so produces estimates *more* precise than the raw
measurements, because the model equations carry information the sensors
do not.

Everything that makes this work is derivative information --- the
constraint Jacobian, the covariance of the estimates, the sensitivity
of an estimate to each measurement --- so a differentiable flowsheet
gets it from ``jax.jacobian`` instead of hand-derived analytical
models. The module is deliberately domain-agnostic: give it any
traceable ``F(x, params) -> residuals`` and it works. See
:mod:`difflow_gas.residuals` for a flowsheet-derived example.

Example:
    >>> import jax.numpy as jnp
    >>> from difflow.reconciliation import reconcile, global_test
    >>> def F(x, params=None):          # one balance: in = out1 + out2
    ...     return jnp.array([x[0] - x[1] - x[2]])
    >>> y = jnp.array([100.0, 62.0, 40.0])       # does not close
    >>> sigma = jnp.array([2.0, 1.0, 1.0])
    >>> res = reconcile(F, y, sigma, names=["feed", "top", "bottom"])
    >>> res.x_named                               # doctest: +SKIP
    {'feed': 100.67, 'top': 61.33, 'bottom': 39.33}
    >>> global_test(res).detected                 # doctest: +SKIP
    False

An entry of ``sigma`` set to ``inf`` marks a variable to be *estimated*
rather than reconciled --- an unknown fouling factor, an unmetered
offtake --- which is how joint parameter estimation and reconciliation
become the same computation. Whether such a variable can be recovered
at all is decided up front by
:func:`~difflow.reconciliation.structure.classify`, so an ill-posed
problem raises :class:`ReconciliationStructureError` naming the
culprits instead of returning NaN.
"""

from difflow.reconciliation.core import (
    Scaling,
    auto_scaling,
    identity_scaling,
    implicit_correction,
    jacobian_of,
    kkt_matrix,
    measured_mask,
    measurement_sensitivity,
    reconciled_covariance,
    solve_reconciliation,
    stationarity,
)
from difflow.reconciliation.design import sensor_ranking, sensor_value
from difflow.reconciliation.gross_error import (
    EliminationStep,
    GlobalTestResult,
    MeasurementTestResult,
    global_test,
    measurement_test,
    serial_elimination,
)
from difflow.reconciliation.reconcile import ReconcileResult, reconcile
from difflow.reconciliation.structure import (
    MEASURED_JUST_DETERMINED,
    MEASURED_REDUNDANT,
    REDUNDANCY_TOL,
    UNMEASURED_OBSERVABLE,
    UNMEASURED_UNOBSERVABLE,
    ReconciliationStructureError,
    StructureReport,
    classify,
)

__all__ = [
    # entry point
    "reconcile",
    "ReconcileResult",
    # KKT core
    "Scaling",
    "auto_scaling",
    "identity_scaling",
    "kkt_matrix",
    "jacobian_of",
    "measured_mask",
    "solve_reconciliation",
    "reconciled_covariance",
    "measurement_sensitivity",
    "stationarity",
    "implicit_correction",
    # structure
    "classify",
    "StructureReport",
    "ReconciliationStructureError",
    "MEASURED_REDUNDANT",
    "MEASURED_JUST_DETERMINED",
    "UNMEASURED_OBSERVABLE",
    "UNMEASURED_UNOBSERVABLE",
    "REDUNDANCY_TOL",
    # gross error detection
    "global_test",
    "GlobalTestResult",
    "measurement_test",
    "MeasurementTestResult",
    "serial_elimination",
    "EliminationStep",
    # sensor placement
    "sensor_value",
    "sensor_ranking",
]
