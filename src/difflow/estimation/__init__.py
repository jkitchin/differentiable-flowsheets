"""Parameter estimation module for difflow.

Provides a structured API for fitting model parameters to experimental data,
computing confidence intervals, bootstrap uncertainty, diagnostics,
cross-validation, and Fisher-information design of the *next* experiment —
all powered by JAX autodiff.

The intended order of operations is:

1. ``check_identifiability`` — can these parameters be told apart at all
   from the measurements available? When this fails, no estimator and no
   experiment design can help; the fix is an added measurement or a
   reparameterization. ``design_experiments`` and ``predicted_covariance``
   enforce the ordering by running this check first and refusing to
   proceed when it fails.
2. ``design_experiments`` / ``predicted_covariance`` — choose the runs and
   see the confidence intervals they would buy, before running them.
3. ``Estimator.fit`` and the uncertainty tools — fit the data once it
   exists.

Example:
    from difflow.estimation import Estimator, Experiment

    experiments = [
        Experiment(inputs={'x': 1.0}, observed={'y': 2.1}),
        Experiment(inputs={'x': 2.0}, observed={'y': 3.9}),
    ]

    def model(theta, exp):
        return {'y': theta['a'] * exp.inputs['x'] + theta['b']}

    est = Estimator(model, param_names=['a', 'b'])
    result = est.fit(experiments, theta_init={'a': 1.0, 'b': 0.0})
    ci = est.confidence_intervals(result, experiments)
    print(est.summary(result, experiments))
"""

from difflow.estimation.experiment import Experiment
from difflow.estimation.objectives import (
    sum_squared_errors,
    weighted_sum_squared_errors,
    negative_log_likelihood,
)
from difflow.estimation.confidence import (
    ConfidenceResult,
    confidence_result_from_covariance,
    fisher_confidence_intervals,
)
from difflow.estimation.diagnostics import DiagnosticsResult, compute_diagnostics
from difflow.estimation.bootstrap import (
    BootstrapResult,
    nonparametric_bootstrap,
    parametric_bootstrap,
)
from difflow.estimation.cross_validation import CrossValidationResult, leave_n_out_cv
from difflow.estimation.identifiability import (
    IdentifiabilityError,
    IdentifiabilityReport,
    check_identifiability,
    sensitivity_matrix,
)
from difflow.estimation.design import (
    CRITERIA,
    DesignResult,
    design_criterion,
    design_experiments,
    fisher_information,
    log_det,
    predicted_covariance,
)
from difflow.estimation.estimator import Estimator, EstimationResult

__all__ = [
    # Core
    "Experiment",
    "Estimator",
    "EstimationResult",
    # Objectives
    "sum_squared_errors",
    "weighted_sum_squared_errors",
    "negative_log_likelihood",
    # Confidence
    "ConfidenceResult",
    "confidence_result_from_covariance",
    "fisher_confidence_intervals",
    # Diagnostics
    "DiagnosticsResult",
    "compute_diagnostics",
    # Bootstrap
    "BootstrapResult",
    "nonparametric_bootstrap",
    "parametric_bootstrap",
    # Cross-validation
    "CrossValidationResult",
    "leave_n_out_cv",
    # Structural identifiability (run this first)
    "IdentifiabilityError",
    "IdentifiabilityReport",
    "check_identifiability",
    "sensitivity_matrix",
    # Experiment design
    "CRITERIA",
    "DesignResult",
    "design_criterion",
    "design_experiments",
    "fisher_information",
    "log_det",
    "predicted_covariance",
]
