"""Nd/Dy Single-Stage Separation Study.

Differentiable technoeconomic analysis of Nd/Dy separation by solvent extraction.

Modules:
    single_stage: Core LLE equilibrium model
    objectives: Purity, recovery, cost objective functions
    sensitivity: Gradient-based sensitivity analysis
    optimization: Single and multi-objective optimization
    economics: Technoeconomic analysis
"""

from .single_stage import (
    SingleStageLLE,
    D2EHPADistribution,
    SeparationResult,
)
from .objectives import (
    dy_purity,
    dy_recovery,
    nd_purity,
    nd_recovery,
    separation_cost,
)
from .sensitivity import (
    compute_sensitivities,
    tornado_data,
    uncertainty_propagation,
)
from .optimization import (
    optimize_purity,
    pareto_front,
)
