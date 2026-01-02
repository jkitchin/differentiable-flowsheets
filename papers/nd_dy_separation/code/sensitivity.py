"""Sensitivity analysis for Nd/Dy separation.

Gradient-based sensitivity analysis using JAX automatic differentiation.
"""

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from jax import grad, jacfwd, jacrev, hessian
from jax import Array

from .single_stage import SingleStageLLE, D2EHPADistribution, create_base_case
from .objectives import dy_purity, dy_recovery, nd_purity, nd_recovery, annualized_cost


# =============================================================================
# Sensitivity Data Structures
# =============================================================================

@dataclass
class SensitivityResult:
    """Container for sensitivity analysis results."""
    parameter_names: list[str]
    output_names: list[str]
    jacobian: Array  # Shape: (n_outputs, n_params)
    base_values: dict[str, float]
    gradients: dict[str, dict[str, float]]  # output -> param -> gradient


@dataclass
class TornadoData:
    """Data for tornado chart visualization."""
    parameter: str
    low_value: float
    high_value: float
    base_output: float
    low_output: float
    high_output: float
    sensitivity: float  # Normalized sensitivity


# =============================================================================
# Core Sensitivity Functions
# =============================================================================

def compute_sensitivities(
    base_pH: float = 3.0,
    base_OA: float = 1.0,
    base_T: float = 298.15,
    base_conc: float = 0.5,
) -> SensitivityResult:
    """Compute gradient-based sensitivities at base case.

    Calculates ∂(output)/∂(parameter) for all output-parameter pairs.

    Args:
        base_pH: Base case pH
        base_OA: Base case O/A ratio
        base_T: Base case temperature (K)
        base_conc: Base case D2EHPA concentration (M)

    Returns:
        SensitivityResult with Jacobian and gradient dictionaries
    """
    params = jnp.array([base_pH, base_OA, base_T, base_conc])

    # Define output functions
    def purity_fn(p):
        return dy_purity(p[0], p[1], p[2], p[3])

    def recovery_fn(p):
        return dy_recovery(p[0], p[1], p[2], p[3])

    def nd_purity_fn(p):
        return nd_purity(p[0], p[1], p[2], p[3])

    def nd_recovery_fn(p):
        return nd_recovery(p[0], p[1], p[2], p[3])

    def cost_fn(p):
        return annualized_cost(p[0], p[1], p[2], p[3])

    # Compute gradients
    grad_purity = grad(purity_fn)(params)
    grad_recovery = grad(recovery_fn)(params)
    grad_nd_purity = grad(nd_purity_fn)(params)
    grad_nd_recovery = grad(nd_recovery_fn)(params)
    grad_cost = grad(cost_fn)(params)

    # Stack into Jacobian
    jacobian = jnp.stack([
        grad_purity,
        grad_recovery,
        grad_nd_purity,
        grad_nd_recovery,
        grad_cost,
    ])

    param_names = ["pH", "O/A ratio", "T (K)", "[D2EHPA] (M)"]
    output_names = ["Dy purity", "Dy recovery", "Nd purity", "Nd recovery", "Annual cost"]

    # Base values
    base_values = {
        "pH": base_pH,
        "O/A ratio": base_OA,
        "T (K)": base_T,
        "[D2EHPA] (M)": base_conc,
        "Dy purity": float(purity_fn(params)),
        "Dy recovery": float(recovery_fn(params)),
        "Nd purity": float(nd_purity_fn(params)),
        "Nd recovery": float(nd_recovery_fn(params)),
        "Annual cost": float(cost_fn(params)),
    }

    # Gradient dictionary
    gradients = {}
    for i, out_name in enumerate(output_names):
        gradients[out_name] = {}
        for j, param_name in enumerate(param_names):
            gradients[out_name][param_name] = float(jacobian[i, j])

    return SensitivityResult(
        parameter_names=param_names,
        output_names=output_names,
        jacobian=jacobian,
        base_values=base_values,
        gradients=gradients,
    )


def compute_d_sensitivities(
    base_pH: float = 3.0,
    base_OA: float = 1.0,
    base_T: float = 298.15,
    base_conc: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Compute sensitivities to distribution coefficient parameters.

    Calculates how purity/recovery change with D_Nd and D_Dy directly.

    Args:
        base_pH: Base case pH
        base_OA: Base case O/A ratio
        base_T: Base case temperature (K)
        base_conc: Base case D2EHPA concentration (M)

    Returns:
        Dictionary of sensitivities to D parameters
    """
    # Create custom model where D values can be directly varied
    def purity_from_D(D_Nd: Array, D_Dy: Array, OA_ratio: Array) -> Array:
        """Compute Dy purity given D values directly."""
        F_Nd = 0.01
        F_Dy = 0.01
        F_aq = 1.0
        F_org = F_aq * OA_ratio

        # Extraction factors
        E_Nd = D_Nd * F_org / F_aq
        E_Dy = D_Dy * F_org / F_aq

        # Fraction remaining in aqueous (single stage, 95% efficiency)
        eta = 0.95
        frac_Nd_aq = 1.0 / (1.0 + eta * E_Nd)
        frac_Dy_aq = 1.0 / (1.0 + eta * E_Dy)

        # Flows in extract
        F_Nd_org = F_Nd * (1.0 - frac_Nd_aq)
        F_Dy_org = F_Dy * (1.0 - frac_Dy_aq)

        purity = F_Dy_org / (F_Nd_org + F_Dy_org + 1e-10)
        return purity

    def recovery_from_D(D_Nd: Array, D_Dy: Array, OA_ratio: Array) -> Array:
        """Compute Dy recovery given D values directly."""
        F_aq = 1.0
        F_org = F_aq * OA_ratio
        E_Dy = D_Dy * F_org / F_aq
        eta = 0.95
        frac_Dy_aq = 1.0 / (1.0 + eta * E_Dy)
        recovery = 1.0 - frac_Dy_aq
        return recovery

    # Get base D values
    dist = D2EHPADistribution()
    D_Nd_base = dist.D("Nd", base_pH, base_T, base_conc)
    D_Dy_base = dist.D("Dy", base_pH, base_T, base_conc)
    OA_base = jnp.array(base_OA)

    # Gradients w.r.t. D values
    d_purity_d_DNd = grad(purity_from_D, argnums=0)(D_Nd_base, D_Dy_base, OA_base)
    d_purity_d_DDy = grad(purity_from_D, argnums=1)(D_Nd_base, D_Dy_base, OA_base)

    d_recovery_d_DNd = grad(recovery_from_D, argnums=0)(D_Nd_base, D_Dy_base, OA_base)
    d_recovery_d_DDy = grad(recovery_from_D, argnums=1)(D_Nd_base, D_Dy_base, OA_base)

    return {
        "Dy purity": {
            "D_Nd": float(d_purity_d_DNd),
            "D_Dy": float(d_purity_d_DDy),
        },
        "Dy recovery": {
            "D_Nd": float(d_recovery_d_DNd),
            "D_Dy": float(d_recovery_d_DDy),
        },
        "base_D_Nd": float(D_Nd_base),
        "base_D_Dy": float(D_Dy_base),
    }


# =============================================================================
# Tornado Chart Data
# =============================================================================

def tornado_data(
    output_fn: Callable,
    base_params: dict,
    variations: dict[str, tuple[float, float]],
) -> list[TornadoData]:
    """Generate tornado chart data for an output function.

    Args:
        output_fn: Function that takes (pH, OA, T, conc) and returns scalar
        base_params: Base case parameter values
        variations: Dict of param_name -> (low_value, high_value)

    Returns:
        List of TornadoData sorted by sensitivity magnitude
    """
    pH = base_params["pH"]
    OA = base_params["OA_ratio"]
    T = base_params["T"]
    conc = base_params["conc"]

    base_output = float(output_fn(
        jnp.array(pH), jnp.array(OA), jnp.array(T), jnp.array(conc)
    ))

    results = []

    for param_name, (low, high) in variations.items():
        # Evaluate at low value
        if param_name == "pH":
            low_out = float(output_fn(jnp.array(low), jnp.array(OA), jnp.array(T), jnp.array(conc)))
            high_out = float(output_fn(jnp.array(high), jnp.array(OA), jnp.array(T), jnp.array(conc)))
        elif param_name == "OA_ratio":
            low_out = float(output_fn(jnp.array(pH), jnp.array(low), jnp.array(T), jnp.array(conc)))
            high_out = float(output_fn(jnp.array(pH), jnp.array(high), jnp.array(T), jnp.array(conc)))
        elif param_name == "T":
            low_out = float(output_fn(jnp.array(pH), jnp.array(OA), jnp.array(low), jnp.array(conc)))
            high_out = float(output_fn(jnp.array(pH), jnp.array(OA), jnp.array(high), jnp.array(conc)))
        elif param_name == "conc":
            low_out = float(output_fn(jnp.array(pH), jnp.array(OA), jnp.array(T), jnp.array(low)))
            high_out = float(output_fn(jnp.array(pH), jnp.array(OA), jnp.array(T), jnp.array(high)))
        else:
            continue

        # Sensitivity (normalized by parameter range)
        sensitivity = abs(high_out - low_out) / (high - low + 1e-10)

        results.append(TornadoData(
            parameter=param_name,
            low_value=low,
            high_value=high,
            base_output=base_output,
            low_output=low_out,
            high_output=high_out,
            sensitivity=sensitivity,
        ))

    # Sort by sensitivity magnitude
    results.sort(key=lambda x: abs(x.high_output - x.low_output), reverse=True)

    return results


# =============================================================================
# Uncertainty Propagation
# =============================================================================

def uncertainty_propagation(
    base_pH: float = 3.0,
    base_OA: float = 1.0,
    base_T: float = 298.15,
    base_conc: float = 0.5,
    sigma_D_Nd: float = 0.15,  # Relative uncertainty in D_Nd
    sigma_D_Dy: float = 0.15,  # Relative uncertainty in D_Dy
) -> dict[str, float]:
    """Propagate D coefficient uncertainty to outputs.

    Uses linear error propagation:
        σ_output = sqrt(Σ (∂output/∂param)² * σ_param²)

    Args:
        base_pH: Base case pH
        base_OA: Base case O/A ratio
        base_T: Base case temperature (K)
        base_conc: Base case D2EHPA concentration (M)
        sigma_D_Nd: Relative uncertainty in D_Nd (e.g., 0.15 = 15%)
        sigma_D_Dy: Relative uncertainty in D_Dy

    Returns:
        Dictionary with uncertainty in purity and recovery
    """
    # Get D sensitivities
    d_sens = compute_d_sensitivities(base_pH, base_OA, base_T, base_conc)

    # Absolute uncertainties in D
    D_Nd = d_sens["base_D_Nd"]
    D_Dy = d_sens["base_D_Dy"]
    abs_sigma_D_Nd = D_Nd * sigma_D_Nd
    abs_sigma_D_Dy = D_Dy * sigma_D_Dy

    # Propagate to purity
    dpurity_dDNd = d_sens["Dy purity"]["D_Nd"]
    dpurity_dDDy = d_sens["Dy purity"]["D_Dy"]

    sigma_purity = jnp.sqrt(
        (dpurity_dDNd * abs_sigma_D_Nd)**2 +
        (dpurity_dDDy * abs_sigma_D_Dy)**2
    )

    # Propagate to recovery
    drecov_dDNd = d_sens["Dy recovery"]["D_Nd"]
    drecov_dDDy = d_sens["Dy recovery"]["D_Dy"]

    sigma_recovery = jnp.sqrt(
        (drecov_dDNd * abs_sigma_D_Nd)**2 +
        (drecov_dDDy * abs_sigma_D_Dy)**2
    )

    # Base values
    params = jnp.array([base_pH, base_OA, base_T, base_conc])
    base_purity = float(dy_purity(params[0], params[1], params[2], params[3]))
    base_recovery = float(dy_recovery(params[0], params[1], params[2], params[3]))

    return {
        "base_purity": base_purity,
        "sigma_purity": float(sigma_purity),
        "purity_95_CI": (base_purity - 1.96*float(sigma_purity),
                         base_purity + 1.96*float(sigma_purity)),
        "base_recovery": base_recovery,
        "sigma_recovery": float(sigma_recovery),
        "recovery_95_CI": (base_recovery - 1.96*float(sigma_recovery),
                           base_recovery + 1.96*float(sigma_recovery)),
        "D_Nd": D_Nd,
        "D_Dy": D_Dy,
        "sigma_D_Nd_rel": sigma_D_Nd,
        "sigma_D_Dy_rel": sigma_D_Dy,
    }


# =============================================================================
# Hessian Analysis
# =============================================================================

def compute_hessian(
    output_fn: Callable,
    base_params: Array,
) -> Array:
    """Compute Hessian matrix of second derivatives.

    Args:
        output_fn: Scalar function of parameters
        base_params: Base case parameter values

    Returns:
        Hessian matrix
    """
    return hessian(output_fn)(base_params)


def curvature_analysis(
    base_pH: float = 3.0,
    base_OA: float = 1.0,
    base_T: float = 298.15,
    base_conc: float = 0.5,
) -> dict:
    """Analyze curvature of purity objective.

    Helps identify whether optimum is sharp or flat.

    Returns:
        Dictionary with Hessian eigenvalues and eigenvectors
    """
    params = jnp.array([base_pH, base_OA, base_T, base_conc])

    def purity_fn(p):
        return dy_purity(p[0], p[1], p[2], p[3])

    H = hessian(purity_fn)(params)

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = jnp.linalg.eigh(H)

    return {
        "hessian": H,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "condition_number": float(jnp.abs(eigenvalues).max() / (jnp.abs(eigenvalues).min() + 1e-10)),
    }


if __name__ == "__main__":
    # Test sensitivity analysis
    print("Computing sensitivities at base case...")
    sens = compute_sensitivities()

    print("\nGradient-based Sensitivities")
    print("=" * 60)
    print(f"Base case: pH={sens.base_values['pH']}, O/A={sens.base_values['O/A ratio']}")
    print(f"           T={sens.base_values['T (K)']}K, [D2EHPA]={sens.base_values['[D2EHPA] (M)']}M")
    print()

    for output in ["Dy purity", "Dy recovery"]:
        print(f"\n{output} = {sens.base_values[output]:.4f}")
        print("-" * 40)
        for param in sens.parameter_names:
            g = sens.gradients[output][param]
            print(f"  ∂({output})/∂({param}) = {g:.6f}")

    # D coefficient sensitivities
    print("\n\nSensitivity to Distribution Coefficients")
    print("=" * 60)
    d_sens = compute_d_sensitivities()
    print(f"Base D_Nd = {d_sens['base_D_Nd']:.3f}")
    print(f"Base D_Dy = {d_sens['base_D_Dy']:.3f}")
    print()
    for output in ["Dy purity", "Dy recovery"]:
        print(f"\n{output}:")
        print(f"  ∂/∂D_Nd = {d_sens[output]['D_Nd']:.6f}")
        print(f"  ∂/∂D_Dy = {d_sens[output]['D_Dy']:.6f}")

    # Uncertainty propagation
    print("\n\nUncertainty Propagation (±15% D uncertainty)")
    print("=" * 60)
    uq = uncertainty_propagation()
    print(f"Dy purity = {uq['base_purity']:.4f} ± {uq['sigma_purity']:.4f}")
    print(f"  95% CI: [{uq['purity_95_CI'][0]:.4f}, {uq['purity_95_CI'][1]:.4f}]")
    print(f"Dy recovery = {uq['base_recovery']:.4f} ± {uq['sigma_recovery']:.4f}")
    print(f"  95% CI: [{uq['recovery_95_CI'][0]:.4f}, {uq['recovery_95_CI'][1]:.4f}]")
