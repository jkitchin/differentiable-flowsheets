"""Uncertainty propagation utilities for difflow.

This module provides tools for propagating parameter uncertainties
through flowsheet calculations using JAX's automatic differentiation.

Methods supported:
1. Linear propagation: First-order Taylor expansion using Jacobians
2. Monte Carlo: Parallel sampling with vmap
3. Sensitivity analysis: Gradient-based importance ranking

All methods leverage JAX's autodiff for efficient computation.

Example:
    from difflow.uncertainty import linear_propagation, monte_carlo_propagation

    # Define a model function
    def model(params):
        T, P = params["T"], params["P"]
        # ... run simulation ...
        return output

    # Nominal values and uncertainties (1-sigma)
    nominal = {"T": 350.0, "P": 101325.0}
    uncertainties = {"T": 5.0, "P": 1000.0}  # ±5 K, ±1000 Pa

    # Linear propagation
    output, output_std = linear_propagation(model, nominal, uncertainties)

    # Monte Carlo (more accurate for nonlinear systems)
    output_mean, output_std, samples = monte_carlo_propagation(
        model, nominal, uncertainties, n_samples=1000
    )
"""

from typing import Callable, Any
from functools import partial
import jax
import jax.numpy as jnp
from jax import Array, vmap

from difflow.numerics import safe_divide
from difflow.constants import EPS_DIVISION


def linear_propagation(
    model: Callable[[dict], Array],
    nominal_params: dict[str, float | Array],
    uncertainties: dict[str, float | Array],
    output_names: list[str] | None = None,
) -> tuple[Array, Array, dict]:
    """Propagate uncertainties using linear (first-order) approximation.

    Uses Jacobian to compute output uncertainty:
        Var(y) ≈ J @ Var(x) @ J.T  (for multivariate)
        σ_y ≈ |∂y/∂x| × σ_x  (for scalar output, single input)

    This is accurate when:
    - Uncertainties are small relative to parameter values
    - Model is approximately linear in the uncertain region

    Args:
        model: Function mapping parameter dict to output (scalar or array)
        nominal_params: Dictionary of nominal parameter values
        uncertainties: Dictionary of 1-sigma uncertainties (same keys as nominal)
        output_names: Optional names for output components

    Returns:
        output: Model output at nominal values
        output_std: Propagated 1-sigma uncertainty
        info: Dictionary with sensitivity information

    Example:
        >>> def model(p):
        ...     return p["k"] * p["C"] ** p["n"]
        >>> nominal = {"k": 0.1, "C": 100.0, "n": 2.0}
        >>> sigma = {"k": 0.01, "C": 5.0, "n": 0.1}
        >>> y, y_std, info = linear_propagation(model, nominal, sigma)
    """
    # Convert to arrays for JAX
    param_names = list(nominal_params.keys())
    x_nominal = jnp.array([float(nominal_params[k]) for k in param_names])
    x_sigma = jnp.array([float(uncertainties.get(k, 0.0)) for k in param_names])

    # Wrapper that takes array input
    def model_array(x):
        params = {k: x[i] for i, k in enumerate(param_names)}
        return jnp.atleast_1d(model(params))

    # Compute output at nominal
    y_nominal = model_array(x_nominal)

    # Compute Jacobian: J[i,j] = ∂y_i/∂x_j
    jacobian = jax.jacobian(model_array)(x_nominal)

    # Propagate variance: Var(y) = J @ diag(σ²) @ J.T
    # For diagonal input covariance:
    # Var(y_i) = sum_j (∂y_i/∂x_j)² × σ_j²
    variance = jnp.sum(jacobian**2 * x_sigma**2, axis=-1)
    y_std = jnp.sqrt(variance)

    # Sensitivity analysis: normalized sensitivity coefficients
    # S_ij = (∂y_i/∂x_j) × (x_j / y_i) = elasticity
    sensitivities = {}
    for j, name in enumerate(param_names):
        # Absolute sensitivity
        sensitivities[name] = {
            "gradient": jacobian[:, j] if jacobian.ndim > 1 else jacobian[j],
            "elasticity": safe_divide(jacobian[:, j] * x_nominal[j], y_nominal)
                if jacobian.ndim > 1
                else safe_divide(jacobian[j] * x_nominal[j], y_nominal[0]),
            "variance_contribution": safe_divide(jacobian[:, j]**2 * x_sigma[j]**2, variance)
                if jacobian.ndim > 1
                else safe_divide(jacobian[j]**2 * x_sigma[j]**2, variance),
        }

    info = {
        "jacobian": jacobian,
        "sensitivities": sensitivities,
        "param_names": param_names,
        "variance": variance,
    }

    # Return scalar if output is scalar
    if y_nominal.size == 1:
        return float(y_nominal[0]), float(y_std[0]), info

    return y_nominal, y_std, info


def monte_carlo_propagation(
    model: Callable[[dict], Array],
    nominal_params: dict[str, float | Array],
    uncertainties: dict[str, float | Array],
    n_samples: int = 1000,
    distribution: str = "normal",
    seed: int = 42,
    return_samples: bool = False,
) -> tuple[Array, Array, dict]:
    """Propagate uncertainties using Monte Carlo sampling.

    Samples input parameters from specified distribution and
    evaluates model for each sample using vectorized execution.

    More accurate than linear propagation for:
    - Large uncertainties
    - Highly nonlinear models
    - Non-Gaussian outputs

    Args:
        model: Function mapping parameter dict to output
        nominal_params: Dictionary of nominal (mean) parameter values
        uncertainties: Dictionary of 1-sigma uncertainties
        n_samples: Number of Monte Carlo samples
        distribution: 'normal' (Gaussian) or 'uniform' (±3σ bounds)
        seed: Random seed for reproducibility
        return_samples: Whether to return all sample outputs

    Returns:
        output_mean: Mean of output samples
        output_std: Standard deviation of output samples
        info: Dictionary with percentiles, samples (if requested)

    Example:
        >>> mean, std, info = monte_carlo_propagation(
        ...     model, nominal, sigma, n_samples=10000
        ... )
        >>> print(f"95% CI: [{info['p2.5']:.2f}, {info['p97.5']:.2f}]")
    """
    key = jax.random.PRNGKey(seed)

    param_names = list(nominal_params.keys())
    n_params = len(param_names)

    x_nominal = jnp.array([float(nominal_params[k]) for k in param_names])
    x_sigma = jnp.array([float(uncertainties.get(k, 0.0)) for k in param_names])

    # Generate samples
    if distribution == "normal":
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, shape=(n_samples, n_params))
        x_samples = x_nominal + z * x_sigma
    elif distribution == "uniform":
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=(n_samples, n_params), minval=-3, maxval=3)
        x_samples = x_nominal + u * x_sigma
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    # Wrapper for array input
    def model_array(x):
        params = {k: x[i] for i, k in enumerate(param_names)}
        return jnp.atleast_1d(model(params))

    # Vectorized evaluation
    outputs = vmap(model_array)(x_samples)

    # Statistics
    output_mean = jnp.mean(outputs, axis=0)
    output_std = jnp.std(outputs, axis=0)

    # Percentiles for confidence intervals
    percentiles = {
        "p2.5": jnp.percentile(outputs, 2.5, axis=0),
        "p5": jnp.percentile(outputs, 5, axis=0),
        "p25": jnp.percentile(outputs, 25, axis=0),
        "p50": jnp.percentile(outputs, 50, axis=0),  # median
        "p75": jnp.percentile(outputs, 75, axis=0),
        "p95": jnp.percentile(outputs, 95, axis=0),
        "p97.5": jnp.percentile(outputs, 97.5, axis=0),
    }

    info = {
        "n_samples": n_samples,
        "distribution": distribution,
        **percentiles,
    }

    if return_samples:
        info["samples"] = outputs
        info["input_samples"] = x_samples

    # Return scalar if output is scalar
    if output_mean.size == 1:
        return float(output_mean[0]), float(output_std[0]), info

    return output_mean, output_std, info


def sensitivity_analysis(
    model: Callable[[dict], Array],
    nominal_params: dict[str, float | Array],
    param_ranges: dict[str, tuple[float, float]] | None = None,
    n_points: int = 10,
) -> dict[str, dict]:
    """Perform local and global sensitivity analysis.

    Computes:
    - Local sensitivity (gradient at nominal point)
    - Normalized sensitivity (elasticity)
    - One-at-a-time (OAT) sensitivity curves

    Args:
        model: Function mapping parameter dict to output
        nominal_params: Dictionary of nominal parameter values
        param_ranges: Optional dict of (min, max) ranges for OAT analysis.
                     If None, uses ±20% of nominal.
        n_points: Number of points for OAT curves

    Returns:
        Dictionary with sensitivity information for each parameter

    Example:
        >>> results = sensitivity_analysis(model, nominal)
        >>> for param, data in results.items():
        ...     print(f"{param}: elasticity = {data['elasticity']:.3f}")
    """
    param_names = list(nominal_params.keys())
    x_nominal = jnp.array([float(nominal_params[k]) for k in param_names])

    def model_array(x):
        params = {k: x[i] for i, k in enumerate(param_names)}
        return jnp.atleast_1d(model(params))

    # Compute gradient at nominal point
    y_nominal = model_array(x_nominal)
    grad = jax.grad(lambda x: jnp.sum(model_array(x)))(x_nominal)

    results = {}

    for i, name in enumerate(param_names):
        # Local sensitivity
        local_sens = grad[i]

        # Elasticity (normalized sensitivity)
        elasticity = safe_divide(local_sens * x_nominal[i], y_nominal[0])

        # One-at-a-time curve
        if param_ranges and name in param_ranges:
            x_min, x_max = param_ranges[name]
        else:
            x_min = 0.8 * x_nominal[i]
            x_max = 1.2 * x_nominal[i]

        x_range = jnp.linspace(x_min, x_max, n_points)
        y_oat = []

        for x_val in x_range:
            x_test = x_nominal.at[i].set(x_val)
            y_oat.append(model_array(x_test))

        y_oat = jnp.array(y_oat).squeeze()

        results[name] = {
            "nominal": float(x_nominal[i]),
            "gradient": float(local_sens),
            "elasticity": float(elasticity),
            "oat_x": x_range,
            "oat_y": y_oat,
        }

    return results


def sobol_indices(
    model: Callable[[dict], Array],
    param_bounds: dict[str, tuple[float, float]],
    n_samples: int = 1024,
    seed: int = 42,
) -> dict[str, dict]:
    """Compute first-order Sobol sensitivity indices.

    Sobol indices measure the fraction of output variance
    attributable to each input parameter.

    Uses Saltelli's sampling scheme for efficiency.

    Args:
        model: Function mapping parameter dict to output
        param_bounds: Dictionary of (min, max) bounds for each parameter
        n_samples: Base number of samples (total = n_samples * (2*d + 2))
        seed: Random seed

    Returns:
        Dictionary with Sobol indices for each parameter

    Note:
        This is a simplified implementation. For production use,
        consider SALib or similar dedicated sensitivity packages.
    """
    key = jax.random.PRNGKey(seed)

    param_names = list(param_bounds.keys())
    n_params = len(param_names)

    # Get bounds as arrays
    bounds_low = jnp.array([param_bounds[k][0] for k in param_names])
    bounds_high = jnp.array([param_bounds[k][1] for k in param_names])

    def model_array(x):
        params = {k: x[i] for i, k in enumerate(param_names)}
        return model(params)

    # Generate two independent sample matrices
    key, k1, k2 = jax.random.split(key, 3)
    A = jax.random.uniform(k1, (n_samples, n_params))
    B = jax.random.uniform(k2, (n_samples, n_params))

    # Scale to bounds
    A = bounds_low + A * (bounds_high - bounds_low)
    B = bounds_low + B * (bounds_high - bounds_low)

    # Evaluate base matrices
    y_A = vmap(model_array)(A)
    y_B = vmap(model_array)(B)

    # Total variance
    y_all = jnp.concatenate([y_A, y_B])
    var_total = jnp.var(y_all)

    results = {}

    for i, name in enumerate(param_names):
        # Create AB_i matrix: A with column i from B
        AB_i = A.at[:, i].set(B[:, i])
        y_AB_i = vmap(model_array)(AB_i)

        # First-order index: S_i = V[E[Y|X_i]] / V[Y]
        # Estimated as: S_i ≈ (1/N) * sum(y_B * (y_AB_i - y_A)) / V[Y]
        S_i = safe_divide(jnp.mean(y_B * (y_AB_i - y_A)), var_total)

        results[name] = {
            "S1": float(jnp.clip(S_i, 0, 1)),  # First-order index
            "bounds": param_bounds[name],
        }

    # Normalize so indices sum to approximately 1 (for additive models)
    total_S1 = sum(r["S1"] for r in results.values())
    for name in results:
        results[name]["S1_normalized"] = float(safe_divide(results[name]["S1"], total_S1))

    return results


def propagate_covariance(
    model: Callable[[dict], Array],
    nominal_params: dict[str, float | Array],
    covariance_matrix: Array,
    param_order: list[str],
) -> tuple[Array, Array, Array]:
    """Propagate a full covariance matrix through the model.

    For correlated input uncertainties, propagates the full
    covariance matrix using the Jacobian:
        Σ_y = J @ Σ_x @ J.T

    Args:
        model: Function mapping parameter dict to output
        nominal_params: Dictionary of nominal parameter values
        covariance_matrix: Input covariance matrix (n_params × n_params)
        param_order: Order of parameters in covariance matrix

    Returns:
        output: Model output at nominal
        output_cov: Output covariance matrix
        jacobian: The Jacobian matrix

    Example:
        >>> # T and P are correlated
        >>> cov = jnp.array([[25.0, 10.0],   # Var(T)=25, Cov(T,P)=10
        ...                  [10.0, 1e6]])   # Cov(P,T)=10, Var(P)=1e6
        >>> y, y_cov, J = propagate_covariance(
        ...     model, {"T": 350, "P": 1e5}, cov, ["T", "P"]
        ... )
    """
    x_nominal = jnp.array([float(nominal_params[k]) for k in param_order])

    def model_array(x):
        params = {k: x[i] for i, k in enumerate(param_order)}
        return jnp.atleast_1d(model(params))

    y_nominal = model_array(x_nominal)
    jacobian = jax.jacobian(model_array)(x_nominal)

    # Propagate covariance: Σ_y = J @ Σ_x @ J.T
    output_cov = jacobian @ covariance_matrix @ jacobian.T

    return y_nominal, output_cov, jacobian
