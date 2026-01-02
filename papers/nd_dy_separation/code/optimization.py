"""Optimization routines for Nd/Dy separation.

Single-objective and multi-objective optimization using gradient descent.
"""

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import Array

from .objectives import (
    dy_purity,
    dy_recovery,
    annualized_cost,
    weighted_objective,
    constrained_purity_objective,
)


# =============================================================================
# Optimization Results
# =============================================================================

@dataclass
class OptimizationResult:
    """Result from optimization."""
    optimal_params: dict[str, float]
    optimal_value: float
    convergence_history: list[float]
    n_iterations: int
    converged: bool


@dataclass
class ParetoPoint:
    """Single point on Pareto front."""
    pH: float
    OA_ratio: float
    T: float
    conc: float
    purity: float
    recovery: float
    cost: float


# =============================================================================
# Single-Objective Optimization
# =============================================================================

def optimize_purity(
    min_recovery: float = 0.80,
    initial_pH: float = 3.0,
    initial_OA: float = 1.0,
    initial_T: float = 298.15,
    initial_conc: float = 0.5,
    learning_rate: float = 0.1,
    max_iterations: int = 200,
    tolerance: float = 1e-6,
) -> OptimizationResult:
    """Maximize Dy purity subject to minimum recovery constraint.

    Uses gradient descent with penalty method for constraints.

    Args:
        min_recovery: Minimum required Dy recovery
        initial_pH: Starting pH
        initial_OA: Starting O/A ratio
        initial_T: Starting temperature (K)
        initial_conc: Starting D2EHPA concentration (M)
        learning_rate: Step size for gradient descent
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance

    Returns:
        OptimizationResult with optimal parameters
    """
    # Bounds
    bounds_low = jnp.array([1.0, 0.5, 283.0, 0.2])
    bounds_high = jnp.array([5.0, 3.0, 333.0, 1.0])

    # Learning rates (different for each parameter)
    lr = jnp.array([0.05, 0.02, 0.5, 0.01])

    # Initial parameters
    params = jnp.array([initial_pH, initial_OA, initial_T, initial_conc])

    # Gradient function
    grad_fn = jit(grad(constrained_purity_objective))

    history = []
    prev_value = float('inf')

    for i in range(max_iterations):
        # Compute gradient
        g = grad_fn(params, min_recovery=min_recovery)

        # Update parameters
        params = params - lr * g

        # Project onto bounds
        params = jnp.clip(params, bounds_low, bounds_high)

        # Evaluate objective
        value = float(constrained_purity_objective(params, min_recovery=min_recovery))
        history.append(value)

        # Check convergence
        if abs(value - prev_value) < tolerance:
            break
        prev_value = value

    # Extract final values
    final_purity = float(dy_purity(params[0], params[1], params[2], params[3]))
    final_recovery = float(dy_recovery(params[0], params[1], params[2], params[3]))

    return OptimizationResult(
        optimal_params={
            "pH": float(params[0]),
            "OA_ratio": float(params[1]),
            "T": float(params[2]),
            "conc": float(params[3]),
            "purity": final_purity,
            "recovery": final_recovery,
        },
        optimal_value=final_purity,
        convergence_history=history,
        n_iterations=i + 1,
        converged=(i + 1 < max_iterations),
    )


def optimize_recovery(
    min_purity: float = 0.90,
    initial_pH: float = 3.0,
    initial_OA: float = 1.0,
    initial_T: float = 298.15,
    initial_conc: float = 0.5,
    learning_rate: float = 0.1,
    max_iterations: int = 200,
    tolerance: float = 1e-6,
) -> OptimizationResult:
    """Maximize Dy recovery subject to minimum purity constraint.

    Args:
        min_purity: Minimum required Dy purity
        initial_pH: Starting pH
        initial_OA: Starting O/A ratio
        initial_T: Starting temperature (K)
        initial_conc: Starting D2EHPA concentration (M)
        learning_rate: Step size
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance

    Returns:
        OptimizationResult with optimal parameters
    """
    bounds_low = jnp.array([1.0, 0.5, 283.0, 0.2])
    bounds_high = jnp.array([5.0, 3.0, 333.0, 1.0])
    lr = jnp.array([0.05, 0.02, 0.5, 0.01])

    def objective(params, min_purity=0.90, penalty_weight=100.0):
        pH, OA, T, conc = params[0], params[1], params[2], params[3]
        purity = dy_purity(pH, OA, T, conc)
        recovery = dy_recovery(pH, OA, T, conc)
        violation = jnp.maximum(min_purity - purity, 0.0)
        penalty = penalty_weight * violation**2
        return -recovery + penalty

    params = jnp.array([initial_pH, initial_OA, initial_T, initial_conc])
    grad_fn = jit(grad(objective))

    history = []
    prev_value = float('inf')

    for i in range(max_iterations):
        g = grad_fn(params, min_purity=min_purity)
        params = params - lr * g
        params = jnp.clip(params, bounds_low, bounds_high)

        value = float(objective(params, min_purity=min_purity))
        history.append(value)

        if abs(value - prev_value) < tolerance:
            break
        prev_value = value

    final_purity = float(dy_purity(params[0], params[1], params[2], params[3]))
    final_recovery = float(dy_recovery(params[0], params[1], params[2], params[3]))

    return OptimizationResult(
        optimal_params={
            "pH": float(params[0]),
            "OA_ratio": float(params[1]),
            "T": float(params[2]),
            "conc": float(params[3]),
            "purity": final_purity,
            "recovery": final_recovery,
        },
        optimal_value=final_recovery,
        convergence_history=history,
        n_iterations=i + 1,
        converged=(i + 1 < max_iterations),
    )


# =============================================================================
# Multi-Objective Optimization (Pareto Front)
# =============================================================================

def pareto_front(
    n_points: int = 50,
    initial_pH: float = 3.0,
    initial_OA: float = 1.0,
    initial_T: float = 298.15,
    initial_conc: float = 0.5,
    max_iterations: int = 100,
) -> list[ParetoPoint]:
    """Generate Pareto front using weighted sum method.

    Varies weights between purity and recovery objectives.

    Args:
        n_points: Number of Pareto points to generate
        initial_pH: Starting pH for each optimization
        initial_OA: Starting O/A ratio
        initial_T: Starting temperature
        initial_conc: Starting concentration
        max_iterations: Iterations per optimization

    Returns:
        List of ParetoPoint on the Pareto front
    """
    bounds_low = jnp.array([1.0, 0.5, 283.0, 0.2])
    bounds_high = jnp.array([5.0, 3.0, 333.0, 1.0])
    lr = jnp.array([0.05, 0.02, 0.5, 0.01])

    pareto_points = []

    # Vary weight on purity from 0 to 1
    for w_purity in jnp.linspace(0.0, 1.0, n_points):
        w_recovery = 1.0 - w_purity

        def objective(params):
            pH, OA, T, conc = params[0], params[1], params[2], params[3]
            purity = dy_purity(pH, OA, T, conc)
            recovery = dy_recovery(pH, OA, T, conc)
            # Maximize weighted sum (minimize negative)
            return -(w_purity * purity + w_recovery * recovery)

        grad_fn = jit(grad(objective))

        params = jnp.array([initial_pH, initial_OA, initial_T, initial_conc])

        for _ in range(max_iterations):
            g = grad_fn(params)
            params = params - lr * g
            params = jnp.clip(params, bounds_low, bounds_high)

        # Evaluate final point
        pH, OA, T, conc = float(params[0]), float(params[1]), float(params[2]), float(params[3])
        purity = float(dy_purity(params[0], params[1], params[2], params[3]))
        recovery = float(dy_recovery(params[0], params[1], params[2], params[3]))
        cost = float(annualized_cost(params[0], params[1], params[2], params[3]))

        pareto_points.append(ParetoPoint(
            pH=pH,
            OA_ratio=OA,
            T=T,
            conc=conc,
            purity=purity,
            recovery=recovery,
            cost=cost,
        ))

    # Remove dominated points
    pareto_points = _remove_dominated(pareto_points)

    return pareto_points


def pareto_front_3d(
    n_points: int = 100,
    max_iterations: int = 100,
) -> list[ParetoPoint]:
    """Generate 3D Pareto front (purity, recovery, cost).

    Uses weighted sum method with random weight combinations.

    Args:
        n_points: Number of Pareto points to generate
        max_iterations: Iterations per optimization

    Returns:
        List of ParetoPoint on the 3D Pareto front
    """
    bounds_low = jnp.array([1.0, 0.5, 283.0, 0.2])
    bounds_high = jnp.array([5.0, 3.0, 333.0, 1.0])
    lr = jnp.array([0.05, 0.02, 0.5, 0.01])

    pareto_points = []

    # Generate random weight combinations
    key = jax.random.PRNGKey(42)
    weights = jax.random.dirichlet(key, jnp.ones(3), shape=(n_points,))

    for w in weights:
        w_purity, w_recovery, w_cost = float(w[0]), float(w[1]), float(w[2])

        def objective(params):
            pH, OA, T, conc = params[0], params[1], params[2], params[3]
            purity = dy_purity(pH, OA, T, conc)
            recovery = dy_recovery(pH, OA, T, conc)
            cost = annualized_cost(pH, OA, T, conc) / 1e6  # Normalize

            # Maximize purity and recovery, minimize cost
            return -(w_purity * purity + w_recovery * recovery - w_cost * cost)

        grad_fn = jit(grad(objective))

        # Random starting point
        key, subkey = jax.random.split(key)
        params = jax.random.uniform(subkey, shape=(4,), minval=bounds_low, maxval=bounds_high)

        for _ in range(max_iterations):
            g = grad_fn(params)
            params = params - lr * g
            params = jnp.clip(params, bounds_low, bounds_high)

        pH, OA, T, conc = float(params[0]), float(params[1]), float(params[2]), float(params[3])
        purity = float(dy_purity(params[0], params[1], params[2], params[3]))
        recovery = float(dy_recovery(params[0], params[1], params[2], params[3]))
        cost = float(annualized_cost(params[0], params[1], params[2], params[3]))

        pareto_points.append(ParetoPoint(
            pH=pH,
            OA_ratio=OA,
            T=T,
            conc=conc,
            purity=purity,
            recovery=recovery,
            cost=cost,
        ))

    return pareto_points


def _remove_dominated(points: list[ParetoPoint]) -> list[ParetoPoint]:
    """Remove dominated points from a list.

    A point is dominated if another point is better in all objectives.
    Here: higher purity AND higher recovery AND lower cost.
    """
    non_dominated = []

    for p in points:
        dominated = False
        for q in points:
            if (q.purity >= p.purity and
                q.recovery >= p.recovery and
                q.cost <= p.cost and
                (q.purity > p.purity or q.recovery > p.recovery or q.cost < p.cost)):
                dominated = True
                break
        if not dominated:
            non_dominated.append(p)

    return non_dominated


# =============================================================================
# Grid Search (for visualization)
# =============================================================================

def grid_search(
    pH_range: tuple[float, float] = (1.5, 4.5),
    OA_range: tuple[float, float] = (0.5, 2.5),
    n_pH: int = 50,
    n_OA: int = 50,
    T: float = 298.15,
    conc: float = 0.5,
) -> dict:
    """Evaluate purity and recovery over a 2D grid.

    Useful for creating contour plots.

    Args:
        pH_range: (min, max) pH values
        OA_range: (min, max) O/A ratio values
        n_pH: Number of pH points
        n_OA: Number of O/A points
        T: Fixed temperature
        conc: Fixed D2EHPA concentration

    Returns:
        Dictionary with pH_grid, OA_grid, purity_grid, recovery_grid
    """
    pH_vals = jnp.linspace(pH_range[0], pH_range[1], n_pH)
    OA_vals = jnp.linspace(OA_range[0], OA_range[1], n_OA)

    pH_grid, OA_grid = jnp.meshgrid(pH_vals, OA_vals, indexing='ij')

    # Vectorized evaluation
    @jit
    def evaluate(pH, OA):
        purity = dy_purity(pH, OA, jnp.array(T), jnp.array(conc))
        recovery = dy_recovery(pH, OA, jnp.array(T), jnp.array(conc))
        return purity, recovery

    # Use vmap for efficient grid evaluation
    evaluate_row = vmap(evaluate, in_axes=(0, 0))
    evaluate_grid = vmap(evaluate_row, in_axes=(0, 0))

    purity_grid, recovery_grid = evaluate_grid(pH_grid, OA_grid)

    return {
        "pH": pH_vals,
        "OA": OA_vals,
        "pH_grid": pH_grid,
        "OA_grid": OA_grid,
        "purity": purity_grid,
        "recovery": recovery_grid,
    }


if __name__ == "__main__":
    print("Single-Objective Optimization: Maximize Dy Purity")
    print("=" * 60)
    print("Constraint: Dy recovery ≥ 80%")
    print()

    result = optimize_purity(min_recovery=0.80)

    print(f"Converged: {result.converged} ({result.n_iterations} iterations)")
    print(f"\nOptimal Parameters:")
    print(f"  pH = {result.optimal_params['pH']:.3f}")
    print(f"  O/A = {result.optimal_params['OA_ratio']:.3f}")
    print(f"  T = {result.optimal_params['T']:.1f} K")
    print(f"  [D2EHPA] = {result.optimal_params['conc']:.3f} M")
    print(f"\nPerformance:")
    print(f"  Dy purity = {result.optimal_params['purity']*100:.1f}%")
    print(f"  Dy recovery = {result.optimal_params['recovery']*100:.1f}%")

    print("\n\nMulti-Objective Pareto Front")
    print("=" * 60)
    pareto = pareto_front(n_points=20)

    print(f"{'Purity':>10} {'Recovery':>10} {'Cost (k$/y)':>12} {'pH':>6} {'O/A':>6}")
    print("-" * 50)
    for p in pareto[:10]:  # Show first 10
        print(f"{p.purity*100:>9.1f}% {p.recovery*100:>9.1f}% {p.cost/1000:>11.1f} {p.pH:>6.2f} {p.OA_ratio:>6.2f}")
