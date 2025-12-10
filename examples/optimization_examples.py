"""Example: Optimization with Differentiable Flowsheets.

This example demonstrates various optimization scenarios enabled by
automatic differentiation through flowsheet calculations:

1. Single-variable optimization (optimal reactor temperature)
2. Multi-variable optimization (V, T jointly)
3. Constrained optimization (conversion target)
4. Economic optimization (profit maximization)
5. Parameter estimation (fitting kinetic parameters)
6. Multi-objective Pareto analysis

All optimizations use JAX's automatic differentiation for gradients.
"""

import jax
import jax.numpy as jnp
from jax import Array
from typing import Callable

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

from difflow.streams import Stream, make_stream, get_flows
from difflow.thermo import IdealThermo, SpeciesData
from difflow.units.cstr import CSTR, CSTRParams


# =============================================================================
# Setup
# =============================================================================

species_data = {
    "A": SpeciesData(
        name="A", MW=100.0, Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
        Hvap_coeffs=(35000.0, 0.38, 500.0), antoine_coeffs=(10.0, 3000.0, -50.0),
    ),
    "B": SpeciesData(
        name="B", MW=100.0, Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
        Hvap_coeffs=(30000.0, 0.38, 450.0), antoine_coeffs=(10.0, 2800.0, -40.0),
        Hf=-50000.0,
    ),
}

thermo = IdealThermo(species_data)
species_order = ["A", "B"]
stoichiometry = jnp.array([[-1.0], [+1.0]])


def rate_function(C: dict[str, Array], T: Array, params: dict) -> Array:
    """First-order reaction: A → B."""
    k = params["A"] * jnp.exp(-params["Ea"] / (8.314 * T))
    return jnp.array([k * C["A"]])


def create_cstr(V: Array, rate_params: dict) -> CSTR:
    """Create a CSTR with given volume and kinetic parameters."""
    params = CSTRParams(
        V=V,
        rate_fn=rate_function,
        stoich=stoichiometry,
        rate_params=rate_params,
        species_order=species_order,
        dH_rxn=jnp.array([-50000.0]),
    )
    return CSTR(params, thermo=thermo, mode="isothermal")


# =============================================================================
# Optimizer Utilities
# =============================================================================

def gradient_descent(
    objective: Callable,
    x0: Array,
    learning_rate: float | Array = 0.01,
    max_iter: int = 100,
    tol: float = 1e-6,
    bounds: tuple | None = None,
    verbose: bool = True,
) -> tuple[Array, list]:
    """Simple gradient descent optimizer.

    Args:
        objective: Function to minimize
        x0: Initial guess
        learning_rate: Step size (scalar or per-dimension array)
        max_iter: Maximum iterations
        tol: Convergence tolerance on gradient norm
        bounds: Optional (lower, upper) bounds tuple
        verbose: Print progress

    Returns:
        (optimal_x, history) where history contains (x, obj, grad_norm) per iteration
    """
    x = x0
    lr = jnp.asarray(learning_rate)
    history = []

    for i in range(max_iter):
        obj = objective(x)
        grad = jax.grad(objective)(x)
        grad_norm = jnp.linalg.norm(grad)

        history.append((x.copy(), float(obj), float(grad_norm)))

        if grad_norm < tol:
            if verbose:
                print(f"Converged at iteration {i+1}")
            break

        x = x - lr * grad

        # Apply bounds
        if bounds is not None:
            lower, upper = bounds
            x = jnp.clip(x, lower, upper)

        if verbose and (i + 1) % 10 == 0:
            print(f"  Iter {i+1:3d}: obj = {float(obj):.6f}, |grad| = {float(grad_norm):.6f}")

    return x, history


def adam_optimizer(
    objective: Callable,
    x0: Array,
    learning_rate: float = 0.01,
    max_iter: int = 100,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    bounds: tuple | None = None,
    verbose: bool = True,
) -> tuple[Array, list]:
    """Adam optimizer for better convergence."""
    x = x0
    m = jnp.zeros_like(x)  # First moment
    v = jnp.zeros_like(x)  # Second moment
    history = []

    for i in range(max_iter):
        obj = objective(x)
        grad = jax.grad(objective)(x)
        grad_norm = jnp.linalg.norm(grad)

        history.append((x.copy(), float(obj), float(grad_norm)))

        # Adam updates
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2

        m_hat = m / (1 - beta1 ** (i + 1))
        v_hat = v / (1 - beta2 ** (i + 1))

        x = x - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)

        if bounds is not None:
            lower, upper = bounds
            x = jnp.clip(x, lower, upper)

        if verbose and (i + 1) % 20 == 0:
            print(f"  Iter {i+1:3d}: obj = {float(obj):.6f}, |grad| = {float(grad_norm):.6f}")

    return x, history


# =============================================================================
# 1. Single-Variable Optimization: Optimal Temperature
# =============================================================================

def demo_optimal_temperature():
    """Find optimal reactor temperature for maximum conversion."""
    print("\n" + "=" * 60)
    print("1. OPTIMAL TEMPERATURE FOR MAXIMUM CONVERSION")
    print("=" * 60)

    def neg_conversion(T: Array) -> Array:
        """Negative conversion (to minimize)."""
        cstr = create_cstr(
            V=jnp.array(1.0),
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
        )
        inlet = make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0)
        _, info = cstr(inlet, T_spec=T)
        return -info["conversion"]["A"]

    print("\nFinding T that maximizes conversion...")
    T_opt, history = gradient_descent(
        neg_conversion,
        x0=jnp.array(350.0),
        learning_rate=5.0,
        max_iter=50,
        bounds=(jnp.array(300.0), jnp.array(500.0)),
        verbose=False,
    )

    print(f"\nOptimal temperature: T* = {float(T_opt):.1f} K")
    print(f"Maximum conversion: {-float(neg_conversion(T_opt))*100:.2f}%")

    # Show conversion vs temperature curve
    print("\nConversion vs Temperature:")
    for T in [300, 350, 400, 450, 500]:
        X = -neg_conversion(jnp.array(float(T)))
        print(f"  T = {T} K: X = {float(X)*100:.2f}%")


# =============================================================================
# 2. Multi-Variable Optimization: V and T Jointly
# =============================================================================

def demo_joint_optimization():
    """Jointly optimize reactor volume and temperature."""
    print("\n" + "=" * 60)
    print("2. JOINT OPTIMIZATION OF VOLUME AND TEMPERATURE")
    print("=" * 60)

    def objective(params: Array) -> Array:
        """Minimize cost subject to conversion constraint.

        Cost = capital_cost(V) + energy_cost(T)
        Capital: $10,000 per m³
        Energy: $100 per K above 300K
        Target: 95% conversion
        """
        V, T = params[0], params[1]

        cstr = create_cstr(
            V=V,
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
        )
        inlet = make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0)
        _, info = cstr(inlet, T_spec=T)

        conversion = info["conversion"]["A"]

        # Cost function
        capital_cost = 10000.0 * V
        energy_cost = 100.0 * (T - 300.0)
        total_cost = capital_cost + energy_cost

        # Penalty for missing conversion target
        target_conversion = 0.95
        penalty = 1e6 * jnp.maximum(0.0, target_conversion - conversion) ** 2

        return total_cost + penalty

    print("\nMinimizing: Capital + Energy cost")
    print("Subject to: Conversion >= 95%")

    x_opt, history = adam_optimizer(
        objective,
        x0=jnp.array([1.0, 400.0]),
        learning_rate=0.05,
        max_iter=200,
        bounds=(jnp.array([0.1, 300.0]), jnp.array([10.0, 500.0])),
        verbose=False,
    )

    V_opt, T_opt = float(x_opt[0]), float(x_opt[1])

    # Verify solution
    cstr = create_cstr(
        V=jnp.array(V_opt),
        rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
    )
    inlet = make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0)
    _, info = cstr(inlet, T_spec=jnp.array(T_opt))

    print(f"\nOptimal design:")
    print(f"  Volume V* = {V_opt:.3f} m³")
    print(f"  Temperature T* = {T_opt:.1f} K")
    print(f"  Conversion = {float(info['conversion']['A'])*100:.2f}%")
    print(f"\nCosts:")
    print(f"  Capital: ${V_opt * 10000:.0f}")
    print(f"  Energy: ${(T_opt - 300) * 100:.0f}")
    print(f"  Total: ${V_opt * 10000 + (T_opt - 300) * 100:.0f}")


# =============================================================================
# 3. Constrained Optimization via Penalty Method
# =============================================================================

def demo_constrained_optimization():
    """Optimize with explicit constraints using penalty method."""
    print("\n" + "=" * 60)
    print("3. CONSTRAINED OPTIMIZATION (Penalty Method)")
    print("=" * 60)

    def objective_with_constraints(params: Array, penalty_weight: float) -> Array:
        """
        Minimize: Reactor volume (capital cost)
        Subject to:
          - Conversion >= 90%
          - Temperature <= 400 K (material limit)
          - Residence time >= 0.5 min
        """
        V, T = params[0], params[1]

        cstr = create_cstr(
            V=V,
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
        )
        inlet = make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0)
        _, info = cstr(inlet, T_spec=T)

        conversion = info["conversion"]["A"]

        # Primary objective: minimize volume
        obj = V

        # Constraint violations (inequality: g(x) <= 0)
        g1 = 0.90 - conversion  # conversion >= 90%
        g2 = T - 400.0  # T <= 400 K

        # Quadratic penalty
        penalty = (
            jnp.maximum(0.0, g1) ** 2 +
            jnp.maximum(0.0, g2) ** 2
        )

        return obj + penalty_weight * penalty

    # Solve with increasing penalty weights
    print("\nSolving with increasing penalty weights...")

    x = jnp.array([2.0, 380.0])
    for penalty_weight in [1.0, 10.0, 100.0, 1000.0]:
        obj_fn = lambda p: objective_with_constraints(p, penalty_weight)
        x, _ = gradient_descent(
            obj_fn, x,
            learning_rate=jnp.array([0.01, 1.0]),
            max_iter=50,
            bounds=(jnp.array([0.1, 300.0]), jnp.array([10.0, 500.0])),
            verbose=False,
        )
        print(f"  penalty={penalty_weight:.0f}: V={float(x[0]):.3f}, T={float(x[1]):.1f}")

    V_opt, T_opt = float(x[0]), float(x[1])

    # Verify constraints
    cstr = create_cstr(
        V=jnp.array(V_opt),
        rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
    )
    inlet = make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0)
    _, info = cstr(inlet, T_spec=jnp.array(T_opt))

    print(f"\nFinal solution:")
    print(f"  V = {V_opt:.3f} m³")
    print(f"  T = {T_opt:.1f} K")
    print(f"\nConstraint verification:")
    print(f"  Conversion = {float(info['conversion']['A'])*100:.2f}% (>= 90%: {'OK' if float(info['conversion']['A']) >= 0.90 else 'VIOLATED'})")
    print(f"  Temperature = {T_opt:.1f} K (<= 400: {'OK' if T_opt <= 400 else 'VIOLATED'})")


# =============================================================================
# 4. Economic Optimization: Profit Maximization
# =============================================================================

def demo_economic_optimization():
    """Maximize profit considering revenues and costs."""
    print("\n" + "=" * 60)
    print("4. ECONOMIC OPTIMIZATION (Profit Maximization)")
    print("=" * 60)

    def profit(params: Array) -> Array:
        """
        Profit = Revenue - Costs

        Revenue: $50 per mol/s of B produced
        Costs:
          - Raw material A: $10 per mol/s
          - Capital: $5000 * V per year (annualized)
          - Energy: $0.1 * Q (heat duty in W)
          - Operating at feed rate of 10 mol/s A
        """
        V, T = params[0], params[1]

        cstr = create_cstr(
            V=V,
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
        )
        inlet = make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0)
        outlet, info = cstr(inlet, T_spec=T)

        F_A_in = 10.0
        F_B_out = outlet["F_B"]
        Q = jnp.abs(info["Q"])  # Heat duty magnitude

        # Annual basis (8000 hours/year)
        hours_per_year = 8000.0
        seconds_per_year = hours_per_year * 3600.0

        revenue = 50.0 * F_B_out * seconds_per_year / 1e6  # $M/year
        raw_material_cost = 10.0 * F_A_in * seconds_per_year / 1e6
        capital_cost = 5000.0 * V / 1e6  # Annualized
        energy_cost = 0.1 * Q * hours_per_year / 1e6

        profit = revenue - raw_material_cost - capital_cost - energy_cost

        return -profit  # Minimize negative profit

    print("\nMaximizing annual profit...")
    print("Revenue: $50/mol B, Costs: A=$10/mol, Capital=$5000/m³, Energy=$0.1/W")

    x_opt, history = adam_optimizer(
        profit,
        x0=jnp.array([1.0, 350.0]),
        learning_rate=0.02,
        max_iter=200,
        bounds=(jnp.array([0.1, 300.0]), jnp.array([5.0, 450.0])),
        verbose=False,
    )

    V_opt, T_opt = float(x_opt[0]), float(x_opt[1])

    # Calculate final economics
    cstr = create_cstr(
        V=jnp.array(V_opt),
        rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
    )
    inlet = make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0)
    outlet, info = cstr(inlet, T_spec=jnp.array(T_opt))

    F_B_out = float(outlet["F_B"])
    Q = abs(float(info["Q"]))

    hours_per_year = 8000.0
    seconds_per_year = hours_per_year * 3600.0

    revenue = 50.0 * F_B_out * seconds_per_year / 1e6
    raw_material = 10.0 * 10.0 * seconds_per_year / 1e6
    capital = 5000.0 * V_opt / 1e6
    energy = 0.1 * Q * hours_per_year / 1e6
    net_profit = revenue - raw_material - capital - energy

    print(f"\nOptimal design:")
    print(f"  V = {V_opt:.3f} m³")
    print(f"  T = {T_opt:.1f} K")
    print(f"  Conversion = {float(info['conversion']['A'])*100:.2f}%")
    print(f"\nAnnual economics ($M/year):")
    print(f"  Revenue (B sales):    ${revenue:.3f}M")
    print(f"  Raw material cost:   -${raw_material:.3f}M")
    print(f"  Capital (annualized):-${capital:.3f}M")
    print(f"  Energy cost:         -${energy:.3f}M")
    print(f"  ─────────────────────────────")
    print(f"  Net Profit:           ${net_profit:.3f}M")


# =============================================================================
# 5. Parameter Estimation
# =============================================================================

def demo_parameter_estimation():
    """Estimate kinetic parameters from experimental data."""
    print("\n" + "=" * 60)
    print("5. PARAMETER ESTIMATION (Fitting to Data)")
    print("=" * 60)

    # Generate synthetic "experimental" data
    true_log_A = jnp.log(1e6)
    true_Ea = 50000.0

    temperatures = jnp.array([320.0, 340.0, 360.0, 380.0, 400.0])

    def true_conversion(T):
        cstr = create_cstr(
            V=jnp.array(1.0),
            rate_params={"A": jnp.exp(true_log_A), "Ea": true_Ea},
        )
        inlet = make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0)
        _, info = cstr(inlet, T_spec=T)
        return info["conversion"]["A"]

    # Add noise to create "experimental" data
    key = jax.random.PRNGKey(42)
    noise = jax.random.normal(key, shape=(5,)) * 0.02
    experimental_X = jnp.array([float(true_conversion(T)) for T in temperatures]) + noise

    print("\nExperimental data (with 2% noise):")
    for T, X in zip(temperatures, experimental_X):
        print(f"  T = {float(T):.0f} K: X = {float(X)*100:.2f}%")

    # Define loss function
    def loss(params: Array) -> Array:
        """Sum of squared errors between model and data."""
        log_A, Ea = params[0], params[1]

        def model_X(T):
            cstr = create_cstr(
                V=jnp.array(1.0),
                rate_params={"A": jnp.exp(log_A), "Ea": Ea},
            )
            inlet = make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0)
            _, info = cstr(inlet, T_spec=T)
            return info["conversion"]["A"]

        predictions = jnp.array([model_X(T) for T in temperatures])
        return jnp.sum((predictions - experimental_X) ** 2)

    # Initial guess (deliberately wrong)
    x0 = jnp.array([jnp.log(1e5), 40000.0])  # A=1e5, Ea=40 kJ/mol

    print(f"\nInitial guess: A = {jnp.exp(x0[0]):.2e}, Ea = {x0[1]/1000:.1f} kJ/mol")
    print(f"Initial loss: {float(loss(x0)):.6f}")

    # Optimize
    print("\nFitting parameters...")
    x_opt, history = adam_optimizer(
        loss,
        x0,
        learning_rate=0.1,
        max_iter=200,
        bounds=(jnp.array([jnp.log(1e4), 30000.0]), jnp.array([jnp.log(1e8), 70000.0])),
        verbose=False,
    )

    estimated_A = jnp.exp(x_opt[0])
    estimated_Ea = x_opt[1]

    print(f"\nEstimated parameters:")
    print(f"  A = {float(estimated_A):.2e} (true: {float(jnp.exp(true_log_A)):.2e})")
    print(f"  Ea = {float(estimated_Ea)/1000:.2f} kJ/mol (true: {true_Ea/1000:.2f})")
    print(f"\nFinal loss: {float(loss(x_opt)):.8f}")

    # Relative errors
    print(f"\nRelative errors:")
    print(f"  A: {abs(float(estimated_A) - float(jnp.exp(true_log_A)))/float(jnp.exp(true_log_A))*100:.2f}%")
    print(f"  Ea: {abs(float(estimated_Ea) - true_Ea)/true_Ea*100:.2f}%")


# =============================================================================
# 6. Multi-Objective Pareto Analysis
# =============================================================================

def demo_pareto():
    """Generate Pareto front for conversion vs cost trade-off."""
    print("\n" + "=" * 60)
    print("6. MULTI-OBJECTIVE PARETO ANALYSIS")
    print("=" * 60)

    def cost(V: Array, T: Array) -> float:
        """Total cost = capital + energy."""
        return float(V) * 10000.0 + float(T - 300.0) * 100.0

    def conversion(V: Array, T: Array) -> float:
        """Reactor conversion."""
        cstr = create_cstr(
            V=V,
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
        )
        inlet = make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0)
        _, info = cstr(inlet, T_spec=T)
        return float(info["conversion"]["A"])

    print("\nGenerating Pareto front: Conversion vs Cost")
    print("(Trade-off between performance and cost)")

    pareto_points = []

    # Weighted sum method with varying weights
    for alpha in jnp.linspace(0.01, 0.99, 20):
        def weighted_obj(params: Array) -> Array:
            V, T = params[0], params[1]

            cstr = create_cstr(
                V=V,
                rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
            )
            inlet = make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0)
            _, info = cstr(inlet, T_spec=T)

            X = info["conversion"]["A"]
            C = V * 10.0 + (T - 300.0) * 0.1  # Scaled cost

            # Minimize: alpha * (-X) + (1-alpha) * C
            return alpha * (-X) + (1 - alpha) * C

        x_opt, _ = gradient_descent(
            weighted_obj,
            x0=jnp.array([1.0, 350.0]),
            learning_rate=jnp.array([0.05, 2.0]),
            max_iter=100,
            bounds=(jnp.array([0.1, 300.0]), jnp.array([5.0, 450.0])),
            verbose=False,
        )

        V_opt, T_opt = x_opt[0], x_opt[1]
        X = conversion(V_opt, T_opt)
        C = cost(V_opt, T_opt)
        pareto_points.append((X, C, float(V_opt), float(T_opt)))

    # Sort by conversion
    pareto_points.sort(key=lambda p: p[0])

    print("\nPareto-optimal solutions:")
    print("  Conversion   Cost($)    V(m³)   T(K)")
    print("  ──────────────────────────────────────")
    for X, C, V, T in pareto_points[::4]:  # Show every 4th point
        print(f"    {X*100:5.1f}%    {C:7.0f}   {V:5.2f}   {T:5.1f}")

    print("\nInterpretation:")
    print("  - Higher conversion requires more capital (larger V) or energy (higher T)")
    print("  - The Pareto front shows optimal trade-offs")
    print("  - Points below the front are dominated (can improve both objectives)")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("OPTIMIZATION EXAMPLES WITH DIFFERENTIABLE FLOWSHEETS")
    print("=" * 60)

    demo_optimal_temperature()
    demo_joint_optimization()
    demo_constrained_optimization()
    demo_economic_optimization()
    demo_parameter_estimation()
    demo_pareto()

    print("\n" + "=" * 60)
    print("All optimization examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
