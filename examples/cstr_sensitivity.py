"""Example: Differentiable CSTR - Sensitivity Analysis and Gradients.

This example demonstrates the differentiability of the CSTR unit operation:
1. dOutput/dInput - How outlet composition changes with inlet conditions
2. dOutput/dParameters - Sensitivity to kinetic parameters (A, Ea)
3. dOutput/dOperating - Sensitivity to operating conditions (V, T)
4. Jacobians and Hessians for advanced analysis
5. Gradient-based optimization of reactor conditions

The CSTR solves: A → B with first-order kinetics k = A * exp(-Ea/RT)
"""

import jax
import jax.numpy as jnp
from jax import Array

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

from difflow.streams import Stream, make_stream, get_flows
from difflow.thermo import IdealThermo, SpeciesData
from difflow.units.cstr import CSTR, CSTRParams


# =============================================================================
# Setup: Species and Thermodynamics
# =============================================================================

species_data = {
    "A": SpeciesData(
        name="A",
        MW=100.0,
        Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
        Hvap_coeffs=(35000.0, 0.38, 500.0),
        antoine_coeffs=(10.0, 3000.0, -50.0),
        Hf=0.0,
    ),
    "B": SpeciesData(
        name="B",
        MW=100.0,
        Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
        Hvap_coeffs=(30000.0, 0.38, 450.0),
        antoine_coeffs=(10.0, 2800.0, -40.0),
        Hf=-50000.0,
    ),
}

thermo = IdealThermo(species_data)
species_order = ["A", "B"]
stoichiometry = jnp.array([[-1.0], [+1.0]])


def rate_function(C: dict[str, Array], T: Array, params: dict) -> Array:
    """First-order reaction: A → B with Arrhenius kinetics."""
    k = params["A"] * jnp.exp(-params["Ea"] / (8.314 * T))
    return jnp.array([k * C["A"]])


# =============================================================================
# 1. dOutput/dInput - Sensitivity to Inlet Conditions
# =============================================================================

def demo_input_sensitivity():
    """Demonstrate sensitivity of outlet to inlet conditions."""
    print("\n" + "=" * 60)
    print("1. SENSITIVITY TO INLET CONDITIONS (dOutput/dInput)")
    print("=" * 60)

    def outlet_B_vs_inlet(F_A_in: Array, F_B_in: Array, T_in: Array) -> Array:
        """Compute outlet F_B as function of inlet conditions."""
        params = CSTRParams(
            V=jnp.array(1.0),
            rate_fn=rate_function,
            stoich=stoichiometry,
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
            species_order=species_order,
            dH_rxn=jnp.array([-50000.0]),
        )
        cstr = CSTR(params, thermo=thermo, mode="isothermal")

        inlet = make_stream({"A": F_A_in, "B": F_B_in}, T=T_in, P=101325.0)
        outlet, _ = cstr(inlet, T_spec=350.0)
        return outlet["F_B"]

    # Base case
    F_A_base, F_B_base, T_base = 10.0, 0.0, 300.0
    F_B_out = outlet_B_vs_inlet(
        jnp.array(F_A_base), jnp.array(F_B_base), jnp.array(T_base)
    )
    print(f"\nBase case: F_A_in={F_A_base}, F_B_in={F_B_base}, T_in={T_base}K")
    print(f"  Outlet F_B = {float(F_B_out):.4f} mol/s")

    # Gradients
    grad_FA = jax.grad(outlet_B_vs_inlet, argnums=0)
    grad_FB = jax.grad(outlet_B_vs_inlet, argnums=1)
    grad_T = jax.grad(outlet_B_vs_inlet, argnums=2)

    dFB_dFA_in = grad_FA(jnp.array(F_A_base), jnp.array(F_B_base), jnp.array(T_base))
    dFB_dFB_in = grad_FB(jnp.array(F_A_base), jnp.array(F_B_base), jnp.array(T_base))
    dFB_dT_in = grad_T(jnp.array(F_A_base), jnp.array(F_B_base), jnp.array(T_base))

    print("\nSensitivities (gradients):")
    print(f"  dF_B_out/dF_A_in = {float(dFB_dFA_in):.4f}")
    print(f"  dF_B_out/dF_B_in = {float(dFB_dFB_in):.4f}")
    print(f"  dF_B_out/dT_in   = {float(dFB_dT_in):.6f} mol/s per K")

    print("\nInterpretation:")
    print(f"  - Increasing inlet A by 1 mol/s increases outlet B by {float(dFB_dFA_in):.2f} mol/s")
    print(f"  - Inlet B passes through unchanged (dF_B/dF_B_in ≈ 1)")


# =============================================================================
# 2. dOutput/dParameters - Sensitivity to Kinetic Parameters
# =============================================================================

def demo_kinetic_sensitivity():
    """Demonstrate sensitivity to kinetic parameters (A, Ea)."""
    print("\n" + "=" * 60)
    print("2. SENSITIVITY TO KINETIC PARAMETERS")
    print("=" * 60)

    def conversion_vs_kinetics(log_A: Array, Ea: Array) -> Array:
        """Compute conversion as function of kinetic parameters.

        Uses log(A) instead of A for better numerical conditioning.
        """
        params = CSTRParams(
            V=jnp.array(1.0),
            rate_fn=rate_function,
            stoich=stoichiometry,
            rate_params={"A": jnp.exp(log_A), "Ea": Ea},
            species_order=species_order,
            dH_rxn=jnp.array([-50000.0]),
        )
        cstr = CSTR(params, thermo=thermo, mode="isothermal")

        inlet = make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0)
        outlet, info = cstr(inlet, T_spec=350.0)
        return info["conversion"]["A"]

    # Base case
    log_A_base = jnp.log(1e6)  # A = 1e6 /s
    Ea_base = jnp.array(50000.0)  # 50 kJ/mol

    X = conversion_vs_kinetics(log_A_base, Ea_base)
    print(f"\nBase case: A = 1e6 /s, Ea = 50 kJ/mol")
    print(f"  Conversion = {float(X)*100:.2f}%")

    # Gradients
    grad_fn = jax.grad(conversion_vs_kinetics, argnums=(0, 1))
    dX_dlogA, dX_dEa = grad_fn(log_A_base, Ea_base)

    print("\nSensitivities:")
    print(f"  dX/d(log A) = {float(dX_dlogA):.4f}")
    print(f"  dX/dEa      = {float(dX_dEa)*1000:.6f} per kJ/mol")

    # Physical interpretation
    print("\nPhysical interpretation:")
    print(f"  - Doubling A (Δlog A = 0.693) increases X by {float(dX_dlogA)*0.693*100:.2f}%")
    print(f"  - Increasing Ea by 1 kJ/mol changes X by {float(dX_dEa)*1000*100:.2f}%")

    # Parameter estimation context
    print("\n  In parameter estimation, these gradients enable:")
    print("  - Gradient-based fitting of A and Ea to experimental data")
    print("  - Uncertainty propagation from parameters to conversion")


# =============================================================================
# 3. dOutput/dOperating - Sensitivity to Operating Conditions
# =============================================================================

def demo_operating_sensitivity():
    """Demonstrate sensitivity to operating conditions (V, T)."""
    print("\n" + "=" * 60)
    print("3. SENSITIVITY TO OPERATING CONDITIONS (V, T)")
    print("=" * 60)

    def outlet_B_vs_operating(V: Array, T_reactor: Array) -> Array:
        """Compute outlet F_B as function of V and T."""
        params = CSTRParams(
            V=V,
            rate_fn=rate_function,
            stoich=stoichiometry,
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
            species_order=species_order,
            dH_rxn=jnp.array([-50000.0]),
        )
        cstr = CSTR(params, thermo=thermo, mode="isothermal")

        inlet = make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0)
        outlet, _ = cstr(inlet, T_spec=T_reactor)
        return outlet["F_B"]

    # Base case
    V_base = jnp.array(1.0)
    T_base = jnp.array(350.0)

    F_B = outlet_B_vs_operating(V_base, T_base)
    print(f"\nBase case: V = 1.0 m³, T = 350 K")
    print(f"  Outlet F_B = {float(F_B):.4f} mol/s")

    # Gradients
    grad_V = jax.grad(outlet_B_vs_operating, argnums=0)
    grad_T = jax.grad(outlet_B_vs_operating, argnums=1)

    dFB_dV = grad_V(V_base, T_base)
    dFB_dT = grad_T(V_base, T_base)

    print("\nSensitivities:")
    print(f"  dF_B/dV = {float(dFB_dV):.4f} mol/s per m³")
    print(f"  dF_B/dT = {float(dFB_dT):.6f} mol/s per K")

    # Normalized sensitivities (elasticities)
    elasticity_V = float(dFB_dV) * float(V_base) / float(F_B)
    elasticity_T = float(dFB_dT) * float(T_base) / float(F_B)

    print("\nElasticities (normalized sensitivities):")
    print(f"  (dF_B/dV)*(V/F_B) = {elasticity_V:.4f}")
    print(f"  (dF_B/dT)*(T/F_B) = {elasticity_T:.4f}")
    print("  (A 1% increase in V/T causes this % change in F_B)")


# =============================================================================
# 4. Jacobian - Full Input-Output Sensitivity Matrix
# =============================================================================

def demo_jacobian():
    """Demonstrate computation of full Jacobian matrix."""
    print("\n" + "=" * 60)
    print("4. JACOBIAN MATRIX (Full Input-Output Sensitivities)")
    print("=" * 60)

    def cstr_function(inputs: Array) -> Array:
        """CSTR as a vector function: [F_A_in, F_B_in, T_in] → [F_A_out, F_B_out, T_out]."""
        F_A_in, F_B_in, T_in = inputs[0], inputs[1], inputs[2]

        params = CSTRParams(
            V=jnp.array(1.0),
            rate_fn=rate_function,
            stoich=stoichiometry,
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
            species_order=species_order,
            dH_rxn=jnp.array([-50000.0]),
        )
        cstr = CSTR(params, thermo=thermo, mode="isothermal")

        inlet = make_stream({"A": F_A_in, "B": F_B_in}, T=T_in, P=101325.0)
        outlet, _ = cstr(inlet, T_spec=350.0)

        return jnp.array([outlet["F_A"], outlet["F_B"], outlet["T"]])

    # Compute Jacobian
    inputs = jnp.array([10.0, 0.0, 300.0])
    jacobian = jax.jacfwd(cstr_function)(inputs)

    outputs = cstr_function(inputs)
    print(f"\nInputs:  F_A_in={inputs[0]:.1f}, F_B_in={inputs[1]:.1f}, T_in={inputs[2]:.1f}")
    print(f"Outputs: F_A_out={outputs[0]:.4f}, F_B_out={outputs[1]:.4f}, T_out={outputs[2]:.1f}")

    print("\nJacobian matrix d(outputs)/d(inputs):")
    print("         dF_A_in   dF_B_in   dT_in")
    print(f"dF_A_out  {jacobian[0, 0]:8.4f}  {jacobian[0, 1]:8.4f}  {jacobian[0, 2]:8.6f}")
    print(f"dF_B_out  {jacobian[1, 0]:8.4f}  {jacobian[1, 1]:8.4f}  {jacobian[1, 2]:8.6f}")
    print(f"dT_out    {jacobian[2, 0]:8.4f}  {jacobian[2, 1]:8.4f}  {jacobian[2, 2]:8.6f}")

    print("\nJacobian interpretation:")
    print("  - Row i shows how output i changes with each input")
    print("  - Column j shows how input j affects each output")
    print("  - The Jacobian enables linear uncertainty propagation")


# =============================================================================
# 5. Hessian - Second-Order Sensitivities
# =============================================================================

def demo_hessian():
    """Demonstrate second-order sensitivities (Hessian)."""
    print("\n" + "=" * 60)
    print("5. HESSIAN MATRIX (Second-Order Sensitivities)")
    print("=" * 60)

    def conversion_vs_VT(params: Array) -> Array:
        """Conversion as function of [V, T]."""
        V, T_reactor = params[0], params[1]

        cstr_params = CSTRParams(
            V=V,
            rate_fn=rate_function,
            stoich=stoichiometry,
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
            species_order=species_order,
            dH_rxn=jnp.array([-50000.0]),
        )
        cstr = CSTR(cstr_params, thermo=thermo, mode="isothermal")

        inlet = make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0)
        outlet, info = cstr(inlet, T_spec=T_reactor)
        return info["conversion"]["A"]

    # Base point
    params = jnp.array([1.0, 350.0])

    # Gradient and Hessian
    X = conversion_vs_VT(params)
    grad = jax.grad(conversion_vs_VT)(params)
    hessian = jax.hessian(conversion_vs_VT)(params)

    print(f"\nAt V = {params[0]:.1f} m³, T = {params[1]:.1f} K:")
    print(f"  Conversion X = {float(X)*100:.2f}%")

    print(f"\nGradient (first derivatives):")
    print(f"  dX/dV = {float(grad[0]):.4f}")
    print(f"  dX/dT = {float(grad[1]):.6f}")

    print(f"\nHessian (second derivatives):")
    print(f"  d²X/dV²   = {float(hessian[0, 0]):.4f}")
    print(f"  d²X/dT²   = {float(hessian[1, 1]):.10f}")
    print(f"  d²X/dVdT  = {float(hessian[0, 1]):.6f}")

    print("\nHessian interpretation:")
    print("  - d²X/dV² < 0: Diminishing returns with volume (concave)")
    print("  - d²X/dT² indicates curvature of T-dependence")
    print("  - d²X/dVdT: Cross-effect (synergy between V and T)")


# =============================================================================
# 6. Gradient-Based Optimization
# =============================================================================

def demo_optimization():
    """Demonstrate gradient-based optimization of reactor conditions."""
    print("\n" + "=" * 60)
    print("6. GRADIENT-BASED OPTIMIZATION")
    print("=" * 60)

    def objective(params: Array) -> Array:
        """Objective: Maximize conversion while minimizing reactor cost.

        Cost = V + 0.001 * (T - 300)^2  (penalize large V and high T)
        Goal: Maximize (conversion - 0.1 * cost)
        """
        V, T_reactor = params[0], params[1]

        cstr_params = CSTRParams(
            V=V,
            rate_fn=rate_function,
            stoich=stoichiometry,
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
            species_order=species_order,
            dH_rxn=jnp.array([-50000.0]),
        )
        cstr = CSTR(cstr_params, thermo=thermo, mode="isothermal")

        inlet = make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0)
        outlet, info = cstr(inlet, T_spec=T_reactor)

        conversion = info["conversion"]["A"]
        cost = V + 0.001 * (T_reactor - 300.0) ** 2

        # Minimize negative of (conversion - cost_penalty)
        return -(conversion - 0.1 * cost)

    # Initial guess
    params = jnp.array([0.5, 320.0])

    print("\nOptimizing: max(conversion - 0.1 * cost)")
    print("where cost = V + 0.001*(T-300)²")
    print(f"\nInitial: V = {params[0]:.2f} m³, T = {params[1]:.1f} K")
    print(f"  Objective = {-float(objective(params)):.4f}")

    # Simple gradient descent
    learning_rate = jnp.array([0.1, 10.0])  # Different scales for V and T

    print("\nGradient descent optimization:")
    for i in range(10):
        grad = jax.grad(objective)(params)
        params = params - learning_rate * grad

        # Bounds
        params = jnp.array([
            jnp.clip(params[0], 0.1, 5.0),
            jnp.clip(params[1], 300.0, 450.0),
        ])

        obj = -float(objective(params))
        if i % 2 == 0:
            print(f"  Step {i+1:2d}: V = {params[0]:.3f} m³, T = {params[1]:.1f} K, obj = {obj:.4f}")

    print(f"\nOptimized: V = {params[0]:.3f} m³, T = {params[1]:.1f} K")

    # Verify optimum
    final_grad = jax.grad(objective)(params)
    print(f"Final gradient magnitude: {jnp.linalg.norm(final_grad):.6f}")


# =============================================================================
# 7. Finite Difference Validation
# =============================================================================

def demo_finite_difference_check():
    """Validate automatic derivatives against finite differences."""
    print("\n" + "=" * 60)
    print("7. FINITE DIFFERENCE VALIDATION")
    print("=" * 60)

    def F_B_out(V: Array) -> Array:
        """Outlet F_B as function of reactor volume."""
        params = CSTRParams(
            V=V,
            rate_fn=rate_function,
            stoich=stoichiometry,
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
            species_order=species_order,
            dH_rxn=jnp.array([-50000.0]),
        )
        cstr = CSTR(params, thermo=thermo, mode="isothermal")

        inlet = make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0)
        outlet, _ = cstr(inlet, T_spec=350.0)
        return outlet["F_B"]

    V = jnp.array(1.0)

    # Automatic differentiation
    ad_grad = float(jax.grad(F_B_out)(V))

    # Finite differences
    eps_values = [1e-2, 1e-4, 1e-6, 1e-8]
    print("\nComparing AD gradient with finite differences:")
    print(f"  AD gradient: {ad_grad:.8f}")
    print("\n  Finite difference approximations:")

    for eps in eps_values:
        fd_grad = (float(F_B_out(V + eps)) - float(F_B_out(V - eps))) / (2 * eps)
        rel_error = abs(fd_grad - ad_grad) / abs(ad_grad) * 100
        print(f"    eps = {eps:.0e}: grad = {fd_grad:.8f}, rel error = {rel_error:.4f}%")

    print("\n  AD gradients are exact (up to floating point precision)")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("DIFFERENTIABLE CSTR - SENSITIVITY ANALYSIS EXAMPLES")
    print("=" * 60)
    print("\nReaction: A → B (first-order, Arrhenius kinetics)")
    print("k = A * exp(-Ea/RT)")

    demo_input_sensitivity()
    demo_kinetic_sensitivity()
    demo_operating_sensitivity()
    demo_jacobian()
    demo_hessian()
    demo_optimization()
    demo_finite_difference_check()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
