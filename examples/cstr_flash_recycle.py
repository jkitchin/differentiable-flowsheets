"""Example: CSTR + Flash Separator with Recycle.

This example demonstrates a classic reaction-separation system:
- Fresh feed of reactant A
- CSTR where A → B (first-order reaction)
- Flash separator to remove product B as vapor
- Liquid recycle of unreacted A back to CSTR

The flowsheet is fully differentiable, allowing:
- Sensitivity analysis (how do outputs change with parameters?)
- Optimization (find optimal reactor volume, temperature, etc.)

Flowsheet:
                    ┌─────────┐
    Fresh A ──────►│         │      ┌─────────┐
                   │  CSTR   ├─────►│  Flash  ├───► Product B (vapor)
    Recycle ──────►│         │      │         │
         ▲         └─────────┘      └────┬────┘
         │                               │
         └───────────────────────────────┘
                  (liquid recycle)
"""

import jax
import jax.numpy as jnp
from jax import Array

# Enable 64-bit precision for better numerical accuracy
jax.config.update("jax_enable_x64", True)

from difflow.streams import Stream, make_stream, get_flows, combine_streams
from difflow.thermo import IdealThermo, SpeciesData
from difflow.units.cstr import CSTR, CSTRParams
from difflow.units.flash import Flash, FlashParams, Mixer
from difflow.flowsheet import Flowsheet, Unit
from difflow.solvers import fixed_point_solve


# =============================================================================
# Define Species and Thermodynamic Properties
# =============================================================================

# Species A: Reactant (heavy, less volatile - like toluene)
# Species B: Product (light, more volatile - like benzene)
# Antoine coefficients: log10(Psat/Pa) = A - B/(T + C)
# At 350K and 101325 Pa, we want K_A < 1 and K_B > 1
species_data = {
    "A": SpeciesData(
        name="A",
        MW=92.0,  # g/mol (like toluene)
        Cp_coeffs=(75.0, 0.0, 0.0, 0.0),  # J/mol/K (constant Cp)
        Hvap_coeffs=(35000.0, 0.38, 590.0),  # Watson correlation
        # At 350K: log10(P) = 10 - 2000/(350-40) = 10 - 6.45 = 3.55, P ≈ 3550 Pa
        # K_A = 3550/101325 = 0.035 (stays in liquid)
        antoine_coeffs=(10.0, 2000.0, -40.0),
        Hf=0.0,
    ),
    "B": SpeciesData(
        name="B",
        MW=78.0,  # g/mol (like benzene)
        Cp_coeffs=(50.0, 0.0, 0.0, 0.0),  # J/mol/K
        Hvap_coeffs=(30000.0, 0.38, 560.0),  # Watson correlation
        # At 350K: log10(P) = 10 - 1500/(350-40) = 10 - 4.84 = 5.16, P ≈ 145000 Pa
        # K_B = 145000/101325 = 1.43 (goes to vapor)
        antoine_coeffs=(10.0, 1500.0, -40.0),
        Hf=-50000.0,  # Exothermic reaction A → B
    ),
}

thermo = IdealThermo(species_data)
species_order = ["A", "B"]


# =============================================================================
# Define Reaction Kinetics (as pure function)
# =============================================================================

def rate_function(C: dict[str, Array], T: Array, params: dict) -> Array:
    """First-order reaction: A → B.

    Rate = k * C_A where k = A * exp(-Ea / RT)

    Args:
        C: Concentrations (mol/m^3)
        T: Temperature (K)
        params: {"A": pre-exponential, "Ea": activation energy}

    Returns:
        Array of reaction rates [r1] (mol/m^3/s)
    """
    A = params["A"]
    Ea = params["Ea"]
    R = 8.314  # J/mol/K

    k = A * jnp.exp(-Ea / (R * T))
    r = k * C["A"]

    return jnp.array([r])


# Stoichiometry: A → B means ν_A = -1, ν_B = +1
stoichiometry = jnp.array([
    [-1.0],  # A
    [+1.0],  # B
])


# =============================================================================
# Pure functional flowsheet solver (JAX-compatible)
# =============================================================================

def solve_cstr_flash_recycle(
    params: dict[str, Array],
    fresh_feed: Stream,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> dict[str, any]:
    """Solve CSTR + Flash with recycle as a pure function.

    This function is fully differentiable with respect to params.

    Args:
        params: Dictionary with keys:
            - 'V_reactor': Reactor volume (m^3)
            - 'T_reactor': Reactor temperature (K)
            - 'T_flash': Flash temperature (K)
            - 'P_flash': Flash pressure (Pa)
            - 'k_A': Arrhenius pre-exponential factor (1/s)
            - 'k_Ea': Activation energy (J/mol)
        fresh_feed: Fresh feed stream
        tol: Convergence tolerance
        max_iter: Maximum iterations

    Returns:
        Dictionary with all streams and info
    """
    # Extract parameters
    V_reactor = params["V_reactor"]
    T_reactor = params["T_reactor"]
    T_flash = params["T_flash"]
    P_flash = params["P_flash"]
    rate_params = {"A": params["k_A"], "Ea": params["k_Ea"]}

    # Create unit operations (these don't depend on traced values)
    cstr_params = CSTRParams(
        V=V_reactor,
        rate_fn=rate_function,
        stoich=stoichiometry,
        rate_params=rate_params,
        species_order=species_order,
        dH_rxn=jnp.array([-50000.0]),
    )
    cstr = CSTR(cstr_params, thermo=thermo, mode="isothermal")

    flash_params = FlashParams(species_order=species_order)
    flash = Flash(flash_params, thermo=thermo)

    mixer = Mixer(species_order, thermo=thermo)

    # Define the fixed-point iteration function
    # Capture unit operations in closure (they're not JAX types)
    def flowsheet_step(recycle_arr, args):
        fresh, T_r, T_f, P_f = args

        # Unpack recycle
        recycle = make_stream(
            {"A": recycle_arr[0], "B": recycle_arr[1]},
            T=recycle_arr[2],
            P=recycle_arr[3],
        )

        # Mix fresh feed with recycle
        reactor_inlet = mixer(fresh, recycle)

        # React in CSTR
        reactor_outlet, _ = cstr(reactor_inlet, T_spec=T_r)

        # Flash separation
        liquid, vapor, _ = flash(reactor_outlet, T=T_f, P=P_f)

        # Liquid is new recycle
        return jnp.array([
            liquid["F_A"],
            liquid["F_B"],
            liquid["T"],
            liquid["P"],
        ])

    # Initial recycle guess
    recycle_init = jnp.array([1.0, 0.1, T_flash, P_flash])

    args = (fresh_feed, T_reactor, T_flash, P_flash)

    # Solve for converged recycle
    recycle_converged = fixed_point_solve(
        flowsheet_step,
        recycle_init,
        args,
        tol=tol,
        max_iter=max_iter,
        damping=0.5,
    )

    # Final evaluation
    recycle = make_stream(
        {"A": recycle_converged[0], "B": recycle_converged[1]},
        T=recycle_converged[2],
        P=recycle_converged[3],
    )

    reactor_inlet = mixer(fresh_feed, recycle)
    reactor_outlet, cstr_info = cstr(reactor_inlet, T_spec=T_reactor)
    liquid, vapor, flash_info = flash(reactor_outlet, T=T_flash, P=P_flash)

    return {
        "fresh_feed": fresh_feed,
        "recycle": recycle,
        "reactor_inlet": reactor_inlet,
        "reactor_outlet": reactor_outlet,
        "liquid": liquid,
        "vapor": vapor,
        "cstr_info": cstr_info,
        "flash_info": flash_info,
    }


# =============================================================================
# Main Example
# =============================================================================

def main():
    print("=" * 60)
    print("CSTR + Flash Separator with Recycle Example")
    print("=" * 60)

    # Define parameters
    params = {
        "V_reactor": jnp.array(1.0),      # m^3
        "T_reactor": jnp.array(400.0),    # K (higher for more conversion)
        "T_flash": jnp.array(350.0),      # K
        "P_flash": jnp.array(101325.0),   # Pa (atmospheric)
        "k_A": jnp.array(1e8),            # 1/s
        "k_Ea": jnp.array(50000.0),       # J/mol
    }

    # Create fresh feed
    fresh_feed = make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0)

    print("\nSolving flowsheet...")
    results = solve_cstr_flash_recycle(params, fresh_feed)

    # Print results
    print("\n" + "-" * 40)
    print("Stream Results:")
    print("-" * 40)

    for name in ["fresh_feed", "recycle", "reactor_inlet", "reactor_outlet", "liquid", "vapor"]:
        stream = results[name]
        flows = get_flows(stream)
        print(f"\n{name}:")
        print(f"  F_A = {float(flows['A']):.4f} mol/s")
        print(f"  F_B = {float(flows['B']):.4f} mol/s")
        print(f"  T   = {float(stream['T']):.2f} K")
        print(f"  P   = {float(stream['P']):.0f} Pa")

    print("\n" + "-" * 40)
    print("CSTR Info:")
    print("-" * 40)
    cstr_info = results["cstr_info"]
    print(f"  Heat duty Q = {float(cstr_info['Q']):.2f} W")
    print(f"  Reaction rate = {float(cstr_info['rates'][0]):.4f} mol/m³/s")
    print(f"  Conversion A = {float(cstr_info['conversion']['A'])*100:.2f}%")

    print("\n" + "-" * 40)
    print("Flash Info:")
    print("-" * 40)
    flash_info = results["flash_info"]
    print(f"  Vapor fraction = {float(flash_info['V_frac'])*100:.2f}%")
    print(f"  K_A = {float(flash_info['K']['A']):.4f}")
    print(f"  K_B = {float(flash_info['K']['B']):.4f}")

    # =========================================================================
    # Demonstrate Differentiability
    # =========================================================================

    print("\n" + "=" * 60)
    print("Sensitivity Analysis (Automatic Differentiation)")
    print("=" * 60)

    def product_B_flow(params: dict) -> Array:
        """Compute product B vapor flow."""
        result = solve_cstr_flash_recycle(params, fresh_feed)
        return result["vapor"]["F_B"]

    def product_A_flow(params: dict) -> Array:
        """Compute unreacted A in liquid (to minimize)."""
        result = solve_cstr_flash_recycle(params, fresh_feed)
        return result["liquid"]["F_A"]

    # Compute gradients using JAX
    print("\nComputing gradients of product B flow w.r.t. parameters...")

    # Gradient w.r.t. reactor volume
    grad_fn = jax.grad(lambda p: product_B_flow(p))

    try:
        grads = grad_fn(params)
        print(f"\n  dF_B/dV_reactor = {float(grads['V_reactor']):.6f} mol/s per m³")
        print(f"  dF_B/dT_reactor = {float(grads['T_reactor']):.6f} mol/s per K")
        print(f"  dF_B/dT_flash   = {float(grads['T_flash']):.6f} mol/s per K")
        print(f"  dF_B/dP_flash   = {float(grads['P_flash']):.10f} mol/s per Pa")
    except Exception as e:
        print(f"  Gradient computation failed: {e}")
        print("  This is expected if the flash produces no vapor (F_B = 0)")

    # =========================================================================
    # Check K-values and vapor pressure
    # =========================================================================

    print("\n" + "-" * 40)
    print("Thermodynamic Check:")
    print("-" * 40)

    T_check = 350.0
    P_check = 101325.0
    print(f"\nAt T = {T_check} K, P = {P_check} Pa:")
    for species in ["A", "B"]:
        Psat = thermo.Psat(species, T_check)
        K = float(Psat / P_check)
        print(f"  {species}: Psat = {float(Psat):.0f} Pa, K = {K:.4f}")

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
