"""Tests for CSTR with various kinetics orders.

This test file demonstrates that the CSTR implementation works correctly
with different reaction kinetics, including:
- First-order kinetics
- Second-order kinetics
- Zero-order kinetics
- Michaelis-Menten kinetics
- Reversible reactions

Each test compares numerical results to analytical solutions where available,
and verifies that gradients flow correctly through the solver.
"""

import jax
import jax.numpy as jnp
import pytest
import numpy as np

from difflow import (
    CSTR,
    CSTRParams,
    IdealThermo,
    SpeciesData,
    make_stream,
    get_flows,
)


# Enable 64-bit precision for tests
jax.config.update("jax_enable_x64", True)


@pytest.fixture
def simple_thermo():
    """Simple thermodynamics for testing (constant Cp, ideal)."""
    species_data = {
        "A": SpeciesData(
            "A",
            MW=100.0,
            Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(35000.0, 0.38, 500.0),
            antoine_coeffs=(10.0, 3000.0, -50.0),
        ),
        "B": SpeciesData(
            "B",
            MW=100.0,
            Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(30000.0, 0.38, 450.0),
            antoine_coeffs=(10.0, 2800.0, -40.0),
        ),
    }
    return IdealThermo(species_data)


class TestFirstOrderKinetics:
    """Tests for first-order kinetics: r = k * C_A"""

    def test_analytical_solution(self, simple_thermo):
        """Compare CSTR to analytical solution for first-order kinetics.

        For A → B with rate = k * C_A:
        Material balance: F_A_out = F_A_in - k * V * C_A_out
                         F_A_out = F_A_in - k * V * (F_A_out / Q)
                         F_A_out * (1 + k*V/Q) = F_A_in
                         F_A_out = F_A_in / (1 + k * tau)

        where tau = V/Q is the residence time.
        """

        def first_order_rate(C, T, params):
            k = params["k"]
            return jnp.array([k * C["A"]])

        # Parameters
        k = 0.5  # 1/s
        V = 2.0  # m³
        Q = 0.1  # m³/s
        F_A_in = 10.0  # mol/s
        tau = V / Q

        # Analytical solution
        F_A_out_analytical = F_A_in / (1 + k * tau)
        conversion_analytical = (F_A_in - F_A_out_analytical) / F_A_in

        # Numerical solution via CSTR
        stoich = jnp.array([[-1.0], [+1.0]])
        params = CSTRParams(
            V=jnp.array(V),
            rate_fn=first_order_rate,
            stoich=stoich,
            rate_params={"k": jnp.array(k)},
            species_order=["A", "B"],
        )
        cstr = CSTR(params, thermo=simple_thermo, mode="isothermal")

        inlet = make_stream({"A": F_A_in, "B": 0.0}, T=350.0, P=101325.0)
        outlet, info = cstr(inlet, T_spec=350.0, volumetric_flow=Q)

        F_A_out_numerical = float(get_flows(outlet)["A"])
        conversion_numerical = float(info["conversion"]["A"])

        # Compare
        assert F_A_out_numerical == pytest.approx(F_A_out_analytical, rel=1e-6)
        assert conversion_numerical == pytest.approx(conversion_analytical, rel=1e-6)

    def test_gradient_wrt_k(self, simple_thermo):
        """Test gradient of conversion with respect to rate constant k."""

        def first_order_rate(C, T, params):
            k = params["k"]
            return jnp.array([k * C["A"]])

        stoich = jnp.array([[-1.0], [+1.0]])

        def get_conversion(k):
            params = CSTRParams(
                V=jnp.array(2.0),
                rate_fn=first_order_rate,
                stoich=stoich,
                rate_params={"k": k},
                species_order=["A", "B"],
            )
            cstr = CSTR(params, thermo=simple_thermo, mode="isothermal")
            inlet = make_stream({"A": 10.0, "B": 0.0}, T=350.0, P=101325.0)
            outlet, info = cstr(inlet, T_spec=350.0, volumetric_flow=0.1)
            return info["conversion"]["A"]

        # Gradient should be positive (higher k = more conversion)
        grad_k = jax.grad(get_conversion)(jnp.array(0.5))
        assert float(grad_k) > 0

        # Verify with finite difference
        eps = 1e-6
        fd_grad = (
            float(get_conversion(jnp.array(0.5 + eps)))
            - float(get_conversion(jnp.array(0.5 - eps)))
        ) / (2 * eps)

        assert float(grad_k) == pytest.approx(fd_grad, rel=1e-4)


class TestSecondOrderKinetics:
    """Tests for second-order kinetics: r = k * C_A^2"""

    def test_analytical_solution(self, simple_thermo):
        """Compare CSTR to analytical solution for second-order kinetics.

        For A → B with rate = k * C_A^2:
        Material balance: F_A_out = F_A_in - k * V * C_A_out^2
                         F_A_out = F_A_in - k * V * (F_A_out/Q)^2

        Let x = F_A_out, then:
        x = F_A_in - k*V/Q^2 * x^2
        k*V/Q^2 * x^2 + x - F_A_in = 0

        Quadratic formula: x = (-1 + sqrt(1 + 4*k*V*F_A_in/Q^2)) / (2*k*V/Q^2)
        Or equivalently:    x = Q^2 / (2*k*V) * (-1 + sqrt(1 + 4*k*V*F_A_in/Q^2))
        """

        def second_order_rate(C, T, params):
            k = params["k"]
            return jnp.array([k * C["A"] ** 2])

        # Parameters
        k = 0.1  # m³/(mol*s)
        V = 2.0  # m³
        Q = 0.1  # m³/s
        F_A_in = 10.0  # mol/s

        # Analytical solution (quadratic formula)
        a = k * V / Q**2
        F_A_out_analytical = (-1 + np.sqrt(1 + 4 * a * F_A_in)) / (2 * a)

        # Numerical solution via CSTR
        stoich = jnp.array([[-1.0], [+1.0]])
        params = CSTRParams(
            V=jnp.array(V),
            rate_fn=second_order_rate,
            stoich=stoich,
            rate_params={"k": jnp.array(k)},
            species_order=["A", "B"],
        )
        cstr = CSTR(params, thermo=simple_thermo, mode="isothermal")

        inlet = make_stream({"A": F_A_in, "B": 0.0}, T=350.0, P=101325.0)
        outlet, info = cstr(inlet, T_spec=350.0, volumetric_flow=Q)

        F_A_out_numerical = float(get_flows(outlet)["A"])

        # Compare
        assert F_A_out_numerical == pytest.approx(F_A_out_analytical, rel=1e-5)

    def test_gradient_wrt_k(self, simple_thermo):
        """Test gradient flows correctly for second-order kinetics."""

        def second_order_rate(C, T, params):
            k = params["k"]
            return jnp.array([k * C["A"] ** 2])

        stoich = jnp.array([[-1.0], [+1.0]])

        def get_product_B(k):
            params = CSTRParams(
                V=jnp.array(2.0),
                rate_fn=second_order_rate,
                stoich=stoich,
                rate_params={"k": k},
                species_order=["A", "B"],
            )
            cstr = CSTR(params, thermo=simple_thermo, mode="isothermal")
            inlet = make_stream({"A": 10.0, "B": 0.0}, T=350.0, P=101325.0)
            outlet, _ = cstr(inlet, T_spec=350.0, volumetric_flow=0.1)
            return get_flows(outlet)["B"]

        # Gradient should be positive (higher k = more B produced)
        grad_k = jax.grad(get_product_B)(jnp.array(0.1))
        assert float(grad_k) > 0

        # Verify with finite difference
        eps = 1e-7
        fd_grad = (
            float(get_product_B(jnp.array(0.1 + eps)))
            - float(get_product_B(jnp.array(0.1 - eps)))
        ) / (2 * eps)

        assert float(grad_k) == pytest.approx(fd_grad, rel=1e-3)


class TestZeroOrderKinetics:
    """Tests for zero-order kinetics: r = k (constant rate)"""

    def test_analytical_solution(self, simple_thermo):
        """Compare CSTR to analytical solution for zero-order kinetics.

        For A → B with rate = k (when C_A > 0):
        Material balance: F_A_out = F_A_in - k * V

        Note: This is valid only when F_A_out > 0, i.e., F_A_in > k * V
        """

        def zero_order_rate(C, T, params):
            k = params["k"]
            # Rate is k as long as there's A present
            # Use smooth approximation to avoid discontinuity
            return jnp.array([k * jnp.tanh(C["A"] * 100)])

        # Parameters (ensure we don't run out of A)
        k = 1.0  # mol/(m³*s)
        V = 2.0  # m³
        Q = 0.1  # m³/s
        F_A_in = 10.0  # mol/s (much more than k*V = 2 mol/s consumed)

        # Analytical solution (zero-order: constant consumption rate)
        F_A_out_analytical = F_A_in - k * V  # = 10 - 2 = 8 mol/s

        # Numerical solution via CSTR
        stoich = jnp.array([[-1.0], [+1.0]])
        params = CSTRParams(
            V=jnp.array(V),
            rate_fn=zero_order_rate,
            stoich=stoich,
            rate_params={"k": jnp.array(k)},
            species_order=["A", "B"],
        )
        cstr = CSTR(params, thermo=simple_thermo, mode="isothermal")

        inlet = make_stream({"A": F_A_in, "B": 0.0}, T=350.0, P=101325.0)
        outlet, info = cstr(inlet, T_spec=350.0, volumetric_flow=Q)

        F_A_out_numerical = float(get_flows(outlet)["A"])

        # Compare (use slightly relaxed tolerance due to tanh smoothing)
        assert F_A_out_numerical == pytest.approx(F_A_out_analytical, rel=1e-3)

    def test_gradient_wrt_volume(self, simple_thermo):
        """Test gradient of output with respect to reactor volume."""

        def zero_order_rate(C, T, params):
            k = params["k"]
            return jnp.array([k * jnp.tanh(C["A"] * 100)])

        stoich = jnp.array([[-1.0], [+1.0]])

        def get_F_A_out(V):
            params = CSTRParams(
                V=V,
                rate_fn=zero_order_rate,
                stoich=stoich,
                rate_params={"k": jnp.array(1.0)},
                species_order=["A", "B"],
            )
            cstr = CSTR(params, thermo=simple_thermo, mode="isothermal")
            inlet = make_stream({"A": 10.0, "B": 0.0}, T=350.0, P=101325.0)
            outlet, _ = cstr(inlet, T_spec=350.0, volumetric_flow=0.1)
            return get_flows(outlet)["A"]

        # For zero-order: dF_A_out/dV should be approximately -k
        grad_V = jax.grad(get_F_A_out)(jnp.array(2.0))

        # Analytical: dF_A_out/dV = -k = -1.0
        assert float(grad_V) == pytest.approx(-1.0, rel=0.05)


class TestMichaelisMentenKinetics:
    """Tests for Michaelis-Menten kinetics: r = V_max * C_A / (K_m + C_A)"""

    def test_mass_balance_conservation(self, simple_thermo):
        """Verify mass balance is conserved with Michaelis-Menten kinetics."""

        def michaelis_menten_rate(C, T, params):
            V_max = params["V_max"]
            K_m = params["K_m"]
            r = V_max * C["A"] / (K_m + C["A"])
            return jnp.array([r])

        stoich = jnp.array([[-1.0], [+1.0]])
        params = CSTRParams(
            V=jnp.array(2.0),
            rate_fn=michaelis_menten_rate,
            stoich=stoich,
            rate_params={"V_max": jnp.array(5.0), "K_m": jnp.array(2.0)},
            species_order=["A", "B"],
        )
        cstr = CSTR(params, thermo=simple_thermo, mode="isothermal")

        inlet = make_stream({"A": 10.0, "B": 0.0}, T=350.0, P=101325.0)
        outlet, info = cstr(inlet, T_spec=350.0, volumetric_flow=0.1)

        # Total moles should be conserved
        flows_in = get_flows(inlet)
        flows_out = get_flows(outlet)
        total_in = float(flows_in["A"]) + float(flows_in["B"])
        total_out = float(flows_out["A"]) + float(flows_out["B"])

        assert total_out == pytest.approx(total_in, rel=1e-6)

    def test_gradient_wrt_vmax(self, simple_thermo):
        """Test gradient with respect to V_max."""

        def michaelis_menten_rate(C, T, params):
            V_max = params["V_max"]
            K_m = params["K_m"]
            r = V_max * C["A"] / (K_m + C["A"])
            return jnp.array([r])

        stoich = jnp.array([[-1.0], [+1.0]])

        def get_product_B(V_max):
            params = CSTRParams(
                V=jnp.array(2.0),
                rate_fn=michaelis_menten_rate,
                stoich=stoich,
                rate_params={"V_max": V_max, "K_m": jnp.array(2.0)},
                species_order=["A", "B"],
            )
            cstr = CSTR(params, thermo=simple_thermo, mode="isothermal")
            inlet = make_stream({"A": 10.0, "B": 0.0}, T=350.0, P=101325.0)
            outlet, _ = cstr(inlet, T_spec=350.0, volumetric_flow=0.1)
            return get_flows(outlet)["B"]

        # Gradient should be positive
        grad_vmax = jax.grad(get_product_B)(jnp.array(5.0))
        assert float(grad_vmax) > 0

        # Verify with finite difference
        eps = 1e-6
        fd_grad = (
            float(get_product_B(jnp.array(5.0 + eps)))
            - float(get_product_B(jnp.array(5.0 - eps)))
        ) / (2 * eps)

        assert float(grad_vmax) == pytest.approx(fd_grad, rel=1e-3)

    def test_gradient_wrt_km(self, simple_thermo):
        """Test gradient with respect to K_m."""

        def michaelis_menten_rate(C, T, params):
            V_max = params["V_max"]
            K_m = params["K_m"]
            r = V_max * C["A"] / (K_m + C["A"])
            return jnp.array([r])

        stoich = jnp.array([[-1.0], [+1.0]])

        def get_product_B(K_m):
            params = CSTRParams(
                V=jnp.array(2.0),
                rate_fn=michaelis_menten_rate,
                stoich=stoich,
                rate_params={"V_max": jnp.array(5.0), "K_m": K_m},
                species_order=["A", "B"],
            )
            cstr = CSTR(params, thermo=simple_thermo, mode="isothermal")
            inlet = make_stream({"A": 10.0, "B": 0.0}, T=350.0, P=101325.0)
            outlet, _ = cstr(inlet, T_spec=350.0, volumetric_flow=0.1)
            return get_flows(outlet)["B"]

        # Gradient should be negative (higher K_m = less affinity = less reaction)
        grad_km = jax.grad(get_product_B)(jnp.array(2.0))
        assert float(grad_km) < 0


class TestReversibleKinetics:
    """Tests for reversible reaction kinetics: A ⇌ B"""

    def test_equilibrium_approach(self, simple_thermo):
        """Test that reversible reaction approaches equilibrium."""

        def reversible_rate(C, T, params):
            k_f = params["k_f"]  # Forward rate constant
            k_r = params["k_r"]  # Reverse rate constant
            # Net forward rate
            r = k_f * C["A"] - k_r * C["B"]
            return jnp.array([r])

        # At equilibrium: K_eq = k_f / k_r = C_B_eq / C_A_eq
        k_f = 1.0
        k_r = 0.5
        K_eq = k_f / k_r  # = 2.0

        stoich = jnp.array([[-1.0], [+1.0]])
        params = CSTRParams(
            V=jnp.array(10.0),  # Large volume for near-equilibrium
            rate_fn=reversible_rate,
            stoich=stoich,
            rate_params={"k_f": jnp.array(k_f), "k_r": jnp.array(k_r)},
            species_order=["A", "B"],
        )
        cstr = CSTR(params, thermo=simple_thermo, mode="isothermal")

        inlet = make_stream({"A": 10.0, "B": 0.0}, T=350.0, P=101325.0)
        outlet, _ = cstr(inlet, T_spec=350.0, volumetric_flow=0.01)  # Low flow = high residence time

        C_A_out = float(get_flows(outlet)["A"]) / 0.01
        C_B_out = float(get_flows(outlet)["B"]) / 0.01

        # Should approach equilibrium ratio
        ratio = C_B_out / C_A_out
        # Won't reach exactly K_eq due to finite residence time, but should be close
        assert ratio > 1.5  # Should be approaching K_eq = 2.0

    def test_gradient_wrt_equilibrium_constant(self, simple_thermo):
        """Test gradient with respect to reverse rate constant (affects equilibrium)."""

        def reversible_rate(C, T, params):
            k_f = params["k_f"]
            k_r = params["k_r"]
            r = k_f * C["A"] - k_r * C["B"]
            return jnp.array([r])

        stoich = jnp.array([[-1.0], [+1.0]])

        def get_product_B(k_r):
            params = CSTRParams(
                V=jnp.array(5.0),
                rate_fn=reversible_rate,
                stoich=stoich,
                rate_params={"k_f": jnp.array(1.0), "k_r": k_r},
                species_order=["A", "B"],
            )
            cstr = CSTR(params, thermo=simple_thermo, mode="isothermal")
            inlet = make_stream({"A": 10.0, "B": 0.0}, T=350.0, P=101325.0)
            outlet, _ = cstr(inlet, T_spec=350.0, volumetric_flow=0.05)
            return get_flows(outlet)["B"]

        # Gradient should be negative (higher k_r = more reverse reaction = less B)
        grad_kr = jax.grad(get_product_B)(jnp.array(0.5))
        assert float(grad_kr) < 0


class TestMultipleReactions:
    """Tests for systems with multiple simultaneous reactions."""

    def test_series_reactions(self, simple_thermo):
        """Test series reactions: A → B → C (both first-order)."""
        species_data = {
            "A": SpeciesData("A", MW=100.0, Cp_coeffs=(75.0, 0, 0, 0),
                           Hvap_coeffs=(35000.0, 0.38, 500.0),
                           antoine_coeffs=(10.0, 3000.0, -50.0)),
            "B": SpeciesData("B", MW=100.0, Cp_coeffs=(75.0, 0, 0, 0),
                           Hvap_coeffs=(30000.0, 0.38, 450.0),
                           antoine_coeffs=(10.0, 2800.0, -40.0)),
            "C": SpeciesData("C", MW=100.0, Cp_coeffs=(75.0, 0, 0, 0),
                           Hvap_coeffs=(28000.0, 0.38, 420.0),
                           antoine_coeffs=(10.0, 2600.0, -30.0)),
        }
        thermo = IdealThermo(species_data)

        def series_rate(C, T, params):
            k1 = params["k1"]  # A → B
            k2 = params["k2"]  # B → C
            r1 = k1 * C["A"]
            r2 = k2 * C["B"]
            return jnp.array([r1, r2])

        # Stoichiometry: A → B (rxn 1), B → C (rxn 2)
        stoich = jnp.array([
            [-1.0, 0.0],   # A consumed in rxn 1
            [+1.0, -1.0],  # B produced in rxn 1, consumed in rxn 2
            [0.0, +1.0],   # C produced in rxn 2
        ])

        params = CSTRParams(
            V=jnp.array(2.0),
            rate_fn=series_rate,
            stoich=stoich,
            rate_params={"k1": jnp.array(0.5), "k2": jnp.array(0.3)},
            species_order=["A", "B", "C"],
        )
        cstr = CSTR(params, thermo=thermo, mode="isothermal")

        inlet = make_stream({"A": 10.0, "B": 0.0, "C": 0.0}, T=350.0, P=101325.0)
        outlet, info = cstr(inlet, T_spec=350.0, volumetric_flow=0.1)

        flows_out = get_flows(outlet)

        # Mass balance: A + B + C should be conserved
        total_in = 10.0
        total_out = float(flows_out["A"]) + float(flows_out["B"]) + float(flows_out["C"])
        assert total_out == pytest.approx(total_in, rel=1e-6)

        # All species should be present
        assert float(flows_out["A"]) > 0
        assert float(flows_out["B"]) > 0
        assert float(flows_out["C"]) > 0


class TestArrheniusTemperatureDependence:
    """Tests for temperature-dependent kinetics using Arrhenius form."""

    def test_arrhenius_gradient_wrt_temperature(self, simple_thermo):
        """Test gradient with respect to temperature for Arrhenius kinetics."""

        def arrhenius_rate(C, T, params):
            A = params["A"]  # Pre-exponential factor
            Ea = params["Ea"]  # Activation energy (J/mol)
            R = 8.314  # Gas constant
            k = A * jnp.exp(-Ea / (R * T))
            return jnp.array([k * C["A"]])

        stoich = jnp.array([[-1.0], [+1.0]])

        def get_product_B(T_spec):
            params = CSTRParams(
                V=jnp.array(2.0),
                rate_fn=arrhenius_rate,
                stoich=stoich,
                rate_params={"A": jnp.array(1e8), "Ea": jnp.array(50000.0)},
                species_order=["A", "B"],
            )
            cstr = CSTR(params, thermo=simple_thermo, mode="isothermal")
            inlet = make_stream({"A": 10.0, "B": 0.0}, T=T_spec, P=101325.0)
            outlet, _ = cstr(inlet, T_spec=T_spec, volumetric_flow=0.1)
            return get_flows(outlet)["B"]

        # Gradient should be positive (higher T = faster reaction = more B)
        grad_T = jax.grad(get_product_B)(jnp.array(350.0))
        assert float(grad_T) > 0

        # Verify with finite difference
        eps = 0.1
        fd_grad = (
            float(get_product_B(jnp.array(350.0 + eps)))
            - float(get_product_B(jnp.array(350.0 - eps)))
        ) / (2 * eps)

        assert float(grad_T) == pytest.approx(fd_grad, rel=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
