"""Tests for Peng-Robinson and SRK equations of state."""

import jax
import jax.numpy as jnp
import pytest

from difflow.eos import (
    PengRobinson,
    SRK,
    CriticalProperties,
    flash_TP_eos,
)
from difflow.thermo import IdealThermo, CubicThermo, SpeciesData


# Enable 64-bit precision for tests
jax.config.update("jax_enable_x64", True)


@pytest.fixture
def methane_ethane_system():
    """Two-component hydrocarbon system."""
    species_data = {
        "methane": CriticalProperties(
            name="methane",
            Tc=190.6,  # K
            Pc=4.6e6,  # Pa
            omega=0.011,
            MW=16.04,
        ),
        "ethane": CriticalProperties(
            name="ethane",
            Tc=305.3,
            Pc=4.87e6,
            omega=0.099,
            MW=30.07,
        ),
    }
    return species_data


@pytest.fixture
def propane_butane_system():
    """Propane-butane system for VLE tests."""
    species_data = {
        "propane": CriticalProperties(
            name="propane",
            Tc=369.8,
            Pc=4.25e6,
            omega=0.152,
            MW=44.10,
        ),
        "butane": CriticalProperties(
            name="butane",
            Tc=425.1,
            Pc=3.80e6,
            omega=0.200,
            MW=58.12,
        ),
    }
    return species_data


class TestPengRobinson:
    """Tests for Peng-Robinson EOS."""

    def test_pr_creation(self, methane_ethane_system):
        """Test PR EOS can be created."""
        pr = PengRobinson(methane_ethane_system)
        assert pr is not None
        assert pr.n_species == 2
        assert pr.species_order == ["methane", "ethane"]

    def test_pr_alpha_function(self, methane_ethane_system):
        """Test alpha function calculation."""
        pr = PengRobinson(methane_ethane_system)
        T = jnp.array(250.0)  # K

        alpha = pr.alpha(T)

        # Alpha should be > 1 for T < Tc, < 1 for T > Tc
        # Methane Tc = 190.6 K, so at 250 K, T > Tc
        assert alpha[0] < 1.0  # methane at T > Tc
        # Ethane Tc = 305.3 K, so at 250 K, T < Tc
        assert alpha[1] > 1.0  # ethane at T < Tc

    def test_pr_compressibility_vapor(self, methane_ethane_system):
        """Test compressibility factor for vapor phase."""
        pr = PengRobinson(methane_ethane_system)

        T = jnp.array(250.0)
        P = jnp.array(1e5)  # 1 bar, low pressure
        y = jnp.array([0.5, 0.5])

        Z = pr.solve_Z(T, P, y, phase="vapor")

        # At low pressure, Z should be close to 1 (ideal gas)
        assert float(Z) > 0.9
        assert float(Z) < 1.1

    def test_pr_molar_volume(self, methane_ethane_system):
        """Test molar volume calculation."""
        pr = PengRobinson(methane_ethane_system)

        T = jnp.array(300.0)
        P = jnp.array(1e5)  # 1 bar
        y = jnp.array([0.5, 0.5])

        V = pr.molar_volume(T, P, y, phase="vapor")

        # V = ZRT/P, at low P and Z ≈ 1
        # V_ideal = 8.314 * 300 / 1e5 = 0.0249 m³/mol
        assert float(V) > 0.02
        assert float(V) < 0.03

    def test_pr_fugacity_coefficient(self, methane_ethane_system):
        """Test fugacity coefficient calculation."""
        pr = PengRobinson(methane_ethane_system)

        T = jnp.array(250.0)
        P = jnp.array(1e5)
        y = jnp.array([0.5, 0.5])

        phi = pr.fugacity_coefficient(T, P, y, phase="vapor")

        # At low pressure, fugacity coefficients should be close to 1
        assert jnp.all(phi > 0.8)
        assert jnp.all(phi < 1.2)

    def test_pr_k_values_wilson(self, propane_butane_system):
        """Test Wilson K-value estimation."""
        pr = PengRobinson(propane_butane_system)

        T = jnp.array(320.0)
        P = jnp.array(5e5)  # 5 bar

        K = pr.K_values_wilson(T, P)

        # Propane is lighter, should have higher K
        assert K[0] > K[1]
        # Both should be positive
        assert jnp.all(K > 0)

    def test_pr_differentiability(self, methane_ethane_system):
        """Test that PR is differentiable w.r.t. temperature."""
        pr = PengRobinson(methane_ethane_system)

        def Z_func(T):
            P = jnp.array(1e6)
            y = jnp.array([0.5, 0.5])
            return pr.solve_Z(T, P, y, phase="vapor")

        # Compute gradient
        grad_T = jax.grad(Z_func)(jnp.array(300.0))

        # Gradient should exist and be finite
        assert jnp.isfinite(grad_T)


class TestSRK:
    """Tests for SRK EOS."""

    def test_srk_creation(self, methane_ethane_system):
        """Test SRK EOS can be created."""
        srk = SRK(methane_ethane_system)
        assert srk is not None
        assert srk.n_species == 2

    def test_srk_compressibility_vapor(self, methane_ethane_system):
        """Test SRK compressibility factor for vapor."""
        srk = SRK(methane_ethane_system)

        T = jnp.array(250.0)
        P = jnp.array(1e5)
        y = jnp.array([0.5, 0.5])

        Z = srk.solve_Z(T, P, y, phase="vapor")

        # At low pressure, Z should be close to 1
        assert float(Z) > 0.9
        assert float(Z) < 1.1

    def test_srk_fugacity_coefficient(self, methane_ethane_system):
        """Test SRK fugacity coefficient."""
        srk = SRK(methane_ethane_system)

        T = jnp.array(250.0)
        P = jnp.array(1e5)
        y = jnp.array([0.5, 0.5])

        phi = srk.fugacity_coefficient(T, P, y, phase="vapor")

        # At low pressure, phi should be close to 1
        assert jnp.all(phi > 0.8)
        assert jnp.all(phi < 1.2)

    def test_srk_vs_pr_consistency(self, propane_butane_system):
        """Test that SRK and PR give similar results at low pressure."""
        pr = PengRobinson(propane_butane_system)
        srk = SRK(propane_butane_system)

        T = jnp.array(300.0)
        P = jnp.array(1e5)  # Low pressure
        y = jnp.array([0.5, 0.5])

        Z_pr = pr.solve_Z(T, P, y, phase="vapor")
        Z_srk = srk.solve_Z(T, P, y, phase="vapor")

        # At low pressure, both should give Z ≈ 1
        # They may differ somewhat but should be in same ballpark
        assert abs(float(Z_pr) - float(Z_srk)) < 0.1


class TestFlashTPEOS:
    """Tests for flash calculation with EOS."""

    def test_flash_two_phase(self, propane_butane_system):
        """Test flash calculation in the two-phase region."""
        pr = PengRobinson(propane_butane_system)

        # 320 K / 8.5 bar is genuinely two-phase for this 50/50 mixture (the PR
        # dew pressure is ~7.3 bar and the bubble ~10 bar). The earlier 5 bar
        # here is actually single-phase vapor (below the dew), and only appeared
        # two-phase because the pre-stability-test flash could return a spurious
        # split near the boundary; flash_TP_eos now runs a phase-stability test
        # first and correctly reports V = 1 there.
        z = jnp.array([0.5, 0.5])
        T = jnp.array(320.0)
        P = jnp.array(8.5e5)

        V, x, y = flash_TP_eos(pr, z, T, P)

        # Genuinely two-phase: 0 < V < 1
        assert 0.0 < float(V) < 1.0

        # Vapor fraction should be between 0 and 1
        assert float(V) >= 0.0
        assert float(V) <= 1.0

        # Compositions should sum to 1
        assert jnp.isclose(jnp.sum(x), 1.0, rtol=1e-3)
        assert jnp.isclose(jnp.sum(y), 1.0, rtol=1e-3)

        # Vapor should be enriched in lighter component (propane)
        assert y[0] > x[0]

    def test_flash_differentiability(self, propane_butane_system):
        """Test that flash is differentiable w.r.t. temperature."""
        pr = PengRobinson(propane_butane_system)

        def V_frac_func(T):
            z = jnp.array([0.5, 0.5])
            P = jnp.array(5e5)
            V, _, _ = flash_TP_eos(pr, z, T, P)
            return V

        # Compute gradient
        grad_T = jax.grad(V_frac_func)(jnp.array(320.0))

        # Gradient should be finite (may be zero at certain conditions)
        assert jnp.isfinite(grad_T)

    def test_flash_single_phase_no_trivial_root(self, propane_butane_system):
        """Single-phase feeds return V=1 (vapor) or V=0 (liquid) robustly.

        Regression test for the trivial-root failure of successive
        substitution near the critical region: without the phase-stability
        gate, the K-value iteration could drift to the spurious (all K_i -> 1)
        solution and report the wrong vapor fraction (e.g. V=0 for a
        single-phase vapor) once given more iterations. With the gate, the
        result is exact and stable across iteration counts.
        """
        pr = PengRobinson(propane_butane_system)
        z = jnp.array([0.5, 0.5])
        T = jnp.array(320.0)

        # Well below the ~7.3 bar dew pressure -> single-phase vapor.
        for n in (30, 100, 400):
            V, _, _ = flash_TP_eos(pr, z, T, jnp.array(3e5), max_iter=n)
            assert float(V) == pytest.approx(1.0, abs=1e-6)

        # Well above the ~10 bar bubble pressure -> single-phase liquid.
        for n in (30, 100, 400):
            V, _, _ = flash_TP_eos(pr, z, T, jnp.array(15e5), max_iter=n)
            assert float(V) == pytest.approx(0.0, abs=1e-6)

    def test_flash_vapor_fraction_monotonic_and_continuous(self, propane_butane_system):
        """V(P) at fixed T is monotone non-increasing and free of the spurious
        jumps the trivial-root drift used to produce near the phase boundary."""
        pr = PengRobinson(propane_butane_system)
        z = jnp.array([0.5, 0.5])
        T = jnp.array(320.0)

        pressures = jnp.linspace(3e5, 12e5, 40)
        Vs = [float(flash_TP_eos(pr, z, T, P)[0]) for P in pressures]

        # Monotone non-increasing (more pressure -> less vapor), no jump > 0.2
        for lo, hi in zip(Vs[1:], Vs[:-1]):
            assert lo <= hi + 1e-6
        assert max(abs(a - b) for a, b in zip(Vs[1:], Vs[:-1])) < 0.2

    def test_flash_wide_k_spread_ngl(self):
        """Regression test for issue #169.

        A cryogenic NGL feed with a wide K-value spread (methane K~3,
        hexane K~0.003) makes the Rachford-Rice function stiff, with poles
        just outside [0, 1]. The old unbracketed Newton solver overshot a pole,
        diverged, and clipped to a spurious single-phase V=1. The bracketed
        bisection solver stays inside [0, 1] and finds the genuine two-phase
        root (VF ~ 0.89, reference `thermo`: 0.890).
        """
        from difflow.database import get_critical_props

        names = [
            "nitrogen", "carbon_dioxide", "methane", "ethane", "propane",
            "isobutane", "n_butane", "isopentane", "n_pentane", "n_hexane",
        ]
        eos = PengRobinson({c: get_critical_props(c) for c in names})
        z = jnp.array(
            [0.005, 0.007, 0.860, 0.070, 0.030, 0.005, 0.010, 0.003, 0.005, 0.005]
        )
        z = z / z.sum()

        V, x, y = flash_TP_eos(eos, z, jnp.array(233.15), jnp.array(50e5))

        # Genuinely two-phase near VF=0.89 -- NOT a spurious single-phase V=1.
        assert float(V) == pytest.approx(0.89, abs=0.02)
        assert jnp.isclose(jnp.sum(x), 1.0, rtol=1e-3)
        assert jnp.isclose(jnp.sum(y), 1.0, rtol=1e-3)


class TestBinaryInteractionParameters:
    """Tests for binary interaction parameter support."""

    def test_k_ij_affects_properties(self, methane_ethane_system):
        """Test that k_ij affects mixture properties."""
        pr = PengRobinson(methane_ethane_system)

        T = jnp.array(250.0)
        P = jnp.array(5e6)  # Higher pressure for more interaction effect
        y = jnp.array([0.5, 0.5])

        # Without k_ij
        Z_no_kij = pr.solve_Z(T, P, y, phase="vapor", k_ij=None)

        # With k_ij
        k_ij = jnp.array([[0.0, 0.02], [0.02, 0.0]])
        Z_with_kij = pr.solve_Z(T, P, y, phase="vapor", k_ij=k_ij)

        # Results should differ
        assert not jnp.isclose(Z_no_kij, Z_with_kij, rtol=1e-6)


class TestEnthalpyDeparture:
    """Enthalpy departure H(T,P) - H_ideal_gas(T) from the cubic EOS.

    Covers both PengRobinson and SRK (SRK.enthalpy_departure is the
    epsilon=0/sigma=1 specialization of the generic-cubic departure).
    """

    @pytest.mark.parametrize("EOS", [PengRobinson, SRK])
    def test_departure_vanishes_at_low_pressure(self, propane_butane_system, EOS):
        """As P -> 0 the gas is ideal, so the departure -> 0."""
        eos = EOS(propane_butane_system)
        y = jnp.array([0.5, 0.5])
        T = jnp.array(320.0)
        h_dep = eos.enthalpy_departure(T, jnp.array(1.0e2), y, "vapor")
        assert abs(float(h_dep)) < 1.0  # J/mol

    @pytest.mark.parametrize("EOS", [PengRobinson, SRK])
    def test_departure_negative_and_grows_for_vapor(self, propane_butane_system, EOS):
        """A real vapor sits below the ideal-gas enthalpy, more so at higher P."""
        eos = EOS(propane_butane_system)
        y = jnp.array([0.5, 0.5])
        T = jnp.array(320.0)
        h_low = eos.enthalpy_departure(T, jnp.array(1e5), y, "vapor")
        h_high = eos.enthalpy_departure(T, jnp.array(8e5), y, "vapor")
        assert float(h_low) < 0.0
        assert float(h_high) < float(h_low)

    @pytest.mark.parametrize("EOS", [PengRobinson, SRK])
    def test_departure_differentiable(self, propane_butane_system, EOS):
        """d(H_dep)/dT is finite (jax.jvp path stays differentiable)."""
        eos = EOS(propane_butane_system)
        y = jnp.array([0.5, 0.5])
        g = jax.grad(
            lambda T: eos.enthalpy_departure(T, jnp.array(8e5), y, "vapor")
        )(jnp.array(320.0))
        assert jnp.isfinite(g)

    def test_pr_and_srk_departure_agree_within_ten_percent(self, propane_butane_system):
        """PR and SRK departures track each other for this near-ideal vapor."""
        pr = PengRobinson(propane_butane_system)
        srk = SRK(propane_butane_system)
        y = jnp.array([0.5, 0.5])
        T = jnp.array(320.0)
        P = jnp.array(8e5)
        hp = float(pr.enthalpy_departure(T, P, y, "vapor"))
        hs = float(srk.enthalpy_departure(T, P, y, "vapor"))
        assert abs(hp - hs) / abs(hp) < 0.1


class TestEntropyDeparture:
    """Entropy departure S(T,P) - S_ideal_gas(T,P) from the cubic EOS (issue #170).

    Covers both PengRobinson and SRK (SRK.entropy_departure is the
    epsilon=0/sigma=1 specialization of the generic-cubic departure).
    """

    @pytest.mark.parametrize("EOS", [PengRobinson, SRK])
    def test_thermodynamic_identity(self, EOS):
        """S_dep must equal (H_dep - G_dep)/T with G_dep = R*T*sum(y_i ln phi_i).

        This ties the new entropy departure to the already-tested enthalpy
        departure and fugacity coefficients, so an error in the closed form
        cannot pass unnoticed.
        """
        from difflow.eos import R
        from difflow.database import get_critical_props
        names = ["nitrogen", "carbon_dioxide", "methane", "ethane", "propane",
                 "n_butane"]
        eos = EOS({c: get_critical_props(c) for c in names})
        y = jnp.array([0.02, 0.03, 0.75, 0.12, 0.05, 0.03]); y = y / y.sum()
        T = jnp.array(250.0)
        P = jnp.array(40e5)
        for phase in ("vapor", "liquid"):
            h_dep = eos.enthalpy_departure(T, P, y, phase)
            s_dep = eos.entropy_departure(T, P, y, phase)
            g_dep = R * T * jnp.sum(y * jnp.log(eos.fugacity_coefficient(T, P, y, phase)))
            assert float(s_dep) == pytest.approx(float((h_dep - g_dep) / T), abs=1e-8)

    @pytest.mark.parametrize("EOS", [PengRobinson, SRK])
    def test_departure_vanishes_at_low_pressure(self, propane_butane_system, EOS):
        """As P -> 0 the gas is ideal, so the entropy departure -> 0."""
        eos = EOS(propane_butane_system)
        y = jnp.array([0.5, 0.5])
        s_dep = eos.entropy_departure(jnp.array(320.0), jnp.array(1.0e2), y, "vapor")
        assert abs(float(s_dep)) < 1e-2  # J/mol/K

    @pytest.mark.parametrize("EOS", [PengRobinson, SRK])
    def test_departure_negative_and_grows_for_vapor(self, propane_butane_system, EOS):
        """A compressed real vapor has lower entropy than the ideal gas, more so
        at higher pressure (departure becomes more negative)."""
        eos = EOS(propane_butane_system)
        y = jnp.array([0.5, 0.5])
        T = jnp.array(320.0)
        s_low = eos.entropy_departure(T, jnp.array(1e5), y, "vapor")
        s_high = eos.entropy_departure(T, jnp.array(8e5), y, "vapor")
        assert float(s_low) < 0.0
        assert float(s_high) < float(s_low)

    @pytest.mark.parametrize("EOS", [PengRobinson, SRK])
    def test_departure_differentiable(self, propane_butane_system, EOS):
        """d(S_dep)/dT is finite (jax.jvp path stays differentiable)."""
        eos = EOS(propane_butane_system)
        y = jnp.array([0.5, 0.5])
        g = jax.grad(
            lambda T: eos.entropy_departure(T, jnp.array(8e5), y, "vapor")
        )(jnp.array(320.0))
        assert jnp.isfinite(g)


def _propane_butane_ideal():
    """IdealThermo with ideal-gas Cp for the same species/order as the EOS."""
    species_data = {
        "propane": SpeciesData(
            name="propane",
            MW=44.10,
            Cp_coeffs=(73.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(18000.0, 0.38, 369.8),
            antoine_coeffs=(13.72, 1872.5, -25.16),
        ),
        "butane": SpeciesData(
            name="butane",
            MW=58.12,
            Cp_coeffs=(98.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(22000.0, 0.38, 425.1),
            antoine_coeffs=(13.98, 2292.4, -27.86),
        ),
    }
    return IdealThermo(species_data)


class TestCubicThermo:
    """CubicThermo enthalpy = ideal-gas sensible + EOS departure."""

    def test_no_pressure_reduces_to_ideal(self, propane_butane_system):
        """With P=None the departure is omitted -> pure ideal-gas sensible H."""
        ideal = _propane_butane_ideal()
        cubic = CubicThermo(ideal, PengRobinson(propane_butane_system))
        flows = {"propane": 1.0, "butane": 0.5}
        T = jnp.array(320.0)
        H_ideal = ideal.stream_enthalpy(flows, T, phase="liquid")
        H_cubic = cubic.stream_enthalpy(flows, T, P=None)
        assert float(H_cubic) == pytest.approx(float(H_ideal), rel=1e-9)

    def test_departure_lowers_vapor_enthalpy(self, propane_butane_system):
        """Adding the (negative) vapor departure lowers the stream enthalpy."""
        ideal = _propane_butane_ideal()
        cubic = CubicThermo(ideal, PengRobinson(propane_butane_system))
        flows = {"propane": 1.0, "butane": 0.5}
        T = jnp.array(320.0)
        H_ideal = cubic.stream_enthalpy(flows, T, P=None)
        H_real = cubic.stream_enthalpy(flows, T, phase="vapor", P=jnp.array(8e5))
        assert float(H_real) < float(H_ideal)

    @pytest.mark.parametrize("EOS", [PengRobinson, SRK])
    def test_works_with_pr_and_srk(self, propane_butane_system, EOS):
        """Regression: SRK gained enthalpy_departure, so CubicThermo(., SRK) works."""
        ideal = _propane_butane_ideal()
        cubic = CubicThermo(ideal, EOS(propane_butane_system))
        flows = {"propane": 1.0, "butane": 0.5}
        H = cubic.stream_enthalpy(
            flows, jnp.array(320.0), phase="vapor", P=jnp.array(8e5)
        )
        assert jnp.isfinite(H)

    def test_flash_enthalpy_single_phase_matches_named_phase(self, propane_butane_system):
        """For a single-phase vapor feed the flash enthalpy equals the vapor path."""
        ideal = _propane_butane_ideal()
        cubic = CubicThermo(ideal, PengRobinson(propane_butane_system))
        flows = {"propane": 1.0, "butane": 1.0}
        T = jnp.array(320.0)
        P = jnp.array(3e5)  # below the ~7.3 bar dew -> single-phase vapor
        H_flash = cubic.stream_enthalpy_flash(flows, T, P)
        H_vapor = cubic.stream_enthalpy(flows, T, phase="vapor", P=P)
        assert float(H_flash) == pytest.approx(float(H_vapor), rel=1e-6)

    def test_flash_enthalpy_differentiable(self, propane_butane_system):
        """stream_enthalpy_flash stays differentiable through the flash."""
        ideal = _propane_butane_ideal()
        cubic = CubicThermo(ideal, PengRobinson(propane_butane_system))
        flows = {"propane": 1.0, "butane": 1.0}
        g = jax.grad(
            lambda T: cubic.stream_enthalpy_flash(flows, T, jnp.array(8e5))
        )(jnp.array(320.0))
        assert jnp.isfinite(g)

    def test_entropy_increases_with_temperature(self, propane_butane_system):
        """Stream entropy rises with temperature at fixed P and composition."""
        ideal = _propane_butane_ideal()
        cubic = CubicThermo(ideal, PengRobinson(propane_butane_system))
        flows = {"propane": 1.0, "butane": 0.5}
        P = jnp.array(3e5)
        S_lo = cubic.stream_entropy(flows, jnp.array(300.0), "vapor", P)
        S_hi = cubic.stream_entropy(flows, jnp.array(340.0), "vapor", P)
        assert float(S_hi) > float(S_lo)

    def test_entropy_decreases_with_pressure(self, propane_butane_system):
        """At fixed T, entropy falls as pressure rises (-R ln(P/Pref) term)."""
        ideal = _propane_butane_ideal()
        cubic = CubicThermo(ideal, PengRobinson(propane_butane_system))
        flows = {"propane": 1.0, "butane": 0.5}
        T = jnp.array(340.0)
        S_lo = cubic.stream_entropy(flows, T, "vapor", jnp.array(1e5))
        S_hi = cubic.stream_entropy(flows, T, "vapor", jnp.array(5e5))
        assert float(S_hi) < float(S_lo)

    def test_isentropic_expansion_cools(self, propane_butane_system):
        """Matching entropy across a pressure drop yields a lower temperature --
        the physical basis of a turboexpander."""
        import optimistix as optx
        ideal = _propane_butane_ideal()
        cubic = CubicThermo(ideal, PengRobinson(propane_butane_system))
        flows = {"propane": 1.0, "butane": 0.5}
        T_in, P_in, P_out = jnp.array(340.0), jnp.array(8e5), jnp.array(2e5)
        S_in = cubic.stream_entropy_flash(flows, T_in, P_in)

        def resid(T, args):
            return cubic.stream_entropy_flash(flows, jnp.clip(T, 200.0, 400.0), P_out) - S_in

        sol = optx.root_find(resid, optx.Newton(rtol=1e-8, atol=1e-4),
                             jnp.array(320.0), max_steps=100, throw=False)
        T_out = float(jnp.clip(sol.value, 200.0, 400.0))
        assert T_out < float(T_in)

    def test_stream_entropy_flash_differentiable(self, propane_butane_system):
        """stream_entropy_flash stays differentiable through the flash."""
        ideal = _propane_butane_ideal()
        cubic = CubicThermo(ideal, PengRobinson(propane_butane_system))
        flows = {"propane": 1.0, "butane": 1.0}
        g = jax.grad(
            lambda T: cubic.stream_entropy_flash(flows, T, jnp.array(8e5))
        )(jnp.array(320.0))
        assert jnp.isfinite(g)
