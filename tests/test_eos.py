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
