"""Tests for liquid-liquid extraction unit operations."""

import jax
import jax.numpy as jnp
import pytest

from difflow import (
    MultistageCascade,
    CascadeParams,
    DifferentialContactor,
    ContactorParams,
    LLEEquilibrium,
    DistributionCoeffs,
    get_K_values,
    separation_factor,
    make_stream,
    get_flows,
)


# Enable 64-bit precision for tests
jax.config.update("jax_enable_x64", True)


@pytest.fixture
def simple_equilibrium():
    """Simple extraction equilibrium with two solutes."""
    K_coeffs = DistributionCoeffs(
        species=("A", "B"),
        K0=(2.0, 0.5),  # A prefers organic, B prefers aqueous
    )

    return LLEEquilibrium(
        solutes=["A", "B"],
        aqueous_carrier="H2O",
        organic_carrier="Solvent",
        K_coeffs=K_coeffs,
    )


class TestDistributionCoeffs:
    def test_get_K_values_constant(self):
        """Test constant K-values (no temperature dependence)."""
        coeffs = DistributionCoeffs(
            species=("A", "B"),
            K0=(2.0, 0.5),
        )
        K = get_K_values(coeffs, T=jnp.array(298.15))

        assert float(K["A"]) == pytest.approx(2.0)
        assert float(K["B"]) == pytest.approx(0.5)

    def test_get_K_values_temperature_dependent(self):
        """Test temperature-dependent K-values."""
        # K(T) = K0 * exp(-dH/R * (1/T - 1/Tref))
        # With positive dH and T > Tref, (1/T - 1/Tref) < 0
        # So -dH/R * (1/T - 1/Tref) > 0, meaning K increases with T
        coeffs = DistributionCoeffs(
            species=("A",),
            K0=(2.0,),
            dH=(10000.0,),  # Positive dH means K increases with T
            Tref=298.15,
        )

        K_ref = get_K_values(coeffs, T=jnp.array(298.15))
        K_hot = get_K_values(coeffs, T=jnp.array(350.0))

        # At Tref, K should equal K0
        assert float(K_ref["A"]) == pytest.approx(2.0)
        # With positive dH, K should increase at higher T
        assert float(K_hot["A"]) > float(K_ref["A"])


class TestSeparationFactor:
    def test_separation_factor(self):
        """Test separation factor calculation."""
        K1 = jnp.array(4.0)
        K2 = jnp.array(2.0)

        SF = separation_factor(K1, K2)
        assert float(SF) == pytest.approx(2.0)


class TestMultistageCascade:
    def test_cascade_creation(self, simple_equilibrium):
        """Test cascade can be created."""
        params = CascadeParams(
            n_stages=5,
            equilibrium=simple_equilibrium,
            flow_config="counter_current",
        )
        cascade = MultistageCascade(params)
        assert cascade is not None

    def test_cascade_mass_balance(self, simple_equilibrium):
        """Test mass balance in cascade."""
        params = CascadeParams(
            n_stages=5,
            equilibrium=simple_equilibrium,
            flow_config="counter_current",
        )
        cascade = MultistageCascade(params)

        feed = make_stream(
            {"H2O": 100.0, "A": 1.0, "B": 1.0, "Solvent": 0.0},
            T=298.15,
            P=101325.0,
        )
        solvent = make_stream(
            {"H2O": 0.0, "A": 0.0, "B": 0.0, "Solvent": 50.0},
            T=298.15,
            P=101325.0,
        )

        raffinate, extract, info = cascade(feed, solvent)

        # Check mass balance for each solute
        feed_flows = get_flows(feed)
        raff_flows = get_flows(raffinate)
        ext_flows = get_flows(extract)

        for species in ["A", "B"]:
            total_in = float(feed_flows.get(species, 0.0))
            total_out = float(raff_flows.get(species, 0.0)) + float(ext_flows.get(species, 0.0))
            assert total_out == pytest.approx(total_in, rel=1e-6)

    def test_cascade_extraction_direction(self, simple_equilibrium):
        """Test that species with high K are extracted more."""
        params = CascadeParams(
            n_stages=5,
            equilibrium=simple_equilibrium,
            flow_config="counter_current",
        )
        cascade = MultistageCascade(params)

        feed = make_stream(
            {"H2O": 100.0, "A": 1.0, "B": 1.0, "Solvent": 0.0},
            T=298.15,
            P=101325.0,
        )
        solvent = make_stream(
            {"H2O": 0.0, "A": 0.0, "B": 0.0, "Solvent": 50.0},
            T=298.15,
            P=101325.0,
        )

        raffinate, extract, info = cascade(feed, solvent)

        ext_flows = get_flows(extract)
        feed_flows = get_flows(feed)

        # A (K=2) should be extracted more than B (K=0.5)
        recovery_A = float(ext_flows["A"]) / float(feed_flows["A"])
        recovery_B = float(ext_flows["B"]) / float(feed_flows["B"])

        assert recovery_A > recovery_B

    def test_cascade_differentiability(self, simple_equilibrium):
        """Test that cascade is differentiable w.r.t. n_stages."""

        def recovery_A(n_stages):
            params = CascadeParams(
                n_stages=n_stages,
                equilibrium=simple_equilibrium,
                flow_config="counter_current",
            )
            cascade = MultistageCascade(params)

            feed = make_stream(
                {"H2O": 100.0, "A": 1.0, "B": 0.0, "Solvent": 0.0},
                T=298.15,
                P=101325.0,
            )
            solvent = make_stream(
                {"H2O": 0.0, "A": 0.0, "B": 0.0, "Solvent": 50.0},
                T=298.15,
                P=101325.0,
            )

            _, extract, _ = cascade(feed, solvent)
            return extract["F_A"]

        # Compute gradient
        grad_n = jax.grad(recovery_A)(jnp.array(5.0))

        # Gradient should be positive (more stages = more extraction)
        assert float(grad_n) > 0


class TestDifferentialContactor:
    def test_contactor_creation(self, simple_equilibrium):
        """Test contactor can be created."""
        params = ContactorParams(
            length=2.0,
            area=0.1,
            equilibrium=simple_equilibrium,
            HETP=0.5,
        )
        contactor = DifferentialContactor(params)
        assert contactor is not None

    def test_contactor_mass_balance(self, simple_equilibrium):
        """Test mass balance in differential contactor."""
        params = ContactorParams(
            length=2.0,
            area=0.1,
            equilibrium=simple_equilibrium,
            HETP=0.5,
        )
        contactor = DifferentialContactor(params)

        feed = make_stream(
            {"H2O": 100.0, "A": 1.0, "B": 1.0, "Solvent": 0.0},
            T=298.15,
            P=101325.0,
        )
        solvent = make_stream(
            {"H2O": 0.0, "A": 0.0, "B": 0.0, "Solvent": 50.0},
            T=298.15,
            P=101325.0,
        )

        raffinate, extract, info = contactor(feed, solvent)

        # Check mass balance
        feed_flows = get_flows(feed)
        raff_flows = get_flows(raffinate)
        ext_flows = get_flows(extract)

        for species in ["A", "B"]:
            total_in = float(feed_flows.get(species, 0.0))
            total_out = float(raff_flows.get(species, 0.0)) + float(ext_flows.get(species, 0.0))
            assert total_out == pytest.approx(total_in, rel=1e-4)
