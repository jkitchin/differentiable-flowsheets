"""Tests for difflow.kinetics.

The load-bearing test is `test_matches_a_hand_written_rate_fn`: a
reactor driven by the declarative rate law must produce exactly what
the same reactor produces from a hand-written callable. Everything else
checks the algebra against closed-form values, or checks that a
specification which cannot be turned into a correct rate law is refused
rather than approximated.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from difflow import CSTR, CSTRParams, get_flows, import_reactions, make_stream
from difflow.kinetics import (
    R_GAS,
    KineticsSpecError,
    ReactionSet,
    mass_action_kinetics,
)

SPECIES = ["A", "B"]


def first_order(A=1.0e6, Ea=50_000.0, n=0.0):
    """A -> B."""
    return [{
        "equation": "A -> B",
        "reactants": {"A": 1.0}, "products": {"B": 1.0},
        "rate_params": {"A": A, "Ea": Ea, "n": n},
    }]


def reversible(K_eq=4.0):
    """A <=> B."""
    return [{
        "equation": "A <=> B",
        "reactants": {"A": 1.0}, "products": {"B": 1.0},
        "rate_params": {"A": 2.0, "Ea": 0.0, "n": 0.0},
        "reversible": True, "K_eq": K_eq,
    }]


# =============================================================================
# Stoichiometry
# =============================================================================


class TestStoichiometry:
    def test_signs_and_coefficients(self):
        """Negative for reactants, positive for products."""
        rxns = [{
            "equation": "2 A -> 3 B",
            "reactants": {"A": 2.0}, "products": {"B": 3.0},
            "rate_params": {"A": 1.0},
        }]
        kin = mass_action_kinetics(rxns, SPECIES)
        np.testing.assert_allclose(np.asarray(kin.stoich), [[-2.0], [3.0]])

    def test_shape_is_species_by_reactions(self):
        kin = mass_action_kinetics(
            first_order() + first_order(), SPECIES
        )
        assert kin.stoich.shape == (2, 2)
        assert kin.n_species == 2 and kin.n_reactions == 2

    def test_species_order_defaults_to_sorted_union(self):
        rxns = [{
            "equation": "B -> A", "reactants": {"B": 1.0}, "products": {"A": 1.0},
            "rate_params": {"A": 1.0},
        }]
        assert mass_action_kinetics(rxns).species_order == ["A", "B"]

    def test_species_order_is_respected(self):
        """Row order must follow the caller's species order, not sorted."""
        kin = mass_action_kinetics(first_order(), ["B", "A"])
        np.testing.assert_allclose(np.asarray(kin.stoich), [[1.0], [-1.0]])

    def test_a_species_on_both_sides_nets_out(self):
        rxns = [{
            "equation": "2 A -> A + B",
            "reactants": {"A": 2.0}, "products": {"A": 1.0, "B": 1.0},
            "rate_params": {"A": 1.0},
        }]
        kin = mass_action_kinetics(rxns, SPECIES)
        # net -1 in A, but the rate is still second order in A
        np.testing.assert_allclose(np.asarray(kin.stoich), [[-1.0], [1.0]])
        assert float(kin.rates({"A": 2.0, "B": 0.0}, 300.0)[0]) == pytest.approx(4.0)


# =============================================================================
# The rate law
# =============================================================================


class TestRateLaw:
    def test_first_order_matches_arrhenius(self):
        kin = mass_action_kinetics(first_order(), SPECIES)
        T, c_a = 350.0, 2.0
        expected = 1.0e6 * np.exp(-50_000.0 / (R_GAS * T)) * c_a
        got = float(kin.rates({"A": c_a, "B": 0.0}, T)[0])
        assert got == pytest.approx(expected, rel=1e-12)

    def test_order_comes_from_reactant_stoichiometry(self):
        rxns = [{
            "equation": "2 A -> B", "reactants": {"A": 2.0}, "products": {"B": 1.0},
            "rate_params": {"A": 3.0, "Ea": 0.0, "n": 0.0},
        }]
        kin = mass_action_kinetics(rxns, SPECIES)
        assert float(kin.rates({"A": 2.0, "B": 0.0}, 300.0)[0]) == pytest.approx(
            3.0 * 2.0**2
        )

    def test_temperature_exponent(self):
        kin = mass_action_kinetics(first_order(A=2.0, Ea=0.0, n=1.5), SPECIES)
        T = 400.0
        assert float(kin.rates({"A": 1.0, "B": 0.0}, T)[0]) == pytest.approx(
            2.0 * T**1.5
        )

    def test_rate_rises_with_temperature(self):
        kin = mass_action_kinetics(first_order(), SPECIES)
        c = {"A": 1.0, "B": 0.0}
        assert float(kin.rates(c, 400.0)[0]) > float(kin.rates(c, 300.0)[0])

    def test_zero_concentration_is_finite_and_negligible(self):
        """A missing reactant must give no rate, and no nan.

        The log-space product floors the concentration rather than
        evaluating 0 ** order, so the rate underflows instead of
        reaching exactly zero. The residue is 300 orders of magnitude
        below anything physical.
        """
        kin = mass_action_kinetics(first_order(Ea=0.0), SPECIES)
        r = kin.rates({"A": 0.0, "B": 0.0}, 300.0)
        assert bool(jnp.all(jnp.isfinite(r)))
        assert abs(float(r[0])) < 1e-280

    def test_a_zero_order_reactant_still_reacts_at_zero(self):
        """0 ** 0 is 1, not 0: a zeroth-order rate is concentration-free."""
        rxns = [{
            "equation": "A -> B", "reactants": {"A": 1.0}, "products": {"B": 1.0},
            "rate_params": {"A": 7.0, "Ea": 0.0, "n": 0.0},
        }]
        kin = mass_action_kinetics(rxns, SPECIES, orders=[{}])
        assert float(kin.rates({"A": 0.0, "B": 0.0}, 300.0)[0]) == pytest.approx(7.0)

    def test_explicit_orders_override_stoichiometry(self):
        """Empirical orders need not equal the coefficients."""
        rxns = [{
            "equation": "2 A -> B", "reactants": {"A": 2.0}, "products": {"B": 1.0},
            "rate_params": {"A": 5.0, "Ea": 0.0, "n": 0.0},
        }]
        kin = mass_action_kinetics(rxns, SPECIES, orders=[{"A": 1.0}])
        # first order in A despite the coefficient of 2 ...
        assert float(kin.rates({"A": 3.0, "B": 0.0}, 300.0)[0]) == pytest.approx(15.0)
        # ... while the stoichiometry is untouched
        np.testing.assert_allclose(np.asarray(kin.stoich), [[-2.0], [1.0]])

    def test_multiple_reactions_are_independent(self):
        rxns = first_order(A=1.0, Ea=0.0) + [{
            "equation": "B -> A", "reactants": {"B": 1.0}, "products": {"A": 1.0},
            "rate_params": {"A": 10.0, "Ea": 0.0, "n": 0.0},
        }]
        kin = mass_action_kinetics(rxns, SPECIES)
        r = kin.rates({"A": 2.0, "B": 3.0}, 300.0)
        assert float(r[0]) == pytest.approx(2.0)
        assert float(r[1]) == pytest.approx(30.0)


# =============================================================================
# Reversibility
# =============================================================================


class TestReversibility:
    def test_refused_by_default(self):
        """Forward Arrhenius alone does not determine the reverse rate."""
        with pytest.raises(KineticsSpecError, match="reversible"):
            mass_action_kinetics(reversible(), SPECIES)

    def test_error_names_the_reaction(self):
        with pytest.raises(KineticsSpecError) as exc:
            mass_action_kinetics(reversible(), SPECIES)
        assert "A <=> B" in str(exc.value)

    def test_forward_only_drops_the_reverse_term(self):
        kin = mass_action_kinetics(reversible(), SPECIES, reverse="forward_only")
        # 2 * C_A, with no dependence on C_B at all
        assert float(kin.rates({"A": 3.0, "B": 8.0}, 300.0)[0]) == pytest.approx(6.0)
        assert float(kin.rates({"A": 3.0, "B": 0.0}, 300.0)[0]) == pytest.approx(6.0)
        assert kin.reverse == "forward_only"

    def test_equilibrium_subtracts_the_reverse_term(self):
        kin = mass_action_kinetics(reversible(K_eq=4.0), SPECIES,
                                   reverse="equilibrium")
        # k (C_A - C_B / K_eq) = 2 (3 - 8/4)
        assert float(kin.rates({"A": 3.0, "B": 8.0}, 300.0)[0]) == pytest.approx(2.0)

    def test_equilibrium_rate_vanishes_at_equilibrium(self):
        """The defining property: r = 0 when C_B / C_A = K_eq."""
        kin = mass_action_kinetics(reversible(K_eq=4.0), SPECIES,
                                   reverse="equilibrium")
        assert float(kin.rates({"A": 2.0, "B": 8.0}, 300.0)[0]) == pytest.approx(
            0.0, abs=1e-12
        )

    def test_equilibrium_reverses_sign_past_equilibrium(self):
        kin = mass_action_kinetics(reversible(K_eq=4.0), SPECIES,
                                   reverse="equilibrium")
        assert float(kin.rates({"A": 1.0, "B": 20.0}, 300.0)[0]) < 0.0

    def test_equilibrium_requires_k_eq(self):
        rxn = reversible()
        del rxn[0]["K_eq"]
        with pytest.raises(KineticsSpecError, match="K_eq"):
            mass_action_kinetics(rxn, SPECIES, reverse="equilibrium")

    def test_irreversible_reactions_ignore_the_mode(self):
        for mode in ("error", "forward_only", "equilibrium"):
            kin = mass_action_kinetics(first_order(Ea=0.0), SPECIES, reverse=mode)
            assert float(kin.rates({"A": 1.0, "B": 5.0}, 300.0)[0]) == pytest.approx(
                1.0e6
            )


# =============================================================================
# Rejected specifications
# =============================================================================


class TestValidation:
    def test_unsupported_reaction_type_is_refused(self):
        """Falloff and three-body need their own rate law, not an approximation."""
        for kind in ("falloff", "three-body", "pressure-dependent-Arrhenius"):
            rxns = [{**first_order()[0], "type": kind}]
            with pytest.raises(KineticsSpecError, match="unsupported reaction type"):
                mass_action_kinetics(rxns, SPECIES)

    def test_unknown_species_is_refused(self):
        with pytest.raises(KineticsSpecError, match="not in species_order"):
            mass_action_kinetics(first_order(), ["A"])

    def test_empty_reaction_list_is_refused(self):
        with pytest.raises(KineticsSpecError, match="no reactions"):
            mass_action_kinetics([], SPECIES)

    def test_a_reaction_with_no_species_is_refused(self):
        """`equation` is a label, not parsed -- so this reacts nothing."""
        with pytest.raises(KineticsSpecError, match="no reactants and no products"):
            mass_action_kinetics(
                [{"equation": "A -> B", "rate_params": {"A": 1.0}}], SPECIES
            )

    def test_the_refusal_says_where_stoichiometry_comes_from(self):
        """The whole mistake is expecting the equation string to be read."""
        with pytest.raises(KineticsSpecError) as exc:
            mass_action_kinetics([{"equation": "A -> B"}], SPECIES)
        assert "'equation' is only a label" in str(exc.value)

    def test_a_half_specified_reaction_is_allowed(self):
        """Decomposition to nothing tracked, or generation from a feed."""
        kin = mass_action_kinetics(
            [{"equation": "A ->", "reactants": {"A": 1.0},
              "rate_params": {"A": 1.0}}], SPECIES
        )
        assert float(kin.stoich[SPECIES.index("A"), 0]) == -1.0

    def test_bad_reverse_mode_is_refused(self):
        with pytest.raises(ValueError, match="reverse="):
            mass_action_kinetics(first_order(), SPECIES, reverse="maybe")

    def test_order_for_unknown_species_is_refused(self):
        with pytest.raises(KineticsSpecError, match="not in species_order"):
            mass_action_kinetics(first_order(), SPECIES, orders=[{"Z": 1.0}])


# =============================================================================
# Differentiability
# =============================================================================


class TestDifferentiability:
    def test_gradient_wrt_concentration(self):
        rxns = [{
            "equation": "2 A -> B", "reactants": {"A": 2.0}, "products": {"B": 1.0},
            "rate_params": {"A": 3.0, "Ea": 0.0, "n": 0.0},
        }]
        kin = mass_action_kinetics(rxns, SPECIES)
        g = jax.grad(lambda c: kin.rates({"A": c, "B": 0.0}, 300.0)[0])(2.0)
        # d/dC (3 C^2) = 6 C
        assert float(g) == pytest.approx(12.0, rel=1e-9)

    def test_gradient_at_zero_concentration_is_finite(self):
        kin = mass_action_kinetics(first_order(Ea=0.0), SPECIES)
        g = jax.grad(lambda c: kin.rates({"A": c, "B": 0.0}, 300.0)[0])(0.0)
        assert bool(jnp.isfinite(g))

    def test_gradient_wrt_temperature_is_positive(self):
        kin = mass_action_kinetics(first_order(), SPECIES)
        g = jax.grad(lambda T: kin.rates({"A": 1.0, "B": 0.0}, T)[0])(350.0)
        assert bool(jnp.isfinite(g)) and float(g) > 0.0

    def test_rate_params_are_a_differentiable_pytree(self):
        """Fitting a rate constant differentiates through rate_params."""
        kin = mass_action_kinetics(first_order(), SPECIES)

        def rate(pre_exponential):
            rp = {**kin.rate_params, "A": jnp.array([pre_exponential])}
            return kin.rate_fn({"A": 1.0, "B": 0.0}, 350.0, rp)[0]

        g = jax.grad(rate)(1.0e6)
        assert bool(jnp.isfinite(g)) and float(g) > 0.0

    def test_can_be_built_inside_a_trace(self):
        """The structure is static, but the coefficients may be traced."""
        def rate(pre_exponential):
            kin = mass_action_kinetics(first_order(A=pre_exponential), SPECIES)
            return kin.rates({"A": 1.0, "B": 0.0}, 350.0)[0]

        assert bool(jnp.isfinite(jax.grad(rate)(1.0e6)))
        assert bool(jnp.isfinite(jax.jit(rate)(1.0e6)))

    def test_jit(self):
        kin = mass_action_kinetics(first_order(), SPECIES)
        fn = jax.jit(lambda c, T: kin.rate_fn({"A": c, "B": 0.0}, T, kin.rate_params))
        assert bool(jnp.all(jnp.isfinite(fn(1.0, 350.0))))


# =============================================================================
# Driving a real reactor
# =============================================================================


class TestReactorIntegration:
    def test_matches_a_hand_written_rate_fn(self):
        """The whole point: a declarative rate law is not an approximation.

        Same reactor, same numbers, one built from a Python callable and
        one from data.
        """
        def hand_rate(C, T, p):
            k = p["k0"] * jnp.exp(-p["Ea"] / (R_GAS * T))
            return jnp.array([k * C["A"]])

        hand = CSTRParams(
            V=1.5, rate_fn=hand_rate, stoich=jnp.array([[-1.0], [1.0]]),
            rate_params={"k0": 1.0e6, "Ea": 50_000.0},
            species_order=SPECIES, molar_density=1000.0,
        )
        kin = mass_action_kinetics(first_order(), SPECIES)
        declarative = CSTRParams(V=1.5, molar_density=1000.0, **kin.params_kwargs())

        feed = make_stream({"A": 1.0, "B": 0.0}, T=350.0, P=101325.0)

        def outlet_flows(params):
            out = CSTR(params)(feed)
            stream = out[0] if isinstance(out, tuple) else out
            return {k: float(v) for k, v in get_flows(stream).items()}

        a, b = outlet_flows(hand), outlet_flows(declarative)
        for species in a:
            assert a[species] == pytest.approx(b[species], abs=1e-12), (
                f"{species}: hand {a[species]} vs declarative {b[species]}"
            )
        assert b["B"] > 0.9, "the reaction should have run"

    def test_params_kwargs_supplies_every_reactor_field(self):
        kin = mass_action_kinetics(first_order(), SPECIES)
        kwargs = kin.params_kwargs()
        assert set(kwargs) == {"rate_fn", "stoich", "rate_params", "species_order"}
        CSTRParams(V=1.0, **kwargs)          # must not raise

    def test_gradient_through_the_reactor(self):
        kin = mass_action_kinetics(first_order(), SPECIES)
        feed = make_stream({"A": 1.0, "B": 0.0}, T=350.0, P=101325.0)

        def product_flow(pre_exponential):
            rp = {**kin.rate_params, "A": jnp.array([pre_exponential])}
            params = CSTRParams(
                V=1.5, molar_density=1000.0, rate_fn=kin.rate_fn,
                stoich=kin.stoich, rate_params=rp, species_order=SPECIES,
            )
            out = CSTR(params)(feed)
            stream = out[0] if isinstance(out, tuple) else out
            return get_flows(stream)["B"]

        g = jax.grad(product_flow)(1.0e6)
        assert bool(jnp.isfinite(g))
        assert float(g) > 0.0, "a faster reaction must make more product"


# =============================================================================
# The Cantera path
# =============================================================================


@pytest.fixture(scope="module")
def reactions():
    """The repo's Cantera test mechanism, as reaction dictionaries."""
    return import_reactions("tests/data/test_mechanism.yaml")


class TestFromCanteraYAML:
    def test_import_reactions_output_is_accepted(self, reactions):
        """No adapter needed between the importer and the rate law."""
        kin = mass_action_kinetics(reactions, reverse="forward_only")
        assert kin.n_reactions == len(reactions)
        assert kin.species_order == ["CH4", "CO", "CO2", "H2", "H2O", "O2"]
        assert kin.stoich.shape == (6, len(reactions))

    def test_rates_are_finite(self, reactions):
        kin = mass_action_kinetics(reactions, reverse="forward_only")
        r = kin.rates({s: 10.0 for s in kin.species_order}, 1200.0)
        assert bool(jnp.all(jnp.isfinite(r)))
        assert bool(jnp.all(r > 0)), "forward rates should be positive"

    def test_a_reversible_mechanism_is_refused_by_default(self, reactions):
        with pytest.raises(KineticsSpecError, match="reversible"):
            mass_action_kinetics(reactions)

    def test_elements_are_conserved_by_the_stoichiometry(self, reactions):
        """Carbon balance over every reaction, from the parsed coefficients."""
        carbon = {"CH4": 1, "CO": 1, "CO2": 1, "H2": 0, "H2O": 0, "O2": 0}
        kin = mass_action_kinetics(reactions, reverse="forward_only")
        counts = jnp.array([carbon[s] for s in kin.species_order], dtype=jnp.float64)
        np.testing.assert_allclose(
            np.asarray(counts @ kin.stoich), np.zeros(kin.n_reactions), atol=1e-12
        )


# =============================================================================
# Reporting
# =============================================================================


class TestReporting:
    def test_equations_follow_the_unit_operation_convention(self):
        kin = mass_action_kinetics(first_order(), SPECIES)
        assert len(kin.equations) == 1
        assert "C_{A}" in kin.equations[0]

    def test_equations_show_the_reverse_term_only_when_used(self):
        fwd = mass_action_kinetics(reversible(), SPECIES, reverse="forward_only")
        eq = mass_action_kinetics(reversible(), SPECIES, reverse="equilibrium")
        assert "K_{eq}" not in fwd.equations[0]
        assert "K_{eq}" in eq.equations[0]

    def test_order_appears_in_the_latex(self):
        rxns = [{
            "equation": "2 A -> B", "reactants": {"A": 2.0}, "products": {"B": 1.0},
            "rate_params": {"A": 1.0},
        }]
        assert "C_{A}^{2}" in mass_action_kinetics(rxns, SPECIES).equations[0]

    def test_summary_reports_the_coefficients(self):
        kin = mass_action_kinetics(first_order(), SPECIES)
        text = kin.summary()
        assert "A -> B" in text and "50.00" in text     # Ea in kJ/mol

    def test_source_reactions_are_kept_for_round_tripping(self):
        rxns = first_order()
        kin = mass_action_kinetics(rxns, SPECIES)
        assert kin.reactions == rxns

    def test_is_a_params_mixin(self):
        kin = mass_action_kinetics(first_order(), SPECIES)
        assert isinstance(kin, ReactionSet)
        assert "stoich" in kin
        assert kin["reverse"] == "error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
