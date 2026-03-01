"""Tests for bio plugin bug fixes (issues #93-#98)."""

import jax
import jax.numpy as jnp
import pytest

from difflow_bio import (
    ContinuousBioreactor,
    BioreactorParams,
    monod_kinetics,
    product_inhibition_kinetics,
    contois_kinetics,
)
from difflow_bio.units.bioreactors import call_kinetics, get_kinetic_arity
from difflow_bio import (
    ProteinAChromatography,
    ProteinAParams,
    IonExchangeChromatography,
    IEXParams,
    SizeExclusionChromatography,
    SECParams,
)
from difflow_bio import (
    Centrifuge,
    CentrifugeParams,
    DiscStackCentrifuge,
    DiscStackParams,
)
from difflow_bio import (
    Ultrafiltration,
    UltrafiltrationParams,
    Diafiltration,
    DiafiltrationParams,
)
from difflow import make_stream, get_flows

jax.config.update("jax_enable_x64", True)


# =========================================================================
# Issue #93: call_kinetics arity dispatch for product inhibition
# =========================================================================

class TestCallKineticsDispatch:
    """Tests for correct kinetic function dispatch (issue #93)."""

    def test_monod_arity(self):
        """Monod kinetics has arity 1 (S only, plus params)."""
        arity = get_kinetic_arity(monod_kinetics)
        assert arity == 1

    def test_product_inhibition_arity(self):
        """Product inhibition has arity 2 (S, P, plus params)."""
        arity = get_kinetic_arity(product_inhibition_kinetics)
        assert arity == 2

    def test_contois_arity(self):
        """Contois has arity 2 (S, X, plus params)."""
        arity = get_kinetic_arity(contois_kinetics)
        assert arity == 2

    def test_product_inhibition_dispatch(self):
        """Product inhibition should receive S and P, not S and X."""
        params = {
            "mu_max": jnp.array(0.5),
            "K_s": jnp.array(0.1),
            "P_max": jnp.array(50.0),
            "n": 1.0,
        }
        S = jnp.array(10.0)
        X = jnp.array(5.0)   # Cell conc (should NOT be passed)
        P = jnp.array(25.0)  # Product conc (should be passed)

        arity = get_kinetic_arity(product_inhibition_kinetics)

        # call_kinetics should dispatch to product_inhibition_kinetics(S, P, params)
        mu_dispatched = call_kinetics(
            product_inhibition_kinetics, arity, S, X, P, params
        )

        # Direct call for comparison
        mu_direct = product_inhibition_kinetics(S, P, params)

        assert float(mu_dispatched) == pytest.approx(float(mu_direct), rel=1e-10), (
            f"Dispatch gave {float(mu_dispatched)}, direct gave {float(mu_direct)}. "
            "Product inhibition is receiving wrong arguments."
        )

    def test_contois_dispatch(self):
        """Contois kinetics should receive S and X."""
        params = {
            "mu_max": jnp.array(0.5),
            "K_s": jnp.array(0.1),
        }
        S = jnp.array(10.0)
        X = jnp.array(5.0)
        P = jnp.array(0.0)

        arity = get_kinetic_arity(contois_kinetics)
        mu_dispatched = call_kinetics(contois_kinetics, arity, S, X, P, params)
        mu_direct = contois_kinetics(S, X, params)

        assert float(mu_dispatched) == pytest.approx(float(mu_direct), rel=1e-10)

    def test_product_inhibition_reduces_growth(self):
        """At high product concentration, growth should be inhibited."""
        params = {
            "mu_max": jnp.array(0.5),
            "K_s": jnp.array(0.1),
            "P_max": jnp.array(50.0),
            "n": 1.0,
        }
        S = jnp.array(10.0)
        X = jnp.array(5.0)

        # Low product
        P_low = jnp.array(1.0)
        mu_low = call_kinetics(
            product_inhibition_kinetics,
            get_kinetic_arity(product_inhibition_kinetics),
            S, X, P_low, params,
        )

        # High product (near P_max)
        P_high = jnp.array(45.0)
        mu_high = call_kinetics(
            product_inhibition_kinetics,
            get_kinetic_arity(product_inhibition_kinetics),
            S, X, P_high, params,
        )

        assert float(mu_high) < float(mu_low), (
            "Growth rate should decrease with higher product concentration"
        )


# =========================================================================
# Issue #94: Chromatography mass balance
# =========================================================================

class TestChromatographyMassBalance:
    """Tests for closed mass balance in chromatography (issue #94)."""

    def test_protein_a_mass_balance(self):
        """Product + waste should equal feed for Protein A."""
        feed = make_stream(
            {"mAb": jnp.array(10.0), "HCP": jnp.array(1.0), "DNA": jnp.array(0.1)},
            T=jnp.array(300.0),
            P=jnp.array(101325.0),
        )
        params = ProteinAParams(
            column_volume=1.0,
            q_max=35.0,
            K_d=0.1,
            target_species="mAb",
            yield_factor=0.95,
            impurity_clearance={"HCP": 2.0, "DNA": 3.0},
        )
        col = ProteinAChromatography(params)
        (product, waste), info = col(feed, load_volume=11.1, feed_volume=11.1)

        product_flows = get_flows(product)
        waste_flows = get_flows(waste)
        feed_flows = get_flows(feed)

        for species in feed_flows:
            total_out = float(product_flows.get(species, 0.0)) + float(
                waste_flows.get(species, 0.0)
            )
            assert total_out == pytest.approx(float(feed_flows[species]), rel=1e-6), (
                f"Mass balance violated for {species}: "
                f"feed={float(feed_flows[species])}, total_out={total_out}"
            )

    def test_protein_a_partial_load_mass_balance(self):
        """When only partial feed is loaded, unloaded mass goes to waste."""
        feed = make_stream(
            {"mAb": jnp.array(10.0), "HCP": jnp.array(1.0)},
            T=jnp.array(300.0),
            P=jnp.array(101325.0),
        )
        params = ProteinAParams(
            column_volume=1.0,
            q_max=35.0,
            K_d=0.1,
            target_species="mAb",
            yield_factor=0.95,
            impurity_clearance={"HCP": 2.0},
        )
        col = ProteinAChromatography(params)
        # Only load half the feed
        (product, waste), info = col(feed, load_volume=5.5, feed_volume=11.0)

        product_flows = get_flows(product)
        waste_flows = get_flows(waste)
        feed_flows = get_flows(feed)

        for species in feed_flows:
            total_out = float(product_flows.get(species, 0.0)) + float(
                waste_flows.get(species, 0.0)
            )
            assert total_out == pytest.approx(float(feed_flows[species]), rel=1e-6), (
                f"Mass balance violated for {species} with partial load"
            )

    def test_iex_mass_balance(self):
        """IEX product + waste should equal feed."""
        feed = make_stream(
            {"mAb": jnp.array(8.0), "HCP": jnp.array(0.5), "aggregate": jnp.array(0.2)},
            T=jnp.array(300.0),
            P=jnp.array(101325.0),
        )
        params = IEXParams(
            column_volume=1.0,
            mode="bind_elute",
            target_species="mAb",
            selectivity={"HCP": 0.3, "aggregate": 0.8},
            yield_factor=0.90,
        )
        col = IonExchangeChromatography(params)
        total_flow = sum(get_flows(feed).values())
        (product, waste), info = col(feed, load_volume=total_flow)

        product_flows = get_flows(product)
        waste_flows = get_flows(waste)
        feed_flows = get_flows(feed)

        for species in feed_flows:
            total_out = float(product_flows.get(species, 0.0)) + float(
                waste_flows.get(species, 0.0)
            )
            assert total_out == pytest.approx(float(feed_flows[species]), rel=1e-6), (
                f"IEX mass balance violated for {species}"
            )

    def test_sec_mass_balance(self):
        """SEC product + aggregates + fragments should equal feed."""
        feed = make_stream(
            {
                "mAb": jnp.array(8.0),
                "aggregates": jnp.array(0.3),
                "fragments": jnp.array(0.2),
            },
            T=jnp.array(300.0),
            P=jnp.array(101325.0),
        )
        params = SECParams(
            column_volume=1.0,
            target_species="mAb",
            aggregate_species="aggregates",
            fragment_species="fragments",
            yield_factor=0.95,
        )
        col = SizeExclusionChromatography(params)
        total_flow = sum(get_flows(feed).values())
        (product, aggregates, fragments), info = col(feed, load_volume=total_flow)

        product_flows = get_flows(product)
        agg_flows = get_flows(aggregates)
        frag_flows = get_flows(fragments)
        feed_flows = get_flows(feed)

        for species in feed_flows:
            total_out = (
                float(product_flows.get(species, 0.0))
                + float(agg_flows.get(species, 0.0))
                + float(frag_flows.get(species, 0.0))
            )
            assert total_out == pytest.approx(float(feed_flows[species]), rel=1e-6), (
                f"SEC mass balance violated for {species}"
            )


# =========================================================================
# Issue #95: Centrifuge mass conservation
# =========================================================================

class TestCentrifugeMassBalance:
    """Tests for centrifuge mass conservation (issue #95)."""

    def test_centrifuge_mass_balance(self):
        """Concentrate + clarified should equal feed for all species."""
        feed = make_stream(
            {
                "cells": jnp.array(5.0),
                "substrate": jnp.array(20.0),
                "product": jnp.array(3.0),
            },
            T=jnp.array(300.0),
            P=jnp.array(101325.0),
        )
        params = CentrifugeParams(sigma=1000.0, efficiency=0.7, cell_species="cells")
        centrifuge = Centrifuge(params)
        concentrate, clarified, info = centrifuge(
            feed,
            Q=1e-4,
            d_particle=5e-6,
            concentrate_fraction=0.1,
        )

        conc_flows = get_flows(concentrate)
        clar_flows = get_flows(clarified)
        feed_flows = get_flows(feed)

        for species in feed_flows:
            total_out = float(conc_flows.get(species, 0.0)) + float(
                clar_flows.get(species, 0.0)
            )
            assert total_out == pytest.approx(float(feed_flows[species]), rel=1e-10), (
                f"Centrifuge mass balance violated for {species}: "
                f"feed={float(feed_flows[species])}, total_out={total_out}"
            )

    def test_disc_stack_mass_balance(self):
        """DiscStack centrifuge should also conserve mass."""
        feed = make_stream(
            {"cells": jnp.array(3.0), "product": jnp.array(10.0)},
            T=jnp.array(300.0),
            P=jnp.array(101325.0),
        )
        params = DiscStackParams(
            n_discs=50,
            r_outer=0.1,
            r_inner=0.04,
            rpm=6000.0,
            efficiency=0.7,
            cell_species="cells",
        )
        centrifuge = DiscStackCentrifuge(params)
        concentrate, clarified, info = centrifuge(
            feed, Q=1e-4, concentrate_fraction=0.15
        )

        conc_flows = get_flows(concentrate)
        clar_flows = get_flows(clarified)
        feed_flows = get_flows(feed)

        for species in feed_flows:
            total_out = float(conc_flows.get(species, 0.0)) + float(
                clar_flows.get(species, 0.0)
            )
            assert total_out == pytest.approx(float(feed_flows[species]), rel=1e-10), (
                f"DiscStack mass balance violated for {species}"
            )


# =========================================================================
# Issue #96: UF batch concentration formula
# =========================================================================

class TestUFBatchConcentration:
    """Tests for exact UF concentration formula (issue #96)."""

    def test_fully_retained_species(self):
        """R=1 species should be fully retained (mass conserved in retentate)."""
        feed = make_stream(
            {"protein": jnp.array(10.0), "salt": jnp.array(5.0)},
            T=jnp.array(300.0),
            P=jnp.array(101325.0),
        )
        params = UltrafiltrationParams(
            membrane_area=1.0,
            rejection={"protein": 1.0, "salt": 0.0},
        )
        uf = Ultrafiltration(params)
        (retentate, permeate), info = uf(feed, concentration_factor=5.0)

        ret_flows = get_flows(retentate)
        perm_flows = get_flows(permeate)

        # Fully retained protein: all in retentate
        assert float(ret_flows["protein"]) == pytest.approx(10.0, rel=1e-6)
        assert float(perm_flows["protein"]) == pytest.approx(0.0, abs=1e-10)

    def test_fully_permeable_species(self):
        """R=0 species at CF=5 should have 1/5 retained."""
        feed = make_stream(
            {"protein": jnp.array(10.0), "salt": jnp.array(5.0)},
            T=jnp.array(300.0),
            P=jnp.array(101325.0),
        )
        params = UltrafiltrationParams(
            membrane_area=1.0,
            rejection={"protein": 1.0, "salt": 0.0},
        )
        uf = Ultrafiltration(params)
        (retentate, permeate), info = uf(feed, concentration_factor=5.0)

        ret_flows = get_flows(retentate)
        perm_flows = get_flows(permeate)

        # R=0: retained_frac = CF^(0-1) = 1/CF = 0.2
        assert float(ret_flows["salt"]) == pytest.approx(5.0 * 0.2, rel=1e-6)
        assert float(perm_flows["salt"]) == pytest.approx(5.0 * 0.8, rel=1e-6)

    def test_partial_rejection_high_cf(self):
        """At high CF with partial rejection, exact formula should match CF^(R-1)."""
        feed = make_stream(
            {"protein": jnp.array(10.0)},
            T=jnp.array(300.0),
            P=jnp.array(101325.0),
        )
        R = 0.5
        CF = 10.0
        params = UltrafiltrationParams(
            membrane_area=1.0,
            rejection={"protein": R},
        )
        uf = Ultrafiltration(params)
        (retentate, permeate), info = uf(feed, concentration_factor=CF)

        ret_flows = get_flows(retentate)

        # Exact formula: retained_frac = CF^(R-1) = 10^(-0.5) = 0.3162...
        expected_retained = 10.0 * CF ** (R - 1.0)
        assert float(ret_flows["protein"]) == pytest.approx(
            expected_retained, rel=1e-4
        ), (
            f"UF batch concentration at high CF should use exact CF^(R-1) formula. "
            f"Got {float(ret_flows['protein'])}, expected {expected_retained}"
        )

    def test_uf_mass_balance(self):
        """Retentate + permeate should equal feed."""
        feed = make_stream(
            {"protein": jnp.array(10.0), "salt": jnp.array(5.0)},
            T=jnp.array(300.0),
            P=jnp.array(101325.0),
        )
        params = UltrafiltrationParams(
            membrane_area=1.0,
            rejection={"protein": 0.99, "salt": 0.1},
        )
        uf = Ultrafiltration(params)
        (retentate, permeate), info = uf(feed, concentration_factor=5.0)

        ret_flows = get_flows(retentate)
        perm_flows = get_flows(permeate)
        feed_flows = get_flows(feed)

        for species in feed_flows:
            total = float(ret_flows.get(species, 0.0)) + float(
                perm_flows.get(species, 0.0)
            )
            assert total == pytest.approx(float(feed_flows[species]), rel=1e-6), (
                f"UF mass balance violated for {species}"
            )


# =========================================================================
# Issue #97: Diafiltration wash-in formula
# =========================================================================

class TestDiafiltrationWashFormula:
    """Tests for correct diafiltration wash equations (issue #97)."""

    def test_impurity_removal(self):
        """Impurity (R=0) should follow exp(-N) exactly."""
        feed = make_stream(
            {"protein": jnp.array(10.0), "impurity": jnp.array(5.0)},
            T=jnp.array(300.0),
            P=jnp.array(101325.0),
        )
        buffer = make_stream(
            {"buffer_salt": jnp.array(1.0)},
            T=jnp.array(300.0),
            P=jnp.array(101325.0),
        )
        N = 5.0
        params = DiafiltrationParams(
            membrane_area=1.0,
            rejection={"protein": 1.0, "impurity": 0.0, "buffer_salt": 0.0},
        )
        df = Diafiltration(params)
        (retentate, permeate), info = df(feed, buffer, n_diavolumes=N)

        ret_flows = get_flows(retentate)

        # R=0 impurity: C/C0 = exp(-N) => remaining = 5.0 * exp(-5)
        expected_impurity = 5.0 * jnp.exp(-N)
        assert float(ret_flows["impurity"]) == pytest.approx(
            float(expected_impurity), rel=1e-4
        ), (
            f"Impurity removal should follow exp(-N*(1-R)). "
            f"Got {float(ret_flows['impurity'])}, expected {float(expected_impurity)}"
        )

    def test_retained_species_stays(self):
        """R=1 species should be fully retained during diafiltration."""
        feed = make_stream(
            {"protein": jnp.array(10.0), "salt": jnp.array(2.0)},
            T=jnp.array(300.0),
            P=jnp.array(101325.0),
        )
        buffer = make_stream(
            {"new_buffer": jnp.array(1.0)},
            T=jnp.array(300.0),
            P=jnp.array(101325.0),
        )
        params = DiafiltrationParams(
            membrane_area=1.0,
            rejection={"protein": 1.0, "salt": 0.0, "new_buffer": 0.0},
        )
        df = Diafiltration(params)
        (retentate, permeate), info = df(feed, buffer, n_diavolumes=5.0)

        ret_flows = get_flows(retentate)
        assert float(ret_flows["protein"]) == pytest.approx(10.0, rel=1e-6)

    def test_buffer_wash_in(self):
        """Buffer species should wash in correctly: C_buf * V * (1 - exp(-N*(1-R)))."""
        feed = make_stream(
            {"protein": jnp.array(10.0)},
            T=jnp.array(300.0),
            P=jnp.array(101325.0),
        )
        buffer = make_stream(
            {"new_buffer": jnp.array(1.0)},
            T=jnp.array(300.0),
            P=jnp.array(101325.0),
        )
        N = 5.0
        R_buf = 0.0  # Fully permeable buffer salt
        params = DiafiltrationParams(
            membrane_area=1.0,
            rejection={"protein": 1.0, "new_buffer": R_buf},
        )
        df = Diafiltration(params)
        (retentate, permeate), info = df(feed, buffer, n_diavolumes=N)

        ret_flows = get_flows(retentate)

        # Buffer wash-in: C_buffer * V_initial * (1 - exp(-N*(1-R)))
        # buffer_conc = 1.0 (only species in buffer)
        # V_initial = sum of feed flows = 10.0
        V_initial = 10.0
        expected_buffer = 1.0 * V_initial * (1.0 - jnp.exp(-N * (1.0 - R_buf)))
        assert float(ret_flows.get("new_buffer", 0.0)) == pytest.approx(
            float(expected_buffer), rel=1e-3
        ), (
            f"Buffer wash-in incorrect. "
            f"Got {float(ret_flows.get('new_buffer', 0.0))}, expected {float(expected_buffer)}"
        )

    def test_diafiltration_mass_balance(self):
        """Retentate + permeate should equal feed + buffer added."""
        feed = make_stream(
            {"protein": jnp.array(10.0), "salt": jnp.array(2.0)},
            T=jnp.array(300.0),
            P=jnp.array(101325.0),
        )
        buffer = make_stream(
            {"new_buffer": jnp.array(1.0)},
            T=jnp.array(300.0),
            P=jnp.array(101325.0),
        )
        N = 3.0
        params = DiafiltrationParams(
            membrane_area=1.0,
            rejection={"protein": 1.0, "salt": 0.0, "new_buffer": 0.0},
        )
        df = Diafiltration(params)
        (retentate, permeate), info = df(feed, buffer, n_diavolumes=N)

        ret_flows = get_flows(retentate)
        perm_flows = get_flows(permeate)
        feed_flows = get_flows(feed)

        # For species in feed
        for species in feed_flows:
            total_out = float(ret_flows.get(species, 0.0)) + float(
                perm_flows.get(species, 0.0)
            )
            assert total_out == pytest.approx(
                float(feed_flows[species]), rel=1e-4
            ), f"DF mass balance violated for feed species {species}"


# =========================================================================
# Issue #98: Bioreactor yield coefficients
# =========================================================================

class TestBioreactorYieldCoefficients:
    """Tests for consistent yield coefficient enforcement (issue #98)."""

    def test_substrate_consumption_matches_growth(self):
        """The bioreactor ODE uses Y_xs consistently in the substrate balance.

        At exact steady state with no death or maintenance:
            mu = D  (from biomass balance with X_in=0, k_d=0)
            S_out = K_s * D / (mu_max - D)  (from Monod)
            X = Y_xs * (S_in - S_out)  (from substrate balance)

        We verify this stoichiometric relationship holds (within solver tolerance).
        """
        Y_xs = 0.5  # 0.5 g cells / g substrate
        mu_max_val = 0.4
        K_s_val = 0.5
        feed = make_stream(
            {
                "cells": jnp.array(0.0),
                "substrate": jnp.array(100.0),
                "product": jnp.array(0.0),
            },
            T=jnp.array(310.0),
            P=jnp.array(101325.0),
        )
        params = BioreactorParams(
            V=10.0,
            Y_xs=Y_xs,
            kinetic_fn=monod_kinetics,
            kinetic_params={"mu_max": jnp.array(mu_max_val), "K_s": jnp.array(K_s_val)},
            k_d=0.0,
            m_s=0.0,
            alpha=0.0,
            beta=0.0,
        )
        bioreactor = ContinuousBioreactor(params)
        F = 1.0  # L/h (D=0.1, well below washout)
        outlet, info = bioreactor(feed, F=F)

        X = float(info["X"])
        S_in = 100.0 / F  # inlet concentration
        S_out = float(info["S"])

        # Analytical: X = Y_xs * (S_in - S_out)
        X_expected = Y_xs * (S_in - S_out)

        assert X == pytest.approx(X_expected, rel=0.15), (
            f"Biomass X ({X}) should equal Y_xs*(S_in - S_out) ({X_expected}). "
            "Yield coefficient Y_xs not properly enforced in substrate balance."
        )

    def test_yield_params_are_jax_arrays(self):
        """Yield coefficients should be JAX arrays for gradient compatibility."""
        params = BioreactorParams(
            V=10.0,
            Y_xs=0.5,
            kinetic_fn=monod_kinetics,
            kinetic_params={"mu_max": jnp.array(0.4), "K_s": jnp.array(0.5)},
        )
        bioreactor = ContinuousBioreactor(params)

        feed = make_stream(
            {
                "cells": jnp.array(0.0),
                "substrate": jnp.array(50.0),
                "product": jnp.array(0.0),
            },
            T=jnp.array(310.0),
            P=jnp.array(101325.0),
        )

        # This should run without type errors from mixed float/JAX operations
        outlet, info = bioreactor(feed, F=5.0)
        assert info["X"] > 0, "Bioreactor should produce biomass"
