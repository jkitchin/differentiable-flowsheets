"""Mass balance verification tests for key unit operations.

Verifies that sum(inlet_flows) == sum(outlet_flows) within tolerance
for each unit operation, confirming conservation of mass/moles.
"""

import jax
import jax.numpy as jnp
import pytest

from difflow.streams import make_stream, get_flows
from difflow import (
    CSTR,
    CSTRParams,
    Flash,
    FlashParams,
    IdealThermo,
    SpeciesData,
    CounterCurrentHX,
    HeatExchangerParams,
)

jax.config.update("jax_enable_x64", True)

TOL = 1e-5


def _sum_flows(stream):
    """Sum all molar flows in a stream."""
    flows = get_flows(stream)
    return sum(float(v) for v in flows.values())


def _flows_dict(stream):
    """Get flows as {species: float} dict."""
    return {k: float(v) for k, v in get_flows(stream).items()}


# =========================================================================
# 1. CSTR - mole-conserving reaction A -> B
# =========================================================================

class TestCSTRMassBalance:
    """CSTR with A -> B (1:1 stoichiometry) conserves total moles."""

    def test_total_moles_conserved(self):
        species = ["A", "B"]
        stoich = jnp.array([[-1.0], [1.0]])  # A -> B

        def rate_fn(C, T, params):
            k = params["k"]
            return jnp.array([k * C["A"]])

        params = CSTRParams(
            V=jnp.array(2.0),
            rate_fn=rate_fn,
            stoich=stoich,
            rate_params={"k": jnp.array(0.5)},
            species_order=species,
        )
        cstr = CSTR(params)

        inlet = make_stream({"A": 5.0, "B": 1.0}, T=350.0, P=101325.0)
        outlet, info = cstr(inlet)

        inlet_total = _sum_flows(inlet)
        outlet_total = _sum_flows(outlet)
        assert outlet_total == pytest.approx(inlet_total, rel=TOL)

    def test_multiple_species_balance(self):
        """A + B -> C + D (2 reactants, 2 products) conserves total moles."""
        species = ["A", "B", "C", "D"]
        # Reaction: A + B -> C + D
        stoich = jnp.array([[-1.0], [-1.0], [1.0], [1.0]])

        def rate_fn(C, T, params):
            k = params["k"]
            return jnp.array([k * C["A"] * C["B"]])

        params = CSTRParams(
            V=jnp.array(1.0),
            rate_fn=rate_fn,
            stoich=stoich,
            rate_params={"k": jnp.array(0.1)},
            species_order=species,
        )
        cstr = CSTR(params)

        inlet = make_stream({"A": 3.0, "B": 2.0, "C": 0.0, "D": 0.0}, T=350.0, P=101325.0)
        outlet, info = cstr(inlet)

        # Total moles conserved (stoich sums to zero per reaction)
        assert _sum_flows(outlet) == pytest.approx(_sum_flows(inlet), rel=TOL)


# =========================================================================
# 2. Flash - feed = vapor + liquid
# =========================================================================

class TestFlashMassBalance:
    """Flash separator conserves moles: feed = vapor + liquid per species."""

    @pytest.fixture
    def thermo(self):
        species_data = {
            "Light": SpeciesData(
                name="Light", MW=72.0,
                Cp_coeffs=(120.0, 0.0, 0.0, 0.0),
                Hvap_coeffs=(26000.0, 0.38, 470.0),
                antoine_coeffs=(10.422, 1687.537, -38.44),
                Hf=0.0,
            ),
            "Heavy": SpeciesData(
                name="Heavy", MW=114.0,
                Cp_coeffs=(190.0, 0.0, 0.0, 0.0),
                Hvap_coeffs=(35000.0, 0.38, 570.0),
                antoine_coeffs=(10.186, 2004.68, -60.53),
                Hf=0.0,
            ),
        }
        return IdealThermo(species_data)

    def test_binary_flash_mass_balance(self, thermo):
        feed = make_stream({"Light": 50.0, "Heavy": 50.0}, T=350.0, P=30000.0)
        flash = Flash(FlashParams(species_order=["Light", "Heavy"]), thermo)

        liquid, vapor, info = flash(feed)

        feed_flows = _flows_dict(feed)
        liq_flows = _flows_dict(liquid)
        vap_flows = _flows_dict(vapor)

        for sp in ["Light", "Heavy"]:
            out = liq_flows.get(sp, 0.0) + vap_flows.get(sp, 0.0)
            assert out == pytest.approx(feed_flows[sp], rel=TOL), (
                f"Flash mass balance violated for {sp}"
            )

    def test_flash_total_moles(self, thermo):
        feed = make_stream({"Light": 30.0, "Heavy": 70.0}, T=360.0, P=25000.0)
        flash = Flash(FlashParams(species_order=["Light", "Heavy"]), thermo)

        liquid, vapor, info = flash(feed)
        total_out = _sum_flows(liquid) + _sum_flows(vapor)
        assert total_out == pytest.approx(_sum_flows(feed), rel=TOL)


# =========================================================================
# 3. HeatExchanger - species flows unchanged, only T changes
# =========================================================================

class TestHeatExchangerMassBalance:
    """CounterCurrentHX preserves per-species flows on both sides."""

    def test_species_conserved_hot_and_cold(self):
        hot_in = make_stream({"N2": 10.0, "O2": 3.0}, T=500.0, P=101325.0)
        cold_in = make_stream({"H2O": 20.0, "CO2": 1.0}, T=300.0, P=101325.0)

        params = HeatExchangerParams(UA=500.0, Cp_hot=30.0, Cp_cold=75.0)
        hx = CounterCurrentHX(params)
        hot_out, cold_out, info = hx(hot_in, cold_in)

        hot_in_flows = _flows_dict(hot_in)
        hot_out_flows = _flows_dict(hot_out)
        for sp in hot_in_flows:
            assert hot_out_flows[sp] == pytest.approx(hot_in_flows[sp], rel=TOL)

        cold_in_flows = _flows_dict(cold_in)
        cold_out_flows = _flows_dict(cold_out)
        for sp in cold_in_flows:
            assert cold_out_flows[sp] == pytest.approx(cold_in_flows[sp], rel=TOL)

    def test_total_moles_both_sides(self):
        hot_in = make_stream({"A": 5.0, "B": 3.0}, T=450.0, P=200000.0)
        cold_in = make_stream({"C": 8.0, "D": 2.0}, T=310.0, P=200000.0)

        params = HeatExchangerParams(UA=1000.0)
        hx = CounterCurrentHX(params)
        hot_out, cold_out, info = hx(hot_in, cold_in)

        total_in = _sum_flows(hot_in) + _sum_flows(cold_in)
        total_out = _sum_flows(hot_out) + _sum_flows(cold_out)
        assert total_out == pytest.approx(total_in, rel=TOL)


# =========================================================================
# 4. REEExtractor - feed + solvent = raffinate + extract (per species)
# =========================================================================

class TestREEExtractorMassBalance:
    """REEExtractor conserves mass for every species."""

    def test_per_species_balance(self):
        from difflow_ree import REEExtractor, REEExtractorParams

        elements = ("Nd", "Dy")
        feed = make_stream(
            {"H2O": 10.0, "Nd": 0.5, "Dy": 0.3, "Fe": 0.1},
            T=298.15, P=101325.0,
        )
        solvent = make_stream(
            {"D2EHPA": 1.0, "kerosene": 5.0, "Nd": 0.0, "Dy": 0.0},
            T=298.15, P=101325.0,
        )

        params = REEExtractorParams(
            n_stages=5, extractant="D2EHPA", elements=elements, pH=3.0,
        )
        extractor = REEExtractor(params)
        raffinate, extract, info = extractor(feed, solvent)

        feed_flows = _flows_dict(feed)
        solvent_flows = _flows_dict(solvent)
        raff_flows = _flows_dict(raffinate)
        ext_flows = _flows_dict(extract)

        all_species = set(feed_flows) | set(solvent_flows)
        for sp in all_species:
            total_in = feed_flows.get(sp, 0.0) + solvent_flows.get(sp, 0.0)
            total_out = raff_flows.get(sp, 0.0) + ext_flows.get(sp, 0.0)
            assert total_out == pytest.approx(total_in, rel=TOL), (
                f"REEExtractor mass balance violated for {sp}: "
                f"in={total_in:.8f}, out={total_out:.8f}"
            )


# =========================================================================
# 5. REEMixerSettler - feed + solvent = aq_out + org_out (per species)
# =========================================================================

class TestREEMixerSettlerMassBalance:
    """REEMixerSettler single stage conserves mass per species."""

    def test_per_species_balance(self):
        from difflow_ree import REEMixerSettler, MixerSettlerParams

        elements = ("La", "Nd")
        aq_in = make_stream(
            {"H2O": 10.0, "La": 0.3, "Nd": 0.4, "Cl": 1.0},
            T=298.15, P=101325.0,
        )
        org_in = make_stream(
            {"D2EHPA": 1.0, "kerosene": 5.0, "La": 0.01, "Nd": 0.02},
            T=298.15, P=101325.0,
        )

        params = MixerSettlerParams(
            extractant="D2EHPA", elements=elements, pH=2.5,
            stage_efficiency=0.90,
        )
        stage = REEMixerSettler(params)
        aq_out, org_out, info = stage(aq_in, org_in)

        aq_in_flows = _flows_dict(aq_in)
        org_in_flows = _flows_dict(org_in)
        aq_out_flows = _flows_dict(aq_out)
        org_out_flows = _flows_dict(org_out)

        all_species = set(aq_in_flows) | set(org_in_flows)
        for sp in all_species:
            total_in = aq_in_flows.get(sp, 0.0) + org_in_flows.get(sp, 0.0)
            total_out = aq_out_flows.get(sp, 0.0) + org_out_flows.get(sp, 0.0)
            assert total_out == pytest.approx(total_in, rel=TOL), (
                f"REEMixerSettler mass balance violated for {sp}: "
                f"in={total_in:.8f}, out={total_out:.8f}"
            )


# =========================================================================
# 6. Centrifuge (bio) - feed = concentrate + clarified
# =========================================================================

class TestCentrifugeMassBalance:
    """Centrifuge conserves mass: feed = concentrate + clarified."""

    def test_per_species_balance(self):
        from difflow_bio import Centrifuge, CentrifugeParams

        feed = make_stream(
            {"cells": 2.0, "protein": 0.5, "H2O": 50.0, "salt": 1.0},
            T=298.15, P=101325.0,
        )
        params = CentrifugeParams(sigma=5000.0, efficiency=0.8, cell_species="cells")
        centrifuge = Centrifuge(params)

        concentrate, clarified, info = centrifuge(
            feed, Q=1e-4, d_particle=5e-6,
            rho_particle=1050.0, rho_fluid=1000.0, viscosity=0.001,
            concentrate_fraction=0.1,
        )

        feed_flows = _flows_dict(feed)
        conc_flows = _flows_dict(concentrate)
        clar_flows = _flows_dict(clarified)

        for sp in feed_flows:
            total_out = conc_flows.get(sp, 0.0) + clar_flows.get(sp, 0.0)
            assert total_out == pytest.approx(feed_flows[sp], rel=TOL), (
                f"Centrifuge mass balance violated for {sp}"
            )

    def test_total_moles(self):
        from difflow_bio import Centrifuge, CentrifugeParams

        feed = make_stream(
            {"cells": 5.0, "product": 1.0, "media": 100.0},
            T=310.0, P=101325.0,
        )
        params = CentrifugeParams(sigma=3000.0, efficiency=0.7, cell_species="cells")
        centrifuge = Centrifuge(params)

        concentrate, clarified, info = centrifuge(
            feed, Q=5e-5, concentrate_fraction=0.15,
        )

        total_in = _sum_flows(feed)
        total_out = _sum_flows(concentrate) + _sum_flows(clarified)
        assert total_out == pytest.approx(total_in, rel=TOL)


# =========================================================================
# 7. MultistageMembrane (cc) - feed = retentate + permeate
# =========================================================================

class TestMultistageMembraneMassBalance:
    """MultistageMembrane conserves mass: feed = retentate + permeate."""

    def test_per_species_balance(self):
        from difflow_cc import MultistageMembrane, MembraneParams

        feed = make_stream(
            {"CO2": 10.0, "N2": 80.0, "O2": 5.0, "H2O": 5.0},
            T=298.15, P=1000000.0,
        )

        params = MembraneParams(
            membrane_type="Matrimid",
            area=500.0,
            pressure_ratio=10.0,
            T_operation=298.15,
            feed_pressure=1000000.0,
        )
        cascade = MultistageMembrane(params, n_stages=2, configuration="series")
        retentate, permeate, info = cascade(feed)

        feed_flows = _flows_dict(feed)
        ret_flows = _flows_dict(retentate)
        perm_flows = _flows_dict(permeate)

        for sp in feed_flows:
            total_out = ret_flows.get(sp, 0.0) + perm_flows.get(sp, 0.0)
            assert total_out == pytest.approx(feed_flows[sp], rel=TOL), (
                f"MultistageMembrane mass balance violated for {sp}: "
                f"feed={feed_flows[sp]:.8f}, out={total_out:.8f}"
            )

    def test_total_moles(self):
        from difflow_cc import MultistageMembrane, MembraneParams

        feed = make_stream(
            {"CO2": 15.0, "N2": 70.0, "H2O": 3.0},
            T=313.15, P=800000.0,
        )

        params = MembraneParams(
            membrane_type="Matrimid",
            area=300.0,
            pressure_ratio=8.0,
            T_operation=313.15,
            feed_pressure=800000.0,
        )
        cascade = MultistageMembrane(params, n_stages=3, configuration="series")
        retentate, permeate, info = cascade(feed)

        total_in = _sum_flows(feed)
        total_out = _sum_flows(retentate) + _sum_flows(permeate)
        assert total_out == pytest.approx(total_in, rel=TOL)
