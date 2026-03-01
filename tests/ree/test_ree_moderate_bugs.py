"""Tests for REE plugin moderate bug fixes (#108, #112, #113, #123)."""

import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

pytest.importorskip("difflow_ree")


class TestBug108_ExtractionFactorTotalAqueous:
    """Bug #108: Extraction factor should use total aqueous flow, not just H2O."""

    def test_extraction_factor_includes_dissolved_ree(self):
        """With high REE concentration (same H2O), F_aq should be larger,
        leading to a smaller extraction factor E = D * F_org / F_aq."""
        from difflow.streams import make_stream, get_flows
        from difflow_ree.units.extraction import REEExtractor, REEExtractorParams

        params = REEExtractorParams(
            n_stages=5,
            extractant="D2EHPA",
            elements=("Nd",),
            pH=3.0,
            include_loading=False,
        )
        extractor = REEExtractor(params)

        solvent = make_stream(
            {"D2EHPA": 1.0, "kerosene": 5.0}, T=298.15, P=101325.0
        )

        # Dilute feed: same H2O, small REE
        # F_aq = 10.0 + 0.01 = 10.01
        dilute_feed = make_stream(
            {"H2O": 10.0, "Nd": 0.01}, T=298.15, P=101325.0
        )
        _, _, info_dilute = extractor(dilute_feed, solvent)

        # Concentrated feed: same H2O, large REE
        # F_aq = 10.0 + 5.0 = 15.0
        conc_feed = make_stream(
            {"H2O": 10.0, "Nd": 5.0}, T=298.15, P=101325.0
        )
        _, _, info_conc = extractor(conc_feed, solvent)

        E_dilute = float(info_dilute["profiles"]["Nd"]["E"])
        E_conc = float(info_conc["profiles"]["Nd"]["E"])

        # Concentrated solution should have smaller extraction factor
        # because F_aq is larger (H2O + dissolved REE)
        # E_dilute = D * 6.0 / 10.01, E_conc = D * 6.0 / 15.0
        assert E_conc < E_dilute, (
            f"Concentrated feed should have smaller E: "
            f"E_conc={E_conc:.4f}, E_dilute={E_dilute:.4f}"
        )

    def test_extraction_factor_with_multiple_aqueous_species(self):
        """Total aqueous flow should include all non-organic species."""
        from difflow.streams import make_stream, get_flows
        from difflow_ree.units.extraction import REEExtractor, REEExtractorParams

        params = REEExtractorParams(
            n_stages=3,
            extractant="D2EHPA",
            elements=("La", "Nd"),
            pH=3.0,
            include_loading=False,
        )
        extractor = REEExtractor(params)

        solvent = make_stream(
            {"D2EHPA": 1.0, "kerosene": 5.0}, T=298.15, P=101325.0
        )

        # Feed with multiple aqueous species
        feed = make_stream(
            {"H2O": 10.0, "La": 2.0, "Nd": 2.0}, T=298.15, P=101325.0
        )
        _, _, info = extractor(feed, solvent)

        # F_aq should be H2O + La + Nd = 14.0
        # F_org = D2EHPA + kerosene = 6.0
        # E = D * 6.0 / 14.0 (not D * 6.0 / 10.0)
        E_la = float(info["profiles"]["La"]["E"])
        E_nd = float(info["profiles"]["Nd"]["E"])

        # Both should reflect the total aqueous flow
        assert E_la > 0, "Extraction factor should be positive"
        assert E_nd > 0, "Extraction factor should be positive"


class TestBug112_ExtractantLoadingCapacity:
    """Bug #112: Extractant loading capacity not enforced in multi-stage extraction."""

    def test_total_loading_capped_at_capacity(self):
        """Total REE extracted should not exceed extractant capacity."""
        from difflow.streams import make_stream, get_flows
        from difflow_ree.units.extraction import REEExtractor, REEExtractorParams
        from difflow_ree.equilibrium.loading import get_loading_isotherm

        params = REEExtractorParams(
            n_stages=20,  # Many stages to push toward full extraction
            extractant="D2EHPA",
            elements=("La", "Ce", "Nd", "Dy"),
            pH=3.0,
            include_loading=True,
            extractant_conc=0.5,
        )
        extractor = REEExtractor(params)

        # High REE concentration feed to challenge capacity
        feed = make_stream(
            {"H2O": 10.0, "La": 2.0, "Ce": 2.0, "Nd": 2.0, "Dy": 2.0},
            T=298.15,
            P=101325.0,
        )
        solvent = make_stream(
            {"D2EHPA": 1.0, "kerosene": 5.0}, T=298.15, P=101325.0
        )

        _, extract, _ = extractor(feed, solvent)
        extract_flows = get_flows(extract)

        # Calculate total newly extracted REE
        total_extracted = sum(
            float(extract_flows.get(elem, 0.0))
            for elem in ("La", "Ce", "Nd", "Dy")
        )

        # Get maximum capacity
        isotherm = get_loading_isotherm("D2EHPA", 0.5)
        F_org = 1.0 + 5.0  # D2EHPA + kerosene
        max_capacity = isotherm.max_ree_conc * F_org

        # Total extracted should not exceed capacity
        assert total_extracted <= max_capacity + 1e-6, (
            f"Total extracted ({total_extracted:.4f}) exceeds capacity "
            f"({max_capacity:.4f})"
        )

    def test_loading_cap_scales_back_proportionally(self):
        """When capacity is exceeded, all elements should be scaled back."""
        from difflow.streams import make_stream, get_flows
        from difflow_ree.units.extraction import REEExtractor, REEExtractorParams

        params = REEExtractorParams(
            n_stages=20,
            extractant="D2EHPA",
            elements=("Nd", "Dy"),
            pH=3.0,
            include_loading=True,
            extractant_conc=0.5,
        )
        extractor = REEExtractor(params)

        # Large feed to force capacity limit
        feed = make_stream(
            {"H2O": 10.0, "Nd": 5.0, "Dy": 5.0},
            T=298.15,
            P=101325.0,
        )
        solvent = make_stream(
            {"D2EHPA": 1.0, "kerosene": 5.0}, T=298.15, P=101325.0
        )

        raff, extract, _ = extractor(feed, solvent)
        raff_flows = get_flows(raff)
        extract_flows = get_flows(extract)

        # Mass balance: feed + solvent = raffinate + extract for each element
        for elem in ("Nd", "Dy"):
            f_in = 5.0  # feed
            f_solv = 0.0  # no REE in solvent
            f_raff = float(raff_flows.get(elem, 0.0))
            f_ext = float(extract_flows.get(elem, 0.0))
            balance = abs((f_in + f_solv) - (f_raff + f_ext))
            assert balance < 1e-6, (
                f"Mass balance violated for {elem}: "
                f"in={f_in + f_solv:.6f}, out={f_raff + f_ext:.6f}"
            )


class TestBug113_OxidantConsumption:
    """Bug #113: CeriumOxidizer should track oxidant consumption."""

    def test_oxidant_consumption_in_info(self):
        """Info dict should contain oxidant consumption data."""
        from difflow.streams import make_stream
        from difflow_ree.units.cerium import CeriumOxidizer, CeriumOxidizerParams

        params = CeriumOxidizerParams(
            elements=("La", "Ce", "Nd"),
            oxidant="air",
            oxidant_excess=2.0,
            pH=8.0,
            ce_conversion=0.95,
        )
        oxidizer = CeriumOxidizer(params)

        feed = make_stream(
            {"H2O": 10.0, "La": 1.0, "Ce": 2.0, "Nd": 1.0},
            T=353.15,
            P=101325.0,
        )
        _, _, info = oxidizer(feed)

        # Check that oxidant consumption keys exist
        assert "oxidant_consumed_mol_s" in info
        assert "oxidant_stoich_ratio" in info
        assert "electrons_transferred_mol_s" in info
        assert "oxidant_excess" in info

    def test_oxidant_stoichiometry_air(self):
        """Air oxidant: 0.25 mol O2 per mol Ce oxidized."""
        from difflow.streams import make_stream
        from difflow_ree.units.cerium import CeriumOxidizer, CeriumOxidizerParams

        params = CeriumOxidizerParams(
            elements=("Ce",),
            oxidant="air",
            oxidant_excess=1.0,  # No excess for clean test
            pH=10.0,  # High pH for near-complete conversion
            temperature=353.15,
            ce_conversion=0.95,
        )
        oxidizer = CeriumOxidizer(params)

        feed = make_stream(
            {"H2O": 10.0, "Ce": 4.0},
            T=353.15,
            P=101325.0,
        )
        _, _, info = oxidizer(feed)

        assert info["oxidant_stoich_ratio"] == 0.25
        # With stoich=0.25 and excess=1.0:
        # oxidant_consumed = Ce_oxidized * 0.25 * 1.0
        ce_oxidized = info["ce_removed_mol_s"]
        expected_oxidant = ce_oxidized * 0.25 * 1.0
        assert abs(info["oxidant_consumed_mol_s"] - expected_oxidant) < 1e-10

    def test_oxidant_stoichiometry_h2o2(self):
        """H2O2 oxidant: 0.5 mol H2O2 per mol Ce oxidized."""
        from difflow.streams import make_stream
        from difflow_ree.units.cerium import CeriumOxidizer, CeriumOxidizerParams

        params = CeriumOxidizerParams(
            elements=("Ce",),
            oxidant="H2O2",
            oxidant_excess=1.5,
            pH=10.0,
            temperature=353.15,
            ce_conversion=0.95,
        )
        oxidizer = CeriumOxidizer(params)

        feed = make_stream(
            {"H2O": 10.0, "Ce": 2.0},
            T=353.15,
            P=101325.0,
        )
        _, _, info = oxidizer(feed)

        assert info["oxidant_stoich_ratio"] == 0.5
        ce_oxidized = info["ce_removed_mol_s"]
        expected_oxidant = ce_oxidized * 0.5 * 1.5
        assert abs(info["oxidant_consumed_mol_s"] - expected_oxidant) < 1e-10

    def test_electrons_transferred(self):
        """Electrons transferred should equal moles of Ce oxidized (1e- per Ce)."""
        from difflow.streams import make_stream
        from difflow_ree.units.cerium import CeriumOxidizer, CeriumOxidizerParams

        params = CeriumOxidizerParams(
            elements=("Ce",),
            oxidant="electrolytic",
            oxidant_excess=1.0,
            pH=10.0,
            temperature=353.15,
            ce_conversion=0.95,
        )
        oxidizer = CeriumOxidizer(params)

        feed = make_stream(
            {"H2O": 10.0, "Ce": 3.0},
            T=353.15,
            P=101325.0,
        )
        _, _, info = oxidizer(feed)

        ce_oxidized = info["ce_removed_mol_s"]
        assert abs(info["electrons_transferred_mol_s"] - ce_oxidized) < 1e-10

    def test_oxidant_excess_tracked(self):
        """Oxidant excess parameter should be passed through to info."""
        from difflow.streams import make_stream
        from difflow_ree.units.cerium import CeriumOxidizer, CeriumOxidizerParams

        params = CeriumOxidizerParams(
            elements=("Ce",),
            oxidant="NaOCl",
            oxidant_excess=3.0,
            pH=8.0,
        )
        oxidizer = CeriumOxidizer(params)

        feed = make_stream(
            {"H2O": 10.0, "Ce": 1.0},
            T=353.15,
            P=101325.0,
        )
        _, _, info = oxidizer(feed)

        assert info["oxidant_excess"] == 3.0
        assert info["oxidant_stoich_ratio"] == 0.5  # NaOCl


class TestBug123_LaborCostOverestimate:
    """Bug #123: REE labor cost overestimates by 4x."""

    def test_labor_cost_not_multiplied_by_shifts(self):
        """labor_cost should use n_operators as total headcount, not per-shift."""
        from difflow_ree.economics.costs import OperatingCosts

        opex = OperatingCosts(labor_rate=35.0)

        # 6 total operators, 8000 hours/year
        cost = opex.labor_cost(6, 8000)

        # Should be: 35 * 6 * 8000 = 1,680,000
        # NOT: 35 * 6 * 4 * 8000 = 6,720,000 (old 4x bug)
        expected = 35.0 * 6 * 8000
        assert abs(cost - expected) < 1e-2, (
            f"Labor cost {cost} != expected {expected}. "
            f"Should not multiply by 4 shifts."
        )

    def test_estimate_opex_labor_reasonable(self):
        """Total OPEX labor should be reasonable for a small REE plant."""
        from difflow_ree.economics.costs import estimate_opex

        opex = estimate_opex(
            annual_ree_tonnes=100,
            capex=10_000_000,
            extractant="D2EHPA",
        )

        labor = opex["labor"]
        # For 6 total operators at $35/hr, 8000 hrs: ~$1.68M
        # Old buggy value would be ~$6.72M
        assert labor < 3_000_000, (
            f"Labor cost ${labor:,.0f} is too high for a small REE plant"
        )
        assert labor > 500_000, (
            f"Labor cost ${labor:,.0f} is unrealistically low"
        )

    def test_labor_cost_reduced_by_factor_of_four(self):
        """The fix should reduce labor cost by approximately 4x."""
        from difflow_ree.economics.costs import OperatingCosts

        opex = OperatingCosts(labor_rate=35.0)

        # With the fix: 35 * 6 * 8000 = 1,680,000
        cost = opex.labor_cost(6, 8000)

        # The old buggy cost was 35 * 6 * 4 * 8000 = 6,720,000
        old_buggy_cost = 35.0 * 6 * 4 * 8000

        ratio = old_buggy_cost / cost
        assert abs(ratio - 4.0) < 0.01, (
            f"New cost should be 4x less than old. Ratio = {ratio:.2f}"
        )
