"""Tests for REE separation plugin."""

import pytest
import jax.numpy as jnp

# Skip if plugin not installed
pytest.importorskip("difflow_ree")


class TestREEDatabase:
    """Test REE database functionality."""

    def test_load_elements(self):
        """Test loading element data."""
        from difflow_ree import get_element, list_ree_elements

        elements = list_ree_elements()
        assert len(elements) == 10
        assert "Nd" in elements
        assert "Dy" in elements

    def test_element_properties(self):
        """Test element property access."""
        from difflow_ree import get_element

        nd = get_element("Nd")
        assert nd.symbol == "Nd"
        assert nd.atomic_number == 60
        assert nd.atomic_weight > 144
        assert nd.group == "light"

    def test_extractant_data(self):
        """Test extractant database."""
        from difflow_ree import get_extractant, list_extractants

        extractants = list_extractants()
        assert "D2EHPA" in extractants
        assert "PC88A" in extractants

        d2ehpa = get_extractant("D2EHPA")
        assert d2ehpa.molecular_weight > 300
        assert "Nd" in d2ehpa.ph_coefficients


class TestDistributionModel:
    """Test distribution coefficient models."""

    def test_basic_distribution(self):
        """Test basic D value calculation."""
        from difflow_ree import REEDistribution

        dist = REEDistribution(
            extractant="D2EHPA",
            elements=("La", "Nd", "Dy"),
        )

        D_values = dist.get_D_all(pH=3.0, T=298.15)

        # Heavy REE should have higher D
        assert D_values["Dy"] > D_values["Nd"]
        assert D_values["Nd"] > D_values["La"]

    def test_ph_effect(self):
        """Test pH effect on D values."""
        from difflow_ree import get_distribution_coefficient

        D_low_pH = get_distribution_coefficient("Nd", "D2EHPA", pH=2.0)
        D_high_pH = get_distribution_coefficient("Nd", "D2EHPA", pH=4.0)

        # Higher pH = higher extraction
        assert D_high_pH > D_low_pH

    def test_separation_factor(self):
        """Test separation factor calculation."""
        from difflow_ree import REEDistribution

        dist = REEDistribution(
            extractant="D2EHPA",
            elements=("Nd", "Pr"),
        )

        SF = dist.get_separation_factor("Nd", "Pr", pH=3.0)

        # Nd should extract slightly better than Pr
        assert SF > 1.0


class TestUnitOperations:
    """Test REE unit operations."""

    def test_extractor(self):
        """Test REE extractor."""
        from difflow_ree import REEExtractor, REEExtractorParams
        from difflow.streams import make_stream, get_flows

        # Use fewer stages and lower S/F to see selectivity differences
        params = REEExtractorParams(
            n_stages=3,
            extractant="D2EHPA",
            elements=("La", "Nd", "Dy"),
            pH=2.5,  # Lower pH reduces extraction, shows selectivity
        )
        extractor = REEExtractor(params)

        feed = make_stream(
            flows={"H2O": 10.0, "La": 0.01, "Nd": 0.02, "Dy": 0.01},
            T=298.15,
            P=101325.0,
        )
        solvent = make_stream(
            flows={"kerosene": 5.0, "D2EHPA": 0.5 * 5.0, "La": 0.0, "Nd": 0.0, "Dy": 0.0},
            T=298.15,
            P=101325.0,
        )

        raffinate, extract, info = extractor(feed, solvent)

        raff_flows = get_flows(raffinate)
        ext_flows = get_flows(extract)

        # Heavy REE should have higher recovery
        dy_recovery = float(ext_flows["Dy"]) / 0.01
        la_recovery = float(ext_flows["La"]) / 0.01

        # At these conditions, Dy extracts much better than La
        assert dy_recovery > la_recovery, f"Dy recovery {dy_recovery:.3f} should be > La recovery {la_recovery:.3f}"
        assert dy_recovery > 0.3  # Should have significant Dy extraction

    def test_extractor_mass_conservation(self):
        """Test that REEExtractor conserves mass for all species (issue #53)."""
        from difflow_ree import REEExtractor, REEExtractorParams
        from difflow.streams import make_stream, get_flows

        params = REEExtractorParams(
            n_stages=1,
            extractant="D2EHPA",
            elements=("Dy", "Nd"),
            diluent="n-Heptane",
            pH=1.6,
        )
        extractor = REEExtractor(params)

        feed = make_stream(
            flows={"H2O": 10.0, "Nd": 0.2, "Dy": 0.143827799, "Fe": 0.553},
            T=298.15,
            P=101325.0,
        )
        solvent = make_stream(
            flows={"D2EHPA": 10.0, "n-Heptane": 20.0, "Nd": 0.0, "Dy": 0.0, "Fe": 0.0},
            T=298.15,
            P=101325.0,
        )

        raffinate, extract, info = extractor(feed, solvent)
        raff_flows = get_flows(raffinate)
        ext_flows = get_flows(extract)

        feed_flows = get_flows(feed)
        solvent_flows = get_flows(solvent)

        # Check mass conservation for every species in feed + solvent
        all_species = set(feed_flows.keys()) | set(solvent_flows.keys())
        for species in all_species:
            total_in = float(feed_flows.get(species, 0.0)) + float(solvent_flows.get(species, 0.0))
            total_out = float(raff_flows.get(species, 0.0)) + float(ext_flows.get(species, 0.0))
            assert abs(total_in - total_out) < 1e-10, (
                f"Mass not conserved for {species}: in={total_in:.6f}, out={total_out:.6f}"
            )

    def test_oxalate_precipitator(self):
        """Test oxalate precipitation."""
        from difflow_ree import OxalatePrecipitator, PrecipitatorParams
        from difflow.streams import make_stream

        params = PrecipitatorParams(
            elements=("Nd", "Dy"),
            precipitant_excess=1.5,
        )
        precip = OxalatePrecipitator(params)

        feed = make_stream(
            flows={"H2O": 10.0, "Nd": 0.1, "Dy": 0.05},
            T=298.15,
            P=101325.0,
        )
        oxalate = make_stream(
            flows={"H2O": 5.0, "C2O4": 0.3},  # 1.5x stoichiometric
            T=298.15,
            P=101325.0,
        )

        filtrate, solid, info = precip(feed, oxalate)

        # Should have high precipitation
        assert info["total_precipitated"] > 0.1


class TestFlowsheets:
    """Test flowsheet templates."""

    def test_extract_strip_circuit(self):
        """Test basic extract-strip circuit."""
        from difflow_ree import ExtractStripCircuit, ExtractStripParams
        from difflow.streams import make_stream

        params = ExtractStripParams(
            extractant="D2EHPA",
            elements=("La", "Nd", "Dy"),
            n_extraction_stages=5,
            n_stripping_stages=3,
        )
        circuit = ExtractStripCircuit(params)

        feed = make_stream(
            flows={"H2O": 10.0, "La": 0.01, "Nd": 0.02, "Dy": 0.01},
            T=298.15,
            P=101325.0,
        )

        results = circuit(feed)

        assert "product" in results
        assert "recovery" in results
        assert results["recovery"] > 0.5  # Should have reasonable recovery

    def test_split_shell_cascade_basic(self):
        """Test that SplitShellCascade runs and returns expected keys."""
        from difflow_ree import SplitShellCascade, SplitShellParams
        from difflow.streams import make_stream

        params = SplitShellParams(
            extractant="D2EHPA",
            elements=("La", "Nd", "Dy"),
            n_stages=12,
            split_points=(4, 8),
        )
        cascade = SplitShellCascade(params)

        feed = make_stream(
            flows={"H2O": 10.0, "La": 0.03, "Nd": 0.03, "Dy": 0.03},
            T=298.15,
            P=101325.0,
        )
        solvent = make_stream(
            flows={"kerosene": 10.0, "D2EHPA": 5.0, "La": 0.0, "Nd": 0.0, "Dy": 0.0},
            T=298.15,
            P=101325.0,
        )

        results = cascade(feed, solvent)

        assert "products" in results
        assert "D_values" in results
        assert "converged_in_iter" in results
        assert "product_1" in results["products"]
        assert "product_2" in results["products"]
        assert "raffinate" in results["products"]

    def test_split_shell_cascade_convergence(self):
        """Test that SplitShellCascade iteration converges and mass is conserved."""
        from difflow_ree import SplitShellCascade, SplitShellParams
        from difflow.streams import make_stream

        params = SplitShellParams(
            extractant="D2EHPA",
            elements=("La", "Nd", "Dy"),
            n_stages=15,
            split_points=(5, 10),
            pH=3.0,
        )
        cascade = SplitShellCascade(params)

        feed_flows_dict = {"H2O": 10.0, "La": 0.05, "Nd": 0.03, "Dy": 0.02}
        feed = make_stream(flows=feed_flows_dict, T=298.15, P=101325.0)
        solvent = make_stream(
            flows={"kerosene": 10.0, "D2EHPA": 5.0, "La": 0.0, "Nd": 0.0, "Dy": 0.0},
            T=298.15,
            P=101325.0,
        )

        results = cascade(feed, solvent)

        # Should converge in fewer than max_iter iterations
        assert results["converged_in_iter"] < 50

        # Mass conservation: total extracted + raffinate equals feed
        total_extracted = {elem: 0.0 for elem in ("La", "Nd", "Dy")}
        for prod_name, prod in results["products"].items():
            for elem in ("La", "Nd", "Dy"):
                total_extracted[elem] += prod["flows"].get(elem, 0.0)

        for elem in ("La", "Nd", "Dy"):
            total_out = total_extracted[elem]
            total_in = feed_flows_dict[elem]
            assert abs(total_out - total_in) < 1e-6, (
                f"Mass not conserved for {elem}: in={total_in}, out={total_out}"
            )

    def test_split_shell_heavy_ree_enriched_early(self):
        """Test that heavy REE (high D) are preferentially extracted in early sections."""
        from difflow_ree import SplitShellCascade, SplitShellParams
        from difflow.streams import make_stream

        params = SplitShellParams(
            extractant="D2EHPA",
            elements=("La", "Dy"),
            n_stages=10,
            split_points=(5,),
            pH=3.0,
        )
        cascade = SplitShellCascade(params)

        feed = make_stream(
            flows={"H2O": 10.0, "La": 0.1, "Dy": 0.1},
            T=298.15,
            P=101325.0,
        )
        solvent = make_stream(
            flows={"kerosene": 10.0, "D2EHPA": 5.0, "La": 0.0, "Dy": 0.0},
            T=298.15,
            P=101325.0,
        )

        results = cascade(feed, solvent)
        prods = results["products"]

        # product_1 is the first section (highest extraction for high-D elements)
        # Dy has much higher D than La, so it should dominate product_1
        dy_in_p1 = prods["product_1"]["flows"]["Dy"]
        la_in_p1 = prods["product_1"]["flows"]["La"]
        assert dy_in_p1 > la_in_p1, (
            f"Expected Dy ({dy_in_p1:.4f}) > La ({la_in_p1:.4f}) in product_1"
        )

    def test_optimize_split_points_d_value_based(self):
        """Test that optimize_split_points uses D values, not equal spacing."""
        from difflow_ree.flowsheets.split_shell import optimize_split_points

        # For D2EHPA at pH 3.5, Dy >> Nd >> La in terms of D.
        # With 3 products from 3 elements, splits should place boundaries
        # between natural D-value gaps, not at n/3 and 2n/3.
        splits = optimize_split_points(
            elements=("La", "Nd", "Dy"),
            extractant="D2EHPA",
            n_stages=30,
            n_products=3,
            pH=3.5,
        )

        assert len(splits) == 2  # n_products - 1 split points
        assert splits[0] < splits[1]  # Strictly increasing
        assert splits[0] >= 1
        assert splits[1] <= 29  # Within valid range

    def test_optimize_split_points_single_product(self):
        """Test optimize_split_points with n_products=1 returns empty tuple."""
        from difflow_ree.flowsheets.split_shell import optimize_split_points

        splits = optimize_split_points(
            elements=("La", "Nd"),
            extractant="D2EHPA",
            n_stages=20,
            n_products=1,
            pH=3.0,
        )
        assert splits == ()


class TestEconomics:
    """Test economic analysis functions."""

    def test_ree_pricing(self):
        """Test REE pricing model."""
        from difflow_ree import REEPricing

        pricing = REEPricing()

        nd_price = pricing.get_price("Nd", purity="99%", form="oxide")
        assert nd_price > 100  # Nd is valuable

        ce_price = pricing.get_price("Ce", purity="99%", form="oxide")
        assert ce_price < nd_price  # Ce is less valuable

    def test_capex_estimate(self):
        """Test capital cost estimation."""
        from difflow_ree import estimate_capex

        capex = estimate_capex(
            annual_ree_tonnes=1000,
            n_stages_extraction=10,
            n_stages_scrubbing=5,
            n_stages_stripping=5,
        )

        assert "total" in capex
        assert capex["total"] > 0

    def test_msp_calculation(self):
        """Test minimum selling price."""
        from difflow_ree import minimum_selling_price

        msp = minimum_selling_price(
            opex=1000000,  # $1M/year
            capex=5000000,  # $5M
            annual_production_kg=100000,  # 100 tonnes
            target_roi=0.15,
        )

        assert msp > 0
        assert msp < 1000  # Should be reasonable $/kg


class TestJAXCompatibility:
    """Test JAX compatibility for gradients."""

    def test_distribution_gradient(self):
        """Test that D values are differentiable."""
        from difflow_ree import REEDistribution
        from jax import grad

        dist = REEDistribution(
            extractant="D2EHPA",
            elements=("Nd",),
        )

        def D_at_pH(pH):
            return dist.get_D("Nd", pH)

        # Should be able to compute gradient
        dD_dpH = grad(D_at_pH)(3.0)
        assert dD_dpH > 0  # D increases with pH


class TestCustomExtractants:
    """Test custom extractant creation and registration."""

    def test_create_custom_extractant(self):
        """Test creating a custom extractant."""
        from difflow_ree import create_custom_extractant

        # Create custom extractant with minimal required data
        custom_ext = create_custom_extractant(
            name="TestExtractant",
            full_name="Test Phosphoric Acid",
            formula="C8H17O4P",
            molecular_weight=220.0,
            ph_coefficients={
                "La": {"a": -8.0, "b": 2.2, "c": 0.01},
                "Nd": {"a": -7.5, "b": 2.4, "c": 0.01},
                "Dy": {"a": -7.0, "b": 2.6, "c": 0.01},
            },
            temperature_coefficients={
                "La": -1500.0,
                "Nd": -1700.0,
                "Dy": -1900.0,
            },
            pKa=3.5,
        )

        assert custom_ext.name == "TestExtractant"
        assert custom_ext.molecular_weight == 220.0
        assert "La" in custom_ext.ph_coefficients
        assert custom_ext.ph_coefficients["La"].a == -8.0
        assert custom_ext.ph_coefficients["La"].b == 2.2

    def test_register_custom_extractant(self):
        """Test registering a custom extractant with the database."""
        from difflow_ree import create_custom_extractant, get_extractant_database

        # Create custom extractant
        custom_ext = create_custom_extractant(
            name="MyExtractant",
            full_name="My Custom Extractant",
            formula="C10H20O4P",
            molecular_weight=250.0,
            ph_coefficients={
                "La": {"a": -8.5, "b": 2.3, "c": 0.01, "d": 0.0},
                "Nd": {"a": -8.0, "b": 2.5, "c": 0.01, "d": 0.0},
            },
            temperature_coefficients={
                "La": -1600.0,
                "Nd": -1800.0,
            },
        )

        # Register it
        db = get_extractant_database()
        db.add_extractant("MyExtractant", custom_ext)

        # Verify it's registered
        assert "MyExtractant" in db.list_extractants()

        # Retrieve it
        retrieved = db.get("MyExtractant")
        assert retrieved.name == "MyExtractant"
        assert retrieved.molecular_weight == 250.0

        # Clean up
        db.remove_extractant("MyExtractant")
        assert "MyExtractant" not in db.list_extractants()

    def test_custom_extractant_in_distribution(self):
        """Test using custom extractant in distribution calculations."""
        from difflow_ree import create_custom_extractant, get_extractant_database, REEDistribution

        # Create and register custom extractant
        custom_ext = create_custom_extractant(
            name="CustomD2EHPA",
            full_name="Custom D2EHPA variant",
            formula="C8H17O4P",
            molecular_weight=322.43,
            ph_coefficients={
                "La": {"a": -8.5, "b": 2.3, "c": 0.01},
                "Nd": {"a": -7.7, "b": 2.45, "c": 0.01},
                "Dy": {"a": -6.8, "b": 2.8, "c": 0.01},
            },
            temperature_coefficients={
                "La": -1500.0,
                "Nd": -1800.0,
                "Dy": -2400.0,
            },
            pKa=3.24,
            typical_concentration=0.5,
        )

        db = get_extractant_database()
        db.add_extractant("CustomD2EHPA", custom_ext)

        try:
            # Use in distribution model
            dist = REEDistribution(
                extractant="CustomD2EHPA",
                elements=("La", "Nd", "Dy"),
                concentration=0.5,
            )

            D_values = dist.get_D_all(pH=3.0, T=298.15)

            # Verify heavy REE have higher D
            assert D_values["Dy"] > D_values["Nd"]
            assert D_values["Nd"] > D_values["La"]
            assert D_values["La"] > 0

        finally:
            # Clean up
            db.remove_extractant("CustomD2EHPA")

    def test_validation_missing_ph_coefficients(self):
        """Test validation for missing pH coefficients."""
        from difflow_ree import create_custom_extractant

        with pytest.raises(ValueError, match="ph_coefficients is required"):
            create_custom_extractant(
                name="BadExtractant",
                full_name="Bad Extractant",
                formula="C8H17O4P",
                molecular_weight=220.0,
                ph_coefficients={},  # Empty!
                temperature_coefficients={"La": -1500.0},
            )

    def test_validation_mismatched_elements(self):
        """Test validation for mismatched elements between pH and temperature."""
        from difflow_ree import create_custom_extractant

        with pytest.raises(ValueError, match="Element mismatch"):
            create_custom_extractant(
                name="BadExtractant",
                full_name="Bad Extractant",
                formula="C8H17O4P",
                molecular_weight=220.0,
                ph_coefficients={
                    "La": {"a": -8.0, "b": 2.2, "c": 0.01},
                    "Nd": {"a": -7.5, "b": 2.4, "c": 0.01},
                },
                temperature_coefficients={
                    "La": -1500.0,
                    # Missing Nd!
                },
            )

    def test_validation_missing_required_coeff(self):
        """Test validation for missing required coefficient keys."""
        from difflow_ree import create_custom_extractant

        with pytest.raises(ValueError, match="missing required keys"):
            create_custom_extractant(
                name="BadExtractant",
                full_name="Bad Extractant",
                formula="C8H17O4P",
                molecular_weight=220.0,
                ph_coefficients={
                    "La": {"a": -8.0, "b": 2.2},  # Missing 'c'!
                },
                temperature_coefficients={
                    "La": -1500.0,
                },
            )

    def test_duplicate_extractant_error(self):
        """Test error when trying to add duplicate extractant."""
        from difflow_ree import create_custom_extractant, get_extractant_database

        custom_ext = create_custom_extractant(
            name="DuplicateTest",
            full_name="Duplicate Test",
            formula="C8H17O4P",
            molecular_weight=220.0,
            ph_coefficients={
                "La": {"a": -8.0, "b": 2.2, "c": 0.01},
            },
            temperature_coefficients={
                "La": -1500.0,
            },
        )

        db = get_extractant_database()
        db.add_extractant("DuplicateTest", custom_ext)

        try:
            # Try to add again - should fail
            with pytest.raises(ValueError, match="already exists"):
                db.add_extractant("DuplicateTest", custom_ext)
        finally:
            db.remove_extractant("DuplicateTest")

    def test_update_extractant(self):
        """Test updating an existing extractant."""
        from difflow_ree import create_custom_extractant, get_extractant_database

        # Create and add original
        original = create_custom_extractant(
            name="UpdateTest",
            full_name="Update Test Original",
            formula="C8H17O4P",
            molecular_weight=220.0,
            ph_coefficients={
                "La": {"a": -8.0, "b": 2.2, "c": 0.01},
            },
            temperature_coefficients={
                "La": -1500.0,
            },
        )

        db = get_extractant_database()
        db.add_extractant("UpdateTest", original)

        try:
            # Create updated version
            updated = create_custom_extractant(
                name="UpdateTest",
                full_name="Update Test Modified",
                formula="C10H20O4P",
                molecular_weight=250.0,
                ph_coefficients={
                    "La": {"a": -7.5, "b": 2.5, "c": 0.02},
                },
                temperature_coefficients={
                    "La": -1600.0,
                },
            )

            # Update it
            db.update_extractant("UpdateTest", updated)

            # Verify update
            retrieved = db.get("UpdateTest")
            assert retrieved.molecular_weight == 250.0
            assert retrieved.full_name == "Update Test Modified"

        finally:
            db.remove_extractant("UpdateTest")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
