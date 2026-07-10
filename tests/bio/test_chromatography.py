"""Tests for chromatography unit operations."""

import jax
import jax.numpy as jnp
import pytest

from difflow_bio import (
    ProteinAChromatography,
    ProteinAParams,
    IonExchangeChromatography,
    IEXParams,
    SizeExclusionChromatography,
    SECParams,
    langmuir_isotherm,
    linear_isotherm,
)
from difflow import make_stream, get_flows


# Enable 64-bit precision for tests
jax.config.update("jax_enable_x64", True)


class TestIsotherms:
    def test_langmuir_at_low_C(self):
        """At low C, Langmuir is linear: q ≈ q_max * C / K_d."""
        q = langmuir_isotherm(
            C=jnp.array(0.01),
            q_max=jnp.array(35.0),
            K_d=jnp.array(0.1),
        )
        expected = 35.0 * 0.01 / 0.1
        assert float(q) == pytest.approx(expected, rel=0.1)

    def test_langmuir_at_high_C(self):
        """At high C, Langmuir saturates: q ≈ q_max."""
        q = langmuir_isotherm(
            C=jnp.array(100.0),
            q_max=jnp.array(35.0),
            K_d=jnp.array(0.1),
        )
        assert float(q) == pytest.approx(35.0, rel=0.01)

    def test_langmuir_at_Kd(self):
        """At C = K_d, q = q_max / 2."""
        q = langmuir_isotherm(
            C=jnp.array(0.1),
            q_max=jnp.array(35.0),
            K_d=jnp.array(0.1),
        )
        assert float(q) == pytest.approx(17.5, rel=1e-6)

    def test_linear_isotherm(self):
        """Test linear isotherm q = K * C."""
        q = linear_isotherm(C=jnp.array(5.0), K=jnp.array(10.0))
        assert float(q) == pytest.approx(50.0, rel=1e-6)


class TestProteinAChromatography:
    @pytest.fixture
    def proa_params(self):
        """Protein A column parameters."""
        return ProteinAParams(
            column_volume=jnp.array(1.0),  # 1 L
            q_max=jnp.array(35.0),  # g/L
            K_d=jnp.array(0.1),
            target_species="mAb",
            yield_factor=0.95,
            impurity_clearance={
                "HCP": 2.0,
                "DNA": 3.0,
            },
        )

    def test_proa_creation(self, proa_params):
        """Test Protein A column can be created."""
        proa = ProteinAChromatography(proa_params)
        assert proa is not None

    def test_proa_captures_mab(self, proa_params):
        """Test Protein A captures mAb."""
        proa = ProteinAChromatography(proa_params)

        # Clarified harvest - lower HCP for realistic purity
        feed = make_stream(
            {"mAb": 10.0, "HCP": 10.0, "DNA": 0.1},
            T=300.0, P=101325.0
        )

        (product, waste), info = proa(feed, load_volume=10.0)

        prod_flows = get_flows(product)

        # mAb should be in product with high yield
        assert float(info["yield"]) > 0.9

        # Purity should be high after Protein A
        assert float(info["purity"]) > 0.95

    def test_proa_clears_impurities(self, proa_params):
        """Test Protein A clears HCP and DNA."""
        proa = ProteinAChromatography(proa_params)

        feed = make_stream(
            {"mAb": 10.0, "HCP": 100.0, "DNA": 1.0},
            T=300.0, P=101325.0
        )

        (product, waste), info = proa(feed, load_volume=10.0)

        prod_flows = get_flows(product)
        waste_flows = get_flows(waste)

        # HCP clearance: 2 LRV = 100-fold reduction
        hcp_in = 100.0  # in feed
        hcp_out = float(prod_flows["HCP"])
        hcp_reduction = hcp_in / (hcp_out + 1e-10)
        assert hcp_reduction > 50  # At least 50-fold

        # DNA clearance: 3 LRV = 1000-fold reduction
        dna_in = 1.0
        dna_out = float(prod_flows["DNA"])
        dna_reduction = dna_in / (dna_out + 1e-10)
        assert dna_reduction > 500  # At least 500-fold

    def test_proa_mass_balance(self, proa_params):
        """Test mass balance is preserved: product + waste = total feed."""
        proa = ProteinAChromatography(proa_params)

        feed = make_stream(
            {"mAb": 10.0, "HCP": 100.0, "DNA": 1.0},
            T=300.0, P=101325.0
        )

        (product, waste), info = proa(feed, load_volume=10.0)

        prod_flows = get_flows(product)
        waste_flows = get_flows(waste)
        feed_flows = get_flows(feed)

        # Total mass out must equal total feed (including unloaded portion)
        for species in feed_flows:
            mass_in = float(feed_flows[species])
            mass_out = float(prod_flows[species]) + float(waste_flows[species])
            assert mass_out == pytest.approx(mass_in, rel=0.01)

    def test_proa_differentiability(self, proa_params):
        """Test Protein A is differentiable w.r.t. column volume."""
        def mab_yield(CV):
            params = ProteinAParams(
                column_volume=CV,
                q_max=jnp.array(35.0),
                K_d=jnp.array(0.1),
                target_species="mAb",
                yield_factor=0.95,
            )
            proa = ProteinAChromatography(params)
            feed = make_stream({"mAb": 10.0, "HCP": 100.0}, T=300.0, P=101325.0)
            (product, _), info = proa(feed, load_volume=10.0)
            return product["F_mAb"]

        grad_CV = jax.grad(mab_yield)(jnp.array(1.0))

        # Gradient should be positive (larger column = more capacity = more yield)
        assert float(grad_CV) >= 0

    def test_proa_calculate_load_volume(self, proa_params):
        """Test load volume calculation."""
        proa = ProteinAChromatography(proa_params)

        load_vol = proa.calculate_load_volume(
            feed_concentration=5.0,  # g/L mAb
            target_utilization=0.8,
        )

        # Expected: 0.9 * 35 * 1.0 * 0.8 / 5.0 = 5.04 L
        expected = 0.9 * 35.0 * 1.0 * 0.8 / 5.0
        assert float(load_vol) == pytest.approx(expected, rel=0.1)


class TestIonExchangeChromatography:
    @pytest.fixture
    def iex_params(self):
        """IEX column parameters."""
        return IEXParams(
            column_volume=jnp.array(1.0),
            mode="bind_elute",
            target_species="mAb",
            selectivity={"mAb": 0.9, "HCP": 0.3, "aggregates": 0.95},
            yield_factor=0.90,
        )

    def test_iex_creation(self, iex_params):
        """Test IEX column can be created."""
        iex = IonExchangeChromatography(iex_params)
        assert iex is not None

    def test_iex_bind_elute(self, iex_params):
        """Test IEX bind-elute mode."""
        iex = IonExchangeChromatography(iex_params)

        feed = make_stream(
            {"mAb": 10.0, "HCP": 1.0, "aggregates": 0.5},
            T=300.0, P=101325.0
        )

        (product, waste), info = iex(feed, load_volume=10.0)

        # mAb should be in product
        assert float(info["yield"]) > 0.8

        # Purity should be reasonable
        assert float(info["purity"]) > 0.8

    def test_iex_flow_through(self):
        """Test IEX flow-through mode."""
        params = IEXParams(
            column_volume=jnp.array(1.0),
            mode="flow_through",
            target_species="mAb",
            selectivity={"mAb": 0.1, "HCP": 0.9},  # mAb doesn't bind, HCP binds
            yield_factor=0.95,
        )
        iex = IonExchangeChromatography(params)

        feed = make_stream({"mAb": 10.0, "HCP": 1.0}, T=300.0, P=101325.0)

        (product, waste), info = iex(feed, load_volume=10.0)

        prod_flows = get_flows(product)
        waste_flows = get_flows(waste)

        # mAb flows through (product)
        assert float(prod_flows["mAb"]) > float(waste_flows["mAb"])

        # HCP binds (waste)
        assert float(waste_flows["HCP"]) > float(prod_flows["HCP"])


class TestSizeExclusionChromatography:
    @pytest.fixture
    def sec_params(self):
        """SEC column parameters."""
        return SECParams(
            column_volume=jnp.array(1.0),
            target_species="mAb",
            aggregate_species="aggregates",
            fragment_species="fragments",
            yield_factor=0.95,
        )

    def test_sec_creation(self, sec_params):
        """Test SEC column can be created."""
        sec = SizeExclusionChromatography(sec_params)
        assert sec is not None

    def test_sec_separates_by_size(self, sec_params):
        """Test SEC separates aggregates, monomer, and fragments."""
        sec = SizeExclusionChromatography(sec_params)

        feed = make_stream(
            {"mAb": 95.0, "aggregates": 3.0, "fragments": 2.0},
            T=300.0, P=101325.0
        )

        (product, aggregates, fragments), info = sec(feed, load_volume=10.0)

        prod_flows = get_flows(product)
        agg_flows = get_flows(aggregates)
        frag_flows = get_flows(fragments)

        # mAb should be mostly in product
        assert float(prod_flows["mAb"]) > float(agg_flows["mAb"])
        assert float(prod_flows["mAb"]) > float(frag_flows["mAb"])

        # Aggregates should be mostly in aggregate fraction
        assert float(agg_flows["aggregates"]) > float(prod_flows["aggregates"])

        # Fragments should be mostly in fragment fraction
        assert float(frag_flows["fragments"]) > float(prod_flows["fragments"])

    def test_sec_aggregate_removal(self, sec_params):
        """Test SEC removes aggregates effectively."""
        sec = SizeExclusionChromatography(sec_params)

        feed = make_stream(
            {"mAb": 95.0, "aggregates": 5.0},
            T=300.0, P=101325.0
        )

        (product, aggregates, fragments), info = sec(feed, load_volume=10.0)

        # Aggregate removal should be > 90%
        assert float(info["aggregate_removal"]) > 0.9

    def test_sec_mass_balance(self, sec_params):
        """Test SEC mass balance: all output streams = total feed."""
        sec = SizeExclusionChromatography(sec_params)

        feed = make_stream(
            {"mAb": 95.0, "aggregates": 3.0, "fragments": 2.0},
            T=300.0, P=101325.0
        )

        total_flow = sum(get_flows(feed).values())
        (product, aggregates, fragments), info = sec(feed, load_volume=total_flow)

        feed_flows = get_flows(feed)

        for species in feed_flows:
            mass_in = float(feed_flows[species])
            mass_out = (
                float(get_flows(product).get(species, 0.0)) +
                float(get_flows(aggregates).get(species, 0.0)) +
                float(get_flows(fragments).get(species, 0.0))
            )
            assert mass_out == pytest.approx(mass_in, rel=0.01)


class TestProteinADynamicBindingCapacity:
    """Issue #100: DBC should depend on residence time / flow rate."""

    def _col(self, k_ads=None):
        return ProteinAChromatography(ProteinAParams(
            column_volume=jnp.array(1.0),  # L
            q_max=35.0,
            K_d=0.1,
            target_species="mAb",
            k_ads=k_ads,
        ))

    def test_static_dbc_backward_compat(self):
        """Without k_ads/flow, DBC is the static q_max fraction."""
        col = self._col(k_ads=None)
        feed = make_stream({"mAb": 10.0, "HCP": 1.0}, T=300.0, P=101325.0)
        (product, waste), info = col(feed, load_volume=5.0)
        assert float(info["DBC"]) == pytest.approx(35.0 * 0.99, rel=1e-6)
        assert "residence_time" not in info

    def test_dbc_decreases_with_flow_rate(self):
        """Higher load flow rate -> shorter residence -> lower DBC (#100)."""
        col = self._col(k_ads=0.5)  # 1/min
        feed = make_stream({"mAb": 10.0, "HCP": 1.0}, T=300.0, P=101325.0)
        (_, _), info_slow = col(feed, load_volume=5.0, load_flow_rate=0.1)  # tr=10 min
        (_, _), info_fast = col(feed, load_volume=5.0, load_flow_rate=2.0)  # tr=0.5 min
        assert float(info_slow["DBC"]) > float(info_fast["DBC"])
        assert float(info_fast["residence_time"]) == pytest.approx(0.5, rel=1e-6)

    def test_dbc_gradient_through_flow_rate(self):
        """DBC is differentiable w.r.t. load flow rate."""
        col = self._col(k_ads=0.5)
        feed = make_stream({"mAb": 10.0}, T=300.0, P=101325.0)

        def dbc(q):
            (_, _), info = col(feed, load_volume=5.0, load_flow_rate=q)
            return info["DBC"]

        g = jax.grad(dbc)(0.5)
        assert jnp.isfinite(g)
        assert float(g) < 0.0  # DBC falls as flow rate rises


class TestProteinAElutionPool:
    """Issue #102: elution pool volume from column dimensions / plate count."""

    def test_no_pool_volume_backward_compat(self):
        col = ProteinAChromatography(ProteinAParams(column_volume=1.0))
        feed = make_stream({"mAb": 10.0}, T=300.0, P=101325.0)
        (_, _), info = col(feed, load_volume=5.0)
        assert "elution_pool_volume" not in info

    def test_pool_volume_and_concentration(self):
        col = ProteinAChromatography(ProteinAParams(
            column_volume=2.0, n_plates=400.0, elution_cv=2.0,
        ))
        feed = make_stream({"mAb": 10.0}, T=300.0, P=101325.0)
        (_, _), info = col(feed, load_volume=5.0)
        # w = 4 * (2 CV * 2 L) / sqrt(400) = 4*4/20 = 0.8 L
        assert float(info["elution_pool_volume"]) == pytest.approx(0.8, rel=1e-6)
        assert float(info["product_concentration"]) == pytest.approx(
            float(info["mass_eluted"]) / 0.8, rel=1e-6
        )

    def test_more_plates_narrower_pool(self):
        feed = make_stream({"mAb": 10.0}, T=300.0, P=101325.0)
        low = ProteinAChromatography(ProteinAParams(column_volume=1.0, n_plates=100.0))
        high = ProteinAChromatography(ProteinAParams(column_volume=1.0, n_plates=900.0))
        (_, _), info_low = low(feed, load_volume=5.0)
        (_, _), info_high = high(feed, load_volume=5.0)
        assert float(info_high["elution_pool_volume"]) < float(info_low["elution_pool_volume"])


class TestSECResolutionOverlap:
    """Issue #156: resolution (axial dispersion) should drive peak overlap."""

    def _sec(self, resolution, use_overlap):
        return SizeExclusionChromatography(SECParams(
            column_volume=jnp.array(1.0),
            resolution=resolution,
            use_resolution_overlap=use_overlap,
        ))

    def test_legacy_overlap_unchanged(self):
        """Default path keeps the fixed 5% overlap."""
        sec = self._sec(resolution=1.5, use_overlap=False)
        feed = make_stream({"mAb": 95.0, "aggregates": 5.0}, T=300.0, P=101325.0)
        (product, _, _), info = sec(feed, load_volume=100.0)
        assert float(info["peak_overlap"]) == pytest.approx(0.05, rel=1e-6)

    def test_higher_resolution_less_overlap(self):
        """Better resolution -> less aggregate carryover into product (#156)."""
        feed = make_stream({"mAb": 95.0, "aggregates": 5.0}, T=300.0, P=101325.0)
        (_, _, _), lo = self._sec(1.0, True)(feed, load_volume=100.0)
        (_, _, _), hi = self._sec(2.0, True)(feed, load_volume=100.0)
        assert float(hi["peak_overlap"]) < float(lo["peak_overlap"])
        # Baseline resolution (Rs=1.5) gives ~0.13% overlap
        (_, _, _), base = self._sec(1.5, True)(feed, load_volume=100.0)
        assert float(base["peak_overlap"]) == pytest.approx(0.00134, abs=5e-4)

    def test_mass_balance_with_resolution_overlap(self):
        sec = self._sec(resolution=1.2, use_overlap=True)
        feed = make_stream({"mAb": 95.0, "aggregates": 3.0, "fragments": 2.0}, T=300.0, P=101325.0)
        total = sum(get_flows(feed).values())
        (product, aggregates, fragments), _ = sec(feed, load_volume=total)
        for s in get_flows(feed):
            mass_in = float(get_flows(feed)[s])
            mass_out = (float(get_flows(product).get(s, 0.0))
                        + float(get_flows(aggregates).get(s, 0.0))
                        + float(get_flows(fragments).get(s, 0.0)))
            assert mass_out == pytest.approx(mass_in, rel=1e-6)
