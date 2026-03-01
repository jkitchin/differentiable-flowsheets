"""Gradient tests for REE unit operations.

Tests that jax.grad works through REE extractors, mixer-settlers,
and precipitators, and that gradient signs are physically meaningful.
"""

import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

pytest.importorskip("difflow_ree")

from difflow.streams import make_stream, get_flows
from difflow_ree.units.extraction import REEExtractor, REEExtractorParams
from difflow_ree.units.extraction import REEMixerSettler, MixerSettlerParams
from difflow_ree.units.precipitation import (
    OxalatePrecipitator,
    HydroxidePrecipitator,
    PrecipitatorParams,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feed():
    """Standard aqueous feed with Nd."""
    return make_stream({"H2O": 10.0, "Nd": 1.0}, T=298.15, P=101325.0)


def _make_solvent():
    """Standard organic solvent."""
    return make_stream({"D2EHPA": 1.0, "kerosene": 5.0}, T=298.15, P=101325.0)


# ---------------------------------------------------------------------------
# REEExtractor gradient tests
# ---------------------------------------------------------------------------

def test_extractor_grad_recovery_wrt_pH():
    """Gradient of Nd recovery through REEExtractor w.r.t. pH is finite and non-zero."""
    params = REEExtractorParams(
        n_stages=5,
        extractant="D2EHPA",
        elements=("Nd",),
        pH=3.0,
        include_loading=False,
    )
    extractor = REEExtractor(params)
    feed = _make_feed()
    solvent = _make_solvent()

    def recovery_fn(pH):
        raff, _, _ = extractor(feed, solvent, pH=pH)
        flows = get_flows(raff)
        nd_out = flows.get("Nd", jnp.float64(0.0))
        return 1.0 - nd_out / 1.0  # recovery

    grad_fn = jax.grad(recovery_fn)
    g = grad_fn(jnp.float64(3.0))

    assert jnp.isfinite(g), f"Gradient is not finite: {g}"
    assert jnp.abs(g) > 1e-10, f"Gradient is effectively zero: {g}"


def test_extractor_grad_pH_sign():
    """Higher pH should increase REE extraction (positive gradient).

    For cation-exchange extractants like D2EHPA, distribution coefficients
    increase with pH because the extraction reaction consumes H+:
        REE3+ + 3(HA)2 -> REE(HA2)3 + 3H+
    """
    params = REEExtractorParams(
        n_stages=5,
        extractant="D2EHPA",
        elements=("Nd",),
        pH=2.5,
        include_loading=False,
    )
    extractor = REEExtractor(params)
    feed = _make_feed()
    solvent = _make_solvent()

    def recovery_fn(pH):
        raff, _, _ = extractor(feed, solvent, pH=pH)
        flows = get_flows(raff)
        nd_out = flows.get("Nd", jnp.float64(0.0))
        return 1.0 - nd_out / 1.0

    g = jax.grad(recovery_fn)(jnp.float64(2.5))
    assert g > 0, (
        f"Expected positive gradient (higher pH -> more extraction), got {g}"
    )


def test_extractor_grad_temperature():
    """Gradient of recovery w.r.t. temperature is finite and non-zero."""
    params = REEExtractorParams(
        n_stages=5,
        extractant="D2EHPA",
        elements=("Nd",),
        pH=3.0,
        include_loading=False,
    )
    extractor = REEExtractor(params)
    feed = _make_feed()
    solvent = _make_solvent()

    def recovery_fn(T):
        raff, _, _ = extractor(feed, solvent, T=T)
        flows = get_flows(raff)
        nd_out = flows.get("Nd", jnp.float64(0.0))
        return 1.0 - nd_out / 1.0

    g = jax.grad(recovery_fn)(jnp.float64(298.15))
    assert jnp.isfinite(g), f"Gradient is not finite: {g}"
    assert jnp.abs(g) > 1e-10, f"Gradient is effectively zero: {g}"


# ---------------------------------------------------------------------------
# REEMixerSettler gradient tests
# ---------------------------------------------------------------------------

def test_mixer_settler_grad_wrt_stage_efficiency():
    """Gradient of extraction w.r.t. stage_efficiency is finite, non-zero, and positive."""
    feed = _make_feed()
    solvent = _make_solvent()

    def extraction_fn(eta):
        params = MixerSettlerParams(
            extractant="D2EHPA",
            elements=("Nd",),
            pH=3.0,
            stage_efficiency=eta,
        )
        ms = REEMixerSettler(params)
        aq_out, _, _ = ms(feed, solvent)
        flows = get_flows(aq_out)
        nd_out = flows.get("Nd", jnp.float64(0.0))
        return 1.0 - nd_out / 1.0

    g = jax.grad(extraction_fn)(jnp.float64(0.85))

    assert jnp.isfinite(g), f"Gradient is not finite: {g}"
    assert jnp.abs(g) > 1e-10, f"Gradient is effectively zero: {g}"
    assert g > 0, (
        f"Expected positive gradient (higher efficiency -> more extraction), got {g}"
    )


def test_mixer_settler_grad_wrt_pH():
    """Gradient of Nd extraction through MixerSettler w.r.t. pH is finite and positive."""
    params = MixerSettlerParams(
        extractant="D2EHPA",
        elements=("Nd",),
        pH=3.0,
    )
    ms = REEMixerSettler(params)
    feed = _make_feed()
    solvent = _make_solvent()

    def extraction_fn(pH):
        aq_out, _, _ = ms(feed, solvent, pH=pH)
        flows = get_flows(aq_out)
        nd_out = flows.get("Nd", jnp.float64(0.0))
        return 1.0 - nd_out / 1.0

    g = jax.grad(extraction_fn)(jnp.float64(3.0))

    assert jnp.isfinite(g), f"Gradient is not finite: {g}"
    assert jnp.abs(g) > 1e-10, f"Gradient is effectively zero: {g}"
    assert g > 0, (
        f"Expected positive gradient (higher pH -> more extraction for D2EHPA), got {g}"
    )


# ---------------------------------------------------------------------------
# Precipitator gradient tests
# ---------------------------------------------------------------------------

def test_oxalate_precipitator_grad_wrt_oxalate_flow():
    """Gradient of yield w.r.t. oxalate flow is finite and non-zero.

    More oxalate means higher excess ratio, which increases precipitation yield.
    Use a low oxalate flow so the conversion is not clamped at target_conversion,
    keeping the gradient alive through jnp.minimum.
    """
    params = PrecipitatorParams(
        elements=("Nd",),
        precipitant_excess=1.5,
        target_conversion=0.9999,  # high ceiling so sqrt branch dominates
    )
    precip = OxalatePrecipitator(params)
    feed = make_stream({"H2O": 10.0, "Nd": 0.5}, T=298.15, P=101325.0)

    # Required oxalate = 1.5 * 0.5 = 0.75 mol/s for stoichiometric.
    # Use sub-stoichiometric flow so actual_excess < 1 and
    # base_conversion * sqrt(actual_excess) < target_conversion.
    def yield_fn(oxalate_flow):
        precipitant = make_stream({"C2O4": oxalate_flow}, T=298.15, P=101325.0)
        filtrate, solid, _ = precip(feed, precipitant)
        solid_flows = get_flows(solid)
        return solid_flows.get("Nd", jnp.float64(0.0)) / 0.5

    g = jax.grad(yield_fn)(jnp.float64(0.1))
    assert jnp.isfinite(g), f"Gradient is not finite: {g}"
    assert jnp.abs(g) > 1e-10, f"Gradient is effectively zero: {g}"
    assert g > 0, (
        f"Expected positive gradient (more oxalate -> higher yield), got {g}"
    )


def test_hydroxide_precipitator_grad_wrt_pH():
    """Gradient of precipitation yield w.r.t. pH is finite and positive.

    Higher pH means more OH-, which drives precipitation of REE(OH)3.
    We use a pH near the onset of precipitation so that conversion is
    not clamped at target_conversion, keeping the gradient alive.
    For Nd (pKsp=20.2), c_feed=0.05 M, the critical pH is ~7.7.
    At pH=7.9, supersaturation S is moderate and conversion < target.
    """
    params = PrecipitatorParams(elements=("Nd",))
    precip = HydroxidePrecipitator(params)

    feed = make_stream({"H2O": 10.0, "Nd": 0.5}, T=298.15, P=101325.0)
    base = make_stream({"NaOH": 5.0, "H2O": 10.0}, T=298.15, P=101325.0)

    def yield_fn(pH):
        filtrate, solid, _ = precip(feed, base, pH=pH)
        solid_flows = get_flows(solid)
        return solid_flows.get("Nd", jnp.float64(0.0)) / 0.5

    g = jax.grad(yield_fn)(jnp.float64(7.9))
    assert jnp.isfinite(g), f"Gradient is not finite: {g}"
    assert jnp.abs(g) > 1e-10, f"Gradient is effectively zero: {g}"
    assert g > 0, (
        f"Expected positive gradient (higher pH -> more precipitation), got {g}"
    )
