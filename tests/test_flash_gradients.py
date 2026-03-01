"""Gradient tests for EOSFlash and PHFlash.

Tests verify that jax.grad produces finite, non-zero gradients through
flash calculations, and that gradient signs match physical intuition.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy.testing as npt

jax.config.update("jax_enable_x64", True)

from difflow.units.flash import (
    EOSFlash, EOSFlashParams, PHFlash, Flash, FlashParams,
)
from difflow.streams import make_stream, get_flows, total_flow
from difflow.thermo import IdealThermo, SpeciesData
from difflow.eos import PengRobinson, CriticalProperties
from difflow.database import get_critical_props


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def hydrocarbon_eos():
    """Peng-Robinson EOS for methane/propane binary."""
    species_data = {
        "methane": get_critical_props("methane"),
        "propane": get_critical_props("propane"),
    }
    return PengRobinson(species_data)


@pytest.fixture
def eos_flash(hydrocarbon_eos):
    """EOSFlash for methane/propane."""
    params = EOSFlashParams(species_order=["methane", "propane"])
    return EOSFlash(params, hydrocarbon_eos)


@pytest.fixture
def binary_thermo():
    """Ideal thermo for Light/Heavy binary (pentane/octane-like)."""
    species_data = {
        "Light": SpeciesData(
            name="Light",
            MW=72.0,
            Cp_coeffs=(120.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(26000.0, 0.38, 470.0),
            antoine_coeffs=(10.422, 1687.537, -38.44),
            Hf=0.0,
        ),
        "Heavy": SpeciesData(
            name="Heavy",
            MW=114.0,
            Cp_coeffs=(190.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(35000.0, 0.38, 570.0),
            antoine_coeffs=(10.186, 2004.68, -60.53),
            Hf=0.0,
        ),
    }
    return IdealThermo(species_data)


# =============================================================================
# EOSFlash gradient tests
# =============================================================================

class TestEOSFlashGradients:
    """Gradient tests for EOS-based flash."""

    def test_gradient_wrt_temperature(self, eos_flash):
        """Gradient of vapor fraction w.r.t. feed temperature.

        Physical intuition: higher T -> more vapor -> positive gradient.
        """
        feed = make_stream(
            {"methane": 40.0, "propane": 60.0},
            T=250.0,
            P=1.5e6,
        )

        def vapor_fraction(T):
            _, _, info = eos_flash(feed, T=T)
            return info["V_frac"]

        grad_val = jax.grad(vapor_fraction)(250.0)

        assert jnp.isfinite(grad_val), "Gradient must be finite"
        assert float(grad_val) != 0.0, "Gradient should be non-zero in two-phase region"
        # Higher T -> more vapor -> positive gradient
        assert float(grad_val) > 0.0, (
            f"dV/dT should be positive (got {float(grad_val)})"
        )

    def test_gradient_wrt_pressure(self, eos_flash):
        """Gradient of vapor fraction w.r.t. feed pressure.

        Physical intuition: higher P -> less vapor -> negative gradient.
        """
        feed = make_stream(
            {"methane": 40.0, "propane": 60.0},
            T=250.0,
            P=1.5e6,
        )

        def vapor_fraction(P):
            _, _, info = eos_flash(feed, P=P)
            return info["V_frac"]

        grad_val = jax.grad(vapor_fraction)(1.5e6)

        assert jnp.isfinite(grad_val), "Gradient must be finite"
        assert float(grad_val) != 0.0, "Gradient should be non-zero in two-phase region"
        # Higher P -> less vapor -> negative gradient
        assert float(grad_val) < 0.0, (
            f"dV/dP should be negative (got {float(grad_val)})"
        )

    def test_gradient_wrt_composition(self, eos_flash):
        """Gradient of vapor-phase methane purity w.r.t. feed methane flow.

        More methane in feed should affect vapor composition.
        """
        def vapor_methane_purity(F_methane):
            feed = make_stream(
                {"methane": F_methane, "propane": 60.0},
                T=250.0,
                P=1.5e6,
            )
            _, vapor, info = eos_flash(feed)
            vapor_flows = get_flows(vapor)
            y_methane = vapor_flows["methane"] / total_flow(vapor)
            return y_methane

        grad_val = jax.grad(vapor_methane_purity)(40.0)

        assert jnp.isfinite(grad_val), "Gradient must be finite"
        assert float(grad_val) != 0.0, "Gradient should be non-zero"


# =============================================================================
# PHFlash gradient tests
# =============================================================================

class TestPHFlashGradients:
    """Gradient tests for PH (isenthalpic) flash."""

    def test_gradient_wrt_feed_temperature(self, binary_thermo):
        """Gradient of vapor fraction w.r.t. feed temperature in PH flash.

        Hotter feed -> more enthalpy -> more vaporization at same P_flash.
        """
        params = FlashParams(species_order=["Light", "Heavy"])
        ph_flash = PHFlash(params, binary_thermo)

        def vapor_fraction(T_feed):
            feed = make_stream(
                {"Light": 50.0, "Heavy": 50.0},
                T=T_feed,
                P=101325.0,
            )
            _, _, info = ph_flash(feed, P=30000.0)
            return info["V_frac"]

        grad_val = jax.grad(vapor_fraction)(380.0)

        assert jnp.isfinite(grad_val), "Gradient must be finite"
        assert float(grad_val) != 0.0, "Gradient should be non-zero"
        # Hotter feed has more enthalpy, so more vaporization
        assert float(grad_val) > 0.0, (
            f"dV/dT_feed should be positive for PH flash (got {float(grad_val)})"
        )


# =============================================================================
# Ideal Flash gradient w.r.t. composition
# =============================================================================

class TestFlashCompositionGradients:
    """Gradient of ideal Flash w.r.t. feed composition."""

    def test_gradient_wrt_light_fraction(self, binary_thermo):
        """Gradient of vapor fraction w.r.t. amount of light component.

        More light component -> easier to vaporize -> more vapor.
        """
        params = FlashParams(species_order=["Light", "Heavy"])
        flash = Flash(params, binary_thermo)

        def vapor_fraction(F_light):
            feed = make_stream(
                {"Light": F_light, "Heavy": 50.0},
                T=350.0,
                P=30000.0,
            )
            _, _, info = flash(feed, T=350.0)
            return info["V_frac"]

        grad_val = jax.grad(vapor_fraction)(50.0)

        assert jnp.isfinite(grad_val), "Gradient must be finite"
        assert float(grad_val) != 0.0, "Gradient should be non-zero"
        # More light component -> higher average volatility -> more vapor
        assert float(grad_val) > 0.0, (
            f"dV/dF_light should be positive (got {float(grad_val)})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
