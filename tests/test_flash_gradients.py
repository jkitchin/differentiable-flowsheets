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

    @pytest.mark.slow
    @pytest.mark.release
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

    @pytest.mark.slow
    @pytest.mark.release
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

    @pytest.mark.slow
    @pytest.mark.release
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


    @pytest.mark.slow
    @pytest.mark.release
    def test_second_derivative_matches_fd(self):
        """d2x/dP2 through ``flash_TP_eos`` matches a central difference of the
        gradient (issue #444). The derivative used to come from unrolling the
        damped substitution, whose second derivative lags the value: here it
        was 10% off (and its sign wrong in a two-flash train)."""
        from difflow.eos import flash_TP_eos

        species = ["methane", "n-butane", "n-decane"]
        eos = PengRobinson({s: get_critical_props(s) for s in species})
        z = jnp.array([0.5, 0.2, 0.3])

        def x_methane(P):
            return flash_TP_eos(eos, z, 320.0, P)[1][0]

        g = jax.jit(jax.grad(x_methane))
        h = jax.jit(jax.grad(jax.grad(x_methane)))
        P, dP = 20e5, 10.0
        h_fd = (g(P + dP) - g(P - dP)) / (2 * dP)
        npt.assert_allclose(float(h(P)), float(h_fd), rtol=1e-5)


# =============================================================================
# PHFlash gradient tests
# =============================================================================

class TestPHFlashGradients:
    """Gradient tests for PH (isenthalpic) flash."""

    @pytest.mark.slow
    @pytest.mark.release
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

    @pytest.mark.release
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


class TestSinglePhaseReverseMode:
    """Reverse-mode gradients through a single-phase flash stay finite.

    The Newton polish of ``flash_TP_eos`` (issue #444) is singular at the
    trivial K = 1 point of a single-phase feed. Its value was protected by a
    select, but reverse mode pulled a zero cotangent through the singular
    solve and returned NaN for every gradient through a supercritical gas,
    e.g. a JT valve's dT_out/dT_in (unit-ops book ex05-05).
    """

    def test_flash_TP_eos_grad_is_finite_and_matches_forward(self):
        from difflow.eos import flash_TP_eos
        sp = ["methane", "ethane", "propane"]
        eos = PengRobinson({s: get_critical_props(s) for s in sp})
        z = jnp.array([0.90, 0.06, 0.04])   # the ex05-05 gas, single phase

        def h(T):   # a smooth function of T through the flash outputs
            V, x, y = flash_TP_eos(eos, z, T, 150e5)
            return eos.enthalpy_departure(T, 150e5, x, "liquid") + V

        rev = jax.grad(h)(323.0)
        fwd = jax.jvp(h, (323.0,), (1.0,))[1]
        assert jnp.isfinite(rev)
        npt.assert_allclose(rev, fwd, rtol=1e-10)

    @pytest.mark.slow
    def test_jt_valve_reverse_gradient(self):
        from difflow import JTValve, JTValveParams
        from difflow.database import get_species_data
        from difflow.thermo import CubicThermo
        sp = ["methane", "ethane", "propane"]
        ng = CubicThermo(IdealThermo({s: get_species_data(s) for s in sp}),
                         PengRobinson({s: get_critical_props(s) for s in sp}))
        flows = {"methane": 90.0, "ethane": 6.0, "propane": 4.0}

        def T_out(T_in):
            out, _ = JTValve(JTValveParams(P_out=70e5), ng)(
                make_stream(flows, T=T_in, P=150e5))
            return out["T"]

        rev = jax.grad(T_out)(323.0)
        fwd = jax.jvp(T_out, (323.0,), (1.0,))[1]
        assert jnp.isfinite(rev)
        npt.assert_allclose(rev, fwd, rtol=1e-8)
