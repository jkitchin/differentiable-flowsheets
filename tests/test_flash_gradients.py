"""jax.grad runs and is finite through PHFlash, the ideal Flash and flash_TP_eos."""

import jax
import jax.numpy as jnp

from difflow.database import get_critical_props
from difflow.eos import PengRobinson, flash_TP_eos
from difflow.streams import make_stream
from difflow.thermo import IdealThermo, SpeciesData
from difflow.units.flash import Flash, FlashParams, PHFlash

THERMO = IdealThermo({
    "Light": SpeciesData(name="Light", MW=72.0, Cp_coeffs=(120.0, 0.0, 0.0, 0.0),
                         Hvap_coeffs=(26000.0, 0.38, 470.0),
                         antoine_coeffs=(10.422, 1687.537, -38.44), Hf=0.0),
    "Heavy": SpeciesData(name="Heavy", MW=114.0, Cp_coeffs=(190.0, 0.0, 0.0, 0.0),
                         Hvap_coeffs=(35000.0, 0.38, 570.0),
                         antoine_coeffs=(10.186, 2004.68, -60.53), Hf=0.0),
})
PARAMS = FlashParams(species_order=["Light", "Heavy"])


def test_ph_flash_grad_is_finite():
    flash = PHFlash(PARAMS, THERMO)

    def V(T_feed):
        feed = make_stream({"Light": 50.0, "Heavy": 50.0}, T=T_feed, P=101325.0)
        return flash(feed, P=30000.0)[2]["V_frac"]
    assert jnp.isfinite(jax.grad(V)(380.0))


def test_ideal_flash_grad_is_finite():
    flash = Flash(PARAMS, THERMO)

    def V(F_light):
        feed = make_stream({"Light": F_light, "Heavy": 50.0}, T=350.0, P=30000.0)
        return flash(feed, T=350.0)[2]["V_frac"]
    assert jnp.isfinite(jax.grad(V)(50.0))


def test_single_phase_reverse_mode_is_not_nan():
    """#444: reverse mode through a single-phase flash_TP_eos returned NaN."""
    sp = ["methane", "ethane", "propane"]
    eos = PengRobinson({s: get_critical_props(s) for s in sp})
    z = jnp.array([0.90, 0.06, 0.04])

    def h(T):
        V, x, _ = flash_TP_eos(eos, z, T, 150e5)
        return eos.enthalpy_departure(T, 150e5, x, "liquid") + V
    assert jnp.isfinite(jax.grad(h)(323.0))


def test_chained_flash_second_derivative_finite():
    """#497: a flash fed by another flash's liquid, differentiated twice."""
    sp = ["methane", "propane", "n_hexane"]
    eos = PengRobinson({s: get_critical_props(s) for s in sp})
    z = jnp.array([0.5, 0.2, 0.3])

    def liquid(P1):
        V1, x1, _ = flash_TP_eos(eos, z, 310.0, P1 * 1e5)
        V2, _, _ = flash_TP_eos(eos, x1, 310.0, 101325.0)
        return (1 - V1) * (1 - V2)
    assert jnp.isfinite(jax.jit(jax.hessian(liquid))(8.0))
