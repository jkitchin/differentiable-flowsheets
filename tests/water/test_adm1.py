"""Anaerobic digestion (ADM1, BSM2 implementation): code checks."""

import jax
import jax.numpy as jnp
import pytest

import difflow_water as dw
from difflow_water.wastewater.adm1 import (
    STATES,
    adm1_test_influent,
    charge_balance_SH,
    constants,
    rhs,
    stream_to_state,
)

jax.config.update("jax_enable_x64", True)

S_MID = jnp.array([0.01, 0.005, 0.1, 0.01, 0.01, 0.02, 0.2, 2e-7, 0.05, 0.15, 0.13, 0.3, 0.3,
                   0.03, 0.1, 0.03, 0.4, 1.2, 0.24, 0.43, 0.14, 0.76, 0.32, 25.6, 0.04, 0.02])
G_MID = jnp.array([1e-5, 1.6, 0.014])


def test_charge_balance_root_and_gradient():
    p = dw.ADM1Params()
    c = constants(p)
    SH = charge_balance_SH(S_MID, c)
    assert 1e-10 < float(SH) < 1e-4
    g = jax.grad(lambda s: charge_balance_SH(S_MID.at[STATES.index("S_cat")].set(s), c))(0.04)
    assert jnp.isfinite(g) and g < 0   # more cations, less H+


@pytest.mark.parametrize("ph", ["hill", "switch"])
def test_rhs_finite(ph):
    q, S_in = stream_to_state(adm1_test_influent())
    dS, dG, d = rhs(S_MID, G_MID, q, S_in, dw.ADM1Params(pH_inhibition=ph))
    assert jnp.isfinite(dS).all() and jnp.isfinite(dG).all() and jnp.isfinite(d["q_gas"])


def test_digester_runs_and_differentiates():
    eff, gas, info = dw.AnaerobicDigester(dw.ADM1Params())(adm1_test_influent())
    assert bool(info["converged"]) and jnp.isfinite(info["pH"])
    assert jnp.isfinite(gas["F_CH4"]) and jnp.isfinite(eff["F_S_ac"])
    g = jax.grad(lambda k: dw.anaerobic_digester(adm1_test_influent(),
                                                 dw.ADM1Params(k_dis=k))["methane"])(0.5)
    assert jnp.isfinite(g)


def test_digester_validation():
    with pytest.raises(ValueError, match="gas_flow"):
        dw.AnaerobicDigester(dw.ADM1Params(gas_flow="bogus"))
    with pytest.raises(ValueError, match="pH_inhibition"):
        dw.anaerobic_digester(adm1_test_influent(), dw.ADM1Params(pH_inhibition="bogus"))
