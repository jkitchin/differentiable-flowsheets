"""Dense-gas viscosity/conductivity and mixture rules (issue 442, Ch. 5)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from difflow.gas_properties import (
    GAS_CRITICAL, dense_gas_conductivity, dense_gas_viscosity, gas_mixture_transport,
    gas_thermal_conductivity, gas_viscosity, mixture_conductivity, mixture_viscosity,
)

jax.config.update("jax_enable_x64", True)

AIR = (["nitrogen", "oxygen", "argon"], [0.7812, 0.2096, 0.0092])


def test_methane_at_50_bar_is_near_nist():
    # The issue's numbers: 10.75 uPa s dilute, about 12.0 at 50 bar (NIST).
    mu0 = float(gas_viscosity("methane", 288.15)) * 1e6
    mu = float(dense_gas_viscosity("methane", 288.15, 50e5)) * 1e6
    assert mu0 == pytest.approx(10.75, abs=0.02)
    assert mu == pytest.approx(12.0, rel=0.03)


def test_jst_by_hand():
    # Methane, rho given: rho_r = rho Vc / M.
    c = GAS_CRITICAL["methane"]
    rho = 40.0
    rr = rho / 16.0428 * c["Vc"] * 1e-3
    xi = c["Tc"] ** (1 / 6) * 16.0428 ** -0.5 * (c["Pc"] / 1.01325) ** (-2 / 3)
    rhs = 0.1023 + 0.023364 * rr + 0.058533 * rr**2 - 0.040758 * rr**3 + 0.0093324 * rr**4
    expect = float(gas_viscosity("methane", 300.0)) + (rhs**4 - 1e-4) / xi * 1e-3
    got = float(dense_gas_viscosity("methane", 300.0, 60e5, rho=rho))
    assert got == pytest.approx(expect, rel=1e-12)


def test_stiel_thodos_branches_are_continuous():
    # The three published branches meet at rho_r = 0.5 and 2.0 to <0.5%.
    c = GAS_CRITICAL["nitrogen"]
    M = 28.0134

    def k_at(rr):
        rho = rr / (c["Vc"] * 1e-6) * M * 1e-3
        return float(dense_gas_conductivity("nitrogen", 300.0, 1e7, rho=rho))

    for rr in (0.5, 2.0):
        assert k_at(rr - 1e-9) == pytest.approx(k_at(rr + 1e-9), rel=5e-3)


def test_residuals_grow_with_pressure():
    P = [1e5, 20e5, 50e5, 100e5, 150e5]
    mu = [float(dense_gas_viscosity("nitrogen", 300.0, p)) for p in P]
    k = [float(dense_gas_conductivity("nitrogen", 300.0, p)) for p in P]
    assert np.all(np.diff(mu) > 0) and np.all(np.diff(k) > 0)
    # Near 1 bar both are within ~2% of the dilute fits.
    assert mu[0] == pytest.approx(float(gas_viscosity("nitrogen", 300.0)), rel=0.02)
    assert k[0] == pytest.approx(float(gas_thermal_conductivity("nitrogen", 300.0)), rel=0.02)


def test_mixture_rules_reproduce_air():
    mu = float(mixture_viscosity(*AIR, 300.0))
    mu_hz = float(mixture_viscosity(*AIR, 300.0, method="herning_zipperer"))
    k = float(mixture_conductivity(*AIR, 300.0))
    assert mu == pytest.approx(float(gas_viscosity("air", 300.0)), rel=0.01)
    assert mu_hz == pytest.approx(float(gas_viscosity("air", 300.0)), rel=0.01)
    assert k == pytest.approx(float(gas_thermal_conductivity("air", 300.0)), rel=0.02)


def test_pure_limit_and_bad_method():
    for n in ("methane", "carbon_dioxide"):
        assert float(mixture_viscosity([n, "nitrogen"], [1.0, 0.0], 350.0)) == pytest.approx(
            float(gas_viscosity(n, 350.0)), rel=1e-12)
        assert float(mixture_conductivity([n, "nitrogen"], [1.0, 0.0], 350.0)) == pytest.approx(
            float(gas_thermal_conductivity(n, 350.0)), rel=1e-12)
    with pytest.raises(ValueError):
        mixture_viscosity(*AIR, 300.0, method="nope")
    with pytest.raises(ValueError):
        dense_gas_viscosity(["methane", "nitrogen"], 300.0, 1e6)  # no y


def test_mixture_transport_natural_gas():
    r = gas_mixture_transport(["methane", "nitrogen", "carbon_dioxide"],
                              [0.9, 0.05, 0.05], 288.15, 70e5)
    lo = gas_mixture_transport(["methane", "nitrogen", "carbon_dioxide"],
                               [0.9, 0.05, 0.05], 288.15, 70e5, dense=False)
    assert 0.8 < float(r["Z"]) < 0.9
    assert float(r["mu"]) > float(lo["mu"]) and float(r["k"]) > float(lo["k"])
    assert float(r["rho"]) == pytest.approx(70e5 * float(r["MW"]) * 1e-3
                                            / (float(r["Z"]) * 8.314462618 * 288.15))


def test_gradients():
    f = lambda T, P: dense_gas_viscosity("methane", T, P * 1e5) * 1e6  # noqa: E731
    check_grads(f, (288.15, 50.0), order=1, modes=["rev"], atol=1e-5, rtol=1e-5)
    g = lambda y1: dense_gas_conductivity(["methane", "nitrogen"], 300.0, 50e5,  # noqa: E731
                                          y=jnp.stack([y1, 1 - y1])) * 1e3
    check_grads(g, (0.8,), order=1, modes=["rev"], atol=1e-5, rtol=1e-5)
    assert float(jax.grad(lambda P: dense_gas_viscosity("methane", 300.0, P))(50e5)) > 0
