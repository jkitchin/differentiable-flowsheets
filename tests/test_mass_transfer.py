"""Smoke and API tests for difflow.mass_transfer."""

import warnings

import jax
import numpy as np
import pytest

from difflow import mass_transfer as mt

jax.config.update("jax_enable_x64", True)


def _finite(*vals):
    for v in vals:
        assert np.all(np.isfinite(np.asarray(v, dtype=float)))


def test_diffusivities_and_tables():
    V = mt.diffusion_volume({"C": 3, "H": 5, "Cl": 1})
    _finite(
        mt.fuller_diffusivity(298.0, 1e5, 76.5, 29.0, V, mt.FULLER_VOLUMES["air"]),
        mt.wilke_chang_diffusivity(298.15, 0.89e-3, 18.015, mt.molar_volume_nbp({"C": 2, "H": 6, "O": 1}), 2.6),
        mt.stokes_einstein_diffusivity(298.15, 0.89e-3, 1e-9),
        mt.diffusion_volume("air", table="1966"),
    )
    with pytest.raises(KeyError):
        mt.diffusion_volume("unobtainium")
    with pytest.raises(ValueError):
        mt.diffusion_volume("air", table="1999")


def test_wilke_chang_small_solute_warns():
    with pytest.warns(mt.WilkeChangAccuracyWarning, match="small solute"):
        mt.wilke_chang_diffusivity(298.15, 0.89e-3, 18.015, mt.MOLAR_VOLUMES_NBP["CO2"], 2.6)


def test_film_penetration_and_conversions():
    _finite(
        mt.penetration_coefficient(2e-9, 0.5),
        mt.surface_renewal_coefficient(2e-9, 2.0),
        mt.falling_film_thickness(0.02, 998.0, 1e-3),
        mt.falling_film_liquid_coefficient(2e-9, 0.02, 998.0, 1e-3, 1.0),
        mt.log_mean(3.0, 1.0), mt.log_mean(2.0, 2.0),
        mt.stagnant_film_flux(2.5e-5, 41.0, 0.3, 0.05, 0.1),
        mt.equimolar_flux(2.5e-5, 41.0, 0.3, 0.05, 0.1),
        mt.mass_transfer_coefficient_conversions(0.02, "c", "y", 320.0, 2e5),
        mt.k_L_from_k_x(mt.k_x_from_k_L(1e-4, 55500.0), 55500.0),
    )
    with pytest.raises(ValueError):
        mt.mass_transfer_coefficient_conversions(1.0, "c", "z", 320.0, 2e5)


def test_sherwood_and_two_resistance():
    _finite(
        mt.ranz_marshall(100.0, 1.0), mt.frossling(100.0, 8.0),
        mt.linton_sherwood(1e4, 1000.0), mt.gilliland_sherwood(1e4, 1.0),
        mt.pipe_sherwood(3000.0, 5.0), mt.flat_plate_sherwood(1e6, 2.0, "mixed"),
        mt.chilton_colburn_jD(150.0, 1e4, 0.7),
        mt.sherwood_from_heat_transfer(150.0, 1e4, 0.7, 0.6),
        mt.film_coefficient(30.0, 0.6e-5, 0.01), mt.sherwood(0.02, 0.01, 0.6e-5),
        mt.schmidt(1.85e-5, 1.18, 0.6e-5),
        mt.overall_gas_coefficient(0.5, 2.0, 3.0), mt.overall_liquid_coefficient(0.5, 2.0, 3.0),
        mt.overall_K_G(1e-6, 1e-4, 3000.0), mt.resistance_fractions(0.5, 2.0, 3.0),
    )
    with pytest.raises(ValueError):
        mt.pipe_sherwood(1e4, 1.0, method="nope")


def test_onda():
    pk = mt.PACKINGS["pall_metal_50"]
    _finite(
        mt.onda_wetted_area(0.01, 1000.0, 1e-3, 0.072, pk),
        mt.onda_gas_coefficient(1.0, 1.2, 1.8e-5, 1.6e-5, pk),
        mt.onda_liquid_coefficient(0.01, 1000.0, 1e-3, 1.5e-9, 0.072, pk),
    )
    with pytest.raises(ValueError):
        mt.onda_wetted_area(0.01, 1000.0, 1e-3, 0.072)
    with pytest.raises(ValueError):
        mt.onda_gas_coefficient(1.0, 1.2, 1.8e-5, 1.6e-5, a_p=100.0)


def test_onda_k_L_regression():
    """The old (rho g/mu)^(1/3) factor gave k_L ~ 0.49 m/s."""
    k_L = float(mt.onda_liquid_coefficient(0.01, 1010.0, 2e-3, 1.5e-9, 0.072,
                                           a_p=250.0, d_p=0.05, sigma_c=0.075))
    assert k_L < 1e-2


def test_htu_ntu_hetp():
    _finite(
        mt.htu_og(2.0, 0.5, 100.0), mt.htu_ol(10.0, 0.5, 100.0),
        mt.htu_og_from_films(0.3, 0.2, 0.5),
        mt.ntu_og_colburn(0.05, 0.002, 0.0, 1.0, 0.7),
        mt.ntu_og_colburn(0.05, 0.002, 0.0, 1.0, 1.0),
        mt.ntu_og_integral(0.2, 0.01, lambda y: 0.5 * y, n=16, concentrated=True),
        mt.hetp_from_htu(0.6, 1.0), mt.hetp_from_htu(0.6, 1.2),
    )


def test_grad_finite():
    f = lambda lam: mt.ntu_og_colburn(0.05, 0.002, 0.0, 1.0, lam) * mt.hetp_from_htu(0.6, lam)  # noqa: E731
    for lam in (0.7, 1.0):
        assert np.isfinite(float(jax.grad(f)(lam)))
    assert np.all(np.isfinite(jax.grad(mt.log_mean, argnums=(0, 1))(2.0, 2.0)))


def test_cc_wrappers_deprecated():
    from difflow_cc import kinetics as K
    from difflow_cc.kinetics.mass_transfer import height_of_transfer_unit, number_of_transfer_units

    with pytest.warns(DeprecationWarning):
        K.gas_film_coefficient(1.0, 0.025, 1.8e-5, 1.2, 1.6e-5, 250.0)
    with pytest.warns(DeprecationWarning):
        K.liquid_film_coefficient(u_L=0.01, d_p=0.05, mu_L=2e-3, rho_L=1010.0, D_L=1.5e-9, a_p=250.0)
    with pytest.warns(DeprecationWarning):
        K.interfacial_area(0.01, 1.0, 1000.0, 1e-3, 0.072, 250.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        _finite(number_of_transfer_units(0.1, 0.01, 0.0), height_of_transfer_unit(1.0, 2.0, 3.0))
