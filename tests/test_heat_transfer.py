"""Smoke tests for the heat-transfer correlation library."""

import jax
import numpy as np
import pytest

from difflow import heat_transfer as ht

# water, ~40 C (inputs only)
RHO, MU, CP, K = 992.0, 653e-6, 4179.0, 0.631
PR = CP * MU / K
WATER_SAT = dict(mu_l=279e-6, h_fg=2257e3, rho_l=957.9, rho_v=0.5955, sigma=58.9e-3, Cp_l=4217.0, Pr_l=1.76)
COND = dict(k_l=0.68, rho_l=957.9, rho_v=0.5955, mu_l=279e-6, h_fg=2257e3, Cp_l=4217.0)


def test_correlations_are_finite():
    values = [
        ht.reynolds(RHO, 1.5, 0.025, MU), ht.prandtl(CP, MU, K), ht.nusselt_to_h(100.0, K, 0.025),
        ht.grashof(3e-4, 20.0, 0.5, 6.6e-7), ht.rayleigh(3e-4, 20.0, 0.5, 6.6e-7, 1.5e-7),
        ht.slab_resistance(0.01, 0.5, 2.0), ht.cylinder_resistance(0.01, 0.02, 0.5, 1.0),
        ht.sphere_resistance(0.01, 0.02, 0.5), ht.composite_wall([1.0, 2.0, 3.5]),
        ht.critical_insulation_radius(0.05, 10.0), ht.fin_efficiency_straight(50.0, 200.0, 0.002, 0.03),
        ht.dittus_boelter(3e4, PR), ht.sieder_tate(3e4, PR), ht.sieder_tate_laminar(1000.0, PR, 0.01),
        ht.gnielinski(3e4, PR, 0.024), ht.internal_nusselt(3e4, PR), ht.internal_h(3e4, PR, K, 0.025, L=3.0),
        ht.overall_U(4500.0, 350.0, 0.0229, 0.0254, 16.0, R_fi=1e-4),
        ht.churchill_bernstein(1e4, 0.7), ht.flat_plate_nusselt(1e5, 0.7, "laminar"),
        ht.churchill_chu_vertical_plate(1e8, 0.7), ht.churchill_chu_horizontal_cylinder(1e6, 0.7),
        ht.nusselt_film_condensation(dT=8.0, L=0.5, geometry="vertical_plate", **COND),
        ht.rohsenow_heat_flux(10.0, **WATER_SAT), ht.rohsenow_superheat(1e5, **WATER_SAT),
        ht.mostinski(1e5, 22064.0, 101.325),
        ht.critical_heat_flux(2257e3, 0.5955, 957.9, 58.9e-3),
        ht.annulus_nusselt(0.56, "inner"), ht.net_radiation(438.15, 296.15, 0.85, 0.314),
        ht.radiation_h(400.0, 300.0, 0.8), ht.blackbody_emissive_power(5800.0),
        ht.two_surface_radiation(600.0, 300.0, 0.6, 0.3, 2.0), ht.biot_number(100.0, 0.005 / 3, 400.0),
        ht.lumped_heating_time(1000.0, 4180.0, 2000.0, 393.15, 293.15, 353.15),
        ht.lumped_temperature(1234.0, 293.15, 393.15, 4.18e6, 2000.0),
        ht.lumped_time_constant(4.18e6, 2000.0),
    ]
    assert all(np.isfinite(float(v)) for v in values)
    lo, hi = ht.typical_U_range("water_to_oil")
    assert 0 < lo < hi


def test_overall_U_grad_is_finite():
    def U(v):
        h_i = ht.internal_h(ht.reynolds(RHO, v, 0.025, MU), PR, K, 0.025, L=4.0)
        return ht.overall_U(h_i, 1500.0, 0.025, 0.029, 16.0, R_fi=1e-4)

    assert np.isfinite(float(jax.grad(U)(1.5)))


@pytest.mark.parametrize("call,exc", [
    (lambda: ht.critical_insulation_radius(0.05, 10.0, "cube"), ValueError),
    (lambda: ht.internal_nusselt(1e4, 5.0, method="nope"), ValueError),
    (lambda: ht.internal_nusselt(1e4, 5.0, boundary="x"), ValueError),
    (lambda: ht.flat_plate_nusselt(1e5, 0.7, "bad"), ValueError),
    (lambda: ht.nusselt_film_condensation(dT=5.0, L=0.03, geometry="sphere", **COND), ValueError),
    (lambda: ht.overall_U(1.0, 1.0, 1.0, 1.0, 1.0, basis="mean"), ValueError),
    (lambda: ht.typical_U_range("nope"), KeyError),
    (lambda: ht.annulus_nusselt(0.5, "both"), ValueError),
], ids=["geometry", "method", "boundary", "regime", "condensation", "basis", "service", "surface"])
def test_bad_arguments_raise(call, exc):
    with pytest.raises(exc):
        call()
