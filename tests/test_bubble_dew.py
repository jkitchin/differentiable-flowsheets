"""Bubble/dew points: EOS functions, modified Raoult, Flash helpers, warnings."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow import (
    IdealThermo, PengRobinson, SpeciesData, make_stream,
    AntoineRangeWarning, BubbleDewConvergenceWarning,
    eos_bubble_P, bubble_T, bubble_P, dew_T, dew_P, txy,
)
from difflow.database import get_critical_props, get_species_data
from difflow.units.flash import Flash, FlashParams
from difflow.units.lle import NRTLParams

jax.config.update("jax_enable_x64", True)

SP_EW = ("ethanol", "water")


def _ethanol_water():
    th = IdealThermo({
        "ethanol": SpeciesData(name="ethanol", MW=46.07, Cp_coeffs=(0, 0, 0, 0),
                               Hvap_coeffs=(1.0, 0.38, 514.0),
                               antoine_coeffs=(10.24677, 1598.673, -46.424)),
        "water": SpeciesData(name="water", MW=18.02, Cp_coeffs=(0, 0, 0, 0),
                             Hvap_coeffs=(1.0, 0.38, 647.0),
                             antoine_coeffs=(9.6543, 1435.264, -64.848)),
    })
    nrtl = NRTLParams(
        species=SP_EW,
        a=jnp.array([[0.0, -0.8009], [3.4578, 0.0]]),
        b=jnp.array([[0.0, 246.18], [-586.0809, 0.0]]),
        alpha=jnp.array([[0.0, 0.3], [0.3, 0.0]]),
    )
    return th, nrtl


def _c1c3():
    return PengRobinson({s: get_critical_props(s) for s in ("methane", "propane")})


def test_eos_bubble_P_smoke():
    r = eos_bubble_P(_c1c3(), jnp.array([0.2, 0.8]), 277.6)
    assert np.isfinite(float(r["P"])) and "converged" in r


def test_modified_raoult_smoke():
    th, nrtl = _ethanol_water()
    x = jnp.array([0.4, 0.6])
    for v in (bubble_P(th, SP_EW, x, 350.0, nrtl), bubble_T(th, SP_EW, x, 101325.0, nrtl),
              dew_P(th, SP_EW, x, 360.0, nrtl), dew_T(th, SP_EW, x, 101325.0, nrtl)):
        assert np.isfinite(float(v))


def test_flash_bubble_point_smoke():
    th, nrtl = _ethanol_water()
    fl = Flash(FlashParams(species_order=list(SP_EW)), th, activity_model=nrtl)
    feed = make_stream({"ethanol": 30.0, "water": 70.0}, T=350.0, P=101325.0)
    assert np.isfinite(float(fl.bubble_point_pressure(feed)))
    assert np.isfinite(float(fl.bubble_point_temperature(feed)))


def test_no_silent_clip_above_700K():
    heavy = SpeciesData(name="H", MW=300.0, Cp_coeffs=(0, 0, 0, 0),
                        Hvap_coeffs=(1.0, 0.38, 900.0), antoine_coeffs=(9.0, 3000.0, 0.0))
    fl = Flash(FlashParams(species_order=["H"]), IdealThermo({"H": heavy}))
    feed = make_stream({"H": 1.0}, T=300.0, P=101325.0)
    assert float(fl.bubble_point_temperature(feed)) > 700.0


def test_unconverged_solve_warns():
    th, _ = _ethanol_water()
    fl = Flash(FlashParams(species_order=list(SP_EW)), th)
    feed = make_stream({"ethanol": 30.0, "water": 70.0}, T=350.0, P=101325.0)
    with pytest.warns(BubbleDewConvergenceWarning):
        T = fl.bubble_point_temperature(feed, T_guess=float("nan"))
    assert not np.isfinite(float(T))


class TestAntoineRangeWarning:
    def _thermo(self):
        return IdealThermo({s: get_species_data(s) for s in ("benzene", "toluene")})

    def test_psat_out_of_range_warns(self):
        with pytest.warns(AntoineRangeWarning, match="'benzene' extrapolated below"):
            self._thermo().Psat("benzene", 250.0)

    def test_no_warning_under_jit_or_grad(self):
        th = self._thermo()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            jax.jit(lambda T: th.Psat("benzene", T))(250.0)
            jax.grad(lambda T: th.Psat("benzene", T))(250.0)

    def test_txy_warns_once_per_species(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            txy(self._thermo(), ("benzene", "toluene"), 20 * 101325.0, n=5)
        msgs = [str(x.message) for x in w if issubclass(x.category, AntoineRangeWarning)]
        assert len(msgs) == 2

    def test_psat_with_info_works_under_jit(self):
        th = self._thermo()
        ok = jax.jit(lambda T: th.Psat_with_info("benzene", T)[1]["antoine_in_range"])
        assert bool(ok(350.0)) and not bool(ok(250.0))

