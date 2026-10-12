"""EOS units near and inside the two-phase dome (book gaps G05-1/2/3).

The Turboexpander, Compressor and JTValve solve their outlet temperature with a
bracketed (P, S) / (P, H) flash (``flash_PS_eos`` / ``flash_PH_eos``). Before
that, an unbracketed Newton returned its last iterate near a dew point: methane
expanded 100 -> 10 bar gave T_isen = 261.5 K (true 152.1 K) and negative work,
and a JT letdown into the dome returned the 100 K bound or an outlet warmer
than the inlet, all without a flag.

Oracles are computed independently of the units' solver: a scipy ``brentq`` on
the TP-flash entropy, the saturation temperature from equal liquid/vapour
fugacity, and the energy balance closed by the reported vapour fraction.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from scipy.optimize import brentq

from difflow.database import get_critical_props, get_species_data
from difflow.eos import PengRobinson, flash_TP_eos
from difflow.streams import get_flows, make_stream
from difflow.thermo import CubicThermo, IdealThermo
from difflow.units.eos_units import (
    Compressor,
    CompressorParams,
    EOSUnitConvergenceWarning,
    JTValve,
    JTValveParams,
    Turboexpander,
    TurboexpanderParams,
    flash_PH_eos,
    flash_PS_eos,
)
from difflow.units.flash import FlashParams, PHFlash

jax.config.update("jax_enable_x64", True)

CH4 = {"methane": 1.0}


def _pr(species):
    return CubicThermo(
        IdealThermo({s: get_species_data(s) for s in species}),
        PengRobinson({s: get_critical_props(s) for s in species}),
    )


@pytest.fixture(scope="module")
def ch4():
    return _pr(["methane"])


def _T_sat(thermo, P):
    """Pure-methane PR saturation temperature: equal liquid/vapour fugacity."""
    eos, z = thermo.eos, jnp.array([1.0])

    def g(T):
        return float(jnp.log(eos.fugacity_coefficient(T, P, z, "liquid")[0])
                     - jnp.log(eos.fugacity_coefficient(T, P, z, "vapor")[0]))

    # Above the vapour spinodal the cubic has one root and g == 0 identically,
    # so bracket the sign change by scanning up from the liquid side.
    grid = np.arange(100.0, 190.0, 0.5)
    hi = next(T for T in grid if g(T) > 0.0)
    return brentq(g, hi - 0.5, hi, xtol=1e-12)


def _H_two_phase(thermo, T, P, beta):
    """Enthalpy of 1 mol/s methane at (T, P) with vapour fraction beta."""
    eos, z = thermo.eos, jnp.array([1.0])
    H_ig = thermo.ideal.stream_enthalpy_ig(CH4, T)
    return float(H_ig + beta * eos.enthalpy_departure(T, P, z, "vapor")
                 + (1.0 - beta) * eos.enthalpy_departure(T, P, z, "liquid"))


class TestTurboexpanderNearDewPoint:
    @pytest.mark.slow
    def test_isentropic_temperature_matches_independent_root(self, ch4):
        """G05-1: methane 300 K, 100 -> 10 bar."""
        out, info = Turboexpander(
            TurboexpanderParams(P_out=10e5, eta_isentropic=0.85), ch4
        )(make_stream(CH4, T=300.0, P=100e5))
        S_in = float(ch4.stream_entropy_flash(CH4, 300.0, 100e5))
        T_ref = brentq(
            lambda T: float(ch4.stream_entropy_flash(CH4, T, 10e5)) - S_in,
            100.0, 300.0, xtol=1e-10,
        )
        assert T_ref == pytest.approx(152.12, abs=0.01)
        assert float(info["T_isen"]) == pytest.approx(T_ref, abs=1e-6)
        assert bool(info["converged"])
        assert float(info["W"]) > 0.0
        # Expander outlet lies between the isentropic and inlet temperatures.
        assert float(info["T_isen"]) < float(out["T"]) < 300.0
        assert abs(float(info["residual_isen"])) < 1e-8

    @pytest.mark.slow
    def test_expansion_into_dome_reports_liquid(self, ch4):
        """Expanding cold methane deep enough condenses part of it."""
        _, info = Turboexpander(
            TurboexpanderParams(P_out=5e5, eta_isentropic=0.85), ch4
        )(make_stream(CH4, T=220.0, P=60e5))
        assert bool(info["converged"])
        assert 0.0 < float(info["vapor_fraction"]) < 1.0
        assert float(info["T_out"]) == pytest.approx(_T_sat(ch4, 5e5), abs=1e-6)
        assert float(info["W"]) > 0.0

    @pytest.mark.slow
    @pytest.mark.release
    def test_gradients_match_finite_differences(self, ch4):
        def W(P_bar):
            return Turboexpander(
                TurboexpanderParams(P_out=P_bar * 1e5, eta_isentropic=0.85), ch4
            )(make_stream(CH4, T=300.0, P=100e5))[1]["W"]

        check_grads(W, (jnp.array(10.0),), order=1, modes=["rev"], eps=1e-3)


@pytest.mark.slow
class TestJTValveTwoPhase:
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "T_in, P_in, P_out", [(200.0, 60e5, 10e5), (220.0, 80e5, 20e5)]
    )
    def test_letdown_into_dome(self, ch4, T_in, P_in, P_out):
        """G05-2: outlet at T_sat(P_out) with the energy balance closed."""
        out, info = JTValve(JTValveParams(P_out=P_out), ch4)(
            make_stream(CH4, T=T_in, P=P_in))
        H_in = float(ch4.stream_enthalpy_flash(CH4, T_in, P_in))
        T_out, beta = float(out["T"]), float(info["vapor_fraction"])
        assert bool(info["converged"])
        assert T_out < T_in  # below the inversion temperature: cools
        assert T_out == pytest.approx(_T_sat(ch4, P_out), abs=1e-6)
        assert 0.0 < beta < 1.0
        H_out = _H_two_phase(ch4, T_out, P_out, beta)
        assert abs(H_out - H_in) < 1e-6 * abs(H_in)

    @pytest.mark.slow
    def test_unbracketed_target_is_flagged(self, ch4):
        """A target outside T_bounds returns converged=False and warns."""
        valve = JTValve(JTValveParams(P_out=10e5, T_bounds=(160.0, 1000.0)), ch4)
        with pytest.warns(EOSUnitConvergenceWarning):
            out, info = valve(make_stream(CH4, T=200.0, P=60e5))
        assert not bool(info["converged"])

    def test_converged_solve_does_not_warn(self, ch4):
        with warnings.catch_warnings():
            warnings.simplefilter("error", EOSUnitConvergenceWarning)
            JTValve(JTValveParams(P_out=10e5), ch4)(make_stream(CH4, T=200.0, P=60e5))

    @pytest.mark.slow
    @pytest.mark.release
    def test_saturation_slope_and_quality_gradients(self, ch4):
        """Inside the dome dT/dP is the saturation (Clausius-Clapeyron) slope."""
        feed = make_stream(CH4, T=200.0, P=60e5)

        def T_out(P_bar):
            return JTValve(JTValveParams(P_out=P_bar * 1e5), ch4)(feed)[1]["T_out"]

        def beta(P_bar):
            return JTValve(JTValveParams(P_out=P_bar * 1e5), ch4)(feed)[1]["vapor_fraction"]

        check_grads(T_out, (jnp.array(10.0),), order=1, modes=["rev"], eps=1e-3)
        check_grads(beta, (jnp.array(10.0),), order=1, modes=["rev"], eps=1e-3)
        dTdP = float(jax.grad(T_out)(jnp.array(10.0)))
        h = 1e-3
        fd_sat = (_T_sat(ch4, 10e5 + h * 1e5) - _T_sat(ch4, 10e5 - h * 1e5)) / (2 * h)
        assert dTdP == pytest.approx(fd_sat, rel=1e-5)


class TestCompressorFlag:
    @pytest.mark.slow
    def test_compressor_converged(self, ch4):
        _, info = Compressor(CompressorParams(P_out=30e5, eta_isentropic=0.75), ch4)(
            make_stream(CH4, T=300.0, P=10e5))
        assert bool(info["converged"])
        assert float(info["vapor_fraction"]) == 1.0
        assert float(info["T_out"]) > float(info["T_isen"]) > 300.0


class TestSpecFlash:
    @pytest.mark.slow
    def test_ps_flash_round_trip(self, ch4):
        S0 = ch4.stream_entropy_flash(CH4, 180.0, 30e5)
        res = flash_PS_eos(ch4, CH4, 30e5, S0)
        assert float(res["T"]) == pytest.approx(180.0, abs=1e-6)
        assert bool(res["converged"])

    @pytest.mark.slow
    def test_ph_flash_mixture_matches_tp_flash(self):
        """For a mixture the vapour fraction is the TP-flash one at the found T."""
        sp = ["methane", "propane"]
        th = _pr(sp)
        flows = {"methane": 0.7, "propane": 0.3}
        H0 = th.stream_enthalpy_flash(flows, 230.0, 30e5)
        res = flash_PH_eos(th, flows, 5e5, H0)
        assert bool(res["converged"])
        V, _, _ = flash_TP_eos(th.eos, jnp.array([0.7, 0.3]), res["T"], 5e5)
        assert 0.0 < float(res["vapor_fraction"]) < 1.0
        assert float(res["vapor_fraction"]) == pytest.approx(float(V), abs=1e-8)
        H1 = th.stream_enthalpy_flash(flows, res["T"], 5e5)
        assert float(H1) == pytest.approx(float(H0), rel=1e-8)

    @pytest.mark.slow
    def test_phflash_unit_eos_path_splits_streams(self, ch4):
        """PHFlash with a CubicThermo returns the two-phase split."""
        flash = PHFlash(FlashParams(species_order=["methane"]), ch4)
        liq, vap, info = flash(make_stream(CH4, T=200.0, P=60e5), P=10e5)
        Fl, Fv = float(get_flows(liq)["methane"]), float(get_flows(vap)["methane"])
        assert Fl + Fv == pytest.approx(1.0)
        assert Fv == pytest.approx(float(info["V_frac"]))
        assert bool(info["converged"])
        assert float(info["T_flash"]) == pytest.approx(_T_sat(ch4, 10e5), abs=1e-6)
