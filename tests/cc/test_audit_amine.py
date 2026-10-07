"""Regression tests for the carbon-capture audit: amine absorber/stripper.

Each test fails on the code before the fix it names (audit items C2, C3,
(a) and C7).
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from difflow.streams import get_flows, make_stream
from difflow_cc import AbsorberParams, AmineAbsorber, AmineStripper, StripperParams



@pytest.fixture
def flue():
    return make_stream({"CO2": 15.0, "N2": 85.0}, T=313.15, P=101325.0)


class TestKremserBelowOne:
    """C2: with A < 1 a Kremser column captures at most the fraction A."""

    @pytest.mark.parametrize("solvent", ["MDEA", "AMP", "DEA"])
    def test_capture_bounded_by_absorption_factor(self, flue, solvent):
        _, _, info = AmineAbsorber(AbsorberParams(solvent=solvent))(flue)
        A = float(info["absorption_factor"])
        assert A < 1.0
        # Before the fix: MDEA 0.973 at A = 0.0017, AMP 0.999 at A = 0.106.
        assert float(info["capture_efficiency"]) <= A + 1e-12

    def test_effective_stages_positive_and_below_actual(self, flue):
        _, _, info = AmineAbsorber(AbsorberParams(solvent="MDEA"))(flue)
        n_eff = float(info["n_stages_effective"])
        assert 0.0 < n_eff <= float(info["n_stages"])

    def test_continuous_through_A_equal_one(self):
        from difflow_cc.units.absorber import _effective_stages, _kremser_unabsorbed
        f = lambda A: _kremser_unabsorbed(A, _effective_stages(10.0, 0.25, A))
        below, at, above = f(1.0 - 2e-6), f(1.0), f(1.0 + 2e-6)
        assert abs(float(below) - float(at)) < 1e-4
        assert abs(float(above) - float(at)) < 1e-4
        g = jax.grad(f)(jnp.asarray(1.0))
        assert jnp.isfinite(g)

    def test_gradient_finite_for_A_below_one(self, flue):
        def cap(L_G):
            p = AbsorberParams(solvent="MDEA", L_G_ratio=L_G)
            return AmineAbsorber(p)(flue)[2]["capture_efficiency"]
        g = jax.grad(cap)(3.0)
        fd = (cap(3.0 + 1e-5) - cap(3.0 - 1e-5)) / 2e-5
        assert g == pytest.approx(float(fd), rel=1e-4)

    def test_required_stages_infeasible_target_is_inf(self, flue):
        ab = AmineAbsorber(AbsorberParams(solvent="AMP"))
        A = float(ab(flue)[2]["absorption_factor"])
        assert jnp.isinf(ab.required_stages(min(0.99, A + 0.2), flue))
        n = float(ab.required_stages(0.5 * A, flue))
        cap = AmineAbsorber(AbsorberParams(solvent="AMP", n_stages=n))(flue)[2]
        assert float(cap["capture_efficiency"]) == pytest.approx(0.5 * A, rel=1e-6)


class TestRichSolventCarriesTotalCO2:
    """C3: rich CO2_absorbed is lean + captured; the stripper reads it."""

    def test_rich_stream_loading_matches_info(self, flue):
        _, rich, info = AmineAbsorber(AbsorberParams(solvent="MEA"))(flue)
        f = get_flows(rich)
        assert float(f["CO2_absorbed"] / f["Amine"]) == pytest.approx(
            float(info["rich_loading"]), rel=1e-12)

    def test_stripper_sees_absorber_rich_loading(self, flue):
        _, rich, ainfo = AmineAbsorber(AbsorberParams(solvent="MEA"))(flue)
        _, _, sinfo = AmineStripper(StripperParams(solvent="MEA"))(rich)
        # Before the fix: absorber 0.50, stripper 0.30.
        assert float(sinfo["rich_loading"]) == pytest.approx(
            float(ainfo["rich_loading"]), rel=1e-12)


class TestAbsorberUsesSolventIn:
    """(a): the fed solvent sets amine, water and lean loading."""

    def test_solvent_in_changes_result(self, flue):
        ab = AmineAbsorber(AbsorberParams(solvent="MEA"))
        lean = make_stream({"H2O": 300.0, "Amine": 40.0, "CO2_absorbed": 16.0},
                           T=313.15, P=101325.0)
        _, rich, info = ab(flue, lean)
        f = get_flows(rich)
        assert float(f["Amine"]) == pytest.approx(40.0)
        assert float(f["H2O"]) == pytest.approx(300.0)
        assert float(info["lean_loading"]) == pytest.approx(0.4)
        # Carbon closes across the unit including the fed solvent's CO2.
        gas_out = get_flows(ab(flue, lean)[0])
        assert float(f["CO2_absorbed"] + gas_out["CO2"]) == pytest.approx(31.0, rel=1e-12)

    def test_solvent_in_without_amine_rejected(self, flue):
        ab = AmineAbsorber(AbsorberParams(solvent="MEA"))
        with pytest.raises(ValueError, match="Amine"):
            ab(flue, make_stream({"H2O": 10.0}, T=313.15, P=101325.0))

    def test_closed_loop_balances(self):
        """Absorber-stripper recycle with makeup water closes on carbon."""
        gas = make_stream({"CO2": 13.0, "N2": 87.0}, T=313.15, P=101325.0)
        ab = AmineAbsorber(AbsorberParams(solvent="MEA", n_stages=15,
                                          L_G_ratio=3.5, lean_loading=0.25))
        st = AmineStripper(StripperParams(solvent="MEA", n_stages=10))
        @jax.jit
        def step(rich):
            lean, prod, sinfo = st(rich)
            lf = dict(get_flows(lean))
            lf["H2O"] = lf["H2O"] + get_flows(prod)["H2O"]  # makeup water
            lean = make_stream(lf, T=313.15, P=101325.0)
            gas_out, rich, ainfo = ab(gas, lean)
            return rich, (gas_out, prod, sinfo, ainfo)

        _, rich, _ = ab(gas)
        for _ in range(200):
            rich, (gas_out, prod, sinfo, ainfo) = step(rich)
        captured = float(ainfo["CO2_captured"])
        stripped = float(sinfo["CO2_stripped"])
        assert captured == pytest.approx(stripped, rel=1e-6)
        assert 13.0 == pytest.approx(
            float(get_flows(gas_out)["CO2"]) + float(get_flows(prod)["CO2"]), rel=1e-6)


class TestStripperOverheadCondenser:
    """#378: the condenser returns reflux; the CO2 product is nearly dry."""

    @pytest.fixture
    def run(self):
        rich = make_stream({"H2O": 70.0, "Amine": 30.0, "CO2_absorbed": 12.0},
                           T=313.15, P=101325.0)

        def go(**kw):
            st = AmineStripper(StripperParams(solvent="MEA", **kw))
            return rich, st(rich)
        return go

    def test_product_is_nearly_dry(self, run):
        _, (lean, prod, info) = run()
        assert float(info["CO2_purity"]) > 0.95
        f = get_flows(prod)
        assert float(f["H2O"]) / float(f["CO2"]) < 0.06

    def test_solvent_water_is_conserved_apart_from_saturation(self, run):
        rich, (lean, prod, info) = run()
        w_in = float(get_flows(rich)["H2O"])
        w_out = float(get_flows(lean)["H2O"]) + float(get_flows(prod)["H2O"])
        assert w_out == pytest.approx(w_in, rel=1e-12)
        assert float(get_flows(lean)["H2O"]) > 0.99 * w_in

    def test_colder_condenser_dries_the_product(self, run):
        _, (_, _, warm) = run(T_condenser=323.15)
        _, (_, _, cold) = run(T_condenser=303.15)
        assert float(cold["CO2_purity"]) > float(warm["CO2_purity"])

    def test_reboiler_duty_still_pays_for_the_stripping_steam(self, run):
        _, (_, _, info) = run(steam_ratio=3.0)
        _, (_, _, base) = run()
        assert float(info["reboiler_duty"]) > float(base["reboiler_duty"])

    def test_loop_closes_without_make_up_water(self):
        gas = make_stream({"CO2": 13.0, "N2": 87.0}, T=313.15, P=101325.0)
        ab = AmineAbsorber(AbsorberParams(solvent="MEA", n_stages=15,
                                          L_G_ratio=3.5, lean_loading=0.25))
        st = AmineStripper(StripperParams(solvent="MEA", n_stages=10))
        _, rich, _ = ab(gas)
        water = []
        for _ in range(40):
            lean, prod, sinfo = st(rich)
            lean = make_stream(dict(get_flows(lean)), T=313.15, P=101325.0)
            gas_out, rich, ainfo = ab(gas, lean)
            water.append(float(get_flows(lean)["H2O"]))
        # the circulating water drifts only by what leaves in the product
        assert water[-1] > 0.9 * water[0]


class TestStripperReboilerConditions:
    """C7: lean loading follows the reboiler equilibrium."""

    @pytest.fixture
    def rich(self):
        return make_stream({"H2O": 70.0, "Amine": 30.0, "CO2_absorbed": 12.0},
                           T=313.15, P=101325.0)

    def _lean(self, rich, **kw):
        return float(AmineStripper(StripperParams(solvent="MEA", **kw))(rich)[2]["lean_loading"])

    def test_temperature_matters(self, rich):
        assert self._lean(rich, T_reboiler=373.15) > self._lean(rich, T_reboiler=403.15)

    def test_pressure_matters(self, rich):
        assert self._lean(rich, P_stripper=1e7) > self._lean(rich, P_stripper=2e5)
        # At 100 bar the reboiler cannot strip this solvent at all.
        assert self._lean(rich, P_stripper=1e7) == pytest.approx(0.4)

    def test_duty_limits_stripping(self, rich):
        st = AmineStripper(StripperParams(solvent="MEA", reboiler_duty=2e5))
        _, prod, info = st(rich)
        free = AmineStripper(StripperParams(solvent="MEA"))(rich)[2]
        assert float(info["CO2_stripped"]) < float(free["CO2_stripped"])
        assert float(info["reboiler_duty"]) == pytest.approx(2e5, rel=1e-9)

    def test_lean_not_below_equilibrium(self, rich):
        _, _, info = AmineStripper(StripperParams(
            solvent="MEA", target_lean_loading=0.05))(rich)
        assert float(info["lean_loading"]) >= float(info["equilibrium_lean_loading"]) - 1e-12

    def test_cold_reboiler_rejected(self, rich):
        with pytest.raises(ValueError, match="below the rich solvent"):
            AmineStripper(StripperParams(solvent="MEA", T_reboiler=300.0))(rich)

    def test_gradient_wrt_reboiler_temperature(self, rich):
        def lean(T):
            return AmineStripper(StripperParams(solvent="MEA", T_reboiler=T,
                                                target_lean_loading=0.05))(rich)[2]["lean_loading"]
        g = jax.grad(lean)(393.15)
        fd = (lean(393.16) - lean(393.14)) / 0.02
        assert float(g) < 0.0
        assert float(g) == pytest.approx(float(fd), rel=1e-4)


def test_equilibrium_loading_inverts_correlation():
    from difflow_cc.equilibrium.vle import co2_equilibrium_pressure, equilibrium_loading
    for P, T in [(24e3, 393.15), (1e3, 313.15), (1.0, 393.15)]:
        a = equilibrium_loading(P, T, "MEA")
        assert float(co2_equilibrium_pressure(a, T, "MEA")) == pytest.approx(P, rel=1e-9)
