"""#448: rich-end VLE pinch on AmineAbsorber and the closed amine loop.

The absorber's rich loading used to be capped only at the stoichiometric
capacity (0.5 for MEA, carbamate), so capture depended on the circulation
alone. It is now bounded by the loading in equilibrium with the inlet CO2
partial pressure at the operating temperature (the rich-end pinch), with the
capacity kept as an upper bound. ``solve_amine_loop`` closes the
absorber-stripper circuit with the lean solvent as the tear.
"""

import jax
import jax.numpy as jnp
import pytest
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)

from difflow.streams import get_flows, make_stream
from difflow_cc import (
    AbsorberParams,
    AmineAbsorber,
    AmineStripper,
    StripperParams,
    get_solvent,
    solve_amine_loop,
)
from difflow_cc.equilibrium.vle import equilibrium_loading

P_ATM = 101325.0


@pytest.fixture
def flue():
    return make_stream({"CO2": 15.0, "N2": 85.0}, T=313.15, P=P_ATM)


def _absorb(flue, **kw):
    kw.setdefault("solvent", "MEA")
    return AmineAbsorber(AbsorberParams(**kw))(flue)


class TestRichEndPinch:

    def test_rich_loading_is_the_vle_loading_not_the_capacity(self, flue):
        # Default MEA at L/G 3 used to load to 0.5; the solvent's VLE holds
        # only ~0.48 in equilibrium with 15 kPa CO2 at 40 C.
        _, rich, info = _absorb(flue)
        alpha_eq = float(equilibrium_loading(0.15 * P_ATM, 313.15, "MEA"))
        assert alpha_eq < get_solvent("MEA").loading_capacity
        assert bool(info["rich_pinch"])
        assert float(info["rich_loading"]) == pytest.approx(alpha_eq, rel=1e-10)
        assert float(info["P_CO2_rich_end"]) == pytest.approx(0.15 * P_ATM)
        f = get_flows(rich)
        assert float(f["CO2_absorbed"] / f["Amine"]) == pytest.approx(alpha_eq, rel=1e-10)

    def test_stoichiometric_option_keeps_the_old_cap(self, flue):
        _, _, info = _absorb(flue, rich_loading_limit="stoichiometric")
        assert float(info["rich_loading"]) == pytest.approx(0.5)
        _, _, pinched = _absorb(flue)
        assert float(pinched["capture_efficiency"]) < float(info["capture_efficiency"])

    def test_carbon_balance_under_the_pinch(self, flue):
        gas, rich, info = _absorb(flue, L_G_ratio=2.0)
        assert bool(info["rich_pinch"])
        out = float(get_flows(gas)["CO2"]) + float(info["CO2_captured"])
        assert out == pytest.approx(15.0, rel=1e-12)

    def test_pinched_capture_scales_with_circulation_not_stages(self, flue):
        caps = {n: float(_absorb(flue, L_G_ratio=2.0, n_stages=n)[2]["capture_efficiency"])
                for n in (10, 30)}
        assert caps[10] == pytest.approx(caps[30], rel=1e-12)
        c2 = float(_absorb(flue, L_G_ratio=2.0)[2]["capture_efficiency"])
        c1 = float(_absorb(flue, L_G_ratio=1.0)[2]["capture_efficiency"])
        assert c2 == pytest.approx(2.0 * c1, rel=1e-10)

    def test_hotter_column_loads_less(self, flue):
        cold = _absorb(flue, T_liquid_in=313.15)[2]
        hot = _absorb(flue, T_liquid_in=333.15)[2]
        assert float(hot["rich_loading_max"]) < float(cold["rich_loading_max"])
        assert float(hot["capture_efficiency"]) < float(cold["capture_efficiency"])

    def test_richer_gas_loads_more(self):
        lean_gas = make_stream({"CO2": 5.0, "N2": 95.0}, T=313.15, P=P_ATM)
        rich_gas = make_stream({"CO2": 25.0, "N2": 75.0}, T=313.15, P=P_ATM)
        a = _absorb(lean_gas, L_G_ratio=0.5)[2]["rich_loading_max"]
        b = _absorb(rich_gas, L_G_ratio=0.5)[2]["rich_loading_max"]
        assert float(b) > float(a)

    def test_lean_above_equilibrium_captures_nothing(self, flue):
        # No desorption either: the rich loading stays at the lean loading.
        _, _, info = _absorb(flue, lean_loading=0.49)
        assert float(info["capture_efficiency"]) == pytest.approx(0.0, abs=1e-14)
        assert float(info["rich_loading"]) == pytest.approx(0.49)

    def test_rich_approach(self, flue):
        full = _absorb(flue)[2]
        part = _absorb(flue, rich_approach=0.8)[2]
        expect = 0.2 + 0.8 * (float(full["equilibrium_rich_loading"]) - 0.2)
        assert float(part["rich_loading_max"]) == pytest.approx(expect, rel=1e-12)
        for bad in (0.0, 1.5):
            with pytest.raises(ValueError, match="rich_approach"):
                AbsorberParams(solvent="MEA", rich_approach=bad)
        with pytest.raises(ValueError, match="rich_loading_limit"):
            AbsorberParams(solvent="MEA", rich_loading_limit="kinetic")

    def test_required_stages_infeasible_beyond_the_pinch(self, flue):
        ab = AmineAbsorber(AbsorberParams(solvent="MEA"))
        pinched = float(ab(flue)[2]["capture_efficiency"])
        assert jnp.isinf(ab.required_stages(pinched + 0.05, flue))
        assert jnp.isfinite(ab.required_stages(0.5 * pinched, flue))

    def test_gradient_through_the_pinch(self, flue):
        def captured(T):
            return _absorb(flue, T_liquid_in=T)[2]["CO2_captured"]
        assert float(jax.grad(captured)(313.15)) < 0.0
        check_grads(captured, (313.15,), order=1, modes=["fwd", "rev"])


class TestAmineLoop:

    @pytest.fixture
    def gas(self):
        return make_stream({"CO2": 13.0, "N2": 87.0}, T=313.15, P=P_ATM)

    @staticmethod
    def _solve(gas, L_G=3.5, **ab_kw):
        return solve_amine_loop(
            gas,
            AbsorberParams(solvent="MEA", n_stages=15, L_G_ratio=L_G, **ab_kw),
            StripperParams(solvent="MEA", n_stages=10))

    def test_converges_and_balances(self, gas):
        treated, co2, info = self._solve(gas)
        assert bool(info["converged"])
        assert float(info["residual"]) < 1e-10
        assert float(info["CO2_captured"]) == pytest.approx(
            float(info["CO2_stripped"]), rel=1e-9)
        assert float(get_flows(treated)["CO2"]) + float(get_flows(co2)["CO2"]) \
            == pytest.approx(13.0, rel=1e-9)
        # The tear closes: the stripper returns the lean loading it was fed.
        assert float(info["stripper"]["lean_loading"]) == pytest.approx(
            float(info["lean_loading"]), abs=1e-10)
        assert float(info["reboiler_duty"]) > 0.0
        assert 2.0 < float(info["specific_energy"]) < 8.0
        assert float(info["makeup_water"]) >= 0.0

    def test_matches_successive_substitution(self, gas):
        _, _, info = self._solve(gas)
        ab = AmineAbsorber(AbsorberParams(solvent="MEA", n_stages=15, L_G_ratio=3.5))
        st = AmineStripper(StripperParams(solvent="MEA", n_stages=10))
        lean0 = info["lean_solvent"]
        lf = get_flows(lean0)

        @jax.jit
        def step(alpha):
            lean = make_stream({**lf, "CO2_absorbed": alpha * lf["Amine"]},
                               313.15, P_ATM)
            _, rich, _ = ab(gas, lean)
            lean_out, _, _ = st(rich)
            return get_flows(lean_out)["CO2_absorbed"] / lf["Amine"]

        alpha = jnp.asarray(0.3)
        for _ in range(100):
            alpha = step(alpha)
        assert float(alpha) == pytest.approx(float(info["lean_loading"]), abs=1e-9)

    def test_capture_rises_with_circulation(self, gas):
        caps = [float(self._solve(gas, L_G=lg)[2]["capture_efficiency"])
                for lg in (1.5, 2.5, 3.5)]
        assert caps[0] < caps[1] < caps[2]

    def test_vle_limit_lowers_capture(self, gas):
        eq = self._solve(gas)[2]
        st = self._solve(gas, rich_loading_limit="stoichiometric")[2]
        assert bool(eq["rich_pinch"])
        alpha_eq = float(equilibrium_loading(0.13 * P_ATM, 313.15, "MEA"))
        assert float(eq["rich_loading"]) == pytest.approx(alpha_eq, rel=1e-9)
        assert float(st["rich_loading"]) == pytest.approx(0.5)
        assert float(eq["capture_efficiency"]) < float(st["capture_efficiency"])
        # A hotter absorber moves the pinch down and capture with it.
        hot = self._solve(gas, T_liquid_in=323.15)[2]
        assert float(hot["rich_loading"]) < float(eq["rich_loading"])
        assert float(hot["capture_efficiency"]) < float(eq["capture_efficiency"])

    def test_gradient_matches_finite_difference(self, gas):
        def capture(L_G):
            return self._solve(gas, L_G=L_G)[2]["capture_efficiency"]
        g = float(jax.grad(capture)(2.5))
        h = 1e-5
        fd = (float(capture(2.5 + h)) - float(capture(2.5 - h))) / (2 * h)
        assert g > 0.0
        assert g == pytest.approx(fd, rel=1e-6)

    def test_solvent_mismatch_rejected(self, gas):
        with pytest.raises(ValueError, match="solvent"):
            solve_amine_loop(gas, AbsorberParams(solvent="MEA"),
                             StripperParams(solvent="PZ"))
