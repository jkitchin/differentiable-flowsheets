"""CCR reformer (#474): activity from coke, moving-bed coke balance, regenerator.

Smoke, balance, gradient-finite and API tests only (the test policy in
CLAUDE.md); the published design points and ranges (AFPM, the patents)
are not per-commit tests.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow_refinery import combustion as cb
from difflow_refinery.reforming import ReformerParams, lean_naphtha
from difflow_refinery.reforming.ccr import (
    CCRReformer, Regenerator, RegeneratorParams, activity,
    ccr_reformer_params, mean_activity)

jax.config.update("jax_enable_x64", True)


class TestActivity:
    def test_fresh_catalyst_is_fully_active(self):
        assert float(activity(0.0, 10.0)) == 1.0

    def test_series_branch_joins_smoothly(self):
        # Either side of |x| = 1e-4 the two branches agree to the series' error.
        a = mean_activity(0.01, 0.01 + 0.99e-5, 10.0)
        b = mean_activity(0.01, 0.01 + 1.01e-5, 10.0)
        assert float(a - b) == pytest.approx(10.0 * np.exp(-0.1) * 0.02e-5 / 2, rel=1e-3)

    def test_gradients(self):
        g = jax.grad(mean_activity, argnums=(0, 1, 2))(0.01, 0.04, 10.0)
        h = jax.hessian(mean_activity, argnums=1)(0.01, 0.04, 10.0)
        g0 = jax.grad(mean_activity, argnums=1)(0.02, 0.02, 10.0)      # on the series branch
        assert np.all(np.isfinite(np.asarray(g))) and np.isfinite(float(h)) and np.isfinite(float(g0))


class TestBedCokes:
    def test_satisfies_the_moving_bed_balance(self):
        u = CCRReformer(ccr_reformer_params())
        s = np.array([1e-4, 2e-4, 3e-4, 5e-4])
        c = u._bed_cokes(s, 0.02, 10.0, 0.001)
        c_in = np.concatenate([[0.001], c[:-1]])
        a = np.array([float(mean_activity(ci, co, 10.0)) for ci, co in zip(c_in, c)])
        np.testing.assert_allclose(c - c_in, s * a / 0.02, rtol=1e-10)

    def test_no_deactivation_is_a_cumulative_sum(self):
        u = CCRReformer(ccr_reformer_params())
        s = np.array([1e-4, 2e-4, 3e-4, 5e-4])
        c = u._bed_cokes(s, 0.02, 0.0, 0.001)
        np.testing.assert_allclose(c, 0.001 + np.cumsum(s) / 0.02, rtol=1e-12)


class TestRegenerator:
    G, C = 0.02, 0.05

    def test_coke_burned_is_the_coke_lifted(self):
        rg = Regenerator()(self.G, self.C)
        B = self.G * (self.C - RegeneratorParams().coke_regen)
        assert float(rg["coke_burn"]) == pytest.approx(B, rel=1e-14)
        assert float(rg["elements"]["C"] + rg["elements"]["H"]) == pytest.approx(B, rel=1e-14)

    def test_coke_composition_from_h_to_c(self):
        el = Regenerator(RegeneratorParams(coke_HC=0.5)).elements(1.0)
        nC = el["C"] / cb.ATOMIC_WEIGHT["C"]
        nH = el["H"] / cb.ATOMIC_WEIGHT["H"]
        assert float(nH / nC) == pytest.approx(0.5, rel=1e-14)

    def test_burn_zone_is_o2_limited_at_the_inlet_o2(self):
        p = RegeneratorParams()
        rg = Regenerator(p)(self.G, self.C)
        gas = rg["burn_gas"]
        assert float(gas["oxygen"] / sum(gas.values())) == pytest.approx(p.y_O2, rel=1e-12)
        assert "oxygen" not in rg["burn_outlet"] or float(rg["burn_outlet"]["oxygen"]) == 0.0
        # The air brings exactly the O2 the coke takes: C + H/4 per mol.
        el = rg["elements"]
        o2 = el["C"] * 1000 / cb.ATOMIC_WEIGHT["C"] + el["H"] * 1000 / cb.ATOMIC_WEIGHT["H"] / 4
        assert float(rg["air"]["oxygen"]) == pytest.approx(float(o2), rel=1e-12)

    def test_element_balance_of_the_burn_zone(self):
        rg = Regenerator()(self.G, self.C)
        i, o, el = rg["burn_gas"], rg["burn_outlet"], rg["elements"]
        C = lambda g: g.get("carbon_dioxide", 0.0) + g.get("carbon_monoxide", 0.0)
        H = lambda g: 2 * g.get("water", 0.0)
        O = lambda g: (2 * g.get("oxygen", 0.0) + 2 * g.get("carbon_dioxide", 0.0)
                       + g.get("carbon_monoxide", 0.0) + g.get("water", 0.0))
        assert float(C(o) - C(i)) == pytest.approx(float(el["C"]) * 1000 / cb.ATOMIC_WEIGHT["C"], rel=1e-12)
        assert float(H(o) - H(i)) == pytest.approx(float(el["H"]) * 1000 / cb.ATOMIC_WEIGHT["H"], rel=1e-12)
        assert float(O(o)) == pytest.approx(float(O(i)), rel=1e-12)
        assert float(o["nitrogen"]) == pytest.approx(float(i["nitrogen"]), rel=1e-14)

    def test_peak_temperature_closes_the_energy_balance(self):
        rg = Regenerator()(self.G, self.C)
        T_in = RegeneratorParams().T_burn_in
        gained = cb.gas_enthalpy(rg["burn_outlet"], rg["T_peak"]) - cb.gas_enthalpy(rg["burn_outlet"], T_in)
        assert float(gained) == pytest.approx(float(rg["heat"]), rel=1e-10)

    def test_gradients(self):
        def f(G, C, y):
            rg = Regenerator(RegeneratorParams(y_O2=y))(G, C)
            return jnp.stack([rg["T_peak"], rg["coke_burn"], rg["burn_capacity_used"],
                              rg["H2_reduction"], rg["recycle"]])
        for J in (jax.jacfwd(f, argnums=(0, 1, 2))(self.G, self.C, 0.008),
                  jax.jacrev(f, argnums=(0, 1, 2))(self.G, self.C, 0.008)):
            assert np.all(np.isfinite(np.asarray(J)))


def test_reformer_activity_length_is_checked():
    with pytest.raises(ValueError, match="activity"):
        ReformerParams(activity=(1.0, 1.0))
    with pytest.raises(ValueError, match="leave it None"):
        CCRReformer().solve(lean_naphtha(), ccr_reformer_params(activity=(1.0,) * 4))


# ----- flowsheet ------------------------------------------------------------------


@pytest.fixture(scope="module")
def base():
    return CCRReformer().solve(lean_naphtha())


class TestCCRFlowsheet:
    def test_converges(self, base):
        assert base.converged
        assert base.reformer.converged

    def test_coke_made_is_coke_burned(self, base):
        b = base.balances()
        assert abs(float(b["coke"])) < 1e-8
        assert abs(float(b["catalyst_coke"])) < 1e-8
        for k in ("mass", "carbon", "hydrogen"):
            assert abs(float(b[k])) < 1e-9, k

def test_ccr_block(base):
    """The block's levers and base point; its unjitted fn (~3 min) is
    evaluated in tests/reference/refinery_checks.py."""
    from difflow_refinery.reforming.planning import CCR_OUTPUTS, ccr_block

    blk = ccr_block(CCRReformer(base.params, base.ccr), lean_naphtha(),
                    levers=["wait", "circulation"])
    assert blk.u_names == ["wait", "circulation"] and blk.y_names == list(CCR_OUTPUTS)
    assert blk.u0[1] == pytest.approx(float(base.ccr.circulation) * 3600.0)
