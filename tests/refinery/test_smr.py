"""SMR hydrogen plant (#473): Xu-Froment kinetics, reformer tube, shift, furnace, plant.

Smoke, balance, gradient-finite and API tests only (the test policy in
CLAUDE.md); the IEAGHG 2017/02 and Xu & Froment comparisons are not
per-commit tests.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow import make_stream
from difflow_refinery import thermochemistry as tc
from difflow_refinery.hydrogen import PSA, H2Plant
from difflow_refinery.hydrogen import smr

jax.config.update("jax_enable_x64", True)

TUBE_FEED = {"methane": 0.4, "water": 1.2, "hydrogen": 0.04, "carbon_monoxide": 0.0,
             "carbon_dioxide": 0.01, "nitrogen": 0.004, "hydrogen_sulfide": 0.0}


def tube_inlet(**kw):
    f = dict(TUBE_FEED)
    T = kw.pop("T", 893.15)
    P = kw.pop("P", 25e5)
    f.update(kw)
    return make_stream(f, T, P)


#: IEAGHG 2017/02 base case, heat and material balance stream 2 (natural gas
#: feedstock, 128 C, 3.71 MPa, 1455.8 kmol/h); n-pentane (0.0001) lumped into
#: n-butane; H2S 5 ppmv (note 1 of the balance).
IEAGHG_NG = {"methane": 0.89, "ethane": 0.07, "propane": 0.01, "n_butane": 0.0011,
             "carbon_dioxide": 0.02, "nitrogen": 0.0089, "hydrogen_sulfide": 5e-6}


def natural_gas(F=0.45, T=401.15, P=37.1e5):
    """The IEAGHG natural gas, ``F`` mol/s (one tube's worth by default)."""
    return make_stream({k: v * F for k, v in IEAGHG_NG.items()}, T, P)


def elements(n, names):
    tab = tc.IdealGasSet(list(names))
    return {e: float(jnp.sum(jnp.asarray(n) * tab.ELEMENTS[e])) for e in ("C", "H", "O", "N", "S")}


# ---------------------------------------------------------------- kinetics
class TestKinetics:
    def test_rates_smoke_and_grad(self):
        n = smr.to_vec(tube_inlet())
        r = smr.xu_froment_rates(n / jnp.sum(n) * 25.0, 900.0)
        assert r.shape == (3,) and np.all(np.isfinite(np.asarray(r)))
        g = jax.grad(lambda T: jnp.sum(smr.xu_froment_rates(n / jnp.sum(n) * 25.0, T)))(900.0)
        assert np.isfinite(float(g))


# ---------------------------------------------------------------- tube
@pytest.fixture(scope="module")
def tube():
    return smr.reformer_tube(tube_inlet(), smr.ReformerTubeParams(), T_out=1123.15)


class TestTube:
    def test_outlet_temperature_and_duty(self, tube):
        out, info = tube
        assert bool(info["converged"])
        assert float(out["T"]) == pytest.approx(1123.15, abs=1e-3)
        n0, n1 = smr.to_vec(tube_inlet()), smr.to_vec(out)
        dH = smr.enthalpy(n1, out["T"]) - smr.enthalpy(n0, 893.15)
        assert float(info["Q"]) == pytest.approx(float(dH), rel=1e-6)

    def test_element_balance(self, tube):
        out, _ = tube
        a = elements(smr.to_vec(tube_inlet()), smr.SYNGAS)
        b = elements(smr.to_vec(out), smr.SYNGAS)
        for e in a:
            assert b[e] == pytest.approx(a[e], abs=1e-10)

    def test_wall_mode(self):
        tp = smr.ReformerTubeParams(mode="wall", T_wall=(1050.0, 1180.0), U=300.0)
        out, info = smr.reformer_tube(tube_inlet(), tp)
        assert bool(info["converged"])
        n0, n1 = smr.to_vec(tube_inlet()), smr.to_vec(out)
        dH = smr.enthalpy(n1, out["T"]) - smr.enthalpy(n0, 893.15)
        assert float(info["Q"]) == pytest.approx(float(dH), rel=1e-12)

    def test_gradient_is_finite(self):
        """Reverse mode (the default adjoint) through the duty solve;
        tests/reference/gradient_checks.py compares both modes with FD."""
        def f(x):
            out, info = smr.reformer_tube(tube_inlet(water=x[1]), smr.ReformerTubeParams(), T_out=x[0])
            return out["F_methane"] + info["Q"] / 1e5

        assert np.all(np.isfinite(np.asarray(jax.grad(f)(jnp.array([1123.15, 1.2])))))


# ---------------------------------------------------------------- front/back end
class TestUnits:
    def test_desulfurize(self):
        out, removed = smr.desulfurize(natural_gas(), 0.99)
        h = float(natural_gas()["F_hydrogen_sulfide"])
        assert float(removed) == pytest.approx(0.99 * h)
        assert float(out["F_hydrogen_sulfide"]) == pytest.approx(0.01 * h)

    def test_prereformer_converts_c2plus(self):
        ng = natural_gas()
        mixed = dict(ng, F_water=3.0 * smr.carbon_flow(ng))
        out, info = smr.prereform(mixed, 763.15)
        a = elements(smr.to_vec(mixed, smr.SPECIES), smr.SPECIES)
        b = elements(smr.to_vec(out), smr.SYNGAS)
        for e in ("C", "H", "O", "N"):
            assert b[e] == pytest.approx(a[e], rel=1e-6)

    def test_shift_smoke_and_balance(self):
        g = make_stream({"methane": 0.05, "water": 0.8, "hydrogen": 1.3, "carbon_monoxide": 0.25,
                         "carbon_dioxide": 0.1, "nitrogen": 0.0, "hydrogen_sulfide": 0.0}, 613.15, 24e5)
        out, info = smr.shift(g, 613.15)
        assert np.isfinite(float(out["T"]))
        a, b = elements(smr.to_vec(g), smr.SYNGAS), elements(smr.to_vec(out), smr.SYNGAS)
        for e in ("C", "H", "O"):
            assert b[e] == pytest.approx(a[e], rel=1e-10)

    def test_knock_out_saturates(self):
        g = make_stream({"methane": 0.05, "water": 0.8, "hydrogen": 1.3, "carbon_monoxide": 0.02,
                         "carbon_dioxide": 0.3, "nitrogen": 0.0, "hydrogen_sulfide": 0.0}, 450.0, 22e5)
        out, cond, duty = smr.knock_out(g, 313.15)
        from difflow.water import water_saturation_pressure
        n = smr.to_vec(out)
        assert float(n[1] / jnp.sum(n) * 22e5) == pytest.approx(float(water_saturation_pressure(313.15)), rel=1e-10)
        assert float(cond + n[1]) == pytest.approx(0.8)
        assert float(duty) > 0

    def test_furnace_closes_duty(self):
        tail = {"hydrogen": 0.15, "methane": 0.09, "carbon_monoxide": 0.06, "carbon_dioxide": 0.33,
                "water": 0.01}
        fuel = {"methane": 1.0}
        fz = smr.furnace(1.0e5, tail, fuel, y_O2=0.02, T_stack=423.15)
        assert abs(float(fz["closure"])) < 1e-6
        flue = np.asarray(fz["flue"])
        assert flue[3] / flue.sum() == pytest.approx(0.02, rel=1e-12)
        assert float(fz["CO2"]) == pytest.approx(0.09 + 0.06 + 0.33 + float(fz["fuel"]), rel=1e-12)


# ---------------------------------------------------------------- plant
SHORTCUT = PSA(recovery=0.89, purity=0.999)


@pytest.fixture(scope="module")
def plant():
    return smr.smr_plant(natural_gas(), smr.SMRParams(psa=SHORTCUT))


class TestPlant:
    def test_converged(self, plant):
        assert bool(plant["converged"])
        assert float(plant["fuel"]) > 0

    def test_element_balance(self, plant):
        """Feed + steam = product + tail + condensate + ZnO, and furnace C = CO2."""
        ng = natural_gas()
        a = elements(smr.to_vec(ng, smr.SPECIES), smr.SPECIES)
        a["H"] += 2 * float(plant["steam"])
        a["O"] += float(plant["steam"])
        prod = plant["psa"]["product"]
        tail = plant["psa"]["tail_gas"]
        out = elements(jnp.stack([prod[k] + tail[k] for k in smr.SYNGAS]), smr.SYNGAS)
        out["H"] += 2 * float(plant["condensate"]) + 2 * float(plant["H2S_removed"])
        out["O"] += float(plant["condensate"])
        out["S"] += float(plant["H2S_removed"])
        for e in a:
            assert out[e] == pytest.approx(a[e], rel=1e-9, abs=1e-14)
        C_fuel = float(plant["fuel"]) * float(smr.carbon_flow(ng) + ng["F_carbon_dioxide"]) / float(
            sum(ng.get(f"F_{k}", 0.0) for k in smr.SPECIES))
        C_tail = sum(float(tail[k]) for k in ("methane", "carbon_monoxide", "carbon_dioxide"))
        assert float(plant["CO2"]) == pytest.approx(C_fuel + C_tail, rel=1e-10)

    def test_h2plant_from_smr(self, plant):
        h = H2Plant.from_smr(plant, ng_price=4.0)
        assert float(h.capacity) == pytest.approx(float(plant["H2"]))
        assert float(h.purity) == pytest.approx(0.999)
        assert float(h.marginal_cost) == pytest.approx(float(plant["ng_lhv"] / plant["H2"]) * 4e-6, rel=1e-12)

    def test_cycle_model_psa(self):
        r = smr.smr_plant(natural_gas(), smr.SMRParams())
        for v in r["psa"]["balance"].values():
            assert abs(float(v)) < 1e-12
        assert 0.0 < float(r["recovery"]) < 1.0


# ---------------------------------------------------------------- planning
class TestPlanning:
    def test_plant_reverse_mode(self):
        """jax.grad through the whole plant (tube on the checkpointed adjoint);
        tests/reference/gradient_checks.py checks it against forward mode."""
        def f(T):
            return smr.smr_plant(natural_gas(), smr.SMRParams(T_out=T, psa=SHORTCUT))["H2"]

        assert np.isfinite(float(jax.grad(f)(1193.15)))

    def test_smr_block_links_to_network(self, plant):
        from difflow_refinery.hydrogen import link_smr, smr_block
        b = smr_block(natural_gas(), smr.SMRParams(psa=SHORTCUT), jit=False)
        assert b.u_names == ["ng", "S_C", "T_out"]
        assert b.y_names == ["h2.net_mol_s", "h2.purity", "ng.lhv_MW", "co2.mol_s"]
        y = np.asarray(b.fn(jnp.asarray(b.u0)))
        r = plant                                     # the same plant, solved once
        np.testing.assert_allclose(y, [float(r["H2"]), 99.9, float(r["ng_lhv"]) / 1e6, float(r["CO2"])],
                                   rtol=1e-9)
        assert link_smr() == [("smr.h2.net_mol_s", "h2.smr.h2"), ("smr.h2.purity", "h2.smr.purity")]
        assert b.metadata["u_units"] == ["mol/s", "mol/mol", "K"]
