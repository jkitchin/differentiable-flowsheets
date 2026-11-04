"""ROStage, PressureExchanger and WaterPump: smoke, gradient, input validation."""

import jax
import jax.numpy as jnp
import pytest

import difflow_water as dw

RO = dict(A_w=4.2e-12, B=3.5e-8, area=50.0, n_steps=10)


def _feed(P=65e5):
    return dw.make_seawater_stream(flow_vol=1e-3, salinity=0.035, P=P)


@pytest.mark.parametrize("kw", [{}, {"k": 5e-5, "pressure_drop": "none"},
                                {"k": 5e-5, "pressure_drop": 1e5}],
                         ids=["channel", "fixed_k", "fixed_dp"])
def test_ro_stage_runs(kw):
    conc, perm, info = dw.ROStage(dw.ROStageParams(**RO, **kw))(_feed())
    assert set(conc) == set(perm) == {"F_H2O", "F_TDS", "T", "P"}
    for key in ("recovery_mass_water", "salt_rejection", "salinity_permeate",
                "deltaP", "min_driving_pressure"):
        assert jnp.isfinite(info[key]), key


def test_ro_stage_gradient_is_finite():
    def rec(P, area):
        p = dw.ROStageParams(A_w=4.2e-12, B=3.5e-8, area=area, n_steps=10)
        return dw.ROStage(p)(_feed(P))[2]["recovery_mass_water"]
    g = jax.grad(rec, argnums=(0, 1))(65e5, 50.0)
    assert all(jnp.isfinite(x) for x in g)


def test_ro_stage_below_osmotic_pressure_is_finite():
    """No driving force (feed below the osmotic pressure): zero flux, still finite."""
    _, perm, info = dw.ROStage(dw.ROStageParams(**RO, k=5e-5))(_feed(P=10e5))
    assert jnp.isfinite(perm["F_H2O"]) and jnp.isfinite(info["salinity_permeate"])


def test_ro_stage_needs_seawater_species():
    bad = {"F_H2O": 1.0, "F_NaCl": 0.1, "T": 298.15, "P": 6e6}
    with pytest.raises(ValueError):
        dw.ROStage(dw.ROStageParams(**RO))(bad)


def test_ro_stage_rejects_unknown_pressure_drop():
    with pytest.raises(ValueError):
        dw.ROStage(dw.ROStageParams(**RO, pressure_drop="ergun"))


def test_pressure_exchanger_runs_and_differentiates():
    brine = dw.make_seawater_stream(flow_vol=5e-4, salinity=0.07, P=60e5)
    feed = dw.make_seawater_stream(flow_vol=1e-3, salinity=0.035, P=2e5)
    hp, lp, bypass, info = dw.PressureExchanger(dw.PressureExchangerParams(mixing=0.03))(brine, feed)
    for s in (hp, lp, bypass):
        assert all(jnp.isfinite(v) for v in s.values())
    assert all(jnp.isfinite(v) for v in info.values())

    def p_hp(eta):
        return dw.PressureExchanger(dw.PressureExchangerParams(efficiency=eta))(brine, feed)[0]["P"]
    assert jnp.isfinite(jax.grad(p_hp)(0.95))


def test_pump_runs_and_differentiates():
    out, info = dw.WaterPump(dw.WaterPumpParams(deltaP=5e6))(_feed(P=1e5))
    assert jnp.isfinite(out["P"]) and jnp.isfinite(info["work_mechanical"])
    g = jax.grad(lambda P: dw.WaterPump(dw.WaterPumpParams(outlet_pressure=P))(_feed(1e5))[1]
                 ["work_mechanical"])(6e6)
    assert jnp.isfinite(g)


def test_pump_needs_a_pressure_spec():
    with pytest.raises(ValueError):
        dw.WaterPump(dw.WaterPumpParams(deltaP=None))


def test_pump_outlet_pressure_takes_precedence():
    """API contract: with both given, outlet_pressure wins."""
    out, _ = dw.WaterPump(dw.WaterPumpParams(outlet_pressure=6e6, deltaP=1e5))(_feed(P=1e5))
    assert float(out["P"]) == 6e6


def test_ro_stage_reports_saturation_indices():
    _, _, info = dw.ROStage(dw.ROStageParams(**RO, scaling=True))(_feed())
    for key in ("SI_concentrate", "SI_wall_outlet"):
        assert all(jnp.isfinite(v) for v in info[key].values()), key


def test_ro_train_runs_and_differentiates():
    s = dw.ROStageParams(**{**RO, "area": 30.0})
    conc, perm, info = dw.ROTrain(dw.ROTrainParams(stages=[s, s], boost=[5e5]))(_feed())
    assert len(info["stages"]) == 2 and jnp.isfinite(info["recovery_mass_water"])

    def rec(boost):
        t = dw.ROTrain(dw.ROTrainParams(stages=[s, s], boost=[boost]))
        return t(_feed())[2]["recovery_mass_water"]
    assert jnp.isfinite(jax.grad(rec)(5e5))


@pytest.mark.parametrize("kw", [{"stages": []}, {"boost": [1e5, 1e5]}])
def test_ro_train_input_validation(kw):
    s = dw.ROStageParams(**RO)
    with pytest.raises(ValueError):
        dw.ROTrain(dw.ROTrainParams(**{"stages": [s, s], **kw}))


def test_pressure_exchanger_with_leakage_runs():
    brine = dw.make_seawater_stream(flow_vol=5e-4, salinity=0.07, P=60e5)
    feed = dw.make_seawater_stream(flow_vol=1e-3, salinity=0.035, P=2e5)
    hp, _, _, info = dw.PressureExchanger(dw.PressureExchangerParams(leakage=0.01))(brine, feed)
    assert jnp.isfinite(hp["F_H2O"]) and jnp.isfinite(info["flow_vol_exchanged"])
