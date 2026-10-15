"""Smoke tests for the evaporator units (single effect, multi-effect, MVR)."""

import jax
import jax.numpy as jnp
import pytest

from difflow import (
    Evaporator, EvaporatorParams, MechanicalVaporRecompression, MultiEffectEvaporator,
    MultiEffectEvaporatorParams, MVRParams, UnverifiedDataWarning, boiling_point_rise,
    make_stream,
)
from difflow.flowsheet import Flowsheet, Unit
from difflow.units.evaporator import (
    water_latent_heat, water_liquid_enthalpy, water_saturation_pressure,
    water_saturation_temperature, water_vapor_enthalpy,
)

MW_NAOH = 40.0
NAOH = dict(solute_species=["NaOH"], solute_MW=[MW_NAOH], bpr_model="ideal",
            vant_hoff_i=2.0)


def naoh_feed(F=10.0, xF=0.10, T=330.0):
    return make_stream({"water": F * (1 - xF) * 1000 / 18.015,
                        "NaOH": F * xF * 1000 / MW_NAOH}, T=T, P=2e5)


def single(**kw):
    return Evaporator(EvaporatorParams(**{**NAOH, "U": 1500.0, "P_vapor_space": 30e3,
                                          "steam_P": 300e3, **kw}))


def _finite(*xs):
    return all(bool(jnp.all(jnp.isfinite(jnp.asarray(x)))) for x in xs)


def test_steam_properties():
    T = jnp.array(373.15)
    assert _finite(water_saturation_pressure(T), water_saturation_temperature(101325.0),
                   water_latent_heat(T), water_liquid_enthalpy(T), water_vapor_enthalpy(T))


def test_boiling_point_rise_models_and_errors():
    vals = [boiling_point_rise(0.2, 1e5, m, solute_MW=40.0, vant_hoff_i=2.0)
            for m in ("ideal", "colligative")]
    vals += [boiling_point_rise(0.5, 1e5, "polynomial", coeffs=[0.0, 1.78, 6.22]),
             boiling_point_rise(0.5, 1e5, "duhring", duhring=([0.0, 0.5], [0.0, -20.0],
                                                              [1.0, 1.1])),
             boiling_point_rise(0.3, 1e5, "callable", bpr_fn=lambda x, Tw, P: 10.0 * x),
             boiling_point_rise(0.5, 1e5, "naoh")]
    assert _finite(*vals)
    with pytest.raises(ValueError):
        boiling_point_rise(0.3, 1e5, "no_such_model")
    with pytest.raises(ValueError):
        boiling_point_rise(0.3, 1e5, "ideal")  # needs solute_MW
    with pytest.warns(UnverifiedDataWarning):
        Evaporator(EvaporatorParams(solute_species=["s"], solute_MW=[40.0], x_product=0.3,
                                    bpr_model="naoh", steam_P=2e5))


def test_single_effect():
    conc, vap, info = single(x_product=0.3)(naoh_feed())
    assert float(info["mass_balance_error"]) == pytest.approx(0.0, abs=1e-10)
    assert float(vap["F_NaOH"]) == 0.0 and _finite(info["steam"], info["A"], conc["T"])


def test_single_effect_grad_is_finite():
    def area(U):
        p = EvaporatorParams(**NAOH, U=U, x_product=0.3, P_vapor_space=30e3, steam_P=300e3)
        return Evaporator(p)(naoh_feed())[2]["A"]
    assert _finite(jax.grad(area)(2500.0))


def test_single_effect_input_errors():
    with pytest.raises(ValueError, match="exactly one"):
        single()
    with pytest.raises(ValueError, match="steam"):
        single(x_product=0.3, steam_P=None)(naoh_feed())
    with pytest.raises(ValueError, match="neither"):
        single(x_product=0.3)(dict(naoh_feed(), F_other=jnp.asarray(1.0)))


def test_multi_effect_in_flowsheet():
    mee = MultiEffectEvaporator(MultiEffectEvaporatorParams(
        **NAOH, n_effects=2, feed="forward", U=[2500.0, 2000.0], P_vapor_space=30e3,
        steam_P=300e3, x_product=0.3))
    fs = Flowsheet(species_order=["water", "NaOH"])
    fs.add_feed("feed", naoh_feed())
    fs.add_unit(Unit("mee", mee, ["feed"], ["conc", "vap"]))
    out = fs.solve()
    assert float(out["vap"]["F_NaOH"]) == 0.0 and _finite(out["conc"]["F_water"])
    with pytest.raises(ValueError, match="feed must be"):
        MultiEffectEvaporator(MultiEffectEvaporatorParams(
            **NAOH, n_effects=2, feed="sideways", U=[1.0, 1.0], x_product=0.3))


def test_mvr():
    p = MVRParams(**NAOH, U=1500.0, x_product=0.3, P_vapor_space=101325.0, dT_drive=10.0,
                  eta=0.75, steam_P=300e3)
    _, cond, info = MechanicalVaporRecompression(p)(naoh_feed())
    assert float(cond["F_NaOH"]) == 0.0 and _finite(info["W_compressor"], info["Q"])


def test_catalog_registration():
    from difflow.catalog import catalog
    cat = catalog()
    for name, roles in [("Evaporator", ["concentrate", "vapor"]),
                        ("MultiEffectEvaporator", ["concentrate", "vapor"]),
                        ("MechanicalVaporRecompression", ["concentrate", "condensate"])]:
        spec = cat[name]
        assert spec.category == "evaporation" and spec.is_buildable
        assert spec.ports.inlets == ["feed"] and spec.ports.outlet_roles == roles
