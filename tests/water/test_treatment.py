"""Conventional treatment units (GAC, disinfection, sedimentation, filtration): code checks."""

import jax
import jax.numpy as jnp
import pytest

import difflow_water as dw
from difflow import make_stream

jax.config.update("jax_enable_x64", True)


def _feed(**extra):
    flows = {"H2O": 5.55e4, "TSS_fine": 20.0, "TSS_coarse": 30.0, "Giardia": 1e3,
             "solute": 0.05}
    flows.update(extra)
    return make_stream(flows, T=288.15, P=101325.0)


def test_gac_runs_and_differentiates():
    treated, removed, info = dw.GACAdsorber(dw.GACParams())(_feed())
    assert jnp.isfinite(info["operational_time"]) and jnp.isfinite(treated["F_solute"])
    assert jnp.allclose(treated["F_solute"] + removed["F_solute"], _feed()["F_solute"])
    g = jax.grad(lambda e: dw.GACAdsorber(dw.GACParams(ebct=e))(_feed())[2]["gac_usage_rate"])(600.0)
    assert jnp.isfinite(g)


def test_gac_rejects_missing_target_and_too_few_elements():
    with pytest.raises(ValueError, match="F_benzene"):
        dw.GACAdsorber(dw.GACParams(target="benzene"))(_feed())
    with pytest.raises(ValueError, match="n_elements"):
        dw.GACAdsorber(dw.GACParams(n_elements=1))


def test_disinfection_runs_and_differentiates():
    out, info = dw.Disinfection(dw.DisinfectionParams())(_feed())
    assert info["log_inactivation"] > 0 and jnp.isfinite(info["CT"])
    assert info["concentration"].shape == (5,)
    g = jax.grad(lambda d: dw.Disinfection(dw.DisinfectionParams(dose=d))(_feed())[1]
                 ["log_inactivation"])(1.5)
    assert jnp.isfinite(g)


def test_disinfection_validation():
    with pytest.raises(ValueError, match="n_tanks"):
        dw.Disinfection(dw.DisinfectionParams(n_tanks=0))
    with pytest.raises(ValueError, match="F_Crypto"):
        dw.Disinfection(dw.DisinfectionParams(organism="Crypto"))(_feed())


def test_sedimentation_runs_and_differentiates():
    clear, sludge, info = dw.Sedimentation(dw.SedimentationParams())(_feed())
    assert jnp.isfinite(info["removal"]).all() and jnp.isfinite(sludge["F_H2O"])
    assert jnp.allclose(clear["F_H2O"] + sludge["F_H2O"], _feed()["F_H2O"])
    g = jax.grad(lambda a: dw.Sedimentation(dw.SedimentationParams(area=a))(_feed())[2]
                 ["removal_total"])(2000.0)
    assert jnp.isfinite(g)


def test_sedimentation_given_velocities_and_validation():
    p = dw.SedimentationParams(settling_velocity=(1e-4, 1e-3))
    _, _, info = dw.Sedimentation(p)(_feed())
    assert jnp.isfinite(info["removal_total"])
    with pytest.raises(ValueError, match="one value per class"):
        dw.Sedimentation(dw.SedimentationParams(diameters=(1e-5,)))
    with pytest.raises(ValueError, match="at least one"):
        dw.Sedimentation(dw.SedimentationParams(classes=()))


def test_filtration_runs_and_differentiates():
    filt, kept, info = dw.Filtration(dw.FiltrationParams())(_feed())
    assert jnp.isfinite(info["head_loss"]) and jnp.isfinite(info["eta"]).all()
    g = jax.grad(lambda d: dw.Filtration(dw.FiltrationParams(media_diameter=d))(_feed())[2]
                 ["removal_total"])(0.5e-3)
    assert jnp.isfinite(g)


def test_filtration_validation():
    with pytest.raises(ValueError, match="one value per class"):
        dw.Filtration(dw.FiltrationParams(particle_density=(1050.0,)))
    with pytest.raises(ValueError, match="F_TSS_fine"):
        dw.Filtration(dw.FiltrationParams())(make_stream({"H2O": 1.0}, T=288.15, P=1e5))
