"""Smoke tests for difflow.units.membranes: gas permeation, reverse osmosis, UF."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow import make_stream
from difflow.units import membranes as M

jax.config.update("jax_enable_x64", True)

F2 = jnp.array([0.5, 0.5])
Q2 = jnp.array([1e-2, 1e-3])


def test_binary_helpers_run():
    y = M.binary_permeate_composition(0.5, 10.0, 0.25)
    a = M.binary_complete_mixing(0.5, 10.0, 0.25, x_o=0.25)
    assert np.isfinite(float(y)) and np.isfinite(float(a["theta"]))
    assert np.isfinite(float(M.binary_reject_composition(y, 10.0, 0.25)))


@pytest.mark.parametrize("pattern", ["complete_mixing", "cross_flow", "cocurrent"])
def test_permeation_stage_conserves(pattern):
    ret, perm, _ = M.permeation_stage(F2, Q2, 200.0, 1.0, 0.25, pattern, 20)
    np.testing.assert_allclose(ret + perm, F2, rtol=1e-8)


def test_large_area_does_not_nan():
    """Past the dry point the stages used to return NaN (also the countercurrent smoke)."""
    GPU = 3.348e-10
    Fi = 1000 / 22.414 / 3.6 * jnp.array([0.15, 0.85])
    Q = jnp.array([100.0, 5.0]) * GPU
    R, V, _ = M.permeation_stage(Fi, Q, 1e4, 13e5, 1.05e5, "countercurrent")
    assert np.all(np.isfinite(np.asarray(V))) and np.all(np.isfinite(np.asarray(R)))
    g = jax.grad(lambda x: M.local_permeate_composition(x, Q, 13e5, 1.05e5)[1])
    assert np.all(np.isfinite(g(jnp.zeros(2))))


def test_gas_permeator_smoke_and_validation():
    feed = make_stream({"CO2": 0.13, "N2": 0.87}, T=313.0, P=2e5)
    unit = M.GasPermeator(M.GasPermeationParams(
        permeances={"CO2": 1000 * M.GPU, "N2": 1000 / 30 * M.GPU}, area=50.0,
        flow_pattern="cross_flow", pressure_ratio=10.0, n_steps=20))
    ret, perm, info = unit(feed)
    assert float(ret["F_CO2"] + perm["F_CO2"]) == pytest.approx(0.13)
    with pytest.raises(ValueError):
        M.GasPermeationParams(permeances={}, area=1.0, flow_pattern="spiral")
    with pytest.raises(ValueError):
        M.GasPermeationParams(permeances={}, area=1.0, pressure_ratio=0.5)
    with pytest.raises(ValueError):
        M.GasPermeationParams(permeances={})


def test_reverse_osmosis_smoke():
    Qf = 100 / 3600
    c = 35 / 58.44 * 1000
    feed = make_stream({"H2O": Qf * 997 / 0.018015, "NaCl": Qf * c}, T=298.15, P=101325 + 60e5)
    ro = M.ReverseOsmosis(M.ReverseOsmosisParams(A_w=2.78e-12, B=1e-8, area=1e3, n_steps=20))
    conc, perm, info = ro(feed)
    assert float(conc["F_NaCl"] + perm["F_NaCl"]) == pytest.approx(Qf * c, rel=1e-8)
    assert np.isfinite(float(info["recovery"]))


def test_ultrafiltration_helpers_run():
    assert np.isfinite(float(M.uf_flux(5e5, 1e12, 1e-3, 1e-5, 10.0, 250.0)))
    assert np.isfinite(float(M.polarization_modulus(1e-5, 1e-5)))
    assert np.isfinite(float(M.batch_concentration_time(1.0, 0.1, 10.0, 1e-5, 5.0, 250.0)))


def test_bio_wall_concentration_regression():
    """A fully retained solute polarizes (R and 1 - R were swapped)."""
    from difflow_bio.units.filtration import concentration_polarization
    assert float(concentration_polarization(10.0, 1e-5, 1e-5, 1.0)) > 10.0


def test_grad_finite():
    f = lambda A: M.cross_flow(F2, Q2, A, 1.0, 0.25, 20)[1][0]  # noqa: E731
    g = lambda dP: M.ro_local_flux(dP, 600.0, 2.8e-12, 1e-8, k=3e-5)["J_v"]  # noqa: E731
    assert np.isfinite(float(jax.grad(f)(150.0)))
    assert np.isfinite(float(jax.grad(g)(6e6)))
