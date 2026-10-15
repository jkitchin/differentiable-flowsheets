"""Smoke tests for the distillation unit operations and design functions."""

import jax
import jax.numpy as jnp
import pytest

from difflow import IdealThermo, SpeciesData, get_flows, make_stream
from difflow.units.distillation import (
    DistillationColumn,
    DistillationColumnParams,
    ShortcutColumn,
    ShortcutColumnParams,
    column_diameter,
    fenske_stages,
    gilliland_stages,
    minimum_reflux_ratio,
    underwood_class2,
)

jax.config.update("jax_enable_x64", True)

SPECIES = ["benzene", "toluene"]


@pytest.fixture(scope="module")
def thermo():
    return IdealThermo({
        "benzene": SpeciesData(name="benzene", MW=78.11, Cp_coeffs=(136.0, 0.0, 0.0, 0.0),
                               Hvap_coeffs=(33900.0, 0.38, 562.0),
                               antoine_coeffs=(13.82, 2788.0, -52.36)),
        "toluene": SpeciesData(name="toluene", MW=92.14, Cp_coeffs=(157.0, 0.0, 0.0, 0.0),
                               Hvap_coeffs=(38000.0, 0.38, 591.8),
                               antoine_coeffs=(13.93, 3096.0, -53.67)),
    })


def _feed():
    return make_stream({"benzene": 50.0, "toluene": 50.0}, T=370.0, P=101325.0)


def _column(thermo, **kw):
    kw = {"n_stages": 8, "feed_stage": 4, "P": 101325.0, **kw}
    return DistillationColumn(DistillationColumnParams(species_order=SPECIES, **kw), thermo)


def _closes(feed, d, b):
    f, fd, fb = get_flows(feed), get_flows(d), get_flows(b)
    for s in SPECIES:
        assert float(fd[s] + fb[s]) == pytest.approx(float(f[s]), rel=1e-6)


def test_shortcut_column(thermo):
    col = ShortcutColumn(ShortcutColumnParams(species_order=SPECIES, light_key="benzene",
                                              heavy_key="toluene"), thermo)
    d, b, info = col(_feed(), R=2.0, P=101325.0)
    assert {"N", "N_min", "R_min"} <= set(info)
    assert jnp.isfinite(info["N"])
    _closes(_feed(), d, b)
    g = jax.grad(lambda R: col(_feed(), R=R)[2]["N"])(2.0)
    assert jnp.isfinite(g)


def test_design_functions():
    assert jnp.isfinite(fenske_stages(0.95, 0.05, 2.5))
    assert jnp.isfinite(minimum_reflux_ratio(0.5, 0.5, 0.95, 2.5))
    assert jnp.isfinite(gilliland_stages(2.0, 1.0, 10.0))
    assert jnp.isfinite(gilliland_stages(2.0, 1.0, 10.0, method="molokanov"))
    with pytest.raises(ValueError):
        gilliland_stages(1.0, 0.5, 5.0, method="nope")
    with pytest.warns(DeprecationWarning):
        D = column_diameter(100.0, 3.0, 800.0, 0.02, 0.6, MW_V=80.0, L=100.0, MW_L=85.0)
    assert jnp.isfinite(D)


def test_multicomponent_shortcut_functions():
    from difflow import (
        fenske_distribution,
        fenske_stages_multicomponent,
        kirkbride_feed_stage,
        underwood_minimum_reflux,
    )

    K = jnp.array([3.12, 1.38, 0.60, 0.28])
    z = jnp.array([0.40, 0.25, 0.20, 0.15])
    x_D = jnp.array([0.62, 0.35, 0.03, 0.0])
    assert jnp.isfinite(fenske_stages_multicomponent(0.35, 0.03, 0.07, 0.5, 2.258))
    R_min, theta = underwood_minimum_reflux(K / K[2], z, x_D, 1.0)
    assert jnp.isfinite(R_min) and jnp.isfinite(theta)
    N_R, N_S = kirkbride_feed_stage(11.0, 0.25, 0.2, 0.07, 0.03, 35.0, 65.0)
    assert float(N_R + N_S) == pytest.approx(11.0)
    d, b, _ = fenske_distribution(100 * z, K, 1, 2, 0.9, 0.9)
    assert jnp.allclose(d + b, 100 * z)


def test_underwood_class2():
    alpha = jnp.array([4.0, 2.0, 1.5, 1.0])
    f = jnp.array([30.0, 30.0, 10.0, 30.0])
    R, d, th = underwood_class2(alpha, f, 29.0, 1.0, light_key=1, heavy_key=3, q=1.0)
    assert jnp.isfinite(R) and th.shape == (2,)
    with pytest.raises(ValueError, match="distributing"):
        underwood_class2(alpha, f, 29.0, 1.0, 0, 2, distributing=[])


@pytest.mark.parametrize("use_mesh", [False, True])
def test_rigorous_column(thermo, use_mesh):
    d, b, info = _column(thermo)(_feed(), R=2.0, D_spec=50.0, use_mesh=use_mesh,
                                 mesh_iter=20, cmo_iter=100)
    assert {"T_profile", "x_profile", "Q_condenser", "Q_reboiler"} <= set(info)
    assert jnp.all(jnp.isfinite(info["T_profile"]))
    assert float(info["balance_error_rel"]) < 0.01


def test_rigorous_column_gradient(thermo):
    col = _column(thermo)

    def x_D(R):
        d, _, _ = col(_feed(), R=R, D_spec=50.0, use_mesh=False, cmo_iter=15)
        return d["F_benzene"]

    assert jnp.isfinite(jax.grad(x_D)(2.0))


def test_partial_condenser_and_murphree_efficiency(thermo):
    col = _column(thermo, condenser_type="partial", stage_efficiency=0.7)
    d, b, info = col(_feed(), R=2.0, D_spec=50.0, use_mesh=False, cmo_iter=100)
    assert info["y_D"].shape == (2,)
    assert int(info["n_equilibrium_stages"]) == 9
    assert jnp.all(jnp.isfinite(info["x_profile"]))


def test_column_params_validation():
    kw = dict(species_order=SPECIES, n_stages=4, feed_stage=2)
    with pytest.raises(NotImplementedError, match="mixed"):
        DistillationColumnParams(**kw, condenser_type="mixed")
    with pytest.raises(ValueError):
        DistillationColumnParams(**kw, condenser_type="bogus")
    with pytest.raises(ValueError):
        DistillationColumnParams(**kw, stage_efficiency=-0.1)


def test_solve_specs(thermo):
    col = _column(thermo)
    _, _, info = col.solve_specs(_feed(), [("x_D", "benzene", 0.9), ("D", 50.0)],
                                 use_mesh=False)
    assert jnp.isfinite(info["R"])
    with pytest.raises(ValueError, match="exactly two"):
        col.solve_specs(_feed(), [("R", 2.0)])
    with pytest.raises(ValueError, match="unknown specification"):
        col.solve_specs(_feed(), [("R", 2.0), ("purity", "benzene", 0.9)])
