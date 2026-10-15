"""Tests for difflow.operability: smoke, API contracts and validation."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from difflow.operability import (
    OperabilityWarning,
    Scaling,
    condition_number,
    disturbance_gain,
    gain_matrix,
    min_singular_value,
    pinv,
    rga,
    screen,
)

UNIT = Scaling(u_span=[1.0, 1.0], y_span=[1.0, 1.0])


def test_rga_2x2():
    R = np.asarray(rga(jnp.array([[1.0, 2.0], [3.0, 4.0]])))
    np.testing.assert_allclose(R, [[-2.0, 3.0], [3.0, -2.0]], atol=1e-10)


def test_screen_flags_negative_pairing():
    rep = screen(jnp.array([[1.0, 2.0], [3.0, 4.0]]), scaling=UNIT)
    assert "rga_negative" in [f.kind for f in rep.findings]


def test_singular_plant_is_diagnosed():
    rep = screen(jnp.array([[1.0, 2.0], [2.0, 4.0]]), scaling=UNIT)
    assert rep.singular
    assert "singular" in [f.kind for f in rep.findings]


def test_pinv_of_exactly_singular_matrix_is_finite():
    """Regression: a zero singular value must not give nan."""
    assert np.all(np.isfinite(np.asarray(pinv(jnp.array([[1.0, 0.0], [0.0, 0.0]])))))
    assert np.all(np.isfinite(np.asarray(pinv(jnp.zeros((2, 2))))))


def test_gain_matrix_and_screen_smoke():
    def plant(u, d):
        return jnp.array([u[0] + 0.5 * u[1] + d[0], 0.3 * u[0] + u[1] ** 2])

    u0, d0 = jnp.array([0.3, 0.4]), jnp.array([0.1])
    G = gain_matrix(plant, u0, d0, scaling=UNIT)
    Gd = disturbance_gain(plant, u0, d0)
    assert G.shape == (2, 2) and Gd.shape == (2, 1)
    sc = Scaling(u_span=[1.0, 1.0], y_span=[1.0, 1.0], d_span=[1.0])
    rep = screen(plant, u0, d0, scaling=sc)
    assert np.isfinite(float(rep.msv))
    assert isinstance(rep.summary(), str)


def test_unscaled_use_is_caveated():
    G = jnp.array([[1.0, 0.0], [0.0, 1e5]])
    with pytest.warns(OperabilityWarning, match="no Scaling"):
        min_singular_value(G)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        min_singular_value(G, assume_scaled=True)
    with pytest.raises(TypeError, match="scaling"):
        screen(G)
    with pytest.raises(TypeError, match="requires a Scaling"):
        screen(G, scaling={"u_span": [1.0, 1.0], "y_span": [1.0, 1.0]})
    with pytest.warns(OperabilityWarning, match="UNSCALED"):
        rep = screen(G, scaling=Scaling.unscaled(2, 2))
    assert rep.summary().startswith("!! UNSCALED")


def test_scaling_validation_and_constructors():
    with pytest.raises(ValueError, match="strictly positive"):
        Scaling(u_span=[1.0, 0.0], y_span=[1.0])
    with pytest.raises(ValueError, match="finite"):
        Scaling(u_span=[1.0, np.inf], y_span=[1.0])
    sc = Scaling(u_span=[1.0, 2.0], y_span=[1.0])
    with pytest.raises(ValueError, match="Scaling is for"):
        sc.scale_gain(jnp.ones((2, 2)))
    with pytest.raises(ValueError, match="no d_span"):
        sc.scale_disturbance(jnp.ones((1, 1)))
    fb = Scaling.from_bounds([0.0, 300.0], [1.0, 400.0],
                             y_tol=[0.01], d_lb=[0.0], d_ub=[5.0])
    assert fb.explicit and fb.u_span.shape == (2,)


def test_scaling_from_block():
    from difflow.planning import Block

    blk = Block(name="sep", fn=lambda u: jnp.array([u[0] * u[1]]),
                u_names=["recovery", "split"], y_names=["product"],
                lb=[0.0, 0.2], ub=[1.0, 0.8])
    assert Scaling.from_block(blk, y_tol=[0.05]).u_span.shape == (2,)
    unbounded = Block(name="open", fn=lambda u: jnp.array([u[0]]),
                      u_names=["x"], y_names=["y"])
    with pytest.raises(ValueError, match="unbounded"):
        Scaling.from_block(unbounded, y_tol=[1.0])


def test_metrics_jit_and_vmap():
    def metrics(u):
        G = gain_matrix(lambda x: jnp.array([x[0] + 0.9 * x[1],
                                             0.9 * x[0] + x[1] ** 2]), u,
                        scaling=UNIT)
        return jnp.array([min_singular_value(G, assume_scaled=True),
                          condition_number(G, assume_scaled=True)])

    out = jax.jit(jax.vmap(metrics))(jnp.array([[0.3, 0.4], [1.0, 2.0]]))
    assert out.shape == (2, 2) and np.all(np.isfinite(np.asarray(out)))


def test_msv_grad_is_finite():
    def f(a):
        return screen(lambda u: jnp.array([u[0] + a * u[1], a * u[0] + u[1]]),
                      jnp.zeros(2), scaling=UNIT).msv

    assert np.isfinite(float(jax.grad(f)(0.4)))


def test_report_is_a_pytree():
    rep = screen(jnp.array([[1.0, 0.2], [0.3, 1.0]]), scaling=UNIT,
                 u_names=["a", "b"])
    leaves, treedef = jax.tree_util.tree_flatten(rep)
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert back.u_names == ("a", "b")


def test_findings_refuse_tracers():
    @jax.jit
    def bad(u):
        return len(screen(lambda x: jnp.array([x[0], x[1]]), u,
                          scaling=UNIT).findings)

    with pytest.raises(ValueError, match="tracers"):
        bad(jnp.zeros(2))


def test_screen_rejects_inconsistent_arguments():
    G = jnp.eye(2)
    with pytest.raises(ValueError, match="expected 2 u names"):
        screen(G, scaling=UNIT, u_names=["only_one"])
    with pytest.raises(ValueError, match="pairing"):
        screen(G, scaling=UNIT, pairing=[0, 5])
    with pytest.raises(ValueError, match="Gd is for the matrix form"):
        screen(lambda u: u, jnp.zeros(2), scaling=UNIT, Gd=jnp.zeros((2, 1)))
    with pytest.raises(ValueError, match="both u0 and d0"):
        disturbance_gain(lambda u, d: u, jnp.zeros(2))
