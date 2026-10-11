"""Orthogonal collocation helper (spike for #440): nodes, weights, analytic ODEs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow.dynamic.collocation import (
    Collocation, collocation_residual, differentiation_matrix, graded_edges, interpolate,
    radau_nodes, radau_weights)

jax.config.update("jax_enable_x64", True)


def _newton(res, Y0, n=12):
    Y = Y0
    flat = lambda y: res(y.reshape(Y0.shape)).ravel()
    for _ in range(n):
        r = flat(Y.ravel())
        J = jax.jacfwd(flat)(Y.ravel())
        Y = (Y.ravel() - jnp.linalg.solve(J, r)).reshape(Y0.shape)
    return Y


def test_radau_nodes_known_values():
    np.testing.assert_allclose(radau_nodes(1), [1.0])
    np.testing.assert_allclose(radau_nodes(2), [1 / 3, 1.0], atol=1e-14)
    s6 = np.sqrt(6.0)
    np.testing.assert_allclose(radau_nodes(3), [(4 - s6) / 10, (4 + s6) / 10, 1.0], atol=1e-14)


@pytest.mark.parametrize("K", [2, 3, 4, 5])
def test_differentiation_matrix_is_exact_on_polynomials(K):
    tau = radau_nodes(K)
    nodes = np.concatenate([[0.0], tau])
    D = differentiation_matrix(tau)
    for p in range(K + 1):                      # degree <= K is in the Lagrange space
        np.testing.assert_allclose(D @ nodes**p, p * tau ** max(p - 1, 0) if p else 0 * tau, atol=1e-11)


@pytest.mark.parametrize("K", [2, 3, 4])
def test_weights_integrate_to_degree_2K_minus_2(K):
    tau = radau_nodes(K)
    w = radau_weights(tau)
    for p in range(2 * K - 1):
        assert float(w @ tau**p) == pytest.approx(1.0 / (p + 1), abs=1e-12)


def test_graded_edges():
    e = graded_edges(4, 2.0)
    assert e[0] == 0.0 and e[-1] == 1.0
    np.testing.assert_allclose(np.diff(e)[1:] / np.diff(e)[:-1], 2.0)


def test_exponential_decay_matches_analytic():
    m = Collocation.make(4, K=3)
    f = lambda t, y: -y
    y0 = jnp.array([1.0])
    Y = _newton(lambda Y: collocation_residual(f, y0, Y, m), jnp.ones((4, 3, 1)))
    assert float(Y[-1, -1, 0]) == pytest.approx(np.exp(-1.0), rel=1e-6)
    # interior: collocation points (stage order K) are less accurate than the ends
    np.testing.assert_allclose(np.asarray(Y[..., 0]), np.exp(-m.t), rtol=1e-4)


def test_endpoint_order_is_2K_minus_1():
    f = lambda t, y: jnp.stack([y[1], -y[0]])      # harmonic oscillator
    y0 = jnp.array([0.0, 1.0])
    errs = []
    for ne in (2, 4):
        m = Collocation.make(ne, K=3)
        Y = _newton(lambda Y: collocation_residual(f, y0, Y, m), jnp.ones((ne, 3, 2)))
        errs.append(abs(float(Y[-1, -1, 0]) - np.sin(1.0)))
    order = np.log(errs[0] / errs[1]) / np.log(2.0)
    assert order > 4.5            # theory 5


def test_stiff_problem_is_l_stable_and_tracks_slow_manifold():
    # y' = -1000 (y - cos t) - sin t, exact y = cos t for y0 = 1; element width >> 1/1000
    f = lambda t, y: -1000.0 * (y - jnp.cos(t)) - jnp.sin(t)
    m = Collocation.make(4, K=3)
    Y = _newton(lambda Y: collocation_residual(f, jnp.array([1.0]), Y, m), jnp.ones((4, 3, 1)))
    assert float(Y[-1, -1, 0]) == pytest.approx(np.cos(1.0), abs=1e-6)


def test_graded_mesh_resolves_an_inlet_layer():
    # y' = -200 (y - 1), y0 = 0: boundary layer of thickness 1/200 at t = 0
    f = lambda t, y: -200.0 * (y - 1.0)
    y0 = jnp.array([0.0])
    err = {}
    for g in (1.0, 2.0):
        m = Collocation.make(6, K=3, grading=g)
        Y = _newton(lambda Y: collocation_residual(f, y0, Y, m), jnp.ones((6, 3, 1)))
        s = np.array([0.005, 0.01, 0.02])
        yi = interpolate(y0, Y, m, s)
        err[g] = float(jnp.max(jnp.abs(yi[:, 0] - (1 - np.exp(-200 * s)))))
    assert err[2.0] < err[1.0]


def test_residual_is_differentiable_in_parameters():
    m = Collocation.make(3, K=3)
    f = lambda t, y, a: -a * y
    y0 = jnp.array([1.0])

    def y_end(a):
        Y = _newton(lambda Y: collocation_residual(f, y0, Y, m, (a,)), jnp.ones((3, 3, 1)))
        return Y[-1, -1, 0]

    g = jax.grad(y_end)(2.0)
    assert float(g) == pytest.approx(-np.exp(-2.0), rel=1e-4)
