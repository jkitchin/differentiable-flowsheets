"""difflow_refinery.implicit: one trace of the model, implicit derivatives."""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow_refinery.implicit import hashable, merge_params, solution, split_params

jax.config.update("jax_enable_x64", True)


def _fixed_point_problem():
    calls = {"n": 0}

    def G(x, th):
        calls["n"] += 1
        a, b = th["a"], th["b"]
        g = jnp.stack([0.5 * jnp.cos(x[1]) + a, 0.3 * jnp.sin(x[0]) + b])
        return g, {"s": jnp.sum(g) * a, "ok": jnp.all(jnp.isfinite(g))}

    def solve(th):
        th0 = jax.tree_util.tree_map(jax.lax.stop_gradient, th)
        x = jnp.zeros(2)
        for _ in range(80):              # converged on stop-gradient inputs
            x = G(x, th0)[0]
        return solution(G, x, th, fixed_point=True)

    return G, solve, calls


def test_traces_once_and_derivatives_match_fd():
    G, solve, calls = _fixed_point_problem()
    th = {"a": jnp.asarray(0.2), "b": jnp.asarray(-0.1)}

    def f(a):
        x, aux = solve(dict(th, a=a))
        return x[0] + aux["s"]

    calls["n"] = 0
    jax.make_jaxpr(lambda a: solve(dict(th, a=a)))(0.2)
    loop = 80
    assert calls["n"] == loop + 1                      # value: one trace of G
    calls["n"] = 0
    jax.make_jaxpr(jax.jacfwd(f))(0.2)
    assert calls["n"] == loop + 1                      # derivative: one linearisation
    g_fwd, g_rev = jax.jacfwd(f)(0.2), jax.grad(f)(0.2)
    h = 1e-6
    fd = (f(0.2 + h) - f(0.2 - h)) / (2 * h)
    assert float(g_fwd) == pytest.approx(float(fd), rel=1e-7)
    assert float(g_rev) == pytest.approx(float(fd), rel=1e-7)


def test_root_form():
    def G(x, th):
        return x ** 3 - th, {"x2": x ** 2}

    th = jnp.asarray(8.0)
    x, aux = solution(G, jnp.array([2.0]), th, fixed_point=False)
    assert float(x[0]) == 2.0
    d = jax.grad(lambda t: solution(G, jnp.cbrt(t)[None], t, fixed_point=False)[1]["x2"][0])(th)
    assert float(d) == pytest.approx(2.0 * 2.0 / (3.0 * 4.0), rel=1e-12)   # d(x^2)/dt = 2x / (3x^2)


@dataclasses.dataclass(frozen=True)
class _P:
    T: object
    n: int = 3
    inner: tuple = ()


def test_split_and_merge_params():
    p = _P(T=jnp.asarray(500.0), inner=(_P(T=1.0), _P(T=np.array(2.0))))
    skel, arrays = split_params(p)
    assert len(arrays) == 2 and hashable(skel)
    back = merge_params(skel, arrays)
    assert float(back.T) == 500.0 and float(back.inner[1].T) == 2.0 and back.inner[0].T == 1.0
    assert split_params(_P(T=1.0))[1] == []
