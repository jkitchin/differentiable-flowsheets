"""Chaining units into one differentiable function (#334): ``difflow_refinery.plant``.

Per commit (cheap, no unit solves): the chain machinery on toy stages that
have exactly the AD restrictions of the real units -- a ``custom_vjp``
(reverse only, as the hydrotreater's checkpointed bed adjoint) and a
``lax.while_loop`` (forward only, as the reformer's ``ForwardMode`` beds).
A mixed chain cannot be differentiated end to end in either mode, and the
chain rule by unit Jacobians gets it right; the AD-mode table names
objects that exist.

The cross-unit gradient on real units (naphtha hydrotreater -> fractionator
-> reformer, against central differences) is ``tests/refinery/test_plant_chain.py``
(release, slow).
"""

from __future__ import annotations

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow_refinery import plant
from difflow_refinery.plant import AD_MODES, Chain, Stage, ad_mode_table, central_difference, timed

jax.config.update("jax_enable_x64", True)


@jax.custom_vjp
def _rev_only(x):
    """``y = [x0 * x1, sin(x0)]``; reverse mode only."""
    return jnp.stack([x[0] * x[1], jnp.sin(x[0])])


def _rev_fwd(x):
    return _rev_only(x), x


def _rev_bwd(x, g):
    return (jnp.stack([g[0] * x[1] + g[1] * jnp.cos(x[0]), g[0] * x[0]]),)


_rev_only.defvjp(_rev_fwd, _rev_bwd)


def _fwd_only(y):
    """``z = sqrt(y0 + y1^2)`` by a Newton ``while_loop`` (forward mode only), and ``y0``."""
    a = y[0] + y[1] ** 2

    def body(c):
        z, k = c
        return 0.5 * (z + a / z), k + 1

    z, _ = jax.lax.while_loop(lambda c: c[1] < 40, body, (jnp.asarray(1.0) + 0.0 * a, 0))
    return {"z": z, "y0": y[0]}


def _exact(x):
    y0, y1 = x[0] * x[1], jnp.sin(x[0])
    return {"z": jnp.sqrt(y0 + y1 ** 2), "y0": y0}


X = jnp.asarray([0.7, 1.3])


@pytest.fixture
def mixed():
    return Chain(Stage("hdt-like", _rev_only, modes="rev"), Stage("reformer-like", _fwd_only, modes="fwd"))


def test_the_toy_stages_have_the_units_restrictions():
    with pytest.raises(TypeError):
        jax.jacfwd(_rev_only)(X)
    with pytest.raises(ValueError, match="while_loop"):
        jax.jacrev(lambda y: _fwd_only(y)["z"])(X)


def test_a_mixed_chain_cannot_be_traced_end_to_end_in_either_mode(mixed):
    assert mixed.modes == ()
    with pytest.raises(TypeError):
        jax.jacfwd(lambda x: mixed(x)["z"])(X)
    with pytest.raises(ValueError, match="while_loop"):
        jax.jacrev(lambda x: mixed(x)["z"])(X)
    with pytest.raises(ValueError, match="hdt-like"):
        mixed.jacobian(X, method="fwd")
    with pytest.raises(ValueError, match="reformer-like"):
        mixed.jacobian(X, method="rev")


def test_the_chain_rule_by_unit_jacobians_is_exact(mixed):
    res = mixed.jacobian(X)
    assert res.method == "chain"
    J_exact = np.asarray(jax.jacfwd(lambda x: jax.flatten_util.ravel_pytree(_exact(x))[0])(X))
    np.testing.assert_allclose(np.asarray(res.jacobian), J_exact, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(float(res.value["z"]), float(_exact(X)["z"]), rtol=1e-14)
    assert set(res.timings) == {"hdt-like", "reformer-like", "total"}
    # the output pytree comes back per input column
    col = res.column(0)
    assert set(col) == {"z", "y0"}
    assert float(col["y0"]) == pytest.approx(float(X[1]))
    # and the finite-difference check agrees
    fd = central_difference(mixed, X, 1e-6)
    np.testing.assert_allclose(np.asarray(res.jacobian), fd, rtol=1e-7, atol=1e-9)
    # one tangent at a time gives the same Jacobian
    seq = mixed.jacobian(X, vectorize=False)
    np.testing.assert_allclose(np.asarray(seq.jacobian), np.asarray(res.jacobian), rtol=1e-14)
    np.testing.assert_allclose(float(seq.value["z"]), float(res.value["z"]), rtol=1e-14)


def test_a_chain_of_one_mode_traces_end_to_end():
    fwd = Chain(Stage("a", lambda x: jnp.stack([x[0] * x[1], jnp.sin(x[0])]), modes="fwd"),
                Stage("b", _fwd_only, modes="fwd"))
    res = fwd.jacobian(X)
    assert res.method == "fwd"
    ref = Chain(Stage("a", _rev_only, modes="rev"), Stage("b", _fwd_only, modes="fwd")).jacobian(X)
    np.testing.assert_allclose(np.asarray(res.jacobian), np.asarray(ref.jacobian), rtol=1e-12)
    seq = fwd.jacobian(X, vectorize=False)
    np.testing.assert_allclose(np.asarray(seq.jacobian), np.asarray(res.jacobian), rtol=1e-14)
    assert float(seq.value["z"]) == pytest.approx(float(res.value["z"]), rel=1e-14)
    rev = Chain(Stage("a", _rev_only, modes="rev"), Stage("b", lambda y: jnp.sum(y ** 2), modes="rev"))
    r = rev.jacobian(X)
    assert r.method == "rev" and r.jacobian.shape == (1, 2)
    np.testing.assert_allclose(np.asarray(rev.jacobian(X, vectorize=False).jacobian), np.asarray(r.jacobian),
                               rtol=1e-14)


def test_auto_chooses_by_shape_when_every_stage_has_both_modes():
    many_out = Chain(Stage("s", lambda x: jnp.outer(x, x).ravel()))
    many_in = Chain(Stage("s", lambda x: jnp.sum(x ** 3)))
    assert many_out.jacobian(X).method == "fwd"
    assert many_in.jacobian(jnp.arange(5.0)).method == "rev"
    assert many_in.jacobian(jnp.arange(5.0), method="chain").method == "chain"
    np.testing.assert_allclose(np.asarray(many_in.jacobian(jnp.arange(5.0), method="chain").jacobian),
                               3.0 * np.arange(5.0)[None, :] ** 2)


def test_bad_stages_and_methods_say_what_is_wrong():
    with pytest.raises(ValueError, match="subset"):
        Stage("x", lambda v: v, modes=("sideways",))
    with pytest.raises(ValueError, match="subset"):
        Stage("x", lambda v: v, modes=())
    with pytest.raises(ValueError, match="unique"):
        Chain(Stage("x", lambda v: v), Stage("x", lambda v: v))
    with pytest.raises(ValueError, match="at least one"):
        Chain()
    with pytest.raises(ValueError, match="unknown method"):
        Chain(Stage("x", lambda v: v)).jacobian(X, method="adjoint")
    assert Stage("x", lambda v: v, modes=("rev", "fwd")).modes == ("fwd", "rev")


def test_timed_reports_each_call():
    f = jax.jit(lambda x: jnp.sin(x) * 2.0)
    out, ts = timed(f, X, repeat=3)
    assert len(ts) == 3 and all(t >= 0.0 for t in ts)
    np.testing.assert_allclose(np.asarray(out), 2.0 * np.sin(np.asarray(X)))


def _resolve(path: str):
    obj = importlib.import_module("difflow_refinery")
    for part in path.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            obj = importlib.import_module(f"{obj.__name__}.{part}")
    return obj


def test_the_ad_mode_table_names_real_objects():
    assert {"hydrotreater", "catalytic reformer", "hydrocracker", "FCC", "gas plant columns",
            "residue desulfurizer", "crude unit"} <= set(AD_MODES)
    for m in AD_MODES.values():
        assert m.modes and set(m.modes) <= set(plant.MODES), m.name
        for part in m.obj.split(","):
            for path in part.split("/"):
                _resolve(path.strip())
    assert AD_MODES["catalytic reformer"].modes == ("fwd",)
    assert AD_MODES["hydrotreater"].modes == ("rev",)
    assert AD_MODES["hydrocracker"].modes == ("rev",)
    assert AD_MODES["FCC"].modes == ("fwd", "rev")
    md = ad_mode_table()
    assert md.count("\n") == len(AD_MODES) + 1 and "| catalytic reformer |" in md
    assert "hydrotreater: rev" in ad_mode_table(markdown=False)


def test_a_jitted_stage_gives_the_same_jacobian(mixed):
    jitted = Chain(Stage("hdt-like", _rev_only, modes="rev", jit=True),
                   Stage("reformer-like", _fwd_only, modes="fwd", jit=True))
    for vec in (True, False):
        np.testing.assert_allclose(np.asarray(jitted.jacobian(X, vectorize=vec).jacobian),
                                   np.asarray(mixed.jacobian(X).jacobian), rtol=1e-13)
