"""Tests for delta attribution from plant data (difflow.planning.attribution).

Every case is synthetic: a model block, a "plant" that differs from it in a
known way, and a sampled operating history.  The checks are the ones the
module's docstring promises -- a level error and an excited slope error are
recovered, an input that never moved is reported as not estimable, collinear
inputs are reported as a combination, autocorrelated noise does not produce
a stream of flags, and the estimates round-trip into the planner.
"""

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from difflow.planning import (
    Block, DeltaBasePlanner, Network, attribute_deltas,
)

MOVE = [1.0, 1.0, 1.0]


def model_fn(u):
    return {"yield": 2.0 * u[0] + 1.0 * u[1] + 0.5 * u[2],
            "purity": 0.9 * jnp.exp(-0.05 * u[0])}


def make_block():
    return Block(name="b", fn=model_fn, u_names=["x0", "x1", "x2"],
                 y_names=["yield", "purity"],
                 lb=[0.0, 0.0, 0.0], ub=[10.0, 10.0, 10.0])


def history(seed, n=200, x2=5.0):
    rng = np.random.default_rng(seed)
    u = np.column_stack([rng.uniform(2, 8, n), rng.uniform(2, 8, n),
                         np.full(n, x2)])
    return u, np.arange(n, dtype=float)


def model_yield(u):
    return 2.0 * u[:, 0] + u[:, 1] + 0.5 * u[:, 2]


def test_level_and_excited_slope_are_recovered():
    blk = make_block()
    u, t = history(0)
    rng = np.random.default_rng(1)
    y = model_yield(u) + 0.8 + 0.3 * u[:, 0] + rng.normal(0, 0.05, len(t))
    res = attribute_deltas(blk, u, {"yield": y}, sigma_y={"yield": 0.05},
                           t=t, move=MOVE)
    o = res.outputs["yield"]

    x0 = o.coefficient("x0")
    assert x0.status == "flag"
    npt.assert_allclose(x0.estimate, 0.3, atol=0.02)
    # The level refers to u_ref (the mean input), so it includes the slope
    # error evaluated there.
    lev = o.coefficient("level")
    assert lev.status == "flag"
    npt.assert_allclose(lev.estimate, 0.8 + 0.3 * res.u_ref[0], atol=0.03)
    assert o.coefficient("x1").status == "ok"
    assert not o.structural
    assert res.unobserved == ["purity"]


def test_input_that_never_moved_is_not_estimable():
    blk = make_block()
    u, t = history(2)
    rng = np.random.default_rng(3)
    # A large slope error on x2 -- invisible, because x2 was held still.
    y = model_yield(u) + 2.0 * (u[:, 2] - 5.0) + rng.normal(0, 0.05, len(t))
    res = attribute_deltas(blk, u, {"yield": y}, sigma_y=0.05, t=t, move=MOVE)
    x2 = res.outputs["yield"].coefficient("x2")
    assert not x2.estimable
    assert x2.status == "not estimable"
    assert np.isnan(x2.estimate)
    assert "x2" in res.outputs["yield"].not_estimable
    assert "not estimable" in res.table()


def test_collinear_inputs_are_reported_as_a_combination():
    blk = make_block()
    u, t = history(4)
    rng = np.random.default_rng(5)
    u[:, 1] = u[:, 0] + rng.normal(0, 1e-4, len(t))   # one controller, two valves
    y = model_yield(u) + 0.4 * (u[:, 1] - 5.0) + rng.normal(0, 0.05, len(t))
    res = attribute_deltas(blk, u, {"yield": y}, sigma_y=0.05, t=t, move=MOVE)
    o = res.outputs["yield"]
    kept = [c for c in o.coefficients if c.term in ("x0", "x1") and c.estimable]
    held = [c for c in o.coefficients if c.term in ("x0", "x1")
            and not c.estimable]
    assert len(kept) == 1 and len(held) == 1
    k = kept[0]
    assert k.status == "combination"
    npt.assert_allclose(k.aliases[held[0].term], 1.0, atol=1e-2)
    # The estimate is the combined slope error, whichever column was kept.
    npt.assert_allclose(k.estimate, 0.4, atol=0.03)


def test_inflation_suppresses_false_flags_under_ar1_noise():
    blk = make_block()
    raw_flags = inflated_flags = 0
    for seed in range(12):
        u, t = history(100 + seed)
        rng = np.random.default_rng(200 + seed)
        e = np.zeros(len(t))
        for i in range(1, len(t)):
            e[i] = 0.95 * e[i - 1] + rng.normal(0, 0.05)
        y = model_yield(u) + e          # no model error at all
        res = attribute_deltas(blk, u, {"yield": y}, sigma_y=0.05, t=t,
                               move=MOVE)
        o = res.outputs["yield"]
        for c in o.coefficients:
            if not c.estimable:
                continue
            big = abs(c.per_move) > 0.05
            raw_flags += big and abs(c.z * o.inflation) > 3
            inflated_flags += c.status == "flag"
    assert raw_flags >= 10
    assert inflated_flags <= 1


def test_input_noise_is_not_mistaken_for_a_gradient_error():
    blk = make_block()
    u_true, t = history(6)
    rng = np.random.default_rng(7)
    y = model_yield(u_true) + rng.normal(0, 0.02, len(t))
    u_meas = u_true + np.column_stack([rng.normal(0, 0.2, len(t)),
                                       np.zeros(len(t)), np.zeros(len(t))])
    naive = attribute_deltas(blk, u_meas, {"yield": y}, sigma_y=0.02, t=t,
                             move=MOVE)
    eiv = attribute_deltas(blk, u_meas, {"yield": y}, sigma_y=0.02, t=t,
                           move=MOVE, sigma_u={"x0": 0.2})
    # Attenuation makes the measured-input fit see a slope error on x0 ...
    assert naive.outputs["yield"].coefficient("x0").status == "flag"
    # ... which the propagated input noise correctly discounts.
    assert eiv.outputs["yield"].coefficient("x0").status != "flag"


def test_residual_along_an_unresolved_direction_is_structural():
    blk = make_block()
    u, t = history(8)
    n = len(t)
    u[n // 2:, 2] += 1e-3          # a barely-visible change in x2 ...
    rng = np.random.default_rng(9)
    y = model_yield(u) + rng.normal(0, 0.05, n)
    y[n // 2:] += 1.0               # ... coinciding with a regime change
    res = attribute_deltas(blk, u, {"yield": y}, sigma_y=0.05, t=t, move=MOVE)
    o = res.outputs["yield"]
    assert not o.coefficient("x2").estimable
    assert o.structural
    assert "STRUCTURAL" in res.table()


def test_log_output_modifiers_reproduce_the_plant_at_u_ref():
    blk = make_block()
    u, t = history(10)
    rng = np.random.default_rng(11)
    purity_model = 0.9 * np.exp(-0.05 * u[:, 0])
    # Relative error: the plant runs 10% richer and loses 0.02 more per x0.
    purity = purity_model * np.exp(0.1 - 0.02 * (u[:, 0] - 5.0))
    purity *= np.exp(rng.normal(0, 0.005, len(t)))
    res = attribute_deltas(blk, u, {"purity": purity}, sigma_y=0.005, t=t,
                           move=MOVE, log_outputs={"purity": 0.0})
    o = res.outputs["purity"]
    npt.assert_allclose(o.coefficient("x0").estimate, -0.02, atol=2e-3)

    mods = res.to_modifiers()
    k = blk.y_index("purity")
    u_ref = jnp.asarray(res.u_ref)
    plant_at = lambda x: 0.9 * np.exp(-0.05 * x[0]) * np.exp(  # noqa: E731
        0.1 - 0.02 * (x[0] - 5.0))
    npt.assert_allclose(float(mods.correct(blk, u_ref)[k]),
                        plant_at(np.asarray(u_ref)), rtol=2e-3)
    # Unobserved outputs are left alone.
    assert float(mods.eps[blk.y_index("yield")]) == 0.0
    assert not np.any(np.asarray(mods.lam[blk.y_index("yield")]))


def test_flagged_only_drops_insignificant_terms():
    blk = make_block()
    u, t = history(12)
    rng = np.random.default_rng(13)
    y = model_yield(u) + 0.8 + rng.normal(0, 0.05, len(t))
    res = attribute_deltas(blk, u, {"yield": y}, sigma_y=0.05, t=t, move=MOVE)
    k = blk.y_index("yield")
    strict = res.to_modifiers()
    loose = res.to_modifiers(flagged_only=False)
    npt.assert_allclose(float(strict.eps[k]), 0.8, atol=0.02)
    assert not np.any(np.asarray(strict.lam[k]))
    assert np.any(np.asarray(loose.lam[k]))


def test_modifiers_and_exposure_through_the_planner():
    blk = make_block()
    u, t = history(14)
    rng = np.random.default_rng(15)
    y = model_yield(u) + 0.8 + rng.normal(0, 0.05, len(t))
    res = attribute_deltas(blk, u, {"yield": y}, sigma_y=0.05, t=t, move=MOVE)

    planner = DeltaBasePlanner(Network([blk]), prices={"b.yield": 3.0},
                               modifiers={"b": res.to_modifiers()},
                               radius=0.3)
    plan = planner.solve()
    assert np.all(np.asarray(plan.decisions) <= 10.0 + 1e-9)
    ex = res.exposure(plan)
    if not ex:
        pytest.skip("solver reported no marginals")
    (row,) = ex
    assert row["output"] == "yield"
    npt.assert_allclose(abs(row["dual"]), 3.0, rtol=1e-6)
    npt.assert_allclose(row["exposure"], 3.0 * abs(row["eps"]), rtol=1e-9)


def test_input_validation():
    blk = make_block()
    u, t = history(16, n=10)
    with pytest.raises(KeyError):
        attribute_deltas(blk, u, {"nope": np.zeros(10)}, sigma_y=1.0)
    with pytest.raises(ValueError):
        attribute_deltas(blk, u[:, :2], {"yield": np.zeros(10)}, sigma_y=1.0)
    unbounded = Block(name="c", fn=lambda x: {"y": x[0]}, u_names=["x"],
                      y_names=["y"])
    with pytest.raises(ValueError, match="move"):
        attribute_deltas(unbounded, u[:, :1], {"y": np.zeros(10)},
                         sigma_y=1.0)
