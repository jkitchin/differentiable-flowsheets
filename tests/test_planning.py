"""Tests for difflow.planning: smoke, API contracts and input validation."""

import warnings
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from difflow.planning import (
    Block,
    CostRatio,
    DeltaBasePlanner,
    DeltaHealthWarning,
    Network,
    PhaseBoundaryWarning,
    PiecewiseSpec,
    apply_backoff,
    build_lp,
    check_delta_health,
    check_phase_transition,
    choose_ad_mode,
    classify_phase,
    constraint_backoff,
    linearize_block,
    price_switch_point,
    run_modifier_adaptation,
    sample_piecewise,
    scaled_jacobian,
)
from difflow.planning import benchmark as benchmark_mod


def quadratic_block(name="q", peak=(0.3, 0.7)):
    a, b = peak
    return Block(name=name, fn=lambda u: jnp.array([-(u[0] - a) ** 2 - (u[1] - b) ** 2]),
                 u_names=["x", "y"], y_names=["f"], lb=[0.0, 0.0], ub=[1.0, 1.0])


def smooth_block(name="s"):
    def fn(u):
        return jnp.array([
            jnp.sin(3.0 * u[0]) * jnp.exp(-0.4 * u[1]) + 0.25 * u[2] ** 2,
            jnp.log1p(u[0] * u[1]) + jnp.tanh(u[2]),
        ])

    return Block(name=name, fn=fn, u_names=["a", "b", "c"],
                 y_names=["p", "q"], lb=[0.1, 0.1, 0.1], ub=[1.5, 1.5, 1.5])


def chain():
    a = Block(name="up", fn=lambda u: jnp.array([u[0] * u[1], 1 - u[0]]),
              u_names=["r", "s"], y_names=["prod", "residue"],
              lb=[0.0, 0.0], ub=[1.0, 1.0])
    b = Block(name="down", fn=lambda u: jnp.array([u[0] * u[1] * 30.0]),
              u_names=["feed", "alloc"], y_names=["power"],
              lb=[0.0, 0.0], ub=[10.0, 1.0])
    return Network([a, b], links=[("up.residue", "down.feed")])


@pytest.fixture(autouse=True)
def _quiet_phase_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PhaseBoundaryWarning)
        yield


def test_block_evaluate_smoke():
    blk = Block(name="b", fn=lambda u, th: {"y2": th["k"] * u[0], "y1": u[0] ** 2},
                u_names=["x"], y_names=["y1", "y2"], lb=[0.0], ub=[2.0], theta={"k": 2.0})
    np.testing.assert_allclose(np.asarray(blk.evaluate(jnp.array([1.5]))), [2.25, 3.0])


def test_block_validation():
    with pytest.raises(ValueError, match="both an input and an output"):
        Block(name="b", fn=lambda u: u, u_names=["T"], y_names=["T"], lb=[0.0], ub=[1.0])
    with pytest.raises(ValueError, match="may not contain"):
        Block(name="a.b", fn=lambda u: u, u_names=["x"], y_names=["y"], lb=[0.0], ub=[1.0])
    with pytest.raises(ValueError, match="ub < lb"):
        Block(name="b", fn=lambda u: u, u_names=["x"], y_names=["y"], lb=[1.0], ub=[0.0])
    blk = Block(name="b", fn=lambda u: jnp.array([1.0]), u_names=["x"],
                y_names=["y1", "y2"], lb=[0.0], ub=[1.0])
    with pytest.raises(ValueError, match="expected \\(2,\\)"):
        blk.evaluate(jnp.array([0.5]))


def test_network_smoke():
    net = chain()
    assert net.decision_names == ["up.r", "up.s", "down.alloc"]
    values = net.evaluate(jnp.array([0.25, 0.5, 1.0])).as_dict()
    np.testing.assert_allclose(values["down.feed"], values["up.residue"])


def test_network_validation():
    net = chain()
    with pytest.raises(ValueError, match="recycle among blocks"):
        Network(net.blocks, links=[("up.residue", "down.feed"), ("down.power", "up.r")])
    with pytest.raises(ValueError, match="more than one link"):
        Network(net.blocks, links=[("up.residue", "down.feed"), ("up.prod", "down.feed")])
    with pytest.raises(ValueError, match="duplicate block names"):
        Network([quadratic_block(), quadratic_block()])


def test_linearize_and_helpers_smoke():
    blk = smooth_block()
    lin = linearize_block(blk)
    assert lin.J.shape == (2, 3) and np.all(np.isfinite(np.asarray(lin.J)))
    assert np.all(np.isfinite(scaled_jacobian(blk, lin)))
    assert choose_ad_mode(80, 1) == "rev" and choose_ad_mode(1, 40) == "fwd"
    np.testing.assert_array_equal(classify_phase(np.array([-0.1, 0.4, 1.2]), (0.0, 1.0)), [0, 1, 2])
    assert check_phase_transition(blk, lin, jnp.array([1.4, 1.4, 1.4])) == []


def test_build_lp_smoke_and_unknown_price():
    blk = smooth_block()
    lin = linearize_block(blk)
    sol = build_lp(Network([blk]), {"s": lin}, prices={"s.p": 1.0},
                   specs=[("s.q", ">=", 1e6)], radius=0.1, penalty=1.0).solve()
    assert sol.success and np.isfinite(sol["s.p"])
    with pytest.raises(KeyError, match="unknown variable"):
        build_lp(Network([blk]), {"s": lin}, prices={"s.nope": 1.0})


def test_pyomo_emission():
    pyo = pytest.importorskip("pyomo.environ")
    blk = smooth_block()
    lp = build_lp(Network([blk]), {"s": linearize_block(blk)}, prices={"s.p": 1.0}, radius=0.2)
    model = lp.to_pyomo()
    assert isinstance(model, pyo.ConcreteModel)
    assert len(list(model.x.keys())) == lp.n_cols


def test_planner_solve_smoke():
    planner = DeltaBasePlanner(Network([quadratic_block()]), prices={"q.f": 1.0},
                               radius=0.3, vertex_seeding=False)
    assert "maximise" in planner.describe()
    result = planner.solve()
    assert np.all(np.isfinite(np.asarray(result.decisions)))
    assert "decisions" in result.summary()
    assert "more columns" in result.lp_model.as_text(max_rows=0)
    sens = result.plan_sensitivity(wrt="prices")
    assert np.all(np.isfinite(np.asarray(sens.d_plan)))
    assert isinstance(planner.check_health().ok, bool)


def test_price_switch_point_smoke():
    blk = Block(name="a", fn=lambda u: jnp.array([u[0]]), u_names=["x"],
                y_names=["hi"], lb=[0.0], ub=[1.0])
    planner = DeltaBasePlanner(Network([blk]), prices={"a.hi": 1.0}, radius=0.4)
    assert price_switch_point(planner, "a.hi", 1.0, 5.0)["price"] is None


def test_modifier_adaptation_smoke_and_unknown_block():
    blk = Block(name="r", fn=lambda u: jnp.array([-(u[0] - 0.3) ** 2]), u_names=["a"],
                y_names=["f"], lb=[0.0], ub=[1.0])
    planner = DeltaBasePlanner(Network([blk]), prices={"r.f": 1.0}, radius=0.3,
                               vertex_seeding=False)
    plant = lambda u: jnp.array([-(u[0] - 0.7) ** 2])  # noqa: E731
    result = run_modifier_adaptation(planner, {"r": plant}, max_iter=3)
    assert np.isfinite(result.plan["r.a"])
    with pytest.raises(KeyError, match="unknown block"):
        run_modifier_adaptation(planner, {"nope": plant})


def test_constraint_backoff_smoke_and_validation():
    blk = Block(name="c", fn=lambda u, th: jnp.array([u[0] * th["k"], -(u[0] - 0.9) ** 2]),
                u_names=["a"], y_names=["T", "f"], lb=[0.0], ub=[1.0], theta={"k": 200.0})
    planner = DeltaBasePlanner(Network([blk]), prices={"c.f": 1.0},
                               specs=[("c.T", "<=", 120.0)], radius=0.3)
    found = constraint_backoff(planner, planner.network.decision_start(),
                               np.array([[25.0]]), ["c.k"], kappa=2.0)
    assert np.all(np.isfinite(found.backoff))
    assert len(apply_backoff(planner.specs, found)) == 1
    bare = DeltaBasePlanner(Network([smooth_block()]), prices={"s.p": 1.0},
                            specs=[("s.q", "<=", 1.0)])
    with pytest.raises(ValueError, match="scalar `theta` parameters"):
        constraint_backoff(bare, bare.network.decision_start(), np.eye(1))


def test_backed_off_plan_keeps_its_margin():
    # Regression (issue 513): the acceptance test scored steps against the
    # stated rhs, so a convex spec output overshot into the back-off.
    blk = Block(name="c", fn=lambda u: jnp.array([100.0 * u[0] + 100.0 * u[0] ** 2, u[0]]),
                u_names=["a"], y_names=["T", "f"], lb=[0.0], ub=[1.0], u0=[0.1])
    planner = DeltaBasePlanner(Network([blk]), prices={"c.f": 1.0},
                               specs=[("c.T", "<=", 120.0)], radius=0.3,
                               vertex_seeding=False)
    planner.specs = apply_backoff(planner.specs, {"c.T": 20.0})
    res = planner.solve()
    assert res.converged
    assert res.values["c.T"] <= 120.0 - 20.0 + 1e-6
    assert res.violations == {"c.T": 0.0}
    assert res.stated_violations == {"c.T": 0.0}


def test_piecewise_smoke_and_unbounded():
    blk = Block(name="w", fn=lambda u: jnp.array([jnp.sin(3 * u[0]) - 0.3 * u[0]]),
                u_names=["a"], y_names=["f"], lb=[0.0], ub=[3.0], u0=[2.6])
    data = sample_piecewise(blk, PiecewiseSpec("w", "a", n_points=7))
    assert data.y.shape == (7, 1) and data.J.shape == (7, 1, 1)
    unb = Block(name="u", fn=lambda u: jnp.array([u[0]]), u_names=["a"],
                y_names=["f"], lb=[0.0], ub=None, u0=[1.0])
    with pytest.raises(ValueError, match="unbounded"):
        sample_piecewise(unb, PiecewiseSpec("u", "a", n_points=5))


def test_best_cost_ratio_retry_policy():
    seen = []
    ratios = [7.0, 5.0, 2.0, 1.0]

    def fake(fn, x0, repeats, warmup, **kwargs):
        seen.append(repeats)
        ad = ratios[len(seen) - 1]
        return CostRatio(n=1, eval_seconds=1.0, ad_seconds=ad, fd_seconds=9.0,
                         ad_ratio=ad, fd_ratio=9.0, speedup=9.0 / ad)

    with mock.patch.object(benchmark_mod, "gradient_cost_ratio", fake):
        best = benchmark_mod.best_cost_ratio(lambda u: u, jnp.zeros(1), limit=3.0,
                                             attempts=4, repeats=2)
    assert best.ad_ratio == pytest.approx(2.0)
    assert seen == [2, 4, 8]


def test_delta_health_smoke_and_errors():
    assert check_delta_health(smooth_block()).ok
    nan = Block(name="b", fn=lambda u: jnp.array([jnp.sqrt(u[0])]), u_names=["x"],
                y_names=["y"], lb=[0.0], ub=[1.0], u0=[0.0])
    with pytest.raises(ValueError, match="health check found"):
        check_delta_health(nan).raise_on_error()
    dead = Block(name="b", fn=lambda u: jnp.array([u[0] + jnp.clip(u[1], 0.0, 1.0)]),
                 u_names=["free", "pinned"], y_names=["y"],
                 lb=[0.0, 2.0], ub=[1.0, 5.0], u0=[0.5, 3.0])
    with pytest.warns(DeltaHealthWarning, match="dead_lever"):
        check_delta_health(dead).warn()
    with pytest.raises(TypeError, match="Block or a Network"):
        check_delta_health("not a block")
