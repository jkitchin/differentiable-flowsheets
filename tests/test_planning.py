"""Tests for difflow.planning — delta-base planning from flowsheets.

The acceptance criteria for the module are covered explicitly:

1. ``test_gradient_cost_ratio_scaling`` — the AD gradient stays under 3x the
   cost of one model evaluation for n up to 80.
2. ``test_trust_region_converges_to_known_optimum`` and
   ``test_does_not_converge_without_acceptance_test`` — both directions.
3. ``test_delta_vectors_match_central_differences`` — agreement to <1e-4.
4. ``test_phase_boundary_warning_from_planner`` — the documented warning.
5. The worked example lives in ``examples/30_delta_base_planning.ipynb``;
   ``test_chain_allocation_lever_switches_with_prices`` covers its claim.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

jax.config.update("jax_enable_x64", True)

from difflow.planning import (
    Block,
    DeltaBasePlanner,
    Network,
    PhaseBoundaryWarning,
    PiecewiseSpec,
    Spec,
    TrustRegionOptions,
    apply_backoff,
    build_lp,
    check_delta_vectors,
    check_phase_transition,
    choose_ad_mode,
    classify_phase,
    constraint_backoff,
    format_scaling_table,
    gradient_cost_ratio,
    linearize_block,
    plan_sensitivity,
    planner_objective,
    price_switch_point,
    run_modifier_adaptation,
    sample_piecewise,
    scaling_study,
)
from difflow.planning import chain as chain_mod


# --------------------------------------------------------------------------
# Small analytic blocks used throughout
# --------------------------------------------------------------------------

def quadratic_block(name="q", peak=(0.3, 0.7)):
    """Concave block whose unique maximum is at ``peak``."""
    a, b = peak

    def fn(u):
        return jnp.array([-(u[0] - a) ** 2 - (u[1] - b) ** 2])

    return Block(name=name, fn=fn, u_names=["x", "y"], y_names=["f"],
                 lb=[0.0, 0.0], ub=[1.0, 1.0])


def smooth_block(name="s"):
    """A smooth, well-scaled block with more inputs than outputs."""

    def fn(u):
        return jnp.array([
            jnp.sin(3.0 * u[0]) * jnp.exp(-0.4 * u[1]) + 0.25 * u[2] ** 2,
            jnp.log1p(u[0] * u[1]) + jnp.tanh(u[2]),
        ])

    return Block(name=name, fn=fn, u_names=["a", "b", "c"],
                 y_names=["p", "q"], lb=[0.1, 0.1, 0.1], ub=[1.5, 1.5, 1.5])


@pytest.fixture(scope="module")
def chain_problem():
    return chain_mod.two_plant_chain()


@pytest.fixture(autouse=True)
def _quiet_phase_warnings():
    """Phase warnings are asserted where they matter; silence them elsewhere."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PhaseBoundaryWarning)
        yield


# --------------------------------------------------------------------------
# Block
# --------------------------------------------------------------------------

class TestBlock:

    def test_evaluate_positional_and_dict(self):
        blk = Block(name="b", fn=lambda u: {"y2": 2 * u[0], "y1": u[0] ** 2},
                    u_names=["x"], y_names=["y1", "y2"], lb=[0.0], ub=[2.0])
        npt.assert_allclose(np.asarray(blk.evaluate(jnp.array([1.5]))),
                            [2.25, 3.0])

    def test_defaults_to_bound_midpoint(self):
        blk = quadratic_block()
        npt.assert_allclose(np.asarray(blk.u0), [0.5, 0.5])

    def test_rejects_name_used_as_input_and_output(self):
        with pytest.raises(ValueError, match="both an input and an output"):
            Block(name="b", fn=lambda u: u, u_names=["T"], y_names=["T"],
                  lb=[0.0], ub=[1.0])

    def test_rejects_dot_in_block_name(self):
        with pytest.raises(ValueError, match="may not contain"):
            Block(name="a.b", fn=lambda u: u, u_names=["x"], y_names=["y"],
                  lb=[0.0], ub=[1.0])

    def test_rejects_inverted_bounds(self):
        with pytest.raises(ValueError, match="ub < lb"):
            Block(name="b", fn=lambda u: u, u_names=["x"], y_names=["y"],
                  lb=[1.0], ub=[0.0])

    def test_rejects_wrong_output_length(self):
        blk = Block(name="b", fn=lambda u: jnp.array([1.0]), u_names=["x"],
                    y_names=["y1", "y2"], lb=[0.0], ub=[1.0])
        with pytest.raises(ValueError, match="expected \\(2,\\)"):
            blk.evaluate(jnp.array([0.5]))

    def test_theta_is_passed_through(self):
        blk = Block(name="b", fn=lambda u, th: jnp.array([th["k"] * u[0]]),
                    u_names=["x"], y_names=["y"], lb=[0.0], ub=[1.0],
                    theta={"k": 3.0})
        npt.assert_allclose(float(blk.evaluate(jnp.array([2.0]))[0]), 6.0)

    def test_jit_gives_the_same_answer(self):
        raw = smooth_block()
        jitted = Block(name="s", fn=raw.fn, u_names=raw.u_names,
                       y_names=raw.y_names, lb=raw.lb, ub=raw.ub, jit=True)
        u = jnp.array([0.4, 0.9, 1.1])
        npt.assert_allclose(np.asarray(raw.evaluate(u)),
                            np.asarray(jitted.evaluate(u)), rtol=1e-12)


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------

class TestNetwork:

    def _chain(self):
        a = Block(name="up", fn=lambda u: jnp.array([u[0] * u[1], 1 - u[0]]),
                  u_names=["r", "s"], y_names=["prod", "residue"],
                  lb=[0.0, 0.0], ub=[1.0, 1.0])
        b = Block(name="down", fn=lambda u: jnp.array([u[0] * u[1] * 30.0]),
                  u_names=["feed", "alloc"], y_names=["power"],
                  lb=[0.0, 0.0], ub=[10.0, 1.0])
        return Network([a, b], links=[("up.residue", "down.feed")])

    def test_linked_input_is_not_a_decision(self):
        net = self._chain()
        assert net.decision_names == ["up.r", "up.s", "down.alloc"]
        assert net.is_linked("down.feed")

    def test_evaluation_propagates_the_link(self):
        net = self._chain()
        values = net.evaluate(jnp.array([0.25, 0.5, 1.0])).as_dict()
        npt.assert_allclose(values["up.residue"], 0.75)
        npt.assert_allclose(values["down.feed"], 0.75)
        npt.assert_allclose(values["down.power"], 22.5)

    def test_topological_order_is_respected(self):
        a = Block(name="a", fn=lambda u: jnp.array([u[0]]), u_names=["x"],
                  y_names=["y"], lb=[0.0], ub=[1.0])
        b = Block(name="b", fn=lambda u: jnp.array([u[0] + 1]), u_names=["x"],
                  y_names=["y"], lb=[0.0], ub=[9.0])
        net = Network([b, a], links=[("a.y", "b.x")])
        assert net.order == ["a", "b"]

    def test_rejects_inter_block_recycle(self):
        net = self._chain()
        with pytest.raises(ValueError, match="recycle among blocks"):
            Network(net.blocks,
                    links=[("up.residue", "down.feed"),
                           ("down.power", "up.r")])

    def test_rejects_two_links_into_one_input(self):
        net = self._chain()
        with pytest.raises(ValueError, match="more than one link"):
            Network(net.blocks, links=[("up.residue", "down.feed"),
                                       ("up.prod", "down.feed")])

    def test_rejects_duplicate_block_names(self):
        blk = quadratic_block()
        with pytest.raises(ValueError, match="duplicate block names"):
            Network([blk, quadratic_block()])


# --------------------------------------------------------------------------
# Delta vectors  (acceptance criterion 3, and the AD-mode requirement)
# --------------------------------------------------------------------------

class TestDeltaVectors:

    def test_delta_vectors_match_central_differences(self):
        """Acceptance criterion 3: AD agrees with FD to <1e-4 relative."""
        result = check_delta_vectors(smooth_block(),
                                     u0=jnp.array([0.6, 0.9, 1.2]), rtol=1e-4)
        assert result["passed"]
        assert result["max_rel_error"] < 1e-4

    def test_delta_vectors_match_on_a_real_flowsheet_block(self):
        """The same check on a block containing an implicit flash solve."""
        result = check_delta_vectors(chain_mod.ngl_block(), rtol=1e-4)
        assert result["passed"], result["max_rel_error"]

    def test_check_can_raise(self):
        blk = smooth_block()
        with pytest.raises(AssertionError, match="disagree with central"):
            check_delta_vectors(blk, rtol=1e-18, raise_on_fail=True)

    def test_ad_mode_follows_the_shape(self):
        """jacrev when n >> m, jacfwd when m >> n; never hard-coded."""
        assert choose_ad_mode(80, 1) == "rev"
        assert choose_ad_mode(1, 40) == "fwd"
        assert choose_ad_mode(5, 5) == "rev"
        assert choose_ad_mode(1, 40, mode="rev") == "rev"

    def test_reverse_mode_is_exercised(self):
        """Reverse mode delivers the scaling, so it must actually be used."""
        blk = smooth_block()  # 3 inputs, 2 outputs
        lin = linearize_block(blk)
        assert lin.mode == "rev"
        fwd = linearize_block(blk, mode="fwd")
        npt.assert_allclose(np.asarray(lin.J), np.asarray(fwd.J), rtol=1e-10)

    def test_forward_mode_for_tall_blocks(self):
        blk = Block(name="t", fn=lambda u: jnp.array([u[0], u[0] ** 2,
                                                      jnp.sin(u[0])]),
                    u_names=["x"], y_names=["a", "b", "c"],
                    lb=[0.0], ub=[2.0])
        assert linearize_block(blk).mode == "fwd"

    def test_taylor_model_is_exact_at_the_centre(self):
        blk = smooth_block()
        lin = linearize_block(blk)
        npt.assert_allclose(np.asarray(lin.predict(lin.u0)),
                            np.asarray(lin.y0), rtol=1e-12)


# --------------------------------------------------------------------------
# Phase boundaries  (acceptance criterion 4)
# --------------------------------------------------------------------------

class TestPhaseBoundary:

    def test_classification_bins_on_the_lower_side(self):
        codes = classify_phase(np.array([-0.1, 0.0, 0.4, 1.0, 1.2]), (0.0, 1.0))
        npt.assert_array_equal(codes, [0, 0, 1, 1, 2])

    def test_crossing_raises_the_warning(self):
        blk = chain_mod.ngl_block()
        lin = linearize_block(blk, jnp.array([0.7, 224.0, 0.4, 4.0e6]))
        assert int(np.asarray(lin.phase_code)[0]) == 0  # subcooled liquid
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            messages = check_phase_transition(
                blk, lin, jnp.array([0.7, 240.0, 0.4, 4.0e6]))
        assert len(messages) == 1
        assert any(w.category is PhaseBoundaryWarning for w in caught)
        assert "phase boundary" in messages[0]

    def test_no_warning_inside_one_regime(self):
        blk = chain_mod.ngl_block()
        lin = linearize_block(blk, jnp.array([0.7, 236.0, 0.4, 3.0e6]))
        assert check_phase_transition(
            blk, lin, jnp.array([0.7, 238.0, 0.4, 3.0e6])) == []

    def test_phase_boundary_warning_from_planner(self):
        """Acceptance criterion 4, through the planner's own machinery."""
        blk = chain_mod.ngl_block()
        blk.u0 = jnp.array([0.7, 222.0, 0.4, 4.0e6])
        net = Network([blk])
        # Refrigeration is a cost, so the LP wants a warmer cold box — which
        # walks the block out of the subcooled branch it was linearised on.
        planner = DeltaBasePlanner(
            net, prices={"ngl.E_refrig": -18.0}, radius=0.5,
            vertex_seeding=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = planner.solve(u0=jnp.array([0.7, 222.0, 0.4, 4.0e6]))
        assert any(w.category is PhaseBoundaryWarning for w in caught)
        assert result.phase_warnings

    def test_blocks_without_a_phase_fn_are_silent(self):
        blk = smooth_block()
        lin = linearize_block(blk)
        assert check_phase_transition(blk, lin, jnp.array([1.4, 1.4, 1.4])) == []


# --------------------------------------------------------------------------
# LP assembly
# --------------------------------------------------------------------------

class TestLP:

    def test_model_rows_reproduce_the_taylor_model(self):
        blk = smooth_block()
        net = Network([blk])
        lin = linearize_block(blk)
        lp = build_lp(net, {"s": lin}, prices={"s.p": 1.0}, radius=0.1)
        sol = lp.solve()
        assert sol.success
        u = np.array([sol[n] for n in blk.qualified_u()])
        npt.assert_allclose([sol["s.p"], sol["s.q"]],
                            np.asarray(lin.predict(jnp.asarray(u))),
                            rtol=1e-8, atol=1e-10)

    def test_trust_region_limits_the_step(self):
        blk = smooth_block()
        lin = linearize_block(blk)
        lp = build_lp(Network([blk]), {"s": lin}, prices={"s.p": 1.0},
                      radius=0.05)
        sol = lp.solve()
        span = np.asarray(blk.range)
        step = np.abs(np.array([sol[n] for n in blk.qualified_u()])
                      - np.asarray(lin.u0))
        assert np.all(step <= 0.05 * span + 1e-9)

    def test_elastic_slack_keeps_an_impossible_spec_feasible(self):
        blk = smooth_block()
        lin = linearize_block(blk)
        lp = build_lp(Network([blk]), {"s": lin}, prices={"s.p": 1.0},
                      specs=[("s.q", ">=", 1e6)], radius=0.1, penalty=1.0)
        sol = lp.solve()
        assert sol.success
        assert sol.slacks()["s.q"] > 0

    def test_link_rows_tie_the_blocks_together(self):
        a = Block(name="a", fn=lambda u: jnp.array([2.0 * u[0]]),
                  u_names=["x"], y_names=["y"], lb=[0.0], ub=[1.0])
        b = Block(name="b", fn=lambda u: jnp.array([3.0 * u[0]]),
                  u_names=["z"], y_names=["w"], lb=[0.0], ub=[5.0])
        net = Network([a, b], links=[("a.y", "b.z")])
        state = net.evaluate(jnp.array([0.5]))
        lins = {n: linearize_block(net.block(n), state.u[n]) for n in net.order}
        sol = build_lp(net, lins, prices={"b.w": 1.0}, centers=state.u,
                       radius=1.0).solve()
        npt.assert_allclose(sol["b.z"], sol["a.y"], atol=1e-9)
        npt.assert_allclose(sol["b.w"], 6.0, atol=1e-8)

    def test_backoff_tightens_the_lp_but_not_the_promise(self):
        spec = Spec("s.q", "<=", 10.0, backoff=2.0)
        assert spec.effective_rhs == 8.0
        # Eating into the margin is not a violation of the stated spec.
        assert spec.violation({"s.q": 9.0}) == 0.0
        assert spec.violation({"s.q": 11.0}) == pytest.approx(1.0)

    def test_unknown_price_is_rejected(self):
        blk = smooth_block()
        with pytest.raises(KeyError, match="unknown variable"):
            build_lp(Network([blk]), {"s": linearize_block(blk)},
                     prices={"s.nope": 1.0})

    def test_pyomo_emission(self):
        pyo = pytest.importorskip("pyomo.environ")
        blk = smooth_block()
        lp = build_lp(Network([blk]), {"s": linearize_block(blk)},
                      prices={"s.p": 1.0}, specs=[("s.q", "<=", 5.0)],
                      radius=0.2)
        model = lp.to_pyomo()
        assert isinstance(model, pyo.ConcreteModel)
        assert len(list(model.x.keys())) == lp.n_cols
        assert model.obj.sense == pyo.minimize


# --------------------------------------------------------------------------
# Trust region  (acceptance criterion 2)
# --------------------------------------------------------------------------

class TestTrustRegion:

    def test_trust_region_converges_to_known_optimum(self):
        """Acceptance criterion 2, first half: it converges from a cold start."""
        net = Network([quadratic_block(peak=(0.3, 0.7))])
        result = DeltaBasePlanner(net, prices={"q.f": 1.0}, radius=0.3).solve()
        assert result.converged
        npt.assert_allclose(result.decisions, [0.3, 0.7], atol=1e-5)
        npt.assert_allclose(result.objective, 0.0, atol=1e-9)

    def test_does_not_converge_without_acceptance_test(self):
        """Acceptance criterion 2, second half: the guard is load-bearing."""
        net = Network([quadratic_block(peak=(0.3, 0.7))])
        result = DeltaBasePlanner(
            net, prices={"q.f": 1.0}, radius=0.3, accept_test=False,
            vertex_seeding=False,
            options=TrustRegionOptions(radius=0.3, max_iter=40)).solve()
        assert not result.converged
        assert np.linalg.norm(np.asarray(result.decisions) - [0.3, 0.7]) > 0.1
        # It does not merely stall: it oscillates, taking steps the nonlinear
        # model does not support.
        assert any(h.rho < 0 for h in result.history)

    def test_every_proposal_is_evaluated_nonlinearly(self):
        calls = {"n": 0}

        def fn(u):
            calls["n"] += 1
            return jnp.array([-(u[0] - 0.4) ** 2])

        blk = Block(name="c", fn=fn, u_names=["x"], y_names=["f"],
                    lb=[0.0], ub=[1.0])
        planner = DeltaBasePlanner(Network([blk]), prices={"c.f": 1.0},
                                   radius=0.3, vertex_seeding=False)
        result = planner.solve()
        # One evaluation per proposal at minimum, on top of linearisation.
        assert calls["n"] >= result.n_iterations

    def test_realised_violation_is_charged_not_predicted(self):
        """A plan that runs off-spec is charged for it by the nonlinear model.

        The LP's own slack reads zero here because the linear model predicts
        feasibility; only evaluating the real block reveals the violation.
        """

        def fn(u):
            # Curves away from the linear model built at the origin.
            return jnp.array([u[0] + 4.0 * u[0] ** 2])

        blk = Block(name="v", fn=fn, u_names=["x"], y_names=["g"],
                    lb=[0.0], ub=[1.0], u0=[0.0])
        planner = DeltaBasePlanner(
            Network([blk]), prices={"v.g": 1.0},
            specs=[("v.g", "<=", 0.5)], penalty=1e3, radius=1.0,
            vertex_seeding=False)

        lin = linearize_block(blk, jnp.array([0.0]))
        lp = build_lp(planner.network, {"v": lin}, planner.prices,
                      planner.specs, centers={"v": lin.u0}, radius=1.0,
                      penalty=1e3)
        sol = lp.solve()
        assert sol.slacks()["v.g"] == pytest.approx(0.0, abs=1e-9)

        # The same proposal, scored against the real block, is off-spec.
        scored = planner.score(jnp.array([sol["v.x"]]))
        assert scored["violations"]["v.g"] > 0.1
        assert scored["merit"] < scored["objective"]

        # And the planner therefore refuses to end up there.
        result = planner.solve()
        assert result.total_violation < 1e-6

    def test_vertex_seeding_finds_the_right_corner(self):
        """A bang-bang lever is not decided by the starting point."""

        def fn(u):
            # Two corners. The far one is much better, but the slope at the
            # interior start points at the near one, so a single interior
            # start walks steadily to the wrong bound.
            x = u[0]
            return jnp.array([0.1 - 0.25 * x + jnp.exp(8.0 * (x - 1.0))])

        blk = Block(name="bb", fn=fn, u_names=["a"], y_names=["f"],
                    lb=[0.0], ub=[1.0], u0=[0.5])
        seeded = DeltaBasePlanner(Network([blk]), prices={"bb.f": 1.0},
                                  radius=0.2, vertex_seeding=True).solve()
        lonely = DeltaBasePlanner(Network([blk]), prices={"bb.f": 1.0},
                                  radius=0.2, vertex_seeding=False).solve()
        npt.assert_allclose(seeded.decisions, [1.0], atol=1e-6)
        npt.assert_allclose(lonely.decisions, [0.0], atol=1e-6)
        assert seeded.objective > lonely.objective
        assert seeded.n_starts > 1

    def test_seeds_can_be_supplied_explicitly(self):
        net = Network([quadratic_block(peak=(0.9, 0.1))])
        planner = DeltaBasePlanner(net, prices={"q.f": 1.0}, radius=0.2)
        result = planner.solve(seeds=[jnp.array([0.95, 0.05])])
        assert result.n_starts == 2
        npt.assert_allclose(result.decisions, [0.9, 0.1], atol=1e-5)

    def test_minimisation_sense(self):
        net = Network([quadratic_block(peak=(0.3, 0.7))])
        result = DeltaBasePlanner(net, prices={"q.f": -1.0}, radius=0.3,
                                  sense="min").solve()
        npt.assert_allclose(result.decisions, [0.3, 0.7], atol=1e-5)

    def test_result_reports_delta_vectors_and_summary(self):
        net = Network([smooth_block()])
        result = DeltaBasePlanner(net, prices={"s.p": 1.0}, radius=0.2).solve()
        assert result.delta_vectors["s"].shape == (2, 3)
        assert "decisions" in result.summary()
        assert "delta vectors" in result.delta_table("s")


# --------------------------------------------------------------------------
# Sensitivity of the plan
# --------------------------------------------------------------------------

class TestPlanSensitivity:

    def _priced_block(self):
        # max p_lin * u - p_quad * u^2  ->  u* = p_lin / (2 p_quad)
        return Block(name="q", fn=lambda u: jnp.array([u[0], -u[0] ** 2]),
                     u_names=["a"], y_names=["lin", "quad"],
                     lb=[0.0], ub=[5.0])

    def test_price_sensitivity_matches_the_analytic_answer(self):
        planner = DeltaBasePlanner(Network([self._priced_block()]),
                                   prices={"q.lin": 2.0, "q.quad": 1.0},
                                   radius=0.5)
        result = planner.solve()
        npt.assert_allclose(result.decisions, [1.0], atol=1e-5)
        sens = result.plan_sensitivity(wrt="prices")
        table = sens.as_dict()["q.a"]
        npt.assert_allclose(table["q.lin"], 0.5, rtol=1e-4)
        npt.assert_allclose(table["q.quad"], -1.0, rtol=1e-3)
        # Envelope theorem: d(objective)/d(price) is the priced activity.
        obj = sens.objective_sensitivity()
        npt.assert_allclose(obj["q.lin"], 1.0, rtol=1e-4)
        npt.assert_allclose(obj["q.quad"], -1.0, rtol=1e-3)
        assert "not a stationary point" not in sens.note

    def test_parameter_sensitivity(self):
        blk = Block(name="p",
                    fn=lambda u, th: jnp.array([-(u[0] - th["target"]) ** 2]),
                    u_names=["a"], y_names=["f"], lb=[0.0], ub=[2.0],
                    theta={"target": 1.2})
        result = DeltaBasePlanner(Network([blk]), prices={"p.f": 1.0},
                                  radius=0.4).solve()
        npt.assert_allclose(result.decisions, [1.2], atol=1e-5)
        sens = result.plan_sensitivity(wrt="theta")
        # The optimum tracks the target one-for-one.
        npt.assert_allclose(sens.as_dict()["p.a"]["p.target"], 1.0, rtol=1e-4)

    def test_sensitivity_at_a_vertex_is_zero_and_says_so(self):
        blk = Block(name="v", fn=lambda u: jnp.array([u[0]]), u_names=["a"],
                    y_names=["f"], lb=[0.0], ub=[1.0])
        result = DeltaBasePlanner(Network([blk]), prices={"v.f": 1.0},
                                  radius=0.5).solve()
        npt.assert_allclose(result.decisions, [1.0], atol=1e-9)
        sens = result.plan_sensitivity(wrt="prices")
        npt.assert_allclose(sens.d_plan, [[0.0]], atol=1e-12)
        assert "switches at a finite parameter change" in sens.note
        assert sens.fixed_decisions == ["v.a"]

    def test_active_spec_multiplier_is_reported(self):
        blk = Block(name="c", fn=lambda u: jnp.array([u[0], 3.0 * u[0]]),
                    u_names=["a"], y_names=["profit", "load"],
                    lb=[0.0], ub=[10.0])
        result = DeltaBasePlanner(
            Network([blk]), prices={"c.profit": 1.0},
            specs=[("c.load", "<=", 6.0)], radius=0.5).solve()
        npt.assert_allclose(result.decisions, [2.0], atol=1e-4)
        sens = result.plan_sensitivity(wrt="prices")
        assert "c.load" in sens.active_constraints

    def test_price_switch_point_brackets_the_corner(self):
        def fn(u):
            return jnp.array([u[0], 1.0 - u[0]])

        blk = Block(name="a", fn=fn, u_names=["x"], y_names=["hi", "lo"],
                    lb=[0.0], ub=[1.0], u0=[0.5])
        planner = DeltaBasePlanner(Network([blk]),
                                   prices={"a.hi": 1.0, "a.lo": 2.0},
                                   radius=0.4)
        found = price_switch_point(planner, "a.hi", 0.5, 4.0, tol=1e-3)
        assert found["decision"] == "a.x"
        npt.assert_allclose(found["price"], 2.0, atol=5e-3)
        # The planner's own prices are left untouched.
        assert planner.prices["a.hi"] == 1.0

    def test_price_switch_point_reports_no_switch(self):
        blk = Block(name="a", fn=lambda u: jnp.array([u[0]]), u_names=["x"],
                    y_names=["hi"], lb=[0.0], ub=[1.0])
        planner = DeltaBasePlanner(Network([blk]), prices={"a.hi": 1.0},
                                   radius=0.4)
        found = price_switch_point(planner, "a.hi", 1.0, 5.0)
        assert found["price"] is None


# --------------------------------------------------------------------------
# Modifier adaptation
# --------------------------------------------------------------------------

class TestModifiers:

    MODEL = staticmethod(lambda u: jnp.array([-(u[0] - 0.3) ** 2]))
    PLANT = staticmethod(
        lambda u: jnp.array([-(u[0] - 0.7) ** 2 - 0.4 * (u[0] - 0.7) ** 3]))

    def _planner(self):
        blk = Block(name="r", fn=self.MODEL, u_names=["a"], y_names=["f"],
                    lb=[0.0], ub=[1.0])
        return DeltaBasePlanner(Network([blk]), prices={"r.f": 1.0},
                                radius=0.3, vertex_seeding=False)

    def test_gradient_modifiers_reach_the_plant_optimum(self):
        result = run_modifier_adaptation(
            self._planner(), {"r": self.PLANT}, max_iter=40, tol=1e-7)
        assert result.converged
        npt.assert_allclose(result.plan["r.a"], 0.7, atol=1e-4)
        npt.assert_allclose(result.plant_objective, 0.0, atol=1e-6)

    def test_value_only_modifiers_stop_at_the_models_optimum(self):
        """Under structural mismatch, eps alone is not enough."""
        result = run_modifier_adaptation(
            self._planner(), {"r": self.PLANT}, max_iter=40, tol=1e-7,
            use_gradients=False)
        npt.assert_allclose(result.plan["r.a"], 0.3, atol=1e-4)
        assert result.plant_objective < -0.1

    def test_modifiers_leave_the_linearisation_consistent(self):
        from difflow.planning.modifiers import update_modifiers
        blk = Block(name="r", fn=self.MODEL, u_names=["a"], y_names=["f"],
                    lb=[0.0], ub=[1.0])
        u = jnp.array([0.45])
        mods = update_modifiers(blk, u, self.PLANT)
        lin = mods.apply(linearize_block(blk, u))
        # At the adaptation point the corrected model matches the plant in
        # both value and gradient.
        npt.assert_allclose(np.asarray(lin.y0),
                            np.asarray(self.PLANT(u)), atol=1e-10)
        plant_grad = np.asarray(jax.jacobian(self.PLANT)(u))
        npt.assert_allclose(np.asarray(lin.J), plant_grad, atol=1e-8)

    def test_unknown_plant_block_is_rejected(self):
        with pytest.raises(KeyError, match="unknown block"):
            run_modifier_adaptation(self._planner(), {"nope": self.PLANT})


# --------------------------------------------------------------------------
# Coefficient covariance and back-off
# --------------------------------------------------------------------------

class TestBackOff:

    def _planner(self):
        blk = Block(name="c",
                    fn=lambda u, th: jnp.array([u[0] * th["k"],
                                                -(u[0] - 0.9) ** 2]),
                    u_names=["a"], y_names=["T", "f"], lb=[0.0], ub=[1.0],
                    theta={"k": 200.0})
        return DeltaBasePlanner(Network([blk]), prices={"c.f": 1.0},
                                specs=[("c.T", "<=", 120.0)], radius=0.3)

    def test_backoff_is_kappa_sigma(self):
        planner = self._planner()
        result = planner.solve()
        npt.assert_allclose(result.values["c.T"], 120.0, atol=1e-4)
        found = constraint_backoff(planner, result.decisions,
                                   np.array([[25.0]]), ["c.k"], kappa=2.0)
        # sigma = |dT/dk| * sigma_k = a * 5 = 0.6 * 5 = 3
        npt.assert_allclose(found.sigma[0], 3.0, rtol=1e-3)
        npt.assert_allclose(found.backoff[0], 6.0, rtol=1e-3)

    def test_backoff_moves_the_plan_off_the_limit(self):
        planner = self._planner()
        result = planner.solve()
        found = constraint_backoff(planner, result.decisions,
                                   np.array([[25.0]]), ["c.k"], kappa=2.0)
        planner.specs = apply_backoff(planner.specs, found)
        backed_off = planner.solve()
        assert backed_off.values["c.T"] < result.values["c.T"] - 1.0
        assert backed_off.total_violation == 0.0

    def test_requires_parameters(self):
        net = Network([smooth_block()])
        planner = DeltaBasePlanner(net, prices={"s.p": 1.0},
                                   specs=[("s.q", "<=", 1.0)])
        with pytest.raises(ValueError, match="scalar `theta` parameters"):
            constraint_backoff(planner, planner.network.decision_start(),
                               np.eye(1))


# --------------------------------------------------------------------------
# Piecewise / MILP
# --------------------------------------------------------------------------

class TestPiecewise:

    def _wavy(self):
        return Block(name="w",
                     fn=lambda u: jnp.array([jnp.sin(3 * u[0]) - 0.3 * u[0]]),
                     u_names=["a"], y_names=["f"], lb=[0.0], ub=[3.0],
                     u0=[2.6])

    def test_sampling_is_exact_at_the_breakpoints(self):
        blk = self._wavy()
        data = sample_piecewise(blk, PiecewiseSpec("w", "a", n_points=7))
        assert data.y.shape == (7, 1)
        assert data.J.shape == (7, 1, 1)
        for k, g in enumerate(data.breakpoints):
            npt.assert_allclose(data.y[k],
                                np.asarray(blk.evaluate(jnp.array([g]))),
                                atol=1e-12)

    def test_piecewise_escapes_a_local_optimum(self):
        blk = self._wavy()  # starts near the wrong hump at a = 2.6
        plain = DeltaBasePlanner(Network([blk]), prices={"w.f": 1.0},
                                 radius=0.1, vertex_seeding=False).solve()
        milp = DeltaBasePlanner(
            Network([blk]), prices={"w.f": 1.0}, radius=0.1,
            vertex_seeding=False,
            piecewise=[PiecewiseSpec("w", "a", n_points=31)]).solve()
        grid = np.linspace(0.0, 3.0, 3001)
        best = float(np.max(np.sin(3 * grid) - 0.3 * grid))
        assert milp.objective > plain.objective
        npt.assert_allclose(milp.objective, best, atol=1e-3)

    def test_milp_columns_are_integral(self):
        blk = self._wavy()
        planner = DeltaBasePlanner(
            Network([blk]), prices={"w.f": 1.0}, radius=0.2,
            piecewise=[PiecewiseSpec("w", "a", n_points=9)])
        result = planner.solve()
        assert len(result.lp_model.integer_cols) == 8
        assert result.lp_model.sos2_sets

    def test_rejects_unbounded_variable(self):
        blk = Block(name="u", fn=lambda u: jnp.array([u[0]]), u_names=["a"],
                    y_names=["f"], lb=[0.0], ub=None, u0=[1.0])
        with pytest.raises(ValueError, match="unbounded"):
            sample_piecewise(blk, PiecewiseSpec("u", "a", n_points=5))


# --------------------------------------------------------------------------
# Scaling  (acceptance criterion 1)
# --------------------------------------------------------------------------

class TestScaling:

    @staticmethod
    def _make(n):
        problem = chain_mod.two_plant_chain(horizon=n // 5)
        return (planner_objective(problem.planner()),
                problem.network.decision_start())

    def test_gradient_cost_ratio_scaling(self):
        """Acceptance criterion 1: AD gradient cost stays under 3x one eval.

        Reverse mode is used throughout — a scalar objective over ``n``
        decisions is exactly the shape it is for — so this also exercises the
        mode that delivers the scaling.
        """
        rows = scaling_study(self._make, [5, 10, 20, 40, 80], repeats=3,
                             warmup=2, mode="rev")
        assert [r.n for r in rows] == [5, 10, 20, 40, 80]
        for row in rows:
            assert row.ad_ratio < 3.0, (
                f"AD gradient cost {row.ad_ratio:.2f}x one evaluation at "
                f"n={row.n}\n" + format_scaling_table(rows))
        # Finite differences must cost more, and increasingly so.
        assert rows[-1].fd_seconds > rows[0].fd_seconds
        assert rows[-1].speedup > 1.0

    def test_gradient_agrees_with_finite_differences(self):
        fn, x0 = self._make(5)
        ratio = gradient_cost_ratio(fn, x0, repeats=1, warmup=1, check=True,
                                    step=1e-6)
        assert ratio.max_abs_error is not None
        scale = max(1.0, float(np.max(np.abs(jax.grad(fn)(x0)))))
        assert ratio.max_abs_error / scale < 1e-4

    def test_table_formats(self):
        rows = scaling_study(self._make, [5], repeats=1, warmup=1)
        text = format_scaling_table(rows)
        assert text.startswith("| n |")
        assert "speedup" in text


# --------------------------------------------------------------------------
# The reference chain  (acceptance criterion 5's claim)
# --------------------------------------------------------------------------

class TestChain:

    def test_chain_structure(self, chain_problem):
        net = chain_problem.network
        assert net.order == ["ngl", "power"]
        assert net.is_linked("power.fuel_F")
        assert chain_problem.n_decisions == 5

    def test_horizon_scales_decisions(self):
        assert chain_mod.two_plant_chain(horizon=4).n_decisions == 20

    def test_plan_respects_the_spec(self, chain_problem):
        result = chain_problem.planner(radius=0.25).solve()
        assert result.converged
        assert result.total_violation < 1e-6
        assert result.values["ngl.T_colfeed"] <= 236.0 + 1e-6

    def test_chain_allocation_lever_switches_with_prices(self, chain_problem):
        """The worked example's claim: the lever flips corner with price."""
        cheap = chain_problem.planner(radius=0.25)
        cheap.prices = dict(chain_problem.prices)
        cheap.prices["power.Power"] = 10.0
        dear = chain_problem.planner(radius=0.25)
        dear.prices = dict(chain_problem.prices)
        dear.prices["power.Power"] = 55.0

        low, high = cheap.solve(), dear.solve()
        npt.assert_allclose(low.plan["power.alloc"], 0.0, atol=1e-6)
        npt.assert_allclose(high.plan["power.alloc"], 1.0, atol=1e-6)
        # Ethane recovery flips with it: rejected ethane is worth more as fuel.
        assert low.plan["ngl.ethane_recovery"] > high.plan["ngl.ethane_recovery"]

    def test_delta_vectors_are_reported_for_both_blocks(self, chain_problem):
        result = chain_problem.planner(radius=0.25).solve()
        assert result.delta_vectors["ngl"].shape == (5, 4)
        assert result.delta_vectors["power"].shape == (3, 2)
