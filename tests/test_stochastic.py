"""Tests for difflow.stochastic.

The tests that matter here are the ones with an answer known in advance.  A
stochastic program is easy to write and hard to check by eye --- a plausible
number is not evidence --- so almost everything below is pinned to a closed
form: the newsvendor's critical quantile, the sample mean, the empirical CVaR,
and the ordering ``WS <= SP <= EEV`` that no correct implementation can
violate.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import difflow.stochastic as st
from difflow.stochastic.risk import weighted_quantile

FAST = st.SAAOptions(steps=250, rounds=3, n_starts=1)


# =============================================================================
# ScenarioSet
# =============================================================================


class TestScenarioSet:
    def test_shapes_and_names(self):
        s = st.ScenarioSet.normal({"a": (1.0, 0.1), "b": (2.0, 0.2)},
                                  n=32, seed=0)
        assert (s.n_scenarios, s.n_parameters) == (32, 2)
        assert s.names == ("a", "b")
        assert float(jnp.sum(s.weights)) == pytest.approx(1.0)

    def test_the_mean_scenario_is_the_weighted_mean(self):
        s = st.ScenarioSet.normal({"a": (1.0, 0.5)}, n=128, seed=1)
        assert float(s.mean_scenario().draws[0, 0]) == pytest.approx(
            float(jnp.mean(s.column("a"))), rel=1e-12)

    def test_a_covariance_survives_the_draw(self):
        """The correlation is the whole reason from_covariance exists."""
        cov = np.array([[0.04, 0.024], [0.024, 0.09]])
        s = st.ScenarioSet.from_covariance(["a", "b"], [1.0, 2.0], cov,
                                           n=20_000, seed=3)
        got = np.cov(np.asarray(s.draws).T)
        np.testing.assert_allclose(got, cov, rtol=0.06)

    def test_a_rank_deficient_covariance_samples_the_directions_it_has(self):
        """Two parameters the data cannot separate must not raise."""
        cov = np.array([[0.04, 0.04], [0.04, 0.04]])       # rank 1
        s = st.ScenarioSet.from_covariance(["a", "b"], [0.0, 0.0], cov,
                                           n=256, seed=0)
        d = np.asarray(s.draws)
        np.testing.assert_allclose(d[:, 0], d[:, 1], atol=1e-9)

    def test_lognormal_draws_are_positive(self):
        s = st.ScenarioSet.lognormal({"D": (5.0, 0.4)}, n=512, seed=0)
        assert float(jnp.min(s.draws)) > 0.0

    def test_lognormal_refuses_a_non_positive_median(self):
        with pytest.raises(ValueError, match="must be positive"):
            st.ScenarioSet.lognormal({"D": (0.0, 0.2)}, n=8)

    def test_uniform_refuses_a_reversed_interval(self):
        with pytest.raises(ValueError, match="reversed"):
            st.ScenarioSet.uniform({"x": (3.0, 1.0)}, n=8)

    def test_a_redraw_is_independent_but_the_same_shape(self):
        s = st.ScenarioSet.normal({"a": (1.0, 0.1)}, n=64, seed=0)
        t = s.redraw(7)
        assert t.draws.shape == s.draws.shape
        assert not np.allclose(np.asarray(t.draws), np.asarray(s.draws))

    def test_a_supplied_sample_cannot_be_redrawn(self):
        s = st.ScenarioSet.from_samples(np.zeros((4, 1)), ["a"])
        with pytest.raises(ValueError, match="no distribution to redraw"):
            s.redraw(1)

    def test_names_must_match_the_columns(self):
        with pytest.raises(ValueError, match="names for"):
            st.ScenarioSet.from_samples(np.zeros((4, 2)), ["a"])

    def test_the_flexibility_bridge(self):
        from difflow.flexibility.sets import UncertaintySet
        T = UncertaintySet(nominal=[1.0, 2.0], lower=[0.2, 0.1],
                           upper=[0.1, 0.3], names=["feed", "T"])
        s = st.ScenarioSet.from_uncertainty_set(T, n=64, seed=0)
        assert s.names == ("feed", "T")
        assert s.n_scenarios == 64
        assert float(jnp.min(s.column("feed"))) >= 0.8 - 1e-9


# =============================================================================
# Risk measures
# =============================================================================


class TestRisk:
    z = jnp.arange(10.0)
    w = jnp.full((10,), 0.1)

    def test_expectation(self):
        assert float(st.Expectation().value(self.z, self.w, jnp.zeros(0))) == \
            pytest.approx(4.5)

    def test_cvar_is_the_tail_mean(self):
        """CVaR_0.8 of 0..9 is the mean of {8, 9}."""
        c = st.CVaR(0.8)
        aux = c.exact_aux(self.z, self.w)
        assert float(c.value(self.z, self.w, aux)) == pytest.approx(8.5)

    def test_the_rockafellar_uryasev_optimum_is_where_exact_aux_says(self):
        c = st.CVaR(0.8)
        best = float(c.value(self.z, self.w, c.exact_aux(self.z, self.w)))
        ts = jnp.linspace(-2.0, 12.0, 4001)
        grid = jax.vmap(lambda t: c.value(self.z, self.w, jnp.array([t])))(ts)
        assert best == pytest.approx(float(jnp.min(grid)), abs=1e-9)

    def test_exact_aux_traces(self):
        """It is called inside the objective, so it must take a tracer."""
        c = st.CVaR(0.9)
        f = jax.jit(lambda z: c.value(z, self.w, c.exact_aux(z, self.w)))
        assert float(f(self.z)) == pytest.approx(9.0)

    def test_the_surrogate_converges_to_the_exact_value(self):
        c = st.CVaR(0.8)
        aux = c.exact_aux(self.z, self.w)
        exact = float(c.value(self.z, self.w, aux))
        coarse = float(c.surrogate(self.z, self.w, aux, 1.0))
        fine = float(c.surrogate(self.z, self.w, aux, 1e-9))
        assert fine == pytest.approx(exact, abs=1e-6)
        assert coarse > exact                     # smoothing is from above

    def test_cvar_alpha_is_the_first_keyword_not_n_aux(self):
        """A positional argument must reach alpha; n_aux is not a field."""
        assert st.CVaR(0.8).alpha == 0.8
        assert st.CVaR(0.8).n_aux == 1
        assert st.as_risk_measure(("cvar", 0.9)).alpha == 0.9

    def test_cvar_rejects_a_degenerate_alpha(self):
        with pytest.raises(ValueError, match="strictly in"):
            st.CVaR(1.0)

    def test_worst_case_and_mean_std(self):
        assert float(st.WorstCase().value(self.z, self.w, jnp.zeros(0))) == 9.0
        m = st.MeanStd(2.0)
        sd = float(jnp.std(self.z))
        assert float(m.value(self.z, self.w, jnp.zeros(0))) == \
            pytest.approx(4.5 + 2.0 * sd)

    def test_every_risk_measure_is_translation_and_scale_equivariant(self):
        """The solver rescales the objective; that is only exact if this holds."""
        a, b = 3.0, 7.0
        for rm in (st.Expectation(), st.MeanStd(1.5), st.CVaR(0.8),
                   st.WorstCase()):
            base = float(rm.value(self.z, self.w, rm.exact_aux(self.z, self.w)))
            zz = a + b * self.z
            got = float(rm.value(zz, self.w, rm.exact_aux(zz, self.w)))
            assert got == pytest.approx(a + b * base, rel=1e-12)

    def test_weighted_quantile_respects_the_weights(self):
        z = jnp.array([0.0, 1.0, 2.0])
        assert float(weighted_quantile(z, jnp.array([0.98, 0.01, 0.01]),
                                       0.5)) == 0.0
        assert float(weighted_quantile(z, jnp.array([0.01, 0.01, 0.98]),
                                       0.5)) == 2.0


class TestConstraints:
    z = jnp.array([0.0, 1.0, 2.0, 3.0])
    w = jnp.full((4,), 0.25)

    def test_signed_residual_convention(self):
        assert float(st.Expected("g", "<=", 2.0).signed(jnp.array([3.0]))[0]) \
            == 1.0
        assert float(st.Expected("g", ">=", 2.0).signed(jnp.array([1.0]))[0]) \
            == 1.0

    def test_expected_robust_and_chance_order_as_they_should(self):
        """Robust is the strictest, expectation the loosest."""
        args = ("g", "<=", 1.0)
        e = st.Expected(*args).residual(self.z, self.w, jnp.zeros(0))
        r = st.Robust(*args).residual(self.z, self.w, jnp.zeros(0))
        c = st.Chance(*args, alpha=0.75)
        cv = c.residual(self.z, self.w, c.exact_aux(self.z, self.w))
        assert float(e) <= float(cv) <= float(r)

    def test_the_chance_surrogate_is_conservative(self):
        """CVaR <= 0 implies the probability; the converse need not hold."""
        c = st.Chance("g", "<=", 2.5, alpha=0.75)
        # 3 of 4 scenarios satisfy g <= 2.5, so the probability holds exactly.
        assert c.violation_rate(self.z) == pytest.approx(0.25)
        res = float(c.residual(self.z, self.w, c.exact_aux(self.z, self.w)))
        assert res > 0.0                          # yet the surrogate refuses it

    def test_positive_rescaling_of_the_residual_keeps_the_sign(self):
        """The solver scales residuals; that is only safe if this holds."""
        for c in (st.Expected("g", "<=", 1.0), st.Robust("g", "<=", 1.0),
                  st.Chance("g", "<=", 1.0, alpha=0.8)):
            r = c.signed(self.z)
            a = float(c.residual_from_signed(
                r, self.w, c.exact_aux_from_signed(r, self.w)))
            b = float(c.residual_from_signed(
                r / 17.0, self.w, c.exact_aux_from_signed(r / 17.0, self.w)))
            assert np.sign(a) == np.sign(b)
            assert b == pytest.approx(a / 17.0, rel=1e-10)

    def test_shorthands(self):
        assert isinstance(st.as_constraint(("p", ">=", 1.0)), st.Expected)
        c = st.as_constraint(("p", ">=", 1.0, 0.9))
        assert isinstance(c, st.Chance) and c.alpha == 0.9

    def test_a_bad_operator_is_refused(self):
        with pytest.raises(ValueError, match="operator"):
            st.Expected("g", "<", 1.0)

    def test_a_chance_equality_is_refused(self):
        with pytest.raises(ValueError, match="probability zero"):
            st.Chance("g", "==", 1.0, alpha=0.9)


# =============================================================================
# The problem statement
# =============================================================================


def quadratic(x, u, theta):
    """A model with a known optimum: x* is the sample mean, u* tracks theta."""
    return {"cost": (x["x"] - theta["t"]) ** 2 + (u["u"] - theta["t"]) ** 2,
            "level": x["x"] - theta["t"]}


@pytest.fixture(scope="module")
def scen():
    return st.ScenarioSet.normal({"t": (5.0, 1.0)}, n=64, seed=0)


class TestProblem:
    def test_a_variable_cannot_be_in_both_stages(self):
        with pytest.raises(ValueError, match="both the first stage"):
            st.TwoStageProblem(model=quadratic, first_stage={"x": (0.0, 1.0)},
                               recourse={"x": (0.0, 1.0)}, objective="cost")

    def test_a_problem_with_no_first_stage_is_refused(self):
        with pytest.raises(ValueError, match="at least one first-stage"):
            st.TwoStageProblem(model=quadratic, first_stage={},
                               objective="cost")

    def test_a_missing_objective_output_says_what_the_model_returned(self, scen):
        p = st.TwoStageProblem(model=quadratic, first_stage={"x": (0.0, 10.0)},
                               recourse={"u": (0.0, 10.0)}, objective="profit")
        with pytest.raises(KeyError, match="no output named"):
            p.objective_values(p.outputs(jnp.array([1.0]),
                                         jnp.zeros((scen.n_scenarios, 1)),
                                         scen))

    def test_a_model_returning_a_bare_value_is_refused(self, scen):
        p = st.TwoStageProblem(model=lambda x, u, th: x["x"],
                               first_stage={"x": (0.0, 1.0)}, objective="cost")
        with pytest.raises(TypeError, match="mapping of named outputs"):
            p.outputs(jnp.array([1.0]), jnp.zeros((scen.n_scenarios, 0)), scen)

    def test_outputs_are_batched_over_scenarios(self, scen):
        p = st.TwoStageProblem(model=quadratic, first_stage={"x": (0.0, 10.0)},
                               recourse={"u": (0.0, 10.0)}, objective="cost")
        out = p.outputs(jnp.array([5.0]), jnp.full((scen.n_scenarios, 1), 5.0),
                        scen)
        assert out["cost"].shape == (scen.n_scenarios,)

    def test_maximize_flips_the_sense_and_flips_it_back(self, scen):
        p = st.TwoStageProblem(model=quadratic, first_stage={"x": (0.0, 10.0)},
                               recourse={"u": (0.0, 10.0)}, objective="cost",
                               maximize=True)
        out = p.outputs(jnp.array([1.0]), jnp.ones((scen.n_scenarios, 1)), scen)
        np.testing.assert_allclose(np.asarray(p.objective_values(out)),
                                   -np.asarray(out["cost"]))


# =============================================================================
# The solve: every case here has a closed form
# =============================================================================


class TestSolve:
    def test_the_first_stage_lands_on_the_sample_mean(self, scen):
        """min_x E[(x - t)^2] is the sample mean, to machine precision."""
        p = st.TwoStageProblem(model=quadratic, first_stage={"x": (0.0, 10.0)},
                               recourse={"u": (0.0, 10.0)}, objective="cost")
        res = st.solve_saa(p, scen, options=FAST)
        assert res.first_stage["x"] == pytest.approx(
            float(jnp.mean(scen.column("t"))), abs=1e-6)

    def test_the_recourse_tracks_each_scenario(self, scen):
        """Non-anticipativity: u sees theta, x does not."""
        p = st.TwoStageProblem(model=quadratic, first_stage={"x": (0.0, 10.0)},
                               recourse={"u": (0.0, 10.0)}, objective="cost")
        res = st.solve_saa(p, scen, options=st.SAAOptions(steps=500, rounds=1,
                                                          n_starts=1))
        np.testing.assert_allclose(res.recourse["u"],
                                   np.asarray(scen.column("t")), atol=1e-6)

    def test_the_newsvendor_finds_its_critical_quantile(self):
        """min_x E[b(t-x)_+ + h(x-t)_+] is the b/(b+h) quantile of t."""
        b, h = 4.0, 1.0

        def news(x, u, theta):
            s = x["x"] - theta["t"]
            return {"cost": b * jnp.maximum(-s, 0.0) + h * jnp.maximum(s, 0.0)}

        sample = st.ScenarioSet.normal({"t": (5.0, 1.0)}, n=512, seed=3)
        p = st.TwoStageProblem(model=news, first_stage={"x": (0.0, 12.0)},
                               objective="cost")
        res = st.solve_saa(p, sample, options=st.SAAOptions(steps=400,
                                                            n_starts=1))
        target = float(np.quantile(np.asarray(sample.column("t")), b / (b + h)))
        assert res.first_stage["x"] == pytest.approx(target, abs=5e-3)

    def test_a_chance_constraint_binds_at_its_cvar(self):
        """min x s.t. P(x >= t) >= alpha lands on CVaR_alpha(t), the surrogate."""
        sample = st.ScenarioSet.normal({"t": (5.0, 1.0)}, n=64, seed=0)

        def m(x, u, theta):
            return {"cost": x["x"], "margin": x["x"] - theta["t"]}

        p = st.TwoStageProblem(model=m, first_stage={"x": (0.0, 12.0)},
                               objective="cost",
                               constraints=[("margin", ">=", 0.0, 0.9)])
        res = st.solve_saa(p, sample, options=st.SAAOptions(steps=400,
                                                            n_starts=1))
        c = st.CVaR(0.9)
        t = sample.column("t")
        target = float(c.value(t, sample.weights, c.exact_aux(t,
                                                              sample.weights)))
        assert res.first_stage["x"] == pytest.approx(target, abs=1e-4)
        assert res.feasible
        assert res.violation_rates[0] <= 0.10

    def test_a_growing_penalty_would_have_frozen_here(self):
        """Regression: the augmented Lagrangian, not a penalty continuation.

        With a quadratic penalty raised to 1e4, a projected-Adam iterate that
        crosses the infeasible region has its second moment inflated so far
        that every later step is negligible, and it stops several units from
        the optimum while reporting itself feasible.  The multipliers keep rho
        at 10 and the answer is exact, so a large miss here means that
        regression is back.
        """
        sample = st.ScenarioSet.normal({"t": (5.0, 1.0)}, n=64, seed=0)

        def m(x, u, theta):
            return {"cost": x["x"], "margin": x["x"] - theta["t"]}

        p = st.TwoStageProblem(model=m, first_stage={"x": (0.0, 12.0)},
                               objective="cost",
                               constraints=[("margin", ">=", 0.0, 0.9)])
        res = st.solve_saa(p, sample, options=st.SAAOptions(steps=400))
        assert res.first_stage["x"] < 7.0        # the frozen answer was 9.7

    def test_evaluate_matches_a_hand_computation(self, scen):
        p = st.TwoStageProblem(model=quadratic, first_stage={"x": (0.0, 10.0)},
                               recourse={"u": (0.0, 10.0)}, objective="cost")
        t = np.asarray(scen.column("t"))
        res = st.evaluate(p, {"x": 3.0}, {"u": t}, scen)
        assert res.risk_value == pytest.approx(float(np.mean((3.0 - t) ** 2)))

    def test_solve_recourse_holds_the_design_where_it_was_put(self, scen):
        p = st.TwoStageProblem(model=quadratic, first_stage={"x": (0.0, 10.0)},
                               recourse={"u": (0.0, 10.0)}, objective="cost")
        res = st.solve_recourse(p, scen, {"x": 3.0}, options=FAST)
        assert res.first_stage["x"] == pytest.approx(3.0, abs=1e-12)
        assert res.risk_value == pytest.approx(
            float(np.mean((3.0 - np.asarray(scen.column("t"))) ** 2)), rel=1e-4)

    def test_a_first_stage_naming_an_undeclared_decision_is_refused(self, scen):
        p = st.TwoStageProblem(model=quadratic, first_stage={"x": (0.0, 10.0)},
                               recourse={"u": (0.0, 10.0)}, objective="cost")
        with pytest.raises(KeyError, match="not declared decisions"):
            st.evaluate(p, {"x": 1.0, "y": 2.0}, None, scen)

    def test_scaling_does_not_move_the_answer(self, scen):
        """The same problem in different units must give the same design."""
        def big(x, u, theta):
            return {"cost": 1e6 * (x["x"] - theta["t"]) ** 2}

        p1 = st.TwoStageProblem(model=quadratic, first_stage={"x": (0.0, 10.0)},
                                recourse={"u": (0.0, 10.0)}, objective="cost")
        p2 = st.TwoStageProblem(model=big, first_stage={"x": (0.0, 10.0)},
                                objective="cost")
        a = st.solve_saa(p1, scen, options=FAST).first_stage["x"]
        b = st.solve_saa(p2, scen, options=FAST).first_stage["x"]
        assert a == pytest.approx(b, abs=1e-5)

    def test_summary_runs_and_names_the_pieces(self, scen):
        p = st.TwoStageProblem(model=quadratic, first_stage={"x": (0.0, 10.0)},
                               recourse={"u": (0.0, 10.0)}, objective="cost",
                               risk=("cvar", 0.8),
                               constraints=[("level", "<=", 2.0, 0.9)])
        text = st.solve_saa(p, scen, options=FAST).summary()
        assert "CVaR_0.8" in text and "recourse" in text and "level" in text


# =============================================================================
# Diagnostics
# =============================================================================


class TestDiagnostics:
    def test_the_three_bounds_are_ordered(self, scen):
        """WS <= SP <= EEV is a theorem; a violation is a bug, not noise."""
        p = st.TwoStageProblem(model=quadratic, first_stage={"x": (0.0, 10.0)},
                               recourse={"u": (0.0, 10.0)}, objective="cost")
        rep = st.bounds(p, scen, options=FAST)
        assert rep.ordered
        assert rep.wait_and_see <= rep.stochastic + 1e-6
        assert rep.stochastic <= rep.expected_value + 1e-6
        assert rep.vss >= -1e-6 and rep.evpi >= -1e-6

    def test_the_bounds_order_holds_when_maximizing_too(self, scen):
        def profit(x, u, theta):
            return {"p": -(x["x"] - theta["t"]) ** 2}

        p = st.TwoStageProblem(model=profit, first_stage={"x": (0.0, 10.0)},
                               objective="p", maximize=True)
        rep = st.bounds(p, scen, options=FAST)
        assert rep.ordered
        assert rep.wait_and_see >= rep.stochastic - 1e-6
        assert rep.stochastic >= rep.expected_value - 1e-6

    def test_vss_is_zero_when_the_mean_value_design_is_already_right(self, scen):
        """min E[(x-t)^2] has its optimum at the mean, so nothing is bought."""
        def m(x, u, theta):
            return {"cost": (x["x"] - theta["t"]) ** 2}

        p = st.TwoStageProblem(model=m, first_stage={"x": (0.0, 10.0)},
                               objective="cost")
        assert st.value_of_stochastic_solution(p, scen, options=FAST) == \
            pytest.approx(0.0, abs=1e-4)

    def test_vss_is_positive_when_the_cost_is_asymmetric(self):
        """A newsvendor's mean-value design is the median, and that is wrong."""
        def news(x, u, theta):
            s = x["x"] - theta["t"]
            return {"cost": 9.0 * jnp.maximum(-s, 0.0)
                    + 1.0 * jnp.maximum(s, 0.0)}

        sample = st.ScenarioSet.normal({"t": (5.0, 1.0)}, n=256, seed=1)
        p = st.TwoStageProblem(model=news, first_stage={"x": (0.0, 12.0)},
                               objective="cost")
        rep = st.bounds(p, sample, options=st.SAAOptions(steps=400,
                                                         n_starts=1))
        assert rep.ordered and rep.vss > 0.5

    def test_perfect_information_is_worthless_when_recourse_can_do_it_all(
            self, scen):
        """u alone can absorb theta, so knowing theta early buys nothing."""
        def m(x, u, theta):
            return {"cost": (u["u"] - theta["t"]) ** 2 + 0.01 * x["x"] ** 2}

        p = st.TwoStageProblem(model=m, first_stage={"x": (0.0, 10.0)},
                               recourse={"u": (0.0, 10.0)}, objective="cost")
        rep = st.bounds(p, scen, options=FAST)
        assert rep.evpi == pytest.approx(0.0, abs=1e-4)

    def test_health_finds_a_dead_lever(self, scen):
        """A clip on the active path gives an exactly-zero gradient column."""
        def m(x, u, theta):
            dead = jnp.minimum(x["dead"], -5.0)    # never above -5, so no slope
            return {"cost": (x["x"] - theta["t"]) ** 2 + 0.0 * dead}

        p = st.TwoStageProblem(
            model=m, first_stage={"x": (0.0, 10.0), "dead": (0.0, 1.0)},
            objective="cost")
        res = st.solve_saa(p, scen, options=FAST)
        health = st.check_scenario_health(p, scen, res)
        assert "dead" in health.dead_levers
        assert not health.healthy

    def test_health_warns_about_an_empty_cvar_tail(self):
        sample = st.ScenarioSet.normal({"t": (5.0, 1.0)}, n=32, seed=0)

        def m(x, u, theta):
            return {"cost": (x["x"] - theta["t"]) ** 2}

        p = st.TwoStageProblem(model=m, first_stage={"x": (0.0, 10.0)},
                               objective="cost", risk=("cvar", 0.99))
        res = st.solve_saa(p, sample, options=FAST)
        health = st.check_scenario_health(p, sample, res)
        assert health.tail_scenarios < st.MIN_TAIL_SCENARIOS
        assert any("tail" in w for w in health.warnings)

    def test_health_warns_about_saturated_recourse(self, scen):
        """u is boxed far away from where every scenario wants it."""
        def m(x, u, theta):
            return {"cost": (u["u"] - theta["t"]) ** 2 + 0.0 * x["x"]}

        p = st.TwoStageProblem(model=m, first_stage={"x": (0.0, 1.0)},
                               recourse={"u": (0.0, 1.0)}, objective="cost")
        res = st.solve_saa(p, scen, options=FAST)
        health = st.check_scenario_health(p, scen, res)
        assert health.saturated.get("u", 0.0) > st.SATURATION_WARN
        assert any("saturated" in w for w in health.warnings)

    def test_the_optimality_gap_needs_a_variance(self, scen):
        p = st.TwoStageProblem(model=quadratic, first_stage={"x": (0.0, 10.0)},
                               recourse={"u": (0.0, 10.0)}, objective="cost")
        res = st.solve_saa(p, scen, options=FAST)
        with pytest.raises(ValueError, match="at least two replications"):
            st.optimality_gap(p, scen, res, n_replications=1)

    def test_the_optimality_gap_is_small_on_a_problem_solved_exactly(self, scen):
        def m(x, u, theta):
            return {"cost": (x["x"] - theta["t"]) ** 2}

        p = st.TwoStageProblem(model=m, first_stage={"x": (0.0, 10.0)},
                               objective="cost")
        res = st.solve_saa(p, scen, options=FAST)
        gap = st.optimality_gap(p, scen, res, n_replications=3,
                                n_evaluation=512, options=FAST)
        # The true optimum of E[(x-t)^2] is 1.0 (the variance); the SAA answer
        # is within sampling noise of it, so the gap must be small.
        assert abs(gap.gap) < 0.5
        assert "optimality gap" in gap.summary()
