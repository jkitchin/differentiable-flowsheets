"""Tests for Fisher-information experiment design and identifiability.

The checks here are deliberately external to the implementation: hand
arithmetic for the Fisher information and the criteria, a brute-force
enumeration of every design for the case where the D-optimal answer is
known analytically, and a Monte-Carlo refit that compares the *predicted*
covariance against the covariance actually obtained by fitting simulated
data from the designed runs.
"""

from itertools import combinations_with_replacement

import pytest
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from difflow.estimation import (
    CRITERIA,
    DesignResult,
    Estimator,
    Experiment,
    IdentifiabilityError,
    IdentifiabilityReport,
    check_identifiability,
    design_criterion,
    design_experiments,
    fisher_information,
    log_det,
    predicted_covariance,
    sensitivity_matrix,
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def linear_model(theta, exp):
    """y = a * x + b."""
    return {"y": theta["a"] * exp.inputs["x"] + theta["b"]}


def product_model(theta, exp):
    """y = (A * B) * x -- A and B are not separately identifiable."""
    return {"y": theta["A"] * theta["B"] * exp.inputs["x"]}


def sum_model(theta, exp):
    """y = (c + d) * x -- c and d are not separately identifiable."""
    return {"y": (theta["c"] + theta["d"]) * exp.inputs["x"]}


def arrhenius_model(theta, exp):
    """Nonlinear: y = A * exp(-E / T)."""
    return {"y": theta["A"] * jnp.exp(-theta["E"] / exp.inputs["T"])}


def two_input_model(theta, exp):
    """y = p*u + q*v -- two inputs, so a candidate is a direction in S-space."""
    return {"y": theta["p"] * exp.inputs["u"] + theta["q"] * exp.inputs["v"]}


def two_output_product_model(theta, exp):
    """A*B*x, plus a second output that resolves A on its own."""
    return {
        "y": theta["A"] * theta["B"] * exp.inputs["x"],
        "z": theta["A"] * exp.inputs["x"],
    }


def line_pool(xs, sigma=1.0):
    return [
        Experiment.candidate({"x": float(x)}, ["y"], {"y": sigma}, name=f"x{i}")
        for i, x in enumerate(xs)
    ]


# ---------------------------------------------------------------------------
# Experiment as a designable object
# ---------------------------------------------------------------------------

class TestCandidateExperiment:
    def test_candidate_has_no_observations(self):
        c = Experiment.candidate({"x": 1.0}, ["y"], {"y": 0.5})
        assert c.is_candidate
        assert c.observed == {}
        assert c.measured_names == ["y"]
        assert float(c.sigma_array[0]) == 0.5

    def test_recorded_experiment_still_reports_its_outputs(self):
        exp = Experiment(inputs={"x": 1.0}, observed={"y": 2.0, "z": 3.0})
        assert not exp.is_candidate
        # Backward compatible: measured_names falls back to observed keys.
        assert exp.measured_names == exp.output_names == ["y", "z"]
        assert np.allclose(np.asarray(exp.sigma_array), [1.0, 1.0])

    def test_missing_uncertainty_defaults_to_one(self):
        c = Experiment.candidate({"x": 1.0}, ["y", "z"], {"y": 0.25})
        assert np.allclose(np.asarray(c.sigma_array), [0.25, 1.0])


# ---------------------------------------------------------------------------
# Sensitivity matrix and Fisher information vs hand arithmetic
# ---------------------------------------------------------------------------

class TestSensitivityAndFIM:
    def test_sensitivity_matrix_of_a_line(self):
        # d(a x + b)/da = x, d/db = 1.
        exps = line_pool([1.0, 3.0])
        s = np.asarray(
            sensitivity_matrix(linear_model, {"a": 2.0, "b": 1.0}, exps)
        )
        assert np.allclose(s, [[1.0, 1.0], [3.0, 1.0]])

    def test_sensitivity_matrix_is_weighted_by_sigma(self):
        exps = [Experiment.candidate({"x": 2.0}, ["y"], {"y": 0.5})]
        s = np.asarray(
            sensitivity_matrix(linear_model, {"a": 1.0, "b": 0.0}, exps)
        )
        assert np.allclose(s, [[2.0 / 0.5, 1.0 / 0.5]])
        raw = np.asarray(
            sensitivity_matrix(
                linear_model, {"a": 1.0, "b": 0.0}, exps, weighted=False
            )
        )
        assert np.allclose(raw, [[2.0, 1.0]])

    def test_fim_matches_hand_arithmetic(self):
        # Two runs, x = 1 and 2, sigma = 0.5 and 2.0.
        #   row 1 weight 1/0.25 = 4  -> 4 * [[1, 1], [1, 1]]
        #   row 2 weight 1/4    = .25 -> .25 * [[4, 2], [2, 1]]
        exps = [
            Experiment.candidate({"x": 1.0}, ["y"], {"y": 0.5}),
            Experiment.candidate({"x": 2.0}, ["y"], {"y": 2.0}),
        ]
        fim = np.asarray(fisher_information(linear_model, {"a": 3.0, "b": -1.0}, exps))
        expected = np.array([[5.0, 4.5], [4.5, 4.25]])
        assert np.allclose(fim, expected)

    def test_sigmas_line_up_with_ragged_vector_outputs(self):
        """A scalar output and a length-3 output, with different sigmas.

        Each output's declared uncertainty must be broadcast across *its
        own* rows. Splitting the rows evenly between the two outputs (the
        obvious ``n_rows // n_outputs`` guess) puts the scalar output's
        sigma on the first two rows of the vector one.
        """

        def ragged(theta, exp):
            x = exp.inputs["x"]
            return {
                "s": theta["a"] * x,
                "v": theta["b"] * jnp.array([x, 2.0 * x, 3.0 * x]),
            }

        exp = Experiment.candidate({"x": 1.0}, ["s", "v"], {"s": 0.5, "v": 0.25})
        s = np.asarray(sensitivity_matrix(ragged, {"a": 1.0, "b": 1.0}, [exp]))
        # rows: s (sigma 0.5), then the three entries of v (sigma 0.25).
        assert np.allclose(
            s,
            [[1.0 / 0.5, 0.0], [0.0, 1.0 / 0.25], [0.0, 2.0 / 0.25],
             [0.0, 3.0 / 0.25]],
        )
        fim = np.asarray(fisher_information(ragged, {"a": 1.0, "b": 1.0}, [exp]))
        # (1/0.5)^2 = 4 and (1 + 4 + 9)/0.25^2 = 224.
        assert np.allclose(fim, [[4.0, 0.0], [0.0, 224.0]])

    def test_fim_is_additive_over_experiments(self):
        pool = line_pool([0.0, 1.0, 2.0, 3.0], sigma=0.7)
        theta = {"a": 1.0, "b": 2.0}
        whole = np.asarray(fisher_information(linear_model, theta, pool))
        parts = sum(
            np.asarray(fisher_information(linear_model, theta, [e])) for e in pool
        )
        assert np.allclose(whole, parts)

    def test_prior_fim_is_added(self):
        pool = line_pool([1.0, 2.0])
        theta = {"a": 1.0, "b": 0.0}
        prior = np.array([[3.0, 0.0], [0.0, 5.0]])
        with_prior = np.asarray(
            fisher_information(linear_model, theta, pool, prior_fim=prior)
        )
        without = np.asarray(fisher_information(linear_model, theta, pool))
        assert np.allclose(with_prior - without, prior)

    def test_fim_does_not_depend_on_measured_values(self):
        # The whole point of design: the FIM is known before the run.
        theta = {"a": 2.0, "b": 1.0}
        cand = Experiment.candidate({"x": 1.5}, ["y"], {"y": 0.3})
        run = Experiment(
            inputs={"x": 1.5}, observed={"y": -99.0}, uncertainties={"y": 0.3}
        )
        assert np.allclose(
            np.asarray(fisher_information(linear_model, theta, [cand])),
            np.asarray(fisher_information(linear_model, theta, [run])),
        )


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

class TestCriteria:
    def test_diagonal_fim_hand_arithmetic(self):
        fim = jnp.diag(jnp.array([4.0, 1.0]))
        assert float(design_criterion(fim, "D")) == pytest.approx(np.log(4.0))
        assert float(design_criterion(fim, "A")) == pytest.approx(1.25)
        assert float(design_criterion(fim, "E")) == pytest.approx(1.0)
        assert float(design_criterion(fim, "ME")) == pytest.approx(4.0)

    def test_correlated_fim_hand_arithmetic(self):
        # [[2, 1], [1, 2]] has eigenvalues 1 and 3, det 3, inverse trace 4/3.
        fim = jnp.array([[2.0, 1.0], [1.0, 2.0]])
        assert float(design_criterion(fim, "D")) == pytest.approx(np.log(3.0))
        assert float(design_criterion(fim, "A")) == pytest.approx(4.0 / 3.0)
        assert float(design_criterion(fim, "E")) == pytest.approx(1.0)
        assert float(design_criterion(fim, "ME")) == pytest.approx(3.0)

    def test_log_det_agrees_with_slogdet_but_not_via_det(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal((5, 5))
        fim = jnp.asarray(a @ a.T + 5.0 * np.eye(5))
        sign, ref = jnp.linalg.slogdet(fim)
        assert float(sign) == 1.0
        assert float(log_det(fim)) == pytest.approx(float(ref), rel=1e-10)

    def test_log_det_survives_scaling_that_overflows_det(self):
        # det of this matrix is 1e-260**8; log_det stays finite.
        fim = jnp.eye(8) * 1e-260
        assert float(log_det(fim)) == pytest.approx(8 * np.log(1e-260))
        assert float(jnp.linalg.det(fim)) == 0.0  # the naive route has failed

    def test_a_badly_scaled_singular_fim_is_not_reported_as_finite(self):
        """A rank-1 FIM whose diagonal spans seven decades is still rank 1.

        Regression test. A *relative* floor on the pivots of the Cholesky
        factor cannot see this: the pivots inherit the scale spread of the
        diagonal, so the second pivot (~1e-7) sits far above ``sqrt(eps)``
        times the first (~6e-3) even though the matrix is exactly singular.
        The old code returned a finite -41.6 here, and greedy selection then
        happily built a design of three replicates at one temperature --
        which determines exactly one combination of two parameters.
        """
        theta = {"A": 1e6, "E": 4000.0}
        replicates = [
            Experiment.candidate({"T": 500.0}, ["y"], {"y": 0.1}) for _ in range(3)
        ]
        fim = np.asarray(fisher_information(arrhenius_model, theta, replicates))
        # Rank 1 by construction: three copies of the same S row.
        assert np.linalg.matrix_rank(fim) == 1
        assert fim[1, 1] / fim[0, 0] > 1e6  # and badly scaled
        assert float(log_det(jnp.asarray(fim))) == -np.inf
        assert float(design_criterion(jnp.asarray(fim), "D")) == -np.inf
        assert float(design_criterion(jnp.asarray(fim), "A")) == np.inf

    def test_cholesky_success_is_not_taken_for_a_rank_test(self):
        """A FIM indistinguishable from singular must not score finitely.

        ``[[1, rho], [rho, 1]]`` with ``rho = 1 - 2*eps`` factors happily --
        the second Cholesky pivot is ``sqrt(1 - rho^2) = 2.1e-8``, positive
        and finite -- and both ``det`` and ``slogdet`` report ``log det =
        -35.35``. Its smallest eigenvalue is 2.8e-16, below the ``n*eps*
        lambda_max`` floor, so every criterion must take the singular limit
        instead: that "design" determines one direction, not two.
        """
        rho = 1.0 - 2e-16
        fim = jnp.array([[1.0, rho], [rho, 1.0]])
        assert rho != 1.0  # the near-degeneracy is representable
        assert np.isfinite(np.linalg.slogdet(np.asarray(fim))[1])  # naive route
        assert float(log_det(fim)) == -np.inf
        assert float(design_criterion(fim, "D")) == -np.inf
        assert float(design_criterion(fim, "A")) == np.inf
        assert float(design_criterion(fim, "E")) == 0.0

    def test_an_ill_conditioned_but_full_rank_fim_stays_usable(self):
        """The other side of the same tolerance: cond 1e10 is not singular.

        Two Arrhenius runs at different temperatures determine both
        parameters, badly but genuinely. A ``sqrt(eps)`` floor on the FIM's
        eigenvalues (rather than ``eps``, their being squares of the
        singular values of S) would call this singular and the design would
        walk away from the only informative pairs in the pool.
        """
        theta = {"A": 1e6, "E": 4000.0}
        exps = [
            Experiment.candidate({"T": 450.0}, ["y"], {"y": 0.1}),
            Experiment.candidate({"T": 500.0}, ["y"], {"y": 0.1}),
        ]
        fim = np.asarray(fisher_information(arrhenius_model, theta, exps))
        w = np.linalg.eigvalsh(fim)
        assert 1e8 < w[-1] / w[0] < 1e13  # genuinely ill-conditioned
        assert np.isfinite(float(log_det(jnp.asarray(fim))))
        # ... and it agrees with the structural rank test, which is the
        # point of tying the two tolerances together.
        assert check_identifiability(arrhenius_model, theta, exps).identifiable

    def test_singular_fim_gives_the_right_limits(self):
        fim = jnp.array([[1.0, 1.0], [1.0, 1.0]])  # rank 1
        assert float(log_det(fim)) == -np.inf
        assert float(design_criterion(fim, "D")) == -np.inf
        assert float(design_criterion(fim, "A")) == np.inf
        assert float(design_criterion(fim, "E")) == 0.0
        assert float(design_criterion(fim, "ME")) == np.inf

    def test_unknown_criterion_raises(self):
        with pytest.raises(ValueError, match="unknown criterion"):
            design_criterion(jnp.eye(2), "Z")


# ---------------------------------------------------------------------------
# Design selection
# ---------------------------------------------------------------------------

def _brute_force_best(model_fn, theta, pool, n, criterion):
    """Best design by exhaustive enumeration (with replacement)."""
    contrib = [
        np.asarray(fisher_information(model_fn, theta, [e])) for e in pool
    ]
    best_val, best_combo = None, None
    minimize = criterion in ("A", "ME")
    for combo in combinations_with_replacement(range(len(pool)), n):
        fim = sum(contrib[i] for i in combo)
        val = float(design_criterion(jnp.asarray(fim), criterion))
        if not np.isfinite(val):
            continue
        if best_val is None or (val < best_val if minimize else val > best_val):
            best_val, best_combo = val, combo
    return best_val, best_combo


class TestDesignSelection:
    def test_d_optimal_line_is_the_endpoints(self):
        # For y = a x + b on a bounded interval the D-optimal design is
        # known analytically: half the runs at each end point.
        pool = line_pool(np.linspace(0.0, 10.0, 11))
        res = design_experiments(linear_model, {"a": 1.0, "b": 0.0}, pool, n=4)
        assert isinstance(res, DesignResult)
        assert sorted(e.inputs["x"] for e in res.selected) == [0.0, 0.0, 10.0, 10.0]

        # And the criterion value is the hand-computed one:
        # FIM = [[sum x^2, sum x], [sum x, n]] = [[200, 20], [20, 4]],
        # det = 800 - 400 = 400.
        assert np.allclose(np.asarray(res.fim), [[200.0, 20.0], [20.0, 4.0]])
        assert res.criterion_value == pytest.approx(np.log(400.0))

    def test_design_result_covariance_is_the_inverse_fim_by_hand(self):
        """``covariance`` and ``std_errors`` are inv(FIM), not 1/diag(FIM).

        The D-optimal four-run design above has FIM [[200, 20], [20, 4]],
        det 400, so inv(FIM) = [[0.01, -0.05], [-0.05, 0.5]] and the
        standard errors are 0.1 and sqrt(0.5). Reading the variances off
        the FIM's own diagonal instead would give 1/sqrt(200) = 0.0707 and
        1/2 -- both plausible-looking, both wrong, and both blind to the
        parameter correlation that is the whole reason the design matters.
        """
        pool = line_pool(np.linspace(0.0, 10.0, 11))
        res = design_experiments(linear_model, {"a": 1.0, "b": 0.0}, pool, n=4)
        expected = np.array([[0.01, -0.05], [-0.05, 0.5]])
        assert np.allclose(np.asarray(res.covariance), expected)
        assert res.std_errors["a"] == pytest.approx(0.1)
        assert res.std_errors["b"] == pytest.approx(np.sqrt(0.5))
        # inv(FIM) really is the inverse, not something that merely looks it.
        assert np.allclose(np.asarray(res.covariance) @ np.asarray(res.fim),
                           np.eye(2), atol=1e-12)
        # ... and it is the same number predicted_covariance reports for the
        # very same runs, so summary() and the CI table cannot disagree.
        ci = predicted_covariance(linear_model, {"a": 1.0, "b": 0.0}, res.selected)
        assert ci.std_errors["a"] == pytest.approx(res.std_errors["a"])
        assert ci.std_errors["b"] == pytest.approx(res.std_errors["b"])

    def test_greedy_matches_exhaustive_enumeration(self):
        pool = line_pool(np.linspace(0.0, 10.0, 11))
        theta = {"a": 1.0, "b": 0.0}
        for criterion in ("D", "A", "E"):
            res = design_experiments(
                linear_model, theta, pool, n=4, criterion=criterion
            )
            best_val, _ = _brute_force_best(linear_model, theta, pool, 4, criterion)
            assert res.criterion_value == pytest.approx(best_val, rel=1e-9)

    def test_exchange_escapes_a_greedy_trap(self):
        """A three-candidate pool where greedy is provably suboptimal.

        Sensitivity rows (sigma = 1): A = [1.7, 0], B = [1, 1], C = [1, -1].
        Greedy's first pick maximizes the one nonzero eigenvalue of a rank-1
        FIM: A gives 1.7^2 = 2.89 and beats B and C, which give 2. It is then
        stuck -- adding B or C to A gives

            FIM = [[2.89 + 1, 1], [1, 1]],  det = 3.89 - 1 = 2.89,

        whereas the pair it passed over gives

            FIM = [[2, 0], [0, 2]],  det = 4.

        One Fedorov swap (A out, C in) finds it. The whole construction turns
        on 1.7 lying in (sqrt(2), 2): above sqrt(2) so greedy takes the bait,
        below 2 so the bait is wrong.
        """
        theta = {"p": 1.0, "q": 1.0}
        pool = [
            Experiment.candidate({"u": 1.7, "v": 0.0}, ["y"], {"y": 1.0}, name="A"),
            Experiment.candidate({"u": 1.0, "v": 1.0}, ["y"], {"y": 1.0}, name="B"),
            Experiment.candidate({"u": 1.0, "v": -1.0}, ["y"], {"y": 1.0}, name="C"),
        ]
        greedy = design_experiments(two_input_model, theta, pool, n=2)
        assert sorted(e.name for e in greedy.selected) == ["A", "B"]
        assert greedy.criterion_value == pytest.approx(np.log(2.89))

        exch = design_experiments(
            two_input_model, theta, pool, n=2, method="exchange"
        )
        assert sorted(e.name for e in exch.selected) == ["B", "C"]
        assert exch.criterion_value == pytest.approx(np.log(4.0))
        assert exch.n_exchanges == 1
        assert np.allclose(np.asarray(exch.fim), [[2.0, 0.0], [0.0, 2.0]])

        best_val, _ = _brute_force_best(two_input_model, theta, pool, 2, "D")
        assert exch.criterion_value == pytest.approx(best_val)

    def test_exchange_is_no_worse_than_greedy_on_a_nonlinear_model(self):
        theta = {"A": 1e6, "E": 4000.0}
        pool = [
            Experiment.candidate({"T": float(T)}, ["y"], {"y": 0.1}, name=f"T{T:.0f}")
            for T in np.linspace(300.0, 500.0, 21)
        ]
        greedy = design_experiments(arrhenius_model, theta, pool, n=5)
        exch = design_experiments(
            arrhenius_model, theta, pool, n=5, method="exchange"
        )
        assert exch.criterion_value >= greedy.criterion_value - 1e-9
        best_val, _ = _brute_force_best(arrhenius_model, theta, pool, 5, "D")
        assert exch.criterion_value <= best_val + 1e-9
        # Greedy itself reaches the optimum on this pool -- an ill-conditioned
        # FIM (cond ~ 1e10) must not be mistaken for a singular one, or the
        # selection settles for replicates at a single temperature.
        assert greedy.criterion_value == pytest.approx(best_val, rel=1e-9)
        assert len({e.inputs["T"] for e in greedy.selected}) >= 2

    def test_each_criterion_picks_its_own_hand_computed_design(self):
        """Four criteria, four different answers, all checkable by hand.

        y = a x + b with x in {0, ..., 10}, sigma = 1, four runs. Writing
        the FIM of a design as [[sum x^2, sum x], [sum x, 4]]:

        - D maximizes the determinant: [0,0,10,10] -> [[200,20],[20,4]],
          det 400. (3-1 gives det 300.)
        - A minimizes trace(FIM^-1) = trace/det: [0,0,0,10] ->
          [[100,10],[10,4]], (100+4)/300 = 26/75. (2-2 gives 204/400 = 0.51.)
        - E maximizes the smallest eigenvalue: same 3-1 design, eigenvalues
          (104 -+ sqrt(104^2 - 4*300))/2, so lambda_min = 52 - sqrt(2404).
        - ME minimizes the condition number, which wants a *round* FIM, not
          an informative one: [0,0,1,2] -> [[5,3],[3,4]], eigenvalues
          (9 -+ sqrt(37))/2, ratio (9+sqrt(37))/(9-sqrt(37)).
        """
        theta = {"a": 1.0, "b": 0.0}
        pool = line_pool(np.linspace(0.0, 10.0, 11))
        picks = {}
        values = {}
        for c in CRITERIA:
            res = design_experiments(linear_model, theta, pool, n=4, criterion=c)
            picks[c] = sorted(e.inputs["x"] for e in res.selected)
            values[c] = res.criterion_value

        assert picks["D"] == [0.0, 0.0, 10.0, 10.0]
        assert picks["A"] == [0.0, 0.0, 0.0, 10.0]
        assert picks["E"] == [0.0, 0.0, 0.0, 10.0]
        assert picks["ME"] == [0.0, 0.0, 1.0, 2.0]

        assert values["D"] == pytest.approx(np.log(400.0))
        assert values["A"] == pytest.approx(26.0 / 75.0)
        assert values["E"] == pytest.approx(52.0 - np.sqrt(2404.0))
        assert values["ME"] == pytest.approx(
            (9.0 + np.sqrt(37.0)) / (9.0 - np.sqrt(37.0))
        )

    def test_criterion_history_improves_monotonically_once_full_rank(self):
        pool = line_pool(np.linspace(0.0, 10.0, 11))
        res = design_experiments(linear_model, {"a": 1.0, "b": 0.0}, pool, n=6)
        finite = [v for v in res.criterion_history if np.isfinite(v)]
        assert len(finite) >= 4
        assert all(b >= a - 1e-9 for a, b in zip(finite, finite[1:]))

    def test_existing_experiments_shift_the_design(self):
        pool = line_pool(np.linspace(0.0, 10.0, 11))
        theta = {"a": 1.0, "b": 0.0}
        # Already ran three replicates at x = 10; the next runs should go
        # to the other end.
        existing = line_pool([10.0, 10.0, 10.0])
        res = design_experiments(linear_model, theta, pool, n=2, existing=existing)
        assert sorted(e.inputs["x"] for e in res.selected) == [0.0, 0.0]
        # The reported FIM includes the existing information.
        assert np.asarray(res.fim)[1, 1] == pytest.approx(5.0)

    def test_replace_false_gives_distinct_runs(self):
        pool = line_pool(np.linspace(0.0, 10.0, 11))
        res = design_experiments(
            linear_model, {"a": 1.0, "b": 0.0}, pool, n=4, replace=False
        )
        assert len(set(res.indices)) == 4
        with pytest.raises(ValueError, match="without replacement"):
            design_experiments(
                linear_model, {"a": 1.0, "b": 0.0}, pool, n=99, replace=False
            )

    def test_empty_pool_and_bad_n_raise(self):
        with pytest.raises(ValueError, match="empty"):
            design_experiments(linear_model, {"a": 1.0, "b": 0.0}, [], n=2)
        with pytest.raises(ValueError, match="n must be positive"):
            design_experiments(
                linear_model, {"a": 1.0, "b": 0.0}, line_pool([1.0]), n=0
            )

    def test_summary_is_a_string(self):
        pool = line_pool(np.linspace(0.0, 10.0, 11))
        res = design_experiments(linear_model, {"a": 1.0, "b": 0.0}, pool, n=3)
        text = res.summary()
        assert "D-optimal design" in text
        assert "Predicted standard errors" in text


# ---------------------------------------------------------------------------
# Predicted covariance
# ---------------------------------------------------------------------------

class TestPredictedCovariance:
    def test_matches_the_closed_form_for_a_line(self):
        # cov = sigma^2 (X^T X)^{-1} for y = a x + b.
        sigma = 0.5
        xs = [0.0, 1.0, 2.0, 3.0]
        exps = line_pool(xs, sigma=sigma)
        ci = predicted_covariance(linear_model, {"a": 2.0, "b": 1.0}, exps)
        x_mat = np.column_stack([xs, np.ones(len(xs))])
        expected = sigma**2 * np.linalg.inv(x_mat.T @ x_mat)
        assert np.allclose(np.asarray(ci.covariance), expected)
        assert ci.std_errors["a"] == pytest.approx(np.sqrt(expected[0, 0]))
        assert ci.correlation[0, 1] == pytest.approx(
            expected[0, 1] / np.sqrt(expected[0, 0] * expected[1, 1])
        )

    def test_intervals_use_the_student_t_of_the_proposed_campaign(self):
        from scipy import stats

        exps = line_pool([0.0, 1.0, 2.0, 3.0], sigma=0.5)
        theta = {"a": 2.0, "b": 1.0}
        ci = predicted_covariance(linear_model, theta, exps)
        # dof = 4 measurements - 2 parameters, the same convention
        # fisher_confidence_intervals uses, since they share an assembler.
        t_val = float(stats.t.ppf(0.975, 2))
        assert ci.ci_upper["a"] - 2.0 == pytest.approx(t_val * ci.std_errors["a"])
        assert 2.0 - ci.ci_lower["a"] == pytest.approx(t_val * ci.std_errors["a"])

        tighter = predicted_covariance(linear_model, theta, exps, alpha=0.32)
        assert tighter.ci_upper["a"] < ci.ci_upper["a"]

    def test_a_singular_campaign_reports_infinite_intervals(self):
        pool = [Experiment.candidate({"x": float(x)}, ["y"]) for x in range(1, 5)]
        ci = predicted_covariance(
            product_model, {"A": 2.0, "B": 3.0}, pool, require_identifiable=False
        )
        assert ci.std_errors["A"] == np.inf
        assert ci.ci_lower["A"] == -np.inf and ci.ci_upper["A"] == np.inf
        assert np.all(np.isinf(np.asarray(ci.covariance)))
        assert np.all(np.isnan(np.asarray(ci.correlation)))

    def test_intervals_bracket_theta(self):
        exps = line_pool([0.0, 1.0, 2.0, 3.0], sigma=0.5)
        ci = predicted_covariance(linear_model, {"a": 2.0, "b": 1.0}, exps)
        assert ci.ci_lower["a"] < 2.0 < ci.ci_upper["a"]
        assert ci.ci_lower["b"] < 1.0 < ci.ci_upper["b"]

    def test_more_replicates_shrink_the_intervals_as_one_over_sqrt_n(self):
        base = line_pool([0.0, 1.0, 2.0, 3.0], sigma=0.5)
        ci1 = predicted_covariance(linear_model, {"a": 2.0, "b": 1.0}, base)
        ci4 = predicted_covariance(linear_model, {"a": 2.0, "b": 1.0}, base * 4)
        assert ci4.std_errors["a"] == pytest.approx(ci1.std_errors["a"] / 2.0)

    @pytest.mark.slow
    def test_predicted_covariance_matches_a_monte_carlo_refit(self):
        """The end-to-end check: design, simulate, fit, compare.

        ``predicted_covariance`` claims to tell you, in advance, the
        covariance of the estimate a campaign will produce. Here that claim
        is tested against the covariance actually obtained by fitting many
        noisy data sets simulated from the designed runs.
        """
        a_true, b_true, sigma = 2.0, 1.0, 0.5
        theta_true = {"a": a_true, "b": b_true}

        pool = line_pool(np.linspace(0.0, 10.0, 11), sigma=sigma)
        design = design_experiments(linear_model, theta_true, pool, n=8)
        predicted = predicted_covariance(linear_model, theta_true, design.selected)
        cov_pred = np.asarray(predicted.covariance)

        est = Estimator(linear_model, ["a", "b"])
        rng = np.random.default_rng(20200)
        n_mc = 400
        fits = np.empty((n_mc, 2))
        for k in range(n_mc):
            noisy = [
                Experiment(
                    inputs=dict(c.inputs),
                    observed={
                        "y": float(
                            a_true * c.inputs["x"]
                            + b_true
                            + sigma * rng.standard_normal()
                        )
                    },
                    uncertainties={"y": sigma},
                )
                for c in design.selected
            ]
            res = est.fit(noisy, theta_init=theta_true, objective="wsse",
                          jit_objective=False)
            fits[k] = [res.theta_opt["a"], res.theta_opt["b"]]

        cov_mc = np.cov(fits, rowvar=False)

        # The estimator is unbiased, so the fitted means sit on the truth
        # within a few Monte-Carlo standard errors.
        se_mean = np.sqrt(np.diag(cov_pred) / n_mc)
        assert abs(fits[:, 0].mean() - a_true) < 4 * se_mean[0]
        assert abs(fits[:, 1].mean() - b_true) < 4 * se_mean[1]

        # Variances agree to within Monte-Carlo error: the relative
        # standard error of a sample variance from n draws is sqrt(2/n),
        # about 7% here, so 25% is a comfortable but non-vacuous band.
        for i in range(2):
            assert cov_mc[i, i] == pytest.approx(cov_pred[i, i], rel=0.25)

        # ... and so does the parameter correlation.
        rho_mc = cov_mc[0, 1] / np.sqrt(cov_mc[0, 0] * cov_mc[1, 1])
        rho_pred = cov_pred[0, 1] / np.sqrt(cov_pred[0, 0] * cov_pred[1, 1])
        assert rho_mc == pytest.approx(rho_pred, abs=0.12)

    @pytest.mark.slow
    def test_nonlinear_predicted_covariance_matches_a_monte_carlo_refit(self):
        """Same check for a nonlinear model, where the FIM is a linearization."""
        theta_true = {"A": 5.0, "E": 800.0}
        sigma = 0.02

        pool = [
            Experiment.candidate({"T": float(T)}, ["y"], {"y": sigma})
            for T in np.linspace(300.0, 500.0, 21)
        ]
        design = design_experiments(arrhenius_model, theta_true, pool, n=10)
        cov_pred = np.asarray(
            predicted_covariance(arrhenius_model, theta_true, design.selected).covariance
        )

        est = Estimator(arrhenius_model, ["A", "E"])
        rng = np.random.default_rng(7)
        n_mc = 250
        fits = np.empty((n_mc, 2))
        for k in range(n_mc):
            noisy = []
            for c in design.selected:
                mean = float(
                    theta_true["A"] * np.exp(-theta_true["E"] / c.inputs["T"])
                )
                noisy.append(
                    Experiment(
                        inputs=dict(c.inputs),
                        observed={"y": mean + sigma * rng.standard_normal()},
                        uncertainties={"y": sigma},
                    )
                )
            res = est.fit(noisy, theta_init=theta_true, objective="wsse",
                          jit_objective=False)
            fits[k] = [res.theta_opt["A"], res.theta_opt["E"]]

        cov_mc = np.cov(fits, rowvar=False)
        for i in range(2):
            assert cov_mc[i, i] == pytest.approx(cov_pred[i, i], rel=0.35)

    def test_a_design_beats_an_arbitrary_campaign_of_the_same_size(self):
        theta = {"a": 1.0, "b": 0.0}
        pool = line_pool(np.linspace(0.0, 10.0, 11))
        design = design_experiments(linear_model, theta, pool, n=6)
        naive = pool[3:9]  # six runs bunched in the middle
        ci_design = predicted_covariance(linear_model, theta, design.selected)
        ci_naive = predicted_covariance(linear_model, theta, naive)
        assert ci_design.std_errors["a"] < ci_naive.std_errors["a"]
        assert ci_design.std_errors["b"] < ci_naive.std_errors["b"]


# ---------------------------------------------------------------------------
# Structural identifiability -- and the enforced ordering
# ---------------------------------------------------------------------------

class TestIdentifiability:
    def test_identifiable_model_passes(self):
        pool = line_pool(np.linspace(0.0, 10.0, 11))
        rep = check_identifiability(linear_model, {"a": 1.0, "b": 2.0}, pool)
        assert isinstance(rep, IdentifiabilityReport)
        assert rep.identifiable
        assert rep.rank == rep.n_params == 2
        assert rep.unidentifiable == []
        assert np.isfinite(rep.condition_number)

    @pytest.mark.parametrize(
        "model, theta, names",
        [
            (product_model, {"A": 2.0, "B": 3.0}, ["A", "B"]),
            (sum_model, {"c": 1.0, "d": 4.0}, ["c", "d"]),
        ],
    )
    def test_product_and_sum_parameters_are_caught(self, model, theta, names):
        pool = [Experiment.candidate({"x": float(x)}, ["y"]) for x in range(1, 8)]
        rep = check_identifiability(model, theta, pool)
        assert not rep.identifiable
        assert rep.rank == 1
        assert sorted(rep.unidentifiable) == sorted(names)
        assert rep.null_space.shape == (2, 1)
        assert rep.combinations  # a readable null direction was reported
        with pytest.raises(IdentifiabilityError, match="not structurally identifiable"):
            rep.raise_if_unidentifiable()

    def test_the_reported_null_direction_really_is_one(self):
        """The reported direction must annihilate S, not merely exist.

        ``unidentifiable`` and ``combinations`` are both non-empty for *any*
        singular vector: reporting the dominant right singular vector instead
        of a null one still names A and B and still renders a readable
        string. Only pushing the vector back through S tells the right answer
        from a plausible wrong one.
        """
        theta = {"A": 2.0, "B": 3.0}
        pool = [Experiment.candidate({"x": float(x)}, ["y"]) for x in range(1, 8)]
        rep = check_identifiability(product_model, theta, pool)
        s = np.asarray(
            sensitivity_matrix(
                product_model, theta, pool, ["A", "B"], scale="theta"
            )
        )
        assert rep.null_space.shape == (2, 1)
        v = rep.null_space[:, 0]
        assert np.linalg.norm(v) == pytest.approx(1.0)

        # S v = 0 ...
        assert np.linalg.norm(s @ v) < 1e-10 * np.linalg.norm(s)
        # ... while the orthogonal direction is emphatically not annihilated,
        # so this is not vacuously true of every unit vector.
        w = np.array([-v[1], v[0]])
        assert np.linalg.norm(s @ w) > 0.1 * np.linalg.norm(s)

        # In theta-scaled coordinates a product trades A off against B one
        # for one: the direction is (1, -1)/sqrt(2) up to an overall sign.
        assert abs(v[0]) == pytest.approx(abs(v[1]))
        assert v[0] * v[1] < 0

        # ... and the rendered string says exactly that.
        (combo,) = rep.combinations
        assert combo.endswith("~ 0")
        assert "A" in combo and "B" in combo
        assert "0.707" in combo
        assert combo.count("-") == 1  # one of the two terms is negated

    def test_more_data_does_not_rescue_an_unidentifiable_model(self):
        """The point of the check: rows can be added forever, rank stays 1."""
        theta = {"A": 2.0, "B": 3.0}
        small = [Experiment.candidate({"x": float(x)}, ["y"]) for x in range(1, 4)]
        huge = [Experiment.candidate({"x": float(x)}, ["y"]) for x in range(1, 400)]
        assert check_identifiability(product_model, theta, small).rank == 1
        assert check_identifiability(product_model, theta, huge).rank == 1

    def test_an_added_measurement_does_rescue_it(self):
        """...but a new *kind* of measurement does, which is the real fix."""
        theta = {"A": 2.0, "B": 3.0}
        xs = [1.0, 2.0, 3.0]
        only_y = [Experiment.candidate({"x": x}, ["y"]) for x in xs]
        y_and_z = [Experiment.candidate({"x": x}, ["y", "z"]) for x in xs]
        assert not check_identifiability(
            two_output_product_model, theta, only_y
        ).identifiable
        assert check_identifiability(
            two_output_product_model, theta, y_and_z
        ).identifiable

    def test_design_refuses_to_run_on_an_unidentifiable_model(self):
        pool = [Experiment.candidate({"x": float(x)}, ["y"]) for x in range(1, 8)]
        with pytest.raises(IdentifiabilityError):
            design_experiments(product_model, {"A": 2.0, "B": 3.0}, pool, n=3)

    def test_predicted_covariance_refuses_too(self):
        pool = [Experiment.candidate({"x": float(x)}, ["y"]) for x in range(1, 8)]
        with pytest.raises(IdentifiabilityError):
            predicted_covariance(product_model, {"A": 2.0, "B": 3.0}, pool)

    def test_the_check_can_be_bypassed_deliberately_and_then_bites(self):
        pool = [Experiment.candidate({"x": float(x)}, ["y"]) for x in range(1, 8)]
        ci = predicted_covariance(
            product_model, {"A": 2.0, "B": 3.0}, pool, require_identifiable=False
        )
        assert ci.std_errors["A"] == np.inf
        res = design_experiments(
            product_model, {"A": 2.0, "B": 3.0}, pool, n=3,
            require_identifiable=False,
        )
        assert res.covariance is None
        assert res.criterion_value == -np.inf

    def test_a_successful_design_carries_its_identifiability_report(self):
        pool = line_pool(np.linspace(0.0, 10.0, 11))
        res = design_experiments(linear_model, {"a": 1.0, "b": 0.0}, pool, n=3)
        assert res.identifiability is not None
        assert res.identifiability.identifiable

    def test_rank_test_is_invariant_to_parameter_units(self):
        """Rescaling a parameter must not change a structural conclusion."""

        def kilo_model(theta, exp):
            # Same physics as linear_model with a expressed in kilo-units.
            return {"y": 1000.0 * theta["a"] * exp.inputs["x"] + theta["b"]}

        pool = line_pool(np.linspace(0.0, 10.0, 11))
        rep = check_identifiability(linear_model, {"a": 1.0, "b": 2.0}, pool)
        rep_k = check_identifiability(kilo_model, {"a": 1e-3, "b": 2.0}, pool)
        assert rep.identifiable == rep_k.identifiable
        assert rep.rank == rep_k.rank
        assert rep_k.condition_number == pytest.approx(rep.condition_number, rel=1e-8)

    def test_summary_is_a_string(self):
        pool = [Experiment.candidate({"x": float(x)}, ["y"]) for x in range(1, 5)]
        text = check_identifiability(product_model, {"A": 2.0, "B": 3.0}, pool).summary()
        assert "identifiable" in text
        assert "null direction" in text


# ---------------------------------------------------------------------------
# JAX transformations
# ---------------------------------------------------------------------------

class TestJax:
    def test_fisher_information_jits(self):
        pool = line_pool([0.0, 1.0, 2.0, 3.0], sigma=0.5)

        @jax.jit
        def fim_of(theta):
            return fisher_information(linear_model, theta, pool, ["a", "b"])

        jitted = np.asarray(fim_of(jnp.array([1.0, 0.0])))
        direct = np.asarray(
            fisher_information(linear_model, {"a": 1.0, "b": 0.0}, pool)
        )
        assert np.allclose(jitted, direct)

    def test_criterion_jits_and_differentiates(self):
        pool = [
            Experiment.candidate({"T": float(T)}, ["y"], {"y": 0.1})
            for T in (300.0, 400.0, 500.0)
        ]

        def d_objective(theta):
            fim = fisher_information(arrhenius_model, theta, pool, ["A", "E"])
            return design_criterion(fim, "D")

        theta = jnp.array([1e6, 4000.0])
        val = float(jax.jit(d_objective)(theta))
        assert np.isfinite(val)
        assert val == pytest.approx(float(d_objective(theta)))

        # The design criterion is itself differentiable w.r.t. theta, which
        # is what a continuous-design optimizer would need.
        g = jax.grad(d_objective)(theta)
        assert np.all(np.isfinite(np.asarray(g)))
        # dD/dA is exact here: the FIM scales like A^2 in its A-A block only
        # through S = [exp(-E/T), ...], and log det picks up 2 log A, so
        # d(logdet)/dA = 2/A.
        assert float(g[0]) == pytest.approx(2.0 / float(theta[0]), rel=1e-6)

    def test_sensitivity_matrix_jits(self):
        pool = line_pool([1.0, 3.0])

        @jax.jit
        def s_of(theta):
            return sensitivity_matrix(linear_model, theta, pool, ["a", "b"])

        assert np.allclose(np.asarray(s_of(jnp.array([1.0, 0.0]))),
                           [[1.0, 1.0], [3.0, 1.0]])


# ---------------------------------------------------------------------------
# Estimator integration
# ---------------------------------------------------------------------------

class TestEstimatorIntegration:
    def test_estimator_methods_delegate(self):
        est = Estimator(linear_model, ["a", "b"])
        pool = line_pool(np.linspace(0.0, 10.0, 11))
        rep = est.check_identifiability({"a": 1.0, "b": 0.0}, pool)
        assert rep.identifiable
        design = est.design_experiments({"a": 1.0, "b": 0.0}, pool, n=4)
        assert sorted(e.inputs["x"] for e in design.selected) == [0.0, 0.0, 10.0, 10.0]
        ci = est.predicted_covariance({"a": 1.0, "b": 0.0}, design.selected)
        assert ci.std_errors["a"] > 0

    def test_estimation_result_can_be_used_as_theta(self):
        est = Estimator(linear_model, ["a", "b"])
        data = [
            Experiment(inputs={"x": float(x)}, observed={"y": 2.0 * x + 1.0})
            for x in (0.0, 1.0, 2.0, 3.0)
        ]
        fit = est.fit(data, theta_init={"a": 1.0, "b": 0.0})
        pool = line_pool(np.linspace(0.0, 10.0, 11))
        design = est.design_experiments(fit, pool, n=2)
        assert len(design.selected) == 2
        assert design.theta["a"] == pytest.approx(2.0, abs=1e-4)
