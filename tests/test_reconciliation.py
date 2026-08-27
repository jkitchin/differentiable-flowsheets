"""Tests for difflow.reconciliation.

The load-bearing tests are the linear ones: on a linear constraint the
KKT system is solved exactly in a single step, so there is no iteration
tolerance to hide behind and the results are checked against the
closed-form weighted least squares solution.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from difflow.reconciliation import (
    MEASURED_JUST_DETERMINED,
    MEASURED_REDUNDANT,
    UNMEASURED_OBSERVABLE,
    ReconciliationStructureError,
    auto_scaling,
    classify,
    global_test,
    identity_scaling,
    kkt_matrix,
    measurement_sensitivity,
    measurement_test,
    reconcile,
    reconciled_covariance,
    sensor_value,
    serial_elimination,
    solve_reconciliation,
)


# =============================================================================
# Fixtures and helpers
# =============================================================================


def one_balance(x, params=None):
    """A single linear constraint: x0 + x1 = 10."""
    return jnp.array([x[0] + x[1] - 10.0])


def closed_form_wls(a, b, y, sigma):
    """x_hat = y - Sigma A^T (A Sigma A^T)^-1 (A y - b), fully measured."""
    sig = np.diag(np.asarray(sigma) ** 2)
    a = np.asarray(a, dtype=float)
    r = a @ np.asarray(y) - np.asarray(b)
    return np.asarray(y) - sig @ a.T @ np.linalg.solve(a @ sig @ a.T, r)


@pytest.fixture(scope="module")
def random_linear():
    """A random full-row-rank linear system, 6 equations in 10 variables."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(6, 10))
    b = rng.normal(size=6)
    sigma = jnp.asarray(rng.uniform(0.5, 2.0, size=10))
    y = jnp.asarray(rng.normal(size=10) * 2.0)

    def residual_fn(x, params=None):
        return jnp.asarray(a) @ x - jnp.asarray(b)

    return residual_fn, a, b, y, sigma


# =============================================================================
# Linear exactness
# =============================================================================


class TestLinearExactness:
    """A linear problem must be solved exactly, in one step."""

    def test_two_variable_closed_form(self):
        """x0 + x1 = 10 with sigma = (1, 2) puts 4x the adjustment on x1."""
        y = jnp.array([6.0, 6.0])
        sigma = jnp.array([1.0, 2.0])
        res = reconcile(one_balance, y, sigma, names=["a", "b"])

        assert res.x[0] == pytest.approx(5.6, abs=1e-12)
        assert res.x[1] == pytest.approx(4.4, abs=1e-12)
        assert res.objective == pytest.approx(0.8, abs=1e-12)
        assert res.converged
        assert res.residual_norm < 1e-12

    def test_matches_closed_form_wls(self, random_linear):
        residual_fn, a, b, y, sigma = random_linear
        res = reconcile(residual_fn, y, sigma)
        expected = closed_form_wls(a, b, y, sigma)
        np.testing.assert_allclose(np.asarray(res.x), expected, atol=1e-10)

    def test_solved_in_one_step(self, random_linear):
        """One KKT solve is exact for a linear system; iterating adds nothing."""
        residual_fn, _, _, y, sigma = random_linear
        sc = auto_scaling(residual_fn, y, sigma)
        x1, _ = solve_reconciliation(
            residual_fn, y, sigma, x0=y, scaling=sc, n_steps=1, correct=False
        )
        x20, _ = solve_reconciliation(
            residual_fn, y, sigma, x0=y, scaling=sc, n_steps=20, correct=False
        )
        np.testing.assert_allclose(np.asarray(x1), np.asarray(x20), atol=1e-14)

    def test_unmeasured_variable_absorbs_the_imbalance(self):
        """With x1 unmeasured, x0 is untouched and the residual is zero."""
        y = jnp.array([6.0, jnp.nan])
        sigma = jnp.array([1.0, jnp.inf])
        res = reconcile(one_balance, y, sigma, names=["a", "b"])

        assert res.x[0] == pytest.approx(6.0, abs=1e-12)
        assert res.x[1] == pytest.approx(4.0, abs=1e-12)
        assert res.objective == pytest.approx(0.0, abs=1e-20)
        assert res.structure.degree_of_redundancy == 0

    def test_nan_measurement_does_not_leak(self):
        """A nan at an unmeasured entry must not poison the solution."""
        y = jnp.array([6.0, jnp.nan])
        sigma = jnp.array([1.0, jnp.inf])
        res = reconcile(one_balance, y, sigma)
        assert bool(jnp.all(jnp.isfinite(res.x)))


# =============================================================================
# Covariance
# =============================================================================


class TestCovariance:
    """The KKT inverse reproduces, and generalizes, the textbook formula."""

    def test_matches_projection_formula(self, random_linear):
        """[K^-1]_11 == Sigma - Sigma A^T (A Sigma A^T)^-1 A Sigma."""
        residual_fn, a, _, y, sigma = random_linear
        sc = auto_scaling(residual_fn, y, sigma)
        cov = np.asarray(
            reconciled_covariance(residual_fn, y, sigma, scaling=sc)
        )

        sig = np.diag(np.asarray(sigma) ** 2)
        proj = sig - sig @ a.T @ np.linalg.solve(a @ sig @ a.T, a @ sig)
        np.testing.assert_allclose(cov, proj, atol=1e-10)

    def test_reduces_uncertainty(self, random_linear):
        """Reconciliation can only sharpen a measurement, never blunt it."""
        residual_fn, _, _, y, sigma = random_linear
        res = reconcile(residual_fn, y, sigma)
        var = np.diag(np.asarray(res.covariance))
        assert np.all(var <= np.asarray(sigma) ** 2 + 1e-12)
        assert np.all(var > 0)

    def test_sensitivity_identity_linear(self, random_linear):
        """S Sigma S^T == Sigma_xhat exactly when the constraints are linear."""
        residual_fn, _, _, y, sigma = random_linear
        sc = auto_scaling(residual_fn, y, sigma)
        s = np.asarray(
            measurement_sensitivity(residual_fn, y, sigma, x0=y, scaling=sc)
        )
        cov = np.asarray(
            reconciled_covariance(residual_fn, y, sigma, scaling=sc)
        )
        sig = np.diag(np.asarray(sigma) ** 2)
        np.testing.assert_allclose(s @ sig @ s.T, cov, atol=1e-10)

    def test_sensitivity_identity_with_unmeasured(self, random_linear):
        """The identity survives entries that carry no measurement."""
        residual_fn, _, _, y, sigma = random_linear
        sigma = sigma.at[jnp.array([2, 5, 7])].set(jnp.inf)
        sc = auto_scaling(residual_fn, y, sigma)
        cov = np.asarray(
            reconciled_covariance(residual_fn, y, sigma, scaling=sc)
        )
        w = np.diag(np.where(np.isfinite(sigma), 1.0 / np.asarray(sigma) ** 2, 0.0))
        # P W P^T == P  is the identity behind S Sigma S^T == P.
        np.testing.assert_allclose(cov @ w @ cov.T, cov, atol=1e-10)

    def test_own_measurement_is_shrunk_not_amplified(self, random_linear):
        residual_fn, _, _, y, sigma = random_linear
        sc = auto_scaling(residual_fn, y, sigma)
        s = np.asarray(
            measurement_sensitivity(residual_fn, y, sigma, x0=y, scaling=sc)
        )
        diag = np.diag(s)
        assert np.all(diag > 0), "an estimate must move with its own measurement"
        assert np.all(diag <= 1.0 + 1e-9), "it must not move by more than it"


# =============================================================================
# Gradients
# =============================================================================


class TestGradients:
    """Reconciliation is differentiable end to end."""

    def test_gradient_wrt_measurement(self, random_linear):
        residual_fn, _, _, y, sigma = random_linear
        sc = auto_scaling(residual_fn, y, sigma)

        def first_estimate(yy):
            return solve_reconciliation(
                residual_fn, yy, sigma, x0=y, scaling=sc, n_steps=6
            )[0][0]

        g = jax.grad(first_estimate)(y)
        assert bool(jnp.all(jnp.isfinite(g)))
        assert float(g[0]) > 0.0

    def test_gradient_wrt_sigma_is_finite(self, random_linear):
        """Loosening a sensor lowers the weighted objective."""
        residual_fn, _, _, y, sigma = random_linear

        def objective(s):
            sc = auto_scaling(residual_fn, y, s)
            x, _ = solve_reconciliation(
                residual_fn, y, s, x0=y, scaling=sc, n_steps=6
            )
            return jnp.sum(((x - y) / s) ** 2)

        g = jax.grad(objective)(sigma)
        assert bool(jnp.all(jnp.isfinite(g)))
        assert bool(jnp.all(g <= 1e-9)), f"expected non-positive, got {g}"

    def test_gradient_wrt_params(self):
        """params is a differentiable argument of the residual function."""
        def residual_fn(x, theta):
            return jnp.array([x[0] + theta * x[1] - 10.0])

        y = jnp.array([6.0, 6.0])
        sigma = jnp.array([1.0, 2.0])

        def estimate(theta):
            sc = auto_scaling(residual_fn, y, sigma, params=theta)
            x, _ = solve_reconciliation(
                residual_fn, y, sigma, x0=y, scaling=sc,
                params=theta, n_steps=6,
            )
            return x[0]

        g = jax.grad(estimate)(1.0)
        assert jnp.isfinite(g)
        assert float(g) != 0.0


# =============================================================================
# Structure: observability, redundancy, solvability
# =============================================================================


class TestStructure:
    """Solvability and observability are one test, decided before solving."""

    def test_all_measured_is_fully_redundant(self, random_linear):
        residual_fn, _, _, y, sigma = random_linear
        res = reconcile(residual_fn, y, sigma)
        st = res.structure
        assert st.solvable
        assert st.degree_of_redundancy == 6
        assert all(c == MEASURED_REDUNDANT for c in st.classes.values())
        assert all(0.0 < r <= 1.0 for r in st.redundancy.values())

    def test_unmeasured_observable_is_classified(self):
        y = jnp.array([6.0, jnp.nan])
        sigma = jnp.array([1.0, jnp.inf])
        res = reconcile(one_balance, y, sigma, names=["a", "b"])
        assert res.structure.classes["b"] == UNMEASURED_OBSERVABLE
        assert res.structure.classes["a"] == MEASURED_JUST_DETERMINED
        assert res.structure.redundancy["a"] == pytest.approx(0.0, abs=1e-9)

    def test_unobservable_raises_naming_the_variable(self):
        """Two unknowns in one equation cannot both be recovered."""
        def residual_fn(x, params=None):
            return jnp.array([x[0] + x[1] + x[2] - 10.0])

        y = jnp.array([6.0, jnp.nan, jnp.nan])
        sigma = jnp.array([1.0, jnp.inf, jnp.inf])
        with pytest.raises(ReconciliationStructureError, match="Unobservable"):
            reconcile(residual_fn, y, sigma, names=["a", "b", "c"])

    def test_unobservable_error_names_the_right_variables(self):
        def residual_fn(x, params=None):
            return jnp.array([x[0] + x[1] + x[2] - 10.0])

        y = jnp.array([6.0, jnp.nan, jnp.nan])
        sigma = jnp.array([1.0, jnp.inf, jnp.inf])
        with pytest.raises(ReconciliationStructureError) as exc:
            reconcile(residual_fn, y, sigma, names=["a", "b", "c"])
        message = str(exc.value)
        assert "b" in message and "c" in message

    def test_prior_restores_solvability(self):
        """A finite sigma on an unobservable unknown makes it estimable."""
        def residual_fn(x, params=None):
            return jnp.array([x[0] + x[1] + x[2] - 10.0])

        y = jnp.array([6.0, 2.0, jnp.nan])
        sigma = jnp.array([1.0, 0.5, jnp.inf])
        res = reconcile(residual_fn, y, sigma, names=["a", "b", "c"])
        assert res.converged

    @staticmethod
    def _classify_linear(a):
        """Classify a system whose variables are all unmeasured."""
        def residual_fn(x, params=None):
            return jnp.asarray(a) @ x

        return classify(
            residual_fn,
            jnp.ones(a.shape[1]),
            jnp.full(a.shape[1], jnp.inf),
            scaling=identity_scaling(*reversed(a.shape)),
            names=["u", "v"],
        )

    def test_ill_conditioned_but_full_rank_is_solvable(self):
        """Singular values well above the threshold keep the problem solvable."""
        st = self._classify_linear(jnp.array([[1.0, 0.0], [0.0, 1e-4]]))
        assert st.solvable
        assert st.rank_A_unmeasured == 2

    def test_genuine_null_direction_is_detected(self):
        st = self._classify_linear(jnp.array([[1.0, 1.0], [1.0, 1.0]]))
        assert not st.solvable
        assert st.rank_A_unmeasured == 1
        assert set(st.unobservable) == {"u", "v"}

    def test_rank_is_taken_from_the_svd_not_the_gram_matrix(self):
        """Rank must come from svd(A), never from eigvals(A.T @ A).

        Squaring the matrix squares its condition number, so a
        singular value at the threshold reappears as its square and a
        rank-deficient system is reported as full rank. Here the
        smallest singular value is genuinely below the threshold, while
        the corresponding Gram eigenvalue is far below anything a
        threshold on ``A.T @ A`` would reject in the same units.
        """
        a = np.array([[1.0, 0.0], [0.0, 1e-9]])
        st = self._classify_linear(jnp.asarray(a))

        sv = np.linalg.svd(a, compute_uv=False)
        assert sv[1] < st.rank_tol, "premise: the small value is below tolerance"
        assert st.rank_A_unmeasured == 1, "SVD sees the deficiency"
        assert not st.solvable

        # The same threshold applied to the Gram spectrum would not.
        gram_eigs = np.linalg.eigvalsh(a.T @ a)
        assert gram_eigs.min() < st.rank_tol**2

    def test_kkt_matrix_shape_and_blocks(self):
        a = jnp.array([[1.0, 2.0, 3.0]])
        w = jnp.array([1.0, 0.0, 4.0])
        k = kkt_matrix(a, w)
        assert k.shape == (4, 4)
        np.testing.assert_allclose(np.asarray(k[:3, :3]), np.diag([1.0, 0.0, 4.0]))
        np.testing.assert_allclose(np.asarray(k[3:, :3]), np.asarray(a))
        np.testing.assert_allclose(np.asarray(k[:3, 3:]), np.asarray(a).T)
        assert float(k[3, 3]) == 0.0


# =============================================================================
# Scaling
# =============================================================================


class TestScaling:
    """Scaling makes the formulation independent of the unit system."""

    @staticmethod
    def _problem(unit):
        """x0 + x1 = 10 * unit, measured in units of `unit`."""
        def residual_fn(x, params=None):
            return jnp.array([x[0] + x[1] - 10.0 * unit])

        y = jnp.array([6.0, 6.0]) * unit
        sigma = jnp.array([1.0, 2.0]) * unit
        return residual_fn, y, sigma

    def test_unit_invariance(self):
        """The same problem in units 1e5 apart gives the same answer."""
        f1, y1, s1 = self._problem(1.0)
        f2, y2, s2 = self._problem(1e5)
        r1 = reconcile(f1, y1, s1)
        r2 = reconcile(f2, y2, s2)
        np.testing.assert_allclose(
            np.asarray(r2.x) / 1e5, np.asarray(r1.x), rtol=1e-10
        )
        assert r2.objective == pytest.approx(r1.objective, rel=1e-10)

    def test_scaled_kkt_is_better_conditioned(self):
        """Scaling is on by default because the raw system degrades fast."""
        from difflow.reconciliation.core import jacobian_of, measured_mask

        f, y, sigma = self._problem(1e5)
        a = jacobian_of(f, y)

        raw = kkt_matrix(a, jnp.where(measured_mask(sigma), 1.0 / sigma**2, 0.0))
        sc = auto_scaling(f, y, sigma)
        scaled = kkt_matrix(
            sc.r[:, None] * a * sc.d[None, :],
            jnp.where(measured_mask(sigma), 1.0, 0.0),
        )
        cond_raw = float(np.linalg.cond(np.asarray(raw)))
        cond_scaled = float(np.linalg.cond(np.asarray(scaled)))
        assert cond_scaled < 1e3, f"scaled condition number {cond_scaled:.3g}"
        assert cond_raw > 1e6 * cond_scaled, (
            f"raw {cond_raw:.3g} vs scaled {cond_scaled:.3g}"
        )


# =============================================================================
# Gross error detection
# =============================================================================


class TestGrossError:
    """The global test's statistic and degrees of freedom must be right."""

    def test_global_test_dof_equals_degree_of_redundancy(self, random_linear):
        """E[objective] == degrees of redundancy, for several unmeasured counts.

        This is the property that fixes both the statistic and its
        distribution: the objective is chi-squared on ``m - rank(A_U)``,
        not on the number of equations.
        """
        residual_fn, a, b, _, sigma = random_linear
        # A truth that satisfies the constraints exactly.
        x_true = jnp.asarray(np.linalg.lstsq(a, b, rcond=None)[0])
        n_draws = 4000

        for n_unmeasured, expected_dof in [(0, 6), (1, 5), (2, 4)]:
            sig = np.asarray(sigma).copy()
            sig[:n_unmeasured] = np.inf
            sig_j = jnp.asarray(sig)
            mask = jnp.isfinite(sig_j)

            # The structure is fixed across draws, so check it once and
            # then vmap the traced core over the whole batch.
            res = reconcile(residual_fn, x_true, sig_j)
            assert res.structure.degree_of_redundancy == expected_dof
            sc = res.scaling

            noise = jax.random.normal(
                jax.random.PRNGKey(7 + n_unmeasured), (n_draws, 10)
            ) * jnp.where(mask, sig_j, 0.0)

            @jax.jit
            @jax.vmap
            def objective(y):
                x, _ = solve_reconciliation(
                    residual_fn, y, sig_j, x0=y, scaling=sc, n_steps=3
                )
                return jnp.sum(
                    jnp.where(mask, ((x - y) / jnp.where(mask, sig_j, 1.0)) ** 2, 0.0)
                )

            mean = float(jnp.mean(objective(x_true + noise)))
            assert mean == pytest.approx(expected_dof, rel=0.06), (
                f"{n_unmeasured} unmeasured: E[objective] = {mean:.3f}, "
                f"expected {expected_dof}"
            )

    def test_global_test_accepts_clean_data(self, random_linear):
        residual_fn, a, b, _, sigma = random_linear
        x_true = jnp.asarray(np.linalg.lstsq(a, b, rcond=None)[0])
        res = reconcile(residual_fn, x_true, sigma)
        gt = global_test(res)
        assert not gt.detected
        assert gt.dof == 6
        assert gt.statistic == pytest.approx(0.0, abs=1e-15)

    def test_global_test_rejects_a_gross_error(self, random_linear):
        residual_fn, a, b, _, sigma = random_linear
        x_true = np.linalg.lstsq(a, b, rcond=None)[0]
        y = np.asarray(x_true).copy()
        y[3] += 20.0 * float(sigma[3])
        res = reconcile(residual_fn, jnp.asarray(y), sigma)

        gt = global_test(res)
        assert gt.detected
        assert gt.statistic > gt.critical

        mt = measurement_test(res)
        assert mt.detected
        assert mt.suspect == res.names[3]

    def test_measurement_test_marks_untestable_entries(self):
        """A measurement nothing checks gets nan, not a spurious z."""
        y = jnp.array([6.0, jnp.nan])
        sigma = jnp.array([1.0, jnp.inf])
        res = reconcile(one_balance, y, sigma, names=["a", "b"])
        mt = measurement_test(res)
        assert not mt.testable["a"], "a is unchecked, so it cannot be tested"
        assert np.isnan(mt.z["a"])
        assert not mt.detected

    def test_serial_elimination_clears_the_global_test(self, random_linear):
        residual_fn, a, b, _, sigma = random_linear
        x_true = np.linalg.lstsq(a, b, rcond=None)[0]
        rng = np.random.default_rng(11)
        y = x_true + rng.normal(size=10) * np.asarray(sigma)
        y[3] += 20.0 * float(sigma[3])

        steps = serial_elimination(residual_fn, jnp.asarray(y), sigma)
        assert steps[0].detected, "the seeded error must be detected"
        assert steps[0].suspect is not None
        assert not steps[-1].detected, "elimination must end on clean data"
        assert steps[-1].removed is not None


# =============================================================================
# Sensor placement
# =============================================================================


class TestSensorPlacement:
    def test_a_sensor_never_increases_uncertainty(self, random_linear):
        residual_fn, _, _, y, sigma = random_linear
        sigma = sigma.at[4].set(jnp.inf)
        out = sensor_value(
            residual_fn, y, sigma, target=4, candidate=4, candidate_sigma=1.0
        )
        assert out["sd_after"] <= out["sd_before"] + 1e-12
        assert out["variance_reduction"] > 0.0

    def test_ranking_is_ordered(self, random_linear):
        residual_fn, _, _, y, sigma = random_linear
        sigma = sigma.at[4].set(jnp.inf)
        ranked = sensor_ranking = __import__(
            "difflow.reconciliation", fromlist=["sensor_ranking"]
        ).sensor_ranking(
            residual_fn, y, sigma, target=4,
            candidates=[0, 1, 2, 4], candidate_sigma=0.5,
        )
        reductions = [d["variance_reduction"] for d in ranked]
        assert reductions == sorted(reductions, reverse=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
