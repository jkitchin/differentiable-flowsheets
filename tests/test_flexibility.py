"""Tests for :mod:`difflow.flexibility`.

Every claim the module makes is checked against something that is *not* the
module.  Where a number can be derived by hand it is derived by hand in the
test and written into the assertion; where it cannot, it is checked against a
brute-force search or a central difference.

The load-bearing tests, and what each would catch:

``test_flexibility_index_matches_hand_arithmetic``
    ``F = (d - 1) / 0.5`` for the textbook one-control problem, at four
    designs.  Catches a bisection that solves the wrong scaling equation.
``test_index_no_recourse_matches_hand_arithmetic``
    Every per-vertex limit, the binding vertex, its direction and the binding
    constraint, all four hand-computed.  Catches a wrong ``argmin``, a
    transposed sign matrix, or a binding constraint read off the wrong row.
``test_psi_sign_tracks_feasibility``
    ``psi <= 0`` exactly when the design is feasible, over a sweep that
    crosses the boundary in both directions.
``test_critical_realization_is_the_true_argmax``
    The reported critical realization and ``psi`` are checked against a
    brute-force ``max_theta min_u max_j`` on a 31^3 x 4001 grid.
``test_gradient_is_the_multiplier_weighted_one``
    The true derivative is ``-0.5``; differentiating whichever constraint
    ``jnp.max`` selected gives ``-1`` or ``0``.  The tolerance is 5e-3, which
    admits neither.
``test_recourse_cuts_the_feed_penalty_but_not_the_backoff``
    The conceptual deliverable: with recourse the feed penalty halves,
    0.2 -> 0.1, while the parameter back-off stays at 0.24.
``test_continuous_finds_interior_critical_point``
    Vertex enumeration reports ``psi = -0.8`` (feasible) on a problem whose
    true ``psi`` is ``+0.2`` at an interior point; the continuous fallback
    finds it.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

jax.config.update("jax_enable_x64", True)

from difflow.flexibility import (
    DEFAULT_OPTIONS,
    MAX_VERTICES,
    NO_CONTROLS,
    ControlSpec,
    FeasibilityResult,
    FlexibilityResult,
    PenaltyReport,
    SolverOptions,
    StochasticFeasibilityResult,
    UncertaintySet,
    as_control_spec,
    as_uncertainty_set,
    draw_flexibility_region,
    draw_penalty_split,
    expected_feasibility,
    feasibility_function,
    feasibility_value,
    flexibility_index,
    inner_value,
    minimax_value,
    sample_set,
    smooth_max,
    uncertainty_penalties,
    vertex_limits,
    vertex_values,
)

# --------------------------------------------------------------------------
# Reference problems, with their closed-form answers written out.
# --------------------------------------------------------------------------

#: The textbook one-control problem.  ``u`` must cover the feed and must not
#: exceed the design::
#:
#:     min_u max(u - d, theta - u)  =  (theta - d) / 2      at  u* = (theta + d)/2
#:
#: so ``psi(d, delta) = (1 + 0.5 delta - d) / 2`` and ``F = (d - 1) / 0.5``.
def linear_model(d, u, theta):
    return jnp.array([u[0] - d[0], theta[0] - u[0]])


LINEAR_SET = {"feed": (1.0, 0.5)}
LINEAR_CONTROLS = {"u": (-10.0, 10.0)}


def linear_psi(d, delta=1.0):
    """Closed form of ``psi`` for :func:`linear_model`."""
    return (1.0 + 0.5 * delta - d) / 2.0


#: A two-parameter problem with no recourse at all, so every quantity is a
#: plain maximum over four hand-listable corners.
def corner_model(d, u, theta):
    return jnp.array([theta[0] + theta[1] - d[0], -theta[0] - 2.0])


CORNER_SET = {"a": (0.0, 1.0), "b": (0.0, 1.0)}


#: A three-parameter, nonlinear-in-theta problem used for the brute-force and
#: finite-difference checks.  Monotone in every parameter, so the critical
#: realization is a vertex and ``method="vertex"`` is exact.
def curved_model(d, u, theta):
    return jnp.array([theta[0] + 0.5 * theta[1] ** 2 - u[0] - 1.0,
                      u[0] - 2.0 - theta[2] - d[0]])


CURVED_SET = {"a": (1.0, 0.3, 0.2), "b": (2.0, 0.5, 0.4), "c": (0.5, 0.25, 0.1)}
CURVED_CONTROLS = {"u": (0.0, 6.0)}


def brute_force_psi(theta_lo, theta_hi, u_lo, u_hi, f_np, n_theta=31, n_u=4001):
    """``max_theta min_u max_j f_j`` on a dense grid, in plain numpy.

    Returns ``(psi, theta_star)``.  This is the independent oracle: it shares
    no code with the module under test.
    """
    grids = [np.linspace(lo, hi, n_theta)
             for lo, hi in zip(theta_lo, theta_hi)]
    thetas = np.array(list(itertools.product(*grids)))
    us = np.linspace(u_lo, u_hi, n_u)
    inner = np.max(f_np(thetas, us), axis=0).min(axis=1)
    k = int(np.argmax(inner))
    return float(inner[k]), thetas[k]


# --------------------------------------------------------------------------
# Sets and recourse boxes
# --------------------------------------------------------------------------


class TestUncertaintySet:
    def test_asymmetric_vertices_are_exact(self):
        T = UncertaintySet(nominal=[1.0, 300.0], lower=[0.2, 10.0],
                           upper=[0.1, 20.0], names=["C", "T"])
        assert T.n == 2 and T.n_vertices == 4
        # signs() enumerates itertools.product((-1, 1)) over varying axes.
        npt.assert_array_equal(T.signs(),
                               [[-1, -1], [-1, 1], [1, -1], [1, 1]])
        npt.assert_allclose(np.asarray(T.vertices()),
                            [[0.8, 290.0], [0.8, 320.0],
                             [1.1, 290.0], [1.1, 320.0]])

    def test_scaling_is_linear_in_delta(self):
        T = as_uncertainty_set({"a": (2.0, 0.5)})
        npt.assert_allclose(np.asarray(T.vertices(0.0)), [[2.0], [2.0]])
        npt.assert_allclose(np.asarray(T.vertices(3.0)), [[0.5], [3.5]])
        lo, hi = T.bounds(2.0)
        npt.assert_allclose(np.asarray(lo), [1.0])
        npt.assert_allclose(np.asarray(hi), [3.0])

    def test_fixed_coordinate_does_not_double_the_vertex_count(self):
        T = UncertaintySet(nominal=[1.0, 5.0, 2.0], lower=[0.1, 0.0, 0.2],
                           upper=[0.1, 0.0, 0.2])
        assert T.n_vertices == 4          # not 8
        npt.assert_array_equal(T.varying, [0, 2])
        assert np.all(np.asarray(T.vertices())[:, 1] == 5.0)

    def test_contains(self):
        T = as_uncertainty_set({"a": (1.0, 0.5)})
        assert bool(T.contains(jnp.array([1.4])))
        assert not bool(T.contains(jnp.array([1.6])))
        assert bool(T.contains(jnp.array([1.6]), scale=2.0))

    def test_mapping_forms(self):
        T = as_uncertainty_set({"C": (1.0, 0.2), "T": (300.0, 5.0, 10.0)})
        assert T.names == ("C", "T")
        npt.assert_allclose(np.asarray(T.lower), [0.2, 5.0])
        npt.assert_allclose(np.asarray(T.upper), [0.2, 10.0])
        assert as_uncertainty_set(T) is T

    def test_params_mixin_access(self):
        T = as_uncertainty_set({"a": (1.0, 0.5)})
        assert "nominal" in T
        npt.assert_allclose(np.asarray(T["nominal"]), [1.0])

    def test_vertex_explosion_is_refused(self):
        T = UncertaintySet(nominal=np.zeros(20), lower=np.ones(20),
                           upper=np.ones(20))
        assert T.n_vertices > MAX_VERTICES
        with pytest.raises(ValueError, match="MAX_VERTICES"):
            T.signs()

    def test_name_length_is_checked(self):
        with pytest.raises(ValueError, match="names has length"):
            UncertaintySet(nominal=[1.0], lower=[1.0], upper=[1.0],
                           names=("a", "b"))

    def test_bad_mapping_entry(self):
        with pytest.raises(ValueError, match="nominal"):
            as_uncertainty_set({"a": (1.0,)})

    def test_describe_lists_every_parameter(self):
        T = as_uncertainty_set({"C": (1.0, 0.2), "T": (300.0, 5.0, 10.0)})
        text = T.describe()
        assert "C" in text and "T" in text and "4 vertices" in text


class TestControlSpec:
    def test_unbounded_recourse_is_refused(self):
        with pytest.raises(ValueError, match="finite"):
            ControlSpec(lower=[0.0], upper=[np.inf])

    def test_reversed_bounds_are_refused(self):
        with pytest.raises(ValueError, match="below lower bound"):
            ControlSpec(lower=[1.0], upper=[0.0])

    def test_starts_lie_in_the_box_and_include_the_midpoint(self):
        spec = ControlSpec(lower=[0.0, -2.0], upper=[4.0, 2.0])
        s = np.asarray(spec.starts(3))
        assert s.shape == (3, 2)
        npt.assert_allclose(s[0], [2.0, 0.0])
        assert np.all(s >= np.asarray(spec.lower) - 1e-12)
        assert np.all(s <= np.asarray(spec.upper) + 1e-12)

    def test_explicit_start_replaces_the_midpoint(self):
        spec = ControlSpec(lower=[0.0], upper=[4.0], start=[3.5])
        npt.assert_allclose(np.asarray(spec.starts(2))[0], [3.5])

    def test_no_controls_is_empty(self):
        assert NO_CONTROLS.n == 0
        assert as_control_spec(None) is NO_CONTROLS
        assert np.asarray(NO_CONTROLS.starts(1)).shape == (1, 0)

    def test_bad_mapping_entry(self):
        with pytest.raises(ValueError, match=r"\(lower, upper\)"):
            as_control_spec({"a": (1.0, 2.0, 3.0)})


# --------------------------------------------------------------------------
# The inner problem
# --------------------------------------------------------------------------


class TestInnerProblem:
    def test_smooth_max_brackets_the_true_max(self):
        f = jnp.array([-1.0, 0.5, 0.25])
        for tau in (1.0, 0.1, 1e-3):
            v = float(smooth_max(f, tau))
            assert 0.5 <= v <= 0.5 + tau * np.log(3) + 1e-12

    def test_minimax_lands_on_the_kink(self):
        # min_u max(u - 1, -u - 3):  balanced at u = -1, value -2.
        spec = ControlSpec(lower=[-5.0], upper=[5.0])
        v, u = minimax_value(
            lambda u: jnp.array([u[0] - 1.0, -u[0] - 3.0]), spec)
        assert abs(float(v) + 2.0) < 1e-4
        assert abs(float(u[0]) + 1.0) < 1e-4

    def test_reported_value_is_the_exact_max_not_the_smoothed_one(self):
        # A smoothed value would sit strictly *below* the true max and so make
        # an infeasible design look feasible.  Here the true max at u* is 0.
        spec = ControlSpec(lower=[-5.0], upper=[5.0])
        v, u = minimax_value(lambda u: jnp.array([u[0], -u[0]]), spec)
        f = np.array([float(u[0]), -float(u[0])])
        npt.assert_allclose(float(v), f.max(), atol=1e-12)

    def test_bound_constrained_minimum(self):
        # min over u in [2, 5] of max(u - 1, -u - 3) is attained at u = 2.
        spec = ControlSpec(lower=[2.0], upper=[5.0])
        v, u = minimax_value(
            lambda u: jnp.array([u[0] - 1.0, -u[0] - 3.0]), spec)
        assert abs(float(u[0]) - 2.0) < 1e-6
        assert abs(float(v) - 1.0) < 1e-6

    def test_gradient_is_the_multiplier_weighted_one(self):
        # d/dtheta min_u max(u - 2, theta - u) = 1/2.  Differentiating
        # whichever row jnp.max selected would give 1 or 0.
        spec = ControlSpec(lower=[-10.0], upper=[10.0])

        def value(theta):
            v, _ = minimax_value(
                lambda u: jnp.array([u[0] - 2.0, theta - u[0]]), spec)
            return v

        g = float(jax.grad(value)(1.0))
        assert abs(g - 0.5) < 5e-3, g

    def test_no_recourse_reduces_to_the_plain_max(self):
        v, u = minimax_value(lambda u: jnp.array([-1.0, 3.0]), NO_CONTROLS)
        assert float(v) == 3.0
        assert u.shape == (0,)

    def test_solver_options_is_a_params_mixin(self):
        opts = SolverOptions(steps=400, n_starts=5)
        assert opts["steps"] == 400
        assert opts.update(steps=10)["steps"] == 10
        assert DEFAULT_OPTIONS.steps == 250


# --------------------------------------------------------------------------
# The feasibility function
# --------------------------------------------------------------------------


class TestFeasibilityFunction:
    def test_matches_hand_arithmetic(self):
        for d in (1.0, 1.5, 2.0, 3.0):
            psi = float(feasibility_value(linear_model, [d], LINEAR_SET,
                                          LINEAR_CONTROLS))
            assert abs(psi - linear_psi(d)) < 1e-4, (d, psi)

    def test_scaling_the_set_scales_psi(self):
        for delta in (0.0, 0.5, 1.0, 2.5):
            psi = float(feasibility_value(linear_model, [2.0], LINEAR_SET,
                                          LINEAR_CONTROLS, scale=delta))
            assert abs(psi - linear_psi(2.0, delta)) < 1e-4, delta

    def test_psi_sign_tracks_feasibility(self):
        # Feasible exactly when d >= 1.5 (the design must cover the richest
        # feed the envelope allows).  Designs within 0.05 of the boundary are
        # skipped: their verdict is a coin flip on solver noise, not on logic.
        for d in np.linspace(0.8, 2.4, 17):
            res = feasibility_function(linear_model, [d], LINEAR_SET,
                                       LINEAR_CONTROLS)
            if abs(d - 1.5) < 0.05:
                continue
            assert res.feasible == bool(d >= 1.5), (d, res.psi)
            assert (res.psi <= 0.0) == res.feasible
            assert res.margin == pytest.approx(-res.psi)

    def test_reports_the_critical_vertex_and_the_binding_constraint(self):
        res = feasibility_function(linear_model, [2.0], LINEAR_SET,
                                   LINEAR_CONTROLS,
                                   constraint_names=("capacity", "coverage"))
        npt.assert_allclose(res.critical_theta, [1.5])       # the rich corner
        assert res.critical_vertex == 1
        assert res.critical_point() == {"feed": 1.5}
        # Both rows are active at the kink; either name is a correct report.
        assert res.binding_constraint in ("capacity", "coverage")
        assert set(res.active_constraints(tol=1e-4)) == {"capacity",
                                                         "coverage"}
        assert abs(float(res.controls[0]) - 1.75) < 1e-4

    def test_critical_realization_is_the_true_argmax(self):
        """The reported critical point beats a 31^3 x 4001 brute-force grid."""
        res = feasibility_function(curved_model, [1.0], CURVED_SET,
                                   CURVED_CONTROLS)

        def f_np(thetas, us):
            f0 = thetas[:, 0:1] + 0.5 * thetas[:, 1:2] ** 2 - us[None, :] - 1.0
            f1 = us[None, :] - 2.0 - thetas[:, 2:3] - 1.0
            return np.stack([f0, f1])

        lo = np.array([0.7, 1.5, 0.25])
        hi = np.array([1.2, 2.4, 0.6])
        psi_bf, theta_bf = brute_force_psi(lo, hi, 0.0, 6.0, f_np)
        assert abs(res.psi - psi_bf) < 1e-3, (res.psi, psi_bf)
        npt.assert_allclose(res.critical_theta, theta_bf, atol=1e-6)
        # The grid can only be beaten by a genuinely better point, so the
        # reported psi may not fall below it by more than grid resolution.
        assert res.psi >= psi_bf - 1e-3

    def test_no_recourse_is_a_plain_worst_corner(self):
        res = feasibility_function(corner_model, [1.0], CORNER_SET, None)
        # max over the 4 corners of max(a + b - 1, -a - 2) = 1 at (+1, +1).
        assert res.psi == pytest.approx(1.0)
        assert not res.feasible
        npt.assert_allclose(res.critical_theta, [1.0, 1.0])
        assert res.binding_constraint == "f0"
        npt.assert_allclose(res.vertex_psi, [-1.0, -1.0, -1.0, 1.0])
        assert "no recourse" in res.summary()

    def test_recourse_strictly_helps(self):
        with_r = float(feasibility_value(linear_model, [2.0], LINEAR_SET,
                                         LINEAR_CONTROLS))
        # Freeze the control at the nominal optimum u = 1.5 by collapsing the
        # box; the same design then has less margin.
        without = float(feasibility_value(linear_model, [2.0], LINEAR_SET,
                                          {"u": (1.5, 1.5)}))
        assert with_r == pytest.approx(-0.25, abs=1e-4)
        assert without == pytest.approx(0.0, abs=1e-6)
        assert with_r < without

    def test_continuous_finds_interior_critical_point(self):
        # psi_true = max over theta in [-1, 1] of (0.5 - theta^2 - d) = 0.2 at
        # theta = 0, but both vertices give -0.8.  Vertex enumeration is
        # structurally unable to see this; the continuous fallback must.
        model = lambda d, u, th: jnp.array([0.5 - th[0] ** 2 - d[0]])
        T = {"th": (0.0, 1.0)}
        rv = feasibility_function(model, [0.3], T, None, method="vertex")
        assert rv.psi == pytest.approx(-0.8)
        assert rv.feasible

        rc = feasibility_function(model, [0.3], T, None, method="continuous")
        assert rc.psi == pytest.approx(0.2, abs=1e-3)
        assert not rc.feasible
        assert rc.critical_vertex == -1
        assert abs(float(rc.critical_theta[0])) < 0.05
        assert "interior point" in rc.summary()

    def test_continuous_never_reports_less_than_vertex(self):
        rv = feasibility_function(curved_model, [1.0], CURVED_SET,
                                  CURVED_CONTROLS, method="vertex")
        rc = feasibility_function(curved_model, [1.0], CURVED_SET,
                                  CURVED_CONTROLS, method="continuous")
        assert rc.psi >= rv.psi - 1e-9

    def test_vertex_values_returns_one_row_per_vertex(self):
        T = as_uncertainty_set(CORNER_SET)
        vals, us, verts = vertex_values(corner_model, jnp.array([1.0]), T,
                                        NO_CONTROLS)
        assert vals.shape == (4,)
        assert verts.shape == (4, 2)
        assert us.shape == (4, 0)

    def test_inner_value_at_one_realization(self):
        v, u = inner_value(linear_model, jnp.array([2.0]), jnp.array([1.2]),
                           as_control_spec(LINEAR_CONTROLS))
        assert float(v) == pytest.approx((1.2 - 2.0) / 2, abs=1e-4)
        assert float(u[0]) == pytest.approx((1.2 + 2.0) / 2, abs=1e-4)

    def test_bad_method_is_refused(self):
        with pytest.raises(ValueError, match="method must be one of"):
            feasibility_function(linear_model, [2.0], LINEAR_SET,
                                 LINEAR_CONTROLS, method="kkt")

    def test_constraint_name_count_is_checked(self):
        with pytest.raises(ValueError, match="expected 2 f names"):
            feasibility_function(linear_model, [2.0], LINEAR_SET,
                                 LINEAR_CONTROLS,
                                 constraint_names=("only-one",))

    def test_reports_render(self):
        res = feasibility_function(linear_model, [2.0], LINEAR_SET,
                                   LINEAR_CONTROLS)
        assert isinstance(res, FeasibilityResult)
        assert "FEASIBLE" in res.summary()
        assert "max_theta min_u max_j" in res.describe()
        assert "psi=" in repr(res)


# --------------------------------------------------------------------------
# Differentiability
# --------------------------------------------------------------------------


class TestDerivatives:
    def test_gradient_is_the_multiplier_weighted_one(self):
        # psi(d) = (1.5 - d)/2, so dpsi/dd = -0.5 exactly.  The two ways of
        # getting this wrong -- taking the derivative of whichever constraint
        # jnp.max selected at the kink -- give -1.0 or 0.0.  Neither is inside
        # the tolerance.
        g = jax.grad(lambda d: feasibility_value(
            linear_model, d, LINEAR_SET, LINEAR_CONTROLS))(jnp.array([2.0]))
        assert np.all(np.isfinite(np.asarray(g)))
        assert abs(float(g[0]) + 0.5) < 5e-3, float(g[0])

    def test_gradient_matches_a_central_difference(self):
        d0 = jnp.array([1.0])
        g = jax.grad(lambda d: feasibility_value(
            curved_model, d, CURVED_SET, CURVED_CONTROLS))(d0)
        h = 1e-3
        fd = (feasibility_value(curved_model, d0 + h, CURVED_SET,
                                CURVED_CONTROLS)
              - feasibility_value(curved_model, d0 - h, CURVED_SET,
                                  CURVED_CONTROLS)) / (2 * h)
        assert np.all(np.isfinite(np.asarray(g)))
        assert abs(float(g[0]) - float(fd)) < 5e-3, (g, fd)
        # ... and the analytic value, which is -1/2 here too.
        assert abs(float(g[0]) + 0.5) < 5e-3

    def test_gradient_of_a_no_recourse_problem(self):
        # psi(d) = 1 - d exactly, so the gradient is -1.
        g = jax.grad(lambda d: feasibility_value(
            corner_model, d, CORNER_SET, None))(jnp.array([1.0]))
        assert float(g[0]) == pytest.approx(-1.0)

    def test_jit_agrees_with_eager(self):
        fn = jax.jit(lambda d: feasibility_value(linear_model, d, LINEAR_SET,
                                                 LINEAR_CONTROLS))
        for d in (1.8, 2.0, 2.5):
            eager = float(feasibility_value(linear_model, jnp.array([d]),
                                            LINEAR_SET, LINEAR_CONTROLS))
            jitted = float(fn(jnp.array([d])))
            # Both must land on the analytic answer; they may disagree only at
            # the level of the inner solver's own noise.
            assert jitted == pytest.approx(linear_psi(d), abs=1e-4)
            assert jitted == pytest.approx(eager, abs=1e-6)

    def test_jit_of_the_gradient(self):
        g = jax.jit(jax.grad(lambda d: feasibility_value(
            linear_model, d, LINEAR_SET, LINEAR_CONTROLS)))(jnp.array([2.0]))
        assert np.all(np.isfinite(np.asarray(g)))
        assert abs(float(g[0]) + 0.5) < 5e-3

    def test_vmap_over_designs(self):
        ds = jnp.array([[1.5], [2.0], [2.5]])
        got = jax.vmap(lambda d: feasibility_value(
            linear_model, d, LINEAR_SET, LINEAR_CONTROLS))(ds)
        want = [linear_psi(float(d[0])) for d in ds]
        npt.assert_allclose(np.asarray(got), want, atol=1e-4)

    def test_vertex_limits_is_jittable(self):
        got = jax.jit(lambda d: vertex_limits(linear_model, d, LINEAR_SET,
                                              LINEAR_CONTROLS))(
            jnp.array([2.0]))
        npt.assert_allclose(np.asarray(got), [4.0, 2.0], atol=1e-4)


# --------------------------------------------------------------------------
# The flexibility index
# --------------------------------------------------------------------------


class TestFlexibilityIndex:
    def test_matches_hand_arithmetic(self):
        # psi(d, delta) = (1 + 0.5 delta - d)/2 = 0  =>  F = (d - 1)/0.5
        for d in (1.25, 1.5, 2.0, 2.5):
            res = flexibility_index(linear_model, [d], LINEAR_SET,
                                    LINEAR_CONTROLS)
            want = (d - 1.0) / 0.5
            assert res.index == pytest.approx(want, abs=2e-3), d
            if abs(want - 1.0) > 0.05:      # d = 1.5 sits exactly on F = 1
                assert res.covers_envelope == bool(want >= 1.0), d

    def test_index_no_recourse_matches_hand_arithmetic(self):
        # f0 = a + b - 1, f1 = -a - 2, with a, b in [-delta, +delta].
        #   (-,-): max(-2d - 1, d - 2) <= 0 until delta = 2
        #   (-,+): max(     -1, d - 2) <= 0 until delta = 2
        #   (+,-): max(     -1, -d - 2) <= 0 always -> saturates at delta_max
        #   (+,+): max(2d - 1, -d - 2) <= 0 until delta = 0.5   <-- binds
        res = flexibility_index(corner_model, [1.0], CORNER_SET, None,
                                delta_max=4.0,
                                constraint_names=("sum", "floor"))
        npt.assert_allclose(res.vertex_limits, [2.0, 2.0, 4.0, 0.5], atol=2e-3)
        assert res.index == pytest.approx(0.5, abs=2e-3)
        assert res.limited_by_vertex == 3
        assert res.direction() == {"a": "+", "b": "+"}
        assert res.binding_constraint == "sum"
        assert not res.covers_envelope
        npt.assert_allclose(res.critical_theta, [0.5, 0.5], atol=2e-3)
        assert res.slack_vertices(factor=1.5) == [0, 1, 2]
        assert res.nominal_feasible and not res.saturated

    def test_index_is_the_largest_feasible_scaling(self):
        res = flexibility_index(linear_model, [2.0], LINEAR_SET,
                                LINEAR_CONTROLS)
        F = res.index
        inside = float(feasibility_value(linear_model, [2.0], LINEAR_SET,
                                         LINEAR_CONTROLS, scale=0.98 * F))
        outside = float(feasibility_value(linear_model, [2.0], LINEAR_SET,
                                          LINEAR_CONTROLS, scale=1.02 * F))
        assert inside <= 0.0 < outside

    def test_index_agrees_with_bisecting_psi_directly(self):
        # An independent route to the same number: bisect the feasibility
        # function itself instead of each vertex direction.
        res = flexibility_index(curved_model, [1.0], CURVED_SET,
                                CURVED_CONTROLS, delta_max=6.0)
        lo, hi = 0.0, 6.0
        for _ in range(45):
            mid = 0.5 * (lo + hi)
            if float(feasibility_value(curved_model, [1.0], CURVED_SET,
                                       CURVED_CONTROLS, scale=mid)) <= 0.0:
                lo = mid
            else:
                hi = mid
        assert res.index == pytest.approx(lo, abs=5e-3)

    def test_zero_when_the_nominal_point_is_already_infeasible(self):
        res = flexibility_index(lambda d, u, th: jnp.array([th[0] - d[0]]),
                                [0.5], {"a": (1.0, 0.5)}, None)
        assert res.index == 0.0
        assert not res.nominal_feasible
        assert not res.covers_envelope
        assert "infeasible at the nominal point" in res.summary()

    def test_saturation_is_flagged_not_hidden(self):
        res = flexibility_index(lambda d, u, th: jnp.array([th[0] - d[0]]),
                                [50.0], {"a": (1.0, 0.5)}, None, delta_max=3.0)
        assert res.saturated
        assert res.index == pytest.approx(3.0, abs=1e-6)
        assert "saturated" in res.summary()

    def test_recourse_raises_the_index(self):
        # The same design, with and without a control that can respond.
        model = lambda d, u, th: jnp.array([th[0] - u.sum() - 1.0,
                                            u.sum() - d[0]])
        free = flexibility_index(model, [2.0], {"a": (1.0, 0.5)},
                                 {"u": (0.0, 5.0)})
        frozen = flexibility_index(model, [2.0], {"a": (1.0, 0.5)}, None)
        # frozen: max(theta - 1, -2) <= 0 until theta = 1, i.e. delta = 0.
        # free:   min_u max(theta - u - 1, u - 2) = (theta - 3)/2 <= 0
        #         until theta = 3, i.e. delta = 4.
        assert frozen.index == pytest.approx(0.0, abs=2e-3)
        assert free.index == pytest.approx(4.0, abs=5e-3)

    def test_reports_render(self):
        res = flexibility_index(corner_model, [1.0], CORNER_SET, None)
        assert isinstance(res, FlexibilityResult)
        text = res.describe()
        assert "flexibility index" in text and "vertex" in text
        assert "index=" in repr(res)
        assert res.critical_point().keys() == {"a", "b"}


# --------------------------------------------------------------------------
# The stochastic counterpart
# --------------------------------------------------------------------------


class TestExpectedFeasibility:
    def test_probability_matches_the_analytic_one(self):
        # psi_sample = (theta - d)/2 <= 0 iff theta <= d, and theta is uniform
        # on [0.5, 1.5], so P(feasible) = d - 0.5.
        for d, want in ((0.75, 0.25), (1.0, 0.5), (1.25, 0.75)):
            res = expected_feasibility(linear_model, [d], LINEAR_SET,
                                       LINEAR_CONTROLS, n_samples=1500, key=7)
            assert abs(res.probability - want) < 4 * res.standard_error, (
                d, res.probability, want)
            assert res.n_samples == 1500

    def test_standard_error_shrinks_with_the_sample_size(self):
        small = expected_feasibility(linear_model, [1.0], LINEAR_SET,
                                     LINEAR_CONTROLS, n_samples=100, key=1)
        big = expected_feasibility(linear_model, [1.0], LINEAR_SET,
                                   LINEAR_CONTROLS, n_samples=1600, key=1)
        assert big.standard_error < 0.35 * small.standard_error

    def test_worst_case_bounds_the_samples(self):
        res = expected_feasibility(linear_model, [2.0], LINEAR_SET,
                                   LINEAR_CONTROLS, n_samples=300, key=2)
        psi = float(feasibility_value(linear_model, [2.0], LINEAR_SET,
                                      LINEAR_CONTROLS))
        assert res.worst <= psi + 1e-6          # samples live inside the set
        assert res.mean <= res.worst
        assert res.probability == 1.0           # psi <= 0 => every sample ok

    def test_blame_names_the_constraint_that_actually_fails(self):
        # f0 fails about half the time; f1 is a constant -5 and never can.
        model = lambda d, u, th: jnp.array([th[0] - d[0], -5.0])
        res = expected_feasibility(model, [1.0], LINEAR_SET, None,
                                   n_samples=400, key=3,
                                   constraint_names=("purity", "duty"))
        npt.assert_allclose(res.blame, [1.0, 0.0])
        assert res.violation_rate[1] == 0.0
        assert 0.4 < res.violation_rate[0] < 0.6
        assert "purity" in res.summary()

    def test_chance_margin_is_the_quantile_and_gates_correctly(self):
        res = expected_feasibility(linear_model, [1.0], LINEAR_SET,
                                   LINEAR_CONTROLS, n_samples=800, key=5)
        assert res.chance_margin(0.9) == pytest.approx(res.quantile(0.9))
        assert res.chance_margin(0.9) > 0.0 and not res.satisfies(0.9)
        # The design is feasible on about half the draws, so a 40% reliability
        # requirement is met and a 90% one is not.
        assert res.satisfies(0.4)
        assert res.quantile(0.9) > res.quantile(0.4)

    def test_worst_sample_is_inside_the_set(self):
        res = expected_feasibility(linear_model, [1.0], LINEAR_SET,
                                   LINEAR_CONTROLS, n_samples=200, key=4)
        w = res.worst_sample()
        assert 0.5 - 1e-9 <= w["feed"] <= 1.5 + 1e-9
        assert res.values[int(np.argmax(res.values))] == pytest.approx(
            res.worst)

    def test_sampling_is_reproducible(self):
        kw = dict(n_samples=64, key=11)
        a = expected_feasibility(linear_model, [1.0], LINEAR_SET,
                                 LINEAR_CONTROLS, **kw)
        b = expected_feasibility(linear_model, [1.0], LINEAR_SET,
                                 LINEAR_CONTROLS, **kw)
        npt.assert_array_equal(a.samples, b.samples)
        assert a.probability == b.probability

    def test_uniform_samples_fill_the_scaled_box(self):
        s = np.asarray(sample_set({"a": (0.0, 1.0, 2.0)}, 2000, 1,
                                  distribution="uniform"))
        assert -1.0 <= s.min() and s.max() <= 2.0
        assert s.min() < -0.95 and s.max() > 1.9
        s2 = np.asarray(sample_set({"a": (0.0, 1.0, 2.0)}, 2000, 1, scale=2.0))
        assert s2.min() < -1.9 and s2.max() > 3.8

    def test_normal_samples_keep_the_asymmetry(self):
        T = as_uncertainty_set({"a": (0.0, 1.0, 2.0)})
        s = np.asarray(sample_set(T, 20000, 1, distribution="normal")).ravel()
        # Two-piece normal: each side's one-sigma half-width is its deviation.
        assert s[s < 0].std() == pytest.approx(1.0 * np.sqrt(1 - 2 / np.pi),
                                               rel=0.06)
        assert s[s > 0].std() == pytest.approx(2.0 * np.sqrt(1 - 2 / np.pi),
                                               rel=0.06)
        # Roughly 68% of draws land inside the delta = 1 box.
        inside = np.mean((s >= -1.0) & (s <= 2.0))
        assert abs(inside - 0.6827) < 0.02

    def test_unknown_distribution_is_refused(self):
        with pytest.raises(ValueError, match="distribution must be one of"):
            sample_set({"a": (1.0, 0.5)}, 4, 0, distribution="beta")

    def test_result_type_and_repr(self):
        res = expected_feasibility(linear_model, [2.0], LINEAR_SET,
                                   LINEAR_CONTROLS, n_samples=32, key=0)
        assert isinstance(res, StochasticFeasibilityResult)
        assert "P=" in repr(res)


# --------------------------------------------------------------------------
# Feed uncertainty versus parameter uncertainty -- the point of the module
# --------------------------------------------------------------------------


def penalty_model(d, u, theta, phi):
    """``u`` may respond to ``theta`` but never to ``phi``.

    Written with ``u.sum()`` so the same model can be posed with and without
    recourse, which is what makes the comparison a controlled experiment.
    """
    x = u.sum()
    return jnp.array([phi[0] * theta[0] - x - 1.0, x - d[0]])


PENALTY_KW = dict(parameters={"K": 1.0}, covariance=[[0.01]])


class TestPenaltySplit:
    def test_matches_hand_arithmetic(self):
        # d = 2, feed 1.0 +/- 0.2, K = 1, sigma_K = 0.1, kappa = 2.
        #   nominal : min_u max(1 - u - 1, u - 2) = -1 at u = 1
        #   theta=0.8: value -1.1;  theta=1.2: value -0.9  -> feed_worst -0.9
        #   feed penalty = -0.9 - (-1) = 0.1 on both rows
        #   frozen at u = 1: f = [theta - 2, -1] -> worst [-0.8, -1]
        #     recourse credit = [-0.8 + 0.9, -1 + 0.9] = [0.1, -0.1]
        #   d f0 / d K = theta* = 1.2, so sigma = 0.12 and back-off = 0.24
        rep = uncertainty_penalties(penalty_model, [2.0], {"feed": (1.0, 0.2)},
                                    {"u": (0.0, 5.0)}, **PENALTY_KW)
        npt.assert_allclose(rep.nominal, [-1.0, -1.0], atol=1e-4)
        npt.assert_allclose(rep.feed_worst, [-0.9, -0.9], atol=1e-4)
        npt.assert_allclose(rep.feed_penalty, [0.1, 0.1], atol=1e-4)
        npt.assert_allclose(rep.recourse_credit, [0.1, -0.1], atol=1e-4)
        npt.assert_allclose(rep.backoff, [0.24, 0.0], atol=1e-9)
        npt.assert_allclose(rep.sigma, [0.12, 0.0], atol=1e-9)
        npt.assert_allclose(rep.required_margin, [0.34, 0.1], atol=1e-4)
        npt.assert_allclose(rep.jacobian, [[1.2], [0.0]], atol=1e-9)
        assert rep.psi_feed == pytest.approx(-0.9, abs=1e-4)
        npt.assert_allclose(rep.critical_theta, [1.2])

    def test_recourse_cuts_the_feed_penalty_but_not_the_backoff(self):
        """The claim the module exists to make measurable.

        Feed variability is answerable -- re-optimizing the control halves its
        penalty.  Parameter uncertainty is not: the back-off is identical
        whether or not there is a control, because you cannot schedule a move
        against a constant you do not know.
        """
        free = uncertainty_penalties(penalty_model, [2.0],
                                     {"feed": (1.0, 0.2)},
                                     {"u": (0.0, 5.0)}, **PENALTY_KW)
        frozen = uncertainty_penalties(penalty_model, [2.0],
                                       {"feed": (1.0, 0.2)}, None,
                                       **PENALTY_KW)
        assert frozen.feed_penalty[0] == pytest.approx(0.2, abs=1e-4)
        assert free.feed_penalty[0] == pytest.approx(0.1, abs=1e-4)
        assert free.feed_penalty[0] < frozen.feed_penalty[0]

        npt.assert_allclose(free.backoff, frozen.backoff, atol=1e-9)
        assert free.backoff[0] == pytest.approx(0.24, abs=1e-9)

        # And with no control there is nothing for recourse to credit.
        npt.assert_allclose(frozen.recourse_credit, [0.0, 0.0], atol=1e-12)

    def test_backoff_scales_with_kappa_and_with_sigma(self):
        a = uncertainty_penalties(penalty_model, [2.0], {"feed": (1.0, 0.2)},
                                  {"u": (0.0, 5.0)}, kappa=3.0, **PENALTY_KW)
        b = uncertainty_penalties(penalty_model, [2.0], {"feed": (1.0, 0.2)},
                                  {"u": (0.0, 5.0)}, parameters={"K": 1.0},
                                  covariance=[[0.04]])
        assert a.backoff[0] == pytest.approx(0.36, abs=1e-9)   # 3 * 1.2 * 0.1
        assert b.backoff[0] == pytest.approx(0.48, abs=1e-9)   # 2 * 1.2 * 0.2

    def test_dominant_names_the_bigger_bill(self):
        rep = uncertainty_penalties(penalty_model, [2.0], {"feed": (1.0, 0.2)},
                                    {"u": (0.0, 5.0)}, **PENALTY_KW)
        assert rep.dominant == {"f0": "parameter", "f1": "feed"}
        # Shrink the parameter uncertainty and the verdict flips.
        cheap = uncertainty_penalties(penalty_model, [2.0],
                                      {"feed": (1.0, 0.2)},
                                      {"u": (0.0, 5.0)},
                                      parameters={"K": 1.0},
                                      covariance=[[1e-6]])
        assert cheap.dominant["f0"] == "feed"

    def test_feasible_charges_both_penalties(self):
        rep = uncertainty_penalties(penalty_model, [2.0], {"feed": (1.0, 0.2)},
                                    {"u": (0.0, 5.0)}, **PENALTY_KW)
        # worst_case = feed_worst + backoff = [-0.9 + 0.24, -0.9] < 0
        npt.assert_allclose(rep.worst_case(), [-0.66, -0.9], atol=1e-4)
        assert rep.feasible()
        # A tighter design fails once the back-off is charged even though the
        # feed worst case alone clears zero.  At d = 0.5 the critical vertex
        # gives psi_feed = (1.2 - 1 - 0.5)/2 = -0.15, and -0.15 + 0.24 > 0.
        tight = uncertainty_penalties(penalty_model, [0.5],
                                      {"feed": (1.0, 0.2)},
                                      {"u": (0.0, 5.0)}, **PENALTY_KW)
        assert tight.psi_feed == pytest.approx(-0.15, abs=1e-4)
        npt.assert_allclose(tight.worst_case(), [0.09, -0.15], atol=1e-4)
        assert not tight.feasible()

    def test_correlated_covariance_is_propagated(self):
        # f0 = (K + 2 M) * theta - x - 1, so d f0/d(K, M) = theta * (1, 2) and
        # sigma^2 = theta^2 (1, 2) Sigma (1, 2)^T.
        def model(d, u, theta, phi):
            x = u.sum()
            return jnp.array([(phi[0] + 2 * phi[1]) * theta[0] - x - 1.0,
                              x - d[0]])

        cov = np.array([[0.01, 0.004], [0.004, 0.0025]])
        rep = uncertainty_penalties(model, [4.0], {"feed": (1.0, 0.2)},
                                    {"u": (0.0, 8.0)},
                                    parameters={"K": 1.0, "M": 0.0},
                                    covariance=cov, kappa=1.0)
        theta_star = float(rep.critical_theta[0])
        g = theta_star * np.array([1.0, 2.0])
        want = np.sqrt(g @ cov @ g)
        assert rep.sigma[0] == pytest.approx(want, rel=1e-8)
        npt.assert_allclose(rep.jacobian[0], g, rtol=1e-8)

    def test_covariance_shape_is_checked(self):
        with pytest.raises(ValueError, match="covariance has shape"):
            uncertainty_penalties(penalty_model, [2.0], {"feed": (1.0, 0.2)},
                                  None, parameters={"K": 1.0},
                                  covariance=np.eye(2))

    def test_missing_nominal_parameter_is_refused(self):
        with pytest.raises(KeyError, match="no nominal value"):
            uncertainty_penalties(penalty_model, [2.0], {"feed": (1.0, 0.2)},
                                  None, parameters={"K": 1.0},
                                  covariance=[[0.01]],
                                  parameter_order=["Q"])

    def test_variance_vector_is_accepted_for_covariance(self):
        a = uncertainty_penalties(penalty_model, [2.0], {"feed": (1.0, 0.2)},
                                  {"u": (0.0, 5.0)}, parameters={"K": 1.0},
                                  covariance=[0.01])
        b = uncertainty_penalties(penalty_model, [2.0], {"feed": (1.0, 0.2)},
                                  {"u": (0.0, 5.0)}, **PENALTY_KW)
        npt.assert_allclose(a.backoff, b.backoff)

    def test_reports_render(self):
        rep = uncertainty_penalties(penalty_model, [2.0], {"feed": (1.0, 0.2)},
                                    {"u": (0.0, 5.0)}, **PENALTY_KW)
        assert isinstance(rep, PenaltyReport)
        text = rep.describe()
        assert "controls and instruments" in text
        assert "experiments" in text
        assert rep.as_dict()["f0"]["total"] == pytest.approx(0.34, abs=1e-4)
        assert "kappa=2" in repr(rep)


# --------------------------------------------------------------------------
# The worked example in docs/flexibility.md
# --------------------------------------------------------------------------

C_MAX = 0.05


def ree_stage(d, u, theta, phi):
    """One mixer-settler stage: ``c_raff = c_feed / (1 + K S)``.

    Purity holds when ``1 + K S >= c_feed / C_MAX`` and capacity when
    ``S <= d``, so with ``K = 1`` the design covers a feed of
    ``(d + 1) * C_MAX``.
    """
    c_feed, = theta
    K, = phi
    S = u.sum()
    return jnp.array([c_feed / (1.0 + K * S) / C_MAX - 1.0,
                      S / d[0] - 1.0])


def ree_model(d, u, theta):
    return ree_stage(d, u, theta, jnp.array([1.0]))


REE_FEED = {"c_feed": (1.0, 0.3)}
REE_CTRL = {"S": (0.0, 60.0)}


class TestDocumentedExample:
    def test_index_matches_the_hand_derivation(self):
        # Covered feed is (26 + 1) * 0.05 = 1.35, i.e. delta = 0.35/0.3.
        res = flexibility_index(ree_model, [26.0], REE_FEED, REE_CTRL,
                                constraint_names=("purity", "capacity"))
        assert res.index == pytest.approx(0.35 / 0.3, abs=1e-3)
        assert res.binding_constraint == "purity"
        assert res.covers_envelope
        npt.assert_allclose(res.critical_theta, [1.35], atol=1e-3)

    def test_freezing_the_control_more_than_halves_the_index(self):
        # Pinned at the nominal optimum S = 22.309, the covered feed is
        # (1 + 22.309) * 0.05 = 1.16545, i.e. delta = 0.5515.
        frozen = flexibility_index(ree_model, [26.0], REE_FEED,
                                   {"S": (22.309, 22.309)})
        assert frozen.index == pytest.approx(0.5515, abs=1e-3)

    def test_backoff_can_sink_a_design_the_index_passes(self):
        """The headline of the documentation, asserted.

        ``F = 1.17`` says the design covers its feed envelope with room to
        spare, and a 5% uncertainty on the distribution coefficient sinks it
        anyway.  Reporting only one of the two numbers would have passed it.
        """
        idx = flexibility_index(ree_model, [26.0], REE_FEED, REE_CTRL)
        rep = uncertainty_penalties(ree_stage, [26.0], REE_FEED, REE_CTRL,
                                    parameters={"K": 1.0},
                                    covariance=[[0.05 ** 2]],
                                    constraint_names=("purity", "capacity"))
        assert idx.covers_envelope                 # flexibility says yes
        assert rep.psi_feed < 0.0                  # so does psi on its own
        assert not rep.feasible()                  # the back-off says no

        # d f0 / d K = -c_feed S / ((1 + K S)^2 C_MAX) at the critical vertex.
        c, S = float(rep.critical_theta[0]), float(rep.critical_controls[0])
        want = -c * S / ((1.0 + S) ** 2 * C_MAX)
        assert rep.jacobian[0, 0] == pytest.approx(want, rel=1e-6)
        assert rep.backoff[0] == pytest.approx(2.0 * abs(want) * 0.05,
                                               rel=1e-6)
        assert rep.backoff[0] > abs(rep.psi_feed)


# --------------------------------------------------------------------------
# Drawings
# --------------------------------------------------------------------------


class TestDiagrams:
    def test_flexibility_region(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        res = flexibility_index(corner_model, [1.0], CORNER_SET, None)
        ax = draw_flexibility_region(res)
        assert ax.get_xlabel() == "a" and ax.get_ylabel() == "b"
        assert len(ax.patches) == 2          # stated and covered envelopes
        plt.close("all")

    def test_flexibility_region_needs_two_parameters(self):
        pytest.importorskip("matplotlib")
        res = flexibility_index(lambda d, u, th: jnp.array([th[0] - d[0]]),
                                [2.0], {"a": (1.0, 0.5)}, None)
        with pytest.raises(ValueError, match="two parameters"):
            draw_flexibility_region(res)

    def test_penalty_split(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rep = uncertainty_penalties(penalty_model, [2.0], {"feed": (1.0, 0.2)},
                                    {"u": (0.0, 5.0)}, **PENALTY_KW)
        ax = draw_penalty_split(rep)
        assert [t.get_text() for t in ax.get_yticklabels()] == ["f0", "f1"]
        assert len(ax.containers) == 2       # feed bar and parameter bar
        plt.close("all")


# --------------------------------------------------------------------------
# Doctests carry worked examples; keep them honest.
# --------------------------------------------------------------------------


def test_module_doctests():
    import doctest

    from difflow.flexibility import (
        feasibility as _feasibility,
        index as _index,
        inner as _inner,
        penalties as _penalties,
        sets as _sets,
        stochastic as _stochastic,
    )

    failures = 0
    for mod in (_sets, _inner, _feasibility, _index, _stochastic, _penalties):
        result = doctest.testmod(mod, verbose=False)
        assert result.attempted > 0, mod.__name__
        failures += result.failed
    assert failures == 0
