"""Tests for the pounce / discopt bridges (:mod:`difflow.solvers`, #203).

The load-bearing claims, and where each is pinned:

* ``as_nlp`` produces ``f`` and ``g`` whose derivatives are the model's --
  :func:`test_objective_gradient_matches_central_difference`,
  :func:`test_constraint_jacobian_matches_central_differences`.
* The equality rows *are* the flowsheet, so a feasible point is a converged
  flowsheet -- :func:`test_residuals_vanish_at_the_sequential_modular_solution`,
  :func:`test_optimum_is_a_converged_flowsheet`.
* **The sparsity trap.** pounce probes at random ``N(0, 1)`` points unless a
  pattern is supplied. A difflow model evaluated at ``T ~ -1.3 K`` returns
  ``inf``/``nan`` derivatives, and ``nan > eps`` is ``False``, so those entries
  are recorded as *structural zeros*: the probe silently drops the entire
  reactor-volume column and the subsequent solve reports
  ``Infeasible_Problem_Detected``. Pinned in
  :func:`test_probing_drops_real_entries_of_the_jacobian` and
  :func:`test_probed_pattern_breaks_the_solve_that_the_adapter_completes`.
* ``mult_g`` is the bound sensitivity, with a sign that was measured in both
  directions -- :func:`test_mult_g_is_d_objective_d_bound_for_a_ge_spec`,
  :func:`test_mult_g_is_d_objective_d_bound_for_a_le_spec`.
* The residual view composes correctly through ``dm.implicit`` --
  :func:`test_discopt_implicit_derivative_matches_the_implicit_function_theorem`.
* The one thing a difflow block in discopt cannot do --
  :func:`test_discopt_refuses_integrality_at_build_time`,
  :func:`test_discopt_itself_raises_on_integrality`.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from difflow import (  # noqa: E402
    CSTR,
    CSTRParams,
    Flowsheet,
    IdealThermo,
    SpeciesData,
    Unit,
    make_stream,
)
from difflow.eo_solver import solve_residual_system  # noqa: E402
from difflow.solvers import (  # noqa: E402
    Decision,
    Parameter,
    Spec,
    SparsityPatternError,
    as_implicit,
    as_nlp,
    as_residual,
    bound_sensitivities,
    check_no_integrality,
    CUSTOMCALL_RESTRICTION,
    dense_hessian_pattern,
    dense_jacobian_pattern,
    differentiable_problem,
    DiscoptIntegralityError,
    optimize_flowsheet,
    require_eo_residuals,
    residual_from_system,
    solve_with_pounce,
    validate_patterns,
)
from difflow.solvers._lazy import have, require  # noqa: E402

HAVE_POUNCE = have("pounce.jax")
HAVE_DISCOPT = have("discopt.modeling")

needs_pounce = pytest.mark.skipif(
    not HAVE_POUNCE, reason="pounce not installed (pip install pounce-solver[jax])"
)
needs_discopt = pytest.mark.skipif(
    not HAVE_DISCOPT, reason="discopt not installed"
)


# =============================================================================
# The model: one isothermal CSTR, A -> B, first order.
#
# Chosen because its optimum is available in closed form. For a first-order
# reaction in a single CSTR at fixed T,
#
#     F_B = F_A0 * tau k / (1 + tau k),   tau = V / Q
#
# so the smallest volume meeting F_B >= b is V* = Q b / (k (F_A0 - b)). Every
# end-to-end assertion below is against that number, not against a golden value
# recorded from a previous run.
# =============================================================================

R_GAS = 8.314
FEED_A = 10.0
FEED_T = 320.0
P0 = 101325.0
QV = 0.01
PRE_EXP = 1.0e6
E_ACT = 50_000.0


def _rate_fn(C, T, p):
    k = p["A"] * jnp.exp(-p["Ea"] / (R_GAS * T))
    return jnp.array([k * C["A"]])


def _thermo():
    data = {
        s: SpeciesData(
            s,
            MW=100.0,
            Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(30000.0, 0.38, 450.0),
            antoine_coeffs=(10.0, 2800.0, -40.0),
        )
        for s in ("A", "B")
    }
    return IdealThermo(data)


def make_flowsheet(T_spec: float = 350.0) -> Flowsheet:
    """One CSTR, feed -> product. A fresh object every call (units are mutated)."""
    params = CSTRParams(
        V=jnp.array(1.0),
        rate_fn=_rate_fn,
        stoich=jnp.array([[-1.0], [1.0]]),
        rate_params={"A": jnp.array(PRE_EXP), "Ea": jnp.array(E_ACT)},
        species_order=["A", "B"],
    )
    cstr = CSTR(params, thermo=_thermo(), mode="isothermal")
    fs = Flowsheet(species_order=["A", "B"])
    fs.add_feed("feed", make_stream({"A": FEED_A, "B": 0.0}, T=FEED_T, P=P0))
    fs.add_unit(
        Unit(
            "reactor",
            cstr,
            ["feed"],
            ["product"],
            params={"T_spec": T_spec, "volumetric_flow": QV},
        )
    )
    return fs


def analytic_volume(F_B: float, T: float) -> float:
    """``V*`` meeting ``F_B`` at temperature ``T`` -- the closed form above."""
    k = PRE_EXP * np.exp(-E_ACT / (R_GAS * T))
    return QV * F_B / (k * (FEED_A - F_B))


def fixed_T_problem(bound: float = 8.0, T: float = 350.0, **kwargs):
    """Minimize reactor volume subject to ``product.F_B >= bound`` at fixed T."""
    return as_nlp(
        make_flowsheet(T),
        [Decision("unit:reactor.params.V", lb=0.01, ub=5.0, x0=0.5)],
        [("product.F_B", ">=", bound)],
        objective=lambda streams, dvals: dvals["unit:reactor.params.V"],
        var_bounds={"product.T": (T, T)},
        **kwargs,
    )


def free_T_problem(**kwargs):
    """Volume and temperature both free; the optimizer runs T to its ceiling."""
    kwargs.setdefault(
        "objective", lambda streams, dvals: dvals["unit:reactor.params.V"]
    )
    return as_nlp(
        make_flowsheet(),
        [
            Decision("unit:reactor.params.V", lb=0.01, ub=5.0, x0=0.5),
            Decision("unit:reactor.T_spec", lb=320.0, ub=420.0, x0=360.0),
        ],
        [("product.F_B", ">=", 8.0)],
        **kwargs,
    )


def central_diff(fn, x, i, h):
    xp = x.at[i].add(h)
    xm = x.at[i].add(-h)
    return (np.asarray(fn(xp)) - np.asarray(fn(xm))) / (2.0 * h)


# =============================================================================
# The flat NLP view
# =============================================================================


def test_variable_and_constraint_layout():
    """x is [decisions | stream state]; g is [unit residuals | specs]."""
    _f, _g, bd = free_T_problem()
    assert bd.n_decisions == 2
    assert bd.var_names == [
        "unit:reactor.params.V",
        "unit:reactor.T_spec",
        "product.F_A",
        "product.F_B",
        "product.T",
        "product.P",
    ]
    # CSTR.eo_residuals gives n_species material balances plus a T row and a P
    # row. In the EO form none of those is an identity: T_out is a stream
    # variable and T_spec is a decision, so the T row is a real equation
    # (unlike the hand-written sequential form, where T_out IS the decision and
    # the row collapses to 0 = 0 and must be dropped).
    assert bd.con_names[:4] == [f"residual[{i}]" for i in range(4)]
    assert bd.con_names[4] == "product.F_B >= 8"
    assert bd.n == 6 and bd.m == 5
    assert np.all(np.asarray(bd.cl)[:4] == 0.0)
    assert np.all(np.asarray(bd.cu)[:4] == 0.0)
    assert np.asarray(bd.cl)[4] == 8.0


def test_residuals_vanish_at_the_sequential_modular_solution():
    """x0 comes from the SM solve, so the model rows start satisfied."""
    _f, g, bd = free_T_problem()
    r = np.asarray(g(bd.x0))[:4]
    assert np.max(np.abs(r)) < 1e-8


def test_objective_gradient_matches_central_difference():
    f, _g, bd = free_T_problem(
        objective=lambda s, d: -2.0 * s["product"]["F_B"]
        + 1.5 * d["unit:reactor.params.V"] ** 0.6
    )
    ad = np.asarray(jax.grad(f)(bd.x0))
    fd = np.array(
        [float(central_diff(f, bd.x0, i, 1e-6 * max(1.0, abs(float(bd.x0[i])))))
         for i in range(bd.n)]
    )
    assert np.allclose(ad, fd, rtol=1e-5, atol=1e-7)


def test_constraint_jacobian_matches_central_differences():
    _f, g, bd = free_T_problem()
    ad = np.asarray(jax.jacobian(g)(bd.x0))
    fd = np.column_stack(
        [central_diff(g, bd.x0, i, 1e-6 * max(1.0, abs(float(bd.x0[i]))))
         for i in range(bd.n)]
    )
    assert np.allclose(ad, fd, rtol=1e-5, atol=1e-6)


def test_parameters_are_differentiable_but_are_not_columns_of_x():
    f, g, bd = as_nlp(
        make_flowsheet(),
        [Decision("unit:reactor.params.V", 0.01, 5.0, 0.5)],
        [("product.F_B", ">=", 8.0)],
        objective=lambda s, d: d["unit:reactor.params.V"],
        parameters=[Parameter("unit:reactor.T_spec", 350.0)],
    )
    assert bd.n == 5  # 1 decision + 4 stream variables; T_spec is not among them
    assert np.asarray(bd.p0) == np.array([350.0])
    # g really does depend on p.
    dp = np.asarray(jax.jacobian(g, argnums=1)(bd.x0, bd.p0))
    assert np.max(np.abs(dp)) > 0.0


def test_flowsheet_builder_callable_is_accepted():
    """The escape hatch for anything the address grammar cannot reach."""
    from dataclasses import replace as dc_replace

    def build(values):
        fs = make_flowsheet()
        u = fs.units[0]
        fs.units[0] = dc_replace(u, params={**u.params, "T_spec": values["T"]})
        return fs

    f, g, bd = as_nlp(
        build,
        [Decision("T", 320.0, 420.0, 360.0)],
        [("product.F_B", ">=", 8.0)],
        objective=lambda s, d: -s["product"]["F_B"],
    )
    assert bd.n == 5
    assert np.max(np.abs(np.asarray(g(bd.x0))[:4])) < 1e-8
    assert float(f(bd.x0)) < 0.0


def test_spec_shorthands_parse():
    assert Spec.parse(("product.F_B", ">=", 8.0)).lo == 8.0
    assert Spec.parse(("product.F_B", "<=", 8.0)).hi == 8.0
    eq = Spec.parse(("product.T", "==", 350.0))
    assert eq.lo == eq.hi == 350.0
    with pytest.raises(ValueError, match="unknown spec operator"):
        Spec.parse(("product.F_B", "~=", 8.0))
    with pytest.raises(ValueError, match="3-tuple"):
        Spec.parse("product.F_B >= 8")


def test_units_without_an_equation_oriented_form_are_refused_up_front():
    """The EOSolver fallback branch is broken; say so before the JAX trace."""

    class NoEO:
        def __call__(self, inlet):
            return inlet

    fs = Flowsheet(species_order=["A", "B"])
    fs.add_feed("feed", make_stream({"A": 1.0, "B": 0.0}, T=300.0, P=P0))
    fs.add_unit(Unit("mystery", NoEO(), ["feed"], ["out"]))
    with pytest.raises(TypeError, match="mystery"):
        require_eo_residuals(fs)
    with pytest.raises(TypeError, match="eo_residuals"):
        as_nlp(fs, [], [])
    with pytest.raises(TypeError, match="eo_residuals"):
        as_residual(fs)


def test_recycle_with_differing_names_is_refused():
    fs = make_flowsheet()
    fs.add_recycle("product", "feed2")
    with pytest.raises(ValueError, match="underdetermined"):
        as_nlp(fs, [Decision("unit:reactor.params.V", 0.01, 5.0, 0.5)], [])


def test_unknown_address_is_reported_with_the_grammar():
    with pytest.raises(KeyError, match="no unit named"):
        as_nlp(make_flowsheet(), [Decision("nope.V", 0.0, 1.0, 0.5)], [])
    with pytest.raises(ValueError, match="too many dots"):
        as_nlp(make_flowsheet(), [Decision("reactor.a.b.c", 0.0, 1.0, 0.5)], [])


# =============================================================================
# Sparsity: supersets by construction, never probed
# =============================================================================


def test_structural_pattern_is_a_superset_across_the_box():
    """Point checks at many feasible points, not just at x0."""
    _f, g, bd = free_T_problem(validate=True)
    mask = np.zeros((bd.m, bd.n), dtype=bool)
    mask[bd.jac_pattern[0], bd.jac_pattern[1]] = True
    rng = np.random.default_rng(7)
    lo, hi = np.asarray(bd.lb), np.asarray(bd.ub)
    hi = np.minimum(hi, 1e4)  # keep the sample inside physical territory
    for _ in range(12):
        x = jnp.asarray(lo + rng.random(bd.n) * (hi - lo))
        J = np.asarray(jax.jacobian(g)(x))
        assert np.all(np.isfinite(J))
        assert ((np.abs(J) > 0) <= mask).all()


def test_structural_pattern_is_tighter_than_dense_but_still_valid():
    _f, _g, bd = free_T_problem()
    dense_nnz = bd.m * bd.n
    assert len(bd.jac_pattern[0]) < dense_nnz
    # The spec row touches exactly one variable, which is where the saving is.
    rows = np.asarray(bd.jac_pattern[0])
    assert (rows == bd.m - 1).sum() == 1


def test_dense_patterns_are_supersets_by_definition():
    r, c = dense_jacobian_pattern(3, 4)
    assert len(r) == 12 and set(zip(r.tolist(), c.tolist())) == {
        (i, j) for i in range(3) for j in range(4)
    }
    hr, hc = dense_hessian_pattern(4)
    assert len(hr) == 10 and np.all(hr >= hc)


def test_validate_patterns_catches_a_missing_entry():
    """The check pounce does not do. A missing entry is otherwise silent."""
    _f, g, bd = free_T_problem()
    rows, cols = bd.jac_pattern
    with pytest.raises(SparsityPatternError, match="silently wrong"):
        validate_patterns(g, bd.x0, (rows[1:], cols[1:]), bd.m, bd.n)


def test_validate_patterns_accepts_the_derived_pattern():
    f, g, bd = free_T_problem(validate=False)
    validate_patterns(
        g, bd.x0, bd.jac_pattern, bd.m, bd.n, f=f, hess_pattern=bd.hess_pattern
    )


def test_hessian_pattern_covers_the_lagrangian():
    """With RANDOM multipliers. At lambda = 1 the CSTR's two material balances
    cancel their shared reaction term exactly (the stoichiometric column sums to
    zero), so the Lagrangian Hessian is identically zero and the check passes
    for any pattern, including an empty one."""
    f, g, bd = free_T_problem()
    hmask = np.zeros((bd.n, bd.n), dtype=bool)
    hr, hc = np.asarray(bd.hess_pattern[0]), np.asarray(bd.hess_pattern[1])
    hmask[hr, hc] = True
    hmask[hc, hr] = True

    ones = np.asarray(
        jax.hessian(lambda z: f(z) + jnp.dot(jnp.ones(bd.m), g(z)))(bd.x0)
    )
    assert np.max(np.abs(ones)) == 0.0, "the cancellation this test guards against"

    rng = np.random.default_rng(3)
    seen = np.zeros((bd.n, bd.n), dtype=bool)
    for _ in range(6):
        lam = jnp.asarray(rng.standard_normal(bd.m))
        H = np.asarray(jax.hessian(lambda z: f(z) + jnp.dot(lam, g(z)))(bd.x0))
        seen |= np.abs(H) > 0
    assert seen.sum() > 0
    assert (seen <= hmask).all()


def test_validate_patterns_catches_a_missing_hessian_entry():
    """The all-ones cancellation must not let a bad Hessian pattern through."""
    f, g, bd = free_T_problem()
    diag = (np.arange(bd.n), np.arange(bd.n))
    with pytest.raises(SparsityPatternError, match="hess_pattern misses"):
        validate_patterns(
            g, bd.x0, bd.jac_pattern, bd.m, bd.n, f=f, hess_pattern=diag
        )


# =============================================================================
# The sparsity trap -- the reason this module exists
# =============================================================================


def test_the_model_is_undefined_at_the_points_pounce_probes():
    """N(0, 1) means T ~ -1.3 K. Arrhenius overflows; the Jacobian is nan/inf."""
    _f, g, bd = free_T_problem()
    xp = jnp.asarray(np.random.default_rng(0).standard_normal(bd.n))
    J = np.asarray(jax.jacobian(g)(xp))
    assert not np.all(np.isfinite(J))
    # And at x0 it is perfectly finite -- the probe point is the problem, not
    # the model.
    assert np.all(np.isfinite(np.asarray(jax.jacobian(g)(bd.x0))))


@needs_pounce
def test_probing_drops_real_entries_of_the_jacobian():
    """`nan > eps` is False, so a nan derivative is recorded as a structural
    zero. The probe loses the whole reactor-volume column -- the one column the
    optimizer has to move."""
    from pounce.jax._build import _JaxProblem

    _f, g, bd = free_T_problem()
    f, _, _ = free_T_problem()
    probed = _JaxProblem(f=f, g=g, n=bd.n, m=bd.m)
    pr, pc = probed.jacobianstructure()
    probed_mask = np.zeros((bd.m, bd.n), dtype=bool)
    probed_mask[np.asarray(pr), np.asarray(pc)] = True

    true_at_x0 = np.abs(np.asarray(jax.jacobian(g)(bd.x0))) > 0
    missed = np.argwhere(true_at_x0 & ~probed_mask)
    assert missed.size > 0, "probing was expected to miss entries here"
    # Column 0 is the reactor volume.
    assert 0 in set(missed[:, 1].tolist())

    ours = np.zeros((bd.m, bd.n), dtype=bool)
    ours[bd.jac_pattern[0], bd.jac_pattern[1]] = True
    assert (true_at_x0 <= ours).all()


@needs_pounce
def test_probed_pattern_breaks_the_solve_that_the_adapter_completes():
    """Same f, g, bounds and start; only the pattern differs."""
    pj = require("pounce.jax")
    f, g, bd = free_T_problem()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probed = pj.from_jax(
            f, g, n=bd.n, m=bd.m,
            lb=np.asarray(bd.lb), ub=np.asarray(bd.ub),
            cl=np.asarray(bd.cl), cu=np.asarray(bd.cu),
        )
    probed.add_option("print_level", 0)
    probed.add_option("tol", 1e-9)
    _x_bad, info_bad = probed.solve(x0=np.asarray(bd.x0))
    assert info_bad["status_msg"] != "Solve_Succeeded"

    _x, info = solve_with_pounce(f, g, bd, options={"tol": 1e-9})
    assert info["status_msg"] == "Solve_Succeeded"


@needs_pounce
def test_the_adapter_never_reaches_pounce_without_a_pattern(monkeypatch):
    """Structural or dense, but never None -- there is no probing code path."""
    pj = require("pounce.jax")
    seen = {}
    real = pj.from_jax

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(pj, "from_jax", spy)
    f, g, bd = free_T_problem()
    solve_with_pounce(f, g, bd)
    assert seen["jac_pattern"] is not None
    assert seen["hess_pattern"] is not None

    # Even when Bounds carries nothing, the wrapper substitutes dense.
    seen.clear()
    bd.jac_pattern = None
    bd.hess_pattern = None
    solve_with_pounce(f, g, bd)
    assert len(seen["jac_pattern"][0]) == bd.m * bd.n
    assert len(seen["hess_pattern"][0]) == bd.n * (bd.n + 1) // 2


# =============================================================================
# End-to-end solves
# =============================================================================


@needs_pounce
def test_solve_reaches_the_analytic_optimum_at_fixed_temperature():
    f, g, bd = fixed_T_problem(bound=8.0, T=350.0)
    x, info = solve_with_pounce(f, g, bd, options={"tol": 1e-10})
    assert info["status_msg"] == "Solve_Succeeded"
    assert float(x[0]) == pytest.approx(analytic_volume(8.0, 350.0), rel=1e-6)


@needs_pounce
def test_optimum_is_a_converged_flowsheet():
    """A feasible point of the NLP satisfies the model equations by
    construction, and the product flows match the closed form."""
    res = optimize_flowsheet(
        make_flowsheet(350.0),
        [Decision("unit:reactor.params.V", 0.01, 5.0, 0.5)],
        [("product.F_B", ">=", 8.0)],
        objective=lambda s, d: d["unit:reactor.params.V"],
        var_bounds={"product.T": (350.0, 350.0)},
        options={"tol": 1e-10},
    )
    assert res.success
    assert float(res.streams["product"]["F_B"]) == pytest.approx(8.0, abs=1e-6)
    assert float(res.streams["product"]["F_A"]) == pytest.approx(2.0, abs=1e-6)
    g = as_nlp(
        make_flowsheet(350.0),
        [Decision("unit:reactor.params.V", 0.01, 5.0, 0.5)],
        [("product.F_B", ">=", 8.0)],
        objective=lambda s, d: d["unit:reactor.params.V"],
        var_bounds={"product.T": (350.0, 350.0)},
    )[1]
    assert np.max(np.abs(np.asarray(g(jnp.asarray(res.x)))[:4])) < 1e-7


@needs_pounce
def test_sparse_colored_ad_gives_the_same_answer_as_dense():
    f, g, bd = fixed_T_problem()
    x_dense, _ = solve_with_pounce(f, g, bd, sparse=False, options={"tol": 1e-10})
    x_sparse, _ = solve_with_pounce(f, g, bd, sparse=True, options={"tol": 1e-10})
    assert np.allclose(np.asarray(x_dense), np.asarray(x_sparse), rtol=1e-7)


@needs_pounce
def test_mult_g_is_d_objective_d_bound_for_a_ge_spec():
    """The post-optimal sensitivity, against a central difference on the bound."""
    def obj_at(bound):
        f, g, bd = fixed_T_problem(bound=bound)
        _x, info = solve_with_pounce(f, g, bd, options={"tol": 1e-11})
        return float(info["obj_val"]), info, bd

    obj0, info, bd = obj_at(8.0)
    h = 1e-5
    fd = (obj_at(8.0 + h)[0] - obj_at(8.0 - h)[0]) / (2.0 * h)
    sens = bound_sensitivities(info, bd)
    assert sens["product.F_B >= 8"] == pytest.approx(fd, rel=1e-5)
    # And the raw multiplier is its negative -- the sign convention, pinned.
    assert float(np.asarray(info["mult_g"])[-1]) == pytest.approx(-fd, rel=1e-5)
    assert fd > 0.0  # a tighter product spec costs volume


@needs_pounce
def test_mult_g_is_d_objective_d_bound_for_a_le_spec():
    """Same convention in the other direction; no per-constraint sign logic."""
    def obj_at(bound):
        f, g, bd = as_nlp(
            make_flowsheet(350.0),
            [Decision("unit:reactor.params.V", 0.01, 5.0, 0.5)],
            [("product.F_A", "<=", bound)],
            objective=lambda s, d: d["unit:reactor.params.V"],
            var_bounds={"product.T": (350.0, 350.0)},
        )
        _x, info = solve_with_pounce(f, g, bd, options={"tol": 1e-11})
        return float(info["obj_val"]), info, bd

    _o, info, bd = obj_at(2.0)
    h = 1e-5
    fd = (obj_at(2.0 + h)[0] - obj_at(2.0 - h)[0]) / (2.0 * h)
    assert fd < 0.0  # a looser conversion spec saves volume
    assert bound_sensitivities(info, bd)["product.F_A <= 2"] == pytest.approx(
        fd, rel=1e-5
    )
    assert float(np.asarray(info["mult_g"])[-1]) == pytest.approx(-fd, rel=1e-5)


@needs_pounce
def test_inactive_constraint_has_zero_sensitivity():
    f, g, bd = fixed_T_problem(bound=1.0)  # trivially met
    _x, info = solve_with_pounce(f, g, bd, options={"tol": 1e-10})
    sens = bound_sensitivities(info, bd)
    assert abs(sens["product.F_B >= 1"]) > 0.0  # still binding: V is minimized
    # A genuinely slack row: put a loose upper bound on T that the optimum
    # does not touch.
    f2, g2, bd2 = as_nlp(
        make_flowsheet(350.0),
        [Decision("unit:reactor.params.V", 0.01, 5.0, 0.5)],
        [("product.F_B", ">=", 8.0), ("product.T", "<=", 900.0)],
        objective=lambda s, d: d["unit:reactor.params.V"],
        var_bounds={"product.T": (350.0, 350.0)},
    )
    _x2, info2 = solve_with_pounce(f2, g2, bd2, options={"tol": 1e-10})
    assert bound_sensitivities(info2, bd2)["product.T <= 900"] == pytest.approx(
        0.0, abs=1e-8
    )


@needs_pounce
def test_differentiable_problem_backward_matches_a_finite_difference():
    """pounce.jax.solve has no pattern arguments and would probe, so the
    differentiable entry point is JaxProblem. d(optimal volume)/d(T_spec)."""
    f, g, bd = as_nlp(
        make_flowsheet(),
        [Decision("unit:reactor.params.V", 0.01, 5.0, 0.5)],
        [("product.F_B", ">=", 8.0)],
        objective=lambda s, d: d["unit:reactor.params.V"],
        parameters=[Parameter("unit:reactor.T_spec", 350.0)],
    )
    jp = differentiable_problem(f, g, bd)

    def loss(p):
        return f(jp.solve(p, bd.x0), p)

    grad = float(jax.grad(loss)(bd.p0)[0])
    h = 1e-4
    fd = (float(loss(bd.p0 + h)) - float(loss(bd.p0 - h))) / (2.0 * h)
    assert grad == pytest.approx(fd, rel=1e-5)
    # Cross-check against the closed form dV*/dT.
    dV = (analytic_volume(8.0, 350.0 + h) - analytic_volume(8.0, 350.0 - h)) / (2 * h)
    assert grad == pytest.approx(dV, rel=1e-4)


# =============================================================================
# The residual view
# =============================================================================


def test_flowsheet_residual_view_is_square_and_vanishes_at_v0():
    view = as_residual(make_flowsheet())
    assert view.u_names == ["feed.F_A", "feed.F_B", "feed.T", "feed.P"]
    assert view.v_names == [
        "product.F_A", "product.F_B", "product.T", "product.P",
    ]
    assert view.n_unknowns == 4 and view.n_inputs == 4
    r = np.asarray(view(view.u0, view.v0))
    assert r.shape == (4,)
    assert np.max(np.abs(r)) < 1e-8


def test_residual_view_solves_back_to_the_flowsheet_solution():
    """as_residual exposes the EO solver's own residual, so solving it
    reproduces solve_eo. Nothing is re-derived."""
    fs = make_flowsheet()
    view = as_residual(fs)
    z, norm, ok = solve_residual_system(
        lambda z, u: view.fn(u, z), jnp.zeros_like(view.v0) + view.v0, view.u0
    )
    assert bool(ok) and float(norm) < 1e-9
    solved = fs.solve_eo()
    for i, name in enumerate(("F_A", "F_B", "T", "P")):
        assert float(z[i]) == pytest.approx(float(solved["product"][name]), rel=1e-8)
    streams = view.unpack_v(z)
    assert float(streams["product"]["F_B"]) == pytest.approx(float(z[1]))


def test_unit_residual_view_requires_a_square_system():
    fs = make_flowsheet()
    unit = fs.units[0]
    feed = fs.feeds["feed"]
    guess = make_stream({"A": 2.0, "B": 8.0}, T=350.0, P=P0)
    view = as_residual(unit, inlets=[feed], outlets=[guess], species_order=["A", "B"])
    assert view.n_unknowns == 4
    z, _norm, ok = solve_residual_system(
        lambda z, u: view.fn(u, z), view.v0, view.u0
    )
    assert bool(ok)
    assert float(z[0] + z[1]) == pytest.approx(FEED_A, rel=1e-9)


def test_residual_from_system_wraps_the_section_scope_convention():
    """difflow's r(z; args) form -- what solve_residual_system takes and what
    difflow_ree's mass-action section returns."""
    n = 3

    def residual(z, args):
        u = z.reshape(n, 2)
        c = jnp.exp(u)
        aq = args["Q_aq"] * c[:, 0]
        org = args["Q_org"] * c[:, 1] * c[:, 0]
        aq_prev = jnp.concatenate([args["feed"][None], aq[:-1]])
        org_next = jnp.concatenate([org[1:], jnp.zeros(1)])
        return jnp.stack(
            [aq + org - (aq_prev + org_next), c[:, 1] - args["K"] * c[:, 0]],
            axis=1,
        ).reshape(-1)

    args = {
        "Q_aq": jnp.asarray(1.0),
        "Q_org": jnp.asarray(0.5),
        "feed": jnp.asarray(1.0),
        "K": jnp.asarray(2.0),
    }
    z0 = jnp.full(2 * n, -1.0)
    view = residual_from_system(residual, z0, args, u_keys=["feed", "K"])
    # ravel_pytree sorts dict keys, so u is [K, feed] -- NOT the u_keys order.
    assert view.u_names == ["u['K']", "u['feed']"]
    assert np.allclose(np.asarray(view.u0), [2.0, 1.0])

    z, _norm, ok = solve_residual_system(lambda z, u: view.fn(u, z), z0, view.u0)
    assert bool(ok)
    # dv/du by the implicit function theorem must match AD through the solve.
    dv_du = np.asarray(
        jax.jacobian(
            lambda u: solve_residual_system(
                lambda zz, uu: view.fn(uu, zz), z0, u
            )[0]
        )(view.u0)
    )
    gv = np.asarray(jax.jacobian(lambda zz: view.fn(view.u0, zz))(z))
    gu = np.asarray(jax.jacobian(lambda uu: view.fn(uu, z))(view.u0))
    assert np.allclose(dv_du, -np.linalg.solve(gv, gu), rtol=1e-7, atol=1e-9)


def test_residual_from_system_rejects_a_non_square_system():
    with pytest.raises(ValueError, match="square"):
        residual_from_system(lambda z, a: jnp.concatenate([z, z]), jnp.ones(3), None)
    with pytest.raises(KeyError, match="not keys of args"):
        residual_from_system(lambda z, a: z, jnp.ones(2), {"a": 1.0}, u_keys=["b"])
    with pytest.raises(TypeError, match="u_keys"):
        residual_from_system(lambda z, a: z, jnp.ones(2), 1.0, u_keys=["b"])
    with pytest.raises(ValueError, match="z0"):
        as_residual(lambda z, a: z)


def test_residual_view_of_a_ree_mass_action_section():
    """The second residual test case the issue asks for: a whole section that
    already is a residual, never wrapped in a Flowsheet."""
    pytest.importorskip("difflow_ree")
    from difflow_ree.equilibrium.mass_action import make_section_residual
    from difflow_ree.equilibrium.network import build_network

    net = build_network(
        "cation_exchange_dimer", ("Nd", "Dy"),
        log10_K={"Nd": -7.0, "Dy": -5.0},
    )
    n_stages = 3
    residual_fn, _ = make_section_residual(net, n_stages)
    args = {
        "ln_K": net.ln_K(),
        "Q_aq": jnp.asarray(1.0),
        "Q_org": jnp.asarray(1.0),
        "feed_totals": jnp.ones(net.n_components),
        "solvent_totals": jnp.ones(net.n_components),
        "scale": jnp.ones(net.n_components),
    }
    z0 = jnp.full(n_stages * net.n_components, -3.0)
    view = residual_from_system(residual_fn, z0, args, u_keys=["feed_totals"])
    assert view.n_unknowns == n_stages * net.n_components
    assert view.n_inputs == net.n_components
    assert np.asarray(view(view.u0, view.v0)).shape == (view.n_unknowns,)
    # The section Jacobian is nonsingular, which is what makes the view usable
    # as an implicit block at all.
    J = np.asarray(jax.jacobian(lambda z: view.fn(view.u0, z))(z0))
    assert np.linalg.matrix_rank(J) == view.n_unknowns


# =============================================================================
# discopt -- and the restriction that shapes the design
# =============================================================================


@needs_discopt
def test_discopt_implicit_derivative_matches_the_implicit_function_theorem():
    """dm.implicit compiles g(u, v) = 0 to a custom_root node; the derivative
    of the node must be -(dg/dv)^-1 (dg/du) computed straight from the view."""
    import discopt.modeling as dm

    view = as_residual(make_flowsheet())
    model = dm.Model()
    F_A = model.continuous("F_A_feed", lb=1.0, ub=20.0)
    node, returned = as_implicit(
        model, None, [F_A, 0.0, FEED_T, P0], view=view
    )
    assert returned is view

    phi = node.fn  # the JAX callable dm.custom wrapped
    u0 = np.array([FEED_A, 0.0, FEED_T, P0])
    v_star = np.asarray(phi(*u0))
    assert np.max(np.abs(np.asarray(view.fn(jnp.asarray(u0), jnp.asarray(v_star))))) < 1e-9

    dv_dFA = np.asarray(jax.jacobian(lambda a: phi(a, 0.0, FEED_T, P0))(FEED_A))
    gv = np.asarray(jax.jacobian(lambda v: view.fn(jnp.asarray(u0), v))(jnp.asarray(v_star)))
    gu = np.asarray(jax.jacobian(lambda u: view.fn(u, jnp.asarray(v_star)))(jnp.asarray(u0)))
    ift = -np.linalg.solve(gv, gu)[:, 0]
    assert np.allclose(dv_dFA, ift, rtol=1e-8, atol=1e-10)
    assert np.max(np.abs(dv_dFA)) > 0.1  # the derivative is not trivially zero


@needs_discopt
def test_discopt_solve_through_a_difflow_block_finds_the_right_feed():
    """End to end: choose the feed rate that gives F_B = 5, with every
    derivative coming through the difflow residual."""
    import discopt.modeling as dm

    view = as_residual(make_flowsheet())
    model = dm.Model()
    F_A = model.continuous("F_A_feed", lb=1.0, ub=20.0)
    node, _ = as_implicit(model, None, [F_A, 0.0, FEED_T, P0], view=view)
    model.minimize((node[1] - 5.0) ** 2)
    result = model.solve()

    # Local NLP path only: no certificate, by design.
    assert result.status == "feasible"
    assert getattr(result, "bound", None) is None
    assert getattr(result, "gap", None) is None

    def F_B_of(a):
        u = jnp.array([a, 0.0, FEED_T, P0])
        z, _n, _ok = solve_residual_system(
            lambda zz, uu: view.fn(uu, zz), jnp.asarray(view.v0), u
        )
        return float(z[1])

    from scipy.optimize import brentq

    reference = brentq(lambda a: F_B_of(a) - 5.0, 1.0, 20.0)
    assert float(result.value(F_A)) == pytest.approx(reference, rel=1e-6)


@needs_discopt
def test_discopt_refuses_integrality_at_build_time():
    import discopt.modeling as dm

    model = dm.Model()
    model.continuous("x", lb=0.0, ub=1.0)
    check_no_integrality(model)  # clean, no exception

    model.binary("y")
    with pytest.raises(DiscoptIntegralityError, match="integer/binary"):
        check_no_integrality(model)
    with pytest.raises(DiscoptIntegralityError):
        as_implicit(model, make_flowsheet(), [1.0, 0.0, FEED_T, P0])
    assert "RAISES if any integer or binary" in CUSTOMCALL_RESTRICTION


@needs_discopt
def test_discopt_itself_raises_on_integrality():
    """The build-time check mirrors discopt's own solve-time refusal. If this
    test ever fails, discopt's CustomCall contract changed and the docs in
    difflow.solvers.discopt_bridge need rewriting."""
    import discopt.modeling as dm

    view = as_residual(make_flowsheet())
    model = dm.Model()
    F_A = model.continuous("F_A_feed", lb=1.0, ub=20.0)
    node, _ = as_implicit(
        model, None, [F_A, 0.0, FEED_T, P0], view=view
    )
    y = model.binary("y")  # added after the block, so the build check passed
    model.minimize((node[1] - 5.0) ** 2 + y)
    with pytest.raises(ValueError, match="integer/binary"):
        model.solve()


@needs_discopt
def test_as_implicit_checks_the_input_arity():
    import discopt.modeling as dm

    view = as_residual(make_flowsheet())
    model = dm.Model()
    x = model.continuous("x", lb=1.0, ub=20.0)
    with pytest.raises(ValueError, match="flatten to 2 scalars"):
        as_implicit(model, None, [x, 0.0], view=view)


# =============================================================================
# Optional dependencies
# =============================================================================


def test_missing_backend_names_the_pypi_distribution(monkeypatch):
    """pounce imports as `pounce` but installs as `pounce-solver`; the error
    has to say so or the user cannot act on it."""
    import importlib

    from difflow.solvers import _lazy

    def boom(name):
        raise ImportError(f"no module named {name!r}")

    monkeypatch.setattr(_lazy.importlib, "import_module", boom)
    with pytest.raises(ImportError, match="pounce-solver"):
        _lazy.require("pounce.jax")
    with pytest.raises(ImportError, match="discopt"):
        _lazy.require("discopt.modeling")
    assert _lazy.have("pounce.jax") is False
    assert importlib is not None


def test_have_reports_installed_backends():
    from difflow.solvers import _lazy

    assert _lazy.have("json") is True
    assert _lazy.have("difflow_no_such_module") is False
