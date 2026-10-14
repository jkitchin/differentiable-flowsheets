"""Smoke and contract tests for the pounce / discopt bridges (:mod:`difflow.solvers`)."""

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
from difflow.solvers import (  # noqa: E402
    Decision,
    Spec,
    SparsityDetectionError,
    SparsityPatternError,
    as_implicit,
    as_nlp,
    as_residual,
    check_no_integrality,
    dense_hessian_pattern,
    dense_jacobian_pattern,
    detect_patterns,
    DiscoptIntegralityError,
    pattern_density,
    require_eo_residuals,
    residual_from_system,
    solve_with_pounce,
    validate_patterns,
)
from difflow.solvers._lazy import have  # noqa: E402

HAVE_POUNCE = have("pounce.jax")
HAVE_DISCOPT = have("discopt.modeling")
HAVE_ASDEX = have("asdex")

needs_pounce = pytest.mark.skipif(not HAVE_POUNCE, reason="pounce not installed")
needs_discopt = pytest.mark.skipif(not HAVE_DISCOPT, reason="discopt not installed")
needs_asdex = pytest.mark.skipif(not HAVE_ASDEX, reason="asdex not installed")

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


@needs_asdex
def test_as_nlp_smoke_and_grad_is_finite():
    f, g, bd = free_T_problem(validate=False)
    assert np.isfinite(float(f(bd.x0)))
    assert np.all(np.isfinite(np.asarray(g(bd.x0))))
    assert np.all(np.isfinite(np.asarray(jax.grad(f)(bd.x0))))
    assert bd.jac_pattern is not None and bd.hess_pattern is not None


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

def test_dense_patterns_and_density():
    r, c = dense_jacobian_pattern(3, 4)
    assert len(r) == 12
    hr, hc = dense_hessian_pattern(4)
    assert len(hr) == 10 and np.all(hr >= hc)
    assert pattern_density((np.arange(3), np.arange(3)), 3, 3) == pytest.approx(1 / 3)


@needs_asdex
def test_detect_patterns_reads_a_hand_built_model_exactly():
    """Not every caller has a flowsheet. On a model whose structure is
    obvious, the derived pattern is the true one -- no slack at all."""

    def f(x):
        return x[0] ** 2 * x[1]

    def g(x):
        return jnp.array([x[0] * x[2], x[1] + x[3]])

    jac, hess = detect_patterns(f, g, jnp.ones(4), 2)
    assert set(zip(jac[0].tolist(), jac[1].tolist())) == {(0, 0), (0, 2), (1, 1), (1, 3)}
    # d2/dx0dx0, d2/dx0dx1 from the objective; d2/dx0dx2 from row 0 of g.
    # x3 is linear everywhere, so it has no Hessian entry at all.
    assert set(zip(hess[0].tolist(), hess[1].tolist())) == {(0, 0), (1, 0), (2, 0)}

def test_missing_asdex_raises_instead_of_silently_going_dense(monkeypatch):
    """An install problem has an obvious fix. Substituting a dense pattern
    would hide it behind a solve that is merely slow -- or, on a real plant,
    one that never finishes."""
    import difflow.solvers.sparsity as sparsity_mod

    def no_asdex(module):
        raise ImportError(f"no {module}: `pip install difflow[solvers]`")

    monkeypatch.setattr(sparsity_mod, "require", no_asdex)
    with pytest.raises(ImportError, match=r"difflow\[solvers\]"):
        free_T_problem()

    # The explicit modes still work without it -- neither uses the analysis.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _f, _g, bd = free_T_problem(sparsity="structural")
    assert bd.sparsity_source == "structural"

def test_detection_failure_falls_back_loudly_and_global_refuses_to(monkeypatch):
    """asdex cannot see through an inner solve. 'auto' may fall back to the
    connectivity pattern -- but it says so; 'global' asks for the analysis and
    gets an error rather than a coarser answer."""
    import difflow.solvers.nlp as nlp_mod

    def boom(*args, **kwargs):
        raise SparsityDetectionError("jaxpr contains linear_solve")

    monkeypatch.setattr(nlp_mod, "detect_jacobian_pattern", boom)

    with pytest.warns(RuntimeWarning, match="falling back"):
        _f, _g, bd = free_T_problem(sparsity="auto")
    assert bd.sparsity_source == "structural"

    with pytest.raises(SparsityDetectionError, match="linear_solve"):
        free_T_problem(sparsity="global")

def test_unknown_sparsity_mode_is_refused_with_the_choices():
    with pytest.raises(ValueError, match="auto"):
        free_T_problem(sparsity="sparse")

@needs_asdex
def test_validate_patterns_catches_a_missing_entry():
    """The check pounce does not do. A missing entry is otherwise silent."""
    _f, g, bd = free_T_problem()
    rows, cols = bd.jac_pattern
    with pytest.raises(SparsityPatternError, match="silently wrong"):
        validate_patterns(g, bd.x0, (rows[1:], cols[1:]), bd.m, bd.n)

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

def test_residual_from_system_rejects_a_non_square_system():
    with pytest.raises(ValueError, match="square"):
        residual_from_system(lambda z, a: jnp.concatenate([z, z]), jnp.ones(3), None)
    with pytest.raises(KeyError, match="not keys of args"):
        residual_from_system(lambda z, a: z, jnp.ones(2), {"a": 1.0}, u_keys=["b"])
    with pytest.raises(TypeError, match="u_keys"):
        residual_from_system(lambda z, a: z, jnp.ones(2), 1.0, u_keys=["b"])
    with pytest.raises(ValueError, match="z0"):
        as_residual(lambda z, a: z)

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

@needs_discopt
def test_as_implicit_checks_the_input_arity():
    import discopt.modeling as dm

    view = as_residual(make_flowsheet())
    model = dm.Model()
    x = model.continuous("x", lb=1.0, ub=20.0)
    with pytest.raises(ValueError, match="flatten to 2 scalars"):
        as_implicit(model, None, [x, 0.0], view=view)

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

def test_require_message_drops_the_differ_clause_when_names_match(monkeypatch):
    import difflow.solvers._lazy as lazy

    monkeypatch.setattr(lazy.importlib, "import_module",
                        lambda m: (_ for _ in ()).throw(ImportError(m)))
    with pytest.raises(ImportError) as same:
        lazy.require("asdex")
    assert "not 'asdex'" not in str(same.value)
    assert "'asdex'" in str(same.value)
    with pytest.raises(ImportError) as differ:
        lazy.require("pounce.jax")
    assert "'pounce-solver', not 'pounce'" in str(differ.value)

@needs_asdex
def test_unpack_reports_feed_decisions_at_their_optimized_values():
    """#207 review: `Bounds.feeds` is frozen at the reference flowsheet.

    A decision addressed `feed:<stream>.<key>` is written into the rebuilt
    flowsheet, so the objective, constraints and residuals all see the moved
    value -- but `unpack` merged the *stale* reference feeds into the streams
    dict it returns. `FlowsheetOptimum.streams` therefore reported a converged
    solution whose feed conditions were not the ones that produced it, and
    reading a feed decision back off the streams gave its starting value with
    no indication anything was wrong.
    """
    feed_T_start, feed_T_moved = 350.0, 377.0
    f, g, bd = as_nlp(
        make_flowsheet(feed_T_start),
        [
            Decision("unit:reactor.params.V", lb=0.01, ub=5.0, x0=0.5),
            Decision("feed:feed.T", lb=300.0, ub=420.0, x0=feed_T_start),
        ],
        [("product.F_B", ">=", 8.0)],
        objective=lambda streams, dvals: dvals["unit:reactor.params.V"],
    )

    # Move the feed-temperature decision away from its start.
    i = bd.var_names.index("feed:feed.T")
    x = np.array(bd.x0, dtype=float)
    x[i] = feed_T_moved

    dvals, streams = bd.unpack(jnp.asarray(x))
    assert float(dvals["feed:feed.T"]) == pytest.approx(feed_T_moved)
    assert float(streams["feed"]["T"]) == pytest.approx(feed_T_moved), (
        "unpack reported the feed at its reference value, not the optimized one"
    )
    # The un-moved decision is untouched, and non-feed streams still come from
    # the layout rather than being overwritten.
    assert float(dvals["unit:reactor.params.V"]) == pytest.approx(bd.x0[0])

def test_solve_with_pounce_is_a_wrapper_over_pounce_problem(monkeypatch):
    """One Problem per pounce_problem call; solve_with_pounce solves it once."""
    import difflow.solvers.pounce_bridge as pb

    built, solved, opts = [], [], {}

    class FakeProblem:
        def add_option(self, k, v):
            opts[k] = v

        def solve(self, x0):
            solved.append(x0)
            return x0, {}

    class FakePJ:
        @staticmethod
        def from_jax(*a, **kw):
            built.append(kw)
            return FakeProblem()

    monkeypatch.setattr(pb, "require", lambda name: FakePJ)
    from types import SimpleNamespace

    bd = SimpleNamespace(
        n=2, m=1, lb=np.zeros(2), ub=np.ones(2), cl=np.zeros(1), cu=np.ones(1),
        x0=np.full(2, 0.5), jac_pattern=([0, 0], [0, 1]),
        hess_pattern=([0, 1], [0, 1]),
    )
    f = g = None

    problem = pb.pounce_problem(f, g, bd, options={"tol": 1e-9})
    problem.solve(x0=bd.x0)
    problem.solve(x0=bd.x0)
    assert len(built) == 1 and len(solved) == 2
    assert built[0]["jac_pattern"] is not None and built[0]["hess_pattern"] is not None
    assert opts == {"print_level": 0, "tol": 1e-9}

    pb.solve_with_pounce(f, g, bd)
    assert len(built) == 2 and len(solved) == 3

@needs_pounce
@needs_asdex
def test_solve_with_pounce_refuses_a_missing_pattern():
    f, g, bd = free_T_problem()
    bd.jac_pattern = None
    bd.hess_pattern = None
    with pytest.raises(SparsityDetectionError, match="probing"):
        solve_with_pounce(f, g, bd)
