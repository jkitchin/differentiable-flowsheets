"""Tests for Flowsheet.make_objective_fn and create_objective (issue #62).

Verifies that the objective function actually varies when unit parameters are
changed, and that JAX gradients flow through the parameter updates correctly.
"""

import jax
import jax.numpy as jnp
import pytest

from difflow import (
    CSTR,
    CSTRParams,
    Flowsheet,
    Unit,
    IdealThermo,
    SpeciesData,
    make_stream,
    get_flows,
    create_objective,
)

# Enable 64-bit precision for numerical stability in tests
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_thermo():
    """Two-component thermodynamics (A and B)."""
    species_data = {
        "A": SpeciesData(
            "A",
            MW=100.0,
            Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(35000.0, 0.38, 500.0),
            antoine_coeffs=(10.0, 3000.0, -50.0),
        ),
        "B": SpeciesData(
            "B",
            MW=100.0,
            Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(30000.0, 0.38, 450.0),
            antoine_coeffs=(10.0, 2800.0, -40.0),
        ),
    }
    return IdealThermo(species_data)


@pytest.fixture
def simple_rate_fn():
    """First-order Arrhenius rate function: A -> B."""
    def rate_fn(C, T, params):
        k = params["A"] * jnp.exp(-params["Ea"] / (8.314 * T))
        return jnp.array([k * C["A"]])
    return rate_fn


@pytest.fixture
def cstr_flowsheet(simple_thermo, simple_rate_fn):
    """A minimal single-CSTR flowsheet (no recycles)."""
    stoich = jnp.array([[-1.0], [+1.0]])  # A -> B
    params = CSTRParams(
        V=jnp.array(1.0),
        rate_fn=simple_rate_fn,
        stoich=stoich,
        rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
        species_order=["A", "B"],
    )
    cstr = CSTR(params, thermo=simple_thermo, mode="isothermal")

    fs = Flowsheet(species_order=["A", "B"])
    fs.add_feed("feed", make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0))
    fs.add_unit(Unit(
        name="reactor",
        operation=cstr,
        inlet_names=["feed"],
        outlet_names=["product"],
        params={"T_spec": 350.0},
    ))
    return fs


# ---------------------------------------------------------------------------
# Tests: objective varies with parameters
# ---------------------------------------------------------------------------


class TestMakeObjectiveFnVariesWithParams:
    """The objective must not be constant w.r.t. the optimisation variables."""

    def test_objective_changes_with_volume(self, cstr_flowsheet):
        """Larger reactor volume -> higher B yield -> different objective."""
        def obj_fn(streams):
            return streams["product"]["F_B"]

        objective = cstr_flowsheet.make_objective_fn(obj_fn)

        val_small = float(objective({"reactor.V": jnp.array(0.5)}))
        val_large = float(objective({"reactor.V": jnp.array(5.0)}))

        assert val_small != val_large, (
            "Objective is constant: parameter update had no effect on the solve"
        )
        # Larger volume -> more conversion -> more B
        assert val_large > val_small

    def test_objective_does_not_modify_original_flowsheet(self, cstr_flowsheet):
        """Calling make_objective_fn must not mutate the source flowsheet."""
        original_V = float(cstr_flowsheet.units[0].operation.params.V)

        def obj_fn(streams):
            return streams["product"]["F_B"]

        objective = cstr_flowsheet.make_objective_fn(obj_fn)
        _ = objective({"reactor.V": jnp.array(99.0)})

        assert float(cstr_flowsheet.units[0].operation.params.V) == pytest.approx(
            original_V
        ), "make_objective_fn mutated the original flowsheet"

    def test_create_objective_function_varies(self, cstr_flowsheet):
        """create_objective (module-level helper) must also vary with params."""
        def obj_fn(streams):
            return streams["product"]["F_B"]

        objective = create_objective(cstr_flowsheet, obj_fn)

        val1 = float(objective({"reactor.V": jnp.array(1.0)}))
        val2 = float(objective({"reactor.V": jnp.array(3.0)}))

        assert val1 != val2


# ---------------------------------------------------------------------------
# Tests: gradient flows through parameter updates
# ---------------------------------------------------------------------------


class TestMakeObjectiveFnGradients:
    """JAX gradients should flow through the parameter update."""

    def test_grad_wrt_volume_is_positive(self, cstr_flowsheet):
        """d(F_B)/d(V) should be positive at V=1."""
        def obj_fn(streams):
            return streams["product"]["F_B"]

        objective = cstr_flowsheet.make_objective_fn(obj_fn)

        grad_fn = jax.grad(lambda V: objective({"reactor.V": V}))
        g = grad_fn(jnp.array(1.0))

        assert jnp.isfinite(g), "Gradient is not finite"
        assert float(g) > 0.0, "Expected positive gradient d(F_B)/d(V)"

    def test_grad_is_finite_and_nonzero(self, cstr_flowsheet):
        """Gradient must be finite and non-zero (not a constant function)."""
        def obj_fn(streams):
            return streams["product"]["F_B"]

        objective = cstr_flowsheet.make_objective_fn(obj_fn)

        grad_fn = jax.grad(lambda V: objective({"reactor.V": V}))
        g = grad_fn(jnp.array(2.0))

        assert jnp.isfinite(g), "Gradient is not finite"
        assert float(g) != 0.0, "Gradient is zero: optimising a constant function"


# ---------------------------------------------------------------------------
# Tests: error handling in _apply_params
# ---------------------------------------------------------------------------


class TestApplyParamsErrors:
    def test_missing_unit_raises_key_error(self, cstr_flowsheet):
        """A dotted key with an unknown unit name must raise KeyError."""
        def obj_fn(streams):
            return streams["product"]["F_B"]

        objective = cstr_flowsheet.make_objective_fn(obj_fn)

        with pytest.raises(KeyError, match="nonexistent"):
            objective({"nonexistent.V": jnp.array(1.0)})

    def test_missing_dot_raises_value_error(self, cstr_flowsheet):
        """A key without a dot must raise ValueError."""
        def obj_fn(streams):
            return streams["product"]["F_B"]

        objective = cstr_flowsheet.make_objective_fn(obj_fn)

        with pytest.raises(ValueError, match="dot notation"):
            objective({"reactor_V": jnp.array(1.0)})


# ---------------------------------------------------------------------------
# Tests: feed streams as levers
# ---------------------------------------------------------------------------


class TestFeedStreamParams:
    """``feed:<name>.<field>`` keys make feed rate and composition levers."""

    def test_total_flow_scales_the_feed(self, cstr_flowsheet):
        fs = cstr_flowsheet._apply_params({"feed:feed.total_flow": 20.0})

        assert float(fs.feeds["feed"]["F_A"]) == pytest.approx(20.0)
        # Composition held: B was zero and stays zero.
        assert float(fs.feeds["feed"]["F_B"]) == pytest.approx(0.0)
        # Original untouched.
        assert float(cstr_flowsheet.feeds["feed"]["F_A"]) == pytest.approx(10.0)

    def test_mole_fraction_holds_the_total(self, cstr_flowsheet):
        fs = cstr_flowsheet._apply_params({"feed:feed.x_B": 0.25})
        feed = fs.feeds["feed"]

        assert float(feed["F_A"] + feed["F_B"]) == pytest.approx(10.0)
        assert float(feed["F_B"]) == pytest.approx(2.5)
        assert float(feed["F_A"]) == pytest.approx(7.5)

    def test_conditions_and_species_flows(self, cstr_flowsheet):
        fs = cstr_flowsheet._apply_params({
            "feed:feed.T": 320.0,
            "feed:feed.P": 2e5,
            "feed:feed.F_B": 1.0,
        })
        feed = fs.feeds["feed"]

        assert float(feed["T"]) == pytest.approx(320.0)
        assert float(feed["P"]) == pytest.approx(2e5)
        assert float(feed["F_B"]) == pytest.approx(1.0)

    def test_feed_and_unit_keys_together(self, cstr_flowsheet):
        fs = cstr_flowsheet._apply_params({
            "reactor.V": jnp.array(3.0),
            "feed:feed.total_flow": 5.0,
        })

        assert float(fs.units[0].operation.params.V) == pytest.approx(3.0)
        assert float(fs.feeds["feed"]["F_A"]) == pytest.approx(5.0)

    def test_unknown_feed_name_raises(self, cstr_flowsheet):
        with pytest.raises(KeyError, match="No feed stream named"):
            cstr_flowsheet._apply_params({"feed:nope.total_flow": 1.0})

    def test_unknown_feed_field_raises(self, cstr_flowsheet):
        with pytest.raises(KeyError, match="Unknown feed field"):
            cstr_flowsheet._apply_params({"feed:feed.enthalpy": 1.0})

    def test_missing_dot_raises(self, cstr_flowsheet):
        with pytest.raises(ValueError, match="feed:<stream_name>"):
            cstr_flowsheet._apply_params({"feed:feed": 1.0})

    def test_objective_varies_with_feed_rate(self, cstr_flowsheet):
        objective = cstr_flowsheet.make_objective_fn(
            lambda streams: streams["product"]["F_B"]
        )

        small = float(objective({"feed:feed.total_flow": jnp.array(5.0)}))
        large = float(objective({"feed:feed.total_flow": jnp.array(20.0)}))

        assert large > small

    def test_gradient_through_a_feed_lever(self, cstr_flowsheet):
        """The whole point: d(product)/d(feed rate) must be finite and right."""
        objective = cstr_flowsheet.make_objective_fn(
            lambda streams: streams["product"]["F_B"]
        )

        def scalar(F):
            return objective({"feed:feed.total_flow": F})

        F0 = jnp.array(10.0)
        g = float(jax.grad(scalar)(F0))
        h = 1e-4
        fd = (float(scalar(F0 + h)) - float(scalar(F0 - h))) / (2 * h)

        assert jnp.isfinite(g)
        assert g == pytest.approx(fd, rel=1e-5)

    def test_gradient_through_a_composition_lever(self, cstr_flowsheet):
        objective = cstr_flowsheet.make_objective_fn(
            lambda streams: streams["product"]["F_B"]
        )

        def scalar(x):
            return objective({"feed:feed.x_B": x})

        x0 = jnp.array(0.2)
        g = float(jax.grad(scalar)(x0))
        h = 1e-5
        fd = (float(scalar(x0 + h)) - float(scalar(x0 - h))) / (2 * h)

        assert jnp.isfinite(g)
        assert g == pytest.approx(fd, rel=1e-4)


# ---------------------------------------------------------------------------
# Unit.params levers and solve keywords (unit-ops book, Ch 24)
# ---------------------------------------------------------------------------


def _purge_loop():
    """Ch 1's methanol loop with a per-pass conversion that falls with CH4.

    The purge fraction lives in the Splitter's ``Unit.params`` dict, not on a
    ParamsMixin, and the loop gain near ``1 - p`` makes a small purge need
    many plain-substitution iterations under ``jax.grad``.
    """
    from difflow import (ComponentSplitter, ComponentSplitterParams, Mixer,
                         Splitter, StoichiometricReactor,
                         StoichiometricReactorParams, total_flow)

    sp = ["CO", "H2", "MeOH", "CH4"]
    nu = {"CO": -1.0, "H2": -2.0, "MeOH": 1.0}

    def reactor(inlet):
        X = 0.30 * (1 - inlet["F_CH4"] / total_flow(inlet))
        return StoichiometricReactor(
            StoichiometricReactorParams(stoich=nu, key="CO", conversion=X))(inlet)

    fs = Flowsheet(species_order=sp, default_P=None)
    fs.add_feed("fresh", make_stream({"CO": 100.0, "H2": 200.0, "MeOH": 0.0, "CH4": 3.0},
                                     T=500.0, P=50e5))
    fs.add_unit(Unit("mixer", Mixer(sp), ["fresh", "recycle"], ["rx_in"]))
    fs.add_unit(Unit("reactor", reactor, ["rx_in"], ["rx_out"]))
    fs.add_unit(Unit("sep", ComponentSplitter(ComponentSplitterParams({"MeOH": 0.0},
                                                                      default_fraction=1.0)),
                     ["rx_out"], ["gas", "crude"]))
    fs.add_unit(Unit("purge", Splitter(sp), ["gas"], ["purge", "recycle"],
                     {"split_frac": 0.10}))
    fs.add_recycle("recycle", "recycle")
    return fs


class TestUnitParamsLevers:
    """``"<unit>.<key>"`` reaches a key of ``Unit.params`` (e.g. split_frac)."""

    def test_apply_params_updates_unit_params_dict(self):
        fs = _purge_loop()
        new = fs._apply_params({"purge.split_frac": 0.05})
        purge = [u for u in new.units if u.name == "purge"][0]
        assert purge.params["split_frac"] == 0.05
        # The source flowsheet is untouched.
        assert [u for u in fs.units if u.name == "purge"][0].params["split_frac"] == 0.10

    def test_unit_params_and_operation_params_together(self, cstr_flowsheet):
        """``T_spec`` is a call keyword, ``V`` a CSTRParams field: both apply."""
        new = cstr_flowsheet._apply_params({"reactor.T_spec": 360.0,
                                            "reactor.V": jnp.array(2.0)})
        assert new.units[0].params["T_spec"] == 360.0
        assert float(new.units[0].operation.params.V) == pytest.approx(2.0)

    def test_unknown_field_on_paramless_operation_raises(self):
        fs = _purge_loop()
        with pytest.raises(AttributeError, match="not keys of its Unit.params"):
            fs._apply_params({"purge.nonsense": 1.0})

    def test_objective_varies_with_split_frac(self):
        fs = _purge_loop()
        obj = fs.make_objective_fn(lambda s: s["crude"]["F_MeOH"])
        assert float(obj({"purge.split_frac": 0.20})) < float(obj({"purge.split_frac": 0.10}))


class TestSolveKeywordsForwarded:
    """``make_objective_fn``/``create_objective`` pass keywords to ``solve``."""

    def test_max_iter_reaches_solve(self):
        fs = _purge_loop()
        seen = {}
        orig = Flowsheet.solve

        def spy(self, *args, **kwargs):
            seen.update(kwargs)
            return orig(self, *args, **kwargs)

        Flowsheet.solve = spy
        try:
            create_objective(fs, lambda s: s["crude"]["F_MeOH"],
                             max_iter=777, tol_basis="error")({"purge.split_frac": 0.1})
        finally:
            Flowsheet.solve = orig
        assert seen["max_iter"] == 777 and seen["tol_basis"] == "error"

    def test_gradient_of_split_frac_matches_finite_difference(self):
        """At p = 0.03 (gain ~0.97) the traced solve needs max_iter well above 100."""
        fs = _purge_loop()
        obj = create_objective(fs, lambda s: s["crude"]["F_MeOH"], max_iter=3000)
        f = lambda p: obj({"purge.split_frac": p})
        p0 = 0.03
        g = float(jax.grad(f)(p0))
        h = 1e-5
        fd = (float(f(p0 + h)) - float(f(p0 - h))) / (2 * h)
        assert g == pytest.approx(fd, rel=1e-4)


def _react(inlet, X=0.5):
    out = dict(inlet)
    out["F_A"], out["F_B"] = inlet["F_A"] * (1 - X), inlet["F_B"] + X * inlet["F_A"]
    return out

def _mix(feed, rec):
    out = {k: feed[k] + rec[k] for k in ("F_A", "F_B")}
    out["T"], out["P"] = feed["T"], feed["P"]
    return out


class TestGradAfterJittedGrad:
    """``Unit`` compares by identity, so a traced ``Unit.params`` is never ``==``-tested.

    A value ``__eq__`` over ``params`` (which hold tracers inside ``jit``) made
    equinox's cache raise on a later trace: after ``jit(grad(f))``,
    ``grad(jit(f))`` and ``grad(f)`` failed with "Exception raised while
    checking equality of metadata fields ... _ClosureConvert".
    """

    def test_units_compare_by_identity(self):
        fs = _purge_loop()
        a = fs.units[0]
        b = Unit(a.name, a.operation, list(a.inlet_names), list(a.outlet_names),
                 dict(a.params))
        assert a == a and a != b
        assert hash(a) == hash(b)

    def test_grad_after_a_jitted_grad(self):
        """A traced value in the ``Unit.params`` of a plain-function unit.

        The functions live at module scope, so both traces build Units with
        equal hashes and the cache falls through to ``__eq__``.
        """
        from difflow import Splitter

        sp = ["A", "B"]

        def f(X):
            fs = Flowsheet(species_order=sp, default_P=None)
            fs.add_feed("feed", make_stream({"A": 1.0, "B": 0.0}, T=300.0, P=1e5))
            fs.add_unit(Unit("mix", _mix, ["feed", "rec"], ["rx_in"]))
            fs.add_unit(Unit("rx", _react, ["rx_in"], ["rx_out"], {"X": X}))
            fs.add_unit(Unit("split", Splitter(sp), ["rx_out"], ["rec", "prod"],
                             {"split_frac": 0.5}))
            fs.add_recycle("rec", "rec")
            return fs.solve(max_iter=200)["prod"]["F_B"]

        X0 = 0.4
        g0 = float(jax.jit(jax.grad(f))(X0))    # one traced solve first ...
        g1 = float(jax.grad(jax.jit(f))(X0))    # ... then these used to raise
        g2 = float(jax.grad(f)(X0))
        h = 1e-5
        fd = (float(f(X0 + h)) - float(f(X0 - h))) / (2 * h)
        assert g0 == pytest.approx(fd, rel=1e-4)
        assert g1 == pytest.approx(g0, rel=1e-10) and g2 == pytest.approx(g0, rel=1e-10)
