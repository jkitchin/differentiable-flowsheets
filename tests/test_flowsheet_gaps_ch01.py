"""Flowsheet gaps found writing Chapter 1 of the unit-operations book.

1. a recycle tear no longer starts (and stays) at 1 atm in a pressurized loop;
2. an unconverged recycle under ``jax.grad`` warns instead of returning a
   silently wrong gradient;
3. ``Mixer`` with thermo solves the enthalpy balance exactly;
4. ``Mixer`` treats a species an inlet lacks as zero flow;
5. ``StoichiometricReactor`` and ``ComponentSplitter``;
6. ``Flowsheet.degrees_of_freedom``;
7. ``last_solve_gain`` after an Anderson solve.
"""

import warnings

import jax
import jax.numpy as jnp
import pytest
from jax.test_util import check_grads

from difflow import (
    ComponentSplitter,
    ComponentSplitterParams,
    ConvergenceError,
    ConvergenceWarning,
    DegreesOfFreedom,
    Mixer,
    Splitter,
    StoichiometricReactor,
    StoichiometricReactorParams,
    make_stream,
)
from difflow.flowsheet import Flowsheet, Unit
from difflow.streams import get_flows, total_flow

jax.config.update("jax_enable_x64", True)

SP = ["CO", "H2", "MeOH", "CH4"]
NU = {"CO": -1.0, "H2": -2.0, "MeOH": 1.0}
P_LOOP = 50e5


def _fresh(P=P_LOOP, T=500.0):
    return make_stream({"CO": 100.0, "H2": 200.0, "MeOH": 0.0, "CH4": 3.0}, T=T, P=P)


def _diluted_reactor(inlet):
    """Conversion falls as inert CH4 builds up: a slow, high-gain loop."""
    F = get_flows(inlet)
    X = 0.30 * (1 - F["CH4"] / total_flow(inlet))
    xi = X * F["CO"]
    return make_stream({s: F[s] + NU.get(s, 0.0) * xi for s in SP}, inlet["T"], inlet["P"])


def methanol_loop(p=0.1, reactor=None, default_P=None, default_T=None, fresh=None):
    """Fresh feed + recycle -> mixer -> 25 % reactor -> MeOH knockout -> purge."""
    reactor = reactor or StoichiometricReactor(StoichiometricReactorParams(
        stoich=NU, key="CO", conversion=0.25))
    fs = Flowsheet(SP, default_P=default_P, default_T=default_T)
    fs.add_feed("fresh", _fresh() if fresh is None else fresh)
    fs.add_unit(Unit("mixer", Mixer(SP), ["fresh", "recycle"], ["rx_in"]))
    fs.add_unit(Unit("reactor", reactor, ["rx_in"], ["rx_out"]))
    fs.add_unit(Unit("sep", ComponentSplitter(ComponentSplitterParams(
        {"MeOH": 0.0}, default_fraction=1.0)), ["rx_out"], ["gas", "crude"]))
    fs.add_unit(Unit("purge", Splitter(SP), ["gas"], ["purge", "recycle"],
                     {"split_frac": p}))
    fs.add_recycle("recycle", "recycle")
    return fs


def _exact_meoh(p, X=0.25):
    return 100.0 * X / (1 - (1 - p) * (1 - X))


# --- 1. tear pressure -----------------------------------------------------

class TestTearPressureInference:

    def test_defaults_are_inferred(self):
        fs = Flowsheet(SP)
        assert fs.infer_tear_T and fs.infer_tear_P
        assert fs.default_P == 101325.0 and fs.default_T == 300.0

    @pytest.mark.slow
    def test_pressurized_loop_stays_at_loop_pressure(self):
        fs = methanol_loop()
        st = fs.solve()
        for name in ("rx_in", "rx_out", "recycle", "purge"):
            assert float(st[name]["P"]) == pytest.approx(P_LOOP)
        assert float(st["crude"]["F_MeOH"]) == pytest.approx(_exact_meoh(0.1), rel=1e-8)

    def test_pressure_derivative_is_one(self):
        def P_out(P):
            fs = methanol_loop(fresh=_fresh(P=P))
            return fs.solve(tol=1e-12, max_iter=500)["purge"]["P"]

        assert float(jax.grad(P_out)(P_LOOP)) == pytest.approx(1.0, abs=1e-8)

    @pytest.mark.slow
    def test_temperature_is_inferred_too(self):
        st = methanol_loop(fresh=_fresh(T=480.0)).solve()
        assert float(st["recycle"]["T"]) == pytest.approx(480.0)

    def test_explicit_default_pins_the_old_way(self):
        # A pinned 1 atm start is what the Mixer's min-pressure rule then
        # locks the loop at: the reason inference is the default.
        fs = methanol_loop(default_P=101325.0)
        assert not fs.infer_tear_P
        st = fs.solve()
        assert float(st["rx_in"]["P"]) == pytest.approx(101325.0)

    def test_traced_path_infers_from_an_upstream_unit(self):
        # feed at 1 bar -> compressor to P_c -> mixer: the tear guess must
        # come from the compressor outlet, which is upstream of every tear.
        def compressor(inlet, P_c):
            return make_stream(get_flows(inlet), inlet["T"], P_c)

        def P_mix(P_c):
            fs = Flowsheet(SP)
            fs.add_feed("raw", _fresh(P=1e5))
            fs.add_unit(Unit("comp", lambda s: compressor(s, P_c), ["raw"], ["fresh"]))
            fs.add_unit(Unit("mixer", Mixer(SP), ["fresh", "recycle"], ["rx_in"]))
            fs.add_unit(Unit("reactor", StoichiometricReactor(StoichiometricReactorParams(
                stoich=NU, key="CO", conversion=0.25)), ["rx_in"], ["rx_out"]))
            fs.add_unit(Unit("purge", Splitter(SP), ["rx_out"], ["purge", "recycle"],
                             {"split_frac": 0.5}))
            fs.add_recycle("recycle", "recycle")
            return fs.solve(tol=1e-12, max_iter=500)["rx_in"]["P"]

        assert float(P_mix(P_LOOP)) == pytest.approx(P_LOOP)
        assert float(jax.grad(P_mix)(P_LOOP)) == pytest.approx(1.0, abs=1e-8)

    def test_inference_survives_serialization_and_copy(self):
        from difflow.serialize import from_dict, to_dict

        fs = methanol_loop()
        d = to_dict(fs)
        assert "P" not in d["defaults"] and "T" not in d["defaults"]
        fs2 = from_dict(d)
        assert fs2.infer_tear_P and fs2.infer_tear_T
        assert float(fs2.solve()["rx_in"]["P"]) == pytest.approx(P_LOOP)

        pinned = to_dict(Flowsheet(SP, default_P=2e5))
        assert pinned["defaults"]["P"] == 2e5
        assert not from_dict(pinned).infer_tear_P

        assert fs._apply_params({}).infer_tear_P


# --- 2. traced non-convergence --------------------------------------------

def _crude_meoh(p, max_iter, **kw):
    fs = methanol_loop(p, reactor=_diluted_reactor)
    return fs.solve(max_iter=max_iter, **kw)["crude"]["F_MeOH"]


class TestTracedNonconvergence:

    def test_unconverged_grad_warns(self):
        with pytest.warns(ConvergenceWarning, match="GRADIENTS"):
            jax.grad(_crude_meoh)(0.02, 20)

    def test_raise_stays_a_warning_under_trace(self):
        # an exception inside a gradient would make the flowsheet
        # undifferentiable (TestUnderTracing records that invariant)
        with pytest.warns(ConvergenceWarning):
            g = jax.grad(lambda p: _crude_meoh(p, 20, on_nonconvergence="raise"))(0.02)
        assert jnp.isfinite(g)

    def test_warning_filter_escalates(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            with pytest.raises(Exception, match="did not converge"):
                jax.grad(_crude_meoh)(0.02, 20)

    def test_concrete_solve_still_raises(self):
        with pytest.raises(ConvergenceError):
            _crude_meoh(0.02, 5, acceleration="none", on_nonconvergence="raise")

    def test_ignore_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            jax.grad(lambda p: _crude_meoh(p, 20, on_nonconvergence="ignore"))(0.02)

    def test_converged_grad_is_silent_and_exact(self):
        fs_kw = dict(tol=1e-12, max_iter=500)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            g = jax.grad(lambda p: methanol_loop(p).solve(**fs_kw)["crude"]["F_MeOH"])(0.1)
        exact = jax.grad(_exact_meoh)(0.1)
        assert float(g) == pytest.approx(float(exact), rel=1e-7)

    def test_warns_under_jit(self):
        with pytest.warns(ConvergenceWarning):
            jax.block_until_ready(jax.jit(jax.grad(lambda p: _crude_meoh(p, 20)))(0.02))


# --- 7. gain after an accelerated solve -----------------------------------

class TestGainAfterAcceleration:

    @pytest.mark.parametrize("method", ["anderson", "wegstein", "none"])
    def test_gain_is_populated(self, method):
        fs = methanol_loop(0.1)
        fs.solve(acceleration=method, tol=1e-10, max_iter=2000)
        assert fs.last_solve_converged
        # the inert CH4 is the slowest mode: it returns with fraction 1 - p
        assert fs.last_solve_gain == pytest.approx(0.9, abs=2e-3)
        assert fs.last_solve_error_estimate is not None

    def test_the_jacobian_probe_leaves_no_tracer_behind(self):
        """A unit that keeps its last result must hold numbers after the solve."""
        class Remembering:
            def __init__(self):
                self.last = None

            def __call__(self, inlet):
                self.last = inlet["F_CH4"]
                return make_stream(get_flows(inlet), inlet["T"], inlet["P"])

        fs = methanol_loop(0.1)
        keeper = Remembering()
        fs.add_unit(Unit("keeper", keeper, ["crude"], ["product"]))
        fs.solve(acceleration="anderson", tol=1e-10)
        assert fs.last_solve_gain is not None
        assert not isinstance(keeper.last, jax.core.Tracer)
        float(keeper.last)

    def test_recycle_free_solve_has_no_gain(self):
        fs = Flowsheet(SP)
        fs.add_feed("fresh", _fresh())
        fs.add_unit(Unit("mixer", Mixer(SP), ["fresh"], ["out"]))
        fs.solve()
        assert fs.last_solve_gain is None


# --- 3, 4. Mixer ------------------------------------------------------------

class TestMixer:

    @pytest.fixture(scope="class")
    def thermo(self):
        from difflow.database import get_species_data
        from difflow.thermo import IdealThermo
        return IdealThermo({s: get_species_data(s) for s in ["methane", "ethane"]})

    def _mix(self, thermo, T1, T2=800.0):
        sp = ["methane", "ethane"]
        a = make_stream({"methane": 1.0, "ethane": 0.0}, T=T1, P=2e5)
        b = make_stream({"methane": 10.0, "ethane": 0.0}, T=T2, P=1e5)
        return Mixer(sp, thermo=thermo, phase="vapor")(a, b)[0]

    def test_enthalpy_balance_is_exact(self, thermo):
        out = self._mix(thermo, 300.0)
        T = float(out["T"])
        assert T == pytest.approx(763.64, abs=0.01)   # not 752.7 (one cp at the mean T)
        H = lambda F, T: thermo.stream_enthalpy({"methane": F, "ethane": 0.0}, T, phase="vapor")
        H_in = H(1.0, 300.0) + H(10.0, 800.0)
        assert float(H(11.0, T)) == pytest.approx(float(H_in), rel=1e-10)
        assert float(out["P"]) == 1e5

    @pytest.mark.slow
    def test_gradient(self, thermo):
        check_grads(lambda T1: self._mix(thermo, T1)["T"], (300.0,), order=1,
                    modes=["rev", "fwd"])
        assert jnp.isfinite(jax.jit(lambda T1: self._mix(thermo, T1)["T"])(300.0))

    def test_missing_species_is_zero(self):
        mixer = Mixer(["A", "B"])
        a = make_stream({"A": 1.0}, T=300.0, P=1e5)
        b = make_stream({"A": 1.0, "B": 2.0}, T=307.5, P=1e5)
        out, _ = mixer(a, b)
        assert float(out["F_A"]) == 2.0 and float(out["F_B"]) == 2.0
        assert float(out["T"]) == pytest.approx((300.0 + 3 * 307.5) / 4)

    def test_missing_species_with_thermo(self, thermo):
        mixer = Mixer(["methane", "ethane"], thermo=thermo, phase="vapor")
        a = make_stream({"methane": 1.0}, T=300.0, P=1e5)
        b = make_stream({"methane": 1.0, "ethane": 1.0}, T=300.0, P=1e5)
        out, _ = mixer(a, b)
        assert float(out["F_ethane"]) == 1.0
        assert float(out["T"]) == pytest.approx(300.0, abs=1e-6)


# --- 5. new units -----------------------------------------------------------

class TestStoichiometricReactor:

    feed = make_stream({"CO": 100.0, "H2": 200.0}, T=500.0, P=5e6)

    def test_single_reaction(self):
        rx = StoichiometricReactor(StoichiometricReactorParams(
            stoich=NU, key="CO", conversion=0.25, pressure_drop=1e5, T_out=520.0))
        out, info = rx(self.feed)
        assert float(out["F_CO"]) == pytest.approx(75.0)
        assert float(out["F_H2"]) == pytest.approx(150.0)
        assert float(out["F_MeOH"]) == pytest.approx(25.0)   # product added
        assert float(out["T"]) == 520.0 and float(out["P"]) == pytest.approx(4.9e6)
        assert float(info["extent"][0]) == pytest.approx(25.0)
        assert float(info["key_consumed"]["CO"]) == pytest.approx(25.0)
        assert float(info["min_outlet_flow"]) > 0

    def test_parallel_reactions(self):
        feed = make_stream({"A": 10.0, "B": 0.0, "C": 0.0}, T=300.0, P=1e5)
        rx = StoichiometricReactor(StoichiometricReactorParams(
            stoich=[{"A": -1, "B": 1}, {"A": -2, "C": 1}], key=["A", "A"],
            conversion=[0.5, 0.2]))
        out, info = rx(feed)
        assert float(out["F_B"]) == pytest.approx(5.0)
        assert float(out["F_C"]) == pytest.approx(1.0)
        assert float(out["F_A"]) == pytest.approx(10.0 * (1 - 0.5 - 0.2))
        assert info["extent"].shape == (2,)

    def test_infeasible_conversions_are_flagged(self):
        feed = make_stream({"A": 10.0}, T=300.0, P=1e5)
        rx = StoichiometricReactor(StoichiometricReactorParams(
            stoich=[{"A": -1, "B": 1}, {"A": -1, "C": 1}], key=["A", "A"],
            conversion=[0.7, 0.6]))
        assert float(rx(feed)[1]["min_outlet_flow"]) < 0

    @pytest.mark.slow
    def test_gradients(self):
        def f(X, F_CO):
            rx = StoichiometricReactor(StoichiometricReactorParams(
                stoich=NU, key="CO", conversion=X))
            out, _ = rx(make_stream({"CO": F_CO, "H2": 200.0}, T=500.0, P=5e6))
            return out["F_MeOH"] + 0.1 * out["F_H2"]

        check_grads(f, (0.25, 100.0), order=2, modes=["rev", "fwd"])
        assert float(jax.grad(f)(0.25, 100.0)) == pytest.approx(100.0 - 0.1 * 200.0)

    @pytest.mark.parametrize("kwargs, match", [
        (dict(stoich=NU, key="MeOH", conversion=0.5), "reactant"),
        (dict(stoich=NU, key="N2", conversion=0.5), "does not appear"),
        (dict(stoich=NU, key="CO", conversion=1.5), r"\[0, 1\]"),
        (dict(stoich=[NU, NU], key="CO", conversion=0.5), "one key per reaction"),
        (dict(stoich=[NU, NU], key=["CO", "CO"], conversion=0.5), "one conversion"),
    ])
    def test_validation(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            StoichiometricReactorParams(**kwargs)


class TestComponentSplitter:

    feed = make_stream({"CO": 10.0, "H2": 20.0, "MeOH": 5.0}, T=320.0, P=5e6)

    def test_split(self):
        sep = ComponentSplitter(ComponentSplitterParams(
            {"MeOH": 0.1, "H2": 1.0}))     # CO takes default_fraction = 0
        first, second, info = sep(self.feed)
        assert float(first["F_MeOH"]) == pytest.approx(0.5)
        assert float(second["F_MeOH"]) == pytest.approx(4.5)
        assert float(first["F_H2"]) == 20.0 and float(second["F_H2"]) == 0.0
        assert float(first["F_CO"]) == 0.0 and float(second["F_CO"]) == 10.0
        for s in (first, second):
            assert float(s["T"]) == 320.0 and float(s["P"]) == 5e6
        assert float(info["flow_first"]) == pytest.approx(20.5)

    def test_gradients(self):
        def f(phi, F):
            sep = ComponentSplitter(ComponentSplitterParams({"MeOH": phi}, default_fraction=0.5))
            first, second, _ = sep(make_stream({"CO": 10.0, "MeOH": F}, T=300.0, P=1e5))
            return first["F_MeOH"] * second["F_CO"]

        check_grads(f, (0.3, 5.0), order=2, modes=["rev", "fwd"])

    def test_validation(self):
        with pytest.raises(ValueError):
            ComponentSplitterParams({"A": 1.2})
        with pytest.raises(ValueError):
            ComponentSplitterParams({"A": 0.5}, default_fraction=-0.1)


def test_new_units_are_in_the_catalog():
    from difflow.catalog import catalog

    cat = catalog()
    assert cat["StoichiometricReactor"].category == "reactors"
    assert cat["ComponentSplitter"].category == "separations"
    assert cat["StoichiometricReactor"].is_declarative
    assert cat["ComponentSplitter"].is_declarative


# --- 6. degrees of freedom --------------------------------------------------

class TestDegreesOfFreedom:

    def test_well_posed_loop(self):
        dof = methanol_loop().degrees_of_freedom()
        assert isinstance(dof, DegreesOfFreedom)
        nps = len(SP) + 2
        # fresh, recycle, rx_in, rx_out, gas, crude, purge
        assert dof.n_variables == 7 * nps
        assert dof.n_specified == nps
        assert dof.n_equations == 6 * nps
        assert dof.unit_equations == {"mixer": nps, "reactor": nps,
                                      "sep": 2 * nps, "purge": 2 * nps}
        assert dof.degrees_of_freedom == 0 and dof.is_well_posed
        assert "conversion" in dof.unit_parameters["reactor"]
        assert "split_frac" in dof.unit_parameters["purge"]
        assert set(dof.unknowns) == set(dof.streams) - {"fresh"}
        assert "Degrees of freedom: 0" in str(dof)

    def test_named_recycle_adds_connection_equations(self):
        fs = methanol_loop()
        fs.units[-1].outlet_names = ["purge", "recycle_out"]
        fs.recycles = {"recycle_out": "recycle"}
        dof = fs.degrees_of_freedom()
        assert dof.recycle_equations == {"recycle_out -> recycle": len(SP) + 2}
        assert dof.degrees_of_freedom == 0 and dof.is_well_posed

    def test_dangling_inlet_is_underspecified(self):
        fs = Flowsheet(SP)
        fs.add_feed("a", _fresh())
        fs.add_unit(Unit("mixer", Mixer(SP), ["a", "b"], ["c"]))
        dof = fs.degrees_of_freedom()
        assert dof.degrees_of_freedom == len(SP) + 2
        assert any("'b'" in p for p in dof.problems)

    def test_stream_made_twice_is_overspecified(self):
        fs = Flowsheet(SP)
        fs.add_feed("a", _fresh())
        fs.add_unit(Unit("m1", Mixer(SP), ["a"], ["c"]))
        fs.add_unit(Unit("m2", Mixer(SP), ["a"], ["c"]))
        dof = fs.degrees_of_freedom()
        assert dof.degrees_of_freedom == -(len(SP) + 2)
        assert not dof.is_well_posed
        assert any("produced by 2 units" in p for p in dof.problems)
