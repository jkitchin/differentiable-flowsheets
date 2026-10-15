"""Flowsheet gaps found writing Chapter 1 of the unit-operations book."""

import jax
import jax.numpy as jnp
import pytest

from difflow import (
    ComponentSplitter,
    ComponentSplitterParams,
    DegreesOfFreedom,
    Mixer,
    Splitter,
    StoichiometricReactor,
    StoichiometricReactorParams,
    make_stream,
)
from difflow.flowsheet import Flowsheet, Unit

jax.config.update("jax_enable_x64", True)

SP = ["CO", "H2", "MeOH", "CH4"]
NU = {"CO": -1.0, "H2": -2.0, "MeOH": 1.0}
P_LOOP = 50e5


def _fresh(P=P_LOOP, T=500.0):
    return make_stream({"CO": 100.0, "H2": 200.0, "MeOH": 0.0, "CH4": 3.0}, T=T, P=P)


def methanol_loop(p=0.1, default_P=None):
    reactor = StoichiometricReactor(StoichiometricReactorParams(
        stoich=NU, key="CO", conversion=0.25))
    fs = Flowsheet(SP, default_P=default_P)
    fs.add_feed("fresh", _fresh())
    fs.add_unit(Unit("mixer", Mixer(SP), ["fresh", "recycle"], ["rx_in"]))
    fs.add_unit(Unit("reactor", reactor, ["rx_in"], ["rx_out"]))
    fs.add_unit(Unit("sep", ComponentSplitter(ComponentSplitterParams(
        {"MeOH": 0.0}, default_fraction=1.0)), ["rx_out"], ["gas", "crude"]))
    fs.add_unit(Unit("purge", Splitter(SP), ["gas"], ["purge", "recycle"],
                     {"split_frac": p}))
    fs.add_recycle("recycle", "recycle")
    return fs


def test_pressurized_loop_keeps_its_pressure():
    """Regression: the tear used to start (and stay) at 1 atm."""
    fs = methanol_loop()
    st = fs.solve(acceleration="anderson", tol=1e-10, max_iter=500)
    assert float(st["recycle"]["P"]) == pytest.approx(P_LOOP)
    assert fs.last_solve_gain is not None


def test_tear_inference_flag_survives_serialization():
    from difflow.serialize import from_dict, to_dict

    assert from_dict(to_dict(methanol_loop())).infer_tear_P
    assert not from_dict(to_dict(Flowsheet(SP, default_P=2e5))).infer_tear_P


def test_mixer_missing_species_is_zero():
    out, _ = Mixer(["A", "B"])(make_stream({"A": 1.0}, T=300.0, P=1e5),
                               make_stream({"A": 1.0, "B": 2.0}, T=300.0, P=1e5))
    assert float(out["F_A"]) == 2.0 and float(out["F_B"]) == 2.0


def test_stoichiometric_reactor_smoke():
    rx = StoichiometricReactor(StoichiometricReactorParams(stoich=NU, key="CO",
                                                           conversion=0.25))
    feed = make_stream({"CO": 100.0, "H2": 200.0}, T=500.0, P=5e6)
    out, info = rx(feed)
    assert float(out["F_CO"]) == pytest.approx(75.0)
    assert "extent" in info and "min_outlet_flow" in info

    def f(X):
        return StoichiometricReactor(StoichiometricReactorParams(
            stoich=NU, key="CO", conversion=X))(feed)[0]["F_MeOH"]

    assert jnp.isfinite(jax.grad(f)(0.25))


@pytest.mark.parametrize("kwargs, match", [
    (dict(stoich=NU, key="MeOH", conversion=0.5), "reactant"),
    (dict(stoich=NU, key="N2", conversion=0.5), "does not appear"),
    (dict(stoich=NU, key="CO", conversion=1.5), r"\[0, 1\]"),
    (dict(stoich=[NU, NU], key="CO", conversion=0.5), "one key per reaction"),
    (dict(stoich=[NU, NU], key=["CO", "CO"], conversion=0.5), "one conversion"),
])
def test_stoichiometric_reactor_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        StoichiometricReactorParams(**kwargs)


def test_component_splitter_smoke():
    feed = make_stream({"CO": 10.0, "H2": 20.0, "MeOH": 5.0}, T=320.0, P=5e6)
    first, second, info = ComponentSplitter(ComponentSplitterParams(
        {"MeOH": 0.1, "H2": 1.0}))(feed)
    assert float(first["F_MeOH"] + second["F_MeOH"]) == pytest.approx(5.0)
    assert "flow_first" in info
    with pytest.raises(ValueError):
        ComponentSplitterParams({"A": 1.2})


def test_new_units_are_in_the_catalog():
    from difflow.catalog import catalog

    cat = catalog()
    assert cat["StoichiometricReactor"].category == "reactors"
    assert cat["ComponentSplitter"].category == "separations"


def test_degrees_of_freedom():
    dof = methanol_loop().degrees_of_freedom()
    assert isinstance(dof, DegreesOfFreedom)
    assert dof.degrees_of_freedom == 0 and dof.is_well_posed
    assert "Degrees of freedom: 0" in str(dof)

    fs = Flowsheet(SP)
    fs.add_feed("a", _fresh())
    fs.add_unit(Unit("mixer", Mixer(SP), ["a", "b"], ["c"]))
    assert any("'b'" in p for p in fs.degrees_of_freedom().problems)

    fs = Flowsheet(SP)
    fs.add_feed("a", _fresh())
    fs.add_unit(Unit("m1", Mixer(SP), ["a"], ["c"]))
    fs.add_unit(Unit("m2", Mixer(SP), ["a"], ["c"]))
    assert any("produced by 2 units" in p for p in fs.degrees_of_freedom().problems)
