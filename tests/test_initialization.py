"""Tests for initialization strategies and acceleration methods."""

import jax
import jax.numpy as jnp
import pytest

from difflow import (
    CSTR,
    CSTRParams,
    PFR,
    PFRParams,
    Flash,
    FlashParams,
    IdealThermo,
    SpeciesData,
    make_stream,
    get_flows,
    Flowsheet,
    Unit,
)
from difflow.initialization import (
    wegstein_acceleration,
    anderson_acceleration_step,
    AndersonAccelerator,
    estimate_cstr_conversion,
    estimate_outlet_temperature,
)


# Enable 64-bit precision for tests
jax.config.update("jax_enable_x64", True)


@pytest.fixture
def simple_thermo():
    """Simple two-component thermodynamics."""
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
    """First-order reaction rate function."""
    def rate_fn(C, T, params):
        k = params["k"]
        return jnp.array([k * C["A"]])
    return rate_fn


@pytest.fixture
def simple_inlet():
    """Simple inlet stream."""
    return make_stream({"A": 1.0, "B": 0.0}, 350.0, 101325.0)


@pytest.fixture
def simple_cstr(simple_thermo, simple_rate_fn):
    """Create a simple CSTR for testing."""
    stoich = jnp.array([[-1.0], [+1.0]])  # A → B
    params = CSTRParams(
        V=jnp.array(1.0),
        rate_fn=simple_rate_fn,
        stoich=stoich,
        rate_params={"k": jnp.array(0.1)},
        species_order=["A", "B"],
    )
    return CSTR(params, thermo=simple_thermo, mode="isothermal")


def test_wegstein_step():
    # Wegstein is exact for a linear map: g(x) = 0.5 x + 0.25 has root 0.5 (#164).
    x_new = wegstein_acceleration(jnp.array([0.0]), jnp.array([0.25]),
                                  jnp.array([0.25]), jnp.array([0.375]))
    assert x_new.shape == (1,)
    assert float(x_new[0]) == pytest.approx(0.5, abs=1e-6)


def test_anderson_step_and_accelerator():
    x_hist = jnp.array([[0.0, 0.0], [0.3, 0.2], [0.45, 0.35]])
    g_hist = jnp.array([[0.3, 0.2], [0.45, 0.35], [0.52, 0.42]])
    x_new = anderson_acceleration_step(x_hist, g_hist, m=2)
    assert x_new.shape == (2,) and jnp.isfinite(x_new).all()

    acc = AndersonAccelerator(m=2)
    for i in range(5):
        acc.step(jnp.array([float(i)]), jnp.array([float(i) + 0.5]))
    assert len(acc.x_hist) <= 3
    acc.reset()
    assert len(acc.x_hist) == 0


def test_estimate_helpers():
    assert 0.0 < estimate_cstr_conversion(0.1, 10.0, order=1) < 1.0
    T = estimate_outlet_temperature(make_stream({"A": 1.0}, 350.0, 101325.0),
                                    heat_duty=0.0, heat_of_reaction=7500.0, Cp_avg=75.0)
    assert jnp.isfinite(T)


def test_cstr_initialize(simple_cstr, simple_inlet):
    result = simple_cstr.initialize(simple_inlet)
    assert {"outlet", "states", "info"} <= set(result)
    assert {"conversion", "residence_time"} <= set(result["states"])


def test_pfr_initialize(simple_thermo, simple_rate_fn, simple_inlet):
    params = PFRParams(V=jnp.array(1.0), rate_fn=simple_rate_fn,
                       stoich=jnp.array([[-1.0], [+1.0]]),
                       rate_params={"k": jnp.array(0.1)}, species_order=["A", "B"])
    result = PFR(params, thermo=simple_thermo, mode="isothermal").initialize(simple_inlet)
    assert {"outlet", "states", "info"} <= set(result)


def test_flash_initialize_maps_onto_outlet_names(simple_thermo):
    from difflow.flowsheet import _initialized_outlets

    flash = Flash(FlashParams(species_order=["A", "B"]), simple_thermo)
    result = flash.initialize(make_stream({"A": 0.5, "B": 0.5}, 350.0, 101325.0))
    assert {"liquid", "vapor", "states"} <= set(result)
    outlets = _initialized_outlets(result, Unit("flash", flash, ["feed"], ["liq", "vap"]))
    assert list(outlets) == ["liq", "vap"]
    assert _initialized_outlets(None, Unit("flash", flash, ["feed"], ["liq", "vap"])) == {}


def _loop(operation):
    def splitter(inlet):
        flows = get_flows(inlet)
        return (make_stream({s: f * 0.7 for s, f in flows.items()}, inlet["T"], inlet["P"]),
                make_stream({s: f * 0.3 for s, f in flows.items()}, inlet["T"], inlet["P"]))

    def mixer(a, b):
        fa, fb = get_flows(a), get_flows(b)
        return make_stream({s: fa[s] + fb[s] for s in fa}, a["T"], a["P"])

    fs = Flowsheet(["A", "B"])
    fs.add_feed("feed", make_stream({"A": 1.0, "B": 0.0}, 350.0, 101325.0))
    fs.add_unit(Unit("mixer", mixer, ["feed", "recycle"], ["mixed"]))
    fs.add_unit(Unit("op", operation, ["mixed"], ["op_out"]))
    fs.add_unit(Unit("splitter", splitter, ["op_out"], ["product", "recycle_source"]))
    fs.add_recycle("recycle_source", "recycle")
    return fs


def test_tear_guess_comes_from_the_source_end():
    """#247: the guess used to be the default stream on every flowsheet."""
    fs = _loop(lambda inlet: inlet)
    guess = {k: float(v) for k, v in get_flows(fs._initialize_tear_stream("recycle_source")).items()}
    zero = {k: float(v) for k, v in get_flows(fs._make_zero_stream()).items()}
    assert guess != pytest.approx(zero)


def test_tear_initialize_stands_in_for_a_unit_that_raised():
    class FailsButEstimates:
        def __call__(self, inlet):
            raise RuntimeError("root find did not converge")

        def initialize(self, inlet, **kwargs):
            return {"outlets": (inlet,)}

    guess = get_flows(_loop(FailsButEstimates())._initialize_tear_stream("recycle_source"))
    assert float(guess["A"]) > 0.01


def test_tear_initialization_warns_without_an_initializer():
    from difflow.initialization import TearInitializationWarning

    class JustFails:
        def __call__(self, inlet):
            raise RuntimeError("root find did not converge")

    with pytest.warns(TearInitializationWarning, match="no initialize"):
        _loop(JustFails())._initialize_tear_stream("recycle_source")


def test_tear_initialization_warns_when_initializer_fails():
    from difflow.initialization import TearInitializationWarning

    class FailsTwice:
        def __call__(self, inlet):
            raise RuntimeError("root find did not converge")

        def initialize(self, inlet, **kwargs):
            raise ValueError("no bracketing interval")

    with pytest.warns(TearInitializationWarning, match="neither could its"):
        _loop(FailsTwice())._initialize_tear_stream("recycle_source")
