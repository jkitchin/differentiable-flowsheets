"""Smoke tests for the EOS-consistent dynamic units."""

import jax
import jax.numpy as jnp
import pytest

from difflow.database import get_critical_props, get_species_data
from difflow.dynamic import DynamicCounterCurrentHX, DynamicEOSFlash
from difflow.eos import PengRobinson
from difflow.streams import make_stream
from difflow.thermo import CubicThermo, IdealThermo
from difflow.units.cstr import CSTR, CSTRParams

jax.config.update("jax_enable_x64", True)

NAMES = ["propane", "n_butane", "n_pentane"]


@pytest.fixture(scope="module")
def pr_cubic():
    eos = PengRobinson({c: get_critical_props(c) for c in NAMES})
    return eos, CubicThermo(IdealThermo({c: get_species_data(c) for c in NAMES}), eos)


def _feed(T=330.0, P=8e5):
    return make_stream({"propane": 5.0, "n_butane": 3.0, "n_pentane": 2.0}, T, P)


def test_cstr_eos_transient(pr_cubic):
    eos, ct = pr_cubic
    params = CSTRParams(
        V=0.5, rate_fn=lambda C, T, p: jnp.array([p["k"] * C["propane"]]),
        stoich=jnp.array([[-1.0], [1.0], [0.0]]), rate_params={"k": 1e-3},
        species_order=NAMES, eos=eos, reaction_phase="vapor")
    cstr = CSTR(params, thermo=ct, mode="isothermal")
    inp = {"inlet": make_stream({"propane": 5.0, "n_butane": 3.0, "n_pentane": 2.0}, 420.0, 20e5)}
    state = cstr.initial_state(inp)
    assert jnp.all(jnp.isfinite(cstr.derivatives(0.0, state, inp)))
    assert "outlet" in cstr.outputs(0.0, state, inp)


def test_dynamic_eos_flash(pr_cubic):
    eos, _ = pr_cubic
    flash = DynamicEOSFlash(eos=eos, species_order=NAMES, P=8e5)
    inp = {"feed": _feed()}
    n0 = flash.initial_state(inp)
    assert jnp.all(jnp.isfinite(flash.derivatives(0.0, n0, inp)))
    out = flash.outputs(0.0, n0, inp)
    assert {"liquid", "vapor"} <= set(out)


def test_dynamic_counter_current_hx(pr_cubic):
    _, ct = pr_cubic
    inp = {"hot": _feed(400.0, 10e5), "cold": _feed(300.0, 10e5)}

    def dQ(UA):
        hx = DynamicCounterCurrentHX(UA=UA, thermo=ct, tau=30.0)
        return hx.derivatives(0.0, jnp.array([4e4]), inp)[0]

    assert jnp.isfinite(dQ(800.0))
    assert jnp.isfinite(jax.grad(dQ)(800.0))
    hx = DynamicCounterCurrentHX(UA=800.0, thermo=ct, tau=30.0)
    assert {"hot_out", "cold_out"} <= set(hx.outputs(0.0, jnp.array([4e4]), inp))
