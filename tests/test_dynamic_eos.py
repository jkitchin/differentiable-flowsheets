"""Tests for EOS-consistent dynamic units.

Covers the CSTR's EOS-consistent transient (dynamic) branch and the two dynamic
units built on the cubic EOS / two-phase enthalpy:

- CSTR (eos set): the transient fixed point reproduces the steady-state solve.
- DynamicEOSFlash: the steady-state split equals the feed's EOS flash.
- DynamicCounterCurrentHX: the steady state is the LMTD duty and conserves
  energy in a physical (no temperature-cross) operating range.
"""

import jax
import jax.numpy as jnp
import pytest
import optimistix as optx

from difflow.eos import PengRobinson, flash_TP_eos
from difflow.thermo import IdealThermo, CubicThermo
from difflow.database import get_critical_props, get_species_data
from difflow.streams import make_stream, get_flows
from difflow.units.cstr import CSTR, CSTRParams
from difflow.dynamic import DynamicEOSFlash, DynamicCounterCurrentHX

jax.config.update("jax_enable_x64", True)


# Session-scoped: the dynamic units' JIT cores key their compilation cache on
# the thermo object's identity, so a fresh thermo per test would miss the cache
# and recompile the nested Newton-over-flash solve every time. Sharing one
# read-only thermo across the file lets each core compile once.
@pytest.fixture(scope="session")
def pr_cubic():
    names = ["propane", "n_butane", "n_pentane"]
    eos = PengRobinson({c: get_critical_props(c) for c in names})
    ct = CubicThermo(IdealThermo({c: get_species_data(c) for c in names}), eos)
    return names, eos, ct


# ---------------------------------------------------------------------------
# CSTR EOS-consistent dynamic branch
# ---------------------------------------------------------------------------
class TestCSTREOSDynamic:
    @staticmethod
    def _cstr(mode):
        names = ["n_butane", "isobutane"]
        eos = PengRobinson({c: get_critical_props(c) for c in names})
        ct = CubicThermo(IdealThermo({c: get_species_data(c) for c in names}), eos)

        def rate_fn(C, T, p):
            k = p["k"] * jnp.exp(-p["Ea"] / (8.314 * T))
            return jnp.array([k * C["n_butane"]])

        params = CSTRParams(
            V=0.5, rate_fn=rate_fn, stoich=jnp.array([[-1.0], [1.0]]),
            rate_params={"k": jnp.array(1e6), "Ea": jnp.array(45000.0)},
            species_order=names, eos=eos, reaction_phase="vapor",
            dH_rxn=jnp.array([-8000.0]))
        inlet = make_stream({"n_butane": 8.0, "isobutane": 0.5}, 420.0, 20e5)
        return names, eos, CSTR(params, thermo=ct, mode=mode), inlet, params

    def _steady_state_vector(self, names, eos, cstr, inlet, params, mode):
        outlet, _ = cstr(inlet)
        of = get_flows(outlet)
        n = jnp.array([of[s] for s in names])
        y = n / jnp.sum(n)
        rho = eos.density(jnp.asarray(outlet["T"]), inlet["P"], y, phase="vapor")
        n_hold = rho * float(params.V) * y
        if mode == "isothermal":
            return outlet, n_hold
        return outlet, jnp.concatenate([n_hold, jnp.array([outlet["T"]])])

    @pytest.mark.parametrize("mode", ["isothermal", "adiabatic"])
    def test_transient_fixed_point_matches_steady_state(self, mode):
        names, eos, cstr, inlet, params = self._cstr(mode)
        outlet, state = self._steady_state_vector(names, eos, cstr, inlet, params, mode)
        # At the steady-state holdup, the transient derivatives vanish.
        d = cstr.derivatives(0.0, state, {"inlet": inlet})
        assert float(jnp.max(jnp.abs(d))) < 1e-3
        # And the dynamic outputs reproduce the steady-state outlet flows.
        out_dyn = get_flows(cstr.outputs(0.0, state, {"inlet": inlet})["outlet"])
        of = get_flows(outlet)
        for s in names:
            assert float(out_dyn[s]) == pytest.approx(float(of[s]), abs=1e-9)

    def test_derivatives_differentiable(self):
        names, eos, cstr, inlet, params = self._cstr("isothermal")
        _, state = self._steady_state_vector(names, eos, cstr, inlet, params, "isothermal")

        def total_dn(scale):
            return jnp.sum(cstr.derivatives(0.0, state * scale, {"inlet": inlet}))

        g = jax.grad(total_dn)(jnp.array(1.0))
        assert jnp.isfinite(g)


# ---------------------------------------------------------------------------
# DynamicEOSFlash
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestDynamicEOSFlash:
    def _steady_state(self, eos, names, feed, P):
        flash = DynamicEOSFlash(eos=eos, species_order=names, P=P,
                                tau_liquid=100.0, tau_vapor=1.0)
        inp = {"feed": feed}
        sol = optx.root_find(
            lambda n, _: flash.derivatives(0.0, n, inp),
            optx.Newton(rtol=1e-8, atol=1e-8),
            flash.initial_state(inp), max_steps=200, throw=False)
        return flash, inp, sol.value

    def test_steady_state_split_matches_feed_flash(self, pr_cubic):
        names, eos, _ = pr_cubic
        feed = make_stream({"propane": 5.0, "n_butane": 3.0, "n_pentane": 2.0}, 330.0, 8e5)
        flash, inp, nss = self._steady_state(eos, names, feed, 8e5)
        out = flash.outputs(0.0, nss, inp)
        L, V = get_flows(out["liquid"]), get_flows(out["vapor"])

        z = jnp.array([get_flows(feed)[s] for s in names]); z = z / z.sum()
        Vf, x_ref, y_ref = flash_TP_eos(eos, z, feed["T"], jnp.asarray(8e5))
        x_dyn = jnp.array([L[s] for s in names]); x_dyn = x_dyn / x_dyn.sum()
        y_dyn = jnp.array([V[s] for s in names]); y_dyn = y_dyn / y_dyn.sum()

        assert float(jnp.max(jnp.abs(x_dyn - x_ref))) < 1e-8
        assert float(jnp.max(jnp.abs(y_dyn - y_ref))) < 1e-8
        beta = sum(float(v) for v in V.values()) / sum(
            float(v) for v in {**{k: L[k] + V[k] for k in names}}.values())
        assert beta == pytest.approx(float(Vf), abs=1e-4)

    def test_steady_state_mass_balance(self, pr_cubic):
        names, eos, _ = pr_cubic
        feed = make_stream({"propane": 5.0, "n_butane": 3.0, "n_pentane": 2.0}, 330.0, 8e5)
        flash, inp, nss = self._steady_state(eos, names, feed, 8e5)
        out = flash.outputs(0.0, nss, inp)
        draw = sum(float(v) for v in get_flows(out["liquid"]).values()) + \
            sum(float(v) for v in get_flows(out["vapor"]).values())
        feed_total = sum(float(v) for v in get_flows(feed).values())
        assert draw == pytest.approx(feed_total, rel=1e-6)

    def test_derivatives_differentiable(self, pr_cubic):
        names, eos, _ = pr_cubic
        feed = make_stream({"propane": 5.0, "n_butane": 3.0, "n_pentane": 2.0}, 330.0, 8e5)
        flash = DynamicEOSFlash(eos=eos, species_order=names, P=8e5)
        inp = {"feed": feed}
        n0 = flash.initial_state(inp)

        def total(scale):
            return jnp.sum(flash.derivatives(0.0, n0 * scale, inp))

        g = jax.grad(total)(jnp.array(1.0))
        assert jnp.isfinite(g)


# ---------------------------------------------------------------------------
# DynamicCounterCurrentHX
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestDynamicCounterCurrentHX:
    def _setup(self, ct, UA):
        hot = make_stream({"propane": 5.0, "n_butane": 3.0, "n_pentane": 2.0}, 400.0, 10e5)
        cold = make_stream({"propane": 4.0, "n_butane": 4.0, "n_pentane": 2.0}, 300.0, 10e5)
        hx = DynamicCounterCurrentHX(UA=UA, thermo=ct, tau=30.0)
        return hx, {"hot": hot, "cold": cold}, hot, cold

    def test_cold_start_heats_up(self, pr_cubic):
        _, _, ct = pr_cubic
        hx, inp, _, _ = self._setup(ct, 800.0)
        d0 = hx.derivatives(0.0, hx.initial_state(inp), inp)
        assert float(d0[0]) > 0.0  # zero-duty start: duty grows

    def test_steady_state_conserves_energy(self, pr_cubic):
        _, _, ct = pr_cubic
        hx, inp, hot, cold = self._setup(ct, 800.0)  # physical UA, no temperature cross
        sol = optx.root_find(lambda Q, _: hx.derivatives(0.0, Q, inp),
                             optx.Newton(rtol=1e-8, atol=1e-3),
                             jnp.array([5e4]), max_steps=80, throw=False)
        Qss = sol.value
        out = hx.outputs(0.0, Qss, inp)
        H_hi = ct.stream_enthalpy_flash(get_flows(hot), hot["T"], hot["P"])
        H_ho = ct.stream_enthalpy_flash(get_flows(out["hot_out"]), out["hot_out"]["T"], hot["P"])
        H_ci = ct.stream_enthalpy_flash(get_flows(cold), cold["T"], cold["P"])
        H_co = ct.stream_enthalpy_flash(get_flows(out["cold_out"]), out["cold_out"]["T"], cold["P"])
        hot_drop = float(H_hi - H_ho)
        cold_gain = float(H_co - H_ci)
        assert hot_drop == pytest.approx(cold_gain, rel=1e-4)   # energy conserved
        assert hot_drop == pytest.approx(float(Qss[0]), rel=1e-4)  # = the duty
        # Physical: hot cools, cold heats, no crossing.
        assert float(out["hot_out"]["T"]) < 400.0
        assert float(out["cold_out"]["T"]) > 300.0
        assert float(out["hot_out"]["T"]) > float(out["cold_out"]["T"])

    def test_duty_differentiable_wrt_UA(self, pr_cubic):
        _, _, ct = pr_cubic
        hot = make_stream({"propane": 5.0, "n_butane": 3.0, "n_pentane": 2.0}, 400.0, 10e5)
        cold = make_stream({"propane": 4.0, "n_butane": 4.0, "n_pentane": 2.0}, 300.0, 10e5)
        inp = {"hot": hot, "cold": cold}

        def duty(UA):
            hx = DynamicCounterCurrentHX(UA=UA, thermo=ct, tau=30.0)
            # one derivative evaluation at a fixed duty is enough to check AD path
            return hx.derivatives(0.0, jnp.array([4e4]), inp)[0]

        g = jax.grad(duty)(jnp.array(800.0))
        assert jnp.isfinite(g)
