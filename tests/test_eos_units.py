"""Tests for EOS-consistent process units (issue #171).

Turboexpander, Compressor, JTValve and ComponentSeparator built on a
two-phase-aware CubicThermo (PR EOS + ideal-gas Cp), with implicit-diff
temperature solves.
"""

import jax
import jax.numpy as jnp
import pytest

from difflow.units.eos_units import (
    Turboexpander,
    TurboexpanderParams,
    Compressor,
    CompressorParams,
    JTValve,
    JTValveParams,
    ComponentSeparator,
    ComponentSeparatorParams,
)
from difflow.thermo import IdealThermo, CubicThermo
from difflow.eos import PengRobinson
from difflow.database import get_critical_props, get_species_data
from difflow.streams import make_stream, get_flows

jax.config.update("jax_enable_x64", True)

# A representative NGL feed (methane-rich, with a light/heavy spread). Kept to
# four components on purpose: these unit tests only check thermodynamic signs,
# monotonicity, mass balance and differentiability -- none of which need the
# full demethanizer slate, so the smaller mixture keeps the fixtures simple and
# free of extra property-database lookups.
NGL = ["methane", "ethane", "propane", "n_butane"]
FEED_FLOWS = [86.0, 7.0, 3.0, 1.0]  # propane flow == 3.0 is asserted below


@pytest.fixture
def thermo():
    ideal = IdealThermo({c: get_species_data(c) for c in NGL})
    eos = PengRobinson({c: get_critical_props(c) for c in NGL})
    return CubicThermo(ideal, eos)


@pytest.fixture
def feed():
    return make_stream({c: f for c, f in zip(NGL, FEED_FLOWS)}, 305.0, 60e5)


class TestTurboexpander:
    def test_expansion_cools_and_extracts_work(self, thermo, feed):
        exp = Turboexpander(TurboexpanderParams(P_out=20e5, eta_isentropic=0.80), thermo)
        out, info = exp(feed)
        assert float(out["T"]) < float(feed["T"])       # expansion cools
        assert float(out["P"]) == pytest.approx(20e5)
        assert float(info["W"]) > 0.0                    # shaft work extracted

    def test_efficiency_below_one_recovers_less_than_isentropic(self, thermo, feed):
        """An inefficient expander does less work, so its outlet is warmer than
        the reversible (isentropic) outlet."""
        exp = Turboexpander(TurboexpanderParams(P_out=20e5, eta_isentropic=0.80), thermo)
        out, info = exp(feed)
        assert float(out["T"]) > float(info["T_isen"])

    def test_ideal_efficiency_is_isentropic(self, thermo, feed):
        """With eta=1 the outlet temperature equals the isentropic temperature."""
        exp = Turboexpander(TurboexpanderParams(P_out=20e5, eta_isentropic=1.0), thermo)
        out, info = exp(feed)
        assert float(out["T"]) == pytest.approx(float(info["T_isen"]), abs=0.05)

    def test_mass_is_conserved(self, thermo, feed):
        exp = Turboexpander(TurboexpanderParams(P_out=20e5), thermo)
        out, _ = exp(feed)
        fin, fout = get_flows(feed), get_flows(out)
        for c in NGL:
            assert float(fout[c]) == pytest.approx(float(fin[c]))

    def test_work_differentiable_wrt_pressure(self, thermo, feed):
        def W(Pout):
            e = Turboexpander(TurboexpanderParams(P_out=Pout, eta_isentropic=0.8), thermo)
            return e(feed)[1]["W"]
        g = jax.grad(W)(jnp.array(20e5))
        assert jnp.isfinite(g)
        assert float(g) < 0.0  # lower discharge pressure -> more work extracted


class TestCompressor:
    def test_compression_heats_and_consumes_work(self, thermo, feed):
        comp = Compressor(CompressorParams(P_out=90e5, eta_isentropic=0.75), thermo)
        out, info = comp(feed)
        assert float(out["T"]) > float(feed["T"])       # compression heats
        assert float(out["P"]) == pytest.approx(90e5)
        assert float(info["W"]) > 0.0                    # shaft work consumed

    def test_lower_efficiency_needs_more_work(self, thermo, feed):
        hi = Compressor(CompressorParams(P_out=90e5, eta_isentropic=0.90), thermo)
        lo = Compressor(CompressorParams(P_out=90e5, eta_isentropic=0.60), thermo)
        assert float(lo(feed)[1]["W"]) > float(hi(feed)[1]["W"])

    def test_expander_then_compressor_round_trip_costs_net_work(self, thermo, feed):
        """Expand then recompress to the original pressure: the irreversible
        round trip must consume more work than it recovered (2nd law)."""
        exp = Turboexpander(TurboexpanderParams(P_out=20e5, eta_isentropic=0.80), thermo)
        low, e_info = exp(feed)
        comp = Compressor(CompressorParams(P_out=60e5, eta_isentropic=0.75), thermo)
        _, c_info = comp(low)
        assert float(c_info["W"]) > float(e_info["W"])

    def test_work_differentiable(self, thermo, feed):
        def W(eta):
            c = Compressor(CompressorParams(P_out=90e5, eta_isentropic=eta), thermo)
            return c(feed)[1]["W"]
        g = jax.grad(W)(jnp.array(0.75))
        assert jnp.isfinite(g)


class TestJTValve:
    def test_isenthalpic(self, thermo, feed):
        valve = JTValve(JTValveParams(P_out=20e5), thermo)
        out, info = valve(feed)
        H_in = thermo.stream_enthalpy_flash(get_flows(feed), feed["T"], feed["P"])
        H_out = thermo.stream_enthalpy_flash(get_flows(out), out["T"], out["P"])
        assert float(H_out) == pytest.approx(float(H_in), rel=1e-6)

    def test_joule_thomson_cooling(self, thermo, feed):
        valve = JTValve(JTValveParams(P_out=20e5), thermo)
        out, _ = valve(feed)
        assert float(out["T"]) < float(feed["T"])

    def test_valve_cools_less_than_expander(self, thermo, feed):
        """Same pressure drop: the irreversible valve extracts no work and so
        cools less than the (work-extracting) turboexpander."""
        valve = JTValve(JTValveParams(P_out=20e5), thermo)
        exp = Turboexpander(TurboexpanderParams(P_out=20e5, eta_isentropic=0.80), thermo)
        T_valve = float(valve(feed)[0]["T"])
        T_exp = float(exp(feed)[0]["T"])
        assert T_valve > T_exp

    def test_outlet_T_differentiable(self, thermo, feed):
        def T_out(Pout):
            v = JTValve(JTValveParams(P_out=Pout), thermo)
            return v(feed)[0]["T"]
        g = jax.grad(T_out)(jnp.array(20e5))
        assert jnp.isfinite(g)


class TestComponentSeparator:
    def test_recoveries_and_mass_balance(self, thermo, feed):
        rec = {"methane": 0.01, "ethane": 0.30, "propane": 0.95,
               "n_butane": 0.99}
        sep = ComponentSeparator(ComponentSeparatorParams(recovery_to_product=rec), thermo)
        residue, product, _ = sep(feed)
        fin, fres, fprod = get_flows(feed), get_flows(residue), get_flows(product)
        for c in NGL:
            # per-component mass balance: residue + product = feed
            assert float(fres[c] + fprod[c]) == pytest.approx(float(fin[c]))
            # product recovery matches the spec (default 0 for unlisted species)
            expected = rec.get(c, 0.0)
            assert float(fprod[c]) == pytest.approx(expected * float(fin[c]))

    def test_default_recovery_applies_to_unlisted(self, thermo, feed):
        sep = ComponentSeparator(
            ComponentSeparatorParams(recovery_to_product={"propane": 1.0},
                                     default_recovery=0.0),
            thermo,
        )
        _, product, _ = sep(feed)
        fprod = get_flows(product)
        assert float(fprod["methane"]) == pytest.approx(0.0)
        assert float(fprod["propane"]) == pytest.approx(3.0)

    def test_recovery_differentiable(self, thermo, feed):
        def prod_propane(r):
            sep = ComponentSeparator(
                ComponentSeparatorParams(recovery_to_product={"propane": r}), thermo)
            return get_flows(sep(feed)[1])["propane"]
        g = jax.grad(prod_propane)(jnp.array(0.9))
        assert float(g) == pytest.approx(3.0, rel=1e-4)  # d/dr (r * 3.0) = 3.0
