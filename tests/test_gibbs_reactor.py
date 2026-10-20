"""Tests for difflow.units.gibbs.GibbsReactor (#468).

Code checks only (the test policy in CLAUDE.md), on a core
:class:`GibbsThermo` built from the NIST-JANAF tables:

* the solver solves its own equations: WGS and steam-methane-reforming
  quotients equal K(T) of the same thermo, to Newton tolerance;
* element balances to round-off; trace species non-negative and finite;
* adiabatic energy closure; the overall and per-reaction approach
  temperatures; the restricted (reaction-set) mode agrees with the
  unrestricted one when the reactions span the space;
* validation errors; finite derivatives w.r.t. T, P, feed and duty; jit.

The comparisons with JANAF log Kf and with IDAES's GibbsReactor
(``tests/reference/gibbs_generate.py``) are not per-commit tests.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow import GibbsReactor, GibbsReactorParams, GibbsThermo, make_stream
from difflow.units.gibbs import R_GAS

jax.config.update("jax_enable_x64", True)

# --- JANAF rows (T K, Cp J/mol/K) used for the Cp fit, plus Hf(298) kJ/mol
# and S(298) J/mol/K. Values copied from the NIST-JANAF tables named above.
_T = np.array([298.15, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500])
JANAF = {
    # name: (table, Hf298 kJ/mol, S298 J/mol/K, Cp row, {element: n})
    "CO": ("C-093", -110.527, 197.653,
           [29.142, 29.342, 29.794, 30.443, 31.171, 31.899, 32.577, 33.183, 33.710,
            34.175, 34.572, 34.920, 35.217], {"C": 1, "O": 1}),
    "CO2": ("C-095", -393.522, 213.795,
            [37.129, 41.325, 44.627, 47.321, 49.564, 51.434, 52.999, 54.308, 55.409,
             56.342, 57.137, 57.802, 58.379], {"C": 1, "O": 2}),
    "H2O": ("H-064", -241.826, 188.834,
            [33.590, 34.262, 35.226, 36.325, 37.495, 38.721, 39.987, 41.268, 42.536,
             43.768, 44.945, 46.054, 47.090], {"H": 2, "O": 1}),
    "H2": ("H-050", 0.0, 130.680,
           [28.836, 29.181, 29.260, 29.327, 29.441, 29.624, 29.881, 30.205, 30.581,
            30.992, 31.423, 31.861, 32.298], {"H": 2}),
    "CH4": ("C-067", -74.873, 186.251,
            [35.639, 40.500, 46.342, 52.227, 57.794, 62.932, 67.601, 71.795, 75.529,
             78.833, 81.744, 84.305, 86.556], {"C": 1, "H": 4}),
}
WGS = {"CO": -1, "H2O": -1, "CO2": 1, "H2": 1}
SMR = {"CH4": -1, "H2O": -1, "CO": 1, "H2": 3}


def janaf_thermo(names=("CH4", "H2O", "CO", "CO2", "H2")) -> GibbsThermo:
    cp = [np.polyfit(_T, JANAF[s][3], 3)[::-1] for s in names]
    return GibbsThermo(names, Hf=[JANAF[s][1] * 1e3 for s in names],
                       S0=[JANAF[s][2] for s in names], cp=cp,
                       elements={s: JANAF[s][4] for s in names})


def quotient(out, rx, P=None):
    """ln of the equilibrium quotient on mole fractions, times (P/P0)^dn."""
    flows = {k[2:]: v for k, v in out.items() if k.startswith("F_")}
    N = sum(flows.values())
    dn = sum(rx.values())
    P = out["P"] if P is None else P
    return (sum(nu * jnp.log(flows[s] / N) for s, nu in rx.items())
            + dn * jnp.log(P / 1e5))


def thermo_lnK(th, rx, T):
    idx = {s: i for i, s in enumerate(th.names)}
    g = th.gibbs(T)
    return -sum(nu * g[idx[s]] for s, nu in rx.items()) / (R_GAS * T)


@pytest.fixture(scope="module")
def th():
    return janaf_thermo()


# --------------------------------------------------------------------------
# closed forms
# --------------------------------------------------------------------------
class TestClosedForms:
    @pytest.mark.parametrize("T", [900.0])
    def test_wgs_quotient_is_K(self, th, T):
        out, info = GibbsReactor(GibbsReactorParams(thermo=th))(
            make_stream({"CO": 1.0, "H2O": 1.5, "H2": 0.2}, T=T, P=3e6))
        assert bool(info["converged"])
        np.testing.assert_allclose(quotient(out, WGS), thermo_lnK(th, WGS, T), atol=1e-9)

    @pytest.mark.parametrize("P", [2e6])
    def test_smr_and_wgs_quotients_are_K(self, th, P):
        T = 1000.0
        out, info = GibbsReactor(GibbsReactorParams(thermo=th))(
            make_stream({"CH4": 1.0, "H2O": 3.0}, T=T, P=P))
        assert bool(info["converged"])
        for rx in (WGS, SMR):
            np.testing.assert_allclose(quotient(out, rx), thermo_lnK(th, rx, T), atol=1e-9)

# --------------------------------------------------------------------------
# balances and trace species
# --------------------------------------------------------------------------
class TestBalances:
    def test_elements_close_to_round_off(self, th):
        out, info = GibbsReactor(GibbsReactorParams(thermo=th))(
            make_stream({"CH4": 1.0, "H2O": 3.0, "CO2": 0.3}, T=950.0, P=2e6))
        for e, d in info["element_balance"].items():
            assert abs(float(d)) < 1e-12, e

    def test_trace_species_are_positive_and_finite(self):
        names = ("CH4", "H2O", "CO", "CO2", "H2")
        th = janaf_thermo(names)
        # at 500 K and 30 bar CO is a trace; at 1500 K CH4 is
        for T in (500.0, 1500.0):
            out, info = GibbsReactor(GibbsReactorParams(thermo=th))(
                make_stream({"CH4": 1.0, "H2O": 3.0}, T=T, P=3e6))
            assert bool(info["converged"])
            for s in names:
                v = float(out[f"F_{s}"])
                assert np.isfinite(v) and v > 0.0, (T, s, v)

    def test_species_of_an_unfed_element_are_exactly_zero(self):
        tc = pytest.importorskip("difflow_refinery.thermochemistry")
        names = ["methane", "water", "carbon_monoxide", "carbon_dioxide", "hydrogen",
                 "nitrogen", "ammonia"]
        out, info = GibbsReactor(GibbsReactorParams(thermo=tc.IdealGasSet(names)))(
            make_stream({"methane": 1.0, "water": 3.0}, T=1000.0, P=2e6))
        assert bool(info["converged"])
        assert float(out["F_nitrogen"]) == 0.0 and float(out["F_ammonia"]) == 0.0
        assert float(out["F_carbon_monoxide"]) > 0.1

    def test_unknown_inlet_species_raises(self, th):
        with pytest.raises(ValueError, match="not in the reactor's thermo"):
            GibbsReactor(GibbsReactorParams(thermo=th))(
                make_stream({"CH4": 1.0, "N2": 1.0}, T=900.0, P=1e5))

    def test_bad_mode_raises(self, th):
        with pytest.raises(ValueError, match="mode"):
            GibbsReactorParams(thermo=th, mode="isobaric")

    def test_unbalanced_reaction_raises(self, th):
        with pytest.raises(ValueError, match="balance"):
            GibbsReactor(GibbsReactorParams(thermo=th, reactions=[{"CO": -1, "CO2": 1}]))


# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------
class TestModes:
    def test_T_out_overrides_the_inlet_temperature(self, th):
        a, _ = GibbsReactor(GibbsReactorParams(thermo=th, T_out=1100.0))(
            make_stream({"CH4": 1.0, "H2O": 3.0}, T=600.0, P=1e6))
        b, _ = GibbsReactor(GibbsReactorParams(thermo=th))(
            make_stream({"CH4": 1.0, "H2O": 3.0}, T=1100.0, P=1e6))
        for k in a:
            np.testing.assert_allclose(a[k], b[k], rtol=1e-10)

    def test_pressure_drop_sets_the_equilibrium_pressure(self, th):
        a, _ = GibbsReactor(GibbsReactorParams(thermo=th, pressure_drop=5e5))(
            make_stream({"CH4": 1.0, "H2O": 3.0}, T=1000.0, P=2e6))
        b, _ = GibbsReactor(GibbsReactorParams(thermo=th))(
            make_stream({"CH4": 1.0, "H2O": 3.0}, T=1000.0, P=1.5e6))
        for k in a:
            np.testing.assert_allclose(a[k], b[k], rtol=1e-10)

    @pytest.mark.parametrize("Q", [0.0, 1.5e5])
    def test_adiabatic_energy_closes(self, th, Q):
        feed = make_stream({"CH4": 1.0, "H2O": 3.0}, T=1100.0, P=2e6)
        out, info = GibbsReactor(GibbsReactorParams(thermo=th, mode="adiabatic",
                                                    heat_duty=Q))(feed)
        assert bool(info["converged"])
        np.testing.assert_allclose(info["heat_duty"], Q, atol=1e-6)
        names = th.names
        h_in = float(jnp.sum(jnp.array([feed.get(f"F_{s}", 0.0) for s in names])
                             * th.enthalpy(1100.0)))
        h_out = float(jnp.sum(jnp.array([out[f"F_{s}"] for s in names])
                              * th.enthalpy(out["T"])))
        np.testing.assert_allclose(h_out - h_in, Q, atol=1e-6)
        # the adiabatic outlet is the isothermal equilibrium at its own T
        iso, _ = GibbsReactor(GibbsReactorParams(thermo=th, T_out=float(out["T"])))(feed)
        for s in names:
            np.testing.assert_allclose(out[f"F_{s}"], iso[f"F_{s}"], rtol=1e-8)

    def test_overall_approach_is_equilibrium_at_T_plus_dT(self, th):
        feed = make_stream({"CH4": 1.0, "H2O": 3.0}, T=1000.0, P=2e6)
        a, _ = GibbsReactor(GibbsReactorParams(thermo=th, approach_dT=-25.0))(feed)
        b, _ = GibbsReactor(GibbsReactorParams(thermo=th, T_out=975.0))(feed)
        assert float(a["T"]) == 1000.0
        for s in th.names:
            np.testing.assert_allclose(a[f"F_{s}"], b[f"F_{s}"], rtol=1e-8)


class TestReactionSet:
    def test_spanning_reactions_reproduce_gibbs_minimisation(self, th):
        feed = make_stream({"CH4": 1.0, "H2O": 3.0}, T=1000.0, P=2e6)
        a, _ = GibbsReactor(GibbsReactorParams(thermo=th))(feed)
        b, info = GibbsReactor(GibbsReactorParams(thermo=th, reactions=[SMR, WGS]))(feed)
        assert bool(info["converged"])
        for s in th.names:
            np.testing.assert_allclose(a[f"F_{s}"], b[f"F_{s}"], rtol=1e-8)

    def test_wgs_only_leaves_methane_inert(self, th):
        out, info = GibbsReactor(GibbsReactorParams(thermo=th, reactions=[WGS]))(
            make_stream({"CO": 1.0, "H2O": 1.0, "CH4": 0.5}, T=1000.0, P=2e6))
        assert bool(info["converged"])
        assert float(out["F_CH4"]) == 0.5
        np.testing.assert_allclose(quotient(out, WGS), thermo_lnK(th, WGS, 1000.0), atol=1e-9)

    def test_per_reaction_approach(self, th):
        # WGS 30 K short of equilibrium, SMR at equilibrium
        out, info = GibbsReactor(GibbsReactorParams(
            thermo=th, reactions=[SMR, WGS], reaction_approach_dT=[0.0, -30.0]))(
            make_stream({"CH4": 1.0, "H2O": 3.0}, T=1000.0, P=2e6))
        assert bool(info["converged"])
        np.testing.assert_allclose(quotient(out, SMR), thermo_lnK(th, SMR, 1000.0), atol=1e-9)
        np.testing.assert_allclose(quotient(out, WGS), thermo_lnK(th, WGS, 970.0), atol=1e-9)

    def test_approach_list_length_is_checked(self, th):
        with pytest.raises(ValueError, match="one value per reaction"):
            GibbsReactor(GibbsReactorParams(thermo=th, reactions=[SMR, WGS],
                                            reaction_approach_dT=[0.0]))


# --------------------------------------------------------------------------
# gradients: implicit differentiation at the converged point
# --------------------------------------------------------------------------
def _finite(f, x):
    for J in (jax.jacfwd(f)(x), jax.jacrev(f)(x)):
        assert np.all(np.isfinite(np.asarray(J)))


class TestGradients:
    def _h2(self, th, mode="isothermal", **kw):
        def f(x):
            T, P, ch4, h2o = x
            out, _ = GibbsReactor(GibbsReactorParams(thermo=th, mode=mode, **kw))(
                make_stream({"CH4": ch4, "H2O": h2o}, T=T, P=P))
            return jnp.stack([out["F_H2"], out["F_CO"], out["T"]])
        return f

    def test_grads_isothermal(self, th):
        f = self._h2(th)
        x = jnp.array([1000.0, 2e6, 1.0, 3.0])
        _finite(f, x)

    def test_grads_adiabatic(self, th):
        f = self._h2(th, mode="adiabatic")
        x = jnp.array([1100.0, 2e6, 1.0, 3.0])
        _finite(f, x)

    def test_grads_heat_duty(self, th):
        def f(Q):
            out, _ = GibbsReactor(GibbsReactorParams(thermo=th, mode="adiabatic",
                                                     heat_duty=Q))(
                make_stream({"CH4": 1.0, "H2O": 3.0}, T=1100.0, P=2e6))
            return jnp.stack([out["T"], out["F_H2"]])
        _finite(f, 5e4)

    def test_jit(self, th):
        rx = GibbsReactor(GibbsReactorParams(thermo=th))

        @jax.jit
        def f(T):
            return rx(make_stream({"CH4": 1.0, "H2O": 3.0}, T=T, P=2e6))[0]["F_H2"]

        out, _ = rx(make_stream({"CH4": 1.0, "H2O": 3.0}, T=950.0, P=2e6))
        np.testing.assert_allclose(f(950.0), out["F_H2"], rtol=1e-12)
