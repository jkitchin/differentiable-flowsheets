"""Tests for the hydrogen-purification PSA model (#469): difflow_refinery.hydrogen.psa."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from scipy import integrate, special

from difflow_refinery.hydrogen import psa
from difflow_refinery.hydrogen.network import PSA

jax.config.update("jax_enable_x64", True)

F = 1.24e-2           # mol/s, Zhang et al. (2025): 1.24e-5 kmol/s
ZHANG_FEED = {"hydrogen": 0.76 * F, "methane": 0.03 * F, "carbon_monoxide": 0.04 * F,
              "carbon_dioxide": 0.17 * F}


def run(feed=ZHANG_FEED, **kw):
    return psa.HydrogenPSA(psa.HydrogenPSAParams(**kw))(feed)


# --- isotherms (Zhang et al. 2025 Table 2, Park et al. 1998) -----------------

def _pure(i, p_atm):
    x = np.zeros(4)
    x[i] = p_atm
    return jnp.asarray(x)


def test_isotherm_magnitudes_mol_per_g():
    """Pure gases at 1 atm, 293.15 K (mol/kg), the table read as mol/g. Expected
    values recomputed outside the module (plain-math DSL on Table 2)."""
    T = 293.15
    ac = [float(psa.dsl_loading("activated_carbon", _pure(i, 1.0), T)[i]) for i in range(4)]
    z5 = [float(psa.dsl_loading("zeolite_5a", _pure(i, 1.0), T)[i]) for i in range(4)]
    # H2, CH4, CO, CO2
    assert ac == pytest.approx([0.0245, 1.2268, 0.5721, 3.1853], abs=1e-4)
    assert z5 == pytest.approx([0.0206, 0.7459, 2.3555, 3.9514], abs=1e-4)


@pytest.mark.parametrize("adsorbent,order", [
    # Zhang et al. (2025): AC "CO2 ... followed by CH4 and CO, while H2 showed much
    # lower affinity"; 5A "CO2 > CO > CH4 > H2"
    ("activated_carbon", ["carbon_dioxide", "methane", "carbon_monoxide", "hydrogen"]),
    ("zeolite_5a", ["carbon_dioxide", "carbon_monoxide", "methane", "hydrogen"]),
])
@pytest.mark.parametrize("p_atm", [0.1, 1.0, 10.0])
def test_affinity_order_as_published(adsorbent, order, p_atm):
    q = {s: float(psa.dsl_loading(adsorbent, _pure(i, p_atm), 298.15)[i])
         for i, s in enumerate(psa.ADSORBED)}
    assert sorted(q, key=q.get, reverse=True) == order


def test_isotherm_competitive_and_temperature():
    y = jnp.array([0.76, 0.03, 0.04, 0.17]) * 12.0
    for ads in psa.DSL_PARAMETERS:
        mix = psa.dsl_loading(ads, y, 298.15)
        for i in range(4):
            alone = psa.dsl_loading(ads, y * (jnp.arange(4) == i), 298.15)[i]
            assert float(mix[i]) < float(alone)          # competition lowers loading
        for i in range(4):                               # pure gases desorb on heating
            pure = y * (jnp.arange(4) == i)
            assert float(psa.dsl_loading(ads, pure, 330.0)[i]) < float(
                psa.dsl_loading(ads, pure, 298.15)[i])


def test_bed_constants_from_zhang_table1():
    assert psa.ADSORBENT_PROPERTIES == {
        "activated_carbon": {"rho_p": 850.0, "eps_b": 0.433, "eps_t": 0.78},
        "zeolite_5a": {"rho_p": 1160.0, "eps_b": 0.357, "eps_t": 0.77}}
    p = psa.HydrogenPSAParams()
    assert (p.bed_length, p.bed_diameter, p.P_ads, p.t_ads, p.purge_to_feed) == (
        1.68, 0.0384, 12.0e5, 200.0, 0.11)


# --- breakthrough: Klinkenberg vs the exact linear-LDF solution ---------------

def anzelius(xi, tau):
    """Exact c/c0 for a linear isotherm with LDF (Anzelius 1926; Thomas 1944):
    ``J(xi, tau) = 1 - int_0^xi exp(-tau - s) I0(2 sqrt(tau s)) ds``."""
    def f(s):
        z = 2 * np.sqrt(tau * s)
        return np.exp(-tau - s + z) * special.i0e(z)
    val, _ = integrate.quad(f, 0.0, xi, limit=400)
    return 1.0 - val


@pytest.mark.parametrize("xi", [5.0, 20.0, 80.0])
def test_klinkenberg_matches_exact(xi):
    for tau in np.linspace(0.2, 2.5, 12) * xi:
        exact = anzelius(xi, tau)
        assert float(psa.klinkenberg(jnp.asarray(xi), jnp.asarray(tau))) == pytest.approx(
            exact, abs=0.005)


def test_klinkenberg_zero_before_arrival():
    assert float(psa.klinkenberg(jnp.asarray(10.0), jnp.asarray(-1.0))) == 0.0


@pytest.mark.parametrize("ac_fraction,first", [(1.0, "carbon_monoxide"), (0.0, "methane")])
def test_breakthrough_order_zhang_fig4(ac_fraction, first):
    """Zhang et al. Fig. 4 (8 atm): CO breaks first in AC, CH4 first in 5A."""
    t = run(ac_fraction=ac_fraction, P_ads=8 * psa.ATM)["breakthrough_time"]
    other = "methane" if first == "carbon_monoxide" else "carbon_monoxide"
    assert float(t[first]) < float(t[other]) < float(t["carbon_dioxide"])


def test_breakthrough_layered_close():
    """Zhang et al. Fig. 4c: with 0.7 AC, CH4 and CO break 'almost simultaneously'
    -- much closer than in either single bed."""
    gaps = {}
    for f in (1.0, 0.7, 0.0):
        t = run(ac_fraction=f, P_ads=8 * psa.ATM)["breakthrough_time"]
        gaps[f] = abs(float(t["methane"] - t["carbon_monoxide"]))
    assert gaps[0.7] < 0.25 * min(gaps[1.0], gaps[0.0])


# --- purge heel: equilibrium theory -----------------------------------------

def test_heel_linear_isotherm_is_skarstrom(monkeypatch):
    """With a linear isotherm the simple-wave heel is (1 - gamma)+ of the load."""
    monkeypatch.setitem(psa.DSL_PARAMETERS, "linear", {
        s: (1e-3, 1e-12, 0.0, 0.0, 1e-12, 0.0) for s in psa.ADSORBED})
    y = jnp.array([0.7, 0.1, 0.1, 0.1])
    P, T, eps, rho = psa.ATM, 300.0, 0.4, 600.0
    c = y * P / (psa.R * T)
    n = eps * c + rho * psa.dsl_loading("linear", y * P / psa.ATM, T)
    feed_vol = n[1] / c[1]                       # gas volume that loads 1 m3 of bed
    for gamma in (0.3, 0.7, 1.3):
        h = psa.purge_heel("linear", y, P, T, gamma * feed_vol, eps, rho)
        assert float(h[1] / n[1]) == pytest.approx(max(1.0 - gamma, 0.0), abs=0.01)


def test_favourable_isotherm_keeps_heel_beyond_skarstrom():
    out = run(purge_to_feed=0.25)
    assert float(out["gamma"]) > 2.0
    assert all(float(v) > 0.0 for v in out["heel_time"].values())


# --- balances and outputs ---------------------------------------------------

@pytest.mark.parametrize("kw", [{}, dict(t_ads=240.0), dict(purge_to_feed=0.2),
                                dict(n_equalizations=0), dict(n_beds_on_feed=2)])
def test_balances_close(kw):
    feed = {**ZHANG_FEED, "nitrogen": 1e-4, "ethane": 2e-4, "water": 1e-5}
    out = run(feed, **kw)
    for s, b in out["balance"].items():
        assert abs(float(b)) < 1e-15 + 1e-12 * float(feed[s])
    prod = out["product"]
    assert float(prod["ethane"]) == 0.0 and float(prod["water"]) == 0.0
    assert float(prod["nitrogen"]) == pytest.approx(1e-4)
    purity = prod["hydrogen"] / sum(prod.values())
    assert float(out["purity"]) == pytest.approx(float(purity), rel=1e-14)
    assert float(out["recovery"]) == pytest.approx(
        float(prod["hydrogen"] / feed["hydrogen"]), rel=1e-12)


def test_unknown_species_raises():
    with pytest.raises(ValueError, match="no rule"):
        run({**ZHANG_FEED, "ammonia": 1e-5})


def test_recovery_independent_h2_balance():
    """Recovery = 1 - (tail H2)/(feed H2), with the tail H2 recomputed by hand
    from the bed inventory (voids + adsorbed) and the purge."""
    p = psa.HydrogenPSAParams()
    out = psa.HydrogenPSA(p)(ZHANG_FEED)
    y = jnp.array([0.76, 0.03, 0.04, 0.17])
    A = np.pi * 0.0384 ** 2 / 4
    P_eq, P_L, T = 0.5 * (12e5 + psa.ATM), psa.ATM, 298.15
    tail = 0.11 * ZHANG_FEED["hydrogen"] * 200.0
    for name, frac in (("activated_carbon", 0.5), ("zeolite_5a", 0.5)):
        pr = psa.ADSORBENT_PROPERTIES[name]
        V, rho_b, et = A * 1.68 * frac, pr["rho_p"] * (1 - pr["eps_b"]), pr["eps_t"]
        phi = float(out["swept"][name])
        pure = jnp.array([1.0, 0, 0, 0])

        def inv(yy, P):
            return et * V * P * yy[0] / (psa.R * T) + rho_b * V * float(
                psa.dsl_loading(name, yy * P / psa.ATM, T)[0])
        tail += phi * inv(y, P_eq) + (1 - phi) * inv(pure, P_eq) - inv(pure, P_L)
    assert float(out["recovery"]) == pytest.approx(1 - tail / (ZHANG_FEED["hydrogen"] * 200),
                                                   rel=1e-10)


# --- the published trade-offs (Zhang et al. 2025, Table 6) -------------------

def _pr(**kw):
    o = run(**kw)
    return float(o["purity"]), float(o["recovery"])


def test_tradeoff_adsorption_time():
    """t_ads 160 -> 240 s: purity down, recovery up (Table 6)."""
    (p1, r1), (p2, r2), (p3, r3) = (_pr(t_ads=t) for t in (160.0, 200.0, 240.0))
    assert p1 > p2 > p3 and r1 < r2 < r3


def test_tradeoff_purge_to_feed():
    """P/F 0.11 -> 0.25: purity up, recovery down (Table 6)."""
    res = [_pr(purge_to_feed=x) for x in (0.11, 0.15, 0.2, 0.25)]
    assert all(a[0] < b[0] and a[1] > b[1] for a, b in zip(res, res[1:]))


def test_tradeoff_adsorption_pressure():
    """P_H 8 -> 14 bar: purity up, recovery down (Table 6)."""
    res = [_pr(P_ads=P * 1e5) for P in (8.0, 10.0, 12.0, 14.0)]
    assert all(a[0] < b[0] and a[1] > b[1] for a, b in zip(res, res[1:]))


def test_recovery_and_purity_bands():
    """Zhang et al. report 59-65 % recovery (Table 6, three layers with UTSA-16);
    the issue's industrial range is 75-90 %. The two-layer model sits between,
    with purity above 99 %. A band, not a fit: the LDF constants are illustrative."""
    p, r = _pr()
    assert 0.6 < r < 0.85
    assert 0.99 < p < 1.0


def test_equalization_raises_recovery():
    assert _pr(n_equalizations=1)[1] > _pr(n_equalizations=0)[1]


# --- network bridge and gradients --------------------------------------------

def test_from_model_bridge():
    out = run()
    s = PSA.from_model(out)
    assert s.recovery == pytest.approx(float(out["recovery"]))
    assert s.purity == pytest.approx(float(out["purity"]))
    t = PSA.from_model(out, target_purity=0.999)
    assert t.target_purity == 0.999


def test_check_grads_cycle_parameters():
    def f(x):
        p = psa.HydrogenPSAParams(t_ads=x[0], purge_to_feed=x[1], P_ads=x[2], T=x[3])
        o = psa.HydrogenPSA(p)(ZHANG_FEED)
        return jnp.stack([o["purity"], o["recovery"], o["tail_gas"]["hydrogen"]])
    x = jnp.array([200.0, 0.11, 12.0e5, 298.15])
    check_grads(f, (x,), order=1, modes=["fwd", "rev"], eps=1e-4, rtol=1e-3)


def test_grads_feed():
    def f(F_h2):
        o = run({**ZHANG_FEED, "hydrogen": F_h2})
        return o["recovery"]
    check_grads(f, (jnp.asarray(ZHANG_FEED["hydrogen"]),), order=1, modes=["fwd", "rev"],
                eps=1e-6, rtol=1e-3)
