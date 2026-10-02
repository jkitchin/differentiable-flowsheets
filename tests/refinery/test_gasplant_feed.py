"""gas_plant_feed: crude-unit products onto a gas-plant component table (#326).

The streams here are synthetic (on the example-38 assay's characterization)
so the tests run without a crude-unit solve; examples/38 runs the helper on
the real CDU products.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from difflow_refinery import Assay, characterize
from difflow_refinery.gasplant import evolved_h2s, gas_components, gas_plant_feed

PCT = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
T_C = [-10, 60, 95, 150, 205, 260, 315, 370, 430, 500, 600, 680, 850]
LE = {"ethane": 0.05, "propane": 0.5, "isobutane": 0.3,
      "n_butane": 1.0, "isopentane": 0.8, "n_pentane": 1.5}
LIGHT = ["hydrogen_sulfide", "ethane", "propane", "isobutane", "n_butane",
         "isopentane", "n_pentane"]


@pytest.fixture(scope="module")
def char():
    a = Assay(PCT, [t + 273.15 for t in T_C], sg=0.86, light_ends=LE, sulfur_wt=1.8)
    return characterize(a)


def _products(char, scale=1.0):
    """CDU-like offgas and naphtha: light ends, cuts decaying with Tb, water."""
    npc = len(char.pseudo_names)
    decay = np.exp(-0.9 * np.arange(npc))           # a few cuts fall below 1e-3
    light_off = dict(ethane=0.9, propane=3.0, isobutane=1.5, n_butane=3.5,
                     isopentane=1.0, n_pentane=1.2)
    light_nap = dict(ethane=0.1, propane=4.0, isobutane=4.0, n_butane=14.0,
                     isopentane=12.0, n_pentane=22.0)

    def stream(light, cut_total, cut_decay, water, T):
        s = {f"F_{n}": jnp.asarray(scale * light.get(n, 0.0)) for n in char.light_names}
        for i, n in enumerate(char.pseudo_names):
            s[f"F_{n}"] = jnp.asarray(scale * cut_total * cut_decay[i])
        s["F_water"] = jnp.asarray(water)
        s["T"], s["P"] = jnp.asarray(T), jnp.asarray(1.3e5)
        return s

    return {"offgas": stream(light_off, 0.05, np.exp(-2.0 * np.arange(npc)), 2.0, 313.0),
            "naphtha": stream(light_nap, 120.0, decay, 0.5, 313.0)}


def _old_hand_code(products, char):
    """examples/38's bridge, verbatim except that it returns its streams."""
    off = {k[2:]: float(v) for k, v in products["offgas"].items() if k.startswith("F_")}
    nap = {k[2:]: float(v) for k, v in products["naphtha"].items() if k.startswith("F_")}
    tot = sum(nap.values())
    cuts = [n for n in char.pseudo_names if nap.get(n, 0.0) / tot > 1e-3]
    folded = [n for n in char.pseudo_names if n not in cuts]
    comps = gas_components(LIGHT, pseudo=char, cuts=cuts)

    def stream(flows, T, P):
        s = {f"F_{n}": 0.0 for n in comps.names}
        for k, v in flows.items():
            if k == "water":
                continue
            s[f"F_{cuts[-1] if k in folded else k}"] += v
        s.update(T=T, P=P)
        return s

    H2S = 0.02 * sum(v for k, v in off.items() if k != "water")
    frac = sum(nap.get(n, 0) + off.get(n, 0) for n in folded) / tot
    return comps, cuts, stream(dict(off, hydrogen_sulfide=H2S), 313.15, 1.3e5), \
        stream(nap, 313.15, 1.3e5), frac


@pytest.fixture(scope="module")
def both(char):
    P = _products(char)
    new = gas_plant_feed(P, char, LIGHT, h2s={"offgas": 0.02}, T=313.15, P=1.3e5)
    return P, new, _old_hand_code(P, char)


def test_matches_the_hand_code_of_example_38(both):
    P, new, (comps, cuts, off, nap, frac) = both
    assert new.components.names == comps.names
    assert list(new.cuts) == cuts
    assert 0 < len(new.folded_cuts) < len(cuts) + len(new.folded_cuts)
    for a, b in ((new["offgas"], off), (new["naphtha"], nap)):
        assert set(a) == set(b)
        for k in b:
            assert abs(float(a[k]) - b[k]) <= 1e-12 * max(1.0, abs(b[k])), k
    np.testing.assert_allclose(np.asarray(new.components.MW), np.asarray(comps.MW), rtol=0, atol=0)
    # mass balance, against the hand code's own
    for n, old in (("offgas", off), ("naphtha", nap)):
        m_old = float(sum(old[f"F_{c}"] * m for c, m in zip(comps.names, np.asarray(comps.MW)))) / 1e3
        assert abs(float(new.mass(new[n])) - m_old) <= 1e-12 * m_old
    assert abs(float(new.folded_fraction) - frac) <= 1e-12 * frac


def test_moles_conserved_and_folded_mass_reported(both, char):
    P, new, _ = both
    mw = dict(zip(char.names, np.asarray(char.component_MW)))
    for n in ("offgas", "naphtha"):
        src = {k[2:]: float(v) for k, v in P[n].items() if k.startswith("F_")}
        h2s = float(new.h2s_mol.get(n, 0.0))
        dry = sum(v for k, v in src.items() if k != "water")
        assert abs(float(jnp.sum(new.flows(new[n]))) - (dry + h2s)) <= 1e-12 * dry
        # dropped water is what the stream carried
        assert float(new.dropped_mol[n]["water"]) == src["water"]
        # folded moles and mass are exactly the non-kept cuts
        fm = sum(src[c] for c in new.folded_cuts)
        fk = sum(src[c] * mw[c] for c in new.folded_cuts) / 1e3
        assert fm > 0
        assert abs(float(new.folded_mol[n]) - fm) <= 1e-15 + 1e-12 * fm
        assert abs(float(new.folded_kg[n]) - fk) <= 1e-15 + 1e-12 * fk
        # mass balance: gas-plant mass = source dry mass + H2S + fold mass change
        m_src = sum(v * mw[k] for k, v in src.items() if k != "water") / 1e3
        m_new = float(new.mass(new[n]))
        m_h2s = h2s * float(new.components.MW[new.components.index("hydrogen_sulfide")]) / 1e3
        assert abs(m_new - (m_src + m_h2s + float(new.mass_change[n]))) <= 1e-12 * m_src
    # folded fraction: relative to the naphtha's total (water included)
    nap_tot = sum(float(v) for k, v in P["naphtha"].items() if k.startswith("F_"))
    assert float(new.folded_fraction) == pytest.approx(
        (float(new.folded_mol["offgas"]) + float(new.folded_mol["naphtha"])) / nap_tot, rel=1e-14)
    assert 0 < float(new.folded_fraction_of("naphtha")) < 1e-2
    assert "folded into" in new.summary()


def test_threshold_and_explicit_cuts(char):
    P = _products(char)
    loose = gas_plant_feed(P, char, LIGHT, min_fraction=1e-2)
    tight = gas_plant_feed(P, char, LIGHT, min_fraction=1e-5)
    assert len(loose.cuts) < len(tight.cuts)
    assert float(loose.folded_fraction) > float(tight.folded_fraction)
    same = gas_plant_feed(P, char, LIGHT, cuts=loose.cuts)
    assert same.cuts == loose.cuts and same.components.names == loose.components.names
    # no H2S asked for: none added; streams keep their own T, P
    assert not loose.h2s_mol and float(loose["naphtha"]["T"]) == 313.0


def test_unknown_light_end_raises(char):
    P = _products(char)
    with pytest.raises(ValueError, match="neither in light"):
        gas_plant_feed(P, char, LIGHT[:-1])


def test_differentiable_in_the_flows(char):
    P = _products(char)
    cuts = gas_plant_feed(P, char, LIGHT).cuts

    def f(scale):
        Q = {n: {k: (v * scale if k.startswith("F_") else v) for k, v in s.items()}
             for n, s in P.items()}
        feed = gas_plant_feed(Q, char, LIGHT, cuts=cuts, h2s={"offgas": 0.02})
        return feed.mass(feed["offgas"]) + feed.mass(feed["naphtha"])

    g = jax.grad(f)(1.0)
    assert abs(float(g) - float(f(1.0))) <= 1e-12 * float(f(1.0))    # mass is linear in scale
    assert abs(float(jax.jit(f)(1.0)) - float(f(1.0))) <= 1e-12 * float(f(1.0))
    # selection without cuts= also works under grad (concrete primal)
    assert jnp.isfinite(jax.grad(lambda s: gas_plant_feed(
        {n: {k: v * s for k, v in st.items()} for n, st in P.items()},
        char, LIGHT).mass_change["naphtha"])(1.0))


def test_selection_under_jit_needs_cuts(char):
    P = _products(char)
    with pytest.raises(TypeError, match="cuts="):
        jax.jit(lambda s: gas_plant_feed(
            {n: {k: v * s for k, v in st.items()} for n, st in P.items()},
            char, LIGHT).basis_total)(1.0)


def test_evolved_h2s_is_a_sulfur_balance(char):
    P = _products(char)
    S = np.asarray(char.sulfur)
    mw = np.asarray(char.component_MW)
    s_g = sum(float(P[n][f"F_{c}"]) * m * s for n in P for c, m, s in zip(char.names, mw, S))
    h = evolved_h2s(P, char, 0.01)
    assert float(h) == pytest.approx(0.01 * s_g / 32.065, rel=1e-12)
    feed = gas_plant_feed(P, char, LIGHT[1:], h2s_flow={"offgas": h})
    assert feed.components.names[0] == "hydrogen_sulfide"
    assert float(feed["offgas"]["F_hydrogen_sulfide"]) == pytest.approx(float(h), rel=1e-15)
    no_s = characterize(Assay(PCT, [t + 273.15 for t in T_C], sg=0.86, light_ends=LE))
    with pytest.raises(ValueError, match="sulfur"):
        evolved_h2s(P, no_s, 0.01)
