"""Du et al. (2025) benchmark: existing difflow units calibrated in (#533).

Run by hand from the repository root, with the benchmark data fetched
(``du2025.fetch()``) and pounce installed::

    PYTHONPATH=src:. python -m pytest tests/reference/du2025_calibration_check.py -v -s

It prints the numbers quoted in the docs ("Calibrating difflow units"):

1. FCC (``Upf3.m9``, ``Upf4.m13``) calibrated to the benchmark lumps:
   every lump within 1 wt% (``Upf3``'s co-processed batches ``m11``/``m12``
   are reported, not required: their coke yields are below what a
   heat-balanced riser makes).
2. Mass, element (C, H, S, N) and energy balances close at the calibrated
   points; the calibrated FCC Jacobian with respect to feed SG and sulfur
   matches central differences.
3. Both FCCs substituted as their own delta-base linearization; the
   benchmark re-solved: the profit change and the limits that change status.
4. The published gasoline blends under difflow's nonlinear rules (RT-70,
   volume-based aromatics and olefins) against the benchmark's mass-linear
   rule: the specs still hold.

The hydrotreater calibration (``Upd0``/``Upd1``) is a longer run (compiles
for several minutes) kept in ``tests/reference/du2025_hdt_check.py``.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow_refinery.benchmarks import du2025

jax.config.update("jax_enable_x64", True)
CASE = du2025.find_case(1)
pytestmark = pytest.mark.skipif(CASE is None, reason="benchmark data not fetched")
BATCHES = ("Upf3.m9", "Upf4.m13", "Upf3.m11", "Upf3.m12")
REQUIRED = ("Upf3.m9", "Upf4.m13")


@pytest.fixture(scope="module")
def ctx():
    import difflow_refinery as dr
    from difflow_refinery.fcc import FCCFeed, FCCParams, FCCUnit
    from difflow_refinery.vacuum import assay as va
    data = du2025.load_case(CASE)
    path = CASE / "solution_baron.json"
    sol = du2025.load_solution(path if path.exists() else CASE / "solution_baron.gdx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        char = dr.characterize(va.light_crude().to_assay(heavy_end=dr.HeavyEnd()))
    base_feed = FCCFeed.from_characterization(char, 50.0, 343 + 273.15, 550 + 273.15)
    return data, sol, FCCUnit(FCCParams()), base_feed


@pytest.fixture(scope="module")
def fcc(ctx):
    from difflow_refinery.benchmarks.du2025.calibration import calibrate, unit_base
    from difflow_refinery.benchmarks.du2025.units import (FCC_KNOBS, fcc_feed, fcc_predict,
                                                          fcc_targets)
    data, sol, unit, base_feed = ctx
    lo, hi, th0 = (np.array([k[i] for k in FCC_KNOBS]) for i in (1, 2, 3))
    out = {}
    for key in BATCHES:
        u, m = key.split(".")
        b = unit_base(data, u, m, solution=sol)
        p = dict(b.props)
        assumed = p.get("Q168", 0.0) < 0.6          # an unconstrained published SG
        if assumed:
            p["Q168"] = 0.92
        f = fcc_feed(base_feed, sg=p["Q168"], sulfur_wt=p.get("Q0", 0.5),
                     ccr_wt=p.get("Q21", 0.0), nickel_vanadium_wppm=p.get("Q25", 0.0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cal = calibrate(lambda th, f=f: fcc_predict(unit, f, th, with_regenerator=True),
                            th0, np.append(fcc_targets(b), 0.0), lower=lo, upper=hi,
                            n_yields=6)
        o = _outputs(unit, f, cal.theta)
        out[key] = (b, p, cal, assumed)
        print(f"\n{key}: max |error| {cal.max_abs_error:.4f} (SG assumed: {assumed}); "
              f"regenerator {float(o['regenerator_T']) - 273.15:.0f} C, "
              f"cat/oil {float(o['cat_oil']):.2f}, converged {bool(o['converged'])}")
        print("  knobs ", dict(zip([k[0] for k in FCC_KNOBS], np.round(cal.theta, 3))))
        print("  pred  ", np.round(cal.predicted[:6], 4))
        print("  target", np.round(cal.target[:6], 4))
    return out


def _outputs(unit, f, theta):
    from difflow_refinery.benchmarks.du2025.units import _rate_multipliers
    a, rot, gcut, lcut, mdg, mck, tf = theta
    k = unit.theta(f)["params"]["k_ref"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = unit.solve(f, activity=a, riser_outlet_T=rot, gasoline_cut=gcut, lco_cut=lcut,
                       feed_T=tf, k_ref=k * _rate_multipliers(unit, mdg, mck, k.shape[0],
                                                   unit.scheme.reactions))
    return {**r["outputs"], "converged": r["converged"], "balances": r["balances"]}


def test_fcc_calibrated(fcc):
    for key in REQUIRED:
        assert fcc[key][2].within_tol, (key, fcc[key][2].max_abs_error)


def test_fcc_balances_at_calibration(ctx, fcc):
    """Mass, C, H, S, N and energy close at the calibrated point, as the FCC's
    own tests require of any point (the calibration moves knobs, not balances)."""
    from difflow_refinery.benchmarks.du2025.units import fcc_feed
    _, _, unit, base_feed = ctx
    for key in REQUIRED:
        _, p, cal, _ = fcc[key]
        f = fcc_feed(base_feed, sg=p["Q168"], sulfur_wt=p["Q0"], ccr_wt=p.get("Q21", 0.0),
                     nickel_vanadium_wppm=p.get("Q25", 0.0))
        bal = _outputs(unit, f, cal.theta)["balances"]
        print(key, {k: f"{float(v):.1e}" for k, v in bal.items()})
        assert all(abs(float(v)) < 1e-8 for v in bal.values())


def test_fcc_feed_jacobian_fd(ctx, fcc):
    from difflow_refinery.benchmarks.du2025.units import fcc_feed, fcc_predict
    _, _, unit, base_feed = ctx
    b, p, cal, _ = fcc["Upf3.m9"]
    th = jnp.asarray(cal.theta)
    f = lambda q: fcc_predict(unit, fcc_feed(base_feed, sg=q[0], sulfur_wt=q[1],  # noqa: E731
                                             ccr_wt=p.get("Q21", 0.0)), th)
    q0 = jnp.asarray([p["Q168"], p["Q0"]])
    J = np.asarray(jax.jacfwd(f)(q0))
    for i, h in enumerate((1e-4, 1e-4)):
        e = jnp.zeros(2).at[i].set(h)
        fd = (np.asarray(f(q0 + e)) - np.asarray(f(q0 - e))) / (2 * h)
        np.testing.assert_allclose(J[:, i], fd, rtol=1e-3, atol=1e-6)


def test_fcc_substituted_objective(ctx, fcc):
    from difflow_refinery.benchmarks.du2025.calibration import (objective_change, substitute,
                                                                unit_base)
    from difflow_refinery.benchmarks.du2025.units import fcc_feed, fcc_outlet_yields, fcc_predict
    data, sol, unit, base_feed = ctx
    new = data
    for key in REQUIRED:
        b, p, cal, _ = fcc[key]
        th = jnp.asarray(cal.theta)
        f = lambda q: fcc_predict(unit, fcc_feed(base_feed, sg=q[0], sulfur_wt=q[1],  # noqa: E731
                                                 ccr_wt=p.get("Q21", 0.0),
                                                 nickel_vanadium_wppm=p.get("Q25", 0.0)), th)
        q0 = jnp.asarray([p["Q168"], p["Q0"]])
        J = np.asarray(jax.jacfwd(f)(q0))
        zero = fcc_outlet_yields(b, np.zeros(6))
        dy = {q: fcc_outlet_yields(b, J[:, i]) - zero for i, q in enumerate(("Q168", "Q0"))}
        print(f"\n{key} dY/dSG", dict(zip(b.outlets, np.round(dy["Q168"], 3))))
        print(f"{key} dY/dS ", dict(zip(b.outlets, np.round(dy["Q0"], 4))))
        new = substitute(new, unit_base(new, *key.split("."), solution=sol),
                         fcc_outlet_yields(b, np.asarray(f(q0))), dy,
                         {"Q168": p["Q168"], "Q0": p["Q0"]})
    x0 = du2025.build_model(data).x_from_solution(sol)
    r = objective_change(data, new, x0)
    print(f"\nprofit {r['profit_before']:,.0f} -> {r['profit_after']:,.0f} "
          f"({r['status_after']}); changed: {r['changed']}")
    assert r["status_after"] == "Solve_Succeeded" and r["violations_after"]["eq"] < 1e-6


def test_gasoline_blends_nonlinear(ctx):
    from difflow_refinery.blending import BlendComponent, BlendPool
    data, sol, _, _ = ctx
    FQ, F = sol["FQ"], sol["FVO"]
    props = {"RON": ("Q2", 1), "MON": ("Q4", 1), "olefins_vol": ("Q51", 1),
             "aromatics_vol": ("Q11", 1), "benzene_vol": ("Q13", 1), "S_ppm": ("Q0", 1e4)}
    qmin, qmax = data.param("FQBMin"), data.param("FQBMax")
    for u in ("Ublend3", "Ublend4", "Ublend5"):
        ins = [s for (uu, s) in data.sets["IU"] if uu == u and F.get((s, "1"), 0) > 1e-9]
        comps, mass = [], []
        for s in ins:
            sg = FQ.get((s, "Q168", "1"), 0.0)
            sg = sg if sg >= 0.5 else 0.73       # an unconstrained published SG: typical
            comps.append(BlendComponent.from_properties(
                s, SG=sg, **{k: FQ.get((s, q, "1"), 0.0) * f for k, (q, f) in props.items()}))
            mass.append(F[(s, "1")])
        vol = np.array(mass) / np.array([float(c.properties["SG"]) for c in comps])
        nl = BlendPool("gasoline", specs=[]).blend_properties(comps, jnp.asarray(vol))
        for k, (q, f) in props.items():
            lin = sum(m * FQ.get((s, q, "1"), 0.0) for m, s in zip(mass, ins)) / sum(mass) * f
            lo, hi = qmin.get((u, q)), qmax.get((u, q))
            print(f"{u} {k:14s} benchmark {lin:8.3f}  nonlinear {float(nl[k]):8.3f}  "
                  f"spec {lo and lo * f} .. {hi and hi * f}")
            if lo is not None:
                assert float(nl[k]) >= lo * f - 1e-6
            if hi is not None:
                assert float(nl[k]) <= hi * f + 1e-6
