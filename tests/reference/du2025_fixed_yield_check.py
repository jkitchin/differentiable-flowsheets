"""Du et al. (2025): fixed-yield blocks and the sensitivity ranking (#537).

    PYTHONPATH=src:. python -m pytest tests/reference/du2025_fixed_yield_check.py -v -s

Needs the benchmark data and pounce. Checks:

1. Each block of :data:`~du2025.units.FIXED_YIELD_UNITS` reproduces the
   published outlet flows of every batch, and the transferred properties,
   at the published solution. Its mass balance closes, with the
   benchmark's own loss carried explicitly.
2. The objective's sensitivity to each unit's yields, by the envelope
   theorem at a local optimum, ranks the units. One entry is checked by a
   finite re-solve.
"""

import copy

import jax
import numpy as np
import pytest

from difflow_refinery.benchmarks import du2025
from difflow_refinery.benchmarks.du2025.units import (FIXED_YIELD_UNITS, fixed_yield_block,
                                                      stream_from_solution)
from difflow_refinery.fixed_yield import balance

jax.config.update("jax_enable_x64", True)
CASE = du2025.find_case(1)
pytestmark = pytest.mark.skipif(CASE is None, reason="benchmark data not fetched")


@pytest.fixture(scope="module")
def ctx():
    data = du2025.load_case(CASE)
    path = CASE / "solution_baron.json"
    sol = du2025.load_solution(path if path.exists() else CASE / "solution_baron.gdx")
    return data, sol


def test_blocks_reproduce_published(ctx):
    data, sol = ctx
    for u in FIXED_YIELD_UNITS:
        for m in sorted({m for (uu, m, s) in data.sets["IM"] if uu == u}):
            blk = fixed_yield_block(data, u, m, named=False)
            feeds = {i: stream_from_solution(sol, u, m, i, named=False) for i in blk.inlets}
            if float(sum(f.mass for f in feeds.values())) < 1e-9:
                continue
            out = blk(feeds)
            for o in blk.outlets:
                assert float(out[o].mass) == pytest.approx(sol["FVM"].get((u, m, o, "1"), 0.0),
                                                           rel=1e-6, abs=1e-8), (u, m, o)
                for q, v in out[o].props.items():
                    if q in blk.transfer.get(o, {}) and (o, q) in set(data.sets["SQ"]):
                        assert float(v) == pytest.approx(sol["FQ"][(o, q, "1")], rel=1e-6), (o, q)
            assert abs(float(balance(blk, feeds, out))) < 1e-12
            print(f"{u}.{m} ({FIXED_YIELD_UNITS[u]}): reproduced; loss {blk.loss:.4f}")


@pytest.fixture(scope="module")
def optimum(ctx):
    data, sol = ctx
    model = du2025.build_model(data)
    x, info = du2025.solve_local(model, model.x_from_solution(sol),
                                 options=du2025.WARM_START_OPTIONS)
    return model, x, info


def test_sensitivity_ranking(ctx, optimum):
    from difflow_refinery.benchmarks.du2025.calibration import yield_sensitivity
    model, x, info = optimum
    sens = yield_sensitivity(model, x, info["mult_g"], list(FIXED_YIELD_UNITS) + ["Upf3", "Upf4"])
    rank = sorted(sens.items(), key=lambda kv: -kv[1]["per_wt_pct"])
    print("\nunit    what                                   profit per 1 wt% yield shift")
    for u, s in rank:
        print(f"{u:6s}  {FIXED_YIELD_UNITS.get(u, 'FCC (reference)'):38s} {s['per_wt_pct']:12,.0f}")
    for u, s in rank:
        print(u, {k: round(v) for k, v in s["by_outlet"].items()})


def test_sensitivity_matches_resolve(ctx, optimum):
    """Envelope theorem vs a finite re-solve, on Upf10 (moving 1 wt% of yield)."""
    from difflow_refinery.benchmarks.du2025.calibration import (objective_change,
                                                                yield_sensitivity)
    data, sol = ctx
    model, x, info = optimum
    s = yield_sensitivity(model, x, info["mult_g"], ["Upf10"])["Upf10"]["by_outlet"]
    (m, a), (_, b) = sorted(s)[:2]
    h = 0.01
    batches = sorted({k[0] for k in s})
    new = copy.deepcopy(data)
    for mm in batches:
        new.params["gamma"][("Upf10", mm, a)] += h
        new.params["gamma"][("Upf10", mm, b)] -= h
    # both batches moved: predict with both
    pred = sum((s[(mm, a)] - s[(mm, b)]) * h for mm in batches)
    r = objective_change(data, new, model.x_from_solution(sol))
    actual = r["profit_after"] - r["profit_before"]
    print(f"\nUpf10 +1 wt% {a} / -1 wt% {b}: envelope {pred:,.0f}, re-solve {actual:,.0f}")
    assert actual == pytest.approx(pred, rel=0.05, abs=2e3)
