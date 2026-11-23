"""Contract tests for the Du et al. (2025) benchmark (#530).

The benchmark data are not vendored (license), so these run on a tiny
synthetic case in the benchmark's file format
(``tests/refinery/du2025_mini/case1``). Reproducing the published optimum is a
by-hand check: ``tests/reference/du2025_case1_check.py``.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow_refinery.benchmarks.du2025 import blocks as bk
from difflow_refinery.benchmarks.du2025 import (build_model, load_case, parse_gams_text)

jax.config.update("jax_enable_x64", True)

MINI = Path(__file__).parent / "du2025_mini" / "case1"


@pytest.fixture(scope="module")
def model():
    return build_model(load_case(MINI))


def _forward_point(model):
    """A consistent operating point built from the blocks alone."""
    F0, frac9 = 5.0, 0.4
    y = jnp.array([[0.4], [0.5], [0.1]])                 # cuts s1, s2, s3
    merge = jnp.array([[1.0, 0.0, 0.3], [0.0, 1.0, 0.5], [0.0, 0.0, 0.2]])
    spg_cut = jnp.array([[0.7], [0.9], [0.8]])
    q0_cut = jnp.array([[[0.1]], [[1.2]], [[0.5]]])
    f_cdu = bk.swing_cut(jnp.array([F0]), y, merge)
    spg_cdu, q_cdu = bk.cut_properties(jnp.array([F0]), y, merge, spg_cut, q0_cut)
    f45 = bk.fixed_yield(f_cdu[:1], jnp.array([1.0]), jnp.array([0.7, 0.3]))
    f67, g_out = bk.delta_base(f_cdu[1:2], jnp.array([1.0]), jnp.array([0.6, 0.4]),
                               q_cdu[1], jnp.array([1.0]), jnp.zeros((1, 1)),
                               jnp.array([[-0.05], [0.05]]), jnp.array([0.5]))
    q4 = bk.transfer(q_cdu[0, 0], 0.5)
    q6 = bk.transfer(q_cdu[1, 0], 0.8)
    spg4, spg6 = 0.72, 0.88                               # untraced upstream: free
    f8, spg8, q8 = bk.blend(jnp.array([f45[0], f67[0]]), jnp.array([spg4, spg6]),
                            jnp.array([[q4], [q6]]))
    (f9, f10), _ = bk.split(f8, jnp.array([frac9, 1 - frac9]), q8)
    vals = {("F", "s0"): F0, ("F", "s1"): f_cdu[0], ("F", "s2"): f_cdu[1],
            ("F", "s3"): f_cdu[2], ("F", "s4"): f45[0], ("F", "s5"): f45[1],
            ("F", "s6"): f67[0], ("F", "s7"): f67[1], ("F", "s8"): f8,
            ("F", "s9"): f9, ("F", "s10"): f10, ("F", "s11"): f9 + f45[1],
            ("FVM", "UCDU0", "m0", "s0"): F0, ("FVM", "Upf0", "m1", "s1"): f_cdu[0],
            ("FVM", "Upd0", "m2", "s2"): f_cdu[1],
            ("FVM", "Umix0", "m3", "s4"): f45[0], ("FVM", "Umix0", "m3", "s6"): f67[0],
            ("GM", "Upd0", "m2", "s2"): 1.0,
            ("GM", "Upd0", "m2", "s6"): g_out[0], ("GM", "Upd0", "m2", "s7"): g_out[1],
            ("VM", "Umix0", "m3", "s4"): f45[0] / spg4,
            ("VM", "Umix0", "m3", "s6"): f67[0] / spg6,
            ("VM", "Umix0", "m3", "s8"): f8 / spg8,
            ("V", "s9"): f9 / spg8, ("V", "s5"): f45[1] / 0.75}
    for k, (a, b) in {"s1": (0, 0), "s2": (1, 1), "s3": (2, 2)}.items():
        vals[("FQ", k, "Q1")] = spg_cdu[a]
        vals[("FQ", k, "Q0")] = q_cdu[b, 0]
    for (u, m, s) in [("UCDU0", "m0", s) for s in ("s1", "s2", "s3")]:
        vals[("FVM", u, m, s)] = vals[("F", s)]
    vals.update({("FVM", "Upf0", "m1", "s4"): f45[0], ("FVM", "Upf0", "m1", "s5"): f45[1],
                 ("FVM", "Upd0", "m2", "s6"): f67[0], ("FVM", "Upd0", "m2", "s7"): f67[1],
                 ("FVM", "Umix0", "m3", "s8"): f8,
                 ("FQ", "s4", "Q0"): q4, ("FQ", "s4", "Q1"): spg4,
                 ("FQ", "s6", "Q0"): q6, ("FQ", "s6", "Q1"): spg6,
                 ("FQ", "s8", "Q0"): q8[0], ("FQ", "s8", "Q1"): spg8})
    for s in ("s9", "s10"):
        vals[("FQ", s, "Q0")] = q8[0]
        vals[("FQ", s, "Q1")] = spg8
    idx = model.index()
    x = np.zeros(model.n)
    for k, v in vals.items():
        x[idx[k]] = float(v)
    return x


def test_parse_counts():
    data = load_case(MINI)
    c = data.counts()
    assert (c["S"], c["IM"], c["OM"], c["FQcut"], c["RB"]) == (12, 5, 8, 6, 0)
    assert data.param("gamma")[("Upf0", "m1", "s4")] == 0.7
    assert data.elements("SPG") == ("Q1",)


def test_parse_unclosed_raises():
    with pytest.raises(ValueError, match="closing"):
        parse_gams_text("S set\n/\ns0, s1\n")


def test_model_smoke(model):
    x = np.ones(model.n)
    assert np.all(np.isfinite(np.asarray(model.eq(x))))
    assert np.all(np.isfinite(np.asarray(model.ineq(x))))
    assert np.isfinite(float(model.profit(x)))


def test_blocks_satisfy_model(model):
    x = _forward_point(model)
    assert model.violations(x)["eq"] < 1e-12


def test_model_jacobian_finite(model):
    x = _forward_point(model)
    J = jax.jacobian(lambda z: model.eq(z))(jnp.asarray(x))
    assert np.all(np.isfinite(np.asarray(J)))
    pat = model.eq.jacobian_pattern(model.n)
    assert not np.any(np.asarray(J)[~pat])


def test_blocks_grad_finite():
    def f(F):
        out, _ = bk.delta_base(F, jnp.array([1.0]), jnp.array([0.6, 0.4]),
                               jnp.array([0.9]), jnp.array([1.0]), jnp.zeros((1, 1)),
                               jnp.array([[-0.05], [0.05]]), jnp.array([0.5]))
        _, spg, q = bk.blend(out, jnp.array([0.8, 0.9]), jnp.array([[1.0], [2.0]]))
        return spg + q[0]
    assert np.isfinite(float(jax.grad(lambda F: f(F))(jnp.array([2.0]))[0]))
    # an empty blend stays finite (0/0 guarded)
    g = jax.grad(lambda F: bk.blend(F, jnp.array([0.8, 0.9]))[1])(jnp.zeros(2))
    assert np.all(np.isfinite(np.asarray(g)))


def test_multiperiod_raises(tmp_path):
    for f in ("all_sets.txt", "all_parameters.txt"):
        text = (MINI / f).read_text()
        if f == "all_sets.txt":
            text = text.replace("T   set of periods\n/\n1\n/", "T   set of periods\n/\n1, 2\n/")
        (tmp_path / f).write_text(text)
    with pytest.raises(NotImplementedError, match="single-period"):
        build_model(load_case(tmp_path))


def test_inventory_raises(tmp_path):
    for f in ("all_sets.txt", "all_parameters.txt"):
        text = (MINI / f).read_text()
        if f == "all_parameters.txt":
            text = text.replace("LMax(s,t)   maximum inventory level of stream s during period t /  /",
                                "LMax(s,t)   maximum inventory level of stream s during period t\n/\ns3.1=1.0\n/")
        (tmp_path / f).write_text(text)
    with pytest.raises(NotImplementedError, match="LMax"):
        build_model(load_case(tmp_path))


def test_local_solve_smoke(model):
    pytest.importorskip("pounce")
    from difflow_refinery.benchmarks.du2025 import solve_local
    x, info = solve_local(model, _forward_point(model), options={"max_iter": 200})
    assert np.all(np.isfinite(x))
    assert np.isfinite(info["profit"])


# --------------------------------------------------------------------------- #
# #532: property map and reconstructed assays                                 #
# --------------------------------------------------------------------------- #

def test_property_map_contract():
    from difflow_refinery.benchmarks.du2025.properties import PROPERTY_MAP, by_name, describe
    from difflow_refinery.blending import PROPERTY_RULES
    from difflow_refinery.assay import Assay
    import dataclasses
    fields = {f.name for f in dataclasses.fields(Assay)}
    for q, p in PROPERTY_MAP.items():
        assert q == p.index
        assert p.confidence in ("high", "medium", "low", "unknown")
        assert p.blend_key is None or p.blend_key in PROPERTY_RULES
        assert p.assay_field is None or p.assay_field in fields
    assert by_name()["SG"] == "Q168"
    assert describe("Q99").endswith("not used in Case 1")


def test_fit_tbp_mini():
    from difflow_refinery.benchmarks.du2025.assays import crude_assay, cut_tables, fit_tbp
    data = load_case(MINI)
    (table,) = cut_tables(data)
    assert table.cuts == ("s1", "s3", "s2")           # light to heavy by SG
    fit = fit_tbp(data, max_steps=50)
    s = fit.summary()["UCDU0"]
    assert np.isfinite(s["max_yield_err"]) and np.isfinite(s["max_sg_err"])
    assert np.all(np.diff(fit.points[0][1]) > 0)
    assay = crude_assay(fit, 0)
    assert assay.basis == "mass" and assay.heavy_end is not None


def test_crude_cut_table_matches_model(model):
    """#531: the swing-cut block on the published tables gives the CDU's
    outlets at a point the residual model accepts."""
    from difflow_refinery.benchmarks.du2025.crude import cut_table
    from difflow_refinery.swing_cut import swing_cut_block
    table, phi = cut_table(load_case(MINI), "UCDU0")
    x = _forward_point(model)
    idx = model.index()
    F, q = swing_cut_block(table, jnp.array([x[idx[("F", "s0")]]]), jnp.asarray(phi))
    for i, p in enumerate(table.products):
        assert float(F[i]) == pytest.approx(x[idx[("F", p)]], rel=1e-12)
        assert float(q["Q1"][i]) == pytest.approx(x[idx[("FQ", p, "Q1")]], rel=1e-12)


# --------------------------------------------------------------------------- #
# #533: calibration framework                                                 #
# --------------------------------------------------------------------------- #

def test_substitute_identity(model):
    """Writing a unit back with its own published coefficients changes nothing."""
    from difflow_refinery.benchmarks.du2025.calibration import substitute, unit_base
    data = load_case(MINI)
    b = unit_base(data, "Upd0")
    new = substitute(data, b, b.yields, {q: v / b.gamma_in.sum() for q, v in b.delta.items()})
    m1 = build_model(new)
    x = np.linspace(0.5, 1.5, model.n)
    assert m1.var_names == model.var_names
    np.testing.assert_allclose(np.asarray(m1.eq(x)), np.asarray(model.eq(x)), atol=1e-12)


def test_substitute_fixed_yield_becomes_delta_base():
    from difflow_refinery.benchmarks.du2025.calibration import substitute, unit_base
    data = load_case(MINI)
    b = unit_base(data, "Upf0")
    new = substitute(data, b, [0.65, 0.35], {"Q0": [0.01, -0.01]}, {"Q0": 0.3})
    assert "Upf0" in new.elements("UPD") and "Upf0" not in new.elements("UPF")
    assert "Upf0" in data.elements("UPF")                  # the original is untouched
    assert np.all(np.isfinite(np.asarray(build_model(new).eq(np.ones(52)))))


def test_calibrate_and_compare_delta():
    from difflow_refinery.benchmarks.du2025.calibration import calibrate, compare_delta, unit_base
    data = load_case(MINI)
    b = unit_base(data, "Upd0")
    # a toy unit: yields move linearly with sulfur at a tunable rate
    def unit(theta, S):
        y0 = jnp.array([theta[0], 1.0 - theta[0]])
        return y0 + theta[1] * (S - 1.0) * jnp.array([-1.0, 1.0])
    cal = calibrate(lambda th: unit(th, 1.0), [0.5, 0.1], b.yields,
                    lower=[0.0, 0.0], upper=[1.0, 1.0])
    assert cal.within_tol and cal.max_abs_error < 1e-6
    cmp = compare_delta(b, lambda p: unit(jnp.asarray(cal.theta), p[0]))
    assert cmp.summary()["n"] == 2 and cmp.sign_agree.all()


def test_calibrate_bad_start_raises():
    from difflow_refinery.benchmarks.du2025.calibration import calibrate
    with pytest.raises(ValueError, match="strictly inside"):
        calibrate(lambda t: t, [1.0], [0.5], lower=[0.0], upper=[1.0])


def test_fcc_lumping_round_trip():
    """FCC lumps -> benchmark outlets -> lumps is the identity (no FCC solve)."""
    from difflow_refinery.benchmarks.du2025.calibration import UnitBase
    from difflow_refinery.benchmarks.du2025.units import (FCC_OUTLETS, fcc_outlet_yields,
                                                          fcc_targets, is_cracking_batch)
    o = FCC_OUTLETS["Upf3"]
    outs = tuple(o[k] for k in ("acid_gas", "dry_gas", "lpg", "gasoline", "lco", "slurry",
                                "coke", "loss"))
    y = np.array([0.02, 0.03, 0.17, 0.48, 0.19, 0.05, 0.059, 0.001])
    b = UnitBase("Upf3", "m9", ("s27",), outs, np.ones(1), y, y, {}, {})
    assert is_cracking_batch(b)
    lumped = fcc_targets(b)
    np.testing.assert_allclose(fcc_outlet_yields(b, lumped), y, atol=1e-15)
    short = UnitBase("Upf3", "m10", ("s1",), outs[:4], np.ones(1), y[:4], y[:4], {}, {})
    assert not is_cracking_batch(short) and fcc_targets(short)[3] == 0.0
