"""Alkylation unit (#310): smoke, validation and API tests."""

import jax
import numpy as np
import pytest

import difflow_refinery.alkylation as al

jax.config.update("jax_enable_x64", True)


def _fed(io_ratio=8.0):
    mix = al.IsobutaneMakeup(al.IsobutaneMakeupParams(io_ratio=io_ratio))
    fed, _ = mix(al.c4_olefin_feed(), al.feed_stream({}, 311.0, 6e5))
    return fed


def test_reactor_smoke():
    fed = _fed()
    eff, info = al.AlkylationReactor()(fed)
    assert all(np.isfinite(float(v)) for v in eff.values())
    assert np.isfinite(float(info["MON"])) and np.isfinite(float(info["alkylate_yield"]))
    for k, (a, b) in al.atom_balance([fed], [eff]).items():
        assert float(b) == pytest.approx(float(a), rel=1e-12), k


def test_reactor_grad_finite():
    fed = _fed()

    def mon(s):
        return al.AlkylationReactor()(dict(fed, F_isobutane=fed["F_isobutane"] * s))[1]["MON"]

    assert np.isfinite(float(jax.grad(mon)(1.0)))


def test_hf_needs_its_consumption():
    with pytest.raises(ValueError, match="hf_acid_lb_per_bbl"):
        al.AlkylationReactor(al.AlkylationReactorParams(acid="HF"))


def test_bad_selectivity_is_refused():
    sel = dict(al.DEFAULT_SELECTIVITY, propylene={"2_2_4_trimethylpentane": 1.0})
    with pytest.raises(ValueError, match="not an alkylation product"):
        al.AlkylationReactor(al.AlkylationReactorParams(selectivity=sel))


def test_range_warning():
    with pytest.warns(al.AlkylationRangeWarning, match="I/O ratio"):
        al.AlkylationReactor()(_fed(io_ratio=2.0))


def test_column_smoke():
    eff, _ = al.AlkylationReactor()(_fed())
    col = al.AlkylationUnit().columns["dib"]
    D, B, info = col(eff)
    assert np.isfinite(float(D["F_isobutane"])) and np.isfinite(float(B["F_isobutane"]))
    assert bool(info["feasible"])


def test_unit_solves():
    r = al.AlkylationUnit().solve(al.c4_olefin_feed())
    assert bool(r.converged)
    assert all(np.isfinite(float(v)) for v in r.outputs.values())


def test_alky_block_builds():
    unit = al.AlkylationUnit()
    blk = al.alky_block(unit, al.c4_olefin_feed(), levers=["io_ratio", "reactor.T", "olefin.bpd"],
                        bounds={"io_ratio": (6.0, 10.0)})
    assert blk.u_names == ["io_ratio", "reactor.T", "olefin.bpd"]
    assert float(blk.lb[0]) == 6.0 and float(blk.ub[0]) == 10.0
    with pytest.raises(KeyError):
        al.alky_block(unit, al.c4_olefin_feed(), levers=["nonsense"])
