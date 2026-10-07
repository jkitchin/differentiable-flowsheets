"""Regression tests for the carbon-capture audit: membranes (C5, (e))."""

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from difflow.streams import get_flows, make_stream, total_flow
from difflow_cc import MembraneParams, MembraneSeparator, list_membranes
from difflow_cc.units.membrane import MultistageMembrane


def _feed(P=1e6):
    return make_stream({"CO2": 1.0, "N2": 9.0}, T=298.15, P=P)


def _partial_pressures(retentate, permeate, sp):
    r, p = get_flows(retentate), get_flows(permeate)
    x = float(r[sp]) / float(total_flow(retentate))
    y = float(p[sp]) / float(total_flow(permeate))
    return x * float(retentate["P"]), y * float(permeate["P"])


@pytest.mark.parametrize("membrane", list_membranes())
def test_driving_force_never_reversed(membrane):
    """C5: Matrimid gave 1.16 kPa CO2 retentate vs 72 kPa permeate."""
    ret, perm, info = MembraneSeparator(MembraneParams(membrane_type=membrane))(_feed())
    if float(info["stage_cut"]) > 1.0 - 1e-9:
        return  # total permeation, no retentate composition to compare
    for sp in ("CO2", "N2"):
        p_ret, p_perm = _partial_pressures(ret, perm, sp)
        assert p_perm <= p_ret * (1 + 1e-12), (membrane, sp, p_ret, p_perm)


def test_feed_stream_pressure_used():
    """C5: a 2 bar feed separated exactly like a 10 bar one."""
    m = MembraneSeparator(MembraneParams(membrane_type="Matrimid"))
    lo = m(_feed(2e5))
    hi = m(_feed(1e6))
    assert float(lo[0]["P"]) == 2e5
    assert float(lo[2]["CO2_recovery"]) < float(hi[2]["CO2_recovery"])


def test_stage_cut_target_sets_area():
    """C5: stage_cut_target was ignored."""
    ret, perm, info = MembraneSeparator(
        MembraneParams(membrane_type="Matrimid", stage_cut_target=0.2))(_feed())
    assert float(info["stage_cut"]) == pytest.approx(0.2, rel=1e-9)
    assert float(info["area_used"]) != 1000.0
    # Feeding the solved area back reproduces the cut.
    _, _, again = MembraneSeparator(
        MembraneParams(membrane_type="Matrimid", area=info["area_used"]))(_feed())
    assert float(again["stage_cut"]) == pytest.approx(0.2, rel=1e-9)


@pytest.mark.parametrize("ratio", [1.0, 0.5])
def test_pressure_ratio_not_above_one_rejected(ratio):
    with pytest.raises(ValueError, match="pressure_ratio"):
        MembraneParams(membrane_type="Matrimid", pressure_ratio=ratio)


def test_permeate_pressure_above_feed_rejected():
    m = MembraneSeparator(MembraneParams(membrane_type="Matrimid"))
    with pytest.raises(ValueError, match="below the feed"):
        m(_feed(1e5), P_permeate=2e5)


def test_area_gradient_matches_fd():
    def rec(a):
        p = MembraneParams(membrane_type="Matrimid", area=a)
        return MembraneSeparator(p)(_feed())[2]["CO2_recovery"]
    g = jax.grad(rec)(1000.0)
    fd = (rec(1000.01) - rec(999.99)) / 0.02
    assert float(g) > 0.0
    assert float(g) == pytest.approx(float(fd), rel=1e-5)


def test_required_area_inverts_model():
    m = MembraneSeparator(MembraneParams(membrane_type="Matrimid"))
    A = m.required_area(_feed(), 0.5)
    _, _, info = MembraneSeparator(MembraneParams(membrane_type="Matrimid", area=A))(_feed())
    assert float(info["CO2_recovery"]) == pytest.approx(0.5, rel=1e-9)


class TestPermeateRecycle:
    """(e): the recycle configuration ran 2 stages whatever was asked."""

    def test_three_stages_rejected(self):
        with pytest.raises(ValueError, match="two-stage"):
            MultistageMembrane(MembraneParams(membrane_type="Matrimid"),
                               n_stages=3, configuration="permeate_recycle")

    def test_recycle_converged_and_balanced(self):
        feed = _feed()
        cas = MultistageMembrane(MembraneParams(membrane_type="Matrimid", area=500.0),
                                 n_stages=2, configuration="permeate_recycle")
        ret, perm, info = cas(feed)
        assert info["n_stages"] == 2
        assert float(info["recycle_flow"]) > 0.0
        assert float(info["recycle_residual"]) < 1e-8
        for sp, f in get_flows(feed).items():
            assert float(get_flows(ret)[sp] + get_flows(perm)[sp]) == pytest.approx(float(f), rel=1e-8)
        # Stage 2 enriches: product purer than a single stage.
        single = MembraneSeparator(MembraneParams(membrane_type="Matrimid", area=500.0))(feed)[2]
        assert float(info["overall_CO2_purity"]) > float(single["CO2_purity"])
