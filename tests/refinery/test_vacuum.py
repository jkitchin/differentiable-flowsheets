"""Tests for difflow_refinery.vacuum.VacuumColumn.

What the column has to get right, in the order the issue lists it: it
converges from the default initialization on a light and a heavy crude, it
closes the mass balance per pseudocomponent, its specs hold, its cut point
moves the right way with furnace temperature and flash-zone pressure, and
its implicit gradients match central differences.

Every solve here shares one compiled executable per spec structure (the
column caches it on its layout), so the cost is the first compile, not the
number of solves.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow_refinery.vacuum import (
    CrackingWarning,
    StageSpec,
    VacuumColumn,
    VacuumColumnParams,
    atmospheric_residue,
    characterize,
    heavy_crude,
    light_crude,
)

C_TO_K = 273.15
PRODUCTS = ("overhead", "lvgo", "hvgo", "slop", "residue")

pytestmark = pytest.mark.slow


def _setup(crude):
    char = characterize(crude)
    feed = atmospheric_residue(char, crude_rate_kg_s=100.0)
    return char, feed


@pytest.fixture(scope="module", params=["light", "heavy"])
def solved(request):
    crude = light_crude() if request.param == "light" else heavy_crude()
    char, feed = _setup(crude)
    vdu = VacuumColumn(VacuumColumnParams(components=char.components))
    with warnings.catch_warnings():
        warnings.simplefilter("error")       # no convergence/cracking warning
        streams = vdu(feed)
    return char, feed, vdu, streams


@pytest.fixture(scope="module")
def heavy():
    char, feed = _setup(heavy_crude())
    return char, feed, VacuumColumnParams(components=char.components)


def _flows(stream, names):
    return np.array([float(stream[f"F_{n}"]) for n in names])


# -- convergence and balances ----------------------------------------------


class TestSerialization:
    def test_json_round_trip(self, heavy):
        import difflow_refinery as dr
        from difflow import serialize
        from difflow.flowsheet import Flowsheet, Unit
        from difflow.plugins import OperationRegistry

        char, feed, params = heavy
        op = VacuumColumn(params)
        reg = OperationRegistry()
        dr.register(reg)
        fs = Flowsheet([k[2:] for k in feed if k.startswith("F_")])
        fs.add_feed("resid", feed)
        fs.add_unit(Unit("vdu", op, ["resid"], [*PRODUCTS, "info"]))
        text = serialize.to_json(fs, registry=reg)
        back = serialize.from_json(text, registry=reg).units[0].operation
        assert isinstance(back, VacuumColumn)
        assert back.params.specs == op.params.specs
        assert serialize.to_json(serialize.from_json(text, registry=reg), registry=reg) == text


class TestConvergence:
    def test_converges_from_the_default_initialization(self, solved):
        *_, info = solved[3]
        assert bool(info["converged"])
        assert int(info["iterations"]) < 30
        assert float(jnp.max(jnp.abs(info["profiles"]["residual"]))) < 1e-10

    def test_mass_balance_per_pseudocomponent(self, solved):
        char, feed, _, streams = solved
        names = char.components.names
        f = _flows(feed, names)
        out = sum(_flows(s, names) for s in streams[:5])
        np.testing.assert_allclose(out, f, rtol=1e-8)
        assert abs(out.sum() / f.sum() - 1.0) < 1e-8

    def test_a_cut_missing_from_the_feed(self, heavy):
        """Log-form balances with a zero feed flow (floored, not log 0)."""
        char, feed, params = heavy
        names = char.components.names
        feed = dict(feed, **{f"F_{names[0]}": 0.0})
        streams = VacuumColumn(params)(feed)
        assert bool(streams[-1]["converged"])
        out = sum(_flows(s, names) for s in streams[:5])
        np.testing.assert_allclose(out[1:], _flows(feed, names)[1:], rtol=1e-8)
        assert out[0] < 1e-25 * out.sum()

    def test_steam_leaves_overhead(self, solved):
        *_, info = solved[3]
        overhead = solved[3][0]
        steam = info["outputs"]["steam.rate"]
        assert float(overhead["F_H2O"]) * 18.01528e-3 == pytest.approx(float(steam))

    def test_energy_balance_closes_overall(self, solved):
        """Furnace in, pumparounds out, steam in: the products carry the rest."""
        char, feed, vdu, streams = solved
        info = streams[-1]
        stages = info["profiles"]["residual"]
        # the per-stage energy rows are part of the residual; all closed
        assert float(jnp.max(jnp.abs(stages))) < 1e-10


# -- specs -----------------------------------------------------------------


class TestSpecs:
    def test_default_specs_hold(self, solved):
        out = solved[3][-1]["outputs"]
        assert float(out["top.T"]) == pytest.approx(70.0 + C_TO_K, abs=1e-6)
        assert float(out["overflash"]) == pytest.approx(0.03, abs=1e-9)
        assert float(out["lvgo.T95"]) == pytest.approx(450.0 + C_TO_K, abs=1e-6)

    def test_pumparound_rates_hold(self, solved):
        char, feed, vdu, streams = solved
        out = streams[-1]["outputs"]
        F = float(out["feed.rate"])
        assert float(out["lvgo_pa.rate"]) == pytest.approx(0.3 * F, rel=1e-9)
        assert float(out["hvgo_pa.rate"]) == pytest.approx(0.8 * F, rel=1e-9)

    def test_a_spec_can_free_the_furnace(self, heavy):
        """Ask for an HVGO end point; get back the furnace T that gives it."""
        char, feed, params = heavy
        base = VacuumColumn(params)(feed)[-1]["outputs"]
        target = float(base["hvgo.T95"])
        specs = tuple(params.specs) + (
            StageSpec("hvgo.T95", target, replaces="furnace.T"),)
        out = VacuumColumn(params.update(furnace_T=650.0, specs=specs))(feed)[-1]
        assert bool(out["converged"])
        assert float(out["outputs"]["furnace.T"]) == pytest.approx(
            float(params.furnace_T), abs=1e-6)

    def test_fixed_duty_reproduces_the_spec_it_replaced(self, heavy):
        """Fixing a pumparound duty at its solved value gives the same column."""
        char, feed, params = heavy
        out = VacuumColumn(params)(feed)[-1]["outputs"]
        specs = tuple(s for s in params.specs if s.replaces != "lvgo_pa.duty")
        fixed = VacuumColumn(params.update(
            specs=specs, lvgo_pa_duty=float(out["lvgo_pa.duty"])))(feed)[-1]
        assert float(fixed["outputs"]["top.T"]) == pytest.approx(
            float(out["top.T"]), abs=1e-6)

    def test_bad_specs_are_rejected(self, heavy):
        char, feed, params = heavy
        with pytest.raises(ValueError, match="neither a draw rate"):
            VacuumColumn(params.update(specs=(StageSpec("top.T", 340.0, replaces="nope"),)))
        with pytest.raises(ValueError, match="no value and no spec"):
            VacuumColumn(params.update(specs=()))


# -- physics -----------------------------------------------------------------


class TestPhysics:
    def test_products_are_ordered_by_boiling_range(self, solved):
        p = solved[3][-1]["properties"]
        t50 = [float(p[n]["T50"]) for n in ("lvgo", "hvgo", "slop", "residue")]
        assert t50 == sorted(t50)

    def test_temperature_profile(self, solved):
        info = solved[3][-1]
        out = info["outputs"]
        T = np.asarray(info["profiles"]["T"])
        assert float(out["top.T"]) < float(out["lvgo.draw_T"]) < float(out["hvgo.draw_T"])
        assert float(out["hvgo.draw_T"]) < float(out["flash_zone.T"])
        assert float(out["flash_zone.T"]) < float(out["furnace.T"])
        # stripping cools the residue (steam takes the heat of vaporization)
        assert T[-1] < float(out["flash_zone.T"])

    def test_vgo_is_cleaner_than_residue(self, solved):
        p = solved[3][-1]["properties"]
        for key in ("ccr_wt", "nickel_vanadium_wppm", "sulfur_wt", "sg"):
            assert float(p["hvgo"][key]) < float(p["residue"][key])
        assert float(p["lvgo"]["ccr_wt"]) < float(p["hvgo"]["ccr_wt"])

    def test_entrainment_carries_metals_into_hvgo(self, heavy):
        char, feed, params = heavy
        clean = VacuumColumn(params.update(entrainment=0.0))(feed)[-1]["properties"]
        dirty = VacuumColumn(params.update(entrainment=0.05,
                                           deentrainment=0.5))(feed)[-1]["properties"]
        assert (float(dirty["hvgo"]["nickel_vanadium_wppm"])
                > 2.0 * float(clean["hvgo"]["nickel_vanadium_wppm"]))

    def test_stage_efficiency_blurs_the_cut(self, heavy):
        char, feed, params = heavy
        sharp = VacuumColumn(params)(feed)[-1]["properties"]
        soft = VacuumColumn(params.update(wash_efficiency=0.5,
                                          strip_efficiency=0.5))(feed)[-1]
        assert bool(soft["converged"])
        # a poorer wash bed lets more heavy material into HVGO
        assert float(soft["properties"]["hvgo"]["T95"]) > float(sharp["hvgo"]["T95"])
        assert float(soft["properties"]["hvgo"]["ccr_wt"]) > float(sharp["hvgo"]["ccr_wt"])

    def test_an_end_point_the_beds_cannot_make_is_infeasible(self, heavy):
        """An HVGO bed at 70% passes 9% of the heavy vapor to the LVGO
        section, so a 450 C LVGO end point cannot be met at any duty; it can
        once the spec allows for it."""
        char, feed, params = heavy
        with pytest.warns(Warning, match="did not converge"):
            bad = VacuumColumn(params.update(hvgo_efficiency=0.7))(feed)[-1]
        assert not bool(bad["converged"])
        specs = tuple(StageSpec(s.output, 520.0 + C_TO_K, s.replaces)
                      if s.output == "lvgo.T95" else s for s in params.specs)
        ok = VacuumColumn(params.update(hvgo_efficiency=0.7, specs=specs))(feed)[-1]
        assert bool(ok["converged"])

    def test_cracking_limit_is_reported_not_imposed(self, heavy):
        char, feed, params = heavy
        with pytest.warns(CrackingWarning):
            info = VacuumColumn(params.update(furnace_T=425.0 + C_TO_K))(feed)[-1]
        assert bool(info["converged"])
        assert float(info["furnace"]["cracking_margin"]) < 0


class TestCutPoint:
    """The VGO/residue cut moves monotonically with furnace T and pressure."""

    def test_furnace_temperature(self, heavy):
        char, feed, params = heavy
        Ts = [385.0, 395.0, 405.0, 415.0]
        outs = [VacuumColumn(params.update(furnace_T=t + C_TO_K))(feed)[-1]["outputs"]
                for t in Ts]
        t95 = [float(o["hvgo.T95"]) for o in outs]
        resid = [float(o["residue.yield"]) for o in outs]
        assert np.all(np.diff(t95) > 0)
        assert np.all(np.diff(resid) < 0)

    def test_flash_zone_pressure(self, heavy):
        char, feed, params = heavy
        Ps = [20.0, 30.0, 45.0, 60.0]
        outs = [VacuumColumn(params.update(flash_zone_P=p * 133.322368))(feed)[-1]["outputs"]
                for p in Ps]
        t95 = [float(o["hvgo.T95"]) for o in outs]
        resid = [float(o["residue.yield"]) for o in outs]
        assert np.all(np.diff(t95) < 0)
        assert np.all(np.diff(resid) > 0)


# -- gradients ---------------------------------------------------------------


def _quantities(char, feed, params):
    info = VacuumColumn(params)(feed)[-1]
    out, hv = info["outputs"], info["properties"]["hvgo"]
    return jnp.stack([out["lvgo.rate"], out["hvgo.rate"], out["residue.rate"],
                      hv["T95"], hv["sg"], hv["sulfur_wt"], hv["nitrogen_wppm"],
                      hv["ccr_wt"]])


@pytest.mark.release
class TestGradients:
    """Implicit gradients against central differences (about 1e-5 relative)."""

    @pytest.mark.parametrize("field,x0,h", [
        ("furnace_T", 400.0 + C_TO_K, 0.02),
        ("flash_zone_P", 4000.0, 1.0),
        ("steam_rate", 0.33, 1e-4),
    ])
    def test_operating_levers(self, heavy, field, x0, h):
        char, feed, params = heavy

        def q(x):
            return _quantities(char, feed, params.update(**{field: x}))

        g = jax.jacfwd(q)(x0)
        fd = (q(x0 + h) - q(x0 - h)) / (2 * h)
        np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-12)

    def test_a_tbp_point_of_the_assay(self):
        crude = heavy_crude()

        def q(t):
            char = characterize(crude.with_tbp_point(9, t))    # the 500 C point
            feed = atmospheric_residue(char, crude_rate_kg_s=100.0)
            return _quantities(char, feed, VacuumColumnParams(components=char.components))

        t0, h = 500.0, 0.02
        g = jax.jacfwd(q)(t0)
        fd = (q(t0 + h) - q(t0 - h)) / (2 * h)
        np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-12)


def test_gradients_flow_through_the_solve(heavy):
    """Per-commit wiring check; the accuracy claim is TestGradients."""
    char, feed, params = heavy

    def hvgo_rate(T):
        return VacuumColumn(params.update(furnace_T=T))(feed)[-1]["outputs"]["hvgo.rate"]

    g = jax.grad(hvgo_rate)(400.0 + C_TO_K)
    assert np.isfinite(float(g)) and float(g) > 0      # hotter furnace, more HVGO
