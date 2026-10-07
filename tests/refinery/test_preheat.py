"""The crude preheat train, desalter and preflash drum in front of the CDU (#313).

Per commit: the train on its own (against hand-checkable balances and a
difflow heat exchanger), the drum's thermodynamics against the column's,
the operations' protocol, registration and JSON round trip, the structural
errors, and one coupled solve on a small column (``slow``).

``release``: the answers on the 30-stage validation column behind an
eight-exchanger train -- convergence on two crudes from the default start,
balances to 1e-8, implicit gradients against central differences, and the
signs a fouling or a drum pressure must have.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import difflow_refinery as dr
from difflow import serialize
from difflow.catalog import describe_class
from difflow.flowsheet import Flowsheet, Unit
from difflow.plugins import OperationRegistry
from difflow.units.heat_exchanger import EnthalpyCounterCurrentHX, EnthalpyHXParams
from difflow_refinery.preheat import ColumnThermoEnthalpy, crude_enthalpy, crude_split
from difflow_refinery.preheat.flash import liquid_enthalpy

from .reference import preheat_case as pc

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def small():
    a, col, tp = pc.small_case()
    crude = dr.characterize(a)
    th = dr.ColumnThermo.from_characterization(crude)
    return a, col, tp, crude, th


def _flows(crude, rate):
    s = crude.stream(rate, T=300.0, P=1e5, basis="mole")
    return jnp.stack([jnp.asarray(s[f"F_{n}"]) for n in crude.names])


def _hot(small):
    """Synthetic hot streams for the train alone: the heavy end of the crude."""
    _, _, _, crude, th = small
    f = _flows(crude, pc.SMALL_RATE)
    heavy = jnp.where(jnp.arange(f.shape[0]) >= f.shape[0] // 2, f, 0.0)
    return {"residue": (0.8 * heavy, jnp.asarray(560.0)), "naphtha": (0.2 * f, jnp.asarray(400.0))}


# =============================================================================
# The train alone
# =============================================================================


@pytest.fixture(scope="module")
def train_solved(small):
    _, _, tp, crude, th = small
    tr = dr.PreheatTrain(tp, th)
    f = _flows(crude, pc.SMALL_RATE)
    return tr, f, tr.solve(f, 300.0, _hot(small))


class TestTrain:
    def test_converges(self, train_solved):
        _, _, r = train_solved
        assert bool(r.converged)
        assert float(r.residual_norm) < 1e-8

    def test_each_exchanger_balances(self, train_solved, small):
        tr, _, r = train_solved
        th = small[4]
        hot = _hot(small)
        for hs in tr.params.hot_streams:
            hf = hot[hs.source][0]
            for name in hs.exchangers:
                e = r.exchangers[name]
                dH = liquid_enthalpy(th, hf, e["T_hot_in"]) - liquid_enthalpy(th, hf, e["T_hot_out"])
                np.testing.assert_allclose(e["Q"], dH, rtol=1e-9)
                ex = tr.params.exchanger(name)
                np.testing.assert_allclose(e["UA"], ex.area / (1.0 / ex.U + ex.Rf), rtol=1e-12)
                assert float(e["approach"]) > 0

    def test_hot_streams_run_hottest_first(self, train_solved):
        _, _, r = train_solved
        # the residue leaves E3 and enters E2 at the same temperature
        np.testing.assert_allclose(r.exchangers["E3"]["T_hot_mixed"], r.exchangers["E2"]["T_hot_in"])

    def test_recovered_is_the_sum_of_duties(self, train_solved):
        _, _, r = train_solved
        np.testing.assert_allclose(r.recovered, sum(e["Q"] for e in r.exchangers.values()), rtol=1e-12)

    def test_fouling_cools_the_furnace_inlet(self, train_solved, small):
        tr, f, _ = train_solved
        tp = small[2]

        def T_out(rf):
            return tr.solve(f, 300.0, _hot(small), params=tp.with_exchanger("E3", Rf=rf)).T_out

        g = jax.grad(T_out)(5e-4)
        h = 1e-5
        fd = (T_out(5e-4 + h) - T_out(5e-4 - h)) / (2 * h)
        assert float(g) < 0
        np.testing.assert_allclose(g, fd, rtol=1e-5)

    def test_bypass_reduces_duty(self, train_solved, small):
        tr, f, r = train_solved
        rb = tr.solve(f, 300.0, _hot(small), params=small[2].with_exchanger("E3", bypass=0.3))
        assert bool(rb.converged)
        assert float(rb.exchangers["E3"]["Q"]) < float(r.exchangers["E3"]["Q"])

    def test_one_exchanger_matches_difflow_enthalpy_hx(self, small):
        """A one-exchanger train is the enthalpy-based exchanger of difflow.units."""
        _, _, _, crude, th = small
        f = _flows(crude, pc.SMALL_RATE)
        hot_f = 0.6 * f
        tp = dr.PreheatTrainParams((dr.PreheatExchanger("E1", 300.0, 200.0),),
                                   (dr.HotStream("residue", ("E1",)),), ("E1",), tank_water=0.0)
        r = dr.PreheatTrain(tp, th).solve(f, 300.0, {"residue": (hot_f, jnp.asarray(450.0))})
        # at 15 bar and these temperatures both sides stay liquid, so flashing
        # the stream (the difflow exchanger) and holding it liquid (the train's
        # hot side) are the same enthalpy
        hx = EnthalpyCounterCurrentHX(EnthalpyHXParams(UA=float(tp.exchangers[0].UA), max_iter=400),
                                      ColumnThermoEnthalpy(th))

        def stream(fl, T):
            return {**{f"F_{n}": fl[i] for i, n in enumerate(th.names)}, "T": T, "P": 15e5}

        _, _, info = hx(stream(hot_f, 450.0), stream(f, 300.0))
        np.testing.assert_allclose(r.exchangers["E1"]["Q"], info["Q"], rtol=1e-6)
        np.testing.assert_allclose(r.T_out, info["T_cold_out"], atol=1e-4)


# =============================================================================
# The drum's thermodynamics are the column's
# =============================================================================


class TestThermo:
    def test_dry_crude_enthalpy_is_the_columns_feed_enthalpy(self, small):
        _, col, _, crude, th = small
        f = _flows(crude, pc.SMALL_RATE)
        column = dr.CrudeColumn(col.update(furnace=dr.Furnace()), th)
        for T, P in ((450.0, 2e5), (560.0, 2e5), (600.0, 1.6e5)):
            args = {"thermo": th, "f": f, "f_water": jnp.asarray(0.0), "T_F": T, "P_F": P}
            E, _ = column._feed_enthalpy(args, T, P)
            np.testing.assert_allclose(crude_enthalpy(th, f, 0.0, T, P), E, rtol=1e-10)

    def test_split_conserves_and_decants_free_water(self, small):
        _, _, _, crude, th = small
        f = _flows(crude, pc.SMALL_RATE)
        sp = crude_split(th, f, 5.0, 380.0, 3e5)
        np.testing.assert_allclose(sp["liquid"] + sp["vapor"], f, rtol=1e-12)
        np.testing.assert_allclose(sp["water_vapor"] + sp["water_liquid"], 5.0, rtol=1e-12)
        assert float(sp["water_liquid"]) > 0  # 380 K at 3 bar: below water's boiling point
        hot = crude_split(th, f, 5.0, 430.0, 3e5)  # above it (Psat ~5.7 bar), all of it flashes
        np.testing.assert_allclose(hot["water_liquid"], 0.0, atol=1e-12)

    def test_vapour_falls_with_pressure(self, small):
        _, _, _, crude, th = small
        f = _flows(crude, pc.SMALL_RATE)
        v = [float(jnp.sum(crude_split(th, f, 1.0, 500.0, P)["vapor"])) for P in (1.5e5, 3e5, 6e5)]
        assert v[0] > v[1] > v[2] > 0


# =============================================================================
# The operations
# =============================================================================


@pytest.fixture(scope="module")
def desalter(small):
    return dr.Desalter(dr.DesalterUnitParams(assay=small[0], desalter=dr.DesalterParams()))


@pytest.fixture(scope="module")
def drum(small):
    return dr.PreflashDrum(dr.PreflashDrumUnitParams(assay=small[0], drum=dr.PreflashDrumParams(P=2.5e5)))


class TestDesalter:
    def test_water_and_energy(self, desalter, small):
        th = desalter.thermo
        feed = desalter.feed(pc.SMALL_RATE, T=410.0, P=10e5)
        out = desalter.solve(feed)
        crude, brine = out["crude"], out["brine"]
        f = jnp.stack([feed[f"F_{n}"] for n in th.names])
        wash = float(out["brine"]["F_water"] + crude["F_water"] - feed["F_water"])
        assert wash > 0
        for n in th.names:
            np.testing.assert_allclose(crude[f"F_{n}"], feed[f"F_{n}"])
        H_in = (crude_enthalpy(th, f, feed["F_water"], 410.0, 10e5)
                + wash * th.water_h_liquid(desalter.params.desalter.T_wash))
        H_out = crude_enthalpy(th, f, crude["F_water"], out["T"], 10e5) + brine["F_water"] * th.water_h_liquid(out["T"])
        np.testing.assert_allclose(H_in, H_out, rtol=1e-10)
        np.testing.assert_allclose(out["margin_low"], out["T"] - 393.15)

    def test_gradient(self, desalter):
        def T(T_in):
            return desalter.solve(desalter.feed(pc.SMALL_RATE, T=T_in, P=10e5))["T"]

        g = jax.grad(T)(410.0)
        fd = (T(410.01) - T(409.99)) / 0.02
        assert 0 < float(g) < 1  # the wash water takes some of the heat
        np.testing.assert_allclose(g, fd, rtol=1e-6)


class TestPreflashDrum:
    def test_adiabatic_flash(self, drum):
        th = drum.thermo
        feed = drum.feed(pc.SMALL_RATE, T=480.0, P=12e5, water=0.002)
        out = drum.solve(feed)
        f = jnp.stack([feed[f"F_{n}"] for n in th.names])
        v, liq, w = out["vapor"], out["liquid"], out["water"]
        for n in th.names:
            np.testing.assert_allclose(v[f"F_{n}"] + liq[f"F_{n}"], feed[f"F_{n}"], rtol=1e-12)
        np.testing.assert_allclose(v["F_water"] + w["F_water"], feed["F_water"], rtol=1e-12)
        np.testing.assert_allclose(crude_enthalpy(th, f, feed["F_water"], out["T"], out["P"]),
                                   crude_enthalpy(th, f, feed["F_water"], 480.0, 12e5), rtol=1e-10)
        assert float(out["T"]) < 480.0 and 0 < float(out["vapor_fraction"]) < 1

    def test_vapor_fraction_mode_finds_the_pressure(self, small, drum):
        op = dr.PreflashDrum(dr.PreflashDrumUnitParams(
            assay=small[0], drum=dr.PreflashDrumParams(P=2.5e5, vapor_fraction=0.1)))
        out = op.solve(drum.feed(pc.SMALL_RATE, T=480.0, P=12e5))
        np.testing.assert_allclose(out["vapor_fraction"], 0.1, atol=1e-10)
        assert 1e5 < float(out["P"]) < 12e5


def _flowsheet(op, feed):
    fs = Flowsheet(list(op.thermo.names) + ["water"])
    fs.add_feed("crude", feed)
    fs.add_unit(Unit("u", op, ["crude"], list(op.outlet_names)))
    return fs


class TestOperations:
    def test_register(self):
        reg = OperationRegistry()
        dr.register(reg)
        ops = reg.list_operations()
        for name in ("Desalter", "PreflashDrum", "CrudeUnitWithPreheat"):
            assert ops[name].cls is getattr(dr, name)
            assert ops[name].category == "refinery"

    @pytest.mark.parametrize("cls,params", [
        (dr.Desalter, "DesalterUnitParams"), (dr.PreflashDrum, "PreflashDrumUnitParams"),
        (dr.CrudeUnitWithPreheat, "CrudeUnitWithPreheatParams")])
    def test_catalog_reads_the_params(self, cls, params):
        spec = describe_class(cls)
        assert spec.params_class == params
        assert all(p.description for p in spec.parameters)
        assert spec.ports.inlets == ["feed"]

    def test_nested_params_are_documented(self):
        from difflow.docstrings import attribute_docs

        for cls in (dr.PreheatExchanger, dr.HotStream, dr.DesalterParams, dr.PreflashDrumParams,
                    dr.PreheatTrainParams, dr.EbertPanchal):
            d = attribute_docs(cls)
            assert all(d.get(f) for f in cls.__dataclass_fields__), cls.__name__

    @pytest.mark.parametrize("which", ["desalter", "drum"])
    def test_json_round_trip(self, which, request):
        op = request.getfixturevalue(which)
        feed = op.feed(pc.SMALL_RATE, T=430.0, P=10e5)
        reg = OperationRegistry()
        dr.register(reg)
        fs = _flowsheet(op, feed)
        streams = fs.solve()
        assert set(op.outlet_names) <= set(streams)
        text = serialize.to_json(fs, registry=reg)
        back = serialize.from_json(text, registry=reg)
        assert type(back.units[0].operation) is type(op)
        assert serialize.to_json(back, registry=reg) == text
        for a, b in zip(back.units[0].operation(feed), op(feed)):
            for k in a:
                np.testing.assert_allclose(a[k], b[k], rtol=1e-12)

    def test_crude_unit_with_preheat_round_trip(self, small):
        a, col, tp, _, _ = small
        op = dr.CrudeUnitWithPreheat(dr.CrudeUnitWithPreheatParams(assay=a, column=col, train=tp))
        assert op.outlet_names == ("naphtha", "residue", "water", "brine", "drum_water")
        reg = OperationRegistry()
        dr.register(reg)
        fs = Flowsheet(list(op.unit.thermo.names) + ["water"])
        fs.add_feed("crude", op.feed(pc.SMALL_RATE, T=300.0, basis="mole"))
        fs.add_unit(Unit("cdu", op, ["crude"], list(op.outlet_names)))
        text = serialize.to_json(fs, registry=reg)
        back = serialize.from_json(text, registry=reg).units[0].operation
        assert back.params.train == tp
        assert back.params.column == col


# =============================================================================
# Structure errors
# =============================================================================


class TestErrors:
    def _train(self, **over):
        kw = dict(exchangers=(dr.PreheatExchanger("E1", 300.0, 50.0), dr.PreheatExchanger("E2", 300.0, 50.0)),
                  hot_streams=(dr.HotStream("residue", ("E2", "E1")),), crude_path=("E1", "E2"))
        kw.update(over)
        return dr.PreheatTrainParams(**kw)

    @pytest.mark.parametrize("over,match", [
        ({"crude_path": ("E1",)}, "not on the crude path"),
        ({"crude_path": ("E1", "E2", "E3")}, "not exchangers"),
        ({"crude_path": ("E1", "desalter", "E2")}, "no DesalterParams"),
        ({"crude_path": ("E1", "preflash", "E2")}, "no PreflashDrumParams"),
        ({"hot_streams": (dr.HotStream("residue", ("E2",)),)}, "on no hot stream"),
        ({"hot_streams": (dr.HotStream("residue", ("E2", "E1")), dr.HotStream("ago", ("E1",)))},
         "on two hot streams"),
    ])
    def test_train_structure(self, small, over, match):
        with pytest.raises(ValueError, match=match):
            dr.PreheatTrain(self._train(**over), small[4])

    def test_desalter_after_drum(self, small):
        with pytest.raises(ValueError, match="before the preflash"):
            dr.PreheatTrain(self._train(crude_path=("E1", "preflash", "desalter", "E2"),
                                        desalter=dr.DesalterParams(), drum=dr.PreflashDrumParams()),
                            small[4])

    def test_pumparound_needs_a_return_temperature(self):
        from .reference import case

        col = case.column_params(0.15)  # pumparounds on duty and delta-T
        tp = self._train(hot_streams=(dr.HotStream("pa1", ("E2", "E1")),))
        with pytest.raises(ValueError, match="pa_return_T"):
            dr.PreheatedCrudeUnit(pc.assay(), col, tp)

    def test_hot_stream_must_be_a_product_or_pumparound(self, small):
        a, col, _, _, _ = small
        with pytest.raises(ValueError, match="neither a product nor a pumparound"):
            dr.PreheatedCrudeUnit(a, col, self._train(hot_streams=(dr.HotStream("kero", ("E2", "E1")),)))

    def test_solve_with_a_different_structure(self, small, train_solved):
        tr, f, _ = train_solved
        other = small[2].update(crude_path=("E1", "desalter", "E3", "E2", "preflash"))
        with pytest.raises(ValueError, match="same structure"):
            tr.solve(f, 300.0, _hot(small), params=other)


# =============================================================================
# Fouling model
# =============================================================================


class TestFouling:
    def test_hot_end_fouls_faster(self, train_solved):
        tr, _, r = train_solved
        # this small train runs below the threshold everywhere: floored at zero
        rates = dr.preheat.fouling_rates(r, tr.params, dr.EbertPanchal())
        assert all(float(v) == 0.0 for v in rates.values())
        # deposition alone goes with the film temperature, hottest last
        dep = dr.preheat.fouling_rates(r, tr.params, dr.EbertPanchal(gamma=0.0))
        assert 0 < float(dep["E1"]) < float(dep["E2"]) < float(dep["E3"])

    def test_threshold(self):
        m = dr.EbertPanchal()
        cold = dr.preheat.fouling_rate(m, 3e4, 10.0, 400.0, 2.0)
        hot = dr.preheat.fouling_rate(m, 3e4, 10.0, 600.0, 2.0)
        assert float(cold) < 0 < float(hot)  # below the threshold, removal wins


# =============================================================================
# Coupled: the small case (per commit, slow)
# =============================================================================


@pytest.fixture(scope="module")
def small_unit(small):
    a, col, tp, _, _ = small
    op = dr.CrudeUnitWithPreheat(dr.CrudeUnitWithPreheatParams(assay=a, column=col, train=tp))
    feed = op.feed(pc.SMALL_RATE, T=300.0, basis="mole")
    return op, feed, op.solve(feed)


@pytest.mark.slow
class TestCoupledSmall:
    def test_converges(self, small_unit):
        _, _, r = small_unit
        assert bool(r.converged) and bool(r.column.converged) and bool(r.train.converged)
        assert float(r.tear_residual) < 1e-6

    def test_balances(self, small_unit):
        op, _, r = small_unit
        for k, (_, _, rel) in op.unit.balances(r).items():
            assert abs(float(rel)) < 1e-8, k

    def test_furnace_sees_the_train(self, small_unit):
        _, _, r = small_unit
        np.testing.assert_allclose(r.furnace_inlet_T, r.train.T_out)
        assert float(r.fired_duty) > float(r.absorbed_duty) > 0

    def test_outlets(self, small_unit):
        op, feed, r = small_unit
        out = op(feed)
        assert len(out) == len(op.outlet_names)
        named = dict(zip(op.outlet_names, out))
        # the residue leaves through the train, not at the column's bottom temperature
        np.testing.assert_allclose(named["residue"]["T"], r.train.hot_outlet_T["residue"])
        np.testing.assert_allclose(named["brine"]["F_water"], r.train.desalter["brine"])

    def test_inlet_water_is_not_discarded(self, small_unit):
        """Audit (2026-10): the operation replaced the stream's water by the
        train's tank_water BS&W, so 0 and 50 mol/s of inlet water gave the
        same outlets and balances() (which recomputed the BS&W) still closed.
        The stream's water now joins the BS&W."""
        op, feed, r = small_unit
        extra = 50.0
        wet = dict(feed, F_water=extra)
        out = op(wet)
        rw = op.last_result
        assert bool(rw.converged)
        named = dict(zip(op.outlet_names, out))
        # The desalter holds the desalted crude's water at its spec, so the
        # extra water leaves in the brine.
        np.testing.assert_allclose(named["brine"]["F_water"],
                                   r.train.desalter["brine"] + extra, rtol=1e-10)
        # Every water outlet, against everything that brought water in.
        p = op.unit.column_params
        steam = float(p.bottom_steam) + sum(float(s.steam) for s in p.side_products
                                            if s.stripper_stages > 0)
        w_in = float(r.feed["F_water"]) + extra + float(rw.train.desalter["wash_water"]) + steam
        w_out = sum(float(s["F_water"]) for s in out)
        assert w_out == pytest.approx(w_in, rel=1e-10)
        b = op.unit.balances(rw)
        assert float(b["water"][0]) == pytest.approx(w_in, rel=1e-10)
        for k, (_, _, rel) in b.items():
            assert abs(float(rel)) < 1e-8, k

    def test_planning_block(self, small_unit):
        from difflow_refinery.column import BARREL
        from difflow_refinery.planning import available_levers, cdu_block

        op, feed, r = small_unit
        lv = available_levers(op.unit)
        assert "preheat.T" not in lv
        assert {"tank.T", "E3.Rf", "E2.area", "E1.bypass", "desalter.wash", "preflash.P"} <= set(lv)
        th = op.unit.thermo
        bpd = float(th.std_volume(jnp.stack([feed[f"F_{n}"] for n in th.names]))) * 86400.0 / BARREL
        blk = cdu_block(op.unit, ["E3.Rf", "preflash.P"], ["furnace.inlet_T", "furnace.fired"],
                        rate=bpd, T=300.0, jit=False)
        np.testing.assert_allclose(blk.u0, [0.5, 2.0])  # m2K/kW and bar
        y = blk.fn(blk.u0)
        np.testing.assert_allclose(y, [r.furnace_inlet_T - 273.15, r.fired_duty / 1e6], rtol=1e-7)


# =============================================================================
# release: the validation column behind an eight-exchanger train
# =============================================================================


@pytest.fixture(scope="module", params=["base", "heavy"])
def coupled(request):
    from .reference.case import BPD

    unit, Vf = pc.preheated_unit(request.param)
    return request.param, unit, unit.solve(BPD, pc.T_TANK)


@pytest.mark.release
@pytest.mark.slow
class TestCoupled:
    def test_converges_from_the_default_start(self, coupled):
        _, unit, r = coupled
        np.testing.assert_allclose(r.furnace_inlet_T, r.train.T_out)
        assert bool(r.converged) and bool(r.column.converged) and bool(r.train.converged)
        assert int(r.iterations) < unit.max_iter
        assert len(unit.train_params.exchangers) == 8

    def test_pumparounds_return_at_the_trains_outlet(self, coupled):
        """The pumparound recycle is closed: the column's return temperature
        is what the train sends back."""
        _, unit, r = coupled
        col = r.column
        for k, pa in enumerate(unit.column_params.pumparounds):
            np.testing.assert_allclose(col.pumparound_return_T[k], r.train.hot_outlet_T[pa.name],
                                       atol=1e-6)

    def test_balances(self, coupled):
        _, unit, r = coupled
        for k, (_, _, rel) in unit.balances(r).items():
            assert abs(float(rel)) < 1e-8, k


@pytest.fixture(scope="module")
def base_unit():
    from .reference.case import BPD

    unit, _ = pc.preheated_unit("base", Rf={"E8": 2e-4})
    return unit, BPD


K_TBP = 5  # the TBP point differentiated against (the 40 % point)


def _outputs(unit, BPD, v):
    """(furnace inlet T, fired duty, preflash vapour) at (E6 area, E8 Rf, drum P, one TBP point)."""
    tp = unit.train_params
    t = tp.with_exchanger("E6", area=v[0]).with_exchanger("E8", Rf=v[1])
    t = t.update(drum=t.drum.update(P=v[2]))
    a = unit.assay
    assay = dr.Assay(a.tbp_percent, jnp.asarray(a.tbp_T).at[K_TBP].set(v[3]), sg=a.sg,
                     light_ends=a.light_ends)
    r = unit.solve(BPD, pc.T_TANK, train=t, assay=assay)
    return jnp.stack([r.furnace_inlet_T, r.fired_duty, r.preflash_vapor])


@pytest.mark.release
@pytest.mark.slow
class TestGradients:
    def test_implicit_gradients_match_central_differences(self, base_unit):
        unit, BPD = base_unit
        v0 = jnp.array([1500.0, 2e-4, pc.DRUM_P, float(unit.assay.tbp_T[K_TBP])])
        J = jax.jacfwd(lambda v: _outputs(unit, BPD, v))(v0)
        steps = [1.0, 1e-6, 100.0, 0.05]
        for i, h in enumerate(steps):
            e = jnp.zeros(4).at[i].set(h)
            fd = (_outputs(unit, BPD, v0 + e) - _outputs(unit, BPD, v0 - e)) / (2 * h)
            np.testing.assert_allclose(J[:, i], fd, rtol=1e-5, atol=1e-9 * jnp.abs(fd).max(),
                                       err_msg=f"lever {i}")
        # signs: fouling cools the furnace inlet and costs fuel; the drum
        # flashes less at a higher pressure
        assert float(J[0, 1]) < 0 < float(J[1, 1])
        assert float(J[2, 2]) < 0

    def test_every_rf_costs_fuel(self, base_unit):
        """Monotone in every exchanger's R_f: a reverse-mode gradient for all
        eight, and a finite fouling of each confirms the sign."""
        unit, BPD = base_unit
        sens = unit.fouling_sensitivity(BPD, pc.T_TANK)
        assert all(float(g) > 0 for g in sens.values()), sens
        base = unit.solve(BPD, pc.T_TANK)
        for e in unit.train_params.exchangers:
            r = unit.solve(BPD, pc.T_TANK, train=unit.train_params.with_exchanger(e.name, Rf=float(e.Rf) + 5e-4))
            assert float(r.furnace_inlet_T) < float(base.furnace_inlet_T), e.name
            assert float(r.fired_duty) > float(base.fired_duty), e.name

    def test_preflash_vapour_falls_with_pressure(self, base_unit):
        unit, BPD = base_unit
        tp = unit.train_params
        v = [float(unit.solve(BPD, pc.T_TANK, train=tp.update(drum=tp.drum.update(P=P))).preflash_vapor)
             for P in (2.0e5, 3.0e5, 4.5e5)]
        assert v[0] > v[1] > v[2] > 0

    def test_cleaning_ranking(self, base_unit):
        unit, BPD = base_unit
        rows = unit.cleaning_ranking(BPD, pc.T_TANK)
        assert rows[0]["name"] == "E8"  # the only fouled exchanger
        assert rows[0]["saving"] > 0
        assert all(abs(r["saving"]) < 1e-3 * rows[0]["saving"] for r in rows[1:])
        # the linear estimate sizes the saving to within the curvature of UA(R_f)
        np.testing.assert_allclose(rows[0]["linear_saving"], rows[0]["saving"], rtol=0.2)
