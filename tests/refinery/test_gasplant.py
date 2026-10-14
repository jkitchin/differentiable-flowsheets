"""The saturated gas plant (#312): columns, compressor, amine, products, block.

One small debutanizer (balances, specs, products, the planning block's wiring,
implicit gradients against central differences) and the cheap units. Every
column solve here is a compile of a minute or more, so each other factory is
not solved: the absorber-deethanizer, C3/C4 splitter and deisobutanizer
from default initialization, the lean-oil deethanizer and the
monotonicity sweeps were cut, and the one regression among them that has a
cheap form -- the deisobutanizer's initial guess must not depend on the
round-off of the feed -- stays.

The IDAES cross-check is in ``test_gasplant_validation.py`` and
``test_gasplant_validation_file.py``; DWSIM's in ``test_dwsim_gasplant.py``.
"""

from __future__ import annotations


import jax
import jax.numpy as jnp
import numpy as np
import pytest

import difflow_refinery as dr
from difflow_refinery import Assay, characterize
from difflow_refinery.gasplant import (
    GPA_2140,
    AmineTreater,
    AmineTreaterParams,
    GasCompressor,
    GasCompressorParams,
    GasPlantColumn,
    GasPlantColumnParams,
    absorber_deethanizer,
    c3c4_splitter,
    column_layout,
    debutanizer,
    deisobutanizer,
    fuel_gas,
    gas_components,
    gasplant_block,
    lpg_quality,
    oconnell_efficiency,
    reid_vapor_pressure,
)
from difflow_refinery.vacuum.column import StageSpec

jax.config.update("jax_enable_x64", True)


def _flows(stream, names):
    return np.array([float(stream.get(f"F_{n}", 0.0)) for n in names])


def _feed(names, w, T, P):
    d = {f"F_{n}": float(w.get(n, 0.0)) for n in names}
    d.update(T=T, P=P)
    return d


def assert_balanced(column, feeds, products, rtol=1e-8):
    """Every component and the total, to ``rtol`` of the feed (a component
    the feed does not carry: to ``rtol`` of the total feed)."""
    names = column.names
    F = sum(_flows(f, names) for f in feeds)
    P = sum(_flows(p, names) for p in products)
    scale = np.where(F > 0, F, F.sum())
    assert np.max(np.abs(P - F) / scale) < rtol
    assert abs(P.sum() - F.sum()) / F.sum() < rtol


# -----------------------------------------------------------------------------
# A small debutanizer: the per-commit column
# -----------------------------------------------------------------------------

SMALL = ["propane", "isobutane", "n_butane", "isopentane", "n_pentane", "n_hexane"]
SMALL_FEED = {"F_propane": 10.0, "F_isobutane": 8.0, "F_n_butane": 15.0, "F_isopentane": 12.0,
              "F_n_pentane": 15.0, "F_n_hexane": 40.0, "T": 370.0, "P": 12e5}


@pytest.fixture(scope="module")
def small():
    c = gas_components(SMALL)
    col = GasPlantColumn(debutanizer(c, n_trays=20, feed_tray=10, naphtha_rvp=80e3))
    lpg, naphtha, info = col(SMALL_FEED)
    return c, col, lpg, naphtha, info


class TestSmallDebutanizer:
    def test_it_converges_and_meets_its_specs(self, small):
        c, col, lpg, naphtha, info = small
        assert bool(info["converged"])
        o = info["outputs"]
        assert float(o["distillate.x.C5+"]) == pytest.approx(0.01, rel=1e-8)
        assert float(o["bottoms.rvp"]) == pytest.approx(80e3, rel=1e-8)

    def test_mass_balance(self, small):
        c, col, lpg, naphtha, info = small
        assert_balanced(col, [SMALL_FEED], [lpg, naphtha])

    def test_energy_balance(self, small):
        o = small[4]["outputs"]
        assert abs(float(o["energy_balance"])) < 1e-8 * float(o["reboiler.duty"])
        assert float(o["condenser.duty"]) > 0 and float(o["reboiler.duty"]) > 0

    def test_tray_efficiency_is_oconnells(self, small):
        c, col, *_ = small
        th = col.theta(SMALL_FEED)
        eta = np.asarray(th["eta"])
        assert eta[0] == 1.0 and eta[-1] == 1.0          # condenser, reboiler
        assert 0.3 < eta[1] < 1.0 and np.all(eta[1:-1] == eta[1])

    def test_streams(self, small):
        c, col, lpg, naphtha, info = small
        assert lpg["phase"] == "liquid" and naphtha["phase"] == "liquid"
        assert float(naphtha["T"]) > float(lpg["T"])
        assert col.products == ("distillate", "bottoms")
        T = np.asarray(info["profiles"]["T"])
        assert T.shape == (22,) and np.all(np.diff(T) > 0)

    def test_products(self, small):
        c, col, lpg, naphtha, info = small
        fl = jnp.asarray(_flows(lpg, c.names))
        q = lpg_quality(fl, c, "commercial_propane")
        assert float(q["values"]["pentanes_plus"]) == pytest.approx(0.01, rel=1e-8)
        assert float(q["values"]["vapor_pressure"]) > 0
        assert set(q["margins"]) == set(GPA_2140["commercial_propane"])
        fg = fuel_gas(fl, c)
        assert float(fg["lhv_mass"]) == pytest.approx(46e6, rel=0.03)   # C3/C4 LHV
        rvp = reid_vapor_pressure(jnp.asarray(_flows(naphtha, c.names)), c)
        assert float(rvp) == pytest.approx(80e3, rel=1e-6)

    def test_lpg_grade_errors(self, small):
        c = small[0]
        with pytest.raises(ValueError, match="grade"):
            lpg_quality(jnp.ones(c.n), c, "HD-6")

    def test_planning_block_wiring(self, small):
        c, col, *_ = small
        blk = gasplant_block(col, [SMALL_FEED], ["distillate.x.C5+", "top.P", "feed.mol"],
                             outputs=["reboiler.duty", "bottom.T", "bottoms.rvp"], jit=False)
        assert blk.fn.u0.tolist() == pytest.approx([0.01, 1000.0, 360.0])
        assert blk.metadata["u_units"] == ["-", "kPa", "kmol/h"]
        assert blk.metadata["y_units"] == ["MW", "C", "kPa"]
        o = small[4]["outputs"]
        y = np.asarray(blk.fn(blk.fn.u0))
        assert y == pytest.approx([float(o["reboiler.duty"]) / 1e6,
                                   float(o["bottom.T"]) - 273.15, 80.0], rel=1e-8)
        with pytest.raises(ValueError, match="no lever"):
            gasplant_block(col, [SMALL_FEED], ["reboiler.duty"])   # replaced by a spec
        with pytest.raises(ValueError, match="are levers"):
            gasplant_block(col, [SMALL_FEED], ["top.P"], outputs=["top.P"])


# -----------------------------------------------------------------------------
# Cheap units and validation
# -----------------------------------------------------------------------------


class TestValidation:
    def test_unknown_species_is_refused(self, small):
        col = small[1]
        with pytest.raises(ValueError, match="not components"):
            col.theta(dict(SMALL_FEED, F_methane=1.0))

    def test_feed_count(self, small):
        col = small[1]
        with pytest.raises(ValueError, match="expected 1 feed"):
            col.theta(SMALL_FEED, SMALL_FEED)

    def test_a_knob_needs_a_value_or_a_spec(self):
        c = gas_components(SMALL)
        with pytest.raises(ValueError, match="reboiler.duty has no value"):
            GasPlantColumn(GasPlantColumnParams(components=c, n_trays=5, feed_trays={"feed": 3},
                                                draw_rates={"distillate": 1.0}))

    def test_oconnell_needs_keys(self):
        c = gas_components(SMALL)
        with pytest.raises(ValueError, match="keys"):
            GasPlantColumn(GasPlantColumnParams(components=c, n_trays=5, feed_trays={"feed": 3},
                                                reboiler_duty=1e6, draw_rates={"distillate": 1.0},
                                                tray_efficiency=None))

    def test_factories_check_their_specs(self):
        c = gas_components(["methane", "ethane", "propane", "n_butane"])
        with pytest.raises(ValueError, match="exactly one"):
            absorber_deethanizer(c, c2_in_bottoms=0.01, bottoms_T=400.0)
        with pytest.raises(ValueError, match="naphtha_rvp or c4"):
            debutanizer(c, c4_in_naphtha=None)

    def test_cuts_keeps_the_named_cuts_with_their_own_constants(self):
        full = gas_components(["propane"], pseudo=NAPHTHA)
        pn = list(NAPHTHA.pseudo_names)
        keep = [pn[0], pn[-1]]
        part = gas_components(["propane"], pseudo=NAPHTHA, cuts=list(reversed(keep)))
        # characterization order, whatever order they were named in
        assert part.names == ("propane", *keep)
        rows = [0, 1, len(pn)]
        for key in ("MW", "Tc", "Pc", "omega", "lhv", "pseudo", "cp"):
            a, b = np.asarray(getattr(full, key)), np.asarray(getattr(part, key))
            np.testing.assert_array_equal(b, a[rows], err_msg=key)
        with pytest.raises(ValueError, match="not cuts"):
            gas_components(["propane"], pseudo=NAPHTHA, cuts=["pc99"])

    def test_oconnell(self):
        # E_o = 0.492 (alpha mu)^-0.245, clipped to [0.1, 1]
        assert float(oconnell_efficiency(1.0, 1.0)) == pytest.approx(0.492)
        assert float(oconnell_efficiency(2.0, 0.1)) == pytest.approx(0.492 * 0.2 ** -0.245)
        assert float(oconnell_efficiency(1.0, 1e-6)) == 1.0

    def test_layouts(self):
        lay = column_layout(10, {"feed": 5}, side_draws={"side": 3})
        assert lay.products == ("distillate", "side", "bottoms")
        assert lay.n_stages == 12
        ab = column_layout(10, {"lean_oil": 1, "feed": 5}, condenser=None)
        assert ab.products == ("overhead", "bottoms") and ab.n_stages == 11
        part = column_layout(10, {"feed": 5}, condenser="partial")
        assert part.products == ("overhead", "bottoms")

    def test_registered(self):
        names = []

        class Reg:
            def register(self, name, **kw):
                names.append(name)

        dr.register(Reg())
        assert {"GasPlantColumn", "WetGasCompressor", "AmineTreater"} <= set(names)


class TestAmine:
    def test_removal_and_balance(self):
        sweet, acid, _ = AmineTreater(AmineTreaterParams())(
            {"F_methane": 10.0, "F_hydrogen_sulfide": 0.5, "T": 310.0, "P": 14e5})
        assert float(sweet["F_hydrogen_sulfide"]) == pytest.approx(0.005)
        assert float(acid["F_hydrogen_sulfide"]) == pytest.approx(0.495)
        assert float(sweet["F_methane"]) == pytest.approx(10.0)
        assert float(acid["F_methane"]) == pytest.approx(0.0)


WET_GAS = ["hydrogen", "methane", "ethane", "propane", "n_butane", "n_pentane"]


def test_solve_T_converges():
    """The compressor's temperature solve converges, and carries the
    implicit derivative. Its Newton slope was ``grad(stop_gradient(fn))`` --
    zero -- so the loop bounced between half and twice the guess and the
    answer was one final Newton step (found against DWSIM:
    ``test_dwsim_gasplant.py``)."""
    from difflow_refinery.gasplant.units import _solve_T

    fn = lambda T, a: a * T ** 3 + jnp.log(T)  # noqa: E731
    T = _solve_T(lambda t: fn(t, 2.0), 2.0 * 350.0 ** 3 + np.log(350.0), 300.0)
    assert float(T) == pytest.approx(350.0, rel=1e-13)
    # implicit derivative dT/da = -(T^3) / (3 a T^2 + 1/T)
    g = jax.grad(lambda a: _solve_T(lambda t: fn(t, a), 2.0 * 350.0 ** 3 + np.log(350.0), 300.0))(2.0)
    assert float(g) == pytest.approx(-350.0 ** 3 / (6.0 * 350.0 ** 2 + 1 / 350.0), rel=1e-8)


class TestCompressor:
    @pytest.fixture(scope="class")
    def run(self):
        c = gas_components(WET_GAS)
        comp = GasCompressor(GasCompressorParams(c, outlet_P=14e5, n_stages=2))
        feed = {"F_hydrogen": 5.0, "F_methane": 30.0, "F_ethane": 15.0, "F_propane": 15.0,
                "F_n_butane": 10.0, "F_n_pentane": 5.0, "T": 313.15, "P": 1.5e5}
        return c, comp, feed, comp(feed)

    def test_balance_and_condensate(self, run):
        c, comp, feed, (gas, cond, info) = run
        F = _flows(feed, c.names)
        P = _flows(gas, c.names) + _flows(cond, c.names)
        assert np.max(np.abs(P - F) / F) < 1e-10
        # compressed and cooled to 40 C, some pentane drops out; hydrogen
        # dissolves only sparingly
        assert float(cond["F_n_pentane"]) > 0.1
        x_h2 = float(cond["F_hydrogen"]) / _flows(cond, c.names).sum()
        y_h2 = float(gas["F_hydrogen"]) / _flows(gas, c.names).sum()
        assert x_h2 < 0.05 * y_h2

    def test_power_and_temperatures(self, run):
        c, comp, feed, (gas, cond, info) = run
        assert float(info["power"]) > 0
        assert float(info["ratio"]) == pytest.approx((14e5 / 1.5e5) ** 0.5)
        assert np.all(np.asarray(info["discharge_T"]) > 313.15)
        assert float(gas["P"]) == pytest.approx(14e5)


# -----------------------------------------------------------------------------
# Column cases (the guess of a hard column)
# -----------------------------------------------------------------------------

#: A characterised heavy naphtha tail: two pseudocomponents (~110 and 150 C).
NAPHTHA = characterize(Assay([0, 50, 100], [360.0, 400.0, 470.0], sg=0.74),
                       cut_points=[385.0, 420.0])
#: CDU light ends only (straight run): saturated, with H2S.
SR = ["hydrogen_sulfide", "methane", "ethane", "propane", "isobutane", "n_butane",
      "isopentane", "n_pentane", "n_hexane"]
#: CDU plus FCC gas: hydrogen, olefins.
FCC = ["hydrogen", "hydrogen_sulfide", "methane", "ethane", "ethylene", "propane", "propylene",
       "isobutane", "n_butane", "1_butene", "isobutylene", "cis_2_butene", "trans_2_butene",
       "isopentane", "n_pentane", "n_hexane"]
#: Debutanizer feeds (mol/s), deethanizer-bottoms-like.
SR_W = {"ethane": 0.2, "propane": 8, "isobutane": 5, "n_butane": 12, "isopentane": 10,
        "n_pentane": 12, "n_hexane": 20, "pc02": 18, "pc03": 15, "hydrogen_sulfide": 0.05}
FCC_W = {"ethane": 0.1, "ethylene": 0.02, "propane": 4, "propylene": 10, "isobutane": 8,
         "n_butane": 3, "1_butene": 4, "isobutylene": 5, "cis_2_butene": 3, "trans_2_butene": 4,
         "isopentane": 12, "n_pentane": 4, "n_hexane": 15, "pc02": 14, "pc03": 14,
         "hydrogen_sulfide": 0.05}
C3C4_KEEP = ("ethane", "ethylene", "propane", "propylene", "isobutane", "n_butane", "1_butene",
             "isobutylene", "cis_2_butene", "trans_2_butene", "isopentane", "hydrogen_sulfide")
DIB_KEEP = ("isobutane", "n_butane", "1_butene", "isobutylene", "cis_2_butene",
            "trans_2_butene", "isopentane")


def factory_case(which, feed_kind):
    names, w = (SR, SR_W) if feed_kind == "sr" else (FCC, FCC_W)
    c = gas_components(names, pseudo=NAPHTHA)
    names = list(c.names)
    if which == "debutanizer":
        return debutanizer(c), (_feed(names, w, 380.0, 12e5),)
    if which == "c3c4_splitter":
        lw = {k: v for k, v in w.items() if k in C3C4_KEEP}
        return c3c4_splitter(c), (_feed(names, lw, 320.0, 18e5),)
    if which == "deisobutanizer":
        lw = {k: v for k, v in w.items() if k in DIB_KEEP}
        # FCC butanes carry the butenes, which boil with isobutane: purity
        # 0.5 is what the C4 olefins leave room for
        return (deisobutanizer(c, ic4_purity=0.9 if feed_kind == "sr" else 0.5),
                (_feed(names, lw, 320.0, 8e5),))
    gw = dict(w, methane=30.0, ethane=15.0)
    if feed_kind == "fcc":
        gw.update(hydrogen=20.0, ethylene=5.0)
    lean = {"n_hexane": 30.0, "pc02": 40.0, "pc03": 30.0}
    return (absorber_deethanizer(c),
            (_feed(names, lean, 310.0, 14e5), _feed(names, gw, 320.0, 14.5e5)))


def test_log_mmatrix_solve_matches_lu_and_resolves_tiny_flows():
    from difflow_refinery.gasplant.column import _log_mmatrix_solve

    rng = np.random.default_rng(1)
    C, N = 3, 12
    M = -rng.uniform(0, 1, (C, N, N)) * (rng.uniform(size=(C, N, N)) < 0.4)
    for c in range(C):
        np.fill_diagonal(M[c], 0.0)
        np.fill_diagonal(M[c], -M[c].sum(axis=0) + rng.uniform(0.1, 1.0, N))
    f = rng.uniform(0.0, 1.0, (C, N)) + 1e-3
    ref = np.log(np.linalg.solve(M, f[..., None])[..., 0])
    np.testing.assert_allclose(_log_mmatrix_solve(jnp.asarray(M), jnp.asarray(f)), ref,
                               atol=1e-12)
    # x_i = a x_(i+1), fed at the bottom: x_i = a^(N-1-i), down to e^-1100,
    # far under what an LU solve (or even a float) resolves
    a = np.exp(-100.0)
    S = np.eye(N)[None].copy()
    S[0, np.arange(N - 1), np.arange(1, N)] = -a
    fb = np.full((1, N), 0.0)
    fb[0, -1] = 1.0
    got = np.asarray(_log_mmatrix_solve(jnp.asarray(S), jnp.asarray(fb)))[0]
    np.testing.assert_allclose(got, -100.0 * np.arange(N - 1, -1, -1), atol=1e-9)


def test_initial_guess_is_insensitive_to_round_off_in_the_feed():
    # the deisobutanizer's heavy cut sits ~e^-600 under its feed at the top:
    # an LU solve put round-off there, a 1e-12 change of a feed moved those
    # log flows by hundreds, and pass 1 stalled for some feeds (0.2.0's
    # Publish gate failed on a CI runner's rounding)
    params, feeds = factory_case("deisobutanizer", "sr")
    col = GasPlantColumn(params)
    rng = np.random.default_rng(0)
    nudged = [{k: (v * (1 + 1e-12 * rng.standard_normal()) if k.startswith("F_") else v)
               for k, v in f.items()} for f in feeds]
    gc = col.column
    x0, x1 = (np.asarray(gc.initial_guess(gc.prepare(col.theta(*fs)))) for fs in (feeds, nudged))
    assert np.abs(x1 - x0).max() < 1e-6


# -----------------------------------------------------------------------------
# Implicit gradients against central differences
# -----------------------------------------------------------------------------

GRAD_OUT = ("reboiler.duty", "condenser.duty", "bottom.T")


def test_gradients_match_central_differences(small):
    """The small debutanizer, differentiated in its two specs and one feed
    flow: the implicit step's gradient against central differences of the
    whole three-pass solve (one jitted value-and-Jacobian: one compile)."""
    c, col, *_ = small
    th0 = col.theta(SMALL_FEED)
    i_nc4 = list(c.names).index("n_butane")

    def f(u):
        th = dict(th0, targets={"distillate.x.C5+": u[0], "bottoms.rvp": u[1]},
                  feeds={"feed": dict(th0["feeds"]["feed"],
                                      flows=th0["feeds"]["feed"]["flows"].at[i_nc4].set(u[2]))})
        o = col.solve_theta(th)["outputs"]
        y = jnp.stack([o[k] for k in GRAD_OUT])
        return y, y

    value_and_jac = jax.jit(lambda u: jax.jacfwd(f, has_aux=True)(u))   # one compile for both
    u0 = jnp.asarray([0.01, 80e3, 15.0])
    J = np.asarray(value_and_jac(u0)[0])
    h = np.array([1e-5, 10.0, 1e-3])
    for k in range(3):
        e = np.zeros(3)
        e[k] = h[k]
        fd = (np.asarray(value_and_jac(u0 + e)[1]) - np.asarray(value_and_jac(u0 - e)[1])) / (2 * h[k])
        scale = np.abs(J[:, k]) + 1e-12 * np.abs(np.asarray(value_and_jac(u0)[1]))
        assert np.all(np.abs(fd - J[:, k]) / scale < 1e-5), (k, fd, J[:, k])
    # purer naphtha (a lower RVP) costs boilup
    assert J[0, 1] < 0
