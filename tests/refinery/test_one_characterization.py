"""Crude unit into vacuum column on one characterization (#301).

One `Assay` with a `HeavyEnd` is characterized once. The crude unit runs on
it, and its residue -- a stream of the same pseudo-components -- feeds a
`VacuumColumn` whose property table is `Characterization.pseudo_components()`
of that same characterization: one grid, no re-cutting between the units. In
a difflow `Flowsheet` the crude's light ends and the CDU's water ride along in
the residue and leave with the vacuum overhead.

Pinned: both units converge, the flowsheet's balance closes per component
and in total (water included), and gradients cross the connection -- one
through the vacuum column, one from the crude rate through both units.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from difflow.flowsheet import Flowsheet, Unit

import difflow_refinery as dr
from difflow_refinery import column as cc
from difflow_refinery.thermo import WATER_MW, ColumnThermo
from difflow_refinery.vacuum import VacuumColumn, VacuumColumnParams
from difflow_refinery.vacuum.thermo import MW_WATER

from .test_column import LIGHT, _atmospheric_params

jax.config.update("jax_enable_x64", True)

# a 30-stage crude unit and a vacuum column, solved once for the module
pytestmark = pytest.mark.slow

PCT = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
T_K = [t + 273.15 for t in (60, 95, 150, 205, 260, 315, 370, 430, 500, 600, 680)]
ASSAY = dr.Assay(PCT, T_K, sg=0.86, light_ends=LIGHT, heavy_end=dr.HeavyEnd(),
                 sulfur_wt=1.8, nitrogen_wppm=1500.0, ccr_wt=6.0,
                 nickel_vanadium_wppm=60.0, asphaltenes_wt=3.0)
BPD = 95_000.0
# the VDU's products renamed: "residue" is already the CDU's outlet
VDU_OUT = ["vac_overhead", "lvgo", "hvgo", "slop", "vac_residue", "vdu_info"]


@pytest.fixture(scope="module")
def plant():
    char = dr.characterize(ASSAY)
    th = ColumnThermo.from_characterization(char)
    kg_s = BPD * cc.BARREL / 86400.0 * float(char.bulk_sg) * 999.016
    sized = char.stream(kg_s, T=600.0, P=1.9e5, basis="mass")
    base = _atmospheric_params(sized, th, pa1_duty=15e6)
    column = _atmospheric_params(sized, th, pa1_duty=15e6,
                                 specs=base.specs + (cc.overflash(0.05),))
    cdu = dr.CrudeDistillationUnit(dr.CrudeDistillationUnitParams(assay=ASSAY, column=column))
    vdu = VacuumColumn(VacuumColumnParams(components=char.pseudo_components()))
    feed = cdu.feed(BPD, T=273.15 + 240.0, P=6e5)

    fs = Flowsheet(list(char.names) + ["water", "H2O"])
    fs.add_feed("crude", feed)
    fs.add_unit(Unit("cdu", cdu, ["crude"], list(cdu.outlet_names)))
    fs.add_unit(Unit("vdu", vdu, ["residue"], VDU_OUT))
    streams = fs.solve()
    return dict(char=char, cdu=cdu, vdu=vdu, feed=feed, streams=streams)


def _flows(stream, names):
    return np.array([float(stream.get(f"F_{n}", 0.0)) for n in names])


def _final_products(plant):
    cdu_out = [n for n in plant["cdu"].outlet_names if n != "residue"]
    return [plant["streams"][n] for n in cdu_out + VDU_OUT[:-1]]


def test_both_units_converge_on_one_grid(plant):
    assert bool(plant["cdu"].last_result.converged)
    assert bool(plant["streams"]["vdu_info"]["converged"])
    char, vdu = plant["char"], plant["vdu"]
    # the vacuum column's components ARE the crude unit's pseudo-components
    assert vdu.params.components.names == char.pseudo_names
    np.testing.assert_array_equal(np.asarray(vdu.params.components.Tb), np.asarray(char.Tb))


def test_balance_closes_per_component(plant):
    names = plant["char"].names
    fin = _flows(plant["feed"], names)
    out = sum(_flows(s, names) for s in _final_products(plant))
    np.testing.assert_allclose(out, fin, rtol=1e-9, atol=1e-12 * fin.sum())


def test_balance_closes_in_total_with_the_steam(plant):
    char, cdu = plant["char"], plant["cdu"]
    mw = np.asarray(char.component_MW) / 1000.0
    names = char.names
    p = cdu.unit.params
    # each unit's own water molar mass (18.015 and 18.01528): the CDU's water
    # passes through the VDU as F_water, its steam leaves as F_H2O
    steam_cdu = (float(p.bottom_steam) + sum(float(s.steam) for s in p.side_products)) * WATER_MW / 1000.0
    steam_vdu = float(plant["streams"]["vdu_info"]["outputs"]["steam.rate"])
    m_in = float(_flows(plant["feed"], names) @ mw) + steam_cdu + steam_vdu
    m_out = 0.0
    for s in _final_products(plant):
        m_out += float(_flows(s, names) @ mw)
        m_out += float(s.get("F_water", 0.0)) * WATER_MW / 1000.0
        m_out += float(s.get("F_H2O", 0.0)) * MW_WATER / 1000.0
    assert m_out == pytest.approx(m_in, rel=1e-9)


@pytest.mark.release
def test_vgo_yield_gradient_in_vacuum_furnace_T(plant):
    """d(VGO yield)/d(VDU coil outlet T) on the crude unit's residue: AD vs FD."""
    residue = plant["streams"]["residue"]
    params = plant["vdu"].params

    def vgo_yield(T):
        *_, info = VacuumColumn(params.update(furnace_T=T))(residue)
        return info["outputs"]["vgo.yield"]

    T0 = params.furnace_T
    g = float(jax.grad(vgo_yield)(T0))
    h = 0.05
    fd = float((vgo_yield(T0 + h) - vgo_yield(T0 - h)) / (2 * h))
    assert g > 0, "a hotter furnace lifts more gas oil"
    assert g == pytest.approx(fd, rel=1e-5)


@pytest.mark.release
def test_vgo_rate_gradient_in_crude_rate_crosses_both_units(plant):
    """d(VGO kg/s)/d(crude rate), through the CDU's and the VDU's implicit solves.

    The CDU's distillate and side draws are fixed rates, so extra crude
    leaves in its residue and the vacuum column turns part of it into VGO.
    """
    cdu, vdu, feed = plant["cdu"], plant["vdu"], plant["feed"]
    i_res = cdu.outlet_names.index("residue")

    def vgo(scale):
        crude = {k: (v * scale if k.startswith("F_") else v) for k, v in feed.items()}
        residue = cdu(crude)[i_res]
        *_, info = vdu(residue)
        return info["outputs"]["vgo.rate"]

    g = float(jax.grad(vgo)(1.0))
    h = 1e-4
    fd = float((vgo(1.0 + h) - vgo(1.0 - h)) / (2 * h))
    assert g == pytest.approx(fd, rel=1e-4)
    assert g > 0
    # and is a sizeable share of the crude's mass, not a trace effect
    assert g > 0.1 * float(plant["streams"]["vdu_info"]["outputs"]["vgo.rate"])


def test_products_blend_on_the_characterization(plant):
    """CDU and VDU product streams are BlendComponent.from_stream inputs.

    The pool's mass-averaged sulfur must equal what the vacuum column itself
    reports for its product: both average the same per-component vectors.
    """
    grid = dr.BlendCharacterization.from_characterization(plant["char"])
    streams = plant["streams"]
    props = streams["vdu_info"]["properties"]
    for name in ("lvgo", "hvgo"):
        comp = dr.BlendComponent.from_stream(name, streams[name], grid)
        assert float(comp.properties["S_ppm"]) == pytest.approx(
            float(props[name]["sulfur_wt"]) * 1e4, rel=1e-10)
        assert float(comp.properties["SG"]) == pytest.approx(float(props[name]["sg"]), rel=1e-10)
    diesel = dr.BlendComponent.from_stream("diesel", streams["diesel"], grid)
    assert 0.0 < float(diesel.properties["S_ppm"]) < float(
        dr.BlendComponent.from_stream("hvgo", streams["hvgo"], grid).properties["S_ppm"])
    lvgo = dr.BlendComponent.from_stream("lvgo", streams["lvgo"], grid)
    blend = dr.BlendPool("ulsd", specs=[("S_ppm", "<=", 15.0)])([diesel, lvgo], [0.7, 0.3])
    # sulfur blends by mass, so it lies between the two components'
    S = sorted(float(c.properties["S_ppm"]) for c in (diesel, lvgo))
    assert S[0] < float(blend.properties["S_ppm"]) < S[1]
