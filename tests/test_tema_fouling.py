"""TEMA design fouling resistances (difflow.fouling).

What is verified: the unit conversion (1 h ft^2 F/Btu = 0.17611 m^2 K/W,
from the definitions of the Btu, foot and degree), the binning of the crude
and water tables, and agreement with the TEMA-derived Incropera & DeWitt
Table 11.1 entries already in ``FOULING_RESISTANCES`` (which were checked
against that table). The TEMA values themselves were recalled from secondary
reproductions of RGP-T-2.4 and could not be checked against a printed copy;
the cross-check with Incropera covers sea water, boiler feedwater, fuel oil,
refrigerant liquids and steam only.
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from difflow.fouling import (
    HFT2F_PER_BTU,
    TEMA_CRUDE_OIL,
    TEMA_FOULING,
    TEMA_WATER,
    tema_crude_oil_fouling,
    tema_fouling_resistance,
    tema_water_fouling,
)
from difflow.heat_transfer import FOULING_RESISTANCES


def F_to_K(F):
    return (F - 32.0) * 5.0 / 9.0 + 273.15


def test_conversion_factor_from_definitions():
    btu = 1055.05585262                     # J (International Table Btu)
    ft2 = 0.3048**2
    dF = 5.0 / 9.0
    si = 3600.0 * ft2 * dF / btu            # (h ft^2 F/Btu) in (s m^2 K / J)
    assert si == pytest.approx(HFT2F_PER_BTU, rel=2e-5)


def _one_sig(x):
    return round(x, 4)


@pytest.mark.parametrize(
    "si_key, tema_si",
    [
        ("seawater_below_50C", lambda: tema_water_fouling("sea_water", 300.0, 0.5)),
        ("seawater_above_50C", lambda: tema_water_fouling("sea_water", 340.0, 0.5)),
        ("treated_boiler_feedwater_below_50C", lambda: tema_water_fouling("treated_boiler_feedwater", 300.0, 1.5)),
        ("treated_boiler_feedwater_above_50C", lambda: tema_water_fouling("treated_boiler_feedwater", 340.0, 1.5)),
        ("fuel_oil", lambda: tema_fouling_resistance("fuel_oil_no6")),
        ("refrigerating_liquids", lambda: tema_fouling_resistance("refrigerant_liquids")),
        ("steam_oil_free", lambda: tema_fouling_resistance("steam_non_oil_bearing")),
    ],
)
def test_agrees_with_incropera_table(si_key, tema_si):
    # Incropera lists the TEMA values in SI, rounded to 4 decimal places.
    assert _one_sig(float(tema_si())) == pytest.approx(FOULING_RESISTANCES[si_key], rel=1e-12)


def test_table_shape_and_ranges():
    for lo, hi in TEMA_FOULING.values():
        assert 0.0 < lo <= hi <= 0.01
    assert len(TEMA_CRUDE_OIL) == 4 and all(len(r) == 3 for r in TEMA_CRUDE_OIL)
    for row in TEMA_WATER.values():
        assert len(row) == 4
    # Crude: never decreasing with temperature, never increasing with velocity.
    import numpy as np
    c = np.array(TEMA_CRUDE_OIL)
    assert (np.diff(c, axis=0) >= 0).all() and (np.diff(c, axis=1) <= 0).all()


def test_lookup_variants():
    assert tema_fouling_resistance("kerosene") == pytest.approx(0.003 * HFT2F_PER_BTU)
    assert tema_fouling_resistance("kerosene", "low") == pytest.approx(0.002 * HFT2F_PER_BTU)
    assert tema_fouling_resistance("kerosene", "mean") == pytest.approx(0.0025 * HFT2F_PER_BTU)
    assert tema_fouling_resistance("engine_lube_oil") == pytest.approx(0.001 * HFT2F_PER_BTU)
    with pytest.raises(KeyError, match="Unknown service"):
        tema_fouling_resistance("crude")
    with pytest.raises(ValueError):
        tema_fouling_resistance("kerosene", "max")


@pytest.mark.parametrize(
    "T_F, v_fts, expect",
    [
        (100.0, 1.0, 0.003), (100.0, 3.0, 0.002), (100.0, 5.0, 0.002),
        (250.0, 1.0, 0.003), (250.0, 5.0, 0.002),
        (400.0, 1.0, 0.004), (400.0, 3.0, 0.003), (400.0, 5.0, 0.002),
        (600.0, 1.0, 0.005), (600.0, 3.0, 0.004), (600.0, 5.0, 0.003),
    ],
)
def test_crude_bins(T_F, v_fts, expect):
    got = tema_crude_oil_fouling(F_to_K(T_F), v_fts * 0.3048)
    assert float(got) == pytest.approx(expect * HFT2F_PER_BTU, rel=1e-12)


def test_water_bins_and_edges():
    row = TEMA_WATER["cooling_tower_untreated"]
    cases = [(F_to_K(100), 0.5, row[0]), (F_to_K(100), 1.5, row[1]),
             (F_to_K(150), 0.5, row[2]), (F_to_K(150), 1.5, row[3]),
             (F_to_K(125), 0.9144, row[0])]        # on both edges: the lower bins
    for T, v, expect in cases:
        assert float(tema_water_fouling("cooling_tower_untreated", T, v)) == pytest.approx(expect * HFT2F_PER_BTU)
    with pytest.raises(KeyError, match="Unknown water"):
        tema_water_fouling("lake", 300.0, 1.0)


def test_jit_and_vectorised():
    f = jax.jit(tema_crude_oil_fouling)
    out = f(jnp.array([F_to_K(100.0), F_to_K(600.0)]), jnp.array([0.3, 0.3]))
    assert out.shape == (2,)
    assert float(out[1]) > float(out[0])
    # Piecewise constant: zero derivative inside a bin.
    assert float(jax.grad(lambda v: tema_crude_oil_fouling(500.0, v))(1.0)) == 0.0
