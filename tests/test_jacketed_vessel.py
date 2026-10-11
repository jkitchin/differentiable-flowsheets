"""Agitated-vessel heat transfer: jacketed-wall and immersed-coil coefficients.

Verified against hand arithmetic of the stated correlations (Chilton-Drew-
Jebens paddle and coil, Uhl's turbine and anchor constants, Oldshue-Gretton
coil) and their gradients. No textbook worked-example number is quoted: the
Kern Ch. 20 and McCabe examples could not be recalled with confidence. The
turbine and anchor constants are recalled from secondary tabulations
(documented on ``AGITATED_VESSEL_CONSTANTS``); a test here pins them so a
change is deliberate.
"""

import jax
import pytest
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)

from difflow.heat_transfer import (
    AGITATED_VESSEL_CONSTANTS,
    agitated_coil_h,
    agitated_vessel_nusselt,
    agitator_reynolds,
    jacketed_vessel_h,
    prandtl,
)

# Water-like batch: 1.5 m tank, 0.5 m impeller at 2 rev/s.
RHO, N, DA, DT, MU, CP, K = 1000.0, 2.0, 0.5, 1.5, 1.0e-3, 4180.0, 0.6


def test_reynolds_uses_rev_per_s_and_impeller_diameter():
    assert float(agitator_reynolds(RHO, N, DA, MU)) == pytest.approx(1000.0 * 2.0 * 0.25 / 1e-3)


def test_paddle_chilton_drew_jebens_hand():
    Re, Pr, visc = 5.0e5, 7.0, 1.3
    expect = 0.36 * Re ** (2.0 / 3.0) * Pr ** (1.0 / 3.0) * visc**0.14
    assert float(agitated_vessel_nusselt(Re, Pr, visc)) == pytest.approx(expect, rel=1e-12)


def test_jacket_h_hand():
    Re = RHO * N * DA**2 / MU
    Pr = CP * MU / K
    mu_w = 0.7e-3
    h = 0.36 * Re ** (2 / 3) * Pr ** (1 / 3) * (MU / mu_w) ** 0.14 * K / DT
    assert float(jacketed_vessel_h(RHO, N, DA, DT, MU, CP, K, mu_wall=mu_w)) == pytest.approx(h, rel=1e-12)


def test_constants_pinned():
    assert AGITATED_VESSEL_CONSTANTS["paddle"][:3] == (0.36, 2.0 / 3.0, 0.14)
    assert AGITATED_VESSEL_CONSTANTS["flat_blade_turbine"][:3] == (0.74, 0.67, 0.14)
    assert AGITATED_VESSEL_CONSTANTS["anchor"][:3] == (0.36, 2.0 / 3.0, 0.18)
    assert AGITATED_VESSEL_CONSTANTS["anchor_low_re"][:3] == (1.0, 0.5, 0.18)
    for a, b, m, (lo, hi), src in AGITATED_VESSEL_CONSTANTS.values():
        assert lo < hi and src


@pytest.mark.parametrize("impeller", sorted(AGITATED_VESSEL_CONSTANTS))
def test_each_impeller_hand(impeller):
    a, b, m, _, _ = AGITATED_VESSEL_CONSTANTS[impeller]
    Re, Pr, visc = 2.0e4, 50.0, 2.0
    assert float(agitated_vessel_nusselt(Re, Pr, visc, impeller)) == pytest.approx(
        a * Re**b * Pr ** (1 / 3) * visc**m, rel=1e-12)


def test_turbine_beats_paddle_at_same_re():
    assert float(agitated_vessel_nusselt(1e5, 5.0, impeller="flat_blade_turbine")) > float(
        agitated_vessel_nusselt(1e5, 5.0))


def test_override_constants():
    assert float(agitated_vessel_nusselt(1e4, 8.0, a=0.5, b=0.6, m=0.0)) == pytest.approx(
        0.5 * 1e4**0.6 * 2.0, rel=1e-12)


def test_unknown_impeller():
    with pytest.raises(KeyError, match="impeller"):
        agitated_vessel_nusselt(1e4, 5.0, impeller="propeller_x")


def test_coil_chilton_drew_jebens_hand():
    Re, Pr, visc = 1.0e5, 5.0, 1.2
    h = 0.87 * Re**0.62 * Pr ** (1 / 3) * visc**0.14 * K / DT
    assert float(agitated_coil_h(Re, Pr, K, DT, visc)) == pytest.approx(h, rel=1e-12)


def test_coil_oldshue_gretton_hand():
    Re, Pr, d_o = 1.0e5, 5.0, 0.0334
    h = 0.17 * Re**0.67 * Pr**0.37 * (DA / DT) ** 0.1 * (d_o / DT) ** 0.5 * K / d_o
    got = agitated_coil_h(Re, Pr, K, DT, method="oldshue_gretton", D_a=DA, d_o=d_o)
    assert float(got) == pytest.approx(h, rel=1e-12)


def test_coil_errors():
    with pytest.raises(ValueError, match="D_a"):
        agitated_coil_h(1e5, 5.0, K, DT, method="oldshue_gretton")
    with pytest.raises(ValueError, match="method"):
        agitated_coil_h(1e5, 5.0, K, DT, method="dittus")


def test_plausible_magnitude():
    # Water in a stirred tank: vessel-side h of order 10^3 W/m^2/K (the usual band).
    h = float(jacketed_vessel_h(RHO, N, DA, DT, MU, CP, K, impeller="flat_blade_turbine"))
    assert 500.0 < h < 5000.0


def test_grads():
    # O(1) multipliers on each argument so the finite-difference step suits them all.
    check_grads(lambda sn, sd, sm: jacketed_vessel_h(RHO, sn * N, sd * DA, DT, sm * MU, CP, K, mu_wall=0.8e-3),
                (1.0, 1.0, 1.0), order=1, modes=["rev", "fwd"])
    check_grads(lambda sr, pr: agitated_coil_h(sr * 1e5, pr, K, DT, method="oldshue_gretton", D_a=DA, d_o=0.03),
                (1.0, 5.0), order=1, modes=["rev"])
    # dh/dN = (2/3) h / N for the paddle (h ~ N^(2/3))
    h = jacketed_vessel_h(RHO, N, DA, DT, MU, CP, K)
    dh = jax.grad(lambda n: jacketed_vessel_h(RHO, n, DA, DT, MU, CP, K))(N)
    assert float(dh) == pytest.approx(2.0 / 3.0 * float(h) / N, rel=1e-12)
    assert float(prandtl(CP, MU, K)) == pytest.approx(CP * MU / K)
