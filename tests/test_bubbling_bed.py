"""Bubbling fluidized beds: Mori-Wen and Darton bubble size, Davidson rise
velocity, two-phase bed expansion, and the Geldart (1973) group boundaries.

What is verified, stated plainly:

- The Mori-Wen functions take SI inputs but the correlation is published in
  cgs units. Each test evaluates the cgs equation **as printed** (cm, cm/s,
  cm^2) with plain Python arithmetic and compares with the SI function, so
  a slip in the unit conversion fails here.
- No Kunii & Levenspiel worked-example number is quoted from memory. The
  correlations are checked by hand arithmetic, by their limits (``d_b ->
  d_b0`` at the plate, ``-> d_bm`` high in the bed) and by monotonicity.
- Gradients with ``jax.test_util.check_grads``.
"""

import math

import jax
import jax.numpy as jnp
import pytest
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)

from difflow.particles import (
    G_STD,
    GELDART_1973_AB,
    GELDART_1973_BD,
    bubble_rise_velocity,
    bubble_velocity,
    bubbling_bed_expansion,
    darton_bubble_diameter,
    geldart_group,
    mori_wen_bubble_diameter,
    mori_wen_initial_bubble_diameter,
    mori_wen_max_bubble_diameter,
)

U0, UMF, DT = 0.30, 0.05, 0.8          # m/s, m/s, m


class TestMoriWenUnits:
    def test_dbm_matches_cgs_equation(self):
        A_cm2 = math.pi / 4.0 * (DT * 100.0) ** 2
        du_cms = (U0 - UMF) * 100.0
        d_cm = 0.652 * (A_cm2 * du_cms) ** 0.4
        assert float(mori_wen_max_bubble_diameter(U0, UMF, DT)) == pytest.approx(d_cm / 100.0, rel=1e-12)

    def test_dbm_si_constant(self):
        # 0.01 * 0.652 * (1e4 * 1e2)^0.4 = 1.638: the SI form quoted in the docstring
        A = math.pi / 4.0 * DT**2
        assert float(mori_wen_max_bubble_diameter(U0, UMF, DT)) == pytest.approx(
            0.01 * 0.652 * 1e6**0.4 * (A * (U0 - UMF)) ** 0.4, rel=1e-12)
        assert 0.01 * 0.652 * 1e6**0.4 == pytest.approx(1.638, abs=5e-4)

    def test_db0_porous_matches_cgs(self):
        d_cm = 0.00376 * ((U0 - UMF) * 100.0) ** 2
        assert float(mori_wen_initial_bubble_diameter(U0, UMF)) == pytest.approx(d_cm / 100.0, rel=1e-12)

    def test_db0_perforated_matches_cgs(self):
        N_or = 2000.0                       # orifices per m^2 -> 0.2 per cm^2
        A_per_orifice_cm2 = 1.0 / (N_or * 1e-4)
        d_cm = 0.347 * (A_per_orifice_cm2 * (U0 - UMF) * 100.0) ** 0.4
        got = mori_wen_initial_bubble_diameter(U0, UMF, "perforated", N_or)
        assert float(got) == pytest.approx(d_cm / 100.0, rel=1e-12)

    def test_db_matches_cgs_growth(self):
        h = 0.4
        dbm = 0.652 * (math.pi / 4.0 * 80.0**2 * 25.0) ** 0.4
        db0 = 0.00376 * 25.0**2
        d_cm = dbm - (dbm - db0) * math.exp(-0.3 * 40.0 / 80.0)
        assert float(mori_wen_bubble_diameter(h, U0, UMF, DT)) == pytest.approx(d_cm / 100.0, rel=1e-12)

    def test_errors(self):
        with pytest.raises(ValueError, match="N_or"):
            mori_wen_initial_bubble_diameter(U0, UMF, "perforated")
        with pytest.raises(ValueError, match="distributor"):
            mori_wen_initial_bubble_diameter(U0, UMF, "bubble_cap")


class TestMoriWenBehaviour:
    def test_limits(self):
        d0 = mori_wen_initial_bubble_diameter(U0, UMF)
        dbm = mori_wen_max_bubble_diameter(U0, UMF, DT)
        assert float(mori_wen_bubble_diameter(0.0, U0, UMF, DT)) == pytest.approx(float(d0), rel=1e-12)
        assert float(mori_wen_bubble_diameter(100.0 * DT, U0, UMF, DT)) == pytest.approx(float(dbm), rel=1e-10)

    def test_increases_with_height_and_excess_gas(self):
        hs = jnp.linspace(0.0, 3.0, 31)
        d = mori_wen_bubble_diameter(hs, U0, UMF, DT)
        assert bool(jnp.all(jnp.diff(d) > 0))
        us = jnp.linspace(0.08, 0.5, 20)
        d_u = jax.vmap(lambda u: mori_wen_bubble_diameter(0.5, u, UMF, DT))(us)
        assert bool(jnp.all(jnp.diff(d_u) > 0))

    def test_given_db0(self):
        d = mori_wen_bubble_diameter(0.0, U0, UMF, DT, d_b0=0.02)
        assert float(d) == pytest.approx(0.02, rel=1e-12)

    def test_grads(self):
        f = lambda h, u0, umf, dt: mori_wen_bubble_diameter(h, u0, umf, dt)
        check_grads(f, (0.5, U0, UMF, DT), order=1, modes=["rev", "fwd"])
        g = lambda u0, n: mori_wen_bubble_diameter(0.5, u0, UMF, DT, "perforated", n)
        check_grads(g, (U0, 2000.0), order=1, modes=["rev"])


class TestDarton:
    def test_hand_value(self):
        h, A0 = 0.5, 1e-4
        expect = 0.54 * (U0 - UMF) ** 0.4 * (h + 4.0 * math.sqrt(A0)) ** 0.8 / G_STD**0.2
        assert float(darton_bubble_diameter(h, U0, UMF, A0)) == pytest.approx(expect, rel=1e-12)

    def test_dimensionally_consistent(self):
        # The same bed in cgs (g = 980.665 cm/s^2) gives the same diameter in cm.
        h, A0 = 0.5, 1e-4
        d_m = float(darton_bubble_diameter(h, U0, UMF, A0))
        d_cm = float(darton_bubble_diameter(h * 100, U0 * 100, UMF * 100, A0 * 1e4, g=G_STD * 100))
        assert d_cm == pytest.approx(100.0 * d_m, rel=1e-12)

    def test_monotone_and_grads(self):
        d = darton_bubble_diameter(jnp.linspace(0.0, 2.0, 11), U0, UMF, 1e-4)
        assert bool(jnp.all(jnp.diff(d) > 0))
        check_grads(lambda h, u0: darton_bubble_diameter(h, u0, UMF, 1e-4), (0.5, U0), order=1, modes=["rev"])


class TestExpansion:
    def test_hand_values(self):
        d_b = 0.12
        u_br = 0.711 * math.sqrt(G_STD * d_b)
        u_b = U0 - UMF + u_br
        delta = (U0 - UMF) / u_b
        out = bubbling_bed_expansion(U0, UMF, d_b, 1.2)
        assert float(bubble_rise_velocity(d_b)) == pytest.approx(u_br, rel=1e-12)
        assert float(bubble_velocity(U0, UMF, d_b)) == pytest.approx(u_b, rel=1e-12)
        assert float(out["delta"]) == pytest.approx(delta, rel=1e-12)
        assert float(out["H"]) == pytest.approx(1.2 / (1.0 - delta), rel=1e-12)

    def test_at_umf_no_expansion(self):
        out = bubbling_bed_expansion(UMF, UMF, 0.1, 1.0)
        assert float(out["delta"]) == 0.0
        assert float(out["H"]) == pytest.approx(1.0)

    def test_expansion_grows_with_gas(self):
        H = [float(bubbling_bed_expansion(u, UMF, 0.1, 1.0)["H"]) for u in (0.1, 0.2, 0.4)]
        assert H[0] < H[1] < H[2]

    def test_composed_with_mori_wen_grads(self):
        def height(u0, umf, H_mf):
            d_b = mori_wen_bubble_diameter(H_mf / 2.0, u0, umf, DT)
            return bubbling_bed_expansion(u0, umf, d_b, H_mf)["H"]

        check_grads(height, (U0, UMF, 1.0), order=1, modes=["rev", "fwd"])


class TestGeldart1973:
    @pytest.mark.parametrize(
        "d_p, rho_p, expect",
        [
            (10e-6, 2500.0, "C"),
            (60e-6, 1500.0, "A"),      # FCC: 90 < 225
            (140e-6, 1500.0, "A"),     # 210 < 225
            (160e-6, 1500.0, "B"),     # 240 > 225
            (200e-6, 2600.0, "B"),     # sand
            (1000e-6, 2600.0, "D"),    # 2.6 * 1000^2 > 1e6
        ],
    )
    def test_groups(self, d_p, rho_p, expect):
        assert geldart_group(d_p, rho_p, 1.2, method="geldart1973") == expect

    def test_boundary_values(self):
        # The A/B line sits at (rho_p - rho_g) d_p = 225 (g/cm^3)(micron).
        rho_p = 2001.2                                    # drho = 2.0 g/cm^3
        d_ab = GELDART_1973_AB / 2.0 * 1e-6               # 112.5 micron
        assert geldart_group(0.99 * d_ab, rho_p, 1.2, method="geldart1973") == "A"
        assert geldart_group(1.01 * d_ab, rho_p, 1.2, method="geldart1973") == "B"
        d_bd = math.sqrt(GELDART_1973_BD / 2.0) * 1e-6    # 707 micron
        assert geldart_group(0.99 * d_bd, rho_p, 1.2, method="geldart1973") == "B"
        assert geldart_group(1.01 * d_bd, rho_p, 1.2, method="geldart1973") == "D"

    def test_default_unchanged_and_methods_differ_near_boundary(self):
        # 170 micron at 1500 kg/m^3: the extrapolated u_mb/u_mf line (near 180
        # micron at this density) calls it A; Geldart's 1973 line puts the edge
        # at 225 / 1.4988 = 150 micron, so it is B there.
        assert geldart_group(60e-6, 1500.0, 1.2) == "A"
        assert geldart_group(170e-6, 1500.0, 1.2) == "A"
        assert geldart_group(170e-6, 1500.0, 1.2, method="geldart1973") == "B"

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="method"):
            geldart_group(60e-6, 1500.0, 1.2, method="chart")
