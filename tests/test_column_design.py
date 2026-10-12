"""Tests for difflow.units.column_design (#429).

What these do and do NOT verify: the correlation constants in the module were
written from memory (flagged in its docstrings). ``TestPublishedExamples``
reproduces printed worked examples (Dutta 2007 Examples 5.1 and 5.2; Perry's 8e
Sec. 14 Example 13) for the Fair fit, dry-plate drop, weir crest, Eckert flood
line and Robbins pressure drop. The other tests check closed-form hand calculations,
independent NumPy re-implementations, scaling laws, limiting cases, sign changes
of margins, and gradients (``check_grads`` plus finite differences). Agreement
with a correlation written twice by the same author says nothing about its
constants; it catches transcription and unit errors only.
"""

import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from difflow import make_stream
from difflow.mass_transfer import PACKINGS, hetp_from_htu
from difflow.units import column_design as cd
from difflow.units.column_design import (
    column_hardware_cost, fair_capacity_factor, flooding_velocity, flow_parameter,
    fractional_entrainment, gpdc_flood, gpdc_pressure_drop, hetp, kister_gill_flood_dp,
    oconnell_efficiency, packed_column_diameter, size_column, tray_column_diameter,
    tray_percent_flood, tray_pressure_drop, weep_point, weir_crest,
)

# benzene-like vapor / liquid
TRAY = dict(V_molar=100.0, MW_V=80.0, rho_V=2.9, L_molar=80.0, MW_L=85.0, rho_L=790.0, sigma=0.02)


def _fair_np(F, t, sigma):
    a, b = 0.0744 * t + 0.01173, 0.0304 * t + 0.015
    return (a * math.log10(1.0 / F) + b) * (sigma / 0.02) ** 0.2


class TestFair:
    def test_capacity_factor_hand_calc(self):
        # t = 0.6 m: alpha = 0.05637, beta = 0.03324; F_LV = 0.1 -> 0.08961 m/s
        assert float(fair_capacity_factor(0.1, 0.6, 0.02)) == pytest.approx(0.08961, rel=1e-4)
        for F, t, s in [(0.02, 0.45, 0.03), (0.05, 0.6, 0.015), (0.3, 0.9, 0.05)]:
            assert float(fair_capacity_factor(F, t, s)) == pytest.approx(_fair_np(F, t, s), rel=1e-4)

    def test_sigma_correction_and_multipliers(self):
        c1 = float(fair_capacity_factor(0.05, 0.6, 0.02))
        c2 = float(fair_capacity_factor(0.05, 0.6, 0.04))
        assert c2 / c1 == pytest.approx(2 ** 0.2, rel=1e-4)
        c3 = float(fair_capacity_factor(0.05, 0.6, 0.02, foaming_factor=0.8, hole_area_factor=0.9))
        assert c3 / c1 == pytest.approx(0.72, rel=1e-4)

    def test_floor_keeps_positive(self):
        assert float(fair_capacity_factor(50.0, 0.6, 0.02)) > 0

    def test_flooding_velocity(self):
        assert float(flooding_velocity(0.09, 790.0, 2.9)) == pytest.approx(
            0.09 * math.sqrt((790 - 2.9) / 2.9), rel=1e-12)

    def test_diameter_independent_reimplementation(self):
        D, info = tray_column_diameter(**TRAY, tray_spacing=0.6, fraction_flood=0.8)
        Vm, Lm = 100 * 80e-3, 80 * 85e-3
        F = Lm / Vm * math.sqrt(2.9 / 790)
        uf = _fair_np(F, 0.6, 0.02) * math.sqrt((790 - 2.9) / 2.9)
        Ac = Vm / 2.9 / (0.8 * uf) / 0.9
        assert float(D) == pytest.approx(math.sqrt(4 * Ac / math.pi), rel=1e-4)
        assert float(info["F_LV"]) == pytest.approx(F, rel=1e-12)
        assert float(info["A_active"]) == pytest.approx(0.8 * float(info["A_column"]), rel=1e-12)
        assert float(info["percent_flood"]) == 80.0

    def test_scaling_with_fraction_flood(self):
        D1, _ = tray_column_diameter(**TRAY, fraction_flood=0.8)
        D2, _ = tray_column_diameter(**TRAY, fraction_flood=0.4)
        assert float(D2 / D1) == pytest.approx(2 ** 0.5, rel=1e-12)

    def test_scaling_with_vapor_load(self):
        # V and L scaled together keep F_LV: D ~ (V MW / rho_V)^(1/2)
        D1, _ = tray_column_diameter(**TRAY)
        big = dict(TRAY, V_molar=400.0, L_molar=320.0)
        D4, _ = tray_column_diameter(**big)
        assert float(D4 / D1) == pytest.approx(2.0, rel=1e-12)
        heavy = dict(TRAY, MW_V=320.0, MW_L=340.0)
        Dh, _ = tray_column_diameter(**heavy)
        assert float(Dh / D1) == pytest.approx(2.0, rel=1e-12)

    def test_mw_changes_diameter_unlike_the_stub(self):
        D1, _ = tray_column_diameter(**TRAY)
        D2, _ = tray_column_diameter(**dict(TRAY, MW_V=160.0))
        assert float(D2) > 1.2 * float(D1)

    def test_rating_inverts_sizing(self):
        D, _ = tray_column_diameter(**TRAY, fraction_flood=0.75)
        pf = tray_percent_flood(**TRAY, diameter=D)
        assert float(pf) == pytest.approx(75.0, rel=1e-10)
        # debottlenecking: +20 % throughput on the same column
        up = dict(TRAY, V_molar=120.0, L_molar=96.0)
        assert float(tray_percent_flood(**up, diameter=D)) == pytest.approx(90.0, rel=1e-10)


class TestTrayHydraulics:
    kw = dict(V_vol=2.76, L_vol=0.0086, rho_V=2.9, rho_L=790.0, diameter=1.67)

    def test_components_add_up(self):
        dP, i = tray_pressure_drop(**self.kw)
        assert float(dP) == pytest.approx(
            float(i["dP_dry"] + i["dP_liquid"] + i["dP_residual"]), rel=1e-12)
        heads = i["h_dry"] + i["h_weir"] + i["h_crest"] + i["h_residual"]
        assert float(dP) == pytest.approx(790 * 9.80665 * float(heads), rel=1e-12)
        assert float(i["dP_residual"]) == pytest.approx(12.5 * 9.80665, rel=1e-12)

    def test_francis_and_dry_head(self):
        _, i = tray_pressure_drop(**self.kw)
        lw = 0.7 * 1.67
        assert float(i["h_crest"]) == pytest.approx((0.0086 / (1.84 * lw)) ** (2 / 3), rel=1e-9)
        Aa = 0.8 * math.pi * 1.67 ** 2 / 4
        uh = 2.76 / (0.1 * Aa)
        assert float(i["u_hole"]) == pytest.approx(uh, rel=1e-12)
        assert float(i["h_dry"]) == pytest.approx(2.9 * uh ** 2 / (2 * 0.64 * 790 * 9.80665), rel=1e-12)
        assert float(weir_crest(0.0, 1.0)) < 1e-9

    def test_dry_drop_scales_with_v_squared(self):
        _, a = tray_pressure_drop(**self.kw)
        _, b = tray_pressure_drop(**dict(self.kw, V_vol=2 * 2.76))
        assert float(b["dP_dry"] / a["dP_dry"]) == pytest.approx(4.0, rel=1e-12)

    def test_weeping_margin_changes_sign_at_turndown(self):
        kw = dict(L_vol=0.0086, rho_V=2.9, diameter=1.67)
        _, i = weep_point(V_vol=2.76, **kw)
        # by hand: u_min = (30 - 0.9 (25.4 - 5)) / sqrt(2.9)
        assert float(i["u_min"]) == pytest.approx((30 - 0.9 * 20.4) / math.sqrt(2.9), rel=1e-12)
        V_turn = float(i["u_min"]) * 0.1 * 0.8 * math.pi * 1.67 ** 2 / 4
        assert float(weep_point(V_vol=V_turn, **kw)[0]) == pytest.approx(0.0, abs=1e-12)
        assert float(weep_point(V_vol=0.9 * V_turn, **kw)[0]) < 0
        assert float(weep_point(V_vol=1.1 * V_turn, **kw)[0]) > 0
        # turndown ratio of the design point (design is 2.76 m^3/s)
        assert 2.76 / V_turn > 1.5

    def test_entrainment_placeholder_trends(self):
        with pytest.warns(cd.EntrainmentPlaceholderWarning):
            assert float(fractional_entrainment(0.1, 80.0)) == pytest.approx(0.01)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", cd.EntrainmentPlaceholderWarning)
            assert float(fractional_entrainment(0.1, 90.0)) > float(fractional_entrainment(0.1, 80.0))
            assert float(fractional_entrainment(0.3, 80.0)) > float(fractional_entrainment(0.1, 80.0))

    def test_weep_K2_callable(self):
        kw = dict(L_vol=0.0086, rho_V=2.9, diameter=1.67)
        # a constant callable reproduces the number
        m0, i0 = weep_point(2.0, **kw)
        m1, i1 = weep_point(2.0, **kw, K2=lambda h: 30.0 + 0.0 * h)
        assert float(m1) == pytest.approx(float(m0), rel=1e-12)
        assert float(i0["K2"]) == 30.0
        # the callable sees h_w + h_ow in mm
        seen = {}

        def k2(h_mm):
            seen["h"] = float(h_mm)
            return 29.0 + 0.02 * h_mm
        _, i = weep_point(2.0, **kw, K2=k2)
        h_mm = 1000.0 * (0.05 + (0.0086 / (1.84 * 0.7 * 1.67)) ** (2 / 3))
        assert seen["h"] == pytest.approx(h_mm, rel=1e-9)
        assert float(i["K2"]) == pytest.approx(29.0 + 0.02 * h_mm, rel=1e-12)
        assert float(i["u_min"]) == pytest.approx(
            (29.0 + 0.02 * h_mm - 0.9 * 20.4) / math.sqrt(2.9), rel=1e-12)
        # more liquid -> larger K2 -> higher u_min -> smaller margin
        assert float(weep_point(2.0, 0.02, 2.9, 1.67, K2=k2)[0]) < float(weep_point(2.0, **kw, K2=k2)[0])
        # differentiable in L through the chart callable
        chart = lambda h: jnp.interp(h, jnp.array([0.0, 100.0]), jnp.array([29.0, 31.0]))
        check_grads(lambda Ls: weep_point(2.0, 1e-3 * Ls, 2.9, 1.67, K2=chart)[0], (8.6,),  # L in L/s
                    order=1, modes=["rev"], rtol=1e-5, atol=1e-6)


class TestDowncomerBackup:
    kw = dict(V_vol=2.76, L_vol=0.0086, rho_V=2.9, rho_L=790.0, diameter=1.67)

    def test_hand_calculation(self):
        # Sinnott Sec. 11.13 by hand, with h_t from the module's tray drop.
        dP, _ = tray_pressure_drop(**self.kw)
        D, lw = 1.67, 0.7 * 1.67
        Ac = math.pi * D ** 2 / 4
        Ad = 0.1 * Ac
        h_ow = (0.0086 / (1.84 * lw)) ** (2 / 3)
        h_t = float(dP) / (790.0 * 9.80665)
        Aap = 0.040 * lw                      # h_ap = h_w - 10 mm
        Am = min(Ad, Aap)
        L_wd = 0.0086 * 790.0                 # kg/s
        h_dc = 166.0 * (L_wd / (790.0 * Am)) ** 2 / 1000.0   # mm -> m
        h_b = 0.05 + h_ow + h_t + h_dc
        margin, i = cd.downcomer_backup(**self.kw, tray_spacing=0.6)
        assert float(i["A_m"]) == pytest.approx(Aap, rel=1e-12) and Aap < Ad
        assert float(i["h_apron"]) == pytest.approx(h_dc, rel=1e-12)
        assert float(i["h_backup"]) == pytest.approx(h_b, rel=1e-12)
        assert float(i["h_backup_max"]) == pytest.approx(0.5 * (0.6 + 0.05), rel=1e-12)
        assert float(margin) == pytest.approx(0.325 / h_b - 1.0, rel=1e-12)
        assert float(i["residence_time"]) == pytest.approx(Ad * h_b * 790.0 / L_wd, rel=1e-12)
        assert float(i["residence_margin"]) == pytest.approx(Ad * h_b / 0.0086 / 3.0 - 1.0, rel=1e-12)

    def test_explicit_inputs_and_area_switch(self):
        # a 0.2 m apron gap: the downcomer area governs
        _, i = cd.downcomer_backup(**self.kw, apron_clearance=0.2, dP_tray=1000.0)
        Ad = 0.1 * math.pi * 1.67 ** 2 / 4
        assert float(i["A_m"]) == pytest.approx(Ad, rel=1e-12)
        assert float(i["h_tray"]) == pytest.approx(1000.0 / (790.0 * 9.80665), rel=1e-12)

    def test_floods_at_high_liquid_load_and_short_spacing(self):
        m_ok, _ = cd.downcomer_backup(**self.kw, tray_spacing=0.6)
        m_bad, _ = cd.downcomer_backup(**dict(self.kw, L_vol=0.08), tray_spacing=0.3)
        assert float(m_ok) > 0 > float(m_bad)

    def test_grads(self):
        check_grads(lambda V, Ls, t: cd.downcomer_backup(V, 1e-3 * Ls, 2.9, 790.0, 1.67,
                                                         tray_spacing=t)[0],
                    (2.76, 8.6, 0.6),  # L in L/s
                    order=1, modes=["fwd", "rev"], rtol=1e-5, atol=1e-6)


class TestOConnell:
    def test_values_and_reexport(self):
        from difflow_refinery.gasplant import units as gp
        assert gp.oconnell_efficiency is oconnell_efficiency
        assert float(oconnell_efficiency(1.0, 1.0)) == pytest.approx(0.492)
        assert float(oconnell_efficiency(2.0, 0.1)) == pytest.approx(0.492 * 0.2 ** -0.245)
        assert float(oconnell_efficiency(1.0, 1e-6)) == 1.0
        assert float(oconnell_efficiency(1e3, 1e3)) == 0.1

    def test_smooth_clip(self):
        g_hard = jax.grad(lambda m: oconnell_efficiency(1.0, m))(1e-6)
        g_soft = jax.grad(lambda m: oconnell_efficiency(1.0, m, smooth=0.02))(1e-6)
        assert float(g_hard) == 0.0 and float(g_soft) != 0.0
        assert float(oconnell_efficiency(1.0, 1.0, smooth=0.02)) == pytest.approx(0.492, abs=1e-6)
        assert float(oconnell_efficiency(1.0, 1e-6, smooth=0.02)) == pytest.approx(1.0, abs=1e-3)


class TestPacked:
    def test_flood_line_values(self):
        # log10 Y = -1.6678 at X = 1; X = 0.1 gives log10 Y = -1.6678 + 1.085 - 0.29655
        G = float(gpdc_flood(1.0, 100.0, 1000.0, 1.2, 1e-3))
        Y = G ** 2 * 100.0 / (1.2 * (1000 - 1.2) * 9.80665)
        assert Y == pytest.approx(10 ** -1.6678, rel=1e-10)
        Y01 = 10 ** (-1.6678 + 1.085 - 0.29655)
        G = float(gpdc_flood(0.1, 100.0, 1000.0, 1.2, 1e-3))
        assert G ** 2 * 100.0 / (1.2 * (1000 - 1.2) * 9.80665) == pytest.approx(Y01, rel=1e-10)

    def test_flood_scaling(self):
        base = float(gpdc_flood(0.1, 100.0, 1000.0, 1.2, 1e-3))
        assert float(gpdc_flood(0.1, 400.0, 1000.0, 1.2, 1e-3)) / base == pytest.approx(0.5, rel=1e-12)
        mu = float(gpdc_flood(0.1, 100.0, 1000.0, 1.2, 32e-3)) / base
        assert mu == pytest.approx(32.0 ** -0.1, rel=1e-12)

    def test_kister_gill_flood_dp_hand_calc(self):
        # F_p = 183 1/m = 55.78 1/ft: 0.115 * 55.78^0.7 in H2O/ft
        ft = 183.0 / 3.280839895
        expected = 0.115 * ft ** 0.7 * 249.0889 / 0.3048
        assert float(kister_gill_flood_dp(183.0)) == pytest.approx(expected, rel=1e-9)
        assert 800 < float(kister_gill_flood_dp(183.0)) < 2500   # 1-3 in H2O/ft

    def test_pressure_drop_anchored_at_flood_and_monotone(self):
        rho_L, rho_V, mu, Fp = 1000.0, 1.2, 1e-3, 183.0
        X = 0.1
        Gf = float(gpdc_flood(X, Fp, rho_L, rho_V, mu))
        Lf = X * Gf / math.sqrt(rho_V / (rho_L - rho_V))
        kw = dict(method="anchored")
        at_flood = float(gpdc_pressure_drop(Gf, Lf, rho_L, rho_V, mu, Fp, **kw))
        assert at_flood == pytest.approx(float(kister_gill_flood_dp(Fp)), rel=1e-10)
        # Fixed L/G: dP ~ (G/Gf)^(2n)
        half = float(gpdc_pressure_drop(0.5 * Gf, 0.5 * Lf, rho_L, rho_V, mu, Fp, **kw))
        assert half / at_flood == pytest.approx(0.25 ** 1.7, rel=1e-10)
        with pytest.raises(ValueError):
            gpdc_pressure_drop(Gf, Lf, rho_L, rho_V, mu, Fp, method="nope")

    def test_robbins_hand_calc_and_default(self):
        # Perry's Eqs. 14-145..149 written out in US units
        G, L, rV, rL, mu, Fpd = 1.5, 8.0, 1.2, 998.0, 1e-3, 150.0
        Gl, Ll = G * 737.3381, L * 737.3381
        rVl, rLl, Fft = rV * 0.06242796, rL * 0.06242796, Fpd * 0.3048
        Gf = Gl * (0.075 / rVl) ** 0.5 * (Fft / 20) ** 0.5
        Lf = Ll * (62.4 / rLl) * (Fft / 20) ** 0.5 * 1.0 ** 0.1
        dry = 7.4e-8 * Gf ** 2 * 10 ** (2.7e-5 * Lf)
        exp = (dry + 0.4 * (Lf / 20000) ** 0.1 * dry ** 4) * 249.0889 / 0.3048
        got = float(cd.robbins_pressure_drop(G, L, rV, rL, mu, Fpd))
        assert got == pytest.approx(exp, rel=1e-5)
        # gpdc_pressure_drop defaults to Robbins with F_pd = F_p
        assert float(gpdc_pressure_drop(G, L, rL, rV, mu, Fpd)) == pytest.approx(got, rel=1e-12)

    def test_robbins_flood_hits_kister_gill(self):
        rV, rL, mu, Fp, Fpd, r = 1.2, 998.0, 1e-3, 183.0, 174.0, 5.0
        Gf = float(cd.robbins_flood(r, rV, rL, mu, Fp, Fpd))
        dp = float(cd.robbins_pressure_drop(Gf, r * Gf, rV, rL, mu, Fpd))
        assert dp == pytest.approx(float(kister_gill_flood_dp(Fp)), rel=1e-9)
        # more liquid floods earlier
        assert float(cd.robbins_flood(2 * r, rV, rL, mu, Fp, Fpd)) < Gf

    def test_robbins_flood_diameter_option(self):
        pk = PACKINGS["pall_metal_25"]
        a = dict(V_molar=100.0, MW_V=29.0, rho_V=1.2, L_molar=500.0, MW_L=18.0, rho_L=1000.0, mu_L=1e-3)
        D, i = packed_column_diameter(**a, packing=pk, flood="robbins")
        dp_flood = float(cd.robbins_pressure_drop(i["G_flood"], 500 * 18e-3 / (100 * 29e-3) * i["G_flood"],
                                                  1.2, 1000.0, 1e-3, cd.DRY_PACKING_FACTORS["pall_metal_25"]))
        assert dp_flood == pytest.approx(float(i["dP_flood_per_m"]), rel=1e-8)
        with pytest.raises(ValueError):
            packed_column_diameter(**a, packing=pk, flood="nope")

    def test_diameter_scaling_and_info(self):
        pk = PACKINGS["pall_metal_25"]
        a = dict(V_molar=100.0, MW_V=29.0, rho_V=1.2, L_molar=500.0, MW_L=18.0, rho_L=1000.0, mu_L=1e-3)
        D7, i7 = packed_column_diameter(**a, packing=pk, fraction_flood=0.7)
        D35, _ = packed_column_diameter(**a, packing=pk, fraction_flood=0.35)
        assert float(D35 / D7) == pytest.approx(2 ** 0.5, rel=1e-12)
        assert float(i7["G"]) == pytest.approx(0.7 * float(i7["G_flood"]), rel=1e-12)
        assert float(i7["u_operating"]) == pytest.approx(float(i7["G"]) / 1.2, rel=1e-12)
        A = math.pi * float(D7) ** 2 / 4
        assert float(i7["G"]) == pytest.approx(100 * 29e-3 / A, rel=1e-12)
        assert float(i7["dP_per_m"]) < float(i7["dP_flood_per_m"])
        # wetting rate by hand
        L_flux = 500 * 18e-3 / A
        assert float(i7["wetting_rate"]) == pytest.approx(L_flux / (1000 * pk.a_p), rel=1e-12)
        # a deeply under-irrigated column is flagged
        dry = dict(a, L_molar=5.0)
        assert float(packed_column_diameter(**dry, packing=pk)[1]["wetting_margin"]) < 0
        assert float(i7["wetting_margin"]) > 0

    def test_explicit_fp_and_errors(self):
        a = dict(V_molar=100.0, MW_V=29.0, rho_V=1.2, L_molar=500.0, MW_L=18.0, rho_L=1000.0, mu_L=1e-3)
        D, _ = packed_column_diameter(**a, F_p=183.0, a_p=223.5)
        D2, _ = packed_column_diameter(**a, packing=PACKINGS["pall_metal_25"])
        assert float(D) == pytest.approx(float(D2), rel=1e-12)
        with pytest.raises(ValueError):
            packed_column_diameter(**a)

    def test_packing_table_has_factors(self):
        for name in ("intalox_ceramic_25", "intalox_ceramic_38", "intalox_ceramic_50",
                     "pall_metal_25", "pall_metal_38", "pall_metal_50", "mellapak_250y"):
            p = PACKINGS[name]
            assert p.F_p and p.F_p > 0 and 0 < p.eps < 1 and p.a_p > 0
            assert "unverified" in p.source

    def test_structured_packing_warns_no_capacity_model(self):
        a = dict(V_molar=100.0, MW_V=29.0, rho_V=1.2, L_molar=500.0, MW_L=18.0, rho_L=1000.0, mu_L=1e-3)
        with pytest.warns(cd.PackingCapacityWarning):
            packed_column_diameter(**a, packing=PACKINGS["mellapak_250y"], flood="robbins")
        with warnings.catch_warnings():
            warnings.simplefilter("error", cd.PackingCapacityWarning)
            packed_column_diameter(**a, packing=PACKINGS["pall_metal_50"])
            packed_column_diameter(**a, packing=PACKINGS["mellapak_250y"], F_pd=66.0)

    def test_hetp(self):
        assert float(hetp("random", d_p=0.025)) == pytest.approx(0.45)
        assert float(hetp("structured", a_p=250.0)) == pytest.approx(0.5)
        # Perry's 8e Eq. 14-159: C_XY = 1.45 for X-type below 300 m^2/m^3
        assert float(hetp("structured", a_p=250.0, C_XY=1.45)) == pytest.approx(0.68)
        assert float(hetp("htu", H_OG=0.4, lam=1.7)) == pytest.approx(float(hetp_from_htu(0.4, 1.7)))
        with pytest.raises(ValueError):
            hetp("nope")


class TestPublishedExamples:
    """Printed worked examples. Values in US units as printed, converted here."""

    def test_perry_example13_robbins(self):
        # Perry's 8e Sec. 14 Example 13: 2-in metal Pall rings, air-water,
        # G = 1500, L = 9000 lb/ft^2/h, rho_G = 0.074, rho_L = 62.4 lb/ft^3, 1 cP,
        # F_pd = 24 1/ft. Robbins chart: 0.40 in H2O/ft; GPDC chart: 0.38.
        kg = 0.45359237 / 0.3048 ** 2 / 3600.0
        rho = 0.45359237 / 0.3048 ** 3
        dp = float(cd.robbins_pressure_drop(1500 * kg, 9000 * kg, 0.074 * rho, 62.4 * rho,
                                            1e-3, 24 / 0.3048))
        in_ft = dp * 0.3048 / 249.0889
        assert in_ft == pytest.approx(0.38, abs=0.01)        # equation vs the printed 0.38-0.40
        assert in_ft == pytest.approx(0.40, rel=0.06)

    def test_dutta_example52_eckert_flood(self):
        # Dutta (2007) Example 5.2: 3/4-in ceramic Raschig rings, F_p = 94.5 1/ft,
        # L = 37525, G = 1500 kg/h, rho_G = 1.299, rho_L = 996 kg/m^3, 0.81 cP.
        # Eckert chart read as Y = 0.021 at X = 0.90; printed D = 0.90 m at 70 % flood.
        X = 37525 / 1500 * math.sqrt(1.299 / (996 - 1.299))
        Y = float(cd._gpdc_Y_flood(X))
        assert Y == pytest.approx(0.021, rel=0.16)
        Gf = float(gpdc_flood(X, 94.5 / 0.3048, 996.0, 1.299, 0.81e-3))
        D = math.sqrt(4 * (1500 / 3600) / (0.7 * Gf) / math.pi)
        assert D == pytest.approx(0.90, rel=0.05)

    def test_dutta_example51_tray(self):
        # Dutta (2007) Example 5.1: F_LV = 0.0357, 18-in spacing, 20.5 dyn/cm,
        # C_SB read 0.285 ft/s; dry drop 1.75 in at u_h = 39.1 ft/s, C_0 = 0.73,
        # rho_V/rho_L = 0.165/50.4; crest 0.48 (gpm/in)^(2/3) in.
        C = float(fair_capacity_factor(0.0357, 18 * 0.0254, 0.0205)) / 0.3048
        assert C == pytest.approx(0.285, rel=0.12)
        u_h = 39.1 * 0.3048
        h_d = 0.165 * u_h ** 2 / (2 * 0.73 ** 2 * 50.4 * 9.80665) / 0.0254
        assert h_d == pytest.approx(1.75, rel=0.01)
        # Francis crest equals 0.48 (gpm / in)^(2/3) inches
        q_gpm, lw_in = 300.0, 40.0
        h = float(weir_crest(q_gpm * 6.30902e-5, lw_in * 0.0254)) / 0.0254
        assert h == pytest.approx(0.48 * (q_gpm / lw_in) ** (2 / 3), rel=0.01)


class TestSizeColumn:
    def test_trays_and_packing(self):
        s = size_column(20, 0.5, 1.5, tray_spacing=0.6, extra_height=3.0)
        assert float(s["n_trays"]) == pytest.approx(40.0)
        assert float(s["height"]) == pytest.approx(27.0)
        area = math.pi * 1.5 ** 2 / 4
        assert float(s["volume"]) == pytest.approx(27.0 * area)
        assert float(s["tray_area"]) == pytest.approx(area)
        p = size_column(10, 1.0, 1.5, hetp=0.5, extra_height=2.0)
        assert float(p["packed_height"]) == pytest.approx(5.0) and float(p["height"]) == pytest.approx(7.0)
        with pytest.raises(ValueError):
            size_column(10, 1.0, 1.5)
        with pytest.raises(ValueError):
            size_column(10, 1.0, 1.5, tray_spacing=0.6, hetp=0.5)

    def test_packed_height_keyword(self):
        p = size_column(None, None, 1.5, packed_height=6.0, extra_height=2.0)
        area = math.pi * 1.5 ** 2 / 4
        assert float(p["packed_height"]) == 6.0 and float(p["height"]) == pytest.approx(8.0)
        assert float(p["n_trays"]) == 0.0 and float(p["volume"]) == pytest.approx(8.0 * area)
        # same as the HETP route with n * HETP = 6
        q = size_column(12, 1.0, 1.5, hetp=0.5, extra_height=2.0)
        for k in p:
            assert float(p[k]) == pytest.approx(float(q[k]), rel=1e-12)
        for bad in (dict(hetp=0.5, packed_height=6.0), dict(tray_spacing=0.6, packed_height=6.0)):
            with pytest.raises(ValueError):
                size_column(10, 1.0, 1.5, **bad)
        g = jax.grad(lambda Z: size_column(None, None, 1.5, packed_height=Z)["volume"])(6.0)
        assert float(g) == pytest.approx(area, rel=1e-12)

    def test_cost_materials_and_basis(self):
        from difflow.economics.capital import column_cost
        s = size_column(20, 0.5, 1.5, tray_spacing=0.6)
        base = column_hardware_cost(s)
        assert float(base["cost_year"]) == 2026.0
        assert float(base["shell_material_factor"]) == 1.0 and float(base["internals_installed"]) == 0.0
        ss = column_hardware_cost(s, material="stainless", internals_material="stainless")
        assert float(ss["shell"]) == pytest.approx(3.1 * float(base["shell"]), rel=1e-12)
        assert float(ss["internals"]) == pytest.approx(1.8 * float(base["internals"]), rel=1e-12)
        num = column_hardware_cost(s, material=2.0, internals_material=1.5)
        assert float(num["total"]) == pytest.approx(
            2.0 * float(base["shell"]) + 1.5 * float(base["internals"]), rel=1e-12)
        old = column_hardware_cost(s, target_year=2001)
        assert float(old["cost_year"]) == 2001.0
        assert float(old["shell"]) == pytest.approx(float(column_cost(s["volume"], "tray_column_shell", 2001)), rel=1e-12)
        with pytest.raises(ValueError):
            column_hardware_cost(s, material="unobtainium")
        with pytest.raises(ValueError):
            column_hardware_cost(s, internals_material="ceramic")   # not a tray material

        p = size_column(None, None, 1.5, packed_height=5.0)
        pe = column_hardware_cost(p, "packing_random")
        for name, f in (("plastic", 1.0), ("ceramic", 4.1), ("metal", 7.1), ("stainless", 7.1)):
            c = column_hardware_cost(p, "packing_random", internals_material=name)
            assert float(c["internals"]) == pytest.approx(f * float(pe["internals"]), rel=1e-12)
            assert float(c["shell"]) == pytest.approx(float(pe["shell"]), rel=1e-12)

        st = column_hardware_cost(p, "packing_structured")
        assert float(st["internals_installed"]) == 1.0
        assert "INSTALLED" in cd.HARDWARE_COST_BASIS["packing_structured"]
        bought = column_hardware_cost(p, "packing_structured", structured_installation_factor=2.0)
        assert float(bought["internals_installed"]) == 0.0
        assert float(bought["internals"]) == pytest.approx(0.5 * float(st["internals"]), rel=1e-12)
        with pytest.raises(ValueError):
            column_hardware_cost(p, "packing_structured", internals_material="plastic")

    def test_cost_with_material_is_differentiable(self):
        def f(D, Z):
            s = size_column(None, None, D, packed_height=Z)
            return column_hardware_cost(s, "packing_random", material="stainless",
                                        internals_material="metal")["total"] / 1e5
        check_grads(f, (1.5, 6.0), order=1, modes=["rev"], rtol=1e-5, atol=1e-5)

    def test_cost_matches_column_cost(self):
        from difflow.economics.capital import column_cost
        s = size_column(20, 0.5, 1.5, tray_spacing=0.6)
        c = column_hardware_cost(s)
        ref = column_cost(s["volume"], "tray_column_shell") + 40.0 * column_cost(s["tray_area"], "sieve_tray")
        assert float(c["total"]) == pytest.approx(float(ref), rel=1e-12)
        p = size_column(10, 1.0, 1.5, hetp=0.5)
        cp = column_hardware_cost(p, "packing_random")
        assert float(cp["total"]) > 0
        with pytest.raises(ValueError):
            column_hardware_cost(s, "nope")


class TestGradients:
    def test_diameter_grads(self):
        def f(ff, t, sigma, V):
            return tray_column_diameter(V, 80.0, 2.9, 0.8 * V, 85.0, 790.0, sigma,
                                        tray_spacing=t, fraction_flood=ff)[0]
        check_grads(f, (0.8, 0.6, 0.02, 100.0), order=1, modes=["rev"], rtol=1e-5, atol=1e-5)

    def test_pressure_drop_grads(self):
        def f(V, rho_V, hw):
            return tray_pressure_drop(V, 0.0086, rho_V, 790.0, 1.67, weir_height=hw)[0]
        check_grads(f, (2.76, 2.9, 0.05), order=1, modes=["rev"], rtol=1e-5, atol=1e-5)
        g = jax.grad(lambda L: tray_pressure_drop(2.76, L, 2.9, 790.0, 1.67)[0])(0.0)
        assert bool(jnp.isfinite(g))

    def test_weeping_grad(self):
        check_grads(lambda V, r: weep_point(V, 0.0086, r, 1.67)[0], (2.0, 2.9),
                    order=1, modes=["rev"], rtol=1e-5, atol=1e-5)

    def test_packed_grads(self):
        a = dict(V_molar=100.0, MW_V=29.0, rho_V=1.2, L_molar=500.0, MW_L=18.0, rho_L=1000.0, mu_L=1e-3)

        def f(ff, Fp):
            D, i = packed_column_diameter(**a, F_p=Fp, a_p=223.5, fraction_flood=ff)
            return jnp.array([D, i["dP_per_m"]])
        check_grads(f, (0.7, 183.0), order=1, modes=["rev"], rtol=1e-5, atol=1e-5)

    def test_robbins_grads(self):
        check_grads(lambda G, L, Fpd: cd.robbins_pressure_drop(G, L, 1.2, 998.0, 1e-3, Fpd) / 100.0,
                    (1.5, 8.0, 150.0), order=1, modes=["fwd", "rev"], rtol=1e-5, atol=1e-5)
        check_grads(lambda r, Fp: cd.robbins_flood(r, 1.2, 998.0, 1e-3, Fp),
                    (5.0, 183.0), order=1, modes=["rev"], rtol=1e-4, atol=1e-5)

    def test_cost_grads(self):
        def cost(ff, t, sigma, V):
            D, _ = tray_column_diameter(V, 80.0, 2.9, 0.8 * V, 85.0, 790.0, sigma,
                                        tray_spacing=t, fraction_flood=ff)
            s = size_column(20.0, 0.6, D, tray_spacing=t)
            return column_hardware_cost(s)["total"] / 1e5
        check_grads(cost, (0.8, 0.6, 0.02, 100.0), order=1, modes=["rev"], rtol=1e-5, atol=1e-5)

    def test_cost_gradient_sign_in_fraction_flood(self):
        # Capital falls as the fraction of flood rises (smaller shell); an optimum needs
        # the pressure-drop/entrainment penalty on the other side.
        def cost(ff):
            D, _ = tray_column_diameter(**TRAY, fraction_flood=ff)
            return column_hardware_cost(size_column(20.0, 0.6, D, tray_spacing=0.6))["total"]
        assert float(jax.grad(cost)(0.8)) < 0


class TestEndToEnd:
    """wex16-1 chain: DistillationColumn -> vapor rate -> diameter -> size -> cost, d/dR."""

    @pytest.fixture
    def column(self):
        from difflow.thermo import IdealThermo, SpeciesData
        from difflow.units.distillation import DistillationColumn, DistillationColumnParams
        sd = {
            "benzene": SpeciesData(name="benzene", MW=78.11, Cp_coeffs=(136.0, 0, 0, 0),
                                   Hvap_coeffs=(33900.0, 0.38, 562.0), antoine_coeffs=(13.82, 2788.0, -52.36)),
            "toluene": SpeciesData(name="toluene", MW=92.14, Cp_coeffs=(157.0, 0, 0, 0),
                                   Hvap_coeffs=(38000.0, 0.38, 591.8), antoine_coeffs=(13.93, 3096.0, -53.67)),
        }
        p = DistillationColumnParams(species_order=["benzene", "toluene"], n_stages=15,
                                     feed_stage=7, condenser_type="total", P=101325.0)
        return DistillationColumn(p, IdealThermo(sd))

    @pytest.mark.slow
    def test_chain_is_differentiable_in_R(self, column):
        feed = make_stream({"benzene": 40.0 / 3.6, "toluene": 60.0 / 3.6}, T=365.0, P=101325.0)

        D_mol = 100.0 * (0.4 - 0.05) / (0.95 - 0.05) / 3.6   # mol/s, from the overall balance

        def total_cost(R):
            _, _, info = column(feed, R=R, D_spec=D_mol)
            V = info["V_profile"][-1]               # rectifying vapor, mol/s
            L = R * D_mol
            D, _ = tray_column_diameter(V, 80.0, 2.9, L, 80.0, 790.0, 0.021)
            s = size_column(15.0, oconnell_efficiency(2.4, 0.3), D, tray_spacing=0.6)
            return column_hardware_cost(s)["total"], V, D

        c, V, D = total_cost(2.0)
        assert float(V) > 0 and float(D) > 0 and float(c) > 0
        g = jax.grad(lambda R: total_cost(R)[0])(2.0)
        assert bool(jnp.isfinite(g)) and float(g) > 0          # more reflux, bigger column
        h = 1e-3
        fd = (total_cost(2.0 + h)[0] - total_cost(2.0 - h)[0]) / (2 * h)
        assert float(g) == pytest.approx(float(fd), rel=1e-4)
        # and the vapor rate is (R+1) D to within the MESH energy-balance correction
        assert float(V) == pytest.approx(3.0 * D_mol, rel=0.1)


class TestFlashDrumSize:
    """Souders-Brown vertical drum: hand calculation, limits, gradients."""

    def test_hand_calculation_vapor_governs(self):
        from difflow.units.column_design import flash_drum_size
        Q_V, Q_L, rho_L, rho_V = 1.0, 0.001, 550.0, 15.0
        D, info = flash_drum_size(Q_V, Q_L, rho_L, rho_V)
        u = 0.107 * np.sqrt((rho_L - rho_V) / rho_V)
        D_v = np.sqrt(4 * Q_V / (np.pi * u))
        assert float(info["u_max"]) == pytest.approx(u, rel=1e-12)
        assert float(D) == pytest.approx(D_v, rel=1e-12)
        assert float(info["fraction_of_u_max"]) == pytest.approx(1.0, rel=1e-12)
        assert float(info["height"]) == pytest.approx(3.0 * D_v, rel=1e-12)

    def test_liquid_holdup_governs(self):
        from difflow.units.column_design import flash_drum_size
        D, info = flash_drum_size(0.01, 0.01, 800.0, 5.0, holdup_time=600.0)
        V_vessel = 0.01 * 600.0 / 0.5
        assert float(D) == pytest.approx((4 * V_vessel / (np.pi * 3.0)) ** (1 / 3), rel=1e-12)
        assert float(info["volume"]) == pytest.approx(V_vessel, rel=1e-12)
        assert float(info["fraction_of_u_max"]) < 1.0

    def test_no_mesh_needs_bigger_drum(self):
        from difflow.units.column_design import flash_drum_size
        D_mesh, _ = flash_drum_size(1.0, 1e-4, 550.0, 15.0, k=0.107)
        D_bare, _ = flash_drum_size(1.0, 1e-4, 550.0, 15.0, k=0.0305)
        assert float(D_bare / D_mesh) == pytest.approx(np.sqrt(0.107 / 0.0305), rel=1e-12)

    def test_gradients(self):
        from jax.test_util import check_grads
        from difflow.units.column_design import flash_drum_size

        def f(Q_V, rho_V, k):
            D, info = flash_drum_size(Q_V, 0.001, 550.0, rho_V, k=k)
            return D + info["volume"]

        check_grads(f, (1.0, 15.0, 0.107), order=1, modes=["fwd", "rev"])
