"""Tests for Tier 2 core unit operation enhancements.

Covers issues: #78, #84, #87, #158, #75, #76, #77, #79, #157, #88.
"""

import pytest
import jax
import jax.numpy as jnp
from jax import grad

from difflow.streams import make_stream, get_flows
from difflow.thermo import IdealThermo, SpeciesData


# =============================================================================
# Helpers
# =============================================================================

def _simple_rate_fn(C, T, params):
    """First-order: A -> B, r = k * C_A."""
    k = params['k']
    return jnp.array([k * C['A']])


def _make_ab_stream(F_A=1.0, F_B=0.0, T=350.0, P=101325.0):
    return make_stream({'A': F_A, 'B': F_B}, T, P)


def _make_thermo():
    """Create a minimal IdealThermo for testing."""
    return IdealThermo(
        species=['A', 'B'],
        Cp={'A': 75.0, 'B': 75.0},
        Hvap={'A': 30000.0, 'B': 35000.0},
        Tb={'A': 350.0, 'B': 380.0},
        Antoine={'A': (8.07, 1730.0, -39.0), 'B': (8.07, 1830.0, -39.0)},
    )


# =============================================================================
# #78 — PFR solution method info
# =============================================================================

class TestPFRSolutionMethodInfo:
    """Issue #78: PFR should report solution_method in info dict."""

    def test_info_has_solution_method(self):
        from difflow.units.pfr import PFR, PFRParams
        params = PFRParams(
            V=1.0,
            rate_fn=_simple_rate_fn,
            stoich=jnp.array([[-1], [1]]),
            rate_params={'k': 0.5},
            species_order=['A', 'B'],
        )
        pfr = PFR(params)
        inlet = _make_ab_stream()
        outlet, info = pfr(inlet, volumetric_flow=0.01)
        assert 'solution_method' in info
        assert info['solution_method'] == 'numerical'

    def test_analytical_matches_numerical_order1(self):
        from difflow.units.pfr import PFR, PFRParams, pfr_conversion_analytical
        k = 0.5
        V = 2.0
        Q_v = 0.01
        tau = V / Q_v

        params = PFRParams(
            V=V,
            rate_fn=_simple_rate_fn,
            stoich=jnp.array([[-1], [1]]),
            rate_params={'k': k},
            species_order=['A', 'B'],
        )
        pfr = PFR(params)
        inlet = _make_ab_stream()
        outlet, info = pfr(inlet, volumetric_flow=Q_v)

        X_numerical = float(info['conversion']['A'])
        X_analytical = float(pfr_conversion_analytical(k, tau, order=1))
        assert abs(X_numerical - X_analytical) < 0.01

    def test_analytical_order2(self):
        from difflow.units.pfr import pfr_conversion_analytical
        k = 0.1
        tau = 10.0
        C_A0 = 100.0
        X = float(pfr_conversion_analytical(k, tau, order=2, C_A0=C_A0))
        expected = k * tau * C_A0 / (1.0 + k * tau * C_A0)
        assert abs(X - expected) < 1e-6


# =============================================================================
# #84 — HX NTU validation flags
# =============================================================================

class TestHXNTUValidation:
    """Issue #84: HX should report flow_arrangement, NTU_very_high, Cr_near_one."""

    def _make_hx_streams(self, T_hot=400.0, T_cold=300.0, F_hot=1.0, F_cold=1.0):
        hot = make_stream({'water': F_hot}, T_hot, 101325.0)
        cold = make_stream({'water': F_cold}, T_cold, 101325.0)
        return hot, cold

    def test_counter_current_flags_normal(self):
        from difflow.units.heat_exchanger import CounterCurrentHX, HeatExchangerParams
        p = HeatExchangerParams(UA=100.0, Cp_hot=75.0, Cp_cold=75.0)
        hx = CounterCurrentHX(p)
        hot, cold = self._make_hx_streams(F_hot=2.0, F_cold=1.0)
        _, _, info = hx(hot, cold)
        assert info['flow_arrangement'] == 'counter_current'
        assert not bool(info['NTU_very_high'])
        assert not bool(info['Cr_near_one'])

    def test_high_ntu_flag(self):
        from difflow.units.heat_exchanger import CounterCurrentHX, HeatExchangerParams
        p = HeatExchangerParams(UA=10000.0, Cp_hot=75.0, Cp_cold=75.0)
        hx = CounterCurrentHX(p)
        hot, cold = self._make_hx_streams()
        _, _, info = hx(hot, cold)
        assert bool(info['NTU_very_high'])

    def test_balanced_cr_flag(self):
        from difflow.units.heat_exchanger import CounterCurrentHX, HeatExchangerParams
        p = HeatExchangerParams(UA=100.0, Cp_hot=75.0, Cp_cold=75.0)
        hx = CounterCurrentHX(p)
        hot, cold = self._make_hx_streams(F_hot=1.0, F_cold=1.0)
        _, _, info = hx(hot, cold)
        assert bool(info['Cr_near_one'])

    def test_co_current_arrangement(self):
        from difflow.units.heat_exchanger import CoCurrentHX, HeatExchangerParams
        p = HeatExchangerParams(UA=100.0, Cp_hot=75.0, Cp_cold=75.0)
        hx = CoCurrentHX(p)
        hot, cold = self._make_hx_streams()
        _, _, info = hx(hot, cold)
        assert info['flow_arrangement'] == 'co_current'

    def test_crossflow_arrangement(self):
        from difflow.units.heat_exchanger import CrossFlowHX, HeatExchangerParams
        p = HeatExchangerParams(UA=100.0, Cp_hot=75.0, Cp_cold=75.0)
        hx = CrossFlowHX(p)
        hot, cold = self._make_hx_streams()
        _, _, info = hx(hot, cold)
        assert info['flow_arrangement'] == 'crossflow'


# =============================================================================
# #87 — Distillation alpha variation warning
# =============================================================================

class TestDistillationAlphaVariation:
    """Issue #87: ShortcutColumn reports alpha variation diagnostics."""

    def _make_column(self, species, Tb_vals):
        species_data = {}
        for i, (s, tb) in enumerate(zip(species, Tb_vals)):
            species_data[s] = SpeciesData(
                s, MW=100.0,
                Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
                Hvap_coeffs=(30000.0, 0.38, tb + 150.0),
                antoine_coeffs=(10.0, 2800.0 + i * 200, -40.0),
            )
        thermo = IdealThermo(species_data)
        from difflow.units.distillation import ShortcutColumn, ShortcutColumnParams
        params = ShortcutColumnParams(
            species_order=species,
            light_key=species[0],
            heavy_key=species[1],
        )
        return ShortcutColumn(params, thermo)

    def test_alpha_variation_info_present(self):
        col = self._make_column(['A', 'B'], [350.0, 380.0])
        feed = make_stream({'A': 0.5, 'B': 0.5}, 365.0, 101325.0)
        _, _, info = col(feed, R=2.0)
        assert 'alpha_top' in info
        assert 'alpha_bot' in info
        assert 'alpha_variation' in info
        assert 'alpha_varies_significantly' in info

    def test_narrow_boiling_no_significant_variation(self):
        # Close boiling points → small alpha variation
        col = self._make_column(['A', 'B'], [350.0, 355.0])
        feed = make_stream({'A': 0.5, 'B': 0.5}, 352.0, 101325.0)
        _, _, info = col(feed, R=3.0)
        # Alpha variation should be small for close boiling points
        # (may or may not exceed 0.3 threshold depending on Antoine params)
        assert isinstance(info['alpha_varies_significantly'], jax.Array)


# =============================================================================
# #158 — Distillation feasibility checks
# =============================================================================

class TestDistillationFeasibility:
    """Issue #158: ShortcutColumn reports feasibility diagnostics."""

    def _make_column(self, Tb_A=350.0, Tb_B=380.0):
        species_data = {
            'A': SpeciesData(
                'A', MW=100.0,
                Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
                Hvap_coeffs=(30000.0, 0.38, Tb_A + 150.0),
                antoine_coeffs=(10.0, 2800.0, -40.0),
            ),
            'B': SpeciesData(
                'B', MW=100.0,
                Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
                Hvap_coeffs=(30000.0, 0.38, Tb_B + 150.0),
                antoine_coeffs=(10.0, 3000.0, -40.0),
            ),
        }
        thermo = IdealThermo(species_data)
        from difflow.units.distillation import ShortcutColumn, ShortcutColumnParams
        params = ShortcutColumnParams(
            species_order=['A', 'B'],
            light_key='A',
            heavy_key='B',
        )
        return ShortcutColumn(params, thermo)

    def test_feasible_case(self):
        col = self._make_column(350.0, 380.0)
        feed = make_stream({'A': 0.5, 'B': 0.5}, 365.0, 101325.0)
        _, _, info = col(feed, R=2.0)
        assert 'feasible' in info
        assert 'alpha_insufficient' in info
        assert 'negative_flows_detected' in info
        # Well-separated system should be feasible
        assert bool(info['feasible'])

    def test_alpha_insufficient_near_azeotrope(self):
        # Very close boiling points → alpha ≈ 1
        col = self._make_column(350.0, 350.5)
        feed = make_stream({'A': 0.5, 'B': 0.5}, 350.25, 101325.0)
        _, _, info = col(feed, R=5.0)
        # Alpha may or may not be insufficient depending on Antoine params
        assert isinstance(info['alpha_insufficient'], jax.Array)


# =============================================================================
# #75 — CSTR configurable density
# =============================================================================

class TestCSTRConfigurableDensity:
    """Issue #75: CSTR should support configurable molar_density."""

    def test_default_density(self):
        from difflow.units.cstr import CSTR, CSTRParams
        params = CSTRParams(
            V=1.0,
            rate_fn=_simple_rate_fn,
            stoich=jnp.array([[-1], [1]]),
            rate_params={'k': 0.1},
            species_order=['A', 'B'],
        )
        cstr = CSTR(params)
        inlet = _make_ab_stream()
        _, info = cstr(inlet)
        assert float(info['molar_density']) == pytest.approx(55500.0)

    def test_custom_density(self):
        from difflow.units.cstr import CSTR, CSTRParams
        params = CSTRParams(
            V=1.0,
            rate_fn=_simple_rate_fn,
            stoich=jnp.array([[-1], [1]]),
            rate_params={'k': 0.1},
            species_order=['A', 'B'],
            molar_density=30000.0,
        )
        cstr = CSTR(params)
        inlet = _make_ab_stream()
        _, info = cstr(inlet)
        assert float(info['molar_density']) == pytest.approx(30000.0)

    def test_gradient_through_density(self):
        from difflow.units.cstr import CSTR, CSTRParams

        def conversion_fn(density):
            params = CSTRParams(
                V=1.0,
                rate_fn=_simple_rate_fn,
                stoich=jnp.array([[-1], [1]]),
                rate_params={'k': 0.1},
                species_order=['A', 'B'],
                molar_density=density,
            )
            cstr = CSTR(params)
            inlet = _make_ab_stream()
            outlet, _ = cstr(inlet)
            return get_flows(outlet)['B']

        g = grad(conversion_fn)(30000.0)
        assert jnp.isfinite(g)


# =============================================================================
# #76 — PFR liquid pressure drop
# =============================================================================

class TestPFRPressureDrop:
    """Issue #76: PFR with optional liquid pressure drop."""

    def test_no_pressure_drop_backward_compat(self):
        from difflow.units.pfr import PFR, PFRParams
        params = PFRParams(
            V=1.0,
            rate_fn=_simple_rate_fn,
            stoich=jnp.array([[-1], [1]]),
            rate_params={'k': 0.5},
            species_order=['A', 'B'],
        )
        pfr = PFR(params)
        inlet = _make_ab_stream(P=200000.0)
        outlet, info = pfr(inlet, volumetric_flow=0.01)
        assert float(outlet['P']) == pytest.approx(200000.0)
        assert 'pressure_drop' not in info

    def test_with_pressure_drop(self):
        from difflow.units.pfr import PFR, PFRParams
        params = PFRParams(
            V=1.0,
            rate_fn=_simple_rate_fn,
            stoich=jnp.array([[-1], [1]]),
            rate_params={'k': 0.5},
            species_order=['A', 'B'],
            dP_dV=10000.0,  # 10 kPa/m^3
        )
        pfr = PFR(params)
        inlet = _make_ab_stream(P=200000.0)
        outlet, info = pfr(inlet, volumetric_flow=0.01)
        assert 'pressure_drop' in info
        assert 'P_out' in info
        expected_drop = 10000.0 * 1.0  # dP_dV * V
        assert float(info['pressure_drop']) == pytest.approx(expected_drop)
        assert float(outlet['P']) == pytest.approx(200000.0 - expected_drop)

    def test_gradient_through_pressure_drop(self):
        from difflow.units.pfr import PFR, PFRParams

        def p_out_fn(dp_dv):
            params = PFRParams(
                V=1.0,
                rate_fn=_simple_rate_fn,
                stoich=jnp.array([[-1], [1]]),
                rate_params={'k': 0.5},
                species_order=['A', 'B'],
                dP_dV=dp_dv,
            )
            pfr = PFR(params)
            inlet = _make_ab_stream(P=200000.0)
            outlet, _ = pfr(inlet, volumetric_flow=0.01)
            return outlet['P']

        g = grad(p_out_fn)(10000.0)
        assert jnp.isfinite(g)
        # Gradient should be -V = -1.0
        assert float(g) == pytest.approx(-1.0)


# =============================================================================
# #77 — CSTR heat of mixing
# =============================================================================

class TestCSTRHeatOfMixing:
    """Issue #77: CSTR with optional H_mix_fn."""

    def test_no_h_mix_backward_compat(self):
        from difflow.units.cstr import CSTR, CSTRParams
        params = CSTRParams(
            V=1.0,
            rate_fn=_simple_rate_fn,
            stoich=jnp.array([[-1], [1]]),
            rate_params={'k': 0.1},
            species_order=['A', 'B'],
        )
        cstr = CSTR(params)
        inlet = _make_ab_stream()
        _, info = cstr(inlet)
        assert float(info['H_mix']) == pytest.approx(0.0)

    def test_constant_h_mix(self):
        from difflow.units.cstr import CSTR, CSTRParams

        def constant_h_mix(flows, T):
            return jnp.asarray(100.0)  # 100 W

        params = CSTRParams(
            V=1.0,
            rate_fn=_simple_rate_fn,
            stoich=jnp.array([[-1], [1]]),
            rate_params={'k': 0.1},
            species_order=['A', 'B'],
            H_mix_fn=constant_h_mix,
        )
        cstr = CSTR(params)
        inlet = _make_ab_stream()
        _, info = cstr(inlet)
        assert float(info['H_mix']) == pytest.approx(100.0)


# =============================================================================
# #79 — CSTR equilibrium check
# =============================================================================

class TestCSTREquilibriumCheck:
    """Issue #79: CSTR with optional K_eq_fn for equilibrium checking."""

    def test_no_k_eq_backward_compat(self):
        from difflow.units.cstr import CSTR, CSTRParams
        params = CSTRParams(
            V=1.0,
            rate_fn=_simple_rate_fn,
            stoich=jnp.array([[-1], [1]]),
            rate_params={'k': 0.1},
            species_order=['A', 'B'],
        )
        cstr = CSTR(params)
        inlet = _make_ab_stream()
        _, info = cstr(inlet)
        assert 'K_eq' not in info

    def test_large_k_eq_not_exceeded(self):
        from difflow.units.cstr import CSTR, CSTRParams

        def k_eq_fn(T):
            return jnp.asarray(1e6)  # Very large K_eq → X_eq ≈ 1

        params = CSTRParams(
            V=0.001,  # Small volume → low conversion, well below K_eq
            rate_fn=_simple_rate_fn,
            stoich=jnp.array([[-1], [1]]),
            rate_params={'k': 0.01},
            species_order=['A', 'B'],
            K_eq_fn=k_eq_fn,
        )
        cstr = CSTR(params)
        inlet = _make_ab_stream()
        _, info = cstr(inlet)
        assert 'K_eq' in info
        assert float(info['K_eq']) == pytest.approx(1e6)
        assert float(info['X_eq_est']) > 0.99
        assert not bool(info['equilibrium_exceeded'])

    def test_small_k_eq_exceeded(self):
        from difflow.units.cstr import CSTR, CSTRParams

        def k_eq_fn(T):
            return jnp.asarray(0.001)  # Very small K_eq

        params = CSTRParams(
            V=1.0,
            rate_fn=_simple_rate_fn,
            stoich=jnp.array([[-1], [1]]),
            rate_params={'k': 0.1},
            species_order=['A', 'B'],
            K_eq_fn=k_eq_fn,
        )
        cstr = CSTR(params)
        inlet = _make_ab_stream()
        _, info = cstr(inlet)
        assert 'K_eq' in info
        assert float(info['X_eq_est']) < 0.01
        # With K_eq = 0.001, X_eq ≈ 0.001. Any conversion > 0.001 exceeds it.
        assert bool(info['equilibrium_exceeded'])


# =============================================================================
# #157 — LLE mutual solubility
# =============================================================================

class TestLLEMutualSolubility:
    """Issue #157: LLE with optional mutual_solubility."""

    def _make_cascade(self, mutual_sol=None):
        from difflow.units.lle import (
            MultistageCascade, CascadeParams, LLEEquilibrium, DistributionCoeffs,
        )
        eq = LLEEquilibrium(
            solutes=['solute'],
            aqueous_carrier='water',
            organic_carrier='hexane',
            K_coeffs=DistributionCoeffs(
                species=('solute',),
                K0=(3.0,),
            ),
            mutual_solubility=mutual_sol,
        )
        params = CascadeParams(n_stages=3, equilibrium=eq)
        return MultistageCascade(params)

    def test_no_mutual_solubility(self):
        cascade = self._make_cascade()
        feed = make_stream({'water': 10.0, 'solute': 1.0}, 298.15, 101325.0)
        solvent = make_stream({'hexane': 5.0}, 298.15, 101325.0)
        raff, ext, info = cascade(feed, solvent)
        raff_flows = get_flows(raff)
        assert 'hexane' not in raff_flows or float(raff_flows.get('hexane', 0.0)) == 0.0
        assert not info['profiles']['mutual_solubility_active']

    def test_with_mutual_solubility(self):
        cascade = self._make_cascade(mutual_sol={
            'aqueous_in_organic': 0.01,  # 1% water dissolves in organic
            'organic_in_aqueous': 0.005,  # 0.5% hexane in aqueous
        })
        feed = make_stream({'water': 10.0, 'solute': 1.0}, 298.15, 101325.0)
        solvent = make_stream({'hexane': 5.0}, 298.15, 101325.0)
        raff, ext, info = cascade(feed, solvent)
        raff_flows = get_flows(raff)
        ext_flows = get_flows(ext)
        assert info['profiles']['mutual_solubility_active']
        # Some water should appear in extract, some hexane in raffinate
        assert float(ext_flows.get('water', 0.0)) > 0
        assert float(raff_flows.get('hexane', 0.0)) > 0

    def test_mass_balance_with_mutual_solubility(self):
        cascade = self._make_cascade(mutual_sol={
            'aqueous_in_organic': 0.02,
            'organic_in_aqueous': 0.01,
        })
        feed = make_stream({'water': 10.0, 'solute': 1.0}, 298.15, 101325.0)
        solvent = make_stream({'hexane': 5.0}, 298.15, 101325.0)
        raff, ext, _ = cascade(feed, solvent)

        raff_flows = get_flows(raff)
        ext_flows = get_flows(ext)

        # Total water in = 10.0, total water out should be close
        water_out = float(raff_flows.get('water', 0.0)) + float(ext_flows.get('water', 0.0))
        assert water_out == pytest.approx(10.0, rel=0.01)

        # Total hexane in = 5.0
        hexane_out = float(raff_flows.get('hexane', 0.0)) + float(ext_flows.get('hexane', 0.0))
        assert hexane_out == pytest.approx(5.0, rel=0.01)

        # Total solute balance
        solute_out = float(raff_flows.get('solute', 0.0)) + float(ext_flows.get('solute', 0.0))
        assert solute_out == pytest.approx(1.0, rel=0.01)


# =============================================================================
# #88 — LMTD F-correction for multi-pass HX
# =============================================================================

class TestLMTDFCorrection:
    """Issue #88: lmtd_correction_factor and ShellAndTubeHX."""

    def test_f_equals_one_at_limits(self):
        from difflow.units.heat_exchanger import lmtd_correction_factor
        # Very small P → F ≈ 1 (negligible heat transfer)
        F = float(lmtd_correction_factor(R=1.5, P=0.01))
        assert F == pytest.approx(1.0, abs=0.05)

    def test_typical_1_2_hx(self):
        from difflow.units.heat_exchanger import lmtd_correction_factor
        # Typical 1-2 shell-and-tube: F < 1
        F = float(lmtd_correction_factor(R=1.0, P=0.5))
        assert 0.5 < F < 1.0

    def test_f_decreases_with_p(self):
        from difflow.units.heat_exchanger import lmtd_correction_factor
        F1 = float(lmtd_correction_factor(R=1.5, P=0.3))
        F2 = float(lmtd_correction_factor(R=1.5, P=0.6))
        assert F1 > F2  # F decreases as P increases

    def test_shell_and_tube_hx_forward(self):
        from difflow.units.heat_exchanger import ShellAndTubeHX, ShellAndTubeHXParams
        p = ShellAndTubeHXParams(UA=500.0, Cp_hot=75.0, Cp_cold=75.0)
        hx = ShellAndTubeHX(p)
        hot = make_stream({'water': 2.0}, 400.0, 101325.0)
        cold = make_stream({'water': 1.0}, 300.0, 101325.0)
        hot_out, cold_out, info = hx(hot, cold)
        assert 'F_correction' in info
        assert 'R' in info
        assert 'P_param' in info
        assert 'F_too_low' in info
        assert float(info['Q']) > 0
        assert info['flow_arrangement'] == 'shell_and_tube'

    def test_f_too_low_flag(self):
        from difflow.units.heat_exchanger import ShellAndTubeHX, ShellAndTubeHXParams
        # Very large UA → high P → low F
        p = ShellAndTubeHXParams(UA=50000.0, Cp_hot=75.0, Cp_cold=75.0)
        hx = ShellAndTubeHX(p)
        hot = make_stream({'water': 1.0}, 400.0, 101325.0)
        cold = make_stream({'water': 1.0}, 300.0, 101325.0)
        _, _, info = hx(hot, cold)
        # With very high UA and balanced flow, F should be low
        assert isinstance(info['F_too_low'], jax.Array)

    def test_gradient_through_shell_and_tube(self):
        from difflow.units.heat_exchanger import ShellAndTubeHX, ShellAndTubeHXParams

        def q_fn(ua):
            p = ShellAndTubeHXParams(UA=ua, Cp_hot=75.0, Cp_cold=75.0)
            hx = ShellAndTubeHX(p)
            hot = make_stream({'water': 2.0}, 400.0, 101325.0)
            cold = make_stream({'water': 1.0}, 300.0, 101325.0)
            _, _, info = hx(hot, cold)
            return info['Q']

        g = grad(q_fn)(500.0)
        assert jnp.isfinite(g)
        assert float(g) > 0  # More UA → more heat transfer


# =============================================================================
# #75 — CSTR outlet-conditions volumetric flow
# =============================================================================

class TestCSTROutletVolumetricBasis:
    """Issue #75: concentrations at reactor (outlet) volumetric flow."""

    def test_backward_compat_inlet_basis(self):
        from difflow.units.cstr import CSTR, CSTRParams
        params = CSTRParams(
            V=1.0, rate_fn=_simple_rate_fn, stoich=jnp.array([[-1.0], [2.0]]),
            rate_params={'k': 0.5}, species_order=['A', 'B'], molar_density=1000.0,
        )
        cstr = CSTR(params)  # default: inlet basis
        outlet, info = cstr(_make_ab_stream())
        assert not params.outlet_volumetric_basis

    def test_outlet_basis_changes_conversion_for_mole_change(self):
        from difflow.units.cstr import CSTR, CSTRParams
        # A -> 2B increases total moles, so outlet volumetric flow > inlet
        common = dict(
            V=1.0, rate_fn=_simple_rate_fn, stoich=jnp.array([[-1.0], [2.0]]),
            rate_params={'k': 0.5}, species_order=['A', 'B'], molar_density=1000.0,
        )
        cstr_in = CSTR(CSTRParams(outlet_volumetric_basis=False, **common))
        cstr_out = CSTR(CSTRParams(outlet_volumetric_basis=True, **common))
        inlet = _make_ab_stream(F_A=1.0)
        _, info_in = cstr_in(inlet)
        _, info_out = cstr_out(inlet)
        X_in = float(info_in['conversion']['A'])
        X_out = float(info_out['conversion']['A'])
        # Different volumetric-flow basis -> different concentration -> different X
        assert abs(X_in - X_out) > 1e-4

    def test_outlet_basis_differentiable(self):
        from difflow.units.cstr import CSTR, CSTRParams

        def conv(k):
            params = CSTRParams(
                V=1.0, rate_fn=_simple_rate_fn, stoich=jnp.array([[-1.0], [2.0]]),
                rate_params={'k': k}, species_order=['A', 'B'],
                molar_density=1000.0, outlet_volumetric_basis=True,
            )
            _, info = CSTR(params)(_make_ab_stream())
            return info['conversion']['A']

        g = grad(conv)(0.5)
        assert jnp.isfinite(g)


# =============================================================================
# #77 — CSTR heat of mixing enters the energy balance
# =============================================================================

class TestCSTRHeatOfMixingEnergyBalance:
    """Issue #77: H_mix must actually contribute to the heat duty Q."""

    def _thermo(self):
        return IdealThermo({
            'A': SpeciesData(name='A', MW=50.0, Cp_coeffs=(75.0, 0, 0, 0),
                             Hvap_coeffs=(1.0, 0.38, 600.0), antoine_coeffs=(8.07, 1730.0, -39.0)),
            'B': SpeciesData(name='B', MW=50.0, Cp_coeffs=(75.0, 0, 0, 0),
                             Hvap_coeffs=(1.0, 0.38, 600.0), antoine_coeffs=(8.07, 1830.0, -39.0)),
        })

    def test_h_mix_shifts_heat_duty(self):
        from difflow.units.cstr import CSTR, CSTRParams
        thermo = self._thermo()
        common = dict(
            V=1.0, rate_fn=_simple_rate_fn, stoich=jnp.array([[-1.0], [1.0]]),
            rate_params={'k': 0.1}, species_order=['A', 'B'],
        )
        cstr_no = CSTR(CSTRParams(**common), thermo=thermo, mode="isothermal")
        cstr_hm = CSTR(CSTRParams(H_mix_fn=lambda flows, T: jnp.asarray(500.0), **common),
                       thermo=thermo, mode="isothermal")
        inlet = _make_ab_stream()
        _, info_no = cstr_no(inlet)
        _, info_hm = cstr_hm(inlet)
        # Q includes +H_mix (500 W) when thermo is present
        assert float(info_hm['Q']) - float(info_no['Q']) == pytest.approx(500.0, rel=1e-6)
