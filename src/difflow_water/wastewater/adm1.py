"""Anaerobic Digestion Model No. 1 (ADM1), the BSM2 implementation, at steady state.

Transcribed from Rosen & Jeppsson, *Aspects on ADM1 implementation within
the BSM2 framework* (Lund University, TEIE-7224, 2006) [W6], read: content
verified. That report is the benchmark implementation of Batstone et al.
(2002) [W2] with closed C and N balances (its Section 3.1.1), temperature
dependence of the physico-chemical constants (3.1.3), the hydrogen-ion
Hill functions for pH inhibition (3.2.2, the BSM2 choice), the process
rates (3.2.1), water and gas phase balances (3.2.3-3.2.4), the head-space
overpressure gas flow (the BSM2 choice) and its parameter tables (Section
6). Its Section 7 is the steady-state verification case this module is
compared with (``benchmarks/watertap/compare_adm1.py``).

Formulation (the report's DAE with an algebraic pH): the acid-base pairs
are at equilibrium, ``S_H+`` solves the charge balance at every state, and
the 24 water-phase states plus 3 gas-phase states are the unknowns. Units
as in the report: kg COD/m^3, kmol C/m^3 (``S_IC``), kmol N/m^3 (``S_IN``),
kmol/m^3 (``S_cat``, ``S_an``), days, bar.

Streams: plain difflow streams. ``F_H2O`` carries the volume
(``Q = F_H2O M_w / 1000``); ``F_<state> = 1000 Q C`` (so ``1 mol = 1 g`` of
COD for COD states and a real mole for ``S_IC``, ``S_IN``, ``S_cat``,
``S_an``). The biogas outlet carries ``F_CH4``, ``F_CO2``, ``F_H2``,
``F_H2O`` in mol/s.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, make_stream
from difflow_water.chemistry.seawater import MW_WATER

#: ADM1 water-phase states in the report's order (1-24), then cations/anions (25-26)
STATES = ["S_su", "S_aa", "S_fa", "S_va", "S_bu", "S_pro", "S_ac", "S_h2", "S_ch4", "S_IC",
          "S_IN", "S_I", "X_c", "X_ch", "X_pr", "X_li", "X_su", "X_aa", "X_fa", "X_c4", "X_pro",
          "X_ac", "X_h2", "X_I", "S_cat", "S_an"]
GAS = ["S_gas_h2", "S_gas_ch4", "S_gas_co2"]
I = {s: i for i, s in enumerate(STATES)}  # noqa: E741
RHO = 1000.0


@dataclass(repr=False)
class ADM1Params(ParamsMixin):
    """ADM1 benchmark parameters ([W6] Section 6) and the digester.

    Attributes:
        V_liq: Liquid volume (m^3).
        V_gas: Head-space volume (m^3).
        T_op: Operating temperature (K).
        f_sI_xc: Soluble inerts from composites (-).
        f_xI_xc: Particulate inerts from composites (-).
        f_ch_xc: Carbohydrates from composites (-).
        f_pr_xc: Proteins from composites (-).
        f_li_xc: Lipids from composites (-).
        N_xc: N content of composites (kmol N / kg COD).
        N_I: N content of inerts (kmol N / kg COD).
        N_aa: N content of amino acids (kmol N / kg COD).
        N_bac: N content of biomass (kmol N / kg COD).
        C_xc: C content of composites (kmol C / kg COD).
        C_sI: C content of soluble inerts (kmol C / kg COD).
        C_ch: C content of carbohydrates (kmol C / kg COD).
        C_pr: C content of proteins (kmol C / kg COD).
        C_li: C content of lipids (kmol C / kg COD).
        C_xI: C content of particulate inerts (kmol C / kg COD).
        C_su: C content of sugars (kmol C / kg COD).
        C_aa: C content of amino acids (kmol C / kg COD).
        C_fa: C content of LCFA (kmol C / kg COD).
        C_bu: C content of butyrate (kmol C / kg COD).
        C_pro: C content of propionate (kmol C / kg COD).
        C_ac: C content of acetate (kmol C / kg COD).
        C_bac: C content of biomass (kmol C / kg COD).
        C_va: C content of valerate (kmol C / kg COD).
        C_ch4: C content of methane (kmol C / kg COD).
        f_fa_li: LCFA from lipids (-).
        f_h2_su: H2 from sugars (-).
        f_bu_su: Butyrate from sugars (-).
        f_pro_su: Propionate from sugars (-).
        f_ac_su: Acetate from sugars (-).
        f_h2_aa: H2 from amino acids (-).
        f_va_aa: Valerate from amino acids (-).
        f_bu_aa: Butyrate from amino acids (-).
        f_pro_aa: Propionate from amino acids (-).
        f_ac_aa: Acetate from amino acids (-).
        Y_su: Sugar degrader yield (-).
        Y_aa: Amino acid degrader yield (-).
        Y_fa: LCFA degrader yield (-).
        Y_c4: Valerate/butyrate degrader yield (-).
        Y_pro: Propionate degrader yield (-).
        Y_ac: Acetoclastic methanogen yield (-).
        Y_h2: Hydrogenotrophic methanogen yield (-).
        k_dis: Disintegration rate (1/d).
        k_hyd_ch: Carbohydrate hydrolysis rate (1/d).
        k_hyd_pr: Protein hydrolysis rate (1/d).
        k_hyd_li: Lipid hydrolysis rate (1/d).
        K_S_IN: Inorganic N half-saturation (kmol/m^3).
        k_m_su: Sugar uptake rate (1/d).
        K_S_su: Sugar half-saturation (kg COD/m^3).
        pH_UL_aa: Upper pH limit, acidogens (-).
        pH_LL_aa: Lower pH limit, acidogens (-).
        k_m_aa: Amino acid uptake rate (1/d).
        K_S_aa: Amino acid half-saturation (kg COD/m^3).
        k_m_fa: LCFA uptake rate (1/d).
        K_S_fa: LCFA half-saturation (kg COD/m^3).
        K_I_h2_fa: H2 inhibition of LCFA uptake (kg COD/m^3).
        k_m_c4: Valerate/butyrate uptake rate (1/d).
        K_S_c4: Valerate/butyrate half-saturation (kg COD/m^3).
        K_I_h2_c4: H2 inhibition of c4 uptake (kg COD/m^3).
        k_m_pro: Propionate uptake rate (1/d).
        K_S_pro: Propionate half-saturation (kg COD/m^3).
        K_I_h2_pro: H2 inhibition of propionate uptake (kg COD/m^3).
        k_m_ac: Acetate uptake rate (1/d).
        K_S_ac: Acetate half-saturation (kg COD/m^3).
        K_I_nh3: Free ammonia inhibition (kmol/m^3).
        pH_UL_ac: Upper pH limit, acetoclasts (-).
        pH_LL_ac: Lower pH limit, acetoclasts (-).
        k_m_h2: H2 uptake rate (1/d).
        K_S_h2: H2 half-saturation (kg COD/m^3).
        pH_UL_h2: Upper pH limit, hydrogenotrophs (-).
        pH_LL_h2: Lower pH limit, hydrogenotrophs (-).
        k_dec: Biomass decay rate, all seven groups (1/d).
        k_La: Gas-liquid transfer coefficient (1/d).
        k_p: Gas outlet friction (m^3/(d bar)).
        P_atm: Atmospheric pressure (bar).
        gas_flow: ``"compensated"``: ``q_gas = k_p (P_gas - P_atm) P_gas / P_atm``,
            the BSM2 choice as [W6] Section 3.2.4 states it and as WaterTAP
            codes it; ``"uncompensated"``: ``k_p (P_gas - P_atm)``, which is
            what reproduces [W6]'s Section 7 steady-state results (with the
            compensated form its gas balances there are off by 5 %).
        pH_inhibition: ``"hill"`` (the BSM2 choice, [W6] Expression 3) or
            ``"switch"`` (Batstone et al.'s form, ``exp(-3 ((pH - pH_UL) /
            (pH_UL - pH_LL))^2)`` below ``pH_UL``, as [W6] 3.2.2 states it and
            WaterTAP codes it).
    """

    V_liq: float | Array = 3400.0
    V_gas: float | Array = 300.0
    T_op: float | Array = 308.15
    f_sI_xc: float | Array = 0.1
    f_xI_xc: float | Array = 0.2
    f_ch_xc: float | Array = 0.2
    f_pr_xc: float | Array = 0.2
    f_li_xc: float | Array = 0.3
    N_xc: float | Array = 0.0376 / 14.0
    N_I: float | Array = 0.06 / 14.0
    N_aa: float | Array = 0.007
    N_bac: float | Array = 0.08 / 14.0
    C_xc: float | Array = 0.02786
    C_sI: float | Array = 0.03
    C_ch: float | Array = 0.0313
    C_pr: float | Array = 0.03
    C_li: float | Array = 0.022
    C_xI: float | Array = 0.03
    C_su: float | Array = 0.0313
    C_aa: float | Array = 0.03
    C_fa: float | Array = 0.0217
    C_bu: float | Array = 0.025
    C_pro: float | Array = 0.0268
    C_ac: float | Array = 0.0313
    C_bac: float | Array = 0.0313
    C_va: float | Array = 0.024
    C_ch4: float | Array = 0.0156
    f_fa_li: float | Array = 0.95
    f_h2_su: float | Array = 0.19
    f_bu_su: float | Array = 0.13
    f_pro_su: float | Array = 0.27
    f_ac_su: float | Array = 0.41
    f_h2_aa: float | Array = 0.06
    f_va_aa: float | Array = 0.23
    f_bu_aa: float | Array = 0.26
    f_pro_aa: float | Array = 0.05
    f_ac_aa: float | Array = 0.40
    Y_su: float | Array = 0.1
    Y_aa: float | Array = 0.08
    Y_fa: float | Array = 0.06
    Y_c4: float | Array = 0.06
    Y_pro: float | Array = 0.04
    Y_ac: float | Array = 0.05
    Y_h2: float | Array = 0.06
    k_dis: float | Array = 0.5
    k_hyd_ch: float | Array = 10.0
    k_hyd_pr: float | Array = 10.0
    k_hyd_li: float | Array = 10.0
    K_S_IN: float | Array = 1e-4
    k_m_su: float | Array = 30.0
    K_S_su: float | Array = 0.5
    pH_UL_aa: float | Array = 5.5
    pH_LL_aa: float | Array = 4.0
    k_m_aa: float | Array = 50.0
    K_S_aa: float | Array = 0.3
    k_m_fa: float | Array = 6.0
    K_S_fa: float | Array = 0.4
    K_I_h2_fa: float | Array = 5e-6
    k_m_c4: float | Array = 20.0
    K_S_c4: float | Array = 0.2
    K_I_h2_c4: float | Array = 1e-5
    k_m_pro: float | Array = 13.0
    K_S_pro: float | Array = 0.1
    K_I_h2_pro: float | Array = 3.5e-6
    k_m_ac: float | Array = 8.0
    K_S_ac: float | Array = 0.15
    K_I_nh3: float | Array = 0.0018
    pH_UL_ac: float | Array = 7.0
    pH_LL_ac: float | Array = 6.0
    k_m_h2: float | Array = 35.0
    K_S_h2: float | Array = 7e-6
    pH_UL_h2: float | Array = 6.0
    pH_LL_h2: float | Array = 5.0
    k_dec: float | Array = 0.02
    k_La: float | Array = 200.0
    k_p: float | Array = 5e4
    P_atm: float | Array = 1.013
    gas_flow: str = "compensated"
    pH_inhibition: str = "hill"


R_ADM = 0.083145   # bar m^3 / (kmol K) [W6]
T_BASE = 298.15


def constants(p: ADM1Params) -> dict:
    """Temperature-dependent physico-chemical constants ([W6] Section 6)."""
    T = p.T_op
    d = 1.0 / T_BASE - 1.0 / T
    f = 1.0 / (R_ADM * 100.0)
    return {
        "K_w": 1e-14 * jnp.exp(55900.0 * f * d),
        "K_a_va": 10.0 ** -4.86, "K_a_bu": 10.0 ** -4.82, "K_a_pro": 10.0 ** -4.88,
        "K_a_ac": 10.0 ** -4.76,
        "K_a_co2": 10.0 ** -6.35 * jnp.exp(7646.0 * f * d),
        "K_a_IN": 10.0 ** -9.25 * jnp.exp(51965.0 * f * d),
        "p_gas_h2o": 0.0313 * jnp.exp(5290.0 * d),
        "K_H_co2": 0.035 * jnp.exp(-19410.0 * f * d),
        "K_H_ch4": 0.0014 * jnp.exp(-14240.0 * f * d),
        "K_H_h2": 7.8e-4 * jnp.exp(-4180.0 * f * d),
    }


def _ions(S, SH, c):
    """Dissociated forms at equilibrium with S_H+ (the DAE-pH formulation)."""
    return {
        "va": c["K_a_va"] * S[I["S_va"]] / (c["K_a_va"] + SH),
        "bu": c["K_a_bu"] * S[I["S_bu"]] / (c["K_a_bu"] + SH),
        "pro": c["K_a_pro"] * S[I["S_pro"]] / (c["K_a_pro"] + SH),
        "ac": c["K_a_ac"] * S[I["S_ac"]] / (c["K_a_ac"] + SH),
        "hco3": c["K_a_co2"] * S[I["S_IC"]] / (c["K_a_co2"] + SH),
        "nh3": c["K_a_IN"] * S[I["S_IN"]] / (c["K_a_IN"] + SH),
    }


def charge_balance_SH(S, c, n_iter: int = 80):
    """``S_H+`` (kmol/m^3) from the charge balance ([W6] 3.2.3).

    Electroneutrality ``S_H + Theta - K_w / S_H = 0`` with
    ``Theta = S_cat + S_nh4 - S_hco3 - S_ac/64 - S_pro/112 - S_bu/160 - S_va/208 - S_an``
    and the ions in equilibrium with S_H (the report's algebraic pH; its
    closed form ``-Theta/2 + sqrt(Theta^2 + 4 K_w)/2`` is the same equation).
    The left side increases monotonically with S_H, so bisection on
    ``ln S_H`` in [1e-14, 1] finds the one root; a value-neutral Newton step
    from the stop-gradiented root carries first-order implicit gradients.
    """
    def resid(SH):
        ion = _ions(S, SH, c)
        nh4 = S[I["S_IN"]] - ion["nh3"]
        theta = (S[I["S_cat"]] + nh4 - ion["hco3"] - ion["ac"] / 64.0 - ion["pro"] / 112.0
                 - ion["bu"] / 160.0 - ion["va"] / 208.0 - S[I["S_an"]])
        return SH + theta - c["K_w"] / SH

    Sg = jax.lax.stop_gradient(S)
    cg = {k: jax.lax.stop_gradient(v) for k, v in c.items()}

    def resid_g(SH):
        ion = _ions(Sg, SH, cg)
        nh4 = Sg[I["S_IN"]] - ion["nh3"]
        theta = (Sg[I["S_cat"]] + nh4 - ion["hco3"] - ion["ac"] / 64.0 - ion["pro"] / 112.0
                 - ion["bu"] / 160.0 - ion["va"] / 208.0 - Sg[I["S_an"]])
        return SH + theta - cg["K_w"] / SH

    def body(_, lohi):
        lo, hi = lohi
        mid = 0.5 * (lo + hi)
        up = resid_g(jnp.exp(mid)) > 0.0
        return jnp.where(up, lo, mid), jnp.where(up, mid, hi)

    lo, hi = jax.lax.fori_loop(0, n_iter, body, (jnp.log(1e-14), jnp.log(1.0)))
    SH0 = jax.lax.stop_gradient(jnp.exp(0.5 * (lo + hi)))
    f, df = jax.jvp(resid, (SH0,), (jnp.ones_like(SH0),))
    return SH0 - (f - jax.lax.stop_gradient(f)) / jax.lax.stop_gradient(df)


def _hill(SH, UL, LL):
    """BSM2 hydrogen-ion pH inhibition ([W6] Expression 3)."""
    n = 3.0 / (UL - LL)
    K = 10.0 ** (-(LL + UL) / 2.0)
    return K**n / (SH**n + K**n)


def rhs(S, G, q_in, S_in, p: ADM1Params):
    """dS/dt and dG/dt of the water (26) and gas (3) phases, plus diagnostics."""
    c = constants(p)
    SH = charge_balance_SH(S, c)
    ion = _ions(S, SH, c)
    s = {k: S[i] for k, i in I.items()}
    S_co2 = s["S_IC"] - ion["hco3"]
    I_IN = 1.0 / (1.0 + p.K_S_IN / s["S_IN"])
    if p.pH_inhibition == "switch":
        pH = -jnp.log10(SH)

        def inh(UL, LL):
            return jnp.exp(-3.0 * (jnp.maximum(0.0, UL - pH) / (UL - LL)) ** 2)
    else:
        def inh(UL, LL):
            return _hill(SH, UL, LL)
    I_aa = inh(p.pH_UL_aa, p.pH_LL_aa) * I_IN
    I_ac = inh(p.pH_UL_ac, p.pH_LL_ac) * I_IN / (1.0 + ion["nh3"] / p.K_I_nh3)
    I_h2 = inh(p.pH_UL_h2, p.pH_LL_h2) * I_IN
    mon = lambda x, K: x / (K + x)  # noqa: E731
    r = jnp.stack([
        p.k_dis * s["X_c"],
        p.k_hyd_ch * s["X_ch"],
        p.k_hyd_pr * s["X_pr"],
        p.k_hyd_li * s["X_li"],
        p.k_m_su * mon(s["S_su"], p.K_S_su) * s["X_su"] * I_aa,
        p.k_m_aa * mon(s["S_aa"], p.K_S_aa) * s["X_aa"] * I_aa,
        p.k_m_fa * mon(s["S_fa"], p.K_S_fa) * s["X_fa"] * I_aa / (1.0 + s["S_h2"] / p.K_I_h2_fa),
        p.k_m_c4 * mon(s["S_va"], p.K_S_c4) * s["X_c4"] * s["S_va"] / (s["S_bu"] + s["S_va"] + 1e-6)
        * I_aa / (1.0 + s["S_h2"] / p.K_I_h2_c4),
        p.k_m_c4 * mon(s["S_bu"], p.K_S_c4) * s["X_c4"] * s["S_bu"] / (s["S_va"] + s["S_bu"] + 1e-6)
        * I_aa / (1.0 + s["S_h2"] / p.K_I_h2_c4),
        p.k_m_pro * mon(s["S_pro"], p.K_S_pro) * s["X_pro"] * I_aa / (1.0 + s["S_h2"] / p.K_I_h2_pro),
        p.k_m_ac * mon(s["S_ac"], p.K_S_ac) * s["X_ac"] * I_ac,
        p.k_m_h2 * mon(s["S_h2"], p.K_S_h2) * s["X_h2"] * I_h2,
        p.k_dec * s["X_su"], p.k_dec * s["X_aa"], p.k_dec * s["X_fa"], p.k_dec * s["X_c4"],
        p.k_dec * s["X_pro"], p.k_dec * s["X_ac"], p.k_dec * s["X_h2"],
    ])
    # gas phase
    RT = R_ADM * p.T_op
    p_h2, p_ch4, p_co2 = G[0] * RT / 16.0, G[1] * RT / 64.0, G[2] * RT
    P_gas = p_h2 + p_ch4 + p_co2 + c["p_gas_h2o"]
    q_gas = p.k_p * (P_gas - p.P_atm)
    if p.gas_flow == "compensated":
        q_gas = q_gas * P_gas / p.P_atm
    rT8 = p.k_La * (s["S_h2"] - 16.0 * c["K_H_h2"] * p_h2)
    rT9 = p.k_La * (s["S_ch4"] - 64.0 * c["K_H_ch4"] * p_ch4)
    rT10 = p.k_La * (S_co2 - c["K_H_co2"] * p_co2)
    # carbon stoichiometry ([W6] Eq. 10)
    Y = p
    s1 = (-p.C_xc + p.f_sI_xc * p.C_sI + p.f_ch_xc * p.C_ch + p.f_pr_xc * p.C_pr
          + p.f_li_xc * p.C_li + p.f_xI_xc * p.C_xI)
    s2 = -p.C_ch + p.C_su
    s3 = -p.C_pr + p.C_aa
    s4 = -p.C_li + (1 - p.f_fa_li) * p.C_su + p.f_fa_li * p.C_fa
    s5 = -p.C_su + (1 - Y.Y_su) * (p.f_bu_su * p.C_bu + p.f_pro_su * p.C_pro + p.f_ac_su * p.C_ac) \
        + Y.Y_su * p.C_bac
    s6 = -p.C_aa + (1 - Y.Y_aa) * (p.f_va_aa * p.C_va + p.f_bu_aa * p.C_bu + p.f_pro_aa * p.C_pro
                                   + p.f_ac_aa * p.C_ac) + Y.Y_aa * p.C_bac
    s7 = -p.C_fa + (1 - Y.Y_fa) * 0.7 * p.C_ac + Y.Y_fa * p.C_bac
    s8 = -p.C_va + (1 - Y.Y_c4) * 0.54 * p.C_pro + (1 - Y.Y_c4) * 0.31 * p.C_ac + Y.Y_c4 * p.C_bac
    s9 = -p.C_bu + (1 - Y.Y_c4) * 0.8 * p.C_ac + Y.Y_c4 * p.C_bac
    s10 = -p.C_pro + (1 - Y.Y_pro) * 0.57 * p.C_ac + Y.Y_pro * p.C_bac
    s11 = -p.C_ac + (1 - Y.Y_ac) * p.C_ch4 + Y.Y_ac * p.C_bac
    s12 = (1 - Y.Y_h2) * p.C_ch4 + Y.Y_h2 * p.C_bac
    s13 = -p.C_bac + p.C_xc
    sum_C = (s1 * r[0] + s2 * r[1] + s3 * r[2] + s4 * r[3] + s5 * r[4] + s6 * r[5] + s7 * r[6]
             + s8 * r[7] + s9 * r[8] + s10 * r[9] + s11 * r[10] + s12 * r[11]
             + s13 * jnp.sum(r[12:19]))
    D = q_in / p.V_liq
    q_out = q_in   # liquid outflow = inflow ([W6]; WaterTAP removes evaporated water)
    D_out = q_out / p.V_liq
    dec = jnp.sum(r[12:19])
    dS = {
        "S_su": r[1] + (1 - p.f_fa_li) * r[3] - r[4],
        "S_aa": r[2] - r[5],
        "S_fa": p.f_fa_li * r[3] - r[6],
        "S_va": (1 - Y.Y_aa) * p.f_va_aa * r[5] - r[7],
        "S_bu": (1 - Y.Y_su) * p.f_bu_su * r[4] + (1 - Y.Y_aa) * p.f_bu_aa * r[5] - r[8],
        "S_pro": ((1 - Y.Y_su) * p.f_pro_su * r[4] + (1 - Y.Y_aa) * p.f_pro_aa * r[5]
                  + (1 - Y.Y_c4) * 0.54 * r[7] - r[9]),
        "S_ac": ((1 - Y.Y_su) * p.f_ac_su * r[4] + (1 - Y.Y_aa) * p.f_ac_aa * r[5]
                 + (1 - Y.Y_fa) * 0.7 * r[6] + (1 - Y.Y_c4) * 0.31 * r[7]
                 + (1 - Y.Y_c4) * 0.8 * r[8] + (1 - Y.Y_pro) * 0.57 * r[9] - r[10]),
        "S_h2": ((1 - Y.Y_su) * p.f_h2_su * r[4] + (1 - Y.Y_aa) * p.f_h2_aa * r[5]
                 + (1 - Y.Y_fa) * 0.3 * r[6] + (1 - Y.Y_c4) * 0.15 * r[7]
                 + (1 - Y.Y_c4) * 0.2 * r[8] + (1 - Y.Y_pro) * 0.43 * r[9] - r[11] - rT8),
        "S_ch4": (1 - Y.Y_ac) * r[10] + (1 - Y.Y_h2) * r[11] - rT9,
        "S_IC": -sum_C - rT10,
        "S_IN": (-Y.Y_su * p.N_bac * r[4] + (p.N_aa - Y.Y_aa * p.N_bac) * r[5]
                 - Y.Y_fa * p.N_bac * r[6] - Y.Y_c4 * p.N_bac * r[7] - Y.Y_c4 * p.N_bac * r[8]
                 - Y.Y_pro * p.N_bac * r[9] - Y.Y_ac * p.N_bac * r[10] - Y.Y_h2 * p.N_bac * r[11]
                 + (p.N_bac - p.N_xc) * dec
                 + (p.N_xc - p.f_xI_xc * p.N_I - p.f_sI_xc * p.N_I - p.f_pr_xc * p.N_aa) * r[0]),
        "S_I": p.f_sI_xc * r[0],
        "X_c": -r[0] + dec,
        "X_ch": p.f_ch_xc * r[0] - r[1],
        "X_pr": p.f_pr_xc * r[0] - r[2],
        "X_li": p.f_li_xc * r[0] - r[3],
        "X_su": Y.Y_su * r[4] - r[12],
        "X_aa": Y.Y_aa * r[5] - r[13],
        "X_fa": Y.Y_fa * r[6] - r[14],
        "X_c4": Y.Y_c4 * r[7] + Y.Y_c4 * r[8] - r[15],
        "X_pro": Y.Y_pro * r[9] - r[16],
        "X_ac": Y.Y_ac * r[10] - r[17],
        "X_h2": Y.Y_h2 * r[11] - r[18],
        "X_I": p.f_xI_xc * r[0],
        "S_cat": 0.0 * r[0],
        "S_an": 0.0 * r[0],
    }
    dSdt = D * S_in - D_out * S + jnp.stack([dS[k] for k in STATES])
    ratio = p.V_liq / p.V_gas
    dGdt = jnp.stack([-G[0] * q_gas / p.V_gas + rT8 * ratio,
                      -G[1] * q_gas / p.V_gas + rT9 * ratio,
                      -G[2] * q_gas / p.V_gas + rT10 * ratio])
    diag = {"S_H": SH, "pH": -jnp.log10(SH), "ions": ion, "S_co2": S_co2, "S_nh4": s["S_IN"] - ion["nh3"],
            "p_gas_h2": p_h2, "p_gas_ch4": p_ch4, "p_gas_co2": p_co2, "P_gas": P_gas,
            "q_gas": q_gas, "rates": r, "p_gas_h2o": c["p_gas_h2o"], "q_out": q_out}
    return dSdt, dGdt, diag


def _sg(obj):
    if dataclasses.is_dataclass(obj):
        return dataclasses.replace(obj, **{f.name: _sg(getattr(obj, f.name))
                                           for f in dataclasses.fields(obj)})
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return jax.lax.stop_gradient(obj)


def stream_to_state(stream: Stream) -> tuple[Array, Array]:
    """``(q m^3/d, S)`` of an ADM1 stream (states absent from it are zero)."""
    q = stream["F_H2O"] * MW_WATER / RHO * 86400.0
    S = jnp.stack([jnp.asarray(stream.get(f"F_{s}", 0.0), dtype=jnp.float64) for s in STATES]) \
        * 86400.0 / (1000.0 * q)
    return q, S


def state_to_stream(q, S, T=308.15, P=101325.0) -> Stream:
    """An ADM1 stream of flow q (m^3/d) and concentrations S (report units)."""
    flows = {"H2O": q / 86400.0 * RHO / MW_WATER}
    flows.update({s: 1000.0 * q * S[i] / 86400.0 for i, s in enumerate(STATES)})
    return make_stream(flows, T=T, P=P)


#: The report's steady-state verification influent ([W6] Section 7), q = 170 m^3/d, 35 C
BSM2_AD_INFLUENT = {"S_su": 0.01, "S_aa": 0.001, "S_fa": 0.001, "S_va": 0.001, "S_bu": 0.001,
                    "S_pro": 0.001, "S_ac": 0.001, "S_h2": 1e-8, "S_ch4": 1e-5, "S_IC": 0.04,
                    "S_IN": 0.01, "S_I": 0.02, "X_c": 2.0, "X_ch": 5.0, "X_pr": 20.0,
                    "X_li": 5.0, "X_su": 0.0, "X_aa": 0.01, "X_fa": 0.01, "X_c4": 0.01,
                    "X_pro": 0.01, "X_ac": 0.01, "X_h2": 0.01, "X_I": 25.0, "S_cat": 0.04,
                    "S_an": 0.02}


def adm1_test_influent() -> Stream:
    """The ADM1 verification influent of [W6] Section 7 as a stream (170 m^3/d, 35 C)."""
    return state_to_stream(170.0, jnp.array([BSM2_AD_INFLUENT[s] for s in STATES]), T=308.15)


def anaerobic_digester(influent: Stream, params: ADM1Params | None = None,
                       t_settle: float = 300.0, inoculum: float = 1.0) -> dict:
    """Steady state of a completely mixed anaerobic digester (ADM1, BSM2 implementation).

    Integrates the dynamics for ``t_settle`` days from a generic digester
    state (diffrax Kvaerno5, all inputs stop-gradiented: nothing in that
    phase is differentiated), then Newton (optimistix) on the 29 steady-state
    equations; gradients are implicit at the solution.

    Args:
        influent: ADM1 stream (see the module docstring).
        params: :class:`ADM1Params` (the BSM2 digester, 3400 + 300 m^3, 35 C).
        t_settle: Days of dynamics before Newton (the report runs 200).
        inoculum: Initial concentration of each of the seven biomass groups
            (kg COD/m^3). ADM1 has a soured steady state (low pH, VFAs
            accumulated, acetogens and methanogens washed out) that a thinly
            seeded start reaches; the methanogenic state needs a seeded one.
            Check ``pH`` and ``gas_fraction_ch4`` of the result.

    Returns:
        dict with ``S`` (26 states), ``G`` (3 gas states), ``pH``, ``S_H``,
        the ions, partial pressures (bar), ``q_gas`` (m^3/d at P_atm),
        ``methane`` (m^3/d at P_atm, from the partial pressure),
        ``gas_fraction_ch4``/``_co2``, ``effluent`` and ``biogas`` streams,
        ``converged``, ``residual``.

    Example:
        >>> r = anaerobic_digester(adm1_test_influent())
        >>> round(float(r["pH"]), 2)
        7.47
    """
    p = params or ADM1Params()
    if p.gas_flow not in ("compensated", "uncompensated"):
        raise ValueError("gas_flow must be 'compensated' or 'uncompensated'")
    if p.pH_inhibition not in ("hill", "switch"):
        raise ValueError("pH_inhibition must be 'hill' or 'switch'")
    q, S_in = stream_to_state(influent)
    flags = (p.gas_flow, p.pH_inhibition)
    x, res, ok = _solve(q, S_in, _fields(p), flags, float(t_settle),
                        jax.lax.stop_gradient(jnp.asarray(inoculum, dtype=jnp.float64)))
    n = len(STATES)
    S, G = x[:n], x[n:]
    _, _, d = rhs(S, G, q, S_in, p)
    RT = R_ADM * p.T_op
    # species molar flows in the gas (kmol/d): p_i V / (R T) with V the gas volume at P_gas
    V_at_Pgas = d["q_gas"] * p.P_atm / d["P_gas"]
    kmol = {k: d[f"p_gas_{k}"] * V_at_Pgas / RT for k in ("h2", "ch4", "co2", "h2o")}
    biogas = make_stream({"CH4": kmol["ch4"] * 1000 / 86400, "CO2": kmol["co2"] * 1000 / 86400,
                          "H2": kmol["h2"] * 1000 / 86400, "H2O": kmol["h2o"] * 1000 / 86400},
                         T=p.T_op, P=d["P_gas"] * 1e5)
    return {
        "S": S, "G": G, "pH": d["pH"], "S_H": d["S_H"], "ions": d["ions"], "S_co2": d["S_co2"],
        "S_nh4": d["S_nh4"], "p_gas_h2": d["p_gas_h2"], "p_gas_ch4": d["p_gas_ch4"],
        "p_gas_co2": d["p_gas_co2"], "P_gas": d["P_gas"], "q_gas": d["q_gas"],
        "methane": d["q_gas"] * d["p_gas_ch4"] / d["P_gas"],
        "gas_fraction_ch4": d["p_gas_ch4"] / d["P_gas"],
        "gas_fraction_co2": d["p_gas_co2"] / d["P_gas"],
        "effluent": state_to_stream(d["q_out"], S, influent["T"], influent["P"]),
        "q_out": d["q_out"],
        "biogas": biogas,
        "converged": ok,
        "residual": res,
    }


_FLAGS = ("gas_flow", "pH_inhibition")
_NUMERIC = [f.name for f in dataclasses.fields(ADM1Params) if f.name not in _FLAGS]


def _fields(p):
    return jnp.stack([jnp.asarray(getattr(p, k), dtype=jnp.float64) for k in _NUMERIC])


@partial(jax.jit, static_argnums=(3, 4))
def _solve(q, S_in, vals, flags, t_settle, inoculum):
    """Settle (stop-gradiented dynamics) then Newton: the jitted core."""
    import diffrax
    import optimistix as optx

    p = ADM1Params(**dict(zip(_FLAGS, flags)), **dict(zip(_NUMERIC, vals)))
    n = len(STATES)

    def f(x, q, S_in, p):
        dS, dG, _ = rhs(x[:n], x[n:], q, S_in, p)
        return jnp.concatenate([dS, dG])

    # generic start: the influent, every biomass group seeded with ``inoculum``
    # (kg COD/m^3; a digester is started from an inoculum, and from a thin one
    # ADM1 sours: acetogens and methanogens wash out at low pH, a second
    # steady state), a nominal head space
    bio = jnp.array([s.startswith("X_") and s not in ("X_c", "X_ch", "X_pr", "X_li", "X_I")
                     for s in STATES])
    # substrates start near zero (a reactor holding the influent's 20+ kg COD/m^3
    # of solids hydrolyses into an acid burst that sours any inoculum); inerts,
    # inorganic C and N and the ions start at the influent's values
    substrate = jnp.array([s in ("S_su", "S_aa", "S_fa", "S_va", "S_bu", "S_pro", "S_ac",
                                 "S_h2", "S_ch4", "X_c", "X_ch", "X_pr", "X_li") for s in STATES])
    S0 = jnp.where(bio, jnp.maximum(S_in, inoculum),
                   jnp.where(substrate, 1e-3, jnp.maximum(S_in, 1e-6)))
    x0 = jnp.concatenate([S0, jnp.array([1e-5, 1.0, 0.01])])
    sg = jax.lax.stop_gradient
    q_s, S_s = sg(q), sg(S_in)
    p_s = ADM1Params(**dict(zip(_FLAGS, flags)), **dict(zip(_NUMERIC, sg(vals))))
    sol = diffrax.diffeqsolve(diffrax.ODETerm(lambda t, x, a: f(x, q_s, S_s, p_s)),
                              diffrax.Kvaerno5(), t0=0.0, t1=t_settle, dt0=1e-6,
                              y0=sg(x0),
                              stepsize_controller=diffrax.PIDController(rtol=1e-7, atol=1e-10),
                              max_steps=50000, throw=False, adjoint=diffrax.ForwardMode())
    x1 = sg(sol.ys[-1])
    root = optx.root_find(lambda x, a: f(x, a[0], a[1], p), optx.Newton(rtol=1e-12, atol=1e-13),
                          x1, args=(q, S_in), max_steps=40, throw=False)
    x = root.value
    res = jnp.max(jnp.abs(f(x, q, S_in, p)))
    return x, res, (root.result == optx.RESULTS.successful) & (res < 1e-8)


class AnaerobicDigester:
    """Completely mixed anaerobic digester with ADM1 (BSM2 implementation), at steady state.

    Inlet: ADM1 stream. Outlets: ``effluent`` (the digestate) and ``biogas``
    (``F_CH4``, ``F_CO2``, ``F_H2``, ``F_H2O``, mol/s). See
    :func:`anaerobic_digester`.

    Example:
        >>> eff, gas, info = AnaerobicDigester(ADM1Params())(adm1_test_influent())
        >>> 0.55 < float(info["gas_fraction_ch4"]) < 0.65
        True
    """

    symbol = "AD"
    equations = [
        r"0 = \frac{q}{V_{liq}}(S_{in} - S) + \textstyle\sum_j \nu_{ij}\rho_j"
        r"\quad\text{(19 processes, 24 states; C and N balances closed)}",
        r"S_{H^+} + S_{cat} + S_{NH_4^+} - S_{HCO_3^-} - \tfrac{S_{ac^-}}{64} - \dots - S_{an} - \tfrac{K_w}{S_{H^+}} = 0",
        r"\rho_{T,i} = k_La (S_i - K_{H,i} p_{gas,i}),\quad q_{gas} = k_p (P_{gas} - P_{atm}) \tfrac{P_{gas}}{P_{atm}}",
    ]
    assumptions = [
        "ADM1 as implemented for BSM2 (Rosen & Jeppsson 2006); steady state; completely mixed.",
        "Acid-base equilibria algebraic (DAE-pH); temperature-dependent physico-chemical constants.",
        "Head-space overpressure gas flow (compensated by default); pH inhibition by Hill functions (BSM2).",
        "Started from an inoculum with low substrate; ADM1 also has a soured steady state.",
    ]
    references = [
        "Rosen & Jeppsson, Aspects on ADM1 implementation within the BSM2 framework, TEIE-7224, Lund University (2006).",
        "Batstone et al., Water Sci. Technol. 45 (2002) 65-73, doi:10.2166/wst.2002.0292.",
    ]
    parameter_symbols = {"V_liq": "V_liq", "V_gas": "V_gas", "T_op": "T", "k_La": "k_La",
                         "k_p": "k_p"}
    parameter_units = {f.name: "-" for f in dataclasses.fields(ADM1Params)} | {
        "V_liq": "m^3", "V_gas": "m^3", "T_op": "K", "k_La": "1/d", "k_p": "m^3/(d bar)",
        "P_atm": "bar", "K_S_IN": "kmol/m^3", "K_I_nh3": "kmol/m^3", "k_dec": "1/d",
        "k_dis": "1/d", "k_hyd_ch": "1/d", "k_hyd_pr": "1/d", "k_hyd_li": "1/d",
        "k_m_su": "1/d", "k_m_aa": "1/d", "k_m_fa": "1/d", "k_m_c4": "1/d", "k_m_pro": "1/d",
        "k_m_ac": "1/d", "k_m_h2": "1/d", "K_S_su": "kg COD/m^3", "K_S_aa": "kg COD/m^3",
        "K_S_fa": "kg COD/m^3", "K_S_c4": "kg COD/m^3", "K_S_pro": "kg COD/m^3",
        "K_S_ac": "kg COD/m^3", "K_S_h2": "kg COD/m^3", "K_I_h2_fa": "kg COD/m^3",
        "K_I_h2_c4": "kg COD/m^3", "K_I_h2_pro": "kg COD/m^3"}
    numerical_method = ("Stiff integration from an inoculum (diffrax Kvaerno5, stop-gradiented), "
                        "then Newton (optimistix); pH by bisection + implicit step.")
    outlet_roles = ["effluent", "biogas"]

    def __init__(self, params: ADM1Params):
        """Initialize.

        Args:
            params: :class:`ADM1Params`.
        """
        if params.gas_flow not in ("compensated", "uncompensated"):
            raise ValueError("gas_flow must be 'compensated' or 'uncompensated'")
        if params.pH_inhibition not in ("hill", "switch"):
            raise ValueError("pH_inhibition must be 'hill' or 'switch'")
        self.params = params

    def __call__(self, influent: Stream) -> tuple[Stream, Stream, dict]:
        """Digest to steady state.

        Returns:
            ``(effluent, biogas, info)``; ``info`` has the scalar results of
            :func:`anaerobic_digester` and the state vector ``S``.
        """
        r = anaerobic_digester(influent, self.params)
        info = {k: v for k, v in r.items()
                if k not in ("effluent", "biogas", "ions") and not isinstance(v, dict)}
        return r["effluent"], r["biogas"], info
