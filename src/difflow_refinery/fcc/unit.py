"""The fluid catalytic cracker: riser, regenerator and main fractionator.

:class:`FCCUnit` solves the riser and the regenerator TOGETHER as one
equation-oriented system -- the FCC heat balance:

    unknowns   x = (C/O, T_rg)        catalyst-to-oil ratio, regenerator T
    equations  T_riser_out(x) - ROT = 0                (riser outlet T spec)
               Q_regenerator(x)     = 0                (regenerator energy)

The riser outlet temperature (ROT) is the spec; catalyst circulation and
regenerator temperature are OUTPUTS, which is how an FCC is run (the
slide valve holds ROT). The coke the riser makes at ``x`` is what the
regenerator burns at ``x``, so the two units are coupled through both
equations. Solved by Newton (optimistix) from ``C/O = 6, T_rg = 980 K``;
gradients of anything downstream come from the implicit function theorem
through ``optimistix``'s implicit adjoint, not by differentiating Newton's
iterations -- the same pattern as :class:`~difflow_refinery.column.Furnace`
solved with :class:`~difflow_refinery.column.CrudeColumn`.

After the solve, the lumps are mapped to products (real light species,
product pseudocomponents) and split by the simplified main fractionator
(:mod:`difflow_refinery.fcc.fractionator`). Elemental bookkeeping:

* Gas species carry their formula's C, H, S.
* Gasoline and coke have a hydrogen content each (parameters).
* Sulfur: gasoline and coke at ``factor * feed S`` concentration, the
  unconverted cycle oil at ``(1 + cycle_oil_sulfur_slope X) * feed S`` (X =
  conversion), and H2S takes the rest. H2S's mass is taken from the dry-gas
  lump.
* Nitrogen: ``n_to_coke`` of the feed N in coke, the rest in the cycle oil
  (NH3 and HCN not modelled).
* Cycle-oil hydrogen BY DIFFERENCE, so the hydrogen balance closes exactly;
  it is reported (``cycle_oil.hydrogen``) so an implausible value is seen.
* Carbon is mass minus H, S, N (feed metals and oxygen are counted as
  carbon -- tens of ppm).

Outlets, in :attr:`FCCUnit.OUTLETS` order (mol/s per species, difflow
streams)::

    dry_gas    hydrogen, methane, ethane, ethylene, hydrogen_sulfide
               (+ any feed species not on the feed's property table)
    c3         propane, propylene                    -> alkylation (#310)
    c4         isobutane, n_butane, 1_butene, isobutylene,
               cis_2_butene, trans_2_butene          -> alkylation (#310)
    gasoline   fcc01..fcc22 product pseudocomponents (mostly fcc01..fcc10,
               the C5-221 C cuts; the split overlaps)
    lco        fcc01..fcc22 (mostly 221-343 C)
    slurry     fcc01..fcc22 (mostly 343 C+)
    sour_water water (the riser steam)
    flue_gas   nitrogen, oxygen, carbon_dioxide, carbon_monoxide, water,
               sulfur_dioxide

The C3/C4 split is ideal -- it stands in for the gas plant (issue #312,
which does not exist yet).
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow_refinery import correlations as corr
from difflow_refinery.blending import cetane_index_d4737, tbp_temperature, tbp_to_d86
from difflow_refinery.fcc import fractionator as frac
from difflow_refinery.fcc import regenerator as rg
from difflow_refinery.fcc import riser
from difflow_refinery.fcc import species as sp
from difflow_refinery.fcc.feed import FCCFeed
from difflow_refinery.fcc.kinetics import GROUPS, LumpScheme, get_scheme

jax.config.update("jax_enable_x64", True)


class FCCConvergenceWarning(UserWarning):
    """The riser-regenerator heat balance did not converge."""


class RegeneratorTemperatureWarning(UserWarning):
    """The solved regenerator temperature is above ``regenerator_T_max``.

    The heat balance can have more than one steady state, and with a very
    active catalyst or a coke-rich feed the solved one can be far outside
    any operable range (a few hundred kelvin of catalyst circulation away
    from the default start). The solve is not wrong -- it is the steady
    state of the equations -- but nobody runs a regenerator there.
    """


#: Default deactivation constant per deactivation kind (illustrative).
DEFAULT_DEACTIVATION_ALPHA = {"time": 0.12, "coke": 60.0}


@dataclass
class FCCParams(ParamsMixin):
    """Parameters of the fluid catalytic cracker.

    Every default below the specs is ILLUSTRATIVE (see the docs' references
    table for what each is and is not sourced from). Units SI: K, Pa, m,
    kg/s, J/kg, J/kg/K; fractions are mass fractions unless named.

    Attributes:
        scheme: Kinetic scheme: ``"ancheyta_5"`` (default), ``"lee_4"``,
            ``"weekman_nace_3"``, or a :class:`~difflow_refinery.fcc.kinetics.LumpScheme`.
        k_ref: Rate constants at 500 C (1/s per unit C/O), one per reaction
            of the scheme; ``None`` takes the scheme's illustrative defaults.
        Ea: Activation energies (J/mol), one per reaction; ``None`` defaults.
        activity: Equilibrium-catalyst activity, a multiplier on every rate
            (1 = the kinetic constants' catalyst). The parameter to track
            from test runs with ``difflow.reconciliation.tracking``.
        riser_outlet_T: Riser outlet temperature spec, ROT (K).
        feed_T: Feed preheat temperature (K), liquid.
        riser_P: Riser pressure (Pa), constant along the riser.
        riser_height: Riser height (m).
        riser_diameter: Riser diameter (m).
        slip: Slip factor, gas velocity over catalyst velocity.
        steam_ratio: Riser (lift and atomising) steam, kg per kg feed.
        steam_T: Riser steam temperature (K).
        cp_vapor: Heat capacity of hydrocarbon vapour and coke (J/kg/K).
        cp_catalyst: Heat capacity of the catalyst (J/kg/K).
        cp_steam: Heat capacity of riser steam (J/kg/K).
        latent_heat: Feed heat of vaporisation at the preheat T (J/kg).
        heat_of_cracking: Endothermic heat per kg gas oil converted (J/kg).
        deactivation: ``"time"`` (Weekman 1968, exp(-alpha t_c)) or
            ``"coke"`` (exp(-alpha C_c)).
        deactivation_alpha: Its constant (1/s or -); ``None`` takes
            :data:`DEFAULT_DEACTIVATION_ALPHA`.
        kw_ref: Watson K at which the feed multiplier is one.
        kw_sensitivity: Crackability ``exp(kw_sensitivity (Kw - kw_ref))``.
        basic_nitrogen_fraction: Basic share of the feed nitrogen.
        nitrogen_poisoning: Basic-N mass fraction that halves the rates.
        ccr_to_coke: Fraction of feed Conradson carbon that becomes coke.
        metals_coke: Contaminant coke per unit Ni+V mass fraction.
        metals_h2: Contaminant H2 per unit Ni+V mass fraction.
        dry_gas_share: Dry-gas share of the gas lump (3- and 4-lump schemes).
        coke_share: Coke share of the gas+coke lump (3-lump scheme).
        dry_gas_split: Mass split of hydrocarbon dry gas over hydrogen,
            methane, ethane, ethylene.
        c3_share: C3 share of the LPG (mass).
        propylene_in_c3: Olefin (propylene) share of the C3 (mass).
        olefins_in_c4: Olefin share of the C4 (mass).
        isobutane_in_c4_paraffins: Isobutane share of the C4 paraffins.
        butene_split: Split of the C4 olefins over 1-butene, isobutylene,
            cis-2-butene, trans-2-butene (mass).
        gasoline_hydrogen: Hydrogen mass fraction of the gasoline lump at a
            feed of ``hydrogen_ref`` hydrogen.
        gasoline_hydrogen_slope: Change of gasoline H per unit change of
            feed H (a hydrogen-poor feed makes a more aromatic gasoline).
        hydrogen_ref: Feed hydrogen mass fraction of ``gasoline_hydrogen``.
        coke_hydrogen: Hydrogen mass fraction of coke.
        gasoline_sulfur_factor: Gasoline S concentration over feed S.
        coke_sulfur_factor: Coke S concentration over feed S.
        cycle_oil_sulfur_slope: Cycle-oil S over feed S is ``1 + slope X``.
        n_to_coke: Fraction of the feed nitrogen that leaves in coke.
        cycle_oil_hydrogen_gradient: Change of cycle-oil H mass fraction per
            K of boiling point across the cycle-oil cuts (mean preserved).
        cycle_oil_sulfur_gradient: Relative change of cycle-oil S per K.
        gasoline_T50, gasoline_width: Gasoline lump TBP distribution (K).
        cycle_oil_T50, cycle_oil_width: Cycle-oil lump TBP distribution (K).
        gasoline_kw, cycle_oil_kw: Watson K of the product cuts per lump.
        gasoline_cut: Main fractionator gasoline/LCO cut point (K).
        lco_cut: Main fractionator LCO/slurry cut point (K).
        split_width: Overlap width of the fractionator split (K).
        co_co2: CO/CO2 molar ratio of the flue gas, or ``"arthur"``.
        arthur_factor: Multiplier on Arthur's CO/CO2.
        flue_o2: Flue-gas O2 mole fraction (wet) spec; the air rate follows.
        air_rate: Air rate (kg/s) instead of ``flue_o2`` (then ``flue_o2``
            is ignored and the excess O2 is an output).
        air_T: Air (blower discharge) temperature (K).
        regen_heat_loss: Regenerator heat loss (W).
        regenerator_T_max: Regenerator temperature above which a
            :class:`RegeneratorTemperatureWarning` is raised (K; 760 C, a
            commonly quoted metallurgical limit -- illustrative). Not
            imposed; reported as ``regenerator_margin``.
        ron_ref, ron_dT, ron_dX: Gasoline RON ``ron_ref + ron_dT (ROT -
            T_octane_ref) + ron_dX (X - X_octane_ref)``.
        mon_ref, mon_dT, mon_dX: The same for MON.
        T_octane_ref, X_octane_ref: Reference ROT (K) and conversion.
        gasoline_pona: Gasoline volume fractions (P, N, A, O).
        n_steps: Riser integration steps.
        max_iter: Newton iteration limit.
        tol: Convergence tolerance on the scaled residuals.
        CTO_guess, T_rg_guess: Newton's starting point.
    """

    scheme: Union[str, LumpScheme] = "ancheyta_5"
    k_ref: Optional[Sequence[float]] = None
    Ea: Optional[Sequence[float]] = None
    activity: float = 1.0
    riser_outlet_T: float = 793.15
    feed_T: float = 500.0
    riser_P: float = 2.5e5
    riser_height: float = 30.0
    riser_diameter: float = 1.0
    slip: float = 2.0
    steam_ratio: float = 0.03
    steam_T: float = 460.0
    cp_vapor: float = 3.0e3
    cp_catalyst: float = 1.15e3
    cp_steam: float = 2.1e3
    latent_heat: float = 2.5e5
    heat_of_cracking: float = 3.5e5
    deactivation: str = "time"
    deactivation_alpha: Optional[float] = None
    kw_ref: float = 11.8
    kw_sensitivity: float = 1.0
    basic_nitrogen_fraction: float = 0.3
    nitrogen_poisoning: float = 1.0e-3
    ccr_to_coke: float = 0.6
    metals_coke: float = 50.0
    metals_h2: float = 5.0
    dry_gas_share: float = 0.2
    coke_share: float = 0.25
    dry_gas_split: Sequence[float] = (0.03, 0.40, 0.28, 0.29)
    c3_share: float = 0.40
    propylene_in_c3: float = 0.73
    olefins_in_c4: float = 0.55
    isobutane_in_c4_paraffins: float = 0.78
    butene_split: Sequence[float] = (0.22, 0.28, 0.22, 0.28)
    gasoline_hydrogen: float = 0.128
    gasoline_hydrogen_slope: float = 0.5
    hydrogen_ref: float = 0.12
    coke_hydrogen: float = 0.07
    gasoline_sulfur_factor: float = 0.15
    coke_sulfur_factor: float = 1.0
    cycle_oil_sulfur_slope: float = 0.8
    n_to_coke: float = 0.4
    cycle_oil_hydrogen_gradient: float = -1.0e-4
    cycle_oil_sulfur_gradient: float = 1.0e-3
    gasoline_T50: float = frac.DEFAULT_LUMP_TBP["gasoline"][0] + frac.C
    gasoline_width: float = frac.DEFAULT_LUMP_TBP["gasoline"][1]
    cycle_oil_T50: float = frac.DEFAULT_LUMP_TBP["cycle_oil"][0] + frac.C
    cycle_oil_width: float = frac.DEFAULT_LUMP_TBP["cycle_oil"][1]
    gasoline_kw: float = frac.DEFAULT_LUMP_KW["gasoline"]
    cycle_oil_kw: float = frac.DEFAULT_LUMP_KW["cycle_oil"]
    gasoline_cut: float = 221.0 + frac.C
    lco_cut: float = 343.0 + frac.C
    split_width: float = 6.0
    co_co2: Union[float, str] = 0.0
    arthur_factor: float = 1.0
    flue_o2: float = 0.02
    air_rate: Optional[float] = None
    air_T: float = 470.0
    regen_heat_loss: float = 0.0
    regenerator_T_max: float = 1033.15
    ron_ref: float = 92.0
    ron_dT: float = 0.06
    ron_dX: float = 8.0
    mon_ref: float = 80.0
    mon_dT: float = 0.03
    mon_dX: float = 4.0
    T_octane_ref: float = 793.15
    X_octane_ref: float = 0.70
    gasoline_pona: Sequence[float] = (0.33, 0.10, 0.30, 0.27)
    n_steps: int = 200
    max_iter: int = 50
    tol: float = 1e-10
    CTO_guess: float = 6.0
    T_rg_guess: float = 980.0


#: Fields of FCCParams that configure the model rather than enter it as numbers.
_STATIC = {"scheme", "k_ref", "Ea", "deactivation", "deactivation_alpha", "co_co2",
           "air_rate", "n_steps", "max_iter", "tol", "CTO_guess", "T_rg_guess"}


def numeric_params(p: FCCParams, scheme: LumpScheme) -> dict[str, Array]:
    """The traceable parameter pytree of a solve."""
    out = {}
    for f in dataclasses.fields(p):
        if f.name in _STATIC:
            continue
        out[f.name] = jnp.asarray(getattr(p, f.name), dtype=float)
    kin = scheme.theta()
    out["k_ref"] = kin["k_ref"] if p.k_ref is None else jnp.asarray(p.k_ref, dtype=float)
    out["Ea"] = kin["Ea"] if p.Ea is None else jnp.asarray(p.Ea, dtype=float)
    if out["k_ref"].shape != kin["k_ref"].shape or out["Ea"].shape != kin["Ea"].shape:
        raise ValueError(f"k_ref and Ea need {len(scheme.reactions)} entries for {scheme.name}")
    alpha = p.deactivation_alpha
    out["deactivation_alpha"] = jnp.asarray(
        DEFAULT_DEACTIVATION_ALPHA[p.deactivation] if alpha is None else alpha, dtype=float)
    out["co_co2"] = jnp.asarray(0.0 if p.co_co2 == "arthur" else p.co_co2, dtype=float)
    out["air_rate"] = jnp.asarray(0.0 if p.air_rate is None else p.air_rate, dtype=float)
    return out


# =============================================================================
# Light-species composition of the gas lumps
# =============================================================================


def dry_gas_fractions(p) -> dict[str, Array]:
    """Mass split of hydrocarbon dry gas (normalised)."""
    s = jnp.asarray(p["dry_gas_split"])
    s = s / jnp.sum(s)
    return dict(zip(sp.DRY_GAS_SPECIES, s))


def lpg_fractions(p) -> dict[str, Array]:
    """Mass split of LPG over the C3 and C4 species, from the split parameters."""
    c3, c4 = p["c3_share"], 1.0 - p["c3_share"]
    ole = p["olefins_in_c4"]
    bs = jnp.asarray(p["butene_split"])
    bs = bs / jnp.sum(bs)
    par = c4 * (1.0 - ole)
    return {
        "propane": c3 * (1.0 - p["propylene_in_c3"]),
        "propylene": c3 * p["propylene_in_c3"],
        "isobutane": par * p["isobutane_in_c4_paraffins"],
        "n_butane": par * (1.0 - p["isobutane_in_c4_paraffins"]),
        "1_butene": c4 * ole * bs[0],
        "isobutylene": c4 * ole * bs[1],
        "cis_2_butene": c4 * ole * bs[2],
        "trans_2_butene": c4 * ole * bs[3],
    }


def _mix_mw(fracs: dict) -> Array:
    return 1.0 / sum(w / sp.MW[n] for n, w in fracs.items())


def _mix_h(fracs: dict) -> Array:
    return sum(w * sp.element_mass_fraction(n, "H") for n, w in fracs.items())


def _product_grid(p) -> dict:
    """Product pseudocomponents and each lump's distribution on them."""
    props = frac.pseudo_properties({"gasoline": p["gasoline_kw"], "cycle_oil": p["cycle_oil_kw"]})
    n_g = frac.GRID["n_gasoline"]
    lo, hi = props["lo"], props["hi"]
    w_g = frac.lump_distribution(lo[:n_g], hi[:n_g], p["gasoline_T50"], p["gasoline_width"])
    w_c = frac.lump_distribution(lo[n_g:], hi[n_g:], p["cycle_oil_T50"], p["cycle_oil_width"])
    props["w_gasoline"] = jnp.concatenate([w_g, jnp.zeros(frac.GRID["n_cycle"])])
    props["w_cycle"] = jnp.concatenate([jnp.zeros(n_g), w_c])
    props["MW_gasoline"] = 1.0 / jnp.sum(w_g / props["MW"][:n_g])
    return props


# =============================================================================
# The solve
# =============================================================================


def _model(theta, scheme: LumpScheme, static: tuple, ts=None):
    """Build the residual and the post-processing for one configuration."""
    n_steps, deact, co_mode, air_mode = static
    p, feed = theta["params"], theta["feed"]
    M = scheme.group_matrix({"dry_gas_share": p["dry_gas_share"], "coke_share": p["coke_share"]})
    grid = _product_grid(p)
    group_mw = {"gas_oil": feed.MW, "gasoline": grid["MW_gasoline"],
                "lpg": _mix_mw(lpg_fractions(p)), "dry_gas": _mix_mw(dry_gas_fractions(p))}
    inv_mw = riser.lump_inverse_mw(scheme, M, group_mw)

    def run_riser(CTO, T_rg):
        return riser.integrate(p, feed, scheme, M, inv_mw, CTO, T_rg, n_steps, deact, ts=ts)

    def coke_of(r):
        groups = r["y"] @ M
        return feed.mass * (groups[GROUPS.index("coke")] + r["coke_ccr"] + r["coke_metals"])

    def residual(x, _):
        CTO, T_rg = x[0], x[1] * 1000.0
        r = run_riser(CTO, T_rg)
        coke = coke_of(r)
        el = rg.coke_elements(p, feed, coke)
        comb = rg.combustion(p, el, T_rg, co_mode, air_mode)
        q = rg.heat_residual(p, feed, CTO, T_rg, r["T_out"], coke, comb)
        return jnp.stack([(r["T_out"] - p["riser_outlet_T"]) / 100.0, q / (feed.mass * 1e6)])

    return M, grid, run_riser, coke_of, residual


@lru_cache(maxsize=32)
def _compiled(scheme: LumpScheme, static: tuple, max_iter: int, tol: float):
    def solve(theta):
        M, grid, run_riser, coke_of, residual = _model(theta, scheme, static)
        x0 = theta["x0"]
        sol = optx.root_find(residual, optx.Newton(rtol=tol, atol=tol), x0,
                             max_steps=max_iter, throw=False)
        x = sol.value
        res = jnp.max(jnp.abs(residual(x, None)))
        out = _post(theta, scheme, static, M, grid, run_riser, coke_of, x)
        out["residual"] = res
        out["iterations"] = sol.stats["num_steps"]
        out["solver_ok"] = sol.result == optx.RESULTS.successful
        return out

    return jax.jit(solve)


def _post(theta, scheme, static, M, grid, run_riser, coke_of, x) -> dict:
    """Everything a caller reads, at the solved ``x``."""
    n_steps, deact, co_mode, air_mode = static
    p, feed = theta["params"], theta["feed"]
    CTO, T_rg = x[0], x[1] * 1000.0
    Fo = feed.mass
    r = run_riser(CTO, T_rg)
    groups = r["y"] @ M                               # gas_oil, gasoline, lpg, dry_gas, coke
    y_go, y_gs, y_lpg, y_dg, y_ck = (groups[i] for i in range(5))
    coke = coke_of(r)
    y_coke = coke / Fo
    X = 1.0 - y_go                                    # conversion (221 C-, coke incl.)

    # --- regenerator ---
    el = rg.coke_elements(p, feed, coke)
    comb = rg.combustion(p, el, T_rg, co_mode, air_mode)
    q_regen = rg.heat_residual(p, feed, CTO, T_rg, r["T_out"], coke, comb)

    # --- sulfur, nitrogen, hydrogen by lump ---
    S_feed = feed.sulfur * Fo
    m_gs, m_co = y_gs * Fo, y_go * Fo
    S_gs = p["gasoline_sulfur_factor"] * feed.sulfur * m_gs
    S_ck = el["S"]
    S_co = (1.0 + p["cycle_oil_sulfur_slope"] * X) * feed.sulfur * m_co
    S_h2s = S_feed - S_gs - S_ck - S_co
    m_h2s = S_h2s * sp.MW["hydrogen_sulfide"] / sp.ATOMIC_WEIGHT["S"]
    m_dg_hc = (y_dg) * Fo - m_h2s                     # hydrocarbon dry gas from the lump
    m_h2_metals = r["h2_metals"] * Fo
    m_lpg = y_lpg * Fo

    dg = {n: w * m_dg_hc for n, w in dry_gas_fractions(p).items()}
    dg["hydrogen"] = dg["hydrogen"] + m_h2_metals
    dg["hydrogen_sulfide"] = m_h2s
    lpg = {n: w * m_lpg for n, w in lpg_fractions(p).items()}

    N_ck = el["N"]
    N_co = feed.nitrogen * Fo - N_ck
    H_feed = feed.hydrogen * Fo
    H_gas = sum(m * sp.element_mass_fraction(n, "H") for n, m in {**dg, **lpg}.items())
    h_gs = p["gasoline_hydrogen"] + p["gasoline_hydrogen_slope"] * (feed.hydrogen - p["hydrogen_ref"])
    H_gs = h_gs * m_gs
    H_co = H_feed - H_gas - H_gs - el["H"]            # by difference

    # --- product pseudocomponents ---
    n_g = frac.GRID["n_gasoline"]
    m_pc = m_gs * grid["w_gasoline"] + m_co * grid["w_cycle"]
    Tb = grid["Tb"]
    wc = grid["w_cycle"]
    Tbar = jnp.sum(wc * Tb)
    h_co = H_co / m_co
    s_co = S_co / m_co
    h_pc = jnp.where(jnp.arange(Tb.shape[0]) < n_g, h_gs,
                     h_co + p["cycle_oil_hydrogen_gradient"] * (Tb - Tbar))
    s_pc = jnp.where(jnp.arange(Tb.shape[0]) < n_g, S_gs / m_gs,
                     s_co * (1.0 + p["cycle_oil_sulfur_gradient"] * (Tb - Tbar)))
    n_pc = jnp.where(jnp.arange(Tb.shape[0]) < n_g, 0.0, N_co / m_co)

    split = frac.split_fractions(Tb, p["gasoline_cut"], p["lco_cut"], p["split_width"])
    prod_m = {name: m_pc * split[:, j] for j, name in enumerate(("gasoline", "lco", "slurry"))}

    def liquid_props(m):
        tot = jnp.sum(m)
        vol = m / grid["SG"]
        phi = vol / jnp.sum(vol)
        sg = tot / jnp.sum(vol)
        tbp = {pct: tbp_temperature(jnp.asarray(pct / 100.0), Tb, phi, 4.0) for pct in (10, 50, 90)}
        return {"mass": tot, "SG": sg, "API": corr.api_from_sg(sg),
                "sulfur": jnp.sum(m * s_pc) / tot, "hydrogen": jnp.sum(m * h_pc) / tot,
                "nitrogen": jnp.sum(m * n_pc) / tot,
                **{f"tbp{k}": v for k, v in tbp.items()}}

    props = {k: liquid_props(m) for k, m in prod_m.items()}
    lco = props["lco"]
    d86 = {k: tbp_to_d86(lco[f"tbp{k}"], k) - frac.C for k in (10, 50, 90)}
    # D4737 wants density at 15 C in g/mL: SG(60F/60F) * 0.999 (water at 15.6 C).
    lco["cetane_index"] = cetane_index_d4737(lco["SG"] * 0.99904, d86[10], d86[50], d86[90])
    ROT = r["T_out"]
    gas = props["gasoline"]
    gas["RON"] = p["ron_ref"] + p["ron_dT"] * (ROT - p["T_octane_ref"]) + p["ron_dX"] * (X - p["X_octane_ref"])
    gas["MON"] = p["mon_ref"] + p["mon_dT"] * (ROT - p["T_octane_ref"]) + p["mon_dX"] * (X - p["X_octane_ref"])
    pona = jnp.asarray(p["gasoline_pona"])
    pona = pona / jnp.sum(pona)
    gas.update({"paraffins": pona[0], "naphthenes": pona[1], "aromatics": pona[2], "olefins": pona[3]})

    # --- streams (mol/s) ---
    mol = lambda d: {n: m * 1000.0 / sp.MW[n] for n, m in d.items()}
    pc_mol = {k: m * 1000.0 / grid["MW"] for k, m in prod_m.items()}
    Fs = p["steam_ratio"] * Fo
    streams = {
        "dry_gas": mol(dg),
        "c3": mol({n: lpg[n] for n in sp.C3_SPECIES}),
        "c4": mol({n: lpg[n] for n in sp.C4_SPECIES}),
        "gasoline": pc_mol["gasoline"], "lco": pc_mol["lco"], "slurry": pc_mol["slurry"],
        "sour_water": {"water": Fs * 1000.0 / sp.MW["water"]},
        "flue_gas": comb["flue"],
    }

    # --- balances ---
    air_mass = sp.mass_of(comb["air"])
    m_in = Fo + Fs + air_mass
    liquids = sum(jnp.sum(m) for m in prod_m.values())
    m_out = (sp.mass_of(streams["dry_gas"]) + sp.mass_of(streams["c3"]) + sp.mass_of(streams["c4"])
             + liquids + Fs + sp.mass_of(comb["flue"]))
    liq_el = {"H": sum(jnp.sum(m * h_pc) for m in prod_m.values()),
              "S": sum(jnp.sum(m * s_pc) for m in prod_m.values()),
              "N": sum(jnp.sum(m * n_pc) for m in prod_m.values())}
    gas_streams = [streams[k] for k in ("dry_gas", "c3", "c4", "flue_gas")]

    def el_out(e):
        g = sum(sp.element_flow(s, e) for s in gas_streams)
        return g + (liq_el[e] if e in liq_el else 0.0)

    el_in = {"H": H_feed + sp.element_flow(streams["sour_water"], "H"),
             "S": S_feed, "N": feed.nitrogen * Fo + sp.element_flow(comb["air"], "N")}
    el_out_ = {"H": el_out("H") + sp.element_flow(streams["sour_water"], "H"),
               "S": el_out("S"), "N": el_out("N")}
    C_in = Fo * (1.0 - feed.hydrogen - feed.sulfur - feed.nitrogen)
    C_liq = liquids - liq_el["H"] - liq_el["S"] - liq_el["N"]
    C_out = sum(sp.element_flow(s, "C") for s in gas_streams) + C_liq
    T0 = sp.T_REF
    e_in = (Fo * (p["cp_vapor"] * (p["feed_T"] - T0) - p["latent_heat"] - p["heat_of_cracking"])
            + Fs * p["cp_steam"] * (p["steam_T"] - T0) + sp.flue_enthalpy(comb["air"], p["air_T"]))
    e_out = (Fo * ((1.0 - y_coke) * p["cp_vapor"] * (ROT - T0) - p["heat_of_cracking"] * y_go)
             + Fs * p["cp_steam"] * (ROT - T0) + sp.flue_enthalpy(comb["flue"], T_rg)
             + p["regen_heat_loss"])
    e_scale = Fo * (p["latent_heat"] + p["heat_of_cracking"]) + abs(sp.flue_enthalpy(comb["flue"], T_rg))
    balances = {
        "mass": (m_out - m_in) / m_in,
        "C": (C_out - C_in) / C_in,
        "H": (el_out_["H"] - el_in["H"]) / el_in["H"],
        "S": (el_out_["S"] - el_in["S"]) / el_in["S"],
        "N": (el_out_["N"] - el_in["N"]) / el_in["N"],
        "energy": (e_out - e_in) / e_scale,
    }

    yields = {"dry_gas": (m_dg_hc + m_h2s + m_h2_metals) / Fo, "lpg": y_lpg,
              "c3": sum(lpg[n] for n in sp.C3_SPECIES) / Fo,
              "c4": sum(lpg[n] for n in sp.C4_SPECIES) / Fo,
              "gasoline_lump": y_gs, "cycle_oil": y_go, "coke": y_coke,
              "h2s": m_h2s / Fo,
              **{k: props[k]["mass"] / Fo for k in ("gasoline", "lco", "slurry")}}
    flue_tot = sum(comb["flue"].values())
    outputs = {
        "conversion": X,
        "cat_oil": CTO,
        "catalyst_circulation": CTO * Fo,
        "regenerator_T": T_rg,
        "regenerator_margin": p["regenerator_T_max"] - T_rg,
        "riser_outlet_T": ROT,
        "mix_T": r["T_mix"],
        "residence_time": r["t_c"],
        "activity_out": r["phi_out"],
        "coke_on_catalyst": r["coke_on_cat"],
        "feed_multiplier": r["multiplier"],
        "air_rate": air_mass,
        "flue_o2": comb["flue"]["oxygen"] / flue_tot,
        "flue_co": comb["flue"]["carbon_monoxide"] / flue_tot,
        "flue_co2": comb["flue"]["carbon_dioxide"] / flue_tot,
        "co_co2": comb["co_co2"],
        "coke_burn": coke,
        "regenerator_heat_residual": q_regen,
        "c3_olefins": lpg["propylene"] / sum(lpg[n] for n in sp.C3_SPECIES),
        "c4_olefins": sum(lpg[n] for n in sp.C4_OLEFINS) / sum(lpg[n] for n in sp.C4_SPECIES),
        "cycle_oil_hydrogen": h_co,
        **{f"yield.{k}": v for k, v in yields.items()},
        **{f"{prod}.{k}": v for prod in props for k, v in props[prod].items()},
    }
    return {"x": x, "outputs": outputs, "streams": streams, "balances": balances,
            "groups": groups, "lumps": r["y"]}


# =============================================================================
# The unit
# =============================================================================


class FCCUnit:
    """Fluid catalytic cracker: riser + regenerator (heat-balanced) + main fractionator.

    Example:
        >>> import difflow_refinery as dr                    # doctest: +SKIP
        >>> from difflow_refinery.fcc import FCCUnit, FCCParams, FCCFeed
        >>> feed = FCCFeed.from_characterization(char, rate=50.0, T_lo=616.0, T_hi=823.0)
        >>> res = FCCUnit(FCCParams(riser_outlet_T=800.0)).solve(feed)
        >>> res["outputs"]["conversion"], res["outputs"]["regenerator_T"]
    """

    OUTLETS = ("dry_gas", "c3", "c4", "gasoline", "lco", "slurry", "sour_water", "flue_gas")
    PSEUDO_NAMES = frac.PSEUDO_NAMES

    def __init__(self, params: FCCParams | None = None, components=None):
        """
        Args:
            params: :class:`FCCParams`.
            components: Property table of the feed stream's species (VDU
                :class:`~difflow_refinery.vacuum.PseudoComponents`, or a
                characterization's ``pseudo_components()``); needed only to
                call the unit on a stream.
        """
        self.params = params or FCCParams()
        self.components = components
        p = self.params
        self.scheme = get_scheme(p.scheme)
        if p.deactivation not in DEFAULT_DEACTIVATION_ALPHA:
            raise ValueError(f"deactivation must be 'time' or 'coke', not {p.deactivation!r}")
        if isinstance(p.co_co2, str) and p.co_co2 != "arthur":
            raise ValueError(f"co_co2 must be a number or 'arthur', not {p.co_co2!r}")
        self.static = (int(p.n_steps), p.deactivation,
                       "arthur" if p.co_co2 == "arthur" else "fixed",
                       "flue_o2" if p.air_rate is None else "air_rate")
        self._run = _compiled(self.scheme, self.static, int(p.max_iter), float(p.tol))

    def theta(self, feed: FCCFeed, **overrides) -> dict:
        """The differentiable inputs of a solve (a pytree)."""
        params = numeric_params(self.params, self.scheme)
        for k, v in overrides.items():
            if k not in params:
                raise KeyError(f"{k!r} is not a numeric FCC parameter")
            params[k] = jnp.asarray(v, dtype=float)
        x0 = jnp.asarray([self.params.CTO_guess, self.params.T_rg_guess / 1000.0])
        return {"params": params, "feed": feed, "x0": x0}

    def solve(self, feed: FCCFeed, theta: dict | None = None, **overrides) -> dict:
        """Solve the heat balance; return outputs, streams (mol/s), balances.

        Args:
            feed: The :class:`~difflow_refinery.fcc.FCCFeed`.
            theta: A pytree from :meth:`theta` (for differentiating with
                respect to any parameter); default built from the params.
            **overrides: Numeric parameters to replace (traceable), e.g.
                ``riser_outlet_T=jnp.asarray(800.0)``.
        """
        if theta is None:
            theta = self.theta(feed, **overrides)
        res = self._run(theta)
        res["converged"] = jnp.logical_and(res["solver_ok"], res["residual"] < 1e-8)
        conv = _concrete(res["converged"])
        if conv is not None and not conv:
            warnings.warn(f"FCC heat balance did not converge: residual "
                          f"{float(res['residual']):.2e} after {int(res['iterations'])} iterations",
                          FCCConvergenceWarning, stacklevel=2)
        margin = _concrete_float(res["outputs"]["regenerator_margin"])
        if margin is not None and margin < 0:
            warnings.warn(f"regenerator temperature is {-margin:.1f} K above "
                          f"regenerator_T_max", RegeneratorTemperatureWarning, stacklevel=2)
        return res

    def profile(self, feed: FCCFeed, x, n_points: int = 61, **overrides) -> dict:
        """Riser profiles along the height at a solved ``x = res["x"]``.

        Returns ``z`` (m), lump mass fractions ``y``, product-group fractions
        ``groups`` (:data:`~difflow_refinery.fcc.kinetics.GROUPS` order),
        catalyst residence time ``t_c`` (s) and temperature ``T`` (K).
        """
        theta = self.theta(feed, **overrides)
        p = theta["params"]
        M, _, run_riser, _, _ = _model(theta, self.scheme, self.static,
                                       ts=jnp.linspace(0.0, p["riser_height"], n_points))
        out = run_riser(x[0], x[1] * 1000.0)
        out["groups"] = out["y"] @ M
        return out

    def __call__(self, feed_stream):
        """Run on a difflow stream. Returns the eight outlet streams and an info dict.

        Feed species not on ``components`` are passed through to ``dry_gas``.
        """
        if self.components is None:
            raise ValueError("FCCUnit needs components= (the feed's property table) "
                             "to be called on a stream; or call solve(FCCFeed)")
        feed = FCCFeed.from_stream(feed_stream, self.components)
        res = self.solve(feed)
        T_rx = res["outputs"]["riser_outlet_T"]
        P = self.params.riser_P
        out = []
        own = {f"F_{n}" for n in self.components.names}
        for name in self.OUTLETS:
            flows = res["streams"][name]
            if name in ("gasoline", "lco", "slurry"):
                st = {f"F_{n}": flows[i] for i, n in enumerate(self.PSEUDO_NAMES)}
                st["phase"] = "liquid"
            else:
                st = {f"F_{n}": v for n, v in flows.items()}
                st["phase"] = "vapor"
            st["T"] = res["outputs"]["regenerator_T"] if name == "flue_gas" else T_rx
            st["P"] = P
            if name == "dry_gas":
                for k in feed_stream.keys():
                    if isinstance(k, str) and k.startswith("F_") and k not in own:
                        st[k] = st.get(k, 0.0) + feed_stream[k]
            out.append(st)
        return (*out, res)


def _concrete(v):
    try:
        return bool(v)
    except Exception:
        return None


def _concrete_float(v):
    try:
        return float(v)
    except Exception:
        return None
