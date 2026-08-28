"""Combustion and gas-turbine (Brayton-cycle) unit operations.

difflow had no combustion or gas-turbine units, so gas-processing flowsheets that
produce power -- combined-cycle plants, cogeneration, fired heaters -- could not
be built. This module adds three ideal-gas units plus a cycle assembly (see
issue #173):

- :class:`Combustor` -- complete-combustion energy balance for a hydrocarbon
  fuel + air. Either solves the adiabatic flame temperature (both feeds fixed)
  or solves the air/fuel ratio to hit a specified turbine-inlet (firing)
  temperature.
- :class:`GasCompressor` -- isentropic compression to a pressure ratio with an
  isentropic-efficiency correction.
- :class:`GasTurbine` -- isentropic expansion to a back-pressure with an
  isentropic-efficiency correction (the Brayton power turbine / expander).
- :func:`brayton_cycle` -- assembles the three into an intensive (per mole of
  fuel) simple- or combined-cycle solve, validated against F-class machines.

The working fluid is air and combustion gas at high temperature and moderate
pressure, where the Peng-Robinson departure is negligible, so these units use
ideal-gas properties with temperature-dependent Cp
(:class:`difflow.combustion.IdealGasThermo`) rather than the cubic-EOS flash
path of :mod:`difflow.units.eos_units`. They are named distinctly from that
module's ``Compressor``/``Turboexpander`` (which are real-gas, two-phase aware)
because they are a different thermodynamic model for a different service.

Every internal temperature (and air-ratio) solve is an ``optimistix`` root find,
so shaft work, firing temperature, air/fuel ratio and efficiency are all
implicitly differentiable with respect to feed conditions, pressures,
efficiencies and fuel composition.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
import optimistix as optx

from difflow.streams import Stream, make_stream, get_flows
from difflow.params_mixin import ParamsMixin
from difflow.combustion import (
    IdealGasThermo,
    CYCLE_SPECIES,
    FUEL_COMPONENTS,
    FUEL_COMBUSTION,
    AIR_COMPOSITION,
)


# Temperature-search window (K) for the enthalpy/entropy root finds: wide enough
# to span ambient air through adiabatic flame temperatures.
_T_BOUNDS = (150.0, 3000.0)


def make_cycle_thermo(fuel_components: tuple[str, ...] = FUEL_COMPONENTS) -> IdealGasThermo:
    """Build an IdealGasThermo spanning the cycle species and the fuel species.

    A single thermo covering air/product species and the fuel components lets one
    object serve the compressor (air), the combustor (fuel + air -> products) and
    the turbine (products).
    """
    species = list(CYCLE_SPECIES) + [c for c in fuel_components if c not in CYCLE_SPECIES]
    return IdealGasThermo(tuple(species))


def _solve_isentropic_T(thermo, n, T_in, P_in, P_out, T_guess) -> Array:
    """Temperature at P_out with the same entropy as (T_in, P_in) for mixture n."""
    s_in = thermo.mixture_entropy(n, T_in, P_in)
    lo, hi = _T_BOUNDS

    def resid(T, args):
        return thermo.mixture_entropy(n, jnp.clip(T, lo, hi), P_out) - s_in

    sol = optx.root_find(resid, optx.Newton(rtol=1e-9, atol=1e-6),
                         jnp.asarray(T_guess, dtype=float), max_steps=100, throw=False)
    return jnp.clip(sol.value, lo, hi)


def _solve_T_from_enthalpy(thermo, n, H_target, T_guess) -> Array:
    """Temperature at which mixture sensible enthalpy equals H_target."""
    lo, hi = _T_BOUNDS

    def resid(T, args):
        return thermo.mixture_enthalpy(n, jnp.clip(T, lo, hi)) - H_target

    sol = optx.root_find(resid, optx.Newton(rtol=1e-9, atol=1e-6),
                         jnp.asarray(T_guess, dtype=float), max_steps=100, throw=False)
    return jnp.clip(sol.value, lo, hi)


# =============================================================================
# Compressor (ideal-gas, isentropic with efficiency)
# =============================================================================
@dataclass
class GasCompressorParams(ParamsMixin):
    """Parameters for :class:`GasCompressor`.

    Attributes:
        pressure_ratio: Discharge/suction pressure ratio (> 1).
        eta_isentropic: Isentropic efficiency in (0, 1].
    """

    pressure_ratio: float
    eta_isentropic: float = 0.89


class GasCompressor:
    """Adiabatic ideal-gas compressor: isentropic compression + efficiency.

    Matches entropy to find the reversible outlet at ``pressure_ratio * P_in``,
    then the isentropic efficiency inflates the enthalpy rise (an inefficient
    machine needs more work)::

        H_out = H_in + (H_isen - H_in) / eta

    The required shaft work is ``W = H_out - H_in`` (positive, in W).
    """

    symbol = "GCOMP"
    equations = [
        r"S(T_{\mathrm{isen}}, r_p P_{\mathrm{in}}) = S(T_{\mathrm{in}}, P_{\mathrm{in}})",
        r"H_{\mathrm{out}} = H_{\mathrm{in}} + \frac{H_{\mathrm{isen}} - H_{\mathrm{in}}}{\eta}",
    ]
    assumptions = [
        "Adiabatic, steady state; ideal-gas properties.",
        "Isentropic efficiency inflates the reversible enthalpy rise.",
        "Fixed pressure ratio.",
    ]
    references = [
        "Moran, Shapiro, Boettner, Bailey. Fundamentals of Engineering Thermodynamics, 8e, Ch. 6-9.",
    ]

    def __init__(self, params: GasCompressorParams, thermo: IdealGasThermo):
        self.params = params
        self.thermo = thermo

    def __call__(self, inlet: Stream) -> tuple[Stream, dict]:
        flows = get_flows(inlet)
        n = self.thermo.flow_vector(flows)
        T_in, P_in = inlet["T"], inlet["P"]
        P_out = jnp.asarray(self.params.pressure_ratio) * P_in
        eta = jnp.asarray(self.params.eta_isentropic)

        H_in = self.thermo.mixture_enthalpy(n, T_in)
        T_isen = _solve_isentropic_T(
            self.thermo, n, T_in, P_in, P_out,
            T_guess=T_in * jnp.asarray(self.params.pressure_ratio) ** 0.28)
        H_isen = self.thermo.mixture_enthalpy(n, T_isen)
        H_out = H_in + (H_isen - H_in) / eta
        T_out = _solve_T_from_enthalpy(self.thermo, n, H_out, T_guess=T_isen)
        W = H_out - H_in
        return make_stream(flows, T_out, P_out), {
            "W": W, "T_isen": T_isen, "T_out": T_out, "H_in": H_in, "H_out": H_out,
        }


# =============================================================================
# Gas turbine / expander (ideal-gas, isentropic with efficiency)
# =============================================================================
@dataclass
class GasTurbineParams(ParamsMixin):
    """Parameters for :class:`GasTurbine`.

    Attributes:
        P_out: Discharge (back) pressure (Pa); must be below the inlet pressure.
        eta_isentropic: Isentropic efficiency in (0, 1].
    """

    P_out: float
    eta_isentropic: float = 0.90


class GasTurbine:
    """Adiabatic ideal-gas expander: isentropic expansion + efficiency.

    Matches entropy to find the reversible outlet at ``P_out``, then applies the
    isentropic efficiency to the enthalpy drop::

        H_out = H_in - eta * (H_in - H_isen)

    The extracted shaft work is ``W = H_in - H_out`` (positive, in W).
    """

    symbol = "GT"
    equations = [
        r"S(T_{\mathrm{isen}}, P_{\mathrm{out}}) = S(T_{\mathrm{in}}, P_{\mathrm{in}})",
        r"H_{\mathrm{out}} = H_{\mathrm{in}} - \eta\,(H_{\mathrm{in}} - H_{\mathrm{isen}})",
    ]
    assumptions = [
        "Adiabatic, steady state; ideal-gas properties.",
        "Isentropic efficiency applied to the enthalpy drop.",
    ]
    references = [
        "Moran, Shapiro, Boettner, Bailey. Fundamentals of Engineering Thermodynamics, 8e, Ch. 6-9.",
    ]

    def __init__(self, params: GasTurbineParams, thermo: IdealGasThermo):
        self.params = params
        self.thermo = thermo

    def __call__(self, inlet: Stream) -> tuple[Stream, dict]:
        flows = get_flows(inlet)
        n = self.thermo.flow_vector(flows)
        T_in, P_in = inlet["T"], inlet["P"]
        P_out = jnp.asarray(self.params.P_out)
        eta = jnp.asarray(self.params.eta_isentropic)

        H_in = self.thermo.mixture_enthalpy(n, T_in)
        T_isen = _solve_isentropic_T(
            self.thermo, n, T_in, P_in, P_out,
            T_guess=T_in * (P_out / P_in) ** 0.28)
        H_isen = self.thermo.mixture_enthalpy(n, T_isen)
        H_out = H_in - eta * (H_in - H_isen)
        T_out = _solve_T_from_enthalpy(self.thermo, n, H_out, T_guess=T_isen)
        W = H_in - H_out
        return make_stream(flows, T_out, P_out), {
            "W": W, "T_isen": T_isen, "T_out": T_out, "H_in": H_in, "H_out": H_out,
        }


# =============================================================================
# Combustor (complete combustion)
# =============================================================================
def _combustion_products(fuel_flows: dict, air_flows: dict) -> tuple[dict, Array]:
    """Complete-combustion product flows and the O2 demanded by the fuel.

    ``CxHy + (x + y/4) O2 -> x CO2 + (y/2) H2O``; diluent CO2/N2 pass through and
    demand no O2. Returns ``(product_flows over CYCLE_SPECIES, o2_demand)``.
    """
    o2_dem = 0.0
    co2 = 0.0
    h2o = 0.0
    n2_fuel = 0.0
    for c, f in fuel_flows.items():
        x, y, _, combustible = FUEL_COMBUSTION[c]
        o2_dem = o2_dem + f * (1.0 if combustible else 0.0) * (x + y / 4.0)
        co2 = co2 + f * x
        h2o = h2o + f * (y / 2.0)
        if c == "nitrogen":
            n2_fuel = n2_fuel + f

    a = air_flows
    products = {
        "carbon_dioxide": co2 + a.get("carbon_dioxide", 0.0),
        "water": h2o + a.get("water", 0.0),
        "nitrogen": n2_fuel + a.get("nitrogen", 0.0),
        "oxygen": a.get("oxygen", 0.0) - o2_dem,
        "argon": a.get("argon", 0.0),
    }
    return products, o2_dem


@dataclass
class CombustorParams(ParamsMixin):
    """Parameters for :class:`Combustor`.

    Attributes:
        mode: ``"adiabatic"`` (both feeds fixed; solve the flame temperature) or
            ``"fixed_T"`` (scale the air stream to hit ``T_out``).
        T_out: Target combustor outlet / firing temperature (K), used only when
            ``mode == "fixed_T"``.
        dp_frac: Fractional pressure loss across the combustor (0-1); the outlet
            pressure is ``(1 - dp_frac) * min(fuel_P, air_P)``.
    """

    mode: str = "adiabatic"
    T_out: float | None = None
    dp_frac: float = 0.0


class Combustor:
    """Complete-combustion reactor for a hydrocarbon fuel and air.

    In ``"adiabatic"`` mode the fuel and air streams are burned as supplied and
    the outlet temperature is solved from the adiabatic energy balance
    (sensible enthalpies referenced to 298.15 K, heat release from the fuel LHV).
    In ``"fixed_T"`` mode the air stream is treated as a composition/temperature
    template and its flow is scaled to the air/fuel ratio that hits ``T_out``
    (the standard gas-turbine firing-temperature specification); the scaling is
    affine in the air amount, so the solve is exact and cheap.

    Returns ``(product_stream, info)`` where ``info`` carries ``T_out``, the
    ``air_scale`` applied, the ``o2_demand`` and the heat released ``Q``.
    """

    symbol = "CMB"
    equations = [
        r"\mathrm{C}_x\mathrm{H}_y + \left(x + \tfrac{y}{4}\right)\mathrm{O}_2 \rightarrow x\,\mathrm{CO}_2 + \tfrac{y}{2}\,\mathrm{H}_2\mathrm{O}",
        r"\sum_i F_i^{\mathrm{in}} H_i(T_{\mathrm{in}}) = \sum_i F_i^{\mathrm{out}} H_i(T_{\mathrm{out}}) \qquad \text{(adiabatic)}",
    ]
    assumptions = [
        "Complete combustion to CO2 and H2O; no dissociation or NOx.",
        "Ideal-gas enthalpies referenced to 298.15 K.",
        "Adiabatic, or a specified outlet temperature.",
    ]
    references = [
        "Turns, S.R. An Introduction to Combustion, 3e, Ch. 2 and 7.",
        "Moran, Shapiro, Boettner, Bailey. Fundamentals of Engineering Thermodynamics, 8e, Ch. 6-9.",
    ]

    def __init__(self, params: CombustorParams, thermo: IdealGasThermo):
        self.params = params
        self.thermo = thermo

    def _heat_release(self, fuel_flows: dict) -> Array:
        return sum(f * FUEL_COMBUSTION[c][2] for c, f in fuel_flows.items())

    def __call__(self, fuel: Stream, air: Stream) -> tuple[Stream, dict]:
        th = self.thermo
        fuel_flows = get_flows(fuel)
        air_flows = get_flows(air)
        T_fuel, T_air = fuel["T"], air["T"]
        P_out = (1.0 - self.params.dp_frac) * jnp.minimum(fuel["P"], air["P"])

        Q = self._heat_release(fuel_flows)
        H_fuel = th.mixture_enthalpy(th.flow_vector(fuel_flows), T_fuel)

        if self.params.mode == "fixed_T":
            if self.params.T_out is None:
                raise ValueError("CombustorParams.T_out is required for mode='fixed_T'")
            T_out = jnp.asarray(self.params.T_out)
            # Products are affine in the air scale s: prod(s) = prod_f + s*prod_air.
            # Energy balance H_prod(s, T_out) = H_fuel + s*H_air(T_air) + Q is
            # linear in s -> solve in closed form.
            prod1, _ = _combustion_products(fuel_flows, air_flows)     # s = 1
            prod0, _ = _combustion_products(fuel_flows, {})           # s = 0 (fuel only)
            n1 = th.flow_vector(prod1)
            n0 = th.flow_vector(prod0)
            h = th.h_sensible(T_out)
            Hp0 = jnp.sum(n0 * h)                       # fuel-derived product H at T_out
            Hp_air = jnp.sum((n1 - n0) * h)             # air-derived product H at T_out, per unit s
            H_air_in = th.mixture_enthalpy(th.flow_vector(air_flows), T_air)  # per unit s
            # Hp0 + s*Hp_air = H_fuel + s*H_air_in + Q  ->  s = (H_fuel + Q - Hp0)/(Hp_air - H_air_in)
            s = (H_fuel + Q - Hp0) / (Hp_air - H_air_in)
            scaled_air = {k: s * v for k, v in air_flows.items()}
            products, o2_dem = _combustion_products(fuel_flows, scaled_air)
            return make_stream(products, T_out, P_out), {
                "T_out": T_out, "air_scale": s, "o2_demand": o2_dem, "Q": Q,
            }

        # Adiabatic: both feeds fixed; solve the flame temperature.
        products, o2_dem = _combustion_products(fuel_flows, air_flows)
        n_prod = th.flow_vector(products)
        H_air = th.mixture_enthalpy(th.flow_vector(air_flows), T_air)
        H_target = H_fuel + H_air + Q
        T_out = _solve_T_from_enthalpy(th, n_prod, H_target, T_guess=jnp.asarray(1500.0))
        return make_stream(products, T_out, P_out), {
            "T_out": T_out, "air_scale": jnp.asarray(1.0), "o2_demand": o2_dem, "Q": Q,
        }


# =============================================================================
# Brayton / combined cycle assembly
# =============================================================================
@dataclass
class BraytonCycleParams(ParamsMixin):
    """Design/operating parameters for :func:`brayton_cycle`.

    Defaults are a modern F-class machine at ISO conditions.

    Attributes:
        pressure_ratio: Compressor pressure ratio.
        eta_compressor: Compressor isentropic efficiency.
        eta_turbine: Turbine isentropic efficiency.
        eta_generator: Generator electrical efficiency.
        mechanical_eff: Shaft/bearing/auxiliary mechanical efficiency.
        combustor_dp_frac: Combustor + inlet/exhaust fractional pressure loss.
        TIT_K: Turbine-inlet (firing) temperature (K).
        T_ambient_K: Ambient air temperature (K).
        P_ambient_Pa: Ambient pressure (Pa).
        T_fuel_K: Fuel supply temperature (K).
        combined_cycle: If True, add a steam bottoming cycle on the exhaust heat.
        T_stack_K: HRSG stack outlet temperature (K).
        eta_bottoming: Steam-Rankine efficiency on the recovered exhaust heat.
    """

    pressure_ratio: float = 18.0
    eta_compressor: float = 0.89
    eta_turbine: float = 0.90
    eta_generator: float = 0.985
    mechanical_eff: float = 0.99
    combustor_dp_frac: float = 0.04
    TIT_K: float = 1673.15
    T_ambient_K: float = 288.15
    P_ambient_Pa: float = 101325.0
    T_fuel_K: float = 298.15
    combined_cycle: bool = True
    T_stack_K: float = 380.0
    eta_bottoming: float = 0.33


def brayton_cycle(
    fuel_composition: dict,
    params: BraytonCycleParams = None,
    fuel_components: tuple[str, ...] = FUEL_COMPONENTS,
) -> dict:
    """Solve the intensive (per mole of fuel) simple- or combined-cycle plant.

    Composes :class:`GasCompressor`, :class:`Combustor` (fixed firing temperature)
    and :class:`GasTurbine`. Because the cycle is intensive, thermal efficiency is
    independent of machine size and power scales linearly with fuel flow.

    Args:
        fuel_composition: Fuel mole fractions ``{species: fraction}``.
        params: :class:`BraytonCycleParams` (defaults to an F-class CCGT).
        fuel_components: Fuel species order for the working-fluid thermo.

    Returns:
        Dict with ``eta_thermal`` (LHV net electrical efficiency), ``eta_gt_only``,
        ``work_net`` (J per mol fuel, electrical), stage temperatures, the
        air/fuel molar ratio, and the products.
    """
    if params is None:
        params = BraytonCycleParams()
    thermo = make_cycle_thermo(fuel_components)

    P1 = jnp.asarray(params.P_ambient_Pa)
    T1 = jnp.asarray(params.T_ambient_K)

    # --- Compressor: work per mole of air (intensive) ------------------
    air_1mol = make_stream(dict(AIR_COMPOSITION), T1, P1)
    comp = GasCompressor(
        GasCompressorParams(pressure_ratio=params.pressure_ratio,
                            eta_isentropic=params.eta_compressor), thermo)
    air_comp, comp_info = comp(air_1mol)
    T2 = air_comp["T"]
    P2 = air_comp["P"]
    w_comp_per_air = comp_info["W"]

    # --- Combustor: air/fuel ratio to hit the firing temperature -------
    fuel_1mol = make_stream(
        {c: fuel_composition.get(c, 0.0) for c in fuel_components},
        jnp.asarray(params.T_fuel_K), P2)
    combustor = Combustor(
        CombustorParams(mode="fixed_T", T_out=params.TIT_K,
                        dp_frac=params.combustor_dp_frac), thermo)
    products, comb_info = combustor(fuel_1mol, air_comp)
    lam = comb_info["air_scale"]           # mol air / mol fuel
    P3 = products["P"]

    # --- Turbine: expand products to ambient (per mole fuel) -----------
    turbine = GasTurbine(
        GasTurbineParams(P_out=params.P_ambient_Pa,
                        eta_isentropic=params.eta_turbine), thermo)
    exhaust, turb_info = turbine(products)
    w_turb = turb_info["W"]                 # J per mol fuel
    T4 = exhaust["T"]

    w_comp = lam * w_comp_per_air           # J per mol fuel
    w_shaft = (w_turb - w_comp) * params.mechanical_eff
    w_gt_electric = w_shaft * params.eta_generator

    # --- Combined-cycle bottoming (optional) ---------------------------
    n_prod = thermo.flow_vector(get_flows(exhaust))
    h_T4 = thermo.mixture_enthalpy(n_prod, T4)
    h_stack = thermo.mixture_enthalpy(n_prod, jnp.asarray(params.T_stack_K))
    q_exhaust = h_T4 - h_stack
    w_steam = jnp.where(
        params.combined_cycle,
        params.eta_bottoming * q_exhaust * params.eta_generator, 0.0)

    LHV = comb_info["Q"]                    # per mol fuel (fuel flow = 1)
    w_net = w_gt_electric + w_steam
    eta = w_net / LHV

    return {
        "eta_thermal": eta,
        "eta_gt_only": w_gt_electric / LHV,
        "work_net": w_net,
        "work_gt_electric": w_gt_electric,
        "work_steam": w_steam,
        "work_turbine": w_turb,
        "work_compressor": w_comp,
        "air_fuel_molar": lam,
        "LHV_molar": LHV,
        "T_compressor_out": T2,
        "T_turbine_out": T4,
        "q_exhaust": q_exhaust,
        "products": products,
    }
