"""Adsorption-based CO2 capture units.

This module provides simplified equilibrium models for cyclic
adsorption processes:
- PSA (Pressure Swing Adsorption)
- TSA (Temperature Swing Adsorption)
- VSA (Vacuum Swing Adsorption)
- TVSA (Combined Temperature-Vacuum Swing)

The models calculate working capacity, productivity, and energy
consumption based on adsorption isotherms and cycle parameters.

For detailed breakthrough curve modeling and cycle optimization,
these can be extended with dynamic PDE-based models.

References:
    Ruthven DM (1984). Principles of Adsorption and Adsorption
        Processes. Wiley-Interscience.
    Ruthven DM, Farooq S, Knaebel KS (1994). Pressure Swing
        Adsorption. VCH Publishers.
    Webley PA (2014). Adsorption technology for CO2 separation
        and capture: a perspective. Adsorption 20:225-231.
"""

__all__ = [
    "AdsorptionParams",
    "PSAUnit",
    "TSAUnit",
    "VSAUnit",
    "TVSAUnit",
]

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream, make_stream, get_flows, total_flow
from difflow.params_mixin import ParamsMixin
from difflow.numerics import safe_divide
from difflow_cc.database import get_adsorbent, Adsorbent
from difflow_cc.equilibrium.isotherms import (
    Isotherm,
    get_isotherm,
    working_capacity_PSA,
    working_capacity_TSA,
)


# Gas constant
R = 8.314  # J/(mol*K)


# =============================================================================
# Adsorption Parameters
# =============================================================================

@dataclass(repr=False)
class AdsorptionParams(ParamsMixin):
    """Parameters for adsorption-based CO2 capture.

    Attributes:
        adsorbent: Adsorbent material name from database
        cycle_type: 'PSA', 'TSA', 'VSA', or 'TVSA'
        bed_mass: Adsorbent mass per bed (kg)
        n_beds: Number of beds for continuous operation
        void_fraction: Bed void fraction

        PSA/VSA parameters:
        P_adsorption: Adsorption pressure (Pa)
        P_desorption: Desorption/blowdown pressure (Pa)

        TSA parameters:
        T_adsorption: Adsorption temperature (K)
        T_desorption: Desorption/regeneration temperature (K)

        Cycle timing:
        t_adsorption: Adsorption step time (s)
        t_blowdown: Blowdown/depressurization time (s)
        t_purge: Purge step time (s)
        t_repressure: Repressurization time (s)

        Performance targets:
        CO2_purity_target: Target CO2 product purity (mol fraction)
        CO2_recovery_target: Target CO2 recovery (fraction)

    References:
        Ruthven DM et al. (1994). Pressure Swing Adsorption.
    """
    adsorbent: str
    cycle_type: Literal["PSA", "TSA", "VSA", "TVSA"] = "PSA"
    bed_mass: float | Array = 1000.0  # kg
    n_beds: int = 2
    void_fraction: float | Array = 0.4

    # Pressure conditions
    P_adsorption: float | Array = 101325.0  # Pa (1 atm)
    P_desorption: float | Array = 10000.0  # Pa (0.1 atm for VSA)

    # Temperature conditions
    T_adsorption: float | Array = 298.15  # K (25°C)
    T_desorption: float | Array = 393.15  # K (120°C for TSA)

    # Cycle timing
    t_adsorption: float | Array = 300.0  # s (5 min)
    t_blowdown: float | Array = 60.0  # s
    t_purge: float | Array = 120.0  # s
    t_repressure: float | Array = 60.0  # s

    # Performance targets
    CO2_purity_target: float | Array = 0.95
    CO2_recovery_target: float | Array = 0.90






# =============================================================================
# Base Adsorption Unit
# =============================================================================

class _AdsorptionBase:
    """Base class for adsorption units.

    Provides common functionality for working capacity calculation
    and cycle performance estimation.
    """

    def __init__(self, params: AdsorptionParams):
        """Initialize adsorption unit.

        Args:
            params: AdsorptionParams dataclass
        """
        self.params = params
        self._adsorbent_data = get_adsorbent(params.adsorbent)
        self._isotherm_CO2 = get_isotherm(params.adsorbent, "CO2")

        # Try to get N2 isotherm for selectivity calculations
        try:
            self._isotherm_N2 = get_isotherm(params.adsorbent, "N2")
        except (KeyError, ValueError):
            self._isotherm_N2 = None

    def _working_capacity(
        self,
        P_ads: Array | float,
        P_des: Array | float,
        T_ads: Array | float,
        T_des: Array | float,
    ) -> Array:
        """Calculate CO2 working capacity.

        Args:
            P_ads: Adsorption pressure (Pa)
            P_des: Desorption pressure (Pa)
            T_ads: Adsorption temperature (K)
            T_des: Desorption temperature (K)

        Returns:
            Working capacity (mol/kg)
        """
        q_ads = self._isotherm_CO2(P_ads, T_ads)
        q_des = self._isotherm_CO2(P_des, T_des)
        return jnp.maximum(q_ads - q_des, 0.0)

    def _cycle_time(self) -> Array:
        """Calculate total cycle time.

        Returns:
            Total cycle time (s)
        """
        p = self.params
        return (
            jnp.asarray(p.t_adsorption) +
            jnp.asarray(p.t_blowdown) +
            jnp.asarray(p.t_purge) +
            jnp.asarray(p.t_repressure)
        )

    def _productivity(
        self,
        working_cap: Array,
        bed_mass: Array,
        cycle_time: Array
    ) -> Array:
        """Calculate CO2 productivity.

        Args:
            working_cap: Working capacity (mol/kg)
            bed_mass: Adsorbent mass (kg)
            cycle_time: Cycle time (s)

        Returns:
            Productivity (mol CO2 / (kg adsorbent * hour))
        """
        # CO2 per cycle = working_cap * bed_mass
        # Productivity = CO2_per_cycle / (cycle_time / 3600) / bed_mass
        return working_cap * 3600 / cycle_time


# =============================================================================
# PSA Unit
# =============================================================================

class PSAUnit(_AdsorptionBase):
    """Pressure Swing Adsorption unit for CO2 capture.

    PSA cycles use pressure reduction for regeneration:
    1. Adsorption at high pressure
    2. Blowdown (depressurization)
    3. Purge with product or inert
    4. Repressurization

    Typical applications:
    - Hydrogen purification (pre-combustion)
    - High-pressure gas streams

    Example:
        >>> params = AdsorptionParams(
        ...     adsorbent='Zeolite_13X',
        ...     cycle_type='PSA',
        ...     P_adsorption=500000,  # 5 bar
        ...     P_desorption=100000,  # 1 bar
        ... )
        >>> psa = PSAUnit(params)
        >>> product, offgas, info = psa(feed)

    References:
        Ruthven DM et al. (1994). Pressure Swing Adsorption.
        Sircar S (2002). Pressure swing adsorption.
            Ind Eng Chem Res 41:1389-1392.
    """

    symbol = "PSA"
    equations = [
        r"q(P,T) = \frac{q_\mathrm{max}\,b(T)\,P}{1 + b(T)\,P}\qquad \text{(Langmuir isotherm)}",
        r"\Delta q_\mathrm{working} = q(P_\mathrm{ads},T) - q(P_\mathrm{des},T)",
        r"\mathrm{recovery} = \frac{\Delta q_\mathrm{working}\,m_\mathrm{bed}}{\dot{n}_{\mathrm{CO}_2,\mathrm{feed}}\,t_\mathrm{cycle}}",
    ]
    assumptions = [
        "Langmuir-type equilibrium isotherm for the target CO2 component.",
        "Cyclic steady state at the specified P_ads / P_des.",
        "Lumped working capacity per cycle; no intra-particle diffusion modelling.",
    ]
    references = [
        "Ruthven, D.M., Farooq, S., Knaebel, K.S. Pressure Swing Adsorption, VCH, 1994.",
        "Sircar, S. Ind. Eng. Chem. Res., 41, 1389 (2002).",
    ]
    parameter_symbols = {
        "P_adsorption": "P_\\mathrm{ads}",
        "P_desorption": "P_\\mathrm{des}",
        "cycle_time": "t_\\mathrm{cyc}",
        "bed_mass": "m_\\mathrm{bed}",
    }
    parameter_units = {
        "P_adsorption": "Pa",
        "P_desorption": "Pa",
        "cycle_time": "s",
        "bed_mass": "kg",
    }
    numerical_method = "Langmuir working-capacity evaluation per cycle; recovery from mass balance."

    def __call__(
        self,
        feed: Stream,
        y_CO2_feed: Array | float | None = None,
    ) -> tuple[Stream, Stream, dict]:
        """Perform PSA separation.

        Args:
            feed: Feed gas stream
            y_CO2_feed: CO2 mole fraction in feed (if not in stream)

        Returns:
            product: CO2 product stream
            offgas: Treated gas (N2-rich)
            info: Dict with operation details
        """
        p = self.params
        P_ads = jnp.asarray(p.P_adsorption)
        P_des = jnp.asarray(p.P_desorption)
        T = jnp.asarray(p.T_adsorption)
        bed_mass = jnp.asarray(p.bed_mass)

        # Get feed composition
        feed_flows = get_flows(feed)
        F_total = total_flow(feed)
        F_CO2_in = feed_flows.get("CO2", jnp.array(0.0))

        if y_CO2_feed is not None:
            y_CO2 = jnp.asarray(y_CO2_feed)
        else:
            y_CO2 = safe_divide(F_CO2_in, F_total)

        # Partial pressure of CO2
        P_CO2_ads = y_CO2 * P_ads
        # During desorption, CO2 is enriched relative to feed
        selectivity = self._adsorbent_data.CO2_selectivity
        y_CO2_enriched = jnp.minimum(y_CO2 * selectivity / (1.0 + y_CO2 * (selectivity - 1.0)), 0.95)
        P_CO2_des = y_CO2_enriched * P_des * 0.5

        # Working capacity (isothermal PSA)
        working_cap = self._working_capacity(P_CO2_ads, P_CO2_des, T, T)

        # CO2 captured per cycle
        CO2_per_cycle = working_cap * bed_mass  # mol

        # Cycle time
        t_cycle = self._cycle_time()

        # CO2 flow rate (average, accounting for n_beds)
        n_beds = p.n_beds
        F_CO2_captured = CO2_per_cycle * n_beds / t_cycle  # mol/s

        # Limit by feed CO2
        F_CO2_captured = jnp.minimum(F_CO2_captured, F_CO2_in * 0.95)

        # Recovery
        recovery = safe_divide(F_CO2_captured, F_CO2_in)

        # Purity estimation (simplified equilibrium model).
        # Based on ideal selectivity modified by pressure ratio.
        # Higher pressure ratio improves purity by reducing co-adsorbed species.
        # Real purity requires full breakthrough curve simulation.
        selectivity = self._adsorbent_data.CO2_selectivity
        # Purity depends on selectivity and pressure ratio
        pressure_selectivity = selectivity * jnp.sqrt(P_ads / jnp.maximum(P_des, 1.0))
        purity = pressure_selectivity / (pressure_selectivity + 1.0)
        purity = jnp.clip(purity, 0.0, 0.999)

        # Energy consumption
        # Compression work for repressurization
        # W = n * R * T * ln(P_high/P_low) / efficiency
        ratio = P_ads / P_des
        F_total_captured = safe_divide(F_CO2_captured, jnp.maximum(purity, 0.01))
        W_compression = F_total_captured * R * T * jnp.log(ratio) / 0.7  # W (assume 70% eff)

        # Per tonne CO2
        m_CO2_per_s = F_CO2_captured * 44 / 1e6  # tonnes/s
        energy_GJ_per_tonne = safe_divide(W_compression, m_CO2_per_s) / 1e9  # GJ/tonne

        # Productivity
        productivity = self._productivity(working_cap, bed_mass, t_cycle)

        # Create output streams
        # Product (CO2 rich, at desorption pressure)
        # Derive N2 in product from feed N2 via selectivity (not from purity spec)
        F_N2_in = jnp.asarray(feed_flows.get("N2", 0.0))
        N2_in_product = safe_divide(F_CO2_captured, selectivity)
        N2_in_product = jnp.minimum(N2_in_product, F_N2_in)  # Can't exceed feed N2
        product_flows = {
            "CO2": F_CO2_captured,
            "N2": N2_in_product,
        }
        product = make_stream(product_flows, T, P_des)

        # Offgas (N2 rich, at adsorption pressure)
        offgas_flows = {}
        for species, flow in feed_flows.items():
            if species == "CO2":
                offgas_flows[species] = jnp.maximum(0.0, flow - F_CO2_captured)
            elif species == "N2":
                offgas_flows[species] = jnp.maximum(0.0, flow - N2_in_product)
            else:
                offgas_flows[species] = flow * (1 - 0.01)

        offgas = make_stream(offgas_flows, T, P_ads)

        info = {
            "working_capacity": working_cap,
            "CO2_captured": F_CO2_captured,
            "recovery": recovery,
            "purity": purity,
            "cycle_time": t_cycle,
            "productivity": productivity,
            "compression_power": W_compression,
            "specific_energy": energy_GJ_per_tonne,
            "P_adsorption": P_ads,
            "P_desorption": P_des,
        }

        return product, offgas, info


# =============================================================================
# VSA Unit
# =============================================================================

class VSAUnit(_AdsorptionBase):
    """Vacuum Swing Adsorption unit for CO2 capture.

    VSA uses vacuum for regeneration, enabling operation at
    atmospheric feed pressure. Lower compression costs than PSA
    but requires vacuum pump.

    Typical applications:
    - Post-combustion CO2 capture
    - Biogas upgrading

    Example:
        >>> params = AdsorptionParams(
        ...     adsorbent='Zeolite_13X',
        ...     cycle_type='VSA',
        ...     P_adsorption=101325,  # 1 atm
        ...     P_desorption=10000,   # 0.1 atm
        ... )
        >>> vsa = VSAUnit(params)
        >>> product, offgas, info = vsa(feed)

    References:
        Zhang J et al. (2008). Performance of a four-bed VSA
            process for CO2 capture. Chem Eng Sci 63:1827-1837.
    """

    symbol = "VSA"
    equations = [
        r"q(P,T) = \frac{q_\mathrm{max}\,b(T)\,P}{1 + b(T)\,P}\qquad \text{(Langmuir)}",
        r"\Delta q_\mathrm{working} = q(P_\mathrm{ads},T) - q(P_\mathrm{vac},T)",
        r"P_\mathrm{vac} \ll P_\mathrm{atm}\qquad \Rightarrow \quad \text{large working capacity at modest }T",
    ]
    assumptions = [
        "Vacuum regeneration; feed near-atmospheric pressure.",
        "Langmuir isotherm with CO2 as the preferred adsorbate.",
    ]
    references = ["Zhang, J. et al. Chem. Eng. Sci., 63, 1827 (2008)."]
    parameter_symbols = {"P_adsorption": "P_\\mathrm{ads}", "P_desorption": "P_\\mathrm{vac}"}
    parameter_units = {"P_adsorption": "Pa", "P_desorption": "Pa", "cycle_time": "s", "bed_mass": "kg"}
    numerical_method = "Langmuir working-capacity evaluation under vacuum regeneration."

    def __call__(
        self,
        feed: Stream,
        y_CO2_feed: Array | float | None = None,
    ) -> tuple[Stream, Stream, dict]:
        """Perform VSA separation.

        Similar to PSA but with vacuum regeneration.
        """
        p = self.params
        P_ads = jnp.asarray(p.P_adsorption)
        P_des = jnp.asarray(p.P_desorption)
        T = jnp.asarray(p.T_adsorption)
        bed_mass = jnp.asarray(p.bed_mass)

        feed_flows = get_flows(feed)
        F_total = total_flow(feed)
        F_CO2_in = feed_flows.get("CO2", jnp.array(0.0))

        if y_CO2_feed is not None:
            y_CO2 = jnp.asarray(y_CO2_feed)
        else:
            y_CO2 = safe_divide(F_CO2_in, F_total)

        P_CO2_ads = y_CO2 * P_ads
        P_CO2_des = P_des * 0.3  # CO2 desorbs first

        working_cap = self._working_capacity(P_CO2_ads, P_CO2_des, T, T)
        CO2_per_cycle = working_cap * bed_mass
        t_cycle = self._cycle_time()

        F_CO2_captured = CO2_per_cycle * p.n_beds / t_cycle
        F_CO2_captured = jnp.minimum(F_CO2_captured, F_CO2_in * 0.95)

        recovery = safe_divide(F_CO2_captured, F_CO2_in)
        # Purity estimation (simplified equilibrium model).
        # Based on ideal selectivity modified by pressure ratio.
        # Higher pressure ratio improves purity by reducing co-adsorbed species.
        # Real purity requires full breakthrough curve simulation.
        selectivity = self._adsorbent_data.CO2_selectivity
        # Purity depends on selectivity and pressure ratio
        pressure_selectivity = selectivity * jnp.sqrt(P_ads / jnp.maximum(P_des, 1.0))
        purity = pressure_selectivity / (pressure_selectivity + 1.0)
        purity = jnp.clip(purity, 0.0, 0.999)

        # Vacuum pump work (more significant than PSA compression)
        # W_vacuum ~ n * R * T * ln(P_atm/P_vac) / efficiency
        ratio = P_ads / P_des
        W_vacuum = CO2_per_cycle * p.n_beds / t_cycle * R * T * jnp.log(ratio) / 0.6

        m_CO2_per_s = F_CO2_captured * 44 / 1e6
        energy_GJ_per_tonne = safe_divide(W_vacuum, m_CO2_per_s) / 1e9

        productivity = self._productivity(working_cap, bed_mass, t_cycle)

        product_flows = {
            "CO2": F_CO2_captured,
            "N2": F_CO2_captured * (1 - purity) / purity,
        }
        product = make_stream(product_flows, T, P_des)

        offgas_flows = {}
        for species, flow in feed_flows.items():
            if species == "CO2":
                offgas_flows[species] = flow - F_CO2_captured
            elif species == "N2":
                N2_in_product = product_flows.get("N2", 0.0)
                offgas_flows[species] = flow - N2_in_product
            else:
                offgas_flows[species] = flow * (1 - 0.01)

        offgas = make_stream(offgas_flows, T, P_ads)

        info = {
            "working_capacity": working_cap,
            "CO2_captured": F_CO2_captured,
            "recovery": recovery,
            "purity": purity,
            "cycle_time": t_cycle,
            "productivity": productivity,
            "vacuum_power": W_vacuum,
            "specific_energy": energy_GJ_per_tonne,
            "P_adsorption": P_ads,
            "P_desorption": P_des,
        }

        return product, offgas, info


# =============================================================================
# TSA Unit
# =============================================================================

class TSAUnit(_AdsorptionBase):
    """Temperature Swing Adsorption unit for CO2 capture.

    TSA uses temperature increase for regeneration:
    1. Adsorption at low temperature
    2. Heating to desorb CO2
    3. Cooling back to adsorption temperature

    Higher purity than PSA but longer cycle times due to
    thermal mass. Good for dilute streams (DAC).

    Example:
        >>> params = AdsorptionParams(
        ...     adsorbent='PEI_Silica',
        ...     cycle_type='TSA',
        ...     T_adsorption=298.15,
        ...     T_desorption=373.15,
        ... )
        >>> tsa = TSAUnit(params)
        >>> product, offgas, info = tsa(feed)

    References:
        Webley PA (2014). Adsorption technology for CO2 separation.
            Adsorption 20:225-231.
    """

    symbol = "TSA"
    equations = [
        r"b(T) = b_0 \exp\!\left(-\Delta H_\mathrm{ads}/(R T)\right)",
        r"\Delta q_\mathrm{working} = q(P, T_\mathrm{ads}) - q(P, T_\mathrm{des})",
        r"Q_\mathrm{regen} = (m_\mathrm{bed} C_p + m_\mathrm{CO_2}\,\Delta H_\mathrm{ads})\,(T_\mathrm{des} - T_\mathrm{ads})",
    ]
    assumptions = [
        "Temperature-dependent Langmuir b(T) via van't Hoff equation.",
        "Thermal regeneration via an external heat source; no direct steam stripping here.",
    ]
    references = ["Webley, P.A. Adsorption, 20, 225 (2014)."]
    parameter_symbols = {"T_adsorption": "T_\\mathrm{ads}", "T_desorption": "T_\\mathrm{des}"}
    parameter_units = {"T_adsorption": "K", "T_desorption": "K"}
    numerical_method = "Van't Hoff-scaled Langmuir with analytical working capacity; sensible-heat duty."

    def __call__(
        self,
        feed: Stream,
        y_CO2_feed: Array | float | None = None,
    ) -> tuple[Stream, Stream, dict]:
        """Perform TSA separation."""
        p = self.params
        P = jnp.asarray(p.P_adsorption)
        T_ads = jnp.asarray(p.T_adsorption)
        T_des = jnp.asarray(p.T_desorption)
        bed_mass = jnp.asarray(p.bed_mass)

        feed_flows = get_flows(feed)
        F_total = total_flow(feed)
        F_CO2_in = feed_flows.get("CO2", jnp.array(0.0))

        if y_CO2_feed is not None:
            y_CO2 = jnp.asarray(y_CO2_feed)
        else:
            y_CO2 = safe_divide(F_CO2_in, F_total)

        P_CO2 = y_CO2 * P

        # Temperature swing working capacity
        working_cap = self._working_capacity(P_CO2, P_CO2, T_ads, T_des)
        CO2_per_cycle = working_cap * bed_mass
        t_cycle = self._cycle_time()

        F_CO2_captured = CO2_per_cycle * p.n_beds / t_cycle
        F_CO2_captured = jnp.minimum(F_CO2_captured, F_CO2_in * 0.95)

        recovery = safe_divide(F_CO2_captured, F_CO2_in)
        # Purity estimation (simplified equilibrium model).
        # Based on ideal selectivity modified by temperature ratio.
        # Temperature swing improves selectivity by preferentially desorbing CO2.
        # Real purity requires full breakthrough curve simulation.
        selectivity = self._adsorbent_data.CO2_selectivity
        # Temperature swing improves selectivity
        T_ratio = T_des / jnp.maximum(T_ads, 1.0)
        pressure_selectivity = selectivity * T_ratio
        purity = pressure_selectivity / (pressure_selectivity + 1.0)
        purity = jnp.clip(purity, 0.0, 0.999)

        # Heating energy
        Cp_ads = self._adsorbent_data.heat_capacity  # J/(kg*K)
        Q_sensible = bed_mass * Cp_ads * (T_des - T_ads)  # J per cycle

        # Heat of desorption
        dH_des = self._adsorbent_data.heat_of_adsorption * 1000  # J/mol
        Q_desorption = CO2_per_cycle * dH_des  # J per cycle

        Q_total = Q_sensible + Q_desorption
        Q_rate = Q_total * p.n_beds / t_cycle  # W

        m_CO2_per_s = F_CO2_captured * 44 / 1e6
        energy_GJ_per_tonne = safe_divide(Q_rate, m_CO2_per_s) / 1e9 * 1.0  # Thermal

        productivity = self._productivity(working_cap, bed_mass, t_cycle)

        product_flows = {
            "CO2": F_CO2_captured,
            "N2": F_CO2_captured * (1 - purity) / purity,
        }
        product = make_stream(product_flows, T_des, P)

        offgas_flows = {}
        for species, flow in feed_flows.items():
            if species == "CO2":
                offgas_flows[species] = flow - F_CO2_captured
            elif species == "N2":
                N2_in_product = product_flows.get("N2", 0.0)
                offgas_flows[species] = flow - N2_in_product
            else:
                offgas_flows[species] = flow * (1 - 0.01)

        offgas = make_stream(offgas_flows, T_ads, P)

        info = {
            "working_capacity": working_cap,
            "CO2_captured": F_CO2_captured,
            "recovery": recovery,
            "purity": purity,
            "cycle_time": t_cycle,
            "productivity": productivity,
            "heating_power": Q_rate,
            "Q_sensible": Q_sensible * p.n_beds / t_cycle,
            "Q_desorption": Q_desorption * p.n_beds / t_cycle,
            "specific_energy": energy_GJ_per_tonne,
            "T_adsorption": T_ads,
            "T_desorption": T_des,
        }

        return product, offgas, info


# =============================================================================
# TVSA Unit
# =============================================================================

class TVSAUnit(_AdsorptionBase):
    """Temperature-Vacuum Swing Adsorption unit.

    Combines temperature and vacuum swing for enhanced working
    capacity and flexibility. Emerging technology for DAC and
    industrial applications.

    Example:
        >>> params = AdsorptionParams(
        ...     adsorbent='Mg_MOF_74',
        ...     cycle_type='TVSA',
        ...     T_adsorption=298.15,
        ...     T_desorption=353.15,  # Lower T than pure TSA
        ...     P_desorption=10000,   # Vacuum assist
        ... )
        >>> tvsa = TVSAUnit(params)
        >>> product, offgas, info = tvsa(feed)

    References:
        Elfving J et al. (2017). Modelling of equilibrium working
            capacity of PSA, TSA and TVSA processes.
            J CO2 Util 22:270-277.
    """

    symbol = "TVSA"
    equations = [
        r"\Delta q_\mathrm{working} = q(P_\mathrm{ads}, T_\mathrm{ads}) - q(P_\mathrm{vac}, T_\mathrm{des})",
        r"b(T) = b_0 \exp\!\left(-\Delta H_\mathrm{ads}/(RT)\right)",
    ]
    assumptions = [
        "Combined vacuum + moderate temperature swing regeneration.",
        "Langmuir isotherm with van't Hoff temperature dependence.",
    ]
    references = ["Elfving, J. et al. J. CO2 Utilization, 22, 270 (2017)."]
    parameter_symbols = {
        "T_adsorption": "T_\\mathrm{ads}",
        "T_desorption": "T_\\mathrm{des}",
        "P_adsorption": "P_\\mathrm{ads}",
        "P_desorption": "P_\\mathrm{vac}",
    }
    parameter_units = {
        "T_adsorption": "K",
        "T_desorption": "K",
        "P_adsorption": "Pa",
        "P_desorption": "Pa",
    }
    numerical_method = "Working-capacity difference with van't Hoff-scaled Langmuir."

    def __call__(
        self,
        feed: Stream,
        y_CO2_feed: Array | float | None = None,
    ) -> tuple[Stream, Stream, dict]:
        """Perform TVSA separation."""
        p = self.params
        P_ads = jnp.asarray(p.P_adsorption)
        P_des = jnp.asarray(p.P_desorption)
        T_ads = jnp.asarray(p.T_adsorption)
        T_des = jnp.asarray(p.T_desorption)
        bed_mass = jnp.asarray(p.bed_mass)

        feed_flows = get_flows(feed)
        F_total = total_flow(feed)
        F_CO2_in = feed_flows.get("CO2", jnp.array(0.0))

        if y_CO2_feed is not None:
            y_CO2 = jnp.asarray(y_CO2_feed)
        else:
            y_CO2 = safe_divide(F_CO2_in, F_total)

        P_CO2_ads = y_CO2 * P_ads
        P_CO2_des = P_des * 0.3

        # Combined T and P swing
        working_cap = self._working_capacity(P_CO2_ads, P_CO2_des, T_ads, T_des)
        CO2_per_cycle = working_cap * bed_mass
        t_cycle = self._cycle_time()

        F_CO2_captured = CO2_per_cycle * p.n_beds / t_cycle
        F_CO2_captured = jnp.minimum(F_CO2_captured, F_CO2_in * 0.95)

        recovery = safe_divide(F_CO2_captured, F_CO2_in)
        # Purity estimation (simplified equilibrium model).
        # Based on ideal selectivity modified by temperature ratio and pressure ratio.
        # Combined T and P swing improves selectivity.
        # Real purity requires full breakthrough curve simulation.
        selectivity = self._adsorbent_data.CO2_selectivity
        # Temperature swing improves selectivity
        T_ratio = T_des / jnp.maximum(T_ads, 1.0)
        pressure_selectivity = selectivity * T_ratio * jnp.sqrt(P_ads / jnp.maximum(P_des, 1.0))
        purity = pressure_selectivity / (pressure_selectivity + 1.0)
        purity = jnp.clip(purity, 0.0, 0.999)

        # Energy: heating + vacuum
        Cp_ads = self._adsorbent_data.heat_capacity
        Q_sensible = bed_mass * Cp_ads * (T_des - T_ads)
        dH_des = self._adsorbent_data.heat_of_adsorption * 1000
        Q_desorption = CO2_per_cycle * dH_des
        Q_thermal = (Q_sensible + Q_desorption) * p.n_beds / t_cycle

        ratio = P_ads / P_des
        W_vacuum = CO2_per_cycle * p.n_beds / t_cycle * R * T_des * jnp.log(ratio) / 0.6

        # Total energy (thermal + electrical)
        # Convert electrical to thermal equivalent (factor ~3 for heat pump)
        total_energy = Q_thermal + W_vacuum * 3

        m_CO2_per_s = F_CO2_captured * 44 / 1e6
        energy_GJ_per_tonne = safe_divide(total_energy, m_CO2_per_s) / 1e9

        productivity = self._productivity(working_cap, bed_mass, t_cycle)

        product_flows = {
            "CO2": F_CO2_captured,
            "N2": F_CO2_captured * (1 - purity) / purity,
        }
        product = make_stream(product_flows, T_des, P_des)

        offgas_flows = {}
        for species, flow in feed_flows.items():
            if species == "CO2":
                offgas_flows[species] = flow - F_CO2_captured
            else:
                offgas_flows[species] = flow * (1 - 0.01)

        offgas = make_stream(offgas_flows, T_ads, P_ads)

        info = {
            "working_capacity": working_cap,
            "CO2_captured": F_CO2_captured,
            "recovery": recovery,
            "purity": purity,
            "cycle_time": t_cycle,
            "productivity": productivity,
            "thermal_power": Q_thermal,
            "vacuum_power": W_vacuum,
            "specific_energy": energy_GJ_per_tonne,
            "T_adsorption": T_ads,
            "T_desorption": T_des,
            "P_adsorption": P_ads,
            "P_desorption": P_des,
        }

        return product, offgas, info
