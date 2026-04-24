"""Amine stripper/regenerator for CO2 capture.

This module provides a simplified equilibrium-based stripper model
for regenerating amine solvents. The key output is the regeneration
energy (GJ/tonne CO2), which is critical for process economics.

References:
    Rochelle GT (2009). Amine scrubbing for CO2 capture. Science 325:1652.
    Abu-Zahra MRM et al. (2007). CO2 capture from power plants.
        Part I. A parametric study of the technical performance.
        Int J Greenhouse Gas Control 1:37-46.
"""

__all__ = [
    "StripperParams",
    "AmineStripper",
]

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream, make_stream, get_flows, total_flow
from difflow.params_mixin import ParamsMixin
from difflow.numerics import safe_divide
from difflow_cc.database import get_solvent
from difflow_cc.equilibrium.vle import AmineVLE


# =============================================================================
# Stripper Parameters
# =============================================================================

@dataclass(repr=False)
class StripperParams(ParamsMixin):
    """Parameters for amine stripper/regenerator.

    Attributes:
        solvent: Amine solvent name
        n_stages: Number of theoretical stages
        T_reboiler: Reboiler temperature (K)
        P_stripper: Operating pressure (Pa)
        reflux_ratio: Condenser reflux ratio
        target_lean_loading: Target lean solvent loading (mol CO2/mol amine)
        reboiler_duty: Fixed reboiler duty (W), if None calculated from target

    Notes:
        The stripper model calculates:
        - Lean loading achievable at given conditions
        - Reboiler duty required
        - Specific regeneration energy (GJ/tonne CO2)
        - CO2 product purity

        Energy components:
        - Sensible heat: heating solvent to reboiler temperature
        - Heat of reaction: reversing CO2-amine reaction
        - Heat of vaporization: generating stripping steam
    """
    solvent: str
    n_stages: int | float | Array = 8
    T_reboiler: float | Array = 393.15  # K (120°C)
    P_stripper: float | Array = 200000.0  # Pa (2 bar)
    reflux_ratio: float | Array = 0.3
    target_lean_loading: float | Array = 0.2  # mol CO2/mol amine
    reboiler_duty: float | Array | None = None  # W

    # Heat integration
    cross_exchanger_approach: float = 10.0  # K






# =============================================================================
# Amine Stripper
# =============================================================================

class AmineStripper:
    """Amine stripper for solvent regeneration.

    Regenerates the rich amine solvent by heating to release CO2.
    The main design trade-off is between lean loading achieved
    and regeneration energy consumed.

    Example:
        >>> params = StripperParams(
        ...     solvent='MEA',
        ...     T_reboiler=393.15,
        ...     target_lean_loading=0.2,
        ... )
        >>> stripper = AmineStripper(params)
        >>> lean_solvent, co2_product, info = stripper(rich_solvent)
        >>> print(f"Regen energy: {info['specific_energy']:.2f} GJ/tonne CO2")

    The model calculates:
    - Reboiler duty for target lean loading
    - Specific regeneration energy
    - CO2 product stream (purity, flow rate)
    - Heat integration benefits

    Key sensitivities for optimization:
    - Lower T_reboiler reduces energy but limits stripping
    - Higher P_stripper reduces column size but increases energy
    - Heat integration via cross-exchanger is critical

    References:
        Rochelle GT (2009). Amine scrubbing for CO2 capture.
            Science 325:1652-1654.
        Abu-Zahra MRM et al. (2007). Int J Greenhouse Gas Control 1:37.
    """

    symbol = "Amine Stripper"
    equations = [
        r"Q_\mathrm{reb} = L\,C_p\,(T_\mathrm{reb} - T_\mathrm{in}) + \dot{n}_{\mathrm{CO}_2,\mathrm{released}}\,\Delta H_\mathrm{abs} + Q_\mathrm{strip}",
        r"\alpha_\mathrm{lean} = \alpha_\mathrm{rich} - \frac{\dot{n}_{\mathrm{CO}_2}}{L\,c_\mathrm{amine}}",
        r"E_\mathrm{specific} = \frac{Q_\mathrm{reb}}{\dot{m}_{\mathrm{CO}_2}}\qquad [\mathrm{GJ/t\,CO_2}]",
    ]
    assumptions = [
        "Reboiler duty covers sensible, reaction and stripping steam contributions.",
        "Column pressure is specified; steam is released at the reboiler.",
        "Amine loading-enthalpy correlation from the selected solvent database entry.",
    ]
    references = [
        "Rochelle, G.T. Science, 325(5948), 1652 (2009).",
        "Abu-Zahra, M.R.M. et al. Int. J. Greenhouse Gas Control, 1, 37 (2007).",
        "Kohl, A.L., Nielsen, R.B. Gas Purification, 5e, Gulf Publishing, 1997.",
    ]
    parameter_symbols = {
        "T_reboiler": "T_\\mathrm{reb}",
        "P_stripper": "P",
        "target_lean_loading": r"\alpha_\mathrm{lean}",
    }
    parameter_units = {"T_reboiler": "K", "P_stripper": "Pa", "target_lean_loading": "mol CO2 / mol amine"}
    numerical_method = "Energy balance + VLE-closure loop; specific energy computed post-hoc."

    def __init__(self, params: StripperParams):
        """Initialize stripper.

        Args:
            params: StripperParams dataclass
        """
        self.params = params
        self._solvent_data = get_solvent(params.solvent)
        self._vle = AmineVLE(params.solvent)

    def __call__(
        self,
        rich_solvent: Stream,
        T_feed: Array | float | None = None,
    ) -> tuple[Stream, Stream, dict]:
        """Regenerate rich solvent.

        Args:
            rich_solvent: Rich solvent from absorber
            T_feed: Feed temperature (K), defaults to estimate from rich stream

        Returns:
            lean_solvent: Regenerated lean solvent
            co2_product: CO2 product stream
            info: Dict with operation details:
                - reboiler_duty: Heat input (W)
                - specific_energy: Energy per tonne CO2 (GJ/tonne)
                - lean_loading: Achieved lean loading
                - CO2_purity: Product purity (mol fraction)
        """
        p = self.params
        T_reboiler = jnp.asarray(p.T_reboiler)
        P_stripper = jnp.asarray(p.P_stripper)
        target_lean = jnp.asarray(p.target_lean_loading)

        # Get rich solvent composition
        rich_flows = get_flows(rich_solvent)
        F_amine = jnp.asarray(rich_flows.get("Amine", 0.0))
        F_H2O = jnp.asarray(rich_flows.get("H2O", 0.0))
        F_CO2_absorbed = jnp.asarray(rich_flows.get("CO2_absorbed", 0.0))

        # Rich loading
        rich_loading = safe_divide(F_CO2_absorbed, F_amine)

        # Feed temperature
        if T_feed is None:
            T_feed = rich_solvent.get("T", 313.15)
        T_feed = jnp.asarray(T_feed)

        # Calculate equilibrium at reboiler conditions
        # At high T, P_CO2_eq is high, driving CO2 out
        P_CO2_eq_lean = self._vle.equilibrium_pressure(target_lean, T_reboiler)

        # Simplified stripping model: lean loading depends on stages and energy
        # More stages and higher reboiler duty -> lower (better) lean loading
        # At minimum stages/energy, lean loading approaches rich loading
        n_stages = jnp.asarray(p.n_stages)
        # Stripping efficiency: fraction of possible stripping achieved
        # Approaches 1.0 with many stages, 0.0 with few
        strip_efficiency = 1.0 - jnp.exp(-0.3 * n_stages)
        lean_loading = rich_loading - strip_efficiency * (rich_loading - target_lean)
        lean_loading = jnp.clip(lean_loading, target_lean, rich_loading)

        # CO2 stripped = rich CO2 - lean CO2
        # Lean CO2 in liquid = F_amine * lean_loading
        F_CO2_lean = F_amine * lean_loading
        F_CO2_stripped = jnp.maximum(F_CO2_absorbed - F_CO2_lean, 0.0)

        # Energy calculations
        # 1. Sensible heat
        # Cp ~ 4000 J/(kg·K) for amine solution
        Cp_solvent = 4000.0  # J/(kg·K)
        MW_amine = self._solvent_data.MW
        MW_water = 18.0

        # Mass flow (approximate)
        m_amine = F_amine * MW_amine / 1000  # kg/s
        m_water = F_H2O * MW_water / 1000  # kg/s
        m_total = m_amine + m_water

        # Rich solvent enters at absorber temperature, is preheated by cross-exchanger
        T_rich_in = jnp.asarray(rich_solvent.get("T", 313.15))
        # Cross-exchanger heats rich solvent to within approach of reboiler temp
        # But cannot heat above what the lean solvent can provide
        T_after_hx = jnp.maximum(T_rich_in, T_reboiler - p.cross_exchanger_approach)
        dT_sensible = T_reboiler - T_after_hx
        Q_sensible = m_total * Cp_solvent * dT_sensible  # W

        # 2. Heat of reaction (desorption)
        dH_absorption = self._solvent_data.heat_of_absorption * 1000  # J/mol
        Q_reaction = F_CO2_stripped * dH_absorption  # W

        # 3. Heat of vaporization (stripping steam)
        # Typical steam ratio for MEA at 120°C is 1.5-3.0 mol H2O/mol CO2
        dH_vap_water = 40650  # J/mol
        steam_ratio = 2.0  # mol H2O per mol CO2
        F_steam = F_CO2_stripped * steam_ratio
        Q_vaporization = F_steam * dH_vap_water  # W

        # Condenser recovers some steam
        reflux = jnp.asarray(p.reflux_ratio)
        Q_condenser = Q_vaporization * reflux

        # Total reboiler duty
        Q_reboiler = Q_sensible + Q_reaction + Q_vaporization

        # Specific energy (GJ/tonne CO2)
        # Convert F_CO2_stripped (mol/s) to tonnes/s: * 44/1e6
        m_CO2_tonnes = F_CO2_stripped * 44 / 1e6  # tonnes/s
        specific_energy = safe_divide(Q_reboiler, m_CO2_tonnes) / 1e9  # GJ/tonne

        # CO2 product purity (after condenser)
        # Most water condenses, leaving >95% CO2
        F_CO2_product = F_CO2_stripped
        F_H2O_product = F_steam * (1 - reflux)  # Some water passes through
        CO2_purity = safe_divide(F_CO2_product, F_CO2_product + F_H2O_product)

        # Create output streams
        # Lean solvent
        lean_flows = {
            "H2O": jnp.maximum(0.0, F_H2O - F_steam * (1 - reflux)),
            "Amine": F_amine,
            "CO2_absorbed": F_CO2_lean,
        }
        lean_solvent = make_stream(lean_flows, T_reboiler, P_stripper)

        # CO2 product
        T_condenser = 313.15  # K (40°C after condensing)
        co2_flows = {
            "CO2": F_CO2_product,
            "H2O": F_H2O_product,
        }
        co2_product = make_stream(co2_flows, T_condenser, P_stripper)

        info = {
            "reboiler_duty": Q_reboiler,
            "Q_sensible": Q_sensible,
            "Q_reaction": Q_reaction,
            "Q_vaporization": Q_vaporization,
            "Q_condenser": Q_condenser,
            "specific_energy": specific_energy,
            "rich_loading": rich_loading,
            "lean_loading": lean_loading,
            "CO2_stripped": F_CO2_stripped,
            "CO2_purity": CO2_purity,
            "T_reboiler": T_reboiler,
            "T_feed": T_feed,
        }

        return lean_solvent, co2_product, info

    def minimum_reboiler_temperature(
        self,
        lean_loading: Array | float,
        P_CO2_overhead: Array | float = 100000.0
    ) -> Array:
        """Calculate minimum reboiler temperature for target lean loading.

        The reboiler must be hot enough that equilibrium P_CO2
        exceeds the overhead pressure, driving stripping.

        Args:
            lean_loading: Target lean loading (mol CO2/mol amine)
            P_CO2_overhead: Required CO2 partial pressure overhead (Pa)

        Returns:
            Minimum reboiler temperature (K)

        Notes:
            This is solved iteratively using the VLE model.
            A safety margin (typically 10-20 K) should be added.
        """
        lean_loading = jnp.asarray(lean_loading)
        P_CO2_overhead = jnp.asarray(P_CO2_overhead)

        # Simple estimation: use correlation
        # T_min increases with lower lean loading (harder to strip)
        # and higher P_CO2_overhead

        dH = self._solvent_data.heat_of_absorption * 1000  # J/mol
        alpha_max = self._solvent_data.loading_capacity

        # From modified van't Hoff:
        # ln(P2/P1) = -dH/R * (1/T2 - 1/T1)
        # Estimate from reference point
        T_ref = 393.15  # K (120°C) - typical MEA reboiler
        P_ref = 100000.0  # Pa - typical overhead pressure

        # Adjust for loading (lower loading needs higher T)
        loading_factor = (alpha_max - lean_loading) / (alpha_max - 0.2)

        T_min = T_ref * loading_factor * jnp.power(P_CO2_overhead / P_ref, 0.1)

        return jnp.clip(T_min, 353.15, 453.15)  # Between 80-180°C
