"""Physical, kinetic and thermal properties of the treating solvents.

Every constant names its source; ``ILLUSTRATIVE`` marks a value that was
NOT read in a source (recalled, or an assumption) and should be replaced by
a measured one before the answer is relied on. The source tables were
checked when the module was written (not a per-commit test).

Units: SI throughout (mol, m, s, Pa, K, J), except the viscosities, which
are mPa s (only their ratios enter).

Sources
-------
* [VvS88k] Versteeg G.F., van Swaaij W.P.M. (1988) "On the kinetics
  between CO2 and alkanolamines both in aqueous and non-aqueous solutions
  II. Tertiary amines", Chem. Eng. Sci. 43:587-591,
  doi:10.1016/0009-2509(88)87018-0. MDEA: ``k2 = 1.19e8 exp(-5103/T)``
  m3/(kmol s), 293-333 K (their Table 3: 4.3 at 298 K, 12.4 at 318 K).
* [BD81] Blanc C., Demarais G. (1981) Entropie 102:53, as tabulated by
  Blauwhoff P.M.M., Versteeg G.F., van Swaaij W.P.M. (1983) Chem. Eng.
  Sci. 38:1411-1429, doi:10.1016/0009-2509(83)80077-3, Table 4. DEA:
  ``log10 k2 = 10.4493 - 2274.5/T`` L/(mol s), rate ``k2 [DEA][CO2]``,
  20-60 C. DEA's kinetics are zwitterionic (the apparent order in DEA
  moves between 1 and 2 with concentration), so this is an APPARENT
  first-order constant (Blauwhoff 1983).
* [VvS88d] Versteeg G.F., van Swaaij W.P.M. (1988) "Solubility and
  diffusivity of acid gases (CO2, N2O) in aqueous alkanolamine solutions",
  J. Chem. Eng. Data 33:29-34, doi:10.1021/je00051a011: eq 8
  ``D_CO2 = 2.35e-6 exp(-2119/T)`` m2/s in water; eq 4 solubility
  ``HE_CO2 = 3.54e-7 exp(2044/T)`` mol/(m3 Pa) in water; the modified
  Stokes-Einstein relation ``(D eta^0.8)_solution = (D eta^0.8)_water``.
* [K20] Karunarathne S.S., Eimer D.A., Oi L.E. (2020) Appl. Sci. 10:3196,
  doi:10.3390/app10093196, Table A5: viscosity of aqueous MDEA, 30-50 wt%,
  298-323 K (fitted here; the fit reproduces the table to 1.3%).
* [VvS88d] eq 10 / Table XIII: viscosity of aqueous DEA at 298 K
  (20/25/30 wt%: 1.86/2.33/2.92 mPa s as evaluated from the polynomial;
  fitted here).
* [H09] Huber M.L. et al. (2009) "New international formulation for the
  viscosity of H2O", J. Phys. Chem. Ref. Data 38:101-125,
  doi:10.1063/1.3088050 (IAPWS 2008), values at 293-393 K evaluated with
  CoolProp; a Vogel form fitted here reproduces them to 0.13%.
* [S15] Sander R. (2015) "Compilation of Henry's law constants (version
  4.0) for water as solvent", Atmos. Chem. Phys. 15:4399-4981,
  doi:10.5194/acp-15-4399-2015 (the literature-review rows, as served by
  henrys-law.org): ``H = H0 exp(B (1/T - 1/298.15))``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp

#: reference temperature of the Henry's-law temperature dependence (K)
T0 = 298.15


@dataclass(frozen=True)
class AmineProperties:
    """Constants of one amine for the treating unit.

    Attributes:
        name: Amine name.
        cp_amine: Partial molar heat capacity of the amine in solution
            (J/mol/K). ILLUSTRATIVE.
        dH_CO2: Heat of absorption of CO2 (J/mol, positive = released).
            ILLUSTRATIVE (checked against the Kent-Eisenberg model's own
            van 't Hoff heat in the tests).
        dH_H2S: Heat of absorption of H2S (J/mol, positive = released).
            ILLUSTRATIVE (as ``dH_CO2``).
        k2_A: Pre-exponential of the CO2 + amine rate constant (m3/(mol s)).
        k2_B: Its activation temperature (K): ``k2 = k2_A exp(-k2_B/T)``.
        visc: ``(a, b, c, d)`` of ``ln(eta/eta_water) = w (a + b w + (c + d w)/T)``
            with ``w`` the amine mass fraction.
        rho_slope: ``rho = 1000 (1 + rho_slope w)`` (kg/m3). ILLUSTRATIVE.
    """

    name: str
    cp_amine: float
    dH_CO2: float
    dH_H2S: float
    k2_A: float
    k2_B: float
    visc: tuple
    rho_slope: float

    def k2(self, T):
        """Second-order rate constant of CO2 with the amine (m3/(mol s))."""
        return self.k2_A * jnp.exp(-self.k2_B / T)


AMINE_PROPERTIES = {
    # cp: ILLUSTRATIVE (pure-liquid MDEA ~2.3 kJ/kg/K, recalled; aqueous
    #   excess Cp ignored). Heats: ILLUSTRATIVE, inside the calorimetric
    #   ranges usually quoted (CO2-MDEA 50-60, H2S 40-50 kJ/mol;
    #   Mathonat 1997 doi:10.1016/S0378-3812(97)00182-9, Oscarson
    #   Thermochim. Acta 170:235 -- tables not read).
    "MDEA": AmineProperties(
        "MDEA", cp_amine=270.0, dH_CO2=52.0e3, dH_H2S=40.0e3,
        k2_A=1.19e8 / 1e3, k2_B=5103.0,                       # [VvS88k]
        visc=(-2.88747272, -7.58117516, 1889.01944, 3002.47335),  # [K20] fit
        rho_slope=0.09),
    # Heats: ILLUSTRATIVE (CO2-DEA ~69 kJ/mol in 20 wt%, Carson 2000
    #   doi:10.1006/jcht.2000.0680, from an abstract only).
    "DEA": AmineProperties(
        "DEA", cp_amine=250.0, dH_CO2=69.0e3, dH_H2S=42.0e3,
        k2_A=10**10.4493 / 1e3, k2_B=2274.5 * math.log(10.0),  # [BD81]
        # [VvS88d] at 298 K only; held at the same ratio to water at
        # every T (ILLUSTRATIVE assumption)
        visc=(3.15869077, 2.6849512, 0.0, 0.0),
        rho_slope=0.12),
}


def amine_properties(name: str) -> AmineProperties:
    """The :class:`AmineProperties` of ``name``."""
    try:
        return AMINE_PROPERTIES[name]
    except KeyError:
        raise ValueError(f"no treating properties for amine {name!r}; "
                         f"have {sorted(AMINE_PROPERTIES)}") from None


#: Vogel fit ``ln eta = A + B/(T - C)`` (mPa s) to IAPWS 2008 [H09]
_WATER_VOGEL = (-3.71985144, 575.4848355, 138.46889247)


def water_viscosity(T):
    """Viscosity of liquid water (mPa s), 293-393 K [H09]."""
    A, B, C = _WATER_VOGEL
    return jnp.exp(A + B / (T - C))


def solution_viscosity(name, w_amine, T):
    """Viscosity of the aqueous amine (mPa s) at amine mass fraction ``w_amine``."""
    a, b, c, d = amine_properties(name).visc
    w = w_amine
    return water_viscosity(T) * jnp.exp(w * (a + b * w + (c + d * w) / T))


def co2_diffusivity(name, w_amine, T):
    """CO2 diffusivity in the solution (m2/s): water's [VvS88d eq 8],
    corrected by the modified Stokes-Einstein relation ``D ~ eta^-0.8``."""
    ratio = water_viscosity(T) / solution_viscosity(name, w_amine, T)
    return 2.35e-6 * jnp.exp(-2119.0 / T) * ratio**0.8


def co2_henry_solution(name, w_amine, T):
    """Physical Henry's constant of CO2 (Pa m3/mol), ``p = He c``.

    Water's [VvS88d eq 4]. The N2O analogy would scale it by the N2O
    solubility in the amine solution relative to water; that table was not
    read, so the solution is taken as water here (ILLUSTRATIVE; typically
    within ~30% for 30-50 wt% amines).
    """
    del name, w_amine
    return 1.0 / (3.54e-7 * jnp.exp(2044.0 / T))


def solution_density(name, w_amine):
    """Solution density (kg/m3). ILLUSTRATIVE linear form."""
    return 1000.0 * (1.0 + amine_properties(name).rho_slope * w_amine)


#: Henry's-law solubility ``H0`` (mol/(m3 Pa)) at 298.15 K and ``B`` (K) [S15]
HENRY = {
    "methane": (1.4e-5, 1600.0),
    "ethane": (1.9e-5, 2400.0),
    "propane": (1.5e-5, 2700.0),
    "hydrogen": (7.8e-6, 530.0),
    "nitrogen": (6.4e-6, 1600.0),   # ILLUSTRATIVE (recalled, not read)
}


def dissolved_gas(name, T):
    """Physical solubility (mol/(m3 Pa)) of a non-acid gas in water [S15];
    0 for a gas not in :data:`HENRY` (it is then not dissolved)."""
    T = jnp.asarray(T, dtype=float)
    if name not in HENRY:
        return jnp.zeros_like(T)
    H0, B = HENRY[name]
    return H0 * jnp.exp(B * (1.0 / T - 1.0 / T0))
