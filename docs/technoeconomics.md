# Technoeconomic Analysis

This document provides comprehensive documentation for the technoeconomic analysis (TEA) capabilities in Difflow, including capital costs, operating costs, utility costs, and profitability metrics.

## Table of Contents

1. [Overview](#overview)
2. [Capital Costs](#capital-costs)
   - [Equipment Cost Correlations](#equipment-cost-correlations)
   - [Installation Factors](#installation-factors)
   - [Total Capital Investment](#total-capital-investment)
3. [Utility Costs](#utility-costs)
   - [Steam](#steam)
   - [Cooling Water](#cooling-water)
   - [Electricity](#electricity)
   - [Refrigeration](#refrigeration)
4. [Operating Costs](#operating-costs)
   - [Raw Materials](#raw-materials)
   - [Labor](#labor)
   - [Overhead and Maintenance](#overhead-and-maintenance)
   - [Total Operating Cost](#total-operating-cost)
5. [Profitability Analysis](#profitability-analysis)
   - [Time Value of Money](#time-value-of-money)
   - [Net Present Value (NPV)](#net-present-value-npv)
   - [Internal Rate of Return (IRR)](#internal-rate-of-return-irr)
   - [Payback Period](#payback-period)
   - [Minimum Selling Price (MSP)](#minimum-selling-price-msp)
   - [Cash Flow Analysis](#cash-flow-analysis)
6. [Cost Indices](#cost-indices)
7. [Examples](#examples)

---

## Overview

The Difflow economics module provides comprehensive technoeconomic analysis capabilities:

```
difflow/economics/
├── capital.py       # Equipment cost correlations
├── utilities.py     # Utility cost models
├── opex.py          # Operating cost calculations
├── profitability.py # Financial metrics (NPV, IRR, MSP)
└── indices.py       # Cost index escalation (CEPCI)
```

**Key Features**:
- All functions are JAX-differentiable for optimization
- CEPCI cost escalation for 2000-2026
- Comprehensive equipment cost database
- Industry-standard financial metrics

---

## Capital Costs

**Location**: `difflow/economics/capital.py`

### Equipment Cost Correlations

Equipment costs follow power-law correlations:

$$C = a + b \cdot S^n$$

Where:
- $C$: Equipment cost ($)
- $S$: Size parameter (characteristic dimension)
- $a, b, n$: Correlation parameters

```python
from typing import NamedTuple
from difflow.economics.capital import CostParams

class CostParams(NamedTuple):
    a: float       # Fixed cost ($)
    b: float       # Scaling coefficient ($)
    n: float       # Scaling exponent
    S_min: float   # Minimum valid size
    S_max: float   # Maximum valid size
    S_units: str   # Size units
    base_year: int # Cost basis year
```

### Equipment Cost Databases

#### Reactors

```python
from difflow.economics.capital import REACTOR_COSTS, CostParams

# Entries have the form CostParams(a, b, n, S_min, S_max, S_units, base_year)
REACTOR_COSTS = {
    'cstr_jacketed': CostParams(17400, 79.0, 0.85, 0.1, 100.0, 'm³', 2019),
    'cstr_coil': CostParams(14000, 68.0, 0.85, 0.1, 100.0, 'm³', 2019),
    'pfr_tube': CostParams(3000, 1200.0, 0.65, 0.01, 10.0, 'm³', 2019),
    'batch_reactor': CostParams(25000, 95.0, 0.85, 0.5, 50.0, 'm³', 2019),
}
```

**Size Parameter**: Reactor volume (m³)

**Example**:
```python
from difflow.economics.capital import reactor_cost

cost = reactor_cost(volume=10.0, reactor_type='cstr_jacketed')
# Returns ~$24,000 for 10 m³ jacketed CSTR (escalated to the default target year)
```

#### Pressure Vessels

```python
from difflow.economics.capital import CostParams, vessel_cost

VESSEL_COSTS = {
    'pressure_vessel_vertical': CostParams(8000, 380.0, 0.72, 0.1, 200.0, 'm³', 2019),
    'pressure_vessel_horizontal': CostParams(7000, 350.0, 0.72, 0.1, 200.0, 'm³', 2019),
    'storage_tank_atmospheric': CostParams(5000, 180.0, 0.65, 1.0, 10000.0, 'm³', 2019),
    'flash_drum': CostParams(6500, 320.0, 0.70, 0.1, 50.0, 'm³', 2019),
}
```

**Pressure Adjustment**:

$$F_P = \frac{(P - 1)(D + 2t)}{2SE - 1.2(P-1)} + 1$$

Where:
- $P$: Design pressure (barg)
- $D$: Diameter
- $t$: Wall thickness
- $S$: Allowable stress
- $E$: Weld efficiency

```python
from difflow.economics.capital import pressure_factor_vessel

F_p = pressure_factor_vessel(pressure=20.0, diameter=2.0)  # ~1.15 for 20 barg
base_cost = vessel_cost(volume=5.0, vessel_type='pressure_vessel_vertical')
cost_adj = base_cost * F_p
```

#### Heat Exchangers

```python
HEAT_EXCHANGER_COSTS = {
    'shell_tube_floating': CostParams(11000, 340.0, 0.60, 10.0, 1000.0, 'm²', 2019),
    'shell_tube_fixed': CostParams(8500, 280.0, 0.60, 10.0, 1000.0, 'm²', 2019),
    'shell_tube_utube': CostParams(9000, 300.0, 0.60, 10.0, 1000.0, 'm²', 2019),
    'double_pipe': CostParams(1500, 120.0, 0.65, 1.0, 50.0, 'm²', 2019),
    'plate_frame': CostParams(3000, 80.0, 0.70, 5.0, 500.0, 'm²', 2019),
    'air_cooler': CostParams(15000, 180.0, 0.65, 20.0, 2000.0, 'm²', 2019),
}
```

**Size Parameter**: Heat transfer area (m²)

```python
from difflow.economics.capital import heat_exchanger_cost

cost = heat_exchanger_cost(area=50.0, hx_type='shell_tube_floating')
```

#### Distillation Columns

```python
COLUMN_COSTS = {
    'tray_column_shell': CostParams(15000, 68.0, 0.85, 0.5, 100.0, 'm³', 2019),
    'packed_column_shell': CostParams(12000, 58.0, 0.85, 0.5, 100.0, 'm³', 2019),
    'sieve_tray': CostParams(200, 450.0, 0.60, 0.5, 5.0, 'm² (per tray)', 2019),
    'valve_tray': CostParams(300, 500.0, 0.60, 0.5, 5.0, 'm² (per tray)', 2019),
    'packing_random': CostParams(0, 800.0, 1.0, 0.1, 100.0, 'm³', 2019),
    'packing_structured': CostParams(0, 3000.0, 1.0, 0.1, 100.0, 'm³', 2019),
}
```

**Size Parameter**: Column shell volume (m³). Trays (sieve/valve, sized by
tray area in m²) and packing (sized by packed volume in m³) are costed
separately and added to the shell cost.

```python
from difflow.economics.capital import column_cost

import math

# Column shell + internals: 2 m diameter x 20 m tall, 30 sieve trays
D, H, n_trays = 2.0, 20.0, 30
shell_volume = math.pi * D**2 / 4 * H
tray_area = math.pi * D**2 / 4
cost = (column_cost(volume=shell_volume, column_type='tray_column_shell')
        + n_trays * column_cost(volume=tray_area, column_type='sieve_tray'))
```

#### Pumps and Compressors

```python
PUMP_COSTS = {
    'centrifugal_single': CostParams(3500, 320.0, 0.55, 0.5, 100.0, 'kW', 2019),
    'centrifugal_multistage': CostParams(6000, 410.0, 0.55, 1.0, 500.0, 'kW', 2019),
    'reciprocating': CostParams(8000, 680.0, 0.50, 1.0, 200.0, 'kW', 2019),
    'gear': CostParams(2500, 250.0, 0.60, 0.1, 50.0, 'kW', 2019),
}

COMPRESSOR_COSTS = {
    'centrifugal': CostParams(50000, 1800.0, 0.65, 100.0, 10000.0, 'kW', 2019),
    'reciprocating': CostParams(25000, 2200.0, 0.60, 10.0, 1000.0, 'kW', 2019),
    'screw': CostParams(15000, 1400.0, 0.65, 50.0, 3000.0, 'kW', 2019),
}
```

**Size Parameter**: Power (kW)

```python
from difflow.economics.capital import pump_cost, compressor_cost

pump = pump_cost(power=10.0, pump_type='centrifugal_single')
comp = compressor_cost(power=500.0, compressor_type='centrifugal')
```

### Installation Factors

Equipment purchase cost must be multiplied by installation factors to get installed cost.
`InstallationFactors` is a dataclass whose fields are fractions of the
purchased cost; the installed cost is `purchased * (1 + sum of fractions)`
(`total_factor`).

```python
from difflow.economics.capital import InstallationFactors

# Defaults shown; every field can be overridden.
InstallationFactors(
    piping=0.35,
    instrumentation=0.20,
    electrical=0.12,
    buildings=0.15,
    yard_improvements=0.05,
    service_facilities=0.15,
    engineering=0.10,
    construction=0.10,
    contingency=0.15,
)
```

**Typical Installation Factors by Equipment Type**:

| Equipment | Bare Module Factor |
|-----------|-------------------|
| Heat exchangers | 3.0-3.5 |
| Pumps | 3.0-4.0 |
| Vessels | 3.5-4.5 |
| Columns | 4.0-5.0 |
| Reactors | 3.5-4.5 |
| Compressors | 2.5-3.5 |

```python
from difflow.economics.capital import installed_cost, installed_cost_detailed

# Quick calculation with typical factor
installed = installed_cost(purchased_cost=100000, lang_factor=3.5)  # $350,000

# Detailed breakdown
detailed = installed_cost_detailed(
    purchased_cost=100000,
    factors=InstallationFactors(
        piping=0.45,
        instrumentation=0.20,
        electrical=0.15,
        buildings=0.10,
        yard_improvements=0.05,
        service_facilities=0.10,
    )
)
```

### Total Capital Investment

Total Capital Investment (TCI) includes all costs to build a functioning plant.

```python
from difflow.economics.capital import (
    total_capital_investment,
    total_capital_investment_detailed,
    CapitalInvestment
)
```

**Capital Investment Breakdown**:

```
Direct Costs (DC):
├── Equipment Purchase (ISBL)
├── Installation
├── Piping
├── Instrumentation
├── Electrical
├── Buildings
├── Site Development
└── Auxiliary Facilities

Indirect Costs (IC):
├── Engineering & Supervision (10-15% DC)
├── Construction & Contractor's Fee (5-15% DC)
└── Contingency (10-20% DC)

Fixed Capital Investment (FCI) = DC + IC

Working Capital (WC) = 10-20% FCI

Total Capital Investment (TCI) = FCI + WC
```

```python
# Quick estimate from equipment costs
equipment_costs = {
    'reactor': 500000,
    'column': 800000,
    'heat_exchangers': 300000,
    'pumps': 100000
}

tci = total_capital_investment(sum(equipment_costs.values()))
# Lang factor approach: FCI = 4.74 x equipment cost, TCI = FCI + 15% working capital

# Detailed breakdown
result = total_capital_investment_detailed(
    equipment_costs=equipment_costs,
    factors=InstallationFactors(),   # piping, instrumentation, ... as fractions
    working_capital_fraction=0.15
)

print(f"Direct costs: ${result.total_direct_costs:,.0f}")
print(f"Indirect costs: ${result.total_indirect_costs:,.0f}")
print(f"Fixed capital: ${result.fixed_capital_investment:,.0f}")
print(f"Working capital: ${result.working_capital:,.0f}")
print(f"Total capital: ${result.total_capital_investment:,.0f}")
```

---

## Utility Costs

**Location**: `difflow/economics/utilities.py`

### Utility Prices

```python
from difflow.economics.utilities import UtilityPrices, DEFAULT_PRICES

# UtilityPrices is a dataclass; heat utilities are priced in $/GJ.
UtilityPrices(
    steam_high_pressure=14.05,    # $/GJ, 4.1 MPa
    steam_medium_pressure=11.80,  # $/GJ, 1.1 MPa
    steam_low_pressure=9.50,      # $/GJ, 0.34 MPa
    cooling_water=0.35,           # $/GJ
    chilled_water=4.50,           # $/GJ
    refrigeration_moderate=8.00,  # $/GJ, -20 C
    refrigeration_low=13.50,      # $/GJ, -50 C
    cryogenic=35.00,              # $/GJ, below -100 C
    electricity=0.07,             # $/kWh
    natural_gas=4.50,             # $/GJ
    fuel_oil=6.00,                # $/GJ
    coal=2.50,                    # $/GJ
    process_water=0.50,           # $/m³
    boiler_feed_water=2.50,       # $/m³
    wastewater_treatment=0.80,    # $/m³
    compressed_air=0.03,          # $/Nm³
    nitrogen=0.08,                # $/Nm³
    oxygen=0.12,                  # $/Nm³
)

# Regional presets: 'us_gulf_coast', 'us_midwest', 'europe_west',
# 'asia_pacific', 'china'
prices_eu = UtilityPrices.from_region('europe_west')
```

`DEFAULT_PRICES` is the US Gulf Coast instance. All cost functions return
rates in $/s (multiply by `3600 * hours_per_year` for an annual figure).

### Steam

Steam is the primary heating utility in chemical processes.

**Steam Properties**:

| Type | Pressure | Temperature | Latent Heat |
|------|----------|-------------|-------------|
| LP | 150 psig (10 barg) | 185°C | ~2200 kJ/kg |
| MP | 400 psig (28 barg) | 250°C | ~1800 kJ/kg |
| HP | 600 psig (41 barg) | 280°C | ~1500 kJ/kg |

```python
from difflow.economics.utilities import (
    steam_cost_from_duty,
    steam_flowrate_from_duty
)

# Calculate steam cost from heat duty
Q = 1e6  # 1 MW heating
cost_per_s = steam_cost_from_duty(Q, steam_level='low_pressure')   # $/s
annual_cost = cost_per_s * 8000 * 3600                             # $/year

# Calculate steam flowrate
steam_rate = steam_flowrate_from_duty(Q, steam_level='low_pressure')  # kg/s
```

**Equations**:

$$\dot{m}_{steam} = \frac{Q}{\Delta H_{vap}}$$

$$C_{steam} = Q \cdot P_{steam} \cdot t_{operation}$$

with the steam price $P_{steam}$ in $/GJ and the duty $Q$ in GJ/s.

### Cooling Water

```python
from difflow.economics.utilities import (
    cooling_water_cost,
    cooling_water_flowrate
)

# Cooling water for 500 kW duty with 10°C rise
Q = 500000  # W
cost_per_s = cooling_water_cost(Q, delta_T=10.0)       # $/s
annual_cost = cost_per_s * 8000 * 3600                 # $/year

# Required flowrate
cw_rate = cooling_water_flowrate(Q, delta_T=10.0)      # kg/s
```

**Equations**:

$$\dot{m}_{cw} = \frac{Q}{C_p \cdot \Delta T}$$

Assuming $C_p = 4.18$ kJ/kg/K for water.

### Electricity

```python
from difflow.economics.utilities import (
    electricity_cost,
    electricity_cost_per_second,
    pump_electricity_cost,
    compressor_electricity_cost
)

# Cost rate from power in kW ($/h)
cost_per_hour = electricity_cost(power=1000.0)  # $70/h at $0.07/kWh

# Cost rate from continuous power in W ($/s)
cost_rate = electricity_cost_per_second(power=100000)

# Pump electricity (includes efficiency), $/s
pump_cost_rate = pump_electricity_cost(
    flowrate=0.01,       # m³/s
    head=50.0,           # m
    efficiency=0.75,
)

# Compressor electricity (isentropic), $/s
comp_cost_rate = compressor_electricity_cost(
    flowrate=10.0,       # mol/s
    pressure_ratio=3.0,
    efficiency=0.80,
)
```

**Pump Power**:

$$P_{pump} = \frac{\rho g Q H}{\eta}$$

**Compressor Power** (isentropic):

$$P_{comp} = \frac{n}{n-1} P_1 Q_1 \left[\left(\frac{P_2}{P_1}\right)^{(n-1)/n} - 1\right] / \eta$$

### Refrigeration

For sub-ambient cooling:

```python
from difflow.economics.utilities import (
    refrigeration_cost,
    refrigeration_cost_continuous
)

# Cost depends on temperature level ($/s)
cost = refrigeration_cost(
    100000,                    # 100 kW cooling
    temperature_level=-20.0,   # °C
)

# Smooth, differentiable version of the temperature dependence
cost_c = refrigeration_cost_continuous(100000, temperature=-20.0)
```

**Refrigeration Cost Factor**:

Lower temperatures require more compression work, increasing cost. The
price per GJ of duty steps up with the temperature level:

| Temperature level | Price field | Default ($/GJ) |
|-------------------|-------------|----------------|
| 5°C and above | `chilled_water` | 4.50 |
| -20°C to 5°C | `refrigeration_moderate` | 8.00 |
| -50°C to -20°C | `refrigeration_low` | 13.50 |
| below -50°C | `cryogenic` | 35.00 |

### Combined Utility Costs

```python
from difflow.economics.utilities import (
    utility_cost_from_heat_duties,
    total_utility_cost,
    UtilityConsumption,
)

# Total utility cost from the summed heating and cooling duties
heating = 500000 + 300000       # W, sum of the heaters
cooling = 400000 + 600000       # W, sum of the coolers (positive)

cost_per_s = utility_cost_from_heat_duties(
    heating_duty=heating,
    cooling_duty=cooling,
    steam_level='medium_pressure',
    prices=DEFAULT_PRICES,
)
print(f"Utilities: ${float(cost_per_s) * 8000 * 3600:,.0f}/year")

# Or describe all consumption in one container
consumption = UtilityConsumption(
    heating_duty=heating, cooling_duty=cooling, electricity=200.0,  # kW
)
total = total_utility_cost(consumption, DEFAULT_PRICES)   # $/s
print(f"Total utilities: ${total * 8000 * 3600:,.0f}/year")
```

---

## Operating Costs

**Location**: `difflow/economics/opex.py`

### Raw Materials

```python
from difflow.economics.opex import (
    RawMaterial,
    raw_material_cost,
    total_raw_material_cost,
    annual_raw_material_cost,
)

# RawMaterial is a dataclass:
#   RawMaterial(name, price [$/kg], consumption_rate [kg/s], molecular_weight [g/mol])
methanol = RawMaterial('methanol', price=0.40, consumption_rate=0.3, molecular_weight=32.04)
```

```python
# Define raw materials as name -> (flowrate [kg/s], price [$/kg])
materials = {
    'methanol': (0.30, 0.40),
    'oxygen': (0.15, 0.05),
    'catalyst': (3e-5, 50.0),
}

rm_rate = total_raw_material_cost(materials)          # $/s
total_rm_cost = annual_raw_material_cost(materials)   # $/year (default schedule)
print(f"Annual raw material cost: ${total_rm_cost:,.0f}")
```

**From Molar Quantities**:

```python
from difflow.economics.opex import raw_material_cost_molar

# Cost from molar flow rates
cost_per_s = raw_material_cost_molar(
    molar_flowrate=100.0,    # mol/s
    price=0.40,              # $/kg
    molecular_weight=32.04,  # g/mol
)
annual = cost_per_s * 8000 * 3600   # $/year
```

### Labor

```python
from difflow.economics.opex import (
    LaborRates,
    DEFAULT_LABOR_RATES,
    operating_labor_cost,
    labor_cost_from_equipment
)

# LaborRates is a dataclass (defaults shown)
LaborRates(
    operator_salary=65000.0,     # $/year per operator
    supervisor_salary=90000.0,   # $/year per supervisor
    overhead_factor=1.4,         # benefits, taxes, etc.
    supervisor_ratio=0.2,        # supervisors per operator
)
```

**Labor Estimation Methods**:

1. **Direct calculation**:
```python
labor_cost = operating_labor_cost(
    n_operators_per_shift=6,
    n_shifts=3,              # 3 shifts/day; uses a 4.5x coverage factor for 24/7
    rates=DEFAULT_LABOR_RATES,
)
```

2. **From equipment count** (correlation):
```python
# Rough estimate: 1 operator per 2-4 major equipment items
labor_cost = labor_cost_from_equipment(
    n_major_equipment=20,
    process_complexity='medium',  # 'simple', 'medium' or 'complex'
    rates=DEFAULT_LABOR_RATES,
)
```

### Overhead and Maintenance

```python
from difflow.economics.opex import (
    OverheadFactors,
    maintenance_cost,
    insurance_taxes_cost,
    plant_overhead_cost
)

# OverheadFactors is a dataclass of fractions (defaults shown):
#   maintenance=0.04, property_taxes=0.02, insurance=0.01     (of FCI)
#   supervision=0.25, laboratory=0.10, plant_overhead=0.60    (of operating labor)
#   general_admin=0.05, distribution_selling=0.05, research_dev=0.03
#                                                            (of manufacturing cost)
OverheadFactors()
```

**Typical Factors**:

| Item | Basis | Default |
|------|-------|---------|
| Maintenance | FCI | 4% |
| Insurance | FCI | 1% |
| Property taxes | FCI | 2% |
| Supervision + laboratory + plant overhead | operating labor | 95% |

```python
FCI = 50e6  # $50M fixed capital

maintenance = maintenance_cost(FCI, OverheadFactors(maintenance=0.03))  # 3% of FCI = $1.5M
taxes_ins = insurance_taxes_cost(FCI)   # (2% property tax + 1% insurance) of FCI
overhead = plant_overhead_cost(500000)  # supervision + lab + overhead on $500k labor
```

### Total Operating Cost

```python
from difflow.economics.opex import (
    calculate_opex,
    simple_opex,
    OperatingCostBreakdown,
    com_from_correlations
)

# Detailed OPEX calculation; maintenance, taxes/insurance and overhead
# are derived from the fixed capital and labor cost
opex = calculate_opex(
    raw_material_cost=4e6,   # $/year
    utility_cost=2e6,        # $/year
    fixed_capital=FCI,       # $
    operating_labor=1e6,     # $/year (omit to estimate from operators/shift)
)

print(f"Total manufacturing cost: ${opex.manufacturing_cost:,.0f}/year")
print(f"Total production cost: ${opex.total_production_cost:,.0f}/year")

# Quick estimate: variable costs + fractions of FCI
simple = simple_opex(raw_materials=4e6, utilities=2e6, fixed_capital=FCI)
print(f"Estimated OPEX: ${simple:,.0f}/year")
```

**Cost of Manufacturing (COM)** correlation:

$$COM = 0.18 \cdot FCI + 2.73 \cdot C_{OL} + 1.23 \cdot (C_{UT} + C_{RM})$$

Where:
- $FCI$: Fixed capital investment
- $C_{OL}$: Operating labor cost
- $C_{UT}$: Utility cost
- $C_{RM}$: Raw material cost

```python
com = com_from_correlations(
    fixed_capital=50e6,
    utility_cost=2e6,
    raw_material_cost=4e6,
    n_operators_per_shift=4,   # sets the operating labor term
)
```

### Operating Schedule

```python
from difflow.economics.opex import (
    OperatingSchedule, DEFAULT_SCHEDULE, HOURS_PER_YEAR, SECONDS_PER_YEAR
)

# OperatingSchedule is a dataclass (defaults shown)
OperatingSchedule(
    hours_per_year=8000.0,   # ~91% availability
    shifts_per_day=3,
    days_per_week=7.0,
    weeks_per_year=50.0,
    stream_factor=0.91,
)

# Constants: HOURS_PER_YEAR = 8000, SECONDS_PER_YEAR = 28_800_000
```

---

## Profitability Analysis

**Location**: `difflow/economics/profitability.py`

### Financial Parameters

```python
from difflow.economics.profitability import FinancialParams

# FinancialParams is a dataclass (defaults shown)
FinancialParams(
    discount_rate=0.10,             # 10% WACC
    tax_rate=0.21,                  # 21% corporate tax
    depreciation_years=10,          # MACRS 10-year
    plant_life=20,                  # 20-year project life
    construction_years=2,           # 2-year construction
    salvage_fraction=0.05,          # 5% salvage value
    working_capital_fraction=0.15,  # 15% of FCI
    inflation_rate=0.02,            # 2% annual inflation
)
```

### Time Value of Money

```python
from difflow.economics.profitability import (
    present_value,
    future_value,
    discount_factor,
    capital_recovery_factor,
    present_value_factor
)

# Present value of future cash
PV = present_value(future_value=1e6, rate=0.10, years=10)  # $385,543

# Future value of present cash
FV = future_value(present_value=1e6, rate=0.10, years=10)  # $2,593,742

# Discount factor
df = discount_factor(rate=0.10, year=10)  # 0.3855

# Capital recovery factor (annuity payment per $ of principal)
crf = capital_recovery_factor(rate=0.10, years=20)  # 0.1175

# Present value factor (sum of discount factors)
pvf = present_value_factor(rate=0.10, years=20)  # 8.514
```

**Equations**:

$$PV = \frac{FV}{(1+r)^n}$$

$$FV = PV \cdot (1+r)^n$$

$$CRF = \frac{r(1+r)^n}{(1+r)^n - 1}$$

$$PVF = \frac{1 - (1+r)^{-n}}{r}$$

### Depreciation

```python
from difflow.economics.profitability import MACRS_SCHEDULES

# Modified Accelerated Cost Recovery System (US tax code)
# (jnp arrays keyed by recovery period: 5, 7, 10 and 15 years)
print(MACRS_SCHEDULES[5])   # [0.2, 0.32, 0.192, 0.1152, 0.1152, 0.0576]
```

**Annual Depreciation**:

$$D_t = FCI \cdot MACRS_t$$

**Tax Savings from Depreciation**:

$$\text{Tax Savings} = D_t \cdot \tau$$

Where $\tau$ is the tax rate.

### Net Present Value (NPV)

```python
from difflow.economics.profitability import (
    npv,
    npv_with_construction
)

import jax.numpy as jnp

# Simple NPV: cash flows for years 1..N, investment at year 0
cash_flows = jnp.array([10e6, 12e6, 14e6, 14e6, 14e6])
npv_value = npv(cash_flows, discount_rate=0.10, initial_investment=50e6)

# NPV with construction period
npv_value = npv_with_construction(
    operating_cash_flows=jnp.full(20, 10e6),   # 20 operating years
    capital_investment=50e6,
    discount_rate=0.10,
    construction_years=2,
)
```

**Equation**:

$$NPV = \sum_{t=0}^{n} \frac{CF_t}{(1+r)^t}$$

**Decision Criteria**:
- NPV > 0: Accept project
- NPV < 0: Reject project
- NPV = 0: Indifferent (earns exactly the required return)

### Internal Rate of Return (IRR)

```python
from difflow.economics.profitability import irr, irr_approx

# IRR calculation (Newton solve via optimistix)
cash_flows = jnp.array([10e6, 12e6, 14e6, 14e6, 14e6])
irr_value = irr(cash_flows, initial_investment=50e6)

# Quick approximation (linear interpolation between two trial rates)
irr_approx_value = irr_approx(jnp.full(20, 10e6), initial_investment=50e6)
```

**Definition**: IRR is the discount rate where NPV = 0:

$$0 = \sum_{t=0}^{n} \frac{CF_t}{(1+IRR)^t}$$

**Decision Criteria**:
- IRR > WACC: Accept project
- IRR < WACC: Reject project

**Typical IRR Targets**:

| Risk Level | Target IRR |
|------------|------------|
| Low (expansion) | 15-20% |
| Medium (new product) | 20-30% |
| High (new technology) | 30-50% |

### Payback Period

```python
from difflow.economics.profitability import (
    simple_payback,
    discounted_payback
)

# Simple payback (no discounting)
payback = simple_payback(annual_cash_flow=10e6, initial_investment=50e6)  # 5 years

# Discounted payback (accounts for time value)
d_payback = discounted_payback(
    cash_flows=jnp.full(20, 10e6),
    initial_investment=50e6,
    discount_rate=0.10,
)  # ~7.3 years
```

**Equations**:

$$\text{Simple Payback} = \frac{I_0}{CF_{annual}}$$

$$\text{Discounted Payback}: \text{Find } n \text{ where } \sum_{t=1}^{n} \frac{CF_t}{(1+r)^t} = I_0$$

### Return on Investment (ROI)

```python
from difflow.economics.profitability import roi, average_roi

# Simple ROI
roi_value = roi(annual_profit=8e6, total_investment=50e6)  # 16%

# Average ROI over project life
avg_roi = average_roi(
    total_profit=160e6,     # Sum over 20 years
    total_investment=50e6,
    years=20,
)
```

**Equation**:

$$ROI = \frac{\text{Annual Net Profit}}{\text{Total Investment}} \times 100\%$$

### Minimum Selling Price (MSP)

MSP is the product price required to achieve a target financial return.

```python
from difflow.economics.profitability import (
    minimum_selling_price,
    msp_with_target_roi,
    msp_with_npv_zero
)

# Simple MSP (break-even)
msp = minimum_selling_price(
    total_annual_cost=40e6,  # $/year
    annual_production=1e7     # kg/year
)  # $4.00/kg

# MSP for target ROI
msp_roi = msp_with_target_roi(
    total_annual_cost=40e6,
    annual_production=1e7,
    total_investment=50e6,
    target_roi=0.20,
)  # $5.00/kg

# MSP for NPV = 0 (most rigorous)
msp_npv = msp_with_npv_zero(
    opex=35e6,
    capex=50e6,
    annual_production=1e7,
    discount_rate=0.10,
    plant_life=20,
)
```

**Equation** (NPV = 0):

$$MSP = \frac{OPEX + \left(CAPEX + WC - PV(WC)\right) / PVF}{Q}$$

where $WC$ is the working capital (a fraction of CAPEX), recovered at the end
of the plant life, and $PVF$ is the present value factor of the annuity. This
is a pre-tax breakeven price.

### Cash Flow Analysis

```python
from difflow.economics.profitability import (
    FinancialParams,
    generate_cash_flows,
    full_cash_flow_analysis,
    CashFlowResult
)

# Generate year-by-year cash flows
params = FinancialParams(
    discount_rate=0.10, tax_rate=0.21, depreciation_years=10, plant_life=20,
)
cash_flows = generate_cash_flows(
    capital_investment=50e6,
    annual_revenue=45e6,
    annual_opex=30e6,
    params=params,
)

# Full analysis (adds working capital recovery and salvage value)
result = full_cash_flow_analysis(
    capital_investment=50e6,
    annual_revenue=45e6,
    annual_opex=30e6,
    params=params,
)

print(f"NPV: ${result.npv:,.0f}")
print(f"IRR: {result.irr:.1%}")
print(f"Payback: {result.payback:.1f} years")
```

**Cash Flow Components**:

```
Revenue
- Operating Costs
- Depreciation
= Taxable Income
- Taxes (21%)
= Net Income
+ Depreciation (add back)
= Operating Cash Flow

Year 0: -Capital Investment
Years 1-n: Operating Cash Flow
Year n: + Working Capital + Salvage Value
```

### Sensitivity Analysis

```python
from difflow.economics.profitability import npv_sensitivity

# NPV sensitivity to parameter changes
sensitivities = npv_sensitivity(
    base_case={'capital': 50e6, 'revenue': 45e6, 'opex': 30e6, 'discount_rate': 0.10},
    parameter_ranges={
        'capital': (40e6, 60e6),     # +/- 20%
        'revenue': (38e6, 52e6),     # +/- 15%
        'opex': (27e6, 33e6),        # +/- 10%
    },
    n_points=10,
)

# Identify most critical parameters
for param, (values, npvs) in sensitivities.items():
    print(f"{param}: NPV ranges from ${npvs.min()/1e6:.1f}M to ${npvs.max()/1e6:.1f}M")
```

---

## Cost Indices

**Location**: `difflow/economics/indices.py`

### CEPCI (Chemical Engineering Plant Cost Index)

The CEPCI allows escalation of historical equipment costs to current year.

```python
from difflow.economics.indices import (
    get_cepci,
    escalate_cost,
    CEPCI_HISTORICAL
)

# CEPCI_HISTORICAL maps year -> CEPCIData(year, index, ...), 2000-2026
# (2024-2026 are estimates). Selected overall index values:
#   2000: 394.1, 2010: 550.8, 2019: 607.5, 2020: 596.2,
#   2021: 708.0, 2022: 816.0, 2023: 797.9

# Get CEPCI for specific year
cepci_2019 = get_cepci(2019)  # 607.5
cepci_2023 = get_cepci(2023)  # 797.9
```

**Cost Escalation**:

$$C_{new} = C_{old} \times \frac{CEPCI_{new}}{CEPCI_{old}}$$

```python
# Escalate 2019 cost to 2023
cost_2019 = 1e6
cost_2023 = escalate_cost(cost_2019, base_year=2019, target_year=2023)
# $1,000,000 × (797.9/607.5) = $1,313,415
```

### Equipment-Specific Escalation

Different equipment types may have different cost trends:

```python
from difflow.economics.indices import CEPCI_RATIOS

# Pre-computed ratios for common escalations
ratio_2019_to_2024 = CEPCI_RATIOS[(2019, 2024)]

# Apply to equipment cost
old_cost = 1e6
current_cost = old_cost * ratio_2019_to_2024
```

### Inflation Factors

For general inflation (not CEPCI):

```python
from difflow.economics.indices import (
    inflation_factor,
    inflation_factor_continuous
)

# Discrete compounding
factor = inflation_factor(base_year=2020, target_year=2025, annual_rate=0.02)  # 1.104

# Differentiable (array-valued) form
factor_cont = inflation_factor_continuous(years=5.0, annual_rate=0.02)  # 1.104
```

---

## Examples

### Complete TEA for a Methanol Plant

```python
import jax.numpy as jnp
from difflow.economics.capital import (
    reactor_cost, heat_exchanger_cost, column_cost, compressor_cost,
    total_capital_investment
)
from difflow.economics.utilities import utility_cost_from_heat_duties
from difflow.economics.opex import calculate_opex, raw_material_cost_molar
from difflow.economics.profitability import (
    FinancialParams, full_cash_flow_analysis, msp_with_npv_zero
)

# Plant specifications
production = 100000  # tonnes/year methanol
hours_per_year = 8000

# Equipment costs
equipment = {
    'reactor': reactor_cost(50.0, 'cstr_jacketed'),
    'distillation': column_cost(3.14 * 1.5**2 * 30.0, 'tray_column_shell'),  # 3 m dia x 30 m
    'heat_exchangers': 3 * heat_exchanger_cost(200.0, 'shell_tube_floating'),
    'compressor': compressor_cost(2000.0, 'centrifugal'),
}
total_equipment = sum(equipment.values())

# Capital investment
TCI = total_capital_investment(total_equipment)
FCI = TCI / 1.15  # fixed capital: TCI less the 15% working capital

# Utilities (heating and cooling duties summed, W)
heating = 8e6 + 2e6       # reboiler + feed heater
cooling = 5e6 + 6e6       # reactor cooling + condenser
utility_rate = utility_cost_from_heat_duties(heating, cooling)   # $/s
utilities = float(utility_rate) * hours_per_year * 3600          # $/year

# Raw materials
syngas_rate = raw_material_cost_molar(
    molar_flowrate=production * 1e6 / 32.04 / (hours_per_year * 3600) * 3,  # mol/s, 3 mol syngas per mol MeOH
    price=0.15,              # $/kg
    molecular_weight=10.0,   # approximate syngas MW, g/mol
)
syngas_cost = float(syngas_rate) * hours_per_year * 3600         # $/year

# Total OPEX (maintenance, taxes, insurance and overhead follow from FCI and labor)
labor = 1.5e6  # $1.5M/year
opex = calculate_opex(
    raw_material_cost=syngas_cost,
    utility_cost=utilities,
    fixed_capital=float(FCI),
    operating_labor=labor,
)

# Revenue (at $400/tonne methanol)
revenue = production * 400

# Profitability analysis
result = full_cash_flow_analysis(
    capital_investment=float(FCI),
    annual_revenue=revenue,
    annual_opex=opex.total_production_cost,
    params=FinancialParams(discount_rate=0.10, tax_rate=0.21, plant_life=20),
)

print(f"\n=== Methanol Plant TEA ===")
print(f"Production: {production:,} tonnes/year")
print(f"Total Capital Investment: ${TCI/1e6:.1f} M")
print(f"Fixed Capital: ${FCI/1e6:.1f} M")
print(f"Annual OPEX: ${opex.total_production_cost/1e6:.1f} M")
print(f"Annual Revenue: ${revenue/1e6:.1f} M")
print(f"\nProfitability Metrics:")
print(f"  NPV (10%): ${result.npv/1e6:.1f} M")
print(f"  IRR: {result.irr:.1%}")
print(f"  Payback: {result.payback:.1f} years")

# Minimum selling price
msp = msp_with_npv_zero(
    opex=opex.total_production_cost,
    capex=FCI,
    annual_production=production * 1000,  # kg/year
    discount_rate=0.10,
    plant_life=20,
)
print(f"  MSP: ${msp:.2f}/kg = ${msp*1000:.0f}/tonne")
```

### Optimization with Gradient-Based Methods

All TEA functions are JAX-differentiable, enabling gradient-based optimization:

```python
import jax
import jax.numpy as jnp
from difflow.economics.capital import reactor_cost, column_cost, total_capital_investment
from difflow.economics.profitability import npv_objective

def plant_npv(design_params):
    """NPV as function of design parameters."""
    reactor_volume, column_volume = design_params

    # Equipment costs depend on design
    equip_cost = reactor_cost(reactor_volume, 'cstr_jacketed')
    equip_cost += column_cost(column_volume, 'tray_column_shell')

    FCI = total_capital_investment(equip_cost) / 1.15

    # Operating costs depend on design (simplified)
    OPEX = 0.15 * FCI  # Rough correlation

    # Revenue depends on conversion (which depends on reactor size)
    conversion = 1 - jnp.exp(-0.1 * reactor_volume)  # Simplified kinetics
    revenue = conversion * 1e8  # Base revenue

    # npv_objective returns the NEGATIVE NPV (for minimization)
    return -npv_objective(revenue, OPEX, FCI, discount_rate=0.10, plant_life=20)

# Gradient of NPV w.r.t. design parameters
grad_npv = jax.grad(plant_npv)

# Optimization loop
design = jnp.array([10.0, 20.0])  # Initial guess
step = 0.5                        # step length in design units

for i in range(100):
    grad = grad_npv(design)
    # Normalized gradient ascent (maximize NPV); NPV is in $, so scale the step
    design = design + step * grad / (jnp.linalg.norm(grad) + 1e-12)
    design = jnp.maximum(design, 0.5)   # stay inside the correlation range

    if i % 20 == 0:
        print(f"Iteration {i}: NPV = ${plant_npv(design)/1e6:.2f}M")
```

---

## Summary Tables

### Typical Equipment Costs (2019 basis)

| Equipment | Size | Approximate Cost |
|-----------|------|------------------|
| CSTR (jacketed) | 10 m³ | $90,000 |
| PFR | 5 m³ | $55,000 |
| Shell-tube HX | 50 m² | $60,000 |
| Distillation column | 2 m dia × 20 m | $250,000 |
| Centrifugal pump | 10 kW | $15,000 |
| Centrifugal compressor | 500 kW | $400,000 |

### Utility Cost Summary (2019 basis)

| Utility | Unit Cost | Typical Usage |
|---------|-----------|---------------|
| LP Steam | $0.015/kg | Heating to 185°C |
| HP Steam | $0.030/kg | Heating to 280°C |
| Cooling Water | $0.05/m³ | Cooling above 30°C |
| Electricity | $0.07/kWh | Pumps, compressors |
| Refrigeration (-20°C) | 2× CW cost | Sub-ambient cooling |

### Financial Parameters Summary

| Parameter | Typical Value |
|-----------|---------------|
| Discount rate (WACC) | 8-12% |
| Corporate tax rate | 21% (US) |
| MACRS depreciation | 7-10 years |
| Plant life | 15-25 years |
| Construction period | 2-3 years |
| Working capital | 10-20% of FCI |
| Contingency | 10-20% of DC |
