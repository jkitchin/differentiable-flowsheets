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
- CEPCI cost escalation from 1957-2023
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
from difflow.economics.capital import REACTOR_COSTS

REACTOR_COSTS = {
    'cstr_jacketed': CostParams(0, 28000, 0.55, 0.1, 100, 'm³', 2019),
    'cstr_coil': CostParams(0, 32000, 0.55, 0.1, 100, 'm³', 2019),
    'pfr_tube': CostParams(0, 15000, 0.65, 0.01, 50, 'm³', 2019),
    'batch_reactor': CostParams(0, 35000, 0.55, 0.5, 200, 'm³', 2019),
}
```

**Size Parameter**: Reactor volume (m³)

**Example**:
```python
from difflow.economics.capital import reactor_cost

cost = reactor_cost(volume=10.0, reactor_type='cstr_jacketed')
# Returns ~$92,000 for 10 m³ jacketed CSTR
```

#### Pressure Vessels

```python
VESSEL_COSTS = {
    'vertical': CostParams(0, 10000, 0.62, 0.5, 200, 'm³', 2019),
    'horizontal': CostParams(0, 8000, 0.65, 0.5, 200, 'm³', 2019),
    'storage_tank': CostParams(0, 5000, 0.55, 1, 5000, 'm³', 2019),
    'flash_drum': CostParams(0, 12000, 0.60, 0.1, 50, 'm³', 2019),
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

F_p = pressure_factor_vessel(P_design=20.0, diameter=2.0)  # ~1.3 for 20 bar
cost_adj = base_cost * F_p
```

#### Heat Exchangers

```python
HEAT_EXCHANGER_COSTS = {
    'shell_tube_floating': CostParams(0, 20000, 0.65, 5, 500, 'm²', 2019),
    'shell_tube_fixed': CostParams(0, 15000, 0.65, 5, 500, 'm²', 2019),
    'shell_tube_utube': CostParams(0, 18000, 0.65, 5, 500, 'm²', 2019),
    'double_pipe': CostParams(0, 3000, 0.75, 1, 50, 'm²', 2019),
    'plate_frame': CostParams(0, 25000, 0.60, 5, 200, 'm²', 2019),
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
    'tray_column': CostParams(0, 25000, 0.70, 0.5, 5.0, 'm_diameter', 2019),
    'packed_column': CostParams(0, 30000, 0.68, 0.3, 4.0, 'm_diameter', 2019),
}
```

**Size Parameter**: Column diameter (m)

Additional costs:
- Trays: ~$500-1500 per tray
- Packing: ~$5000-15000 per m³

```python
from difflow.economics.capital import column_cost

# Column shell + internals
cost = column_cost(diameter=2.0, height=20.0, n_trays=30, column_type='tray_column')
```

#### Pumps and Compressors

```python
PUMP_COSTS = {
    'centrifugal': CostParams(0, 4000, 0.55, 1, 500, 'kW', 2019),
    'positive_displacement': CostParams(0, 6000, 0.60, 0.5, 100, 'kW', 2019),
}

COMPRESSOR_COSTS = {
    'reciprocating': CostParams(0, 15000, 0.70, 10, 5000, 'kW', 2019),
    'centrifugal': CostParams(0, 25000, 0.65, 100, 20000, 'kW', 2019),
}
```

**Size Parameter**: Power (kW)

```python
from difflow.economics.capital import pump_cost, compressor_cost

pump = pump_cost(power=10.0, pump_type='centrifugal')
comp = compressor_cost(power=500.0, comp_type='centrifugal')
```

### Installation Factors

Equipment purchase cost must be multiplied by installation factors to get installed cost.

```python
from difflow.economics.capital import InstallationFactors

class InstallationFactors(NamedTuple):
    bare_module: float      # Bare module factor
    piping: float           # Piping factor
    instrumentation: float  # Instrumentation
    electrical: float       # Electrical
    buildings: float        # Buildings
    site_prep: float        # Site preparation
    auxiliaries: float      # Service facilities
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
installed = installed_cost(purchase_cost=100000, factor=3.5)  # $350,000

# Detailed breakdown
detailed = installed_cost_detailed(
    purchase_cost=100000,
    factors=InstallationFactors(
        bare_module=1.0,
        piping=0.45,
        instrumentation=0.20,
        electrical=0.15,
        buildings=0.10,
        site_prep=0.05,
        auxiliaries=0.10
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
# Uses Lang factor approach: TCI ≈ 4.7 × Equipment Cost

# Detailed breakdown
result = total_capital_investment_detailed(
    equipment_costs=equipment_costs,
    installation_factor=1.5,
    indirect_factor=0.3,
    working_capital_fraction=0.15
)

print(f"Direct costs: ${result.direct_costs:,.0f}")
print(f"Indirect costs: ${result.indirect_costs:,.0f}")
print(f"Fixed capital: ${result.fixed_capital:,.0f}")
print(f"Working capital: ${result.working_capital:,.0f}")
print(f"Total capital: ${result.total_capital:,.0f}")
```

---

## Utility Costs

**Location**: `difflow/economics/utilities.py`

### Utility Prices

```python
from difflow.economics.utilities import UtilityPrices, DEFAULT_PRICES

class UtilityPrices(NamedTuple):
    steam_lp: float      # Low pressure steam ($/kg)
    steam_mp: float      # Medium pressure steam ($/kg)
    steam_hp: float      # High pressure steam ($/kg)
    cooling_water: float # Cooling water ($/m³)
    electricity: float   # Electricity ($/kWh)
    fuel_gas: float      # Fuel gas ($/GJ)
    process_water: float # Process water ($/m³)

DEFAULT_PRICES = UtilityPrices(
    steam_lp=0.015,      # 150 psig
    steam_mp=0.020,      # 400 psig
    steam_hp=0.030,      # 600 psig
    cooling_water=0.05,
    electricity=0.07,
    fuel_gas=4.0,
    process_water=0.50
)
```

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
annual_cost = steam_cost_from_duty(Q_W=Q, steam_type='lp', hours_per_year=8000)

# Calculate steam flowrate
steam_rate = steam_flowrate_from_duty(Q_W=Q, steam_type='lp')  # kg/h
```

**Equations**:

$$\dot{m}_{steam} = \frac{Q}{\Delta H_{vap}}$$

$$C_{steam} = \dot{m}_{steam} \cdot P_{steam} \cdot t_{operation}$$

### Cooling Water

```python
from difflow.economics.utilities import (
    cooling_water_cost,
    cooling_water_flowrate
)

# Cooling water for 500 kW duty with 10°C rise
Q = 500000  # W
annual_cost = cooling_water_cost(Q_W=Q, dT=10.0, hours_per_year=8000)

# Required flowrate
cw_rate = cooling_water_flowrate(Q_W=Q, dT=10.0)  # kg/h
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

# Annual electricity cost from power consumption
annual_cost = electricity_cost(energy_kWh=1e6, rate=0.07)  # $70,000

# Cost rate from continuous power
cost_rate = electricity_cost_per_second(power_W=100000)  # $/s

# Pump electricity (includes efficiency)
pump_cost = pump_electricity_cost(
    flow_rate=0.01,      # m³/s
    head=50.0,           # m
    efficiency=0.75,
    hours_per_year=8000
)

# Compressor electricity
comp_cost = compressor_electricity_cost(
    power_shaft=500000,  # W
    efficiency=0.80,
    hours_per_year=8000
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

# Cost depends on temperature level
cost = refrigeration_cost(
    Q_W=100000,          # 100 kW cooling
    T_refrigerant=253.0, # -20°C
    hours_per_year=8000
)
```

**Refrigeration Cost Factor**:

Lower temperatures require more compression work, increasing cost:

| Temperature | Relative Cost |
|-------------|---------------|
| 5°C | 1.0× |
| -20°C | 2.0× |
| -40°C | 3.5× |
| -60°C | 6.0× |

### Combined Utility Costs

```python
from difflow.economics.utilities import (
    utility_cost_from_heat_duties,
    total_utility_cost
)

# Calculate all utilities from process heat duties
heat_duties = {
    'heater_1': 500000,     # W (heating)
    'heater_2': 300000,
    'cooler_1': -400000,    # W (cooling, negative)
    'cooler_2': -600000,
}

utility_costs = utility_cost_from_heat_duties(
    duties=heat_duties,
    prices=DEFAULT_PRICES,
    hours_per_year=8000
)

print(f"Steam cost: ${utility_costs['steam']:,.0f}/year")
print(f"Cooling water: ${utility_costs['cooling_water']:,.0f}/year")
print(f"Total utilities: ${utility_costs['total']:,.0f}/year")
```

---

## Operating Costs

**Location**: `difflow/economics/opex.py`

### Raw Materials

```python
from difflow.economics.opex import (
    RawMaterial,
    raw_material_cost,
    total_raw_material_cost
)

class RawMaterial(NamedTuple):
    name: str
    molecular_weight: float  # g/mol
    cost_per_kg: float       # $/kg
    quantity_annual: float   # kg/year or mol/year
```

```python
# Define raw materials
materials = [
    RawMaterial('methanol', 32.04, 0.40, 1e7),    # 10,000 tonnes/year
    RawMaterial('oxygen', 32.00, 0.05, 5e6),      # 5,000 tonnes/year
    RawMaterial('catalyst', 100.0, 50.0, 1000),   # 1 tonne/year
]

total_rm_cost = total_raw_material_cost(materials)
print(f"Annual raw material cost: ${total_rm_cost:,.0f}")
```

**From Molar Quantities**:

```python
from difflow.economics.opex import raw_material_cost_molar

# Cost from molar flow rates
cost = raw_material_cost_molar(
    molar_flow=100.0,        # mol/s
    MW=32.04,                # g/mol
    cost_per_kg=0.40,        # $/kg
    hours_per_year=8000
)
```

### Labor

```python
from difflow.economics.opex import (
    LaborRates,
    operating_labor_cost,
    labor_cost_from_equipment
)

class LaborRates(NamedTuple):
    operator: float      # $/hour
    supervisor: float    # $/hour
    engineer: float      # $/hour

DEFAULT_LABOR_RATES = LaborRates(
    operator=35.0,
    supervisor=50.0,
    engineer=75.0
)
```

**Labor Estimation Methods**:

1. **Direct calculation**:
```python
labor_cost = operating_labor_cost(
    n_operators=6,           # Operators per shift
    n_shifts=5,              # 5-shift rotation for 24/7
    hourly_rate=35.0,
    hours_per_year=8760
)
```

2. **From equipment count** (correlation):
```python
# Rough estimate: 1 operator per 2-4 major equipment items
labor_cost = labor_cost_from_equipment(
    n_equipment=20,
    processing_type='fluids',  # 'fluids' or 'solids'
    hourly_rate=35.0
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

class OverheadFactors(NamedTuple):
    maintenance: float     # % of FCI
    insurance: float       # % of FCI
    property_tax: float    # % of FCI
    admin_overhead: float  # % of operating labor
```

**Typical Factors**:

| Item | % of FCI |
|------|----------|
| Maintenance | 2-4% |
| Insurance | 0.5-1% |
| Property taxes | 1-2% |
| Plant overhead | 50-60% of labor |

```python
FCI = 50e6  # $50M fixed capital

maintenance = maintenance_cost(FCI, rate=0.03)  # 3% of FCI = $1.5M
insurance = insurance_taxes_cost(FCI, rate=0.015)  # 1.5% = $750k
overhead = plant_overhead_cost(labor_cost=500000, rate=0.55)  # $275k
```

### Total Operating Cost

```python
from difflow.economics.opex import (
    calculate_opex,
    simple_opex,
    OperatingCostBreakdown,
    com_from_correlations
)

# Detailed OPEX calculation
opex = calculate_opex(
    raw_materials=4e6,       # $/year
    utilities=2e6,           # $/year
    labor=1e6,               # $/year
    maintenance=1.5e6,       # $/year
    insurance=0.75e6,        # $/year
    overhead=0.55e6,         # $/year
)

print(f"Total OPEX: ${opex.total:,.0f}/year")

# Quick estimate from FCI
simple = simple_opex(FCI=50e6, production_rate=1e7)  # 10,000 tonnes/year
print(f"Estimated OPEX: ${simple:,.0f}/year")
```

**Cost of Manufacturing (COM)** correlation:

$$COM = 0.280 \cdot FCI + 2.73 \cdot C_{OL} + 1.23 \cdot (C_{UT} + C_{RM})$$

Where:
- $FCI$: Fixed capital investment
- $C_{OL}$: Operating labor cost
- $C_{UT}$: Utility cost
- $C_{RM}$: Raw material cost

```python
com = com_from_correlations(
    FCI=50e6,
    labor=1e6,
    utilities=2e6,
    raw_materials=4e6
)
```

### Operating Schedule

```python
from difflow.economics.opex import OperatingSchedule, DEFAULT_SCHEDULE

class OperatingSchedule(NamedTuple):
    hours_per_year: int    # Operating hours
    days_per_year: int     # Operating days
    batch_duration: float  # For batch processes (hours)

DEFAULT_SCHEDULE = OperatingSchedule(
    hours_per_year=8000,   # ~91% availability
    days_per_year=330,
    batch_duration=None
)

# Constants
HOURS_PER_YEAR = 8000
SECONDS_PER_YEAR = 28800000
```

---

## Profitability Analysis

**Location**: `difflow/economics/profitability.py`

### Financial Parameters

```python
from difflow.economics.profitability import FinancialParams

@dataclass
class FinancialParams:
    discount_rate: float = 0.10      # 10% WACC
    tax_rate: float = 0.21           # 21% corporate tax
    depreciation_years: int = 10     # MACRS 10-year
    plant_life: int = 20             # 20-year project life
    construction_years: int = 2      # 2-year construction
    salvage_fraction: float = 0.05   # 5% salvage value
    working_capital_fraction: float = 0.15  # 15% of FCI
    inflation_rate: float = 0.02     # 2% annual inflation
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
PV = present_value(FV=1e6, rate=0.10, years=10)  # $385,543

# Future value of present cash
FV = future_value(PV=1e6, rate=0.10, years=10)  # $2,593,742

# Discount factor
df = discount_factor(rate=0.10, years=10)  # 0.3855

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
MACRS_SCHEDULES = {
    5: [0.200, 0.320, 0.192, 0.115, 0.115, 0.058],
    7: [0.143, 0.245, 0.175, 0.125, 0.089, 0.089, 0.089, 0.045],
    10: [0.100, 0.180, 0.144, 0.115, 0.092, 0.074, 0.066, 0.066, 0.066, 0.066, 0.033],
    15: [...],  # 15-year schedule
}
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

# Simple NPV
cash_flows = [-50e6, 10e6, 12e6, 14e6, 14e6, 14e6]  # Year 0-5
npv_value = npv(cash_flows, discount_rate=0.10)

# NPV with construction period
npv_value = npv_with_construction(
    capital=50e6,
    annual_cash_flow=10e6,
    discount_rate=0.10,
    construction_years=2,
    plant_life=20
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

# IRR calculation (iterative)
cash_flows = [-50e6, 10e6, 12e6, 14e6, 14e6, 14e6]
irr_value = irr(cash_flows)

# Quick approximation
irr_approx_value = irr_approx(
    capital=50e6,
    annual_cash_flow=10e6,
    plant_life=20
)
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
payback = simple_payback(capital=50e6, annual_cash_flow=10e6)  # 5 years

# Discounted payback (accounts for time value)
d_payback = discounted_payback(
    capital=50e6,
    annual_cash_flow=10e6,
    discount_rate=0.10
)  # 7.3 years
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
    plant_life=20
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
    total_cost=40e6,
    target_roi=0.20,
    capital=50e6,
    production=1e7
)  # $5.00/kg

# MSP for NPV = 0 (most rigorous)
msp_npv = msp_with_npv_zero(
    FCI=50e6,
    OPEX=35e6,
    tax_rate=0.21,
    discount_rate=0.10,
    plant_life=20,
    production=1e7
)
```

**Equation** (NPV = 0):

$$MSP = \frac{CRF \cdot FCI + OPEX - \text{Tax Savings}}{(1-\tau) \cdot Q}$$

### Cash Flow Analysis

```python
from difflow.economics.profitability import (
    generate_cash_flows,
    full_cash_flow_analysis,
    CashFlowResult
)

# Generate year-by-year cash flows
cash_flows = generate_cash_flows(
    FCI=50e6,
    OPEX=30e6,
    revenue=45e6,
    tax_rate=0.21,
    depreciation_schedule='MACRS_10',
    plant_life=20,
    construction_years=2
)

# Full analysis
result = full_cash_flow_analysis(
    FCI=50e6,
    annual_revenue=45e6,
    annual_OPEX=30e6,
    tax_rate=0.21,
    discount_rate=0.10,
    plant_life=20
)

print(f"NPV: ${result.npv:,.0f}")
print(f"IRR: {result.irr:.1%}")
print(f"Payback: {result.payback_period:.1f} years")
print(f"ROI: {result.roi:.1%}")
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
    base_case={'FCI': 50e6, 'revenue': 45e6, 'OPEX': 30e6},
    variations={'FCI': 0.20, 'revenue': 0.15, 'OPEX': 0.10},
    discount_rate=0.10
)

# Identify most critical parameters
for param, (npv_low, npv_high) in sensitivities.items():
    print(f"{param}: NPV ranges from ${npv_low/1e6:.1f}M to ${npv_high/1e6:.1f}M")
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

# Historical CEPCI values
CEPCI_HISTORICAL = {
    1957: 100.0,    # Base year
    1980: 261.2,
    1990: 357.6,
    2000: 394.1,
    2010: 532.9,
    2019: 607.5,
    2020: 596.2,
    2021: 708.8,
    2022: 816.0,
    2023: 797.9,
}

# Get CEPCI for specific year
cepci_2019 = get_cepci(2019)  # 607.5
cepci_2023 = get_cepci(2023)  # 797.9
```

**Cost Escalation**:

$$C_{new} = C_{old} \times \frac{CEPCI_{new}}{CEPCI_{old}}$$

```python
# Escalate 2019 cost to 2023
cost_2019 = 1e6
cost_2023 = escalate_cost(cost_2019, base_year=2019, current_year=2023)
# $1,000,000 × (797.9/607.5) = $1,313,415
```

### Equipment-Specific Escalation

Different equipment types may have different cost trends:

```python
from difflow.economics.indices import CEPCI_RATIOS

# Pre-computed ratios for common escalations
ratio_2019_to_2024 = CEPCI_RATIOS[(2019, 2024)]

# Apply to equipment cost
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
factor = inflation_factor(rate=0.02, years=5)  # 1.104

# Continuous compounding
factor_cont = inflation_factor_continuous(rate=0.02, years=5)  # 1.105
```

---

## Examples

### Complete TEA for a Methanol Plant

```python
import jax.numpy as jnp
from difflow.economics.capital import (
    reactor_cost, heat_exchanger_cost, column_cost,
    total_capital_investment
)
from difflow.economics.utilities import utility_cost_from_heat_duties
from difflow.economics.opex import calculate_opex, raw_material_cost_molar
from difflow.economics.profitability import full_cash_flow_analysis, msp_with_npv_zero

# Plant specifications
production = 100000  # tonnes/year methanol
hours_per_year = 8000

# Equipment costs
equipment = {
    'reactor': reactor_cost(50.0, 'cstr_jacketed'),
    'distillation': column_cost(3.0, 30.0, 40, 'tray_column'),
    'heat_exchangers': 3 * heat_exchanger_cost(200.0, 'shell_tube_floating'),
    'compressor': compressor_cost(2000.0, 'centrifugal'),
}
total_equipment = sum(equipment.values())

# Capital investment
TCI = total_capital_investment(total_equipment)
FCI = TCI * 0.85  # 85% of TCI is fixed capital

# Utilities
heat_duties = {
    'reactor_cooling': -5e6,    # 5 MW cooling
    'reboiler': 8e6,            # 8 MW heating
    'condenser': -6e6,          # 6 MW cooling
    'feed_heater': 2e6,         # 2 MW heating
}
utilities = utility_cost_from_heat_duties(heat_duties, hours_per_year=hours_per_year)

# Raw materials
syngas_cost = raw_material_cost_molar(
    molar_flow=production * 1000 / 32.04 / hours_per_year / 3600 * 3,  # 3:1 H2:CO ratio
    MW=10.0,  # approximate syngas MW
    cost_per_kg=0.15,
    hours_per_year=hours_per_year
)

# Labor and overhead
labor = 1.5e6  # $1.5M/year
maintenance = 0.03 * FCI
insurance = 0.015 * FCI

# Total OPEX
opex = calculate_opex(
    raw_materials=syngas_cost,
    utilities=utilities['total'],
    labor=labor,
    maintenance=maintenance,
    insurance=insurance,
    overhead=0.55 * labor
)

# Revenue (at $400/tonne methanol)
revenue = production * 400

# Profitability analysis
result = full_cash_flow_analysis(
    FCI=FCI,
    annual_revenue=revenue,
    annual_OPEX=opex.total,
    tax_rate=0.21,
    discount_rate=0.10,
    plant_life=20
)

print(f"\n=== Methanol Plant TEA ===")
print(f"Production: {production:,} tonnes/year")
print(f"Total Capital Investment: ${TCI/1e6:.1f} M")
print(f"Fixed Capital: ${FCI/1e6:.1f} M")
print(f"Annual OPEX: ${opex.total/1e6:.1f} M")
print(f"Annual Revenue: ${revenue/1e6:.1f} M")
print(f"\nProfitability Metrics:")
print(f"  NPV (10%): ${result.npv/1e6:.1f} M")
print(f"  IRR: {result.irr:.1%}")
print(f"  Payback: {result.payback_period:.1f} years")

# Minimum selling price
msp = msp_with_npv_zero(
    FCI=FCI,
    OPEX=opex.total,
    tax_rate=0.21,
    discount_rate=0.10,
    plant_life=20,
    production=production * 1000  # kg/year
)
print(f"  MSP: ${msp:.2f}/kg = ${msp*1000:.0f}/tonne")
```

### Optimization with Gradient-Based Methods

All TEA functions are JAX-differentiable, enabling gradient-based optimization:

```python
import jax
from difflow.economics.profitability import npv_objective

def plant_npv(design_params):
    """NPV as function of design parameters."""
    reactor_volume, column_stages = design_params

    # Equipment costs depend on design
    equip_cost = reactor_cost(reactor_volume, 'cstr_jacketed')
    equip_cost += column_cost(2.0, 20.0, column_stages, 'tray_column')

    FCI = total_capital_investment(equip_cost) * 0.85

    # Operating costs depend on design (simplified)
    OPEX = 0.15 * FCI  # Rough correlation

    # Revenue depends on conversion (which depends on reactor size)
    conversion = 1 - jnp.exp(-0.1 * reactor_volume)  # Simplified kinetics
    revenue = conversion * 1e8  # Base revenue

    return npv_objective(FCI, revenue, OPEX, discount_rate=0.10, plant_life=20)

# Gradient of NPV w.r.t. design parameters
grad_npv = jax.grad(plant_npv)

# Optimization loop
design = jnp.array([10.0, 20.0])  # Initial guess
learning_rate = 0.1

for i in range(100):
    grad = grad_npv(design)
    design = design + learning_rate * grad  # Gradient ascent (maximize NPV)

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
