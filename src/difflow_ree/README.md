# difflow_ree - Rare Earth Element Separation Plugin

A comprehensive plugin for modeling and optimizing rare earth element (REE) solvent extraction processes using JAX-based automatic differentiation.

## Installation

```bash
pip install difflow[ree]
# or for development
pip install -e ".[ree]"
```

## Submodules

### `database.py`
Property databases for REE separation:
- **REEDatabase**: Properties for 10 commercial rare earth elements (La, Ce, Pr, Nd, Sm, Eu, Gd, Tb, Dy, Y)
- **ExtractantDatabase**: Properties for 4 extractant systems (D2EHPA, PC88A, Cyanex272, TBP)
- **SeparationFactorDatabase**: Element pair separation factors
- Convenience functions: `get_element()`, `get_extractant()`, `get_separation_factor()`
- JAX-compatible accessors: `get_atomic_weight_array()`, `get_price_array()`, `get_ionic_radius_array()`

### `equilibrium/`
Thermodynamic equilibrium models for solvent extraction:
- **distribution.py**: `REEDistribution` - pH-dependent distribution coefficient models
- **loading.py**: `LoadingIsotherm` - Organic phase loading corrections, Langmuir isotherms
- **speciation.py**: `REESpeciation` - Aqueous speciation (sulfate, chloride complexes)
- **network.py** / **mass_action.py**: reaction networks as data and the closed mass-action section, where pH is an output (#196)
- **saponification.py**: `SaponifiedSection` - pre-neutralized extractant, the counter-ion balance and the organic buffer (#197)

### `units/`
Unit operation models (all differentiable):
- **extraction.py**: `REEExtractor`, `REEMixerSettler` - Multi-stage extraction cascades
- **scrubbing.py**: `REEScrubber` - Impurity removal sections
- **stripping.py**: `REEStripper` - Product recovery from organic phase
- **precipitation.py**: `OxalatePrecipitator`, `CarbonatePrecipitator`, `HydroxidePrecipitator` - Solid product recovery
- **cerium.py**: `CeriumOxidizer` - Selective Ce(IV) oxidation and CeO2 precipitation
- **saponification.py**: `Saponifier` - contact the organic with base before the cascade, tracking the reagent duty (#197)

### `flowsheets/`
Pre-built flowsheet templates for common configurations:
- **extract_strip.py**: `ExtractStripCircuit` - Basic 2-section circuit
- **extract_scrub_strip.py**: `ExtractScrubStripCircuit` - Industrial 3-section circuit
- **split_shell.py**: `SplitShellCascade` - Multi-product split-shell design
- **full_train.py**: `FullSeparationTrain`, `GroupSeparator` - Complete separation train

### `economics/`
Technoeconomic analysis:
- **costs.py**: `REEPricing`, `ReagentCosts`, `OperatingCosts`
- **saponification.py**: kg base per kg REO, ammonium-nitrogen and dissolved-salt effluent loads, `compare_counter_ions()` (#197)
- Functions: `estimate_capex()`, `estimate_opex()`, `calculate_revenue()`, `calculate_profit()`, `minimum_selling_price()`

### `data/`
YAML data files with element properties, extractant parameters, and separation factors.

## Quick Start

```python
from difflow_ree import (
    get_element, get_extractant,
    ExtractStripCircuit, ExtractStripParams,
)
from difflow.streams import make_stream

# Get neodymium properties
nd = get_element('Nd')
print(f"Nd price: ${nd.price_usd_kg}/kg")
print(f"Atomic weight: {nd.atomic_weight} g/mol")

# Get D2EHPA extractant
d2ehpa = get_extractant('D2EHPA')
print(f"pKa: {d2ehpa.pKa}")

# Create extraction circuit
params = ExtractStripParams(
    extractant='D2EHPA',
    elements=('La', 'Ce', 'Nd', 'Dy'),
    n_extraction_stages=10,
    n_stripping_stages=5,
)
circuit = ExtractStripCircuit(params)

# Create aqueous feed stream with REE
feed = make_stream(
    flows={"H2O": 10.0, "La": 0.01, "Ce": 0.02, "Nd": 0.02, "Dy": 0.01},
    T=298.15,
    P=101325.0,
)

# Run the circuit (solvent and strip acid are created internally)
results = circuit(feed)
print(f"Overall recovery: {results['recovery']:.3f}")
print(f"Element recoveries: {results['element_recovery']}")
```

## Key Features

- **Fully differentiable**: All models compatible with JAX's `grad`, `jit`, `vmap`
- **Multi-element**: Handles mixtures of any REE combination
- **pH-dependent**: Realistic extraction chemistry with pH effects
- **Loading corrections**: Accounts for organic phase saturation
- **Saponified extractants**: Counter-ion exchange, the organic acting as a buffer, and kg base per kg REO as an economic and environmental metric (#197)
- **Explicit organic phase**: Extractant + diluent species keys (e.g., `"D2EHPA"` + `"kerosene"`) enable multi-organic flowsheets
- **Pre-built flowsheets**: Common industrial configurations ready to use
- **Economic analysis**: Revenue, profit, and minimum selling price calculations

## Supported Elements

| Element | Symbol | Atomic # | Category |
|---------|--------|----------|----------|
| Lanthanum | La | 57 | Light |
| Cerium | Ce | 58 | Light |
| Praseodymium | Pr | 59 | Light |
| Neodymium | Nd | 60 | Light |
| Samarium | Sm | 62 | Middle |
| Europium | Eu | 63 | Middle |
| Gadolinium | Gd | 64 | Middle |
| Terbium | Tb | 65 | Heavy |
| Dysprosium | Dy | 66 | Heavy |
| Yttrium | Y | 39 | Heavy |

## Supported Extractants

| Extractant | Type | Mechanism | Driving variable | Best For |
|------------|------|-----------|------------------|----------|
| D2EHPA | Acidic | `cation_exchange` | pH | Light REE separation |
| PC88A | Acidic | `cation_exchange` | pH | General REE separation |
| Cyanex272 | Acidic | `cation_exchange` | pH | Heavy REE, Co/Ni |
| TBP | Neutral | `solvating` | `[NO3-]` | Nitrate systems |

The mechanism is carried by the extractant record and decides which correlation
drives `D` (#195). TBP therefore requires a `nitrate_conc`:

```python
REEDistribution(
    extractant="TBP", elements=("Nd",), nitrate_conc=3.0, concentration=1.1
)
```

and raises without one. There is no other path: **TBP's `ph_coefficients` block
has been deleted**, so `REEDistribution(extractant="TBP", ...,
mechanism="cation_exchange")` now raises a `ValueError` naming TBP and pointing
back at the nitrate path. The block modelled a neutral extractant (`pKa: null`,
`protons_released: 0`) as a weak cation exchanger — there is no proton to
exchange, no source reports a pH slope for TBP, and it carried the same refuted
selectivity spread as the old nitrate block. `Extractant.ph_coefficients` is
therefore `dict | None`.

> **TBP's nitrate coefficients are now refitted from primary literature.**
> They are the *only* literature-derived numbers in `data/extractants.yaml`;
> D2EHPA / PC88A / Cyanex272 remain hand-tuned with no recorded source. Fit
> basis: Kraikaew, Srinuttrakul & Chayavadhanakur (2005), *J. Metals, Materials
> and Minerals* **15**(2), 89-95, Table 1, corrected to 3 M NO3⁻ / 1 M TBP /
> 298.15 K; heat of extraction from Ganesh & Pandey (2019), *J. Rad. Nucl.
> Appl.* **4**(2), 109-115 (`dH_Sm = -43.3 kJ/mol`).
>
> At the reference this now gives `D_La = 0.023` … `D_Dy = 0.24` — every value
> below 1, `D_Dy/D_La = 10.2` (was 100), mean adjacent-pair separation factor
> 1.29 per unit atomic number (La(57) to Dy(66) is 9 steps, Pm included). `b = 3.0` for every element is *stoichiometric*, not fitted. Every
> temperature coefficient is now **positive**, i.e. exothermic (they were all
> negative, which asserted the opposite).
>
> **Validity window:** neutral nitrate salt, ≤ 0.5 M free acid, 1–6 M NO3⁻,
> 283–326 K. Use for relative REE selectivity, stage-count intuition, and trend
> or sensitivity studies. Do **not** use for stage counts, solvent inventories,
> absolute recoveries, HNO3-supplied nitrate, loaded solvent, or Y at any other
> acidity. Tb is *interpolated*, not measured; nine of the ten temperature
> coefficients are assumed equal to Sm. The full provenance, the per-element
> measured/interpolated/assumed labels and the known gaps are written out on the
> TBP record in `data/extractants.yaml` — read it before using any TBP number.
>
> The 0.5 M `extractant_conc` default is a cation-exchange default and knocks
> TBP's `D` down 8x; TBP is run at ~30% v/v = 1.1 M.

`pH` is on the concentration scale and `ionic_strength=None` (no activity
correction, a conditional constant at the operating ionic strength) is the
default and the right choice for a concentrated liquor (#194). When an ionic
strength *is* supplied, the value fed to the activity model is clamped at that
model's documented range limit, because the Davies bracket changes sign at
I = 1.9404 M and above that the correction would multiply `D` instead of
reducing it (6.5x at 3 M). Pass `extrapolate_activity_model=True` to opt into
the raw extrapolation. See `docs/unit-operations-ree.md`.

## References

- Xie F et al. (2014). A critical review on solvent extraction of rare earths. Miner. Eng. 56:10-28
- Krishnamurthy N, Gupta CK (2004). Extractive Metallurgy of Rare Earths
