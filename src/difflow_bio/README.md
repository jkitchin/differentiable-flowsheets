# difflow_bio - Bio Manufacturing Plugin

A comprehensive plugin for modeling and optimizing biopharmaceutical manufacturing processes using JAX-based automatic differentiation.

## Installation

```bash
pip install difflow[bio]
# or for development
pip install -e ".[bio]"
```

## Submodules

### `units/`
All unit operations for biopharmaceutical manufacturing:

#### `bioreactors.py` - Upstream Processing
Bioreactor models for cell culture:
- **ContinuousBioreactor**: Chemostat/continuous stirred-tank bioreactor
- **FedBatchBioreactor**: Fed-batch reactor with substrate feeding profiles
- **BioreactorParams**, **FedBatchParams**: Parameter dataclasses

Growth kinetics functions:
- `monod_kinetics()`: Classic Monod growth model
- `substrate_inhibition_kinetics()`: Andrews/Haldane inhibition model
- `product_inhibition_kinetics()`: Product feedback inhibition
- `contois_kinetics()`: Biomass-dependent specific growth rate

Utility functions:
- `dilution_rate()`, `residence_time()`, `optimal_dilution_rate()`

#### `centrifuge.py` - Cell Harvesting
Centrifugation models for cell separation:
- **Centrifuge**: General centrifuge using Sigma factor theory
- **DiscStackCentrifuge**: Industrial disc-stack centrifuge
- **CentrifugeParams**, **DiscStackParams**: Parameter dataclasses

Theory functions:
- `stokes_velocity()`: Terminal settling velocity
- `critical_particle_diameter()`: Minimum separable particle size
- `disc_stack_sigma()`, `tubular_bowl_sigma()`: Sigma factor calculations
- `centrifuge_scale_up()`: Scale-up correlations
- `g_force()`: Relative centrifugal force

#### `filtration.py` - Concentration & Buffer Exchange
Membrane filtration models:
- **Ultrafiltration**: Protein concentration via UF membranes
- **Diafiltration**: Buffer exchange via constant-volume diafiltration
- **TFF**: Tangential flow filtration system (combined UF/DF)
- **UltrafiltrationParams**, **DiafiltrationParams**: Parameter dataclasses

Transport functions:
- `concentration_polarization()`: Boundary layer effects
- `gel_layer_flux()`: Gel-polarization model for flux prediction
- `diavolumes_required()`: Calculate wash volumes for target removal
- `rejection_from_mw()`: Estimate rejection from molecular weight

#### `chromatography.py` - Purification
Chromatography models for protein purification:
- **ProteinAChromatography**: Affinity capture for monoclonal antibodies
- **IonExchangeChromatography**: CEX/AEX polishing steps
- **SizeExclusionChromatography**: SEC for aggregate removal
- **ProteinAParams**, **IEXParams**, **SECParams**: Parameter dataclasses

Isotherm models:
- `langmuir_isotherm()`: Single-component Langmuir
- `linear_isotherm()`: Linear partition coefficient
- `langmuir_freundlich_isotherm()`: Heterogeneous binding sites

Column performance:
- `dynamic_binding_capacity()`: DBC at breakthrough
- `column_productivity()`: Throughput calculations
- `resolution()`: Peak separation
- `plate_count()`, `hetp()`: Column efficiency metrics

## Quick Start

```python
from difflow_bio import (
    ContinuousBioreactor, BioreactorParams,
    monod_kinetics, optimal_dilution_rate,
    ProteinAChromatography, ProteinAParams,
    Ultrafiltration, UltrafiltrationParams,
)

# Create a chemostat bioreactor
params = BioreactorParams(
    V=1000.0,  # 1000 L working volume
    mu_max=0.4,  # 1/h
    Ks=0.1,  # g/L
    Yxs=0.5,  # g biomass / g substrate
)
bioreactor = ContinuousBioreactor(params)

# Calculate optimal dilution rate
D_opt = optimal_dilution_rate(params.mu_max, params.Ks, S_in=10.0)

# Run bioreactor
outlet = bioreactor(feed_stream, D=D_opt)

# Protein A capture step
pA_params = ProteinAParams(
    column_volume=5.0,  # L
    binding_capacity=35.0,  # g/L resin
    flow_rate=2.0,  # CV/h
)
capture = ProteinAChromatography(pA_params)
eluate = capture(harvest)

# All operations support automatic differentiation
from jax import grad
d_yield_d_flow = grad(lambda flow: mab_yield(flow))(2.0)
```

## Key Features

- **Fully differentiable**: All models compatible with JAX's `grad`, `jit`, `vmap`
- **Upstream processing**: Chemostat and fed-batch bioreactor models
- **Downstream processing**: Complete DSP train (harvest → capture → polish)
- **Industry-standard kinetics**: Monod, inhibition models, Langmuir isotherms
- **Scale-up ready**: Sigma factors, column productivity metrics

## Typical mAb Process Train

```
Cell Culture → Harvest → Capture → Viral Inactivation → Polish → UF/DF → Fill
     ↓            ↓          ↓                              ↓        ↓
  FedBatch    Centrifuge  ProteinA                        IEX/SEC   TFF
```

## References

- Carta G, Jungbauer A (2010). Protein Chromatography: Process Development and Scale-Up
- Shuler ML, Kargi F (2002). Bioprocess Engineering: Basic Concepts
- van Reis R, Zydney A (2007). Bioprocess membrane technology. J. Membr. Sci. 297:16-50
