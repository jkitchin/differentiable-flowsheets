---
title: 'difflow: A JAX-Based Differentiable Framework for Chemical Process Simulation and Optimization'
tags:
  - Python
  - JAX
  - chemical engineering
  - process simulation
  - automatic differentiation
  - optimization
authors:
  - name: John R. Kitchin
    orcid: 0000-0003-2625-9232
    affiliation: 1
affiliations:
  - name: Department of Chemical Engineering, Carnegie Mellon University, Pittsburgh, PA, USA
    index: 1
date: 5 January 2026
bibliography: paper.bib
---

# Summary

Chemical process simulation is fundamental to the design and optimization of industrial plants, yet most existing tools rely on finite-difference approximations or derivative-free methods for optimization. `difflow` is a Python framework that enables fully differentiable simulation of chemical processes using JAX [@jax2018github], providing exact gradients through automatic differentiation for gradient-based optimization, sensitivity analysis, and uncertainty quantification.

The framework implements common unit operations—reactors (CSTR, PFR, fed-batch), separators (flash, distillation, liquid-liquid extraction), and heat exchangers—with thermodynamic models ranging from ideal mixtures to cubic equations of state (Peng-Robinson, SRK). Flowsheets with recycle streams are solved using acceleration methods (Anderson, Wegstein), with gradients computed through implicit differentiation of the converged solution. A technoeconomic analysis module provides differentiable capital and operating cost correlations, enabling simultaneous technical and economic optimization.

# Statement of Need

Process systems engineers routinely optimize designs involving thousands of decision variables subject to complex constraints. Traditional simulators like Aspen Plus and HYSYS are powerful but closed-source, expensive, and provide only finite-difference gradients. Open-source alternatives such as DWSIM and COCO lack automatic differentiation entirely. IDAES [@idaes] built on Pyomo [@pyomo] offers equation-oriented modeling with symbolic derivatives but requires algebraic problem formulations rather than the procedural simulation approach familiar to most engineers.

`difflow` fills this gap by combining the intuitive sequential-modular simulation paradigm with JAX's automatic differentiation. Users define unit operations as Python functions, compose them into flowsheets, and obtain exact gradients automatically. This enables:

- **Gradient-based optimization** of reactor volumes, separation stages, and operating conditions
- **Sensitivity analysis** computing $\partial y / \partial x$ for any output with respect to any input
- **Uncertainty propagation** through linear (Jacobian-based) or Monte Carlo methods
- **Integration with machine learning** frameworks for hybrid physics-ML models

The framework includes domain-specific plugins for biopharmaceutical manufacturing (bioreactors, chromatography, membrane filtration) and rare earth element separations (solvent extraction, precipitation), demonstrating extensibility to specialized applications.

# Key Features

**Differentiable Unit Operations.** All operations support `jax.grad`, `jax.jacobian`, and `jax.jit`:

```python
from difflow import CSTR, make_stream
import jax

def conversion(volume):
    reactor = CSTR(V=volume, rate_fn=rate_fn, stoich=stoich)
    outlet, info = reactor(inlet_stream)
    return info["conversion"]["A"]

# Exact gradient via automatic differentiation
dX_dV = jax.grad(conversion)(2.0)  # dConversion/dVolume
```

**Flowsheets with Recycles.** Implicit differentiation enables gradients through converged recycle loops without differentiating through iteration history.

**Differentiable Economics.** Capital costs, operating costs, NPV, and minimum selling price are all JAX-compatible, enabling optimization of economic objectives directly.

# Example: Sensitivity Analysis

The following demonstrates computing the sensitivity of reactor conversion to both volume and rate constant simultaneously:

```python
import jax
import jax.numpy as jnp
from difflow import CSTR, CSTRParams, make_stream

def experiment(params):
    """Returns conversion as function of (V, k)."""
    cstr = CSTR(CSTRParams(V=params["V"], rate_params={"k": params["k"]}, ...))
    outlet, info = cstr(inlet)
    return info["conversion"]["A"]

# Compute gradient with respect to all parameters
params = {"V": jnp.array(2.0), "k": jnp.array(0.5)}
sensitivities = jax.grad(experiment)(params)
# sensitivities["V"] = dX/dV, sensitivities["k"] = dX/dk
```

These gradients match analytical derivatives exactly and are computed efficiently via reverse-mode automatic differentiation.

# Availability

`difflow` is available on GitHub at [https://github.com/jkitchin/differentiable-flowsheets](https://github.com/jkitchin/differentiable-flowsheets) under the MIT license. Documentation and example notebooks covering reactors, separations, optimization, uncertainty quantification, and technoeconomic analysis are included.

# AI Disclosure

This software was developed with assistance from Claude Code (Anthropic), an AI coding assistant. The author directed the development, made architectural decisions, and reviewed all generated code.

# References
