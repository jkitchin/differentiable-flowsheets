# Literature Review: Differentiable Flowsheets and Related Work

## Executive Summary

This literature review surveys the emerging field of differentiable simulation for chemical process engineering, with a focus on automatic differentiation (AD) for flowsheet modeling, optimization, and uncertainty quantification. The **difflow** project represents a novel contribution to this space by combining JAX-based automatic differentiation with comprehensive chemical engineering unit operations, thermodynamic modeling, and technoeconomic analysis in a unified, fully differentiable framework.

## Table of Contents

1. [Introduction](#1-introduction)
2. [Differentiable Process Simulation Frameworks](#2-differentiable-process-simulation-frameworks)
3. [JAX Scientific Computing Ecosystem](#3-jax-scientific-computing-ecosystem)
4. [Julia Ecosystem for Chemical Engineering](#4-julia-ecosystem-for-chemical-engineering)
5. [Traditional Process Systems Engineering Tools](#5-traditional-process-systems-engineering-tools)
6. [Automatic Differentiation in Chemical Engineering](#6-automatic-differentiation-in-chemical-engineering)
7. [Neural Networks and Machine Learning for Process Engineering](#7-neural-networks-and-machine-learning-for-process-engineering)
8. [Physics-Informed Neural Networks](#8-physics-informed-neural-networks)
9. [Implicit Differentiation and Deep Equilibrium Networks](#9-implicit-differentiation-and-deep-equilibrium-networks)
10. [Comparison with difflow](#10-comparison-with-difflow)
11. [Conclusions](#11-conclusions)
12. [References](#12-references)

---

## 1. Introduction

The intersection of automatic differentiation and chemical process simulation represents a significant advancement in process systems engineering. Traditional flowsheet simulators rely on numerical differentiation (finite differences) for sensitivity analysis and optimization, which can be computationally expensive, numerically unstable, and limited in accuracy. The emergence of differentiable programming frameworks—particularly JAX, Julia, and PyTorch—has opened new possibilities for creating fully differentiable process simulators that provide exact gradients for optimization, uncertainty propagation, and machine learning integration.

This review examines the landscape of differentiable simulation tools relevant to chemical engineering, comparing their capabilities with the difflow project developed here.

---

## 2. Differentiable Process Simulation Frameworks

### 2.1 A Novel Perspective Process Simulation Framework (Yang, 2023)

Yang (2023) presented a process simulation framework based on automatic differentiation for thermodynamic and flash equilibrium calculations. The key contributions include:

- **Approach**: Uses state-of-the-art AD frameworks to obtain precise derivatives without altering algorithm logic
- **Focus**: PT, PV, and PH flash calculations with enhanced convergence
- **Key Finding**: AD methods show more uniform gradient distributions and require fewer convergence iterations than numerical differentiation
- **Generalizability**: The method extends to various chemical simulation modules

**Comparison with difflow**:
| Aspect | Yang (2023) | difflow |
|--------|-------------|---------|
| Scope | Flash calculations | Full flowsheet simulation |
| Thermodynamics | Basic flash | Ideal, PR, SRK, NRTL, UNIQUAC |
| Unit operations | Limited | Comprehensive library |
| Economics | No | Full TEA module |
| Framework | Not specified | JAX |

### 2.2 JAX-Fluids (Bezgin et al., 2023)

JAX-Fluids is a fully-differentiable CFD solver for compressible two-phase flows, representing one of the most sophisticated applications of JAX to fluid dynamics:

- **Capabilities**: 3D compressible single-phase and two-phase flows
- **ML Integration**: Seamless hybridization of ML with CFD
- **Scalability**: Demonstrated on up to 512 NVIDIA A100 GPUs and 1024 TPU v3 cores
- **AD Features**: Stable gradient computation across extended integration trajectories

**URL**: https://github.com/tumaer/JAXFLUIDS

**Comparison with difflow**:
| Aspect | JAX-Fluids | difflow |
|--------|------------|---------|
| Domain | CFD | Process engineering |
| Scale | High-performance computing | Desktop/workstation |
| Physics | Navier-Stokes | Mass/energy balances, VLE |
| Application | Fluid mechanics | Chemical plant design |

### 2.3 DiffTaichi (Hu et al., 2020)

DiffTaichi is a differentiable programming language for physical simulation:

- **Performance**: 4.2x shorter code than hand-engineered CUDA while matching speed
- **Speedup**: 188x faster than TensorFlow implementations
- **Features**: Source code transformations preserving arithmetic intensity and parallelism
- **Applications**: Soft body simulation, cloth simulation, fluid dynamics

**URL**: https://arxiv.org/abs/1910.00935

**Comparison with difflow**:
- DiffTaichi focuses on general physical simulation (graphics, robotics)
- difflow is domain-specific for chemical engineering
- Both use source code transformation for gradient computation

---

## 3. JAX Scientific Computing Ecosystem

### 3.1 Diffrax (Kidger, 2021)

Diffrax provides numerical differential equation solvers in JAX with comprehensive features:

- **Equation Types**: ODEs, SDEs, CDEs (ordinary, stochastic, controlled)
- **Solvers**: Tsit5, Dopri8, symplectic solvers, implicit solvers
- **Features**: vmappable, PyTree states, dense solutions, multiple adjoint methods
- **Performance**: Similar to Julia libraries, ~100x faster than PyTorch equivalents

**URL**: https://github.com/patrick-kidger/diffrax

**Relationship to difflow**: difflow uses diffrax as an optional backend for advanced ODE/DAE integration, particularly for stiff systems requiring implicit solvers like Kvaerno5.

### 3.2 Equinox (Kidger & Garcia, 2021)

Equinox provides PyTorch-like neural networks in JAX:

- **Philosophy**: Models are PyTrees, no magic behind the scenes
- **Compatibility**: Works seamlessly with all JAX transformations
- **Extensions**: Runtime errors, PyTree manipulation, filtered transformations

**URL**: https://github.com/patrick-kidger/equinox

**Relationship to difflow**: difflow could integrate equinox-based neural network surrogates for unit operations or property predictions.

### 3.3 JAX MD (Schoenholz & Cubuk, 2020)

JAX MD is a molecular dynamics simulation framework:

- **Features**: NVE, NVT (Nose-Hoover), Brownian dynamics
- **Differentiability**: Entire trajectories can be differentiated for meta-optimization
- **Integration**: Physics simulation environments with neural network potentials

**URL**: https://github.com/jax-md/jax-md

**Comparison with difflow**:
| Aspect | JAX MD | difflow |
|--------|--------|---------|
| Scale | Molecular | Process |
| Time scales | Femtoseconds | Seconds to hours |
| Physics | Interatomic potentials | Thermodynamics, kinetics |
| Application | Materials science | Chemical manufacturing |

---

## 4. Julia Ecosystem for Chemical Engineering

### 4.1 Clapeyron.jl (Walker et al., 2022)

Clapeyron.jl is an extensible, open-source fluid thermodynamics toolkit:

- **Models**: 30+ thermodynamic models including SAFT, cubics, activity coefficients, COSMO-SAC
- **Properties**: Bulk, VLE, LLE, VLLE, critical properties
- **AD Support**: Built-in automatic differentiation via Julia's AD ecosystem
- **Extensibility**: User-contributed models encouraged

**URL**: https://github.com/ClapeyronThermo/Clapeyron.jl

**Publication**: Industrial & Engineering Chemistry Research, 2022

**Comparison with difflow**:
| Aspect | Clapeyron.jl | difflow |
|--------|--------------|---------|
| Language | Julia | Python/JAX |
| Thermodynamics | Extensive (30+ models) | Focused (ideal, PR, SRK, activity) |
| Flowsheet | Property calculations only | Full unit operations |
| Integration | Julia ecosystem | JAX ecosystem |

### 4.2 ProcessSimulator.jl (Riedemann et al., 2024)

Presented at JuliaCon 2024, ProcessSimulator.jl is a differentiable chemical process simulator:

- **Foundation**: Built on ModelingToolkit.jl for symbolic equation representation
- **Thermodynamics**: Interfaces with Clapeyron.jl
- **Simulation**: Steady-state (NonlinearSolve.jl) and dynamic (DifferentialEquations.jl)
- **Optimization**: Interface to JuMP.jl for MINLP optimization

**URL**: https://pretalx.com/juliacon2024/talk/LP3XAL/

**Comparison with difflow**:
| Aspect | ProcessSimulator.jl | difflow |
|--------|---------------------|---------|
| Language | Julia | Python/JAX |
| Symbolic | ModelingToolkit.jl | Not symbolic |
| Thermodynamics | Clapeyron.jl | Built-in |
| Optimization | JuMP.jl | JAX grad + optimistix |
| Maturity | Emerging (2024) | Active development |

### 4.3 SciML Ecosystem

The Julia SciML ecosystem provides comprehensive tools:

- **SciMLSensitivity.jl**: Forward and adjoint sensitivity analysis for differential equations
- **DifferentialEquations.jl**: Comprehensive ODE/DAE/SDE solvers
- **ModelingToolkit.jl**: Symbolic-numeric modeling

**Key Insight**: Benchmarks show forward-mode AD is more efficient for small systems (<100 parameters), while continuous adjoint methods scale better for large systems.

---

## 5. Traditional Process Systems Engineering Tools

### 5.1 IDAES (Lee et al., 2021)

The IDAES Process Systems Engineering Framework is developed by DOE national laboratories:

- **Foundation**: Built on Pyomo for algebraic modeling
- **Capabilities**: Steady-state and dynamic optimization, multi-scale modeling
- **Application**: Power systems, carbon capture, advanced energy systems
- **Optimization**: Interface to IPOPT and other NLP solvers

**URL**: https://github.com/IDAES/idaes-pse

**Comparison with difflow**:
| Aspect | IDAES | difflow |
|--------|-------|---------|
| Approach | Equation-oriented (Pyomo) | Sequential modular |
| Differentiation | Algebraic (AMPL/ASL) | Automatic (JAX) |
| Scale | Large-scale industrial | Research/education |
| Application focus | Power/energy | General chemical |
| GPU support | Limited | Native JAX |

### 5.2 BioSTEAM (Cortés-Peña et al., 2020)

BioSTEAM is a biorefinery simulation and TEA platform:

- **Focus**: Design, simulation, and TEA of biorefineries under uncertainty
- **Validation**: Results match Aspen Plus and SuperPro Designer
- **Uncertainty**: Built-in Monte Carlo analysis
- **Applications**: Biofuels, bioproducts, biorefinery design

**URL**: https://github.com/BioSTEAMDevelopmentGroup/biosteam

**Comparison with difflow**:
| Aspect | BioSTEAM | difflow |
|--------|----------|---------|
| Domain | Biorefineries | General chemical |
| Uncertainty | Monte Carlo | Monte Carlo + gradient-based |
| Differentiation | No AD | Full AD |
| Economics | Comprehensive | Comprehensive |
| Optimization | Scipy | JAX-native |

### 5.3 DWSIM (Medeiros, 2008)

DWSIM is an open-source CAPE-OPEN compliant process simulator:

- **Platform**: Windows, Linux, macOS, Android, iOS
- **Thermodynamics**: Extensive models including CoolProp integration
- **Features**: GUI, petroleum characterization, reaction systems
- **License**: GPL v3

**URL**: https://dwsim.org/

**Comparison with difflow**:
| Aspect | DWSIM | difflow |
|--------|-------|---------|
| UI | Full GUI | Code-first |
| Language | VB.NET/C# | Python/JAX |
| Differentiation | Numerical only | Automatic |
| CAPE-OPEN | Compliant | Not applicable |
| Focus | Industry standard | Research/ML |

### 5.4 OpenModelica for Chemical Process Simulation (Nayak et al., 2019)

OpenModelica has been extended for chemical process simulation:

- **Integration**: ChemSep database and DWSIM thermodynamics ported to Modelica
- **Methods**: NRTL, Peng-Robinson, UNIFAC, UNIQUAC available
- **Validation**: Results compared favorably with Aspen Plus

**Comparison with difflow**:
- OpenModelica uses equation-based acausal modeling
- difflow uses functional programming with explicit causality
- Both support steady-state and dynamic simulation

---

## 6. Automatic Differentiation in Chemical Engineering

### 6.1 Historical Context

Perkins & Sargent (1986) introduced chain-rule differentiation for sequential modular flowsheet optimization, recognizing that:
- Gradient evaluation was the most time-consuming optimization step
- Exact gradients from modular sensitivities significantly reduce computation time

### 6.2 Modern AD Approaches

Soares & Secchi (2003) examined AD tools for dynamic simulation of chemical processes:
- AD provides exact derivatives (to roundoff) for DAE systems
- Critical for determination of iteration matrices and consistent initial conditions
- Both forward and reverse mode AD are applicable

### 6.3 Comparison of Sensitivity Methods

Ma et al. (2021) compared automatic differentiation with continuous sensitivity analysis:
- **Small systems (<100 parameters)**: Forward-mode DSAAD is most efficient
- **Large systems**: Continuous adjoint methods are more efficient
- **Stability**: Discrete sensitivity analysis is more stable; continuous methods are more efficient

### 6.4 Adjoint Methods in Chemical Kinetics

Sandu et al. (2003) developed KPP (Kinetic PreProcessor) for adjoint sensitivity:
- Direct method: Propagates uncertainties forward
- Adjoint method: Identifies sources of uncertainty in outputs
- KPP-1.2 supports both approaches with automatic code generation

**Key Insight for difflow**: The choice between forward and reverse mode AD should depend on the problem structure. For optimization with few parameters, forward mode is efficient; for many parameters (e.g., training neural networks), reverse mode (backpropagation) is preferred.

---

## 7. Neural Networks and Machine Learning for Process Engineering

### 7.1 Neural ODEs for Chemical Kinetics

#### ChemNODE (Owoyele & Pal, 2022)

- **Approach**: Integrate neural network predictions during training
- **Application**: Hydrogen-air autoignition
- **Benefit**: Fraction of computational cost of detailed mechanisms

#### jaxkineticmodel (Douwes et al., 2025)

- **Framework**: JAX/Diffrax implementation
- **Features**: SBML compatibility, hybrid mechanistic-neural models
- **Application**: Large-scale kinetic models (141 parameters)

**URL**: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012733

#### GRxnODE (Hua et al., 2023)

- **Approach**: Residence time distribution-inspired neural ODE architecture
- **Features**: Physical interpretability, data efficiency
- **Application**: Dynamic modeling of flow reactors

### 7.2 Surrogate Modeling

Neural network surrogates accelerate optimization of complex flowsheets:

- **Benefits**: Reduced computational load, faster convergence
- **Approaches**: MLP, RBF networks, Support Vector Machines
- **Advanced Training**: Sobolev training uses gradient information for higher accuracy (Huster et al., 2021)

**Comparison with difflow**: difflow's notebook 13 demonstrates neural network surrogates for equipment (pumps), though the approach could be extended to entire unit operations.

---

## 8. Physics-Informed Neural Networks

### 8.1 PINNs for Reactor Modeling

Wu et al. (2024) developed a decoupling-coupling framework for chemical reactor systems:
- **Challenge**: Multiphysics coupling in PINNs
- **Solution**: Pre-train on decoupled subdomains (flow, heat, mass transfer), then couple
- **Result**: Improved accuracy for complex reactor systems

### 8.2 PINNs for Process Operations

Sitapure & Kwon (2024) addressed limited physical knowledge and data:
- **Approach**: Combine first-principles models with plant data
- **Advantage**: Handle plant-model mismatch
- **Comparison**: Outperformed RNNs in predictive capability

### 8.3 Key Challenges

- Proper weighting of empirical and physics-based loss terms
- Performance in highly turbulent or non-ideal conditions
- Generalization across operating regimes

**Comparison with difflow**: difflow takes a different approach—rather than using neural networks to approximate physics, it makes the physics directly differentiable. This preserves physical accuracy while enabling gradient-based optimization.

---

## 9. Implicit Differentiation and Deep Equilibrium Networks

### 9.1 Theoretical Foundations

Bolte et al. (2021) established nonsmooth implicit differentiation theory:
- **Applicability**: Most practical problems (definable problems)
- **Key Feature**: Compatible with algorithmic differentiation (backpropagation)
- **Applications**: Deep equilibrium networks, conic optimization layers, hyperparameter tuning

### 9.2 Deep Equilibrium Networks

Deep Equilibrium Networks (DEQs) define layers implicitly through fixed-point equations:
- **Benefit**: Infinite depth at fixed memory cost
- **Differentiation**: Via implicit function theorem
- **Relevance**: Similar to recycle loop convergence in flowsheets

### 9.3 Relevance to Process Simulation

Zucchet & Baldi (2022) surveyed bilevel optimization:
- **Connection**: Flowsheet recycle convergence is a fixed-point problem
- **Gradient Computation**: Implicit differentiation through converged solutions
- **Implementation**: difflow uses optimistix for implicit differentiation through recycle loops

**Key Insight for difflow**: The implicit differentiation approach used in difflow for recycle stream convergence is theoretically grounded in the same mathematics as deep equilibrium networks, representing a principled approach to differentiating through iterative solvers.

---

## 10. Comparison with difflow

### 10.1 Unique Features of difflow

| Feature | difflow | Closest Alternative |
|---------|---------|---------------------|
| JAX-based chemical process simulation | Yes | ProcessSimulator.jl (Julia) |
| Full flowsheet with recycle | Yes | IDAES (Pyomo) |
| Integrated TEA with AD | Yes | BioSTEAM (no AD) |
| Uncertainty propagation via AD | Yes | None directly comparable |
| Dynamic simulation with AD | Yes | SciML (Julia) |
| Plugin architecture | Yes | IDAES |

### 10.2 Advantages of difflow

1. **Unified Framework**: Process simulation, economics, and uncertainty quantification in one differentiable pipeline
2. **JAX Ecosystem Integration**: Access to equinox, diffrax, optimistix, and broader ML tools
3. **GPU/TPU Ready**: Native JAX compilation to accelerators
4. **Research-Friendly**: Python-based, Jupyter notebook examples
5. **Domain-Specific**: Purpose-built for chemical engineering, not adapted from general tools

### 10.3 Comparison Matrix

| Capability | difflow | IDAES | BioSTEAM | ProcessSimulator.jl | DWSIM |
|------------|---------|-------|----------|---------------------|-------|
| Automatic Differentiation | JAX | Algebraic | No | Julia AD | No |
| GPU Support | Native | Limited | No | Julia GPU | No |
| Thermodynamics | Moderate | Extensive | Moderate | Extensive (Clapeyron) | Extensive |
| Dynamic Simulation | Yes | Yes | Limited | Yes | Yes |
| Uncertainty Quantification | AD-based | PSUADE | Monte Carlo | AD-based | Limited |
| Economics/TEA | Built-in | Costing module | Built-in | Limited | No |
| Open Source | Yes | Yes | Yes | Yes | Yes |
| Language | Python | Python | Python | Julia | VB.NET |

### 10.4 Gaps and Future Opportunities

Based on this literature review, potential enhancements for difflow include:

1. **Expanded Thermodynamics**: Integration of more SAFT-type equations (following Clapeyron.jl's breadth)
2. **Symbolic Capabilities**: Optional ModelingToolkit.jl-style symbolic manipulation
3. **CAPE-OPEN Compatibility**: Industry standard interfaces
4. **Neural Network Surrogates**: Deeper integration with equinox for learned unit operations
5. **Stochastic Simulation**: Differentiable Gillespie algorithm integration (following Jeong et al., 2025)

---

## 11. Conclusions

The difflow project occupies a unique position in the landscape of differentiable process simulation:

1. **First-of-its-kind**: No existing tool combines JAX-based AD with comprehensive chemical engineering capabilities and integrated technoeconomic analysis

2. **Complementary to Julia Tools**: While ProcessSimulator.jl and Clapeyron.jl offer similar AD capabilities in Julia, difflow serves the Python/ML community

3. **Research Enabler**: The framework enables novel research in:
   - Gradient-based process optimization
   - Uncertainty quantification via automatic differentiation
   - Integration of ML with rigorous process models
   - End-to-end differentiable design optimization

4. **Practical Applications**: Already demonstrated for:
   - Rare earth element processing
   - Biopharmaceutical manufacturing
   - Reactor design optimization
   - Dynamic process control

The field of differentiable process simulation is rapidly evolving, with significant contributions from both the machine learning and chemical engineering communities. difflow represents an important bridge between these fields, making modern differentiable programming accessible to chemical engineers while maintaining the rigor expected in process simulation.

---

## 12. References

See the accompanying `references.bib` file for complete bibliographic information.

### Key URLs and Project Links

#### Differentiable Simulation Frameworks
- JAX-Fluids: https://github.com/tumaer/JAXFLUIDS
- DiffTaichi: https://arxiv.org/abs/1910.00935

#### JAX Ecosystem
- Diffrax: https://github.com/patrick-kidger/diffrax
- Equinox: https://github.com/patrick-kidger/equinox
- JAX MD: https://github.com/jax-md/jax-md
- Optimistix: https://github.com/patrick-kidger/optimistix

#### Julia Ecosystem
- Clapeyron.jl: https://github.com/ClapeyronThermo/Clapeyron.jl
- ProcessSimulator.jl (JuliaCon 2024): https://pretalx.com/juliacon2024/talk/LP3XAL/
- SciMLSensitivity.jl: https://docs.sciml.ai/SciMLSensitivity/stable/
- DifferentialEquations.jl: https://diffeq.sciml.ai/

#### Process Systems Engineering
- IDAES: https://github.com/IDAES/idaes-pse
- BioSTEAM: https://github.com/BioSTEAMDevelopmentGroup/biosteam
- DWSIM: https://dwsim.org/
- Pyomo: https://www.pyomo.org/

#### Thermodynamics
- CoolProp: https://github.com/CoolProp/CoolProp
- Thermo (ChEDL): https://github.com/CalebBell/thermo

#### Educational Resources
- Physics-based Deep Learning: https://physicsbaseddeeplearning.org/
- Deep Implicit Layers Tutorial: http://implicit-layers-tutorial.org/

#### Neural ODEs and Chemical Kinetics
- jaxkineticmodel: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012733
- ChemNODE: https://www.sciencedirect.com/science/article/pii/S2666546821000677

---

*Literature review compiled December 2025 for the difflow project.*
