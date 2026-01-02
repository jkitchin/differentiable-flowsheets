# Literature Review and Novelty Statement

## Differentiable Technoeconomic Analysis of Single-Stage Nd/Dy Separation

This document summarizes the literature landscape and articulates the novel contributions of this work for the IECR manuscript.

---

## 1. Literature Landscape

### 1.1 REE Separation by Solvent Extraction

Rare earth element separation by liquid-liquid extraction is the dominant industrial method, with extensive literature on:

- **Extractant chemistry:** D2EHPA, PC88A, Cyanex 272, TBP
- **Distribution coefficient correlations:** pH-dependent models from Gupta & Krishnamurthy (2005)
- **Flowsheet design:** Multi-stage counter-current cascades

**Key References:**
1. Gupta, C. K.; Krishnamurthy, N. *Extractive Metallurgy of Rare Earths*; CRC Press, 2005.
2. Xie, F.; Zhang, T. A.; Dreisinger, D.; Doyle, F. A Critical Review on Solvent Extraction of Rare Earths from Aqueous Solutions. *Miner. Eng.* 2014, 56, 10–28.
3. Comprehensive review on SX technologies: https://www.sciencedirect.com/science/article/abs/pii/S1226086X2400282X

### 1.2 Current Optimization Approaches

| Approach | Description | Limitations |
|----------|-------------|-------------|
| **Response Surface Methodology (RSM)** | Fit quadratic model to experimental data, optimize within local region | Requires many experiments; local approximation only |
| **Trial-and-error** | Systematic variation of conditions | Low-throughput; inefficient |
| **Finite difference sensitivity** | Numerical perturbation of each parameter | Numerical errors (~10⁻⁶); expensive for many parameters |
| **McCabe-Thiele graphical** | Stage-by-stage construction | No sensitivity information; manual |
| **Neural network prediction** | ML model for D coefficient prediction | Black-box; no mechanistic insight |

**Representative Works:**
- RSM for REE extraction: Box-Behnken designs common in Hydrometallurgy journals
- Multi-stage SX simulation: https://www.mdpi.com/2075-163X/13/6/714
- ML for D prediction: https://pubs.acs.org/doi/10.1021/jacsau.2c00122 (R² = 0.85)

### 1.3 Automatic Differentiation in Engineering

AD has been applied to:
- **Computational fluid dynamics:** JAX-Fluids (https://www.sciencedirect.com/science/article/abs/pii/S0010465522002466)
- **Quantum chemistry:** Variational Hartree-Fock (https://pmc.ncbi.nlm.nih.gov/articles/PMC5968443/)
- **Global optimization:** Convex relaxations (https://www.sciencedirect.com/science/article/pii/S2772508123000157)
- **Machine learning force fields:** https://pubs.acs.org/doi/10.1021/acs.jpclett.2c02632

**Gap Identified:** No prior work applies automatic differentiation to rare earth solvent extraction or liquid-liquid equilibrium modeling.

### 1.4 Equation-Oriented Frameworks: IDAES and Pyomo

The [IDAES (Institute for Design of Advanced Energy Systems)](https://idaes.org/) framework, built on [Pyomo](https://www.pyomo.org/), represents the state-of-the-art in equation-oriented process modeling and optimization.

**Key Capabilities:**
- Algebraic modeling language with NLP/MINLP solver interfaces (IPOPT, etc.)
- Extensive unit operation library (flash, distillation, reactors, heat exchangers)
- Built-in support for steady-state and dynamic optimization
- Uncertainty quantification via stochastic programming

**Reference:** Lee, A., et al. "The IDAES process modeling framework and model library—Flexibility for process simulation and optimization." *J. Adv. Manuf. Process.* **2021**, 3, e10095. https://aiche.onlinelibrary.wiley.com/doi/10.1002/amp2.10095

**How IDAES/Pyomo provides "automatic differentiation":**
- Symbolic/algebraic differentiation of model equations
- Solver (e.g., IPOPT) computes gradients via exact symbolic derivatives or finite differences
- Gradient information used internally by NLP solver

**Comparison: JAX vs. IDAES/Pyomo**

| Aspect | JAX (This Work) | IDAES/Pyomo |
|--------|-----------------|-------------|
| **AD Paradigm** | Operator overloading + tracing | Equation-oriented / symbolic |
| **Model Representation** | Python functions | Algebraic constraints |
| **Gradient Access** | Direct via `jax.grad()` | Internal to solver |
| **Hardware Acceleration** | GPU/TPU native | CPU only (solver-dependent) |
| **Higher-Order Derivatives** | Hessian via `jax.hessian()` | Limited / expensive |
| **UQ Approach** | Gradient propagation | Stochastic programming |
| **Unit Operation Library** | Limited (build as needed) | Extensive |
| **Solver Interface** | Custom (gradient descent) | IPOPT, CBC, Gurobi, etc. |

**Why JAX for this study (not IDAES)?**

1. **Direct gradient access:** JAX exposes gradients as first-class objects for sensitivity analysis and UQ. In IDAES, gradients are internal to the solver.

2. **Hardware acceleration:** JAX vectorizes over parameter sweeps on GPU. IDAES is CPU-bound.

3. **Composability with ML:** JAX integrates seamlessly with neural networks, surrogate models. IDAES has limited ML integration.

4. **REE-specific models:** IDAES does not include REE solvent extraction units. We would need to build custom models anyway.

5. **Educational clarity:** JAX's functional style makes gradient computation explicit and pedagogically clear.

**When IDAES/Pyomo is better:**

- Large-scale flowsheet optimization with integer decisions (MINLP)
- Leveraging extensive thermodynamic libraries (IDAES-Experiment)
- Problems requiring global optimization guarantees
- Teams already using Pyomo/IDAES infrastructure

**Manuscript Acknowledgment (suggested text):**
> "We note that equation-oriented frameworks such as IDAES/Pyomo [Lee et al., 2021] provide automatic differentiation through algebraic modeling and NLP solvers. Our JAX-based approach differs in providing direct programmatic access to gradients, enabling explicit sensitivity analysis, GPU acceleration, and seamless integration with machine learning workflows. The approaches are complementary: IDAES excels at large-scale flowsheet optimization, while our framework prioritizes gradient-based analysis and uncertainty quantification."

### 1.5 Uncertainty Quantification in REE Separation

- Distribution coefficient measurements typically have ±15-20% uncertainty
- Current practice: Monte Carlo simulation (10,000+ samples)
- IDAES supports stochastic programming but requires reformulation
- No prior work on analytical gradient-based UQ for REE LLE

---

## 2. Novel Contributions

### Contribution 1: First Differentiable REE LLE Framework

**Claim:** "To our knowledge, this is the first application of automatic differentiation to rare earth liquid-liquid extraction modeling."

**Evidence:**
- Literature search found no prior work combining AD with REE separation
- JAX-Fluids (CFD), quantum chemistry AD papers exist, but not for LLE
- Enables capabilities not available in conventional simulation tools

**Significance:**
- Exact gradients in single backward pass
- Enables efficient optimization, sensitivity analysis, uncertainty quantification
- Foundation for differentiable multi-stage cascade design

### Contribution 2: Exact Gradients vs. Finite Differences

**Claim:** "Automatic differentiation provides gradients accurate to machine precision in a single backward pass."

| Metric | Automatic Differentiation | Finite Differences |
|--------|--------------------------|-------------------|
| Accuracy | Machine precision (~10⁻¹⁵) | ~10⁻⁶ to 10⁻⁸ |
| Cost for n parameters | O(1) backward pass | O(n) function evaluations |
| Higher derivatives | Hessian readily available | Numerically unstable |
| Implementation | Automatic via JAX | Manual perturbation |

**Significance:**
- More accurate sensitivity analysis
- Faster optimization convergence
- Enables second-order methods (Newton, quasi-Newton)

### Contribution 3: Analytical Uncertainty Propagation

**Claim:** "Gradient-based uncertainty propagation provides instant quantification of how distribution coefficient measurement uncertainty affects product purity predictions."

**Method:**
```
σ²(Purity) = (∂Purity/∂D_Nd)² × σ²(D_Nd) + (∂Purity/∂D_Dy)² × σ²(D_Dy)
```

**Comparison to Monte Carlo:**
| Method | Samples | Time | Accuracy |
|--------|---------|------|----------|
| Monte Carlo | 10,000+ | Minutes | Statistical |
| Gradient propagation | 1 | Milliseconds | Analytical (linear approx.) |

**Significance:**
- Instant uncertainty quantification during optimization
- Identifies which D coefficient uncertainties matter most
- Enables robust design accounting for measurement variability

### Contribution 4: Integrated Technoeconomic Sensitivity

**Claim:** "The differentiable framework enables direct computation of economic sensitivities, revealing which parameters most impact minimum selling price."

**Enabled Analyses:**
- ∂(MSP)/∂(pH): How pH control precision affects economics
- ∂(MSP)/∂(D_Nd): How D measurement error propagates to cost
- ∂(Profit)/∂(O/A): Marginal value of solvent ratio changes

**Significance:**
- Direct connection between operating conditions and economics
- Guides experimental priorities (measure D more precisely vs. control pH better)
- Enables gradient-based technoeconomic optimization

### Contribution 5: Multi-Objective Pareto with Exact Gradients

**Claim:** "Weighted-sum scalarization with exact gradients enables efficient exploration of the purity-recovery-cost Pareto front."

**Method:**
- Vary weights between purity, recovery, cost objectives
- Use gradient descent with exact gradients for each weight combination
- Efficient generation of Pareto-optimal points

**Significance:**
- Faster than derivative-free multi-objective methods
- Reveals trade-off structure between competing objectives
- Supports decision-making for process design

---

## 3. Positioning Against Specific Literature

### vs. MDPI Simulation of SX Circuits (2023)
**Reference:** https://www.mdpi.com/2075-163X/13/6/714

| Their Work | This Work |
|------------|-----------|
| Equilibrium modeling | Equilibrium modeling |
| McCabe-Thiele analysis | Kremser equation + AD |
| No sensitivity analysis | Exact gradient-based sensitivities |
| No uncertainty quantification | Analytical UQ via gradients |

**Advance:** Adds differentiability, enabling optimization and UQ not possible with conventional simulation.

### vs. JACS Au Machine Learning for D Prediction (2022)
**Reference:** https://pubs.acs.org/doi/10.1021/jacsau.2c00122

| Their Work | This Work |
|------------|-----------|
| Neural network D prediction | Mechanistic D correlations |
| Black-box model | Interpretable physics-based model |
| No process optimization | Full process optimization |
| R² = 0.85 on D | Literature-validated D correlations |

**Advance:** Mechanistic model provides physical insight; differentiability enables end-to-end process optimization.

### vs. ScienceDirect Comprehensive SX Review (2024)
**Reference:** https://www.sciencedirect.com/science/article/abs/pii/S1226086X2400282X

| Their Work | This Work |
|------------|-----------|
| Review of extractants, conditions | Focus on D2EHPA for Nd/Dy |
| Qualitative comparisons | Quantitative optimization framework |
| No optimization methodology | Gradient-based optimization |

**Advance:** Provides computational framework rather than experimental review.

### vs. Nonaqueous SX for Nd/Dy (IECR 2021)
**Reference:** https://pubs.acs.org/doi/10.1021/acs.iecr.1c02287

| Their Work | This Work |
|------------|-----------|
| Novel nonaqueous solvents | Conventional aqueous/D2EHPA |
| Experimental focus | Computational/modeling focus |
| SF up to 69 in PEG 200 | SF ~18 in aqueous (standard) |

**Advance:** Demonstrates optimization framework applicable to any solvent system; complements experimental discovery.

---

## 4. Suggested Text for Manuscript

### Abstract Statement
> "We present a fully differentiable model for Nd/Dy separation by solvent extraction, enabling exact gradient computation via automatic differentiation. This approach provides machine-precision sensitivities, efficient multi-objective optimization, and analytical uncertainty propagation—capabilities not available in conventional simulation tools."

### Introduction Gap Statement
> "While machine learning has been applied to predict distribution coefficients [JACS Au, 2022], and simulation tools exist for flowsheet design [MDPI, 2023], no prior work combines mechanistic LLE modeling with automatic differentiation. This gap limits the ability to (1) efficiently optimize operating conditions, (2) quantify how measurement uncertainties propagate to product quality, and (3) compute economic sensitivities for process design."

### Contribution Statement
> "The contributions of this work are:
> 1. Development of a fully differentiable single-stage LLE model using JAX automatic differentiation
> 2. Demonstration of exact gradient-based sensitivity analysis for REE separation
> 3. Analytical uncertainty propagation from D coefficient measurements to purity predictions
> 4. Multi-objective Pareto optimization of the purity-recovery-cost trade-off
> 5. Integrated technoeconomic analysis with gradient-based MSP sensitivity"

### Conclusions Statement
> "The differentiable framework demonstrated here for single-stage Nd/Dy separation generalizes to multi-stage cascades and other separation systems. By making process models differentiable 'by design,' chemical engineers gain access to the same gradient-based optimization and uncertainty quantification tools that have transformed machine learning."

---

## 5. Recommended Additional Analyses

To further strengthen the novelty claims, consider adding:

### 5.1 Gradient Accuracy Comparison
Compare AD gradients to finite difference approximations:
```python
# Finite difference
grad_fd = (f(x + h) - f(x - h)) / (2*h)
# Automatic differentiation
grad_ad = jax.grad(f)(x)
# Compare: |grad_fd - grad_ad| / |grad_ad|
```

### 5.2 Computational Time Scaling
Benchmark gradient computation time vs. number of parameters:
- AD: Nearly constant (one backward pass)
- FD: Linear scaling with parameters

### 5.3 Monte Carlo vs. Analytical UQ
Compare uncertainty estimates:
- Run 10,000 MC samples with uncertain D values
- Compare to analytical propagation formula
- Should match for small uncertainties (linear regime)

### 5.4 Hessian Analysis
Compute Hessian at optimum to characterize:
- Curvature (sharp vs. flat optimum)
- Condition number (sensitivity to perturbations)
- Principal directions of sensitivity

---

## 6. References

1. Gupta, C. K.; Krishnamurthy, N. *Extractive Metallurgy of Rare Earths*; CRC Press: Boca Raton, FL, 2005.

2. Xie, F.; Zhang, T. A.; Dreisinger, D.; Doyle, F. A Critical Review on Solvent Extraction of Rare Earths from Aqueous Solutions. *Miner. Eng.* **2014**, 56, 10–28. https://doi.org/10.1016/j.mineng.2013.10.021

3. Florez, D. H. A.; et al. Simulation of Solvent Extraction Circuits for the Separation of Rare Earth Elements. *Minerals* **2023**, 13, 714. https://www.mdpi.com/2075-163X/13/6/714

4. Gensch, T.; et al. Advancing Rare-Earth Separation by Machine Learning. *JACS Au* **2022**, 2, 1615–1623. https://pubs.acs.org/doi/10.1021/jacsau.2c00122

5. Bezanson, D.; et al. JAX-Fluids: A fully-differentiable high-order computational fluid dynamics solver. *Comput. Phys. Commun.* **2023**, 282, 108527. https://www.sciencedirect.com/science/article/abs/pii/S0010465522002466

6. Tamayo-Mendoza, T.; et al. Automatic Differentiation in Quantum Chemistry with Applications to Fully Variational Hartree–Fock. *ACS Cent. Sci.* **2018**, 4, 559–566. https://pmc.ncbi.nlm.nih.gov/articles/PMC5968443/

7. Khan, K. A.; Experiment, C. Automatic differentiation rules for Tsoukalas–Mitsos convex relaxations in global process optimization. *Optim. Methods Softw.* **2023**. https://www.sciencedirect.com/science/article/pii/S2772508123000157

8. Turton, R.; Shaeiwitz, J. A.; Bhattacharyya, D.; Whiting, W. B. *Analysis, Synthesis, and Design of Chemical Processes*, 5th ed.; Prentice Hall: Boston, 2018.

9. Bradbury, J.; et al. JAX: Composable Transformations of Python+NumPy Programs. 2018. http://github.com/google/jax

10. Lee, A.; Ghouse, J. H.; Eslick, J. C.; et al. The IDAES process modeling framework and model library—Flexibility for process simulation and optimization. *J. Adv. Manuf. Process.* **2021**, 3, e10095. https://aiche.onlinelibrary.wiley.com/doi/10.1002/amp2.10095

11. Hart, W. E.; Laird, C. D.; Watson, J.-P.; et al. *Pyomo—Optimization Modeling in Python*, 2nd ed.; Springer: Cham, 2017.
