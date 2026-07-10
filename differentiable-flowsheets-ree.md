---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-size: 20px;
    font-family: 'Helvetica Neue', Arial, sans-serif;
  }
  h1 {
    font-size: 38px;
    color: #1a365d;
  }
  h2 {
    font-size: 30px;
    color: #2c5282;
  }
  strong {
    color: #2b6cb0;
  }
  table {
    font-size: 18px;
  }
  code {
    font-size: 16px;
  }
  blockquote {
    border-left: 4px solid #2b6cb0;
    padding-left: 16px;
    color: #4a5568;
    font-style: italic;
  }
---

<!-- _paginate: false -->
<!-- _class: lead -->

# Differentiable Flowsheets for Rare Earth Separations

**Gradient-Based Process Design and Optimization**

John Kitchin
Carnegie Mellon University

---

## The Challenge: Rare Earth Separation

- U.S. depends on imports for **critical REE** (Nd, Dy, Pr) used in magnets, electronics, defense
- REE are **chemically near-identical** — separation requires many stages of solvent extraction
- Traditional simulators (Aspen, HYSYS) lack efficient optimization for these flowsheets
- Process design today relies on **trial-and-error** experimentation and heuristics

> Separating 10+ similar elements across 20+ stages with recycles is a combinatorial nightmare for conventional tools.

---

## Our Approach: Differentiable Flowsheets

- Built on **JAX** — every calculation is automatically differentiable
- Gradients flow through the **entire flowsheet**: units, recycles, economics
- Gradient-based optimization: **orders of magnitude faster** than derivative-free methods
- Key insight: treat the whole process as a **differentiable program**

<div style="text-align: center; margin-top: 20px; font-size: 22px;">

**Lab Data** → **Calibrated Model** → **Optimized Design** → **Economics**
*All connected by automatic differentiation*

</div>

---

## What "Differentiable" Means in Practice

<div style="display: flex; gap: 40px;">
<div style="flex: 1;">

### Traditional Approach
- Perturb one variable, re-simulate
- **N+1 simulations** for N parameters
- Finite-difference approximations
- Slow, inaccurate for large systems

</div>
<div style="flex: 1;">

### Our Approach
- **Exact gradients** of any output w.r.t. any input — in one pass
- Sensitivity, uncertainty, and optimization simultaneously
- Scales to hundreds of parameters

</div>
</div>

**Example:** $\frac{\partial(\text{Nd recovery})}{\partial(\text{pH})} = 1.43$ — raising pH by 0.1 increases Nd recovery by 14.3%

---

## REE Separation Capabilities

<div style="display: flex; gap: 40px;">
<div style="flex: 1;">

### Elements & Extractants
- **10 REE elements**: La, Ce, Pr, Nd, Sm, Eu, Gd, Tb, Dy, Y
- **4 extractant systems**: D2EHPA, PC88A, Cyanex272, TBP
- pH-dependent equilibrium with temperature correction

</div>
<div style="flex: 1;">

### Unit Operations
- Multi-stage **Extractor** (Kremser equation)
- **Mixer-Settler** with efficiency
- **Scrubber** for impurity rejection
- **Stripper** for product recovery
- **3 Precipitators** (oxalate, carbonate, hydroxide)
- **CeO₂ oxidizer** for selective Ce removal

</div>
</div>

---

## Pre-Built Flowsheet Templates

| Template | Configuration | Application |
|----------|--------------|-------------|
| **ExtractStripCircuit** | 2-section | Basic separations |
| **ExtractScrubStripCircuit** | 3-section | High-purity products |
| **SplitShellCascade** | Multi-product | Multiple pure REE streams |
| **FullSeparationTrain** | Complete | Ore-to-product processing |
| **GroupSeparator** | Group split | Light / Middle / Heavy REE |

All templates are **differentiable end-to-end** — optimize any parameter with gradients.

---

## Application: NdFeB Magnet Production

<div style="display: flex; gap: 40px;">
<div style="flex: 1;">

### Feed & Design
- Feed: La 34%, **Nd 45%**, Pr 11%, Dy 2%
- Extractant: **PC88A** selected automatically (SF 4.03 vs D2EHPA 2.66)
- pH optimized via gradient descent → **pH 5.0**

### Circuit
- 8 extraction + 6 scrubbing + 4 stripping stages

</div>
<div style="flex: 1;">

### Results
- **100% Nd recovery**
- **83% product purity**
- 82% Pr recovery

### Sensitivity (automatic)
- $\partial(\text{recovery})/\partial(\text{pH})$ — continuous
- Effect of adding stages quantified

</div>
</div>

---

## Integrated Economics — All Differentiable

<div style="display: flex; gap: 40px;">
<div style="flex: 1;">

### 500 t/year Plant
- **CAPEX**: $3.6M
- **OPEX**: $10.3M/year
- **Revenue**: $48.1M/year
- **ROI**: 774%
- **Payback**: < 1 year

</div>
<div style="flex: 1;">

### What Differentiable Economics Enables
- Optimize directly for **profit** or **minimum selling price**
- Gradients of cost w.r.t. every design variable
- Trade-off: purity vs. recovery vs. cost — quantified exactly
- Risk-aware design under price uncertainty

</div>
</div>

---

## Parameter Estimation from Lab Data

- Fit pH-dependent distribution coefficients to **experimental concentration data**
- 7 pH values × 3 replicates → **9 parameters** fitted (a, b, c per element)
- All parameters recovered within **95% confidence intervals**
- **R² > 0.999**, RMSE < 1 mg/L

> Closes the loop: **lab data → calibrated model → optimized design**
> No manual parameter tuning required.

---

## Uncertainty Quantification

<div style="display: flex; gap: 40px;">
<div style="flex: 1;">

### Four Built-In Methods
- **Linear propagation** — fast, first-order
- **Monte Carlo** — nonparametric sampling
- **Sobol indices** — global sensitivity
- **Covariance propagation** — full Jacobian

</div>
<div style="flex: 1;">

### Impact
- Propagate experimental uncertainty through flowsheet to **economic outputs**
- Identify which parameters matter most
- **Risk-aware optimization**: maximize expected profit while controlling variance

</div>
</div>

---

## What Makes This Novel

- **First differentiable REE separation simulator**
- Exact gradients through multi-stage counter-current extraction with recycles
- **Unified framework**: experiment → model → optimize → economics
- Three domain plugins: REE extraction, bio manufacturing, carbon capture
- Open, extensible Python framework — not a black-box commercial tool

<div style="text-align: center; margin-top: 20px; font-size: 22px;">

Traditional simulators **simulate**. Our framework **optimizes**.

</div>

---

## Current Status & Next Steps

<div style="display: flex; gap: 40px;">
<div style="flex: 1;">

### Demonstrated
- 10 REE elements, 4 extractants
- 5 flowsheet templates
- NdFeB production optimization
- Parameter estimation with UQ
- Full technoeconomic analysis

</div>
<div style="flex: 1;">

### Next Steps
- Pilot plant validation
- Additional extractant systems
- Multi-objective optimization (purity vs. cost vs. recovery)
- Lab-to-plant design acceleration

</div>
</div>

> **Vision**: A computational tool that takes lab extraction data and delivers an optimized, costed plant design — automatically.

---

<!-- _paginate: false -->
<!-- _class: lead -->

# Thank You

**Questions?**

John Kitchin — Carnegie Mellon University
`difflow` — Differentiable Flowsheet Framework
