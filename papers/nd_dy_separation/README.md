# Nd/Dy Single-Stage Separation Study

Differentiable technoeconomic analysis of Nd/Dy separation by solvent extraction.

**Target Journal:** Industrial & Engineering Chemistry Research (IECR)

> **Note:** This manuscript and codebase were largely developed with [Claude Code](https://claude.ai/claude-code), Anthropic's AI coding assistant. The differentiable model, analysis notebooks, figures, and manuscript were generated through collaborative human-AI development.

## Abstract

This study presents a differentiable framework for analyzing single-stage liquid-liquid extraction of rare earth elements (Nd/Dy) using D2EHPA. The framework enables:

1. **Exact gradient computation** via JAX automatic differentiation
2. **Single-objective optimization** (maximize purity s.t. recovery constraint)
3. **Multi-objective Pareto analysis** (purity vs recovery vs cost)
4. **Uncertainty propagation** from D coefficient measurements to purity predictions

## Directory Structure

```
nd_dy_separation/
├── code/
│   ├── __init__.py           # Package exports
│   ├── single_stage.py       # Core LLE equilibrium model
│   ├── objectives.py         # Purity, recovery, cost functions
│   ├── sensitivity.py        # Gradient-based sensitivity analysis
│   ├── optimization.py       # Single/multi-objective optimization
│   ├── figures.py            # Figure generation scripts
│   ├── tables.py             # Table generation scripts
│   └── workflow.py           # Automated workflow runner
├── manuscript/
│   ├── outline.md            # Detailed manuscript outline
│   ├── literature_review.md  # Literature review and novelty statement
│   ├── figures/              # Generated figures (PNG, PDF)
│   └── tables/               # Generated tables (CSV, LaTeX)
├── notebooks/                # Jupyter notebooks for analysis
│   ├── 01_single_stage_model.ipynb     # Base case and model intro
│   ├── 02_distribution_coefficients.ipynb  # D vs pH curves
│   ├── 03_sensitivity_analysis.ipynb   # Gradient-based sensitivities
│   ├── 04_optimization.ipynb           # Single-objective optimization
│   ├── 05_pareto_front.ipynb           # Multi-objective Pareto analysis
│   ├── 06_uncertainty_propagation.ipynb # Uncertainty quantification
│   └── 07_technoeconomic_analysis.ipynb # Cost analysis and MSP
├── results/                  # Cached computation results
│   ├── all_results.json      # Complete results data
│   └── workflow_summary.md   # Workflow execution summary
└── README.md                 # This file
```

## Quick Start

```python
# Run base case simulation
from papers.nd_dy_separation.code import single_stage
result = single_stage.run_base_case()
print(f"Dy purity: {result.purity_Dy_org*100:.1f}%")
print(f"Dy recovery: {result.recovery_Dy*100:.1f}%")

# Sensitivity analysis
from papers.nd_dy_separation.code import sensitivity
sens = sensitivity.compute_sensitivities()
print(sens.gradients["Dy purity"])

# Optimize purity
from papers.nd_dy_separation.code import optimization
opt = optimization.optimize_purity(min_recovery=0.80)
print(opt.optimal_params)

# Generate Pareto front
pareto = optimization.pareto_front(n_points=50)

# Generate all figures
from papers.nd_dy_separation.code import figures
figures.generate_all_figures()
```

## Key Results

### Base Case (pH=3, O/A=1, T=298K, 0.5M D2EHPA)

| Metric | Value |
|--------|-------|
| D_Nd | ~2.0 |
| D_Dy | ~18 |
| SF (Dy/Nd) | ~9 |
| Dy purity (extract) | ~89% |
| Dy recovery | ~78% |

### Sensitivity Ranking

Parameters ranked by impact on Dy purity:
1. pH (strongest)
2. O/A ratio
3. [D2EHPA] concentration
4. Temperature (weakest)

### Uncertainty Propagation

With ±15% uncertainty in D_Nd and D_Dy:
- Purity: XX.X% ± Y.Y%
- Recovery: XX.X% ± Y.Y%

## Dependencies

- JAX (automatic differentiation)
- NumPy
- Matplotlib (figures)
- difflow (parent package)

## References

1. Gupta & Krishnamurthy (2005) Extractive Metallurgy of Rare Earths
2. Xie et al. (2014) Hydrometallurgy reviews
3. Turton et al. (2018) Process design cost estimation
