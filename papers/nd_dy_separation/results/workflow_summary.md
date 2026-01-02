# Workflow Execution Summary
Generated: 2026-01-02T18:44:40.537635

## Task Results

- **base_case**: cached
  - Base case simulation at pH=3, O/A=1
- **d_sensitivities**: cached
  - Sensitivity to D coefficients
  - Depends on: base_case
- **fig1_schematic**: 0.85s
  - Figure 1: Mixer-settler schematic
- **fig2_distribution**: 1.38s
  - Figure 2: D vs pH curves
- **grid_search**: cached
  - Grid search for contour plot
  - Depends on: base_case
- **optimization**: cached
  - Single-objective purity optimization
  - Depends on: base_case
- **cost_breakdown**: cached
  - Cost breakdown at optimum
  - Depends on: optimization
- **fig3_contour**: 1.68s
  - Figure 3: Contour + optimization
  - Depends on: grid_search, optimization
- **fig7_cost**: 0.61s
  - Figure 7: Cost breakdown
  - Depends on: cost_breakdown
- **msp_sensitivity**: cached
  - Minimum selling price analysis
  - Depends on: optimization, cost_breakdown
- **pareto_front**: cached
  - Multi-objective Pareto front
  - Depends on: base_case
- **fig4_pareto**: 6.33s
  - Figure 4: Pareto front
  - Depends on: pareto_front
- **sensitivities**: cached
  - Gradient-based sensitivity analysis
  - Depends on: base_case
- **fig5_tornado**: 0.41s
  - Figure 5: Tornado chart
  - Depends on: sensitivities
- **uncertainty**: cached
  - Uncertainty propagation (±15% D error)
  - Depends on: base_case, d_sensitivities
- **fig6_uncertainty**: 1.61s
  - Figure 6: Uncertainty bands
  - Depends on: uncertainty

## Output Files

- Table: `table10_msp-sensitivity.csv`
- Table: `table1_correlation.csv`
- Table: `table2_base-conditions.csv`
- Table: `table3_base-results.csv`
- Table: `table4_optimization.csv`
- Table: `table5_pareto.csv`
- Table: `table6_sensitivities.csv`
- Table: `table7_d-sensitivities.csv`
- Table: `table8_uncertainty.csv`
- Table: `table9_cost-breakdown.csv`
- Figure: `fig1_schematic.png`
- Figure: `fig2_distribution_coeffs.png`
- Figure: `fig3_contour_optimization.png`
- Figure: `fig4_pareto_front.png`
- Figure: `fig5_tornado.png`
- Figure: `fig6_uncertainty_bands.png`
- Figure: `fig7_cost_breakdown.png`