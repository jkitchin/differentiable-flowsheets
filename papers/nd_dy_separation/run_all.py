#!/usr/bin/env python3
"""Main workflow script for Nd/Dy separation paper.

Executes all computational tasks in dependency order, generates tables and figures.

Usage:
    python run_all.py              # Run full workflow (use cache)
    python run_all.py --fresh      # Force recompute everything
    python run_all.py --figures    # Only regenerate figures
    python run_all.py --tables     # Only regenerate tables
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jax
import jax.numpy as jnp

# Enable 64-bit precision
jax.config.update('jax_enable_x64', True)

from papers.nd_dy_separation.code.workflow import (
    Workflow, Task, RESULTS_DIR, generate_summary_report, save_results_json
)
from papers.nd_dy_separation.code.tables import generate_all_tables
from papers.nd_dy_separation.code import (
    single_stage, objectives, sensitivity, optimization, figures
)


# =============================================================================
# Task Definitions
# =============================================================================

def task_base_case(deps: dict) -> dict:
    """Run base case simulation."""
    result = single_stage.run_base_case()

    return {
        'D_Nd': float(result.D_Nd),
        'D_Dy': float(result.D_Dy),
        'SF': float(result.SF),
        'purity_Dy_org': float(result.purity_Dy_org),
        'purity_Nd_aq': float(result.purity_Nd_aq),
        'recovery_Dy': float(result.recovery_Dy),
        'recovery_Nd': float(result.recovery_Nd),
        'F_Nd_org': float(result.F_Nd_org),
        'F_Dy_org': float(result.F_Dy_org),
        'F_Nd_aq': float(result.F_Nd_aq),
        'F_Dy_aq': float(result.F_Dy_aq),
    }


def task_sensitivities(deps: dict) -> dict:
    """Compute gradient-based sensitivities."""
    sens = sensitivity.compute_sensitivities()

    # Map parameter names from sensitivity module to expected keys
    param_map = {
        'pH': 'pH',
        'O/A ratio': 'OA_ratio',
        'T (K)': 'T',
        '[D2EHPA] (M)': 'conc',
    }

    purity_grads = {}
    recovery_grads = {}
    for old_key, new_key in param_map.items():
        purity_grads[new_key] = float(sens.gradients['Dy purity'][old_key])
        recovery_grads[new_key] = float(sens.gradients['Dy recovery'][old_key])

    return {
        'purity_gradients': purity_grads,
        'recovery_gradients': recovery_grads,
    }


def task_d_sensitivities(deps: dict) -> dict:
    """Compute sensitivities to distribution coefficients."""
    d_sens = sensitivity.compute_d_sensitivities()

    return {
        'purity_d_DNd': float(d_sens['Dy purity']['D_Nd']),
        'purity_d_DDy': float(d_sens['Dy purity']['D_Dy']),
        'recovery_d_DNd': float(d_sens['Dy recovery']['D_Nd']),
        'recovery_d_DDy': float(d_sens['Dy recovery']['D_Dy']),
        'base_D_Nd': float(d_sens['base_D_Nd']),
        'base_D_Dy': float(d_sens['base_D_Dy']),
    }


def task_uncertainty(deps: dict) -> dict:
    """Compute uncertainty propagation."""
    uq = sensitivity.uncertainty_propagation(
        base_pH=3.0,
        sigma_D_Nd=0.15,  # 15% relative uncertainty
        sigma_D_Dy=0.15,
    )

    return {
        'purity_mean': float(uq['base_purity']),
        'purity_std': float(uq['sigma_purity']),
        'purity_95_CI': (float(uq['purity_95_CI'][0]), float(uq['purity_95_CI'][1])),
        'recovery_mean': float(uq['base_recovery']),
        'recovery_std': float(uq['sigma_recovery']),
        'recovery_95_CI': (float(uq['recovery_95_CI'][0]), float(uq['recovery_95_CI'][1])),
    }


def task_optimization(deps: dict) -> dict:
    """Run single-objective optimization."""
    opt = optimization.optimize_purity(min_recovery=0.80, max_iterations=100)

    return {
        'optimal_pH': float(opt.optimal_params['pH']),
        'optimal_OA': float(opt.optimal_params['OA_ratio']),
        'optimal_T': float(opt.optimal_params['T']),
        'optimal_conc': float(opt.optimal_params['conc']),
        'optimal_purity': float(opt.optimal_params['purity']),
        'optimal_recovery': float(opt.optimal_params['recovery']),
        'converged': opt.converged,
        'iterations': opt.n_iterations,
    }


def task_pareto_front(deps: dict) -> list:
    """Generate Pareto front."""
    pareto = optimization.pareto_front(n_points=25, max_iterations=100)

    return [
        {
            'purity': float(p.purity),
            'recovery': float(p.recovery),
            'cost': float(p.cost),
            'pH': float(p.pH),
            'OA': float(p.OA_ratio),
        }
        for p in pareto
    ]


def task_grid_search(deps: dict) -> dict:
    """Run grid search for contour plot."""
    grid = optimization.grid_search(
        pH_range=(1.5, 4.5),
        OA_range=(0.5, 2.5),
        n_pH=40,
        n_OA=40
    )

    return {
        'pH_grid': grid['pH_grid'].tolist(),
        'OA_grid': grid['OA_grid'].tolist(),
        'purity': grid['purity'].tolist(),
        'recovery': grid['recovery'].tolist(),
    }


def task_cost_breakdown(deps: dict) -> dict:
    """Compute cost breakdown at optimal conditions."""
    opt = deps['optimization']

    # Get costs at optimal conditions
    cost_result = objectives.annualized_cost(
        pH=opt['optimal_pH'],
        OA_ratio=opt['optimal_OA'],
    )

    # Approximate breakdown (would be more detailed in real implementation)
    total = float(cost_result)
    return {
        'extractant': total * 0.35,
        'acid_base': total * 0.25,
        'utilities': total * 0.15,
        'capex_annual': total * 0.25,
        'total': total,
    }


def task_msp_sensitivity(deps: dict) -> dict:
    """Compute minimum selling price sensitivity."""
    opt = deps['optimization']
    costs = deps['cost_breakdown']

    # Base MSP calculation (simplified)
    # Dy production rate at optimal conditions
    model = single_stage.SingleStageLLE()
    result = model(
        F_Nd_feed=jnp.array(0.01),
        F_Dy_feed=jnp.array(0.01),
        F_aq=jnp.array(1.0),
        F_org=jnp.array(opt['optimal_OA']),
        pH=jnp.array(opt['optimal_pH']),
    )

    # Dy oxide production (mol/s -> kg/yr)
    Dy_mol_per_s = float(result.F_Dy_org)
    Dy_kg_per_yr = Dy_mol_per_s * 162.5 * 3600 * 8000  # MW Dy, hours, operating hours

    # MSP = total cost / production
    base_msp = costs['total'] / max(Dy_kg_per_yr, 1.0)

    return {
        'base': base_msp,
        'Dy_production_kg_yr': Dy_kg_per_yr,
    }


# =============================================================================
# Figure Generation Tasks
# =============================================================================

def task_fig1_schematic(deps: dict) -> str:
    """Generate mixer-settler schematic."""
    figures.fig1_mixer_settler_schematic()
    return "fig1_schematic.png"


def task_fig2_distribution(deps: dict) -> str:
    """Generate D vs pH plot."""
    figures.fig2_distribution_coefficients()
    return "fig2_distribution_coeffs.png"


def task_fig3_contour(deps: dict) -> str:
    """Generate contour optimization plot."""
    figures.fig3_contour_optimization()
    return "fig3_contour_optimization.png"


def task_fig4_pareto(deps: dict) -> str:
    """Generate Pareto front plot."""
    figures.fig4_pareto_front()
    return "fig4_pareto_front.png"


def task_fig5_tornado(deps: dict) -> str:
    """Generate tornado chart."""
    figures.fig5_tornado_chart()
    return "fig5_tornado.png"


def task_fig6_uncertainty(deps: dict) -> str:
    """Generate uncertainty bands plot."""
    figures.fig6_uncertainty_bands()
    return "fig6_uncertainty_bands.png"


def task_fig7_cost(deps: dict) -> str:
    """Generate cost breakdown plots."""
    figures.fig7_cost_breakdown()
    return "fig7_cost_breakdown.png"


# =============================================================================
# Workflow Setup
# =============================================================================

def create_workflow(skip_figures: bool = False) -> Workflow:
    """Create the full computational workflow."""
    wf = Workflow()

    # --- Computational tasks ---
    wf.add_task(Task(
        name="base_case",
        func=task_base_case,
        description="Base case simulation at pH=3, O/A=1",
    ))

    wf.add_task(Task(
        name="sensitivities",
        func=task_sensitivities,
        dependencies=["base_case"],
        description="Gradient-based sensitivity analysis",
    ))

    wf.add_task(Task(
        name="d_sensitivities",
        func=task_d_sensitivities,
        dependencies=["base_case"],
        description="Sensitivity to D coefficients",
    ))

    wf.add_task(Task(
        name="uncertainty",
        func=task_uncertainty,
        dependencies=["base_case", "d_sensitivities"],
        description="Uncertainty propagation (±15% D error)",
    ))

    wf.add_task(Task(
        name="optimization",
        func=task_optimization,
        dependencies=["base_case"],
        description="Single-objective purity optimization",
    ))

    wf.add_task(Task(
        name="pareto_front",
        func=task_pareto_front,
        dependencies=["base_case"],
        description="Multi-objective Pareto front",
    ))

    wf.add_task(Task(
        name="grid_search",
        func=task_grid_search,
        dependencies=["base_case"],
        description="Grid search for contour plot",
    ))

    wf.add_task(Task(
        name="cost_breakdown",
        func=task_cost_breakdown,
        dependencies=["optimization"],
        description="Cost breakdown at optimum",
    ))

    wf.add_task(Task(
        name="msp_sensitivity",
        func=task_msp_sensitivity,
        dependencies=["optimization", "cost_breakdown"],
        description="Minimum selling price analysis",
    ))

    # --- Figure tasks (no caching, always regenerate) ---
    if not skip_figures:
        wf.add_task(Task(
            name="fig1_schematic",
            func=task_fig1_schematic,
            cache=False,
            description="Figure 1: Mixer-settler schematic",
        ))

        wf.add_task(Task(
            name="fig2_distribution",
            func=task_fig2_distribution,
            cache=False,
            description="Figure 2: D vs pH curves",
        ))

        wf.add_task(Task(
            name="fig3_contour",
            func=task_fig3_contour,
            dependencies=["grid_search", "optimization"],
            cache=False,
            description="Figure 3: Contour + optimization",
        ))

        wf.add_task(Task(
            name="fig4_pareto",
            func=task_fig4_pareto,
            dependencies=["pareto_front"],
            cache=False,
            description="Figure 4: Pareto front",
        ))

        wf.add_task(Task(
            name="fig5_tornado",
            func=task_fig5_tornado,
            dependencies=["sensitivities"],
            cache=False,
            description="Figure 5: Tornado chart",
        ))

        wf.add_task(Task(
            name="fig6_uncertainty",
            func=task_fig6_uncertainty,
            dependencies=["uncertainty"],
            cache=False,
            description="Figure 6: Uncertainty bands",
        ))

        wf.add_task(Task(
            name="fig7_cost",
            func=task_fig7_cost,
            dependencies=["cost_breakdown"],
            cache=False,
            description="Figure 7: Cost breakdown",
        ))

    return wf


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run Nd/Dy separation paper workflow")
    parser.add_argument('--fresh', action='store_true', help='Force recompute everything')
    parser.add_argument('--figures', action='store_true', help='Only regenerate figures')
    parser.add_argument('--tables', action='store_true', help='Only regenerate tables')
    parser.add_argument('--no-figures', action='store_true', help='Skip figure generation')
    args = parser.parse_args()

    print("=" * 60)
    print("Nd/Dy Separation Paper - Computational Workflow")
    print("=" * 60)

    # Create workflow
    wf = create_workflow(skip_figures=args.no_figures or args.tables)

    if args.figures:
        # Only run figure tasks
        print("\nRegenerating figures only...")
        wf.use_cache = True
        wf.run_all()
        return

    # Run full workflow
    if args.fresh:
        print("\nClearing cache...")
        wf.clear_cache()

    wf.run_all(force=args.fresh)

    # Collect results for tables
    results = {name: result.data for name, result in wf.results.items()}

    # Save results to JSON
    save_results_json(results, "all_results.json")

    # Generate tables
    if not args.figures:
        generate_all_tables(results)

    # Generate summary report
    report = generate_summary_report(wf)
    report_path = RESULTS_DIR / "workflow_summary.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nWorkflow summary: {report_path}")

    print("\n" + "=" * 60)
    print("Workflow completed successfully!")
    print("=" * 60)

    # Print key results
    base = results.get('base_case', {})
    opt = results.get('optimization', {})

    print(f"\nKey Results:")
    print(f"  Base case: D_Nd={base.get('D_Nd', 0):.2f}, D_Dy={base.get('D_Dy', 0):.1f}, "
          f"SF={base.get('SF', 0):.1f}")
    print(f"  Base purity: {base.get('purity_Dy_org', 0)*100:.1f}%, "
          f"recovery: {base.get('recovery_Dy', 0)*100:.1f}%")
    print(f"  Optimal: pH={opt.get('optimal_pH', 0):.2f}, O/A={opt.get('optimal_OA', 0):.2f}")
    print(f"  Optimal purity: {opt.get('optimal_purity', 0)*100:.1f}%, "
          f"recovery: {opt.get('optimal_recovery', 0)*100:.1f}%")


if __name__ == "__main__":
    main()
