"""Figure generation for Nd/Dy separation paper.

Generates all figures for the IECR manuscript.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

import jax.numpy as jnp

from .single_stage import D2EHPADistribution, run_base_case
from .objectives import dy_purity, dy_recovery, annualized_cost
from .sensitivity import compute_sensitivities, compute_d_sensitivities, tornado_data, uncertainty_propagation
from .optimization import optimize_purity, pareto_front, grid_search


# Output directory
FIGURE_DIR = Path(__file__).parent.parent / "manuscript" / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
FONTSIZE = 11
plt.rcParams.update({
    'font.size': FONTSIZE,
    'axes.labelsize': FONTSIZE,
    'axes.titlesize': FONTSIZE + 1,
    'legend.fontsize': FONTSIZE - 1,
    'xtick.labelsize': FONTSIZE - 1,
    'ytick.labelsize': FONTSIZE - 1,
})


def fig1_mixer_settler_schematic():
    """Figure 1: Schematic of single-stage mixer-settler."""
    fig, ax = plt.subplots(figsize=(8, 4))

    # Mixer (left box)
    mixer = plt.Rectangle((0.1, 0.2), 0.3, 0.6, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(mixer)
    ax.text(0.25, 0.5, 'MIXER', ha='center', va='center', fontsize=12, fontweight='bold')

    # Settler (right box)
    settler = plt.Rectangle((0.5, 0.2), 0.4, 0.6, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(settler)
    ax.text(0.7, 0.5, 'SETTLER', ha='center', va='center', fontsize=12, fontweight='bold')

    # Phase separation in settler
    ax.axhline(y=0.5, xmin=0.5, xmax=0.9, color='blue', linestyle='--', alpha=0.7)
    ax.text(0.7, 0.65, 'Organic', ha='center', va='center', fontsize=10, color='orange')
    ax.text(0.7, 0.35, 'Aqueous', ha='center', va='center', fontsize=10, color='blue')

    # Arrows and labels
    # Aqueous feed
    ax.annotate('', xy=(0.1, 0.35), xytext=(-0.05, 0.35),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.text(-0.08, 0.35, 'Aqueous Feed\n(Nd, Dy)', ha='right', va='center', fontsize=9)

    # Organic feed
    ax.annotate('', xy=(0.1, 0.65), xytext=(-0.05, 0.65),
                arrowprops=dict(arrowstyle='->', color='orange', lw=2))
    ax.text(-0.08, 0.65, 'Organic Solvent\n(D2EHPA)', ha='right', va='center', fontsize=9)

    # Mixed flow
    ax.annotate('', xy=(0.5, 0.5), xytext=(0.4, 0.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Raffinate (aqueous out)
    ax.annotate('', xy=(1.05, 0.35), xytext=(0.9, 0.35),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.text(1.08, 0.35, 'Raffinate\n(Nd-rich)', ha='left', va='center', fontsize=9)

    # Extract (organic out)
    ax.annotate('', xy=(1.05, 0.65), xytext=(0.9, 0.65),
                arrowprops=dict(arrowstyle='->', color='orange', lw=2))
    ax.text(1.08, 0.65, 'Extract\n(Dy-rich)', ha='left', va='center', fontsize=9)

    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'fig1_schematic.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURE_DIR / 'fig1_schematic.pdf', bbox_inches='tight')
    plt.close()


def fig2_distribution_coefficients():
    """Figure 2: D vs pH curves for Nd and Dy."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    dist = D2EHPADistribution()
    pH_vals = np.linspace(1.0, 5.0, 100)

    D_Nd = [float(dist.D("Nd", pH, 298.15, 0.5)) for pH in pH_vals]
    D_Dy = [float(dist.D("Dy", pH, 298.15, 0.5)) for pH in pH_vals]

    ax.semilogy(pH_vals, D_Nd, 'b-', linewidth=2, label='Nd')
    ax.semilogy(pH_vals, D_Dy, 'r-', linewidth=2, label='Dy')

    # Literature data points (placeholder - would add actual literature values)
    # ax.plot([2.0, 3.0, 4.0], [0.1, 1.0, 10.0], 'bo', markersize=8, label='Nd (lit.)')
    # ax.plot([2.0, 3.0, 4.0], [1.0, 10.0, 100.0], 'ro', markersize=8, label='Dy (lit.)')

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.text(4.5, 1.2, 'D = 1', fontsize=9, color='gray')

    ax.set_xlabel('pH')
    ax.set_ylabel('Distribution Coefficient, D')
    ax.set_title('D2EHPA Extraction (0.5 M, 25°C)')
    ax.legend(loc='lower right')
    ax.set_xlim(1, 5)
    ax.set_ylim(0.01, 1000)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'fig2_distribution_coeffs.png', dpi=300)
    plt.savefig(FIGURE_DIR / 'fig2_distribution_coeffs.pdf')
    plt.close()


def fig3_contour_optimization():
    """Figure 3: Contour plot of Dy purity with optimization trajectory."""
    fig, ax = plt.subplots(figsize=(7, 5))

    # Grid search
    grid = grid_search(pH_range=(1.5, 4.5), OA_range=(0.5, 2.5), n_pH=60, n_OA=60)

    # Convert to numpy for plotting
    pH_grid = np.array(grid['pH_grid'])
    OA_grid = np.array(grid['OA_grid'])
    purity_grid = np.array(grid['purity']) * 100  # Convert to %

    # Contour plot
    levels = np.linspace(50, 95, 10)
    cs = ax.contourf(pH_grid, OA_grid, purity_grid, levels=levels, cmap='viridis')
    ax.contour(pH_grid, OA_grid, purity_grid, levels=levels, colors='white', linewidths=0.5, alpha=0.5)

    # Colorbar
    cbar = plt.colorbar(cs, ax=ax)
    cbar.set_label('Dy Purity (%)')

    # Optimization trajectory (would compute actual trajectory)
    # Placeholder: straight line from start to optimum
    opt_result = optimize_purity(min_recovery=0.80, max_iterations=50)
    ax.plot([3.0, opt_result.optimal_params['pH']],
            [1.0, opt_result.optimal_params['OA_ratio']],
            'w--', linewidth=2, label='Optimization path')
    ax.plot(3.0, 1.0, 'wo', markersize=10, markeredgecolor='black', label='Start')
    ax.plot(opt_result.optimal_params['pH'], opt_result.optimal_params['OA_ratio'],
            'w*', markersize=15, markeredgecolor='black', label='Optimum')

    ax.set_xlabel('pH')
    ax.set_ylabel('O/A Ratio')
    ax.set_title('Dy Purity Optimization (Recovery ≥ 80%)')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'fig3_contour_optimization.png', dpi=300)
    plt.savefig(FIGURE_DIR / 'fig3_contour_optimization.pdf')
    plt.close()


def fig4_pareto_front():
    """Figure 4: Pareto front of purity vs recovery."""
    fig, ax = plt.subplots(figsize=(6, 5))

    # Generate Pareto front
    pareto = pareto_front(n_points=30, max_iterations=150)

    # Extract data
    purities = [p.purity * 100 for p in pareto]
    recoveries = [p.recovery * 100 for p in pareto]
    costs = [p.cost / 1000 for p in pareto]  # k$/year

    # Scatter plot colored by cost
    sc = ax.scatter(recoveries, purities, c=costs, cmap='RdYlGn_r', s=80, edgecolors='black')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Annual Cost (k$/year)')

    # Connect points
    sorted_idx = np.argsort(recoveries)
    ax.plot([recoveries[i] for i in sorted_idx],
            [purities[i] for i in sorted_idx],
            'k--', alpha=0.5, linewidth=1)

    # Highlight key points
    # Find balanced point (closest to ideal)
    ideal_dist = [(100 - p.purity*100)**2 + (100 - p.recovery*100)**2 for p in pareto]
    balanced_idx = np.argmin(ideal_dist)
    ax.plot(recoveries[balanced_idx], purities[balanced_idx],
            'k*', markersize=15, label='Balanced')

    ax.set_xlabel('Dy Recovery (%)')
    ax.set_ylabel('Dy Purity (%)')
    ax.set_title('Pareto Front: Purity vs Recovery')
    ax.set_xlim(40, 100)
    ax.set_ylim(60, 100)
    ax.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'fig4_pareto_front.png', dpi=300)
    plt.savefig(FIGURE_DIR / 'fig4_pareto_front.pdf')
    plt.close()


def fig5_tornado_chart():
    """Figure 5: Tornado chart of parameter sensitivities."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Get sensitivity data
    base_params = {"pH": 3.0, "OA_ratio": 1.0, "T": 298.15, "conc": 0.5}
    variations = {
        "pH": (2.0, 4.0),
        "OA_ratio": (0.5, 1.5),
        "T": (283.0, 313.0),
        "conc": (0.3, 0.7),
    }

    tornado = tornado_data(dy_purity, base_params, variations)
    base_purity = tornado[0].base_output * 100

    # Prepare data
    params = [t.parameter for t in tornado]
    low_vals = [(t.low_output - t.base_output) * 100 for t in tornado]
    high_vals = [(t.high_output - t.base_output) * 100 for t in tornado]

    # Nice parameter names
    param_labels = {
        "pH": "pH (2.0 → 4.0)",
        "OA_ratio": "O/A (0.5 → 1.5)",
        "T": "T (283 → 313 K)",
        "conc": "[D2EHPA] (0.3 → 0.7 M)",
    }

    y_pos = np.arange(len(params))

    # Plot bars
    for i, (param, low, high) in enumerate(zip(params, low_vals, high_vals)):
        ax.barh(i, high, left=0, height=0.6, color='green', alpha=0.7, label='High' if i == 0 else '')
        ax.barh(i, low, left=0, height=0.6, color='red', alpha=0.7, label='Low' if i == 0 else '')

    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([param_labels.get(p, p) for p in params])
    ax.set_xlabel('Change in Dy Purity (%)')
    ax.set_title(f'Sensitivity Analysis (Base: {base_purity:.1f}% purity)')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'fig5_tornado.png', dpi=300)
    plt.savefig(FIGURE_DIR / 'fig5_tornado.pdf')
    plt.close()


def fig6_uncertainty_bands():
    """Figure 6: Purity prediction with D uncertainty bands."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Vary pH and show uncertainty bands
    pH_vals = np.linspace(1.5, 4.5, 50)

    purities = []
    upper_bounds = []
    lower_bounds = []

    for pH in pH_vals:
        uq = uncertainty_propagation(base_pH=pH, sigma_D_Nd=0.15, sigma_D_Dy=0.15)
        purities.append(uq['base_purity'] * 100)
        upper_bounds.append(uq['purity_95_CI'][1] * 100)
        lower_bounds.append(uq['purity_95_CI'][0] * 100)

    ax.plot(pH_vals, purities, 'b-', linewidth=2, label='Predicted purity')
    ax.fill_between(pH_vals, lower_bounds, upper_bounds, alpha=0.3, color='blue',
                    label='95% CI (±15% D uncertainty)')

    ax.set_xlabel('pH')
    ax.set_ylabel('Dy Purity (%)')
    ax.set_title('Purity Prediction with D Coefficient Uncertainty')
    ax.legend(loc='lower right')
    ax.set_xlim(1.5, 4.5)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'fig6_uncertainty_bands.png', dpi=300)
    plt.savefig(FIGURE_DIR / 'fig6_uncertainty_bands.pdf')
    plt.close()


def fig7_cost_breakdown():
    """Figure 7: Cost breakdown pie chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Pie chart of cost components (placeholder values)
    categories = ['Extractant', 'Acid/Base', 'Utilities', 'CAPEX\n(annualized)']
    costs = [35, 25, 15, 25]  # Percentages
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

    ax1.pie(costs, labels=categories, colors=colors, autopct='%1.0f%%', startangle=90)
    ax1.set_title('Operating Cost Breakdown')

    # MSP sensitivity bar chart
    params = ['D_Nd ±15%', '[D2EHPA] ±20%', 'O/A ±30%', 'pH ±0.5']
    msp_changes = [5, 8, 12, 3]  # Placeholder $/kg changes

    y_pos = np.arange(len(params))
    ax2.barh(y_pos, msp_changes, color='steelblue', alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(params)
    ax2.set_xlabel('MSP Change ($/kg Dy)')
    ax2.set_title('Minimum Selling Price Sensitivity')

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'fig7_cost_breakdown.png', dpi=300)
    plt.savefig(FIGURE_DIR / 'fig7_cost_breakdown.pdf')
    plt.close()


def generate_all_figures():
    """Generate all figures for the manuscript."""
    print("Generating figures...")

    print("  Figure 1: Mixer-settler schematic")
    fig1_mixer_settler_schematic()

    print("  Figure 2: Distribution coefficients")
    fig2_distribution_coefficients()

    print("  Figure 3: Contour + optimization")
    fig3_contour_optimization()

    print("  Figure 4: Pareto front")
    fig4_pareto_front()

    print("  Figure 5: Tornado chart")
    fig5_tornado_chart()

    print("  Figure 6: Uncertainty bands")
    fig6_uncertainty_bands()

    print("  Figure 7: Cost breakdown")
    fig7_cost_breakdown()

    print(f"\nAll figures saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    generate_all_figures()
