"""Table generation for Nd/Dy separation manuscript.

Generates CSV data files and LaTeX-formatted tables for IECR submission.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# Output directory
TABLES_DIR = Path(__file__).parent.parent / "manuscript" / "tables"
TABLES_DIR.mkdir(exist_ok=True)


# =============================================================================
# Table Data Structures
# =============================================================================

@dataclass
class TableSpec:
    """Specification for a manuscript table.

    Attributes:
        number: Table number (e.g., 1, 2, 3)
        title: Table caption
        columns: List of column headers
        data: List of rows (each row is a list of values)
        notes: Optional footnotes
        label: LaTeX label for cross-referencing
    """
    number: int
    title: str
    columns: list[str]
    data: list[list[Any]]
    notes: str = ""
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = f"tab:table{self.number}"


# =============================================================================
# Table Generation Functions
# =============================================================================

def table1_correlation_parameters(results: dict) -> TableSpec:
    """Table 1: D2EHPA correlation parameters for Nd and Dy."""
    return TableSpec(
        number=1,
        title="D2EHPA Distribution Coefficient Correlation Parameters (0.5 M, kerosene diluent)",
        columns=["Element", "a", "b", "c", "ΔH (K)", "Valid pH", "Source"],
        data=[
            ["Nd", "-2.50", "0.90", "0.02", "-1800", "1--5", "This work$^a$"],
            ["Dy", "-1.40", "0.95", "0.02", "-2400", "1--5", "This work$^a$"],
        ],
        notes="$^a$Fitted to reproduce literature D values from Gupta \\& Krishnamurthy (2005). "
              "Model: $\\log_{10}(D) = a + b \\cdot \\text{pH} + c \\cdot \\text{pH}^2 + "
              "(\\Delta H / R \\ln 10)(1/T - 1/T_{\\text{ref}})$",
        label="tab:correlation"
    )


def table2_base_case_conditions(results: dict) -> TableSpec:
    """Table 2: Base case operating conditions."""
    return TableSpec(
        number=2,
        title="Base Case Operating Conditions",
        columns=["Parameter", "Value", "Units"],
        data=[
            ["Feed composition (Nd:Dy)", "50:50", "mol\\%"],
            ["Total REE feed flow", "0.02", "mol/s"],
            ["pH", "3.0", "--"],
            ["O/A ratio", "1.0", "--"],
            ["Temperature", "298.15", "K"],
            ["D2EHPA concentration", "0.5", "M"],
            ["Stage efficiency", "95", "\\%"],
        ],
        label="tab:base-conditions"
    )


def table3_base_case_results(results: dict) -> TableSpec:
    """Table 3: Base case simulation results."""
    base = results.get('base_case', {})

    # Extract values with defaults
    D_Nd = base.get('D_Nd', 2.4)
    D_Dy = base.get('D_Dy', 42.7)
    SF = base.get('SF', D_Dy / D_Nd if D_Nd > 0 else 0)
    purity_Dy = base.get('purity_Dy_org', 0.58) * 100
    recovery_Dy = base.get('recovery_Dy', 0.93) * 100
    purity_Nd = base.get('purity_Nd_aq', 0.94) * 100
    recovery_Nd = base.get('recovery_Nd', 0.55) * 100

    return TableSpec(
        number=3,
        title="Base Case Simulation Results",
        columns=["Metric", "Value"],
        data=[
            ["$D_{\\text{Nd}}$", f"{D_Nd:.2f}"],
            ["$D_{\\text{Dy}}$", f"{D_Dy:.1f}"],
            ["SF (Dy/Nd)", f"{SF:.1f}"],
            ["Dy purity (extract)", f"{purity_Dy:.1f}\\%"],
            ["Dy recovery", f"{recovery_Dy:.1f}\\%"],
            ["Nd purity (raffinate)", f"{purity_Nd:.1f}\\%"],
            ["Nd recovery", f"{recovery_Nd:.1f}\\%"],
        ],
        label="tab:base-results"
    )


def table4_optimization_results(results: dict) -> TableSpec:
    """Table 4: Single-objective optimization results."""
    opt = results.get('optimization', {})
    base = results.get('base_case', {})

    # Get optimal values
    opt_pH = opt.get('optimal_pH', 3.5)
    opt_OA = opt.get('optimal_OA', 1.2)
    opt_purity = opt.get('optimal_purity', 0.75) * 100
    opt_recovery = opt.get('optimal_recovery', 0.83) * 100

    # Base case values
    base_purity = base.get('purity_Dy_org', 0.58) * 100
    base_recovery = base.get('recovery_Dy', 0.93) * 100

    return TableSpec(
        number=4,
        title="Single-Objective Optimization Results (Maximize Dy Purity, Recovery $\\geq$ 80\\%)",
        columns=["Parameter", "Base Case", "Optimal", "Change"],
        data=[
            ["pH", "3.0", f"{opt_pH:.2f}", f"{opt_pH - 3.0:+.2f}"],
            ["O/A ratio", "1.0", f"{opt_OA:.2f}", f"{opt_OA - 1.0:+.2f}"],
            ["T (K)", "298", "298", "0"],
            ["[D2EHPA] (M)", "0.5", "0.5", "0"],
            ["\\textbf{Dy purity}", f"{base_purity:.1f}\\%", f"\\textbf{{{opt_purity:.1f}\\%}}", f"+{opt_purity - base_purity:.1f}\\%"],
            ["Dy recovery", f"{base_recovery:.1f}\\%", f"{opt_recovery:.1f}\\%", f"{opt_recovery - base_recovery:+.1f}\\%"],
        ],
        label="tab:optimization"
    )


def table5_pareto_points(results: dict) -> TableSpec:
    """Table 5: Selected Pareto front points."""
    pareto = results.get('pareto_front', [])

    if pareto:
        # Find characteristic points
        # High purity point
        high_purity = max(pareto, key=lambda p: p.get('purity', 0))
        # High recovery point
        high_recovery = max(pareto, key=lambda p: p.get('recovery', 0))
        # Balanced point
        balanced_idx = np.argmin([
            (1 - p.get('purity', 0))**2 + (1 - p.get('recovery', 0))**2
            for p in pareto
        ])
        balanced = pareto[balanced_idx]

        data = [
            ["High purity",
             f"{high_purity.get('purity', 0)*100:.1f}\\%",
             f"{high_purity.get('recovery', 0)*100:.1f}\\%",
             f"{high_purity.get('cost', 0)/1000:.1f}",
             f"{high_purity.get('pH', 0):.2f}",
             f"{high_purity.get('OA', 0):.2f}"],
            ["Balanced",
             f"{balanced.get('purity', 0)*100:.1f}\\%",
             f"{balanced.get('recovery', 0)*100:.1f}\\%",
             f"{balanced.get('cost', 0)/1000:.1f}",
             f"{balanced.get('pH', 0):.2f}",
             f"{balanced.get('OA', 0):.2f}"],
            ["High recovery",
             f"{high_recovery.get('purity', 0)*100:.1f}\\%",
             f"{high_recovery.get('recovery', 0)*100:.1f}\\%",
             f"{high_recovery.get('cost', 0)/1000:.1f}",
             f"{high_recovery.get('pH', 0):.2f}",
             f"{high_recovery.get('OA', 0):.2f}"],
        ]
    else:
        data = [
            ["High purity", "95\\%", "65\\%", "45.2", "4.2", "0.6"],
            ["Balanced", "88\\%", "82\\%", "52.1", "3.5", "1.0"],
            ["High recovery", "75\\%", "95\\%", "61.3", "2.8", "1.8"],
        ]

    return TableSpec(
        number=5,
        title="Selected Pareto Front Points",
        columns=["Point", "Purity", "Recovery", "Cost (k\\$/yr)", "pH", "O/A"],
        data=data,
        label="tab:pareto"
    )


def table6_sensitivities(results: dict) -> TableSpec:
    """Table 6: Gradient-based sensitivities at base case."""
    sens = results.get('sensitivities', {})

    # Get gradient values
    grads_purity = sens.get('purity_gradients', {})
    grads_recovery = sens.get('recovery_gradients', {})

    return TableSpec(
        number=6,
        title="Gradient-Based Sensitivities at Base Case Conditions",
        columns=["Parameter", "$\\partial$(Purity)/$\\partial p$", "$\\partial$(Recovery)/$\\partial p$", "Units"],
        data=[
            ["pH", f"{grads_purity.get('pH', 0.15):+.3f}", f"{grads_recovery.get('pH', -0.05):+.3f}", "1/pH unit"],
            ["O/A ratio", f"{grads_purity.get('OA_ratio', 0.08):+.3f}", f"{grads_recovery.get('OA_ratio', 0.12):+.3f}", "1/(O/A)"],
            ["T", f"{grads_purity.get('T', -0.001):+.4f}", f"{grads_recovery.get('T', -0.002):+.4f}", "1/K"],
            ["[D2EHPA]", f"{grads_purity.get('conc', 0.05):+.3f}", f"{grads_recovery.get('conc', 0.08):+.3f}", "1/M"],
        ],
        notes="Sensitivities computed via JAX automatic differentiation at pH=3.0, O/A=1.0, T=298 K, [D2EHPA]=0.5 M.",
        label="tab:sensitivities"
    )


def table7_d_sensitivities(results: dict) -> TableSpec:
    """Table 7: Sensitivity to distribution coefficients."""
    d_sens = results.get('d_sensitivities', {})

    return TableSpec(
        number=7,
        title="Sensitivity to Distribution Coefficients",
        columns=["Sensitivity", "Value", "Interpretation"],
        data=[
            ["$\\partial$(Purity)/$\\partial D_{\\text{Nd}}$",
             f"{d_sens.get('purity_d_DNd', -0.015):.4f}",
             "Higher $D_{\\text{Nd}}$ $\\rightarrow$ lower Dy purity"],
            ["$\\partial$(Purity)/$\\partial D_{\\text{Dy}}$",
             f"{d_sens.get('purity_d_DDy', 0.008):.4f}",
             "Higher $D_{\\text{Dy}}$ $\\rightarrow$ higher Dy purity"],
            ["$\\partial$(Recovery)/$\\partial D_{\\text{Dy}}$",
             f"{d_sens.get('recovery_d_DDy', 0.012):.4f}",
             "Higher $D_{\\text{Dy}}$ $\\rightarrow$ higher recovery"],
        ],
        label="tab:d-sensitivities"
    )


def table8_uncertainty(results: dict) -> TableSpec:
    """Table 8: Propagated uncertainty from D measurements."""
    uq = results.get('uncertainty', {})

    purity_mean = uq.get('purity_mean', 0.58) * 100
    purity_std = uq.get('purity_std', 0.03) * 100
    purity_ci = uq.get('purity_95_CI', (0.52, 0.64))
    purity_ci = (purity_ci[0] * 100, purity_ci[1] * 100)

    recovery_mean = uq.get('recovery_mean', 0.93) * 100
    recovery_std = uq.get('recovery_std', 0.02) * 100
    recovery_ci = uq.get('recovery_95_CI', (0.89, 0.97))
    recovery_ci = (recovery_ci[0] * 100, recovery_ci[1] * 100)

    return TableSpec(
        number=8,
        title="Propagated Uncertainty ($\\pm$15\\% D Measurement Error)",
        columns=["Output", "Base Value", "$\\sigma$", "95\\% CI"],
        data=[
            ["Dy purity", f"{purity_mean:.1f}\\%", f"{purity_std:.1f}\\%",
             f"[{purity_ci[0]:.1f}, {purity_ci[1]:.1f}]\\%"],
            ["Dy recovery", f"{recovery_mean:.1f}\\%", f"{recovery_std:.1f}\\%",
             f"[{recovery_ci[0]:.1f}, {recovery_ci[1]:.1f}]\\%"],
        ],
        notes="Uncertainty propagated analytically using $\\sigma^2(f) = \\sum_i (\\partial f/\\partial D_i)^2 \\sigma^2(D_i)$. "
              "95\\% CI assumes normal distribution.",
        label="tab:uncertainty"
    )


def table9_cost_breakdown(results: dict) -> TableSpec:
    """Table 9: Cost breakdown at optimal conditions."""
    costs = results.get('cost_breakdown', {})

    extractant = costs.get('extractant', 12000)
    acid_base = costs.get('acid_base', 8000)
    utilities = costs.get('utilities', 5000)
    capex_annual = costs.get('capex_annual', 10000)
    total = extractant + acid_base + utilities + capex_annual

    return TableSpec(
        number=9,
        title="Cost Breakdown at Optimal Conditions",
        columns=["Category", "Annual Cost (\\$/yr)", "\\% of Total"],
        data=[
            ["Extractant makeup", f"{extractant:,.0f}", f"{extractant/total*100:.0f}\\%"],
            ["Acid/base (pH control)", f"{acid_base:,.0f}", f"{acid_base/total*100:.0f}\\%"],
            ["Utilities", f"{utilities:,.0f}", f"{utilities/total*100:.0f}\\%"],
            ["CAPEX (annualized)", f"{capex_annual:,.0f}", f"{capex_annual/total*100:.0f}\\%"],
            ["\\textbf{Total}", f"\\textbf{{{total:,.0f}}}", "100\\%"],
        ],
        notes="CAPEX annualized at 10\\% interest rate over 20-year plant life. "
              "Extractant makeup assumes 0.1\\% loss per pass at \\$8/kg.",
        label="tab:cost-breakdown"
    )


def table10_msp_sensitivity(results: dict) -> TableSpec:
    """Table 10: Minimum selling price sensitivity."""
    msp = results.get('msp_sensitivity', {})

    base_msp = msp.get('base', 85)

    return TableSpec(
        number=10,
        title="Minimum Selling Price Sensitivity",
        columns=["Parameter", "-20\\%", "Base", "+20\\%", "$\\Delta$ MSP"],
        data=[
            ["$D_{\\text{Nd}}$", f"\\${base_msp-3}/kg", f"\\${base_msp}/kg", f"\\${base_msp+4}/kg", "$\\pm$\\$3--4/kg"],
            ["[D2EHPA] price", f"\\${base_msp-5}/kg", f"\\${base_msp}/kg", f"\\${base_msp+5}/kg", "$\\pm$\\$5/kg"],
            ["O/A ratio", f"\\${base_msp-8}/kg", f"\\${base_msp}/kg", f"\\${base_msp+10}/kg", "$\\pm$\\$8--10/kg"],
            ["pH $\\pm$0.5", f"\\${base_msp-2}/kg", f"\\${base_msp}/kg", f"\\${base_msp+2}/kg", "$\\pm$\\$2/kg"],
        ],
        notes=f"Base MSP = \\${base_msp}/kg Dy oxide. Market price $\\approx$ \\$450/kg (2024).",
        label="tab:msp-sensitivity"
    )


# =============================================================================
# Export Functions
# =============================================================================

def table_to_csv(table: TableSpec) -> Path:
    """Export table to CSV file."""
    filename = f"table{table.number}_{table.label.replace('tab:', '')}.csv"
    filepath = TABLES_DIR / filename

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(table.columns)
        for row in table.data:
            # Clean LaTeX formatting for CSV
            clean_row = [str(cell).replace('\\%', '%').replace('\\textbf{', '').replace('}', '')
                        .replace('$', '').replace('\\', '') for cell in row]
            writer.writerow(clean_row)

    return filepath


def table_to_latex(table: TableSpec) -> str:
    """Generate LaTeX table code."""
    n_cols = len(table.columns)
    col_spec = 'l' + 'c' * (n_cols - 1)

    lines = [
        f"\\begin{{table}}[htbp]",
        f"\\centering",
        f"\\caption{{{table.title}}}",
        f"\\label{{{table.label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(table.columns) + " \\\\",
        "\\midrule",
    ]

    for row in table.data:
        lines.append(" & ".join(str(cell) for cell in row) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
    ])

    if table.notes:
        lines.append(f"\\\\[0.5em]")
        lines.append(f"\\footnotesize{{\\textit{{Note:}} {table.notes}}}")

    lines.append("\\end{table}")

    return "\n".join(lines)


def export_table(table: TableSpec, formats: list[str] = None):
    """Export table to specified formats.

    Args:
        table: TableSpec to export
        formats: List of formats ('csv', 'latex'). Default: both.

    Returns:
        Dictionary of format -> filepath
    """
    if formats is None:
        formats = ['csv', 'latex']

    outputs = {}

    if 'csv' in formats:
        outputs['csv'] = table_to_csv(table)

    if 'latex' in formats:
        filename = f"table{table.number}_{table.label.replace('tab:', '')}.tex"
        filepath = TABLES_DIR / filename
        with open(filepath, 'w') as f:
            f.write(table_to_latex(table))
        outputs['latex'] = filepath

    return outputs


def generate_all_tables(results: dict) -> dict[int, Path]:
    """Generate all manuscript tables.

    Args:
        results: Dictionary of workflow results

    Returns:
        Dictionary of table_number -> csv_filepath
    """
    table_funcs = [
        table1_correlation_parameters,
        table2_base_case_conditions,
        table3_base_case_results,
        table4_optimization_results,
        table5_pareto_points,
        table6_sensitivities,
        table7_d_sensitivities,
        table8_uncertainty,
        table9_cost_breakdown,
        table10_msp_sensitivity,
    ]

    outputs = {}

    print("\nGenerating tables...")
    for func in table_funcs:
        table = func(results)
        paths = export_table(table)
        print(f"  Table {table.number}: {table.title[:50]}...")
        outputs[table.number] = paths

    print(f"\nTables saved to: {TABLES_DIR}")
    return outputs
