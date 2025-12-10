"""Rare Earth Extraction Example.

This example demonstrates liquid-liquid extraction for separating
rare earth elements (REE) using the difflow framework.

Rare earth separation is a critical industrial process for producing
materials used in magnets, electronics, and clean energy technologies.
The most common method is solvent extraction using organophosphorus
extractants like D2EHPA or PC88A.

This example models separation of:
- La (Lanthanum) - light REE
- Nd (Neodymium) - light REE, used in magnets
- Dy (Dysprosium) - heavy REE, used in magnets

The distribution coefficients increase with atomic number, allowing
separation of heavy REE from light REE.
"""

import jax
import jax.numpy as jnp
from jax import grad, jacfwd

from difflow.streams import make_stream, get_flows
from difflow.units.lle import (
    MultistageCascade,
    CascadeParams,
    DifferentialContactor,
    ContactorParams,
    LLEEquilibrium,
    DistributionCoeffs,
    separation_factor,
    minimum_solvent_ratio,
    stages_for_recovery,
)

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


def main():
    print("=" * 70)
    print("RARE EARTH EXTRACTION - Differentiable Simulation")
    print("=" * 70)

    # =========================================================================
    # Problem Setup
    # =========================================================================
    print("\n1. Problem Setup")
    print("-" * 50)

    # Species definition
    # - H2O: Aqueous phase carrier
    # - Organic: Organic phase carrier (kerosene + extractant)
    # - La, Nd, Dy: Rare earth elements (as dissolved ions)

    # Distribution coefficients for D2EHPA extraction
    # These are typical values at pH ~3, and increase with atomic number
    # K = [RE]_org / [RE]_aq (concentration basis)
    K_La = 0.5   # Light REE, low extraction
    K_Nd = 2.0   # Medium, primary target
    K_Dy = 8.0   # Heavy REE, high extraction

    # Temperature dependence (extraction is typically exothermic)
    # dH < 0 means K decreases with temperature
    dH_La = -15000.0  # J/mol
    dH_Nd = -18000.0
    dH_Dy = -22000.0

    dist_coeffs = DistributionCoeffs(
        species=("La", "Nd", "Dy"),
        K0=(K_La, K_Nd, K_Dy),
        dH=(dH_La, dH_Nd, dH_Dy),
        Tref=298.15,
    )

    # LLE equilibrium calculator
    lle_eq = LLEEquilibrium(
        solutes=["La", "Nd", "Dy"],
        aqueous_carrier="H2O",
        organic_carrier="Organic",
        K_coeffs=dist_coeffs,
        activity_model="K",
    )

    # Print separation factors
    print(f"Distribution coefficients at 25°C:")
    print(f"  K_La = {K_La:.2f}")
    print(f"  K_Nd = {K_Nd:.2f}")
    print(f"  K_Dy = {K_Dy:.2f}")
    print(f"\nSeparation factors:")
    print(f"  SF(Nd/La) = {separation_factor(K_Nd, K_La):.2f}")
    print(f"  SF(Dy/Nd) = {separation_factor(K_Dy, K_Nd):.2f}")
    print(f"  SF(Dy/La) = {separation_factor(K_Dy, K_La):.2f}")

    # =========================================================================
    # Feed Streams
    # =========================================================================
    print("\n2. Feed Streams")
    print("-" * 50)

    # Aqueous feed: REE leach solution
    # Typical concentrations in g/L, converted to mol/s assuming 1 L/s
    # MW: La=138.9, Nd=144.2, Dy=162.5
    feed = make_stream(
        flows={
            "H2O": 55.5,    # ~1 L/s of water (55.5 mol)
            "La": 0.01,     # ~1.4 g/L
            "Nd": 0.02,     # ~2.9 g/L (main target)
            "Dy": 0.005,    # ~0.8 g/L
        },
        T=298.15,
        P=101325.0,
    )

    # Organic solvent: D2EHPA in kerosene
    # Assume "Organic" represents the total organic phase
    solvent = make_stream(
        flows={
            "Organic": 10.0,  # mol/s of organic phase
            "La": 0.0,
            "Nd": 0.0,
            "Dy": 0.0,
        },
        T=298.15,
        P=101325.0,
    )

    feed_flows = get_flows(feed)
    print(f"Aqueous feed (mol/s):")
    print(f"  H2O: {feed_flows['H2O']:.2f}")
    print(f"  La:  {feed_flows['La']:.4f}")
    print(f"  Nd:  {feed_flows['Nd']:.4f}")
    print(f"  Dy:  {feed_flows['Dy']:.4f}")

    solvent_flows = get_flows(solvent)
    print(f"\nOrganic solvent (mol/s):")
    print(f"  Organic: {solvent_flows['Organic']:.2f}")

    # =========================================================================
    # Multi-Stage Cascade Extraction
    # =========================================================================
    print("\n3. Multi-Stage Cascade Extraction")
    print("-" * 50)

    cascade_params = CascadeParams(
        n_stages=5,
        equilibrium=lle_eq,
        flow_config="counter_current",
    )
    cascade = MultistageCascade(cascade_params)

    raffinate, extract, info = cascade(feed, solvent, T=298.15)

    raff_flows = get_flows(raffinate)
    ext_flows = get_flows(extract)

    print(f"Counter-current cascade with {cascade_params.n_stages} stages:")
    print(f"\nRaffinate (aqueous outlet, mol/s):")
    print(f"  La: {float(raff_flows['La']):.6f}")
    print(f"  Nd: {float(raff_flows['Nd']):.6f}")
    print(f"  Dy: {float(raff_flows['Dy']):.6f}")

    print(f"\nExtract (organic outlet, mol/s):")
    print(f"  La: {float(ext_flows['La']):.6f}")
    print(f"  Nd: {float(ext_flows['Nd']):.6f}")
    print(f"  Dy: {float(ext_flows['Dy']):.6f}")

    # Calculate recoveries
    rec_La = float(ext_flows['La']) / feed_flows['La'] * 100
    rec_Nd = float(ext_flows['Nd']) / feed_flows['Nd'] * 100
    rec_Dy = float(ext_flows['Dy']) / feed_flows['Dy'] * 100

    print(f"\nRecoveries to extract:")
    print(f"  La: {rec_La:.1f}%")
    print(f"  Nd: {rec_Nd:.1f}%")
    print(f"  Dy: {rec_Dy:.1f}%")

    # =========================================================================
    # Differential Contactor (Packed Column)
    # =========================================================================
    print("\n4. Differential Contactor (Packed Column)")
    print("-" * 50)

    contactor_params = ContactorParams(
        length=3.0,  # 3 m column
        area=0.1,    # 0.1 m² cross-section
        equilibrium=lle_eq,
        n_segments=50,
        flow_config="counter_current",
        mass_transfer_model="equilibrium",
        HETP=0.5,  # 0.5 m per theoretical stage
    )
    contactor = DifferentialContactor(contactor_params)

    raff_cont, ext_cont, info_cont = contactor(feed, solvent, T=298.15)

    raff_cont_flows = get_flows(raff_cont)
    ext_cont_flows = get_flows(ext_cont)

    print(f"Packed column: L={contactor_params.length}m, HETP={contactor_params.HETP}m")
    print(f"Equivalent stages: {info_cont['n_stages_equivalent']:.1f}")

    print(f"\nRecoveries to extract:")
    rec_La_c = float(ext_cont_flows['La']) / feed_flows['La'] * 100
    rec_Nd_c = float(ext_cont_flows['Nd']) / feed_flows['Nd'] * 100
    rec_Dy_c = float(ext_cont_flows['Dy']) / feed_flows['Dy'] * 100
    print(f"  La: {rec_La_c:.1f}%")
    print(f"  Nd: {rec_Nd_c:.1f}%")
    print(f"  Dy: {rec_Dy_c:.1f}%")

    # =========================================================================
    # Sensitivity Analysis using Automatic Differentiation
    # =========================================================================
    print("\n5. Sensitivity Analysis (Automatic Differentiation)")
    print("-" * 50)

    # Define function that returns Nd recovery as function of parameters
    def nd_recovery(n_stages: float, S_F_ratio: float, T: float) -> float:
        """Calculate Nd recovery to extract."""
        # Create solvent stream with adjusted flow
        solvent_adj = make_stream(
            flows={
                "Organic": 10.0 * S_F_ratio,  # Adjust solvent flow
                "La": 0.0,
                "Nd": 0.0,
                "Dy": 0.0,
            },
            T=T,
            P=101325.0,
        )

        params = CascadeParams(
            n_stages=n_stages,
            equilibrium=lle_eq,
            flow_config="counter_current",
        )
        cascade_fn = MultistageCascade(params)

        _, extract, _ = cascade_fn(feed, solvent_adj, T=T)
        ext_flows = get_flows(extract)

        return ext_flows['Nd'] / feed_flows['Nd']

    # Compute gradients
    n_stages_val = 5.0
    SF_ratio_val = 1.0
    T_val = 298.15

    # Gradient w.r.t. number of stages
    d_recovery_d_stages = grad(nd_recovery, argnums=0)(n_stages_val, SF_ratio_val, T_val)
    print(f"∂(Nd recovery)/∂(n_stages) = {float(d_recovery_d_stages):.4f}")
    print(f"  → Adding 1 stage increases recovery by {float(d_recovery_d_stages)*100:.2f}%")

    # Gradient w.r.t. S/F ratio
    d_recovery_d_SF = grad(nd_recovery, argnums=1)(n_stages_val, SF_ratio_val, T_val)
    print(f"\n∂(Nd recovery)/∂(S/F ratio) = {float(d_recovery_d_SF):.4f}")
    print(f"  → 10% more solvent increases recovery by {float(d_recovery_d_SF)*0.1*100:.2f}%")

    # Gradient w.r.t. temperature
    d_recovery_d_T = grad(nd_recovery, argnums=2)(n_stages_val, SF_ratio_val, T_val)
    print(f"\n∂(Nd recovery)/∂T = {float(d_recovery_d_T):.6f} K⁻¹")
    print(f"  → 10K increase changes recovery by {float(d_recovery_d_T)*10*100:.2f}%")

    # =========================================================================
    # Optimization: Maximize Nd Purity in Extract
    # =========================================================================
    print("\n6. Optimization: Maximize Nd Purity in Extract")
    print("-" * 50)

    def nd_purity(params_arr):
        """Nd purity in extract (mole fraction among REEs)."""
        n_stages, S_F_ratio, T = params_arr

        solvent_adj = make_stream(
            flows={
                "Organic": 10.0 * S_F_ratio,
                "La": 0.0,
                "Nd": 0.0,
                "Dy": 0.0,
            },
            T=T,
            P=101325.0,
        )

        params = CascadeParams(
            n_stages=n_stages,
            equilibrium=lle_eq,
            flow_config="counter_current",
        )
        cascade_fn = MultistageCascade(params)

        _, extract, _ = cascade_fn(feed, solvent_adj, T=T)
        ext_flows = get_flows(extract)

        total_REE = ext_flows['La'] + ext_flows['Nd'] + ext_flows['Dy']
        purity = ext_flows['Nd'] / (total_REE + 1e-10)

        return purity

    def neg_nd_purity(params_arr):
        """Negative purity for minimization."""
        return -nd_purity(params_arr)

    # Gradient descent optimization
    params = jnp.array([5.0, 1.0, 298.15])  # [n_stages, S/F, T]
    learning_rates = jnp.array([0.5, 0.01, 1.0])  # Different rates for each param

    print("Starting optimization (gradient descent)...")
    print(f"Initial: n_stages={params[0]:.1f}, S/F={params[1]:.2f}, T={params[2]:.1f}K")
    print(f"Initial Nd purity: {float(nd_purity(params))*100:.2f}%")

    for i in range(50):
        grads = grad(neg_nd_purity)(params)
        params = params - learning_rates * grads

        # Enforce bounds
        params = jnp.array([
            jnp.clip(params[0], 2.0, 15.0),   # 2-15 stages
            jnp.clip(params[1], 0.5, 3.0),    # S/F ratio 0.5-3.0
            jnp.clip(params[2], 280.0, 350.0), # T in 280-350 K
        ])

        if (i + 1) % 10 == 0:
            purity = nd_purity(params)
            print(f"Iter {i+1}: n={params[0]:.1f}, S/F={params[1]:.2f}, "
                  f"T={params[2]:.1f}K, purity={float(purity)*100:.2f}%")

    final_purity = nd_purity(params)
    print(f"\nOptimized parameters:")
    print(f"  n_stages = {float(params[0]):.1f}")
    print(f"  S/F ratio = {float(params[1]):.2f}")
    print(f"  T = {float(params[2]):.1f} K")
    print(f"  Nd purity = {float(final_purity)*100:.2f}%")

    # =========================================================================
    # Multi-Objective: Recovery vs Purity Trade-off
    # =========================================================================
    print("\n7. Multi-Objective Analysis: Recovery vs Purity")
    print("-" * 50)

    print("\nVarying S/F ratio to show trade-off:")
    print(f"{'S/F':>6} {'Nd Rec%':>10} {'Nd Purity%':>12} {'La Rec%':>10} {'Dy Rec%':>10}")
    print("-" * 50)

    for sf in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        solvent_adj = make_stream(
            flows={
                "Organic": 10.0 * sf,
                "La": 0.0,
                "Nd": 0.0,
                "Dy": 0.0,
            },
            T=298.15,
            P=101325.0,
        )

        params = CascadeParams(
            n_stages=5,
            equilibrium=lle_eq,
            flow_config="counter_current",
        )
        cascade_fn = MultistageCascade(params)

        _, extract, _ = cascade_fn(feed, solvent_adj, T=298.15)
        ext_flows = get_flows(extract)

        nd_rec = float(ext_flows['Nd']) / feed_flows['Nd'] * 100
        la_rec = float(ext_flows['La']) / feed_flows['La'] * 100
        dy_rec = float(ext_flows['Dy']) / feed_flows['Dy'] * 100

        total_REE = float(ext_flows['La'] + ext_flows['Nd'] + ext_flows['Dy'])
        nd_pur = float(ext_flows['Nd']) / total_REE * 100 if total_REE > 0 else 0

        print(f"{sf:>6.2f} {nd_rec:>10.1f} {nd_pur:>12.1f} {la_rec:>10.1f} {dy_rec:>10.1f}")

    # =========================================================================
    # Effect of Number of Stages
    # =========================================================================
    print("\n8. Effect of Number of Stages")
    print("-" * 50)

    print(f"{'Stages':>8} {'Nd Rec%':>10} {'La Rec%':>10} {'Dy Rec%':>10}")
    print("-" * 40)

    for n in [1, 2, 3, 5, 7, 10]:
        params = CascadeParams(
            n_stages=n,
            equilibrium=lle_eq,
            flow_config="counter_current",
        )
        cascade_fn = MultistageCascade(params)

        _, extract, _ = cascade_fn(feed, solvent, T=298.15)
        ext_flows = get_flows(extract)

        nd_rec = float(ext_flows['Nd']) / feed_flows['Nd'] * 100
        la_rec = float(ext_flows['La']) / feed_flows['La'] * 100
        dy_rec = float(ext_flows['Dy']) / feed_flows['Dy'] * 100

        print(f"{n:>8} {nd_rec:>10.1f} {la_rec:>10.1f} {dy_rec:>10.1f}")

    # =========================================================================
    # Jacobian: Full Sensitivity Matrix
    # =========================================================================
    print("\n9. Jacobian Analysis")
    print("-" * 50)

    def all_recoveries(params_arr):
        """Return all three recoveries."""
        n_stages, S_F_ratio, T = params_arr

        solvent_adj = make_stream(
            flows={
                "Organic": 10.0 * S_F_ratio,
                "La": 0.0,
                "Nd": 0.0,
                "Dy": 0.0,
            },
            T=T,
            P=101325.0,
        )

        params = CascadeParams(
            n_stages=n_stages,
            equilibrium=lle_eq,
            flow_config="counter_current",
        )
        cascade_fn = MultistageCascade(params)

        _, extract, _ = cascade_fn(feed, solvent_adj, T=T)
        ext_flows = get_flows(extract)

        return jnp.array([
            ext_flows['La'] / feed_flows['La'],
            ext_flows['Nd'] / feed_flows['Nd'],
            ext_flows['Dy'] / feed_flows['Dy'],
        ])

    # Compute Jacobian
    params_eval = jnp.array([5.0, 1.0, 298.15])
    J = jacfwd(all_recoveries)(params_eval)

    print("Jacobian matrix ∂(recoveries)/∂(parameters):")
    print("                  n_stages      S/F ratio          T")
    print(f"  ∂(La rec)     {J[0,0]:10.4f}   {J[0,1]:10.4f}   {J[0,2]:10.6f}")
    print(f"  ∂(Nd rec)     {J[1,0]:10.4f}   {J[1,1]:10.4f}   {J[1,2]:10.6f}")
    print(f"  ∂(Dy rec)     {J[2,0]:10.4f}   {J[2,1]:10.4f}   {J[2,2]:10.6f}")

    print("\nInterpretation:")
    print(f"  - Dy extraction is most sensitive to stages (∂Dy/∂n = {J[2,0]:.4f})")
    print(f"  - La extraction benefits most from more solvent (∂La/∂(S/F) = {J[0,1]:.4f})")
    print(f"  - Temperature effects are small due to assumed dH values")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
