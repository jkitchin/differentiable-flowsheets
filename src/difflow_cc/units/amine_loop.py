"""Closed absorber-stripper solvent loop for amine CO2 capture.

``solve_amine_loop`` closes the amine circuit: the lean solvent leaving the
stripper is cooled and returned to the absorber, so the lean loading is no
longer a design input but the fixed point of

    lean loading -> absorber (rich loading) -> stripper (lean loading).

The lean solvent is the tear. Because the amine is conserved and the water
lost with the treated gas and the CO2 product is made up, the tear reduces
to one scalar, the lean loading ``alpha``, and the residual
``alpha - F(alpha)`` is bracketed on ``[0, loading_capacity]``
(``F(0) >= 0`` and ``F(capacity) <= capacity``), so a bisection always
finds it. Derivatives come from the implicit-function theorem
(optimistix ``ImplicitAdjoint``), not from unrolling the iterations, so
``jax.grad`` of capture or reboiler duty with respect to circulation,
stages, reboiler temperature, ... is exact and cheap.

The lean/rich cross exchanger is the stripper's own
(``StripperParams.cross_exchanger_approach``: the rich feed is preheated to
``T_reboiler - approach``); a separate ``LeanRichExchanger`` is not inserted,
which would count the same heat recovery twice.
"""

__all__ = ["solve_amine_loop"]

import jax.numpy as jnp
import optimistix as optx
from jax import Array

from difflow.numerics import safe_divide
from difflow.streams import Stream, get_flows, make_stream, total_flow
from difflow_cc.units.absorber import AbsorberParams, AmineAbsorber
from difflow_cc.units.stripper import AmineStripper, StripperParams

_MW_WATER = 18.0  # g/mol, as in AmineAbsorber


def solve_amine_loop(
    gas_in: Stream,
    absorber: AbsorberParams | AmineAbsorber,
    stripper: StripperParams | AmineStripper,
    *,
    tol: float = 1e-12,
    max_steps: int = 256,
) -> tuple[Stream, Stream, dict]:
    """Solve the closed absorber-stripper loop with the lean amine as tear.

    The circulation is set by the absorber parameters: ``L_G_ratio`` times
    the gas flow of liquid at ``solvent_conc`` wt% amine
    (``AbsorberParams.lean_loading`` is NOT used -- the lean loading is
    solved for). The lean solvent enters the absorber at ``T_liquid_in``
    and ``P_absorber`` (a trim cooler and pump), with make-up water
    restoring the water lost to the treated gas and the CO2 product.

    Args:
        gas_in: Flue gas (must contain ``F_CO2``).
        absorber: ``AbsorberParams`` or an ``AmineAbsorber``.
        stripper: ``StripperParams`` or an ``AmineStripper``; it must use
            the same solvent as the absorber.
        tol: Absolute tolerance on the lean loading (mol/mol).
        max_steps: Bisection step limit.

    Returns:
        treated_gas: Absorber gas outlet at the converged loop.
        co2_product: Stripper overhead CO2 product.
        info: Dict with
            - capture_efficiency: CO2 captured / CO2 fed
            - CO2_captured, CO2_stripped: mol/s (equal at convergence)
            - lean_loading, rich_loading: mol CO2/mol amine
            - reboiler_duty (W), specific_energy (GJ/t CO2)
            - rich_pinch: True where the absorber's rich-end limit binds
            - makeup_water: water added to the lean solvent (mol/s)
            - amine_flow, water_flow: circulating solvent (mol/s)
            - residual: ``|alpha - F(alpha)|`` at the solution
            - converged: ``residual`` within ``10 * tol`` and the solver
              reported success
            - lean_solvent, rich_solvent: the tear and rich streams
            - absorber, stripper: the units' own info dicts

    Example:
        >>> gas = make_stream({"CO2": 13.0, "N2": 87.0}, T=313.15, P=101325.0)
        >>> treated, co2, info = solve_amine_loop(
        ...     gas, AbsorberParams(solvent="MEA", n_stages=15, L_G_ratio=3.5),
        ...     StripperParams(solvent="MEA", n_stages=10))
        >>> float(info["capture_efficiency"])   # doctest: +SKIP
    """
    ab = absorber if isinstance(absorber, AmineAbsorber) else AmineAbsorber(absorber)
    st = stripper if isinstance(stripper, AmineStripper) else AmineStripper(stripper)
    pa = ab.params
    if st.params.solvent != pa.solvent:
        raise ValueError(
            f"absorber solvent {pa.solvent!r} and stripper solvent "
            f"{st.params.solvent!r} differ; the loop circulates one solvent"
        )

    # Circulating solvent, built as AmineAbsorber builds its own lean feed.
    F_gas = total_flow(gas_in)
    MW_amine = ab._solvent_data.MW
    w = pa.solvent_conc / 100.0
    x_amine = (w / MW_amine) / (w / MW_amine + (1.0 - w) / _MW_WATER)
    F_liquid = jnp.asarray(pa.L_G_ratio) * F_gas
    F_amine = F_liquid * x_amine
    F_H2O = F_liquid * (1.0 - x_amine)
    T_lean = jnp.asarray(pa.T_liquid_in)
    P_lean = jnp.asarray(pa.P_absorber)
    capacity = float(ab._solvent_data.loading_capacity)

    def one_pass(alpha):
        lean = make_stream(
            {"H2O": F_H2O, "Amine": F_amine, "CO2_absorbed": alpha * F_amine},
            T_lean, P_lean)
        gas_out, rich, a_info = ab(gas_in, lean)
        lean_out, co2, s_info = st(rich)
        alpha_next = safe_divide(get_flows(lean_out)["CO2_absorbed"], F_amine)
        return alpha_next, (lean, gas_out, rich, lean_out, co2, a_info, s_info)

    def residual(alpha, _args):
        return alpha - one_pass(alpha)[0]

    sol = optx.root_find(
        residual,
        optx.Bisection(rtol=0.0, atol=tol),
        jnp.asarray(0.5 * capacity),
        options={"lower": 0.0, "upper": capacity},
        max_steps=max_steps,
        throw=False,
    )
    alpha = sol.value
    alpha_next, (lean, gas_out, rich, lean_out, co2, a_info, s_info) = one_pass(alpha)
    res = jnp.abs(alpha - alpha_next)
    converged = (sol.result == optx.RESULTS.successful) & (res <= 10.0 * tol)

    lean_out_H2O = get_flows(lean_out)["H2O"]
    info = {
        "capture_efficiency": a_info["capture_efficiency"],
        "CO2_captured": a_info["CO2_captured"],
        "CO2_stripped": s_info["CO2_stripped"],
        "lean_loading": alpha,
        "rich_loading": a_info["rich_loading"],
        "reboiler_duty": s_info["reboiler_duty"],
        "specific_energy": s_info["specific_energy"],
        "rich_pinch": a_info["rich_pinch"],
        "makeup_water": F_H2O - lean_out_H2O,
        "amine_flow": F_amine,
        "water_flow": F_H2O,
        "residual": res,
        "converged": converged,
        "lean_solvent": lean,
        "rich_solvent": rich,
        "absorber": a_info,
        "stripper": s_info,
    }
    return gas_out, co2, info
