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

__all__ = ["SolventMakeup", "SolventMakeupParams", "solve_amine_loop"]

from dataclasses import dataclass

import jax.numpy as jnp
import optimistix as optx
from jax import Array

from difflow.numerics import safe_divide
from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, get_flows, make_stream, total_flow
from difflow_cc.units.absorber import AbsorberParams, AmineAbsorber
from difflow_cc.units.stripper import AmineStripper, StripperParams

_MW_WATER = 18.0  # g/mol, as in AmineAbsorber



@dataclass
class SolventMakeupParams(ParamsMixin):
    """Design circulation of an amine loop, restored on every pass.

    Attributes:
        amine_flow: circulating amine (mol/s). For a design given as L/G,
            it is ``L_G_ratio * F_gas * x_amine`` with ``x_amine`` the amine
            mole fraction of the solvent, as ``AmineAbsorber`` builds it.
        water_flow: circulating water (mol/s).
        T: lean solvent temperature leaving the trim cooler (K).
        P: lean solvent pressure leaving the pump (Pa).
    """

    amine_flow: float
    water_flow: float
    T: float = 313.15
    P: float = 101325.0


class SolventMakeup:
    """Return lean amine to the absorber at the design circulation.

    The block that closes an absorber-stripper loop drawn as a flowsheet:
    the stripper's lean outlet comes in, and the absorber's lean solvent
    goes out with its amine and water flows reset to the design values,
    its loading (``CO2_absorbed / Amine``) carried through, at the trim
    cooler's temperature and the pump's pressure.

    A recycle of conserved amine with no purge has no unique steady state
    --- any circulating inventory is one --- and the water the treated gas
    and the CO2 product carry away would drain the loop. Fixing the
    circulation here removes both, leaving the lean loading as the one
    quantity the tear has to find. It is the same closure
    :func:`solve_amine_loop` makes internally, so a flowsheet built with
    this unit converges to the same answer.
    """

    symbol = "Makeup"
    equations = [
        r"F_\mathrm{amine} = F_\mathrm{amine}^\mathrm{design},\quad"
        r"F_\mathrm{H_2O} = F_\mathrm{H_2O}^\mathrm{design}",
        r"F_\mathrm{CO_2,abs} = \alpha\,F_\mathrm{amine},\quad"
        r"\alpha = F_\mathrm{CO_2,abs}^\mathrm{in}/F_\mathrm{amine}^\mathrm{in}",
    ]
    references = [
        "Kohl, A.L. and Nielsen, R.B. Gas Purification, 5e, Gulf, 1997.",
    ]
    parameter_units = {
        "amine_flow": "mol/s",
        "water_flow": "mol/s",
        "T": "K",
        "P": "Pa",
    }
    numerical_method = "Explicit: flows reset, loading carried through."

    def __init__(self, params: SolventMakeupParams):
        self.params = params

    def __call__(self, lean_in: Stream) -> tuple[Stream, dict]:
        """Restore the design circulation.

        Args:
            lean_in: Lean solvent from the stripper, with ``Amine``,
                ``H2O`` and ``CO2_absorbed`` flows.

        Returns:
            lean_out: Lean solvent for the absorber's ``solvent_in``.
            info: Dict with
                - lean_loading: mol CO2 per mol amine, carried through
                - makeup_water: water added (mol/s); negative is a purge
                - makeup_amine: amine added (mol/s); zero at a converged
                  loop, where the stripper returns what it was sent
        """
        p = self.params
        flows = get_flows(lean_in)
        amine_in = jnp.asarray(flows.get("Amine", 0.0))
        loading = safe_divide(jnp.asarray(flows.get("CO2_absorbed", 0.0)),
                              amine_in)
        amine = jnp.asarray(p.amine_flow)
        water = jnp.asarray(p.water_flow)
        lean_out = make_stream(
            {"H2O": water, "Amine": amine, "CO2_absorbed": loading * amine},
            jnp.asarray(p.T), jnp.asarray(p.P))
        info = {
            "lean_loading": loading,
            "makeup_water": water - jnp.asarray(flows.get("H2O", 0.0)),
            "makeup_amine": amine - amine_in,
        }
        return lean_out, info


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
