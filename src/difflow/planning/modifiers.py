"""Zeroth- and first-order modifiers for plant-model mismatch.

Under *structural* mismatch — the model has the wrong form, not merely the
wrong parameters — correcting only the predicted *values* converges to the
model's optimum, not the plant's.  Matching the plant's *gradients* at the
operating point is what makes the converged point satisfy the plant's
necessary conditions of optimality.  This is modifier adaptation:

    y_mod(u) = y_model(u) + eps + lam . (u - u_ad)

    eps  = y_plant(u_ad)      - y_model(u_ad)
    lam  = grad y_plant(u_ad) - grad y_model(u_ad)

Reference: Marchetti, Chachuat and Bonvin, "Modifier-Adaptation Methodology
for Real-Time Optimization", Ind. Eng. Chem. Res. 48(13), 2009,
doi:10.1021/ie801352x.

The fit with a delta-base plan is unusually tight: the delta vectors *are*
the first-order terms the method corrects, so ``lam`` is added straight onto
``J`` and the LP needs no change at all.  Both modifiers are first-order
filtered, because an unfiltered gradient correction from noisy plant data is
a fast route to oscillation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.planning.block import Block
from difflow.planning.linearize import Linearization, linearize_block


@dataclass
class Modifiers:
    """Value and gradient corrections for one block.

    Attributes:
        eps: Zeroth-order correction, shape ``(n_y,)``.
        lam: First-order correction, shape ``(n_y, n_u)``.
        u_ad: Adaptation point the corrections were computed at.
    """

    eps: Array
    lam: Array
    u_ad: Array

    @classmethod
    def zeros(cls, block: Block, u_ad: Array | None = None) -> "Modifiers":
        """All-zero modifiers, i.e. the uncorrected model."""
        u = block.u0 if u_ad is None else jnp.asarray(u_ad, dtype=float)
        return cls(eps=jnp.zeros(block.n_y),
                   lam=jnp.zeros((block.n_y, block.n_u)),
                   u_ad=jnp.asarray(u, dtype=float))

    def apply(self, lin: Linearization) -> Linearization:
        """Fold the corrections into a linearisation.

        The corrected model is affine in ``u``, so at the linearisation point
        ``u0`` the value shifts by ``eps + lam (u0 - u_ad)`` and the delta
        vectors shift by ``lam``.
        """
        eps = jnp.asarray(self.eps)
        lam = jnp.asarray(self.lam)
        y0 = lin.y0 + eps + lam @ (lin.u0 - jnp.asarray(self.u_ad))
        return replace(lin, y0=y0, J=lin.J + lam)

    def correct(self, block: Block, u: Array,
                theta: Mapping[str, Any] | None = None) -> Array:
        """Evaluate the corrected model at ``u``."""
        u = jnp.atleast_1d(jnp.asarray(u, dtype=float))
        y = block.evaluate(u, theta)
        return y + jnp.asarray(self.eps) + jnp.asarray(self.lam) @ (
            u - jnp.asarray(self.u_ad))

    @property
    def is_zero(self) -> bool:
        """True when no correction is applied."""
        return (not np.any(np.asarray(self.eps))
                and not np.any(np.asarray(self.lam)))

    def __repr__(self) -> str:
        return (f"Modifiers(|eps|={float(jnp.linalg.norm(self.eps)):.4g}, "
                f"|lam|={float(jnp.linalg.norm(self.lam)):.4g})")


def update_modifiers(block: Block, u_ad: Array,
                     plant_fn: Callable[..., Array],
                     previous: Modifiers | None = None,
                     gain_eps: float = 0.5, gain_lam: float = 0.5,
                     theta: Mapping[str, Any] | None = None,
                     use_gradients: bool = True) -> Modifiers:
    """Recompute filtered modifiers from a plant observation.

    Args:
        block: The model block.
        u_ad: Adaptation point (the current plan).
        plant_fn: The plant, with the same signature as ``block.fn``.  Its
            gradients are obtained by AD; in a real deployment they would come
            from a gradient estimator, and this function accepts whatever
            ``plant_fn`` provides.
        previous: Modifiers from the last adaptation, for filtering.
        gain_eps: Filter gain on the value correction, in ``(0, 1]``.
        gain_lam: Filter gain on the gradient correction.
        theta: Parameter override for the model block.
        use_gradients: Set ``False`` to update only ``eps``.  This reproduces
            the value-correction-only scheme, which converges to the *model's*
            optimum under structural mismatch — provided so the difference can
            be measured.

    Returns:
        Filtered :class:`Modifiers`.
    """
    if not 0.0 < gain_eps <= 1.0 or not 0.0 < gain_lam <= 1.0:
        raise ValueError("filter gains must lie in (0, 1]")

    u_ad = jnp.atleast_1d(jnp.asarray(u_ad, dtype=float))
    plant_block = replace(block, fn=plant_fn, phase_fn=None)
    plant_lin = linearize_block(plant_block, u_ad, theta)
    model_lin = linearize_block(block, u_ad, theta)

    eps_new = plant_lin.y0 - model_lin.y0
    lam_new = (plant_lin.J - model_lin.J if use_gradients
               else jnp.zeros_like(model_lin.J))

    if previous is None:
        return Modifiers(eps=eps_new, lam=lam_new, u_ad=u_ad)
    eps = (1 - gain_eps) * jnp.asarray(previous.eps) + gain_eps * eps_new
    lam = (1 - gain_lam) * jnp.asarray(previous.lam) + gain_lam * lam_new
    return Modifiers(eps=eps, lam=lam, u_ad=u_ad)


@dataclass
class ModifierAdaptationResult:
    """Outcome of a modifier-adaptation loop.

    Attributes:
        plan: The final plan.
        modifiers: The final modifiers per block.
        history: One record per adaptation step, each with the plan, the
            model's predicted objective and the plant's realised objective.
        converged: Whether successive plans stopped moving.
        n_iterations: Adaptation steps taken.
    """

    plan: dict[str, float]
    modifiers: dict[str, Modifiers]
    history: list[dict[str, Any]]
    converged: bool
    n_iterations: int

    @property
    def plant_objective(self) -> float:
        """Plant objective at the final plan."""
        return float(self.history[-1]["plant_objective"])

    def summary(self) -> str:
        lines = [f"Modifier adaptation ({self.n_iterations} steps, "
                 f"{'converged' if self.converged else 'not converged'})",
                 f"  {'step':>4s} {'model obj':>14s} {'plant obj':>14s} "
                 f"{'step size':>12s}"]
        for h in self.history:
            lines.append(f"  {h['step']:4d} {h['model_objective']:14.6g} "
                         f"{h['plant_objective']:14.6g} "
                         f"{h['step_size']:12.4g}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"ModifierAdaptationResult(n_iterations={self.n_iterations}, "
                f"plant_objective={self.plant_objective:.6g}, "
                f"converged={self.converged})")


def run_modifier_adaptation(planner, plant_fns: Mapping[str, Callable],
                            max_iter: int = 20, gain_eps: float = 0.5,
                            gain_lam: float = 0.5, tol: float = 1e-6,
                            use_gradients: bool = True,
                            **solve_kwargs) -> ModifierAdaptationResult:
    """Iterate plan / measure / correct until the plan stops moving.

    Args:
        planner: A :class:`~difflow.planning.planner.DeltaBasePlanner`.  Its
            ``modifiers`` are updated in place as the loop runs.
        plant_fns: ``{block name: plant callable}``.  Blocks absent from the
            mapping are assumed exact.
        max_iter: Maximum adaptation steps.
        gain_eps: Filter gain on the value correction.
        gain_lam: Filter gain on the gradient correction.
        tol: Convergence tolerance on the change in the plan.
        use_gradients: ``False`` disables the first-order correction.
        **solve_kwargs: Passed to ``planner.solve``.

    Returns:
        A :class:`ModifierAdaptationResult`.

    Example:
        >>> res = run_modifier_adaptation(planner, {"reactor": real_reactor})
        >>> res.plant_objective
        2461.3...
    """
    network = planner.network
    unknown = sorted(set(plant_fns) - {b.name for b in network.blocks})
    if unknown:
        raise KeyError(f"plant callables given for unknown block(s) {unknown}")

    history: list[dict[str, Any]] = []
    previous_plan: np.ndarray | None = None
    converged = False
    result = None

    for step in range(max_iter):
        result = planner.solve(**solve_kwargs)
        plan = np.asarray(result.decisions, dtype=float)
        state = network.evaluate(plan, planner.theta)

        # Score against the plant, which is what the plan must actually earn.
        plant_values = _plant_values(network, plan, plant_fns, planner.theta)
        plant_obj = float(sum(float(p) * float(plant_values[v])
                              for v, p in planner.prices.items()))

        step_size = (float("inf") if previous_plan is None
                     else float(np.linalg.norm(plan - previous_plan)))
        history.append({
            "step": step,
            "plan": {n: float(v) for n, v in
                     zip(network.decision_names, plan)},
            "model_objective": float(result.objective),
            "plant_objective": plant_obj,
            "step_size": step_size,
        })

        if step_size <= tol:
            converged = True
            break
        previous_plan = plan

        for bname, plant_fn in plant_fns.items():
            block = network.block(bname)
            th = None if planner.theta is None else planner.theta.get(bname)
            planner.modifiers[bname] = update_modifiers(
                block, state.u[bname], plant_fn,
                previous=planner.modifiers.get(bname),
                gain_eps=gain_eps, gain_lam=gain_lam, theta=th,
                use_gradients=use_gradients)

    assert result is not None
    return ModifierAdaptationResult(
        plan=result.plan, modifiers=dict(planner.modifiers), history=history,
        converged=converged, n_iterations=len(history))


def _plant_values(network, decisions, plant_fns, theta) -> dict[str, float]:
    """Evaluate the network with plant callables substituted for the models."""
    swapped = []
    for b in network.blocks:
        fn = plant_fns.get(b.name)
        swapped.append(b if fn is None else replace(b, fn=fn, phase_fn=None))
    from difflow.planning.network import Network
    plant_net = Network(swapped, [(l.source, l.target) for l in network.links])
    return plant_net.evaluate(decisions, theta).values
