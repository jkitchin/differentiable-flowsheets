"""GasNetworkFlowsheet: a difflow Flowsheet tuned for gas networks.

Differences from the base :class:`difflow.flowsheet.Flowsheet`:

* **Signed flows.** Gas network tear streams are signed (flow against
  an arc's reference direction is routine), so :meth:`solve` defaults
  ``clip_negative_flows`` to False. See difflow issue #164.
* **Default tear guesses.** Builders attach neutral guesses in
  :attr:`tear_guess`; :meth:`solve` and :meth:`solve_differentiable`
  fall back to them.
* **A differentiable damped tear solve.** difflow's default
  ``solve()`` (Anderson acceleration) converges robustly but checks
  convergence with Python ``float()``, so it cannot be traced by
  ``jax.grad`` / ``jax.jit``. Its optimistix path
  (``acceleration="none"``) is traceable but iterates the *undamped*
  map, and gas network tear maps routinely have real negative
  eigenvalues of magnitude above one (loop flows overshoot and
  oscillate; GasLib-40's map has eigenvalues in about [-3.1, -0.6]),
  for which undamped substitution diverges or limit-cycles.
  :meth:`solve_differentiable` runs optimistix fixed-point iteration
  on the damped map ``x + alpha (g(x) - x)``, which has the same fixed
  point but a contractive iteration, and optimistix differentiates
  through the converged solution with the implicit function theorem,
  so gradients are exact regardless of alpha or iteration count.

Choosing the damping ``alpha``: for a tear map with (estimated)
eigenvalue spectrum in ``[-m, 0)``, damped iteration contracts for
``alpha < 2 / (1 + m)`` and the optimal scalar damping is about
``2 / (2 + m)``. When in doubt, measure the spectral radius with a
finite-difference Jacobian of the tear map at a solved state, or start
at ``alpha = 0.3`` (safe for spectral radius up to about 5) and
increase if iteration counts allow.
"""

from __future__ import annotations

from typing import Callable

import optimistix as optx
from difflow.flowsheet import Flowsheet
from difflow.streams import Stream
from jax import Array


class GasNetworkFlowsheet(Flowsheet):
    """Flowsheet with signed flows and a differentiable tear solver."""

    #: default tear guesses, set by the builder ({dest_name: Stream})
    tear_guess: dict[str, Stream] | None = None

    def solve(self, tear_initial=None, clip_negative_flows: bool = False,
              **kwargs):
        """Solve the flowsheet (see :meth:`difflow.Flowsheet.solve`).

        Identical to the base method except that
        ``clip_negative_flows`` defaults to False (gas network tear
        flows are signed) and ``tear_initial`` falls back to
        :attr:`tear_guess`.
        """
        if tear_initial is None:
            tear_initial = self.tear_guess
        return super().solve(
            tear_initial=tear_initial,
            clip_negative_flows=clip_negative_flows,
            **kwargs,
        )

    def solve_differentiable(
        self,
        tear_initial: dict[str, Stream] | None = None,
        alpha: float = 0.3,
        rtol: float = 1e-10,
        atol: float = 1e-6,
        max_iter: int = 500,
        return_stats: bool = False,
    ):
        """Solve with damped fixed-point iteration (jit- and grad-safe).

        Args:
            tear_initial: initial guesses for tear (recycle
                destination) streams; falls back to :attr:`tear_guess`.
            alpha: damping factor in (0, 1]; see the module docstring
                for how to choose it from the tear-map spectrum.
            rtol, atol: optimistix convergence tolerances on the packed
                tear vector [flow (kg/s), T (K), P (Pa)].
            max_iter: iteration cap; the solve returns the last iterate
                without raising (verify residuals downstream, e.g. with
                :func:`difflow_gas.verify.residual_report`).
            return_stats: also return the optimistix stats dict
                (e.g. ``num_steps``).

        Returns:
            Dict of all streams, or ``(streams, stats)`` if
            ``return_stats``.
        """
        if not self.recycles:
            streams = self._solve_sequential()
            return (streams, {"num_steps": 0}) if return_stats else streams

        if tear_initial is None:
            tear_initial = self.tear_guess
        if tear_initial is None:
            tear_initial = {
                dest: self._make_zero_stream()
                for dest in self.recycles.values()
            }

        tear_names = list(tear_initial.keys())
        x0 = self._streams_to_array(tear_initial)

        def damped_map(x: Array, args) -> Array:
            tear = self._array_to_streams(x, tear_names)
            streams = dict(self.feeds)
            streams.update(tear)
            streams = self._run_units(streams)
            new_tear = {
                dest: streams[src] for src, dest in self.recycles.items()
            }
            g = self._streams_to_array(new_tear)
            return x + alpha * (g - x)

        solver = optx.FixedPointIteration(rtol=rtol, atol=atol)
        sol = optx.fixed_point(
            damped_map, solver, x0, max_steps=max_iter, throw=False
        )

        tear = self._array_to_streams(sol.value, tear_names)
        streams = dict(self.feeds)
        streams.update(tear)
        streams = self._run_units(streams)
        if return_stats:
            return streams, dict(sol.stats)
        return streams

    def _apply_params(self, params: dict) -> "GasNetworkFlowsheet":
        """Like Flowsheet._apply_params but preserving this subclass."""
        base = super()._apply_params(params)
        new_fs = GasNetworkFlowsheet(
            self.species_order,
            default_flow=self.default_flow,
            default_T=self.default_T,
            default_P=self.default_P,
        )
        new_fs.feeds = base.feeds
        new_fs.recycles = base.recycles
        new_fs.units = base.units
        new_fs.tear_guess = self.tear_guess
        return new_fs

    def make_objective_fn(
        self, objective_fn: Callable[[dict[str, Stream]], Array]
    ) -> Callable[[dict], Array]:
        """Differentiable objective; solves via
        :meth:`solve_differentiable`.

        Parameter keys use difflow's dot notation
        ``"<unit_name>.<param_name>"``, e.g. ``{"cs_station1.ratio":
        1.1, "src_node0.P_set": 51e5}``.
        """

        def objective(params: dict) -> Array:
            updated = self._apply_params(params)
            streams = updated.solve_differentiable()
            return objective_fn(streams)

        return objective
