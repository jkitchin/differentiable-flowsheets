"""The recycle-gas loop: knock-out, amine scrubber, purge, compressor, makeup -- and its tear.

Around a hydroprocessing reactor the gas goes round::

    reactor -> effluent cooler -> HP separator --vapour--> KO drum -> amine
       ^                                                               |
       |                                                  purge <------+
       +--- treat gas = recycle + makeup <--- recycle compressor <-----+

Every unit here is a pure function of :class:`~.layout.Flows`:

* :func:`knockout` -- the recycle compressor's suction drum: any pseudo-
  component (and its attributes) in the separator vapour is returned to the
  separator liquid. A modelling choice that makes the recycle gas carry only
  real species: at 40-60 C and 30-150 bar the cuts in the separator vapour
  are a few hundred ppm of it, and recycling them would add every cut
  attribute to the tear. Light ends kept as real species (C5, C6) do recycle.
* :func:`amine_scrub` -- a fixed fraction of the H2S removed (the amine
  contactor, issue #306), and a fixed fraction of the NH3 (the wash water
  injected upstream of the separator in a real unit, which takes the NH3 as
  ammonium bisulfide; here a removal fraction). Nothing else is absorbed.
* :func:`purge_split` -- a fraction of the scrubbed gas leaves as purge.
* :func:`compress` -- ideal-gas isentropic compression with temperature-
  dependent Cp, ``int Cp/T dT = R ln(P2/P1)`` for the isentropic outlet and
  ``W = (H_s - H_1)/eta``; both temperature equations solved by Newton with
  implicit differentiation. Ideal gas, not Peng-Robinson: the recycle gas is
  80-95 % hydrogen at a compression ratio of 1.1-1.3, where the compressibility
  correction to the work is a few per cent (stated, not computed). difflow's
  :class:`difflow.units.eos_units.Compressor` is the EOS-consistent unit, but
  its ``CubicThermo`` is built from concrete species data, which a gradient
  with respect to the assay cannot pass through.
* :func:`makeup_for_ratio` -- the makeup gas that brings the treat gas's H2 to
  a target, explicitly ``M = (H2_target - R_H2) / y_H2`` (so the H2/oil ratio
  is a spec, and the makeup rate an output).

The loop is closed by :func:`solve_tear`: Newton on ``g(R) - R = 0`` over the
recycle-gas flows (one unknown per gas species), with the optimistix implicit
adjoint, so a gradient through a converged loop is ``-(J - I)^-1 dg/dtheta``
and never differentiates the iterations. It is a difflow ``Flowsheet``
recycle in substance (a tear on the recycle gas, converged and implicitly
differentiated) without the Flowsheet object, whose stream packing does not
carry per-cut attributes and whose accelerated paths are Python loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array

from difflow_refinery.hydroprocessing.layout import Flows, Layout
from difflow_refinery.hydroprocessing.solve import newton_solve
from difflow_refinery.hydroprocessing.thermo import R_GAS, Components

jax.config.update("jax_enable_x64", True)

#: mol per normal cubic metre (0 C, 1 atm) of an ideal gas.
MOL_PER_NM3 = 101325.0 / (R_GAS * 273.15)


def knockout(vapor: Flows, liquid: Flows) -> tuple[Flows, Flows]:
    """Move every cut (molecules and attributes) from ``vapor`` to ``liquid``."""
    z = jnp.zeros_like(vapor.cut)
    moved = Flows(jnp.zeros_like(vapor.gas), vapor.cut, vapor.attr)
    return Flows(vapor.gas, z, jnp.zeros_like(vapor.attr)), liquid + moved


def amine_scrub(gas: Flows, layout: Layout, h2s_removal, nh3_removal=1.0) -> tuple[Flows, Flows]:
    """``(sweet gas, absorbed)``: fractions of H2S and NH3 removed."""
    frac = jnp.zeros(layout.n_gas)
    if layout.has_gas("hydrogen_sulfide"):
        frac = frac.at[layout.gas_index("hydrogen_sulfide")].set(h2s_removal)
    if layout.has_gas("ammonia"):
        frac = frac.at[layout.gas_index("ammonia")].set(nh3_removal)
    absorbed = Flows(gas.gas * frac, jnp.zeros_like(gas.cut), jnp.zeros_like(gas.attr))
    return gas - absorbed, absorbed


def purge_split(gas: Flows, fraction) -> tuple[Flows, Flows]:
    """``(recycle, purge)``."""
    purge = gas.scale(fraction)
    return gas - purge, purge


def _gas_z(gas: Flows, layout: Layout, comps: Components):
    """Flows and constants of the gases that have them (water by its own Cp)."""
    names = comps.names[:comps.n_gas]
    z = jnp.stack([gas.gas[layout.gas_index(g)] for g in names])
    return z, comps.cp_ig[:comps.n_gas]


def _h(z, cp, T):
    from difflow_refinery.thermo import _cp_integral
    return jnp.sum(z * _cp_integral(cp, T))


def _s_int(cp, T1, T2):
    """``int_{T1}^{T2} Cp/T dT`` for cubic Cp coefficients."""
    a, b, c, d = (cp[..., k] for k in range(4))

    def F(T):
        return a * jnp.log(T) + b * T + c * T**2 / 2 + d * T**3 / 3

    return F(T2) - F(T1)


def compress(gas: Flows, layout: Layout, comps: Components, T_in, P_in, P_out, eta) -> tuple[Array, Array]:
    """Ideal-gas compressor: ``(T_out, power W)``.

    Isentropic outlet from ``sum z_i int Cp_i/T dT = (sum z_i) R ln(P_out/P_in)``,
    actual outlet enthalpy ``H_1 + (H_s - H_1)/eta`` (the convention of
    :class:`difflow.units.eos_units.Compressor`). Water and any cut flows are
    ignored (the loop's gas carries none after :func:`knockout`).
    """
    z, cp = _gas_z(gas, layout, comps)
    T_in = jnp.asarray(T_in, dtype=float)
    lnr = jnp.log(jnp.asarray(P_out, dtype=float) / jnp.asarray(P_in, dtype=float))

    def r_s(T, args):
        z, cp, T_in, lnr = args
        n = jnp.sum(z)
        return (jnp.sum(z * _s_int(cp, T_in, T)) - n * R_GAS * lnr) / jnp.maximum(n, 1e-300)

    solver = optx.Newton(rtol=1e-12, atol=1e-10)
    Ts = optx.root_find(r_s, solver, jax.lax.stop_gradient(T_in * (1.0 + 0.3 * lnr)), args=(z, cp, T_in, lnr),
                        max_steps=50, throw=False).value
    H1 = _h(z, cp, T_in)
    Hs = _h(z, cp, Ts)
    H2 = H1 + (Hs - H1) / eta

    def r_h(T, args):
        z, cp, H2 = args
        return (_h(z, cp, T) - H2) / jnp.maximum(jnp.sum(z), 1e-300) / 30.0

    T2 = optx.root_find(r_h, solver, jax.lax.stop_gradient(Ts), args=(z, cp, H2), max_steps=50,
                        throw=False).value
    return T2, H2 - H1


def makeup_for_ratio(recycle: Flows, layout: Layout, makeup_y: Array, h2_target) -> tuple[Flows, Array]:
    """``(makeup flows, makeup mol/s)`` so the treat gas carries ``h2_target`` mol/s of H2."""
    i = layout.gas_index("hydrogen")
    M = (h2_target - recycle.gas[i]) / makeup_y[i]
    return Flows(makeup_y * M, jnp.zeros_like(recycle.cut), jnp.zeros_like(recycle.attr)), M


def makeup_vector(layout: Layout, composition: Mapping[str, float]) -> Array:
    """A makeup-gas mole-fraction vector on ``layout``'s gases (normalised)."""
    unknown = [k for k in composition if k not in layout.gases]
    if unknown:
        raise ValueError(f"makeup species {unknown} are not in the layout's gases")
    v = jnp.asarray([float(composition.get(g, 0.0)) for g in layout.gases])
    return v / jnp.sum(v)


@dataclass(frozen=True)
class TearSolution:
    """A converged tear.

    Attributes:
        value: The tear variables at the solution.
        residual: ``max |g(x) - x| / scale``.
        steps: Newton steps taken.
        converged: Whether ``residual`` is inside the tolerance.
    """

    value: Array
    residual: Array
    steps: Array
    converged: Array


jax.tree_util.register_dataclass(TearSolution, data_fields=["value", "residual", "steps", "converged"],
                                 meta_fields=[])


def solve_tear(g: Callable, x0: Array, args=None, scale: Array | None = None, tol: float = 1e-11,
               max_steps: int = 40, max_step: float | None = None, g_iter: Callable | None = None,
               jac: str = "rev") -> TearSolution:
    """Solve ``g(x, args) = x`` by Newton, implicitly differentiable in ``args``.

    ``g`` must be a pure function of ``(x, args)``: everything the loop
    depends on -- the feed, the characterization, the specs -- goes in
    ``args``, never in a closure (the Newton loop runs on stop-gradient
    copies). ``scale`` (default ``|x0|`` floored) makes the residual
    relative. Newton is :func:`.solve.newton_solve` (reverse-mode Jacobian,
    as the reactor's diffrax adjoint requires), and the returned value
    carries the implicit-function derivative ``-(dg/dx - I)^-1 dg/dargs``.
    ``g_iter``/``jac``: a forward-differentiable copy of ``g`` for the
    iterations and ``jac="fwd"`` (see :func:`.solve.newton_solve`) -- much
    cheaper to compile than reverse-mode Jacobians through a diffrax solve.
    """
    x0 = jnp.asarray(x0, dtype=float)
    s = jnp.maximum(jnp.abs(x0), 1e-6 * jnp.max(jnp.abs(x0))) if scale is None else scale
    s = jax.lax.stop_gradient(s)

    def res(u, args):
        x = u * s
        return (g(x, args) - x) / s

    res_iter = None
    if g_iter is not None:
        def res_iter(u, args):
            x = u * s
            return (g_iter(x, args) - x) / s

    sol = newton_solve(res, x0 / s, args, tol=tol, max_steps=max_steps, max_step=max_step, jac=jac,
                       f_iter=res_iter)
    return TearSolution(value=sol.value * s, residual=sol.residual, steps=sol.steps, converged=sol.converged)


__all__ = ["MOL_PER_NM3", "knockout", "amine_scrub", "purge_split", "compress", "makeup_for_ratio",
           "makeup_vector", "TearSolution", "solve_tear"]
