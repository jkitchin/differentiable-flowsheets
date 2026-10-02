"""Adiabatic reforming reactor and fired interstage heater.

The reactor is a fixed catalyst bed treated as one-dimensional plug flow in
catalyst mass ``W`` (radial-flow hydraulics are out of scope, and the bed is
isobaric at the pressure it is given). With ``F`` the species flows (mol/s),
``nu`` the stoichiometry, ``r`` the rates (mol/s/kg) and ``p_i = y_i P`` the
partial pressures::

    dF/dW  = nu^T r(p, T)
    H(F, T, P) = H_in                      (adiabatic: total enthalpy constant)
    d(coke)/dW = r_coke(p, T)

The energy balance is written as the *enthalpy* being constant along the bed
and the temperature recovered from it at every point by Newton's method
(:func:`~difflow_refinery.reforming.thermo.temperature_from_enthalpy`),
rather than as an ODE for ``T``. Element balances are then exact because the
stoichiometry conserves them (and Runge-Kutta methods preserve linear
invariants), and the energy balance is exact to the Newton tolerance, at any
ODE tolerance -- which is what lets the reactor section's balances close to
round-off. The enthalpy is the reformer's single basis (heats of formation +
ideal-gas sensible heat + Peng-Robinson vapour departure), so the heats of
reaction are those of the thermochemistry, and the temperature drop is the
model's most visible check against operation.

The ODE is integrated by ``diffrax`` in the normalised bed coordinate
``w = W / W_total`` with ``F / F_in`` as the state, by Tsit5 (explicit,
adaptive) by default: the fastest relaxation, the dehydrogenation
equilibrium at the bed inlet, is a few hundred per unit ``w`` with the
default kinetics, which an explicit method handles in a few hundred steps
and compiles several times faster than an implicit one. ``solver_name=
"kvaerno5"`` (an L-stable ESDIRK) is there for much faster kinetics.
Gradients use ``diffrax.ForwardMode`` by default, because a reformer is
differentiated through the recycle's implicit fixed point, which needs
forward-mode (JVP) derivatives of everything inside the loop; pass
``adjoint="reverse"`` for a stand-alone reactor under ``jax.grad``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import diffrax
import jax
import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, make_stream
from difflow_refinery.reforming import species as sp
from difflow_refinery.reforming import thermo as th
from difflow_refinery.reforming.kinetics import RateModel, ReformingKinetics

PA_PER_BAR = 1.0e5


def flows_of(stream: Stream) -> Array:
    """``(N_SPECIES,)`` flows of the reformer species in a difflow stream (missing = 0)."""
    z = jnp.asarray(0.0)
    return jnp.stack([jnp.asarray(stream.get(f"F_{k}", z), dtype=float) for k in sp.NAMES])


def stream_of(F: Array, T: Array, P: Array) -> Stream:
    """A difflow stream of reformer species flows ``F``."""
    return make_stream({k: F[i] for i, k in enumerate(sp.NAMES)}, T, P)


def _rate_model(iso_fraction: float) -> RateModel:
    key = float(iso_fraction)
    if key not in _RATE_MODELS:
        with jax.ensure_compile_time_eval():  # concrete arrays even when first built under jit
            _RATE_MODELS[key] = RateModel(key)
    return _RATE_MODELS[key]


_RATE_MODELS: dict[float, RateModel] = {}


@partial(jax.jit, static_argnames=("iso_fraction", "adjoint", "n_save", "rtol", "solver_name"))
def integrate_bed(F_in: Array, T_in: Array, P: Array, W: Array, kin: ReformingKinetics,
                  iso_fraction: float = 0.5, adjoint: str = "forward", n_save: int = 0,
                  rtol: float = 1e-8, solver_name: str = "tsit5"):
    """Integrate one adiabatic bed. Returns ``(F_out, T_out, coke_kg_s, profile)``.

    ``profile`` is ``None`` unless ``n_save > 0``, then ``(w, F, T)`` at
    ``n_save`` evenly spaced bed fractions (inlet and outlet included).
    """
    rm = _rate_model(iso_fraction)
    F_in = jnp.asarray(F_in, dtype=float)
    F0 = jnp.sum(F_in)
    P_bar = P / PA_PER_BAR
    h_in = th.total_enthalpy(F_in / F0, T_in, P)

    def temperature(x):
        return th.temperature_from_enthalpy(x, h_in, P, T_in)

    def rhs(w, state, args):
        x = state[:-1]
        T = temperature(x)
        y = x / jnp.sum(x)
        p = y * P_bar
        r = rm.rates(kin, p, T, P_bar)
        dx = (W / F0) * (r @ rm.nu_j)
        dc = W * rm.coke_rate(kin, p, T)
        return jnp.concatenate([dx, jnp.atleast_1d(dc)])

    y0 = jnp.concatenate([F_in / F0, jnp.zeros(1)])
    term = diffrax.ODETerm(rhs)
    solver = diffrax.Tsit5() if solver_name == "tsit5" else diffrax.Kvaerno5()
    ctrl = diffrax.PIDController(rtol=rtol, atol=rtol * 1e-3)
    adj = diffrax.ForwardMode() if adjoint == "forward" else diffrax.RecursiveCheckpointAdjoint()
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, 1.0, n_save)) if n_save else diffrax.SaveAt(t1=True)
    sol = diffrax.diffeqsolve(term, solver, t0=0.0, t1=1.0, dt0=1e-4, y0=y0, stepsize_controller=ctrl,
                              adjoint=adj, saveat=saveat, max_steps=8192)
    ys = sol.ys
    x_out = ys[-1, :-1]
    coke = ys[-1, -1]
    F_out = F0 * x_out
    T_out = temperature(x_out)
    profile = None
    if n_save:
        Ts = jax.vmap(temperature)(ys[:, :-1])
        profile = (sol.ts, F0 * ys[:, :-1], Ts)
    return F_out, T_out, coke, profile


@dataclass
class ReformingReactorParams(ParamsMixin):
    """Parameters for :class:`ReformingReactor`.

    Attributes:
        catalyst_mass: Catalyst in the bed (kg).
        kinetics: The rate parameters (:class:`ReformingKinetics`).
        P: Bed pressure (Pa); ``None`` uses the inlet stream's.
        rtol: Relative tolerance of the ODE integration.
    """

    catalyst_mass: float
    kinetics: ReformingKinetics
    P: float | None = None
    rtol: float = 1e-8


class ReformingReactor:
    """One adiabatic reforming bed (see the module docstring).

    Returns ``(outlet, info)`` with ``info``: ``T_in``, ``T_out``, ``dT``
    (outlet minus inlet, negative for the endothermic beds), ``coke`` (coke
    make, kg/s) and ``H_in``/``H_out`` (W, equal to the Newton tolerance).
    """

    symbol = "RX"
    parameter_units = {"catalyst_mass": "kg", "P": "Pa", "rtol": "-"}

    def __init__(self, params: ReformingReactorParams, adjoint: str = "forward"):
        self.params = params
        self.adjoint = adjoint

    def __call__(self, inlet: Stream) -> tuple[Stream, dict]:
        p = self.params
        F_in = flows_of(inlet)
        T_in = jnp.asarray(inlet["T"], dtype=float)
        P = jnp.asarray(inlet["P"] if p.P is None else p.P, dtype=float)
        kin = p.kinetics
        F_out, T_out, coke, _ = integrate_bed(F_in, T_in, P, jnp.asarray(p.catalyst_mass, dtype=float),
                                              kin, iso_fraction=float(kin.iso_fraction),
                                              adjoint=self.adjoint, rtol=p.rtol)
        info = {"T_in": T_in, "T_out": T_out, "dT": T_out - T_in, "coke": coke,
                "H_in": th.total_enthalpy_flash(F_in, T_in, P),
                "H_out": th.total_enthalpy_flash(F_out, T_out, P)}
        return stream_of(F_out, T_out, P), info


@dataclass
class FiredHeaterParams(ParamsMixin):
    """Parameters for :class:`FiredHeater`.

    Attributes:
        T_out: Outlet temperature (K) -- the next reactor's inlet temperature.
        efficiency: Absorbed over fired duty (as :class:`difflow_refinery.Furnace`).
        P_out: Outlet pressure (Pa); ``None`` keeps the inlet's.
    """

    T_out: float
    efficiency: float = 0.85
    P_out: float | None = None


class FiredHeater:
    """A fired heater raising one or more inlet streams to ``T_out``.

    The reformer's charge heater takes the naphtha and the recycle gas
    together, and the interstage heaters one reactor effluent each. Every
    inlet's enthalpy is :func:`~difflow_refinery.reforming.thermo.total_enthalpy_flash`
    (two-phase aware: the charge heater vaporizes the naphtha), the outlet is
    vapour at ``T_out``, and the absorbed duty is the difference, so the
    heater's energy balance closes on the reformer's one enthalpy basis.

    Returns ``(outlet, info)`` with ``duty`` (absorbed, W) and ``fired``
    (``duty / efficiency``, W).
    """

    symbol = "H"
    parameter_units = {"T_out": "K", "efficiency": "-", "P_out": "Pa"}

    def __init__(self, params: FiredHeaterParams):
        self.params = params

    def __call__(self, *inlets: Stream) -> tuple[Stream, dict]:
        p = self.params
        F = sum(flows_of(s) for s in inlets)
        P = jnp.asarray(inlets[0]["P"] if p.P_out is None else p.P_out, dtype=float)
        H_in = sum(th.total_enthalpy_flash(flows_of(s), s["T"], s["P"]) for s in inlets)
        T_out = jnp.asarray(p.T_out, dtype=float)
        H_out = th.total_enthalpy_flash(F, T_out, P)
        duty = H_out - H_in
        return stream_of(F, T_out, P), {"duty": duty, "fired": duty / p.efficiency,
                                        "H_in": H_in, "H_out": H_out}
