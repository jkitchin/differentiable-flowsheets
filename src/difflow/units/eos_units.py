"""EOS-consistent process units: turboexpander, compressor, JT valve, separator.

difflow's core had no EOS-consistent turboexpander/compressor, no rigorous
(EOS-enthalpy) valve, and no component separator: ``PHFlash``'s inner flash is
ideal-K (Raoult), so it cannot model a Peng-Robinson Joule-Thomson valve, and
the ``difflow_gas`` valve/compressor are pressure-node elements with no
thermodynamics. That blocks cryogenic/gas-processing flowsheets (expander
plants, NGL recovery, refrigeration). See issue #171.

Each unit takes a :class:`~difflow.thermo.CubicThermo` (ideal-gas Cp + PR/SRK
departures, two-phase aware) and a difflow ``Stream`` and returns an outlet
stream plus an info dict. Internal temperature solves use ``optimistix`` root
finds on the two-phase EOS enthalpy/entropy, so every unit is implicitly
differentiable: gradients of an outlet temperature, duty, or shaft work with
respect to feed conditions, pressures, or efficiencies flow through by the
implicit function theorem.

Units
-----
- :class:`Turboexpander` -- isentropic expansion to ``P_out`` with an
  isentropic-efficiency enthalpy correction; extracts shaft work.
- :class:`Compressor` -- isentropic compression to ``P_out`` with an
  isentropic-efficiency correction; consumes shaft work.
- :class:`JTValve` -- isenthalpic pressure letdown on the EOS enthalpy (a real
  Joule-Thomson valve).
- :class:`ComponentSeparator` -- fixed per-component recoveries to a product
  stream (a black-box column/separator surrogate).

The turboexpander and compressor need the EOS entropy departure added in
issue #170 (:meth:`difflow.eos.PengRobinson.entropy_departure`).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
import optimistix as optx

from difflow.streams import Stream, make_stream, get_flows
from difflow.params_mixin import ParamsMixin
from difflow.thermo import CubicThermo


# Default temperature-search window (K) for the enthalpy/entropy root finds.
# Wide enough for cryogenic-to-moderate gas processing; override via a unit's
# ``T_bounds`` when a stream sits outside it.
DEFAULT_T_BOUNDS: tuple[float, float] = (100.0, 1000.0)


def _flow_dict(stream: Stream, species_order) -> dict:
    """Extract ``{species: molar flow}`` in ``species_order`` from a Stream."""
    f = get_flows(stream)
    return {s: f[s] for s in species_order}


def _solve_stream_T(
    residual,
    T_guess: float | Array,
    bounds: tuple[float, float] = DEFAULT_T_BOUNDS,
) -> Array:
    """Find T with ``residual(T) == 0`` (implicitly differentiable).

    ``residual`` is a scalar function of temperature (an enthalpy or entropy
    match). A Newton solve is used; T is clipped to ``bounds`` inside the
    residual so the iteration cannot wander into a nonphysical region where the
    EOS root solve would fail. Because ``optimistix.root_find`` differentiates
    through the converged solution implicitly, gradients of T w.r.t. any
    upstream parameter flow through this call.
    """
    lo, hi = bounds

    def fn(T, args):
        return residual(jnp.clip(T, lo, hi))

    solver = optx.Newton(rtol=1e-9, atol=1e-6)
    sol = optx.root_find(
        fn, solver, jnp.asarray(T_guess, dtype=float), max_steps=100, throw=False
    )
    return jnp.clip(sol.value, lo, hi)


# =============================================================================
# Turboexpander (isentropic expansion with efficiency)
# =============================================================================
@dataclass
class TurboexpanderParams(ParamsMixin):
    """Parameters for :class:`Turboexpander`.

    Attributes:
        P_out: Discharge pressure (Pa); must be below the inlet pressure.
        eta_isentropic: Isentropic (adiabatic) efficiency in (0, 1].
        T_bounds: (T_min, T_max) window for the internal temperature solves (K).
    """

    P_out: float
    eta_isentropic: float = 0.80
    T_bounds: tuple[float, float] = DEFAULT_T_BOUNDS


class Turboexpander:
    """Adiabatic turboexpander: isentropic expansion corrected by efficiency.

    Solves ``S(T_isen, P_out) = S(T_in, P_in)`` for the reversible outlet
    temperature, then applies the isentropic efficiency to the enthalpy drop::

        H_out = H_in + eta * (H_isen - H_in)

    and finds ``T_out`` from the two-phase enthalpy at ``P_out``. The extracted
    shaft work is ``W = H_in - H_out`` (positive, in W). Both the entropy and
    enthalpy are two-phase aware, so an expander that partly condenses its outlet
    (common in cryogenic service) is handled correctly.
    """

    def __init__(self, params: TurboexpanderParams, thermo: CubicThermo):
        self.params = params
        self.thermo = thermo

    def __call__(self, inlet: Stream) -> tuple[Stream, dict]:
        flows = _flow_dict(inlet, self.thermo.species_order)
        T_in, P_in = inlet["T"], inlet["P"]
        P_out = jnp.asarray(self.params.P_out)
        eta = jnp.asarray(self.params.eta_isentropic)
        bounds = self.params.T_bounds

        S_in = self.thermo.stream_entropy_flash(flows, T_in, P_in)
        H_in = self.thermo.stream_enthalpy_flash(flows, T_in, P_in)

        T_isen = _solve_stream_T(
            lambda T: self.thermo.stream_entropy_flash(flows, T, P_out) - S_in,
            T_guess=T_in - 20.0,
            bounds=bounds,
        )
        H_isen = self.thermo.stream_enthalpy_flash(flows, T_isen, P_out)
        H_out = H_in + eta * (H_isen - H_in)

        T_out = _solve_stream_T(
            lambda T: self.thermo.stream_enthalpy_flash(flows, T, P_out) - H_out,
            T_guess=T_isen,
            bounds=bounds,
        )
        W = H_in - H_out
        return make_stream(flows, T_out, P_out), {
            "W": W,
            "T_isen": T_isen,
            "T_out": T_out,
            "H_in": H_in,
            "H_out": H_out,
        }


# =============================================================================
# Compressor (isentropic compression with efficiency)
# =============================================================================
@dataclass
class CompressorParams(ParamsMixin):
    """Parameters for :class:`Compressor`.

    Attributes:
        P_out: Discharge pressure (Pa); must be above the inlet pressure.
        eta_isentropic: Isentropic (adiabatic) efficiency in (0, 1].
        T_bounds: (T_min, T_max) window for the internal temperature solves (K).
    """

    P_out: float
    eta_isentropic: float = 0.75
    T_bounds: tuple[float, float] = DEFAULT_T_BOUNDS


class Compressor:
    """Adiabatic compressor: isentropic compression corrected by efficiency.

    Solves ``S(T_isen, P_out) = S(T_in, P_in)`` for the reversible outlet, then
    the isentropic efficiency inflates the enthalpy rise (an inefficient machine
    needs *more* work than the reversible one)::

        H_out = H_in + (H_isen - H_in) / eta

    ``T_out`` follows from the two-phase enthalpy at ``P_out``. The required
    shaft work is ``W = H_out - H_in`` (positive, in W).
    """

    def __init__(self, params: CompressorParams, thermo: CubicThermo):
        self.params = params
        self.thermo = thermo

    def __call__(self, inlet: Stream) -> tuple[Stream, dict]:
        flows = _flow_dict(inlet, self.thermo.species_order)
        T_in, P_in = inlet["T"], inlet["P"]
        P_out = jnp.asarray(self.params.P_out)
        eta = jnp.asarray(self.params.eta_isentropic)
        bounds = self.params.T_bounds

        S_in = self.thermo.stream_entropy_flash(flows, T_in, P_in)
        H_in = self.thermo.stream_enthalpy_flash(flows, T_in, P_in)

        T_isen = _solve_stream_T(
            lambda T: self.thermo.stream_entropy_flash(flows, T, P_out) - S_in,
            T_guess=T_in + 20.0,
            bounds=bounds,
        )
        H_isen = self.thermo.stream_enthalpy_flash(flows, T_isen, P_out)
        H_out = H_in + (H_isen - H_in) / eta

        T_out = _solve_stream_T(
            lambda T: self.thermo.stream_enthalpy_flash(flows, T, P_out) - H_out,
            T_guess=T_isen,
            bounds=bounds,
        )
        W = H_out - H_in
        return make_stream(flows, T_out, P_out), {
            "W": W,
            "T_isen": T_isen,
            "T_out": T_out,
            "H_in": H_in,
            "H_out": H_out,
        }


# =============================================================================
# JT valve (isenthalpic pressure letdown)
# =============================================================================
@dataclass
class JTValveParams(ParamsMixin):
    """Parameters for :class:`JTValve`.

    Attributes:
        P_out: Downstream pressure (Pa); must be below the inlet pressure.
        T_bounds: (T_min, T_max) window for the internal temperature solve (K).
    """

    P_out: float
    T_bounds: tuple[float, float] = DEFAULT_T_BOUNDS


class JTValve:
    """Joule-Thomson valve: adiabatic, isenthalpic pressure letdown.

    Holds the two-phase EOS enthalpy constant across the pressure drop and
    solves for the outlet temperature ``H(T_out, P_out) = H(T_in, P_in)``. On a
    real gas this produces the Joule-Thomson temperature change (cooling for
    most gases below their inversion temperature) that an ideal-gas or
    ideal-K model misses entirely.
    """

    def __init__(self, params: JTValveParams, thermo: CubicThermo):
        self.params = params
        self.thermo = thermo

    def __call__(self, inlet: Stream) -> tuple[Stream, dict]:
        flows = _flow_dict(inlet, self.thermo.species_order)
        T_in, P_in = inlet["T"], inlet["P"]
        P_out = jnp.asarray(self.params.P_out)

        H_in = self.thermo.stream_enthalpy_flash(flows, T_in, P_in)
        T_out = _solve_stream_T(
            lambda T: self.thermo.stream_enthalpy_flash(flows, T, P_out) - H_in,
            T_guess=T_in - 10.0,
            bounds=self.params.T_bounds,
        )
        return make_stream(flows, T_out, P_out), {"T_out": T_out, "H": H_in}


# =============================================================================
# Component separator (fixed per-component recoveries)
# =============================================================================
@dataclass
class ComponentSeparatorParams(ParamsMixin):
    """Parameters for :class:`ComponentSeparator`.

    Attributes:
        recovery_to_product: ``{species: fraction of inlet routed to the product
            stream}``. Each fraction is in [0, 1]; the complement goes to the
            residue stream. Species absent from the dict default to
            ``default_recovery``.
        default_recovery: Recovery used for any species not listed (default 0,
            i.e. it all leaves in the residue).
    """

    recovery_to_product: dict
    default_recovery: float = 0.0


class ComponentSeparator:
    """Split each component by a fixed recovery (a black-box separator surrogate).

    Mirrors the ``ComponentSeparator`` block of common process simulators: the
    product stream gets ``recovery_to_product[species]`` of each component and
    the residue gets the complement. Both products inherit the inlet T and P.
    The reported duty ``Q`` is the enthalpy imbalance needed to hold both
    products at the inlet temperature (a rigorous column would differ); it lets
    the block close an energy balance as a surrogate for a real separation.

    Returns ``(residue, product, info)``.
    """

    def __init__(self, params: ComponentSeparatorParams, thermo: CubicThermo):
        self.params = params
        self.thermo = thermo

    def __call__(self, inlet: Stream) -> tuple[Stream, Stream, dict]:
        flows = _flow_dict(inlet, self.thermo.species_order)
        T, P = inlet["T"], inlet["P"]
        rec = self.params.recovery_to_product
        default = self.params.default_recovery

        product = {
            s: flows[s] * jnp.asarray(rec.get(s, default)) for s in flows
        }
        residue = {
            s: flows[s] * (1.0 - jnp.asarray(rec.get(s, default))) for s in flows
        }

        H_in = self.thermo.stream_enthalpy_flash(flows, T, P)
        H_out = (
            self.thermo.stream_enthalpy_flash(residue, T, P)
            + self.thermo.stream_enthalpy_flash(product, T, P)
        )
        Q = H_out - H_in
        return (
            make_stream(residue, T, P),
            make_stream(product, T, P),
            {"Q": Q, "H_in": H_in, "H_out": H_out},
        )
