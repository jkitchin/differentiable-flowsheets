"""EOS-consistent process units: turboexpander, compressor, JT valve, separator.

difflow's core had no EOS-consistent turboexpander/compressor, no rigorous
(EOS-enthalpy) valve, and no component separator: ``PHFlash``'s inner flash is
ideal-K (Raoult), so it cannot model a Peng-Robinson Joule-Thomson valve, and
the ``difflow_gas`` valve/compressor are pressure-node elements with no
thermodynamics. That blocks cryogenic/gas-processing flowsheets (expander
plants, NGL recovery, refrigeration). See issue #171.

Each unit takes a :class:`~difflow.thermo.CubicThermo` (ideal-gas Cp + PR/SRK
departures, two-phase aware) and a difflow ``Stream`` and returns an outlet
stream plus an info dict. Internal temperature solves are (P, H) and (P, S)
flashes on the two-phase EOS enthalpy/entropy (:func:`flash_PH_eos`,
:func:`flash_PS_eos`): a bracketed bisection on ``T_bounds`` (no initial guess,
so no wrong root or Newton oscillation near a dew point), with the lever-rule
vapour fraction when a pure component lands inside its dome. Every unit is
implicitly differentiable: gradients of an outlet temperature, vapour fraction,
duty, or shaft work with respect to feed conditions, pressures, or efficiencies
are attached by the implicit function theorem.

Every unit's info carries ``converged`` (bracketed and balance closed),
``residual``, ``vapor_fraction`` and the phase compositions ``x``/``y`` (EOS
species order). With concrete inputs an unconverged solve raises
:class:`EOSUnitConvergenceWarning`. The outlet *stream* holds only flows, T
and P; a downstream unit that re-flashes a pure component at exactly its
saturation temperature cannot recover the split, so read ``vapor_fraction``
from the info (or use :class:`~difflow.units.flash.PHFlash`, which returns the
liquid and vapour streams).

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
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
import warnings

from difflow.streams import Stream, make_stream, get_flows
from difflow.params_mixin import ParamsMixin
from difflow.thermo import CubicThermo
from difflow.eos import flash_TP_eos


# Default temperature-search window (K) for the enthalpy/entropy root finds.
# Wide enough for cryogenic-to-moderate gas processing; override via a unit's
# ``T_bounds`` when a stream sits outside it.
DEFAULT_T_BOUNDS: tuple[float, float] = (100.0, 1000.0)

#: Bisection steps of the bracketed (P, H) / (P, S) flash. The default window
#: (900 K) halves to ~1e-15 K in 60 steps, below float64 resolution of T.
N_BISECT: int = 60

#: Relative tolerance on the energy (or entropy) balance of a (P, H) / (P, S)
#: flash, scaled by ``|target| + F R T``. Above it the solve is reported as not
#: converged (``info["converged"] == False``).
SPEC_FLASH_TOL: float = 1e-7


class EOSUnitConvergenceWarning(UserWarning):
    """A (P, H) or (P, S) temperature solve in an EOS unit did not converge.

    Raised (outside of tracing only) when the target enthalpy or entropy is not
    bracketed by the unit's ``T_bounds`` or the balance is not closed. The
    returned temperature is then a bound or a best estimate, not a solution.
    """


def _flow_dict(stream: Stream, species_order) -> dict:
    """Extract ``{species: molar flow}`` in ``species_order`` from a Stream."""
    f = get_flows(stream)
    return {s: f[s] for s in species_order}


def _phase_terms(thermo: CubicThermo, flows: dict, T, P):
    """TP flash at (T, P) and the phase-resolved H and S pieces.

    Returns a dict with the total flow ``F``, the TP-flash vapour fraction
    ``V`` and phase compositions ``x``, ``y``, the ideal-gas totals ``H_ig``
    (W) and ``S_ig`` (W/K), and the molar departures of each phase
    (``hV``, ``hL``, ``sV``, ``sL``). ``H = H_ig + F [V hV + (1 - V) hL]``
    reproduces :meth:`CubicThermo.stream_enthalpy_flash` (likewise ``S``).
    """
    eos = thermo.eos
    F_total, z, s_ig = thermo._ideal_gas_entropy(flows, T, P)
    V, x, y = flash_TP_eos(eos, z, T, P)
    V = jnp.clip(V, 0.0, 1.0)
    return {
        "F": F_total, "z": z, "V": V, "x": x, "y": y,
        "H_ig": thermo.ideal.stream_enthalpy_ig(flows, T),
        "S_ig": F_total * s_ig,
        "hV": eos.enthalpy_departure(T, P, y, "vapor"),
        "hL": eos.enthalpy_departure(T, P, x, "liquid"),
        "sV": eos.entropy_departure(T, P, y, "vapor"),
        "sL": eos.entropy_departure(T, P, x, "liquid"),
    }


def _two_phase_value(t: dict, beta, kind: str):
    """Total H (W) or S (W/K) of the state with vapour fraction ``beta``."""
    if kind == "H":
        return t["H_ig"] + t["F"] * (beta * t["hV"] + (1.0 - beta) * t["hL"])
    return t["S_ig"] + t["F"] * (beta * t["sV"] + (1.0 - beta) * t["sL"])


def _sat_residual(thermo: CubicThermo, z, T, P):
    """Pure-fluid saturation residual ``sum_i z_i ln(phi_i^L / phi_i^V)`` at z.

    Zero where the liquid and vapour roots of the feed composition have equal
    fugacity: the saturation temperature of a pure component, and the point at
    which :func:`~difflow.eos.flash_TP_eos` switches a single-phase feed from
    liquid to vapour.
    """
    eos = thermo.eos
    lnL = jnp.log(eos.fugacity_coefficient(T, P, z, "liquid"))
    lnV = jnp.log(eos.fugacity_coefficient(T, P, z, "vapor"))
    return jnp.sum(z * (lnL - lnV))


def _flash_spec(
    thermo: CubicThermo,
    flows: dict,
    P,
    target,
    kind: str,
    bounds: tuple[float, float] = DEFAULT_T_BOUNDS,
    n_bisect: int = N_BISECT,
) -> dict:
    """(P, H) or (P, S) flash on the two-phase cubic-EOS property.

    Finds the outlet state at pressure ``P`` whose total enthalpy (``kind="H"``,
    W) or entropy (``kind="S"``, W/K) equals ``target``.

    Method. At fixed P both properties increase monotonically with T, so the
    temperature is found by bisection on the bracket ``bounds`` (``n_bisect``
    halvings, no initial guess, cannot oscillate). For a mixture the flashed
    property is continuous in T and the bisection converges on the root. For a
    pure component (or any feed the TP flash leaves single-phase on both sides
    of a phase change) the property *jumps* by the latent heat at the
    saturation temperature: the bisection then converges on the jump, the
    outlet is two-phase at that temperature, and the vapour fraction follows
    from the lever rule on the phase properties::

        beta = V_TP - (X(T) - target) / (F (x_V - x_L))

    which reduces to the TP-flash vapour fraction wherever the balance already
    closes.

    Gradients. The bisection runs on stop-gradient inputs and the derivative is
    attached by the implicit function theorem on the residual that defines the
    temperature: the property balance away from a jump, and the saturation
    condition (equal liquid/vapour fugacity, Clausius-Clapeyron) at one. So
    ``dT/dP`` is the saturation slope inside a pure fluid's dome, not the
    single-phase slope of either side.

    Returns:
        dict with ``T`` (K), ``vapor_fraction`` (molar, in [0, 1]), ``x`` and
        ``y`` (phase mole fractions in the EOS species order), ``H`` (W) and
        ``S`` (W/K) of the outlet state, ``residual`` (the balance error,
        W or W/K), ``bracketed`` and ``converged`` (boolean arrays).
    """
    lo, hi = float(bounds[0]), float(bounds[1])
    sg = jax.lax.stop_gradient
    flows_c = jax.tree_util.tree_map(sg, flows)
    P_c, target_c = sg(jnp.asarray(P, dtype=float)), sg(jnp.asarray(target, dtype=float))

    def r(T):
        t = _phase_terms(thermo, flows_c, T, P_c)
        return _two_phase_value(t, t["V"], kind) - target_c

    r_lo = r(jnp.asarray(lo))
    r_hi = r(jnp.asarray(hi))
    bracketed = (r_lo <= 0.0) & (r_hi >= 0.0)

    def body(_, ab):
        a, b = ab
        m = 0.5 * (a + b)
        below = r(m) < 0.0
        return jnp.where(below, m, a), jnp.where(below, b, m)

    a, b = jax.lax.fori_loop(0, n_bisect, body, (jnp.asarray(lo), jnp.asarray(hi)))
    T_s = jnp.where(r_lo > 0.0, lo, jnp.where(r_hi < 0.0, hi, 0.5 * (a + b)))

    # Is there a jump at T_s (latent heat not spanned by the TP flash)?
    t_s = _phase_terms(thermo, flows_c, T_s, P_c)
    res_s = _two_phase_value(t_s, t_s["V"], kind) - target_c
    scale = jnp.abs(target_c) + t_s["F"] * 8.314462618 * T_s
    jump = bracketed & (jnp.abs(res_s) > SPEC_FLASH_TOL * scale)

    # Implicit-function derivative: T_s carries no gradient; add one through
    # the residual that defines T (balance, or saturation at a jump).
    def g_balance(T):
        t = _phase_terms(thermo, flows, T, P)
        return _two_phase_value(t, t["V"], kind) - target

    def g_sat(T):
        F = thermo._flow_vector(flows)
        z = F / jnp.maximum(jnp.sum(F), 1e-30)
        return _sat_residual(thermo, z, T, P)

    def attach(g):
        val = g(T_s)
        dgdT = sg(jax.grad(g)(T_s))
        dgdT = jnp.where(jnp.abs(dgdT) > 0.0, dgdT, 1.0)
        return T_s - (val - sg(val)) / dgdT

    T_bal = attach(g_balance)
    T_sat = attach(g_sat)
    T = jnp.where(jump, T_sat, T_bal)
    T = jnp.where(bracketed, T, T_s)

    t = _phase_terms(thermo, flows, T, P)
    res_tp = _two_phase_value(t, t["V"], kind) - target
    if kind == "H":
        d_phase = t["F"] * (t["hV"] - t["hL"])
    else:
        d_phase = t["F"] * (t["sV"] - t["sL"])
    d_phase = jnp.where(jnp.abs(d_phase) > 1e-12, d_phase, 1e-12)
    beta = jnp.where(jump, t["V"] - res_tp / d_phase, t["V"])
    beta = jnp.clip(beta, 0.0, 1.0)

    H = _two_phase_value(t, beta, "H")
    S = _two_phase_value(t, beta, "S")
    residual = (H if kind == "H" else S) - target
    converged = bracketed & (jnp.abs(sg(residual)) <= SPEC_FLASH_TOL * scale)
    return {
        "T": T, "vapor_fraction": beta, "x": t["x"], "y": t["y"],
        "H": H, "S": S, "residual": residual,
        "bracketed": bracketed, "converged": converged,
    }


# ---------------------------------------------------------------------------
# JIT-compiled numeric cores.
#
# Each temperature solve builds a large jaxpr (a bisection over the two-phase
# EOS enthalpy/entropy, each step a TP flash). Called un-jitted, that graph is
# re-traced, re-lowered and re-compiled on *every* invocation. Wrapping the
# core in ``jax.jit`` compiles it once and reuses the executable on subsequent
# calls with the same thermo and array shapes. ``thermo`` and ``bounds`` are
# static arguments (hashed by object identity / value), so reusing a single
# thermo object -- as a flowsheet or a session-scoped test fixture does -- hits
# the compilation cache instead of paying the trace/compile cost again.
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnames=("thermo", "kind", "bounds"))
def _flash_spec_jit(thermo, flows, P, target, kind, bounds):
    return _flash_spec(thermo, flows, P, target, kind, bounds)


def flash_PH_eos(
    thermo: CubicThermo,
    flows: dict,
    P: Array | float,
    H: Array | float,
    T_bounds: tuple[float, float] = DEFAULT_T_BOUNDS,
) -> dict:
    """Pressure-enthalpy flash on a cubic EOS, with the vapour fraction.

    Finds the equilibrium state at pressure ``P`` whose two-phase total
    enthalpy (:meth:`~difflow.thermo.CubicThermo.stream_enthalpy_flash`
    convention) equals ``H``. Bracketed bisection on ``T_bounds``, so it has no
    initial guess and cannot return a wrong root; a pure component that lands
    inside its dome comes back at its saturation temperature with
    ``0 < vapor_fraction < 1`` from the lever rule. Implicitly differentiable
    in ``flows``, ``P`` and ``H`` (see :func:`_flash_spec`).

    Args:
        thermo: A :class:`~difflow.thermo.CubicThermo`.
        flows: ``{species: molar flow}`` (mol/s).
        P: Pressure (Pa).
        H: Target total enthalpy flow (W).
        T_bounds: (T_min, T_max) search bracket (K).

    Returns:
        dict with ``T`` (K), ``vapor_fraction``, ``x``, ``y`` (EOS species
        order), ``H`` (W), ``S`` (W/K), ``residual`` (W), ``bracketed`` and
        ``converged``.

    Example:
        >>> H0 = thermo.stream_enthalpy_flash({"methane": 1.0}, 200.0, 60e5)
        >>> out = flash_PH_eos(thermo, {"methane": 1.0}, 10e5, H0)
        >>> float(out["T"]), float(out["vapor_fraction"])   # ~149 K, ~0.7
    """
    return _flash_spec_jit(thermo, flows, jnp.asarray(P, dtype=float),
                           jnp.asarray(H, dtype=float), "H", tuple(T_bounds))


def flash_PS_eos(
    thermo: CubicThermo,
    flows: dict,
    P: Array | float,
    S: Array | float,
    T_bounds: tuple[float, float] = DEFAULT_T_BOUNDS,
) -> dict:
    """Pressure-entropy flash on a cubic EOS, with the vapour fraction.

    The entropy counterpart of :func:`flash_PH_eos` (the isentropic outlet of
    an expander or compressor). ``S`` is a total entropy flow (W/K) in the
    :meth:`~difflow.thermo.CubicThermo.stream_entropy_flash` convention.

    Args:
        thermo: A :class:`~difflow.thermo.CubicThermo`.
        flows: ``{species: molar flow}`` (mol/s).
        P: Pressure (Pa).
        S: Target total entropy flow (W/K).
        T_bounds: (T_min, T_max) search bracket (K).

    Returns:
        The same dict as :func:`flash_PH_eos` (``residual`` in W/K).

    Example:
        >>> S0 = thermo.stream_entropy_flash({"methane": 1.0}, 300.0, 100e5)
        >>> float(flash_PS_eos(thermo, {"methane": 1.0}, 10e5, S0)["T"])  # ~152 K
    """
    return _flash_spec_jit(thermo, flows, jnp.asarray(P, dtype=float),
                           jnp.asarray(S, dtype=float), "S", tuple(T_bounds))


def _warn_unconverged(unit: str, info: dict) -> None:
    """Warn when a unit's temperature solve failed (concrete values only)."""
    try:
        ok = bool(info["converged"])
    except Exception:  # traced under jit/grad: no concrete flag to report
        return
    if not ok:
        warnings.warn(
            f"{unit}: the (P, H)/(P, S) temperature solve did not converge "
            f"(T = {float(info['T_out']):.2f} K, balance residual "
            f"{float(info['residual']):.3g}). The target is probably outside "
            "T_bounds; widen them. The outlet is not a solution.",
            EOSUnitConvergenceWarning,
            stacklevel=3,
        )


@partial(jax.jit, static_argnames=("thermo", "bounds", "expander"))
def _isentropic_core(thermo, flows, T_in, P_in, P_out, eta, bounds, expander):
    S_in = thermo.stream_entropy_flash(flows, T_in, P_in)
    H_in = thermo.stream_enthalpy_flash(flows, T_in, P_in)
    isen = _flash_spec(thermo, flows, P_out, S_in, "S", bounds)
    H_isen = isen["H"]
    if expander:
        H_out = H_in + eta * (H_isen - H_in)
    else:
        H_out = H_in + (H_isen - H_in) / eta
    out = _flash_spec(thermo, flows, P_out, H_out, "H", bounds)
    return {
        "W": (H_in - H_out) if expander else (H_out - H_in),
        "T_isen": isen["T"],
        "T_out": out["T"],
        "H_in": H_in,
        "H_out": H_out,
        "vapor_fraction": out["vapor_fraction"],
        "vapor_fraction_isen": isen["vapor_fraction"],
        "x": out["x"],
        "y": out["y"],
        "residual": out["residual"],
        "residual_isen": isen["residual"],
        "converged": isen["converged"] & out["converged"],
    }


@partial(jax.jit, static_argnames=("thermo", "bounds"))
def _jtvalve_core(thermo, flows, T_in, P_in, P_out, bounds):
    H_in = thermo.stream_enthalpy_flash(flows, T_in, P_in)
    out = _flash_spec(thermo, flows, P_out, H_in, "H", bounds)
    return {
        "T_out": out["T"],
        "H": H_in,
        "vapor_fraction": out["vapor_fraction"],
        "x": out["x"],
        "y": out["y"],
        "residual": out["residual"],
        "converged": out["converged"],
    }


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
    (common in cryogenic service) is handled correctly: ``info["vapor_fraction"]``
    (< 1 when liquid forms) and ``info["vapor_fraction_isen"]`` report it.

    Info keys: ``W``, ``T_isen``, ``T_out``, ``H_in``, ``H_out``,
    ``vapor_fraction``, ``vapor_fraction_isen``, ``x``, ``y``, ``residual``,
    ``residual_isen``, ``converged``.

    Example:
        >>> out, info = Turboexpander(TurboexpanderParams(P_out=10e5, eta_isentropic=0.85),
        ...                           thermo)(make_stream({"methane": 1.0}, T=300.0, P=100e5))
        >>> float(info["T_isen"]), bool(info["converged"])   # ~152.1 K, True
    """

    symbol = "EXP"
    equations = [
        r"S(T_{\mathrm{isen}}, P_{\mathrm{out}}) = S(T_{\mathrm{in}}, P_{\mathrm{in}})",
        r"H_{\mathrm{out}} = H_{\mathrm{in}} + \eta\,(H_{\mathrm{isen}} - H_{\mathrm{in}})",
        r"\dot{W} = \dot{n}\,(H_{\mathrm{in}} - H_{\mathrm{out}})",
    ]
    assumptions = [
        "Adiabatic, steady state; kinetic and potential energy neglected.",
        "Isentropic efficiency applied to the enthalpy drop.",
        "Real-fluid properties from the supplied cubic EOS.",
    ]
    references = [
        "Smith, Van Ness, Abbott. Introduction to Chemical Engineering Thermodynamics, 7e, Ch. 6-7.",
        "Moran, Shapiro, Boettner, Bailey. Fundamentals of Engineering Thermodynamics, 8e, Ch. 6-9.",
    ]
    parameter_units = {"P_out": "Pa", "eta_isentropic": "-", "T_bounds": "K"}

    def __init__(self, params: TurboexpanderParams, thermo: CubicThermo):
        self.params = params
        self.thermo = thermo

    def __call__(self, inlet: Stream) -> tuple[Stream, dict]:
        flows = _flow_dict(inlet, self.thermo.species_order)
        P_out = jnp.asarray(self.params.P_out)
        eta = jnp.asarray(self.params.eta_isentropic)

        info = _isentropic_core(
            self.thermo, flows, inlet["T"], inlet["P"], P_out, eta,
            tuple(self.params.T_bounds), True,
        )
        _warn_unconverged("Turboexpander", info)
        return make_stream(flows, info["T_out"], P_out), info


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
    shaft work is ``W = H_out - H_in`` (positive, in W). Info keys as for
    :class:`Turboexpander`.
    """

    symbol = "COMP"
    equations = [
        r"S(T_{\mathrm{isen}}, P_{\mathrm{out}}) = S(T_{\mathrm{in}}, P_{\mathrm{in}})",
        r"H_{\mathrm{out}} = H_{\mathrm{in}} + \frac{H_{\mathrm{isen}} - H_{\mathrm{in}}}{\eta}",
        r"\dot{W} = \dot{n}\,(H_{\mathrm{out}} - H_{\mathrm{in}})",
    ]
    assumptions = [
        "Adiabatic, steady state; kinetic and potential energy neglected.",
        "Isentropic efficiency inflates the reversible enthalpy rise.",
        "Real-fluid properties from the supplied cubic EOS.",
    ]
    references = [
        "Smith, Van Ness, Abbott. Introduction to Chemical Engineering Thermodynamics, 7e, Ch. 6-7.",
        "Moran, Shapiro, Boettner, Bailey. Fundamentals of Engineering Thermodynamics, 8e, Ch. 6-9.",
    ]
    parameter_units = {"P_out": "Pa", "eta_isentropic": "-", "T_bounds": "K"}

    def __init__(self, params: CompressorParams, thermo: CubicThermo):
        self.params = params
        self.thermo = thermo

    def __call__(self, inlet: Stream) -> tuple[Stream, dict]:
        flows = _flow_dict(inlet, self.thermo.species_order)
        P_out = jnp.asarray(self.params.P_out)
        eta = jnp.asarray(self.params.eta_isentropic)

        info = _isentropic_core(
            self.thermo, flows, inlet["T"], inlet["P"], P_out, eta,
            tuple(self.params.T_bounds), False,
        )
        _warn_unconverged("Compressor", info)
        return make_stream(flows, info["T_out"], P_out), info


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
    ideal-K model misses entirely. Letting down into the two-phase region gives
    the saturation temperature (pure component) and ``info["vapor_fraction"] < 1``.

    Info keys: ``T_out``, ``H``, ``vapor_fraction``, ``x``, ``y``,
    ``residual``, ``converged``.
    """

    symbol = "JT"
    equations = [
        r"H(T_{\mathrm{out}}, P_{\mathrm{out}}) = H(T_{\mathrm{in}}, P_{\mathrm{in}}) \qquad \text{(isenthalpic)}",
        r"\mu_{\mathrm{JT}} = \left(\frac{\partial T}{\partial P}\right)_H",
    ]
    assumptions = [
        "Adiabatic throttling with no shaft work.",
        "Kinetic and potential energy changes neglected.",
        "Real-fluid enthalpy from the supplied cubic EOS, so the Joule-Thomson temperature change is captured.",
    ]
    references = [
        "Smith, Van Ness, Abbott. Introduction to Chemical Engineering Thermodynamics, 7e, Ch. 6-7.",
    ]
    parameter_units = {"P_out": "Pa", "T_bounds": "K"}

    def __init__(self, params: JTValveParams, thermo: CubicThermo):
        self.params = params
        self.thermo = thermo

    def __call__(self, inlet: Stream) -> tuple[Stream, dict]:
        flows = _flow_dict(inlet, self.thermo.species_order)
        P_out = jnp.asarray(self.params.P_out)

        info = _jtvalve_core(
            self.thermo, flows, inlet["T"], inlet["P"], P_out,
            tuple(self.params.T_bounds),
        )
        _warn_unconverged("JTValve", info)
        return make_stream(flows, info["T_out"], P_out), info


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

    symbol = "CSEP"
    equations = [
        r"F_i^{\mathrm{prod}} = r_i\, F_i^{\mathrm{in}}",
        r"F_i^{\mathrm{res}} = (1 - r_i)\, F_i^{\mathrm{in}}",
    ]
    assumptions = [
        "Specified per-component recovery; no equilibrium is solved.",
        "A black-box surrogate for a separation train, not a physical model.",
        "Outlet temperature and pressure inherited from the inlet.",
    ]
    references = [
        "Seader, Henley, Roper. Separation Process Principles, 3e, Ch. 8 (liquid-liquid extraction).",
    ]
    parameter_units = {"default_recovery": "-"}

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
