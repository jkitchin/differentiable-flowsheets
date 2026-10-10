"""Evaporators: single effect, multiple effect and mechanical vapor recompression.

An evaporator concentrates a solution of a **non-volatile solute** by boiling
off the solvent (water) with steam. Three things set it apart from a
``Heater`` plus a ``Flash``: the solute raises the boiling point
(boiling-point rise, BPR), the heat is supplied through a heating surface with
a given ``U`` and area, and in a multiple-effect train the vapor of one effect
is the heating medium of the next.

Solute representation
---------------------
The solute is carried as ordinary **difflow species** in the stream (flows in
mol/s under ``F_<name>``) whose vapor flow is exactly zero: the vapor outlet
holds the same keys with the solute entries set to ``0.0``. That is the
difflow-native form of "Psat = 0", and it needs no special case in the
gradients (nothing is divided by a vapor pressure or a vapor flow). Mass
fractions are computed from the molar flows and the ``solute_MW`` given on the
params; ``x`` below is always the **solute mass fraction**.

Steam properties
----------------
Water/steam properties are self-contained correlations (below); the module
does not use ``IdealThermo``'s Antoine/Watson water entry because evaporators
run at 20-200 C where a 1-2 % error in the latent heat goes straight into the
steam economy. See :func:`water_saturation_pressure` and friends for what each
correlation is and what was checked.

All functions are JAX-only and differentiable; the multiple-effect solve uses
``optimistix`` (Newton) so it differentiates with respect to U, the feed and the
steam pressure by the implicit function theorem.
"""

import warnings
import dataclasses
from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, get_flows

# =============================================================================
# Water / steam properties (SI: K, Pa, J/kg)
# =============================================================================

#: 0 C in K; the liquid enthalpy reference (h = 0 for saturated liquid at 0 C)
T_REF = 273.15
#: Critical temperature of water (K), IAPWS
T_CRIT_WATER = 647.096
#: Specific gas constant of steam, J/kg/K
R_STEAM = 461.5

# IAPWS-IF97 region-4 saturation-line coefficients (Wagner & Pruss 1993;
# IAPWS-IF97 Eq. 29-31). Checked numerically in tests (round trip, and
# Psat(100 C) = 101.418 kPa).
_N = (
    0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2,
    0.12020824702470e5, -0.32325550322333e7, 0.14915108613530e2,
    -0.48232657361591e4, 0.40511340542057e6, -0.23855557567849,
    0.65017534844798e3,
)

# Saturated-liquid enthalpy h_f(t) = c3 t^3 + c2 t^2 + c1 t + c0, t in C,
# J/kg, and latent heat h_fg = a (1 - T/Tc)^(b + c T/Tc). Both are least-squares
# fits to eight saturated-steam-table points (25, 50, 75, 100, 120, 150, 180,
# 200 C) recalled from a standard steam table (Cengel-type, IAPWS-95 values),
# NOT regenerated from IAPWS here. Max fit residuals over those points: h_f
# 0.04 %, h_fg 0.09 %. Valid 20-200 C; extrapolation outside that is
# unchecked.
_HF_COEFFS = (4.53143312e-03, -6.69677099e-01, 4.21481377e03, -2.36660783e02)
_HFG_COEFFS = (2.92757368e06, 2.64717681e-01, 6.57306406e-02)


def water_saturation_pressure(T: Array | float) -> Array:
    """Saturation pressure of water (Pa) at temperature ``T`` (K).

    IAPWS-IF97 region-4 equation (Wagner & Pruss), 273.15-647 K.
    """
    T = jnp.asarray(T, dtype=jnp.float64)
    th = T + _N[8] / (T - _N[9])
    A = th**2 + _N[0] * th + _N[1]
    B = _N[2] * th**2 + _N[3] * th + _N[4]
    C = _N[5] * th**2 + _N[6] * th + _N[7]
    return (2.0 * C / (-B + jnp.sqrt(B * B - 4.0 * A * C))) ** 4 * 1.0e6


def water_saturation_temperature(P: Array | float) -> Array:
    """Saturation temperature of water (K) at pressure ``P`` (Pa).

    Closed-form inverse of :func:`water_saturation_pressure` (IAPWS-IF97
    backward equation), 611 Pa to 22 MPa.
    """
    P = jnp.asarray(P, dtype=jnp.float64)
    b = (P / 1.0e6) ** 0.25
    E = b * b + _N[2] * b + _N[5]
    F = _N[0] * b * b + _N[3] * b + _N[6]
    G = _N[1] * b * b + _N[4] * b + _N[7]
    D = 2.0 * G / (-F - jnp.sqrt(F * F - 4.0 * E * G))
    return 0.5 * (_N[9] + D - jnp.sqrt((_N[9] + D) ** 2 - 4.0 * (_N[8] + _N[9] * D)))


def water_latent_heat(T: Array | float) -> Array:
    """Latent heat of vaporization of water (J/kg) at saturation temperature ``T`` (K).

    Fit of the form ``a (1 - Tr)^(b + c Tr)`` to steam-table points; see the
    module notes (valid 20-200 C, about 0.1 %).
    """
    Tr = jnp.asarray(T, dtype=jnp.float64) / T_CRIT_WATER
    a, b, c = _HFG_COEFFS
    return a * (1.0 - Tr) ** (b + c * Tr)


def water_liquid_enthalpy(T: Array | float) -> Array:
    """Saturated-liquid enthalpy of water (J/kg), zero at 0 C (valid 20-200 C)."""
    t = jnp.asarray(T, dtype=jnp.float64) - T_REF
    c3, c2, c1, c0 = _HF_COEFFS
    return ((c3 * t + c2) * t + c1) * t + c0


def water_vapor_enthalpy(T: Array | float) -> Array:
    """Saturated-vapor enthalpy of water (J/kg) at saturation temperature ``T`` (K)."""
    return water_liquid_enthalpy(T) + water_latent_heat(T)


# =============================================================================
# Boiling-point rise
# =============================================================================


class UnverifiedDataWarning(UserWarning):
    """A built-in correlation or table that was not checked against its source.

    Raised (as a warning) when a built-in boiling-point-rise table is selected.
    Turn it into an error with ``warnings.simplefilter("error",
    UnverifiedDataWarning)`` to refuse to run on unverified data, or silence
    it once the numbers have been replaced by your own.
    """


#: Built-in BPR tables: solute -> (mass fractions, BPR at 1 atm in K, note).
#: EVERY NUMBER HERE IS AN APPROXIMATE RECOLLECTION, not a transcription of the
#: cited chart (McCabe, Smith & Harriott, Unit Operations of Chemical
#: Engineering, Dühring chart for NaOH; Perry's Chemical Engineers' Handbook,
#: boiling points of aqueous solutions). They are good to the shape of the
#: curve (a few K at most for NaOH below 50 %), and they are NOT a substitute
#: for reading the chart. Pass your own table through ``duhring_x`` /
#: ``duhring_a`` / ``duhring_b`` or a ``bpr_fn`` when the number matters.
DUHRING_TABLES: dict[str, tuple[tuple[float, ...], tuple[float, ...], str]] = {
    "naoh": (
        (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70),
        (0.0, 2.6, 7.0, 14.5, 27.0, 43.0, 63.0, 88.0),
        "NaOH: only the ~50 wt % point (about 43 K above water at 1 atm, "
        "boiling near 143 C) is a firm recollection; the others are "
        "interpolated shape and unverified",
    ),
    "nacl": (
        (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.264),
        (0.0, 0.9, 1.9, 3.1, 4.6, 7.0, 8.7),
        "NaCl: the saturated solution (26.4 wt %) boiling at about 108.7 C is "
        "the firm point; the others are unverified",
    ),
}

#: Polynomial BPR (K versus solute mass fraction, coefficients ascending) for
#: sucrose solutions, the correlation used in Geankoplis, Transport Processes
#: and Separation Process Principles, Example 8.4-1: BPR = 1.78 x + 6.22 x^2,
#: taken as independent of pressure. Recalled from the textbook, unverified.
SUCROSE_BPR_COEFFS = (0.0, 1.78, 6.22)

#: Reference temperature of the built-in Dühring tables (1 atm saturation)
_T_1ATM = 373.15


def _duhring_lines(x_tab, bpr_1atm) -> tuple[Array, Array]:
    """Linear Dühring lines ``T_soln = a + b T_water`` from a 1 atm BPR table.

    Each line passes through (373.15 K, 373.15 + BPR) and has the slope
    ``b = 1 + BPR d ln(T^2/lambda)/dT``, i.e. the colligative scaling
    ``BPR ~ R T^2 / lambda(T)`` carried to other pressures. The slope is
    evaluated with the Watson exponent 0.38 for lambda. This is an
    approximation to the chart's measured slopes.
    """
    bpr = jnp.asarray(bpr_1atm, dtype=jnp.float64)
    slope = 2.0 / _T_1ATM + 0.38 / (T_CRIT_WATER - _T_1ATM)
    b = 1.0 + bpr * slope
    a = (_T_1ATM + bpr) - b * _T_1ATM
    return a, b


def boiling_point_rise(
    x_solute: Array | float,
    P: Array | float,
    model: str = "ideal",
    *,
    solute_MW: float | None = None,
    MW_solvent: float = 18.015,
    vant_hoff_i: float = 1.0,
    coeffs=None,
    duhring=None,
    bpr_fn: Callable | None = None,
) -> Array:
    """Boiling-point rise (K) of a solution at vapor-space pressure ``P``.

    Args:
        x_solute: Solute mass fraction (-).
        P: Pressure of the vapor space (Pa).
        model: ``"ideal"`` (Raoult's law with unit activity: the solution
            boils where ``x_w Psat(T) = P``), ``"colligative"``
            (``dT = R Tw^2 i m / lambda``, the dilute limit), ``"duhring"``
            (user lines, ``duhring=(x, a, b)``), ``"naoh"`` / ``"nacl"``
            (built-in tables, **unverified**, see :data:`DUHRING_TABLES`),
            ``"sucrose"`` (:data:`SUCROSE_BPR_COEFFS`, unverified),
            ``"polynomial"`` (``coeffs``, K versus ``x``, ascending powers),
            ``"callable"`` (``bpr_fn(x, Tw, P)``) or ``"none"``.
        solute_MW: Solute molar mass (g/mol); needed by ``ideal`` and
            ``colligative``.
        MW_solvent: Solvent molar mass (g/mol).
        vant_hoff_i: Moles of particles per mole of solute (2 for fully
            dissociated NaOH or NaCl), used by ``ideal`` and ``colligative``.
        coeffs: Polynomial coefficients for ``"polynomial"``.
        duhring: ``(x, a, b)`` arrays for ``"duhring"``: the Dühring line
            ``T_soln = a(x) + b(x) T_water`` at each tabulated mass fraction,
            linearly interpolated (flat beyond the table).
        bpr_fn: Callable for ``"callable"``.

    Returns:
        Boiling-point rise in K, evaluated at the water saturation
        temperature of ``P``.

    Note:
        The tabulated models are piecewise linear in ``x``, so their
        derivative is discontinuous at the table points.
    """
    x = jnp.asarray(x_solute, dtype=jnp.float64)
    P = jnp.asarray(P, dtype=jnp.float64)
    Tw = water_saturation_temperature(P)
    m = (model or "none").lower()

    if m == "none":
        return jnp.zeros_like(x + P)
    if m in ("ideal", "colligative"):
        if solute_MW is None:
            raise ValueError(f"bpr_model {model!r} needs solute_MW")
        n_s = vant_hoff_i * x / solute_MW
        n_w = (1.0 - x) / MW_solvent
        if m == "ideal":
            x_w = n_w / (n_w + n_s)
            return water_saturation_temperature(P / x_w) - Tw
        molality = n_s / ((1.0 - x) / 1000.0)
        return 8.314462618 * Tw**2 * molality / (water_latent_heat(Tw))
    if m in DUHRING_TABLES:
        x_tab, bpr_tab, _ = DUHRING_TABLES[m]
        a, b = _duhring_lines(x_tab, bpr_tab)
        xt = jnp.asarray(x_tab, dtype=jnp.float64)
        return jnp.interp(x, xt, a) + (jnp.interp(x, xt, b) - 1.0) * Tw
    if m == "duhring":
        if duhring is None:
            raise ValueError("bpr_model 'duhring' needs duhring=(x, a, b)")
        xt, a, b = (jnp.asarray(v, dtype=jnp.float64) for v in duhring)
        return jnp.interp(x, xt, a) + (jnp.interp(x, xt, b) - 1.0) * Tw
    if m == "sucrose" or m == "polynomial":
        c = SUCROSE_BPR_COEFFS if m == "sucrose" else coeffs
        if c is None:
            raise ValueError("bpr_model 'polynomial' needs coeffs")
        out = jnp.zeros_like(x)
        for k in reversed(range(len(c))):
            out = out * x + c[k]
        return out
    if m == "callable":
        if bpr_fn is None:
            raise ValueError("bpr_model 'callable' needs bpr_fn")
        return jnp.asarray(bpr_fn(x, Tw, P), dtype=jnp.float64)
    raise ValueError(
        f"unknown bpr_model {model!r}; choose ideal, colligative, duhring, "
        "naoh, nacl, sucrose, polynomial, callable or none"
    )


_UNVERIFIED_MODELS = ("naoh", "nacl", "sucrose")


# =============================================================================
# Solution model shared by every evaporator
# =============================================================================


class _Solution:
    """Species bookkeeping, BPR and solution enthalpy for one set of params."""

    def __init__(self, p, bpr_fn=None, enthalpy_fn=None):
        self.p = p
        self.solvent = p.solvent
        self.solutes = list(p.solute_species)
        if len(self.solutes) != len(p.solute_MW):
            raise ValueError("solute_species and solute_MW must have the same length")
        if not self.solutes:
            raise ValueError("at least one solute species is required")
        self.MW_w = float(p.MW_solvent)
        self.MW_s = [float(m) for m in p.solute_MW]
        self.bpr_fn = bpr_fn
        self.enthalpy_fn = enthalpy_fn
        model = (p.bpr_model or "ideal").lower()
        if bpr_fn is not None and model == "ideal":
            model = "callable"
        self.model = model
        if model in _UNVERIFIED_MODELS:
            note = (DUHRING_TABLES[model][2] if model in DUHRING_TABLES
                    else "sucrose BPR = 1.78 x + 6.22 x^2 (Geankoplis Ex. 8.4-1), "
                         "recalled, not checked against the book")
            warnings.warn(
                f"bpr_model={model!r} uses a built-in correlation that has not been "
                f"verified against its source ({note}). Supply duhring=/coeffs= or "
                "a bpr_fn for a number you can cite.",
                UnverifiedDataWarning, stacklevel=4,
            )
        # effective molar mass of the solute mixture on a mass basis, for the
        # colligative models (mass-weighted harmonic: moles per kg of solute)
        self._mol_per_kg_solute = None

    # -- streams ------------------------------------------------------------

    def feed_state(self, feed: Stream) -> dict:
        """Mass flows (kg/s), solute fraction and temperature of a feed stream."""
        flows = get_flows(feed)
        extra = set(flows) - {self.solvent} - set(self.solutes)
        if extra:
            raise ValueError(
                f"the feed carries species {sorted(extra)} that are neither the "
                f"solvent {self.solvent!r} nor listed in solute_species"
            )
        if self.solvent not in flows:
            raise ValueError(f"the feed has no solvent species {self.solvent!r}")
        m_w = flows[self.solvent] * self.MW_w / 1000.0
        m_s_each = [flows.get(s, 0.0) * mw / 1000.0
                    for s, mw in zip(self.solutes, self.MW_s)]
        m_s = sum(m_s_each)
        F = m_w + m_s
        return {"F": F, "m_s": m_s, "x": m_s / F, "T": jnp.asarray(feed["T"]),
                "P": jnp.asarray(feed["P"]), "solute_mol": [flows.get(s, 0.0) for s in self.solutes]}

    def _mixed_MW(self, fs) -> Array:
        """Mass-averaged solute molar mass (g/mol) of the feed's solute mix."""
        mol = sum(fs["solute_mol"])
        return (fs["m_s"] * 1000.0) / mol

    def make_streams(self, fs, L, x_L, T, P, V, T_vap, P_vap):
        """Concentrate and vapor streams from kg/s flows."""
        conc, vap = {}, {}
        w_L = L * (1.0 - x_L) * 1000.0 / self.MW_w
        w_V = V * 1000.0 / self.MW_w
        conc["F_" + self.solvent] = w_L
        vap["F_" + self.solvent] = w_V
        for s, mol in zip(self.solutes, fs["solute_mol"]):
            conc["F_" + s] = jnp.asarray(mol)
            vap["F_" + s] = jnp.zeros(())
        conc["T"], conc["P"] = T, P
        vap["T"], vap["P"] = T_vap, P_vap
        return conc, vap

    # -- properties ---------------------------------------------------------

    def bpr(self, x, P, fs) -> Array:
        p = self.p
        coeffs = list(p.bpr_coeffs) if p.bpr_coeffs is not None else None
        duhring = None
        if self.model == "duhring":
            duhring = (p.duhring_x, p.duhring_a, p.duhring_b)
        return boiling_point_rise(
            x, P, self.model, solute_MW=self._mixed_MW(fs), MW_solvent=self.MW_w,
            vant_hoff_i=p.vant_hoff_i, coeffs=coeffs, duhring=duhring,
            bpr_fn=self.bpr_fn,
        )

    def h_liquid(self, T, x) -> Array:
        """Specific enthalpy of the solution (J/kg), 0 at 0 C for pure liquid water."""
        if self.enthalpy_fn is not None:
            return jnp.asarray(self.enthalpy_fn(T, x), dtype=jnp.float64)
        p = self.p
        if p.cp_coeffs is not None:
            cp = jnp.zeros_like(x)
            for k in reversed(range(len(p.cp_coeffs))):
                cp = cp * x + p.cp_coeffs[k]
            return cp * (T - T_REF)
        return (1.0 - x) * water_liquid_enthalpy(T) + x * p.Cp_solute * (T - T_REF)

    def H_vapor(self, Tw, T_soln) -> Array:
        """Vapor leaving at the solution temperature: saturated plus superheat."""
        return water_vapor_enthalpy(Tw) + self.p.Cp_vapor * (T_soln - Tw)


def _steam_temperature(p, steam: Stream | None) -> Array:
    """Saturation temperature of the heating steam (K)."""
    if steam is not None:
        return water_saturation_temperature(steam["P"])
    if p.steam_P is not None:
        return water_saturation_temperature(p.steam_P)
    if p.steam_T is not None:
        return jnp.asarray(p.steam_T, dtype=jnp.float64)
    raise ValueError("specify steam_P, steam_T, or pass a steam stream")


_X_CAP = 0.995  # solute mass fraction above which the model is clipped (not physical)


# =============================================================================
# Single effect
# =============================================================================


@dataclass
class EvaporatorParams(ParamsMixin):
    """Parameters for a single-effect evaporator.

    Attributes:
        solute_species: Names of the non-volatile solute species in the stream.
        solute_MW: Molar mass of each solute species (g/mol).
        solvent: Name of the volatile solvent species.
        MW_solvent: Molar mass of the solvent (g/mol).
        U: Overall heat transfer coefficient (W/m^2/K).
        A: Heat transfer area (m^2). Give it for *rating* (product
            concentration is computed).
        x_product: Product solute mass fraction (-). Give it for *design*
            (area is computed). Exactly one of ``A`` and ``x_product``.
        P_vapor_space: Pressure of the vapor space (Pa).
        steam_P: Pressure of the saturated heating steam (Pa).
        steam_T: Saturation temperature of the heating steam (K), instead of
            ``steam_P``. A steam stream passed to the call overrides both.
        bpr_model: Boiling-point-rise model: ideal, colligative, duhring, naoh,
            nacl, sucrose, polynomial, callable or none. See
            :func:`boiling_point_rise`; naoh, nacl and sucrose are built-in
            *unverified* data.
        vant_hoff_i: Particles per solute formula unit (ideal, colligative).
        bpr_coeffs: Polynomial BPR coefficients (K versus mass fraction,
            ascending), for bpr_model polynomial.
        duhring_x: Mass fractions of a user Dühring table (bpr_model duhring).
        duhring_a: Intercepts a(x) of the lines T_soln = a + b T_water (K).
        duhring_b: Slopes b(x) of the lines (-).
        Cp_solute: Solute heat capacity (J/kg/K) for the default solution
            enthalpy (mixing rule, no heat of dilution).
        cp_coeffs: Solution heat capacity polynomial in mass fraction
            (J/kg/K, ascending); replaces the mixing rule when given.
        Cp_vapor: Vapor heat capacity (J/kg/K), for the superheat of the
            vapor leaving at the boiling-point-raised temperature.
    """

    solute_species: list[str]
    solute_MW: list[float]
    solvent: str = "water"
    MW_solvent: float = 18.015
    U: float = 2000.0
    A: float | None = None
    x_product: float | None = None
    P_vapor_space: float = 101325.0
    steam_P: float | None = None
    steam_T: float | None = None
    bpr_model: str = "ideal"
    vant_hoff_i: float = 1.0
    bpr_coeffs: list[float] | None = None
    duhring_x: list[float] | None = None
    duhring_a: list[float] | None = None
    duhring_b: list[float] | None = None
    Cp_solute: float = 1500.0
    cp_coeffs: list[float] | None = None
    Cp_vapor: float = 1880.0


_COMMON_UNITS = {
    "solute_MW": "g/mol", "MW_solvent": "g/mol", "U": "W/m^2/K", "A": "m^2",
    "x_product": "kg/kg", "P_vapor_space": "Pa", "steam_P": "Pa", "steam_T": "K",
    "vant_hoff_i": "-", "bpr_coeffs": "K", "duhring_x": "kg/kg", "duhring_a": "K",
    "duhring_b": "-", "Cp_solute": "J/kg/K", "cp_coeffs": "J/kg/K",
    "Cp_vapor": "J/kg/K",
}
_COMMON_SYMBOLS = {
    "U": "U", "A": "A", "x_product": "x_P", "P_vapor_space": "P_v", "steam_P": "P_s",
    "steam_T": "T_s", "vant_hoff_i": "i",
}
_COMMON_ASSUMPTIONS = [
    "The solute is non-volatile: it has zero vapor flow (the vapor outlet carries "
    "the solute species at 0 mol/s).",
    "Steady state; saturated steam, condensate leaves saturated at the steam "
    "temperature (no condensate subcooling or flash).",
    "The vapor leaves at the solution temperature, i.e. superheated by the "
    "boiling-point rise; heat of dilution is zero unless an enthalpy_fn is given.",
    "Perfectly mixed boiling liquid: the product leaves at the boiling "
    "temperature of the product concentration (no concentration profile).",
    "Pressure effects on liquid enthalpy are neglected; no entrainment, no "
    "hydrostatic head, no non-condensables.",
    "Water properties are correlations valid 20-200 C (see module notes).",
]


def _only(params_cls, table: dict) -> dict:
    """The entries of ``table`` that name a field of ``params_cls``."""
    names = {f.name for f in dataclasses.fields(params_cls)}
    return {k: v for k, v in table.items() if k in names}


class Evaporator:
    """Single-effect, steam-heated evaporator with boiling-point rise.

    Two modes, picked by which of ``A`` and ``x_product`` is set:

    * **design** (``x_product`` given): steam, vapor, duty and the required
      area come out in closed form;
    * **rating** (``A`` given): the product concentration is solved (1-D
      bisection with implicit differentiation, so it is differentiable).

    The call is ``evaporator(feed, steam=None)``; heating steam comes from
    ``steam_P`` / ``steam_T`` on the params unless a steam stream is passed.

    Example:
        >>> from difflow import make_stream
        >>> from difflow.units import Evaporator, EvaporatorParams
        >>> p = EvaporatorParams(solute_species=["NaOH"], solute_MW=[40.0],
        ...                      U=1500.0, x_product=0.4, bpr_model="ideal",
        ...                      vant_hoff_i=2.0, P_vapor_space=30e3, steam_P=300e3)
        >>> feed = make_stream({"water": 400.0, "NaOH": 25.0}, T=330.0, P=1e5)
        >>> conc, vap, info = Evaporator(p)(feed)
    """

    symbol = "Evaporator"
    equations = [
        r"x_F F = x_P L, \qquad V = F - L",
        r"T = T_w(P_v) + \mathrm{BPR}(x_P, P_v), \qquad T_w = T_\mathrm{sat}(P_v)",
        r"Q = L\,h_L(T, x_P) + V\,H_V(T, P_v) - F\,h_F(T_F, x_F), "
        r"\qquad H_V = h_g(T_w) + c_{p,v}\,(T - T_w)",
        r"S\,\lambda_s = Q, \qquad \lambda_s = h_g(T_s) - h_f(T_s)",
        r"Q = U A\,(T_s - T), \qquad \mathrm{economy} = V/S",
    ]
    assumptions = _COMMON_ASSUMPTIONS
    references = [
        "McCabe, Smith, Harriott. Unit Operations of Chemical Engineering, Ch. 16 "
        "(Evaporation): single-effect energy balance, boiling-point rise, Duhring rule.",
        "Geankoplis. Transport Processes and Separation Process Principles, Ch. 8 "
        "(Evaporation).",
        "IAPWS-IF97 region-4 saturation equations (Wagner & Pruss) for Psat(T), Tsat(P).",
    ]
    parameter_symbols = _only(EvaporatorParams, _COMMON_SYMBOLS)
    parameter_units = _only(EvaporatorParams, _COMMON_UNITS)
    numerical_method = (
        "Design mode is closed form. Rating mode solves the evaporated fraction "
        "with optimistix Bisection (implicit-function-theorem gradients)."
    )

    def __init__(self, params: EvaporatorParams, *, bpr_fn: Callable | None = None,
                 enthalpy_fn: Callable | None = None):
        """Initialize the evaporator.

        Args:
            params: Evaporator parameters.
            bpr_fn: Optional ``bpr_fn(x, T_water, P) -> K`` boiling-point rise;
                selects ``bpr_model='callable'`` when the model is left at
                its default.
            enthalpy_fn: Optional ``enthalpy_fn(T, x) -> J/kg`` solution
                enthalpy (zero at 0 C for pure water), replacing the Cp-based
                default. Use it for a heat of dilution or an enthalpy-
                concentration chart (NaOH is the classic case).
        """
        self.params = params
        self._sol = _Solution(params, bpr_fn, enthalpy_fn)
        p = params
        if (p.A is None) == (p.x_product is None):
            raise ValueError("give exactly one of A (rating) and x_product (design)")

    # -- core ---------------------------------------------------------------

    def _design_state(self, fs, x_P, Ts, P_v, U, hF):
        """Everything from a known product concentration."""
        sol, p = self._sol, self.params
        Tw = water_saturation_temperature(P_v)
        L = fs["m_s"] / x_P
        V = fs["F"] - L
        bpr = sol.bpr(x_P, P_v, fs)
        T = Tw + bpr
        HV = sol.H_vapor(Tw, T)
        hL = sol.h_liquid(T, x_P)
        Q = L * hL + V * HV - fs["F"] * hF
        lam_s = water_latent_heat(Ts)
        S = Q / lam_s
        A = Q / (U * (Ts - T))
        return {"Tw": Tw, "L": L, "V": V, "bpr": bpr, "T": T, "Q": Q, "S": S,
                "A": A, "lam_s": lam_s, "HV": HV, "hL": hL}

    def __call__(self, feed: Stream, steam: Stream | None = None
                 ) -> tuple[Stream, Stream, dict[str, Array]]:
        """Run the evaporator.

        Args:
            feed: Feed stream: solvent and solute species in mol/s, T, P.
            steam: Optional heating-steam stream; only its pressure is used
                (saturated steam), its flow is an output.

        Returns:
            concentrate: Product solution (solute flows unchanged, T = boiling
                temperature, P = vapor-space pressure).
            vapor: Solvent vapor leaving the effect (solutes at 0 mol/s), at
                the solution temperature (superheated by the BPR).
            info: Dictionary with ``steam`` (kg/s), ``steam_economy``
                (vapor/steam), ``Q`` (W), ``T_soln`` (K), ``BPR`` (K), ``A``
                (m^2), ``x_product``, ``V`` and ``L`` (kg/s), ``T_steam``,
                ``T_sat_vapor``, ``driving_force`` (K) and the closure
                residuals ``energy_balance_error`` (W) and
                ``mass_balance_error`` (kg/s).
        """
        sol, p = self._sol, self.params
        fs = sol.feed_state(feed)
        Ts = _steam_temperature(p, steam)
        P_v = jnp.asarray(p.P_vapor_space, dtype=jnp.float64)
        hF = sol.h_liquid(fs["T"], fs["x"])
        U = jnp.asarray(p.U, dtype=jnp.float64)

        if p.x_product is not None:
            x_P = jnp.asarray(p.x_product, dtype=jnp.float64)
            st = self._design_state(fs, x_P, Ts, P_v, U, hF)
            converged = jnp.asarray(True)
        else:
            A = jnp.asarray(p.A, dtype=jnp.float64)
            y_hi = 1.0 - fs["x"] / _X_CAP

            def resid(y, args):
                x_P = fs["x"] / (1.0 - y)
                s = self._design_state(fs, x_P, Ts, P_v, U, hF)
                # scaled by F*2.2e6 W so the bisection's absolute tolerance is relative
                return (U * A * (Ts - s["T"]) - s["Q"]) / (fs["F"] * 2.2e6)

            sol_y = optx.root_find(
                resid, optx.Bisection(rtol=1e-12, atol=1e-12),
                jnp.asarray(0.5 * y_hi), None,
                options={"lower": jnp.asarray(1e-9), "upper": jnp.asarray(y_hi)},
                throw=False, max_steps=200,
            )
            y = sol_y.value
            x_P = fs["x"] / (1.0 - y)
            st = self._design_state(fs, x_P, Ts, P_v, U, hF)
            converged = sol_y.result == optx.RESULTS.successful

        T = st["T"]
        conc, vap = sol.make_streams(fs, st["L"], x_P, T, P_v, st["V"], T, P_v)
        # closure, from the outlets and the condensate: steam in at h_g(Ts), out
        # as saturated condensate at h_f(Ts)
        H_in = st["S"] * water_vapor_enthalpy(Ts) + fs["F"] * hF
        H_out = (st["S"] * water_liquid_enthalpy(Ts) + st["L"] * st["hL"]
                 + st["V"] * st["HV"])
        info = {
            "steam": st["S"],
            "steam_economy": st["V"] / st["S"],
            "Q": st["Q"],
            "T_soln": T,
            "BPR": st["bpr"],
            "A": st["A"] if p.x_product is not None else jnp.asarray(p.A),
            "U": U,
            "x_product": x_P,
            "V": st["V"],
            "L": st["L"],
            "T_steam": Ts,
            "T_sat_vapor": st["Tw"],
            "driving_force": Ts - T,
            "converged": converged,
            "energy_balance_error": H_in - H_out,
            "mass_balance_error": fs["F"] - st["L"] - st["V"],
        }
        return conc, vap, info

    # -- optional EO hook is deliberately absent: the rating solve is internal.


# =============================================================================
# Multiple effect
# =============================================================================


@dataclass
class MultiEffectEvaporatorParams(ParamsMixin):
    """Parameters for a multiple-effect evaporator.

    Attributes:
        solute_species: Names of the non-volatile solute species in the stream.
        solute_MW: Molar mass of each solute species (g/mol).
        n_effects: Number of effects N (effect 1 receives the live steam).
        feed: Feed arrangement: forward, backward or parallel.
        solvent: Name of the volatile solvent species.
        MW_solvent: Molar mass of the solvent (g/mol).
        U: Overall heat transfer coefficient per effect (W/m^2/K); a single
            value is used for every effect.
        A: Heat transfer area per effect (m^2), for *rating*; a single value is
            used for every effect.
        x_product: Final product solute mass fraction (-), for *design*: the
            temperatures, steam rate and the common area of all effects are
            solved so that every effect has the same area.
        P_vapor_space: Pressure of the last effect's vapor space (Pa).
        steam_P: Pressure of the saturated heating steam (Pa).
        steam_T: Saturation temperature of the heating steam (K).
        feed_split: Parallel feed, rating mode only: fraction of the feed to
            each effect (default equal). In design mode the split is solved.
        bpr_model: Boiling-point-rise model, as for the single effect.
        vant_hoff_i: Particles per solute formula unit.
        bpr_coeffs: Polynomial BPR coefficients (K versus mass fraction).
        duhring_x: Mass fractions of a user Dühring table.
        duhring_a: Intercepts of the Dühring lines (K).
        duhring_b: Slopes of the Dühring lines (-).
        Cp_solute: Solute heat capacity (J/kg/K).
        cp_coeffs: Solution heat capacity polynomial in mass fraction (J/kg/K).
        Cp_vapor: Vapor heat capacity (J/kg/K).
        max_steps: Newton iteration limit of the multi-effect solve.
    """

    solute_species: list[str]
    solute_MW: list[float]
    n_effects: int = 3
    feed: str = "forward"
    solvent: str = "water"
    MW_solvent: float = 18.015
    U: list[float] = field(default_factory=lambda: [2000.0])
    A: list[float] | None = None
    x_product: float | None = None
    P_vapor_space: float = 101325.0
    steam_P: float | None = None
    steam_T: float | None = None
    feed_split: list[float] | None = None
    bpr_model: str = "ideal"
    vant_hoff_i: float = 1.0
    bpr_coeffs: list[float] | None = None
    duhring_x: list[float] | None = None
    duhring_a: list[float] | None = None
    duhring_b: list[float] | None = None
    Cp_solute: float = 1500.0
    cp_coeffs: list[float] | None = None
    Cp_vapor: float = 1880.0
    max_steps: int = 100


_MULTI_UNITS = dict(_COMMON_UNITS, feed_split="-", n_effects="-", max_steps="-")


class MultiEffectEvaporator:
    """Multiple-effect evaporator: forward, backward or parallel feed.

    The vapor of effect *i* condenses in effect *i+1*; effect 1 is heated by
    live steam and the last effect vents to the condenser at
    ``P_vapor_space``. Each effect has its own boiling-point rise, ``U`` and
    energy balance. ``Q_i = U_i A_i (T_h,i - T_i)`` with ``T_h,i`` the
    saturation temperature of the vapor heating effect *i*.

    * **Design** (``x_product`` given): unknown effect temperatures, steam and
      one common area ``A`` are solved so that every effect has the same area
      (the classic textbook iteration, done as one Newton solve).
    * **Rating** (``A`` given): areas are fixed, the product concentration
      follows.

    ``MultiEffectEvaporator`` with ``n_effects=1`` agrees with
    :class:`Evaporator`.

    The call is ``mee(feed, steam=None)``.
    """

    symbol = "Multi-effect evaporator"
    equations = [
        r"L_i = L_{i,\mathrm{in}} - V_i,\qquad x_i = m_s / L_i",
        r"T_i = T_{w,i} + \mathrm{BPR}(x_i, P_i),\qquad P_i = P_\mathrm{sat}(T_{w,i})",
        r"Q_i = m_{h,i}\,\lambda_{h,i} = L_i h_L(T_i,x_i) + V_i H_V(T_i) - "
        r"L_{i,\mathrm{in}} h_{L}(T_{i,\mathrm{in}}, x_{i,\mathrm{in}})",
        r"\lambda_{h,1}=\lambda(T_s),\quad \lambda_{h,i}=\lambda(T_{w,i-1}) + "
        r"c_{p,v}\,\mathrm{BPR}_{i-1}\quad (i>1)",
        r"Q_i = U_i A_i (T_{h,i} - T_i),\qquad \mathrm{economy} = \sum_i V_i / S",
        r"A_1 = \dots = A_N \quad (\text{design})",
    ]
    assumptions = _COMMON_ASSUMPTIONS + [
        "Condensate from every effect leaves saturated; it is not flashed into the "
        "next effect.",
        "No vapor-line pressure drop or heat loss between effects; no "
        "feed-preheating or flashing credits beyond the effect energy balances.",
        "Parallel feed: the concentrate is the flow-weighted mixture of the "
        "effect products (temperature mixed by mass).",
    ]
    references = Evaporator.references + [
        "Equal-area multiple-effect design (iterative temperature-drop split) as in "
        "Geankoplis Ch. 8.4 and McCabe Ch. 16; here solved simultaneously by Newton.",
    ]
    parameter_symbols = _only(MultiEffectEvaporatorParams, dict(_COMMON_SYMBOLS, n_effects="N"))
    parameter_units = _only(MultiEffectEvaporatorParams, _MULTI_UNITS)
    numerical_method = (
        "Newton (optimistix) on the stacked effect energy balances, rate "
        "equations and overall solute balance; implicit-function-theorem gradients."
    )

    def __init__(self, params: MultiEffectEvaporatorParams, *,
                 bpr_fn: Callable | None = None, enthalpy_fn: Callable | None = None):
        """Initialize the multiple-effect evaporator.

        Args:
            params: Parameters.
            bpr_fn: Optional ``bpr_fn(x, T_water, P) -> K``.
            enthalpy_fn: Optional ``enthalpy_fn(T, x) -> J/kg`` solution enthalpy.
        """
        self.params = params
        self._sol = _Solution(params, bpr_fn, enthalpy_fn)
        p = params
        if p.feed not in ("forward", "backward", "parallel"):
            raise ValueError("feed must be 'forward', 'backward' or 'parallel'")
        if int(p.n_effects) < 1:
            raise ValueError("n_effects must be >= 1")
        if (p.A is None) == (p.x_product is None):
            raise ValueError("give exactly one of A (rating) and x_product (design)")

    # -- helpers --------------------------------------------------------------

    def _vector(self, value, N, name):
        v = jnp.atleast_1d(jnp.asarray(value, dtype=jnp.float64))
        if v.shape[0] == 1:
            v = jnp.broadcast_to(v, (N,))
        if v.shape[0] != N:
            raise ValueError(f"{name} must have 1 or n_effects={N} entries")
        return v

    def _effects(self, fs, hF, Ts, V, Tw_all, Fin, U):
        """Per-effect state for given vapor flows and water temperatures.

        Returns lists/arrays indexed by effect. ``Fin`` is the parallel feed
        split (kg/s), used only for ``feed='parallel'``.
        """
        sol, p = self._sol, self.params
        N = int(p.n_effects)
        arr = p.feed
        m_s = fs["m_s"]
        P_eff = water_saturation_pressure(Tw_all)
        Lout = [None] * N
        xout = [None] * N
        T = [None] * N
        bpr = [None] * N
        Lin = [None] * N
        hin = [None] * N
        order = list(range(N)) if arr in ("forward", "parallel") else list(range(N - 1, -1, -1))
        prev = None
        for i in order:
            if arr == "parallel":
                Lin[i] = Fin[i]
                hin[i] = hF
            elif prev is None:
                Lin[i] = fs["F"]
                hin[i] = hF
            else:
                Lin[i] = Lout[prev]
                hin[i] = sol.h_liquid(T[prev], xout[prev])
            if arr == "parallel":
                Lout[i] = Lin[i] - V[i]
                xout[i] = Lin[i] * fs["x"] / jnp.maximum(Lout[i], Lin[i] * fs["x"] / _X_CAP)
            else:
                Lout[i] = Lin[i] - V[i]
                xout[i] = m_s / jnp.maximum(Lout[i], m_s / _X_CAP)
            bpr[i] = sol.bpr(xout[i], P_eff[i], fs)
            T[i] = Tw_all[i] + bpr[i]
            prev = i
        # heating
        Th = [Ts] + [Tw_all[i - 1] for i in range(1, N)]
        return {"P": P_eff, "Lout": Lout, "x": xout, "T": T, "bpr": bpr, "Lin": Lin,
                "hin": hin, "Th": Th}

    def _residual(self, z, args):
        """Scaled residual vector of the effect equations."""
        (fs, hF, Ts, Tw_N, U, A_fix, x_P, split) = args
        p, sol = self.params, self._sol
        N = int(p.n_effects)
        design = p.x_product is not None
        F = fs["F"]
        V = z[:N] * F
        S = z[N] * F
        Tw_in = z[N + 1:N + N] * 100.0
        k = 2 * N
        if p.feed == "parallel" and design:
            Fin = z[k:k + N] * F
            k += N
        else:
            Fin = split * F if p.feed == "parallel" else None
        A = (z[k] * A_fix) if design else A_fix
        Tw_all = jnp.concatenate([Tw_in, jnp.atleast_1d(Tw_N)])
        e = self._effects(fs, hF, Ts, V, Tw_all, Fin, U)
        lam1 = water_latent_heat(Ts)
        Escale = F * 2.2e6
        res = []
        for i in range(N):
            if i == 0:
                Qh = S * lam1
            else:
                Qh = V[i - 1] * (water_latent_heat(Tw_all[i - 1])
                                 + p.Cp_vapor * e["bpr"][i - 1])
            HV = sol.H_vapor(Tw_all[i], e["T"][i])
            hL = sol.h_liquid(e["T"][i], e["x"][i])
            Qb = e["Lout"][i] * hL + V[i] * HV - e["Lin"][i] * e["hin"][i]
            Ai = A[i] if (not design) else A
            res.append((Qh - Qb) / Escale)
            res.append((Qh - U[i] * Ai * (e["Th"][i] - e["T"][i])) / Escale)
        if design:
            if p.feed == "parallel":
                res.append(jnp.sum(Fin) / F - 1.0)
                for i in range(N):
                    res.append((V[i] - Fin[i] * (1.0 - fs["x"] / x_P)) / F)
            else:
                res.append(jnp.sum(V) / F - (1.0 - fs["x"] / x_P))
        return jnp.stack(res)

    def _initial_guess(self, fs, hF, Ts, Tw_N, U, A_fix, x_P, split):
        p, sol = self.params, self._sol
        N = int(p.n_effects)
        F = fs["F"]
        design = p.x_product is not None
        lam0 = water_latent_heat(Ts)
        P_N = water_saturation_pressure(Tw_N)
        x_end = x_P if design else jnp.minimum(3.0 * fs["x"], 0.5 * (fs["x"] + 1.0))
        bpr_N = sol.bpr(x_end, P_N, fs)
        dT_tot = jnp.maximum(Ts - Tw_N - bpr_N, 5.0 * N)
        d = (1.0 / U) / jnp.sum(1.0 / U) * dT_tot  # temperature drop of each effect
        if design:
            Vtot = F * (1.0 - fs["x"] / x_P)
            V0 = jnp.full((N,), Vtot / N)
        else:
            Afix = jnp.broadcast_to(A_fix, (N,))
            Q0 = U * Afix * d
            V0 = jnp.minimum(Q0 / lam0, 0.5 * F * (1.0 - fs["x"] / jnp.minimum(x_end, 0.9)) / N)
        S0 = V0[0] * 1.05 + 0.0
        # temperatures: Th_1 = Ts; effect i boils at Th_i - d_i; Tw_i = T_i - BPR_i
        x_i = fs["x"] / jnp.maximum(1.0 - jnp.cumsum(V0) / F, 1e-3)
        if p.feed == "backward":
            x_i = fs["x"] / jnp.maximum(1.0 - jnp.cumsum(V0[::-1])[::-1] / F, 1e-3)
        elif p.feed == "parallel":
            x_i = jnp.full((N,), x_end)
        Tw_list = []
        Th = Ts
        for i in range(N - 1):
            Ti = Th - d[i]
            bpr_i = sol.bpr(jnp.minimum(x_i[i], 0.9), water_saturation_pressure(Ti - 3.0), fs)
            Tw_i = Ti - bpr_i
            Tw_list.append(Tw_i)
            Th = Tw_i
        Tw0 = jnp.stack(Tw_list) if Tw_list else jnp.zeros((0,))
        z = [V0 / F, jnp.atleast_1d(S0 / F), Tw0 / 100.0]
        if p.feed == "parallel" and design:
            z.append(jnp.full((N,), 1.0 / N))
        if design:
            Q1 = S0 * lam0
            z.append(jnp.atleast_1d(Q1 / (U[0] * d[0]) / A_fix))
        return jnp.concatenate(z)

    # -- call -----------------------------------------------------------------

    def __call__(self, feed: Stream, steam: Stream | None = None
                 ) -> tuple[Stream, Stream, dict[str, Array]]:
        """Run the multiple-effect evaporator.

        Args:
            feed: Feed stream: solvent and solute species in mol/s, T, P.
            steam: Optional heating-steam stream (only its pressure is used).

        Returns:
            concentrate: Product solution at the final-product effect
                (effect N for forward feed, effect 1 for backward, the mixture
                of all effects for parallel).
            vapor: Vapor leaving the last effect, to the condenser.
            info: Dictionary with ``steam`` (kg/s), ``steam_economy`` (total
                vapor produced / steam), ``A`` (per-effect areas, m^2),
                ``T``, ``T_water``, ``P``, ``BPR``, ``V``, ``Q``, ``x`` (per
                effect, index 0 is effect 1), ``x_product``, ``total_area``,
                ``converged``, ``residual_norm``, ``energy_balance_error`` (W)
                and ``mass_balance_error`` (kg/s), plus ``feed_split`` (parallel
                feed: the fraction of the feed each effect receives, which a
                rating run can take back as ``feed_split``).
        """
        sol, p = self._sol, self.params
        N = int(p.n_effects)
        design = p.x_product is not None
        fs = sol.feed_state(feed)
        F = fs["F"]
        Ts = _steam_temperature(p, steam)
        P_N = jnp.asarray(p.P_vapor_space, dtype=jnp.float64)
        Tw_N = water_saturation_temperature(P_N)
        hF = sol.h_liquid(fs["T"], fs["x"])
        U = self._vector(p.U, N, "U")
        x_P = jnp.asarray(p.x_product if design else 0.5, dtype=jnp.float64)
        if p.feed == "parallel" and not design:
            split = (self._vector(p.feed_split, N, "feed_split")
                     if p.feed_split is not None else jnp.full((N,), 1.0 / N))
        else:
            split = jnp.full((N,), 1.0 / N)
        if design:
            # area scale for the unknown common area: a plausible value, only
            # conditions the Newton step (never enters the answer)
            A_fix = jnp.asarray(100.0, dtype=jnp.float64)
        else:
            A_fix = self._vector(p.A, N, "A")
        args = (fs, hF, Ts, Tw_N, U, A_fix, x_P, split)

        z0 = self._initial_guess(*args)
        # (the initial guess rescales the design area by the same A_fix)
        solution = optx.root_find(
            self._residual, optx.Newton(rtol=1e-10, atol=1e-10), z0, args,
            max_steps=int(p.max_steps), throw=False,
        )
        z = solution.value
        r = self._residual(z, args)
        rnorm = jnp.max(jnp.abs(r))
        converged = (solution.result == optx.RESULTS.successful) & (rnorm < 1e-8)

        V = z[:N] * F
        S = z[N] * F
        Tw_in = z[N + 1:2 * N] * 100.0
        k = 2 * N
        if p.feed == "parallel" and design:
            Fin = z[k:k + N] * F
            k += N
        else:
            Fin = split * F if p.feed == "parallel" else None
        A = jnp.full((N,), z[k] * A_fix) if design else A_fix
        Tw_all = jnp.concatenate([Tw_in, jnp.atleast_1d(Tw_N)])
        e = self._effects(fs, hF, Ts, V, Tw_all, Fin, U)
        T = jnp.stack(e["T"])
        bpr = jnp.stack(e["bpr"])
        xs = jnp.stack(e["x"])
        Lout = jnp.stack(e["Lout"])
        Th = jnp.stack(e["Th"])
        Q = U * A * (Th - T)

        if p.feed == "forward":
            j = N - 1
        elif p.feed == "backward":
            j = 0
        if p.feed == "parallel":
            L = jnp.sum(Lout)
            x_prod = fs["m_s"] / L
            T_prod = jnp.sum(Lout * T) / L
            hL_prod = jnp.sum(Lout * jnp.stack(
                [sol.h_liquid(T[i], xs[i]) for i in range(N)])) / L
        else:
            L = Lout[j]
            x_prod = xs[j]
            T_prod = T[j]
            hL_prod = sol.h_liquid(T_prod, x_prod)
        P_conc = e["P"][0] if p.feed == "backward" else P_N
        conc, vap = sol.make_streams(fs, L, x_prod, T_prod, P_conc, V[N - 1],
                                     T[N - 1], P_N)

        # overall closure from outlets and condensate enthalpies
        H_in = S * water_vapor_enthalpy(Ts) + F * hF
        H_out = S * water_liquid_enthalpy(Ts) + L * hL_prod
        for i in range(N):
            HV = sol.H_vapor(Tw_all[i], T[i])
            H_out = H_out + (V[i] * HV if i == N - 1
                             else V[i] * water_liquid_enthalpy(Tw_all[i]))
        info = {
            "steam": S,
            "steam_economy": jnp.sum(V) / S,
            "V_total": jnp.sum(V),
            "A": A,
            "total_area": jnp.sum(A),
            "T": T,
            "T_water": Tw_all,
            "P": e["P"],
            "BPR": bpr,
            "V": V,
            "Q": Q,
            "x": xs,
            "x_product": x_prod,
            "L": L,
            "T_steam": Ts,
            "converged": converged,
            "residual_norm": rnorm,
            "energy_balance_error": H_in - H_out,
            "mass_balance_error": F - L - jnp.sum(V),
        }
        if p.feed == "parallel":
            info["feed_split"] = jnp.stack(e["Lin"]) / F
        return conc, vap, info


# =============================================================================
# Mechanical vapor recompression
# =============================================================================


@dataclass
class MVRParams(ParamsMixin):
    """Parameters for a single-effect evaporator with mechanical vapor recompression.

    Attributes:
        solute_species: Names of the non-volatile solute species in the stream.
        solute_MW: Molar mass of each solute species (g/mol).
        solvent: Name of the volatile solvent species.
        MW_solvent: Molar mass of the solvent (g/mol).
        U: Overall heat transfer coefficient (W/m^2/K).
        x_product: Product solute mass fraction (-). Design mode.
        P_vapor_space: Pressure of the vapor space (Pa).
        dT_drive: Temperature difference between the condensing compressed
            vapor and the boiling solution (K).
        eta: Isentropic efficiency of the compressor (-).
        gamma: Heat-capacity ratio of steam, for the ideal-gas isentropic
            work (about 1.3).
        steam_P: Pressure of make-up saturated steam (Pa), used only to
            report the steam an MVR-less effect would need.
        bpr_model: Boiling-point-rise model.
        vant_hoff_i: Particles per solute formula unit.
        bpr_coeffs: Polynomial BPR coefficients.
        duhring_x: Mass fractions of a user Dühring table.
        duhring_a: Intercepts of the Dühring lines (K).
        duhring_b: Slopes of the Dühring lines (-).
        Cp_solute: Solute heat capacity (J/kg/K).
        cp_coeffs: Solution heat capacity polynomial in mass fraction (J/kg/K).
        Cp_vapor: Vapor heat capacity (J/kg/K).
    """

    solute_species: list[str]
    solute_MW: list[float]
    solvent: str = "water"
    MW_solvent: float = 18.015
    U: float = 2000.0
    x_product: float = 0.3
    P_vapor_space: float = 101325.0
    dT_drive: float = 8.0
    eta: float = 0.75
    gamma: float = 1.3
    steam_P: float | None = None
    bpr_model: str = "ideal"
    vant_hoff_i: float = 1.0
    bpr_coeffs: list[float] | None = None
    duhring_x: list[float] | None = None
    duhring_a: list[float] | None = None
    duhring_b: list[float] | None = None
    Cp_solute: float = 1500.0
    cp_coeffs: list[float] | None = None
    Cp_vapor: float = 1880.0


class MechanicalVaporRecompression:
    """Single-effect evaporator whose vapor is recompressed and reused as its steam.

    The vapor leaves the effect at ``P_vapor_space`` (superheated by the
    BPR), is compressed isentropically (ideal-gas steam, ``gamma``) with
    efficiency ``eta`` to the pressure at which it condenses ``dT_drive`` above
    the boiling solution, and condenses on the heating surface. Any shortfall
    between the duty and the compressed-vapor heat is a (signed) make-up
    steam requirement. Design mode: the product concentration is given, the
    area comes out.

    Reports the compressor work against the steam a plain single effect
    would need for the same duty (``steam_saved``).

    The call is ``mvr(feed)``.
    """

    symbol = "MVR evaporator"
    equations = [
        r"T_c = T_\mathrm{soln} + \Delta T,\qquad P_c = P_\mathrm{sat}(T_c)",
        r"w_s = c_{p,v} T_1\left[(P_c/P_1)^{(\gamma-1)/\gamma} - 1\right],"
        r"\qquad W = V w_s / \eta",
        r"Q = V\,(H_{V} + w_s/\eta - h_f(T_c)) + Q_\mathrm{makeup},\qquad Q = U A\,\Delta T",
    ]
    assumptions = _COMMON_ASSUMPTIONS + [
        "Steam is an ideal gas with constant gamma and cp for the compression "
        "(real-gas deviation at these pressures is a few percent of the work).",
        "No heat loss, no compressor inlet superheat control; the compressed vapor "
        "condenses to saturated liquid at its saturation temperature.",
    ]
    references = Evaporator.references + [
        "Mechanical vapor recompression: McCabe Ch. 16 (vapor recompression); "
        "isentropic ideal-gas compression, Smith, Van Ness, Abbott Ch. 7.",
    ]
    parameter_symbols = {"dT_drive": r"\Delta T", "eta": r"\eta", "gamma": r"\gamma",
                         "U": "U", "x_product": "x_P"}
    parameter_units = _only(MVRParams, dict(_COMMON_UNITS, dT_drive="K", eta="-", gamma="-"))
    numerical_method = "Closed form (design mode)."

    def __init__(self, params: MVRParams, *, bpr_fn: Callable | None = None,
                 enthalpy_fn: Callable | None = None):
        """Initialize.

        Args:
            params: MVR parameters.
            bpr_fn: Optional boiling-point-rise callable ``(x, T_water, P) -> K``.
            enthalpy_fn: Optional solution enthalpy callable ``(T, x) -> J/kg``.
        """
        self.params = params
        self._sol = _Solution(params, bpr_fn, enthalpy_fn)

    def __call__(self, feed: Stream) -> tuple[Stream, Stream, dict[str, Array]]:
        """Run the MVR evaporator.

        Args:
            feed: Feed stream.

        Returns:
            concentrate: Product solution.
            condensate: Condensed compressed vapor (saturated liquid water at
                the condensing temperature, solutes at 0).
            info: Dictionary with ``A`` (m^2), ``Q`` (W), ``V`` (kg/s),
                ``W_compressor`` (W), ``compression_ratio``, ``P_discharge``
                (Pa), ``T_condensing`` (K), ``Q_makeup`` (W, signed: negative
                means the compressed vapor over-supplies the duty),
                ``steam_makeup`` (kg/s, at ``steam_P`` when given),
                ``steam_without_mvr`` and ``steam_saved`` (kg/s, if
                ``steam_P`` is given), ``steam_economy`` (V / equivalent
                live steam) and ``specific_work`` (J per kg evaporated).
        """
        sol, p = self._sol, self.params
        fs = sol.feed_state(feed)
        P1 = jnp.asarray(p.P_vapor_space, dtype=jnp.float64)
        Tw = water_saturation_temperature(P1)
        x_P = jnp.asarray(p.x_product, dtype=jnp.float64)
        L = fs["m_s"] / x_P
        V = fs["F"] - L
        bpr = sol.bpr(x_P, P1, fs)
        T = Tw + bpr
        hF = sol.h_liquid(fs["T"], fs["x"])
        hL = sol.h_liquid(T, x_P)
        HV = sol.H_vapor(Tw, T)
        Q = L * hL + V * HV - fs["F"] * hF
        Tc = T + p.dT_drive
        Pc = water_saturation_pressure(Tc)
        ratio = Pc / P1
        g = p.gamma
        w_s = p.Cp_vapor * T * (ratio ** ((g - 1.0) / g) - 1.0)
        w = w_s / p.eta
        H_out = HV + w
        Q_comp = V * (H_out - water_liquid_enthalpy(Tc))
        Q_makeup = Q - Q_comp
        A = Q / (jnp.asarray(p.U, dtype=jnp.float64) * p.dT_drive)
        info = {
            "A": A, "Q": Q, "V": V, "L": L, "x_product": x_P, "BPR": bpr,
            "T_soln": T, "T_condensing": Tc, "P_discharge": Pc,
            "compression_ratio": ratio, "W_compressor": V * w,
            "specific_work": w, "Q_compressor_heat": Q_comp, "Q_makeup": Q_makeup,
        }
        if p.steam_P is not None:
            Ts = water_saturation_temperature(p.steam_P)
            lam_s = water_latent_heat(Ts)
            info["steam_without_mvr"] = Q / lam_s
            info["steam_makeup"] = Q_makeup / lam_s
            info["steam_saved"] = Q / lam_s - Q_makeup / lam_s
            info["steam_economy"] = V / (Q_makeup / lam_s)
        conc, _ = sol.make_streams(fs, L, x_P, T, P1, V, T, P1)
        cond = {"F_" + sol.solvent: V * 1000.0 / sol.MW_w, "T": Tc, "P": Pc}
        for s in sol.solutes:
            cond["F_" + s] = jnp.zeros(())
        return conc, cond, info
