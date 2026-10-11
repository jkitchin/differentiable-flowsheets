"""Liquid pipe flow unit operation.

:class:`Pipe` is an incompressible Darcy-Weisbach pipe run with fittings and
elevation change. It lowers the pressure of the stream; temperature and molar
flows pass through unchanged. See :mod:`difflow.fluids` for the friction factor
methods, the fittings table and the flow meters.

Liquid density and viscosity: ``PipeParams`` takes ``rho`` (kg/m^3) and
``mu`` (Pa s) as numbers, as callables of the inlet stream, or as None (the
default), in which case they come from :mod:`difflow.liquid_properties`
(:func:`~difflow.liquid_properties.stream_liquid_properties`) at the inlet
temperature, and the volumetric flow is the stream's own liquid volume. With
explicit ``rho``, converting molar flow to volumetric flow needs a molar mass:
pass ``MW`` (g/mol; a number for a pure liquid or ``{species: MW}`` for a
mixture) or build the unit with a ``thermo`` that carries ``SpeciesData``.

The outlet pressure is checked against the liquid's vapour pressure
(``info["P_out_below_vapour"]``, and a :class:`PipeCavitationWarning` outside
a trace).
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array

from difflow import fluids
from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, get_flows, get_species, make_stream

#: Commercial steel absolute roughness (m), Crane TP-410 / Moody (1944).
STEEL_ROUGHNESS = 4.6e-5


class PipeCavitationWarning(UserWarning):
    """A :class:`Pipe` outlet pressure is below the liquid's vapour pressure.

    The incompressible model has no phase change: a pressure below the vapour
    pressure (or below zero absolute) means the liquid would flash or the
    line would cavitate, so the computed pressure is not physical. Raised on
    concrete values only; under ``jit``/``grad`` read
    ``info["P_out_below_vapour"]`` instead.
    """


@dataclass(repr=False)
class PipeParams(ParamsMixin):
    """Parameters for :class:`Pipe`.

    Attributes:
        L: Straight pipe length (m).
        D: Inside diameter (m).
        rho: Liquid density (kg/m^3): a number, a callable ``rho(stream)``
            returning one (it sees the inlet stream, so it may depend on T),
            or None to use :mod:`difflow.liquid_properties` at the inlet T.
        mu: Liquid dynamic viscosity (Pa s): a number, a callable
            ``mu(stream)``, or None (liquid_properties).
        roughness: Absolute roughness epsilon (m); default commercial steel, 4.6e-5.
        dz: Elevation gain from inlet to outlet (m); positive is uphill.
        fittings: Minor losses: a dict ``{fitting name: count}`` using the keys
            of :data:`difflow.fluids.FITTINGS`, optionally with a numeric
            ``"K"`` entry added as is (a jacket, a vendor's strainer); or a
            number giving the total loss coefficient K. None for no fittings.
        friction_method: ``"colebrook"``, ``"churchill"``, ``"haaland"`` or
            ``"swamee_jain"``.
        MW: Molar mass (g/mol) used to turn molar flow into mass flow: a
            number, or ``{species: MW}``. May be omitted when the unit is
            built with a ``thermo``, or when ``rho`` is None.
        flow_area: True flow cross-section (m^2) for a non-circular duct; then
            ``D`` is the hydraulic diameter ``4A/P_wetted`` used in Re, e/D and
            L/D, and the velocity is ``Q / flow_area``. None (default) is a
            full circular pipe, ``pi D^2/4``.
        P_vap: Vapour pressure (Pa) for the cavitation check: a number, a
            callable ``P_vap(stream)``, or None to take it from the database
            (water: IAPWS saturation curve; other species: Antoine; a mixture:
            Raoult's-law bubble pressure over the species that have data).
    """

    L: float
    D: float
    rho: float | None = None  # a callable rho(stream) is also accepted
    mu: float | None = None  # likewise a callable mu(stream)
    roughness: float = STEEL_ROUGHNESS
    dz: float = 0.0
    fittings: dict | float | None = None
    friction_method: str = "colebrook"
    MW: dict | float | None = None
    flow_area: float | None = None
    P_vap: float | None = None


def evaluate_property(value, stream: Stream | None = None) -> Array:
    """Liquid property as a float64 array: a number, or a callable of the stream."""
    if callable(value):
        if stream is None:
            raise ValueError("a callable liquid property needs a stream to be evaluated on")
        value = value(stream)
    return jnp.asarray(value, dtype=jnp.float64)


@functools.lru_cache(maxsize=None)
def _psat_model(name: str):
    """``("water", None)``, ``("antoine", (A, B, C))`` or None for a species name."""
    from difflow.database import get_species_info, resolve_alias

    try:
        canon = resolve_alias(name)
    except Exception:
        canon = name
    if canon in ("water", "H2O"):
        return ("water", None)
    try:
        thermo = get_species_info(canon).get("ideal_thermo")
    except KeyError:
        return None
    if not thermo or "antoine" not in thermo:
        return None
    return ("antoine", tuple(float(c) for c in thermo["antoine"]))


def stream_vapour_pressure(stream: Stream) -> Array | None:
    """Raoult's-law bubble pressure (Pa) of a liquid stream, from the database.

    ``P_bub = sum_i x_i Psat_i(T)`` over the species that have vapour-pressure
    data (water: :func:`difflow.water.water_saturation_pressure`; others:
    their Antoine equation). Species without data are left out (so the
    result is a lower bound). Returns None if no species has data. JAX-only
    on the values, so it is jit-safe.
    """
    from difflow.water import water_saturation_pressure

    flows = get_flows(stream)
    T = jnp.asarray(stream["T"], dtype=jnp.float64)
    total = sum(flows.values())
    P = None
    for name, F in flows.items():
        model = _psat_model(name)
        if model is None:
            continue
        if model[0] == "water":
            psat = water_saturation_pressure(T)
        else:
            A, B, C = model[1]
            psat = 10.0 ** (A - B / (T + C))
        term = F / total * psat
        P = term if P is None else P + term
    return P


def _is_concrete(*xs) -> bool:
    return not any(isinstance(x, jax.core.Tracer) for x in xs)


def stream_mass_flow(inlet: Stream, MW=None, thermo=None) -> Array:
    """Mass flow (kg/s) of a stream from molar flows and molar masses.

    Args:
        inlet: Stream with molar flows (mol/s).
        MW: Molar mass (g/mol): a number, or ``{species: MW}``; None to use
            ``thermo.species[s].MW``.
        thermo: Object with a ``species`` map of ``SpeciesData`` (used when
            ``MW`` is None).

    Returns:
        Mass flow (kg/s).
    """
    flows = get_flows(inlet)
    if MW is None:
        if thermo is None or not hasattr(thermo, "species"):
            raise ValueError(
                "A molar mass is needed to turn molar flow into mass flow: "
                "set MW=... (g/mol, or {species: MW}) or pass a thermo."
            )
        MW = {s: thermo.species[s].MW for s in flows}
    total = 0.0
    for s in get_species(inlet):
        mw = MW[s] if isinstance(MW, dict) else MW
        total = total + flows[s] * mw
    return total * 1e-3


class Pipe:
    """Incompressible Darcy-Weisbach pipe with fittings and elevation change.

    ``P_out = P_in - dP_friction - dP_minor - rho g dz``. Temperature and molar
    flows are unchanged (isothermal, adiabatic-wall, no phase change). Returns
    ``(outlet, info)``; ``info`` holds ``Q`` (m^3/s), ``v`` (m/s), ``Re``, ``f``,
    ``K_total``, ``dP_friction``, ``dP_minor``, ``dP_static``, ``dP`` (their sum,
    Pa) and ``head_loss`` (friction plus minor loss, m of fluid); ``__call__``
    adds ``P_vap`` (Pa; NaN when unknown) and ``P_out_below_vapour`` (a bool
    array: the outlet is below the vapour pressure, or below zero when that is
    unknown), and raises :class:`PipeCavitationWarning` on concrete values.

    :meth:`flow_at_dp` and :meth:`flow_at_head` invert the hydraulics (flow
    from an available pressure or head) with implicit derivatives.

    Differentiable in every numeric parameter, notably ``D`` for economic
    pipe-diameter optimization.
    """

    symbol = "Pipe"
    equations = [
        r"\Delta P_f = f\,\frac{L}{D}\,\frac{\rho v^2}{2},\qquad v = \frac{Q}{A},\ A = \frac{\pi D^2}{4}\ \text{(or flow\_area)}",
        r"\frac{1}{\sqrt{f}} = -2\log_{10}\!\left(\frac{\varepsilon/D}{3.7} + \frac{2.51}{\mathrm{Re}\sqrt{f}}\right),\quad f = \frac{64}{\mathrm{Re}}\ (\mathrm{Re}<2100)",
        r"\Delta P_m = \sum_j n_j K_j\,\frac{\rho v^2}{2}",
        r"P_{\mathrm{out}} = P_{\mathrm{in}} - \Delta P_f - \Delta P_m - \rho g\,\Delta z",
    ]
    assumptions = [
        "Incompressible, single-phase Newtonian liquid; rho and mu are inputs, constant along the pipe.",
        "Full circular pipe of constant diameter (or a duct with D = hydraulic diameter and flow_area = true area), steady state, fully developed flow.",
        "No phase change: an outlet below the vapour pressure is flagged (P_out_below_vapour, PipeCavitationWarning), not modelled.",
        "Isothermal: T and molar flows pass through unchanged.",
        "Friction factor from the chosen correlation, blended to 64/Re below Re = 2100 (transition 2100-4000 is an interpolation, not a model).",
        "Fitting K-values (Crane TP-410) are approximate, about +/-25 percent.",
    ]
    references = [
        "Crane Co. Flow of Fluids Through Valves, Fittings, and Pipe, Technical Paper 410.",
        "Colebrook, C.F. (1939). J. Inst. Civil Eng. 11, 133-156.",
        "Churchill, S.W. (1977). Chem. Eng. 84(24), 91-92.",
        "Perry's Chemical Engineers' Handbook, 9e, Sec. 6.",
    ]
    parameter_symbols = {"L": "L", "D": "D", "rho": r"\rho", "mu": r"\mu", "roughness": r"\varepsilon", "dz": r"\Delta z"}
    parameter_units = {
        "L": "m", "D": "m", "rho": "kg/m^3", "mu": "Pa s", "roughness": "m",
        "dz": "m", "fittings": "-", "friction_method": "-", "MW": "g/mol",
        "flow_area": "m^2", "P_vap": "Pa",
    }

    def __init__(self, params: PipeParams, thermo=None):
        """Initialize the unit.

        Args:
            params: Pipe parameters.
            thermo: Optional thermo whose ``species`` map holds ``SpeciesData``
                (for ``MW``). Not needed when ``params.MW`` is given.
        """
        self.params = params
        self.thermo = thermo

    # ------------------------------------------------------------------
    def _mass_flow(self, inlet: Stream) -> Array:
        """Mass flow (kg/s) of the inlet stream."""
        return stream_mass_flow(inlet, self.params.MW, self.thermo)

    @staticmethod
    def _property(value, inlet: Stream) -> Array:
        return evaluate_property(value, inlet)

    def _K_total(self, rel_roughness: Array) -> Array:
        return fluids.fittings_K(self.params.fittings, rel_roughness=rel_roughness)

    def _rho_mu(self, inlet: Stream | None):
        """Density and viscosity, from the parameters or liquid_properties."""
        p = self.params
        lp = None
        if p.rho is None or p.mu is None:
            if inlet is None:
                raise ValueError(
                    "rho/mu are None (read from difflow.liquid_properties at the inlet), "
                    "so an inlet stream is needed; pass inlet=... or set rho and mu."
                )
            from difflow.liquid_properties import stream_liquid_properties

            lp = stream_liquid_properties(inlet)
        rho = lp["rho"] if p.rho is None else evaluate_property(p.rho, inlet)
        mu = lp["mu"] if p.mu is None else evaluate_property(p.mu, inlet)
        return rho, mu, lp

    def hydraulics_at_flow(self, Q: Array | float, inlet: Stream | None = None) -> dict:
        """Hydraulics at a given volumetric flow ``Q`` (m^3/s).

        Same quantities as ``info`` from ``__call__``. ``inlet`` is only needed
        when ``rho`` or ``mu`` are callables of the stream or None. Used by
        :func:`difflow.units.pump.system_curve`.
        """
        D = jnp.asarray(self.params.D, dtype=jnp.float64)
        rho, mu, _ = self._rho_mu(inlet)
        Q = jnp.asarray(Q, dtype=jnp.float64)
        return self._hydraulics_Q(Q, rho, mu, D)

    def _hydraulics(self, inlet: Stream) -> dict:
        p = self.params
        D = jnp.asarray(p.D, dtype=jnp.float64)
        rho, mu, lp = self._rho_mu(inlet)
        if p.rho is None and p.MW is None and self.thermo is None:
            Q = lp["Q"]
        else:
            Q = self._mass_flow(inlet) / rho
        return self._hydraulics_Q(Q, rho, mu, D)

    def _area(self, D: Array) -> Array:
        if self.params.flow_area is None:
            return jnp.pi / 4.0 * D**2
        return jnp.asarray(self.params.flow_area, dtype=jnp.float64)

    def flow_at_dp(self, dP: Array | float, inlet: Stream | None = None,
                   Q0: Array | float | None = None) -> Array:
        """Volumetric flow (m^3/s) that gives a total pressure drop ``dP`` (Pa).

        Solves ``info["dP"](Q) = dP`` (friction + fittings + ``rho g dz``) by
        Newton's method in ``ln Q`` (optimistix), with implicit-function-theorem
        derivatives: ``jax.grad`` of the flow with respect to ``dP``, ``D``,
        roughness, ``mu`` or any K is ``-(d dP/d theta)/(d dP/dQ)`` at the
        solution, not a derivative of the iterations. When ``dP`` does not
        exceed the static head ``rho g dz`` there is no forward flow and 0 is
        returned.

        Args:
            dP: Available pressure difference, inlet minus outlet (Pa).
            inlet: Stream, only needed when ``rho``/``mu`` are callables or None.
            Q0: Initial guess (m^3/s); default from a fully rough estimate.

        Returns:
            Q (m^3/s).
        """
        p = self.params
        D = jnp.asarray(p.D, dtype=jnp.float64)
        rho, mu, _ = self._rho_mu(inlet)
        dP = jnp.asarray(dP, dtype=jnp.float64)
        dP_s = rho * fluids.GRAVITY * jnp.asarray(p.dz, dtype=jnp.float64)
        drive = dP - dP_s
        ok = drive > 0.0
        drive_safe = jnp.where(ok, drive, 1.0)
        A = self._area(D)
        if Q0 is None:
            rel = jnp.asarray(p.roughness, dtype=jnp.float64) / D
            f0 = fluids.fully_rough_friction_factor(jnp.maximum(rel, 1e-6))
            K0 = self._K_total(jnp.maximum(rel, 1e-6))
            v0 = jnp.sqrt(2.0 * drive_safe / (rho * (f0 * p.L / D + K0 + 1e-12)))
            Q0 = v0 * A
        y0 = jax.lax.stop_gradient(jnp.log(jnp.asarray(Q0, dtype=jnp.float64)))

        def resid(y, args):
            rho_, mu_, D_, drive_ = args
            h = self._hydraulics_Q(jnp.exp(y), rho_, mu_, D_)
            return jnp.log(h["dP_friction"] + h["dP_minor"]) - jnp.log(drive_)

        sol = optx.root_find(resid, optx.Newton(rtol=1e-12, atol=1e-12), y0,
                             (rho, mu, D, drive_safe), max_steps=100, throw=False)
        return jnp.where(ok, jnp.exp(sol.value), 0.0)

    def flow_at_head(self, H: Array | float, inlet: Stream | None = None,
                     Q0: Array | float | None = None) -> Array:
        """Volumetric flow (m^3/s) for an available head ``H`` (m of liquid).

        ``head_loss(Q) + dz = H``, i.e. :meth:`flow_at_dp` with
        ``dP = rho g H``. For a gravity feed between two free surfaces, ``H``
        is the level difference and the fittings should include the
        ``"exit"`` (and an entrance) so the discharged velocity head is counted.
        Implicit derivatives, as in :meth:`flow_at_dp`.

        Example:
            >>> from difflow import Pipe, PipeParams
            >>> fit = {"entrance_sharp": 1, "elbow_90_standard": 4, "gate_valve": 1, "exit": 1}
            >>> pipe = Pipe(PipeParams(L=120.0, D=0.0525, rho=998.96, mu=1.0214e-3, fittings=fit))
            >>> round(float(pipe.flow_at_head(15.0)) * 1e3, 3)   # L/s
            5.113
        """
        rho, _, _ = self._rho_mu(inlet)
        return self.flow_at_dp(rho * fluids.GRAVITY * jnp.asarray(H, dtype=jnp.float64), inlet, Q0)

    def _hydraulics_Q(self, Q: Array, rho: Array, mu: Array, D: Array) -> dict:
        p = self.params
        v = Q / self._area(D)
        rel = jnp.asarray(p.roughness, dtype=jnp.float64) / D
        Re = fluids.reynolds_number(rho, v, D, mu)
        f = fluids.friction_factor(Re, rel, p.friction_method)
        K = self._K_total(rel)
        dP_f = fluids.darcy_pressure_drop(f, p.L, D, rho, v)
        dP_m = fluids.minor_loss(K, rho, v)
        dP_s = rho * fluids.GRAVITY * jnp.asarray(p.dz, dtype=jnp.float64)
        return {
            "Q": Q, "v": v, "Re": Re, "f": f, "K_total": K,
            "dP_friction": dP_f, "dP_minor": dP_m, "dP_static": dP_s,
            "dP": dP_f + dP_m + dP_s,
            "head_loss": (dP_f + dP_m) / (rho * fluids.GRAVITY),
        }

    def __call__(self, inlet: Stream) -> tuple[Stream, dict]:
        """Compute the outlet stream and hydraulic diagnostics.

        Args:
            inlet: Liquid stream.

        Returns:
            ``(outlet, info)`` as described on the class.
        """
        info = self._hydraulics(inlet)
        P_out = inlet["P"] - info["dP"]
        p = self.params
        if p.P_vap is not None:
            P_vap = evaluate_property(p.P_vap, inlet)
        else:
            P_vap = stream_vapour_pressure(inlet)
        P_ref = jnp.asarray(0.0) if P_vap is None else P_vap
        info["P_vap"] = jnp.asarray(jnp.nan) if P_vap is None else P_vap
        info["P_out_below_vapour"] = P_out < P_ref
        if _is_concrete(P_out, P_ref):
            try:
                below = bool(P_out < P_ref)
            except Exception:  # array-valued: warn if any element is below
                below = bool(jnp.any(P_out < P_ref))
            if below:
                what = ("zero (absolute)" if P_vap is None
                        else f"the vapour pressure ({float(jnp.max(P_ref)):.4g} Pa)")
                warnings.warn(
                    f"Pipe outlet pressure {float(jnp.min(P_out)):.4g} Pa is below {what}: "
                    "the liquid would flash or cavitate, and the incompressible result is "
                    "not physical. Check the elevation (dz), the flow and the inlet pressure; "
                    "set P_vap=... if the database has no vapour pressure for this liquid.",
                    PipeCavitationWarning,
                    stacklevel=2,
                )
        outlet = make_stream(get_flows(inlet), inlet["T"], P_out)
        return outlet, info

    # ------------------------------------------------------------------
    def eo_residuals(self, inlets: list[Stream], outlets: list[Stream], **kwargs) -> Array:
        """Residuals for the EO solver.

        Residuals:
            F_out_i - F_in_i = 0                        (n_species)
            T_out - T_in = 0                            (1)
            P_out - (P_in - dP(inlet)) = 0              (1)

        The pressure row is the same algebraic expression ``__call__`` uses,
        evaluated at the inlet, so EO roots are the sequential unit's.

        Args:
            inlets: [inlet_stream]
            outlets: [outlet_stream]

        Returns:
            Flat residual array, length n_species + 2
        """
        inlet, outlet = inlets[0], outlets[0]
        inflows, outflows = get_flows(inlet), get_flows(outlet)
        resid = [jnp.atleast_1d(outflows[s] - inflows[s]) for s in get_species(inlet)]
        resid.append(jnp.atleast_1d(outlet["T"] - inlet["T"]))
        dP = self._hydraulics(inlet)["dP"]
        resid.append(jnp.atleast_1d(outlet["P"] - (inlet["P"] - dP)))
        return jnp.concatenate(resid)


__all__ = [
    "Pipe",
    "PipeParams",
    "PipeCavitationWarning",
    "STEEL_ROUGHNESS",
    "evaluate_property",
    "stream_mass_flow",
    "stream_vapour_pressure",
]
