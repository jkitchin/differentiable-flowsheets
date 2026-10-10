"""Liquid pipe flow unit operation.

:class:`Pipe` is an incompressible Darcy-Weisbach pipe run with fittings and
elevation change. It lowers the pressure of the stream; temperature and molar
flows pass through unchanged. See :mod:`difflow.fluids` for the friction factor
methods, the fittings table and the flow meters.

Liquid properties are explicit inputs: core ``SpeciesData`` has no liquid
density or viscosity, so ``PipeParams`` takes ``rho`` (kg/m^3) and ``mu``
(Pa s) as numbers or as callables of the inlet stream, e.g. a correlation in
``T``. Converting molar flow to volumetric flow needs a molar mass: pass ``MW``
(g/mol; a number for a pure liquid or ``{species: MW}`` for a mixture) or build
the unit with a ``thermo`` that carries ``SpeciesData``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from difflow import fluids
from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, get_flows, get_species, make_stream

#: Commercial steel absolute roughness (m), Crane TP-410 / Moody (1944).
STEEL_ROUGHNESS = 4.6e-5


@dataclass(repr=False)
class PipeParams(ParamsMixin):
    """Parameters for :class:`Pipe`.

    Attributes:
        L: Straight pipe length (m).
        D: Inside diameter (m).
        rho: Liquid density (kg/m^3): a number, or a callable ``rho(stream)``
            returning one (it sees the inlet stream, so it may depend on T).
        mu: Liquid dynamic viscosity (Pa s): a number or a callable ``mu(stream)``.
        roughness: Absolute roughness epsilon (m); default commercial steel, 4.6e-5.
        dz: Elevation gain from inlet to outlet (m); positive is uphill.
        fittings: Minor losses: a dict ``{fitting name: count}`` using the keys
            of :data:`difflow.fluids.FITTINGS`, or a number giving the total
            loss coefficient K. None for no fittings.
        friction_method: ``"colebrook"``, ``"churchill"``, ``"haaland"`` or
            ``"swamee_jain"``.
        MW: Molar mass (g/mol) used to turn molar flow into mass flow: a
            number, or ``{species: MW}``. May be omitted when the unit is
            built with a ``thermo``.
    """

    L: float
    D: float
    rho: float  # a callable rho(stream) is also accepted; typed float so forms treat it as a number
    mu: float  # likewise a callable mu(stream)
    roughness: float = STEEL_ROUGHNESS
    dz: float = 0.0
    fittings: dict | float | None = None
    friction_method: str = "colebrook"
    MW: dict | float | None = None


def evaluate_property(value, stream: Stream | None = None) -> Array:
    """Liquid property as a float64 array: a number, or a callable of the stream."""
    if callable(value):
        if stream is None:
            raise ValueError("a callable liquid property needs a stream to be evaluated on")
        value = value(stream)
    return jnp.asarray(value, dtype=jnp.float64)


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
    Pa) and ``head_loss`` (friction plus minor loss, m of fluid).

    Differentiable in every numeric parameter, notably ``D`` for economic
    pipe-diameter optimization.
    """

    symbol = "Pipe"
    equations = [
        r"\Delta P_f = f\,\frac{L}{D}\,\frac{\rho v^2}{2},\qquad v = \frac{4Q}{\pi D^2}",
        r"\frac{1}{\sqrt{f}} = -2\log_{10}\!\left(\frac{\varepsilon/D}{3.7} + \frac{2.51}{\mathrm{Re}\sqrt{f}}\right),\quad f = \frac{64}{\mathrm{Re}}\ (\mathrm{Re}<2100)",
        r"\Delta P_m = \sum_j n_j K_j\,\frac{\rho v^2}{2}",
        r"P_{\mathrm{out}} = P_{\mathrm{in}} - \Delta P_f - \Delta P_m - \rho g\,\Delta z",
    ]
    assumptions = [
        "Incompressible, single-phase Newtonian liquid; rho and mu are inputs, constant along the pipe.",
        "Full circular pipe of constant diameter, steady state, fully developed flow.",
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
        fit = self.params.fittings
        if fit is None:
            return jnp.asarray(0.0)
        if isinstance(fit, dict):
            K = jnp.asarray(0.0)
            for name, count in fit.items():
                K = K + count * fluids.fitting_K(name, rel_roughness=rel_roughness)
            return K
        return jnp.asarray(fit, dtype=jnp.float64)

    def hydraulics_at_flow(self, Q: Array | float, inlet: Stream | None = None) -> dict:
        """Hydraulics at a given volumetric flow ``Q`` (m^3/s).

        Same quantities as ``info`` from ``__call__``. ``inlet`` is only needed
        when ``rho`` or ``mu`` are callables of the stream. Used by
        :func:`difflow.units.pump.system_curve`.
        """
        p = self.params
        D = jnp.asarray(p.D, dtype=jnp.float64)
        rho = evaluate_property(p.rho, inlet)
        mu = evaluate_property(p.mu, inlet)
        Q = jnp.asarray(Q, dtype=jnp.float64)
        return self._hydraulics_Q(Q, rho, mu, D)

    def _hydraulics(self, inlet: Stream) -> dict:
        p = self.params
        D = jnp.asarray(p.D, dtype=jnp.float64)
        rho = self._property(p.rho, inlet)
        mu = self._property(p.mu, inlet)
        return self._hydraulics_Q(self._mass_flow(inlet) / rho, rho, mu, D)

    def _hydraulics_Q(self, Q: Array, rho: Array, mu: Array, D: Array) -> dict:
        p = self.params
        v = Q / (jnp.pi / 4.0 * D**2)
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
        outlet = make_stream(get_flows(inlet), inlet["T"], inlet["P"] - info["dP"])
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


__all__ = ["Pipe", "PipeParams", "STEEL_ROUGHNESS", "evaluate_property", "stream_mass_flow"]
