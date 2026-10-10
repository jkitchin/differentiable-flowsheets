"""Centrifugal pump with performance curves, affinity laws, NPSH and operating point.

:class:`CentrifugalPump` raises the pressure of a liquid stream by
``rho g H(Q)`` where the head curve ``H(Q)`` is a polynomial in the volumetric
flow, scaled by the affinity laws for the operating speed ``N`` and impeller
diameter ``D``. Around it:

* :class:`PumpCurve` -- the (scaled) head / efficiency / NPSHr curves as a
  JAX pytree, with :func:`pumps_in_series` and :func:`pumps_in_parallel`
  returning combined curves with the same interface;
* :func:`system_curve` -- ``H_sys(Q)`` from a static head and
  :class:`~difflow.units.pipe.Pipe` runs (so friction and fittings come from
  :mod:`difflow.fluids`);
* :func:`operating_point` -- the flow where pump and system curves cross,
  differentiable by the implicit function theorem;
* :func:`npsh_available`, :func:`fit_pump_curve`.

All quantities are SI: Q in m^3/s, H and NPSH in m of fluid, power in W.
Polynomial coefficients are in ascending order, ``c0 + c1 Q + c2 Q^2``, so
``h1`` has units m/(m^3/s) and ``h2`` m/(m^3/s)^2. Liquid density is an explicit
input (see :mod:`difflow.units.pipe` for why).

The :class:`difflow_cc.Pump <difflow_cc.units.compression.Pump>` for dense
CO2 is a different, fixed-outlet-pressure model (``W = Q dP / eta``); this
module answers the question it cannot: what flow does this pump deliver into
this system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array

from difflow.fluids import GRAVITY
from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, get_flows, get_species, make_stream
from difflow.units.pipe import Pipe, evaluate_property, stream_mass_flow

#: Floor on the efficiency curve so shaft power stays finite far off the curve.
ETA_FLOOR = 0.01


def _poly(coeffs: Array, x: Array) -> Array:
    """Evaluate an ascending-order polynomial (Horner)."""
    c = jnp.atleast_1d(jnp.asarray(coeffs, dtype=jnp.float64))
    return jnp.polyval(c[::-1], x)


def bep_efficiency_coeffs(eta_max: Array | float, Q_bep: Array | float) -> Array:
    """Quadratic coefficients of ``eta = eta_max (1 - ((Q - Q_bep)/Q_bep)^2)``.

    Args:
        eta_max: Peak efficiency (0-1) at the best efficiency point.
        Q_bep: Flow at the best efficiency point (m^3/s).

    Returns:
        Ascending coefficients ``[0, 2 eta_max/Q_bep, -eta_max/Q_bep^2]``.
    """
    eta_max = jnp.asarray(eta_max, dtype=jnp.float64)
    Q_bep = jnp.asarray(Q_bep, dtype=jnp.float64)
    return jnp.stack([jnp.zeros_like(eta_max), 2.0 * eta_max / Q_bep, -eta_max / Q_bep**2])


def fit_pump_curve(Q_data: Array, H_data: Array, order: int = 2) -> Array:
    """Least-squares polynomial fit of vendor head-flow (or NPSHr) data.

    Args:
        Q_data: Flows (m^3/s).
        H_data: Heads (m).
        order: Polynomial order (2 gives ``h0 + h1 Q + h2 Q^2``).

    Returns:
        Ascending coefficients, length ``order + 1``, ready for
        ``CentrifugalPumpParams(head_coeffs=...)``. Differentiable in the data.
    """
    Q = jnp.asarray(Q_data, dtype=jnp.float64)
    H = jnp.asarray(H_data, dtype=jnp.float64)
    scale = jnp.max(jnp.abs(Q))  # conditioning only; coefficients are rescaled back
    x = Q / scale
    A = x[:, None] ** jnp.arange(order + 1)[None, :]
    c, *_ = jnp.linalg.lstsq(A, H)
    return c / scale ** jnp.arange(order + 1)


def npsh_available(
    P_suction: Array | float,
    Psat: Array | float,
    rho: Array | float,
    v: Array | float = 0.0,
    z_suction: Array | float = 0.0,
    losses: Array | float = 0.0,
) -> Array:
    """Net positive suction head available (m of fluid).

    ``NPSHa = (P_suction - Psat)/(rho g) + v^2/(2 g) + z_suction - losses``.

    Two common uses: (a) ``P_suction`` is the absolute pressure at the pump
    flange and ``v`` the suction-line velocity, ``z_suction = losses = 0``; (b)
    ``P_suction`` is the absolute pressure on a tank liquid surface, ``v = 0``,
    ``z_suction`` the liquid level above the pump centreline (negative for a
    suction lift) and ``losses`` the suction-line friction and fitting head.

    Args:
        P_suction: Absolute pressure (Pa).
        Psat: Vapour pressure at the pumping temperature (Pa).
        rho: Liquid density (kg/m^3).
        v: Velocity in the suction line (m/s), for the velocity head.
        z_suction: Static elevation of the source above the pump (m).
        losses: Suction-side head loss (m).

    Returns:
        NPSHa (m).
    """
    rho = jnp.asarray(rho, dtype=jnp.float64)
    return (
        (jnp.asarray(P_suction) - jnp.asarray(Psat)) / (rho * GRAVITY)
        + jnp.asarray(v) ** 2 / (2.0 * GRAVITY)
        + jnp.asarray(z_suction)
        - jnp.asarray(losses)
    )


# ----------------------------------------------------------------------
# Curves
# ----------------------------------------------------------------------
@jax.tree_util.register_pytree_node_class
class PumpCurve:
    """Head, efficiency and NPSHr curves of one pump at an operating speed and diameter.

    The coefficient arrays describe the pump at the reference speed and
    diameter. The affinity laws (``s = (N/N_ref)(D/D_ref)``) map a reference
    point ``(Q_ref, H_ref)`` to ``Q = s Q_ref``, ``H = s^2 H_ref`` and shaft
    power ``P = s^3 P_ref``; efficiency is unchanged and NPSHr scales like the
    head. Methods take the *operating* flow ``Q`` (m^3/s).

    Args:
        head_coeffs: Ascending polynomial coefficients of ``H_ref(Q_ref)`` (m).
        eta_coeffs: Ascending polynomial coefficients of ``eta(Q_ref)``, or None.
        npshr_coeffs: Ascending polynomial coefficients of ``NPSHr_ref(Q_ref)``
            (m), or None.
        speed_ratio: ``N / N_ref``.
        diameter_ratio: ``D / D_ref``.
    """

    def __init__(self, head_coeffs, eta_coeffs=None, npshr_coeffs=None, speed_ratio=1.0, diameter_ratio=1.0):
        asarr = lambda c: None if c is None else jnp.atleast_1d(jnp.asarray(c, dtype=jnp.float64))
        self.head_coeffs = asarr(head_coeffs)
        self.eta_coeffs = asarr(eta_coeffs)
        self.npshr_coeffs = asarr(npshr_coeffs)
        self.speed_ratio = jnp.asarray(speed_ratio, dtype=jnp.float64)
        self.diameter_ratio = jnp.asarray(diameter_ratio, dtype=jnp.float64)

    def tree_flatten(self):
        return (self.head_coeffs, self.eta_coeffs, self.npshr_coeffs, self.speed_ratio, self.diameter_ratio), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

    # -- scaling --------------------------------------------------------
    @property
    def flow_scale(self) -> Array:
        """``s = (N/N_ref)(D/D_ref)``: ``Q = s Q_ref``."""
        return self.speed_ratio * self.diameter_ratio

    @property
    def head_scale(self) -> Array:
        """``s_H = (N/N_ref)^2 (D/D_ref)^2``: ``H = s_H H_ref``."""
        return self.flow_scale**2

    # -- curves ---------------------------------------------------------
    def head(self, Q: Array | float) -> Array:
        """Pump head (m) at flow ``Q`` (m^3/s)."""
        return self.head_scale * _poly(self.head_coeffs, jnp.asarray(Q) / self.flow_scale)

    def efficiency(self, Q: Array | float) -> Array:
        """Hydraulic efficiency (-), floored at :data:`ETA_FLOOR`."""
        if self.eta_coeffs is None:
            raise ValueError("this pump has no efficiency curve (set eta_coeffs, or eta_max and Q_bep)")
        return jnp.maximum(_poly(self.eta_coeffs, jnp.asarray(Q) / self.flow_scale), ETA_FLOOR)

    def npshr(self, Q: Array | float) -> Array:
        """Required NPSH (m) at flow ``Q``."""
        if self.npshr_coeffs is None:
            raise ValueError("this pump has no NPSHr curve (set npshr_coeffs)")
        return self.head_scale * _poly(self.npshr_coeffs, jnp.asarray(Q) / self.flow_scale)

    def shaft_power(self, Q: Array | float, rho: Array | float) -> Array:
        """Shaft power ``rho g Q H / eta`` (W)."""
        Q = jnp.asarray(Q)
        return jnp.asarray(rho) * GRAVITY * Q * self.head(Q) / self.efficiency(Q)

    def runout_flow(self) -> Array:
        """Largest real flow where the head is zero (m^3/s); a gradient-free bracket for solvers."""
        c = jax.lax.stop_gradient(self.head_coeffs)
        lead = c[-1]
        bound = 1.0 + jnp.max(jnp.abs(c[:-1] / lead))  # Cauchy bound >= every real root

        def newton(_, x):
            return x - _poly(c, x) / _poly(jnp.polyder(c[::-1])[::-1], x)

        x = jax.lax.fori_loop(0, 200, newton, bound)
        return jax.lax.stop_gradient(x * self.flow_scale)


class SeriesCurve:
    """Pumps in series: same flow through each, heads (and powers) add."""

    def __init__(self, curves):
        self.curves = tuple(curves)

    def head(self, Q):
        return sum(c.head(Q) for c in self.curves)

    def shaft_power(self, Q, rho):
        return sum(c.shaft_power(Q, rho) for c in self.curves)

    def efficiency(self, Q, rho=1000.0):
        """Overall ``rho g Q H / sum(P_i)`` (independent of ``rho``)."""
        Q = jnp.asarray(Q)
        return rho * GRAVITY * Q * self.head(Q) / self.shaft_power(Q, rho)

    def npshr(self, Q):
        """NPSHr of the first (suction) pump."""
        return self.curves[0].npshr(Q)

    def runout_flow(self):
        return jnp.max(jnp.stack([c.runout_flow() for c in self.curves]))


class ParallelCurve:
    """Pumps in parallel: same head across each, flows add.

    ``head(Q)`` solves ``sum_i Q_i(H) = Q`` with ``H_i(Q_i) = H`` for the branch
    flows by Newton (implicit-function-theorem gradients). Each pump is assumed
    to have a check valve: a pump whose shutoff head is below the common head is
    *not* switched off here (its branch flow goes negative), so keep the
    operating point inside every pump's range.
    """

    def __init__(self, curves):
        self.curves = tuple(curves)

    def split(self, Q: Array | float) -> Array:
        """Branch flows ``Q_i`` (m^3/s) at total flow ``Q``."""
        n = len(self.curves)
        Q = jnp.asarray(Q, dtype=jnp.float64)
        if n == 1:
            return jnp.atleast_1d(Q)
        scale = jax.lax.stop_gradient(jnp.maximum(jnp.abs(Q), 1e-12))
        curves = self.curves

        def resid(x, Qt):
            q = x * scale
            h0 = curves[0].head(q[0])
            eq = [jnp.sum(q) - Qt] + [curves[i].head(q[i]) - h0 for i in range(1, n)]
            return jnp.stack(eq)

        y0 = jnp.full((n,), 1.0 / n)
        sol = optx.root_find(resid, optx.Newton(rtol=1e-12, atol=1e-10), y0, args=Q, throw=False)
        return sol.value * scale

    def head(self, Q):
        q = self.split(Q)
        return self.curves[0].head(q[0])

    def shaft_power(self, Q, rho):
        q = self.split(Q)
        return sum(c.shaft_power(q[i], rho) for i, c in enumerate(self.curves))

    def efficiency(self, Q, rho=1000.0):
        Q = jnp.asarray(Q)
        return rho * GRAVITY * Q * self.head(Q) / self.shaft_power(Q, rho)

    def npshr(self, Q):
        """Largest NPSHr over the branches at their own flows."""
        q = self.split(Q)
        return jnp.max(jnp.stack([c.npshr(q[i]) for i, c in enumerate(self.curves)]))

    def runout_flow(self):
        return sum(c.runout_flow() for c in self.curves)


def _as_curve(pump) -> Any:
    if hasattr(pump, "curve") and not hasattr(pump, "head"):
        return pump.curve
    return pump


def pumps_in_series(*pumps) -> SeriesCurve:
    """Combined curve of pumps in series: heads add at equal flow.

    Args:
        *pumps: :class:`CentrifugalPump`, :class:`PumpCurve` or combined curves.

    Returns:
        A curve with ``head``, ``efficiency``, ``npshr``, ``shaft_power`` and
        ``runout_flow``, accepted by :func:`operating_point`.
    """
    return SeriesCurve([_as_curve(p) for p in pumps])


def pumps_in_parallel(*pumps) -> ParallelCurve:
    """Combined curve of pumps in parallel: flows add at equal head.

    Args:
        *pumps: :class:`CentrifugalPump`, :class:`PumpCurve` or combined curves.

    Returns:
        A curve as for :func:`pumps_in_series` (plus ``split(Q)`` for the branch flows).
    """
    return ParallelCurve([_as_curve(p) for p in pumps])


# ----------------------------------------------------------------------
# System curve and operating point
# ----------------------------------------------------------------------
class SystemCurve:
    """``H_sys(Q) = static_head + delta_P/(rho g) + sum(pipe losses + pipe dz)``.

    Build with :func:`system_curve`. ``head(Q)`` is the head (m of fluid) the
    pump must supply to push flow ``Q`` through the system; ``losses(Q)`` is the
    flow-dependent part. The loss is odd in ``Q`` (``Q |Q|`` behaviour).
    """

    def __init__(self, static_head, pipes, delta_P, rho, inlet):
        self.static_head = static_head
        self.pipes = tuple(pipes)
        self.delta_P = delta_P
        self.rho = rho
        self.inlet = inlet

    def losses(self, Q: Array | float) -> Array:
        """Friction + fitting head loss (m) of all pipes at flow ``Q``."""
        Q = jnp.asarray(Q, dtype=jnp.float64)
        Qa = jnp.maximum(jnp.abs(Q), 1e-12)
        total = jnp.zeros(())
        for pipe in self.pipes:
            total = total + pipe.hydraulics_at_flow(Qa, self.inlet)["head_loss"]
        return jnp.sign(Q) * total

    def head(self, Q: Array | float) -> Array:
        """System head (m) at flow ``Q`` (m^3/s)."""
        H = jnp.asarray(self.static_head, dtype=jnp.float64) + self.losses(Q)
        for pipe in self.pipes:
            H = H + jnp.asarray(pipe.params.dz, dtype=jnp.float64)
        if self.delta_P is not None:
            H = H + jnp.asarray(self.delta_P) / (evaluate_property(self.rho, self.inlet) * GRAVITY)
        return H


def system_curve(
    static_head: Array | float = 0.0,
    *pipes: Pipe,
    delta_P: Array | float | None = None,
    rho: Array | float | None = None,
    inlet: Stream | None = None,
) -> SystemCurve:
    """System curve ``H_sys(Q)`` for a pump discharging through pipes.

    ``H_sys = static_head + delta_P/(rho g) + sum_i (head_loss_i(Q) + dz_i)``.
    The ``dz`` of each :class:`~difflow.units.pipe.Pipe` is counted, so give
    elevation either there or in ``static_head``, not both.

    Args:
        static_head: Elevation difference between the discharge and suction
            liquid levels (m), e.g. tank-to-tank lift.
        *pipes: :class:`~difflow.units.pipe.Pipe` runs in series (suction and
            discharge lines, with their fittings; add an ``exit`` K for a
            submerged discharge).
        delta_P: Pressure difference to be added (Pa), discharge minus
            suction vessel, e.g. a pressurised receiver. None for none.
        rho: Liquid density (kg/m^3) for the ``delta_P`` term; defaults to the
            first pipe's.
        inlet: Stream, only needed when a pipe's ``rho`` or ``mu`` is a
            callable of the stream.

    Returns:
        A :class:`SystemCurve`.
    """
    if delta_P is not None and rho is None:
        if not pipes:
            raise ValueError("delta_P needs rho (or at least one Pipe to take it from)")
        rho = pipes[0].params.rho
    return SystemCurve(static_head, pipes, delta_P, rho, inlet)


def operating_point(pump, system, Q_guess: Array | float | None = None, return_info: bool = False):
    """Flow ``Q*`` (m^3/s) where the pump head equals the system head.

    Solves ``H_pump(Q) - H_sys(Q) = 0`` by Newton in the scaled flow,
    started at the pump's zero-head (run-out) flow, from where the iteration is
    monotone for the usual concave pump curve against a convex system curve.
    Gradients with respect to pump and system parameters come from the implicit
    function theorem (``optimistix``), not by differentiating through the
    iterations. If the shutoff head is below the static head there is no
    positive root and the result is not meaningful (check ``H(0)`` first).

    Args:
        pump: :class:`CentrifugalPump`, :class:`PumpCurve` or a combined curve.
        system: A :class:`SystemCurve` (or any object with ``head(Q)``).
        Q_guess: Optional starting flow (m^3/s).
        return_info: Also return a dict with ``Q``, ``head``, ``efficiency`` and
            ``shaft_power`` (the last needs ``rho``: taken from the pump if it
            is a :class:`CentrifugalPump`, otherwise omitted).

    Returns:
        ``Q*``, or ``(Q*, info)`` when ``return_info``.
    """
    curve = _as_curve(pump)
    Q0 = jnp.asarray(Q_guess, dtype=jnp.float64) if Q_guess is not None else curve.runout_flow()
    Q0 = jax.lax.stop_gradient(Q0)

    def resid(x, _):
        Q = x * Q0
        return curve.head(Q) - system.head(Q)

    sol = optx.root_find(resid, optx.Newton(rtol=1e-12, atol=1e-10), jnp.asarray(1.0), throw=False)
    Q = sol.value * Q0
    if not return_info:
        return Q
    info = {"Q": Q, "head": curve.head(Q)}
    try:
        info["efficiency"] = curve.efficiency(Q)
    except ValueError:
        pass
    if hasattr(pump, "params") and "efficiency" in info:
        info["shaft_power"] = curve.shaft_power(Q, evaluate_property(pump.params.rho))
    return Q, info


# ----------------------------------------------------------------------
# Unit operation
# ----------------------------------------------------------------------
@dataclass(repr=False)
class CentrifugalPumpParams(ParamsMixin):
    """Parameters for :class:`CentrifugalPump`.

    Attributes:
        head_coeffs: Ascending polynomial coefficients ``[h0, h1, h2, ...]`` of
            the head curve ``H(Q)`` at the reference speed and diameter,
            Q in m^3/s and H in m (see :func:`fit_pump_curve`).
        rho: Liquid density (kg/m^3): a number or a callable ``rho(stream)``.
        eta_coeffs: Ascending polynomial coefficients of the efficiency curve
            ``eta(Q)`` at the reference conditions. Alternatively give
            ``eta_max`` and ``Q_bep`` for the BEP-centred parabola.
        eta_max: Peak efficiency (0-1), with ``Q_bep``.
        Q_bep: Flow at the best efficiency point at the reference speed and
            diameter (m^3/s).
        npshr_coeffs: Ascending polynomial coefficients of the required NPSH
            ``NPSHr(Q)`` (m) at the reference conditions; optional.
        N_ref: Reference speed of the curves (rpm).
        D_ref: Reference impeller diameter of the curves (m).
        N: Operating speed (rpm); None means ``N_ref``.
        D: Operating impeller diameter (m); None means ``D_ref``.
        motor_efficiency: Motor (and drive) efficiency (0-1); electric power
            is shaft power divided by it.
        MW: Molar mass (g/mol): a number or ``{species: MW}``; may be omitted
            when the unit is built with a ``thermo``.
        Psat: Vapour pressure (Pa) for NPSHa: a number or a callable
            ``Psat(stream)``. When None and the unit has a ``thermo``, the
            Antoine ``thermo.Psat(species, T)`` is used.
        npsh_species: Species whose vapour pressure sets NPSHa (default: the
            only species of the stream).
        D_suction: Suction-nozzle inside diameter (m) for the velocity head in
            NPSHa; None neglects it.
        cp: Liquid heat capacity (J/kg/K); with it the outlet is warmed by the
            dissipated shaft work ``g H (1 - eta)/(eta cp)``; None keeps the
            stream isothermal.
    """

    head_coeffs: list
    rho: float  # a callable rho(stream) is also accepted; typed float so forms treat it as a number
    eta_coeffs: list | None = None
    eta_max: float | None = None
    Q_bep: float | None = None
    npshr_coeffs: list | None = None
    N_ref: float = 1.0
    D_ref: float = 1.0
    N: float | None = None
    D: float | None = None
    motor_efficiency: float = 1.0
    MW: dict | float | None = None
    Psat: float | None = None
    npsh_species: str | None = None
    D_suction: float | None = None
    cp: float | None = None


class CentrifugalPump:
    """Centrifugal pump with a head-flow curve, affinity laws and NPSH.

    The inlet stream fixes the flow ``Q = m_dot / rho``; the outlet pressure is
    ``P_in + rho g H(Q)``. Returns ``(outlet, info)``; ``info`` holds ``Q``,
    ``head`` (m), ``efficiency``, ``hydraulic_power``, ``shaft_power``,
    ``electric_power`` (W), and, when the curve / vapour pressure are available,
    ``NPSHr``, ``NPSHa`` and ``npsh_margin = NPSHa - NPSHr`` (m; negative means
    cavitation risk). ``pump.curve`` is the :class:`PumpCurve`; use
    :func:`operating_point` with a :func:`system_curve` to find the flow instead.

    Differentiable in every numeric parameter, including ``N`` and ``D``.
    """

    symbol = "Pump"
    equations = [
        r"P_{\mathrm{out}} = P_{\mathrm{in}} + \rho g\,H(Q),\qquad H(Q) = h_0 + h_1 Q + h_2 Q^2",
        r"Q \propto N D,\quad H \propto N^2 D^2,\quad P \propto N^3 D^3",
        r"P_{\mathrm{shaft}} = \frac{\rho g Q H}{\eta(Q)},\qquad \eta = \eta_{\max}\left(1 - \left(\frac{Q - Q_{\mathrm{bep}}}{Q_{\mathrm{bep}}}\right)^2\right)",
        r"\mathrm{NPSH}_a = \frac{P_s - P_{\mathrm{sat}}}{\rho g} + \frac{v_s^2}{2g} + z_s - h_{f,s}",
        r"H_{\mathrm{pump}}(Q^*) = H_{\mathrm{sys}}(Q^*) = \Delta z + \sum_i h_{f,i}(Q^*)",
    ]
    assumptions = [
        "Incompressible single-phase Newtonian liquid; rho (and mu in the system curve) are inputs.",
        "Steady state; the head, efficiency and NPSHr curves are polynomials in Q fitted to vendor data and trusted only inside the fitted range.",
        "Affinity laws Q ~ N D, H ~ N^2 D^2 (so P ~ N^3 D^3) with efficiency and the NPSHr/H ratio unchanged: good for speed changes of tens of percent and small impeller trims, optimistic for large changes (efficiency drops with speed and trim).",
        "Isothermal unless cp is given; then all dissipated shaft work heats the liquid.",
        "Parallel pumps are assumed to have check valves but are not switched off below their shutoff head.",
    ]
    references = [
        "McCabe, W.L., Smith, J.C., Harriott, P. Unit Operations of Chemical Engineering, 7e, Ch. 8.",
        "Perry's Chemical Engineers' Handbook, 9e, Sec. 10.",
        "Hydraulic Institute, ANSI/HI 9.6.1 Rotodynamic Pumps: Guideline for NPSH Margin.",
        "Karassik, I.J. et al. Pump Handbook, 4e (affinity laws, system curves).",
    ]
    parameter_symbols = {
        "rho": r"\rho", "eta_max": r"\eta_{\max}", "Q_bep": r"Q_{\mathrm{bep}}",
        "N_ref": r"N_{\mathrm{ref}}", "D_ref": r"D_{\mathrm{ref}}", "N": "N", "D": "D",
        "motor_efficiency": r"\eta_m", "D_suction": r"D_s", "cp": r"c_p",
    }
    parameter_units = {
        "head_coeffs": "m, m/(m^3/s), ...", "rho": "kg/m^3", "eta_coeffs": "-", "eta_max": "-",
        "Q_bep": "m^3/s", "npshr_coeffs": "m, m/(m^3/s), ...", "N_ref": "rpm", "D_ref": "m",
        "N": "rpm", "D": "m", "motor_efficiency": "-", "MW": "g/mol", "Psat": "Pa",
        "npsh_species": "-", "D_suction": "m", "cp": "J/kg/K",
    }

    def __init__(self, params: CentrifugalPumpParams, thermo=None):
        """Initialize the unit.

        Args:
            params: Pump parameters.
            thermo: Optional thermo with a ``species`` map (for ``MW``) and a
                ``Psat(species, T)`` method (for NPSHa).
        """
        self.params = params
        self.thermo = thermo

    # ------------------------------------------------------------------
    @property
    def curve(self) -> PumpCurve:
        """The :class:`PumpCurve` at the operating speed and diameter."""
        p = self.params
        eta = p.eta_coeffs
        if eta is None and p.eta_max is not None and p.Q_bep is not None:
            eta = bep_efficiency_coeffs(p.eta_max, p.Q_bep)
        N = p.N_ref if p.N is None else p.N
        D = p.D_ref if p.D is None else p.D
        return PumpCurve(
            p.head_coeffs, eta, p.npshr_coeffs,
            jnp.asarray(N, dtype=jnp.float64) / p.N_ref,
            jnp.asarray(D, dtype=jnp.float64) / p.D_ref,
        )

    def _psat(self, inlet: Stream):
        p = self.params
        if p.Psat is not None:
            return evaluate_property(p.Psat, inlet)
        if self.thermo is not None and hasattr(self.thermo, "Psat"):
            species = p.npsh_species
            if species is None:
                names = get_species(inlet)
                if len(names) != 1:
                    raise ValueError("a multi-species stream needs CentrifugalPumpParams(npsh_species=...)")
                species = names[0]
            return self.thermo.Psat(species, inlet["T"])
        return None

    def _performance(self, inlet: Stream) -> dict:
        p = self.params
        rho = evaluate_property(p.rho, inlet)
        Q = stream_mass_flow(inlet, p.MW, self.thermo) / rho
        curve = self.curve
        H = curve.head(Q)
        info = {"Q": Q, "head": H, "dP": rho * GRAVITY * H}
        if curve.eta_coeffs is not None:
            eta = curve.efficiency(Q)
            hyd = rho * GRAVITY * Q * H
            shaft = hyd / eta
            info.update(
                efficiency=eta, hydraulic_power=hyd, shaft_power=shaft,
                electric_power=shaft / jnp.asarray(p.motor_efficiency, dtype=jnp.float64),
            )
        psat = self._psat(inlet)
        if psat is not None:
            v = 0.0
            if p.D_suction is not None:
                v = Q / (jnp.pi / 4.0 * jnp.asarray(p.D_suction, dtype=jnp.float64) ** 2)
            info["NPSHa"] = npsh_available(inlet["P"], psat, rho, v)
            if curve.npshr_coeffs is not None:
                info["NPSHr"] = curve.npshr(Q)
                info["npsh_margin"] = info["NPSHa"] - info["NPSHr"]
        return info

    def _T_out(self, inlet: Stream, info: dict) -> Array:
        cp = self.params.cp
        if cp is None or "efficiency" not in info:
            return inlet["T"]
        eta = info["efficiency"]
        return inlet["T"] + GRAVITY * info["head"] * (1.0 - eta) / (eta * jnp.asarray(cp, dtype=jnp.float64))

    def __call__(self, inlet: Stream) -> tuple[Stream, dict]:
        """Compute the outlet stream and pump performance at the inlet flow.

        Args:
            inlet: Liquid stream at the suction flange (``P`` is the absolute
                suction pressure).

        Returns:
            ``(outlet, info)`` as described on the class.
        """
        info = self._performance(inlet)
        outlet = make_stream(get_flows(inlet), self._T_out(inlet, info), inlet["P"] + info["dP"])
        return outlet, info

    # ------------------------------------------------------------------
    def eo_residuals(self, inlets: list[Stream], outlets: list[Stream], **kwargs) -> Array:
        """Residuals for the EO solver.

        Residuals:
            F_out_i - F_in_i = 0                        (n_species)
            T_out - T_out(inlet) = 0                    (1)
            P_out - (P_in + rho g H(Q_in)) = 0          (1)

        The temperature and pressure rows use the algebraic expressions
        ``__call__`` uses, evaluated at the inlet.

        Args:
            inlets: [inlet_stream]
            outlets: [outlet_stream]

        Returns:
            Flat residual array, length n_species + 2
        """
        inlet, outlet = inlets[0], outlets[0]
        inflows, outflows = get_flows(inlet), get_flows(outlet)
        resid = [jnp.atleast_1d(outflows[s] - inflows[s]) for s in get_species(inlet)]
        info = self._performance(inlet)
        resid.append(jnp.atleast_1d(outlet["T"] - self._T_out(inlet, info)))
        resid.append(jnp.atleast_1d(outlet["P"] - (inlet["P"] + info["dP"])))
        return jnp.concatenate(resid)


__all__ = [
    "CentrifugalPump", "CentrifugalPumpParams", "PumpCurve", "SeriesCurve", "ParallelCurve",
    "SystemCurve", "system_curve", "operating_point", "pumps_in_series", "pumps_in_parallel",
    "fit_pump_curve", "npsh_available", "bep_efficiency_coeffs", "ETA_FLOOR",
]
