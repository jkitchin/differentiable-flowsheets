"""Electrical physics: per-unit conversions, branch admittances, generator costs.

Steady-state, positive-sequence, balanced three-phase modelling in the
standard form used by MATPOWER, PYPOWER, PowerModels and pandapower, so
that a network built here can be compared number for number against
those tools.

Everything in this module is a pure function of JAX-compatible values:
the branch parameters ``r``, ``x``, ``b``, ``tap`` and ``shift`` may be
traced, so ``jax.grad`` with respect to a line reactance or a
transformer tap is available without any special support elsewhere.

Conventions
-----------

===============  ==========================================
quantity         unit
===============  ==========================================
voltage          per unit (pu) on the bus base kV
angle            radians
admittance       per unit on ``base_mva``
power (state)    per unit on ``base_mva``
power (reports)  MW / MVAr
cost             $/h, with real power in MW
===============  ==========================================

Per unit is not a convenience here, it is what makes the problem
solvable: on a 100 MVA base every voltage sits near 1.0 and every
injection is O(1), so the power-flow Jacobian is naturally scaled and
an interior-point method does not have to fight a 10^5 spread between
volts and watts. Costs are the one exception -- utilities quote
``$/MWh``, and a cost curve in per unit would be unreadable -- so
:func:`polynomial_cost` takes MW and the OPF converts at the boundary.

The branch model
----------------

A single model covers transmission lines, transformers and phase
shifters. The series admittance ``ys = 1 / (r + jx)`` sits between two
halves of the total charging susceptance ``b``, and an ideal
transformer with complex ratio ``t = tau * exp(j theta)`` sits at the
*from* end::

    Yff = (ys + j b / 2) / tau^2      Yft = -ys / conj(t)
    Ytf = -ys / t                     Ytt =  ys + j b / 2

so that ``[I_from; I_to] = [[Yff, Yft], [Ytf, Ytt]] [V_from; V_to]``.
With ``tau = 1`` and ``theta = 0`` this is a plain pi-model line; with
``b = 0`` and ``tau != 1`` a tap-changing transformer; with
``theta != 0`` a phase-shifting transformer. There is no separate unit
operation for the three, and no place for their conventions to drift
apart.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax import Array

#: default system power base (MVA); the near-universal choice
DEFAULT_BASE_MVA = 100.0

#: nominal system frequency (Hz), for reactance/inductance conversions
DEFAULT_FREQUENCY_HZ = 60.0

#: smoothing (pu) used where a magnitude would otherwise be kinked at 0
EPS_POWER = 1.0e-8

PI = math.pi


# =============================================================================
# Per-unit conversions
# =============================================================================


def base_impedance(base_kv: float, base_mva: float = DEFAULT_BASE_MVA) -> float:
    """Base impedance ``Z_base = kV^2 / MVA`` in ohms."""
    return base_kv * base_kv / base_mva


def ohms_to_pu(
    z_ohm, base_kv: float, base_mva: float = DEFAULT_BASE_MVA
):
    """Convert an impedance in ohms to per unit."""
    return z_ohm / base_impedance(base_kv, base_mva)


def pu_to_ohms(
    z_pu, base_kv: float, base_mva: float = DEFAULT_BASE_MVA
):
    """Convert a per-unit impedance to ohms."""
    return z_pu * base_impedance(base_kv, base_mva)


def siemens_to_pu(
    y_siemens, base_kv: float, base_mva: float = DEFAULT_BASE_MVA
):
    """Convert an admittance in siemens to per unit."""
    return y_siemens * base_impedance(base_kv, base_mva)


def mw_to_pu(p_mw, base_mva: float = DEFAULT_BASE_MVA):
    """Convert MW (or MVAr, or MVA) to per unit."""
    return p_mw / base_mva


def pu_to_mw(p_pu, base_mva: float = DEFAULT_BASE_MVA):
    """Convert per unit to MW (or MVAr, or MVA)."""
    return p_pu * base_mva


def line_reactance_pu(
    length_km: float,
    x_ohm_per_km: float,
    base_kv: float,
    base_mva: float = DEFAULT_BASE_MVA,
) -> float:
    """Series reactance (pu) of a line from its per-km rating."""
    return ohms_to_pu(length_km * x_ohm_per_km, base_kv, base_mva)


def line_charging_pu(
    length_km: float,
    c_nf_per_km: float,
    base_kv: float,
    base_mva: float = DEFAULT_BASE_MVA,
    frequency_hz: float = DEFAULT_FREQUENCY_HZ,
) -> float:
    """Total charging susceptance (pu) of a line from its capacitance.

    ``b_total = 2 pi f C L`` in siemens, converted to per unit. This is
    the ``b`` that :func:`branch_admittances` splits in half between the
    two ends.
    """
    b_siemens = 2.0 * PI * frequency_hz * (c_nf_per_km * 1e-9) * length_km
    return siemens_to_pu(b_siemens, base_kv, base_mva)


# =============================================================================
# Branch admittances
# =============================================================================


def branch_admittances(r, x, b, tap=1.0, shift=0.0, g=0.0):
    """The 2x2 admittance block of a branch, as four complex arrays.

    Args:
        r: series resistance (pu).
        x: series reactance (pu). Must be non-zero somewhere in
            ``r + jx``; a zero-impedance branch has no admittance.
        b: TOTAL line charging susceptance (pu), split half at each end.
        tap: off-nominal turns ratio magnitude at the *from* end
            (1.0 for a line). MATPOWER's convention of 0 meaning 1 is
            resolved in :class:`~difflow_power.network.Branch`, not
            here.
        shift: phase shift angle (radians), positive meaning the from
            side leads.
        g: total line charging conductance (pu). Zero for every
            standard case file; carried because the pi-model admits it.

    Returns:
        ``(yff, yft, ytf, ytt)``, complex, broadcast to the shape of the
        inputs.

    Example:
        >>> yff, yft, ytf, ytt = branch_admittances(0.01, 0.1, 0.05)
        >>> bool(jnp.allclose(yft, ytf))   # symmetric with no tap
        True
    """
    r = jnp.asarray(r, dtype=jnp.float64)
    x = jnp.asarray(x, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    g = jnp.asarray(g, dtype=jnp.float64) * jnp.ones_like(r)
    tap = jnp.asarray(tap, dtype=jnp.float64) * jnp.ones_like(r)
    shift = jnp.asarray(shift, dtype=jnp.float64) * jnp.ones_like(r)

    ys = 1.0 / (r + 1j * x)
    y_charge_half = 0.5 * (g + 1j * b)
    t = tap * jnp.exp(1j * shift)

    yff = (ys + y_charge_half) / (tap * tap)
    yft = -ys / jnp.conj(t)
    ytf = -ys / t
    ytt = ys + y_charge_half
    return yff, yft, ytf, ytt


def build_ybus(
    n_bus: int,
    from_idx,
    to_idx,
    yff,
    yft,
    ytf,
    ytt,
    y_shunt=None,
) -> Array:
    """Assemble the bus admittance matrix from branch blocks.

    ``Ybus = Cf^T Yf + Ct^T Yt + diag(y_shunt)`` written as scatter-adds
    so the whole assembly is traceable: differentiate a solved state
    with respect to a line reactance and the derivative flows through
    here.

    The matrix is DENSE. That is the right default for the sizes this
    plugin targets (a few hundred buses, where a dense ``n^2`` complex
    matrix is a few MB and JAX's dense linear algebra is far faster than
    any Python-level sparse path), and the wrong one beyond a few
    thousand.

    Args:
        n_bus: number of buses.
        from_idx: branch from-bus indices, shape ``(n_branch,)``.
        to_idx: branch to-bus indices, shape ``(n_branch,)``.
        yff, yft, ytf, ytt: branch admittance blocks, shape
            ``(n_branch,)``, from :func:`branch_admittances`.
        y_shunt: per-bus shunt admittance (pu), shape ``(n_bus,)``.

    Returns:
        Complex ``(n_bus, n_bus)`` bus admittance matrix.
    """
    from_idx = jnp.asarray(from_idx)
    to_idx = jnp.asarray(to_idx)
    ybus = jnp.zeros((n_bus, n_bus), dtype=jnp.complex128)
    ybus = ybus.at[from_idx, from_idx].add(yff)
    ybus = ybus.at[from_idx, to_idx].add(yft)
    ybus = ybus.at[to_idx, from_idx].add(ytf)
    ybus = ybus.at[to_idx, to_idx].add(ytt)
    if y_shunt is not None:
        ybus = ybus + jnp.diag(jnp.asarray(y_shunt, dtype=jnp.complex128))
    return ybus


def bus_injections(v_complex: Array, ybus: Array) -> Array:
    """Complex power injected INTO the network at each bus (pu).

    ``S = V . conj(Ybus V)``, the single equation the whole plugin turns
    on. Positive real part means power flowing from the bus into the
    network, i.e. net generation.
    """
    return v_complex * jnp.conj(ybus @ v_complex)


def voltage_rectangular(vm: Array, va: Array) -> Array:
    """Complex bus voltages from magnitude (pu) and angle (rad)."""
    return vm * jnp.exp(1j * va)


def branch_power_flows(v_complex, from_idx, to_idx, yff, yft, ytf, ytt):
    """Complex power entering each branch at its two ends (pu).

    Both are measured INTO the branch, so ``s_from + s_to`` is the
    branch's own loss -- a positive real number for any passive branch,
    which is a useful invariant to assert on.

    Returns:
        ``(s_from, s_to)``, complex, shape ``(n_branch,)``.
    """
    vf = v_complex[jnp.asarray(from_idx)]
    vt = v_complex[jnp.asarray(to_idx)]
    i_from = yff * vf + yft * vt
    i_to = ytf * vf + ytt * vt
    return vf * jnp.conj(i_from), vt * jnp.conj(i_to)


# =============================================================================
# Generator cost curves
# =============================================================================


def polynomial_cost(p_mw, coefficients) -> Array:
    """Generator cost ($/h) as a polynomial in real power (MW).

    ``coefficients`` is in MATPOWER's ``gencost`` order, HIGHEST power
    first: ``[c2, c1, c0]`` means ``c2 P^2 + c1 P + c0``. Any degree is
    accepted; quadratic is what every standard case file uses.

    Args:
        p_mw: real power output (MW), scalar or array.
        coefficients: polynomial coefficients, highest order first.

    Returns:
        Cost in $/h, same shape as ``p_mw``.
    """
    coefficients = jnp.asarray(coefficients, dtype=jnp.float64)
    p_mw = jnp.asarray(p_mw, dtype=jnp.float64)
    total = jnp.zeros_like(p_mw)
    for c in coefficients:            # Horner; length is static Python
        total = total * p_mw + c
    return total


def marginal_cost(p_mw, coefficients) -> Array:
    """Derivative of :func:`polynomial_cost` w.r.t. MW, in $/MWh.

    Written out rather than obtained by ``jax.grad`` so it also works
    for plain floats and reads as the economics it is: for the usual
    quadratic this is ``2 c2 P + c1``, the offer curve a generator
    bids.
    """
    coefficients = jnp.asarray(coefficients, dtype=jnp.float64)
    p_mw = jnp.asarray(p_mw, dtype=jnp.float64)
    n = coefficients.shape[0]
    total = jnp.zeros_like(p_mw)
    for i, c in enumerate(coefficients):
        power = n - 1 - i
        if power >= 1:
            total = total + c * power * p_mw ** (power - 1)
    return total


def apparent_power_squared(s_complex) -> Array:
    """``|S|^2`` (pu^2) from a complex power.

    Thermal limits are always posed on the SQUARE. ``|S|`` has an
    unbounded second derivative at the origin, and a branch carrying
    almost nothing is exactly where an interior-point method will put a
    Newton step early on; the squared form is a smooth quadratic there
    and gives the identical feasible set for a non-negative rating.
    """
    return jnp.real(s_complex) ** 2 + jnp.imag(s_complex) ** 2
