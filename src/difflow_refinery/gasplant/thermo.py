"""Cubic-EOS thermodynamics for the gas plant, vectorised over stages.

One equation of state covers the real light ends and the naphtha
pseudocomponents alike (see :mod:`~difflow_refinery.gasplant.components`).
Every function broadcasts over leading axes: ``T`` and ``P`` of shape
``(N,)`` with compositions ``(N, C)`` evaluate a whole column in one call,
which is what the stage-network Newton differentiates.

The model is pluggable. A column reads only three methods of its thermo
object --

``log_K(T, P, x, y)``
    ``ln K_i = ln phi_i^L(x) - ln phi_i^V(y)``, ``(..., C)``;
``h_liquid(T, P, x)`` / ``h_vapor(T, P, y)``
    molar enthalpy of the phase (J/mol, ideal gas at 298.15 K reference),
    ``(...,)``;

so any object with those (an activity-coefficient model, a tabulated K) can
stand in for :class:`CubicThermo`. Two cubics are provided:
:data:`PR` (Peng & Robinson 1976) and :data:`SRK` (Soave 1972).

The cubic root
    The compressibility is taken from the closed form (Cardano, or the
    trigonometric form when there are three real roots) with gradients
    stopped, polished by one Newton step without and one with gradients. The
    last step has the root's exact implicit derivative, ``dZ = -df/f'``,
    so nothing differentiates through ``arccos`` or ``cbrt``. A liquid takes
    the smallest root above ``B``, a vapor the largest. Where the cubic has
    one real root both phases take it -- the EOS reporting a single phase --
    and ``K`` is identically one, as for every cubic-EOS package.

Departure functions (generic two-parameter cubic, ``delta1, delta2`` the
roots of the attractive-term denominator; PR ``1 +/- sqrt 2``, SRK ``1, 0``)::

    ln phi_i = b_i/b (Z-1) - ln(Z-B)
               - A/(B (d1-d2)) (2 sum_j x_j a_ij / a - b_i/b) ln((Z+d1 B)/(Z+d2 B))
    H^R = RT(Z-1) + (T a' - a)/(b (d1-d2)) ln((Z+d1 B)/(Z+d2 B))
    S^R = R ln(Z-B) + a'/(b (d1-d2)) ln((Z+d1 B)/(Z+d2 B))

References
    Peng, D.-Y. & Robinson, D. B., *Ind. Eng. Chem. Fundam.* 15, 59-64 (1976).
    Soave, G., *Chem. Eng. Sci.* 27, 1197-1203 (1972).
    Poling, Prausnitz & O'Connell, *The Properties of Gases and Liquids*,
    5th ed. (2001), Ch. 4-6.
    Wilson, G. M., "A modified Redlich-Kwong equation of state ..." (1968)
    for the K-value initial estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from difflow_refinery.gasplant.components import GasComponents, R, T_REF

SQRT2 = 2.0 ** 0.5


@dataclass(frozen=True)
class Cubic:
    """A two-parameter cubic equation of state.

    Attributes:
        name: Label.
        omega_a: ``a_c = omega_a R^2 Tc^2 / Pc``.
        omega_b: ``b = omega_b R Tc / Pc``.
        delta1: First root of the attractive-term denominator.
        delta2: Second root.
        kappa: ``omega -> kappa`` in ``alpha = (1 + kappa (1 - sqrt Tr))^2``.
    """

    name: str
    omega_a: float
    omega_b: float
    delta1: float
    delta2: float
    kappa: Callable


#: Peng-Robinson (1976), with the 1976 kappa -- the form IDAES's
#: ``CubicType.PR`` and :class:`difflow.eos.PengRobinson` use.
PR = Cubic("PR", 0.45724, 0.07780, 1.0 + SQRT2, 1.0 - SQRT2,
           lambda w: 0.37464 + 1.54226 * w - 0.26992 * w * w)
#: Soave-Redlich-Kwong (1972).
SRK = Cubic("SRK", 0.42748, 0.08664, 1.0, 0.0,
            lambda w: 0.480 + 1.574 * w - 0.176 * w * w)


def _cubic_roots(c2, c1, c0):
    """Smallest and largest real roots of ``Z^3 + c2 Z^2 + c1 Z + c0``."""
    p = c1 - c2 * c2 / 3.0
    q = 2.0 * c2 ** 3 / 27.0 - c2 * c1 / 3.0 + c0
    disc = (q / 2.0) ** 2 + (p / 3.0) ** 3
    sq = jnp.sqrt(jnp.maximum(disc, 0.0))
    one = jnp.cbrt(-q / 2.0 + sq) + jnp.cbrt(-q / 2.0 - sq)
    pn = jnp.minimum(p, -1e-300)
    m = 2.0 * jnp.sqrt(-pn / 3.0)
    arg = jnp.clip(3.0 * q / (pn * m), -1.0, 1.0)
    phi = jnp.arccos(arg) / 3.0
    hi3 = m * jnp.cos(phi)
    lo3 = m * jnp.cos(phi - 4.0 * jnp.pi / 3.0)
    three = disc < 0.0
    shift = -c2 / 3.0
    lo = jnp.where(three, lo3, one) + shift
    hi = jnp.where(three, hi3, one) + shift
    return lo, hi


class CubicThermo:
    """Phase properties of a :class:`GasComponents` mixture on a cubic EOS.

    Args:
        comps: The components (arrays may be traced).
        eos: :data:`PR` (default) or :data:`SRK`.
    """

    def __init__(self, comps: GasComponents, eos: Cubic = PR):
        self.comps = comps
        self.eos = eos
        Tc, Pc = comps.Tc, comps.Pc
        self.ac = eos.omega_a * (R * Tc) ** 2 / Pc
        self.b_i = eos.omega_b * R * Tc / Pc
        self.kappa = eos.kappa(comps.omega)
        self.one_minus_k = 1.0 - comps.kij

    # -- pure-component pieces -------------------------------------------

    def a_i(self, T):
        """``a_i(T)`` and ``da_i/dT``, each ``(..., C)``."""
        Tc = self.comps.Tc
        T = jnp.asarray(T)[..., None]
        sq = jnp.sqrt(T / Tc)
        m = 1.0 + self.kappa * (1.0 - sq)
        a = self.ac * m * m
        da = -self.ac * self.kappa * m / jnp.sqrt(T * Tc)
        return a, da

    def h_ig(self, T):
        """Ideal-gas enthalpy of each component, ``(..., C)`` (J/mol)."""
        c = self.comps.cp
        T = jnp.asarray(T)[..., None]

        def anti(t):
            return (c[:, 0] * t + c[:, 1] * t ** 2 / 2 + c[:, 2] * t ** 3 / 3
                    + c[:, 3] * t ** 4 / 4)
        return anti(T) - anti(T_REF)

    def s_ig(self, T):
        """Ideal-gas entropy of each component at 1 bar, ``(..., C)`` (J/mol/K),
        relative to 298.15 K."""
        c = self.comps.cp
        T = jnp.asarray(T)[..., None]

        def anti(t):
            return (c[:, 0] * jnp.log(t) + c[:, 1] * t + c[:, 2] * t ** 2 / 2
                    + c[:, 3] * t ** 3 / 3)
        return anti(T) - anti(T_REF)

    def log_K_wilson(self, T, P):
        """Wilson's estimate of ``ln K``, ``(..., C)``."""
        c = self.comps
        T = jnp.asarray(T)[..., None]
        P = jnp.asarray(P)[..., None]
        return jnp.log(c.Pc / P) + 5.373 * (1.0 + c.omega) * (1.0 - c.Tc / T)

    # -- mixture ----------------------------------------------------------

    def _mix(self, T, P, x):
        """Mixture parameters and both cubic roots, gradients exact."""
        e = self.eos
        T = jnp.asarray(T)
        P = jnp.asarray(P)
        x = x / jnp.sum(x, -1, keepdims=True)
        a, da = self.a_i(T)                                  # (..., C)
        sa = jnp.sqrt(a)
        aij = sa[..., :, None] * sa[..., None, :] * self.one_minus_k
        # d sqrt(a_i a_j)/dT = sqrt(a_i a_j) (a_i'/a_i + a_j'/a_j) / 2
        r = da / a
        daij = 0.5 * aij * (r[..., :, None] + r[..., None, :])
        sum_xa = jnp.einsum("...ij,...j->...i", aij, x)       # (..., C)
        am = jnp.sum(x * sum_xa, -1)
        dam = jnp.einsum("...i,...ij,...j->...", x, daij, x)
        bm = jnp.sum(x * self.b_i, -1)
        A = am * P / (R * T) ** 2
        B = bm * P / (R * T)
        u = e.delta1 + e.delta2
        w = e.delta1 * e.delta2
        c2 = -(1.0 + B - u * B)
        c1 = A + w * B * B - u * B - u * B * B
        c0 = -(A * B + w * B * B + w * B ** 3)
        return dict(T=T, P=P, x=x, sum_xa=sum_xa, am=am, dam=dam, bm=bm,
                    A=A, B=B, c=(c2, c1, c0))

    @staticmethod
    def _polish(Z, c):
        c2, c1, c0 = c
        f = Z ** 3 + c2 * Z * Z + c1 * Z + c0
        fp = 3 * Z * Z + 2 * c2 * Z + c1
        fp = jnp.where(jnp.abs(fp) < 1e-12, jnp.where(fp < 0, -1e-12, 1e-12), fp)
        return Z - f / fp

    def Z(self, m, phase):
        """Compressibility of ``phase`` ('liquid' or 'vapor')."""
        c = jax.lax.stop_gradient(m["c"])
        lo, hi = _cubic_roots(*c)
        B = jax.lax.stop_gradient(m["B"])
        if phase == "liquid":
            Z0 = jnp.where(lo > B, lo, hi)
        else:
            Z0 = hi
        Z0 = jax.lax.stop_gradient(self._polish(Z0, c))
        return self._polish(Z0, m["c"])

    def _log_term(self, Z, B):
        e = self.eos
        return jnp.log((Z + e.delta1 * B) / (Z + e.delta2 * B)) / (e.delta1 - e.delta2)

    def ln_phi(self, T, P, x, phase):
        """``ln`` fugacity coefficients, ``(..., C)``."""
        m = self._mix(T, P, x)
        Z = self.Z(m, phase)
        A, B, am, bm = m["A"], m["B"], m["am"], m["bm"]
        bb = self.b_i / bm[..., None]
        L = self._log_term(Z, B)
        return (bb * (Z - 1.0)[..., None] - jnp.log(Z - B)[..., None]
                - (A / B * L)[..., None] * (2.0 * m["sum_xa"] / am[..., None] - bb))

    def log_K(self, T, P, x, y):
        """``ln K = ln phi^L(x) - ln phi^V(y)``, ``(..., C)``."""
        return self.ln_phi(T, P, x, "liquid") - self.ln_phi(T, P, y, "vapor")

    def h_departure(self, T, P, x, phase):
        """``H - H^ig`` (J/mol), ``(...,)``."""
        m = self._mix(T, P, x)
        Z = self.Z(m, phase)
        return (R * m["T"] * (Z - 1.0)
                + (m["T"] * m["dam"] - m["am"]) / m["bm"] * self._log_term(Z, m["B"]))

    def s_departure(self, T, P, x, phase):
        """``S - S^ig`` at the same T and P (J/mol/K), ``(...,)``."""
        m = self._mix(T, P, x)
        Z = self.Z(m, phase)
        return R * jnp.log(Z - m["B"]) + m["dam"] / m["bm"] * self._log_term(Z, m["B"])

    def h(self, T, P, x, phase):
        """Molar enthalpy of ``phase`` at composition ``x`` (J/mol)."""
        xn = x / jnp.sum(x, -1, keepdims=True)
        return jnp.sum(xn * self.h_ig(T), -1) + self.h_departure(T, P, x, phase)

    def s(self, T, P, x, phase):
        """Molar entropy of ``phase`` (J/mol/K), ideal gas at 298.15 K, 1 bar."""
        xn = x / jnp.sum(x, -1, keepdims=True)
        xs = jnp.maximum(xn, 1e-300)
        s_ig = (jnp.sum(xn * self.s_ig(T), -1) - R * jnp.log(jnp.asarray(P) / 1e5)
                - R * jnp.sum(xn * jnp.log(xs), -1))
        return s_ig + self.s_departure(T, P, x, phase)

    def h_liquid(self, T, P, x):
        return self.h(T, P, x, "liquid")

    def h_vapor(self, T, P, y):
        return self.h(T, P, y, "vapor")

    def molar_volume(self, T, P, x, phase):
        """Molar volume (m^3/mol). Untranslated: a PR liquid volume is
        typically 5-15% high for light hydrocarbons."""
        m = self._mix(T, P, x)
        return self.Z(m, phase) * R * m["T"] / m["P"]


# ---------------------------------------------------------------------------
# Flash, bubble point and the feed state
# ---------------------------------------------------------------------------


def _rr(z, K, lo, hi):
    """Rachford-Rice vapor fraction in ``[lo, hi]`` by bisection on its
    monotone function (negative flash allowed)."""

    def g(beta):
        return jnp.sum(z * (K - 1.0) / (1.0 + beta * (K - 1.0)))

    def body(_, b):
        a, c = b
        mid = 0.5 * (a + c)
        pos = g(mid) > 0
        return (jnp.where(pos, mid, a), jnp.where(pos, c, mid))

    a, c = jax.lax.fori_loop(0, 80, body, (lo, hi))
    return 0.5 * (a + c)


def flash_tp(thermo, z, T, P, n_iter: int = 60):
    """Isothermal flash of ``z`` at ``(T, P)`` by successive substitution.

    No gradients flow through this; it decides the phase state and
    supplies the starting point for :func:`feed_state`'s Newton step.

    Returns:
        ``(beta, lnK)``, with ``beta`` the vapor fraction from the negative
        flash (below 0: subcooled liquid; above 1: superheated vapor).
    """
    z = jax.lax.stop_gradient(z / jnp.sum(z))
    T, P = jax.lax.stop_gradient(T), jax.lax.stop_gradient(P)
    lnK0 = thermo.log_K_wilson(T, P)

    def body(_, lnK):
        K = jnp.exp(lnK)
        kmax, kmin = jnp.max(K), jnp.min(K)
        lo = jnp.where(kmax > 1.0, 1.0 / (1.0 - kmax), -1e3) + 1e-9
        hi = jnp.where(kmin < 1.0, 1.0 / (1.0 - kmin), 1e3) - 1e-9
        lo, hi = jnp.maximum(lo, -50.0), jnp.minimum(hi, 50.0)
        beta = _rr(z, K, lo, hi)
        x = z / (1.0 + beta * (K - 1.0))
        x = jnp.maximum(x, 1e-300)
        y = K * x
        new = thermo.log_K(T, P, x / jnp.sum(x), y / jnp.sum(y))
        return jnp.where(jnp.isfinite(new), new, lnK)

    lnK = jax.lax.fori_loop(0, n_iter, body, lnK0)
    K = jnp.exp(lnK)
    kmax, kmin = jnp.max(K), jnp.min(K)
    lo = jnp.maximum(jnp.where(kmax > 1.0, 1.0 / (1.0 - kmax), -1e3) + 1e-9, -50.0)
    hi = jnp.minimum(jnp.where(kmin < 1.0, 1.0 / (1.0 - kmin), 1e3) - 1e-9, 50.0)
    beta = _rr(z, K, lo, hi)
    trivial = jnp.max(jnp.abs(lnK)) < 1e-4
    beta = jnp.where(trivial, jnp.where(thermo.Z(thermo._mix(T, P, z), "vapor") > 0.3,
                                        2.0, -1.0), beta)
    return beta, lnK


def _two_phase_residual(thermo, u, z, T, P):
    """Flash equations in ``(ln K, beta)``: ``C`` equilibria plus RR."""
    lnK, beta = u[:-1], u[-1]
    K = jnp.exp(lnK)
    x = z / (1.0 + beta * (K - 1.0))
    y = K * x
    r_eq = lnK - thermo.log_K(T, P, x, y)
    r_rr = jnp.sum(z * (K - 1.0) / (1.0 + beta * (K - 1.0)))
    return jnp.concatenate([r_eq, r_rr[None]])


def feed_state(thermo, flows, T, P):
    """Total enthalpy and vapor fraction of a stream at ``(T, P)``.

    Phase from :func:`flash_tp` (no gradients); then one Newton step on the
    flash equations with the Jacobian frozen, so the result carries the
    exact implicit derivative with respect to ``flows``, ``T``, ``P`` and
    any traced component constant. ``lax.cond`` evaluates (and
    differentiates) only the branch the phase state selects.

    Returns:
        dict with ``H`` (W, flow times molar enthalpy), ``beta`` (molar vapor
        fraction, clipped to [0, 1]), ``x`` and ``y`` (phase compositions; the
        feed composition for an absent phase).
    """
    Ft = jnp.sum(flows)
    z = flows / Ft
    beta_s, lnK_s = flash_tp(thermo, z, T, P)
    idx = jnp.where(beta_s <= 0.0, 0, jnp.where(beta_s >= 1.0, 1, 2))

    def liquid(_):
        return Ft * thermo.h_liquid(T, P, z), jnp.zeros(()), z, z

    def vapor(_):
        return Ft * thermo.h_vapor(T, P, z), jnp.ones(()), z, z

    def two(_):
        u0 = jax.lax.stop_gradient(jnp.concatenate([lnK_s, beta_s[None]]))
        zs = jax.lax.stop_gradient(z)
        Ts, Ps = jax.lax.stop_gradient(T), jax.lax.stop_gradient(P)

        def body(_, u):
            r = _two_phase_residual(thermo, u, zs, Ts, Ps)
            J = jax.jacfwd(_two_phase_residual, argnums=1)(thermo, u, zs, Ts, Ps)
            return u - jnp.linalg.solve(J, r)

        u0 = jax.lax.stop_gradient(jax.lax.fori_loop(0, 4, body, u0))
        J = jax.lax.stop_gradient(
            jax.jacfwd(_two_phase_residual, argnums=1)(thermo, u0, zs, Ts, Ps))
        u = u0 - jnp.linalg.solve(J, _two_phase_residual(thermo, u0, z, T, P))
        K = jnp.exp(u[:-1])
        beta = u[-1]
        x = z / (1.0 + beta * (K - 1.0))
        y = K * x
        H = Ft * ((1.0 - beta) * thermo.h_liquid(T, P, x) + beta * thermo.h_vapor(T, P, y))
        return H, beta, x, y

    H, beta, x, y = jax.lax.switch(idx, [liquid, vapor, two], None)
    return dict(H=H, beta=beta, x=x, y=y)


def bubble_pressure(thermo, x, T, P0=None, n_iter: int = 40):
    """Bubble-point pressure of liquid ``x`` at ``T`` (implicit gradients).

    Unknowns ``(ln y, ln P)``; equations ``ln y_i = ln K_i(x, y) + ln x_i``
    and ``sum y = 1``. Newton without gradients, then one with the
    Jacobian frozen.
    """
    return vapor_pressure_vl(thermo, x, T, 0.0, P0=P0, n_iter=n_iter)


def vapor_pressure_vl(thermo, z, T, vl_ratio, P0=None, n_iter: int = 40):
    """Equilibrium pressure of liquid ``z`` expanded into ``vl_ratio`` times
    its own volume of vapor space at ``T`` -- the Reid vapor pressure
    construction of ASTM D323 (``vl_ratio = 4``, 100 F), on the EOS.

    The vapor space is ``vl_ratio`` times the liquid's molar volume
    (untranslated EOS liquid volume, see :meth:`CubicThermo.molar_volume`);
    its contents are ``n_V = P V / (Z^V R T)`` moles per mole of liquid
    charged, and the liquid left behind is ``x = (z - n_V y)/(1 - n_V)``.
    ``vl_ratio = 0`` gives the bubble point (true vapor pressure).

    Returns:
        The pressure (Pa), with implicit-function gradients.
    """
    z = z / jnp.sum(z)
    T = jnp.asarray(T, dtype=float)
    zs, Ts = jax.lax.stop_gradient(z), jax.lax.stop_gradient(T)
    v_L = thermo.molar_volume(T, 1e5, z, "liquid")

    def res(u, z, T, v_L):
        lny, lnP = u[:-1], u[-1]
        y = jnp.exp(lny)
        P = jnp.exp(lnP)
        Zv = thermo.Z(thermo._mix(T, P, y), "vapor")
        nV = P * vl_ratio * v_L / (Zv * R * T)
        x = jnp.maximum((z - nV * y) / (1.0 - nV), 1e-300)
        r = lny - thermo.log_K(T, P, x, y) - jnp.log(x)
        return jnp.concatenate([r, (jnp.sum(y) - 1.0)[None]])

    # Start: a bubble point on Wilson K, then Newton.
    if P0 is None:
        lnKw = thermo.log_K_wilson(Ts, 1.0)              # K = Kw(1 Pa)/P
        P0 = jnp.sum(zs * jnp.exp(lnKw))
    P0 = jax.lax.stop_gradient(jnp.asarray(P0, dtype=float))
    y0 = zs * jnp.exp(thermo.log_K_wilson(Ts, P0))
    u0 = jnp.concatenate([jnp.log(jnp.maximum(y0 / jnp.sum(y0), 1e-300)), jnp.log(P0)[None]])
    vLs = jax.lax.stop_gradient(v_L)
    J_of = jax.jacfwd(res)

    def body(_, u):
        r = res(u, zs, Ts, vLs)
        J = J_of(u, zs, Ts, vLs)
        du = jnp.linalg.solve(J, -r)
        du = jnp.where(jnp.isfinite(du), du, 0.0)
        s = jnp.minimum(1.0, 1.0 / jnp.maximum(jnp.max(jnp.abs(du)), 1e-300))
        return u + s * du

    u = jax.lax.stop_gradient(jax.lax.fori_loop(0, n_iter, body, u0))
    J = jax.lax.stop_gradient(J_of(u, zs, Ts, vLs))
    u = u - jnp.linalg.solve(J, res(u, z, T, v_L))
    return jnp.exp(u[-1])


def bubble_temperature(thermo, x, P, T0=None, n_iter: int = 40):
    """Bubble-point temperature of liquid ``x`` at ``P`` (implicit gradients)."""
    x = x / jnp.sum(x)
    P = jnp.asarray(P, dtype=float)
    xs, Ps = jax.lax.stop_gradient(x), jax.lax.stop_gradient(P)

    def res(u, x, P):
        lny, T = u[:-1], u[-1]
        y = jnp.exp(lny)
        r = lny - thermo.log_K(T, P, x, y) - jnp.log(x)
        return jnp.concatenate([r, (jnp.sum(y) - 1.0)[None]])

    if T0 is None:
        T0 = wilson_bubble_T(thermo, xs, Ps)
    T0 = jax.lax.stop_gradient(jnp.asarray(T0, dtype=float))
    y0 = xs * jnp.exp(thermo.log_K_wilson(T0, Ps))
    u0 = jnp.concatenate([jnp.log(jnp.maximum(y0 / jnp.sum(y0), 1e-300)), T0[None]])
    J_of = jax.jacfwd(res)
    caps = jnp.concatenate([jnp.full(x.shape[-1], 1.0), jnp.array([10.0])])

    def body(_, u):
        du = jnp.linalg.solve(J_of(u, xs, Ps), -res(u, xs, Ps))
        du = jnp.where(jnp.isfinite(du), du, 0.0)
        s = jnp.minimum(1.0, jnp.min(caps / jnp.maximum(jnp.abs(du), 1e-300)))
        return u + s * du

    u = jax.lax.stop_gradient(jax.lax.fori_loop(0, n_iter, body, u0))
    u = u - jnp.linalg.solve(jax.lax.stop_gradient(J_of(u, xs, Ps)), res(u, x, P))
    return u[-1]


def wilson_bubble_T(thermo, x, P, n_iter: int = 30):
    """Bubble temperature on Wilson K (Newton on ``ln sum K x``), any batch."""
    x = x / jnp.sum(x, -1, keepdims=True)
    T = jnp.full(jnp.shape(x)[:-1], 300.0)

    def f(T):
        return jnp.log(jnp.sum(x * jnp.exp(thermo.log_K_wilson(T, P)), -1))

    def body(_, T):
        v, d = jax.jvp(f, (T,), (jnp.ones_like(T),))
        return jnp.clip(T - jnp.clip(v / d, -50.0, 50.0), 80.0, 900.0)

    return jax.lax.fori_loop(0, n_iter, body, T)


def wilson_dew_T(thermo, y, P, n_iter: int = 30):
    """Dew temperature on Wilson K, any batch."""
    y = y / jnp.sum(y, -1, keepdims=True)
    T = jnp.full(jnp.shape(y)[:-1], 300.0)

    def f(T):
        return jnp.log(jnp.sum(y * jnp.exp(-thermo.log_K_wilson(T, P)), -1))

    def body(_, T):
        v, d = jax.jvp(f, (T,), (jnp.ones_like(T),))
        return jnp.clip(T - jnp.clip(v / d, -50.0, 50.0), 80.0, 900.0)

    return jax.lax.fori_loop(0, n_iter, body, T)
