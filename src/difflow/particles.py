"""Flow past particles and through particle beds (pure JAX functions, SI).

Building blocks for adsorbers, catalytic beds, settlers, cyclones and filters:

- single particles: :func:`drag_coefficient`, :func:`terminal_velocity`,
  :func:`hindered_settling_velocity`;
- packed beds: :func:`ergun_pressure_gradient`, :func:`kozeny_carman`,
  :func:`ergun_alpha` (the lumped coefficient ``GasPFR`` uses);
- fluidization: :func:`minimum_fluidization_velocity`, :func:`bed_expansion`,
  :func:`fluidization_window`, :func:`geldart_group` (not differentiable).

These are plain functions, not unit operations: they are not registered in
the GUI palette. Everything is differentiable with ``jax.grad`` /
``jax.jacobian`` (the implicit terminal-velocity solve differentiates through
``optimistix`` implicit differentiation, not by unrolling Newton steps).

Sources
-------
- Haider, A. and Levenspiel, O., Powder Technol., 58, 63-70 (1989): drag of
  spheres and non-spheres, Re < 2.6e5, sphericity 0.5-1.
- Turton, R. and Levenspiel, O., Powder Technol., 47, 83-86 (1986): sphere drag.
- Schiller, L. and Naumann, A., Z. Ver. Dtsch. Ing., 77, 318 (1933).
- Ergun, S., Chem. Eng. Prog., 48(2), 89-94 (1952).
- Wen, C. Y. and Yu, Y. H., AIChE J., 12, 610-612 (1966).
- Richardson, J. F. and Zaki, W. N., Trans. Inst. Chem. Eng., 32, 35-53 (1954).
- Kunii, D. and Levenspiel, O., Fluidization Engineering, 2nd ed. (1991).
- Geldart, D., Powder Technol., 7, 285-292 (1973).
"""

from __future__ import annotations

import jax.numpy as jnp
import optimistix as optx
from jax import Array

#: Standard gravity (m/s^2).
G_STD = 9.80665

DRAG_METHODS = ("schiller_naumann", "haider_levenspiel", "turton_levenspiel")


def _sphericity_guard(phi):
    return jnp.asarray(phi, dtype=jnp.result_type(float))


def drag_coefficient(
    Re_p: Array | float,
    sphericity: Array | float = 1.0,
    method: str = "haider_levenspiel",
) -> Array:
    r"""Drag coefficient of a single particle versus particle Reynolds number.

    :math:`C_D = F_D / (\tfrac12 \rho_f u^2 A_p)` with :math:`A_p` the projected
    area of the volume-equivalent sphere, :math:`Re_p = \rho_f u d_p/\mu`.

    Args:
        Re_p: Particle Reynolds number (-), must be positive.
        sphericity: Sphericity (0.5 to 1; 1 = sphere). Used by
            ``"haider_levenspiel"`` only; ignored by the sphere correlations.
        method: One of

            - ``"haider_levenspiel"`` (default): Haider & Levenspiel (1989),
              ``24/Re (1 + A Re^B) + C / (1 + D/Re)`` with sphericity-dependent
              A-D; Re < 2.6e5. Smooth over the whole range.
            - ``"turton_levenspiel"``: Turton & Levenspiel (1986), spheres,
              ``24/Re (1 + 0.173 Re^0.657) + 0.413 / (1 + 16300 Re^-1.09)``;
              Re < 2.6e5.
            - ``"schiller_naumann"``: ``24/Re (1 + 0.15 Re^0.687)`` for
              Re < 800, floored at the Newton value 0.44 (the floor is a
              kink in the derivative at Re ~ 800).

    Returns:
        Drag coefficient (-). Tends to ``24/Re`` as ``Re -> 0``.

    Note:
        ``method`` is a Python string (static under ``jit``).

    Example:
        >>> from difflow.particles import drag_coefficient
        >>> round(float(drag_coefficient(1e-4)) * 1e-4 / 24, 3)   # Stokes limit
        1.0
    """
    Re = jnp.asarray(Re_p, dtype=float)
    if method == "haider_levenspiel":
        phi = _sphericity_guard(sphericity)
        A = jnp.exp(2.3288 - 6.4581 * phi + 2.4486 * phi**2)
        B = 0.0964 + 0.5565 * phi
        C = jnp.exp(4.905 - 13.8944 * phi + 18.4222 * phi**2 - 10.2599 * phi**3)
        D = jnp.exp(1.4681 + 12.2584 * phi - 20.7322 * phi**2 + 15.8855 * phi**3)
        return 24.0 / Re * (1.0 + A * Re**B) + C / (1.0 + D / Re)
    if method == "turton_levenspiel":
        return 24.0 / Re * (1.0 + 0.173 * Re**0.657) + 0.413 / (1.0 + 16300.0 * Re**-1.09)
    if method == "schiller_naumann":
        return jnp.maximum(24.0 / Re * (1.0 + 0.15 * Re**0.687), 0.44)
    raise ValueError(f"unknown drag method {method!r}; choose from {DRAG_METHODS}")


def archimedes_number(
    d_p: Array | float,
    rho_p: Array | float,
    rho_f: Array | float,
    mu: Array | float,
    g: float = G_STD,
) -> Array:
    r"""Archimedes number :math:`Ar = d_p^3 \rho_f (\rho_p-\rho_f) g/\mu^2`.

    Args:
        d_p: Particle diameter (m).
        rho_p: Particle density (kg/m^3).
        rho_f: Fluid density (kg/m^3).
        mu: Fluid viscosity (Pa s).
        g: Gravitational acceleration (m/s^2).

    Returns:
        Ar (-).
    """
    return d_p**3 * rho_f * (rho_p - rho_f) * g / mu**2


def _terminal_residual(x, args):
    # x = ln Re. Force balance: C_D(Re) Re^2 = (4/3) Ar.
    log_target, phi, method = args
    Re = jnp.exp(x)
    return jnp.log(drag_coefficient(Re, phi, method)) + 2.0 * x - log_target


def terminal_reynolds(
    d_p: Array | float,
    rho_p: Array | float,
    rho_f: Array | float,
    mu: Array | float,
    sphericity: Array | float = 1.0,
    method: str = "haider_levenspiel",
    g: float = G_STD,
) -> Array:
    r"""Particle Reynolds number at terminal velocity (see :func:`terminal_velocity`).

    Solves :math:`C_D(Re)\,Re^2 = \tfrac43 Ar` for :math:`\ln Re` with Newton's
    method, started from the explicit Haider & Levenspiel dimensionless
    velocity :math:`u^* = [18/d^{*2} + (2.335 - 1.744\phi)/d^{*1/2}]^{-1}`
    (:math:`d^* = Ar^{1/3}`, :math:`u^* = Re / Ar^{1/3}`). Gradients use
    implicit differentiation of the converged root.

    Args:
        d_p, rho_p, rho_f, mu, sphericity, method, g: As :func:`terminal_velocity`.

    Returns:
        Re at terminal velocity (-).
    """
    phi = _sphericity_guard(sphericity)
    Ar = archimedes_number(d_p, rho_p, rho_f, mu, g)
    dstar = Ar ** (1.0 / 3.0)
    ustar = 1.0 / (18.0 / dstar**2 + (2.335 - 1.744 * phi) / jnp.sqrt(dstar))
    x0 = jnp.log(ustar * dstar)
    log_target = jnp.log(4.0 / 3.0 * Ar)
    sol = optx.root_find(
        _terminal_residual,
        optx.Newton(rtol=1e-12, atol=1e-12),
        x0,
        args=(log_target, phi, method),
        max_steps=50,
        throw=False,
    )
    return jnp.exp(sol.value)


def terminal_velocity(
    d_p: Array | float,
    rho_p: Array | float,
    rho_f: Array | float,
    mu: Array | float,
    sphericity: Array | float = 1.0,
    method: str = "haider_levenspiel",
    g: float = G_STD,
) -> Array:
    r"""Terminal settling velocity of a single particle in an unbounded fluid.

    Solves the force balance (weight - buoyancy = drag)

    .. math:: \frac{\pi}{6} d_p^3 (\rho_p-\rho_f) g
              = C_D(Re_p)\,\tfrac12 \rho_f u_t^2\,\frac{\pi}{4} d_p^2

    with the chosen drag correlation, across the Stokes, intermediate and
    Newton regimes, by a root find in :math:`\ln Re_p` (implicit
    differentiation; no unrolled iterations).

    Args:
        d_p: Particle (volume-equivalent) diameter (m).
        rho_p: Particle density (kg/m^3).
        rho_f: Fluid density (kg/m^3). Particle must be denser than the fluid.
        mu: Fluid viscosity (Pa s).
        sphericity: Sphericity (-); effective with ``"haider_levenspiel"``.
        method: Drag correlation, see :func:`drag_coefficient`.
        g: Gravitational acceleration (m/s^2).

    Returns:
        Terminal velocity (m/s), positive downward.

    Example:
        >>> from difflow.particles import terminal_velocity
        >>> # 50 micron quartz-like sphere in water: Stokes regime
        >>> v = terminal_velocity(50e-6, 2650.0, 998.0, 1.0e-3)
        >>> round(float(v), 4)
        0.0022
    """
    Re = terminal_reynolds(d_p, rho_p, rho_f, mu, sphericity, method, g)
    return Re * mu / (rho_f * d_p)


def hindered_settling_velocity(
    v_t: Array | float,
    voidage: Array | float,
    n: Array | float | None = None,
    Re_t: Array | float | None = None,
) -> Array:
    r"""Richardson-Zaki hindered settling / fluidization velocity :math:`u = v_t \varepsilon^n`.

    The exponent follows Richardson & Zaki (1954) for an unbounded column:
    ``n = 4.65`` (Re_t < 0.2), ``4.35 Re_t^-0.03`` (0.2-1), ``4.45 Re_t^-0.1``
    (1-500), ``2.39`` (Re_t > 500). The piecewise choice has small jumps at
    the boundaries (at most about 2%); pass ``n`` to avoid them. Wall effects
    are not included.

    Args:
        v_t: Single-particle terminal velocity (m/s).
        voidage: Void fraction of the suspension, :math:`\varepsilon` (-).
        n: Richardson-Zaki exponent. If None, computed from ``Re_t``.
        Re_t: Terminal Reynolds number, required when ``n`` is None.

    Returns:
        Superficial velocity at which the suspension has the given voidage (m/s).

    Raises:
        ValueError: if neither ``n`` nor ``Re_t`` is given.
    """
    if n is None:
        n = richardson_zaki_exponent(_require(Re_t))
    return v_t * jnp.asarray(voidage) ** n


def _require(Re_t):
    if Re_t is None:
        raise ValueError("give either the exponent n or the terminal Reynolds number Re_t")
    return Re_t


def richardson_zaki_exponent(Re_t: Array | float) -> Array:
    """Richardson & Zaki (1954) exponent ``n`` versus terminal Reynolds number.

    Args:
        Re_t: Terminal Reynolds number (-).

    Returns:
        Exponent n (-), piecewise as described in
        :func:`hindered_settling_velocity`.
    """
    Re = jnp.asarray(Re_t, dtype=float)
    return jnp.where(
        Re < 0.2,
        4.65,
        jnp.where(Re < 1.0, 4.35 * Re**-0.03, jnp.where(Re < 500.0, 4.45 * Re**-0.1, 2.39)),
    )


def ergun_pressure_gradient(
    u_s: Array | float,
    d_p: Array | float,
    voidage: Array | float,
    rho: Array | float,
    mu: Array | float,
    sphericity: Array | float = 1.0,
) -> tuple[Array, dict[str, Array]]:
    r"""Packed-bed pressure gradient by the full Ergun (1952) equation.

    .. math::
        \frac{\Delta P}{L} = \underbrace{150\,\frac{(1-\varepsilon)^2}{\varepsilon^3}
        \frac{\mu u_s}{(\phi d_p)^2}}_{\text{Blake-Kozeny (viscous)}}
        + \underbrace{1.75\,\frac{1-\varepsilon}{\varepsilon^3}
        \frac{\rho u_s^2}{\phi d_p}}_{\text{Burke-Plummer (inertial)}}

    Args:
        u_s: Superficial velocity (m/s).
        d_p: Particle diameter (m) (volume-equivalent sphere).
        voidage: Bed void fraction (-).
        rho: Fluid density (kg/m^3).
        mu: Fluid viscosity (Pa s).
        sphericity: Particle sphericity (-).

    Returns:
        ``(dP_dL, info)``: pressure drop per unit length (Pa/m, positive in
        the flow direction) and ``info`` with ``viscous`` (Blake-Kozeny term,
        Pa/m), ``inertial`` (Burke-Plummer term, Pa/m) and ``Re_bed``
        (``rho u_s phi d_p / (mu (1-eps))``).

    Example:
        >>> from difflow.particles import ergun_pressure_gradient
        >>> dp, info = ergun_pressure_gradient(1.0, 3e-3, 0.4, 1.2, 1.8e-5)
        >>> round(float(dp))
        8250
    """
    eps = jnp.asarray(voidage)
    dp_eff = sphericity * d_p
    viscous = 150.0 * (1.0 - eps) ** 2 / eps**3 * mu * u_s / dp_eff**2
    inertial = 1.75 * (1.0 - eps) / eps**3 * rho * u_s * u_s / dp_eff
    info = {
        "viscous": viscous,
        "inertial": inertial,
        "Re_bed": rho * u_s * dp_eff / (mu * (1.0 - eps)),
    }
    return viscous + inertial, info


def kozeny_carman(
    u_s: Array | float,
    d_p: Array | float,
    voidage: Array | float,
    mu: Array | float,
    sphericity: Array | float = 1.0,
    k0: float = 5.0,
) -> Array:
    r"""Kozeny-Carman laminar pressure gradient (Pa/m).

    .. math:: \frac{\Delta P}{L} = 36 k_0 \frac{(1-\varepsilon)^2}{\varepsilon^3}
              \frac{\mu u_s}{(\phi d_p)^2}

    With the usual Kozeny constant ``k0 = 5`` the prefactor is 180 (Carman),
    versus Ergun's 150 (the two differ by the constant fitted to the data).
    Reusable for filter-cake resistance: the specific cake resistance is
    ``36 k0 (1-eps)/(eps^3 rho_p (phi d_p)^2)``.

    Args:
        u_s: Superficial velocity (m/s).
        d_p: Particle diameter (m).
        voidage: Void fraction (-).
        mu: Viscosity (Pa s).
        sphericity: Sphericity (-).
        k0: Kozeny constant (-).

    Returns:
        Pressure gradient (Pa/m).
    """
    eps = jnp.asarray(voidage)
    dp_eff = sphericity * d_p
    return 36.0 * k0 * (1.0 - eps) ** 2 / eps**3 * mu * u_s / dp_eff**2


def ergun_alpha(
    u_s0: Array | float,
    d_p: Array | float,
    voidage: Array | float,
    rho0: Array | float,
    mu: Array | float,
    area: Array | float,
    sphericity: Array | float = 1.0,
) -> Array:
    r"""Lumped pressure-drop coefficient :math:`\alpha` (Pa/m^3) from the Ergun equation.

    ``GasPFR`` integrates :math:`dP/dV = -\alpha (P_0/P)(T/T_0)(F/F_0)` with
    :math:`V` the bed volume, so :math:`\alpha` is the inlet Ergun gradient
    divided by the bed cross-section: :math:`\alpha = (\Delta P/L)_0 / A_c`.

    The ``(P0/P)(T/T0)(F/F0)`` factor in ``GasPFR`` scales the gradient with
    the local volumetric flow ratio; it is the Ergun inertial term
    (:math:`G^2/\rho` at constant mass flux) exactly, and treats the viscous
    term the same way (an approximation; viscosity is also held at its
    inlet value). Fogler (Ch. 5) makes the same lumping.

    Args:
        u_s0: Inlet superficial velocity (m/s).
        d_p: Particle diameter (m).
        voidage: Void fraction (-).
        rho0: Inlet gas density (kg/m^3).
        mu: Gas viscosity (Pa s).
        area: Bed cross-sectional area (m^2).
        sphericity: Particle sphericity (-).

    Returns:
        alpha (Pa/m^3).
    """
    grad, _ = ergun_pressure_gradient(u_s0, d_p, voidage, rho0, mu, sphericity)
    return grad / area


def minimum_fluidization_velocity(
    d_p: Array | float,
    rho_p: Array | float,
    rho_f: Array | float,
    mu: Array | float,
    voidage_mf: Array | float = 0.45,
    sphericity: Array | float = 1.0,
    method: str = "ergun",
    g: float = G_STD,
) -> Array:
    r"""Minimum fluidization velocity :math:`u_{mf}`.

    ``method="ergun"`` equates the Ergun pressure gradient to the bed weight
    per unit volume, :math:`(1-\varepsilon_{mf})(\rho_p-\rho_f)g`. In
    dimensionless form (Kunii & Levenspiel)

    .. math:: Ar = \frac{150(1-\varepsilon_{mf})}{\phi^2\varepsilon_{mf}^3} Re_{mf}
              + \frac{1.75}{\phi\varepsilon_{mf}^3} Re_{mf}^2,

    a quadratic in :math:`Re_{mf} = \rho_f u_{mf} d_p/\mu` solved in closed form.

    ``method="wen_yu"`` uses Wen & Yu (1966),
    :math:`Re_{mf} = \sqrt{33.7^2 + 0.0408\,Ar} - 33.7`, which needs neither
    voidage nor sphericity (both arguments are then ignored).

    Args:
        d_p: Particle diameter (m).
        rho_p: Particle density (kg/m^3).
        rho_f: Fluid density (kg/m^3).
        mu: Fluid viscosity (Pa s).
        voidage_mf: Bed voidage at minimum fluidization (-).
        sphericity: Sphericity (-).
        method: ``"ergun"`` or ``"wen_yu"``.
        g: Gravitational acceleration (m/s^2).

    Returns:
        u_mf (m/s).

    Example:
        >>> from difflow.particles import minimum_fluidization_velocity
        >>> u = minimum_fluidization_velocity(300e-6, 2600.0, 1.2, 1.8e-5, 0.45, 0.8)
        >>> 0.02 < float(u) < 0.2
        True
    """
    Ar = archimedes_number(d_p, rho_p, rho_f, mu, g)
    if method == "ergun":
        eps = jnp.asarray(voidage_mf)
        a = 1.75 / (sphericity * eps**3)
        b = 150.0 * (1.0 - eps) / (sphericity**2 * eps**3)
        Re_mf = (-b + jnp.sqrt(b * b + 4.0 * a * Ar)) / (2.0 * a)
    elif method == "wen_yu":
        Re_mf = jnp.sqrt(33.7**2 + 0.0408 * Ar) - 33.7
    else:
        raise ValueError(f"unknown method {method!r}; choose 'ergun' or 'wen_yu'")
    return Re_mf * mu / (rho_f * d_p)


def bed_expansion(
    u: Array | float,
    u_t: Array | float,
    n: Array | float | None = None,
    Re_t: Array | float | None = None,
    voidage_mf: Array | float = 0.45,
    H_mf: Array | float = 1.0,
) -> tuple[Array, Array]:
    r"""Fluidized-bed voidage and height from Richardson-Zaki.

    :math:`\varepsilon = (u/u_t)^{1/n}` (valid for :math:`u_{mf} \le u < u_t`),
    and, with a fixed solids inventory,
    :math:`H = H_{mf}(1-\varepsilon_{mf})/(1-\varepsilon)`. This is the
    homogeneous (particulate) expansion; gas beds with bubbles
    expand differently. The Richardson-Zaki line is anchored at ``u_t``, so
    the voidage it gives at ``u_mf`` need not equal ``voidage_mf``; the height
    ratio is then only meaningful well above ``u_mf``.

    Args:
        u: Superficial velocity (m/s).
        u_t: Particle terminal velocity (m/s).
        n: Richardson-Zaki exponent; if None, from ``Re_t``.
        Re_t: Terminal Reynolds number (needed if ``n`` is None).
        voidage_mf: Voidage at minimum fluidization (-).
        H_mf: Bed height at minimum fluidization (m).

    Returns:
        ``(voidage, height)``.
    """
    if n is None:
        n = richardson_zaki_exponent(_require(Re_t))
    eps = (jnp.asarray(u, dtype=float) / u_t) ** (1.0 / n)
    return eps, H_mf * (1.0 - voidage_mf) / (1.0 - eps)


def fluidization_window(
    d_p: Array | float,
    rho_p: Array | float,
    rho_f: Array | float,
    mu: Array | float,
    voidage_mf: Array | float = 0.45,
    sphericity: Array | float = 1.0,
    method: str = "ergun",
    drag: str = "haider_levenspiel",
    g: float = G_STD,
) -> tuple[Array, Array]:
    """Operating window ``(u_mf, u_t)`` for a fluidized bed.

    Below ``u_mf`` the bed is fixed; above ``u_t`` single particles are
    entrained.

    Args:
        d_p, rho_p, rho_f, mu, voidage_mf, sphericity, g: See
            :func:`minimum_fluidization_velocity` and :func:`terminal_velocity`.
        method: u_mf method (``"ergun"`` or ``"wen_yu"``).
        drag: Drag correlation for u_t.

    Returns:
        ``(u_mf, u_t)`` in m/s.
    """
    u_mf = minimum_fluidization_velocity(d_p, rho_p, rho_f, mu, voidage_mf, sphericity, method, g)
    u_t = terminal_velocity(d_p, rho_p, rho_f, mu, sphericity, drag, g)
    return u_mf, u_t


def geldart_group(d_p: float, rho_p: float, rho_f: float) -> str:
    """Approximate Geldart (1973) powder group: ``"A"``, ``"B"``, ``"C"`` or ``"D"``.

    NOT differentiable and not jit-able (returns a Python string from
    concrete numbers). The boundaries are the commonly quoted approximations
    of Geldart's chart, not an exact digitization: C below 20 micron; A
    below 100 micron with density difference under 1400 kg/m^3; D above
    the line ``d_p [micron] * (rho_p - rho_f)^0.934 = 1e6``
    (``rho`` in kg/m^3); B otherwise. Treat results near a boundary as
    indicative only.

    Args:
        d_p: Particle diameter (m).
        rho_p: Particle density (kg/m^3).
        rho_f: Fluid density (kg/m^3).

    Returns:
        One of ``"A"``, ``"B"``, ``"C"``, ``"D"``.
    """
    d_um = float(d_p) * 1e6
    drho = float(rho_p) - float(rho_f)
    if d_um < 20.0:
        return "C"
    if d_um * drho**0.934 > 1.0e6:
        return "D"
    if d_um < 100.0 and drho < 1400.0:
        return "A"
    return "B"
