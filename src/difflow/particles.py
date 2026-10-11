"""Flow past particles and through particle beds (pure JAX functions, SI).

Building blocks for adsorbers, catalytic beds, settlers, cyclones and filters:

- single particles: :func:`drag_coefficient`, :func:`terminal_velocity`,
  :func:`hindered_settling_velocity`;
- packed beds: :func:`ergun_pressure_gradient`, :func:`kozeny_carman`,
  :func:`ergun_alpha` (the lumped coefficient ``GasPFR`` uses);
- fluidization: :func:`minimum_fluidization_velocity`, :func:`bed_expansion`,
  :func:`fluidization_window`, :func:`minimum_bubbling_velocity`,
  :func:`geldart_group` (not differentiable).

These are plain functions, not unit operations: they are not registered in
the GUI palette. The units that apply them are in
:mod:`difflow.units.mechanical`: ``Settler`` (terminal and hindered settling
velocity for overflow-rate and thickener design), ``CakeFilter`` (the
Kozeny-Carman specific cake resistance, via
``specific_cake_resistance``) and ``Cyclone``. Everything is differentiable with ``jax.grad`` /
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
- Baeyens, J. and Geldart, D., Chem. Eng. Sci., 29, 255-265 (1974): u_mf.
- Abrahamsen, A. R. and Geldart, D., Powder Technol., 26, 35-46 (1980):
  minimum bubbling velocity and the A/B boundary.
- Mehta, D. and Hawley, M. C., Ind. Eng. Chem. Process Des. Dev., 8, 280-282
  (1969): wall correction to the Ergun equation.
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
    D_bed: Array | float | None = None,
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
        D_bed: Bed (column) diameter (m). If given, the Mehta-Hawley (1969)
            wall correction is applied: the wall adds wetted surface, so with
            ``M = 1 + 2 phi d_p / (3 D_bed (1 - eps))`` the viscous term is
            multiplied by ``M^2`` and the inertial term by ``M``. It matters
            for ``D_bed / d_p`` below about 10 (laboratory and pilot columns);
            it tends to 1 for a wide bed. None (default) is the
            unbounded-bed equation.

    Returns:
        ``(dP_dL, info)``: pressure drop per unit length (Pa/m, positive in
        the flow direction) and ``info`` with ``viscous`` (Blake-Kozeny term,
        Pa/m), ``inertial`` (Burke-Plummer term, Pa/m), ``Re_bed``
        (``rho u_s phi d_p / (mu (1-eps))``) and ``wall_factor`` (``M``, 1
        without ``D_bed``).

    Example:
        >>> from difflow.particles import ergun_pressure_gradient
        >>> dp, info = ergun_pressure_gradient(1.0, 3e-3, 0.4, 1.2, 1.8e-5)
        >>> round(float(dp))
        8250
    """
    eps = jnp.asarray(voidage)
    dp_eff = sphericity * d_p
    M = jnp.asarray(1.0) if D_bed is None else 1.0 + 2.0 * dp_eff / (3.0 * D_bed * (1.0 - eps))
    viscous = 150.0 * M**2 * (1.0 - eps) ** 2 / eps**3 * mu * u_s / dp_eff**2
    inertial = 1.75 * M * (1.0 - eps) / eps**3 * rho * u_s * u_s / dp_eff
    info = {
        "viscous": viscous,
        "inertial": inertial,
        "Re_bed": rho * u_s * dp_eff / (mu * (1.0 - eps)),
        "wall_factor": M,
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
    u_mf: Array | float | None = None,
) -> tuple[Array, Array]:
    r"""Fluidized-bed voidage and height from Richardson-Zaki.

    :math:`\varepsilon = (u/u_t)^{1/n}` (valid for :math:`u_{mf} \le u < u_t`),
    and, with a fixed solids inventory,
    :math:`H = H_{mf}(1-\varepsilon_{mf})/(1-\varepsilon)`. This is the
    homogeneous (particulate) expansion; gas beds with bubbles
    expand differently.

    By default the Richardson-Zaki line is anchored at ``u_t``, so the
    voidage it gives at ``u_mf`` need not equal ``voidage_mf``: just above
    ``u_mf`` it can predict a bed *shorter* than the settled one, and the height
    is only meaningful well above ``u_mf``. Pass ``u_mf`` to anchor the line at
    ``(u_mf, voidage_mf)`` instead, :math:`\varepsilon = \varepsilon_{mf}
    (u/u_{mf})^{1/n}` (capped at 1). This keeps the Richardson-Zaki slope
    but makes the height exactly ``H_mf`` at ``u_mf``, which is the usual
    engineering fix when no measured intercept is available.

    Args:
        u: Superficial velocity (m/s).
        u_t: Particle terminal velocity (m/s).
        n: Richardson-Zaki exponent; if None, from ``Re_t``.
        Re_t: Terminal Reynolds number (needed if ``n`` is None).
        voidage_mf: Voidage at minimum fluidization (-).
        H_mf: Bed height at minimum fluidization (m).
        u_mf: Minimum fluidization velocity (m/s). If given, the line is
            anchored at ``(u_mf, voidage_mf)`` instead of ``(u_t, 1)``.

    Returns:
        ``(voidage, height)``.
    """
    if n is None:
        n = richardson_zaki_exponent(_require(Re_t))
    u = jnp.asarray(u, dtype=float)
    if u_mf is None:
        eps = (u / u_t) ** (1.0 / n)
    else:
        eps = jnp.minimum(voidage_mf * (u / u_mf) ** (1.0 / n), 1.0)
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


def minimum_bubbling_velocity(
    d_p: Array | float,
    rho_f: Array | float,
    mu: Array | float,
    fines_fraction: Array | float = 0.0,
) -> Array:
    r"""Minimum bubbling velocity of a gas-fluidized powder (Abrahamsen-Geldart, 1980).

    .. math::
        u_{mb} = 2.07\,e^{0.716 F}\,\frac{d_p\,\rho_g^{0.06}}{\mu^{0.347}}

    in SI units, with ``F`` the mass fraction of powder finer than 45 micron.
    Between ``u_mf`` and ``u_mb`` a Group A powder expands homogeneously; for
    Group B powders ``u_mb <= u_mf`` and bubbles appear as soon as the bed
    fluidizes. A correlation of experiments on powders of about 20-72 micron
    (mean); use it as an estimate.

    Args:
        d_p: Mean (surface-volume) particle diameter (m).
        rho_f: Gas density (kg/m^3).
        mu: Gas viscosity (Pa s).
        fines_fraction: Mass fraction finer than 45 micron (-).

    Returns:
        ``u_mb`` (m/s).
    """
    return 2.07 * jnp.exp(0.716 * jnp.asarray(fines_fraction)) * d_p * jnp.asarray(rho_f) ** 0.06 / jnp.asarray(mu) ** 0.347


def _baeyens_geldart_umf(d_p, drho, rho_f, mu, g=G_STD):
    """Baeyens-Geldart (1974) u_mf, the correlation the A/B criterion was built with."""
    return (drho * g) ** 0.934 * d_p**1.8 / (1111.0 * mu**0.87 * rho_f**0.066)


def geldart_group(
    d_p: float,
    rho_p: float,
    rho_f: float,
    mu: float = 1.8e-5,
    fines_fraction: float = 0.0,
) -> str:
    """Geldart (1973) powder group: ``"A"``, ``"B"``, ``"C"`` or ``"D"``.

    NOT differentiable and not jit-able (returns a Python string from
    concrete numbers). The boundaries:

    * **C/A**: 20 micron, the commonly quoted approximation of Geldart's chart.
    * **A/B**: ``u_mb / u_mf = 1`` (Geldart's own definition of the boundary:
      Group A beds expand without bubbling between ``u_mf`` and ``u_mb``), with
      :func:`minimum_bubbling_velocity` and the Baeyens-Geldart ``u_mf``. In SI
      units the ratio is
      ``2300 rho_f^0.126 mu^0.523 exp(0.716 F) / (d_p^0.8 (g (rho_p - rho_f))^0.934)``.
      It is a sloping line on the log-log chart: FCC catalyst (60 micron,
      1500 kg/m^3) is A and 200 micron sand is B.
    * **B/D**: Archimedes number ``Ar = d_p^3 rho_f (rho_p - rho_f) g / mu^2
      = 1.45e5`` (Grace, 1986, Can. J. Chem. Eng. 64, 353), about 1.1 mm
      for sand in ambient air.

    Treat results near a boundary as indicative only; the A/B line depends on
    the gas (through ``rho_f`` and ``mu``) and the fines content. The
    Abrahamsen-Geldart ``u_mb`` was fitted to fine powders (tens of microns),
    so the A/B line is an extrapolation away from them: it sits near 180
    micron at 1500 kg/m^3, larger than Geldart's chart suggests, and at low
    density (``rho_p - rho_f`` of a few hundred kg/m^3) it reaches millimetre
    sizes, where group A meets group D with no B between. Treat an "A" for a
    coarse or very light powder as an extrapolation.

    Args:
        d_p: Mean particle diameter (m).
        rho_p: Particle density (kg/m^3).
        rho_f: Fluid density (kg/m^3).
        mu: Gas viscosity (Pa s); default air at ambient conditions.
        fines_fraction: Mass fraction finer than 45 micron (-).

    Returns:
        One of ``"A"``, ``"B"``, ``"C"``, ``"D"``.
    """
    d_p, rho_f, mu = float(d_p), float(rho_f), float(mu)
    d_um = d_p * 1e6
    drho = float(rho_p) - rho_f
    if d_um < 20.0:
        return "C"
    if d_p**3 * rho_f * drho * G_STD / mu**2 > 1.45e5:
        return "D"
    ratio = float(minimum_bubbling_velocity(d_p, rho_f, mu, fines_fraction)) / _baeyens_geldart_umf(
        d_p, drho, rho_f, mu
    )
    return "A" if ratio > 1.0 else "B"


__all__ = [
    "G_STD",
    "drag_coefficient",
    "archimedes_number",
    "terminal_reynolds",
    "terminal_velocity",
    "hindered_settling_velocity",
    "richardson_zaki_exponent",
    "ergun_pressure_gradient",
    "kozeny_carman",
    "ergun_alpha",
    "minimum_fluidization_velocity",
    "bed_expansion",
    "fluidization_window",
    "minimum_bubbling_velocity",
    "geldart_group",
]
