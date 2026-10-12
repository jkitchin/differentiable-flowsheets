"""Liquid-Liquid Extraction unit operations.

This module provides differentiable LLE operations:
- MultistageCascade: Counter-current or co-current extraction cascade
- DifferentialContactor: Packed column or similar continuous contactor
- ternary_lle_flash: differentiable two-liquid-phase flash with mutual solubility,
  and ``CascadeParams(model="rigorous")``, the partially miscible cascade built on it

Supports both distribution coefficient (K-value) models and activity
coefficient models (NRTL, UNIQUAC) for computing equilibrium.

All operations are fully differentiable using JAX.
"""

from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Literal
import jax
import jax.numpy as jnp
from jax import Array, lax

from difflow.streams import Stream, get_flows, make_stream
from difflow.params_mixin import ParamsMixin
from difflow.numerics import safe_divide
from difflow.staged import kremser_fraction, kremser_stages  # noqa: F401  (re-exported)
from difflow.staged import _taylor_log_fraction


# =============================================================================
# Utility Functions
# =============================================================================

def soft_clip_positive(x: Array, sharpness: float = 10.0) -> Array:
    """Soft clipping to ensure non-negative values with smooth gradients.

    Uses softplus function shifted to pass through origin:
        soft_clip(x) = softplus(x * sharpness) / sharpness

    This is approximately:
        - x when x >> 0
        - 0 when x << 0
        - smooth transition near x = 0

    Args:
        x: Input array
        sharpness: Controls transition sharpness (higher = closer to hard clip)

    Returns:
        Soft-clipped array with values >= 0
    """
    return jax.nn.softplus(x * sharpness) / sharpness


# =============================================================================
# Distribution Coefficient Models
# =============================================================================

class DistributionCoeffs(NamedTuple):
    """Simple distribution coefficient model.

    K = concentration in extract (organic) / concentration in raffinate (aqueous)

    Convention: K > 1 means solute favors the organic/extract phase.

    For temperature-dependent K:
        K(T) = K0 * exp(-dH / R * (1/T - 1/Tref))

    Attributes:
        species: List of species that transfer between phases
        K0: Distribution coefficients at reference temperature (C_org / C_aq)
        dH: Heat of extraction (J/mol), for temperature dependence
        Tref: Reference temperature (K)
    """
    species: tuple[str, ...]
    K0: tuple[float, ...]  # Distribution coefficients at Tref
    dH: tuple[float, ...] | None = None  # Heat of extraction for T-dependence
    Tref: float = 298.15


def get_K_values(
    coeffs: DistributionCoeffs,
    T: Array,
) -> dict[str, Array]:
    """Calculate distribution coefficients at given temperature.

    Args:
        coeffs: Distribution coefficient parameters
        T: Temperature (K)

    Returns:
        Dictionary of K-values for each transferring species
    """
    R = 8.314  # J/(mol·K)
    K_dict = {}

    for i, species in enumerate(coeffs.species):
        K0 = coeffs.K0[i]
        if coeffs.dH is not None:
            dH = coeffs.dH[i]
            K = K0 * jnp.exp(-dH / R * (1/T - 1/coeffs.Tref))
        else:
            K = jnp.asarray(K0)
        K_dict[species] = K

    return K_dict


# =============================================================================
# Activity Coefficient Models
# =============================================================================

class NRTLParams(NamedTuple):
    """NRTL activity coefficient model parameters.

    For temperature-dependent parameters:
        tau_ij = a_ij + b_ij/T

    G_ij = exp(-alpha_ij * tau_ij)

    Attributes:
        species: List of all species (in order)
        a: Constant part of tau parameter matrix (n x n)
        b: Temperature-dependent part of tau (n x n), in K
        alpha: Non-randomness parameter matrix (n x n)
    """
    species: tuple[str, ...]
    a: Array  # (n, n) matrix of constant tau coefficients
    b: Array  # (n, n) matrix of temperature coefficients
    alpha: Array  # (n, n) non-randomness parameters

    def gamma(self, x: Array, T: Array) -> Array:
        """Activity coefficients (the ``ActivityModel`` protocol adapter)."""
        return nrtl_activity_coefficients(x, T, self)


def nrtl_activity_coefficients(
    x: Array,
    T: Array,
    params: NRTLParams,
) -> Array:
    """Calculate activity coefficients using NRTL model.

    Fully vectorized implementation for JAX compatibility.

    Args:
        x: Mole fractions (array, same order as params.species)
        T: Temperature (K)
        params: NRTL parameters

    Returns:
        Activity coefficients for each species
    """
    # Temperature-dependent tau
    tau = params.a + params.b / T

    # G matrix
    G = jnp.exp(-params.alpha * tau)

    # Compute activity coefficients (vectorized)
    # ln(gamma_i) = (sum_j tau_ji * G_ji * x_j) / (sum_k G_ki * x_k)
    #             + sum_j (x_j * G_ij / sum_k G_kj * x_k) * (tau_ij - sum_m x_m * tau_mj * G_mj / sum_k G_kj * x_k)

    # Term 1: (sum_j tau_ji * G_ji * x_j) / (sum_k G_ki * x_k)
    # Denominator for all i: G.T @ x, shape (n,)
    # Numerator for all i: (tau * G).T @ x, shape (n,)
    denom = G.T @ x  # shape (n,)
    numer_tau_G = (tau * G).T @ x  # shape (n,)
    term1 = numer_tau_G / denom

    # Term 2: sum_j (x_j * G_ij / sum_k G_kj * x_k) * (tau_ij - sum_m x_m * tau_mj * G_mj / sum_k G_kj * x_k)
    # denom[j] = sum_k G_kj * x_k (already computed above)
    # numer_tau_G[j] = sum_m x_m * tau_mj * G_mj (already computed above)
    # weight[i,j] = x_j * G_ij / denom[j]
    # inner[i,j] = tau_ij - numer_tau_G[j] / denom[j]
    weight = G * (x / denom)  # broadcasts (n,) over columns of (n,n)
    inner = tau - (numer_tau_G / denom)  # broadcasts (n,) over columns of (n,n)
    term2 = jnp.sum(weight * inner, axis=1)  # sum over j

    ln_gamma = term1 + term2
    return jnp.exp(ln_gamma)


class UNIQUACParams(NamedTuple):
    """UNIQUAC activity coefficient model parameters.

    For temperature-dependent parameters:
        tau_ij = exp(-(a_ij + b_ij/T))

    Attributes:
        species: List of all species (in order)
        r: Volume parameters for each species
        q: Surface area parameters for each species
        a: Constant part of interaction parameters (n x n)
        b: Temperature-dependent part (n x n), in K
        z: Coordination number (default 10, typical range 6-12)
    """
    species: tuple[str, ...]
    r: Array  # Volume parameters
    q: Array  # Surface parameters
    a: Array  # (n, n) interaction parameter matrix
    b: Array  # (n, n) temperature coefficients
    z: float = 10.0  # Coordination number

    def gamma(self, x: Array, T: Array) -> Array:
        """Activity coefficients (the ``ActivityModel`` protocol adapter)."""
        return uniquac_activity_coefficients(x, T, self)


def uniquac_activity_coefficients(
    x: Array,
    T: Array,
    params: UNIQUACParams,
) -> Array:
    """Calculate activity coefficients using UNIQUAC model.

    Fully vectorized implementation for JAX compatibility.

    Args:
        x: Mole fractions (array, same order as params.species)
        T: Temperature (K)
        params: UNIQUAC parameters

    Returns:
        Activity coefficients for each species
    """
    r = params.r
    q = params.q
    z = params.z

    # Segment and area fractions
    phi = x * r / jnp.sum(x * r)  # Segment fraction
    theta = x * q / jnp.sum(x * q)  # Area fraction

    # l parameter
    l = (z / 2) * (r - q) - (r - 1)

    # Temperature-dependent tau
    tau = jnp.exp(-(params.a + params.b / T))

    # Combinatorial contribution (already vectorized)
    ln_gamma_C = jnp.log(phi / x) + (z / 2) * q * jnp.log(theta / phi) + l - (phi / x) * jnp.sum(x * l)

    # Residual contribution (vectorized)
    # sum_theta_tau[i] = sum_j theta[j] * tau[j,i] = (tau.T @ theta)[i]
    sum_theta_tau = tau.T @ theta  # shape (n,)

    # Term 1: -q * log(sum_theta_tau)
    term1 = -q * jnp.log(sum_theta_tau)

    # Term 2: q * (1 - sum_j (theta[j] * tau[i,j] / sum_k theta[k] * tau[k,j]))
    # sum_theta_tau_j[j] = sum_k theta[k] * tau[k,j] = (tau.T @ theta)[j] (same as above)
    # inner[i,j] = theta[j] * tau[i,j] / sum_theta_tau[j]
    inner = tau * (theta / sum_theta_tau)  # broadcasts (n,) over columns of (n,n)
    term2 = q * (1 - jnp.sum(inner, axis=1))  # sum over j

    ln_gamma_R = term1 + term2

    return jnp.exp(ln_gamma_C + ln_gamma_R)


# =============================================================================
# LLE Equilibrium Calculation
# =============================================================================

@dataclass
class LLEEquilibrium:
    """Liquid-liquid equilibrium calculator.

    Computes how solutes distribute between aqueous and organic phases
    based on either distribution coefficients or activity coefficient models.

    Attributes:
        solutes: Species that transfer between phases
        aqueous_carrier: Species that stays in aqueous phase
        organic_carrier: Species that stays in organic phase
        K_coeffs: Distribution coefficient model (optional)
        nrtl_params: NRTL parameters (optional)
        uniquac_params: UNIQUAC parameters (optional)
    """

    symbol = "LLE"
    equations = [
        r"K_i = \frac{y_i^{\mathrm{org}}}{x_i^{\mathrm{aq}}} \qquad \text{(distribution coefficient)}",
        r"F z_i = E\, y_i^{\mathrm{org}} + R\, x_i^{\mathrm{aq}} \qquad \text{(component balance)}",
        r"\gamma_i^{\mathrm{aq}} x_i^{\mathrm{aq}} = \gamma_i^{\mathrm{org}} y_i^{\mathrm{org}} \qquad \text{(isoactivity, NRTL/UNIQUAC)}",
    ]
    assumptions = [
        "Two liquid phases at equilibrium, completely separated.",
        "Either constant distribution coefficients or an activity-coefficient model.",
        "Isothermal; no phase change other than partitioning.",
    ]
    references = [
        "Seader, Henley, Roper. Separation Process Principles, 3e, Ch. 8 (liquid-liquid extraction).",
    ]
    solutes: list[str]
    aqueous_carrier: str
    organic_carrier: str
    K_coeffs: DistributionCoeffs | None = None
    nrtl_params: NRTLParams | None = None
    uniquac_params: UNIQUACParams | None = None
    activity_model: Literal["K", "NRTL", "UNIQUAC"] = "K"
    mutual_solubility: dict[str, float] | None = None

    def get_distribution_coefficients(
        self,
        x_aq: dict[str, Array],
        x_org: dict[str, Array],
        T: Array,
    ) -> dict[str, Array]:
        """Calculate distribution coefficients at equilibrium.

        Convention: K = C_extract / C_raffinate = C_org / C_aq
        For K-value model: Returns constant (or T-dependent) K values
        For activity models: K = gamma_aq / gamma_org (from LLE equilibrium x*gamma equality)

        Args:
            x_aq: Mole fractions in aqueous phase
            x_org: Mole fractions in organic phase
            T: Temperature (K)

        Returns:
            Distribution coefficients for each solute
        """
        if self.activity_model == "K":
            if self.K_coeffs is None:
                raise ValueError("K_coeffs required for K model")
            return get_K_values(self.K_coeffs, T)

        elif self.activity_model == "NRTL":
            if self.nrtl_params is None:
                raise ValueError("nrtl_params required for NRTL model")

            # Convert dicts to arrays
            species = self.nrtl_params.species
            x_aq_arr = jnp.array([x_aq.get(s, 0.0) for s in species])
            x_org_arr = jnp.array([x_org.get(s, 0.0) for s in species])

            gamma_aq = nrtl_activity_coefficients(x_aq_arr, T, self.nrtl_params)
            gamma_org = nrtl_activity_coefficients(x_org_arr, T, self.nrtl_params)

            K_dict = {}
            for i, s in enumerate(species):
                if s in self.solutes:
                    # K = gamma_aq / gamma_org (equilibrium condition)
                    K_dict[s] = gamma_aq[i] / gamma_org[i]
            return K_dict

        elif self.activity_model == "UNIQUAC":
            if self.uniquac_params is None:
                raise ValueError("uniquac_params required for UNIQUAC model")

            species = self.uniquac_params.species
            x_aq_arr = jnp.array([x_aq.get(s, 0.0) for s in species])
            x_org_arr = jnp.array([x_org.get(s, 0.0) for s in species])

            gamma_aq = uniquac_activity_coefficients(x_aq_arr, T, self.uniquac_params)
            gamma_org = uniquac_activity_coefficients(x_org_arr, T, self.uniquac_params)

            K_dict = {}
            for i, s in enumerate(species):
                if s in self.solutes:
                    K_dict[s] = gamma_aq[i] / gamma_org[i]
            return K_dict

        else:
            raise ValueError(f"Unknown activity model: {self.activity_model}")


# =============================================================================
# Partially miscible two-liquid-phase flash
# =============================================================================

_FLASH_FLOOR = 1e-14


def _lle_activity(lle_model):
    """``(model, species or None)`` of an LLEEquilibrium (NRTL/UNIQUAC) or a bare activity model."""
    from difflow.phase_diagrams import _lle_model
    model = _lle_model(lle_model)
    return model, getattr(model, "species", None)


def _stop_model(model):
    """``stop_gradient`` on the numeric leaves of an activity model (species names pass through)."""
    return jax.tree_util.tree_map(
        lambda v: lax.stop_gradient(v) if isinstance(v, (jax.Array, float, jnp.ndarray)) else v, model)


def _flash_seeds(zz: Array, pair: tuple[int, int]) -> tuple[Array, Array]:
    """Starting phase compositions: pair-0-rich phase I, pair-1-rich phase II, three solute loadings."""
    n = zz.shape[0]
    e0 = jnp.zeros(n).at[pair[0]].set(1.0)
    e1 = jnp.zeros(n).at[pair[1]].set(1.0)
    others = zz * (1.0 - e0 - e1)
    xs_I, xs_II = [], []
    for k in (0.3, 1.0, 2.0):
        xI = 0.9 * e0 + 0.1 * e1 + k * others
        xII = 0.1 * e0 + 0.9 * e1 + k * others
        xs_I.append(xI / jnp.sum(xI))
        xs_II.append(xII / jnp.sum(xII))
    return jnp.stack(xs_I), jnp.stack(xs_II)


def _rachford_rice(zz: Array, K: Array, n_bisect: int = 60) -> tuple[Array, Array, Array]:
    """Phase-II fraction ``beta`` in [0, 1] solving ``sum z (K-1)/(1 + beta (K-1)) = 0``.

    Returns ``(beta, f(0), f(1))``; the split is two-phase iff ``f(0) > 0 > f(1)``
    (otherwise ``beta`` is clipped to the single-phase end).
    """
    def f(b):
        return jnp.sum(zz * (K - 1.0) / (1.0 + b * (K - 1.0)))

    f0, f1 = f(0.0), f(1.0)

    def body(_, ab):
        a, b = ab
        m = 0.5 * (a + b)
        pos = f(m) > 0.0
        return jnp.where(pos, m, a), jnp.where(pos, b, m)

    a, b = lax.fori_loop(0, n_bisect, body, (jnp.asarray(0.0), jnp.asarray(1.0)))
    beta = jnp.where(f0 <= 0.0, 0.0, jnp.where(f1 >= 0.0, 1.0, 0.5 * (a + b)))
    return beta, f0, f1


def _flash_resid(w: Array, zz: Array, T: Array, model) -> Array:
    n = zz.shape[0]
    xI, xII, beta = w[:n], w[n:2 * n], w[2 * n]
    from difflow.activity import activity_gamma
    r_iso = xI * activity_gamma(model, xI, T) - xII * activity_gamma(model, xII, T)
    r_bal = zz - (1.0 - beta) * xI - beta * xII
    return jnp.concatenate([r_iso, r_bal, jnp.reshape(jnp.sum(xI) - 1.0, (1,))])


def ternary_lle_flash(lle_model, z: Array, T: Array | float, guess: tuple | None = None,
                      pair: tuple[int, int] = (0, 1), n_ss: int = 60, n_newton: int = 12) -> dict:
    """Two-liquid-phase split of a feed ``z`` with mutual solubility (differentiable).

    Solves the isothermal LLE flash

    .. math:: x^{I}_i\\gamma^{I}_i = x^{II}_i\\gamma^{II}_i,\\qquad
              z_i = (1-\\beta)\\,x^{I}_i + \\beta\\,x^{II}_i,\\qquad \\sum_i x^{I}_i = 1

    for any number of species. The stage is a Rachford-Rice split in the phase
    fraction ``beta`` wrapped around isoactivity K updates (``K_i = gamma^I_i/gamma^II_i``),
    a fixed number of successive-substitution iterations from three seeds
    (``pair[0]``-rich phase I, ``pair[1]``-rich phase II, three solute loadings;
    the seed with the largest phase separation is kept), then a fixed number of
    damped Newton iterations on the full system. There is no data-dependent
    Python control flow, so it is ``jit``-able and ``vmap``-able.

    **Gradients** are implicit: the converged state is stop-gradient'd and one
    Newton step with the Jacobian held fixed is applied, whose derivative is the
    implicit-function derivative ``-J^{-1} dF/dtheta``, so ``jax.grad`` with respect to
    ``z``, ``T`` and the activity-model parameters (NRTL ``tau``) are exact where the
    flash converged (``converged=True``). Where the feed is a single liquid
    (``two_phase=False``) the result is *masked*: both phases are the feed
    composition, ``beta = 0`` (everything in "phase I") and the gradients are those of
    that identity.

    Every species must have ``z_i > 0`` (a species with zero flow makes the
    system singular); use a tiny positive amount.

    Args:
        lle_model: An :class:`LLEEquilibrium` using NRTL or UNIQUAC, ``NRTLParams``,
            ``UNIQUACParams`` or any activity model with a ``gamma(x, T)`` method.
        z: Feed amounts or mole fractions ``(n,)`` (normalized internally).
        T: Temperature (K).
        guess: Optional starting ``(x_I, x_II)`` compositions, replacing the seeds.
        pair: Indices of the partially miscible pair that define the phase labels:
            phase I is rich in ``pair[0]``, phase II in ``pair[1]``.
        n_ss: Successive-substitution iterations.
        n_newton: Newton iterations on the full system.

    Returns:
        dict with ``x_I``, ``x_II`` (mole fractions), ``beta`` (fraction of the
        moles in phase II), ``amount_I``, ``amount_II`` (``sum(z)`` times the
        fractions), ``K`` (``x_II/x_I``), ``two_phase`` (bool), ``converged``
        (bool: the full-system residual is below 1e-9 and the phases are
        distinct) and ``residual``.

    Example:
        >>> import jax.numpy as jnp
        >>> from difflow.units.lle import NRTLParams, ternary_lle_flash
        >>> tau = jnp.array([[0., 2.5, .5], [2.5, 0., .3], [.8, .6, 0.]])
        >>> m = NRTLParams(("a", "b", "s"), tau, jnp.zeros((3, 3)), 0.3 * (1 - jnp.eye(3)))
        >>> r = ternary_lle_flash(m, jnp.array([0.4, 0.5, 0.1]), 300.0)
        >>> bool(r["two_phase"]), round(float(r["beta"]), 3)
        (True, 0.575)
    """
    from difflow.activity import activity_gamma
    model, _ = _lle_activity(lle_model)
    z = jnp.asarray(z, dtype=float)
    T = jnp.asarray(T, dtype=float)
    n = z.shape[0]
    tot = jnp.sum(z)
    zz = z / tot

    def gam(x):
        return activity_gamma(model, x, T)

    # ---- stage A: seeds -> successive substitution (not differentiated) -------
    zs, Ts = lax.stop_gradient(zz), lax.stop_gradient(T)
    ms = _stop_model(model)

    def gam_s(x):
        return activity_gamma(ms, x, Ts)

    if guess is None:
        xI0, xII0 = _flash_seeds(zs, pair)
    else:
        xI0 = jnp.asarray(guess[0], float)[None, :]
        xII0 = jnp.asarray(guess[1], float)[None, :]

    def ss_one(xI, xII):
        def step(c, _):
            xI, xII = c
            K = gam_s(xI) / gam_s(xII)
            beta, _, _ = _rachford_rice(zs, K)
            nI = zs / (1.0 + beta * (K - 1.0))
            nII = K * nI
            nI = jnp.maximum(nI, _FLASH_FLOOR)
            nII = jnp.maximum(nII, _FLASH_FLOOR)
            return (nI / jnp.sum(nI), nII / jnp.sum(nII)), None
        (xI, xII), _ = lax.scan(step, (xI, xII), None, length=n_ss)
        return xI, xII

    xIs, xIIs = jax.vmap(ss_one)(xI0, xII0)
    sep = jnp.sum(jnp.abs(xIs - xIIs), axis=1)
    k_best = jnp.argmax(sep)
    xI, xII = xIs[k_best], xIIs[k_best]
    K = gam_s(xI) / gam_s(xII)
    beta, f0, f1 = _rachford_rice(zs, K)
    beta = jnp.clip(beta, 1e-6, 1.0 - 1e-6)

    # ---- stage B: damped Newton on the full system (not differentiated) ------
    w = jnp.concatenate([xI, xII, jnp.reshape(beta, (1,))])

    def newton(w, _):
        f = _flash_resid(w, zs, Ts, ms)
        J = jax.jacfwd(lambda v: _flash_resid(v, zs, Ts, ms))(w)
        d = jnp.linalg.solve(J + 1e-14 * jnp.eye(2 * n + 1), -f)
        # keep the compositions positive and beta inside (0, 1)
        x, dx = w[:2 * n], d[:2 * n]
        a_x = jnp.where(dx < 0.0, 0.95 * x / jnp.where(dx < 0.0, -dx, 1.0), jnp.inf)
        db = d[2 * n]
        b = w[2 * n]
        a_b = jnp.where(db > 0.0, 0.95 * (1.0 - b) / jnp.where(db > 0.0, db, 1.0),
                        jnp.where(db < 0.0, 0.95 * b / jnp.where(db < 0.0, -db, 1.0), jnp.inf))
        alpha = jnp.minimum(1.0, jnp.minimum(jnp.min(a_x), a_b))
        wn = w + alpha * d
        ok = jnp.all(jnp.isfinite(wn))
        return jnp.where(ok, wn, w), None

    w, _ = lax.scan(newton, w, None, length=n_newton)
    res = jnp.max(jnp.abs(_flash_resid(w, zs, Ts, ms)))
    xI_f, xII_f, beta_f = w[:n], w[n:2 * n], w[2 * n]
    separated = jnp.sum(jnp.abs(xI_f - xII_f)) > 1e-6
    two_phase = (res < 1e-9) & separated & (beta_f > 0.0) & (beta_f < 1.0) \
        & (jnp.min(xI_f) > -1e-12) & (jnp.min(xII_f) > -1e-12)

    # ---- stage C: implicit-gradient polish (one Newton step, J held fixed) ---
    w_safe = jnp.where(two_phase, w, jnp.concatenate([xI0[0], xII0[0], jnp.array([0.5])]))
    w_safe = lax.stop_gradient(w_safe)
    f_w = _flash_resid(w_safe, zz, T, model)
    J = lax.stop_gradient(jax.jacfwd(lambda v: _flash_resid(v, zs, Ts, ms))(w_safe))
    w_pol = w_safe - jnp.linalg.solve(J + 1e-14 * jnp.eye(2 * n + 1), f_w)
    xI_o = jnp.where(two_phase, w_pol[:n], zz)
    xII_o = jnp.where(two_phase, w_pol[n:2 * n], zz)
    beta_o = jnp.where(two_phase, w_pol[2 * n], 0.0)
    K_o = jnp.where(two_phase, xII_o / jnp.where(two_phase, xI_o, 1.0), 1.0)
    return {"x_I": xI_o, "x_II": xII_o, "beta": beta_o,
            "amount_I": tot * (1.0 - beta_o), "amount_II": tot * beta_o, "K": K_o,
            "two_phase": two_phase, "converged": two_phase, "residual": res}


# =============================================================================
# Multi-Stage Cascade Extractor
# =============================================================================

# =============================================================================
# Rigorous (partially miscible) countercurrent cascade
# =============================================================================

_N_CASCADE_NEWTON = 60


def _cascade_resid(u: Array, F: Array, S: Array, T: Array, model, N: int, nsp: int) -> Array:
    """Stage balances and isoactivity for ``N`` countercurrent stages (flows ``u = [R; E]``).

    ``R_n`` is the raffinate (phase I) leaving stage ``n``, ``E_n`` the extract (phase II);
    stage 1 gets the feed ``F``, stage ``N`` the solvent ``S``.
    """
    from difflow.activity import activity_gamma
    R = u[:N * nsp].reshape(N, nsp)
    E = u[N * nsp:].reshape(N, nsp)
    R_in = jnp.concatenate([F[None, :], R[:-1]], axis=0)
    E_in = jnp.concatenate([E[1:], S[None, :]], axis=0)
    bal = R_in + E_in - R - E
    xR = R / jnp.sum(R, axis=1, keepdims=True)
    xE = E / jnp.sum(E, axis=1, keepdims=True)
    gam = jax.vmap(lambda x: activity_gamma(model, x, T))
    iso = xR * gam(xR) - xE * gam(xE)
    return jnp.concatenate([bal.reshape(-1), iso.reshape(-1)])


def _cascade_seed(model, F, S, T, N: int, nsp: int, pair: tuple[int, int]) -> Array:
    """Kremser-profile seed: K and phase ratio from one flash of ``F + S``."""
    fl = ternary_lle_flash(model, F + S, T, pair=pair)
    tot = jnp.sum(F + S)
    R1 = jnp.where(fl["two_phase"], fl["amount_I"] * fl["x_I"], F)
    E1 = jnp.where(fl["two_phase"], fl["amount_II"] * fl["x_II"], S)
    floor = 1e-6 * tot
    R1 = jnp.maximum(R1, floor)
    E1 = jnp.maximum(E1, floor)
    A = (E1 / jnp.sum(E1)) / (R1 / jnp.sum(R1)) * jnp.sum(E1) / jnp.sum(R1)   # K_i * E/R
    A = jnp.maximum(A, 1e-6)
    n_idx = jnp.arange(1, N + 1, dtype=float)[:, None]
    near = jnp.abs(A - 1.0) < 1e-8
    As = jnp.where(near, 2.0, A)
    krem = jnp.where(near, (N - n_idx + 1.0) / (N + 1.0),
                     (As ** (N - n_idx + 1.0) - 1.0) / (As ** (N + 1.0) - 1.0))
    carrier = jnp.zeros(nsp).at[pair[0]].set(1.0).at[pair[1]].set(1.0)
    feed_like = jnp.where(carrier > 0, R1, F + S)         # solutes: Kremser on feed + solvent solute
    R = jnp.where(carrier > 0, R1[None, :], feed_like[None, :] * krem)
    E = jnp.where(carrier > 0, E1[None, :], A[None, :] * R)
    return jnp.maximum(jnp.concatenate([R.reshape(-1), E.reshape(-1)]), floor)


def _rigorous_cascade(model, F: Array, S: Array, T: Array, N: int, pair: tuple[int, int]):
    """Solve the ``N``-stage countercurrent cascade with the full LLE equations.

    Damped Newton on the flat system of ``2 N n`` flows (fixed iteration count, fraction-to-
    boundary step limit) from a Kremser-profile seed, not differentiated; then one Newton step
    with the Jacobian held fixed, which carries the implicit-function gradient with respect to
    ``F``, ``S``, ``T`` and the activity-model parameters.

    Returns ``(R, E, info)`` with ``R, E`` of shape ``(N, n)`` (flows), plus ``converged``,
    ``residual`` and ``two_phase`` (every stage has distinct phases).
    """
    nsp = F.shape[0]
    sc = jnp.sum(F + S)
    Fs, Ss = F / sc, S / sc
    sg = lax.stop_gradient
    ms = _stop_model(model)
    Fsg, Ssg, Tsg = sg(Fs), sg(Ss), sg(T)

    def resid_sg(u):
        return _cascade_resid(u, Fsg, Ssg, Tsg, ms, N, nsp)

    u0 = _cascade_seed(ms, Fsg, Ssg, Tsg, N, nsp, pair)
    dim = 2 * N * nsp

    def newton(u, _):
        f = resid_sg(u)
        J = jax.jacfwd(resid_sg)(u)
        d = jnp.linalg.solve(J + 1e-14 * jnp.eye(dim), -f)
        a = jnp.where(d < 0.0, 0.95 * u / jnp.where(d < 0.0, -d, 1.0), jnp.inf)
        alpha = jnp.minimum(1.0, jnp.min(a))
        un = u + alpha * d
        return jnp.where(jnp.all(jnp.isfinite(un)), un, u), None

    u, _ = lax.scan(newton, u0, None, length=_N_CASCADE_NEWTON)
    res = jnp.max(jnp.abs(resid_sg(u)))
    Rr, Er = u[:N * nsp].reshape(N, nsp), u[N * nsp:].reshape(N, nsp)
    xR = Rr / jnp.sum(Rr, axis=1, keepdims=True)
    xE = Er / jnp.sum(Er, axis=1, keepdims=True)
    two_phase = jnp.min(jnp.sum(jnp.abs(xR - xE), axis=1)) > 1e-6
    converged = (res < 1e-10) & two_phase & jnp.all(jnp.isfinite(u)) & jnp.all(u > 0.0)

    # implicit-gradient polish
    u_safe = sg(jnp.where(converged, u, u0))
    J = sg(jax.jacfwd(resid_sg)(u_safe))
    u_pol = u_safe - jnp.linalg.solve(J + 1e-14 * jnp.eye(dim),
                                       _cascade_resid(u_safe, Fs, Ss, T, model, N, nsp))
    u_out = jnp.where(converged, u_pol, u) * sc
    R_out, E_out = u_out[:N * nsp].reshape(N, nsp), u_out[N * nsp:].reshape(N, nsp)
    return R_out, E_out, {"converged": converged, "residual": res, "two_phase": two_phase}


def _static_stage_count(n_stages, profile_stages: int | None = None) -> tuple[int, bool]:
    """Number of profile rows to tabulate, and whether ``n_stages`` is traced.

    A concrete ``n_stages`` gives ``int(n_stages)`` rows. Under ``jit`` with a
    traced ``n_stages`` the row count cannot follow it, so ``profile_stages``
    rows are tabulated (rows past ``n_stages`` are NaN-masked by the caller);
    without ``profile_stages`` the profiles hold only the inlet row.
    """
    try:
        return int(n_stages), False
    except (jax.errors.ConcretizationTypeError, jax.errors.TracerIntegerConversionError, TypeError):
        if profile_stages is None:
            import warnings
            warnings.warn("MultistageCascade: n_stages is traced, so info['profiles'] holds only the "
                          "inlet row; set CascadeParams.profile_stages to tabulate a fixed number of "
                          "stages under jit", stacklevel=4)
            return 0, True
        return int(profile_stages), True


def _mask_rows(x: Array, n: int, n_stages: Array, traced: bool) -> Array:
    """NaN for a profile row ``n`` past a traced ``n_stages`` (rows exist up to ``profile_stages``)."""
    return jnp.where(n <= n_stages + 1e-9, x, jnp.nan) if traced else x


def _overall_efficiency(eta: Array, E: Array) -> Array:
    """Ideal stages per actual stage for a raffinate-phase Murphree efficiency ``eta``.

    Lewis relation for straight equilibrium and operating lines (constant ``K``,
    immiscible carriers), written for the raffinate phase, whose stripping-factor
    analogue is ``1/E``: ``E_o = ln(1 + eta (1/E - 1)) / ln(1/E)``, smooth
    through ``E = 1`` (limit ``eta``). Exact for the counter-current cascade:
    ``N`` real stages give the outlet of ``E_o N`` ideal stages (Treybal,
    *Mass-Transfer Operations* 3e, Sec. 10.4; Wankat 4e Sec. 13.5).
    """
    return _taylor_log_fraction(eta, 1.0 / E - 1.0)


@dataclass(repr=False)
class CascadeParams(ParamsMixin):
    """Parameters for multi-stage cascade extractor.

    Attributes:
        n_stages: Number of equilibrium stages (can be continuous for optimization)
        equilibrium: LLE equilibrium calculator
        flow_config: ``'counter_current'`` (Kremser), ``'crossflow'`` (fresh
            solvent split equally over the stages, ``(1 + E/N)^-N`` left) or
            ``'co_current'``
        stage_efficiency: Murphree raffinate-phase stage efficiency (0-1]:
            ``(X_{n-1} - X_n) = eta (X_{n-1} - X_n^*)`` with ``X_n^*`` in
            equilibrium with the extract leaving stage ``n``. ``None``
            (default) means ideal stages for counter-current and crossflow and
            the legacy 0.8 for co-current. Counter-current applies it through
            the Lewis overall efficiency (``E_o N`` ideal stages, exact for
            straight lines); crossflow stage by stage.
        profile_stages: Number of stage rows to tabulate in ``info["profiles"]``
            when ``n_stages`` is traced (``jax.jit`` over it). Rows past
            ``n_stages`` are NaN. ``None`` (default): a concrete ``n_stages``
            tabulates ``int(n_stages)`` rows; a traced one only the inlet row
            (with a warning).
        model: ``"immiscible"`` (default; Kremser, immiscible solvents, constant phase
            flows) or ``"rigorous"``: ``n_stages`` countercurrent ideal stages solved
            with the full two-liquid-phase equations (isoactivity from the NRTL or
            UNIQUAC parameters of ``equilibrium``, mutual solubility of all species,
            variable phase flows), from a Kremser-profile seed. The rigorous model
            needs an integer ``n_stages``, ``flow_config="counter_current"``, an
            activity-model ``equilibrium`` without ``mutual_solubility`` (which it
            supersedes), and ideal stages (``stage_efficiency`` must be
            ``None`` or 1).
    """
    n_stages: int | float | Array
    equilibrium: LLEEquilibrium
    flow_config: Literal["counter_current", "crossflow", "co_current"] = "counter_current"
    stage_efficiency: float | Array | None = None
    model: Literal["immiscible", "rigorous"] = "immiscible"
    profile_stages: int | None = None

    def __post_init__(self):
        if self.model not in ("immiscible", "rigorous"):
            raise ValueError(f"model must be 'immiscible' or 'rigorous', got {self.model!r}")
        if self.flow_config not in ("counter_current", "crossflow", "co_current"):
            raise ValueError("flow_config must be 'counter_current', 'crossflow' or 'co_current', "
                             f"got {self.flow_config!r}")


class MultistageCascade:
    """Multi-stage liquid-liquid extraction cascade.

    Models a series of mixer-settler stages or a tray column.
    Supports counter-current, crossflow (fresh solvent split equally over the
    stages) and co-current flow configurations. The immiscible model works on
    the solute-free (ratio) basis: ``K = Y/X`` with ``X`` = solute/aqueous
    carrier and ``Y`` = solute/organic carrier, so ``E = K S/F`` uses carrier
    flows; ``info["profiles"]`` holds ``X_n`` and ``Y_n`` on that basis.

    For optimization, n_stages can be treated as a continuous variable
    using stage efficiency interpolation.
    """

    symbol = "LLE Cascade"
    equations = [
        r"\frac{X_N - Y_{in}/K}{X_0 - Y_{in}/K} = \frac{E - 1}{E^{N+1} - 1},\ E = \frac{K S}{F} \qquad \text{(Kremser, counter-current)}",
        r"\frac{X_N - Y_{in}/K}{X_0 - Y_{in}/K} = \left(1 + \frac{E}{N}\right)^{-N} \qquad \text{(crossflow, equal solvent split)}",
        r"\gamma_i^{R}x_i^{R}=\gamma_i^{E}x_i^{E},\ \ R_{n-1}+E_{n+1}=R_n+E_n \qquad \text{(rigorous model, every stage)}",
        r"E_i = \frac{y_i^\ast}{K_i\, x_i}\qquad \text{(extraction factor)}",
        r"\text{MB at stage } n:\ L x_{i,n-1} + S y_{i,n+1} = L x_{i,n} + S y_{i,n}",
    ]
    assumptions = [
        "Each stage reaches equilibrium, or a Murphree raffinate-phase stage_efficiency of it.",
        "Immiscible solvents; constant phase flows along the cascade.",
        "Constant distribution coefficients K_i with respect to temperature within a stage.",
        "model='rigorous' drops the immiscible and constant-flow assumptions: ideal stages with the full "
        "two-liquid-phase equations (NRTL/UNIQUAC isoactivity, mutual solubility, variable phase flows); "
        "integer n_stages, counter-current only.",
    ]
    references = [
        "Kremser, A. National Petroleum News, 22(21), 43 (1930).",
        "Seader, Henley, Roper. Separation Process Principles, 3e, Ch. 8.",
    ]
    parameter_symbols = {"n_stages": "N", "stage_efficiency": r"\eta_M"}
    parameter_units = {"n_stages": "-", "stage_efficiency": "-", "profile_stages": "-"}
    numerical_method = ("Analytical Kremser for counter-current (solute in the entering solvent included); "
                        "closed form (1 + E/N)^-N for crossflow; approach-to-equilibrium for co-current. "
                        "A Murphree stage_efficiency enters counter-current through the Lewis overall "
                        "efficiency, crossflow stage by stage.")

    def __init__(self, params: CascadeParams):
        """Initialize cascade.

        Args:
            params: Cascade parameters
        """
        self.params = params

    def __call__(
        self,
        feed: Stream,
        solvent: Stream,
        T: Array | float | None = None,
    ) -> tuple[Stream, Stream, dict]:
        """Perform multi-stage extraction.

        Args:
            feed: Aqueous feed stream (contains solutes + aqueous carrier)
            solvent: Organic solvent stream (fresh solvent)
            T: Operating temperature (K). If None, uses feed temperature.

        Returns:
            raffinate: Aqueous outlet (depleted in solutes)
            extract: Organic outlet (enriched in solutes)
            info: Stage-by-stage profiles and other information
        """
        p = self.params
        eq = p.equilibrium

        T = jnp.asarray(T) if T is not None else feed["T"]

        if p.model == "rigorous":
            return self._call_rigorous(feed, solvent, T)

        # Get flows
        feed_flows = get_flows(feed)
        solvent_flows = get_flows(solvent)

        # Total aqueous and organic flows (carriers don't transfer)
        F_aq = feed_flows[eq.aqueous_carrier]
        F_org = solvent_flows[eq.organic_carrier]

        # Convert n_stages to JAX array for differentiability
        n_stages = jnp.asarray(p.n_stages, dtype=jnp.float64)

        # Solve stage-by-stage using Kremser equation (continuous n_stages)
        if p.flow_config == "counter_current":
            raffinate, extract, profiles = self._solve_counter_current(
                feed_flows, solvent_flows, F_aq, F_org, T, n_stages
            )
        elif p.flow_config == "crossflow":
            raffinate, extract, profiles = self._solve_crossflow(
                feed_flows, solvent_flows, F_aq, F_org, T, n_stages
            )
        else:
            raffinate, extract, profiles = self._solve_co_current(
                feed_flows, solvent_flows, F_aq, F_org, T, n_stages
            )

        # Create output streams
        P = feed["P"]
        raffinate_stream = make_stream(raffinate, T, P)
        extract_stream = make_stream(extract, T, P)

        info = {
            "n_stages": n_stages,
            "profiles": profiles,
            "T": T,
        }

        return raffinate_stream, extract_stream, info

    def _call_rigorous(self, feed: Stream, solvent: Stream, T: Array) -> tuple[Stream, Stream, dict]:
        """``model="rigorous"``: countercurrent stages with the two-liquid-phase flash equations."""
        p = self.params
        eq = p.equilibrium
        eta = p.stage_efficiency
        if eta is not None and (isinstance(eta, jax.core.Tracer) or float(eta) != 1.0):
            raise ValueError("model='rigorous' solves ideal stages; stage_efficiency must be None or 1")
        if p.flow_config != "counter_current":
            raise ValueError("model='rigorous' is implemented for flow_config='counter_current' only")
        if eq.activity_model not in ("NRTL", "UNIQUAC"):
            raise ValueError("model='rigorous' needs an LLEEquilibrium with activity_model 'NRTL' or "
                             "'UNIQUAC' (a K-value model has no mutual solubility)")
        if eq.mutual_solubility is not None:
            raise ValueError("model='rigorous' supersedes LLEEquilibrium.mutual_solubility (it solves "
                             "the mutual solubility); remove mutual_solubility or use model='immiscible'")
        if isinstance(p.n_stages, jax.core.Tracer):
            raise ValueError("model='rigorous' needs a concrete integer n_stages (not a traced value)")
        n_f = float(p.n_stages)
        if n_f != int(n_f) or n_f < 1:
            raise ValueError(f"model='rigorous' needs an integer n_stages >= 1, got {p.n_stages}")
        N = int(n_f)
        model, species = _lle_activity(eq)
        for c in (eq.aqueous_carrier, eq.organic_carrier):
            if c not in species:
                raise ValueError(f"carrier {c!r} is not in the activity model's species {species}")
        pair = (species.index(eq.aqueous_carrier), species.index(eq.organic_carrier))
        ff, sf = get_flows(feed), get_flows(solvent)
        F = jnp.stack([jnp.asarray(ff.get(s, 0.0), float) for s in species])
        S = jnp.stack([jnp.asarray(sf.get(s, 0.0), float) for s in species])
        if not isinstance(F, jax.core.Tracer) and not isinstance(S, jax.core.Tracer):
            tot = F + S
            for i, s in enumerate(species):
                if float(tot[i]) <= 0.0:
                    raise ValueError(f"species {s!r} has zero flow in the feed and the solvent; the rigorous "
                                     "model needs every species of the activity model present "
                                     "(give it a tiny positive flow or drop it from the model)")
        R, E, rinfo = _rigorous_cascade(model, F, S, jnp.asarray(T, float), N, pair)
        P = feed["P"]
        raffinate = make_stream({s: R[-1, i] for i, s in enumerate(species)}, T, P)
        extract = make_stream({s: E[0, i] for i, s in enumerate(species)}, T, P)
        xR = R / jnp.sum(R, axis=1, keepdims=True)
        xE = E / jnp.sum(E, axis=1, keepdims=True)
        xF = F / jnp.sum(F)
        yS = S / jnp.sum(S)
        profiles = {
            "x": {s: jnp.concatenate([xF[i:i + 1], xR[:, i]]) for i, s in enumerate(species)},
            "y": {s: jnp.concatenate([yS[i:i + 1], xE[:, i]]) for i, s in enumerate(species)},
            "R": R, "E": E, "species": tuple(species),
            "carrier_transfer": {},
            "mutual_solubility_active": True,
        }
        info = {"n_stages": jnp.asarray(float(N)), "profiles": profiles, "T": T, "model": "rigorous",
                "converged": rinfo["converged"], "residual": rinfo["residual"],
                "two_phase": rinfo["two_phase"]}
        return raffinate, extract, info

    def _solve_counter_current(
        self,
        feed_flows: dict,
        solvent_flows: dict,
        F_aq: Array,
        F_org: Array,
        T: Array,
        n_stages: Array,
    ) -> tuple[dict, dict, dict]:
        """Solve counter-current cascade using Kremser equation.

        Stage numbering: 1 = feed end, N = solvent end
        Aqueous flows: 1 -> 2 -> ... -> N
        Organic flows: N -> N-1 -> ... -> 1

        Uses the Kremser equation which is fully differentiable
        with respect to n_stages (continuous relaxation).

        Fully vectorized over solutes for JAX compatibility.
        """
        eq = self.params.equilibrium
        solutes = eq.solutes

        # Estimate phase compositions for activity coefficient models
        # Use feed composition for aqueous, solvent composition for organic
        F_in_arr = jnp.array([feed_flows.get(s, 0.0) for s in solutes])
        F_solvent_arr = jnp.array([solvent_flows.get(s, 0.0) for s in solutes])

        # Estimate mole fractions (including carriers)
        total_aq = F_aq + jnp.sum(F_in_arr)
        total_org = F_org + jnp.sum(F_solvent_arr)

        x_aq_est = {eq.aqueous_carrier: safe_divide(F_aq, total_aq)}
        x_org_est = {eq.organic_carrier: safe_divide(F_org, total_org)}
        for i, s in enumerate(solutes):
            x_aq_est[s] = safe_divide(F_in_arr[i], total_aq)
            x_org_est[s] = safe_divide(F_solvent_arr[i], total_org)

        # Get distribution coefficients with estimated compositions
        K_dict = eq.get_distribution_coefficients(x_aq_est, x_org_est, T)

        # Convert to arrays for vectorized computation
        K_arr = jnp.array([K_dict[s] for s in solutes])

        # Extraction factors for all solutes (ratio basis: K = Y/X with
        # Y = solute/organic carrier, X = solute/aqueous carrier)
        E_arr = K_arr * F_org / F_aq

        # Kremser equation with solute in the entering solvent (Treybal;
        # Wankat 4e Sec. 13.3; Geankoplis 5e Sec. 12.7):
        #   (X_N - X*) / (X_0 - X*) = (E - 1) / (E^(N+1) - 1),  X* = Y_in / K
        # The small fixed offset keeps the E = 1 limit, 1/(N+1), smooth.
        E_safe = E_arr + 1e-7
        eta = self.params.stage_efficiency
        # Murphree efficiency: N real stages act as E_o N ideal ones (Lewis).
        stage_scale = 1.0 if eta is None else _overall_efficiency(jnp.asarray(eta, float), E_safe)
        N_eq = n_stages * stage_scale
        E_Np1 = E_safe ** (N_eq + 1)
        delta_E = E_safe - 1.0
        frac_remaining = jnp.clip(delta_E / (E_Np1 - 1.0), 0.0, 1.0)

        X0 = safe_divide(F_in_arr, F_aq)
        Y_in = safe_divide(F_solvent_arr, F_org)
        X_star = Y_in / K_arr
        X_N = X_star + (X0 - X_star) * frac_remaining

        F_raffinate_arr = X_N * F_aq
        # Overall solute balance: what enters with feed AND solvent leaves
        # in the raffinate or the extract.
        F_extracted_arr = F_in_arr + F_solvent_arr - F_raffinate_arr

        # Convert back to dicts
        raffinate_flows = {eq.aqueous_carrier: F_aq}
        extract_flows = {eq.organic_carrier: F_org}
        for i, s in enumerate(solutes):
            raffinate_flows[s] = F_raffinate_arr[i]
            extract_flows[s] = F_extracted_arr[i]

        # Mutual solubility: transfer carrier amounts between phases (#157)
        # aq_in_org: mole fraction of aqueous carrier that dissolves in organic
        # org_in_aq: mole fraction of organic carrier that dissolves in aqueous
        carrier_transfer = {}
        if eq.mutual_solubility is not None:
            ms = eq.mutual_solubility
            aq_in_org = ms.get('aqueous_in_organic', 0.0)
            org_in_aq = ms.get('organic_in_aqueous', 0.0)
            aq_transferred = F_aq * aq_in_org
            org_transferred = F_org * org_in_aq
            raffinate_flows[eq.aqueous_carrier] = F_aq - aq_transferred
            raffinate_flows[eq.organic_carrier] = org_transferred
            extract_flows[eq.organic_carrier] = F_org - org_transferred
            extract_flows[eq.aqueous_carrier] = aq_transferred
            carrier_transfer = {
                'aqueous_to_organic': aq_transferred,
                'organic_to_aqueous': org_transferred,
            }

        # Stage profiles on the solute-free (ratio) basis of the Kremser
        # solution, at integer stages n = 0 .. N (stage 1 = feed end):
        #   X_n = X* + (X_0 - X*) (E^(N+1-n) - 1) / (E^(N+1) - 1)
        #   Y_n = K X_n  (extract leaving stage n, n >= 1)
        # Entry 0 holds the inlets: x[0] = feed X_0, y[0] = solvent Y_in.
        # With a Murphree efficiency real stage n sits at ideal stage E_o n
        # (straight lines), and the extract leaving it is off equilibrium, so
        # Y_n comes from the balance over stages n..N:
        #   Y_n = Y_in + (F/S) (X_{n-1} - X_N).
        profiles_x = {s: [] for s in solutes}
        profiles_y = {s: [] for s in solutes}
        for i, s in enumerate(solutes):
            profiles_x[s].append(X0[i])
            profiles_y[s].append(Y_in[i])

        N_int, traced = _static_stage_count(n_stages, self.params.profile_stages)
        x_prev = X0
        for n in range(1, N_int + 1):
            num = E_safe ** (N_eq + 1.0 - n * stage_scale) - 1.0
            frac_n = jnp.clip(num / (E_Np1 - 1.0), 0.0, 1.0)
            x_n = X_star + (X0 - X_star) * frac_n
            if eta is None:
                y_n = K_arr * x_n
            else:
                y_n = Y_in + F_aq / F_org * (x_prev - X_N)
            x_prev = x_n
            x_n = _mask_rows(x_n, n, n_stages, traced)
            y_n = _mask_rows(y_n, n, n_stages, traced)
            for i, s in enumerate(solutes):
                profiles_x[s].append(x_n[i])
                profiles_y[s].append(y_n[i])

        profiles = {
            "x": profiles_x,
            "y": profiles_y,
            "carrier_transfer": carrier_transfer,
            "mutual_solubility_active": eq.mutual_solubility is not None,
        }

        return raffinate_flows, extract_flows, profiles

    def _solve_crossflow(
        self,
        feed_flows: dict,
        solvent_flows: dict,
        F_aq: Array,
        F_org: Array,
        T: Array,
        n_stages: Array,
    ) -> tuple[dict, dict, dict]:
        """Crossflow (cross-current) cascade: fresh solvent split equally over N stages.

        The raffinate passes through stages 1..N; each stage gets ``S/N`` of
        fresh solvent and its own extract leaves. With immiscible carriers and
        a constant ``K`` (ratio basis) each ideal stage gives
        ``X_n - X* = (X_{n-1} - X*) / (1 + E/N)`` with ``E = K S / F`` and
        ``X* = Y_in / K``, so

            (X_N - X*) / (X_0 - X*) = (1 + E/N)^(-N),

        smooth in ``N`` (Geankoplis 5e Sec. 12.7; Wankat 4e Sec. 13.2). The
        extract stream returned is the combined extract of all stages.
        """
        eq = self.params.equilibrium
        solutes = eq.solutes
        F_in_arr = jnp.array([feed_flows.get(s, 0.0) for s in solutes])
        F_solvent_arr = jnp.array([solvent_flows.get(s, 0.0) for s in solutes])
        total_aq = F_aq + jnp.sum(F_in_arr)
        total_org = F_org + jnp.sum(F_solvent_arr)
        x_aq_est = {eq.aqueous_carrier: safe_divide(F_aq, total_aq)}
        x_org_est = {eq.organic_carrier: safe_divide(F_org, total_org)}
        for i, s in enumerate(solutes):
            x_aq_est[s] = safe_divide(F_in_arr[i], total_aq)
            x_org_est[s] = safe_divide(F_solvent_arr[i], total_org)
        K_dict = eq.get_distribution_coefficients(x_aq_est, x_org_est, T)
        K_arr = jnp.array([K_dict[s] for s in solutes])

        E_arr = K_arr * F_org / F_aq
        X0 = safe_divide(F_in_arr, F_aq)
        Y_in = safe_divide(F_solvent_arr, F_org)
        X_star = Y_in / K_arr
        # Stage n gets S/N of fresh solvent. With a raffinate Murphree
        # efficiency eta, X_n - X* = (X_{n-1} - X*) (E + eta N - eta E)/(E + eta N)
        # (balance F (X_{n-1} - X_n) = (S/N)(Y_n - Y_in) and
        # X_{n-1} - X_n = eta (X_{n-1} - Y_n/K)); eta = 1 is 1/(1 + E/N).
        eta = self.params.stage_efficiency
        if eta is None:
            per_stage = 1.0 / (1.0 + E_arr / n_stages)
        else:
            eta = jnp.asarray(eta, float)
            per_stage = (E_arr + eta * n_stages - eta * E_arr) / (E_arr + eta * n_stages)
        X_N = X_star + (X0 - X_star) * per_stage ** n_stages
        F_raffinate_arr = X_N * F_aq
        F_extracted_arr = F_in_arr + F_solvent_arr - F_raffinate_arr

        raffinate_flows = {eq.aqueous_carrier: F_aq}
        extract_flows = {eq.organic_carrier: F_org}
        for i, s in enumerate(solutes):
            raffinate_flows[s] = F_raffinate_arr[i]
            extract_flows[s] = F_extracted_arr[i]
        carrier_transfer = {}
        if eq.mutual_solubility is not None:
            ms = eq.mutual_solubility
            aq_transferred = F_aq * ms.get('aqueous_in_organic', 0.0)
            org_transferred = F_org * ms.get('organic_in_aqueous', 0.0)
            raffinate_flows[eq.aqueous_carrier] = F_aq - aq_transferred
            raffinate_flows[eq.organic_carrier] = org_transferred
            extract_flows[eq.organic_carrier] = F_org - org_transferred
            extract_flows[eq.aqueous_carrier] = aq_transferred
            carrier_transfer = {'aqueous_to_organic': aq_transferred,
                                'organic_to_aqueous': org_transferred}

        # Profiles (ratio basis): x[n] raffinate leaving stage n, y[n] the
        # extract leaving stage n (= K x[n] for ideal stages, else from the
        # stage balance); entry 0 is the feed / fresh solvent.
        profiles_x = {s: [X0[i]] for i, s in enumerate(solutes)}
        profiles_y = {s: [Y_in[i]] for i, s in enumerate(solutes)}
        N_int, traced = _static_stage_count(n_stages, self.params.profile_stages)
        x_prev = X0
        for n in range(1, N_int + 1):
            x_n = X_star + (X0 - X_star) * per_stage ** n
            if eta is None:
                y_n = K_arr * x_n
            else:
                y_n = Y_in + n_stages * F_aq / F_org * (x_prev - x_n)
            x_prev = x_n
            x_n = _mask_rows(x_n, n, n_stages, traced)
            y_n = _mask_rows(y_n, n, n_stages, traced)
            for i, s in enumerate(solutes):
                profiles_x[s].append(x_n[i])
                profiles_y[s].append(y_n[i])
        profiles = {
            "x": profiles_x,
            "y": profiles_y,
            "carrier_transfer": carrier_transfer,
            "mutual_solubility_active": eq.mutual_solubility is not None,
        }
        return raffinate_flows, extract_flows, profiles

    def _solve_co_current(
        self,
        feed_flows: dict,
        solvent_flows: dict,
        F_aq: Array,
        F_org: Array,
        T: Array,
        n_stages: Array,
    ) -> tuple[dict, dict, dict]:
        """Solve co-current cascade.

        Both phases flow in same direction: stage 1 -> 2 -> ... -> N
        Less efficient than counter-current but simpler.

        For co-current flow, equilibrium is approached asymptotically.
        The extraction efficiency increases with number of stages but
        is limited by single-stage equilibrium.

        Fully vectorized over solutes for JAX compatibility.
        """
        eq = self.params.equilibrium
        solutes = eq.solutes

        # Convert to arrays for vectorized computation
        F_in_arr = jnp.array([feed_flows.get(s, 0.0) for s in solutes])
        F_solvent_arr = jnp.array([solvent_flows.get(s, 0.0) for s in solutes])

        # Estimate phase compositions for activity coefficient models
        total_aq = F_aq + jnp.sum(F_in_arr)
        total_org = F_org + jnp.sum(F_solvent_arr)

        x_aq_est = {eq.aqueous_carrier: safe_divide(F_aq, total_aq)}
        x_org_est = {eq.organic_carrier: safe_divide(F_org, total_org)}
        for i, s in enumerate(solutes):
            x_aq_est[s] = safe_divide(F_in_arr[i], total_aq)
            x_org_est[s] = safe_divide(F_solvent_arr[i], total_org)

        # Get distribution coefficients with estimated compositions
        K_dict = eq.get_distribution_coefficients(x_aq_est, x_org_est, T)
        K_arr = jnp.array([K_dict[s] for s in solutes])

        # Total solute amounts
        F_total_arr = F_in_arr + F_solvent_arr

        # Equilibrium concentrations
        # At equilibrium: y = K * x and mass balance
        # F_aq * x + F_org * y = F_total
        # x = F_total / (F_aq + K * F_org)
        x_eq_arr = F_total_arr / (F_aq + K_arr * F_org)

        # Approach to equilibrium with multiple stages
        # Each stage achieves stage_efficiency fraction of remaining driving force
        # Total efficiency = 1 - (1 - eff)^N (differentiable in N)
        eff_per_stage = 0.8 if self.params.stage_efficiency is None else self.params.stage_efficiency
        total_eff = 1.0 - (1.0 - eff_per_stage) ** n_stages

        x_feed_arr = safe_divide(F_in_arr, F_aq)
        x_final_arr = x_feed_arr + total_eff * (x_eq_arr - x_feed_arr)

        F_raffinate_arr = x_final_arr * F_aq
        F_extracted_arr = F_total_arr - F_raffinate_arr

        # Convert back to dicts
        raffinate_flows = {eq.aqueous_carrier: F_aq}
        extract_flows = {eq.organic_carrier: F_org}
        for i, s in enumerate(solutes):
            raffinate_flows[s] = F_raffinate_arr[i]
            extract_flows[s] = F_extracted_arr[i]

        # Mutual solubility: transfer carrier amounts between phases (#157)
        carrier_transfer = {}
        if eq.mutual_solubility is not None:
            ms = eq.mutual_solubility
            aq_in_org = ms.get('aqueous_in_organic', 0.0)
            org_in_aq = ms.get('organic_in_aqueous', 0.0)
            aq_transferred = F_aq * aq_in_org
            org_transferred = F_org * org_in_aq
            raffinate_flows[eq.aqueous_carrier] = F_aq - aq_transferred
            raffinate_flows[eq.organic_carrier] = org_transferred
            extract_flows[eq.organic_carrier] = F_org - org_transferred
            extract_flows[eq.aqueous_carrier] = aq_transferred
            carrier_transfer = {
                'aqueous_to_organic': aq_transferred,
                'organic_to_aqueous': org_transferred,
            }

        # Stage profiles: compute aqueous (x) and organic (y) mole fractions
        # at each integer stage 1 .. N_int using the approach-to-equilibrium formula.
        # x_n = x_feed + total_eff_n * (x_eq - x_feed)
        # y_n = K * x_n  (organic in equilibrium with aqueous at stage n)
        # Initial feed-end compositions (stage 0, before any extraction):
        y_solvent_arr = safe_divide(F_solvent_arr, F_org)

        profiles_x = {s: [] for s in solutes}
        profiles_y = {s: [] for s in solutes}

        # Append stage-0 feed-end boundary
        for i, s in enumerate(solutes):
            profiles_x[s].append(x_feed_arr[i])
            profiles_y[s].append(y_solvent_arr[i])

        N_int, traced = _static_stage_count(n_stages, self.params.profile_stages)
        for n in range(1, N_int + 1):
            n_arr = jnp.asarray(float(n), dtype=jnp.float64)
            eff_n = 1.0 - (1.0 - eff_per_stage) ** n_arr
            x_n = _mask_rows(x_feed_arr + eff_n * (x_eq_arr - x_feed_arr), n, n_stages, traced)
            y_n = K_arr * x_n
            for i, s in enumerate(solutes):
                profiles_x[s].append(x_n[i])
                profiles_y[s].append(y_n[i])

        profiles = {
            "x": profiles_x,
            "y": profiles_y,
            "carrier_transfer": carrier_transfer,
            "mutual_solubility_active": eq.mutual_solubility is not None,
        }

        return raffinate_flows, extract_flows, profiles


# =============================================================================
# Differential Contactor
# =============================================================================

@dataclass(repr=False)
class ContactorParams(ParamsMixin):
    """Parameters for differential contactor.

    Attributes:
        length: Contactor length (m)
        area: Cross-sectional area (m^2)
        equilibrium: LLE equilibrium calculator
        n_segments: Number of discretization segments
        flow_config: 'counter_current' or 'co_current'
        mass_transfer_model: 'equilibrium' or 'rate_based'
        Kla: Overall volumetric mass-transfer coefficient based on the
             raffinate (aqueous) phase, K_x a / c (1/s); the rate per unit
             volume is ``Kla * aqueous_molar_density * (X - Y/K)``.
             Only used for rate_based model
        aqueous_molar_density: Molar density of the aqueous phase (mol/m^3)
             that turns ``Kla`` into a molar rate. Default 55500 (water).
             Only used for rate_based model
        HETP: Height equivalent to theoretical plate (m)
              Only used for equilibrium model
    """
    length: float | Array
    area: float | Array
    equilibrium: LLEEquilibrium
    n_segments: int = 50
    flow_config: Literal["counter_current", "co_current"] = "counter_current"
    mass_transfer_model: Literal["equilibrium", "rate_based"] = "equilibrium"
    Kla: float | Array | dict[str, float] = 0.01  # 1/s, per solute or global
    HETP: float | Array = 0.5  # m
    aqueous_molar_density: float | Array = 55500.0  # mol/m^3


class DifferentialContactor:
    """Differential (packed column) liquid-liquid extractor.

    Models a continuous contacting device like a packed column.
    Supports both equilibrium stage equivalent (HETP) and rate-based
    mass transfer models.
    """

    symbol = "LLE Contactor"
    equations = [
        r"F_{aq}\frac{dX_i}{dz} = -K_L a_i\,c_{aq} A\,(X_i - Y_i/K_i)\qquad \text{(rate-based)}",
        r"F_{org}\frac{dY_i}{dz} = \mp K_L a_i\,c_{aq} A\,(X_i - Y_i/K_i)\quad\text{(counter / co-current)}",
        r"H_{OR} = \frac{F_{aq}}{K_L a\,c_{aq} A},\qquad N_{OR} = \ell / H_{OR}",
        r"N_\mathrm{stages} = \frac{\ell}{\mathrm{HETP}}\qquad \text{(equilibrium model)}",
    ]
    assumptions = [
        "Plug flow of both phases along the contactor.",
        "Rate-based model uses a lumped overall K_L a per solute.",
        "Equilibrium model approximates the contactor as length/HETP stages.",
    ]
    references = [
        "Treybal, R.E. Mass Transfer Operations, 3e, McGraw-Hill, 1980.",
        "Seader, Henley, Roper. Separation Process Principles, 3e, Ch. 8.",
    ]
    parameter_symbols = {
        "length": r"\ell",
        "area": "A",
        "Kla": r"K_L a",
        "HETP": r"\mathrm{HETP}",
    }
    parameter_units = {
        "length": "m",
        "area": "m^2",
        "n_segments": "-",
        "Kla": "1/s",
        "HETP": "m",
        "aqueous_molar_density": "mol/m^3",
    }
    numerical_method = ("Equilibrium model: length/HETP Kremser stages. Rate-based model: exact "
                        "matrix-exponential solution of the linear transfer-unit ODEs, profiles "
                        "sampled at n_segments + 1 points.")

    def __init__(self, params: ContactorParams):
        """Initialize contactor.

        Args:
            params: Contactor parameters
        """
        self.params = params

    def __call__(
        self,
        feed: Stream,
        solvent: Stream,
        T: Array | float | None = None,
    ) -> tuple[Stream, Stream, dict]:
        """Perform extraction in differential contactor.

        Args:
            feed: Aqueous feed stream
            solvent: Organic solvent stream
            T: Operating temperature (K)

        Returns:
            raffinate: Aqueous outlet
            extract: Organic outlet
            info: Axial profiles and other information
        """
        p = self.params
        eq = p.equilibrium

        T = jnp.asarray(T) if T is not None else feed["T"]

        if p.mass_transfer_model == "equilibrium":
            return self._solve_equilibrium_model(feed, solvent, T)
        else:
            return self._solve_rate_based_model(feed, solvent, T)

    def _solve_equilibrium_model(
        self,
        feed: Stream,
        solvent: Stream,
        T: Array,
    ) -> tuple[Stream, Stream, dict]:
        """Solve using equilibrium stage equivalent (HETP) model.

        Number of theoretical stages = L / HETP
        Then use cascade equations.
        """
        p = self.params
        eq = p.equilibrium

        # Equivalent number of stages
        n_stages = p.length / p.HETP

        # Use cascade solver with continuous stages
        cascade_params = CascadeParams(
            n_stages=n_stages,
            equilibrium=eq,
            flow_config=p.flow_config,
        )
        cascade = MultistageCascade(cascade_params)

        raffinate, extract, cascade_info = cascade(feed, solvent, T)

        # Add height profile info
        info = {
            "n_stages_equivalent": n_stages,
            "HETP": p.HETP,
            "length": p.length,
            **cascade_info,
        }

        return raffinate, extract, info

    def _solve_rate_based_model(
        self,
        feed: Stream,
        solvent: Stream,
        T: Array,
    ) -> tuple[Stream, Stream, dict]:
        """Solve the rate-based (transfer-unit) model exactly.

        On the solute-free (ratio) basis, ``X`` = solute / aqueous carrier and
        ``Y`` = solute / organic carrier, the transfer rate per unit column
        volume is ``Kla * c_aq * (X - Y/K)`` (``c_aq`` = ``aqueous_molar_density``),
        so with ``alpha = Kla * c_aq * area`` (mol/(m s)):

            F_aq  dX/dz = -alpha (X - Y/K)
            F_org dY/dz = -alpha (X - Y/K)   (counter-current, organic flows -z)
            F_org dY/dz = +alpha (X - Y/K)   (co-current)

        With constant ``K`` the system is linear, so it is solved with a
        matrix exponential (counter-current: the two-point boundary problem
        ``X(0) = X_0``, ``Y(L) = Y_in``). The height of a raffinate-phase
        overall transfer unit is ``H_OR = F_aq / alpha`` and ``N_OR = L / H_OR``
        (Treybal, Mass Transfer Operations 3e, Ch. 10).
        """
        p = self.params
        eq = p.equilibrium

        feed_flows = get_flows(feed)
        solvent_flows = get_flows(solvent)
        F_aq = feed_flows[eq.aqueous_carrier]
        F_org = solvent_flows[eq.organic_carrier]
        K_dict = eq.get_distribution_coefficients({}, {}, T)
        solutes = eq.solutes
        n_seg = p.n_segments

        X0 = jnp.array([feed_flows.get(s, 0.0) / F_aq for s in solutes])
        Y_in = jnp.array([solvent_flows.get(s, 0.0) / F_org for s in solutes])
        K_arr = jnp.array([K_dict[s] for s in solutes])
        if isinstance(p.Kla, dict):
            Kla_arr = jnp.array([p.Kla[s] for s in solutes])
        else:
            Kla_arr = jnp.full(len(solutes), 1.0) * p.Kla

        alpha = Kla_arr * p.aqueous_molar_density * p.area
        a = alpha / F_aq
        b = alpha / F_org
        counter = p.flow_config == "counter_current"
        sgn = -1.0 if counter else 1.0
        row0 = jnp.stack([-a, a / K_arr], axis=-1)
        row1 = jnp.stack([sgn * b, -sgn * b / K_arr], axis=-1)
        A_mats = jnp.stack([row0, row1], axis=1)  # (n_solutes, 2, 2)

        from jax.scipy.linalg import expm

        L = p.length
        M = jax.vmap(lambda A: expm(A * L))(A_mats)
        if counter:
            Y0 = (Y_in - M[:, 1, 0] * X0) / M[:, 1, 1]
        else:
            Y0 = Y_in

        z = jnp.linspace(0.0, 1.0, n_seg + 1) * L
        Mz = jax.vmap(lambda zz: jax.vmap(lambda A: expm(A * zz))(A_mats))(z)
        X_prof = Mz[:, :, 0, 0] * X0 + Mz[:, :, 0, 1] * Y0
        Y_prof = Mz[:, :, 1, 0] * X0 + Mz[:, :, 1, 1] * Y0
        X_L = M[:, 0, 0] * X0 + M[:, 0, 1] * Y0
        Y_out = Y0 if counter else M[:, 1, 0] * X0 + M[:, 1, 1] * Y0

        F_raff = X_L * F_aq
        F_in = X0 * F_aq
        F_sin = Y_in * F_org
        F_ext = F_in + F_sin - F_raff  # exact for the linear model; closes the balance

        raffinate_flows = {s: F_raff[i] for i, s in enumerate(solutes)}
        extract_flows = {s: F_ext[i] for i, s in enumerate(solutes)}
        raffinate_flows[eq.aqueous_carrier] = F_aq
        extract_flows[eq.organic_carrier] = F_org

        P = feed["P"]
        raffinate = make_stream(raffinate_flows, T, P)
        extract = make_stream(extract_flows, T, P)

        HTU = F_aq / alpha
        info = {
            "length": p.length,
            "n_segments": n_seg,
            "HTU_OR": {s: HTU[i] for i, s in enumerate(solutes)},
            "NTU_OR": {s: L / HTU[i] for i, s in enumerate(solutes)},
            "extraction_factor": {s: (K_arr * F_org / F_aq)[i] for i, s in enumerate(solutes)},
            "Y_out": {s: Y_out[i] for i, s in enumerate(solutes)},
            "profiles": {
                "z": z,
                "c_aq": {s: X_prof[:, i] for i, s in enumerate(solutes)},
                "c_org": {s: Y_prof[:, i] for i, s in enumerate(solutes)},
            },
            "T": T,
        }
        return raffinate, extract, info


# =============================================================================
# Convenience Functions
# =============================================================================

def separation_factor(K1: Array, K2: Array) -> Array:
    """Calculate separation factor between two solutes.

    SF = K1 / K2

    Higher SF means easier separation.
    """
    return K1 / K2


def minimum_solvent_ratio(K: Array, recovery: float = 0.99) -> Array:
    """Calculate minimum solvent-to-feed ratio for desired recovery.

    For counter-current extraction with immiscible carriers, a solute-free
    solvent and infinite stages, the pinch is at the feed end:
    (S/F)_min = recovery / K

    ``K`` must be on the same basis as the flows: with ``K = Y/X`` on the
    solute-free (ratio) basis, ``S`` and ``F`` are the organic and aqueous
    *carrier* flows (the basis of :class:`MultistageCascade`). A tabulated
    concentration-basis ``K_c`` (mol/L over mol/L) goes with volumetric flows.

    Args:
        K: Distribution coefficient (organic over aqueous)
        recovery: Desired fraction of solute recovered

    Returns:
        Minimum solvent-to-feed (carrier) ratio
    """
    return recovery / K


def stages_for_recovery(
    K: Array,
    SF_ratio: Array,
    recovery: float = 0.99,
) -> Array:
    """Calculate stages needed for desired recovery.

    Using Kremser equation for counter-current extraction.

    Convention: K = C_extract / C_raffinate (K > 1 favors organic phase).
    Extraction factor E = K * (S/F), with ``S/F`` on the same basis as ``K``
    (ratio-basis ``K`` with carrier flows, as in :class:`MultistageCascade`;
    concentration-basis ``K`` with volumetric flows). Solute-free solvent.
    The result is real (continuous); round it up for a design.

    Args:
        K: Distribution coefficient (C_org / C_aq)
        SF_ratio: Actual solvent-to-feed molar flow ratio (F_org / F_aq).
                  Must be greater than minimum_solvent_ratio(K, recovery).
        recovery: Desired solute recovery

    Returns:
        Number of theoretical stages needed
    """
    E = K * SF_ratio  # Extraction factor = K * (S/F)

    # From Kremser equation for counter-current extraction:
    #   fraction_remaining = (E - 1) / (E^(N+1) - 1)
    #   recovery = 1 - fraction_remaining
    # Solving for N:
    #   E^(N+1) = (E - recovery) / (1 - recovery)
    #   N = ln((E - recovery) / (1 - recovery)) / ln(E) - 1
    N = jnp.log((E - recovery) / (1.0 - recovery)) / jnp.log(E) - 1.0

    return jnp.maximum(N, 1.0)
