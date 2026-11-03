"""Pitzer activity model for seawater and its brines (JAX, differentiable).

The equations are those of Harvie & Weare (1980) and Harvie, Moller & Weare
(1984) as written in PHRQPITZ (Plummer, Parkhurst, Fleming & Dunkle 1988,
USGS WRIR 88-4153 [C6]); every equation below was read on the report's
pages 2-5 (Eqs. 1-21) and in its Fortran listing (E-theta, Chebyshev
coefficients, dielectric constant). Content verified.

- Osmotic coefficient and ion activity coefficients: [C6] Eqs. 1-3, 6-17.
- Unsymmetrical mixing ``E-theta``, ``E-theta'``: [C6] Eqs. 18-20 with the
  Chebyshev approximation to J0 and J1 of Harvie (coefficients ``AK`` and
  subroutines ``ETHETA``, ``JAY``, ``BDK`` of the PHRQPITZ listing).
- Neutral species: [C6] Eq. 21, plus the ``zeta`` (neutral-cation-anion)
  term PHREEQC adds (Clegg & Whitfield). The neutral species' effect on the
  ions' activity coefficients and on the osmotic coefficient is not
  included: CO2(aq) and MgCO3(aq) are micromolal in seawater, so those terms
  are below 1e-5 in ln gamma.
- ``A_phi``: the defining [C6] Eq. 4 with CODATA constants, the dielectric
  constant ``D`` of Bradley & Pitzer (1979) at 1 atm (function ``DC`` of the
  PHRQPITZ listing) and the pure-water density of Sharqawy et al. 2010 [S1].
  (The printed Eq. 5 constant is inconsistent with Eq. 4; see :func:`a_phi`.)
- Parameter temperature dependence: the six-term form of PHREEQC's PITZER
  keyword (PHREEQC v3 manual [C4], p. 134).

The parameters (``data/pitzer_seawater.py``) are transcribed mechanically
from PHREEQC's ``pitzer.dat``: second-hand, cross-checked against PHREEQC.

Molality units throughout (mol/kg water). Activity coefficients are on the
unscaled (not MacInnes) convention, as PHREEQC computes them.
"""

from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow_water.chemistry import seawater as sw
from difflow_water.chemistry.data import pitzer_seawater as db

T_REF = 298.15
MW_H2O = 0.01801528  # kg/mol
B_PITZER = 1.2

CATIONS = list(db.CATIONS)
ANIONS = list(db.ANIONS)
NEUTRALS = list(db.NEUTRALS)
SPECIES = CATIONS + ANIONS + NEUTRALS
Z_C = np.array([db.CATIONS[s] for s in CATIONS], float)
Z_A = np.array([abs(db.ANIONS[s]) for s in ANIONS], float)
NC, NA, NN = len(CATIONS), len(ANIONS), len(NEUTRALS)

# PHRQPITZ listing, DATA AK (Chebyshev coefficients for J0/J1), [C6]
_AK = np.array([
    [1.925154014814667, -0.060076477753119, -0.029779077456514, -0.007299499690937,
     0.000388260636404, 0.000636874599598, 0.000036583601823, -0.000045036975204,
     -0.000004537895710, 0.000002937706971, 0.000000396566462, -0.000000202099617,
     -0.000000025267769, 0.000000013522610, 0.000000001229405, -0.000000000821969,
     -0.000000000050847, 0.000000000046333, 0.000000000001943, -0.000000000002563,
     -0.000000000010991],
    [0.628023320520852, 0.462762985338493, 0.150044637187895, -0.028796057604906,
     -0.036552745910311, -0.001668087945272, 0.006519840398744, 0.001130378079086,
     -0.000887171310131, -0.000242107641309, 0.000087294451594, 0.000034682122751,
     -0.000004583768938, -0.000003548684306, -0.000000250453880, 0.000000216991779,
     0.000000080779570, 0.000000004558555, -0.000000006944757, -0.000000002849257,
     0.000000000237816],
])

# Bradley & Pitzer (1979) dielectric constant, PHRQPITZ function DC, [C6]
_U = (3.4279e2, -5.0866e-3, 9.4690e-7, -2.0525, 3.1159e3, -1.8289e2, -8.0325e3, 4.2142e6, 2.1417)


def dielectric_constant(T, P_bar=1.01325) -> Array:
    """Relative dielectric constant of water, Bradley & Pitzer (1979) [C6]."""
    T = jnp.asarray(T, dtype=jnp.float64)
    u = _U
    d1000 = u[0] * jnp.exp(u[1] * T + u[2] * T * T)
    c = u[3] + u[4] / (u[5] + T)
    b = u[6] + u[7] / T + u[8] * T
    return d1000 + c * jnp.log((b + P_bar) / (b + 1000.0))


# CODATA 2018 exact/recommended constants (SI)
_N_A = 6.02214076e23
_E = 1.602176634e-19
_K_B = 1.380649e-23
_EPS0 = 8.8541878128e-12


def a_phi(T) -> Array:
    """Debye-Hückel osmotic slope ``A_phi`` (kg^0.5 mol^-0.5).

    The defining equation, [C6] Eq. 4, in SI::

        A_phi = (1/3) sqrt(2 pi N_A rho_w) (e^2 / (4 pi eps0 D k T))^(3/2)

    with ``D`` from :func:`dielectric_constant` and ``rho_w`` the pure-water
    density of [S1]. It gives 0.39147 at 25 C, the value [C6] reports for its
    Eq. 5. Eq. 5 as printed (and coded in the PHRQPITZ listing),
    ``1400684 (rho_w/(D T))^1.5``, gives 0.3904 with the same D and rho_w,
    so it is not used. PHREEQC's results match 0.3915 (benchmarks/phreeqc).
    """
    T = jnp.asarray(T, dtype=jnp.float64)
    rho = sw.density_water(T)  # kg/m^3
    lb = _E**2 / (4.0 * jnp.pi * _EPS0 * dielectric_constant(T) * _K_B * T)  # m
    return jnp.sqrt(2.0 * jnp.pi * _N_A * rho) * lb**1.5 / 3.0


def _temperature_form(co, T):
    A0, A1, A2, A3, A4, A5 = co
    Tr = T_REF
    return (A0 + A1 * (1.0 / T - 1.0 / Tr) + A2 * jnp.log(T / Tr) + A3 * (T - Tr)
            + A4 * (T * T - Tr * Tr) + A5 * (1.0 / (T * T) - 1.0 / (Tr * Tr)))


@lru_cache(maxsize=1)
def _index_tables():
    """Parameter coefficient arrays (numpy), indexed by species position."""
    ci = {s: i for i, s in enumerate(CATIONS)}
    ai = {s: i for i, s in enumerate(ANIONS)}
    ni = {s: i for i, s in enumerate(NEUTRALS)}
    z6 = (0.0,) * 6
    tab = {
        "B0": np.zeros((NC, NA, 6)), "B1": np.zeros((NC, NA, 6)), "B2": np.zeros((NC, NA, 6)),
        "C0": np.zeros((NC, NA, 6)), "THC": np.zeros((NC, NC, 6)), "THA": np.zeros((NA, NA, 6)),
        "PSIC": np.zeros((NC, NC, NA, 6)), "PSIA": np.zeros((NA, NA, NC, 6)),
        "LNC": np.zeros((NN, NC, 6)), "LNA": np.zeros((NN, NA, 6)), "ZETA": np.zeros((NN, NC, NA, 6)),
    }
    del z6
    for kind, sp, co in db.PARAMS:
        co = np.array(co)
        if kind in ("B0", "B1", "B2", "C0"):
            c = next(s for s in sp if s in ci)
            a = next(s for s in sp if s in ai)
            tab[kind][ci[c], ai[a]] = co
        elif kind == "THETA":
            if sp[0] in ci:
                i, j = ci[sp[0]], ci[sp[1]]
                tab["THC"][i, j] = tab["THC"][j, i] = co
            else:
                i, j = ai[sp[0]], ai[sp[1]]
                tab["THA"][i, j] = tab["THA"][j, i] = co
        elif kind == "PSI":
            cs = [s for s in sp if s in ci]
            as_ = [s for s in sp if s in ai]
            if len(cs) == 2:
                i, j, k = ci[cs[0]], ci[cs[1]], ai[as_[0]]
                tab["PSIC"][i, j, k] = tab["PSIC"][j, i, k] = co
            else:
                i, j, k = ai[as_[0]], ai[as_[1]], ci[cs[0]]
                tab["PSIA"][i, j, k] = tab["PSIA"][j, i, k] = co
        elif kind == "LAMDA":
            ns = [s for s in sp if s in ni]
            other = [s for s in sp if s not in ni]
            if not other:  # neutral-neutral (CO2-CO2): micromolal, omitted
                continue
            n, o = ni[ns[0]], other[0]
            if o in ci:
                tab["LNC"][n, ci[o]] = co
            else:
                tab["LNA"][n, ai[o]] = co
        elif kind == "ZETA":
            n = ni[next(s for s in sp if s in ni)]
            tab["ZETA"][n, ci[next(s for s in sp if s in ci)], ai[next(s for s in sp if s in ai)]] = co
    # alphas, [C6] Eqs. 8-15: 2-2 pairs a1 = 1.4, a2 = 12; otherwise a1 = 2, a2 = 12
    zz = np.outer(Z_C, Z_A)
    alpha1 = np.where(zz >= 4, 1.4, 2.0)
    alpha2 = np.full((NC, NA), 12.0)
    return tab, alpha1, alpha2


def parameters(T) -> dict:
    """Pitzer parameters at temperature T (K), as jnp arrays."""
    T = jnp.asarray(T, dtype=jnp.float64)
    tab, a1, a2 = _index_tables()
    out = {k: _temperature_form(np.moveaxis(v, -1, 0), T) for k, v in tab.items()}
    out["alpha1"] = jnp.asarray(a1)
    out["alpha2"] = jnp.asarray(a2)
    return out


def _bdk(x):
    """Chebyshev sums BK(0), BK(2), DK(0), DK(2) for J0/J1 (PHRQPITZ ``BDK``).

    The downward recursion ``BK(M) = Z BK(M+1) - BK(M+2) + AK(M)``,
    ``DK(M) = BK(M+1) + Z DK(M+1) - DK(M+2)``, M = 20..0, as a ``lax.scan``.
    """
    small = x <= 1.0
    xs = jnp.maximum(x, 1e-300)
    z = jnp.where(small, 4.0 * xs ** 0.2 - 2.0, 40.0 * xs ** (-0.1) / 9.0 - 22.0 / 9.0)
    ak = jnp.where(small[..., None], jnp.asarray(_AK[0]), jnp.asarray(_AK[1]))  # (..., 21)
    ak_desc = jnp.moveaxis(ak[..., ::-1], -1, 0)  # M = 20..0 along axis 0

    def body(carry, akm):
        bk1, bk2, dk1, dk2 = carry
        b0 = z * bk1 - bk2 + akm
        d0 = bk1 + z * dk1 - dk2
        return (b0, bk1, d0, dk1), (b0, d0)

    zero = jnp.zeros_like(x)
    _, (bk, dk) = jax.lax.scan(body, (zero, zero, zero, zero), ak_desc)
    # bk[-1] is BK(0), bk[-3] is BK(2) (the scan runs M = 20 down to 0)
    return bk[-1], bk[-3], dk[-1], dk[-3]


def _jay_jprime(x):
    """J0 (``JAY``) and J1 (``JPRIME``) of the PHRQPITZ listing, from one BDK pass."""
    b0, b2, d0, d2 = _bdk(x)
    xs = jnp.maximum(x, 1e-300)
    dz = jnp.where(x <= 1.0, 0.8 * xs ** (-0.8), -4.0 * xs ** (-1.1) / 9.0)
    return x / 4.0 - 1.0 + 0.5 * (b0 - b2), x * (0.25 + dz * (d0 - d2) / 2.0)


def _etheta(zj, zk, I, aphi):
    """E-theta and E-theta' for a pair of charges ([C6] listing, Eqs. A1-A3)."""
    sI = jnp.sqrt(I)
    xcon = 6.0 * aphi * sI
    zz = zj * zk
    x = jnp.stack(jnp.broadcast_arrays(xcon * zz, xcon * zj * zj, xcon * zk * zk))
    j, jp = _jay_jprime(x)
    eth = zz * (j[0] - j[1] / 2.0 - j[2] / 2.0) / (4.0 * I)
    ethp = zz * (jp[0] - jp[1] / 2.0 - jp[2] / 2.0) / (8.0 * I * I) - eth / I
    same = jnp.abs(zj - zk) < 1e-12
    return jnp.where(same, 0.0, eth), jnp.where(same, 0.0, ethp)


def _g(x):
    x = jnp.maximum(x, 1e-12)
    return 2.0 * (1.0 - (1.0 + x) * jnp.exp(-x)) / (x * x)


def _gp(x):
    x = jnp.maximum(x, 1e-12)
    return -2.0 * (1.0 - (1.0 + x + 0.5 * x * x) * jnp.exp(-x)) / (x * x)


def activity(mc, ma, mn, T, params=None) -> dict:
    """Pitzer activity coefficients, osmotic coefficient and water activity.

    Args:
        mc: Cation molalities, in :data:`CATIONS` order (mol/kg water).
        ma: Anion molalities, :data:`ANIONS` order.
        mn: Neutral molalities, :data:`NEUTRALS` order.
        T: Temperature (K).
        params: Output of :func:`parameters` (computed when None).

    Returns:
        dict with ``ln_gamma_c``, ``ln_gamma_a``, ``ln_gamma_n``, ``phi``
        (osmotic coefficient), ``ln_aw`` (water activity), ``I``.
    """
    p = parameters(T) if params is None else params
    mc, ma, mn = (jnp.asarray(v, dtype=jnp.float64) for v in (mc, ma, mn))
    zc, za = jnp.asarray(Z_C), jnp.asarray(Z_A)
    I = 0.5 * (jnp.sum(mc * zc**2) + jnp.sum(ma * za**2))
    I = jnp.maximum(I, 1e-14)
    sI = jnp.sqrt(I)
    Z = jnp.sum(mc * zc) + jnp.sum(ma * za)
    aphi = a_phi(T)
    x1, x2 = p["alpha1"] * sI, p["alpha2"] * sI
    B = p["B0"] + p["B1"] * _g(x1) + p["B2"] * _g(x2)
    Bp = (p["B1"] * _gp(x1) + p["B2"] * _gp(x2)) / I
    Bphi = p["B0"] + p["B1"] * jnp.exp(-x1) + p["B2"] * jnp.exp(-x2)
    C = p["C0"] / (2.0 * jnp.sqrt(jnp.outer(zc, za)))
    # mixing terms, Eqs. 18-20
    ethc, ethpc = _etheta(zc[:, None], zc[None, :], I, aphi)
    etha, ethpa = _etheta(za[:, None], za[None, :], I, aphi)
    Phic, Phipc = p["THC"] + ethc, ethpc
    Phia, Phipa = p["THA"] + etha, ethpa
    Phiphic = p["THC"] + ethc + I * ethpc
    Phiphia = p["THA"] + etha + I * ethpa
    offc = 1.0 - jnp.eye(NC)
    offa = 1.0 - jnp.eye(NA)
    # Eq. 7
    F = (-aphi * (sI / (1.0 + B_PITZER * sI) + 2.0 / B_PITZER * jnp.log1p(B_PITZER * sI))
         + jnp.einsum("c,a,ca->", mc, ma, Bp)
         + 0.5 * jnp.einsum("c,d,cd->", mc, mc, Phipc * offc)
         + 0.5 * jnp.einsum("a,b,ab->", ma, ma, Phipa * offa))
    mcma_C = jnp.einsum("c,a,ca->", mc, ma, C)
    # Eq. 2, cations
    ln_gc = (zc**2 * F + jnp.einsum("a,ca->c", ma, 2.0 * B + Z * C)
             + jnp.einsum("d,cd->c", mc, 2.0 * Phic * offc)
             + jnp.einsum("d,a,cda->c", mc, ma, p["PSIC"])
             + 0.5 * jnp.einsum("a,b,abc->c", ma, ma, p["PSIA"] * offa[:, :, None])
             + zc * mcma_C)
    # Eq. 3, anions
    ln_ga = (za**2 * F + jnp.einsum("c,ca->a", mc, 2.0 * B + Z * C)
             + jnp.einsum("b,ab->a", ma, 2.0 * Phia * offa)
             + jnp.einsum("b,c,abc->a", ma, mc, p["PSIA"])
             + 0.5 * jnp.einsum("c,d,cda->a", mc, mc, p["PSIC"] * offc[:, :, None])
             + za * mcma_C)
    # Eq. 21 (+ zeta)
    ln_gn = (2.0 * jnp.einsum("c,nc->n", mc, p["LNC"]) + 2.0 * jnp.einsum("a,na->n", ma, p["LNA"])
             + jnp.einsum("c,a,nca->n", mc, ma, p["ZETA"]))
    # Eq. 1, osmotic coefficient
    msum = jnp.sum(mc) + jnp.sum(ma) + jnp.sum(mn)
    phi_m1 = (2.0 / msum) * (
        -aphi * I * sI / (1.0 + B_PITZER * sI)
        + jnp.einsum("c,a,ca->", mc, ma, Bphi + Z * C)
        + 0.5 * jnp.einsum("c,d,cd->", mc, mc,
                           (Phiphic + jnp.einsum("a,cda->cd", ma, p["PSIC"])) * offc)
        + 0.5 * jnp.einsum("a,b,ab->", ma, ma,
                           (Phiphia + jnp.einsum("c,abc->ab", mc, p["PSIA"])) * offa))
    phi = 1.0 + phi_m1
    return {"ln_gamma_c": ln_gc, "ln_gamma_a": ln_ga, "ln_gamma_n": ln_gn,
            "phi": phi, "ln_aw": -phi * msum * MW_H2O, "I": I}
