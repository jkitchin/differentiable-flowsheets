r"""Steam-methane-reforming hydrogen plant (#473). A library.

Natural gas -> ZnO guard -> steam addition at ``S/C`` -> adiabatic
prereformer -> reformer tubes -> process-gas boiler -> HT shift (-> LT
shift) -> condensate knock-out -> purification PSA, with the PSA tail gas
and a supplementary natural-gas fuel fired in the reformer furnace::

    smr_plant(feed, SMRParams()) -> dict

* **Desulfurisation** (:func:`desulfurize`): H2S removed at a given fraction
  (a ZnO guard bed); the slip rides through as an inert and leaves in the
  PSA tail gas.
* **Prereformer** (:func:`prereform`): adiabatic Gibbs equilibrium
  (:class:`difflow.GibbsReactor`) over every species: the C2+ go to CH4/CO/
  CO2/H2, methanation and shift equilibrate. Carbon formation is not
  modelled.
* **Reformer tubes** (:func:`reformer_tube`): a 1-D pseudo-homogeneous
  packed tube, Xu & Froment (1989) intrinsic kinetics (:func:`xu_froment_rates`)
  times effectiveness factors, integrated in axial position by diffrax.
  Heat input either a specified flux profile whose total is solved so the
  outlet reaches ``T_out`` (``mode="flux"``), or a wall-temperature profile
  with an overall coefficient (``mode="wall"``). The approach to the
  reforming equilibrium is reported.
* **Shift** (:func:`shift`): adiabatic equilibrium restricted to
  ``CO + H2O = CO2 + H2``, with an approach to equilibrium.
* **Knock-out** (:func:`knock_out`): water condensed to saturation
  (IAPWS-IF97).
* **PSA**: :class:`~difflow_refinery.hydrogen.psa.HydrogenPSA`, numbered up
  from the reference bed (one bed per :data:`PSA_FEED_PER_BED` of feed).
* **Furnace** (:func:`furnace`): tail gas + supplementary fuel burnt with
  air at a flue O2; the radiant duty is the fuel's heat release down to the
  bridge-wall temperature, and the supplementary fuel is what closes it
  against the reformer duty (exact: the balance is affine in the fuel).

All enthalpies are on one formation basis (:mod:`difflow_refinery.thermochemistry`).
The reformer and everything after it use the 298-1800 K ``"wide"`` Cp fits;
the prereformer, which also carries the C2+, uses the table fits (valid to
1000 K).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import diffrax
import jax
import jax.numpy as jnp
from jax import lax

from difflow import GibbsReactor, GibbsReactorParams, make_stream
from difflow.water import water_latent_heat, water_saturation_pressure
from difflow_refinery import thermochemistry as tc
from difflow_refinery.hydrogen.psa import HydrogenPSA, HydrogenPSAParams
from difflow_refinery.sulfur.vapor import bracket_root

CH4, H2O, H2, CO, CO2, N2, H2S = ("methane", "water", "hydrogen", "carbon_monoxide",
                                  "carbon_dioxide", "nitrogen", "hydrogen_sulfide")
#: hydrocarbons a natural-gas feed may carry (C2+ go to the prereformer)
HYDROCARBONS = (CH4, "ethane", "propane", "n_butane")
#: every species of the feed train
SPECIES = (*HYDROCARBONS, H2O, H2, CO, CO2, N2, H2S)
#: the species after the prereformer (C2+ are gone)
SYNGAS = (CH4, H2O, H2, CO, CO2, N2, H2S)
FEED_GAS = tc.IdealGasSet(list(SPECIES))                       # table Cp, <= 1000 K
GAS = tc.IdealGasSet(list(SYNGAS), cp_basis="wide")            # 298-1800 K
_I = {n: i for i, n in enumerate(SYNGAS)}
_IF = {n: i for i, n in enumerate(SPECIES)}
#: carbon atoms per molecule, in :data:`SPECIES` order
C_ATOMS = jnp.asarray(FEED_GAS.ELEMENTS["C"])

#: stoichiometry of the three Xu-Froment reactions over :data:`SYNGAS`:
#: (1) CH4 + H2O = CO + 3 H2, (2) CO + H2O = CO2 + H2, (3) CH4 + 2 H2O = CO2 + 4 H2
REACTIONS = (
    {CH4: -1.0, H2O: -1.0, CO: 1.0, H2: 3.0},
    {CO: -1.0, H2O: -1.0, CO2: 1.0, H2: 1.0},
    {CH4: -1.0, H2O: -2.0, CO2: 1.0, H2: 4.0},
)
NU = jnp.asarray([[r.get(n, 0.0) for n in SYNGAS] for r in REACTIONS])
WGS = REACTIONS[1]

R = 8.314462618          # J/mol/K (CODATA 2018 exact)
P_BAR = 1.0e5

# ------------------------------------------------------------------ kinetics
#: Xu J., Froment G.F. (1989) "Methane steam reforming, methanation and
#: water-gas shift: I. Intrinsic kinetics", AIChE J. 35(1):88-96,
#: doi:10.1002/aic.690350109: pre-exponential factors from Table 6
#: ("reference activity"), activation energies and adsorption enthalpies
#: from Table 5 (rate form eq 3, temperature form eqs 6-7). ``A`` in
#: kmol bar^0.5/(kg_cat h) for k1, k3 and kmol/(kg_cat h bar) for k2; ``E``
#: in J/mol; adsorption ``A`` in bar^-1 (K_H2O dimensionless), ``dH`` in
#: J/mol. The paper (p. 94) multiplies the rate coefficients by 2.246 for
#: FRESH catalyst (k1 -> 9.49e15); these are the reference (aged) values.
#: The paper's data span 773-848 K (reforming) and 3-15 bar: a reformer
#: tube at 900-1200 K and 30 bar is an extrapolation, as in Part II.
XU_FROMENT = {
    "k1": (4.225e15, 240.1e3),
    "k2": (1.955e6, 67.13e3),
    "k3": (1.020e15, 243.9e3),
    "K_CO": (8.23e-5, -70.65e3),
    "K_H2": (6.12e-9, -82.90e3),
    "K_CH4": (6.65e-4, -38.28e3),
    "K_H2O": (1.77e5, 88.68e3),
}
XU_FROMENT_SOURCE = "Xu & Froment 1989, AIChE J. 35:88, doi:10.1002/aic.690350109, Tables 5-6"
_KMOL_H = 1000.0 / 3600.0      # kmol/h -> mol/s


def _arr(A, E, T):
    return A * jnp.exp(-E / (R * T))


def xu_froment_equilibrium(T):
    """The equilibrium-constant expressions used with the Xu-Froment rates:
    ``K1`` (bar^2), ``K2`` (-), ``K3 = K1 K2`` (bar^2).

    Part I gives no expressions for them; these are Hou & Hughes (2001),
    Chem. Eng. J. 82:311-328, doi:10.1016/S1385-8947(00)00367-3, Table 3
    (citing Xu & Froment), converted from kPa^2 to bar^2:
    ``K1 = 1.198e13 exp(-26830/T)``, ``K2 = 1.767e-2 exp(4400/T)``, and
    their ``K3 = 2.117e11 exp(-22430/T)`` is ``K1 K2``. The rates use
    :func:`equilibrium_constants` (the formation table) instead; the tests
    hold the two within 11% in K (700-1200 K)."""
    T = jnp.asarray(T, dtype=float)
    K1 = jnp.exp(-26830.0 / T + 30.114)
    K2 = jnp.exp(4400.0 / T - 4.036)
    return jnp.stack([K1, K2, K1 * K2], axis=-1)


def equilibrium_constants(T):
    """``K1, K2, K3`` (pressures in bar) from the formation table
    (:mod:`difflow_refinery.thermochemistry`, the same basis as the
    :class:`~difflow.GibbsReactor`), so the kinetic limit IS the Gibbs
    equilibrium."""
    G = GAS.gibbs(jnp.asarray(T, dtype=float))
    dG = NU @ G
    return jnp.exp(-dG / (R * jnp.asarray(T, dtype=float)))


def xu_froment_rates(p, T, K=None):
    """Intrinsic rates (mol/(kg_cat s)) of the three reactions.

    Args:
        p: Partial pressures (bar) over :data:`SYNGAS`.
        T: Temperature (K).
        K: Equilibrium constants ``(3,)``; default :func:`equilibrium_constants`.

    .. math::

        r_1 = \frac{k_1}{p_{H_2}^{2.5}}\left(p_{CH_4}p_{H_2O} - \frac{p_{H_2}^3 p_{CO}}{K_1}\right)/D^2,\quad
        r_2 = \frac{k_2}{p_{H_2}}\left(p_{CO}p_{H_2O} - \frac{p_{H_2} p_{CO_2}}{K_2}\right)/D^2,

        r_3 = \frac{k_3}{p_{H_2}^{3.5}}\left(p_{CH_4}p_{H_2O}^2 - \frac{p_{H_2}^4 p_{CO_2}}{K_3}\right)/D^2,\quad
        D = 1 + K_{CO}p_{CO} + K_{H_2}p_{H_2} + K_{CH_4}p_{CH_4} + K_{H_2O}p_{H_2O}/p_{H_2}
    """
    K = equilibrium_constants(T) if K is None else K
    c = {k: _arr(*v, T) for k, v in XU_FROMENT.items()}
    pCH4, pH2O, pCO, pCO2 = p[_I[CH4]], p[_I[H2O]], p[_I[CO]], p[_I[CO2]]
    pH2 = jnp.maximum(p[_I[H2]], 1e-8)
    D = 1.0 + c["K_CO"] * pCO + c["K_H2"] * pH2 + c["K_CH4"] * pCH4 + c["K_H2O"] * pH2O / pH2
    r1 = c["k1"] / pH2**2.5 * (pCH4 * pH2O - pH2**3 * pCO / K[0]) / D**2
    r2 = c["k2"] / pH2 * (pCO * pH2O - pH2 * pCO2 / K[1]) / D**2
    r3 = c["k3"] / pH2**3.5 * (pCH4 * pH2O**2 - pH2**4 * pCO2 / K[2]) / D**2
    return jnp.stack([r1, r2, r3]) * _KMOL_H


# ------------------------------------------------------------------ helpers
def to_vec(stream, names=SYNGAS):
    return jnp.stack([jnp.asarray(stream.get(f"F_{n}", 0.0), dtype=float) for n in names])


def to_stream(n, T, P, names=SYNGAS):
    return make_stream(dict(zip(names, n)), T, P)


def enthalpy(n, T, gas=GAS):
    return jnp.sum(n * gas.enthalpy(T))


def lhv(names):
    """Lower heating value (J/mol) of each species at 298.15 K: the enthalpy
    of combustion to CO2(g), H2O(g), N2 and SO2 on the formation table."""
    tab = tc.IdealGasSet(list(names))
    E = tab.ELEMENTS
    h = {k: tc.Hf(k) for k in (CO2, H2O, "sulfur_dioxide", "oxygen")}
    prod = E["C"] * h[CO2] + E["H"] / 2 * h[H2O] + E["S"] * h["sulfur_dioxide"]
    return jnp.asarray(tab.HF - prod)


def approach_to_equilibrium(n, T, P):
    """``T - T_eq`` (K) of steam reforming (reaction 1): ``T_eq`` is where
    ``K1`` equals the outlet's reaction quotient. Positive when the gas is
    short of equilibrium (an industrial 'approach')."""
    y = n / jnp.sum(n)
    p = y * P / P_BAR
    lnQ = 3 * jnp.log(p[_I[H2]]) + jnp.log(p[_I[CO]]) - jnp.log(p[_I[CH4]]) - jnp.log(p[_I[H2O]])

    def f(Te, lq):
        return jnp.log(equilibrium_constants(Te)[0]) - lq

    return T - bracket_root(f, 500.0, 1700.0, lnQ, iters=60)


# ------------------------------------------------------------------ front end
def desulfurize(feed, removal: float = 1.0):
    """ZnO guard: ``removal`` of the H2S taken out. Returns ``(stream, removed mol/s)``."""
    h2s = jnp.asarray(feed.get(f"F_{H2S}", 0.0), dtype=float)
    out = dict(feed)
    out[f"F_{H2S}"] = h2s * (1.0 - removal)
    return out, h2s * removal


def carbon_flow(stream):
    """Hydrocarbon carbon (mol/s) of a stream: the C of ``S/C``."""
    return sum(tc.elements(n).get("C", 0) * jnp.asarray(stream.get(f"F_{n}", 0.0), dtype=float)
               for n in HYDROCARBONS)


def prereform(feed, T_in, pressure_drop=0.0):
    """Adiabatic prereformer at inlet ``T_in`` (unrestricted Gibbs over
    :data:`SPECIES`). Returns ``(outlet over SYNGAS, info)``; ``info``
    adds ``C2plus`` (C2+ left at equilibrium, mol/s: a few ppm, carried on
    in the outlet as its element-equivalent CH4 and H2)."""
    n = to_vec(feed, SPECIES)
    P = jnp.asarray(feed["P"], dtype=float)
    out, info = GibbsReactor(GibbsReactorParams(thermo=FEED_GAS, mode="adiabatic",
                                                pressure_drop=pressure_drop))(to_stream(n, T_in, P, SPECIES))
    c2 = sum(out[f"F_{k}"] for k in HYDROCARBONS[1:])
    m = to_vec(out, SYNGAS)
    # the equilibrium C2+ left (ppm) rides on as CH4 and H2, element-exact:
    # CnH(2n+2) = n CH4 + (1 - n) H2
    for k in HYDROCARBONS[1:]:
        nc = tc.elements(k)["C"]
        m = m.at[_I[CH4]].add(nc * out[f"F_{k}"]).at[_I[H2]].add((1 - nc) * out[f"F_{k}"])
    return to_stream(m, out["T"], out["P"]), dict(info, C2plus=c2)


# ------------------------------------------------------------------ reformer tube
@dataclass(frozen=True)
class ReformerTubeParams:
    """One reformer tube (a 1-D pseudo-homogeneous packed bed).

    Attributes:
        length: Heated (catalyst) length (m).
        d_in: Tube inside diameter (m).
        rho_bed: Catalyst bulk density (kg/m3 of bed).
        eta: Effectiveness factors of reactions 1-3.
        pressure_drop: Tube pressure drop (Pa), linear in length.
        mode: ``"flux"`` (heat-flux profile, total solved for ``T_out``)
            or ``"wall"`` (wall-T profile and ``U``; outlet T follows).
        flux_shape: ``shape(x)`` on ``x = z/L`` in [0, 1] with mean 1;
            default uniform.
        T_wall: ``(T_in, T_out)`` of a linear wall profile (K), wall mode.
        U: Overall wall-to-gas coefficient on the inside area (W/m2/K), wall mode.
        rtol, atol: diffrax step-size controller tolerances.
        adjoint: ``"reverse"`` (``RecursiveCheckpointAdjoint``: ``jax.grad``)
            or ``"forward"`` (``ForwardMode``: ``jax.jacfwd``).
    """

    # Xu & Froment (1989) Part II, "Methane steam reforming: II. Diffusional
    # limitations and reactor simulation", AIChE J. 35:97,
    # doi:10.1002/aic.690350110: Table 2 heated length and inside diameter;
    # eta1, eta3 ~0.03 near the inlet falling to ~0.02 (Fig. 6, read off the
    # figure); 29 -> ~25.7 bar over the tube (Fig. 3, read off the figure).
    length: float = 11.12
    d_in: float = 0.1016
    rho_bed: float = 1362.0          # ILLUSTRATIVE (Part II gives only the solid density, 2355.2)
    eta: tuple = (0.03, 0.03, 0.03)
    pressure_drop: float = 3.3e5
    mode: str = "flux"
    flux_shape: Callable | None = None
    T_wall: tuple = (1000.0, 1200.0)  # ILLUSTRATIVE
    U: float = 300.0                  # ILLUSTRATIVE
    rtol: float = 1e-8
    atol: float = 1e-10
    adjoint: str = "reverse"


def _integrate(n0, T0, P0, q_per_len, tp: ReformerTubeParams, n_save: int = 0):
    """Integrate a tube from ``(n0, T0, P0)`` with heat input ``q_per_len(z, T)``
    (W/m). Returns the outlet state (and ``n_save`` interior profiles)."""
    L = tp.length
    A = jnp.pi * tp.d_in**2 / 4.0
    eta = jnp.asarray(tp.eta, dtype=float)

    def rhs(z, y, args):
        n, T = y[:-1], y[-1]
        # a REJECTED trial step can leave the physical range; without these
        # guards its inf/NaN reaches the reverse pass (0 * NaN) even though
        # the step is discarded. They never bind on an accepted step.
        T = jnp.clip(T, 400.0, 1800.0)
        n = jnp.maximum(n, 1e-14)
        P = P0 - tp.pressure_drop * z / L
        p = n / jnp.sum(n) * P / P_BAR
        r = xu_froment_rates(p, T) * eta
        dn = A * tp.rho_bed * (NU.T @ r)
        dT = (q_per_len(z, T) - jnp.dot(GAS.enthalpy(T), dn)) / jnp.dot(n, GAS.cp(T))
        return jnp.concatenate([dn, dT[None]])

    adj = diffrax.ForwardMode() if tp.adjoint == "forward" else diffrax.RecursiveCheckpointAdjoint()
    saveat = (diffrax.SaveAt(ts=jnp.linspace(0.0, L, n_save)) if n_save else diffrax.SaveAt(t1=True))
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs), diffrax.Tsit5(), 0.0, L, L / 200.0,
        jnp.concatenate([n0, jnp.asarray(T0, dtype=float)[None]]),
        stepsize_controller=diffrax.PIDController(rtol=tp.rtol, atol=tp.atol),
        saveat=saveat, adjoint=adj, max_steps=8192, throw=False)
    return sol.ys, sol.result == diffrax.RESULTS.successful


def _flux_heat(Q, tp):
    shape = tp.flux_shape or (lambda x: 1.0)
    return lambda z, T: Q / tp.length * shape(z / tp.length)


def _wall_heat(tp):
    Tw0, Tw1 = tp.T_wall
    return lambda z, T: tp.U * jnp.pi * tp.d_in * (Tw0 + (Tw1 - Tw0) * z / tp.length - T)


def _deriv(tp):
    return jax.jacfwd if tp.adjoint == "forward" else jax.grad


def reformer_tube(inlet, tp: ReformerTubeParams = ReformerTubeParams(), T_out=None,
                  tol: float = 1e-4, max_iter: int = 30, n_profile: int = 0):
    """One reformer tube.

    Args:
        inlet: Stream over :data:`SYNGAS` (per tube).
        tp: :class:`ReformerTubeParams`.
        T_out: Outlet temperature (K), required in ``"flux"`` mode: the
            tube's heat input ``Q`` (W) is solved for it (secant on
            stop-gradient inputs, then one Newton step with the slope held,
            so ``dQ/dtheta`` is the implicit derivative).
        n_profile: Points of an axial profile to return (0: none).

    Returns:
        ``(outlet, info)``: ``info`` has ``Q`` (W), ``T_out``, ``CH4_conversion``,
        ``approach`` (K, :func:`approach_to_equilibrium`), ``heat_flux``
        (mean, W/m2 inside area), ``converged`` and, with ``n_profile``,
        ``profile`` (``z``, ``n``, ``T``).
    """
    n0 = to_vec(inlet)
    T0 = jnp.asarray(inlet["T"], dtype=float)
    P0 = jnp.asarray(inlet["P"], dtype=float)
    P1 = P0 - tp.pressure_drop
    ok_solve = jnp.asarray(True)
    if tp.mode == "flux":
        if T_out is None:
            raise ValueError("flux mode needs T_out")
        th = (n0, T0, P0, jnp.asarray(T_out, dtype=float))
        th0 = jax.tree_util.tree_map(lax.stop_gradient, th)

        def f(Q, th):
            n0, T0, P0, Tt = th
            y, _ = _integrate(n0, T0, P0, _flux_heat(Q, tp), tp)
            return y[-1, -1] - Tt

        # start from the duty to equilibrium at T_out (the tube's limit)
        eq, _ = GibbsReactor(GibbsReactorParams(thermo=GAS, mode="isothermal", T_out=th0[3]))(
            to_stream(th0[0], th0[1], P1))
        Qa = enthalpy(to_vec(eq), th0[3]) - enthalpy(th0[0], th0[1])
        Qb = 1.05 * Qa

        def cond(s):
            a, b, fa, fb, k = s
            return (jnp.abs(fb) > tol) & (fb != fa) & (k < max_iter)

        def body(s):
            a, b, fa, fb, k = s
            c = b - fb * (b - a) / (fb - fa)
            return b, c, fb, f(c, th0), k + 1

        _, Qs, _, fs, k = lax.while_loop(cond, body, (Qa, Qb, f(Qa, th0), f(Qb, th0), 0))
        Qs = lax.stop_gradient(Qs)
        Q = Qs - f(Qs, th) / lax.stop_gradient(_deriv(tp)(f)(Qs, th0))
        heat = _flux_heat(Q, tp)
        ok_solve = jnp.abs(fs) <= tol
    elif tp.mode == "wall":
        heat = _wall_heat(tp)
        Q = None
    else:
        raise ValueError(f"mode must be 'flux' or 'wall', not {tp.mode!r}")
    y, ok = _integrate(n0, T0, P0, heat, tp)
    n1, T1 = y[-1, :-1], y[-1, -1]
    if Q is None:
        Q = enthalpy(n1, T1) - enthalpy(n0, T0)
    out = to_stream(n1, T1, P1)
    info = {"Q": Q, "T_out": T1, "P_out": P1,
            "CH4_conversion": 1.0 - n1[_I[CH4]] / n0[_I[CH4]],
            "approach": approach_to_equilibrium(n1, T1, P1),
            "heat_flux": Q / (jnp.pi * tp.d_in * tp.length),
            "converged": ok & ok_solve}
    if n_profile:
        yp, _ = _integrate(n0, T0, P0, heat, tp, n_save=n_profile)
        info["profile"] = {"z": jnp.linspace(0.0, tp.length, n_profile), "n": yp[:, :-1], "T": yp[:, -1]}
    return out, info


# ------------------------------------------------------------------ back end
def cool(stream, T):
    """Sensible cooling of a gas over :data:`SYNGAS` to ``T``. Returns ``(stream, duty W)``."""
    n = to_vec(stream)
    return to_stream(n, T, stream["P"]), enthalpy(n, stream["T"]) - enthalpy(n, T)


def shift(stream, T_in, approach_dT=0.0, pressure_drop=0.0):
    """Adiabatic water-gas shift at inlet ``T_in`` (equilibrium of ``CO +
    H2O = CO2 + H2`` at ``T + approach_dT``; negative falls short)."""
    n = to_vec(stream)
    return GibbsReactor(GibbsReactorParams(thermo=GAS, mode="adiabatic", reactions=[WGS],
                                           approach_dT=approach_dT, pressure_drop=pressure_drop))(
        to_stream(n, T_in, stream["P"]))


#: molar mass of water (kg/mol), IUPAC atomic weights
M_H2O = 18.015e-3


def knock_out(stream, T):
    """Cool to ``T``, condensing water to saturation (IAPWS-IF97 ``p_sat``).

    Returns:
        ``(gas, condensate mol/s, duty W)``; the condensate leaves as liquid
        at ``T`` (latent heat from :func:`difflow.water.water_latent_heat`).
    """
    n = to_vec(stream)
    P = jnp.asarray(stream["P"], dtype=float)
    y = water_saturation_pressure(T) / P
    dry = jnp.sum(n) - n[_I[H2O]]
    w = jnp.minimum(n[_I[H2O]], dry * y / (1.0 - y))
    cond = n[_I[H2O]] - w
    m = n.at[_I[H2O]].set(w)
    h_l = GAS.enthalpy(T)[_I[H2O]] - water_latent_heat(T) * M_H2O
    duty = enthalpy(n, stream["T"]) - enthalpy(m, T) - cond * h_l
    return to_stream(m, T, P), cond, duty


#: reference feed per PSA bed (mol/s): Zhang, Hu & Xin (2025), 1.24e-5 kmol/s,
#: the bed of :class:`HydrogenPSAParams`. The plant PSA is that bed numbered up.
PSA_FEED_PER_BED = 1.24e-2


def purify(stream, psa=None):
    """PSA on a knocked-out syngas.

    ``psa`` is either

    * a :class:`~difflow_refinery.hydrogen.network.PSA` shortcut: H2 to
      product at its ``recovery``; the product's impurity (``1 - purity``)
      is the feed's non-H2, non-water gas in proportion to the feed; or
    * :class:`HydrogenPSAParams` (or None): the cycle model, numbered up
      from the reference bed (``n_beds_on_feed = F / PSA_FEED_PER_BED``) at
      the stream's pressure unless ``P_ads`` is set.

    Returns the :class:`HydrogenPSA` result dict (``product``, ``tail_gas``,
    ``purity``, ``recovery``, ``balance``).
    """
    from difflow_refinery.hydrogen.network import PSA

    n = to_vec(stream)
    feed = dict(zip(SYNGAS, n))
    if isinstance(psa, PSA):
        rec = jnp.asarray(psa.recovery, dtype=float)
        y = jnp.asarray(psa.purity, dtype=float)
        h2 = rec * n[_I[H2]]
        imp = n.at[_I[H2]].set(0.0).at[_I[H2O]].set(0.0)
        imp = imp / jnp.sum(imp) * h2 * (1.0 - y) / y
        prod = {k: (h2 if k == H2 else imp[i]) for i, k in enumerate(SYNGAS)}
        tail = {k: n[i] - prod[k] for i, k in enumerate(SYNGAS)}
        return {"product": prod, "tail_gas": tail, "purity": y, "recovery": rec,
                "balance": {k: n[i] - prod[k] - tail[k] for i, k in enumerate(SYNGAS)}}
    psa = psa or HydrogenPSAParams(P_ads=stream["P"])
    psa = psa.update(n_beds_on_feed=jnp.sum(n) / PSA_FEED_PER_BED)
    return HydrogenPSA(psa)(feed)


# ------------------------------------------------------------------ furnace
FLUE = (CO2, H2O, N2, "oxygen", "sulfur_dioxide")
FLUE_GAS = tc.IdealGasSet(list(FLUE), cp_basis="wide")
#: dry-air O2 mole fraction, U.S. Standard Atmosphere 1976 (the rest as N2);
#: the value :data:`difflow_refinery.sulfur.claus.AIR_O2` uses
AIR_O2 = 0.20946


def _burn(fuel, T_fuel, y_O2, T_air):
    """Fuel ``{species: mol/s}`` (over :data:`SPECIES`) burnt with air at flue
    O2 mole fraction ``y_O2`` (wet). Returns ``(H_in W, flue flows, air O2 mol/s)``."""
    nf = jnp.stack([jnp.asarray(fuel.get(k, 0.0), dtype=float) for k in SPECIES])
    E = {e: jnp.asarray(FEED_GAS.ELEMENTS[e]) for e in ("C", "H", "O", "N", "S")}
    C, Hh, O, N, S = (jnp.sum(nf * E[e]) for e in ("C", "H", "O", "N", "S"))
    s = C + Hh / 4.0 + S - O / 2.0                 # stoichiometric O2
    base = C + Hh / 2.0 + N / 2.0 + S             # flue moles before air
    a = (s + y_O2 * (base - s)) / (1.0 - y_O2 / AIR_O2)
    flue = jnp.stack([C, Hh / 2.0, N / 2.0 + a * (1.0 - AIR_O2) / AIR_O2, a - s, S])
    air = jnp.zeros(len(FLUE)).at[2].set(a * (1.0 - AIR_O2) / AIR_O2).at[3].set(a)
    H_in = jnp.sum(nf * FEED_GAS.enthalpy(T_fuel)) + jnp.sum(air * FLUE_GAS.enthalpy(T_air))
    return H_in, flue, a


def furnace(duty, tail_gas, fuel, y_O2=0.02, T_air=298.15, T_fuel=298.15, T_bridgewall=1273.15,
            T_stack=None):
    """The reformer furnace: tail gas + supplementary ``fuel`` (composition,
    ``{species: mol fraction}``) burnt so that the heat released down to
    ``T_bridgewall`` equals ``duty`` (W).

    Returns:
        dict: ``fuel`` (supplementary, mol/s), ``air_O2`` (mol/s), ``flue``
        (flows over :data:`FLUE`), ``CO2`` (mol/s), ``tail_gas_duty`` (W the
        tail gas alone gives), ``convection_duty`` (W, flue from
        ``T_bridgewall`` to ``T_stack``, if given).
    """
    def released(f):
        tot = {k: jnp.asarray(tail_gas.get(k, 0.0), dtype=float) + f * fuel.get(k, 0.0) for k in SPECIES}
        H_in, flue, a = _burn(tot, T_fuel, y_O2, T_air)
        return H_in - jnp.sum(flue * FLUE_GAS.enthalpy(T_bridgewall)), flue, a

    q0, _, _ = released(0.0)
    q1, _, _ = released(1.0)
    f = (duty - q0) / (q1 - q0)                       # exact: affine in f
    q, flue, a = released(f)
    out = {"fuel": f, "air_O2": a, "flue": flue, "CO2": flue[0], "tail_gas_duty": q0,
           "closure": q - duty}
    if T_stack is not None:
        out["convection_duty"] = jnp.sum(flue * (FLUE_GAS.enthalpy(T_bridgewall) - FLUE_GAS.enthalpy(T_stack)))
    return out


# ------------------------------------------------------------------ plant
@dataclass(frozen=True)
class SMRParams:
    """The hydrogen plant.

    Attributes:
        S_C: Steam / hydrocarbon carbon (mol/mol).
        T_out: Reformer outlet T (K).
        P: Reformer inlet pressure (Pa).
        n_tubes: Reformer tubes.
        tube: :class:`ReformerTubeParams`.
        h2s_removal: ZnO guard H2S removal fraction.
        T_prereformer: Prereformer inlet T (K); None skips the prereformer
            (then the feed must be C1 only).
        T_reformer_in: Reformer inlet T after reheat (K).
        T_hts: HT shift inlet T (K); ``hts_approach_dT`` its approach (K).
        T_lts: LT shift inlet T (K) or None (no LT shift); ``lts_approach_dT``.
        T_ko: Knock-out T (K).
        shift_dP: Pressure drop per shift reactor (Pa).
        psa: :class:`HydrogenPSAParams` or None (reference bed at the
            knock-out pressure, numbered up).
        fuel: Supplementary-fuel composition (mole fractions); None burns
            the feed natural gas.
        y_O2: Flue O2 (wet mole fraction).
        T_air, T_bridgewall, T_stack: Furnace temperatures (K).
        T_steam: Process steam temperature at the mixing point (K).
    """

    # Defaults follow the IEAGHG 2017/02 base case (Amec Foster Wheeler,
    # "Techno-Economic Evaluation of SMR Based Standalone (Merchant) Hydrogen
    # Plant with CCS", Feb 2017; sec. 2 text and the base-case heat and
    # material balance, streams 4-7, 14): S/C "around 2.7-2.8"; prereformer
    # inlet 500 C; reformer outlet "around 900-950 C" (its stream 5 is at
    # reforming equilibrium at ~918 C); HTS inlet 320 C, 2.80 -> 2.77 MPa;
    # PSA inlet 35 C; flue O2 1.02% wet; furnace exit "800 to 900 C"; HP
    # steam to process 400 C.
    S_C: float = 2.75
    T_out: float = 1193.15
    P: float = 32.0e5                # between IEAGHG streams 4 (3.39) and 5 (2.80 MPa)
    n_tubes: int = 1
    tube: ReformerTubeParams = field(default_factory=ReformerTubeParams)
    h2s_removal: float = 1.0
    T_prereformer: float | None = 773.15
    T_reformer_in: float = 893.15    # ILLUSTRATIVE (not in the IEAGHG balance)
    T_hts: float = 593.15
    hts_approach_dT: float = 0.0
    T_lts: float | None = None
    lts_approach_dT: float = 0.0
    T_ko: float = 308.15
    shift_dP: float = 0.03e6
    psa: HydrogenPSAParams | None = None
    fuel: dict | None = None
    y_O2: float = 0.0102
    T_air: float = 298.15            # ILLUSTRATIVE (IEAGHG preheats; T not given)
    T_bridgewall: float = 1123.15
    T_stack: float = 423.15          # ILLUSTRATIVE
    T_steam: float = 673.15


#: kg/mol of H2 and CO2 (IUPAC atomic weights)
M_H2 = 2.016e-3
M_CO2 = 44.009e-3


def smr_plant(feed, p: SMRParams = SMRParams()):
    """The SMR hydrogen plant on ``feed`` (natural gas stream over
    :data:`SPECIES`, mol/s for the whole plant; ``n_tubes`` share it).

    Returns:
        dict: per-stage streams and infos (``desulfurized``, ``prereformer``,
        ``reformer``, ``hts``, ``lts``, ``ko``, ``psa``, ``furnace``), and
        ``H2`` (product, mol/s), ``purity``, ``recovery``, ``CH4_slip``
        (reformer outlet, dry mol fraction), ``approach`` (K), ``steam``
        (mol/s), ``reformer_duty`` (W), ``fuel`` (supplementary, mol/s),
        ``natural_gas`` (feed + fuel, mol/s and ``ng_lhv`` W),
        ``CO2`` (emitted, mol/s), ``CO2_per_H2`` (kg/kg), ``efficiency``
        (H2 LHV / natural-gas LHV), ``converged``.
    """
    d, h2s_removed = desulfurize(feed, p.h2s_removal)
    C = carbon_flow(d)
    steam = p.S_C * C
    mixed = dict(d)
    mixed[f"F_{H2O}"] = jnp.asarray(d.get(f"F_{H2O}", 0.0), dtype=float) + steam
    nt = p.n_tubes
    if p.T_prereformer is not None:
        pre, pinfo = prereform(mixed, p.T_prereformer)
    else:
        pre, pinfo = to_stream(to_vec(mixed), p.T_prereformer or feed["T"], feed["P"]), {"converged": True}
    pre = dict(pre, P=jnp.asarray(p.P, dtype=float))
    per_tube = to_stream(to_vec(pre) / nt, p.T_reformer_in, p.P)
    ref1, rinfo = reformer_tube(per_tube, p.tube, T_out=p.T_out)
    ref = to_stream(to_vec(ref1) * nt, ref1["T"], ref1["P"])
    n_ref = to_vec(ref)
    dry = jnp.sum(n_ref) - n_ref[_I[H2O]]
    pgb_in, pgb = cool(ref, p.T_hts)
    hts, hinfo = shift(pgb_in, p.T_hts, p.hts_approach_dT, p.shift_dP)
    duties = {"process_gas_boiler": pgb}
    if p.T_lts is not None:
        c2, d2 = cool(hts, p.T_lts)
        duties["hts_cooler"] = d2
        lts, linfo = shift(c2, p.T_lts, p.lts_approach_dT, p.shift_dP)
    else:
        lts, linfo = hts, {"converged": True}
    ko, cond, ko_duty = knock_out(lts, p.T_ko)
    duties["knock_out"] = ko_duty
    ps = purify(ko, p.psa)
    tail = {k: v for k, v in ps["tail_gas"].items()}
    fuel_comp = p.fuel
    if fuel_comp is None:
        tot = sum(jnp.asarray(feed.get(f"F_{k}", 0.0), dtype=float) for k in SPECIES)
        fuel_comp = {k: jnp.asarray(feed.get(f"F_{k}", 0.0), dtype=float) / tot for k in SPECIES}
    Q_ref = rinfo["Q"] * nt
    fz = furnace(Q_ref, tail, fuel_comp, p.y_O2, p.T_air, T_bridgewall=p.T_bridgewall, T_stack=p.T_stack)
    # process heating: steam raised from liquid water at 298.15 K to T_steam,
    # feed preheat to the prereformer inlet, reheat to the reformer inlet
    h_liq = GAS.enthalpy(298.15)[_I[H2O]] - water_latent_heat(298.15) * M_H2O
    nf = to_vec(d, SPECIES)
    T_pre = p.T_prereformer if p.T_prereformer is not None else p.T_reformer_in
    duties["steam_raising"] = steam * (GAS.enthalpy(p.T_steam)[_I[H2O]] - h_liq)
    duties["feed_preheat"] = (jnp.sum(nf * (FEED_GAS.enthalpy(T_pre) - FEED_GAS.enthalpy(feed["T"])))
                              + steam * (FEED_GAS.enthalpy(T_pre)[_IF[H2O]] - FEED_GAS.enthalpy(p.T_steam)[_IF[H2O]]))
    duties["reformer_preheat"] = enthalpy(to_vec(pre), p.T_reformer_in) - enthalpy(to_vec(pre), pre["T"])
    duties["convection"] = fz["convection_duty"]
    lhv_f = lhv(SPECIES)
    ng_feed = jnp.sum(nf * lhv_f)
    fuel_vec = jnp.stack([jnp.asarray(fuel_comp.get(k, 0.0), dtype=float) for k in SPECIES])
    ng_fuel = fz["fuel"] * jnp.sum(fuel_vec * lhv_f)
    H2p = ps["product"][H2]
    lhv_h2 = lhv([H2])[0]
    return {
        "desulfurized": d, "H2S_removed": h2s_removed, "steam": steam,
        "prereformer": pre, "prereformer_info": pinfo,
        "reformer": ref, "reformer_info": rinfo, "hts": hts, "hts_info": hinfo,
        "lts": lts, "lts_info": linfo, "ko": ko, "condensate": cond, "psa": ps, "furnace": fz,
        "duties": duties, "H2": H2p, "H2_kg_h": H2p * M_H2 * 3600.0,
        "purity": ps["purity"], "recovery": ps["recovery"],
        "CH4_slip": n_ref[_I[CH4]] / dry, "approach": rinfo["approach"],
        "reformer_duty": Q_ref, "fuel": fz["fuel"],
        "natural_gas": jnp.sum(nf[:len(HYDROCARBONS)]) + fz["fuel"] * jnp.sum(fuel_vec[:len(HYDROCARBONS)]),
        "ng_lhv": ng_feed + ng_fuel, "ng_feed_lhv": ng_feed, "ng_fuel_lhv": ng_fuel,
        "CO2": fz["CO2"], "CO2_per_H2": fz["CO2"] * M_CO2 / (H2p * M_H2),
        "efficiency": H2p * lhv_h2 / (ng_feed + ng_fuel),
        "converged": rinfo["converged"] & pinfo["converged"] & hinfo["converged"] & linfo["converged"],
    }


__all__ = [
    "SPECIES", "SYNGAS", "HYDROCARBONS", "REACTIONS", "XU_FROMENT", "XU_FROMENT_SOURCE",
    "PSA_FEED_PER_BED", "ReformerTubeParams", "SMRParams", "xu_froment_rates",
    "xu_froment_equilibrium", "equilibrium_constants", "approach_to_equilibrium", "desulfurize",
    "carbon_flow", "prereform", "reformer_tube", "cool", "shift", "knock_out", "purify", "furnace",
    "lhv", "smr_plant",
]
