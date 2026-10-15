r"""Continuous catalyst regeneration (CCR) for the catalytic reformer (#474).

A CCR reformer moves its catalyst. It flows down through the reactors in
series (R1, then R2, ... Rn), is lifted to a regenerator, and returns fresh
to R1. The gas side is the semi-regen train of :mod:`.unit`, unchanged. What
CCR adds is the catalyst's coke, carried bed to bed:

* **Activity from coke.** Every rate constant, the coke make's included, is
  multiplied by :func:`activity` ``a(C) = exp(-alpha C)``. ``C`` is the coke
  on the catalyst in kg/kg. ``alpha`` (and the form) is ILLUSTRATIVE
  (see Sources).
* **Moving bed.** Catalyst enters bed ``k`` with the coke ``C_{k-1}`` it
  left bed ``k-1`` with (``C_0`` is the regenerated catalyst's residual
  coke). It leaves with ``C_k = C_{k-1} + m_k / G``. Here ``m_k`` is the
  bed's coke make (kg/s) and ``G`` the catalyst circulation (kg/s). The
  coke rises linearly along the catalyst's path, so the bed runs at the
  path-mean activity :func:`mean_activity`,
  ``(exp(-alpha C_{k-1}) - exp(-alpha C_k)) / (alpha (C_k - C_{k-1}))``.
  This is exact for rates linear in activity. Composition makes them
  nonlinear, so it is an approximation.
* **The coke-activity fixed point.** The coke make depends on the
  activities, and the activities depend on the coke. :meth:`CCRReformer.solve`
  converges the outlet cokes ``C_1..C_n`` alongside the H2 recycle. Each
  outer pass solves the reformer, warm-started at the last recycle gas. It
  then re-solves the beds' coke balances with the coke make per unit
  activity held. Gradients come from the implicit function theorem: one
  Newton step ``C = C* + (I - J)^-1 (Phi(C*, theta) - C*)``, with
  ``J = dPhi/dC`` at the converged point. This is the pattern of
  :func:`difflow_refinery.sulfur.tgtu.solve_sru`. Differentiate with
  ``jax.jacfwd``, never under ``jit``: the outer loop is Python on concrete
  values, and the reformer's beds are ``diffrax.ForwardMode``.
* **Regenerator** (:class:`Regenerator`). The spent catalyst's coke (H/C
  atomic ratio ``coke_HC``) burns in an O2-LIMITED burn zone. Its inlet gas
  is recycled flue gas plus make-up air, mixed to ``y_O2`` O2. The recycle
  dilutes the O2, and the O2 sets the burn-front temperature. Full burn to
  CO2 and water uses :func:`difflow_refinery.combustion.burn`. The peak is
  the adiabatic temperature of the gas that has given up all its O2. Its
  enthalpies come from the shared table
  (:func:`~difflow_refinery.combustion.gas_enthalpy`). The catalyst's
  sensible heat is left out, so this is the front's upper bound.
  Oxychlorination and calcination are a specified chloride make-up and a
  specified duty. The reduction zone reduces the platinum with hydrogen
  (``PtO2 + 2 H2 -> Pt + 2 H2O``, the PtO2 reading of the oxidised metal).

Outputs (:meth:`CCRResult.outputs`) are the reformer's plus the following
keys. ``ccr.spent_coke_wt`` is the coke on catalyst leaving the last
reactor, in wt%. ``ccr.coke_burn_kg_h`` is the burn rate.
``ccr.burn_capacity_used`` is the burn over the design burn capacity.
``ccr.burn_T_margin_K`` is the limit minus the peak.
``ccr.reduction_H2_mol_s`` is the reduction zone's hydrogen, and
``h2.export_mol_s`` is the net hydrogen less it. There are also
``ccr.activity_rx<k>`` and ``ccr.coke_rx<k>_wt``.

Limits of the model. Circulation going to infinity with zero regenerated
coke gives coke-free catalyst everywhere, which is the semi-regen
start-of-run reformer (tested exactly). Circulation going to zero is the
opposite limit, an infinitely coked bed. The catalyst's own sensible heat
and the regenerator's internal gas loop (dryer, heater) are not modelled.

Sources (every number below was read in the source; see
``docs/unit-operations-refinery.md``, "CCR reformer: references", for quotes):

* [AFPM16] Crocker & Philoon (Honeywell UOP), AFPM Q&A 2016, Question 12:
  CCRs are "designed for operation at 5 wt% carbon on catalyst or about
  5.25 wt.% coke at the design catalyst circulation rate", "operate
  normally at coke levels 40% above the design (about 7.4 wt.% coke)";
  "peak burn-zone temperatures are kept below 1100 F (595 C)".
  https://afpm.org/data-reports/technical-papers/qa-search/question-12-what-operating-strategies-do-you-employ
* [AFPM18] Ghosh (UOP), AFPM Q&A 2018, Question 10: low-coke turndown at
  "about 3 wt% coke at design catalyst circulation rate".
* [AFPM13] Bullen (UOP), AFPM Q&A 2013, Question 19: "a 1000 pph CCR
  section is designed to burn 52 pounds of coke per hour" -- the design
  burn capacity is 5.2 % of the design circulation.
* [US7638101] UOP patent US 7,638,101: regenerated catalyst "approximately
  0.2% or less" coke.
* [US5498756] UOP patent US 5,498,756: burn-zone gas "oxygen usually on the
  order of 0.5 to 1.5% by volume"; catalyst chloride 0.4-1.4 wt% (0.8-1.2
  preferred); make-up air about 3 % of the recycle gas.
* [US7285205] IFP patent US 7,285,205: moving-bed units at 0.3-0.8 MPa;
  H2/HC "generally between 2 and 3"; design RON 102; spent coke 3-10 wt%.
* [US5376607] patent US 5,376,607 (citing Beltramini et al., Appl. Catal.
  19, 203 (1985)): reforming coke "from about 0.5 to about 0.7 H/C molar
  ratio". UOP's 5 wt% C = 5.25 wt% coke implies H/C = 0.60.
* [Hameed17] Hameed, Iraqi J. Chem. Pet. Eng. 18(2), 109-123 (2017),
  Table 1 (an industrial CCR catalyst): Pt 0.3 wt%, Sn 0.3 wt%.
* Pt atomic weight 195.08: IUPAC CIAAW 2021 (abridged).

NOT sourced (no value could be verified): the deactivation constant
``alpha`` and the exponential form itself as applied here (reforming models
use it, but no published alpha on a coke basis was found), the burn-zone
inlet temperature (inside the patent's 370-550 C claim, value ILLUSTRATIVE),
the chloride strip per pass, the oxychlorination/calcination duty, and the
PtO2 reduction stoichiometry. The burn-front rise is first principles; no
published "K per 0.1 % O2" rule was found to check it against.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow_refinery import combustion as cb
from difflow_refinery.reforming.feed import NaphthaFeed
from difflow_refinery.reforming.unit import CatalyticReformer, ReformerParams, ReformerResult

#: Atomic weight of platinum (g/mol), IUPAC CIAAW 2021 (abridged): 195.08.
PT_MASS = 195.08
#: Design coke burn per catalyst circulated: 52 lb coke/h per 1000 lb/h [AFPM13].
DESIGN_BURN_RATIO = 0.052
#: Design spent-catalyst coke (kg/kg): 5.25 wt% at design circulation [AFPM16].
DESIGN_SPENT_COKE = 0.0525


def activity(coke: Array, alpha: Array) -> Array:
    """Catalyst activity ``exp(-alpha C)`` at coke ``C`` (kg coke / kg catalyst)."""
    return jnp.exp(-jnp.asarray(alpha) * jnp.asarray(coke))


def mean_activity(c_in: Array, c_out: Array, alpha: Array) -> Array:
    """Path-mean of :func:`activity` for coke rising linearly from ``c_in`` to ``c_out``.

    ``(exp(-alpha c_in) - exp(-alpha c_out)) / (alpha (c_out - c_in))``. The
    series ``1 - x/2 + x^2/6`` (``x = alpha (c_out - c_in)``) is used where
    ``|x| < 1e-4``. That keeps it exact, and differentiable, as the bed's
    coke pick-up goes to zero.
    """
    alpha = jnp.asarray(alpha, dtype=float)
    x = alpha * (jnp.asarray(c_out) - jnp.asarray(c_in))
    small = jnp.abs(x) < 1e-4
    xs = jnp.where(small, 1.0, x)
    ratio = jnp.where(small, 1.0 - x / 2.0 + x * x / 6.0, -jnp.expm1(-xs) / xs)
    return activity(c_in, alpha) * ratio


@dataclass
class RegeneratorParams(ParamsMixin):
    """The CCR regenerator: burn, oxychlorination/calcination, reduction.

    Attributes:
        coke_regen: Residual coke on the regenerated catalyst (kg/kg); 0.2 wt%
            ("0.2% or less", [US7638101]).
        coke_HC: Atomic H/C ratio of the coke; 0.6 (0.5-0.7 [US5376607];
            0.60 implied by [AFPM16]).
        y_O2: O2 mole fraction of the burn-zone inlet gas (recycled flue gas
            plus make-up air); 0.8 % (0.5-1.5 % [US5498756]).
        T_burn_in: Burn-zone inlet gas temperature (K). ILLUSTRATIVE (477 C,
            inside the 370-550 C of [US7638101]).
        T_burn_max: Highest burn temperature allowed (K): 595 C [AFPM16]. The
            margin is reported, not enforced.
        burn_capacity: Design coke burning rate (kg/s). ``None`` (default)
            takes 5.2 % of the circulation, the design ratio of [AFPM13].
        chloride_wt: Chloride on the catalyst held by oxychlorination (wt%);
            1.0 (0.8-1.2 preferred, [US5498756]).
        chloride_strip: Share of the catalyst's chloride stripped per pass and
            replaced by the oxychlorination zone's injection. ILLUSTRATIVE.
        oxy_calc_duty: Heat to the oxychlorination and calcination zones per
            kg of catalyst circulated (J/kg). ILLUSTRATIVE.
        pt_wt: Platinum on catalyst (wt%); 0.3 [Hameed17].
        H2_per_Pt: Mol H2 per mol Pt the reduction zone consumes (2, the
            PtO2 reading of the oxidised metal).
    """

    coke_regen: float = 0.002
    coke_HC: float = 0.6
    y_O2: float = 0.008
    T_burn_in: float = 750.15
    T_burn_max: float = 868.15
    burn_capacity: float | None = None
    chloride_wt: float = 1.0
    chloride_strip: float = 0.05
    oxy_calc_duty: float = 1.0e5
    pt_wt: float = 0.3
    H2_per_Pt: float = 2.0


class Regenerator:
    """Burn the spent catalyst's coke and restore the catalyst (see the module docstring).

    ``Regenerator(params)(circulation, coke_spent)`` returns a dict:
    ``coke_burn`` (kg/s), ``air`` and ``flue`` (mol/s, from
    :func:`~difflow_refinery.combustion.burn`), ``burn_gas`` (the burn-zone
    inlet, mol/s), ``recycle`` (recycled flue gas, mol/s), ``T_peak`` (K),
    ``burn_T_margin`` (K), ``burn_capacity_used`` (-), ``heat`` (W released),
    ``chloride_injection`` (kg Cl/s), ``oxy_calc_duty`` (W), ``H2_reduction``
    (mol/s) and ``water_reduction`` (mol/s).
    """

    def __init__(self, params: RegeneratorParams | None = None):
        self.params = params or RegeneratorParams()

    def elements(self, coke_burn: Array) -> dict[str, Array]:
        """Element mass flows (kg/s) of ``coke_burn`` kg/s of coke CH_x, x = ``coke_HC``."""
        x = jnp.asarray(self.params.coke_HC, dtype=float)
        wC = cb.ATOMIC_WEIGHT["C"] / (cb.ATOMIC_WEIGHT["C"] + x * cb.ATOMIC_WEIGHT["H"])
        z = jnp.zeros_like(jnp.asarray(coke_burn, dtype=float))
        return {"C": coke_burn * wC, "H": coke_burn * (1.0 - wC), "S": z, "N": z}

    def __call__(self, circulation: Array, coke_spent: Array) -> dict:
        p = self.params
        G = jnp.asarray(circulation, dtype=float)
        B = G * (jnp.asarray(coke_spent) - jnp.asarray(p.coke_regen))
        el = self.elements(B)
        # Full burn; the air is exactly the O2 the coke takes (O2-limited zone).
        out = cb.burn(el, 0.0, air_rate=0.0)
        o2 = out["o2_required"]
        k = cb.AIR_N2 / cb.AIR_O2
        air = {"oxygen": o2, "nitrogen": k * o2}
        products = {n: v for n, v in out["flue"].items() if n != "oxygen"}
        products["nitrogen"] = air["nitrogen"]
        vent = sum(products.values())
        # Recycle R mol/s of the O2-free flue so the inlet is y_O2 O2.
        R = o2 / jnp.asarray(p.y_O2) - o2 - air["nitrogen"]
        frac = {n: v / vent for n, v in products.items()}
        inlet = {n: R * frac[n] for n in products}
        inlet["nitrogen"] = inlet["nitrogen"] + air["nitrogen"]
        inlet["oxygen"] = o2
        outlet = {n: inlet.get(n, 0.0) + (products[n] if n != "nitrogen" else 0.0)
                  for n in products}
        T_in = jnp.asarray(p.T_burn_in, dtype=float)
        # Coke on the zero-formation-enthalpy convention (combustion module).
        H_target = cb.gas_enthalpy(inlet, T_in)
        T_peak = _gas_temperature(outlet, H_target, T_in)
        G_pt = G * jnp.asarray(p.pt_wt) / 100.0 * 1000.0 / PT_MASS
        H2 = jnp.asarray(p.H2_per_Pt) * G_pt
        return {
            "coke_burn": B, "elements": el, "air": air, "flue": products,
            "burn_gas": inlet, "burn_outlet": outlet, "recycle": R,
            "T_peak": T_peak, "burn_T_margin": jnp.asarray(p.T_burn_max) - T_peak,
            "burn_capacity_used": B / (DESIGN_BURN_RATIO * G if p.burn_capacity is None
                                       else jnp.asarray(p.burn_capacity)),
            "heat": cb.heat_of_combustion(el, 0.0, T_in),
            "chloride_injection": G * jnp.asarray(p.chloride_wt) / 100.0 * jnp.asarray(p.chloride_strip),
            "oxy_calc_duty": G * jnp.asarray(p.oxy_calc_duty),
            "H2_reduction": H2, "water_reduction": H2,
        }


def _gas_temperature(moles: dict, H: Array, T0: Array, iters: int = 30) -> Array:
    """Temperature at which ``gas_enthalpy(moles, T) = H``: Newton on stop-gradient
    inputs, then one Newton step with the slope held (the implicit derivative)."""
    m0 = jax.tree_util.tree_map(jax.lax.stop_gradient, moles)
    H0 = jax.lax.stop_gradient(H)

    def f(T, m, h):
        return cb.gas_enthalpy(m, T) - h

    def body(_, T):
        return T - f(T, m0, H0) / jax.grad(f)(T, m0, H0)

    T = jax.lax.fori_loop(0, iters, body, jax.lax.stop_gradient(jnp.asarray(T0, dtype=float)))
    slope = jax.lax.stop_gradient(jax.grad(f)(T, m0, H0))
    return T - f(T, moles, H) / slope


@dataclass
class CCRParams(ParamsMixin):
    """Catalyst circulation and the regenerator.

    Attributes:
        circulation: Catalyst circulation (kg/s). The default is the design
            circulation (:meth:`CCRReformer.design_circulation`, 5.25 wt%
            spent coke [AFPM16]) of ``lean_naphtha()`` in
            :func:`ccr_reformer_params`. On the reformer's ILLUSTRATIVE coke
            law that is ~11 kg/h, an ~80-day catalyst turnover: far slower
            than a plant's, because the coke make is low. Not a plant number.
        alpha: Deactivation constant of ``a = exp(-alpha C)`` (kg cat/kg
            coke). ILLUSTRATIVE.
        regenerator: :class:`RegeneratorParams`.
        tol: Convergence of the outlet cokes (kg/kg).
        max_iter: Outer (coke-activity) passes allowed.
    """

    circulation: float = 0.0030221
    alpha: float = 10.0
    regenerator: RegeneratorParams = field(default_factory=RegeneratorParams)
    tol: float = 1e-9
    max_iter: int = 30


def ccr_reformer_params(**overrides) -> ReformerParams:
    """A four-reactor, low-pressure CCR train: 3.5 bar separator, H2/HC 2.5.

    The reactor pressure and H2/HC sit in the CCR envelope (:data:`CCR_ENVELOPE`).
    The inlet temperatures (500 C, which gives RON 102.0 -- [US7285205]'s
    design target -- on ``lean_naphtha()`` at the design circulation), the
    catalyst split and the loop dP are ILLUSTRATIVE.
    """
    base = dict(inlet_T=(773.15,) * 4, catalyst_split=(0.10, 0.15, 0.25, 0.50),
                P_separator=3.5e5, loop_dP=2.0e5, H2_HC=2.5)
    base.update(overrides)
    return ReformerParams(**base)


#: The published envelope the CCR tests check against: reactor pressure (Pa,
#: 0.3-0.8 MPa [US7285205]), H2/HC (2-3 [US7285205]), reformate RON (100-104
#: per #474; 102 is [US7285205]'s design target), spent-catalyst coke (wt%,
#: 3 [AFPM18] turndown to 7.4 [AFPM16] normal maximum).
CCR_ENVELOPE = {"P_reactor": (3.0e5, 8.0e5), "H2_HC": (2.0, 3.0), "RON": (100.0, 104.0),
                "spent_coke_wt": (3.0, 7.4)}


@dataclass
class CCRResult:
    """A solved CCR reformer.

    Attributes:
        reformer: The gas side (:class:`~difflow_refinery.reforming.unit.ReformerResult`);
            its ``params`` carry the solved activities, so re-solve with :attr:`params`.
        params: The :class:`ReformerParams` it was solved at (``activity=None``).
        ccr: The :class:`CCRParams` it was solved at.
        coke: Coke on catalyst leaving each reactor (kg/kg), first first.
        coke_in: Coke entering each reactor (kg/kg).
        activity: Path-mean activity of each bed.
        regenerator: The :class:`Regenerator`'s dict.
        iterations: Outer coke-activity passes.
        converged: Whether the outlet cokes met ``tol`` (``None`` when traced).
    """

    reformer: ReformerResult
    params: ReformerParams
    ccr: CCRParams
    coke: Array
    coke_in: Array
    activity: Array
    regenerator: dict
    iterations: int
    converged: bool | None

    @property
    def tear(self):
        return self.reformer.tear

    @property
    def coke_make(self) -> Array:
        """Coke make of each bed (kg/s)."""
        return jnp.stack([r["coke"] for r in self.reformer.reactors])

    def balances(self) -> dict[str, Array]:
        """The reformer's closures plus ``coke``: (made - burned) / made, and
        ``catalyst_coke``: each bed's ``C_k - C_{k-1} - m_k/G`` over ``C_n``."""
        res = dict(self.reformer.balances())
        made = jnp.sum(self.coke_make)
        res["coke"] = (made - self.regenerator["coke_burn"]) / made
        G = jnp.asarray(self.ccr.circulation, dtype=float)
        res["catalyst_coke"] = jnp.max(jnp.abs(self.coke - self.coke_in - self.coke_make / G)) / self.coke[-1]
        return res

    def outputs(self) -> dict[str, Array]:
        out = dict(self.reformer.outputs())
        rg = self.regenerator
        out.update({
            "ccr.spent_coke_wt": 100.0 * self.coke[-1],
            "ccr.coke_burn_kg_h": rg["coke_burn"] * 3600.0,
            "ccr.burn_capacity_used": rg["burn_capacity_used"],
            "ccr.T_peak_C": rg["T_peak"] - 273.15,
            "ccr.burn_T_margin_K": rg["burn_T_margin"],
            "ccr.reduction_H2_mol_s": rg["H2_reduction"],
            "ccr.chloride_kg_h": rg["chloride_injection"] * 3600.0,
            "ccr.oxy_calc_duty_MW": rg["oxy_calc_duty"] / 1e6,
            "ccr.circulation_kg_h": jnp.asarray(self.ccr.circulation, dtype=float) * 3600.0,
            "h2.export_mol_s": out["h2.net_mol_s"] - rg["H2_reduction"],
        })
        for k in range(len(self.coke)):
            out[f"ccr.activity_rx{k + 1}"] = self.activity[k]
            out[f"ccr.coke_rx{k + 1}_wt"] = 100.0 * self.coke[k]
        return out

    def summary(self) -> str:
        o = {k: float(v) for k, v in self.outputs().items()}
        n = len(self.coke)
        return "\n".join([
            self.reformer.summary(),
            f"CCR: circulation {o['ccr.circulation_kg_h']:.1f} kg/h, spent coke "
            f"{o['ccr.spent_coke_wt']:.2f} wt%, activity "
            + ", ".join(f"{o[f'ccr.activity_rx{k}']:.3f}" for k in range(1, n + 1)),
            f"regenerator: burn {o['ccr.coke_burn_kg_h']:.2f} kg/h "
            f"({100 * o['ccr.burn_capacity_used']:.0f}% of capacity), peak {o['ccr.T_peak_C']:.0f} C "
            f"(margin {o['ccr.burn_T_margin_K']:.0f} K), reduction H2 {o['ccr.reduction_H2_mol_s']:.4f} mol/s",
        ])


def _has_tracer(obj) -> bool:
    found = []

    def visit(x):
        if isinstance(x, jax.core.Tracer):
            found.append(True)
        elif dataclasses.is_dataclass(x) and not isinstance(x, type):
            for f in dataclasses.fields(x):
                visit(getattr(x, f.name))
        elif isinstance(x, (tuple, list)):
            for v in x:
                visit(v)
        elif isinstance(x, dict):
            for v in x.values():
                visit(v)

    visit(obj)
    return bool(found)


def _primal(obj):
    """``obj`` with every array leaf replaced by its primal value (``stop_gradient``).

    Eager ``jax.jacfwd`` carries concrete primals, so this gives concrete
    numbers there; under ``jit`` it does not (and the CCR is not jit-able).
    """
    if isinstance(obj, jax.core.Tracer) or isinstance(obj, (jax.Array, np.ndarray)):
        return jax.lax.stop_gradient(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        vals = {f.name: _primal(getattr(obj, f.name)) for f in dataclasses.fields(obj) if f.init}
        new = replace(obj, **vals)
        for f in dataclasses.fields(obj):          # non-init fields (e.g. derived arrays)
            if not f.init:
                object.__setattr__(new, f.name, _primal(getattr(obj, f.name)))
        return new
    if isinstance(obj, tuple):
        return tuple(_primal(v) for v in obj)
    if isinstance(obj, list):
        return [_primal(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _primal(v) for k, v in obj.items()}
    return obj


class CCRReformer:
    """A catalytic reformer with continuous catalyst regeneration (module docstring).

    Example::

        from difflow_refinery.reforming import lean_naphtha
        from difflow_refinery.reforming.ccr import CCRReformer, ccr_reformer_params
        res = CCRReformer(ccr_reformer_params()).solve(lean_naphtha())
        print(res.summary())
    """

    def __init__(self, params: ReformerParams | None = None, ccr: CCRParams | None = None,
                 adjoint: str = "forward"):
        self.params = params or ccr_reformer_params()
        self.ccr = ccr or CCRParams()
        self.reformer = CatalyticReformer(self.params, adjoint=adjoint)

    # ----- the coke map -------------------------------------------------------

    def _activities(self, c: Array, ccr: CCRParams) -> tuple[Array, Array]:
        c0 = jnp.asarray(ccr.regenerator.coke_regen, dtype=float)
        c_in = jnp.concatenate([c0[None], c[:-1]])
        return c_in, mean_activity(c_in, c, ccr.alpha)

    def _reform(self, feed, p, ccr, c, tear, tol, max_iter):
        c_in, a = self._activities(c, ccr)
        pa = replace(p, activity=tuple(a[k] for k in range(p.n_reactors)))
        res = self.reformer.solve(feed, pa, tear_initial=tear, tol=tol, max_iter=max_iter,
                                  on_nonconvergence="ignore")
        return res, c_in, a

    def coke_map(self, c: Array, feed: NaphthaFeed, params: ReformerParams, ccr: CCRParams,
                 tear=None, tol: float = 1e-10, max_iter: int = 400) -> Array:
        """``Phi(C)``: the outlet cokes the beds make running at the activities of ``C``."""
        res, c_in, _ = self._reform(feed, params, ccr, c, tear, tol, max_iter)
        m = jnp.stack([r["coke"] for r in res.reactors])
        return c_in + m / jnp.asarray(ccr.circulation, dtype=float)

    def _bed_cokes(self, s: np.ndarray, G: float, alpha: float, c0: float) -> np.ndarray:
        """Outlet cokes with each bed's coke make per unit activity ``s`` (kg/s) held:
        ``C_k = C_{k-1} + s_k abar(C_{k-1}, C_k) / G``, bed by bed (bisection; the
        residual is monotone in ``C_k``)."""
        out, c_prev = [], c0
        for sk in s:
            lo, hi = c_prev, c_prev + sk * float(np.exp(-alpha * c_prev)) / G
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                g = mid - c_prev - sk * float(mean_activity(c_prev, mid, alpha)) / G
                lo, hi = (mid, hi) if g < 0 else (lo, mid)
                if hi - lo < 1e-15:
                    break
            c_prev = 0.5 * (lo + hi)
            out.append(c_prev)
        return np.array(out)

    def _converge(self, feed, p, ccr, tear, tol, max_iter):
        G, alpha = float(ccr.circulation), float(ccr.alpha)
        c0 = float(ccr.regenerator.coke_regen)
        n = p.n_reactors
        c = np.full(n, c0)
        res = None
        for it in range(1, int(ccr.max_iter) + 1):
            res, _, a = self._reform(feed, p, ccr, jnp.asarray(c), tear, tol, max_iter)
            tear = res.tear
            m = np.array([float(r["coke"]) for r in res.reactors])
            c_new = self._bed_cokes(m / np.asarray(a), G, alpha, c0)
            dc = float(np.max(np.abs(c_new - c)))
            c = c_new
            if dc < float(ccr.tol):
                return c, tear, it, True
        return c, tear, int(ccr.max_iter), False

    def design_circulation(self, feed: NaphthaFeed, params: ReformerParams | None = None,
                           spent_coke: float = DESIGN_SPENT_COKE, tol: float = 1e-7,
                           max_iter: int = 20) -> tuple[float, CCRResult]:
        """The circulation (kg/s) at which the spent catalyst carries ``spent_coke``
        (kg/kg; default the 5.25 wt% design basis of [AFPM16]), by secant on
        ``log G``. Concrete only. Returns ``(circulation, result)``."""
        p = params or self.params
        ccr = self.ccr
        G = float(ccr.circulation)
        res = self.solve(feed, p, ccr)
        f = float(res.coke[-1]) - spent_coke
        # Spent coke ~ 1/G: the first step is that guess.
        G1 = G * (float(res.coke[-1]) - float(ccr.regenerator.coke_regen)) / (
            spent_coke - float(ccr.regenerator.coke_regen))
        x0, f0, tear = np.log(G), f, res.tear
        x1 = np.log(G1)
        for _ in range(max_iter):
            res = self.solve(feed, p, replace(ccr, circulation=float(np.exp(x1))), tear_initial=tear)
            f1 = float(res.coke[-1]) - spent_coke
            if abs(f1) < tol:
                return float(np.exp(x1)), res
            x0, x1, f0 = x1, x1 - f1 * (x1 - x0) / (f1 - f0), f1
            tear = res.tear
        warnings.warn("design_circulation did not converge")
        return float(np.exp(x1)), res

    # ----- solve --------------------------------------------------------------

    def solve(self, feed: NaphthaFeed, params: ReformerParams | None = None,
              ccr: CCRParams | None = None, *, tear_initial=None, tol: float = 1e-10,
              max_iter: int = 400) -> CCRResult:
        """Solve the CCR reformer for a feed (see the module docstring).

        Concrete inputs converge the coke-activity fixed point by outer passes.
        Under eager ``jax.jacfwd`` the converged point is found on the inputs'
        primal values, and one implicit Newton step carries the tangents (the
        Jacobian ``dPhi/dC`` by ``jax.jacfwd`` at the converged point).
        """
        p = params or self.params
        ccr = ccr or self.ccr
        if p.activity is not None:
            raise ValueError("the CCR sets ReformerParams.activity itself; leave it None "
                             "(re-solve with CCRResult.params, not .reformer.params)")
        traced = _has_tracer((p, ccr, feed))
        p0, ccr0, feed0 = (_primal(p), _primal(ccr), _primal(feed)) if traced else (p, ccr, feed)
        c_star, tear, it, ok = self._converge(feed0, p0, ccr0, tear_initial, tol, max_iter)
        if not ok:
            warnings.warn(f"CCR coke-activity loop did not converge in {it} passes")
        c = jnp.asarray(c_star)
        if traced:
            J = jax.jacfwd(lambda cc: self.coke_map(cc, feed0, p0, ccr0, tear, tol, max_iter))(c)
            phi = self.coke_map(c, feed, p, ccr, tear, tol, max_iter)
            c = c + jnp.linalg.solve(jnp.eye(len(c_star)) - J, phi - c)
        res, c_in, a = self._reform(feed, p, ccr, c, tear, tol, max_iter)
        regen = Regenerator(ccr.regenerator)(ccr.circulation, c[-1])
        return CCRResult(reformer=res, params=p, ccr=ccr, coke=c, coke_in=c_in, activity=a,
                         regenerator=regen, iterations=it, converged=None if traced else ok)


__all__ = [
    "PT_MASS", "DESIGN_BURN_RATIO", "DESIGN_SPENT_COKE", "activity", "mean_activity", "RegeneratorParams", "Regenerator", "CCRParams",
    "ccr_reformer_params", "CCR_ENVELOPE", "CCRResult", "CCRReformer",
]
