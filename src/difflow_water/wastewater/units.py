"""Palette operations for activated sludge: an ASM1 reactor, a Takacs settler, the whole plant."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import diffrax
import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream
from difflow_water.wastewater.asm1 import (
    IDX,
    PART_MASK,
    S_O_SAT,
    ASM1Params,
    conversion_rates,
    state_to_stream,
    stream_to_state,
    suspended_solids,
)
from difflow_water.wastewater.bsm1 import (
    ActivatedSludgeParams,
    TakacsParams,
    activated_sludge_plant,
    takacs_residual,
)

_ASM_FIELDS = ("Y_A", "Y_H", "f_P", "i_XB", "i_XP", "mu_H", "K_S", "K_OH", "K_NO", "b_H",
               "eta_g", "eta_h", "k_h", "K_X", "mu_A", "K_NH", "b_A", "K_OA", "k_a")


@dataclass(repr=False)
class ASM1ReactorParams(ParamsMixin):
    """A steady-state ASM1 CSTR (BSM1 kinetics by default, [W5] Tables 2-3).

    Attributes:
        volume: Reactor volume (m^3).
        KLa: Oxygen transfer coefficient (1/d); 0 for an anoxic reactor.
        DO_setpoint: Dissolved oxygen held at this value (g/m^3) instead of
            using KLa; negative means "use KLa".
        Y_A: Autotrophic yield (g COD / g N).
        Y_H: Heterotrophic yield (g COD / g COD).
        f_P: Fraction of biomass to particulate products (-).
        i_XB: N content of biomass (g N / g COD).
        i_XP: N content of particulate products (g N / g COD).
        mu_H: Heterotrophic maximum growth rate (1/d).
        K_S: Half-saturation, readily biodegradable substrate (g COD/m^3).
        K_OH: Oxygen half-saturation, heterotrophs (g O2/m^3).
        K_NO: Nitrate half-saturation (g N/m^3).
        b_H: Heterotrophic decay (1/d).
        eta_g: Anoxic growth correction (-).
        eta_h: Anoxic hydrolysis correction (-).
        k_h: Hydrolysis rate (g COD / (g COD d)).
        K_X: Hydrolysis half-saturation (g COD / g COD).
        mu_A: Autotrophic maximum growth rate (1/d).
        K_NH: Ammonia half-saturation (g N/m^3).
        b_A: Autotrophic decay (1/d).
        K_OA: Oxygen half-saturation, autotrophs (g O2/m^3).
        k_a: Ammonification rate (m^3 / (g COD d)).
    """

    volume: float | Array = 1333.0
    KLa: float | Array = 240.0
    DO_setpoint: float | Array = -1.0
    Y_A: float | Array = 0.24
    Y_H: float | Array = 0.67
    f_P: float | Array = 0.08
    i_XB: float | Array = 0.08
    i_XP: float | Array = 0.06
    mu_H: float | Array = 4.0
    K_S: float | Array = 10.0
    K_OH: float | Array = 0.2
    K_NO: float | Array = 0.5
    b_H: float | Array = 0.3
    eta_g: float | Array = 0.8
    eta_h: float | Array = 0.8
    k_h: float | Array = 3.0
    K_X: float | Array = 0.1
    mu_A: float | Array = 0.5
    K_NH: float | Array = 1.0
    b_A: float | Array = 0.05
    K_OA: float | Array = 0.4
    k_a: float | Array = 0.05

    def asm(self) -> ASM1Params:
        return ASM1Params(**{k: getattr(self, k) for k in _ASM_FIELDS})


class ASM1Reactor:
    """Continuous stirred activated-sludge reactor with ASM1 kinetics, at steady state.

    Inlet: an ASM1 stream (``F_H2O`` and ``F_<state>``, 1 mol = 1 g). Outlet:
    the reactor contents. The steady state is found by integrating the
    reactor in time for 20 residence times from its inlet composition and
    polishing with Newton (implicit gradients).

    Example:
        >>> from difflow_water.wastewater import bsm1_influent
        >>> out, info = ASM1Reactor(ASM1ReactorParams(volume=1000.0, KLa=0.0))(bsm1_influent())
        >>> bool(info["converged"])
        True
    """

    symbol = "ASM"
    equations = [
        r"0 = \frac{Q}{V}(Z_{in} - Z) + r(Z) + K_La\,(S_O^* - S_O)\,\delta_{O}",
        r"r_i = \textstyle\sum_j \nu_{ij}\rho_j\qquad \text{(ASM1, 8 processes, 13 states)}",
    ]
    assumptions = [
        "ASM1 as stated in the BSM1 report; BSM1 parameter values (about 15 C) by default.",
        "Completely mixed; steady state; oxygen saturation 8 g/m^3.",
        "Dissolved oxygen either from KLa or held at a setpoint (aeration reported).",
    ]
    references = [
        "Alex et al., Benchmark Simulation Model no. 1 (BSM1), LTH-IEA-7229, Lund University (2008).",
        "Henze et al., Activated Sludge Models ASM1, ASM2, ASM2d and ASM3, IWA STR 9 (2000), doi:10.2166/9781780402369.",
    ]
    parameter_symbols = {"volume": "V", "KLa": "K_La", "DO_setpoint": "S_O,sp"}
    parameter_units = {"volume": "m^3", "KLa": "1/d", "DO_setpoint": "g/m^3",
                       "Y_A": "g COD/g N", "Y_H": "g COD/g COD", "f_P": "-", "i_XB": "g N/g COD",
                       "i_XP": "g N/g COD", "mu_H": "1/d", "K_S": "g COD/m^3", "K_OH": "g O2/m^3",
                       "K_NO": "g N/m^3", "b_H": "1/d", "eta_g": "-", "eta_h": "-",
                       "k_h": "g COD/(g COD d)", "K_X": "g COD/g COD", "mu_A": "1/d",
                       "K_NH": "g N/m^3", "b_A": "1/d", "K_OA": "g O2/m^3", "k_a": "m^3/(g COD d)"}
    numerical_method = "Stiff integration (diffrax Kvaerno5) to near steady state, then Newton."

    def __init__(self, params: ASM1ReactorParams):
        """Initialize.

        Args:
            params: :class:`ASM1ReactorParams`.
        """
        self.params = params

    def __call__(self, inlet: Stream) -> tuple[Stream, dict]:
        """Run the reactor to steady state.

        Returns:
            ``(outlet, info)``; ``info`` has ``Z`` (13 states), ``rates``
            (conversion rates, per day), ``aeration`` (g O2/d), ``converged``.
        """
        p = self.params
        Q, Zin = stream_to_state(inlet)
        kin = jnp.stack([jnp.asarray(getattr(p, k), dtype=jnp.float64) for k in _ASM_FIELDS])
        Z, rates, aer, ok, res = _reactor_solve(
            Zin, Q, jnp.asarray(p.volume, dtype=jnp.float64), jnp.asarray(p.KLa, dtype=jnp.float64),
            jnp.asarray(p.DO_setpoint, dtype=jnp.float64), kin)
        out = state_to_stream(Q, Z, inlet["T"], inlet["P"])
        return out, {"Z": Z, "rates": rates, "aeration": aer, "TSS": suspended_solids(Z),
                     "converged": ok, "residual": res}


@jax.jit
def _reactor_solve(Zin, Q, volume, KLa, DO_setpoint, kin):
    """Steady state of one ASM1 CSTR (jitted once per process)."""
    o = IDX["S_O"]
    setpoint = DO_setpoint >= 0.0

    def make_f(Zin, Q, volume, KLa, DO_setpoint, kin):
        asm = ASM1Params(**dict(zip(_ASM_FIELDS, kin)))

        def f(Z):
            dz = Q * (Zin - Z) / volume + conversion_rates(Z, asm)
            dz = dz.at[o].add(KLa * (S_O_SAT - Z[o]))
            return dz.at[o].set(jnp.where(setpoint, -1e3 * (Z[o] - DO_setpoint), dz[o]))
        return f

    f = make_f(Zin, Q, volume, KLa, DO_setpoint, kin)
    sg = jax.lax.stop_gradient
    f_sg = make_f(sg(Zin), sg(Q), sg(volume), sg(KLa), sg(DO_setpoint), sg(kin))
    asm = ASM1Params(**dict(zip(_ASM_FIELDS, kin)))
    tau = volume / Q
    x0 = jax.lax.stop_gradient(jnp.maximum(Zin, 1e-3))
    sol = diffrax.diffeqsolve(diffrax.ODETerm(lambda t, x, a: f_sg(x)), diffrax.Kvaerno5(),
                              t0=0.0, t1=jax.lax.stop_gradient(20.0 * tau),
                              dt0=jax.lax.stop_gradient(1e-4 * tau), y0=x0,
                              stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-8),
                              max_steps=5000, throw=False, adjoint=diffrax.ForwardMode())
    x1 = jax.lax.stop_gradient(sol.ys[-1])
    root = optx.root_find(lambda Z, _: f(Z), optx.Newton(rtol=1e-10, atol=1e-9), x1,
                          max_steps=30, throw=False)
    Z = root.value
    res = jnp.max(jnp.abs(f(Z)))
    rates = conversion_rates(Z, asm)
    aer = jnp.where(setpoint, -(Q * (Zin[o] - Z[o]) + rates[o] * volume),
                    KLa * volume * (S_O_SAT - Z[o]))
    return Z, rates, aer, (root.result == optx.RESULTS.successful) & (res < 1e-6), res


@dataclass(repr=False)
class TakacsSettlerParams(ParamsMixin):
    """Takacs secondary settler with a given underflow (BSM1 values, [W5] Table 4).

    Attributes:
        Q_underflow: Underflow (recycle + wastage) flow (m^3/d).
        area: Surface area (m^2).
        height: Height (m).
        n_layers: Number of layers (-).
        feed_layer: Feed layer from the bottom (1-based).
        v0_max: Maximum settling velocity (m/d).
        v0: Maximum Vesilind settling velocity (m/d).
        r_h: Hindered settling parameter (m^3/g SS).
        r_p: Flocculant settling parameter (m^3/g SS).
        f_ns: Non-settleable fraction (-).
        X_t: Clarification threshold (g SS/m^3).
    """

    Q_underflow: float | Array = 18831.0
    area: float | Array = 1500.0
    height: float | Array = 4.0
    n_layers: int = 10
    feed_layer: int = 6
    v0_max: float | Array = 250.0
    v0: float | Array = 474.0
    r_h: float | Array = 0.000576
    r_p: float | Array = 0.00286
    f_ns: float | Array = 0.00228
    X_t: float | Array = 3000.0


class TakacsSettler:
    """Secondary settler: 10-layer Takacs model at steady state (BSM1).

    Inlet: ASM1 stream. Outlets: ``effluent`` and ``underflow``
    (``Q_underflow``). Solubles pass unchanged; particulates keep the feed's
    fractions.

    Example:
        >>> from difflow_water.wastewater import activated_sludge_plant, bsm1_influent
        >>> from difflow_water.wastewater.asm1 import state_to_stream
        >>> r = activated_sludge_plant(bsm1_influent())
        >>> feed = state_to_stream(r["Q"]["settler_feed"], r["reactors"][-1])
        >>> eff, und, info = TakacsSettler(TakacsSettlerParams())(feed)
        >>> round(float(info["solids"][-1]), 1)
        12.5
    """

    symbol = "SET"
    equations = [
        r"v_s = \max(0, \min(v_0', v_0 (e^{-r_h (X - X_{min})} - e^{-r_p (X - X_{min})})))",
        r"J_s = v_s X,\qquad \text{flux limited by } \min(J_{s,m}, J_{s,m\pm1})",
    ]
    assumptions = [
        "Non-reactive, one-dimensional, 10 layers; feed layer 6 (BSM1).",
        "Solubles unchanged; particulate fractions those of the feed.",
        "Steady state.",
    ]
    references = [
        "Takacs, Patry & Nolasco, Water Res. 25 (1991) 1263-1271, doi:10.1016/0043-1354(91)90066-Y.",
        "Alex et al., Benchmark Simulation Model no. 1 (BSM1), LTH-IEA-7229, Lund University (2008).",
    ]
    parameter_symbols = {"Q_underflow": "Q_u", "area": "A", "v0": "v_0", "v0_max": "v_0'",
                         "r_h": "r_h", "r_p": "r_p", "f_ns": "f_ns", "X_t": "X_t"}
    parameter_units = {"Q_underflow": "m^3/d", "area": "m^2", "height": "m", "n_layers": "-",
                       "feed_layer": "-", "v0_max": "m/d", "v0": "m/d", "r_h": "m^3/g",
                       "r_p": "m^3/g", "f_ns": "-", "X_t": "g/m^3"}
    numerical_method = "Stiff integration (diffrax Kvaerno5) to near steady state, then Newton."
    outlet_roles = ["effluent", "underflow"]

    def __init__(self, params: TakacsSettlerParams):
        """Initialize.

        Args:
            params: :class:`TakacsSettlerParams`.
        """
        if not 1 < int(params.feed_layer) < int(params.n_layers):
            raise ValueError("feed_layer must be strictly between 1 and n_layers")
        self.params = params

    def __call__(self, feed: Stream) -> tuple[Stream, Stream, dict]:
        """Settle to steady state.

        Returns:
            ``(effluent, underflow, info)``; ``info`` has ``solids`` (per
            layer, bottom first, g SS/m^3), ``TSS_effluent``, ``converged``.
        """
        p = self.params
        Qf, Zf = stream_to_state(feed)
        Xf = suspended_solids(Zf)
        geo = jnp.stack([jnp.asarray(getattr(p, k), dtype=jnp.float64) for k in _TAKACS_FLOATS])
        X, ok, res = _settler_solve(Xf, Qf, jnp.asarray(p.Q_underflow, dtype=jnp.float64), geo,
                                    int(p.n_layers), int(p.feed_layer))
        Qu = jnp.asarray(p.Q_underflow, dtype=jnp.float64)
        Qe = Qf - Qu
        Zu = jnp.where(PART_MASK, Zf * X[0] / Xf, Zf)
        Ze = jnp.where(PART_MASK, Zf * X[-1] / Xf, Zf)
        eff = state_to_stream(Qe, Ze, feed["T"], feed["P"])
        und = state_to_stream(Qu, Zu, feed["T"], feed["P"])
        return eff, und, {"solids": X, "TSS_effluent": X[-1], "TSS_underflow": X[0],
                          "converged": ok, "residual": res}


_TAKACS_FLOATS = ("area", "height", "v0_max", "v0", "r_h", "r_p", "f_ns", "X_t")


@partial(jax.jit, static_argnums=(4, 5))
def _settler_solve(Xf, Qf, Qu, geo, n_layers, feed_layer):
    """Steady state of the Takacs settler's solids (jitted once per layer count)."""
    t = TakacsParams(n_layers=n_layers, feed_layer=feed_layer,
                     **dict(zip(_TAKACS_FLOATS, geo)))
    Qe = Qf - Qu
    sg = jax.lax.stop_gradient
    t_sg = TakacsParams(n_layers=n_layers, feed_layer=feed_layer,
                        **dict(zip(_TAKACS_FLOATS, sg(geo))))

    def f(X):
        return takacs_residual(X, Xf, Qf, Qu, Qe, t)

    def f_sg(X):
        return takacs_residual(X, sg(Xf), sg(Qf), sg(Qu), sg(Qe), t_sg)

    x0 = jax.lax.stop_gradient(jnp.full(n_layers, Xf))
    sol = diffrax.diffeqsolve(diffrax.ODETerm(lambda tt, x, a: f_sg(x)), diffrax.Kvaerno5(),
                              t0=0.0, t1=50.0, dt0=1e-4, y0=x0,
                              stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6),
                              max_steps=20000, throw=False, adjoint=diffrax.ForwardMode())
    root = optx.root_find(lambda X, _: f(X), optx.Newton(rtol=1e-10, atol=1e-8),
                          jax.lax.stop_gradient(sol.ys[-1]), max_steps=30, throw=False)
    X = root.value
    res = jnp.max(jnp.abs(f(X)))
    return X, (root.result == optx.RESULTS.successful) & (res < 1e-6), res


@dataclass(repr=False)
class ActivatedSludgePlantParams(ParamsMixin):
    """The BSM1 plant as one block (BSM1 open-loop defaults, [W5]).

    Attributes:
        Q_a: Internal recycle (m^3/d).
        Q_r: External recycle (m^3/d).
        Q_w: Wastage (m^3/d).
        KLa_3: Oxygen transfer, reactor 3 (1/d).
        KLa_4: Oxygen transfer, reactor 4 (1/d).
        KLa_5: Oxygen transfer, reactor 5 (1/d).
        V_anoxic: Volume of each of the two anoxic reactors (m^3).
        V_aerobic: Volume of each of the three aerobic reactors (m^3).
        settler_area: Settler area (m^2).
    """

    Q_a: float | Array = 55338.0
    Q_r: float | Array = 18446.0
    Q_w: float | Array = 385.0
    KLa_3: float | Array = 240.0
    KLa_4: float | Array = 240.0
    KLa_5: float | Array = 84.0
    V_anoxic: float | Array = 1000.0
    V_aerobic: float | Array = 1333.0
    settler_area: float | Array = 1500.0


class ActivatedSludgePlant:
    """A BSM1-layout activated sludge plant (2 anoxic + 3 aerobic ASM1 reactors, Takacs settler).

    Inlet: the influent. Outlets: ``effluent`` and ``wastage``. One block
    solved equation-oriented (see
    :func:`~difflow_water.wastewater.bsm1.activated_sludge_plant`).

    Example:
        >>> from difflow_water.wastewater import bsm1_influent
        >>> eff, waste, info = ActivatedSludgePlant(ActivatedSludgePlantParams())(bsm1_influent())
        >>> round(float(info["sludge_age"]), 1)
        9.1
    """

    symbol = "AS"
    equations = [
        r"0 = Q_{in} Z_{in} - Q Z_k + r(Z_k) V_k + K_La V_k (S_O^* - S_O)",
        r"\text{Takacs 10-layer settler; } Q_a, Q_r, Q_w \text{ given}",
    ]
    assumptions = [
        "BSM1 layout and ASM1 kinetics (BSM1 values); steady state.",
        "Solved as one equation set: stiff integration for 200 d, then Newton.",
    ]
    references = [
        "Alex et al., Benchmark Simulation Model no. 1 (BSM1), LTH-IEA-7229, Lund University (2008).",
        "Takacs, Patry & Nolasco, Water Res. 25 (1991) 1263-1271, doi:10.1016/0043-1354(91)90066-Y.",
    ]
    parameter_symbols = {"Q_a": "Q_a", "Q_r": "Q_r", "Q_w": "Q_w"}
    parameter_units = {"Q_a": "m^3/d", "Q_r": "m^3/d", "Q_w": "m^3/d", "KLa_3": "1/d",
                       "KLa_4": "1/d", "KLa_5": "1/d", "V_anoxic": "m^3", "V_aerobic": "m^3",
                       "settler_area": "m^2"}
    numerical_method = "Equation-oriented: stiff integration (diffrax) then Newton (optimistix)."
    outlet_roles = ["effluent", "wastage"]

    def __init__(self, params: ActivatedSludgePlantParams):
        """Initialize.

        Args:
            params: :class:`ActivatedSludgePlantParams`.
        """
        self.params = params

    def __call__(self, influent: Stream) -> tuple[Stream, Stream, dict]:
        """Solve the plant.

        Returns:
            ``(effluent, wastage, info)``; ``info`` has the scalar results of
            :func:`~difflow_water.wastewater.bsm1.activated_sludge_plant`.
        """
        p = self.params
        a = ActivatedSludgeParams(
            volumes=(p.V_anoxic, p.V_anoxic, p.V_aerobic, p.V_aerobic, p.V_aerobic),
            KLa=(0.0, 0.0, p.KLa_3, p.KLa_4, p.KLa_5), Q_a=p.Q_a, Q_r=p.Q_r, Q_w=p.Q_w,
            takacs=TakacsParams(area=p.settler_area))
        r = activated_sludge_plant(influent, a)
        info = {k: v for k, v in r.items() if not isinstance(v, dict) and k not in
                ("effluent", "underflow", "wastage")}
        return r["effluent"], r["wastage"], info


def build_bsm1_flowsheet(params: ActivatedSludgePlantParams | None = None):
    """The BSM1 plant as a difflow ``Flowsheet`` of palette units, with two recycles.

    Influent ``influent``; units ``M1`` (mixer), ``R1``-``R5``
    (:class:`ASM1Reactor`), ``SP_a`` (internal recycle split), ``SET``
    (:class:`TakacsSettler`), ``SP_r`` (sludge recycle / wastage). Recycles
    ``int_rec_src -> int_rec`` and ``ext_rec_src -> ext_rec``. The split
    fractions follow BSM1's flows at its design influent (18446 m^3/d).

    Returns:
        A :class:`~difflow.flowsheet.Flowsheet` (feed ``influent`` set).
    """
    from difflow.flowsheet import Flowsheet, Unit
    from difflow.units.flash import Mixer, Splitter
    from difflow_water.wastewater.asm1 import BSM1_Q0, STATES, bsm1_influent

    p = params or ActivatedSludgePlantParams()
    species = ["H2O"] + list(STATES)
    fs = Flowsheet(species_order=species)
    fs.add_feed("influent", bsm1_influent())
    fs.add_unit(Unit("M1", Mixer(species), ["influent", "int_rec", "ext_rec"], ["r1_in"]))
    vols = (p.V_anoxic, p.V_anoxic, p.V_aerobic, p.V_aerobic, p.V_aerobic)
    klas = (0.0, 0.0, p.KLa_3, p.KLa_4, p.KLa_5)
    prev = "r1_in"
    for k in range(5):
        out = f"r{k + 1}_out"
        fs.add_unit(Unit(f"R{k + 1}", ASM1Reactor(ASM1ReactorParams(volume=vols[k], KLa=klas[k])),
                         [prev], [out]))
        prev = out
    Q1 = BSM1_Q0 + p.Q_a + p.Q_r
    split_a = Splitter(species)
    fs.add_unit(Unit("SP_a", lambda s: split_a(s, split_frac=p.Q_a / Q1)[:2],
                     [prev], ["int_rec_src", "settler_in"]))
    fs.add_unit(Unit("SET", TakacsSettler(TakacsSettlerParams(Q_underflow=p.Q_r + p.Q_w,
                                                               area=p.settler_area)),
                     ["settler_in"], ["effluent", "underflow"]))
    split_r = Splitter(species)
    fs.add_unit(Unit("SP_r", lambda s: split_r(s, split_frac=p.Q_r / (p.Q_r + p.Q_w))[:2],
                     ["underflow"], ["ext_rec_src", "wastage"]))
    fs.add_recycle("int_rec_src", "int_rec")
    fs.add_recycle("ext_rec_src", "ext_rec")
    return fs
