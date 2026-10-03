"""DWSIM 9.0.5 reactors from Python: Gibbs, equilibrium and conversion.

An additive extension of :mod:`.dwsim_session` (which it does not modify):
:class:`DWSIMSession` gives a flowsheet with one material stream; the
functions here add a reactor, its outlet streams and an energy stream to
that flowsheet, run it, and read the answer back as plain numbers. Like the
rest of the DWSIM harness it runs only in a reference **generator**
(``dwsim_reactions_generate.py``); tests read the JSON.

What DWSIM's reactors are (read from the IL of ``DWSIM.UnitOperations``
9.0.5 and ``PropertyPackage.AUX_DELGF_T``; worth knowing before reading a
number they produce):

* **Formation Gibbs energy at T.** Both the Gibbs and the equilibrium reactor
  take each compound's ``G_f(T)`` from ``AUX_DELGF_T(298.15, T)``::

      G_f(T)/RT = G_f/(R T0) + H_f/R (1/T - 1/T0)
                  + int_T0^T Cp dT/(R T) - int_T0^T Cp/T dT/R

  with ``R = 8.314``, ``G_f`` and ``H_f`` the database's 25 C ideal-gas
  values and the integrals by the midpoint rule (``AUX_INT_CPDTi``, as in
  the harness's enthalpy). It is the compound's own Cp, not the formation
  reaction's ``dCp``; the element terms cancel in a balanced reaction, so
  reaction free energies are exact Gibbs-Helmholtz integrals -- the same
  route as difflow's ``H - T S`` with absolute entropies.
* **Standard-state pressure.** Both reactors divide pressures by
  ``P0 = 101325 Pa`` (``Reactor_Gibbs.FunctionValue2N``,
  ``Reactor_Equilibrium.P0``), one atmosphere, while difflow's
  equilibrium constants are on 1 bar. For a reaction that changes the
  moles of gas by ``dn`` the two conventions differ by ``dn ln(1.01325)``
  in ``ln K`` (0.039 for ``Bz + 3 H2 = CH``) unless the database's ``G_f``
  are themselves 1-atm values -- which convention ChemSep's are on is not
  stated in DWSIM; the generator records both readings.
* **Gibbs reactor.** Minimises ``sum n_i mu_i`` over the selected compounds
  subject to element balances (C, H, O, S, N from each compound's formula),
  ``mu_i/RT = G_f,i(T)/RT + ln x_i + ln phi_i + ln(P/P0)``, phi from the
  property package. IPOPT is DWSIM's default solver but its native library
  is not in the 9.0.5 Linux package (``libIpopt39``): the process aborts.
  :func:`run_gibbs` switches it off (DWSIM's own Newton/"DirectMinimization"
  path) and sets ``InitializeFromPreviousSolution = False`` (otherwise DWSIM
  raises "invalid initial estimates" on a first solve).
* **Equilibrium reactor.** Reactions with ``K`` from the same ``G_f(T)``
  (``KOpt.Gibbs``), basis fugacity, extents solved by DWSIM's nested loops.
* **Conversion reactor.** Reactions by rank; a reaction's conversion is a
  percentage of its base compound *present when its rank runs* (measured:
  40 % then 50 % of the product of the first). Its heat is
  ``sum_j xi_j dH_r,j(25 C)`` from the formation enthalpies, plus the
  stream enthalpies of the property package, so its energy balance is a
  state function: any reaction set giving the same outlet gives the same
  outlet temperature.
* Energy streams are in kW; flows here are mol/s and heats W.
* **Check an equilibrium answer before believing it** (measured on the
  reaction reference): the equilibrium reactor raises "Solution led to
  negative mole fractions" on larger networks or an absent product (give
  every species a trace), fails in adiabatic mode, and once converged
  silently 6e-2 off; the Gibbs reactor can leave a minor species at zero or
  at its start, 5e-6 to 1e-2 off, with no error, and with inert species
  stops ~1e-4 short. ``tests/refinery/_dwsim_rx_compare.acceptance`` is
  the check (the equilibrium of DWSIM's own G_f(T)).

Hypothetical compounds for reactions: :class:`ThermoHypo` is the harness's
:class:`~.dwsim_session.HypoCompound` plus the two things a reactor needs and
the harness's hypo leaves out -- an element formula and a Gibbs energy of
formation -- so DWSIM can run a reactor on difflow's ``H_f``, ``S`` and Cp
exactly.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np

from .dwsim_session import HypoCompound

#: DWSIM's gas constant in AUX_DELGF_T and its reactors (J/mol/K).
R_DWSIM = 8.314
#: Standard-state pressure of DWSIM's Gibbs and equilibrium reactors (Pa).
P0_DWSIM = 101325.0
T0 = 298.15
#: Element entropies for building a hypo's Gibbs energy of formation from an
#: absolute entropy: graphite (CODATA 5.74 J/mol/K) and, per H atom, half of
#: hydrogen's. Any values cancel in a balanced reaction PROVIDED hydrogen the
#: species and hydrogen the element agree; :func:`gibbs_of_formation` takes
#: hydrogen's from the caller's own H2 entropy for that reason.
S_GRAPHITE = 5.74


def gibbs_of_formation(Hf: float, S: float, formula: Mapping[str, float], S_H2: float,
                       S_elements: Optional[Mapping[str, float]] = None) -> float:
    """``G_f(298.15) = H_f - T0 (S - sum_e n_e S_e)`` (J/mol) from an absolute
    entropy; ``S_e`` per atom: carbon :data:`S_GRAPHITE`, hydrogen
    ``S_H2/2`` (so the species H2 has ``G_f = 0`` exactly)."""
    Se = {"C": S_GRAPHITE, "H": S_H2 / 2.0}
    Se.update(S_elements or {})
    return float(Hf - T0 * (S - sum(n * Se[e] for e, n in formula.items())))


@dataclass
class ThermoHypo(HypoCompound):
    """A :class:`HypoCompound` with an element formula and ``Gf`` (J/mol,
    ideal gas, 25 C), so DWSIM's reactors can use it."""

    formula: Mapping[str, float] = field(default_factory=dict)
    Gf: float = 0.0

    def as_dict(self) -> dict:
        d = super().as_dict()
        d.update(formula=dict(self.formula), Gf=self.Gf)
        return d

    def _constant_properties(self, cid: int):
        c = super()._constant_properties(cid)
        c.IG_Gibbs_Energy_of_Formation_25C = float(self.Gf) / float(self.MW)   # kJ/kg
        c.Elements.Clear()
        for e, n in self.formula.items():
            c.Elements.Add(e, int(n) if float(n).is_integer() else n)
        c.Formula = "".join(f"{e}{int(n)}" for e, n in self.formula.items())
        return c


def _impl(o):
    return getattr(o, "__implementation__", o)


def _net_dict(d):
    from System.Collections.Generic import Dictionary

    D = Dictionary[str, float]()
    for k, v in d.items():
        D.Add(str(k), float(v))
    return D


def pure_functions(fs, name: str, T: float) -> dict:
    """DWSIM's ideal-gas functions of one compound at ``T`` (J/mol units):
    ``Hf``/``Gf`` (database, 25 C), ``cp``, ``h_sens`` (= int Cp dT from
    298.15 K, midpoint rule), ``s_sens`` (int Cp/T dT) and ``g_f`` (=
    ``AUX_DELGF_T * R T MW``, the formation Gibbs energy DWSIM's reactors use)."""
    pp = fs.pp
    with fs.session.cwd():
        pp.CurrentMaterialStream = fs.stream
        c = fs.sim.SelectedCompounds[name]
        mw = float(c.Molar_Weight)
        return {"T": float(T), "MW": mw,
                "Hf": float(c.IG_Enthalpy_of_Formation_25C) * mw,
                "Gf": float(c.IG_Gibbs_Energy_of_Formation_25C) * mw,
                "cp": float(pp.AUX_CPi(name, float(T))) * mw,
                "h_sens": float(pp.AUX_INT_CPDTi(T0, float(T), name)) * mw,
                "s_sens": float(pp.AUX_INT_CPDT_Ti(T0, float(T), name)) * mw,
                "g_f": float(pp.AUX_DELGF_T(T0, float(T), name, False)) * R_DWSIM * float(T) * mw,
                "elements": {str(k): float(c.Elements[k]) for k in c.Elements.Keys},
                "db": str(c.OriginalDB)}


class _Rig:
    """Inlet stream (the flowsheet's ``S1``) -> reactor -> vapour + liquid
    outlets, with an energy stream."""

    def __init__(self, fs, object_type):
        from DWSIM.Interfaces.Enums.GraphicObjects import ObjectType

        sim = fs.sim
        self.fs = fs
        self.inlet = fs.stream
        add = lambda t, tag: _impl(sim.AddObject(t, 0, 0, tag).GetAsObject())  # noqa: E731
        self.vap = add(ObjectType.MaterialStream, "OUT_V")
        self.liq = add(ObjectType.MaterialStream, "OUT_L")
        self.q = add(ObjectType.EnergyStream, "Q")
        self.rx = add(getattr(ObjectType, object_type), "R")
        sim.ConnectObjects(self.inlet.GraphicObject, self.rx.GraphicObject, -1, -1)
        sim.ConnectObjects(self.rx.GraphicObject, self.vap.GraphicObject, 0, 0)
        sim.ConnectObjects(self.rx.GraphicObject, self.liq.GraphicObject, 1, 0)
        sim.ConnectObjects(self.q.GraphicObject, self.rx.GraphicObject, 0, 1)
        for o in (self.vap, self.liq, self.rx):
            o.PropertyPackage = fs.pp

    def feed(self, F: Sequence[float], T: float, P: float):
        import System
        from DWSIM.Interfaces.Enums import StreamSpec

        F = np.asarray(F, float)
        s = self.inlet
        s.SetOverallComposition(System.Array[float]([float(v) for v in F / F.sum()]))
        s.SetMolarFlow(float(F.sum()))
        s.SetTemperature(float(T))
        s.SetPressure(float(P))
        s.SpecType = StreamSpec.Temperature_and_Pressure

    def mode(self, T_out: Optional[float]):
        from DWSIM.UnitOperations.Reactors import OperationMode

        if T_out is None:
            self.rx.ReactorOperationMode = OperationMode.Adiabatic
        else:
            self.rx.ReactorOperationMode = OperationMode.OutletTemperature
            self.rx.OutletTemperature = float(T_out)

    def run(self) -> dict:
        fs = self.fs
        with fs.session.cwd():
            errs = fs.session.automation.CalculateFlowsheet4(fs.sim)
        errs = [str(e).split("\n")[0] for e in errs] if errs is not None else []
        out = {"errors": errs, "names": list(fs.names)}

        def flows(st):
            n = float(st.GetMolarFlow() or 0.0)
            if not math.isfinite(n) or n <= 0.0:
                return [0.0] * len(fs.names), 0.0
            return [n * float(v) for v in st.GetOverallComposition()], n

        Fv, nv = flows(self.vap)
        Fl, nl = flows(self.liq)
        try:
            vf_in = float(self.inlet.GetPhase("Vapor").Properties.molarfraction)
        except Exception:  # noqa: BLE001
            vf_in = None
        out.update(inlet_vapor_fraction=vf_in, T=float(self.vap.GetTemperature()), P=float(self.vap.GetPressure()),
                   F_vapor=Fv, F_liquid=Fl, F=[a + b for a, b in zip(Fv, Fl)],
                   Q=float(self.q.EnergyFlow or 0.0) * 1000.0,
                   T_liquid=float(self.liq.GetTemperature()) if nl > 0 else None)
        return out


def run_gibbs(fs, F, T_in: float, P: float, T_out: Optional[float] = None,
              components: Optional[Sequence[str]] = None, phase: str = "CalculateEquilibrium",
              element_matrix=None, tol: float = 1e-12, max_iter: int = 1000,
              alternate: bool = False) -> dict:
    """DWSIM's Gibbs reactor on the flowsheet's inlet. ``T_out=None`` is
    adiabatic, else the outlet temperature is fixed. ``components`` (DWSIM
    names) are the reacting ones, default all; ``phase`` is
    ``Reactor_Gibbs.ReactivePhaseType`` (``"Vapor"`` restricts the
    minimisation to one vapour phase)."""
    import System
    from System.Collections.Generic import List

    with fs.session.cwd():
        rig = _Rig(fs, "RCT_Gibbs")
        rx = rig.rx
        rig.feed(F, T_in, P)
        rig.mode(T_out)
        ids = List[str]()
        for n in (components or fs.names):
            ids.Add(n)
        rx.ComponentIDs = ids
        rx.CreateElementMatrix()
        if element_matrix is not None:
            labels, rows = element_matrix
            M = System.Array.CreateInstance(System.Double, len(labels), len(rows[0]))
            for i, row in enumerate(rows):
                for j, v in enumerate(row):
                    M[i, j] = float(v)
            rx.Elements = System.Array[str]([str(e) for e in labels])
            rx.ElementMatrix = M
        rx.InitializeFromPreviousSolution = False
        rx.UseIPOPTSolver = False
        rx.ReactivePhaseBehavior = System.Enum.Parse(rx.ReactivePhaseBehavior.GetType(), phase)
        rx.InternalTolerance = float(tol)
        rx.MaximumInternalIterations = int(max_iter)
        rx.AlternateSolvingMethod = bool(alternate)
    out = rig.run()
    with fs.session.cwd():
        out.update(elements=[str(e) for e in rx.Elements],
                   reacting=[str(c) for c in rx.ComponentIDs], phase=phase,
                   final_gibbs=float(rx.FinalGibbsEnergy),
                   element_balance=float(rx.ElementBalance), kind="gibbs")
    return out


def _reaction_set(fs, reactions, kind: str):
    sim = fs.sim
    rs = sim.CreateReactionSet(f"{kind}_set", "")
    sim.AddReactionSet(rs)
    made = []
    for rank, r in enumerate(reactions):
        if kind == "equilibrium":
            rx = sim.CreateEquilibriumReaction(r["name"], "", _net_dict(r["nu"]), r["base"], "Vapor",
                                               r.get("basis", "Fugacity"), "", 0.0, "")
        else:
            rx = sim.CreateConversionReaction(r["name"], "", _net_dict(r["nu"]), r["base"],
                                              r.get("phase", "Vapor"), repr(float(r["conversion"])))
        sim.AddReaction(rx)
        sim.AddReactionToSet(rx.ID, rs.ID, True, int(r.get("rank", rank)))
        made.append(rx)
    return rs, made


def run_equilibrium(fs, reactions: Sequence[Mapping], F, T_in: float, P: float,
                    T_out: Optional[float] = None, tol: float = 1e-10, max_iter: int = 1000) -> dict:
    """DWSIM's equilibrium reactor. Each reaction: ``{"name", "nu": {dwsim
    name: coeff}, "base"}``; ``K`` from DWSIM's Gibbs energies, vapour phase,
    fugacity basis."""
    with fs.session.cwd():
        rs, made = _reaction_set(fs, reactions, "equilibrium")
        rig = _Rig(fs, "RCT_Equilibrium")
        rx = rig.rx
        rx.ReactionSetID = rs.ID
        rig.feed(F, T_in, P)
        rig.mode(T_out)
        for k, v in (("InternalLoopTolerance", tol), ("ExternalLoopTolerance", tol)):
            setattr(rx, k, float(v))
        rx.InternalLoopMaximumIterations = int(max_iter)
        rx.ExternalLoopMaximumIterations = int(max_iter)
        with contextlib.suppress(Exception):
            rx.UseIPOPTSolver = False
    out = rig.run()
    with fs.session.cwd():
        ext = rx.ReactionExtents
        byid = {str(r.ID): r for r in made}
        out.update(kind="equilibrium",
                   extents={str(byid[str(k)].Name) if str(k) in byid else str(k): float(ext[k])
                            for k in ext.Keys} if ext is not None else {},
                   reaction_heat_298={str(r.Name): float(r.ReactionHeat) for r in made},
                   reaction_gibbs_298={str(r.Name): float(r.ReactionGibbsEnergy) for r in made})
    return out


def run_conversion(fs, reactions: Sequence[Mapping], F, T_in: float, P: float,
                   T_out: Optional[float] = None) -> dict:
    """DWSIM's conversion reactor. Each reaction: ``{"name", "nu", "base",
    "conversion"}`` (percent of the base compound present when its rank
    runs), ranks in list order unless ``"rank"`` is given."""
    with fs.session.cwd():
        rs, made = _reaction_set(fs, reactions, "conversion")
        rig = _Rig(fs, "RCT_Conversion")
        rig.rx.ReactionSetID = rs.ID
        rig.feed(F, T_in, P)
        rig.mode(T_out)
    out = rig.run()
    with fs.session.cwd():
        out.update(kind="conversion",
                   reaction_heat_298={str(r.Name): float(r.ReactionHeat) for r in made},
                   stoich_balance={str(r.Name): float(r.StoichBalance) for r in made})
    return out


def sequential_conversions(names: Sequence[str], F_in: Sequence[float], F_out: Sequence[float],
                           formula: Mapping[str, Mapping[str, float]], pivot: str,
                           hydrogen: str) -> list[dict]:
    """Conversion reactions that take ``F_in`` exactly to ``F_out``.

    Each hydrocarbon ``CxHy`` is written against a ``pivot`` (methane) and
    hydrogen: ``CxHy + (2x - y/2) H2 = x CH4``. Every species that is
    consumed goes forward first (base: itself, conversion the share of it
    consumed), then every species that is made comes back from the pivot
    (base: the pivot). The energy balance of a conversion reactor is a state
    function, so this reproduces the heat of whatever network produced
    ``F_out``; the generator checks DWSIM's outlet against ``F_out``.
    """
    F = dict(zip(names, map(float, F_in)))
    dF = {n: float(b) - float(a) for n, a, b in zip(names, F_in, F_out)}
    out = []

    def nu_of(n):
        x = formula[n].get("C", 0.0)
        y = formula[n].get("H", 0.0)
        return x, 2.0 * x - y / 2.0

    for n in names:
        if n in (pivot, hydrogen) or dF[n] >= 0.0:
            continue
        x, h2 = nu_of(n)
        conv = -dF[n] / F[n]
        out.append({"name": f"c_{n}", "nu": {n: -1.0, hydrogen: -h2, pivot: x}, "base": n,
                    "conversion": 100.0 * conv})
        F[n] -= -dF[n]
        F[hydrogen] -= h2 * (-dF[n])
        F[pivot] += x * (-dF[n])
    for n in names:
        if n in (pivot, hydrogen) or dF[n] <= 0.0:
            continue
        x, h2 = nu_of(n)
        conv = x * dF[n] / F[pivot]
        out.append({"name": f"m_{n}", "nu": {pivot: -x, n: 1.0, hydrogen: h2}, "base": pivot,
                    "conversion": 100.0 * conv})
        F[pivot] -= x * dF[n]
        F[hydrogen] += h2 * dF[n]
        F[n] += dF[n]
    return out


__all__ = ["R_DWSIM", "P0_DWSIM", "T0", "S_GRAPHITE", "ThermoHypo", "gibbs_of_formation",
           "pure_functions", "run_gibbs", "run_equilibrium", "run_conversion",
           "sequential_conversions"]
