"""DWSIM 9.0.5 unit operations for the crude-side comparison: columns, and
the compounds and characterization details the harness does not expose.

An additive extension of :mod:`.dwsim_session` (which it imports and does not
change). Like the harness, nothing here loads .NET until it is called with a
live :class:`~.dwsim_session.DWSIMSession`, and it runs only in a generator
(``dwsim_cdu_generate.py``), never in the pytest process.

What it adds
------------
:class:`FlatCompound`
    A :class:`~.dwsim_session.HypoCompound` that DWSIM treats as a
    *database* compound (``IsHYPO = False``, ``OriginalDB = "DWSIM"``). The
    difference matters only for the heat of vaporization, and it is why this
    class exists. For a hypo, DWSIM's ``AUX_HVAPi`` is Watson's
    ``Hvap(Tb) ((1-Tr)/(1-Tbr))^0.375`` -- exponent fixed at 0.375; for a
    non-hypo ``"DWSIM"`` compound it is ``A (1-Tr)^(B + C Tr + D Tr^2)``
    (``A`` in J/kmol), which with ``B = 0.38``, ``C = D = 0`` is *exactly*
    difflow's Watson form (``ColumnThermo.dhvap``: ``A (1-T/Tc)^0.38``),
    except that difflow smooths ``1 - Tr`` near zero and DWSIM returns 0 at
    and above ``Tc``. Ideal-gas Cp (the polynomial) and vapour pressure
    (DIPPR 101 form) are read the same way for both (``AUX_CPi``,
    ``AUX_PVAPi`` in DWSIM's ``PropertyPackage.vb``), so with
    :data:`~.dwsim_session.IDEAL_RAOULT` DWSIM's Raoult's Law package then
    computes difflow's ``ColumnThermo`` model on difflow's constants.

:func:`distcurve_characterization`
    The harness's headless run of DWSIM's distillation-curve
    characterization, returning in addition what the UI editor keeps
    internally: each cut's volume fractions at its ends and middle
    (``fv0 fvf fvm``) and the temperature DWSIM gives it (``tbpm``, the TBP
    polynomial at the cut's MIDPOINT fraction), and the SG the cut had
    *before* DWSIM rescaled the SGs to the bulk gravity (the SG its MW was
    computed from).

:func:`property_methods`
    DWSIM's petroleum-fraction correlations (``PropertyMethods`` and
    ``PROPS.Cpig_lk``) evaluated at given ``(Tb, SG)`` -- a correlation
    comparison with the characterization pipeline out of the way.

:class:`DWSIMColumn`
    A DWSIM rigorous column with a condenser and an adiabatic bottom stage
    (what an atmospheric crude column is; see the class for why it is not
    DWSIM's own "refluxed absorber") on a
    :class:`~.dwsim_session.DWSIMFlowsheet`: feeds on any stage, liquid side
    draws at fixed molar rates, a total condenser with a distillate rate
    spec, stage pressures, the column solver by name. ``.solve()`` runs
    DWSIM and ``.solve_adiabatic()`` holds the bottom stage adiabatic; both
    return the stage profile, the products and the duties as plain numbers.

What a DWSIM rigorous column can NOT be (read from ``RigorousColumn.vb``,
DWSIM master, and confirmed against the 9.0.5 assemblies): it has no side
strippers and no pumparounds (a side operation enum exists, ``SideOp*``,
but nothing solves it), and no free-water phase: the column's K-values come
from ``DW_CalcKvalue`` on one liquid phase, and the flash setting
``ImmiscibleWaterOption`` does not reach the column solvers. Steam stripping
therefore condenses water INTO the hydrocarbon liquid wherever the water
partial pressure reaches its ideal-solution value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .dwsim_session import HypoCompound

# ---------------------------------------------------------------------------
# Compounds
# ---------------------------------------------------------------------------


@dataclass
class FlatCompound(HypoCompound):
    """A compound on given constants that DWSIM treats as a database entry.

    Attributes (beyond :class:`~.dwsim_session.HypoCompound`'s):
        hvap_A: Watson ``A`` (J/mol): ``Hvap = A (1 - T/Tc)^hvap_n``.
            Required: DWSIM returns zero without it.
        hvap_n: Watson exponent (difflow's 0.38).
    """

    hvap_A: float = 0.0
    hvap_n: float = 0.38

    def _constant_properties(self, cid: int):
        c = super()._constant_properties(cid)
        c.IsHYPO = False
        # AUX_HVAPi, OriginalDB "DWSIM", not hypo, not PF:
        #   A (1 - Tr)^(B + C Tr + D Tr^2) / MW / 1000 kJ/kg  -> A in J/kmol
        c.HVap_A = float(self.hvap_A) * 1000.0
        c.HVap_B = float(self.hvap_n)
        c.HVap_C = 0.0
        c.HVap_D = 0.0
        c.HVap_E = 0.0
        return c

    def as_dict(self) -> dict:
        d = super().as_dict()
        d.update(hvap_A=self.hvap_A, hvap_n=self.hvap_n, dwsim_kind="database-like (IsHYPO False)")
        return d


def eq101_constant(P: float) -> tuple:
    """DIPPR-101 coefficients of a vapour pressure that is ``P`` (Pa) at
    every temperature: ``ln P = A``. Used for a water that never condenses
    (difflow's water in the column: vapour only, never in the hydrocarbon
    liquid)."""
    return (math.log(P), 0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# DWSIM's characterization, with the internals the harness drops
# ---------------------------------------------------------------------------

def distcurve_characterization(session, tbp_K, cum_frac, sg_bulk, cut_temps_K,
                               basis="liquid_volume", Tc_corr="Riazi-Daubert (1985)",
                               Pc_corr="Riazi-Daubert (1985)", omega_corr="Lee-Kesler (1976)",
                               mw_corr="Winn (1956)", adjust_omega=True, adjust_rackett=True,
                               sg_curve=None, name="ASSAY") -> dict:
    """DWSIM's distillation-curve characterization (TBP curve, cut
    temperatures), as :meth:`DWSIMSession.petroleum_characterization`, plus
    the editor's per-cut bookkeeping.

    Returns:
        ``{"rows": [...], "cuts": [...]}``: ``rows`` are the harness's rows
        (light to heavy); ``cuts[i]`` has ``tbp0 tbpf fv0 fvf fvm tbpm`` (K
        and volume fractions of the curve) and ``sg_unscaled`` for the same
        cut.
    """
    clr = session._clr
    from System import Activator, Array, Boolean, Double, Int32, Object  # noqa: F401
    from System.Collections.Generic import List
    from System.Reflection import BindingFlags
    from System.Runtime.CompilerServices import RuntimeHelpers

    with session.cwd():
        for a in ("DWSIM.UI.Desktop.Shared", "DWSIM.UI.Desktop.Editors"):
            clr.AddReference(str(session.path / f"{a}.dll"))
        import DWSIM.UI.Desktop.Shared as Shared
        from DWSIM.SharedClasses.SystemsOfUnits import SI
        from DWSIM.UI.Desktop.Editors import DistCurvePCharacterization

        T = clr.GetClrType(DistCurvePCharacterization)
        obj = RuntimeHelpers.GetUninitializedObject(T)
        F = BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Instance

        def lst(xs):
            out = List[Double]()
            for x in xs or ():
                out.Add(float(x))
            return out

        tmp = T.GetNestedType("tmpcomp", BindingFlags.NonPublic | BindingFlags.Public)
        fields = {
            "tccol": Activator.CreateInstance(clr.GetClrType(List).MakeGenericType(tmp)),
            "ncomps": Int32(len(cut_temps_K) + 1),
            "Tccorr": Tc_corr, "Pccorr": Pc_corr, "AFcorr": omega_corr, "MWcorr": mw_corr,
            "cb": lst(cum_frac), "tbp": lst(tbp_K), "mwc": lst([]),
            "sgc": lst(sg_curve), "visc100": lst([]), "visc210": lst([]),
            "hasmwc": Boolean(False), "hassgc": Boolean(sg_curve is not None),
            "hasvisc100c": Boolean(False), "hasvisc210c": Boolean(False),
            "pseudomode": Int32(1), "cuttemps": lst(cut_temps_K),
            "pseudocuts": Int32(len(cut_temps_K) + 1),
            "mwb": Double(0.0),
            # the editor reads this field as API gravity (see dwsim_session)
            "sgb": Double(141.5 / float(sg_bulk) - 131.5),
            "tbpcurvetype": Int32(0),
            "curvebasis": Int32({"liquid_volume": 0, "mole": 1, "mass": 2}[basis]),
            "adjustAf": Boolean(adjust_omega), "adjustZR": Boolean(adjust_rackett),
            "decsep1": ".", "decsep2": ".", "assayname": name,
            "flowsheet": Shared.Flowsheet(),
        }
        for k, v in fields.items():
            T.GetField(k, F).SetValue(obj, v)
        comps = T.GetMethod("GenerateCompounds", F).Invoke(obj, Array[Object]([SI()]))
        T.GetMethod("CalculateMolarFractions", F).Invoke(obj, Array[Object]([comps]))
        rows = session._pf_rows(comps)
        # the ConstantProperties objects themselves, light to heavy, for
        # pf_flowsheet (not serialisable; callers drop them before JSON)
        cps = sorted((comps[k].ConstantProperties for k in comps.Keys), key=lambda c: float(c.NBP))
        tccol = T.GetField("tccol", F).GetValue(obj)
        cuts = []
        for tc in tccol:
            tt = tc.GetType()
            cuts.append({k: float(tt.GetField(k, F).GetValue(tc))
                         for k in ("tbp0", "tbpf", "fv0", "fvf", "fvm", "tbpm")})
    # the SG before DWSIM rescaled it to the bulk (what the MW correlation saw):
    # the editor's d15_Riazi of its Tb -> MW guess, when no SG curve is given
    if sg_curve is None:
        for c in cuts:
            Tb = c["tbpm"]
            mm = (1.0 / 0.01964 * (6.97996 - math.log(1080.0 - Tb))) ** 1.5 if Tb < 1080 else \
                (1.0 / 0.01964 * (6.97996 + math.log(Tb - 1080.0))) ** 1.5
            c["mw_guess"] = mm
            c["sg_unscaled"] = 1.07 - math.exp(3.56073 - 2.93886 * mm ** 0.1)
    rows[0]["settings"] = dict(curve="TBP", basis=basis, cut_temps_K=list(cut_temps_K),
                               Tc_corr=Tc_corr, Pc_corr=Pc_corr, omega_corr=omega_corr,
                               mw_corr=mw_corr, adjust_omega=adjust_omega,
                               adjust_rackett=adjust_rackett, sg_bulk=sg_bulk,
                               sg_curve=None if sg_curve is None else list(sg_curve))
    return {"rows": rows, "cuts": cuts, "_constant_properties": cps}


def pf_flowsheet(session, package: str, compounds: Sequence[str] = (), pf=(), kij="dwsim",
                 options=None, flash_tol: float = 1e-10, max_iter: int = 1000):
    """A one-stream DWSIM flowsheet on database ``compounds`` plus DWSIM's OWN
    petroleum fractions ``pf`` (``ConstantProperties`` objects from
    :func:`distcurve_characterization`, ``IsPF = True``: Lee-Kesler Cp and
    Psat, Chao-Seader parameters, fitted omega and Rackett parameter).

    :meth:`DWSIMSession.flowsheet` takes database names and hypos on given
    constants only; this is the same construction (and returns the same
    :class:`~.dwsim_session.DWSIMFlowsheet`) for DWSIM's own fractions.
    """
    from .dwsim_session import PROPERTY_PACKAGES, DWSIMFlowsheet

    pkg = PROPERTY_PACKAGES.get(package, package)
    with session.cwd():
        from DWSIM.Interfaces.Enums.GraphicObjects import ObjectType

        sim = session.automation.CreateFlowsheet()
        names = list(compounds)
        for c in pf:
            n = str(c.Name)
            if sim.AvailableCompounds.ContainsKey(n):
                sim.AvailableCompounds.Remove(n)
            sim.AvailableCompounds.Add(n, c)
            names.append(n)
        for n in names:
            sim.AddCompound(n)
        pp = sim.CreateAndAddPropertyPackage(pkg).__implementation__
        st = sim.AddObject(ObjectType.MaterialStream, 50, 50, "S1").GetAsObject()
        st = getattr(st, "__implementation__", st)
        st.PropertyPackage = pp
        pp.CurrentMaterialStream = st
        fsh = DWSIMFlowsheet(session, sim, pp, st, names, pkg, [], {})
        fsh._set_options(options or {})
        fsh._set_flash_settings(flash_tol, max_iter)
        fsh._set_kij(kij)
    return fsh


def no_ideal_fallback(fs):
    """Switch off DWSIM's silent fallbacks: by default a PV flash that fails
    is redone with ideal K-values (``PVFlash_TryIdealCalcOnFailure``) and the
    answer returned as the package's. With it on, a Lee-Kesler-Plocker
    bubble point of the crude came back equal to Raoult's to every digit."""
    from DWSIM.Interfaces.Enums import FlashSetting

    fs.pp.FlashSettings[FlashSetting.PVFlash_TryIdealCalcOnFailure] = "False"
    fs._flash_settings = {str(k): str(fs.pp.FlashSettings[k]) for k in fs.pp.FlashSettings.Keys}
    return fs


def phase_enthalpy(fs, comp: Sequence[float], T: float, P: float, phase: str) -> float:
    """DWSIM's own molar enthalpy (J/mol) of one phase of composition
    ``comp`` -- ``PropertyPackage.DW_CalcEnthalpy`` called directly, not the
    value a material stream reports."""
    import System
    from DWSIM.Thermodynamics.PropertyPackages import State

    with fs.session.cwd():
        pp = fs.pp
        pp.CurrentMaterialStream = fs.stream
        z = System.Array[float]([float(v) for v in comp])
        h = float(pp.DW_CalcEnthalpy(z, float(T), float(P),
                                     State.Vapor if phase == "vapor" else State.Liquid))
        mw = float(pp.AUX_MMM(z))
    return h * mw


def property_methods(session, Tb: Sequence[float], SG: Sequence[float],
                     T_cp: Sequence[float] = (300.0, 600.0, 800.0)) -> dict:
    """DWSIM's petroleum-fraction correlations at given ``(Tb [K], SG)``.

    Every option of DWSIM's characterization editor that takes only
    ``(Tb, SG)``: Tc (Riazi-Daubert 1985, Riazi 2005, Lee-Kesler 1976), Pc
    (Riazi-Daubert 1985, Lee-Kesler 1976), MW (Winn 1956, Riazi 1986,
    Lee-Kesler 1974); the Lee-Kesler acentric factor on each (Tc, Pc) pair;
    and the Lee-Kesler ideal-gas Cp (``PROPS.Cpig_lk``, kJ/kg/K) from the
    Watson K and the Riazi-Daubert-based omega, at ``T_cp``.
    """
    with session.cwd():
        from DWSIM.Thermodynamics.PropertyPackages.Auxiliary import PROPS
        from DWSIM.Thermodynamics.Utilities.PetroleumCharacterization.Methods import PropertyMethods as PM

        out = {k: [] for k in ("Tc_RiaziDaubert", "Pc_RiaziDaubert", "Tc_Riazi", "Tc_LeeKesler",
                               "Pc_LeeKesler", "MW_Winn", "MW_Riazi", "MW_LeeKesler",
                               "omega_LeeKesler_RD", "omega_LeeKesler_LK", "watson_K",
                               "cp_ig_lk_RD")}
        for tb, sg in zip(Tb, SG):
            tb, sg = float(tb), float(sg)
            tc, pc = float(PM.Tc_RiaziDaubert(tb, sg)), float(PM.Pc_RiaziDaubert(tb, sg))
            tcl, pcl = float(PM.Tc_LeeKesler(tb, sg)), float(PM.Pc_LeeKesler(tb, sg))
            out["Tc_RiaziDaubert"].append(tc)
            out["Pc_RiaziDaubert"].append(pc)
            out["Tc_Riazi"].append(float(PM.Tc_Riazi(tb, sg)))
            out["Tc_LeeKesler"].append(tcl)
            out["Pc_LeeKesler"].append(pcl)
            mw_w = float(PM.MW_Winn(tb, sg))
            out["MW_Winn"].append(mw_w)
            out["MW_Riazi"].append(float(PM.MW_Riazi(tb, sg)))
            out["MW_LeeKesler"].append(float(PM.MW_LeeKesler(tb, sg)))
            w = float(PM.AcentricFactor_LeeKesler(tc, pc, tb))
            out["omega_LeeKesler_RD"].append(w)
            out["omega_LeeKesler_LK"].append(float(PM.AcentricFactor_LeeKesler(tcl, pcl, tb)))
            kw = (1.8 * tb) ** 0.33333 / sg      # DWSIM's PF_Watson_K (exponent 0.33333)
            out["watson_K"].append(kw)
            # Cpig_lk is kJ/kg/K (a cut's J/mol/K is this times its MW)
            out["cp_ig_lk_RD"].append([float(PROPS.Cpig_lk(kw, w, float(T))) for T in T_cp])
        out["T_cp"] = list(map(float, T_cp))
    return out


# ---------------------------------------------------------------------------
# The rigorous column
# ---------------------------------------------------------------------------

#: DWSIM's column solvers, by the substring its ``Column.Calculate`` matches.
SOLVERS = {"naphtali-sandholm": "Napthali-Sandholm (Simultaneous Correction)",
           "wang-henke": "Wang-Henke (Bubble Point)",
           "modified-wang-henke": "Modified Wang-Henke (Bubble Point)"}


class DWSIMColumn:
    """A DWSIM rigorous column whose bottom stage is an ordinary adiabatic
    stage: an atmospheric crude column (condenser, no reboiler).

    Why not DWSIM's own "refluxed absorber" (a ``DistillationColumn`` with
    ``RefluxedAbsorber = True``, no reboiler) or a reboiler duty spec of zero:
    both fail in DWSIM 9.0.5's Naphtali-Sandholm solver, for reasons read from
    its source (``NewtonRaphson.vb``, DWSIM master) and reproduced here on a
    five-component column:

    * refluxed absorber + total condenser: the condenser's vapour is set to
      zero, the distillate is then taken as that vapour's sum (zero), and the
      condenser equilibrium rows divide by it -- NaN on the first function
      evaluation ("Error evaluating error functions").
    * ``DistillationColumn`` + a ``Heat_Duty`` reboiler spec: the reboiler's
      energy balance is replaced by ``spec_function / spec_value`` and the
      heat-duty branch never sets the spec function, so the row is ``0/Q``:
      NaN for ``Q = 0`` and an empty equation otherwise (the solver then
      stops at its iteration cap).

    So this builds a ``DistillationColumn`` whose "reboiler" is the bottom
    stage, specifies its TEMPERATURE (a spec the solver handles), and finds
    by secant the bottom temperature at which DWSIM's computed reboiler duty
    is zero (:meth:`solve_adiabatic`). At the answer every stage satisfies
    DWSIM's own MESH equations with no heat added -- the column difflow
    solves -- to the secant's tolerance on the duty.

    Args:
        fs: A :class:`~.dwsim_session.DWSIMFlowsheet` (its property package
            and compounds are the column's).
        n_stages: Equilibrium stages, 1 (top) to ``n_stages`` (bottom); the
            condenser is DWSIM's stage 0 and extra.
        P_top, P_bottom: Stage 1 and stage ``n_stages`` pressures (Pa),
            linear in between (difflow's convention).
        P_condenser: Condenser pressure (Pa).
        solver: A key of :data:`SOLVERS`.
    """

    def __init__(self, fs, n_stages: int, P_top: float, P_bottom: float, P_condenser: float,
                 solver: str = "naphtali-sandholm", max_iter: int = 200, tol: float = 1e-8):
        from DWSIM.Interfaces.Enums.GraphicObjects import ObjectType
        from DWSIM.UnitOperations.UnitOperations import Column

        self.fs = fs
        self.session = fs.session
        self.sim = fs.sim
        self.n = int(n_stages)
        with self.session.cwd():
            col = self.sim.AddObject(ObjectType.DistillationColumn, 300, 50, "COLUMN").GetAsObject()
            col = getattr(col, "__implementation__", col)
            self.col = col
            col.SetNumberOfStages(self.n + 1)          # + the condenser
            col.CondenserType = Column.condtype.Total_Condenser
            col.PropertyPackage = fs.pp
            col.SolvingMethodName = SOLVERS[solver]
            col.MaxIterations = int(max_iter)
            col.InternalLoopTolerance = float(tol)
            col.ExternalLoopTolerance = float(tol)
            col.CondenserDeltaP = 0.0
            stages = list(col.Stages)
            stages[0].P = float(P_condenser)
            for j in range(1, self.n + 1):
                frac = (j - 1) / max(self.n - 1, 1)
                stages[j].P = float(P_top + (P_bottom - P_top) * frac)
        self.P = [float(s.P) for s in stages]
        self._out_port = 2
        self.feeds = {}
        self.draws = {}
        self._streams = {}
        self.history = []

    # -- streams --------------------------------------------------------------

    def _stream(self, name):
        from DWSIM.Interfaces.Enums.GraphicObjects import ObjectType

        st = self.sim.AddObject(ObjectType.MaterialStream, 50, 50 + 20 * len(self._streams),
                                name).GetAsObject()
        st = getattr(st, "__implementation__", st)
        st.PropertyPackage = self.fs.pp
        self._streams[name] = st
        return st

    def add_feed(self, name: str, stage: int, flows: Sequence[float], T: float, P: float):
        """A feed of molar ``flows`` (mol/s, in ``fs.names`` order) at ``T``
        and ``P`` to ``stage`` (1-based)."""
        import System
        from DWSIM.Interfaces.Enums import StreamSpec

        f = np.asarray(flows, float)
        with self.session.cwd():
            st = self._stream(name)
            st.SetOverallComposition(System.Array[float]([float(v) for v in f / f.sum()]))
            st.SetMolarFlow(float(f.sum()))
            st.SetTemperature(float(T))
            st.SetPressure(float(P))
            st.SpecType = StreamSpec.Temperature_and_Pressure
            self.col.ConnectFeed(st, int(stage))
        self.feeds[name] = dict(stage=int(stage), flows=f.tolist(), T=float(T), P=float(P))
        return st

    def add_side_draw(self, name: str, stage: int, rate: float, phase: str = "L"):
        """A side draw of ``rate`` mol/s from ``stage`` (liquid by default)."""
        from DWSIM.UnitOperations.UnitOperations.Auxiliary.SepOps import StreamInformation

        with self.session.cwd():
            st = self._stream(name)
            self.sim.ConnectObjects(self.col.GraphicObject, st.GraphicObject, self._out_port, 0)
            self._out_port += 1
            si = StreamInformation()
            si.ID = st.Name               # DWSIM's object id, not the tag
            si.StreamID = st.Name
            si.AssociatedStage = self.col.Stages[int(stage)].Name
            si.StreamBehavior = StreamInformation.Behavior.Sidedraw
            si.StreamType = StreamInformation.Type.Material
            si.StreamPhase = getattr(StreamInformation.Phase, phase)
            si.FlowRate.Value = float(rate)
            si.FlowRate.Unit = "mol/s"
            self.col.MaterialStreams.Add(st.Name, si)
        self.draws[name] = dict(stage=int(stage), rate=float(rate), phase=phase)
        return st

    def connect_products(self, distillate="DIST", bottoms="BTMS"):
        from DWSIM.Interfaces.Enums.GraphicObjects import ObjectType

        with self.session.cwd():
            self.col.ConnectDistillate(self._stream(distillate))
            self.col.ConnectBottoms(self._stream(bottoms))
            for nm, conn in (("QC", self.col.ConnectCondenserDuty), ("QR", self.col.ConnectReboilerDuty)):
                q = self.sim.AddObject(ObjectType.EnergyStream, 400, 10, nm).GetAsObject()
                conn(getattr(q, "__implementation__", q))

    def specs(self, distillate_rate: float, T_bottom: float):
        """Distillate molar rate (mol/s) and bottom-stage temperature (K)."""
        with self.session.cwd():
            self.col.SetCondenserSpec("Product_Molar_Flow_Rate", float(distillate_rate), "mol/s", "")
            self.col.SetReboilerSpec("Temperature", float(T_bottom), "K", "")
        self.spec = dict(distillate_rate=float(distillate_rate), T_bottom=float(T_bottom))

    def estimates(self, T=None, V=None, L=None):
        """Initial estimates, stage 0 (condenser) to ``n_stages``."""
        import System

        with self.session.cwd():
            if T is not None:
                self.col.SetInitialTemperatureEstimates(System.Array[float]([float(v) for v in T]))
                self.col.UseTemperatureEstimates = True
            if V is not None:
                self.col.SetInitialVaporMolarFlowEstimates(System.Array[float]([float(v) for v in V]))
                self.col.UseVaporFlowEstimates = True
            if L is not None:
                self.col.SetInitialLiquidMolarFlowEstimates(System.Array[float]([float(v) for v in L]))
                self.col.UseLiquidFlowEstimates = True

    def _warm_start(self):
        """Start the next solve from this one's solution (DWSIM's own)."""
        with self.session.cwd():
            last = self.col.GetLastSolution()
            if last is not None:
                self.col.SetInitialEstimates(last)
                for f in ("UseTemperatureEstimates", "UseVaporFlowEstimates",
                          "UseLiquidFlowEstimates", "UseCompositionEstimates"):
                    setattr(self.col, f, True)

    # -- solve ------------------------------------------------------------------

    def solve(self) -> dict:
        import time

        t0 = time.time()
        with self.session.cwd():
            errs = self.session.automation.CalculateFlowsheet4(self.sim)
        errors = [str(e)[:400] for e in errs] if errs is not None else []
        out = self.results() if not errors else {"converged": False}
        out["errors"] = errors
        out["seconds"] = round(time.time() - t0, 2)
        return out

    def solve_adiabatic(self, T0: float, T1: float, duty_tol: float = 10.0,
                        max_outer: int = 20, warm_start: bool = False) -> dict:
        """Secant on the bottom-stage temperature spec until DWSIM's reboiler
        duty is zero to ``duty_tol`` W. Each solve starts from the same
        estimates (DWSIM's own, or those given) unless ``warm_start``; a
        solve that fails is retried halfway back to the last temperature
        that converged (a secant step can leave DWSIM's region of
        convergence). Returns the last :meth:`solve` result plus ``outer``
        (the (T, duty) history, failures included)."""
        good = []
        T = float(T0)
        nxt = float(T1)
        res = None
        for k in range(max_outer):
            with self.session.cwd():
                self.col.SetReboilerSpec("Temperature", float(T), "K", "")
            res = self.solve()
            if not res["converged"]:
                self.history.append({"T_bottom": T, "reboiler_duty_W": None,
                                     "errors": [e[:200] for e in res["errors"]],
                                     "seconds": res["seconds"]})
                print(f"    outer {k}: T_bottom {T:.6f} K failed ({res['seconds']} s)", flush=True)
                if not good:
                    raise RuntimeError(f"DWSIM column failed at T_bottom={T}: {res['errors']}")
                T = 0.5 * (T + good[-1][0])
                continue
            Q = res["reboiler_duty_W"]
            good.append((T, Q))
            self.history.append({"T_bottom": T, "reboiler_duty_W": Q, "seconds": res["seconds"]})
            print(f"    outer {k}: T_bottom {T:.6f} K, reboiler duty {Q:.3f} W "
                  f"({res['seconds']} s)", flush=True)
            if abs(Q) < duty_tol:
                break
            if len(good) < 2:
                T = nxt
            else:
                (Ta, Qa), (Tb, Qb) = good[-2], good[-1]
                T = Tb - Qb * (Tb - Ta) / (Qb - Qa) if Qb != Qa else Tb + 0.5
                T = min(max(T, Tb - 15.0), Tb + 15.0)
            if warm_start:
                self._warm_start()
        else:
            raise RuntimeError(f"bottom-stage secant did not reach {duty_tol} W: {self.history}")
        res["outer"] = list(self.history)
        self.spec = dict(getattr(self, "spec", {}), T_bottom=good[-1][0])
        return res

    def results(self) -> dict:
        c = self.col

        def arr(a):
            return [float(v) for v in a] if a is not None else None

        T, V, L = arr(c.Tf), arr(c.Vf), arr(c.Lf)
        if T is None or len(T) == 0:
            return {"converged": False}
        x = [[float(v) for v in r] for r in c.xf]
        y = [[float(v) for v in r] for r in c.yf]
        LSS, VSS = arr(c.LSSf), arr(c.VSSf)
        n = self.n
        prods = {"distillate": [LSS[0] * v for v in x[0]],
                 "bottoms": [L[n] * v for v in x[n]]}
        for name, d in self.draws.items():
            j = d["stage"]
            prods[name] = ([LSS[j] * v for v in x[j]] if d["phase"] == "L"
                           else [VSS[j] * v for v in y[j]])
        return {
            "converged": True,
            "names": list(self.fs.names),
            "T": T, "P": [float(s.P) for s in c.Stages], "V": V, "L": L,
            "LSS": LSS, "VSS": VSS, "x": x, "y": y,
            "condenser_duty_W": float(c.CondenserDuty) * 1000.0,    # kW -> W
            "reboiler_duty_W": float(c.ReboilerDuty) * 1000.0,
            "products": prods,
        }

    def provenance(self) -> dict:
        c = self.col
        return {"object": "DistillationColumn, total condenser, bottom stage held adiabatic "
                          "(temperature spec found by secant on zero reboiler duty)",
                "column_class": str(c.GetType().FullName),
                "solver": str(c.SolvingMethodName),
                "max_iterations": int(c.MaxIterations),
                "tolerances": [float(c.InternalLoopTolerance), float(c.ExternalLoopTolerance)],
                "n_stages": self.n, "stage_pressures": self.P,
                "feeds": {k: {kk: vv for kk, vv in v.items() if kk != "flows"}
                          for k, v in self.feeds.items()},
                "side_draws": self.draws, "specs": getattr(self, "spec", None),
                "outer_iterations": self.history}
