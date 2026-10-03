"""DWSIM unit operations for the light-ends references: a rigorous
distillation column, a compressor train with intercoolers and knock-out
drums, and DWSIM's Reid vapour pressure.

An additive companion to :mod:`.dwsim_session` (which it does not modify):
every function takes a :class:`.dwsim_session.DWSIMFlowsheet` -- one
flowsheet on one property package, with its material stream ``fs.stream``
-- adds DWSIM objects to it, solves the flowsheet with DWSIM's own
sequential solver and returns plain numbers (SI, J/mol, W). Like the
harness, it is imported only by generators; nothing here loads .NET until a
:class:`~.dwsim_session.DWSIMSession` exists.

What DWSIM does, read from ``DWSIM.UnitOperations.dll`` 9.0.5 (IL of
``Column.Calculate``, ``DistillationColumn.SetCondenserSpec`` /
``SetReboilerSpec``) and checked against the IDAES debutanizer:

* **Column.** ``DistillationColumn`` numbers its stages from 0 (the
  condenser) to ``N - 1`` (the reboiler); ``ConnectFeed(stream, j)`` feeds
  the WHOLE stream (both phases) to stage ``j``, so difflow's tray ``k``
  (1-based from the top) is DWSIM's stage ``k``. ``SetCondenserSpec("Reflux
  Ratio", R)`` and ``SetReboilerSpec("Boilup Ratio", B)`` both become a
  ``Stream_Ratio`` spec: ``L/D`` at the condenser, ``V/B`` at the reboiler
  -- the definitions difflow and IDAES use. Solver names are matched by
  substring: ``"Wang-Henke"`` (contains "Bubble"), ``"Napthali"`` (sic;
  Naphtali-Sandholm simultaneous correction), ``"Rates"`` (Burningham-Otto
  sum rates). After a solve DWSIM checks every component's balance against
  ``10 * min(loop tolerances)`` and raises if it fails; with tolerances of
  1e-10 that check (1e-9) fails on a converged column (measured 1.8e-9 on
  the debutanizer), so the references use 1e-9 (check at 1e-8) and record
  the column's own balance error.
* **Duties** are kW on DWSIM's energy streams; ``CondenserDuty`` is the heat
  removed (positive), ``ReboilerDuty`` reads negative on the column object
  and positive on its energy stream. Returned here in W, both positive.
* **Compressor** in "OutletPressure" mode, adiabatic path, its
  ``AdiabaticEfficiency`` in PERCENT (0.75 there is 0.75 %, a 133-fold
  power): the isentropic
  outlet from a PS flash at the discharge pressure, the actual enthalpy
  ``h1 + (h_s - h1)/eta`` and the outlet from a PH flash -- difflow's
  :class:`~difflow_refinery.gasplant.GasCompressor` construction.
* **Cooler** in "OutletTemperature" mode; **Vessel** (separator) flashes its
  inlet at the inlet's pressure and enthalpy ("Legacy" mode), vapour out of
  connector 0, liquid out of 1.
* **RVP.** DWSIM's only Reid vapour pressure is in the classic Windows UI's
  "Petroleum Cold Flow Properties" utility (``DWSIM.FrmColdProperties.
  Update1`` in ``DWSIM.exe``): ``TVP`` is the stream package's bubble
  pressure at 310.95 K (100 F) and ``RVP = 6894.76 * 10**((ln(TVP/6894.76)
  + 12.972789480266567 - 12.82) / 2.773803987779386)`` -- a correlation on
  the TVP, not the D323 construction. :func:`dwsim_rvp_from_tvp` is that
  line; the form (``ln`` inside a power of ten) is as read from the IL.
  :func:`d323_rvp` is difflow's D323 construction rebuilt on DWSIM's
  flashes (vapour space four times the liquid's volume at 100 F, the
  liquid's molar volume from DWSIM's PR liquid root at 1 bar).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

#: DWSIM's column solvers by the substring ``Column.Calculate`` matches.
COLUMN_SOLVERS = {"wang-henke": "Wang-Henke (Bubble Point)",
                  "naphtali-sandholm": "Napthali-Sandholm (Simultaneous Correction)",
                  "burningham-otto": "Burningham-Otto (Sum Rates)"}

#: 100 F in K, as DWSIM's cold-flow utility writes it.
T_100F = 310.95


def _impl(o):
    return getattr(o, "__implementation__", o)


def add(fs, kind: str, name: str):
    """Add a DWSIM object (``ObjectType`` name) to ``fs`` on its property package."""
    from DWSIM.Interfaces.Enums.GraphicObjects import ObjectType

    with fs.session.cwd():
        o = _impl(fs.sim.AddObject(getattr(ObjectType, kind), 100 + 60 * len(fs.sim.SimulationObjects),
                                   100, name).GetAsObject())
        if kind != "EnergyStream":
            o.PropertyPackage = fs.pp
    return o


def connect(fs, a, b, i: int = -1, j: int = -1):
    """Connect ``a`` to ``b`` (DWSIM picks free connectors with -1)."""
    fs.sim.ConnectObjects(a.GraphicObject, b.GraphicObject, i, j)


def set_kij(fs, K) -> None:
    """Give a flowsheet made with ``kij="zero"`` the ``(C, C)`` matrix ``K``.

    A workaround: :meth:`.dwsim_session.DWSIMFlowsheet._set_kij` with an
    array removes and adds pairs in one loop over ``(i, j)``, so the removal
    for ``(j, i)`` deletes the pair it added for ``(i, j)`` and every kij
    comes out zero (it then raises "DWSIM did not take the kij"). Here the
    pairs are added after the harness has removed DWSIM's, then checked.
    """
    from System.Collections.Generic import Dictionary

    K = np.asarray(K, float)
    n = len(fs.names)
    if K.shape != (n, n):
        raise ValueError(f"K must be {n}x{n}")
    ip = fs._ip_stores()[0]
    ipcls = None
    for a in ip.Keys:
        for b in ip[a].Keys:
            ipcls = type(ip[a][b])
            break
        if ipcls is not None:
            break
    for i, a in enumerate(fs.names):
        for j, b in enumerate(fs.names):
            if i < j and K[i, j] != 0.0:
                if not ip.ContainsKey(a):
                    ip.Add(a, Dictionary[str, ipcls]())
                d = ipcls()
                d.kij = float(K[i, j])
                ip[a][b] = d
    fs._refresh_kij()
    got = fs.kij()
    off = ~np.eye(n, dtype=bool)
    if not np.allclose(got[off], K[off], atol=1e-12):
        raise RuntimeError(f"DWSIM did not take the kij: asked {K}, has {got}")
    fs._kij_mode = "given"


def set_feed(fs, stream, z: Sequence[float], F: float, T: float, P: float):
    """Specify a material stream by temperature and pressure (molar flow mol/s)."""
    import System
    from DWSIM.Interfaces.Enums import StreamSpec

    z = np.asarray(z, float)
    with fs.session.cwd():
        stream.SetOverallComposition(System.Array[float]([float(v) for v in z / z.sum()]))
        stream.SetMolarFlow(float(F))
        stream.SetTemperature(float(T))
        stream.SetPressure(float(P))
        stream.SpecType = StreamSpec.Temperature_and_Pressure


def solve(fs):
    """Solve the flowsheet; raise with DWSIM's messages if anything failed."""
    with fs.session.cwd():
        errs = fs.session.automation.CalculateFlowsheet4(fs.sim)
    msgs = [str(e) for e in errs] if errs is not None else []
    if msgs:
        raise RuntimeError("DWSIM flowsheet failed: " + " | ".join(m.splitlines()[0] for m in msgs))


def stream_state(fs, s) -> dict:
    """Flow (mol/s), component flows, T, P, vapour fraction and molar enthalpy of a stream."""
    F = float(s.GetMolarFlow())
    z = [float(v) for v in s.GetOverallComposition()]
    ph = s.GetPhase("Vapor").Properties
    mix = s.GetPhase("Mixture").Properties
    mw = float(mix.molecularWeight) if mix.molecularWeight is not None else None
    h = float(mix.enthalpy) * mw if (mix.enthalpy is not None and mw) else None
    return {"F": F, "flows": {n: F * zi for n, zi in zip(fs.names, z)}, "z": z,
            "T": float(s.GetTemperature()), "P": float(s.GetPressure()),
            "vapor_fraction": float(ph.molarfraction or 0.0), "h": h}


# ---------------------------------------------------------------------------
# Distillation column
# ---------------------------------------------------------------------------

def column(fs, *, z, F, T, P, n_trays: int, feed_tray: int, reflux_ratio: float,
           boilup_ratio: float, solver: str = "wang-henke", tol: float = 1e-9,
           max_iter: int = 500, dP: float = 0.0) -> dict:
    """A rigorous DWSIM ``DistillationColumn``: total condenser at the bubble
    point (no subcooling), ``n_trays`` equilibrium trays, kettle reboiler,
    one feed (``fs.stream``) on tray ``feed_tray`` (1-based from the top),
    reflux and boilup ratios specified, top pressure ``P``, no pressure drop
    unless ``dP`` (Pa per stage).

    Returns ``stages`` (condenser to reboiler: ``T P L V x y K``; ``L``/``V``
    the liquid and vapour leaving the stage, mol/s), ``distillate`` and
    ``bottoms`` (:func:`stream_state`), ``condenser_duty`` and
    ``reboiler_duty`` (W, both positive), the specs as DWSIM computed them,
    the solver settings and the column's component balance closure.
    """
    feed = fs.stream
    set_feed(fs, feed, z, F, T, P)
    D = add(fs, "MaterialStream", "DIST")
    B = add(fs, "MaterialStream", "BTMS")
    QC = add(fs, "EnergyStream", "QC")
    QR = add(fs, "EnergyStream", "QR")
    col = add(fs, "DistillationColumn", "COL")
    with fs.session.cwd():
        col.SetNumberOfStages(int(n_trays) + 2)
        col.ConnectFeed(feed, int(feed_tray))
        col.ConnectDistillate(D)
        col.ConnectBottoms(B)
        col.ConnectCondenserDuty(QC)
        col.ConnectReboilerDuty(QR)
        col.SetTopPressure(float(P))
        col.ColumnPressureDrop = float(dP) * (int(n_trays) + 1)
        col.CondenserDeltaP = 0.0
        col.TotalCondenserSubcoolingDeltaT = 0.0
        col.SetCondenserSpec("Reflux Ratio", float(reflux_ratio), "", "")
        col.SetReboilerSpec("Boilup Ratio", float(boilup_ratio), "", "")
        col.SolvingMethodName = COLUMN_SOLVERS[solver]
        col.MaxIterations = int(max_iter)
        col.InternalLoopTolerance = float(tol)
        col.ExternalLoopTolerance = float(tol)
    solve(fs)
    names = fs.names
    sol = _final_solution(col)
    stages = []
    for j, st in enumerate(col.Stages):
        stages.append({"name": str(st.Name), "T": sol["Tf"][j], "P": float(st.P),
                       "L": sol["Lf"][j], "V": sol["Vf"][j], "x": sol["xf"][j], "y": sol["yf"][j],
                       "K": sol["Kf"][j], "T_stage_object": float(st.T)})
    d, b = stream_state(fs, D), stream_state(fs, B)
    fz = np.asarray(z, float) / np.sum(z) * F
    out_fl = np.array([d["flows"][n] + b["flows"][n] for n in names])
    return {
        "stages": stages, "distillate": d, "bottoms": b,
        "condenser_duty": 1000.0 * float(QC.EnergyFlow),
        "reboiler_duty": 1000.0 * float(QR.EnergyFlow),
        "reflux_ratio": stages[0]["L"] / d["F"],
        "boilup_ratio": stages[-1]["V"] / b["F"],
        "component_balance": float(np.max(np.abs(out_fl - fz) / np.maximum(fz, 1e-300))),
        "iterations": {"internal": int(sol["ic"]), "external": int(sol["ec"])},
        "settings": {"solver": COLUMN_SOLVERS[solver], "tolerance": tol, "max_iter": max_iter,
                     "condenser": str(col.CondenserType), "stages": int(n_trays) + 2,
                     "feed_stage_index": int(feed_tray), "subcooling_K": 0.0,
                     "column_pressure_drop": float(col.ColumnPressureDrop)},
    }


def _final_solution(col) -> dict:
    """The converged column's profiles: ``Column.Tf``, ``Lf``, ``Vf`` (liquid
    and vapour leaving each stage, mol/s; the condenser's ``L`` is the
    reflux), ``xf``, ``yf``, ``Kf`` (per stage, in component order), and the
    iteration counters ``ic``/``ec`` (Wang-Henke counts; Naphtali-Sandholm
    leaves them 0). (``Stage.Lout``/``Vout``/``Kvalues`` are not filled in by
    a solve in 9.0.5.)"""
    from System.Reflection import BindingFlags

    F = BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Instance
    t = col.GetType()

    def field(key):
        tt = t
        while tt is not None:
            f = tt.GetField(key, F)
            if f is not None:
                return f.GetValue(col)
            tt = tt.BaseType
        raise AttributeError(key)

    out = {}
    for key in ("Tf", "Lf", "Vf"):
        out[key] = [float(v) for v in field(key)]
    for key in ("xf", "yf", "Kf"):
        arr = field(key)
        out[key] = [[float(v) for v in arr[j]] for j in range(arr.Count)]
    for key in ("ic", "ec"):
        out[key] = int(field(key))
    return out


# ---------------------------------------------------------------------------
# Compressor train
# ---------------------------------------------------------------------------

def compressor_train(fs, *, z, F, T, P, outlet_P: float, n_stages: int, efficiency: float,
                     cooler_T: float, cooler_dP: float = 0.0) -> dict:
    """Inlet knock-out drum, then ``n_stages`` of (adiabatic compressor at an
    equal pressure ratio -> cooler to ``cooler_T`` -> knock-out drum).

    Returns per stage: compressor ``power`` (W), ``discharge_T``, the drum's
    vapour and liquid (:func:`stream_state`); and the inlet drum's.
    """
    feed = fs.stream
    set_feed(fs, feed, z, F, T, P)
    ratio = (outlet_P / P) ** (1.0 / n_stages)
    drum0 = add(fs, "Vessel", "KO0")
    v0 = add(fs, "MaterialStream", "V0")
    l0 = add(fs, "MaterialStream", "L0")
    connect(fs, feed, drum0, -1, 0)
    connect(fs, drum0, v0, 0, -1)
    connect(fs, drum0, l0, 1, -1)
    stages = []
    gas = v0
    Pk = P
    objs = []
    for k in range(1, n_stages + 1):
        c = add(fs, "Compressor", f"K{k}")
        e = add(fs, "EnergyStream", f"W{k}")
        dis = add(fs, "MaterialStream", f"DIS{k}")
        cool = add(fs, "Cooler", f"E{k}")
        q = add(fs, "EnergyStream", f"Q{k}")
        cin = add(fs, "MaterialStream", f"C{k}")
        drum = add(fs, "Vessel", f"KO{k}")
        v = add(fs, "MaterialStream", f"V{k}")
        lq = add(fs, "MaterialStream", f"L{k}")
        import System

        connect(fs, gas, c, -1, 0)
        connect(fs, e, c, -1, 1)
        connect(fs, c, dis, 0, -1)
        connect(fs, dis, cool, -1, 0)
        connect(fs, cool, cin, 0, -1)
        connect(fs, cool, q, -1, -1)
        connect(fs, cin, drum, -1, 0)
        connect(fs, drum, v, 0, -1)
        connect(fs, drum, lq, 1, -1)
        P2 = Pk * ratio
        with fs.session.cwd():
            c.CalcMode = System.Enum.Parse(clr_type(c, "CalcMode"), "OutletPressure")
            c.ProcessPath = System.Enum.Parse(clr_type(c, "ProcessPath"), "Adiabatic")
            c.POut = float(P2)
            c.AdiabaticEfficiency = 100.0 * float(efficiency)      # DWSIM takes percent
            cool.CalcMode = System.Enum.Parse(clr_type(cool, "CalcMode"), "OutletTemperature")
            cool.OutletTemperature = float(cooler_T)
            cool.DeltaP = float(cooler_dP)
        objs.append((c, e, dis, cool, q, v, lq))
        gas = v
        Pk = P2 - cooler_dP
    solve(fs)
    for c, e, dis, cool, q, v, lq in objs:
        stages.append({"power": 1000.0 * float(e.EnergyFlow), "discharge_T": float(dis.GetTemperature()),
                       "discharge_P": float(dis.GetPressure()), "cooler_duty": 1000.0 * float(q.EnergyFlow),
                       "vapor": stream_state(fs, v), "liquid": stream_state(fs, lq),
                       "efficiency_percent": float(c.AdiabaticEfficiency)})
    return {"ratio": ratio, "inlet_drum": {"vapor": stream_state(fs, v0), "liquid": stream_state(fs, l0)},
            "stages": stages}


def clr_type(obj, prop: str):
    """The .NET type of ``obj.prop`` (for ``System.Enum.Parse``)."""
    return getattr(obj, prop).GetType()


# ---------------------------------------------------------------------------
# Vapour pressure
# ---------------------------------------------------------------------------

def dwsim_rvp_from_tvp(tvp: float) -> float:
    """DWSIM's RVP (Pa) from a true vapour pressure (Pa): the line of
    ``FrmColdProperties.Update1`` (DWSIM 9.0.5 classic UI), transcribed."""
    psi = 6894.76
    return psi * 10.0 ** ((math.log(tvp / psi) + 12.972789480266567 - 12.82) / 2.773803987779386)


def liquid_Z(fs, z, T: float, P: float) -> float:
    """DWSIM's PR liquid compressibility factor of ``z`` at ``(T, P)`` (its
    ``m_pr.Z_PR(..., "L")``, the root its liquid phase uses), whatever the
    stable phase."""
    import System

    pp = fs.pp
    n = len(fs.names)
    zz = np.asarray(z, float) / np.sum(z)
    K = fs.kij()
    Kij = System.Array.CreateInstance(System.Double, n, n)
    for i in range(n):
        for j in range(n):
            Kij[i, j] = float(K[i, j])
    c = fs.constants()
    arr = lambda v: System.Array[float]([float(x) for x in v])  # noqa: E731
    with fs.session.cwd():
        return float(pp.m_pr.Z_PR(float(T), float(P), arr(zz), Kij, arr([r["Tc"] for r in c]),
                                  arr([r["Pc"] for r in c]), arr([r["omega"] for r in c]), "L"))


def d323_rvp(fs, z, T: float = T_100F, vl_ratio: float = 4.0, P_liquid: float = 1e5,
             tol: float = 1e-12, max_iter: int = 60) -> dict:
    """difflow's D323 construction on DWSIM's thermodynamics.

    Per mole of liquid charged, the vapour space is ``vl_ratio`` times the
    liquid's molar volume ``v_L = Z_L R T / P_liquid`` (DWSIM's PR liquid
    root at ``P_liquid``). Find the vapour fraction ``beta`` such that a
    DWSIM T-VF flash at ``(T, beta)`` has ``beta = P vl_ratio v_L / (Z_V R
    T)`` (R cancels). Returns the pressure, ``beta``, ``Z_L`` and DWSIM's
    bubble pressure at ``T`` (the TVP).
    """
    ZL = liquid_Z(fs, z, T, P_liquid)
    tvp = fs.flash_tvf(z, T, 0.0)

    def resid(beta):
        r = fs.flash_tvf(z, T, beta)
        return beta - r["P"] * vl_ratio * ZL / (r["Z_vap"] * P_liquid), r

    a, fa = 1e-4, resid(1e-4)[0]
    b, fb = 0.05, resid(0.05)[0]
    r = None
    for _ in range(max_iter):
        c = b - fb * (b - a) / (fb - fa)
        c = min(max(c, 1e-8), 0.5)
        fc, r = resid(c)
        a, fa, b, fb = b, fb, c, fc
        if abs(fc) < tol:
            break
    return {"rvp": r["P"], "beta": b, "Z_L": ZL, "residual": abs(fb), "tvp": tvp["P"],
            "y": r["y"], "x": r["x"], "Z_vap": r["Z_vap"]}
