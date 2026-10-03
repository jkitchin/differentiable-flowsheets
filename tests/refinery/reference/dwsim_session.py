"""A shared harness for comparing ``difflow_refinery`` with DWSIM 9.0.5.

DWSIM (https://dwsim.org) is a .NET process simulator. This module drives it
from Python through pythonnet's CoreCLR runtime, so a reference generator can
build a DWSIM flowsheet on *the same component constants* difflow uses, flash
it, and write plain numbers to a JSON file. Like every other reference in
this package, DWSIM runs only in the **generator**: the tests read the
committed JSON and never import this module's DWSIM half (nothing here loads
.NET until :class:`DWSIMSession` is constructed).

Install (see ``scripts/install_dwsim.sh``; GitHub releases are not reachable
from the build box, SourceForge is)::

    DWSIM 9.0.5 .deb  -> dpkg-deb -x to a directory of your choice
    apt-get install dotnet-runtime-8.0
    pip install pythonnet
    export DWSIM_PATH=<that directory>/usr/local/lib/dwsim

The API, in one place
---------------------
``DWSIMSession(path=None)``
    Loads CoreCLR and DWSIM's assemblies (once per process; see Gotchas) and
    opens an ``Automation3`` instance. Raises :class:`DWSIMUnavailable` with
    the install recipe when DWSIM, .NET or pythonnet is missing.

    * ``.version`` / ``.provenance()`` -- DWSIM, .NET, pythonnet versions.
    * ``.property_packages()`` -- the names DWSIM 9.0.5 offers.
    * ``.has_compound(name)`` / ``.compound(name)`` -- a database compound's
      constants (``MW Tc Pc omega Tb Zc Vc Hf`` in SI/J-mol units, ``db``,
      ``cas``).
    * ``.compounds()`` / ``.find_by_cas(cas)`` -- the database (1488
      compounds: ChemSep 556, ChEDL Thermo 706, CoolProp 90, ...).
    * ``.flowsheet(package="PR", compounds=(), hypos=(), overrides=None,
      kij="zero", options=None, flash_tol=1e-10, max_iter=1000)`` ->
      :class:`DWSIMFlowsheet` (one material stream; ``compounds`` are DWSIM
      database names -- :data:`difflow.dwsim_import.DWSIM_NAMES` maps
      difflow's -- then ``hypos``, in that order).
    * ``.petroleum_characterization(tbp_K, cum_frac, sg_bulk, mw_bulk=None,
      n_cuts=10, cut_temps_K=None, curve="TBP", basis="liquid_volume", ...)``
      -> ``list[dict]``, light to heavy: DWSIM's OWN distillation-curve
      characterization. Each row: ``name mole_fraction mass_fraction Tb SG
      MW Tc Pc omega Zc Vc Z_rackett watson_K cp_ig_300 cp_ig_600
      Hf_dwsim_raw is_pf``; ``rows[0]["settings"]`` records the inputs.
      Feed a row back as ``HypoCompound(name, MW=r["MW"], Tc=r["Tc"], ...)``
      to put DWSIM's cuts into difflow, or the other way round.
    * ``.bulk_characterization(mw, sg, tb=None, n=6, ...)`` -> the same rows:
      DWSIM's bulk C7+ characterization.

:class:`HypoCompound`
    A hypothetical (pseudo) compound for DWSIM, given by name, ``MW``,
    ``Tc``, ``Pc``, ``omega`` and optionally ``Tb``, ``SG``, an ideal-gas Cp
    polynomial and a vapour-pressure correlation (see "Hypothetical
    compounds" below). ``HypoCompound.from_gas_components(comps)`` makes one
    per component of a :class:`difflow_refinery.gasplant.GasComponents`, so
    DWSIM computes on difflow's constants exactly.

:class:`DWSIMFlowsheet`
    * ``.names`` -- component order of every vector in and out.
    * ``.flash_tp(z, T, P)``, ``.flash_ph(z, P, h)`` (``h`` J/mol),
      ``.flash_pvf(z, P, vf)``, ``.flash_tvf(z, T, vf)`` -> a flat dict:
      ``T P vapor_fraction liquid_fraction liquid2_fraction z x y x2 K h h_vap
      h_liq s rho_vap rho_liq Z_vap Z_liq MW_vap MW_liq phi_vap phi_liq
      equilibrium_residual phases``; ``x``/``y``/``K``/``phi_*`` are ``None``
      for a phase that is absent, ``phi_*`` are DWSIM's fugacity
      coefficients at its own phase compositions, ``equilibrium_residual``
      is ``max |ln(y/x) - ln(phi_L/phi_V)|`` (how converged DWSIM's own
      flash is), ``h*`` in J/mol
      (ideal gas at 298.15 K is zero, no heat of formation -- difflow's
      reference), ``rho`` kg/m^3.
    * ``.kij()`` -> the ``(C, C)`` matrix DWSIM is actually using.
    * ``.pure(name, T)`` -> ``cp_ig`` (J/mol/K), ``psat`` (Pa) and ``hvap``
      (J/mol) as DWSIM evaluates them for one component.
    * ``.provenance()`` -- package, flash algorithm and its settings, kij
      mode, component table as DWSIM holds it.

Property packages in DWSIM 9.0.5
--------------------------------
:data:`PROPERTY_PACKAGES` maps short keys to DWSIM's names. Measured with
``.property_packages()``: Peng-Robinson (PR), Peng-Robinson 1978 (PR78),
Peng-Robinson 1978 (PR78) Advanced, Peng-Robinson-Stryjek-Vera 2 (PRSV2-M /
PRSV2-VL), Soave-Redlich-Kwong (SRK), SRK Advanced, Lee-Kesler-Plöcker,
Grayson-Streed, Chao-Seader, Raoult's Law, NRTL, UNIQUAC, UNIFAC, UNIFAC-LL,
Modified UNIFAC (Dortmund), Modified UNIFAC (NIST), Wilson, PC-SAFT, GERG-2008,
CoolProp (+ incompressible), Steam Tables (IAPWS-IF97), Seawater IAPWS-08,
Black Oil, Ideal Solution (Aqueous Electrolytes), ThermoC Bridge, CAPE-OPEN.

What the cubic packages are (read from DWSIM.Thermodynamics' IL, 9.0.5):

* ``"Peng-Robinson (PR)"`` uses the **1976** ``kappa`` for every omega (no
  1978 branch above omega 0.49) -- the same form as
  :data:`difflow_refinery.gasplant.thermo.PR`. ``R = 8.314`` J/mol/K in the
  EOS, so departure enthalpies differ from difflow's (``R = 8.314462618``) by
  5.6e-5 relative; ``A`` and ``B``, hence Z, phi and K, do not depend on R.
* DWSIM loads **its own kij** for every pair it has (H2/propane -0.131,
  CO2/H2S 0.0978, ...). ``kij="zero"`` (the default here) removes them so
  the comparison runs at kij = 0 like difflow's references;
  ``kij="dwsim"`` keeps DWSIM's; an array sets them.
* DWSIM's PR fugacity routine (``ThermoPlugs.PR.CalcLnFugCPU``) writes
  the log term's ``1 + sqrt 2``, ``1 - sqrt 2``, ``2 sqrt 2`` as 2.414213,
  -0.414213, 2.828426 (its Z cubic and enthalpy use the exact values):
  1.4e-6 in a liquid phi. ``tests/refinery/test_dwsim_smoke.py``
  (``_ln_phi_dwsim``) reproduces DWSIM's phi to 2e-12 with that change.
* The ideal-gas enthalpy is the **midpoint-rule** integral of Cp from
  298.15 K (``AUX_INT_CPDTi``: ``round(|dT|/10)`` intervals clipped to
  10..100), not the analytic one: ~1e-5 of ``H_ig`` for a naphtha cut at
  400 K (``_dwsim_h_ig`` in the smoke test reproduces it).
* The default flash is DWSIM's "Universal Flash" over Nested Loops, with
  loop tolerances of **1e-4**. :meth:`DWSIMSession.flowsheet` tightens
  ``PTFlash_*`` / ``PHFlash_*`` tolerances to ``flash_tol`` (1e-10) and
  raises the iteration caps. Even so the answer can sit up to 1.6e-5 (in
  ``ln K``) from DWSIM's own equilibrium condition -- every flash result
  reports this as ``equilibrium_residual``; hold a comparison to a few
  times it, not to the tolerance.
* Package defaults that are NOT the textbook model (all recorded by
  ``DWSIMFlowsheet.provenance()["options"]``): liquid density is Rackett
  + experimental data with Peneloux translation, not the EOS
  (:data:`EOS_ONLY` switches to the EOS); "Raoult's Law" has a Poynting
  factor and Henry's law for supercritical components switched on
  (:data:`IDEAL_RAOULT` switches them off; on a C6/C7 hypo pair at 1.1 bar
  DWSIM's K equals ``Psat/P`` to 1e-12 either way). Lee-Kesler-Plöcker's
  ``k_ij`` is multiplicative (``Tc_ij = k_ij sqrt(Tci Tcj)``) and DWSIM
  takes a missing (zero) one as 1, so ``kij="zero"`` there means "no
  interaction" too; an explicit array is passed to LKP as given.

Hypothetical compounds
----------------------
A :class:`HypoCompound` becomes a ``DWSIM.Thermodynamics.BaseClasses.
ConstantProperties`` with ``IsHYPO = True``, ``IsPF = False`` and
``OriginalDB = "DWSIM"``, added to the flowsheet's ``AvailableCompounds``
and then selected like a database compound. What DWSIM then uses for it
(from the IL of ``PropertyPackage.AUX_CPi`` / ``AUX_PVAPi``):

* **Ideal-gas Cp**: for ``OriginalDB == "DWSIM"`` the polynomial
  ``(A + B T + C T^2 + D T^3 + E T^4) / MW`` kJ/kg/K, i.e. ``A..E`` in
  J/mol/K -- exactly difflow's ``cp`` cubic with ``E = 0``. Given that,
  DWSIM's ideal-gas enthalpy is difflow's. (A petroleum fraction --
  ``IsPF = True``, what DWSIM's own characterization makes -- instead gets
  the Lee-Kesler Cp correlation from its Watson K and omega,
  ``PROPS.Cpig_lk``.)
* **Vapour pressure**: ``exp(A + B/T + C ln T + D T^E)`` Pa (DIPPR 101
  form) for ``OriginalDB == "DWSIM"``. With no coefficients that is 1 Pa
  at every T, and DWSIM's flash *starts* from ``K = Psat/P`` -- so a hypo
  always gets coefficients: those given, or else the **Lee-Kesler**
  correlation on its Tc, Pc, omega written exactly in that form
  (:func:`lee_kesler_eq101`). For a cubic EOS this only seeds the flash
  (a converged EOS flash does not depend on it); for Raoult's Law it IS the
  model, so pass the Psat difflow uses. (``IsPF`` compounds use
  ``PROPS.Pvp_leekesler`` directly.)
* **Enthalpy of vaporization**: ``HVap_A ((1 - Tr)/(1 - Tbr))^0.375``
  (Watson) for ``OriginalDB == "DWSIM"``, ``HVap_A`` the value at Tb. With
  none DWSIM returns **zero** for a hypo, so the hypo always sets it: the
  given ``hvap_tb``, else Vetere's estimate from Tc, Pc, Tb
  (``HYP.DHvb_Vetere`` -- DWSIM's own fallback for its databases). A hypo
  always gets a Tb (given, else where its Psat is 1 atm). The cubic EOS
  packages never read Hvap (enthalpy is ideal gas + EOS departure);
  Raoult's Law and the activity packages do (``H_L = H_ig - Hvap``).
* **Liquid density, transport**: the Rackett parameter (``Z_rackett``,
  default Zc by Pitzer's ``0.2905 - 0.085 omega``) is all a hypo carries;
  use :data:`EOS_ONLY` for a cubic comparison. Heat of formation defaults
  to 0 (DWSIM's stream enthalpy does not include it).

DWSIM's own characterization
----------------------------
``petroleum_characterization`` runs the distillation-curve method of DWSIM's
cross-platform UI (``DWSIM.UI.Desktop.Editors.DistCurvePCharacterization.
GenerateCompounds``) headless: the editor is created without its
constructor (no Eto/GTK) and its data fields are filled in. Defaults are
the UI's: Tc and Pc Riazi-Daubert (1985), omega Lee-Kesler (1976), MW Winn
(1956), acentric factors adjusted to reproduce each cut's NBP on PR, Rackett
parameters adjusted to its SG. Two facts read from the IL, worth knowing:

* With no SG curve, cut SGs come from the curve shape and are then scaled
  by one factor so their **mass-fraction-weighted** mean is the bulk SG.
* The editor converts its bulk-SG field as if it held **API gravity**
  (``SG = 141.5/(131.5 + value)``); entering an SG in the UI therefore
  targets an SG near 1.07. This function passes ``141.5/SG - 131.5`` so the
  target is the SG you give (``ui_sg_bug=True`` reproduces the UI).

The pseudo-components it returns are DWSIM's petroleum fractions
(``IsPF``): Cp by Lee-Kesler from Watson K, Psat by Lee-Kesler.

Gotchas
-------
* **One CoreCLR per process.** pythonnet can load a runtime once; load
  CoreCLR *before* anything imports ``clr`` (Mono is the Linux default and
  DWSIM 9 needs .NET 8). Run each generator as its own process -- that is
  also why DWSIM never runs inside the pytest process.
* **Working directory.** Not needed for anything here in 9.0.5 (the
  Automation layer and the calculator start from any directory, verified);
  :class:`DWSIMSession` still ``chdir``\\ s to the DWSIM directory and back
  around its own calls, defensively, since DWSIM resolves some optional
  data files relative to it. Calling DWSIM objects yourself, use
  ``with session.cwd():``.
* **Threads.** DWSIM parallelizes flashes and solves on its own thread
  pool; the session switches parallel and GPU processing off
  (``GlobalSettings.Settings``) so results are deterministic. Do not call a
  session from more than one Python thread.
* **Interfaces.** pythonnet returns objects typed by the interface the
  method declares (``IPropertyPackage``, ``ISimulationObject``); the
  concrete object is ``obj.__implementation__``.
* **Speed.** Start-up (CoreCLR + DWSIM + compound databases) is ~4 s;
  the first flash of a flowsheet ~0.2 s, then 2-5 ms per flash; the
  distillation-curve characterization ~1 s.
* **The hypo name space.** ``AvailableCompounds`` is per flowsheet but a
  hypo with a database compound's name would shadow it; prefix hypo names
  (the smoke generator uses ``DF_``).
* **Heats of formation of DWSIM's petroleum fractions** come out 1000x
  off its database compounds' basis (a C6 cut reads -1.99 "kJ/kg");
  rows carry the raw number only, as ``Hf_dwsim_raw``.
"""

from __future__ import annotations

import contextlib
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

#: Where the build box keeps DWSIM 9.0.5 (``scripts/install_dwsim.sh``);
#: ``DWSIM_PATH`` overrides it.
DEFAULT_DWSIM_PATH = ("/tmp/claude-0/-home-user-differentiable-flowsheets/"
                      "78ada8bb-bcb2-5a4d-8f9f-7ec49d26275f/scratchpad/dwsim/root/usr/local/lib/dwsim")

#: Short keys -> DWSIM 9.0.5 property package names.
PROPERTY_PACKAGES: dict[str, str] = {
    "PR": "Peng-Robinson (PR)",
    "PR78": "Peng-Robinson 1978 (PR78)",
    "SRK": "Soave-Redlich-Kwong (SRK)",
    "RAOULT": "Raoult's Law",
    "LKP": "Lee-Kesler-Plöcker",
    "GS": "Grayson-Streed",
    "CS": "Chao-Seader",
    "PRSV2": "Peng-Robinson-Stryjek-Vera 2 (PRSV2-M)",
    "NRTL": "NRTL",
    "UNIQUAC": "UNIQUAC",
    "STEAM": "Steam Tables (IAPWS-IF97)",
}

INSTALL_HINT = (
    "DWSIM 9.0.5 is not available. Install it with scripts/install_dwsim.sh (the "
    "SourceForge .deb extracted with dpkg-deb -x, apt-get install dotnet-runtime-8.0, "
    "pip install pythonnet) and point DWSIM_PATH at the directory holding "
    "DWSIM.Automation.dll (.../usr/local/lib/dwsim).")

_ASSEMBLIES = ("DWSIM.Automation", "DWSIM.Interfaces", "DWSIM.GlobalSettings",
               "DWSIM.SharedClasses", "DWSIM.Thermodynamics", "DWSIM.UnitOperations",
               "DWSIM.FlowsheetBase")
_UI_ASSEMBLIES = ("DWSIM.UI.Desktop.Shared", "DWSIM.UI.Desktop.Editors")


class DWSIMUnavailable(RuntimeError):
    """DWSIM, the .NET runtime or pythonnet is missing (the message says how
    to install them)."""


def dwsim_path(path: Optional[str] = None) -> Path:
    """The DWSIM directory: ``path``, else ``$DWSIM_PATH``, else the default."""
    p = Path(path or os.environ.get("DWSIM_PATH") or DEFAULT_DWSIM_PATH)
    if not (p / "DWSIM.Automation.dll").exists():
        raise DWSIMUnavailable(f"no DWSIM.Automation.dll in {p}. {INSTALL_HINT}")
    return p


_LOADED: dict = {}


def _load(path: Path):
    """Load CoreCLR and DWSIM's assemblies, once per process."""
    if _LOADED:
        if _LOADED["path"] != path:
            raise DWSIMUnavailable(f"DWSIM is already loaded from {_LOADED['path']} in this "
                                   "process; one DWSIM per process")
        return _LOADED
    if "clr" not in sys.modules:
        try:
            from pythonnet import load
        except ImportError as e:  # pragma: no cover - only without pythonnet
            raise DWSIMUnavailable(f"pythonnet is not installed. {INSTALL_HINT}") from e
        try:
            load("coreclr")
        except Exception as e:  # pragma: no cover - only without .NET
            raise DWSIMUnavailable(f"could not start the .NET (CoreCLR) runtime: {e}. "
                                   f"{INSTALL_HINT}") from e
    import clr  # noqa: F401  (pythonnet)

    sys.path.append(str(path))
    for name in _ASSEMBLIES:
        clr.AddReference(str(path / f"{name}.dll"))
    _LOADED.update(path=path, clr=clr)
    return _LOADED


# ---------------------------------------------------------------------------
# Hypothetical compounds
# ---------------------------------------------------------------------------

def lee_kesler_eq101(Tc: float, Pc: float, omega: float) -> tuple:
    """Lee-Kesler (1975) vapour pressure, exactly, as DWSIM's equation 101
    coefficients ``(A, B, C, D, E)``: ``ln P[Pa] = A + B/T + C ln T + D T^E``.

    ``ln Pr = f0 + omega f1`` with ``f0 = 5.92714 - 6.09648/Tr - 1.28862 ln Tr
    + 0.169347 Tr^6`` and ``f1 = 15.2518 - 15.6875/Tr - 13.4721 ln Tr +
    0.43577 Tr^6`` -- linear in ``1/T``, ``ln T`` and ``T^6``, so the mapping
    is exact, not a fit.
    """
    c0 = 5.92714 + 15.2518 * omega
    c1 = 6.09648 + 15.6875 * omega
    c2 = 1.28862 + 13.4721 * omega
    c6 = 0.169347 + 0.43577 * omega
    return (math.log(Pc) + c0 + c2 * math.log(Tc), -c1 * Tc, -c2, c6 / Tc ** 6, 6.0)


@dataclass
class HypoCompound:
    """A hypothetical compound to define in DWSIM on given constants.

    Attributes:
        name: DWSIM compound name (unique in the flowsheet).
        MW: Molar mass (g/mol).
        Tc: Critical temperature (K).
        Pc: Critical pressure (Pa).
        omega: Acentric factor.
        Tb: Normal boiling point (K), optional (informational for the cubics).
        SG: Specific gravity 60/60 F, optional.
        cp_ig: Ideal-gas Cp polynomial ``(A, B, C, D[, E])`` in J/mol/K with
            T in K (difflow's ``cp`` rows); ``None`` leaves DWSIM at zero Cp.
        psat_eq101: ``(A, B, C, D, E)`` of ``ln P[Pa] = A + B/T + C ln T +
            D T^E``; ``None`` takes :func:`lee_kesler_eq101` of the constants.
        Zc: Critical compressibility; default ``0.2905 - 0.085 omega`` (Pitzer).
        Z_rackett: Rackett parameter for DWSIM's liquid density; default Zc.
        Hf: Ideal-gas heat of formation at 25 C (J/mol); default 0.
        hvap_tb: Enthalpy of vaporization at ``Tb`` (J/mol). DWSIM scales it
            with Watson's ``((1 - Tr)/(1 - Tbr))^0.375``; default Vetere's
            estimate from Tc, Pc, Tb (``HYP.DHvb_Vetere``, what DWSIM itself
            falls back to). Only Raoult's Law and the activity-coefficient
            packages read it (their liquid enthalpy is ``H_ig - Hvap``).
    """

    name: str
    MW: float
    Tc: float
    Pc: float
    omega: float
    Tb: Optional[float] = None
    SG: Optional[float] = None
    cp_ig: Optional[Sequence[float]] = None
    psat_eq101: Optional[Sequence[float]] = None
    Zc: Optional[float] = None
    Z_rackett: Optional[float] = None
    Hf: float = 0.0
    hvap_tb: Optional[float] = None

    @classmethod
    def from_gas_components(cls, comps, names: Optional[Sequence[str]] = None,
                            prefix: str = "") -> list["HypoCompound"]:
        """One hypo per component of a gas-plant ``GasComponents`` (or the
        named subset), on exactly its MW, Tc, Pc, omega and ideal-gas Cp."""
        out = []
        for i, n in enumerate(comps.names):
            if names is not None and n not in names:
                continue
            out.append(cls(name=prefix + n, MW=float(comps.MW[i]), Tc=float(comps.Tc[i]),
                           Pc=float(comps.Pc[i]), omega=float(comps.omega[i]),
                           cp_ig=[float(v) for v in comps.cp[i]]))
        return out

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("name", "MW", "Tc", "Pc", "omega", "Tb", "SG", "Zc",
                                            "Z_rackett", "Hf", "hvap_tb")}
        d["Tb_used"] = self.tb()
        d["cp_ig"] = None if self.cp_ig is None else [float(v) for v in self.cp_ig]
        d["psat_eq101"] = list(self.psat())
        return d

    def psat(self) -> tuple:
        if self.psat_eq101 is not None:
            return tuple(float(v) for v in self.psat_eq101)
        return lee_kesler_eq101(self.Tc, self.Pc, self.omega)

    def tb(self) -> float:
        """``Tb`` as given, else where :meth:`psat` is one atmosphere."""
        if self.Tb is not None:
            return float(self.Tb)
        A, B, C, D, E = self.psat()
        lo, hi = 0.2 * self.Tc, self.Tc
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if A + B / mid + C * math.log(mid) + D * mid ** E < math.log(101325.0):
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _constant_properties(self, cid: int):
        from DWSIM.Thermodynamics.BaseClasses import ConstantProperties

        c = ConstantProperties()
        c.Name = self.name
        c.ID = cid
        c.IsHYPO = True
        c.IsPF = False
        c.OriginalDB = "DWSIM"
        c.CurrentDB = "DWSIM"
        c.Formula = self.name
        c.Molar_Weight = float(self.MW)
        c.Critical_Temperature = float(self.Tc)
        c.Critical_Pressure = float(self.Pc)
        c.Acentric_Factor = float(self.omega)
        Zc = self.Zc if self.Zc is not None else 0.2905 - 0.085 * self.omega
        c.Critical_Compressibility = float(Zc)
        c.Critical_Volume = float(Zc * 8.314 * self.Tc / self.Pc * 1000.0)   # m3/kmol
        c.Z_Rackett = float(self.Z_rackett if self.Z_rackett is not None else Zc)
        Tb = self.tb()             # DWSIM's Hvap scaling needs a Tb
        c.Normal_Boiling_Point = Tb
        c.NBP = Tb
        if self.hvap_tb is not None:
            hv = float(self.hvap_tb)
        else:
            from DWSIM.Thermodynamics.Utilities.Hypos.Methods import HYP

            hv = float(HYP().DHvb_Vetere(float(self.Tc), float(self.Pc), Tb))   # J/mol
        c.HVap_A = hv / float(self.MW)                                            # kJ/kg
        if self.SG is not None:
            c.PF_SG = float(self.SG)
        cp = list(self.cp_ig or [0.0, 0.0, 0.0, 0.0]) + [0.0] * 5
        c.Ideal_Gas_Heat_Capacity_Const_A = float(cp[0])
        c.Ideal_Gas_Heat_Capacity_Const_B = float(cp[1])
        c.Ideal_Gas_Heat_Capacity_Const_C = float(cp[2])
        c.Ideal_Gas_Heat_Capacity_Const_D = float(cp[3])
        c.Ideal_Gas_Heat_Capacity_Const_E = float(cp[4])
        A, B, C, D, E = self.psat()
        c.Vapor_Pressure_Constant_A = A
        c.Vapor_Pressure_Constant_B = B
        c.Vapor_Pressure_Constant_C = C
        c.Vapor_Pressure_Constant_D = D
        c.Vapor_Pressure_Constant_E = E
        c.Vapor_Pressure_TMIN = 0.3 * float(self.Tc)
        c.Vapor_Pressure_TMAX = float(self.Tc)
        # kJ/kg in DWSIM's table
        c.IG_Enthalpy_of_Formation_25C = float(self.Hf) / float(self.MW)
        return c


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

def _constants_of(c) -> dict:
    """A ConstantProperties object as a plain dict (SI, J/mol)."""
    def f(v):
        try:
            return float(v)
        except Exception:  # noqa: BLE001 - None / nullable
            return None
    MW = f(c.Molar_Weight)
    hf = f(c.IG_Enthalpy_of_Formation_25C)
    return {"name": str(c.Name), "db": str(c.OriginalDB), "cas": str(c.CAS_Number),
            "formula": str(c.Formula), "MW": MW, "Tc": f(c.Critical_Temperature),
            "Pc": f(c.Critical_Pressure), "omega": f(c.Acentric_Factor),
            "Tb": f(c.Normal_Boiling_Point), "Zc": f(c.Critical_Compressibility),
            "Vc": f(c.Critical_Volume),
            "Hf": None if hf is None or MW is None else hf * MW,
            "is_hypo": bool(c.IsHYPO), "is_pf": bool(c.IsPF)}


class DWSIMSession:
    """One DWSIM ``Automation3`` instance in this process.

    Args:
        path: DWSIM directory (default ``$DWSIM_PATH`` or
            :data:`DEFAULT_DWSIM_PATH`).
    """

    def __init__(self, path: Optional[str] = None):
        self.path = dwsim_path(path)
        self._clr = _load(self.path)["clr"]
        with self.cwd():
            from DWSIM.Automation import Automation3
            from DWSIM.GlobalSettings import Settings

            for attr, val in (("EnableParallelProcessing", False), ("EnableGPUProcessing", False)):
                with contextlib.suppress(Exception):
                    setattr(Settings, attr, val)
            self.automation = Automation3()
            self.version = str(self.automation.GetVersion())
            self._probe = self.automation.CreateFlowsheet()
        self._next_id = 990000

    # -- process plumbing ----------------------------------------------------

    @contextlib.contextmanager
    def cwd(self):
        """Run with the DWSIM directory as the working directory."""
        old = os.getcwd()
        os.chdir(self.path)
        try:
            yield
        finally:
            os.chdir(old)

    def provenance(self) -> dict:
        import System
        from System.Reflection import Assembly
        from System.Runtime.InteropServices import RuntimeInformation

        from importlib.metadata import version as _v

        th = Assembly.LoadFrom(str(self.path / "DWSIM.Thermodynamics.dll"))
        return {"dwsim": self.version,
                "dwsim_thermodynamics_assembly": str(th.GetName().Version),
                "dotnet": str(RuntimeInformation.FrameworkDescription),
                "os": str(RuntimeInformation.OSDescription),
                "pythonnet": _v("pythonnet"),
                "clr_version": str(System.Environment.Version),
                "parallel_processing": False}

    # -- catalogue -----------------------------------------------------------

    def property_packages(self) -> list[str]:
        return sorted(str(k) for k in self._probe.AvailablePropertyPackages.Keys)

    def compounds(self) -> list[str]:
        return sorted(str(k) for k in self._probe.AvailableCompounds.Keys)

    def has_compound(self, name: str) -> bool:
        return bool(self._probe.AvailableCompounds.ContainsKey(name))

    def compound(self, name: str) -> dict:
        """Database constants of compound ``name`` (DWSIM's name)."""
        if not self.has_compound(name):
            raise KeyError(f"DWSIM has no compound {name!r}")
        return _constants_of(self._probe.AvailableCompounds[name])

    def find_by_cas(self, cas: str) -> list[dict]:
        """Every database entry with this CAS number (ChemSep first)."""
        out = [_constants_of(c) for c in self._probe.AvailableCompounds.Values
               if str(c.CAS_Number) == cas]
        return sorted(out, key=lambda d: (d["db"] != "ChemSep", d["name"]))

    # -- flowsheets ----------------------------------------------------------

    def flowsheet(self, package: str = "PR", compounds: Sequence[str] = (),
                  hypos: Iterable[HypoCompound] = (),
                  overrides: Optional[Mapping[str, Mapping[str, float]]] = None,
                  kij="zero", options: Optional[Mapping[str, object]] = None,
                  flash_tol: float = 1e-10, max_iter: int = 1000) -> "DWSIMFlowsheet":
        """A flowsheet with one material stream on property package ``package``.

        Args:
            package: A key of :data:`PROPERTY_PACKAGES` or DWSIM's full name.
            compounds: DWSIM database names (see
                :data:`difflow.dwsim_import.DWSIM_NAMES` for difflow's).
            hypos: :class:`HypoCompound` s, appended after ``compounds``.
            overrides: ``{dwsim_name: {"Tc": .., "Pc": .., "omega": .., "MW":
                ..}}`` replaces constants of one of ``compounds`` in THIS
                flowsheet (on a copy; DWSIM's database entry, which every
                flowsheet in the process shares, is untouched). Its Cp,
                Psat and kij stay DWSIM's -- for difflow's numbers
                throughout, use a hypo.
            kij: ``"zero"`` (all binary parameters removed: 0 for the cubics,
                no interaction for LKP), ``"dwsim"`` (DWSIM's database
                values), or a ``(C, C)`` array.
            options: Property-package settings, ``{attribute: value}``; an
                enum by its name (``{"LiquidDensityCalculationMode_Subcritical":
                "EOS"}``). :data:`EOS_ONLY` and :data:`IDEAL_RAOULT` are the
                like-for-like presets. Everything in :data:`PACKAGE_OPTIONS`
                is recorded in :meth:`DWSIMFlowsheet.provenance`.
            flash_tol: Loop tolerance for DWSIM's PT/PH flashes.
            max_iter: Iteration cap for those loops.
        """
        pkg = PROPERTY_PACKAGES.get(package, package)
        hypos = list(hypos)
        with self.cwd():
            from DWSIM.Interfaces.Enums.GraphicObjects import ObjectType

            sim = self.automation.CreateFlowsheet()
            names = list(compounds)
            for n in names:
                if not sim.AvailableCompounds.ContainsKey(n):
                    raise KeyError(f"DWSIM has no compound {n!r}")
            for n in (overrides or {}):
                if n not in names:
                    raise KeyError(f"override for {n!r}, which is not among compounds={names}")
            for h in hypos:
                if sim.AvailableCompounds.ContainsKey(h.name):
                    sim.AvailableCompounds.Remove(h.name)
                self._next_id += 1
                sim.AvailableCompounds.Add(h.name, h._constant_properties(self._next_id))
                names.append(h.name)
            if len(set(names)) != len(names):
                raise ValueError(f"duplicate compounds: {names}")
            for n in names:
                sim.AddCompound(n)
            # AvailableCompounds is shared by every flowsheet in the process: an
            # override goes on a copy held by this flowsheet's SelectedCompounds
            from System.Reflection import BindingFlags

            clr_type_method = None
            for n, fields in (overrides or {}).items():
                orig = sim.SelectedCompounds[n]
                if clr_type_method is None:
                    clr_type_method = orig.GetType().GetMethod(
                        "MemberwiseClone", BindingFlags.NonPublic | BindingFlags.Instance)
                clone = clr_type_method.Invoke(orig, None)
                for k, v in fields.items():
                    setattr(clone, _OVERRIDE_FIELDS.get(k, k), float(v))
                sim.SelectedCompounds[n] = clone
            if pkg not in [str(k) for k in sim.AvailablePropertyPackages.Keys]:
                raise KeyError(f"DWSIM has no property package {pkg!r}; see "
                               "DWSIMSession.property_packages()")
            pp = sim.CreateAndAddPropertyPackage(pkg).__implementation__
            st = sim.AddObject(ObjectType.MaterialStream, 50, 50, "S1").GetAsObject()
            st = getattr(st, "__implementation__", st)
            st.PropertyPackage = pp
            pp.CurrentMaterialStream = st
            fsh = DWSIMFlowsheet(self, sim, pp, st, names, pkg, list(hypos), dict(overrides or {}))
            fsh._set_options(options or {})
            fsh._set_flash_settings(flash_tol, max_iter)
            fsh._set_kij(kij)
        return fsh

    # -- DWSIM's own characterization -----------------------------------------

    def petroleum_characterization(self, tbp_K: Sequence[float], cum_frac: Sequence[float],
                                   sg_bulk: float, mw_bulk: Optional[float] = None,
                                   n_cuts: int = 10, cut_temps_K: Optional[Sequence[float]] = None,
                                   curve: str = "TBP", basis: str = "liquid_volume",
                                   Tc_corr: str = "Riazi-Daubert (1985)",
                                   Pc_corr: str = "Riazi-Daubert (1985)",
                                   omega_corr: str = "Lee-Kesler (1976)",
                                   mw_corr: str = "Winn (1956)",
                                   adjust_omega: bool = True, adjust_rackett: bool = True,
                                   sg_curve: Optional[Sequence[float]] = None,
                                   mw_curve: Optional[Sequence[float]] = None,
                                   name: str = "ASSAY", ui_sg_bug: bool = False) -> list[dict]:
        """DWSIM's distillation-curve petroleum characterization, headless.

        Args:
            tbp_K: Boiling temperatures of the curve (K).
            cum_frac: Cumulative fractions distilled at ``tbp_K`` (0-1, the
                ``basis``).
            sg_bulk: Bulk specific gravity (60/60 F) of the whole sample.
            mw_bulk: Bulk molar mass, if known: cut MWs are scaled to it.
            n_cuts: Number of pseudo-components (ignored with ``cut_temps_K``).
            cut_temps_K: Interior cut temperatures instead of a number.
            curve: ``"TBP"`` (D2892), ``"D86"``, ``"D1160"`` or ``"D2887"``.
            basis: ``"liquid_volume"``, ``"mole"`` or ``"mass"``.
            Tc_corr: ``"Riazi-Daubert (1985)"``, ``"Riazi (2005)"``,
                ``"Lee-Kesler (1976)"`` or ``"Farah (2006)"``.
            Pc_corr: ``"Riazi-Daubert (1985)"``, ``"Lee-Kesler (1976)"``,
                ``"Farah (2006)"``.
            omega_corr: ``"Lee-Kesler (1976)"`` or ``"Korsten (2000)"``.
            mw_corr: ``"Winn (1956)"``, ``"Riazi (1986)"``,
                ``"Lee-Kesler (1974)"``.
            adjust_omega: Fit each cut's omega so DWSIM's PR reproduces its
                NBP (the UI default).
            adjust_rackett: Fit the Rackett parameter to the cut's SG.
            sg_curve / mw_curve: Optional SG / MW at each curve point.
            name: Assay name (prefix of the compound names).
            ui_sg_bug: Reproduce the UI, which reads its bulk-SG field as API.

        Returns:
            One dict per pseudo-component, light to heavy: ``name``,
            ``mole_fraction``, ``mass_fraction``, ``Tb``, ``SG``, ``MW``, ``Tc``,
            ``Pc``, ``omega``, ``Zc``, ``Vc``, ``Z_rackett``, ``watson_K``, and
            ``cp_ig_300``/``cp_ig_600`` (J/mol/K, DWSIM's Lee-Kesler Cp), with
            ``inputs`` recorded on the first element's ``"settings"`` key.
        """
        with self.cwd():
            comps = self._run_distcurve(tbp_K, cum_frac, sg_bulk, mw_bulk, n_cuts, cut_temps_K,
                                        curve, basis, Tc_corr, Pc_corr, omega_corr, mw_corr,
                                        adjust_omega, adjust_rackett, sg_curve, mw_curve, name,
                                        ui_sg_bug)
            rows = self._pf_rows(comps)
        rows[0]["settings"] = dict(curve=curve, basis=basis, n_cuts=n_cuts,
                                   cut_temps_K=None if cut_temps_K is None else list(cut_temps_K),
                                   Tc_corr=Tc_corr, Pc_corr=Pc_corr, omega_corr=omega_corr,
                                   mw_corr=mw_corr, adjust_omega=adjust_omega,
                                   adjust_rackett=adjust_rackett, sg_bulk=sg_bulk,
                                   mw_bulk=mw_bulk, ui_sg_bug=ui_sg_bug)
        return rows

    def _run_distcurve(self, tbp_K, cum_frac, sg_bulk, mw_bulk, n_cuts, cut_temps_K, curve, basis,
                       Tc_corr, Pc_corr, omega_corr, mw_corr, adjust_omega, adjust_rackett,
                       sg_curve, mw_curve, name, ui_sg_bug):
        clr = self._clr
        for a in _UI_ASSEMBLIES:
            clr.AddReference(str(self.path / f"{a}.dll"))
        import DWSIM.UI.Desktop.Shared as Shared
        from DWSIM.SharedClasses.SystemsOfUnits import SI
        from DWSIM.UI.Desktop.Editors import DistCurvePCharacterization
        from System import Activator, Array, Boolean, Double, Int32, Object
        from System.Collections.Generic import List
        from System.Reflection import BindingFlags
        from System.Runtime.CompilerServices import RuntimeHelpers

        if len(tbp_K) != len(cum_frac):
            raise ValueError("tbp_K and cum_frac differ in length")
        T = clr.GetClrType(DistCurvePCharacterization)
        obj = RuntimeHelpers.GetUninitializedObject(T)      # no Eto: skip the UI constructor
        F = BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Instance

        def setf(n, v):
            T.GetField(n, F).SetValue(obj, v)

        def lst(xs):
            out = List[Double]()
            for x in xs or ():
                out.Add(float(x))
            return out

        tmp = T.GetNestedType("tmpcomp", BindingFlags.NonPublic | BindingFlags.Public)
        by_cuts = cut_temps_K is not None
        fields = {
            "tccol": Activator.CreateInstance(clr.GetClrType(List).MakeGenericType(tmp)),
            "ncomps": Int32(int(n_cuts)),
            "Tccorr": Tc_corr, "Pccorr": Pc_corr, "AFcorr": omega_corr, "MWcorr": mw_corr,
            "cb": lst(cum_frac), "tbp": lst(tbp_K),
            "mwc": lst(mw_curve), "sgc": lst(sg_curve),
            "visc100": lst([]), "visc210": lst([]),
            "hasmwc": Boolean(mw_curve is not None), "hassgc": Boolean(sg_curve is not None),
            "hasvisc100c": Boolean(False), "hasvisc210c": Boolean(False),
            "pseudomode": Int32(1 if by_cuts else 0),
            "cuttemps": lst(cut_temps_K if by_cuts else []),
            "pseudocuts": Int32(len(cut_temps_K) + 1 if by_cuts else int(n_cuts)),
            "mwb": Double(float(mw_bulk or 0.0)),
            # the editor reads this field as API gravity (see the module docstring)
            "sgb": Double(float(sg_bulk) if ui_sg_bug else 141.5 / float(sg_bulk) - 131.5),
            "tbpcurvetype": Int32({"TBP": 0, "D86": 1, "D1160": 2, "D2887": 3}[curve]),
            "curvebasis": Int32({"liquid_volume": 0, "mole": 1, "mass": 2}[basis]),
            "adjustAf": Boolean(adjust_omega), "adjustZR": Boolean(adjust_rackett),
            "decsep1": ".", "decsep2": ".",
            "assayname": name,
            "flowsheet": Shared.Flowsheet(),   # empty: the fits use a fresh PR package
        }
        for k, v in fields.items():
            setf(k, v)
        comps = T.GetMethod("GenerateCompounds", F).Invoke(obj, Array[Object]([SI()]))
        T.GetMethod("CalculateMolarFractions", F).Invoke(obj, Array[Object]([comps]))
        return comps

    def _pf_rows(self, comps) -> list[dict]:
        from DWSIM.Thermodynamics.PropertyPackages.Auxiliary import PROPS

        rows = []
        for k in comps.Keys:
            c = comps[k]
            cp = c.ConstantProperties
            d = _constants_of(cp)
            wk = float(cp.PF_Watson_K) if cp.PF_Watson_K is not None else None
            # DWSIM stores a petroleum fraction's heat of formation on a basis
            # 1000x off its database compounds' (a C6 cut reads -1.99 "kJ/kg");
            # keep the raw number, and no J/mol value that would look trustworthy
            d["Hf_dwsim_raw"] = float(cp.IG_Enthalpy_of_Formation_25C)
            d["Hf"] = None
            d.update(name=str(k), mole_fraction=float(c.MoleFraction or 0.0),
                     mass_fraction=float(c.MassFraction or 0.0),
                     Tb=float(cp.NBP) if cp.NBP is not None else d["Tb"],
                     SG=float(cp.PF_SG) if cp.PF_SG is not None else None,
                     Z_rackett=float(cp.Z_Rackett), watson_K=wk)
            if wk is not None:
                # DWSIM's Lee-Kesler ideal-gas Cp (kJ/kg/K) * MW = J/mol/K
                d["cp_ig_300"] = float(PROPS.Cpig_lk(wk, float(cp.Acentric_Factor), 300.0)) * d["MW"]
                d["cp_ig_600"] = float(PROPS.Cpig_lk(wk, float(cp.Acentric_Factor), 600.0)) * d["MW"]
            rows.append(d)
        if not any(r["mass_fraction"] for r in rows):     # the bulk method leaves them unset
            tot = sum(r["mole_fraction"] * r["MW"] for r in rows)
            for r in rows:
                r["mass_fraction"] = r["mole_fraction"] * r["MW"] / tot
        return sorted(rows, key=lambda r: r["Tb"])

    def bulk_characterization(self, mw: Optional[float], sg: Optional[float],
                              tb: Optional[float] = None, n: int = 6, prefix: str = "C7P",
                              mw0: float = 80.0, sg0: float = 0.70, tb0: float = 333.0,
                              Tc_corr: str = "Riazi-Daubert (1985)",
                              Pc_corr: str = "Riazi-Daubert (1985)",
                              omega_corr: str = "Lee-Kesler (1976)",
                              mw_corr: str = "Winn (1956)",
                              adjust_omega: bool = True, adjust_rackett: bool = True) -> list[dict]:
        """DWSIM's bulk C7+ characterization (``Utilities.PetroleumCharacterization.
        GenerateCompounds``, what its "Bulk C7+" editor calls): ``n``
        pseudo-components distributed from the bulk ``mw``, ``sg`` and
        (optional) average NBP ``tb`` (K) of the plus fraction, starting from
        the lightest cut's ``mw0``, ``sg0``, ``tb0`` (the editor's defaults:
        80, 0.70, 333 K). Viscosities are not given (the editor's zeros at
        311.15 and 372.05 K). Returns the same row dicts as
        :meth:`petroleum_characterization`."""
        from System import Double, Nullable
        from DWSIM.Thermodynamics.Utilities.PetroleumCharacterization import GenerateCompounds

        def nul(v):
            return Nullable[Double](float(v)) if v is not None else None

        with self.cwd():
            g = GenerateCompounds()
            comps = g.GenerateCompounds(prefix, int(n), Tc_corr, Pc_corr, omega_corr, mw_corr,
                                        bool(adjust_omega), bool(adjust_rackett), nul(mw), nul(sg),
                                        nul(tb), nul(0.0), nul(0.0), nul(311.15), nul(372.05),
                                        float(mw0), float(sg0), float(tb0))
            rows = self._pf_rows(comps)
        rows[0]["settings"] = dict(mw=mw, sg=sg, tb=tb, n=n, mw0=mw0, sg0=sg0, tb0=tb0,
                                   Tc_corr=Tc_corr, Pc_corr=Pc_corr, omega_corr=omega_corr,
                                   mw_corr=mw_corr, adjust_omega=adjust_omega,
                                   adjust_rackett=adjust_rackett)
        return rows


#: Property-package settings that change answers, all recorded in provenance.
PACKAGE_OPTIONS = ("LiquidDensityCalculationMode_Subcritical",
                   "LiquidDensityCalculationMode_Supercritical",
                   "LiquidDensity_UsePenelouxVolumeTranslation",
                   "LiquidDensity_CorrectExpDataForPressure",
                   "LiquidEnthalpyEntropyCpCvCalculationMode_EOS",
                   "EnthalpyEntropyCpCvCalculationMode",
                   "VaporPhaseFugacityCalculationMode",
                   "LiquidFugacity_UsePoyntingCorrectionFactor",
                   "UseHenryConstants")

#: A cubic EOS and nothing else: liquid density from the EOS without Peneloux
#: volume translation (DWSIM's default is Rackett + experimental data, with
#: Peneloux), liquid enthalpy from the EOS (the default).
EOS_ONLY = {"LiquidDensityCalculationMode_Subcritical": "EOS",
            "LiquidDensityCalculationMode_Supercritical": "EOS",
            "LiquidDensity_UsePenelouxVolumeTranslation": False,
            "LiquidEnthalpyEntropyCpCvCalculationMode_EOS": "EOS"}

#: Raoult's law as the textbook has it: ``y P = x Psat``. DWSIM's "Raoult's
#: Law" package multiplies in a Poynting factor and treats supercritical
#: components (H2, N2, CH4 ...) by Henry's law by default.
IDEAL_RAOULT = {"LiquidFugacity_UsePoyntingCorrectionFactor": False,
                "UseHenryConstants": False,
                "VaporPhaseFugacityCalculationMode": "Ideal"}

_OVERRIDE_FIELDS = {"Tc": "Critical_Temperature", "Pc": "Critical_Pressure",
                    "omega": "Acentric_Factor", "MW": "Molar_Weight",
                    "Tb": "Normal_Boiling_Point", "Zc": "Critical_Compressibility"}


class DWSIMFlowsheet:
    """A DWSIM flowsheet with one material stream, for flashes."""

    def __init__(self, session, sim, pp, stream, names, package, hypos, overrides):
        self.session = session
        self.sim = sim
        self.pp = pp
        self.stream = stream
        self.names = list(names)
        self.package = package
        self.hypos = hypos
        self.overrides = overrides
        self._kij_mode = None
        self._flash_settings = {}

    # -- configuration -------------------------------------------------------

    def _set_options(self, options):
        import System

        for k, v in options.items():
            cur = getattr(self.pp, k)          # AttributeError names a bad key
            if isinstance(v, str) and cur is not None and cur.GetType().IsEnum:
                v = System.Enum.Parse(cur.GetType(), v)
            setattr(self.pp, k, v)

    def options(self) -> dict:
        """The :data:`PACKAGE_OPTIONS` this package has, as strings."""
        out = {}
        for k in PACKAGE_OPTIONS:
            v = getattr(self.pp, k, None)
            if v is not None:
                out[k] = str(v)
        return out

    def _set_flash_settings(self, tol, max_iter):
        fs = self.pp.FlashSettings
        new = {}
        for key in list(fs.Keys):
            k = str(key)
            if k.endswith("_Loop_Tolerance"):
                new[key] = f"{tol:g}"
            elif "Maximum_Number_Of" in k:
                new[key] = str(int(max_iter))
        for key, v in new.items():
            fs[key] = v
        self._flash_settings = {str(k): str(fs[k]) for k in fs.Keys}

    def _ip_stores(self):
        """The package's interaction-parameter tables, the one its
        ``RET_KIJ`` reads first (``m_lk`` for Lee-Kesler-Plocker, else
        ``m_pr``; Grayson-Streed and Chao-Seader carry both)."""
        order = ("m_lk", "m_pr") if "LKP" in str(self.pp.GetType().Name) else ("m_pr", "m_lk")
        out = []
        for attr in order:
            aux = getattr(self.pp, attr, None)
            if aux is not None and hasattr(aux, "InteractionParameters"):
                out.append(aux.InteractionParameters)
        return out

    def _set_kij(self, kij):
        """``"zero"``: remove every stored pair among the components, so DWSIM
        uses its value for a missing pair (0 for the cubics; LKP's
        multiplicative ``k_ij`` has its own neutral value, recorded);
        ``"dwsim"``: leave DWSIM's database values; array: set them."""
        n = len(self.names)
        if isinstance(kij, str) and kij == "dwsim":
            self._kij_mode = "dwsim"
            return
        explicit = not isinstance(kij, str)
        if not explicit and kij != "zero":
            raise ValueError(f"kij must be 'zero', 'dwsim' or a {n}x{n} array, not {kij!r}")
        K = np.asarray(kij, float) if explicit else None
        if explicit and K.shape != (n, n):
            raise ValueError(f"kij must be 'zero', 'dwsim' or a {n}x{n} array")
        stores = self._ip_stores()
        if not stores:
            if explicit and np.any(K):
                raise ValueError(f"{self.package} has no binary interaction parameters to set")
            self._kij_mode = "none (package has no kij)"
            return
        from System.Collections.Generic import Dictionary

        ip = stores[0]
        ipcls = None
        for a in ip.Keys:
            for b in ip[a].Keys:
                ipcls = type(ip[a][b])
                break
            if ipcls is not None:
                break
        for i, a in enumerate(self.names):
            for j, b in enumerate(self.names):
                if i == j:
                    continue
                for st in stores:
                    for p, q in ((a, b), (b, a)):
                        if st.ContainsKey(p) and st[p].ContainsKey(q):
                            st[p].Remove(q)
                if explicit and i < j:
                    if not ip.ContainsKey(a):
                        ip.Add(a, Dictionary[str, ipcls]())
                    d = ipcls()
                    d.kij = float(K[i, j])
                    ip[a].Add(b, d)
        self._refresh_kij()
        got = self.kij()
        off = got[~np.eye(n, dtype=bool)]
        if explicit:
            if not np.allclose(off, K[~np.eye(n, dtype=bool)], atol=1e-12):
                raise RuntimeError(f"DWSIM did not take the kij: asked {K}, has {got}")
            self._kij_mode = "given"
        else:
            if off.size and np.ptp(off) > 0:
                raise RuntimeError(f"kij still pair-dependent after removing DWSIM's: {got}")
            neutral = float(off[0]) if off.size else 0.0
            if "LKP" in str(self.pp.GetType().Name):
                # Plocker's k_ij multiplies sqrt(Tci Tcj); MixCritProp_LK replaces
                # a zero (missing) k_ij, and the diagonal, by 1: no interaction
                self._kij_mode = "removed (LKP: a missing k_ij is taken as 1)"
            else:
                self._kij_mode = ("zero" if neutral == 0.0
                                  else f"removed (missing-pair value {neutral})")

    def _refresh_kij(self):
        from System.Reflection import BindingFlags

        F = BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Instance
        t = self.pp.GetType()
        m = t.GetMethod("SetKijMatrix", F)
        if m is not None:
            with self.session.cwd():
                self.pp.CurrentMaterialStream = self.stream
                m.Invoke(self.pp, None)

    def kij(self) -> np.ndarray:
        """The kij matrix DWSIM uses, in :attr:`names` order."""
        self.pp.CurrentMaterialStream = self.stream
        n = len(self.names)
        try:
            vk = self.pp.RET_VKij()
        except Exception:  # noqa: BLE001 - packages without kij
            return np.zeros((n, n))
        return np.array([[float(vk[i, j]) for j in range(n)] for i in range(n)])

    # -- flashes -------------------------------------------------------------

    def _z(self, z):
        z = np.asarray(z, dtype=float)
        if z.shape != (len(self.names),):
            raise ValueError(f"z needs {len(self.names)} entries ({self.names})")
        return z / z.sum()

    def _run(self, spec, z, **kw):
        import System
        from DWSIM.Interfaces.Enums import StreamSpec

        s = self.stream
        with self.session.cwd():
            s.SetOverallComposition(System.Array[float]([float(v) for v in self._z(z)]))
            s.SetMolarFlow(1.0)
            if "P" in kw:
                s.SetPressure(float(kw["P"]))
            if "T" in kw:
                s.SetTemperature(float(kw["T"]))
            if spec == "TP":
                s.SpecType = StreamSpec.Temperature_and_Pressure
            elif spec == "PH":
                s.SpecType = StreamSpec.Pressure_and_Enthalpy
                mw = float(np.dot(self._z(z), self._mw()))
                s.SetMassEnthalpy(float(kw["h"]) / mw)          # J/mol / (g/mol) = kJ/kg
            elif spec == "PVF":
                s.SpecType = StreamSpec.Pressure_and_VaporFraction
                s.GetPhase("Vapor").Properties.molarfraction = float(kw["vf"])
            elif spec == "TVF":
                s.SpecType = StreamSpec.Temperature_and_VaporFraction
                s.GetPhase("Vapor").Properties.molarfraction = float(kw["vf"])
            errs = self.session.automation.CalculateFlowsheet4(self.sim)
            if errs is not None and len(list(errs)) > 0:
                raise RuntimeError(f"DWSIM {spec} flash failed: {[str(e) for e in errs]}")
            return self._read()

    def _mw(self):
        return np.array([float(self.sim.SelectedCompounds[n].Molar_Weight) for n in self.names])

    def _phase(self, label):
        p = self.stream.GetPhase(label)
        frac = float(p.Properties.molarfraction or 0.0)
        comp = [float(p.Compounds[n].MoleFraction or 0.0) for n in self.names]
        pr = p.Properties

        def f(v):
            try:
                return float(v)
            except Exception:  # noqa: BLE001
                return None
        mw = f(pr.molecularWeight)
        h = f(pr.enthalpy)
        s = f(pr.entropy)
        phi = [f(p.Compounds[n].FugacityCoeff) for n in self.names]
        return frac, comp, {"h": None if h is None or mw is None else h * mw,
                            "s": None if s is None or mw is None else s * mw,
                            "rho": f(pr.density), "Z": f(pr.compressibilityFactor), "MW": mw,
                            "phi": phi}

    def _read(self) -> dict:
        s = self.stream
        out = {"T": float(s.GetTemperature()), "P": float(s.GetPressure()),
               "z": [float(v) for v in s.GetOverallComposition()]}
        vf, y, pv = self._phase("Vapor")
        lf, x, pl = self._phase("Liquid1")
        l2, x2, _ = self._phase("Liquid2")
        _, _, pm = self._phase("Mixture")
        tol = 1e-12
        has_v, has_l = vf > tol, lf > tol
        out.update(vapor_fraction=vf, liquid_fraction=lf, liquid2_fraction=l2,
                   x=x if has_l else None, y=y if has_v else None,
                   x2=x2 if l2 > tol else None,
                   K=[yi / xi if xi > 0 else None for xi, yi in zip(x, y)] if (has_v and has_l) else None,
                   h=pm["h"], s=pm["s"],
                   h_vap=pv["h"] if has_v else None, h_liq=pl["h"] if has_l else None,
                   rho_vap=pv["rho"] if has_v else None, rho_liq=pl["rho"] if has_l else None,
                   Z_vap=pv["Z"] if has_v else None, Z_liq=pl["Z"] if has_l else None,
                   MW_vap=pv["MW"] if has_v else None, MW_liq=pl["MW"] if has_l else None,
                   phi_vap=pv["phi"] if has_v else None, phi_liq=pl["phi"] if has_l else None,
                   phases=[p for p, ok in (("vapor", has_v), ("liquid", has_l),
                                           ("liquid2", l2 > tol)) if ok])
        out["equilibrium_residual"] = None
        if has_v and has_l and all(v for v in pv["phi"]) and all(v for v in pl["phi"]):
            # how far DWSIM's own answer is from its own equilibrium condition:
            # max |ln(y/x) - ln(phi_L/phi_V)| over components present
            r = [abs(math.log(yi / xi) - math.log(fl / fv))
                 for xi, yi, fl, fv in zip(x, y, pl["phi"], pv["phi"]) if xi > 1e-12 and yi > 1e-12]
            out["equilibrium_residual"] = max(r) if r else None
        return out

    def flash_tp(self, z, T: float, P: float) -> dict:
        """PT flash of overall mole fractions ``z`` (any scale; normalized)."""
        return self._run("TP", z, T=T, P=P)

    def flash_ph(self, z, P: float, h: float) -> dict:
        """PH flash; ``h`` molar enthalpy in J/mol on DWSIM's (= difflow's)
        reference (ideal gas at 298.15 K is zero)."""
        return self._run("PH", z, P=P, h=h)

    def flash_pvf(self, z, P: float, vf: float) -> dict:
        """P-vapor-fraction flash (``vf = 0`` bubble point, ``1`` dew point)."""
        return self._run("PVF", z, P=P, vf=vf)

    def flash_tvf(self, z, T: float, vf: float) -> dict:
        """T-vapor-fraction flash (bubble/dew pressure)."""
        return self._run("TVF", z, T=T, vf=vf)

    # -- pure-component functions ----------------------------------------------

    def pure(self, name: str, T: float) -> dict:
        """DWSIM's ideal-gas Cp (J/mol/K), Psat (Pa) and Hvap (J/mol) of one
        selected component at ``T``."""
        pp = self.pp
        with self.session.cwd():
            pp.CurrentMaterialStream = self.stream
            mw = float(self.sim.SelectedCompounds[name].Molar_Weight)
            cp = float(pp.AUX_CPi(name, float(T))) * mw
            psat = float(pp.AUX_PVAPi(name, float(T)))
            try:
                hv = float(pp.AUX_HVAPi(name, float(T))) * mw
            except Exception:  # noqa: BLE001 - not every package defines it
                hv = None
        return {"cp_ig": cp, "psat": psat, "hvap": hv}

    def constants(self) -> list[dict]:
        """The component table as DWSIM holds it (after overrides/hypos)."""
        return [_constants_of(self.sim.SelectedCompounds[n]) for n in self.names]

    def provenance(self) -> dict:
        fb = self.pp.FlashBase
        return {"property_package": self.package,
                "property_package_class": str(self.pp.GetType().FullName),
                "flash_algorithm": str(fb.GetType().FullName) if fb is not None else None,
                "flash_calculation_approach": str(self.pp.FlashCalculationApproach),
                "flash_settings": dict(self._flash_settings),
                "options": self.options(),
                "kij": self._kij_mode,
                "kij_matrix": self.kij().tolist(),
                "compounds": self.names,
                "hypos": [h.as_dict() for h in self.hypos],
                "overrides": {k: dict(v) for k, v in self.overrides.items()},
                "constants": self.constants()}
