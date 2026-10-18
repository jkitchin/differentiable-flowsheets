"""Write the 1 atm gas tables behind the light-hydrocarbon and H2S fits.

Generates ``tests/reference/coolprop_gases_1atm/<gas>.csv`` (same columns as
the NIST WebBook tables in ``nist_gases_1atm/``) for the natural-gas and
refinery-gas components the WebBook tables do not cover. Needs CoolProp
(``pip install CoolProp``)::

    python scripts/make_coolprop_gas_tables.py
    python scripts/fit_gas_transport.py

CoolProp evaluates the reference equations of state and transport models
(the same correlations REFPROP and the NIST WebBook use):

=========  ======================  ===================  =========================
gas        EOS                     viscosity            conductivity
=========  ======================  ===================  =========================
ethane     Buecker & Wagner 2006   Friend et al. 1991   Friend et al. 1991
propane    Lemmon et al. 2009      Vogel et al. 1998    Marsh et al. 2002
n-butane   Buecker & Wagner 2006   Vogel et al. 1999    Perkins et al. 2002
isobutane  Buecker & Wagner 2006   Vogel et al. 2000    Perkins et al. 2002
H2S        Lemmon & Span 2006      Quinones-Cisneros    Chung et al. 1988 (below)
                                   et al. 2012
=========  ======================  ===================  =========================

H2S has no reference conductivity model; its conductivity is Chung et al.
(Ind. Eng. Chem. Res. 27 (1988) 671; Poling et al. 5e, Eq. 10-3.14) from the
tabulated viscosity and ideal-gas heat capacity. The script prints Chung's
deviation from the reference conductivity of the four hydrocarbons as a
check on that estimate.

Each table is the vapour on the 1 atm isobar, from just above the normal
boiling point (the first multiple of 10 K at least 2 K above it) to the EOS upper limit, in 10 K steps.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import CoolProp
import CoolProp.CoolProp as CP

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "reference" / "coolprop_gases_1atm"
P = 101325.0
R = 8.314462618

FLUIDS = {
    "ethane": "Ethane",
    "propane": "Propane",
    "n_butane": "n-Butane",
    "isobutane": "IsoButane",
    "hydrogen_sulfide": "HydrogenSulfide",
}


def chung_conductivity(fluid, T, mu):
    """Chung et al. (1988) low-pressure conductivity (W/m/K) from mu (Pa s)."""
    M = CP.PropsSI("M", fluid)
    omega = CP.PropsSI("acentric", fluid)
    Tr = T / CP.PropsSI("Tcrit", fluid)
    cv = CP.PropsSI("CP0MOLAR", "T", T, "P", P, fluid) / R - 1.0
    alpha = cv - 1.5
    beta = 0.7862 - 0.7109 * omega + 1.3168 * omega**2
    Z = 2.0 + 10.5 * Tr**2
    psi = 1.0 + alpha * ((0.215 + 0.28288 * alpha - 1.061 * beta + 0.26665 * Z)
                         / (0.6366 + beta * Z + 1.061 * alpha * beta))
    return 3.75 * psi * mu * R / M


def table(fluid):
    Tb = CP.PropsSI("T", "P", P, "Q", 1, fluid)
    T = np.arange(math.ceil((Tb + 2.0) / 10.0) * 10.0, CP.PropsSI("Tmax", fluid) + 1e-9, 10.0)
    cp = np.array([CP.PropsSI("C", "T", t, "P", P, fluid) for t in T]) / 1e3
    mu = np.array([CP.PropsSI("V", "T", t, "P", P, fluid) for t in T])
    try:
        k = np.array([CP.PropsSI("L", "T", t, "P", P, fluid) for t in T])
        k_src = f"reference conductivity ({CP.get_BibTeXKey(fluid, 'CONDUCTIVITY')})"
    except ValueError:
        k = np.array([chung_conductivity(fluid, t, m) for t, m in zip(T, mu)])
        k_src = "Chung et al. 1988 conductivity from the tabulated mu and ideal-gas cp"
    return T, cp, mu, k, k_src


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fluid in FLUIDS.items():
        T, cp, mu, k, k_src = table(fluid)
        if "Chung" not in k_src:
            dev = max(abs(chung_conductivity(fluid, t, m) / kk - 1) for t, m, kk in zip(T, mu, k))
            print(f"# {name:16s} Chung k vs reference: max dev {100 * dev:.1f} %")
        header = (f"# {name} gas at 0.101325 MPa, CoolProp {CoolProp.__version__}: "
                  f"EOS {CP.get_BibTeXKey(fluid, 'EOS')}, viscosity "
                  f"{CP.get_BibTeXKey(fluid, 'VISCOSITY')}, {k_src}.\n")
        with open(OUT / f"{name}.csv", "w") as f:
            f.write(header)
            f.write("T_K,cp_J_gK,mu_Pa_s,k_W_m_K\n")
            for row in zip(T, cp, mu, k):
                f.write(f"{row[0]:.3f},{row[1]:.6g},{row[2]:.6g},{row[3]:.6g}\n")
        print(f"wrote {OUT / name}.csv: {T[0]:g}-{T[-1]:g} K")


if __name__ == "__main__":
    main()
