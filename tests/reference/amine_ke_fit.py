"""Fit the Kent-Eisenberg apparent constants of ``difflow_cc.equilibrium.acid_gas`` (#471).

Run from the repository root::

    PYTHONPATH=src:. python tests/reference/amine_ke_fit.py

Prints the fitted constants, which are pasted into ``KE_AMINES``;
``tests/cc/test_acid_gas_vle.py`` refits (marked ``slow``) and checks that
they are still the optimum.

Every thermodynamic constant (water, CO2, HCO3-, H2S, HS- dissociation; the
amine protonation constants; Henry's constants) is fixed at its literature
value. What is fitted, by least squares in ``ln p`` (the usual Kent-Eisenberg
practice; Patil et al. 2006 regress ``ln K_app``, which is equivalent), is:

* MDEA: ``ln K_f,app - ln K_f = d0 + d1 ln(alpha + 0.001)``, two numbers, on
  Austgen (1989) Table 5.3 (CO2, 40 C) and Huttenhuis et al. (2007)
  Tables 3-4 (H2S, 10 and 25 C). Huttenhuis Table 1 (CO2, 10 and 25 C) is
  held out.
* DEA: the same two, plus the apparent carbamate constant
  ``ln K_h = a + b/T``, on Barreau et al. (2006) Tables 1-4. The smoothed
  literature values tabulated beside them (Lee et al. 1972; Lawson &
  Garst 1976) are held out.

A concentration term ``d2 ln m`` (Patil's D) was tried and left at zero: it
moved the MDEA fit by less than the scatter (AAPD 17.5/6.9 % -> 15.0/10.1 % on CO2/H2S).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from difflow_cc.equilibrium import acid_gas as ag
from tests.cc import acid_gas_data as D

jax.config.update("jax_enable_x64", True)

FIT_SETS = {
    "MDEA": [(D.AUSTGEN_CO2_MDEA, "CO2"), (D.HUTTENHUIS_H2S_MDEA, "H2S")],
    "DEA": [(D.BARREAU_CO2_DEA, "CO2"), (D.BARREAU_H2S_DEA, "H2S")],
}


def with_params(base: ag.KentEisenbergAmine, x) -> ag.KentEisenbergAmine:
    """The amine with apparent constants ``x = (d0, d1[, a, b])``."""
    carb = None if base.carbamate is None else (x[2], x[3])
    return dataclasses.replace(base, apparent=(x[0], x[1], 0.0), carbamate=carb)


def log_error(am: ag.KentEisenbergAmine, rows, gas: str):
    """``ln(p_model / p_measured)`` for every row ``(T, w, alpha, P kPa)``."""
    R = jnp.asarray(np.array(rows, dtype=float))

    def one(T, w, a):
        aC, aS = (a, 0.0) if gas == "CO2" else (0.0, a)
        pc, ps = ag.ke_partial_pressures(am, T, w, aC, aS)
        return pc if gas == "CO2" else ps

    return jnp.log(jax.vmap(one)(R[:, 0], R[:, 1], R[:, 2]) / 1e3 / R[:, 3])


def aapd(r) -> float:
    """Average absolute percent deviation in p from ``ln`` errors."""
    return float(100.0 * np.mean(np.abs(np.exp(np.asarray(r)) - 1.0)))


def fit(name: str):
    """Least-squares fit; returns the parameter vector."""
    from scipy.optimize import least_squares

    base = ag.KE_AMINES[name]
    x0 = np.array([0.0, 0.0] + ([-3.0, 0.0] if base.carbamate is not None else []))
    resid = jax.jit(lambda x: jnp.concatenate(
        [log_error(with_params(base, x), rows, gas) for rows, gas in FIT_SETS[name]]))
    sol = least_squares(lambda x: np.asarray(resid(jnp.asarray(x))), x0,
                        xtol=1e-12, ftol=1e-12)
    return sol.x


def main():
    for name in FIT_SETS:
        x = fit(name)
        am = with_params(ag.KE_AMINES[name], x)
        print(name, repr(tuple(float(v) for v in x)))
        for rows, gas in FIT_SETS[name]:
            print(f"   {gas}: AAPD {aapd(log_error(am, rows, gas)):.1f} %")


if __name__ == "__main__":
    main()
