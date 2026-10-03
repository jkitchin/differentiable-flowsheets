"""The DWSIM smoke cases: inputs shared by ``dwsim_smoke_generate.py`` and
``tests/refinery/test_dwsim_smoke.py``.

Two mixtures on Peng-Robinson (1976 kappa), ``kij = 0``, both sides
computing on the SAME constants:

* ``light_ends`` -- hydrogen, H2S and C1-C5 paraffins, the gas plant's
  components (:func:`difflow_refinery.gasplant.gas_components`), flashed at
  four (T, P) points and one PH point. DWSIM gets every component as a
  hypothetical compound carrying difflow's MW, Tc, Pc, omega and ideal-gas
  Cp cubic, so DWSIM's database constants (and its own kij) play no part.
* ``naphtha`` -- five pseudo-components from TBP cut points and SGs written
  below, their constants from difflow's correlations (Twu 1984 Tc/Pc/MW,
  Lee-Kesler omega, the ideal-gas Cp fit of
  :func:`difflow_refinery.correlations.cp_ideal_gas_coeffs`), with a little
  propane and n-butane; flashed at two (T, P) points.

What it tests is the *implementation* -- the same model and constants on
both sides -- not how well PR describes either mixture. The constants are
an input to both sides and are frozen into the JSON; a change to them is a
staleness failure, not a disagreement.
"""

from __future__ import annotations

#: Real components of ``light_ends`` (difflow database names).
LIGHT_ENDS = ["hydrogen", "hydrogen_sulfide", "methane", "ethane", "propane",
              "isobutane", "n_butane", "n_pentane"]

#: Naphtha pseudo-components: name, mid-boiling point (K) and SG 60/60 F.
NAPHTHA_CUTS = [("NBP_345", 345.0, 0.680), ("NBP_375", 375.0, 0.715),
                ("NBP_405", 405.0, 0.745), ("NBP_435", 435.0, 0.770),
                ("NBP_465", 465.0, 0.790)]
NAPHTHA_LIGHT = ["propane", "n_butane"]

CASES: dict[str, dict] = {
    "light_ends": {
        "light": LIGHT_ENDS, "pseudo": [],
        "z": [0.25, 0.03, 0.22, 0.12, 0.13, 0.07, 0.10, 0.08],
        "tp": [[240.0, 30e5], [280.0, 20e5], [300.0, 10e5], [330.0, 35e5]],
        "ph": [{"P": 20e5, "T_from": 280.0}],
    },
    "naphtha": {
        "light": NAPHTHA_LIGHT, "pseudo": NAPHTHA_CUTS,
        "z": [0.05, 0.10, 0.15, 0.20, 0.20, 0.18, 0.12],
        "tp": [[380.0, 2e5], [410.0, 3e5]],
        "ph": [],
    },
}


def component_data(case: dict) -> dict:
    """difflow's constants for ``case``: ``names``, ``MW``, ``Tc``, ``Pc``,
    ``omega``, ``cp_ig`` (rows ``A..D``, J/mol/K), and for pseudo-components
    their ``Tb`` and ``SG`` (``None`` for real ones)."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from difflow_refinery.correlations import (acentric_factor, cp_ideal_gas_coeffs,
                                               critical_properties)
    from difflow_refinery.gasplant import gas_components

    c = gas_components(case["light"])
    out = {"names": list(c.names), "MW": [float(v) for v in c.MW],
           "Tc": [float(v) for v in c.Tc], "Pc": [float(v) for v in c.Pc],
           "omega": [float(v) for v in c.omega],
           "cp_ig": [[float(v) for v in row] for row in c.cp],
           "Tb": [None] * c.n, "SG": [None] * c.n}
    if case["pseudo"]:
        Tb = jnp.asarray([p[1] for p in case["pseudo"]])
        SG = jnp.asarray([p[2] for p in case["pseudo"]])
        MW, Tc, Pc = critical_properties(Tb, SG, method="twu")
        w = acentric_factor(Tb, Tc, Pc, SG)
        cp = cp_ideal_gas_coeffs(Tb, SG, MW)
        out["names"] += [p[0] for p in case["pseudo"]]
        out["MW"] += [float(v) for v in MW]
        out["Tc"] += [float(v) for v in Tc]
        out["Pc"] += [float(v) for v in Pc]
        out["omega"] += [float(v) for v in w]
        out["cp_ig"] += [[float(v) for v in row] for row in cp]
        out["Tb"] += [float(v) for v in Tb]
        out["SG"] += [float(v) for v in SG]
    return out


def difflow_thermo(comp: dict):
    """difflow's gas-plant PR (``CubicThermo``) on ``comp``, ``kij = 0``."""
    import jax.numpy as jnp
    import numpy as np

    from difflow_refinery.gasplant import CubicThermo, GasComponents
    from difflow_refinery.gasplant.thermo import PR

    n = len(comp["names"])
    gc = GasComponents(MW=jnp.asarray(comp["MW"]), Tc=jnp.asarray(comp["Tc"]),
                       Pc=jnp.asarray(comp["Pc"]), omega=jnp.asarray(comp["omega"]),
                       cp=jnp.asarray(comp["cp_ig"]), kij=jnp.zeros((n, n)),
                       lhv=jnp.zeros(n), pseudo=jnp.asarray(np.array(
                           [0.0 if t is None else 1.0 for t in comp["Tb"]])),
                       names=tuple(comp["names"]))
    return CubicThermo(gc, PR)
