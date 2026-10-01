"""Component data for the gas plant: real light ends and pseudocomponents
on one Peng-Robinson equation of state.

A gas plant sees hydrogen, H2S, C1-C5 paraffins and (from an FCC) olefins,
plus naphtha that is only described by a TBP curve. All of them go into one
:class:`GasComponents` -- the same arrays (``Tc``, ``Pc``, ``omega``,
ideal-gas Cp, a ``kij`` matrix) whatever their origin -- so a single cubic
equation of state covers the mixture. The pseudocomponents' constants come
from :class:`difflow_refinery.Characterization` (Twu or Riazi-Daubert
critical properties, Kesler-Lee acentric factor, Watson-Nelson ideal-gas
Cp); the real components' from :mod:`difflow.database` and the tables
below.

Sources of the tables here
    Ideal-gas Cp cubics (J/mol/K, T in K), where :mod:`difflow.database`
    has none or only a constant:

    * hydrogen, ethylene, propylene, 1-butene: Reid, Prausnitz & Poling,
      *The Properties of Gases and Liquids*, 4th ed. (1987), App. A, as
      IDAES's ``HC_PR`` example package transcribes them -- except
      hydrogen's ``C``, which is taken as RPP4's -1.381e-5 rather than
      IDAES's -1.981e-5: the former gives 28.86 J/mol/K at 300 K against
      the JANAF 28.85, the latter 28.32;
    * hydrogen sulfide: a cubic fitted here to the NIST-JANAF Shomate
      equation (Chase 1998) over 298-1000 K, worst point 0.25%.

    Lower heating values are computed from the heats of formation
    (gas, 298 K) by ``LHV = dHf(species) - c dHf(CO2) - h/2 dHf(H2O, g)
    - s dHf(SO2)``, with CO2 -393.51, H2O(g) -241.826 and SO2 -296.84
    kJ/mol (NIST-JANAF). Heats of formation are from
    :mod:`difflow.database` where it has one; hydrogen sulfide -20.6
    kJ/mol (CODATA, Cox, Wagman & Medvedev 1984) and 1-butene -0.63 kJ/mol
    (Prosen, Maron & Rossini 1951), both via the NIST WebBook.

Binary interaction parameters
    :data:`PR_KIJ` holds the nonzero Peng-Robinson ``kij`` used by default.
    Hydrocarbon-hydrocarbon pairs default to zero, the usual choice for
    paraffins of similar size (Peng & Robinson 1976 fit small values, of
    either sign). The tabulated pairs are the ones that matter in a gas
    plant -- CO2, H2S and N2 with the light paraffins -- recalled from the
    Knapp et al. DECHEMA compilation (*Vapor-Liquid Equilibria for Mixtures
    of Low Boiling Substances*, Chemistry Data Series VI, 1982). **Every
    entry is marked ``verify``**: they were not checked against the printed
    table, and should be before a design number rests on them. Pass ``kij``
    to :func:`gas_components` to replace any of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

#: Reference temperature of every enthalpy (K).
T_REF = 298.15
#: Gas constant (J/mol/K), as :mod:`difflow.eos` uses it.
R = 8.314462618

# dHf (J/mol, gas, 298 K) of the combustion products (NIST-JANAF).
_HF_CO2 = -393510.0
_HF_H2O_GAS = -241826.0
_HF_SO2 = -296840.0

#: Ideal-gas Cp cubics for components :mod:`difflow.database` lacks or holds
#: only as a constant (see module docstring for sources).
CP_IG: dict[str, tuple[float, float, float, float]] = {
    "hydrogen": (27.14, 9.274e-3, -1.381e-5, 7.645e-9),
    "ethylene": (3.806, 1.566e-1, -8.348e-5, 1.755e-8),
    "propylene": (3.710, 2.345e-1, -1.160e-4, 2.205e-8),
    "1_butene": (-2.994, 3.532e-1, -1.990e-4, 4.463e-8),
    "hydrogen_sulfide": (31.69, 2.0829e-3, 2.3835e-5, -1.1894e-8),
}

#: Heats of formation (J/mol, gas, 298 K) for components the database lacks.
HF_GAS: dict[str, float] = {
    "hydrogen": 0.0,
    "nitrogen": 0.0,
    "hydrogen_sulfide": -20600.0,
    "1_butene": -630.0,
    "water": _HF_H2O_GAS,
}

#: Elemental formula (C, H, S) of the combustible light components.
FORMULA: dict[str, tuple[int, int, int]] = {
    "hydrogen": (0, 2, 0), "hydrogen_sulfide": (0, 2, 1),
    "methane": (1, 4, 0), "ethane": (2, 6, 0), "propane": (3, 8, 0),
    "isobutane": (4, 10, 0), "n_butane": (4, 10, 0),
    "isopentane": (5, 12, 0), "n_pentane": (5, 12, 0), "neopentane": (5, 12, 0),
    "n_hexane": (6, 14, 0), "n_heptane": (7, 16, 0), "n_octane": (8, 18, 0),
    "ethylene": (2, 4, 0), "propylene": (3, 6, 0), "1_butene": (4, 8, 0),
    "cis_2_butene": (4, 8, 0), "trans_2_butene": (4, 8, 0), "isobutylene": (4, 8, 0),
    "benzene": (6, 6, 0), "toluene": (7, 8, 0),
    "nitrogen": (0, 0, 0), "carbon_dioxide": (0, 0, 0), "water": (0, 0, 0),
}

#: Lower heating value assumed for a pseudocomponent (J/kg): a typical
#: naphtha, 44 MJ/kg. Pseudocomponents are trace in fuel gas; this only
#: keeps them from counting as inert.
PSEUDO_LHV_J_PER_KG = 44.0e6

_V = "verify: recalled from Knapp et al. (1982), not checked against the table"
#: Default nonzero PR binary interaction parameters, ``{(a, b): (kij, note)}``.
PR_KIJ: dict[tuple[str, str], tuple[float, str]] = {
    ("carbon_dioxide", "methane"): (0.0919, _V),
    ("carbon_dioxide", "ethane"): (0.1322, _V),
    ("carbon_dioxide", "propane"): (0.1241, _V),
    ("carbon_dioxide", "isobutane"): (0.1200, _V),
    ("carbon_dioxide", "n_butane"): (0.1333, _V),
    ("nitrogen", "methane"): (0.0311, _V),
    ("nitrogen", "ethane"): (0.0515, _V),
    ("nitrogen", "propane"): (0.0852, _V),
    ("hydrogen_sulfide", "methane"): (0.0850, _V),
    ("hydrogen_sulfide", "ethane"): (0.0833, _V),
    ("hydrogen_sulfide", "propane"): (0.0878, _V),
    ("carbon_dioxide", "hydrogen_sulfide"): (0.0974, _V),
}


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GasComponents:
    """Every component of a gas-plant mixture, on one cubic EOS.

    Registered as a pytree: the arrays may be traced (a pseudocomponent's
    ``Tc`` from a differentiated assay, a ``kij`` under study), and
    ``names`` is static.

    Attributes:
        MW: Molar mass (g/mol), ``(C,)``.
        Tc: Critical temperature (K).
        Pc: Critical pressure (Pa).
        omega: Acentric factor (the EOS one).
        cp: Ideal-gas Cp cubic ``(C, 4)``, J/mol/K with T in K.
        kij: Binary interaction parameters ``(C, C)``, symmetric, zero
            diagonal.
        lhv: Lower heating value (J/mol, gas, 298 K).
        pseudo: 1.0 for a pseudocomponent, 0.0 for a real one.
        names: Component names, in array order (static).
    """

    MW: jax.Array
    Tc: jax.Array
    Pc: jax.Array
    omega: jax.Array
    cp: jax.Array
    kij: jax.Array
    lhv: jax.Array
    pseudo: jax.Array
    names: tuple = field(metadata=dict(static=True), default=())

    @property
    def n(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        """Array position of component ``name``."""
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"no component {name!r}; have {self.names}") from None

    def mask(self, names: Sequence[str]) -> jax.Array:
        """0/1 vector selecting ``names``."""
        m = np.zeros(self.n)
        for n in names:
            m[self.index(n)] = 1.0
        return jnp.asarray(m)


def _lhv_light(name: str, hf: float) -> float:
    c, h, s = FORMULA[name]
    if c == 0 and h == 0:
        return 0.0
    return hf - (c * _HF_CO2 + 0.5 * h * _HF_H2O_GAS + s * _HF_SO2)


def light_component_data(name: str) -> dict:
    """Constants of one real component: Tc, Pc, omega, MW, Cp, dHf, LHV."""
    from difflow.database import _IDEAL_THERMO_DATA, get_critical_props, resolve_alias

    key = resolve_alias(name.lower().replace("-", "_"))
    crit = get_critical_props(key)
    ideal = _IDEAL_THERMO_DATA.get(key, {})
    if key in CP_IG:
        cp = CP_IG[key]
    elif "Cp" in ideal and any(ideal["Cp"][1:]):
        cp = ideal["Cp"]
    else:
        raise KeyError(f"no ideal-gas Cp cubic for {key!r}; add it to "
                       "difflow_refinery.gasplant.components.CP_IG")
    hf = HF_GAS.get(key, ideal.get("Hf"))
    if key not in FORMULA:
        raise KeyError(f"no elemental formula for {key!r} (needed for its LHV)")
    return dict(name=key, Tc=crit.Tc, Pc=crit.Pc, omega=crit.omega, MW=crit.MW,
                cp=tuple(cp), lhv=_lhv_light(key, hf if hf is not None else 0.0))


def default_kij(names: Sequence[str]) -> np.ndarray:
    """The :data:`PR_KIJ` matrix over ``names`` (zero where none is tabulated)."""
    n = len(names)
    k = np.zeros((n, n))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            v = PR_KIJ.get((a, b)) or PR_KIJ.get((b, a))
            if v is not None and i != j:
                k[i, j] = v[0]
    return k


def gas_components(light: Sequence[str], pseudo=None,
                   kij: Optional[Mapping] = None,
                   cuts: Optional[Sequence[str]] = None) -> GasComponents:
    """Build the component set: real ``light`` components, then ``pseudo``.

    Args:
        light: Database names of the real components (``"hydrogen"``,
            ``"methane"``, ``"propylene"``, ``"isobutylene"``, ...).
        pseudo: Optional :class:`difflow_refinery.Characterization`; its
            *cuts* (not its light ends -- list those in ``light``) are
            appended, with the characterization's EOS constants and
            ideal-gas Cp. Its arrays may be traced.
        kij: Overrides, ``{(a, b): value}``; symmetric. Any pair not given
            takes :data:`PR_KIJ` or zero.
        cuts: Only these of ``pseudo``'s cuts, in its order (a naphtha
            from a whole-crude characterization carries the light cuts
            only; the heavy ones would be dead weight in every EOS call).

    Returns:
        A :class:`GasComponents`.
    """
    rows = [light_component_data(n) for n in light]
    names = [r["name"] for r in rows]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate components in {list(light)}")
    MW = jnp.asarray([r["MW"] for r in rows], dtype=float).reshape(-1)
    Tc = jnp.asarray([r["Tc"] for r in rows], dtype=float).reshape(-1)
    Pc = jnp.asarray([r["Pc"] for r in rows], dtype=float).reshape(-1)
    w = jnp.asarray([r["omega"] for r in rows], dtype=float).reshape(-1)
    cp = jnp.asarray([r["cp"] for r in rows], dtype=float).reshape(-1, 4)
    lhv = jnp.asarray([r["lhv"] for r in rows], dtype=float).reshape(-1)
    ps = jnp.zeros(len(rows))
    if pseudo is not None:
        pn = tuple(pseudo.pseudo_names)
        sel = list(range(len(pn)))
        if cuts is not None:
            unknown = sorted(set(cuts) - set(pn))
            if unknown:
                raise ValueError(f"{unknown} are not cuts of the characterization")
            sel = [i for i, n in enumerate(pn) if n in set(cuts)]
            pn = tuple(pn[i] for i in sel)
        sel = jnp.asarray(sel, dtype=int)
        clash = set(pn) & set(names)
        if clash:
            raise ValueError(f"pseudocomponent names clash with real ones: {clash}")
        names += list(pn)
        MW = jnp.concatenate([MW, jnp.asarray(pseudo.MW, dtype=float)[sel]])
        Tc = jnp.concatenate([Tc, jnp.asarray(pseudo.Tc, dtype=float)[sel]])
        Pc = jnp.concatenate([Pc, jnp.asarray(pseudo.Pc, dtype=float)[sel]])
        w = jnp.concatenate([w, jnp.asarray(pseudo.omega, dtype=float)[sel]])
        cp = jnp.concatenate([cp, jnp.asarray(pseudo.cp_ig_coeffs, dtype=float)[sel]])
        lhv = jnp.concatenate([lhv, PSEUDO_LHV_J_PER_KG * jnp.asarray(pseudo.MW)[sel] / 1000.0])
        ps = jnp.concatenate([ps, jnp.ones(len(pn))])
    K = default_kij(names)
    for (a, b), v in dict(kij or {}).items():
        i, j = names.index(a), names.index(b)
        K[i, j] = K[j, i] = float(v)
    return GasComponents(MW=MW, Tc=Tc, Pc=Pc, omega=w, cp=cp, kij=jnp.asarray(K),
                         lhv=lhv, pseudo=ps, names=tuple(names))
