"""Lumped residue hydroprocessing kinetics: HDS by class (and refractory residue sulfur), HDM, CCR, conversion.

A :class:`RDSKinetics` is a kinetic model for
:class:`~difflow_refinery.hydroprocessing.reactor.TrickleBedReactor`. It is the
hydrotreating model (:class:`~difflow_refinery.hydrotreating.kinetics.HDTKinetics`:
HDS of the five sulfur classes, HDN, reversible aromatics saturation, olefins)
run with residue constants, plus four reactions that only a residue needs.
Its per-cut attributes (:data:`RDS_ATTRIBUTES`) are the hydrotreater's, then

* ``S_residue`` -- sulfur atoms in the asphaltene and resin molecules of the
  heaviest cuts. These are the sulfur a residue desulfurizer finds hardest:
  large molecules whose sulfur is held in condensed aromatic sheets and
  whose diffusion into the catalyst pores is hindered. Counted as S.
* ``NiV`` -- nickel plus vanadium atoms (porphyrins and asphaltene-bound
  metals), as moles of a nominal metal of :data:`METAL_MW` g/mol. Counted as
  no element: metals are left out of a cut's mass, as everywhere in the
  hydroprocessing layout.
* ``CCR`` -- carbon atoms that the Conradson carbon residue test would leave
  as coke, mol of C. A *subset* of the cut's carbon, so counted as no element
  (counting it as C would count those atoms twice).

Reactions added to :class:`HDTKinetics` (per cut; ``c`` are the
fugacity-equivalent liquid concentrations of
:class:`~difflow_refinery.hydroprocessing.reactor.ReactionContext`)::

    refractory HDS:  S_res + nu_S H2 -> H2S     r = f k_Sr c_Sres h(c_H2) / D^2
                                                D = 1 + K_H2S c_H2S + K_N c_Nbasic
    HDM:             NiV -> deposit on catalyst r = f k_M c_NiV h(c_H2)^m_M
    CCR reduction:   C_CCR + nu_C H2 -> C       r = f k_C c_CCR h(c_H2)
    conversion:      molecule of cut i + (m_i - 1) H2 -> m_i molecules of cut j(i)
                                                r = f k_X c_i     (cuts at or above T_conv only)

with the same ``f = activity * effectiveness * wetting``, Arrhenius
``k(T)`` and ``h(c_H2) = (c_H2/c_ref)^m`` as the hydrotreating model (whose
inhibition denominator ``D`` is reused unchanged for the refractory class).

* **Forms.** First-order HDS per class with Langmuir-Hinshelwood H2S and
  basic-nitrogen inhibition is the hydrotreating model's form (Korsten &
  Hoffmann 1996; Froment et al. 1994; see that module). The lumped residue
  reactions -- first order in sulfur, metals and CCR, with an H2-pressure
  term, and residue conversion as a first-order lump with a high activation
  energy -- are the forms reviewed by Ancheyta, Sanchez & Rodriguez (2005)
  for heavy-oil hydrotreating; the metals deposit on the catalyst, which is
  what limits a residue unit's cycle length (Rana et al. 2007). None of the
  papers' constants is used.
* **Conversion** sends every carbon atom of a cracked molecule of cut ``i``
  to ``m_i = n_C,i / n_C,j`` molecules of a lighter cut ``j(i)`` (the cut
  whose carbon number is nearest half of ``i``'s, :func:`conversion_targets`)
  and hydrogenates the ``m_i - 1`` new chain ends with one H2 each, so the
  element balance is exact. The fragments inherit the parent's sulfur,
  nitrogen, metals, CCR and ring classes, per atom. Only cuts boiling at or
  above ``T_conv`` (538 C by default: the "residue" of a conversion figure)
  crack. The hydrotreating model's own cracking leak is switched off in the
  residue defaults (``crack_k = 0``), since this replaces it.
* **Stoichiometry and heat.** Refractory HDS: ``nu_S`` = 3 H2 and the heat of
  benzothiophene HDS (the hydrotreating model's model compound), so that the
  class is a slow benzothiophene. CCR reduction: ``nu_C`` = 0.5 H2 per CCR
  carbon (one aromatic ring of six carbons saturated with three H2: an
  ILLUSTRATIVE stoichiometry) at the heat of benzene -> cyclohexane per H2.
  Conversion: the heat of n-hexane + H2 -> n-butane + ethane per bond broken.
  HDM: neither hydrogen nor heat (metals are ppm; their H2 is ~1e-4 of the
  unit's).

Where the numbers come from -- read this before trusting a wt%:

* The RATE CONSTANTS, activation energies and the refractory share of the
  residue sulfur (:data:`DEFAULT_RESIDUE_S_SHARE` in :mod:`.feed`) are
  ILLUSTRATIVE: chosen here so that an atmospheric residue of about 3 wt%
  sulfur at a WABT near 390 C, LHSV 0.25 1/h, 150 bar and 1000 Nm3/m3
  desulfurizes by 85-90 %, demetallizes by 70-85 % and converts 10-20 % of
  its 538 C+ -- the ranges that ARDS units report (Speight 2000, Ch. 7;
  Rana et al. 2007). They are not fitted to any unit or catalyst, and the
  HDM/HDS catalyst grading of a real unit is lumped into one average
  catalyst.
* The heats of reaction are the hydrotreating model's model-compound values.

References:
    Ancheyta, J., Sanchez, S. and Rodriguez, M.A., "Kinetic modeling of
        hydrocracking of heavy oil fractions: a review", Catal. Today 109,
        76-92 (2005), doi:10.1016/j.cattod.2005.08.015 (title, pages and DOI
        as recalled -- unverified).
    Rana, M.S., Samano, V., Ancheyta, J. and Diaz, J.A.I., "A review of
        recent advances on process technologies for upgrading of heavy oils
        and residua", Fuel 86, 1216-1231 (2007),
        doi:10.1016/j.fuel.2006.08.004 (unverified).
    Speight, J.G., "The Desulfurization of Heavy Oils and Residua", 2nd ed.,
        Marcel Dekker (2000) (chapter as recalled -- unverified).
    Korsten, H. and Hoffmann, U. (1996), AIChE J. 42, 1350-1360 -- the HDS
        form, via :mod:`difflow_refinery.hydrotreating.kinetics` (unverified).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from difflow.params_mixin import ParamsMixin
from difflow_refinery.hydroprocessing.layout import Layout
from difflow_refinery.hydroprocessing.reactor import Rates, ReactionContext
from difflow_refinery.hydrotreating.kinetics import (
    AROMATIC_THERMO, CRACK_HEAT, HDS_HEAT, HDT_ATTRIBUTE_ELEMENTS, HDT_ATTRIBUTES, R_GAS, HDTKineticParams,
    HDTKinetics)

jax.config.update("jax_enable_x64", True)

#: The residue unit's per-cut attributes: the hydrotreater's, then the residue ones.
RDS_ATTRIBUTES: tuple[str, ...] = HDT_ATTRIBUTES + ("S_residue", "NiV", "CCR")
#: Element counted by each attribute (``None``: not an element of the balance).
RDS_ATTRIBUTE_ELEMENTS: tuple[str | None, ...] = HDT_ATTRIBUTE_ELEMENTS + ("S", None, None)

#: Nominal molar mass (g/mol) of the ``NiV`` attribute: a 3:1 V:Ni atom mix
#: (V 50.9415, Ni 58.6934). The choice cancels: feed and product Ni+V wppm
#: are both converted with it. ILLUSTRATIVE ratio.
METAL_MW: float = (3.0 * 50.9415 + 58.6934) / 4.0
#: Molar mass of a CCR carbon atom, g/mol.
CCR_MW: float = 12.0107

#: H2 per refractory sulfur atom (a benzothiophene-like model, see module docstring).
RESIDUE_S_H2: float = 3.0
#: Heat per refractory sulfur atom removed, J/mol (benzothiophene HDS).
RESIDUE_S_HEAT: float = HDS_HEAT[2]
#: H2 per CCR carbon atom made non-coke-forming (ILLUSTRATIVE, see module docstring).
CCR_H2: float = 0.5
#: Heat per mol of H2 in CCR reduction: benzene + 3 H2 -> cyclohexane, per H2.
CCR_HEAT_PER_H2: float = AROMATIC_THERMO[2][0] / 3.0
#: Heat per C-C bond broken in residue conversion (n-hexane + H2 -> n-butane + ethane).
CONVERSION_HEAT: float = CRACK_HEAT


def residue_hdt_params(**kw) -> HDTKineticParams:
    """The hydrotreating constants the residue model runs with (ILLUSTRATIVE).

    Relative to the distillate defaults: an effectiveness factor of 0.35
    (diffusion of residue molecules into the pores), the cracking leak off
    (residue conversion replaces it), and a weaker basic-nitrogen adsorption
    constant, since a residue's basic nitrogen is mostly in molecules too
    large to reach the sites the distillate constant describes.
    """
    base = dict(effectiveness=0.35, crack_k=0.0, K_nbasic=0.01, T_ref=653.15,
                hds_k=(7.5e-6, 5.0e-6, 3.75e-6, 3.0e-6, 2.0e-6),
                hdn_k=(1.0e-7, 1.5e-7), hda_k=(6.0e-7, 2.2e-7, 4.5e-8))
    base.update(kw)
    return HDTKineticParams(**base)


@dataclass
class RDSKineticParams(ParamsMixin):
    """Constants of :class:`RDSKinetics` (all ILLUSTRATIVE; see the module docstring).

    Rate constants are first order, m^3 of liquid per kg catalyst per s, at
    the base model's ``T_ref``.

    Attributes:
        base: The hydrotreating constants (:func:`residue_hdt_params`); its
            ``activity``, ``effectiveness`` and ``wetting`` multiply every rate.
        sres_k: Refractory-sulfur HDS rate constant, m^3/kg/s.
        sres_E: Its activation energy, J/mol.
        hdm_k: Ni+V removal rate constant, m^3/kg/s.
        hdm_E: Its activation energy, J/mol.
        hdm_h2_order: Order in H2 of the HDM rate.
        ccr_k: CCR-reduction rate constant, m^3/kg/s.
        ccr_E: Its activation energy, J/mol.
        conv_k: Residue-conversion rate constant (per molecule), m^3/kg/s.
        conv_E: Its activation energy, J/mol (thermal-like: high).
    """

    base: HDTKineticParams = field(default_factory=residue_hdt_params)
    sres_k: float = 1.2e-6
    sres_E: float = 130e3
    hdm_k: float = 5.0e-7
    hdm_E: float = 110e3
    hdm_h2_order: float = 0.5
    ccr_k: float = 3.0e-7
    ccr_E: float = 100e3
    conv_k: float = 2.5e-8
    conv_E: float = 180e3


jax.tree_util.register_dataclass(
    RDSKineticParams, data_fields=[f.name for f in dataclasses.fields(RDSKineticParams)], meta_fields=[])


def _arr(k, E, T, T_ref):
    return jnp.asarray(k, dtype=float) * jnp.exp(-jnp.asarray(E, dtype=float) / R_GAS * (1.0 / T - 1.0 / T_ref))


def conversion_targets(carbon_per_molecule, Tb, T_conv: float) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Where a converted molecule of each cut goes, and into how many molecules.

    ``carbon_per_molecule`` and ``Tb`` are concrete ``(n_cut,)``, lightest
    first. A cut boiling at or above ``T_conv`` (K) cracks to the lighter cut
    whose carbon number is nearest half of its own, giving ``m = n_C,i /
    n_C,j`` molecules; a lighter cut does not convert (target ``-1``,
    ``m = 1``).
    """
    c = np.asarray(carbon_per_molecule, dtype=float)
    Tb = np.asarray(Tb, dtype=float)
    tgt, mult = [], []
    for i in range(c.size):
        if Tb[i] < T_conv or i == 0:
            tgt.append(-1)
            mult.append(1.0)
            continue
        j = int(np.argmin(np.abs(c[:i] - 0.5 * c[i])))
        tgt.append(j)
        mult.append(float(c[i] / c[j]))
    return tuple(tgt), tuple(mult)


@dataclass(frozen=True)
class RDSKinetics:
    """The residue kinetic model (see the module docstring).

    Attributes:
        layout: The stream layout; its attributes must be :data:`RDS_ATTRIBUTES`.
        crack_targets: The hydrotreating model's cracking-leak targets (its
            leak is off in the residue defaults, but the base model needs them).
        conv_targets: For each cut, the cut its converted molecules land in
            (``-1``: does not convert); from :func:`conversion_targets`.
        conv_mult: Molecules of the target cut per converted molecule.
    """

    layout: Layout
    crack_targets: tuple[int, ...]
    conv_targets: tuple[int, ...]
    conv_mult: tuple[float, ...]
    attributes: tuple[str, ...] = RDS_ATTRIBUTES
    attribute_elements: tuple = RDS_ATTRIBUTE_ELEMENTS

    def __post_init__(self):
        if tuple(self.layout.attributes) != RDS_ATTRIBUTES:
            raise ValueError("the layout's attributes must be RDS_ATTRIBUTES")
        if not (len(self.conv_targets) == len(self.conv_mult) == self.layout.n_cut):
            raise ValueError("conv_targets and conv_mult need one entry per cut")

    @property
    def base(self) -> HDTKinetics:
        return HDTKinetics(self.layout, self.crack_targets)

    def rates(self, ctx: ReactionContext, p: RDSKineticParams) -> Rates:
        lay = self.layout
        b = p.base
        r0 = self.base.rates(ctx, b)
        T = ctx.T
        Tr = b.T_ref
        ai = {a: i for i, a in enumerate(lay.attributes)}
        hi, si = lay.gas_index("hydrogen"), lay.gas_index("hydrogen_sulfide")
        f = b.activity * b.effectiveness * b.wetting
        cA = ctx.c_attr
        h2r = jnp.maximum(ctx.c_gas("hydrogen"), 0.0) / b.c_ref
        c_h2s = ctx.c_gas("hydrogen_sulfide")
        K_h2s = b.K_h2s * jnp.exp(-b.dH_h2s / R_GAS * (1.0 / T - 1.0 / Tr))
        K_nb = b.K_nbasic * jnp.exp(-b.dH_nbasic / R_GAS * (1.0 / T - 1.0 / Tr))
        D = 1.0 + K_h2s * c_h2s + K_nb * jnp.sum(cA[:, ai["N_basic"]])

        d_attr, d_gas, d_cut, heat = r0.attr, r0.gas, r0.cut, r0.heat

        # --- refractory residue sulfur ---------------------------------------
        rS = f * _arr(p.sres_k, p.sres_E, T, Tr) * cA[:, ai["S_residue"]] * h2r ** b.hds_h2_order / D**2
        d_attr = d_attr.at[:, ai["S_residue"]].add(-rS)
        d_attr = d_attr.at[:, ai["H"]].add((2.0 * RESIDUE_S_H2 - 2.0) * rS)
        d_gas = d_gas.at[si].add(jnp.sum(rS)).at[hi].add(-RESIDUE_S_H2 * jnp.sum(rS))
        heat = heat - RESIDUE_S_HEAT * jnp.sum(rS)

        # --- HDM: metals leave the stream onto the catalyst ---------------------
        rM = f * _arr(p.hdm_k, p.hdm_E, T, Tr) * cA[:, ai["NiV"]] * h2r ** p.hdm_h2_order
        d_attr = d_attr.at[:, ai["NiV"]].add(-rM)

        # --- CCR reduction ------------------------------------------------------
        rC = f * _arr(p.ccr_k, p.ccr_E, T, Tr) * cA[:, ai["CCR"]] * h2r
        d_attr = d_attr.at[:, ai["CCR"]].add(-rC)
        d_attr = d_attr.at[:, ai["H"]].add(2.0 * CCR_H2 * rC)
        d_gas = d_gas.at[hi].add(-CCR_H2 * jnp.sum(rC))
        heat = heat - CCR_HEAT_PER_H2 * CCR_H2 * jnp.sum(rC)

        # --- residue conversion ---------------------------------------------------
        tgt = np.asarray(self.conv_targets)
        mask = jnp.asarray(tgt >= 0, dtype=float)
        if np.any(tgt >= 0):
            m = jnp.asarray(self.conv_mult)
            rX = f * _arr(p.conv_k, p.conv_E, T, Tr) * ctx.c_cut * mask       # (nc,) parent molecules
            moved = rX[:, None] * ctx.per_molecule                             # every attribute, per atom
            t_safe = jnp.asarray(np.where(tgt >= 0, tgt, 0))
            d_attr = d_attr - moved
            d_attr = d_attr.at[t_safe].add(moved)
            bonds = (m - 1.0) * rX
            d_attr = d_attr.at[t_safe, ai["H"]].add(2.0 * bonds)
            d_cut = d_cut - rX
            d_cut = d_cut.at[t_safe].add(m * rX)
            d_gas = d_gas.at[hi].add(-jnp.sum(bonds))
            heat = heat - CONVERSION_HEAT * jnp.sum(bonds)

        return Rates(gas=d_gas, cut=d_cut, attr=d_attr, heat=heat)


__all__ = ["RDS_ATTRIBUTES", "RDS_ATTRIBUTE_ELEMENTS", "METAL_MW", "CCR_MW", "RESIDUE_S_H2", "RESIDUE_S_HEAT",
           "CCR_H2", "CCR_HEAT_PER_H2", "CONVERSION_HEAT", "residue_hdt_params", "RDSKineticParams",
           "RDSKinetics", "conversion_targets"]
