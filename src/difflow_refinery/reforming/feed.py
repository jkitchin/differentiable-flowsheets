"""Reformer feed: hydrotreated naphtha as P/N/A lumps by carbon number.

A reformer cannot run on a boiling curve and a gravity: its chemistry is
naphthenes to aromatics, paraffins to naphthenes, and both to gas, so it
needs the paraffin, naphthene and aromatic content of the feed by carbon
number. :class:`NaphthaFeed` holds that, as molar flows of the reformer's
species (:data:`~difflow_refinery.reforming.species.NAMES`), and builds it
from one of two sources:

* :meth:`NaphthaFeed.from_piona` -- a measured PIONA analysis by carbon
  number (ASTM D5134 / D6730 detailed hydrocarbon analysis, grouped). This is
  the preferred input.
* :meth:`NaphthaFeed.from_characterization` -- the naphtha cut of the unified
  characterization (#301) with the hydrocarbon-type estimate of #305
  (``Characterization.composition``). The estimate is coarse at carbon-number
  resolution; the mapping below is what turns it into lumps.
* :meth:`NaphthaFeed.from_hydrotreater` -- a hydrotreater's (or its
  fractionator's) product, on the TREATED cuts of its product grid: the
  same mapping, with the grid's molar masses, its hydrocarbon types after
  saturation, and its sulfur (#327). This is the path for a reformer that
  follows a naphtha hydrotreater, as every reformer does.

Mapping a characterization to lumps (``from_characterization``):

1. Each pseudo-component ``i`` (mass flow ``m_i``) carries hydrocarbon-type
   VOLUME fractions ``phi_it`` over P, N, A, O (#305). They are turned into
   mass fractions with the density of the type's model compound at the cut's
   carbon number, ``w_it ~ phi_it rho_t``, so the cut's mass is conserved
   exactly. Olefins are counted as paraffins: the feed is hydrotreated, and
   a hydrotreater saturates them.
2. Each type's mass is placed on the carbon-number grid by the cut's boiling
   point: ``c = interp(Tb_i, Tb_t(6..10), 6..10)``, the boiling points of the
   type's model compounds (n-paraffins for P; cyclohexane, then the
   n-alkylcyclohexanes for N; benzene, then the n-alkylbenzenes for A), and
   split linearly between the two neighbouring carbon numbers. Boiling
   points outside the C6-C10 range are clipped to it, so pentanes and lighter
   must come in as light ends, not as cuts.
3. Paraffin mass is split into normal and iso by ``iso_fraction`` (by mass,
   the two isomers having the same molar mass); C6 naphthene mass into
   methylcyclopentane and cyclohexane by ``mcp_fraction``. Neither split is
   in a Tb/SG characterization; both defaults are ILLUSTRATIVE (straight-run
   naphthas commonly carry iso- and n-paraffins in comparable amounts, and
   more methylcyclopentane than cyclohexane), and a measured PIONA replaces
   them.
4. Light ends the characterization keeps as real species map to themselves
   (``n_hexane`` -> ``nP6``, ``isopentane`` -> ``iC5``...); water and
   anything else without a reformer species is dropped.

Molar flows are then mass over the lump's molar mass. The hydrogen content of
the lumped feed is that of the model compounds, not the #305 hydrogen
estimate; :meth:`NaphthaFeed.hydrogen_wt` reports it so the two can be
compared.

Every step is a ``jax.numpy`` expression (the carbon-number split is
piecewise linear in ``Tb``), so a feed built from a characterization is
differentiable in the assay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow_refinery.reforming import species as sp
from difflow_refinery.reforming import thermo as th

#: Light ends a characterization may carry, and their reformer species.
LIGHT_END_MAP: dict[str, str] = {
    "methane": "C1", "ethane": "C2", "propane": "C3", "isobutane": "iC4", "n_butane": "nC4",
    "isopentane": "iC5", "n_pentane": "nC5", "n_hexane": "nP6",
    "2_methylpentane": "iP6", "methylcyclopentane": "N5_6", "cyclohexane": "N6",
    "benzene": "A6", "n_heptane": "nP7", "toluene": "A7",
}

#: Hydrocarbon groups of a PIONA analysis, and the lump kind each maps to.
PIONA_GROUPS: tuple[str, ...] = ("n_paraffins", "iso_paraffins", "naphthenes", "aromatics")
_GROUP_KIND = {"n_paraffins": "nP", "iso_paraffins": "iP", "naphthenes": "N", "aromatics": "A"}


def _series(kind: str, attr: str) -> np.ndarray:
    keys = {"P": [f"nP{n}" for n in sp.CARBON_NUMBERS],
            "N": [f"N{n}" for n in sp.CARBON_NUMBERS],
            "A": [f"A{n}" for n in sp.CARBON_NUMBERS]}[kind]
    return np.array([getattr(sp.SPECIES[k], attr) for k in keys])



def _lump_cuts(m_cut: Array, Tb: Array, phi: Array, iso_fraction, mcp_fraction) -> Array:
    """Reformer-species molar flows (mol/s) of cuts: steps 1-3 of the module docstring.

    ``m_cut`` is each cut's mass flow (kg/s), ``Tb`` its boiling point (K) and
    ``phi`` its ``(n_cuts, 4)`` hydrocarbon-type VOLUME fractions over P, N,
    A, O (olefins are counted as paraffins). Mass is conserved to round-off:
    the type weights and the hat weights each sum to one per cut.
    """
    out = jnp.zeros(sp.N_SPECIES)
    cn = np.asarray(sp.CARBON_NUMBERS, dtype=float)
    kinds = ("P", "N", "A")
    # Carbon-number position and the type's density there, per cut and type.
    rho = []
    pos = []
    for t in kinds:
        Tb_t = jnp.asarray(_series(t, "Tb"))
        c = jnp.interp(Tb, Tb_t, jnp.asarray(cn))               # clipped at both ends
        pos.append(c)
        rho.append(jnp.interp(c, jnp.asarray(cn), jnp.asarray(_series(t, "rho60"))))
    rho = jnp.stack(rho, axis=1)                                 # (n_cuts, 3)
    vol = jnp.stack([phi[:, 0] + phi[:, 3], phi[:, 1], phi[:, 2]], axis=1)
    w = vol * rho
    w = w / jnp.sum(w, axis=1, keepdims=True)
    m_type = m_cut[:, None] * w                                  # (n_cuts, 3) kg/s
    # Hat-function weights onto C6..C10.
    for ti, t in enumerate(kinds):
        c = pos[ti]
        hat = jnp.clip(1.0 - jnp.abs(c[:, None] - jnp.asarray(cn)[None, :]), 0.0, 1.0)  # (n_cuts, 5)
        m_cn = m_type[:, ti] @ hat                                # (5,) kg/s
        for slot, n in enumerate(sp.CARBON_NUMBERS):
            if t == "P":
                targets = ((f"iP{n}", iso_fraction), (f"nP{n}", 1.0 - jnp.asarray(iso_fraction)))
            elif t == "N" and n == 6:
                targets = (("N5_6", mcp_fraction), ("N6", 1.0 - jnp.asarray(mcp_fraction)))
            else:
                targets = ((f"{t}{n}", 1.0),)
            for key, share in targets:
                j = sp.INDEX[key]
                out = out.at[j].add(share * m_cn[slot] / (sp.MW[j] / 1000.0))
    return out

@dataclass(frozen=True)
class NaphthaFeed:
    """A reformer feed: molar flows of the reformer species.

    Attributes:
        flows: ``(N_SPECIES,)`` molar flows (mol/s), in
            :data:`~difflow_refinery.reforming.species.NAMES` order. Hydrogen
            should be zero (the recycle supplies it).
        T: Feed temperature (K).
        P: Feed pressure (Pa).
        sulfur_wppm: Organic sulfur in the feed (wppm, mass basis), a trace
            element outside the species list. :class:`~.unit.CatalyticReformer`
            reads it when ``ReformerParams.feed_sulfur_wppm`` is ``None``
            (#330). :meth:`from_hydrotreater` sets it from the treated
            product; the other builders leave it at zero.
    """

    flows: Array
    T: Array = 311.0
    P: Array = 15.0e5
    sulfur_wppm: Array = 0.0

    # ----- builders -------------------------------------------------------

    @classmethod
    def from_piona(cls, piona: Mapping[str, Mapping[int, float]], total: float | Array,
                   basis: Literal["mass", "volume", "mole"] = "mass",
                   mcp_fraction: float | Array = 0.6, T: float = 311.0, P: float = 15.0e5,
                   light_ends: Mapping[str, float] | None = None) -> "NaphthaFeed":
        """A feed from a PIONA analysis by carbon number.

        Args:
            piona: ``{group: {carbon_number: fraction}}`` with groups from
                :data:`PIONA_GROUPS`, carbon numbers 6..10. Fractions are on
                ``basis`` and are normalised (with ``light_ends``) to one.
                Olefins, if analysed, should be added to the paraffins (the
                feed is hydrotreated).
            total: Feed rate: kg/s (``"mass"``), m^3/s at 60 F (``"volume"``)
                or mol/s (``"mole"``).
            basis: What the fractions and ``total`` measure.
            mcp_fraction: Share of the C6 naphthenes that is methylcyclopentane
                (when the analysis does not separate the two rings).
            light_ends: ``{reformer species: fraction}`` for C5 and lighter
                (``"iC5"``, ``"nC5"`` ...), on the same basis.
        """
        unknown = sorted(set(piona) - set(PIONA_GROUPS))
        if unknown:
            raise ValueError(f"unknown PIONA groups {unknown}; use {PIONA_GROUPS}")
        x = jnp.zeros(sp.N_SPECIES)
        for g, by_c in piona.items():
            for n, v in by_c.items():
                if n not in sp.CARBON_NUMBERS:
                    raise ValueError(f"carbon number {n} is outside {sp.CARBON_NUMBERS}")
                kind = _GROUP_KIND[g]
                v = jnp.asarray(v, dtype=float)
                if kind == "N" and n == 6:
                    x = x.at[sp.INDEX["N5_6"]].add(mcp_fraction * v)
                    x = x.at[sp.INDEX["N6"]].add((1.0 - mcp_fraction) * v)
                else:
                    x = x.at[sp.INDEX[f"{kind}{n}"]].add(v)
        for k, v in (light_ends or {}).items():
            if k not in sp.INDEX or k == "H2":
                raise ValueError(f"light end {k!r} is not a reformer species")
            x = x.at[sp.INDEX[k]].add(jnp.asarray(v, dtype=float))
        x = x / jnp.sum(x)
        MW = jnp.asarray(sp.MW)
        if basis == "mole":
            F = jnp.asarray(total, dtype=float) * x
        elif basis == "mass":
            F = jnp.asarray(total, dtype=float) * x / (MW / 1000.0)
        elif basis == "volume":
            v_per_mol = jnp.asarray(np.where(np.isfinite(sp.RHO60), sp.MW / 1000.0 / sp.RHO60, 0.0))
            F = jnp.asarray(total, dtype=float) * x / jnp.where(v_per_mol > 0, v_per_mol, 1.0)
        else:
            raise ValueError(f"basis must be 'mass', 'volume' or 'mole', not {basis!r}")
        return cls(F, jnp.asarray(T, dtype=float), jnp.asarray(P, dtype=float))

    @classmethod
    def from_characterization(cls, char, flows: Array | Mapping[str, Array],
                              iso_fraction: float | Array = 0.5, mcp_fraction: float | Array = 0.6,
                              T: float = 311.0, P: float = 15.0e5) -> "NaphthaFeed":
        """A feed from characterization components and their molar flows.

        Args:
            char: A :class:`~difflow_refinery.Characterization` carrying a
                ``composition`` (``characterize(assay, composition=True)``).
            flows: Molar flows (mol/s) of ``char.names`` -- an ``(n,)`` array,
                or a difflow stream / ``{name: flow}`` mapping (missing names
                are zero), e.g. the CDU's naphtha product.
            iso_fraction: Mass share of each paraffin lump that is branched.
            mcp_fraction: Mass share of the C6 naphthenes that is
                methylcyclopentane.

        See the module docstring for the mapping.
        """
        comp = char.composition
        if comp is None:
            raise ValueError("the characterization has no composition: use "
                             "characterize(assay, composition=True) or char.with_composition()")
        names = char.names
        if isinstance(flows, Mapping):
            z = jnp.asarray(0.0)
            Fc = jnp.stack([jnp.asarray(flows.get(f"F_{n}", flows.get(n, z)), dtype=float) for n in names])
        else:
            Fc = jnp.asarray(flows, dtype=float)
        mass = Fc * jnp.asarray(char.component_MW) / 1000.0          # kg/s per component
        out = jnp.zeros(sp.N_SPECIES)
        k = len(char.light_names)
        for i, n in enumerate(char.light_names):
            if n in LIGHT_END_MAP:
                j = sp.INDEX[LIGHT_END_MAP[n]]
                out = out.at[j].add(mass[i] / (sp.MW[j] / 1000.0))
        phi = jnp.asarray(comp.hc_type)[k:]                         # (n_cuts, 4) P N A O
        out = out + _lump_cuts(mass[k:], jnp.asarray(char.Tb), phi, iso_fraction, mcp_fraction)
        return cls(out, jnp.asarray(T, dtype=float), jnp.asarray(P, dtype=float))

    @classmethod
    def from_hydrotreater(cls, source, product: str | tuple | None = None, *, char=None,
                          iso_fraction: float | Array = 0.5, mcp_fraction: float | Array = 0.6,
                          drop_gases: bool = False, T: float = 311.0, P: float = 15.0e5) -> "NaphthaFeed":
        """A feed from a hydrotreated naphtha, on its TREATED composition (#327).

        Args:
            source: A ``HydrotreaterResult`` (outlet ``product``, default the
                stripper bottoms ``"product"``; a tuple such as ``("product",
                "wild_naphtha")`` sums outlets), a ``FractionationResult``
                (``product`` names one of its products, e.g.
                ``"heavy_naphtha"``), anything else with ``product_stream`` /
                ``product_char`` (a hydrocracker's naphtha), or an ``F_``
                stream with ``char=`` its
                :class:`~difflow_refinery.BlendCharacterization`
                (:func:`difflow_refinery.gasplant.hydroprocessed.resolve_product`).
            iso_fraction, mcp_fraction: As :meth:`from_characterization`.
            drop_gases: Leave out the dissolved H2, H2S, NH3 (and water) a
                wild naphtha carries; without it they are refused. Methane
                and ethane map to ``C1`` and ``C2`` either way.

        What differs from :meth:`from_characterization` is where the cut data
        come from: the hydrotreater's product grid (``product_char``), not the
        crude's characterization.

        * **Molar masses.** Each component's mass is its flow times the
          GRID's molar mass -- the treated cut's, and for light ends and gases
          the atomic weights the hydrotreater's balances close on -- so the
          feed's mass is the product's (``res.stream_mass``) to round-off.
          ``from_characterization`` on the same flows uses the untreated
          crude's molar masses (example 40 lost 4e-4 of the mass that way).
        * **Types.** The cuts' hydrocarbon types are the grid's
          ``paraffins_vol``, ``naphthenes_vol``, ``aromatics_vol`` and
          ``olefins_vol`` qualities: what is left after the hydrotreater's
          aromatics saturation and olefin hydrogenation. (The hydrotreater
          tracks them per molecule and reports them back on the rule it read
          them in by, so an unreacted cut maps exactly as
          ``from_characterization`` maps it.) Olefins, if any are left, count
          as paraffins. Placement on the carbon-number grid (by the grid's
          ``Tb``) and the iso/normal and MCP/cyclohexane splits are the
          module docstring's steps 2-3.
        * **Sulfur.** The grid's ``S_ppm`` quality, mass-averaged over the
          feed, becomes :attr:`sulfur_wppm`, which
          :class:`~difflow_refinery.reforming.CatalyticReformer` takes as the
          feed's organic sulfur when ``ReformerParams.feed_sulfur_wppm`` is
          ``None`` (#330). Dissolved H2S is not organic sulfur and is not
          counted.

        Traceable in the flows and the grid's arrays, so a reformer output is
        differentiable back through the hydrotreater.
        """
        from difflow_refinery.gasplant.hydroprocessed import resolve_product
        from difflow_refinery.hydroprocessing.layout import gas_mw
        from difflow_refinery.thermo import LIGHT_ENDS

        stream, grid = resolve_product(source, product, char)
        names = list(grid.names)
        q = grid.qualities
        missing = [k for k in ("paraffins_vol", "naphthenes_vol", "aromatics_vol") if k not in q]
        if missing:
            raise ValueError(f"the product grid has no {missing} qualities: it carries no hydrocarbon types")
        z = jnp.asarray(0.0)
        Fg = jnp.stack([jnp.asarray(stream.get(f"F_{n}", z), dtype=float) for n in names])
        mass = Fg * jnp.asarray(grid.MW, dtype=float) / 1000.0     # kg/s per grid component
        out = jnp.zeros(sp.N_SPECIES)
        cut_i = []
        for i, n in enumerate(names):
            if n in LIGHT_END_MAP:
                j = sp.INDEX[LIGHT_END_MAP[n]]
                out = out.at[j].add(mass[i] / (sp.MW[j] / 1000.0))
            elif n in LIGHT_ENDS:
                raise ValueError(f"light end {n!r} has no reformer species")
            else:
                cut_i.append(i)
        m_hc = jnp.sum(mass)
        for k, v in stream.items():
            if not k.startswith("F_") or k[2:] in names:
                continue
            g = k[2:]
            if g in LIGHT_END_MAP:
                j = sp.INDEX[LIGHT_END_MAP[g]]
                m = jnp.asarray(v, dtype=float) * gas_mw(g) / 1000.0
                out = out.at[j].add(m / (sp.MW[j] / 1000.0))
                m_hc = m_hc + m
            elif not drop_gases:
                raise ValueError(f"{g!r} is dissolved in the product and is not a reformer feed species; "
                                 "pass drop_gases=True (or fractionate it off first)")
        if cut_i:
            ci = jnp.asarray(cut_i)
            o = q["olefins_vol"][ci] if "olefins_vol" in q else jnp.zeros(len(cut_i))
            phi = jnp.stack([q["paraffins_vol"][ci], q["naphthenes_vol"][ci], q["aromatics_vol"][ci], o],
                            axis=1) / 100.0
            out = out + _lump_cuts(mass[ci], jnp.asarray(grid.Tb, dtype=float)[ci], phi,
                                   iso_fraction, mcp_fraction)
        S = (jnp.sum(mass * q["S_ppm"]) / jnp.maximum(m_hc, 1e-300)) if "S_ppm" in q else jnp.asarray(0.0)
        return cls(out, jnp.asarray(T, dtype=float), jnp.asarray(P, dtype=float), S)

    # ----- edits and views -------------------------------------------------

    def with_group_fraction(self, group: Literal["paraffins", "naphthenes", "aromatics"],
                            value: Array, basis: Literal["mass", "volume"] = "volume") -> "NaphthaFeed":
        """A feed with one group's total fraction set to ``value`` (0..1).

        The group's carbon-number (and isomer) distribution is kept; the other
        groups are rescaled together, so the total mass (``basis="mass"``) or
        standard volume (``"volume"``) is unchanged. The lever behind "the
        naphtha's N content".
        """
        kind = {"paraffins": ("nP", "iP", "light"), "naphthenes": ("N",), "aromatics": ("A",)}[group]
        mask = jnp.asarray([sp.SPECIES[k].kind in kind for k in sp.NAMES])
        per_mol = (jnp.asarray(sp.MW) / 1000.0 if basis == "mass"
                   else jnp.asarray(np.where(np.isfinite(sp.RHO60), sp.MW / 1000.0 / sp.RHO60, 0.0)))
        q = self.flows * per_mol
        Q = jnp.sum(q)
        q_in = jnp.sum(jnp.where(mask, q, 0.0))
        value = jnp.asarray(value, dtype=float)
        scale = jnp.where(mask, value * Q / q_in, (1.0 - value) * Q / (Q - q_in))
        return NaphthaFeed(self.flows * scale, self.T, self.P, self.sulfur_wppm)

    def scaled(self, factor: Array) -> "NaphthaFeed":
        """The same feed at ``factor`` times the rate (sulfur content unchanged)."""
        return NaphthaFeed(self.flows * factor, self.T, self.P, self.sulfur_wppm)

    @property
    def mass_flow(self) -> Array:
        """kg/s."""
        return self.flows @ jnp.asarray(sp.MW / 1000.0)

    @property
    def volume_flow(self) -> Array:
        """Standard liquid volume, m^3/s at 60 F."""
        return th.std_liquid_volume(self.flows)

    @property
    def hydrocarbon_moles(self) -> Array:
        """mol/s of everything but hydrogen (the H2/HC ratio's denominator)."""
        return jnp.sum(self.flows) - self.flows[sp.INDEX["H2"]]

    def group_fractions(self, basis: Literal["mass", "volume", "mole"] = "volume") -> dict[str, Array]:
        """P (n + iso + light), N, A fractions on ``basis``."""
        per = {"mass": jnp.asarray(sp.MW), "mole": jnp.ones(sp.N_SPECIES),
               "volume": jnp.asarray(np.where(np.isfinite(sp.RHO60), sp.MW / sp.RHO60, 0.0))}[basis]
        q = self.flows * per
        out = {}
        for g, kinds in (("paraffins", ("nP", "iP", "light")), ("naphthenes", ("N",)), ("aromatics", ("A",))):
            m = jnp.asarray([sp.SPECIES[k].kind in kinds for k in sp.NAMES])
            out[g] = jnp.sum(jnp.where(m, q, 0.0)) / jnp.sum(q)
        return out

    def n_plus_2a(self) -> Array:
        """``N + 2A`` (vol%), the reformability index of a naphtha."""
        g = self.group_fractions("volume")
        return 100.0 * (g["naphthenes"] + 2.0 * g["aromatics"])

    def hydrogen_wt(self) -> Array:
        """Hydrogen content (wt%) of the lumped feed."""
        m = self.flows * jnp.asarray(sp.MW)
        h = self.flows * jnp.asarray(sp.ELEMENTS[1]) * 1.00794
        return 100.0 * jnp.sum(h) / jnp.sum(m)

    def stream(self):
        """The feed as a difflow stream."""
        from difflow_refinery.reforming.reactor import stream_of
        return stream_of(self.flows, self.T, self.P)


jax.tree_util.register_dataclass(NaphthaFeed, data_fields=["flows", "T", "P", "sulfur_wppm"], meta_fields=[])


# -----------------------------------------------------------------------------
# Two illustrative naphthas
# -----------------------------------------------------------------------------

def lean_naphtha(total_kg_s: float = 10.0) -> NaphthaFeed:
    """An illustrative LEAN (high-paraffin) hydrotreated heavy naphtha.

    About 67 vol% paraffins, 24 % naphthenes, 8 % aromatics (N + 2A ~ 41).
    The distribution is made up for this module to be typical of a
    paraffinic straight-run heavy naphtha -- not any crude's assay.
    """
    return NaphthaFeed.from_piona({
        "n_paraffins": {6: 4.0, 7: 9.0, 8: 9.0, 9: 7.0, 10: 4.0},
        "iso_paraffins": {6: 4.0, 7: 9.0, 8: 9.0, 9: 7.0, 10: 3.0},
        "naphthenes": {6: 4.0, 7: 7.0, 8: 7.0, 9: 5.0, 10: 3.0},
        "aromatics": {6: 1.0, 7: 3.0, 8: 3.0, 9: 2.0, 10: 1.0},
    }, total_kg_s, basis="mass")


def rich_naphtha(total_kg_s: float = 10.0) -> NaphthaFeed:
    """An illustrative RICH (high-naphthene) hydrotreated heavy naphtha.

    About 39 vol% paraffins, 47 % naphthenes, 14 % aromatics (N + 2A ~ 75),
    made up for this module to be typical of a naphthenic crude's heavy
    naphtha -- not any crude's assay.
    """
    return NaphthaFeed.from_piona({
        "n_paraffins": {6: 2.0, 7: 5.0, 8: 5.0, 9: 4.0, 10: 2.0},
        "iso_paraffins": {6: 2.0, 7: 5.0, 8: 5.0, 9: 4.0, 10: 2.0},
        "naphthenes": {6: 6.0, 7: 13.0, 8: 13.0, 9: 10.0, 10: 6.0},
        "aromatics": {6: 1.0, 7: 4.0, 8: 5.0, 9: 4.0, 10: 2.0},
    }, total_kg_s, basis="mass")
