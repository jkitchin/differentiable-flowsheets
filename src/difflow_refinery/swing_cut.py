"""Swing-cut crude unit planning block (#531).

A planning model represents a crude unit as cut yields per crude. Each crude
is split into heart cuts and *swing cuts*. A swing cut is the material
between two products that the planner may send to either one. This is the
formulation of Du et al. (2025), arXiv:2503.22057, sections 3.1-3.2 (``y``,
``FQcut``, ``phi``, ``SC``), and :func:`swing_cut_block` evaluates it:

* the decisions are the crude rates and the swing fractions ``phi`` in
  [0, 1];
* the outputs are the product flows and properties. Specific gravity blends
  by volume and every other property by mass, unless it is declared
  volume-based.

The cut table comes from one of three places:

* the benchmark's published ``y`` / ``FQcut`` / ``phi``
  (:func:`difflow_refinery.benchmarks.du2025.crude.cut_table`);
* an assay cut at TBP cut points (:func:`cut_table_from_assays`). This is the
  ideal-fractionation limit, and it is differentiable with respect to the
  cut points;
* the rigorous :class:`~difflow_refinery.unit.CrudeUnit`
  (:func:`cut_table_from_crude_unit`). There each swing cut is the material
  a product gains when its yield spec moves up by the swing's width. The cut
  yields and properties are then the column's own response, and AD through
  the column solve gives their Jacobian.

**Swing-cut properties.** The published formulation gives a swing cut one
set of properties, whichever way it goes. The light end of a swing cut is
lighter than its heavy end, though, so the share that goes to the lighter
product carries lighter properties. With ``varying=True``, each property is
taken linear across the swing cut's cumulative span, centred on the cut's
own value ``q`` with slope ``(q_next - q_prev) / 2``, where ``q_prev`` and
``q_next`` are its neighbouring cuts in the table. Its mean stays ``q``,
and consecutive slices go to the recipients from light to heavy. The two
forms agree at ``phi`` = 0 and 1. :func:`swing_property_error` measures the
difference in between.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

_TINY = 1e-30


def _safe_div(a, b):
    ok = jnp.abs(b) > _TINY
    return jnp.where(ok, a / jnp.where(ok, b, 1.0), 0.0)


@dataclass(frozen=True)
class SwingCutTable:
    """Cut yields and properties for a crude unit, with its swing structure.

    Attributes:
        cuts: Cut names, light to heavy (heart and swing cuts).
        products: Product (outlet) names.
        own: ``{product: cut}`` -- the cut a product takes whole (its heart
            cut); a product may have none.
        swings: ``((product, swing cut), ...)`` -- the swing cut may go,
            in part, to the product. A swing cut that is also a product keeps
            what its recipients do not take.
        crudes: Crude names.
        yields: ``(n_crude, n_cut)`` mass yields.
        props: ``{q: (n_crude, n_cut)}`` cut properties; NaN where unknown
            (it blends as 0, as in the benchmark's Eq. 5).
        spg: The key of specific gravity in ``props``, blended by volume.
        volume_props: Other keys blended by volume (weighted ``F / SG``).
    """

    cuts: tuple[str, ...]
    products: tuple[str, ...]
    own: Mapping[str, str]
    swings: tuple[tuple[str, str], ...]
    crudes: tuple[str, ...]
    yields: np.ndarray
    props: Mapping[str, np.ndarray]
    spg: str = "SG"
    volume_props: frozenset = field(default_factory=frozenset)

    def __post_init__(self):
        cut_set, prod_set = set(self.cuts), set(self.products)
        for p, c in self.own.items():
            if p not in prod_set or c not in cut_set:
                raise ValueError(f"own[{p!r}] = {c!r}: unknown product or cut")
        for p, c in self.swings:
            if p not in prod_set or c not in cut_set:
                raise ValueError(f"swing ({p!r}, {c!r}): unknown product or cut")
        if self.yields.shape != (len(self.crudes), len(self.cuts)):
            raise ValueError("yields must be (n_crude, n_cut)")
        if self.spg not in self.props:
            raise ValueError(f"props has no specific gravity under {self.spg!r}")

    @property
    def n_swing(self) -> int:
        return len(self.swings)

    def merge_matrix(self, phi: Array) -> Array:
        """``(n_product, n_cut)`` share of each cut that ends in each product.

        Eq. 4 of the benchmark: 1 for a product's own cut, ``phi`` for a swing
        cut merged into it, and ``-sum(phi)`` on a swing cut's own outlet.
        """
        phi = jnp.asarray(phi, dtype=float)
        pi = {p: i for i, p in enumerate(self.products)}
        ci = {c: j for j, c in enumerate(self.cuts)}
        M = jnp.zeros((len(self.products), len(self.cuts)))
        for p, c in self.own.items():
            M = M.at[pi[p], ci[c]].add(1.0)
        for k, (p, c) in enumerate(self.swings):
            M = M.at[pi[p], ci[c]].add(phi[k])
            if c in self.own.values():
                owner = [q for q, cc in self.own.items() if cc == c][0]
                M = M.at[pi[owner], ci[c]].add(-phi[k])
        return M

    def swing_slices(self, phi: Array):
        """For ``varying=True``: the cumulative span ``[a, b]`` of each swing
        cut that each recipient takes, as ``(lo, hi)``, ``(n_product, n_cut)``
        arrays; ``lo = hi = 0`` where no slice.

        Recipients of a swing cut are ordered light to heavy by their own
        cut's position in :attr:`cuts` (a product with no own cut goes
        first). The swing cut's own outlet sits at the swing cut's position.
        """
        phi = jnp.asarray(phi, dtype=float)
        pi = {p: i for i, p in enumerate(self.products)}
        ci = {c: j for j, c in enumerate(self.cuts)}
        lo = jnp.zeros((len(self.products), len(self.cuts)))
        hi = jnp.zeros_like(lo)
        by_cut: dict[str, list] = {}
        for k, (p, c) in enumerate(self.swings):
            by_cut.setdefault(c, []).append((p, phi[k]))
        for c, recips in by_cut.items():
            owner = [q for q, cc in self.own.items() if cc == c]
            entries = [(ci.get(self.own.get(p), -1), p, share) for p, share in recips]
            if owner:
                rest = 1.0 - sum(s for _, s in recips)
                entries.append((ci[c], owner[0], rest))
            entries.sort(key=lambda e: e[0])
            start = 0.0
            for _, p, share in entries:
                lo = lo.at[pi[p], ci[c]].set(start)
                hi = hi.at[pi[p], ci[c]].set(start + share)
                start = start + share
        return lo, hi


def _edge_values(table: SwingCutTable, q: str) -> tuple[np.ndarray, np.ndarray]:
    """Light and heavy edge values of every cut's property ``q`` (neighbour
    means; a cut with no neighbour, or no value there, is flat)."""
    v = np.asarray(table.props[q], dtype=float)
    prev = np.concatenate([v[:, :1], v[:, :-1]], axis=1)
    nxt = np.concatenate([v[:, 1:], v[:, -1:]], axis=1)
    prev = np.where(np.isfinite(prev), prev, v)
    nxt = np.where(np.isfinite(nxt), nxt, v)
    return 0.5 * (prev + v), 0.5 * (v + nxt)


def swing_cut_block(table: SwingCutTable, crude_rates: Array, phi: Array, *,
                    varying: bool | Sequence[str] = False) -> tuple[Array, dict[str, Array]]:
    """Product flows and properties of a swing-cut crude unit.

    Args:
        table: The :class:`SwingCutTable`.
        crude_rates: ``(n_crude,)`` crude mass rates.
        phi: ``(n_swing,)`` swing fractions, in ``table.swings`` order.
        varying: Let swing-cut properties vary across the cut (module
            docstring): ``True`` for every swing cut, or the names of the
            swing cuts that vary. ``False`` is the published
            constant-property form.

    Returns:
        ``(flows (n_product,), {q: (n_product,)})``. Specific gravity is by
        volume, ``volume_props`` by volume, every other property by mass.
        A product with no flow gets 0.

    Example:
        >>> t = SwingCutTable(("a", "s", "b"), ("A", "B"), {"A": "a", "B": "b"},
        ...                   (("A", "s"), ("B", "s")), ("c",),
        ...                   np.array([[0.4, 0.2, 0.4]]),
        ...                   {"SG": np.array([[0.7, 0.8, 0.9]])})
        >>> F, q = swing_cut_block(t, jnp.array([10.0]), jnp.array([0.5, 0.5]))
        >>> [round(float(f), 6) for f in F]
        [5.0, 5.0]
    """
    rates = jnp.asarray(crude_rates, dtype=float)
    M = table.merge_matrix(phi)
    mass = jnp.asarray(table.yields) * rates[:, None]          # (crude, cut)
    flows = M @ jnp.sum(mass, axis=0)
    sg = jnp.asarray(np.nan_to_num(table.props[table.spg]))

    if not varying:
        def avg_mass(q_cut):                                  # (crude, cut)
            return _safe_div(M @ jnp.sum(mass * q_cut, axis=0), flows)
        vol = M @ jnp.sum(_safe_div(mass, sg), axis=0)
        out = {table.spg: _safe_div(flows, vol)}
        for q, v in table.props.items():
            if q == table.spg:
                continue
            vq = jnp.asarray(np.nan_to_num(v))
            if q in table.volume_props:
                out[q] = _safe_div(M @ jnp.sum(_safe_div(mass, sg) * vq, axis=0), vol)
            else:
                out[q] = avg_mass(vq)
        return flows, out

    # varying: a recipient's slice [a, b] of a swing cut carries the mean of
    # the cut's linear profile over [a, b], i.e. the value at (a + b) / 2
    lo, hi = table.swing_slices(phi)
    is_swing = np.zeros(len(table.cuts), bool)
    chosen = None if varying is True else set(varying)
    for _, c in table.swings:
        if chosen is None or c in chosen:
            is_swing[table.cuts.index(c)] = True
    if chosen is not None and not chosen <= {c for _, c in table.swings}:
        raise ValueError(f"varying names cuts that are not swing cuts: "
                         f"{sorted(chosen - {c for _, c in table.swings})}")
    width = hi - lo                                            # (prod, cut)
    share = jnp.where(is_swing[None, :], width, M)             # mass share
    mid = 0.5 * (lo + hi)

    def slice_value(q):
        e_lo, e_hi = (jnp.asarray(np.nan_to_num(e)) for e in _edge_values(table, q))
        vq = jnp.asarray(np.nan_to_num(table.props[q]))
        # (prod, crude, cut): the value each product sees of each cut
        # centred on the cut's own value: its mean over [0, 1] stays vq
        prof = vq[None] + (e_hi - e_lo)[None] * (mid[:, None, :] - 0.5)
        return jnp.where(is_swing[None, None, :], prof, vq[None])

    flows = jnp.einsum("pk,ck->p", share, mass)
    sgv = slice_value(table.spg)
    vol = jnp.einsum("pk,ck,pck->p", share, mass, 1.0 / jnp.where(sgv > 0, sgv, 1.0))
    out = {table.spg: _safe_div(flows, vol)}
    for q in table.props:
        if q == table.spg:
            continue
        qv = slice_value(q)
        if q in table.volume_props:
            num = jnp.einsum("pk,ck,pck->p", share, mass, qv / jnp.where(sgv > 0, sgv, 1.0))
            out[q] = _safe_div(num, vol)
        else:
            out[q] = _safe_div(jnp.einsum("pk,ck,pck->p", share, mass, qv), flows)
    return flows, out


def swing_property_error(table: SwingCutTable, crude_rates: Array, phi0: Array,
                         swing: int, q: str, n: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Varying minus constant swing-cut property ``q`` across one swing range.

    Args:
        table, crude_rates: As :func:`swing_cut_block`.
        phi0: Swing fractions held for the other swings.
        swing: Index (into ``table.swings``) of the swing moved from 0 to 1.
        q: Property.
        n: Points.

    Returns:
        ``(phi grid (n,), error (n, n_product))``. Only this swing's cut
        varies, so the error is this swing's alone.
    """
    grid = np.linspace(0.0, 1.0, n)
    errs = []
    for f in grid:
        ph = jnp.asarray(phi0, dtype=float).at[swing].set(f)
        # keep a shared swing cut's fractions summing to at most 1
        c = table.swings[swing][1]
        others = [k for k, (_, cc) in enumerate(table.swings) if cc == c and k != swing]
        tot = f + sum(float(ph[k]) for k in others)
        if tot > 1.0 and others:
            scale = (1.0 - f) / max(sum(float(ph[k]) for k in others), _TINY)
            for k in others:
                ph = ph.at[k].multiply(scale)
        _, a = swing_cut_block(table, crude_rates, ph, varying=False)
        _, b = swing_cut_block(table, crude_rates, ph, varying=[c])
        errs.append(np.asarray(b[q] - a[q]))
    return grid, np.array(errs)


# --------------------------------------------------------------------------- #
# Generators                                                                  #
# --------------------------------------------------------------------------- #

def _aggregate(char, n_cut: int):
    """Mass fraction, SG, sulfur wt%, N wppm, CCR wt% per cut of a
    characterization cut at ``n_cut - 1`` cut points: light ends join the
    first cut, a residue lump (and anything past the last cut point) the
    last."""
    mf = jnp.asarray(char.mass_fraction)
    sg = jnp.asarray(char.component_SG)
    n_light = len(char.light_names)
    n_pc = len(char.pseudo_names)
    idx = np.concatenate([np.zeros(n_light, int),
                          np.minimum(np.arange(n_pc), n_cut - 1)])
    seg = lambda v: jax.ops.segment_sum(v, jnp.asarray(idx), num_segments=n_cut)  # noqa: E731
    mass = seg(mf)
    out = {"SG": _safe_div(mass, seg(_safe_div(mf, sg)))}
    pad = lambda v: jnp.concatenate([jnp.zeros(n_light), jnp.asarray(v)])  # noqa: E731
    for key, attr, unit in (("S_wt", "sulfur", 100.0), ("N_wppm", "nitrogen", 1e6),
                            ("CCR_wt", "ccr", 100.0)):
        v = getattr(char, attr, None)
        if v is None:
            continue
        v = jnp.asarray(v)
        v = v if v.shape[0] == mf.shape[0] else pad(v)
        out[key] = _safe_div(seg(mf * v), mass) * unit
    return mass, out


def cut_table_from_assays(assays: Sequence, cut_points: Array, cuts: Sequence[str],
                          products: Sequence[str], own: Mapping[str, str],
                          swings: Sequence[tuple[str, str]], *, method: str | None = None,
                          as_arrays: bool = False):
    """A :class:`SwingCutTable` from assays cut at TBP cut points (ideal cuts).

    Args:
        assays: One :class:`~difflow_refinery.assay.Assay` per crude.
        cut_points: ``(n_cut - 1,)`` TBP cut points (K), light to heavy.
        cuts, products, own, swings: The structure, as in
            :class:`SwingCutTable`; ``cuts`` light to heavy.
        method: Critical-property method for ``characterize``.
        as_arrays: Return ``(yields, props)`` as JAX arrays instead of a
            table -- the traceable form, for a Jacobian with respect to
            ``cut_points``.

    Returns:
        A :class:`SwingCutTable` (SG, ``S_wt``, ``N_wppm``, ``CCR_wt``), or
        ``(yields (n_crude, n_cut), {q: (n_crude, n_cut)})``.
    """
    from difflow_refinery.assay import characterize

    cps = jnp.asarray(cut_points, dtype=float)
    if cps.shape[0] != len(cuts) - 1:
        raise ValueError(f"{len(cuts)} cuts need {len(cuts) - 1} cut points, got {cps.shape[0]}")
    ys, ps = [], []
    for a in assays:
        m, p = _aggregate(characterize(a, cut_points=cps, method=method), len(cuts))
        ys.append(m)
        ps.append(p)
    yields = jnp.stack(ys)
    props = {k: jnp.stack([p[k] for p in ps]) for k in ps[0]}
    if as_arrays:
        return yields, props
    return SwingCutTable(tuple(cuts), tuple(products), dict(own), tuple(swings),
                         tuple(getattr(a, "name", f"crude{i}") for i, a in enumerate(assays)),
                         np.asarray(yields), {k: np.asarray(v) for k, v in props.items()})


def cut_table_from_crude_unit(unit, rate, T, P, swings: Sequence[tuple[str, str, float]], *,
                              basis: str = "bpd") -> SwingCutTable:
    """A one-crude :class:`SwingCutTable` from the rigorous crude unit.

    The base solve has every swing cut in its heavier product, at the
    column's own specs. For swing ``(light, heavy, width)``, the light
    product's volume-rate spec is raised by ``width``, as a volume fraction
    of the crude, and the column is solved again. The swing cut is the
    material the light product gains: its mass, and the mass-weighted
    difference of the light product's properties. The heavy product's heart
    cut is its base flow less its swing cuts. So cut yields sum to 1, and
    every number is the column's response, not a TBP idealization. Each
    swing needs one extra solve, so this is slow: build it once, not inside
    a loop.

    Args:
        unit: A converged :class:`~difflow_refinery.unit.CrudeUnit` whose
            specs include a volume ``product_rate`` for every light product.
        rate, T, P, basis: As :meth:`CrudeUnit.solve`.
        swings: ``(light product, heavy product, width)`` per swing cut.

    Returns:
        A table with products = the column's products (no offgas), cuts =
        one heart cut per product plus one swing cut per swing (named
        ``"light/heavy"``), and properties ``SG`` and ``S_wt`` (from the
        characterization's sulfur, zero if the assay gives none).

    Raises:
        ValueError: If a light product has no volume rate spec, or a solve
            does not converge.
    """
    from dataclasses import replace

    from difflow_refinery.column import BARREL

    crude = unit.crude
    S = jnp.asarray(crude.sulfur) if getattr(crude, "sulfur", None) is not None else None

    def solve(params=None):
        r = unit.solve(rate, T, P, basis=basis, params=params)
        if not bool(r.converged):
            raise ValueError("crude unit did not converge while generating the cut table")
        return r

    def summary(r):
        out = {}
        for n, p in r.properties.items():
            if n == "offgas":
                continue
            s_mass = 0.0
            if S is not None:
                comp = r.products[n]
                flows = jnp.asarray([comp.get(f"F_{c}", 0.0) for c in crude.names])
                s_mass = jnp.sum(flows * jnp.asarray(crude.component_MW) * 1e-3 * S)
            out[n] = (p.mass, p.volume, s_mass)
        return out

    base = summary(solve())
    p0 = unit.params
    swing_data = []
    for light, heavy, width in swings:
        idx = [i for i, s in enumerate(p0.specs)
               if s.kind == "product_rate" and s.target == light and s.basis == "volume"]
        if not idx:
            raise ValueError(f"{light!r} has no volume product_rate spec to move")
        i = idx[0]
        feed_volume = rate * BARREL / 86400.0 if basis == "bpd" else rate
        specs = list(p0.specs)
        specs[i] = replace(specs[i], value=specs[i].value + width * feed_volume)
        moved = summary(solve(replace(p0, specs=tuple(specs))))
        dm = moved[light][0] - base[light][0]
        dv = moved[light][1] - base[light][1]
        ds = moved[light][2] - base[light][2]
        swing_data.append((light, heavy, dm, dv, ds))

    products = tuple(base)
    heart = {n: list(base[n]) for n in products}
    for light, heavy, dm, dv, ds in swing_data:
        heart[heavy][0] = heart[heavy][0] - dm
        heart[heavy][1] = heart[heavy][1] - dv
        heart[heavy][2] = heart[heavy][2] - ds
    total_mass = sum(base[n][0] for n in products)   # = hearts + swings
    cuts, ys, sgs, ss = [], [], [], []
    rho = lambda m, v: _safe_div(m, v) / 999.016  # noqa: E731  water at 60 F, kg/m3
    for n in products:
        cuts.append(n)
        m, v, s = heart[n]
        ys.append(m / total_mass)
        sgs.append(rho(m, v))
        ss.append(100.0 * _safe_div(s, m))
    sw = []
    for light, heavy, dm, dv, ds in swing_data:
        name = f"{light}/{heavy}"
        cuts.append(name)
        ys.append(dm / total_mass)
        sgs.append(rho(dm, dv))
        ss.append(100.0 * _safe_div(ds, dm))
        sw.append(((light, name), (heavy, name)))
    order = np.argsort([float(g) for g in sgs])
    cuts = tuple(cuts[k] for k in order)
    yl = np.array([[float(ys[k]) for k in order]])
    props = {"SG": np.array([[float(sgs[k]) for k in order]]),
             "S_wt": np.array([[float(ss[k]) for k in order]])}
    swings_t = tuple(pair for two in sw for pair in two)
    # the heavy product does not "own" the swing cut: both recipients take
    # a share, phi_light + phi_heavy = 1
    return SwingCutTable(cuts, products, {n: n for n in products}, swings_t,
                         (unit.assay.name,), yl, props)
