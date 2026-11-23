"""Crude assays reconstructed from the benchmark's cut data (#532).

The benchmark gives each crude only as cut yields ``y`` and cut properties
``FQcut`` on two crude units. The cut temperatures are anonymized and
``FQcrd`` is empty. :func:`difflow_refinery.assay.characterize` needs a TBP
curve, so this module fits one per crude.

**Model.** Every crude in a crude unit is cut at the same temperatures,
which the data anonymize. So a crude's cumulative yield at each cut point of
each unit is a point on its TBP curve at an unknown temperature that it
shares with the other crudes. :func:`fit_tbp` fits those temperatures jointly
over all 25 crudes and both units. The criterion is that each crude's
Watson K, computed from each cut's mid boiling point and its SG, follows a
smooth linear trend in the cumulative yield (Riazi, *Characterization and
Properties of Petroleum Fractions*, ASTM MNL50, 2005, ch. 2-4). A prior
toward :data:`TYPICAL_CUTS_C` sets the scale. The data alone do not fix it:
multiplying every temperature by ``f`` and every K by ``f^(1/3)`` changes
nothing. The cut order is light to heavy by mean SG, and the swing cuts fall
between the heart cuts they merge into.

The TBP curve then passes through the crude's points from both units by
construction (the yields are reproduced; a crude whose two units disagree on
the order shows it as a yield error). Its gravity curve is the cut SG at
each cut's mid-point. Read the temperatures as plausible, not measured.

**Feed properties.** :func:`crude_assay` turns a fitted crude into an
:class:`~difflow_refinery.assay.Assay` on a mass basis, with a heavy end. The
measured cut sulfur (``Q0``), nitrogen (``Q18``) and Conradson carbon
(``Q21``) become its ``*_curve`` fields, and the cut Ni+V (``Q25``) becomes
its bulk value. The benchmark does not carry asphaltenes; they are
ILLUSTRATIVE, set from CCR by :data:`ASPHALTENES_PER_CCR`.
``characterize(..., composition=True)`` then gives PNA and hydrogen per cut
from difflow's correlations.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from difflow_refinery.assay import Assay, HeavyEnd
from difflow_refinery.benchmarks.du2025.data import Du2025Data

#: Target tolerances from #532: cut yields within 0.5 wt%, SG within 0.005.
YIELD_TOL = 0.005
SG_TOL = 0.005
#: Prior spread of a cut point around its typical value (K).
PRIOR_SIGMA_K = 30.0

#: Typical TBP cut points (deg C) by position between consecutive cuts, light
#: to heavy: light naphtha | swing | heavy naphtha | swing | kerosene |
#: diesel | swing | swing | gas oil | VGO | (HVGO | swing) | vacuum residue.
#: A crude unit with fewer cuts uses the leading entries and the last ones.
TYPICAL_CUTS_C = (90.0, 100.0, 165.0, 175.0, 230.0, 270.0, 290.0, 310.0,
                  360.0, 500.0, 520.0, 540.0)

#: ILLUSTRATIVE: C7 asphaltenes per unit CCR of the residue. No data; vacuum
#: residues run 0.2-0.6, so 0.35 is a mid value.
ASPHALTENES_PER_CCR = 0.35

#: ILLUSTRATIVE: nitrogen (wppm) anchored at the lightest cut's mid-point;
#: straight-run naphtha carries ~0.1-1 wppm.
LIGHT_NITROGEN_WPPM = 0.5

_SPG = "SPG"
_S, _N, _CCR, _NIV = "Q0", "Q18", "Q21", "Q25"


@dataclass(frozen=True)
class CutTable:
    """One crude unit's cuts for every crude, light to heavy.

    Attributes:
        unit: The crude unit, e.g. ``"UCDU0"``.
        batch: Its batch, e.g. ``"m0"``.
        cuts: Cut stream names, light to heavy.
        crudes: Raw-material stream of each crude (``S_M``), in a fixed order
            shared by every unit.
        feeds: The unit's inlet stream for each crude (the ``FQcut`` index).
        yields: ``(n_crude, n_cut)`` mass yields (rows sum to 1).
        props: ``{q: (n_crude, n_cut)}`` cut properties (NaN where absent);
            the gravity is also under ``"SPG"``.
    """

    unit: str
    batch: str
    cuts: tuple[str, ...]
    crudes: tuple[str, ...]
    feeds: tuple[str, ...]
    yields: np.ndarray
    props: dict


def cut_tables(data: Du2025Data) -> list[CutTable]:
    """The crude units' cut data, one :class:`CutTable` per unit.

    Each crude is named by its raw material, which a splitter sends to every
    crude unit. So crude ``i`` is the same oil in every table.
    """
    y, fqcut = data.param("y"), data.param("FQcut")
    out_of = {s: u for (u, s) in data.sets["OU"]}
    raw_in = {}
    for (u, s) in data.sets["IU"]:
        raw_in.setdefault(u, []).append(s)
    material = set(data.elements("S_M"))
    tables = []
    crude_order = None
    for u in data.elements("UCDU"):
        batches = sorted({m for (uu, m, s) in data.sets["IM"] if uu == u})
        if len(batches) != 1:
            raise NotImplementedError("one batch per crude unit")
        m = batches[0]
        feeds = [s for (uu, mm, s) in data.sets["IM"] if (uu, mm) == (u, m)]
        crude_of = {}
        for f in feeds:
            up = out_of.get(f)
            src = [s for s in raw_in.get(up, []) if s in material]
            crude_of[f] = src[0] if src else f
        if crude_order is None:
            crude_order = sorted(crude_of.values(), key=lambda s: int(s[1:]))
        feed_of = {c: f for f, c in crude_of.items()}
        feeds = [feed_of[c] for c in crude_order]
        cuts = sorted({k[2] for k in y if k[:2] == (u, m)})
        spg = data.elements("SPG")[0]
        sg_mean = {c: np.mean([fqcut[(m, c, f, spg)] for f in feeds]) for c in cuts}
        cuts = sorted(cuts, key=sg_mean.get)
        yl = np.array([[y.get((u, m, c, f), 0.0) for c in cuts] for f in feeds])
        qs = sorted({k[3] for k in fqcut if k[0] == m})
        props = {q: np.array([[fqcut.get((m, c, f, q), np.nan) for c in cuts]
                              for f in feeds]) for q in qs}
        props[_SPG] = props[spg]          # gravity under one key, whatever its index
        tables.append(CutTable(u, m, tuple(cuts), tuple(crude_order), tuple(feeds),
                               yl, props))
    return tables


def typical_cut_points(n_cuts: int) -> np.ndarray:
    """Typical cut points (K) for a unit with ``n_cuts`` cuts."""
    n = n_cuts - 1
    t = TYPICAL_CUTS_C
    if n > len(t):
        raise ValueError(f"no typical cut set for {n_cuts} cuts")
    k = n - 3
    return np.array(t[:k] + t[-3:]) + 273.15 if n >= 3 else np.array(t[-n:]) + 273.15


# --------------------------------------------------------------------------- #
# Joint cut-point fit                                                         #
# --------------------------------------------------------------------------- #

#: Spread allowed around each crude's linear Watson-K trend.
K_SIGMA = 0.05
#: Spread of each cut's temperature width about its typical width, in log
#: units: a swing cut keeps a finite boiling range.
WIDTH_SIGMA = 0.5


def _ndtri(x):
    return jax.scipy.special.ndtri(jnp.clip(x, 1e-9, 1 - 1e-9))


@dataclass(frozen=True)
class TBPFit:
    """Reconstructed TBP and gravity curves for every crude.

    Each crude's TBP curve passes through its cumulative yield at every cut
    point of the reference (first) crude unit, at the jointly fitted cut
    temperatures. It is interpolated linearly on a probability scale
    (``z = Phi^-1(x)``, which is close to linear for crude oils) and extended
    linearly at both ends. Its gravity curve is piecewise linear, with knots
    at the reference unit's cut mid-points; the knots are corrected so that
    every cut's mean SG, as ``characterize`` averages it, is the
    benchmark's.

    Attributes:
        crudes: Raw-material stream per crude.
        cut_points: ``{unit: (n_cut - 1,) K}`` fitted cut temperatures.
        points: Per crude, ``(x, T)`` arrays: cumulative mass fraction and
            TBP (K) at the reference unit's cut points.
        sg_points: Per crude, ``(x_mid, SG knot)`` of the gravity curve.
        K0, K1: ``(n_crude,)`` each crude's Watson-K trend ``K0 + K1 x``.
        k_residual: ``{unit: (n_crude, n_cut)}`` Watson K of each interior
            cut minus the trend (NaN at the two end cuts).
        tables: The :class:`CutTable` s fitted.
        yield_error: ``{unit: (n_crude, n_cut)}`` reconstructed minus
            benchmark cut yield.
        sg_error: ``{unit: (n_crude, n_cut)}`` reconstructed minus benchmark
            cut SG (the gravity curve's mean over the cut).
    """

    crudes: tuple[str, ...]
    cut_points: dict
    points: tuple
    sg_points: tuple
    K0: np.ndarray
    K1: np.ndarray
    k_residual: dict
    tables: tuple
    yield_error: dict
    sg_error: dict

    def tbp(self, i: int, x) -> np.ndarray:
        """TBP (K) of crude ``i`` at cumulative mass fraction ``x``."""
        xp, Tp = self.points[i]
        z, zp = np.asarray(_ndtri(jnp.asarray(x))), np.asarray(_ndtri(jnp.asarray(xp)))
        lo = Tp[0] + (z - zp[0]) * (Tp[1] - Tp[0]) / (zp[1] - zp[0])
        hi = Tp[-1] + (z - zp[-1]) * (Tp[-1] - Tp[-2]) / (zp[-1] - zp[-2])
        return np.where(z < zp[0], lo, np.where(z > zp[-1], hi, np.interp(z, zp, Tp)))

    def sg(self, i: int, x) -> np.ndarray:
        """Gravity of crude ``i`` at cumulative mass fraction ``x``."""
        xm, sg = self.sg_points[i]
        return np.interp(x, xm, sg)

    def summary(self) -> dict:
        """Largest yield, SG and Watson-K deviations per unit."""
        out = {}
        for u in self.yield_error:
            ye, se = np.abs(self.yield_error[u]), np.abs(self.sg_error[u])
            out[u] = {"max_yield_err": float(ye.max()), "max_sg_err": float(se.max()),
                      "yield_within_tol": float(np.mean(ye <= YIELD_TOL)),
                      "sg_within_tol": float(np.mean(se <= SG_TOL)),
                      "k_rms": float(np.sqrt(np.nanmean(self.k_residual[u] ** 2)))}
        return out


def _x_of_T(T, Tp, zp):
    """Cumulative fraction at ``T`` on a curve through ``(zp, Tp)`` (inverse of
    :meth:`TBPFit.tbp`), differentiable in ``T``."""
    lo = zp[0] + (T - Tp[0]) * (zp[1] - zp[0]) / (Tp[1] - Tp[0])
    hi = zp[-1] + (T - Tp[-1]) * (zp[-1] - zp[-2]) / (Tp[-1] - Tp[-2])
    z = jnp.where(T < Tp[0], lo, jnp.where(T > Tp[-1], hi, jnp.interp(T, Tp, zp)))
    return jax.scipy.stats.norm.cdf(z)


def _cut_sg(fit_sg, edges, n=400):
    """Each cut's SG from a gravity curve, as
    :func:`~difflow_refinery.assay.characterize` takes it: the mean of the
    curve over the cut's span. (Over the steep residue cut this exceeds the
    volume average by up to ~0.005; matching characterize is the point.)"""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        xx = np.linspace(lo, hi, n)
        out.append(np.mean(fit_sg(xx)))
    return np.array(out)


def fit_tbp(data: Du2025Data, *, max_steps: int = 200) -> TBPFit:
    """Reconstruct every crude's TBP and gravity curves from the cut data.

    1. **Reference unit** (the first crude unit, which has the most cuts).
       Its cut temperatures, shared by all crudes, are fitted jointly so that
       each interior cut's Watson K, computed from the cut's mid boiling
       point ``(T_lo + T_hi) / 2`` and its SG, follows its crude's linear
       trend ``K0 + K1 x`` (weight ``1 / K_SIGMA``). A prior toward
       :data:`TYPICAL_CUTS_C` (``1 / PRIOR_SIGMA_K``) sets the scale.
       Each crude's TBP curve passes through its cumulative yields at those
       temperatures, so this unit's yields are reproduced exactly.
    2. **Gravity curve.** Its knots sit at the cut mid-points. They are
       corrected until each cut's mean SG, taken as ``characterize`` takes it,
       equals the benchmark's.
    3. **Other units.** Their cut temperatures, again shared by all crudes,
       are fitted to their own cut yields on the curves from step 1. What
       remains is a measure of how consistent the data are between units,
       not a fitting error the model could remove: for the same crude, the
       two Case 1 units put the residue cut at different yields.

    Args:
        data: The benchmark case.
        max_steps: Iteration cap for each least-squares step.

    Returns:
        A :class:`TBPFit`; ``fit.summary()`` reports the agreement.
    """
    import optimistix as optx

    tables = cut_tables(data)
    ref = tables[0]
    n = len(ref.crudes)
    m = len(ref.cuts) - 1
    X = np.cumsum(ref.yields, axis=1)[:, :-1]
    xm = np.cumsum(ref.yields, axis=1) - 0.5 * ref.yields
    sgd = ref.props[_SPG]
    prior = typical_cut_points(len(ref.cuts))
    lm = optx.LevenbergMarquardt(rtol=1e-9, atol=1e-9)
    logw0 = jnp.asarray(np.log(np.diff(np.concatenate([[0.0], prior]))))

    def unpack(z):
        return jnp.cumsum(jnp.exp(z[:m])), z[m:m + n], z[m + n:]

    def res_ref(z, _):
        cp, K0, K1 = unpack(z)
        tb = 0.5 * (cp[:-1] + cp[1:])
        K = (1.8 * tb[None, :]) ** (1 / 3) / jnp.asarray(sgd[:, 1:-1])
        rk = (K - K0[:, None] - K1[:, None] * xm[:, 1:-1]) / K_SIGMA
        # widths in log space about the typical ones: no cut of zero width
        rw = (z[:m] - logw0) / WIDTH_SIGMA
        return jnp.concatenate([rk.ravel(), (cp - prior) / PRIOR_SIGMA_K, rw])

    z0 = jnp.asarray(np.concatenate([np.log(np.diff(np.concatenate([[0.0], prior]))),
                                     np.full(n, 12.0), np.zeros(n)]))
    sol = optx.least_squares(res_ref, lm, z0, max_steps=max_steps, throw=False)
    cp_ref, K0, K1 = (np.asarray(v) for v in unpack(sol.value))

    points = tuple((X[i].copy(), cp_ref.copy()) for i in range(n))
    # gravity knots: correct until the cut means match
    sg_points = []
    for i in range(n):
        edges = np.concatenate([[0.0], np.cumsum(ref.yields[i])])
        knots = sgd[i].copy()
        for _ in range(200):
            avg = _cut_sg(lambda x, k=knots: np.interp(x, xm[i], k), edges)
            err = sgd[i] - avg
            knots = knots + err
            if np.max(np.abs(err)) < 1e-6:
                break
        sg_points.append((xm[i].copy(), knots))
    sg_points = tuple(sg_points)

    cut_points = {ref.unit: cp_ref}
    zp = [np.asarray(_ndtri(jnp.asarray(p[0]))) for p in points]
    for t in tables[1:]:
        mt = len(t.cuts) - 1
        Xt = jnp.asarray(np.cumsum(t.yields, axis=1)[:, :-1])
        pr = typical_cut_points(len(t.cuts))

        def res_other(z, _, Xt=Xt, pr=pr, mt=mt):
            cp = jnp.cumsum(jnp.exp(z))
            r = [(_x_of_T(cp, jnp.asarray(points[i][1]), jnp.asarray(zp[i])) - Xt[i])
                 / YIELD_TOL for i in range(n)]
            r.append((cp - pr) / (10 * PRIOR_SIGMA_K))
            return jnp.concatenate(r)

        z0 = jnp.log(jnp.diff(jnp.concatenate([jnp.zeros(1), jnp.asarray(pr)])))
        sol = optx.least_squares(res_other, lm, z0, max_steps=max_steps, throw=False)
        cut_points[t.unit] = np.asarray(jnp.cumsum(jnp.exp(sol.value)))

    proto = TBPFit(ref.crudes, cut_points, points, sg_points, K0, K1, {}, tuple(tables), {}, {})
    kres, ye, se = {}, {}, {}
    for t in tables:
        cp = cut_points[t.unit]
        xmt = np.cumsum(t.yields, axis=1) - 0.5 * t.yields
        tb = 0.5 * (cp[:-1] + cp[1:])
        kr = np.full(t.yields.shape, np.nan)
        kr[:, 1:-1] = ((1.8 * tb[None, :]) ** (1 / 3) / t.props[_SPG][:, 1:-1]
                       - K0[:, None] - K1[:, None] * xmt[:, 1:-1])
        kres[t.unit] = kr
        yerr = np.zeros(t.yields.shape)
        serr = np.zeros(t.yields.shape)
        for i in range(n):
            xb = np.asarray(_x_of_T(jnp.asarray(cp), jnp.asarray(points[i][1]),
                                    jnp.asarray(zp[i])))
            yerr[i] = np.diff(np.concatenate([[0.0], xb, [1.0]])) - t.yields[i]
            edges = np.concatenate([[0.0], np.cumsum(t.yields[i])])
            serr[i] = _cut_sg(lambda x, i=i: proto.sg(i, x), edges) - t.props[_SPG][i]
        ye[t.unit], se[t.unit] = yerr, serr
    return TBPFit(ref.crudes, cut_points, points, sg_points, K0, K1, kres,
                  tuple(tables), ye, se)


# --------------------------------------------------------------------------- #
# Assays                                                                      #
# --------------------------------------------------------------------------- #

def _curve(table: CutTable, fit: TBPFit, i: int, q: str):
    """``(T at cut mid-point, value)`` for property ``q`` of crude ``i``."""
    v = table.props.get(q)
    if v is None:
        return None
    X = np.concatenate([[0.0], np.cumsum(table.yields[i])])
    mid = 0.5 * (X[:-1] + X[1:])
    T = fit.tbp(i, np.minimum(mid, 0.999))
    ok = np.isfinite(v[i])
    if ok.sum() < 2:
        return None
    return T[ok], v[i][ok]


def crude_assay(fit: TBPFit, i: int, *, unit: int = 0) -> Assay:
    """A difflow :class:`~difflow_refinery.assay.Assay` for fitted crude ``i``.

    Args:
        fit: From :func:`fit_tbp`.
        i: Crude index (``fit.crudes[i]`` is its raw-material stream).
        unit: Which crude unit's cut table supplies the property curves.

    Returns:
        A mass-basis ``Assay`` with a :class:`HeavyEnd`. Its TBP points are
        the crude's reconstructed points, its gravity curve is the cut SG,
        and the cut sulfur, nitrogen and CCR are measured curves. Ni+V is the
        cut-weighted bulk value spread by the default shape. Asphaltenes are
        the ILLUSTRATIVE :data:`ASPHALTENES_PER_CCR` times the bulk CCR.
    """
    t = fit.tables[unit]
    x, T = fit.points[i]
    # a point inside the lightest cut (on the curve's extension), so the
    # pseudo-components start below the first cut point and every
    # benchmark cut is its own pseudo-component
    x0 = 0.5 * x[0]
    x, T = np.concatenate([[x0], x]), np.concatenate([fit.tbp(i, [x0]), T])
    xm, sg = fit.sg_points[i]
    he = HeavyEnd(residue_sg=float(sg[-1]))      # the lump on the gravity curve
    keep = T < he.T_max - 1.0
    x, T = x[keep], T[keep]
    curves = {k: _curve(t, fit, i, q) for k, q in
              (("sulfur_curve", _S), ("nitrogen_curve", _N), ("ccr_curve", _CCR))}
    if curves["nitrogen_curve"] is not None:
        # the benchmark gives nitrogen from gas oil up; without an anchor the
        # Assay would hold it flat over the naphtha. ILLUSTRATIVE anchor.
        Tn, vn = curves["nitrogen_curve"]
        T_light = float(fit.tbp(i, [0.5 * fit.tables[unit].yields[i][0]])[0])
        if T_light < Tn[0]:
            curves["nitrogen_curve"] = (np.concatenate([[T_light], Tn]),
                                        np.concatenate([[LIGHT_NITROGEN_WPPM], vn]))
    y = t.yields[i]

    def bulk(q):
        v = t.props.get(q)
        if v is None or not np.isfinite(v[i]).any():
            return None
        return float(np.sum(np.where(np.isfinite(v[i]), v[i], 0.0) * y))

    ccr = bulk(_CCR)
    return Assay(
        tbp_percent=tuple(100.0 * x), tbp_T=tuple(T), basis="mass",
        sg_curve=(tuple(100.0 * xm), tuple(sg)), name=fit.crudes[i], heavy_end=he,
        sulfur_curve=curves["sulfur_curve"], nitrogen_curve=curves["nitrogen_curve"],
        ccr_curve=curves["ccr_curve"], nickel_vanadium_wppm=bulk(_NIV),
        asphaltenes_wt=None if ccr is None else ASPHALTENES_PER_CCR * ccr,
    )


def monotone_violations(fit: TBPFit, q: str) -> dict[str, int]:
    """Per unit: how many (crude, adjacent cut pair) steps of ``q`` fall with
    boiling point in the BENCHMARK data. Sulfur, SG and CCR should rise."""
    out = {}
    for t in fit.tables:
        v = t.props.get(q)
        if v is None:
            continue
        n = 0
        for row in v:
            r = row[np.isfinite(row)]
            n += int(np.sum(np.diff(r) < 0))
        out[t.unit] = n
    return out
