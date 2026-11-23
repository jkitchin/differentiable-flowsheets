"""The Du et al. (2025) superstructure as one differentiable JAX model.

Equation-oriented, as published: every equation of the paper's Section 3 is a
residual over one flat vector ``x`` of the GAMS model's variables. Each
equation is, after clearing denominators, a polynomial of degree at most two
in ``x`` (the model is bilinear: pooling, delta-base yields, volume blending),
so the whole constraint body is stored as COO triples -- a constant, linear
terms ``(row, j, c)`` and bilinear terms ``(row, i, j, c)`` -- and evaluated
with two segment sums. ``jax.jacobian`` of that is exact and cheap, and the
sparsity pattern is known without probing.

Scope: Case 1 (single period, no inventory, no binaries). With no inventory
``FVLI = FVLO = 0`` and Eq. 21 makes ``FVO_s = FVI_s``, so one flow ``F_s``
stands for both. The sets Case 1 leaves empty (``CDUMQ``, ``CRU``, ``RB``,
inventory bounds) raise rather than being silently ignored.

Cleared forms (the paper's equation numbers):

* (1) ``F_s = sum_m FVM[u,m,s]`` for every CDU / process-unit / mixer port.
* (2) mixer batch balance; (3) splitter and blender balance.
* (4) swing-cut CDU yields with the published merge ratios ``phi`` held fixed
  (#531 makes them variables).
* (5a) ``F_s = FQ[s,SPG] * sum(y * FVM / FQcut[.., SPG])`` -- the CDU
  product's gravity by volume; (5c) ``F_s * FQ[s,q] = sum(y * FVM * FQcut)``
  for weight-based ``q``. Qv is empty in Case 1, so (5b) does not occur.
* (8) fixed yield ``FVM[s] * sum_in gamma = gamma[s] * sum_in FVM``.
* (9) ``GM - gamma = sum (FQ - B) * delta / Del``; (10) as (8) with ``GM``.
* (11) property transfer ``FQ[s'] = alpha * FQ[s]``.
* (12) ``VM * FQ[s,SPG] = FVM`` on mixer ports; (13a) volume balance per
  batch; (13c) weight-based property balance per batch.
* (16) component-ratio limits on virtual batches (``w``, ``Qp``).
* (17) splitter property pass-through.
* (18) ``V * FQ[s,SPG] = F_s`` on the inlets of blenders with an SPG spec
  (the only ones that need a volume); (19a, c) blend specs.
* (25, 26) capacities; (27) property bounds; (28) fixed properties;
  (30, 31) supply and demand bounds; (32) profit.

Where the paper is silent and the published GAMS model is not (read from the
variables and bounds stored with its solution, every equation of which this
model satisfies there):

* ``FQ[s, SPG] <= 10``; a fixed property without an ``FQ0`` value is 0.
* A property an equation references but ``SQ`` does not trace (a mixer inlet,
  say) is a free nonnegative variable (:attr:`Du2025Model.untracked`).
* (13a) is skipped for a mixer outlet whose SPG is fixed, like (13b, c).
* (18) and the volumes ``V`` exist only for blenders with an SPG spec.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow_refinery.benchmarks.du2025.data import Du2025Data

#: Upper bound on specific gravity in the published GAMS model.
SPG_UPPER = 10.0

_T = "1"  # the single period of Case 1


class _Poly:
    """A degree-<=2 polynomial in the model variables (a builder, not traced)."""

    __slots__ = ("const", "lin", "bil")

    def __init__(self):
        self.const = 0.0
        self.lin: dict[int, float] = defaultdict(float)
        self.bil: dict[tuple[int, int], float] = defaultdict(float)

    def add(self, coef: float, *factors) -> "_Poly":
        """Add ``coef * prod(factors)``; a factor is a var index or a float."""
        idx = []
        for f in factors:
            if isinstance(f, (int, np.integer)) and not isinstance(f, bool):
                idx.append(int(f))
            else:
                coef *= float(f)
        if coef == 0.0:
            return self
        if len(idx) == 0:
            self.const += coef
        elif len(idx) == 1:
            self.lin[idx[0]] += coef
        elif len(idx) == 2:
            i, j = sorted(idx)
            self.bil[(i, j)] += coef
        else:  # pragma: no cover - the model is bilinear
            raise ValueError("term of degree > 2")
        return self


@dataclass
class _Rows:
    names: list[tuple] = field(default_factory=list)
    polys: list[_Poly] = field(default_factory=list)

    def add(self, name: tuple, p: _Poly):
        self.names.append(name)
        self.polys.append(p)


@dataclass(frozen=True)
class PolySystem:
    """``g(x) = c + A x + sum_k b_k x_i x_j`` stored as COO arrays.

    Attributes:
        n_rows: Number of rows.
        const: ``(n_rows,)`` constants.
        lin_row, lin_col, lin_val: Linear terms.
        bil_row, bil_i, bil_j, bil_val: Bilinear terms.
        names: One tuple per row, ``(equation, *indices)``.
    """

    n_rows: int
    const: np.ndarray
    lin_row: np.ndarray
    lin_col: np.ndarray
    lin_val: np.ndarray
    bil_row: np.ndarray
    bil_i: np.ndarray
    bil_j: np.ndarray
    bil_val: np.ndarray
    names: tuple

    @classmethod
    def from_rows(cls, rows: _Rows) -> "PolySystem":
        lr, lc, lv, br, bi, bj, bv = [], [], [], [], [], [], []
        const = np.zeros(len(rows.polys))
        for r, p in enumerate(rows.polys):
            const[r] = p.const
            for j, c in p.lin.items():
                if c != 0.0:
                    lr.append(r), lc.append(j), lv.append(c)
            for (i, j), c in p.bil.items():
                if c != 0.0:
                    br.append(r), bi.append(i), bj.append(j), bv.append(c)
        a = lambda v, t=float: np.asarray(v, dtype=t)  # noqa: E731
        return cls(len(rows.polys), const, a(lr, np.int32), a(lc, np.int32), a(lv),
                   a(br, np.int32), a(bi, np.int32), a(bj, np.int32), a(bv),
                   tuple(rows.names))

    def __call__(self, x: Array) -> Array:
        x = jnp.asarray(x)
        out = jnp.asarray(self.const, dtype=x.dtype)
        out = out.at[self.lin_row].add(self.lin_val * x[self.lin_col])
        out = out.at[self.bil_row].add(self.bil_val * x[self.bil_i] * x[self.bil_j])
        return out

    def jacobian_pattern(self, n_vars: int) -> np.ndarray:
        """Boolean ``(n_rows, n_vars)`` structural pattern (exact superset)."""
        pat = np.zeros((self.n_rows, n_vars), dtype=bool)
        pat[self.lin_row, self.lin_col] = True
        pat[self.bil_row, self.bil_i] = True
        pat[self.bil_row, self.bil_j] = True
        return pat


@dataclass(frozen=True)
class Du2025Model:
    """The Case 1 planning model: variables, residuals, bounds, profit.

    Build it with :func:`build_model`. All callables are pure JAX in ``x``.

    Attributes:
        data: The :class:`~difflow_refinery.benchmarks.du2025.data.Du2025Data`
            it was built from.
        var_names: One key per entry of ``x``: ``("F", s)``, ``("FQ", s, q)``,
            ``("FVM", u, m, s)``, ``("GM", u, m, s)``, ``("V", s)`` or
            ``("VM", u, m, s)``.
        lower, upper: Variable bounds (``-inf``/``inf`` where unbounded).
        eq: Equality residuals, ``eq(x) == 0``.
        ineq: Inequality residuals, ``ineq(x) <= 0``.
        profit_coef: ``(n,)`` linear objective coefficients (Eq. 32).
        fixed_props: ``{(s, q): value}`` for the properties in ``FIX``,
            which are constants here, not variables.
        untracked: ``(s, q)`` pairs outside ``SQ`` that an equation
            references (e.g. a mixer inlet whose property is not traced).
            They are free nonnegative variables, as in the published GAMS
            model -- a degree of freedom of the benchmark, not of the plant.
    """

    data: Du2025Data
    var_names: tuple
    lower: np.ndarray
    upper: np.ndarray
    eq: PolySystem
    ineq: PolySystem
    profit_coef: np.ndarray
    fixed_props: dict
    untracked: tuple = ()

    @property
    def n(self) -> int:
        """Number of variables."""
        return len(self.var_names)

    def index(self) -> dict:
        """``{var key: position in x}``."""
        return {k: i for i, k in enumerate(self.var_names)}

    def profit(self, x: Array) -> Array:
        """Eq. 32 (Case 1: no inventory terms)."""
        return jnp.dot(jnp.asarray(self.profit_coef), jnp.asarray(x))

    def x_from_solution(self, sol: dict) -> np.ndarray:
        """Pack a published solution (:func:`~.data.load_solution`) into ``x``.

        Variables the solution file omits are zero, as GAMS stores them.
        """
        fvo = sol.get("FVO", {})
        x = np.zeros(self.n)
        for i, key in enumerate(self.var_names):
            kind, rest = key[0], key[1:]
            if kind == "F":
                x[i] = fvo.get((*rest, _T), 0.0)
            else:
                x[i] = sol.get(kind, {}).get((*rest, _T), 0.0)
        return x

    def violations(self, x: Array) -> dict[str, float]:
        """Largest equality residual, inequality excess and bound excess."""
        x = np.asarray(x)
        e = np.asarray(self.eq(x))
        g = np.asarray(self.ineq(x))
        b = np.maximum(self.lower - x, x - self.upper)
        return {"eq": float(np.max(np.abs(e), initial=0.0)),
                "ineq": float(np.max(g, initial=0.0)),
                "bounds": float(np.max(b, initial=0.0))}

    def worst(self, x: Array, k: int = 10) -> list[tuple[tuple, float]]:
        """The ``k`` largest equality residuals by name (a debugging aid)."""
        e = np.abs(np.asarray(self.eq(np.asarray(x))))
        order = np.argsort(-e)[:k]
        return [(self.eq.names[i], float(e[i])) for i in order]


def _require_empty(data: Du2025Data, names):
    for n in names:
        if data.sets.get(n) or data.params.get(n):
            raise NotImplementedError(
                f"{n} is non-empty: only Case 1 (single period, no inventory, "
                "no CDU feed-property limits, no proportional blending) is "
                "implemented")


def build_model(data: Du2025Data) -> Du2025Model:
    """Assemble the Case 1 superstructure from the benchmark data.

    Args:
        data: From :func:`~difflow_refinery.benchmarks.du2025.data.load_case`.

    Returns:
        A :class:`Du2025Model`.

    Raises:
        NotImplementedError: If the case uses a feature Case 1 does not
            (inventory, several periods, CDUMQ/CRU, RB).

    Example:
        >>> model = build_model(load_case(path))   # doctest: +SKIP
        >>> model.violations(model.x_from_solution(sol))  # doctest: +SKIP
    """
    if len(data.sets.get("T", ())) != 1:
        raise NotImplementedError("only single-period cases are implemented")
    _require_empty(data, ("CDUMQ", "CRU", "RB", "LMin", "LMax", "L0",
                          "MFQMin", "MFQMax", "FQcrd", "ci_P", "ci_M"))

    S = data.elements("S")
    spg = data.elements("SPG")[0]
    QW = set(data.elements("Qw"))
    QV = set(data.elements("Qv"))
    QP = set(data.elements("Qp"))
    UCDU, UPF, UPD = (set(data.elements(k)) for k in ("UCDU", "UPF", "UPD"))
    UMIX, USPL, UBLD = (set(data.elements(k)) for k in ("UMIX", "USPL", "UBLD"))
    IU, OU = data.sets["IU"], data.sets["OU"]
    IM, OM = data.sets["IM"], data.sets["OM"]
    SQ = set(data.sets["SQ"])
    FIX = set(data.sets["FIX"])
    FQ0 = data.param("FQ0")
    P = data.param

    # ---- variables ------------------------------------------------------- #
    names: list[tuple] = []
    lo: list[float] = []
    hi: list[float] = []
    idx: dict[tuple, int] = {}

    def new(key, a=0.0, b=np.inf):
        idx[key] = len(names)
        names.append(key)
        lo.append(a)
        hi.append(b)

    for s in S:
        new(("F", s))
    fixed_props = {}
    untracked: list[tuple[str, str]] = []
    for (s, q) in data.sets["SQ"]:
        if (s, q) in FIX:
            continue
        a = P("FQMin").get((s, q), 0.0)
        b = P("FQMax").get((s, q), np.inf)
        if q == spg:
            b = min(b, SPG_UPPER)
        new(("FQ", s, q), a, b)
    for (s, q) in data.sets["FIX"]:
        fixed_props[(s, q)] = FQ0.get((s, q), 0.0)

    ins = defaultdict(list)    # (u, m) -> inlet streams of batch m
    outs = defaultdict(list)
    for (u, m, s) in IM:
        ins[(u, m)].append(s)
    for (u, m, s) in OM:
        outs[(u, m)].append(s)
    for (u, m, s) in list(IM) + list(OM):
        if ("FVM", u, m, s) not in idx:
            new(("FVM", u, m, s))
    for (u, m, s) in list(IM) + list(OM):
        if u in UPD and ("GM", u, m, s) not in idx:
            new(("GM", u, m, s))
    for (u, m, s) in list(IM) + list(OM):
        if u in UMIX and (s, spg) in SQ and ("VM", u, m, s) not in idx:
            new(("VM", u, m, s))

    def lazy(key):
        """A volume variable an equation references; free if untraced."""
        if key not in idx:
            new(key)
        return idx[key]

    F = lambda s: idx[("F", s)]  # noqa: E731
    FVM = lambda u, m, s: idx[("FVM", u, m, s)]  # noqa: E731

    def FQ(s, q):
        """Variable index, or the constant for a fixed property."""
        if (s, q) in fixed_props:
            return fixed_props[(s, q)]
        key = ("FQ", s, q)
        if key not in idx:
            # Untracked (s, q) that an equation references: a free,
            # nonnegative variable, as in the published GAMS model.
            new(key, 0.0, SPG_UPPER if q == spg else np.inf)
            untracked.append((s, q))
        return idx[key]

    eq, ineq = _Rows(), _Rows()

    # ---- (1) port aggregation -------------------------------------------- #
    batched = UCDU | UPF | UPD | UMIX
    port_batches = defaultdict(list)
    for (u, m, s) in IM:
        port_batches[("in", u, s)].append(m)
    for (u, m, s) in OM:
        port_batches[("out", u, s)].append(m)
    for tag, pairs in (("in", IU), ("out", OU)):
        for (u, s) in pairs:
            if u not in batched:
                continue
            p = _Poly().add(1.0, F(s))
            for m in port_batches[(tag, u, s)]:
                p.add(-1.0, FVM(u, m, s))
            eq.add((f"1{'a' if tag == 'in' else 'b'}", u, s), p)

    # ---- (2) mixer batch balance; (3) splitter / blender balance --------- #
    for (u, m), sin in ins.items():
        if u in UMIX:
            p = _Poly()
            for s in sin:
                p.add(1.0, FVM(u, m, s))
            for s in outs[(u, m)]:
                p.add(-1.0, FVM(u, m, s))
            eq.add(("2", u, m), p)
    unit_in, unit_out = defaultdict(list), defaultdict(list)
    for (u, s) in IU:
        unit_in[u].append(s)
    for (u, s) in OU:
        unit_out[u].append(s)
    for u in sorted(USPL | UBLD):
        p = _Poly()
        for s in unit_in[u]:
            p.add(1.0, F(s))
        for s in unit_out[u]:
            p.add(-1.0, F(s))
        eq.add(("3", u), p)

    # ---- (4), (5) CDU ---------------------------------------------------- #
    y, phi, fqcut = P("y"), P("phi"), P("FQcut")
    sc_into = defaultdict(list)   # (u, m, s) -> swing cuts s'' merged into s
    sc_from = defaultdict(list)   # (u, m, s'') -> products s that take part of s''
    for (u, m, s, s2) in data.sets["SC"]:
        sc_into[(u, m, s)].append(s2)
        sc_from[(u, m, s2)].append(s)

    def cdu_terms(u, m, s):
        """[(coef, crude stream s', cut whose FQcut applies)] for product s."""
        terms = []
        for sp in ins[(u, m)]:
            terms.append((y.get((u, m, s, sp), 0.0), sp, s))
            for s2 in sc_into[(u, m, s)]:
                terms.append((y.get((u, m, s2, sp), 0.0) * phi[(u, m, s, s2)], sp, s2))
            for s2 in sc_from[(u, m, s)]:
                terms.append((-y.get((u, m, s, sp), 0.0) * phi[(u, m, s2, s)], sp, s))
        return [t for t in terms if t[0] != 0.0]

    for (u, m), souts in outs.items():
        if u not in UCDU:
            continue
        for s in souts:
            p = _Poly().add(1.0, FVM(u, m, s))
            for c, sp, _ in cdu_terms(u, m, s):
                p.add(-c, FVM(u, m, sp))
            eq.add(("4", u, m, s), p)
    for (u, s) in OU:
        if u not in UCDU:
            continue
        ms = port_batches[("out", u, s)]
        if (s, spg) in SQ and (s, spg) not in FIX:
            p = _Poly().add(1.0, F(s))
            for m in ms:
                for c, sp, cut in cdu_terms(u, m, s):
                    p.add(-c / fqcut[(m, cut, sp, spg)], FQ(s, spg), FVM(u, m, sp))
            eq.add(("5a", u, s), p)
        for (s_, q) in SQ:
            if s_ != s or q == spg or (s, q) in FIX:
                continue
            if q in QV:
                raise NotImplementedError("volume-based properties (Eq. 5b)")
            if q not in QW:
                continue
            p = _Poly().add(1.0, F(s), FQ(s, q))
            for m in ms:
                for c, sp, cut in cdu_terms(u, m, s):
                    p.add(-c * fqcut.get((m, cut, sp, q), 0.0), FVM(u, m, sp))
            eq.add(("5c", u, s, q), p)

    # ---- (8)-(10) process units ------------------------------------------ #
    gamma = P("gamma")
    B, delta, Del = P("B"), P("delta"), P("Del")
    dbsq = defaultdict(list)
    for (u, m, s, q) in data.sets["DBSQ"]:
        dbsq[(u, m)].append((s, q))
    for (u, m), sin in ins.items():
        if u in UPF:
            gin = sum(gamma.get((u, m, s), 0.0) for s in sin)
            for s in sin + outs[(u, m)]:
                p = _Poly().add(gin, FVM(u, m, s))
                for sp in sin:
                    p.add(-gamma.get((u, m, s), 0.0), FVM(u, m, sp))
                eq.add(("8", u, m, s), p)
        elif u in UPD:
            for s in sin + outs[(u, m)]:
                g = idx[("GM", u, m, s)]
                p = _Poly().add(1.0, g).add(-gamma.get((u, m, s), 0.0))
                for (sp, q) in dbsq[(u, m)]:
                    k = delta.get((u, m, s, q), 0.0) / Del[(u, m, q)]
                    p.add(-k, FQ(sp, q)).add(k * B[(u, m, q)])
                eq.add(("9", u, m, s), p)
                p = _Poly()
                for sp in sin:
                    p.add(1.0, FVM(u, m, s), idx[("GM", u, m, sp)])
                    p.add(-1.0, g, FVM(u, m, sp))
                eq.add(("10", u, m, s), p)

    # ---- (11) property transfer ------------------------------------------ #
    alpha = P("alpha")
    for (s, s2, q) in data.sets["QT"]:
        if (s, q) in SQ and (s2, q) not in FIX:
            eq.add(("11", s, s2, q),
                   _Poly().add(1.0, FQ(s2, q)).add(-alpha[(s, s2, q)], FQ(s, q)))

    # ---- (12), (13) mixers ----------------------------------------------- #
    for (u, m), sin in ins.items():
        if u not in UMIX:
            continue
        for s in sin + outs[(u, m)]:
            if (s, spg) in SQ:
                eq.add(("12", u, m, s), _Poly().add(1.0, idx[("VM", u, m, s)], FQ(s, spg))
                       .add(-1.0, FVM(u, m, s)))
        for s in outs[(u, m)]:
            # (13a) like (13b, c): not for an outlet whose SPG is fixed
            if (s, spg) in SQ and (s, spg) not in FIX:
                p = _Poly().add(1.0, idx[("VM", u, m, s)])
                for sp in sin:
                    p.add(-1.0, lazy(("VM", u, m, sp)))
                eq.add(("13a", u, m, s), p)
            for (s_, q) in SQ:
                if s_ != s or (s, q) in FIX or q == spg:
                    continue
                if q in QV:
                    raise NotImplementedError("volume-based properties (Eq. 13b)")
                if q not in QW:
                    continue
                p = _Poly().add(1.0, FVM(u, m, s), FQ(s, q))
                for sp in sin:
                    p.add(-1.0, FVM(u, m, sp), FQ(sp, q))
                eq.add(("13c", u, m, s, q), p)

    # ---- (14)-(16) virtual batches --------------------------------------- #
    vmq = defaultdict(list)
    for (u, m, s, q) in data.sets["VMQ"]:
        vmq[(u, m, q)].append(s)
    w = P("w")
    for (u, m, q), members in vmq.items():
        if q not in QP:
            raise NotImplementedError("virtual-batch SPG/QV/QW limits (Eq. 14-15)")
        qmin = P("FQVMin").get((u, m, q))
        qmax = P("FQVMax").get((u, m, q))
        for bound, sign in ((qmin, 1.0), (qmax, -1.0)):
            if bound is None:
                continue
            # sign * (bound * sum F - sum F * w) <= 0
            p = _Poly()
            for s in unit_in[u]:
                p.add(sign * bound, F(s))
            for s in members:
                p.add(-sign * w.get((u, m, s, q), 0.0), F(s))
            ineq.add(("16", u, m, q, "min" if sign > 0 else "max"), p)

    # ---- (17) splitters --------------------------------------------------- #
    for u in sorted(USPL):
        for s in unit_in[u]:
            for s2 in unit_out[u]:
                for (s_, q) in SQ:
                    if s_ == s and (s2, q) not in FIX and (s2, q) in SQ:
                        eq.add(("17", u, s, s2, q),
                               _Poly().add(1.0, FQ(s2, q)).add(-1.0, FQ(s, q)))

    # ---- (18), (19) blenders ---------------------------------------------- #
    qbmin, qbmax = P("FQBMin"), P("FQBMax")
    spg_spec = {u for (u, q) in list(qbmin) + list(qbmax) if q == spg}
    for u in sorted(UBLD):
        for s in unit_in[u]:
            # (18) only where a volume is needed: blenders with an SPG spec
            if u in spg_spec and (s, spg) in SQ:
                eq.add(("18", u, s), _Poly().add(1.0, lazy(("V", s)), FQ(s, spg))
                       .add(-1.0, F(s)))
        for q in sorted({q for (u_, q) in list(qbmin) + list(qbmax) if u_ == u}):
            for bound, sign, tag in ((qbmin.get((u, q)), 1.0, "min"),
                                     (qbmax.get((u, q)), -1.0, "max")):
                if bound is None:
                    continue
                p = _Poly()
                if q == spg:     # (19a) bound * sum V  vs  sum F
                    for s in unit_in[u]:
                        p.add(sign * bound, lazy(("V", s))).add(-sign, F(s))
                elif q in QW:    # (19c)
                    for s in unit_in[u]:
                        p.add(sign * bound, F(s)).add(-sign, F(s), FQ(s, q))
                else:
                    raise NotImplementedError(f"blend property {q} not weight-based")
                ineq.add(("19", u, q, tag), p)

    # ---- (25), (26) capacities -------------------------------------------- #
    caps = defaultdict(list)
    for (c, s) in data.sets["CAPS"]:
        caps[c].append(s)
    for c, members in caps.items():
        for bound, sign, tag in ((P("FVCMin").get((c, _T)), -1.0, "min"),
                                 (P("FVCMax").get((c, _T)), 1.0, "max")):
            if bound is None:
                continue
            p = _Poly().add(-sign * bound)
            for s in members:
                p.add(sign, F(s))
            ineq.add(("25" if (c,) in data.sets["CAPIN"] else "26", c, tag), p)

    # ---- (30), (31) flow bounds and (32) profit --------------------------- #
    lo_a, hi_a = np.array(lo), np.array(hi)
    for (s, t), v in P("FVMin").items():
        lo_a[F(s)] = max(lo_a[F(s)], v)
    for (s, t), v in P("FVMax").items():
        hi_a[F(s)] = min(hi_a[F(s)], v)
    coef = np.zeros(len(names))
    for (s,), c in P("c_P").items():
        coef[F(s)] += c
    for (s,), c in P("c_M").items():
        coef[F(s)] -= c

    return Du2025Model(data=data, var_names=tuple(names), lower=lo_a, upper=hi_a,
                       eq=PolySystem.from_rows(eq), ineq=PolySystem.from_rows(ineq),
                       profit_coef=coef, fixed_props=fixed_props,
                       untracked=tuple(untracked))


#: Profit is O(1e7); the solver sees ``-profit * PROFIT_SCALE``.
PROFIT_SCALE = 1e-6

#: pounce options for a solve started at (or near) a solution. A small
#: initial barrier keeps the iterates from being pushed far into the interior
#: first: from the published Case 1 point the default ``mu_init`` (0.1) ends
#: 68 below it (rel 2e-6), this ends 1.0 above it.
WARM_START_OPTIONS = {"mu_init": 1e-6, "tol": 1e-8, "max_iter": 1000}


def _patterns(model: Du2025Model):
    eq, ineq = model.eq, model.ineq
    rows = np.concatenate([eq.lin_row, eq.bil_row, eq.bil_row,
                           eq.n_rows + ineq.lin_row, eq.n_rows + ineq.bil_row,
                           eq.n_rows + ineq.bil_row])
    cols = np.concatenate([eq.lin_col, eq.bil_i, eq.bil_j,
                           ineq.lin_col, ineq.bil_i, ineq.bil_j])
    jac = np.unique(np.stack([rows, cols]), axis=1)
    hi = np.concatenate([eq.bil_i, ineq.bil_i])
    hj = np.concatenate([eq.bil_j, ineq.bil_j])
    r, c = np.maximum(hi, hj), np.minimum(hi, hj)          # lower triangle
    hess = np.unique(np.stack([r, c]), axis=1)
    return ((jac[0].astype(np.int64), jac[1].astype(np.int64)),
            (hess[0].astype(np.int64), hess[1].astype(np.int64)))


def solve_local(model: Du2025Model, x0: Array, *, options: dict | None = None,
                sparse: bool = True):
    """Maximize profit from ``x0`` with pounce (a local interior-point NLP).

    The model is nonconvex (bilinear pooling and delta-base yields): this
    finds a local optimum near ``x0``, not the global one. The published
    Case 1 point is a KKT point of this model -- a solve started there stays.

    Args:
        model: From :func:`build_model`.
        x0: Starting point, ``(model.n,)``.
        options: pounce options (``tol``, ``max_iter``, ...);
            ``print_level`` defaults to 0.
        sparse: Colored (CPR) derivatives from the exact pattern of
            :class:`PolySystem`. The Case 1 Jacobian is ~0.1 % dense, so
            this is the default; ``False`` forms dense derivatives.

    Returns:
        ``(x, info)``; ``info["profit"]`` is the profit at ``x``.

    Raises:
        ImportError: If pounce is not installed (``pip install pounce-solver``).
    """
    from difflow.solvers._lazy import require

    pj = require("pounce.jax")
    jac, hess = _patterns(model)
    m_eq, m_in = model.eq.n_rows, model.ineq.n_rows
    coef = jnp.asarray(model.profit_coef) * PROFIT_SCALE

    def f(x):
        return -jnp.dot(coef, x)

    def g(x):
        return jnp.concatenate([model.eq(x), model.ineq(x)])

    problem = pj.from_jax(
        f, g, n=model.n, m=m_eq + m_in,
        lb=np.asarray(model.lower), ub=np.asarray(model.upper),
        cl=np.concatenate([np.zeros(m_eq), np.full(m_in, -np.inf)]),
        cu=np.zeros(m_eq + m_in),
        jac_pattern=jac, hess_pattern=hess, sparse=sparse,
    )
    opts = {"print_level": 0}
    opts.update(options or {})
    for k, v in opts.items():
        problem.add_option(k, v)
    x, info = problem.solve(x0=np.clip(np.asarray(x0, float), model.lower, model.upper))
    info = dict(info)
    info["profit"] = float(model.profit(x))
    return np.asarray(x), info
