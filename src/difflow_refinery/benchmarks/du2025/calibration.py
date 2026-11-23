"""Calibrating difflow units to the benchmark's base yields (#533).

A benchmark process unit is ``gamma`` (base yield coefficients per inlet and
outlet stream), plus, for a delta-base unit, ``B`` / ``delta`` / ``Del`` (how
the coefficients move with feed properties). This module connects a difflow
unit to that representation in three steps:

1. :func:`unit_base` reads a unit's base point: its inlets and outlets,
   ``gamma``, the yields ``gamma_out / sum(gamma_in)`` on total feed, and the
   base feed properties (``B`` for a delta-base unit, the published
   solution's inlet properties otherwise).
2. :func:`calibrate` fits a unit's tunable parameters to those yields by
   bounded Levenberg-Marquardt with AD Jacobians. The ``predict(theta)`` callable
   maps parameters to the benchmark's outlet yields; it is the unit plus its
   lumping map.
3. :func:`compare_delta` compares the calibrated unit's AD sensitivities to
   the published ``delta / Del``, reporting sign agreement and magnitude
   ratio. Disagreements are reported, never tuned away. Then
   :func:`substitute` writes the unit back into the benchmark *as its own
   delta-base linearization* at the base point (``gamma`` from the unit, and
   ``delta / Del`` from its Jacobian), so the re-solved benchmark stays
   bilinear and :func:`~.model.solve_local` still applies.

A unit enters as a pure function, so a rigorous difflow model, a reduced
model calibrated to it, or the fixed-yield block itself can all be
substituted the same way.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from difflow_refinery.benchmarks.du2025.data import Du2025Data

_T = "1"


@dataclass(frozen=True)
class UnitBase:
    """One benchmark process unit (one batch) at its base point.

    Attributes:
        unit, batch: e.g. ``"Upd0"``, ``"m22"``.
        inlets, outlets: Stream names.
        gamma_in, gamma_out: Base yield coefficients.
        yields: ``gamma_out / sum(gamma_in)`` -- outlet mass per unit total
            inlet mass (hydrogen co-feed included).
        props: ``{q: base value}`` -- ``B`` for a delta-base unit; for a
            fixed-yield unit, the main inlet's properties in the published
            solution (when given) else ``{}``.
        delta: ``{q: (n_out,) published delta / Del}`` (delta-base only).
    """

    unit: str
    batch: str
    inlets: tuple[str, ...]
    outlets: tuple[str, ...]
    gamma_in: np.ndarray
    gamma_out: np.ndarray
    yields: np.ndarray
    props: dict
    delta: dict


def unit_base(data: Du2025Data, unit: str, batch: str | None = None,
              solution: dict | None = None) -> UnitBase:
    """The base point of a benchmark process unit (see :class:`UnitBase`).

    Args:
        data: The benchmark case.
        unit: A ``UPF`` or ``UPD`` unit.
        batch: Its batch; default the first.
        solution: A published solution, for a fixed-yield unit's feed
            properties.
    """
    ms = sorted({m for (u, m, s) in data.sets["IM"] if u == unit}, key=lambda m: int(m[1:]))
    if not ms:
        raise ValueError(f"{unit!r} has no batches")
    m = ms[0] if batch is None else batch
    ins = tuple(s for (u, mm, s) in data.sets["IM"] if (u, mm) == (unit, m))
    outs = tuple(s for (u, mm, s) in data.sets["OM"] if (u, mm) == (unit, m))
    g = data.param("gamma")
    gi = np.array([g.get((unit, m, s), 0.0) for s in ins])
    go = np.array([g.get((unit, m, s), 0.0) for s in outs])
    props, delta = {}, {}
    if unit in data.elements("UPD"):
        B, d, D = data.param("B"), data.param("delta"), data.param("Del")
        for (u, mm, s, q) in data.sets["DBSQ"]:
            if (u, mm) == (unit, m):
                props[q] = B[(unit, m, q)]
                delta[q] = np.array([d.get((unit, m, o, q), 0.0) for o in outs]) / D[(unit, m, q)]
    elif solution is not None:
        main = ins[int(np.argmax(gi))]
        for (s, q, t), v in solution.get("FQ", {}).items():
            if s == main and (s, q) in set(data.sets["SQ"]):
                props[q] = v
    return UnitBase(unit, m, ins, outs, gi, go, go / gi.sum(), props, delta)


@dataclass(frozen=True)
class Calibration:
    """Result of :func:`calibrate`.

    Attributes:
        theta: Calibrated parameters.
        predicted: Predicted yields at ``theta``.
        target: The benchmark yields.
        max_abs_error: Largest ``|predicted - target|`` (mass fraction).
        within_tol: Whether every yield is within ``tol``.
        jacobian: ``d predicted / d theta`` at ``theta``.
    """

    theta: np.ndarray
    predicted: np.ndarray
    target: np.ndarray
    max_abs_error: float
    within_tol: bool
    jacobian: np.ndarray


def calibrate(predict: Callable, theta0: Sequence[float], target: Sequence[float], *,
              lower: Sequence[float], upper: Sequence[float],
              weights: Sequence[float] | None = None, tol: float = 0.01,
              max_steps: int = 40, jacobian: Callable | None = None,
              n_yields: int | None = None) -> Calibration:
    """Fit ``predict(theta)`` to ``target`` inside ``[lower, upper]``.

    Levenberg-Marquardt on ``theta = lower + (upper - lower) * sigmoid(z)``,
    so the documented physical bounds of each knob are never left. AD gives
    the Jacobian, by ``jax.jacfwd`` unless ``jacobian`` is given (pass
    ``jax.jacrev`` for a reverse-only unit). It is a plain Python loop, so a
    trial point where the unit fails (non-finite output, or a solver that
    raises) is rejected and the damping raised, instead of breaking the
    solve.

    Args:
        predict: ``theta -> (n,) yields`` (the unit and its lumping map).
        theta0: Start, strictly inside the bounds.
        target: ``(n,)`` benchmark yields.
        lower, upper: Bounds per parameter.
        weights: Per-yield weights (default 1).
        tol: Acceptance tolerance on every yield (#533: 1 wt% = 0.01).
        max_steps: Accepted-step cap.
        jacobian: ``f -> (theta -> J)``; default ``jax.jacfwd``.
        n_yields: Only the first ``n_yields`` outputs are yields judged
            against ``tol``; later ones are soft steering targets (e.g. an
            operating temperature). Default: all.

    Returns:
        A :class:`Calibration`.
    """
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    t0 = np.asarray(theta0, float)
    if np.any(t0 <= lo) or np.any(t0 >= hi):
        raise ValueError("theta0 must lie strictly inside [lower, upper]")
    target = np.asarray(target, float)
    w = np.ones_like(target) if weights is None else np.asarray(weights, float)
    jac = jax.jacfwd if jacobian is None else jacobian
    to_theta = lambda z: lo + (hi - lo) / (1.0 + jnp.exp(-z))  # noqa: E731

    def res(z):
        return w * (predict(to_theta(z)) - target)

    jres = jac(res)

    def safe(f, z):
        """``f(z)``, or NaN where the unit raises (a failed trial point)."""
        try:
            return np.asarray(f(z))
        except Exception:  # noqa: BLE001 - any unit failure rejects the step
            return np.full(target.shape if f is res else (target.size, z.size), np.nan)
    z = np.log((t0 - lo) / (hi - t0))
    r = np.asarray(res(z))
    if not np.all(np.isfinite(r)):
        raise ValueError("predict is not finite at theta0")
    lam = 1e-2
    for _ in range(max_steps):
        J = safe(jres, z)
        if not np.all(np.isfinite(J)):
            break
        g, H = J.T @ r, J.T @ J
        accepted = False
        for _ in range(12):
            step = -np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), g)
            r_new = safe(res, z + step)
            if np.all(np.isfinite(r_new)) and r_new @ r_new < r @ r:
                z, r, lam, accepted = z + step, r_new, max(lam / 3.0, 1e-9), True
                break
            lam *= 4.0
        if not accepted or np.max(np.abs(step)) < 1e-9:
            break
    theta = np.asarray(to_theta(z))
    pred = np.asarray(predict(theta))
    J = np.asarray(jac(predict)(jnp.asarray(theta)))
    k = target.size if n_yields is None else n_yields
    err = float(np.max(np.abs(pred[:k] - target[:k])))
    return Calibration(theta, pred, target, err, err <= tol, J)


@dataclass(frozen=True)
class DeltaComparison:
    """AD sensitivities of a calibrated unit vs the published delta-base.

    Attributes:
        props: Property order.
        ad: ``(n_out, n_q)`` AD ``d yield / d property`` at the base point.
        published: ``(n_out, n_q)`` published ``delta / Del``, as yields on
            total feed (see :func:`compare_delta`).
        sign_agree: ``(n_out, n_q)`` bool, where both are nonzero.
        ratio: ``(n_out, n_q)`` ``ad / published`` (NaN where published = 0).
    """

    props: tuple[str, ...]
    ad: np.ndarray
    published: np.ndarray
    sign_agree: np.ndarray
    ratio: np.ndarray

    def summary(self) -> dict:
        """Share of nonzero published entries whose sign AD reproduces, and
        the median ``|ratio|`` over them."""
        nz = self.published != 0
        return {"n": int(nz.sum()),
                "sign_agreement": float(np.mean(self.sign_agree[nz])) if nz.any() else float("nan"),
                "median_abs_ratio": float(np.nanmedian(np.abs(self.ratio[nz]))) if nz.any()
                else float("nan")}


def compare_delta(base: UnitBase, yields_of_props: Callable) -> DeltaComparison:
    """AD Jacobian of a calibrated unit's yields vs the published delta.

    The published delta moves the *coefficients* ``Gamma`` (Eq. 9), which are
    then renormalized by the inlet coefficients (Eq. 10). So the published
    yield sensitivity on total feed is
    ``d(G_out / sum G_in) = (delta_out - y_out * sum delta_in) / sum gamma_in``.
    Only the outlets' delta is stored in ``base.delta``; the inlet delta of
    the benchmark's delta-base units is zero for the main feed, and this
    function takes it as zero.

    Args:
        base: From :func:`unit_base` for a delta-base unit.
        yields_of_props: ``(n_q,) props -> (n_out,) yields`` on total feed,
            props in ``tuple(base.props)`` order.
    """
    qs = tuple(base.props)
    x0 = jnp.asarray([base.props[q] for q in qs], float)
    ad = np.asarray(jax.jacfwd(yields_of_props)(x0))
    pub = np.stack([base.delta[q] for q in qs], axis=1) / base.gamma_in.sum()
    nz = (pub != 0) & (ad != 0)
    sign = np.where(nz, np.sign(ad) == np.sign(pub), False)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(pub != 0, ad / pub, np.nan)
    return DeltaComparison(qs, ad, pub, sign, ratio)


def substitute(data: Du2025Data, base: UnitBase, yields: Sequence[float],
               dyields: Mapping[str, Sequence[float]] | None = None,
               props: Mapping[str, float] | None = None) -> Du2025Data:
    """A copy of the benchmark with one unit batch replaced by a linearization.

    The unit batch becomes delta-base (Eqs. 9-10) with ``gamma_in`` kept,
    ``gamma_out = yields * sum(gamma_in)``, base properties ``props`` and
    ``delta / Del = dyields * sum(gamma_in)``, with ``Del = 1``. A
    fixed-yield unit is moved from ``UPF`` to ``UPD`` (all its batches are
    re-expressed; the other batches keep their gamma with no delta).

    Args:
        data: The benchmark.
        base: From :func:`unit_base`.
        yields: ``(n_out,)`` the substituted unit's yields on total feed at
            the base point.
        dyields: ``{q: (n_out,) d yield / d FQ[main inlet, q]}``; ``None``
            or empty keeps the unit fixed-yield (only gamma changes).
        props: Base properties for ``dyields`` (default ``base.props``).

    Returns:
        A new :class:`Du2025Data`; ``data`` is not modified.
    """
    d = copy.deepcopy(data)
    u, m = base.unit, base.batch
    sum_in = base.gamma_in.sum()
    for o, y in zip(base.outlets, yields):
        d.params.setdefault("gamma", {})[(u, m, o)] = float(y) * sum_in
    if not dyields:
        return d
    props = dict(base.props if props is None else props)
    main = base.inlets[int(np.argmax(base.gamma_in))]
    if u in d.elements("UPF"):
        d.sets["UPF"] = tuple(k for k in d.sets["UPF"] if k[0] != u)
        d.sets["UPD"] = tuple(d.sets["UPD"]) + ((u,),)
    for key in ("B", "Del", "delta"):
        d.params.setdefault(key, {})
        for k in [k for k in d.params[key] if k[:2] == (u, m)]:
            del d.params[key][k]
    d.sets["DBSQ"] = tuple(k for k in d.sets["DBSQ"] if k[:2] != (u, m))
    for q, dy in dyields.items():
        d.params["B"][(u, m, q)] = float(props[q])
        d.params["Del"][(u, m, q)] = 1.0
        for o, v in zip(base.outlets, dy):
            d.params["delta"][(u, m, o, q)] = float(v) * sum_in
        d.sets["DBSQ"] = tuple(d.sets["DBSQ"]) + ((u, m, main, q),)
        # the delta-base equation reads FQ[main, q]: it must be traced
        if (main, q) not in set(d.sets["SQ"]):
            d.sets["SQ"] = tuple(d.sets["SQ"]) + ((main, q),)
    return d


def objective_change(data: Du2025Data, new: Du2025Data, x0, *, options=None) -> dict:
    """Re-solve the benchmark with a unit substituted; compare with the original.

    Args:
        data, new: The original and substituted benchmarks.
        x0: Start for both solves (a published solution packed by the
            original model's ``x_from_solution``; mapped by variable name).
        options: pounce options (default ``WARM_START_OPTIONS``).

    Returns:
        ``{"profit_before", "profit_after", "status_after", "changed":
        [(inequality name, before_active, after_active)]}``. A limit is
        active when its slack is below 1e-6 (relative).
    """
    from difflow_refinery.benchmarks.du2025.model import WARM_START_OPTIONS, build_model, solve_local

    opts = dict(WARM_START_OPTIONS if options is None else options)
    opts.setdefault("max_iter", 3000)
    m0, m1 = build_model(data), build_model(new)
    x0 = np.asarray(x0)
    xa, ia = solve_local(m0, x0, options=opts)
    idx0 = m0.index()
    gamma = new.param("gamma")

    def start(k):
        if k in idx0:
            return x0[idx0[k]]
        if k[0] == "GM":            # a unit made delta-base: start at its gamma
            return gamma.get(k[1:], 0.0)
        return 0.0

    x1 = np.array([start(k) for k in m1.var_names])
    xb, ib = solve_local(m1, x1, options=opts)

    def active(m, x):
        g = np.asarray(m.ineq(x))
        return {n: bool(v > -1e-6) for n, v in zip(m.ineq.names, g)}

    a, b = active(m0, xa), active(m1, xb)
    changed = [(n, a[n], b[n]) for n in a if n in b and a[n] != b[n]]
    return {"profit_before": ia["profit"], "profit_after": ib["profit"],
            "status_before": ia.get("status_msg"), "status_after": ib.get("status_msg"),
            "violations_after": m1.violations(xb), "changed": changed}
