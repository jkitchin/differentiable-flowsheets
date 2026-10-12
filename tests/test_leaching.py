"""Tests for the countercurrent Leacher and the washing helpers (#430).

What is and is not verified
---------------------------
No published numerical example is reproduced. The issue cites McCabe-Smith-Harriott
Ch. 23 and Geankoplis Ch. 12 examples "from recollection"; their inputs and printed answers
could not be retrieved while writing this, so nothing is claimed to agree with them. The unit
is instead checked against

- an independent NumPy assembly of the stage balances as a dense linear system (constant
  underflow, both bases, first stage separate, solute in the fresh solvent, ``r = 1``);
- the closed forms ``(1-w)/(1-w^(N+1))`` for pure washing and the one-stage hand result
  ``y = (V y_s + m0)/(V + L0)`` that holds for any underflow law;
- an independent SciPy ``fsolve`` of the variable-underflow balances (an oilseed-like
  table that is *synthetic*, not a published one);
- the constant limit of the variable path equalling the closed form, per-species mass
  balances to 1e-10, round trips of the two design modes, and ``check_grads``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from scipy.optimize import fsolve

from difflow import make_stream
from difflow.units.leaching import (
    Leacher, LeacherParams, tabulate_underflow, washing_recovery, washing_stages,
)

T, P = 300.0, 1e5


def streams(B=100.0, sugar=20.0, water=180.0, V=150.0, fresh_sugar=0.0):
    feed = make_stream({"pulp": B, "sugar": sugar, "water": water}, T, P)
    solv = make_stream({"water": V, "sugar": fresh_sugar}, T, P)
    return feed, solv


def prm(**kw):
    base = dict(inert_species="pulp", solute="sugar", solvent="water")
    base.update(kw)
    return LeacherParams(**base)


def numpy_constant(B, u, L0, m0, V, m_S, N):
    """Dense stage balances, constant solution underflow ``L = B u`` (independent of the unit)."""
    L = B * u
    V1 = V + L0 - L
    Vs = [V1] + [V] * N          # V_1 .. V_{N+1}
    Ls = [L0] + [L] * N          # L_0 .. L_N
    A = np.zeros((N, N))
    b = np.zeros(N)
    for n in range(1, N + 1):
        A[n - 1, n - 1] = Vs[n - 1] + Ls[n]       # (V_n + L_n) y_n
        if n < N:
            A[n - 1, n] = -Vs[n]                  # V_{n+1} y_{n+1}
        else:
            b[n - 1] += m_S                       # fresh solvent solute
        if n > 1:
            A[n - 1, n - 2] = -Ls[n - 1]          # L_{n-1} y_{n-1}
        else:
            b[n - 1] += m0                        # solute entering with the solids
    y = np.linalg.solve(A, b)
    return y, V1, L


class TestConstantSolution:
    @pytest.mark.parametrize("N,u,V,fresh", [
        (3, 2.0, 150.0, 0.0),    # first stage separate? L0 = 200 = L: pure washing
        (4, 1.5, 120.0, 0.0),    # L0 = 200 != L = 150: first stage separate
        (5, 1.0, 100.0, 0.0),    # r = L/V = 1
        (2, 2.5, 90.0, 3.0),     # solute in the fresh solvent, V1 = 90 + 200 - 250 < 0 is excluded below
        (6, 0.8, 70.0, 2.0),
    ])
    def test_matches_dense_linear_system(self, N, u, V, fresh):
        feed, solv = streams(V=V, fresh_sugar=fresh)
        Vtot = V + fresh                       # fresh solvent solution
        L0 = 200.0
        y, V1, L = numpy_constant(100.0, u, L0, 20.0, Vtot, fresh, N)
        if V1 <= 0:
            pytest.skip("infeasible combination (V1 <= 0)")
        under, over, info = Leacher(prm(n_stages=float(N), underflow=u))(feed, solv)
        np.testing.assert_allclose(info["stage_y"], y, rtol=1e-12)
        np.testing.assert_allclose(info["y_underflow"], y[-1], rtol=1e-12)
        np.testing.assert_allclose(info["y_overflow"], y[0], rtol=1e-12)
        assert bool(info["feasible"])

    def test_pure_washing_closed_form(self):
        # L0 = L, solute-free wash liquid: fraction left (1-w)/(1-w^(N+1)), w = V/L
        for V, N in [(150.0, 3), (200.0, 4), (400.0, 2)]:
            feed, solv = streams(V=V)
            _, _, info = Leacher(prm(n_stages=float(N), underflow=2.0))(feed, solv)
            w = V / 200.0
            left = (1 - w) / (1 - w ** (N + 1)) if w != 1 else 1 / (N + 1)
            np.testing.assert_allclose(info["solute_in_underflow"], 20.0 * left, rtol=1e-12)
            np.testing.assert_allclose(info["recovery"], washing_recovery(w, N), rtol=1e-12)

    def test_hand_one_stage(self):
        # one ideal stage: y = (V y_s + m0)/(V + L0) whatever the underflow law
        feed, solv = streams(V=150.0)
        _, _, info = Leacher(prm(n_stages=1.0, underflow=3.0))(feed, solv)
        np.testing.assert_allclose(info["y_underflow"], 20.0 / 350.0, rtol=1e-14)

    def test_washing_helpers(self):
        np.testing.assert_allclose(washing_recovery(1.0, 3.0), 0.75, rtol=1e-12)
        np.testing.assert_allclose(washing_recovery(0.5, 1.0), 0.5 / 1.5, rtol=1e-12)   # w/(1+w)
        for w in (0.9, 1.0, 1.0 + 1e-9, 2.0):
            f = 0.5 if w >= 1 else 0.4
            n = washing_stages(w, f)
            np.testing.assert_allclose(washing_recovery(w, n), f, rtol=1e-9)
        assert not np.isfinite(float(washing_stages(0.5, 0.6)))   # f > w unreachable

    def test_lambda_one_is_smooth(self):
        feed, solv = streams(V=200.0)
        vals = []
        for eps in (0.0, 1e-9, 1e-5, -1e-5):
            _, _, i = Leacher(prm(n_stages=4.0, underflow=2.0 * (1 + eps)))(feed, solv)
            vals.append(float(i["recovery"]))
        np.testing.assert_allclose(vals, 4.0 / 5.0, atol=1e-4)
        np.testing.assert_allclose(vals[0], 0.8, rtol=1e-12)

    def test_continuous_n_interpolates(self):
        feed, solv = streams(V=150.0)
        r = [float(Leacher(prm(n_stages=n, underflow=2.0))(feed, solv)[2]["recovery"]) for n in (2.0, 2.5, 3.0)]
        assert r[0] < r[1] < r[2]

    def test_efficiency_scales_the_stage_count(self):
        feed, solv = streams(V=150.0)
        a = Leacher(prm(n_stages=6.0, underflow=2.0, stage_efficiency=0.5))(feed, solv)[2]
        b = Leacher(prm(n_stages=3.0, underflow=2.0))(feed, solv)[2]
        np.testing.assert_allclose(a["recovery"], b["recovery"], rtol=1e-14)
        np.testing.assert_allclose(a["N_ideal"], 3.0)

    def test_infeasible_first_overflow_is_flagged(self):
        feed, solv = streams(V=50.0)            # V1 = 50 + 200 - 250 = 0 -> not positive
        _, _, info = Leacher(prm(n_stages=3.0, underflow=2.5))(feed, solv)
        assert not bool(info["feasible"])

    def test_washing_efficiency(self):
        feed, solv = streams(V=150.0)
        _, _, info = Leacher(prm(n_stages=3.0, underflow=2.0))(feed, solv)
        # pure solvent, L0 = L: efficiency = recovery here
        np.testing.assert_allclose(info["washing_efficiency"], info["recovery"], rtol=1e-12)
        assert float(info["solution_to_inert_out"]) == pytest.approx(2.0)


class TestConstantSolvent:
    def test_matches_dense_system_on_ratio_basis(self):
        # solvent basis: carrier = water, composition = sugar/water ratio
        B, u, N = 100.0, 1.5, 3
        feed, solv = streams(V=150.0, water=180.0)
        L, L0, V = B * u, 180.0, 150.0
        V1 = V + L0 - L
        Vs, Ls = [V1] + [V] * N, [L0] + [L] * N
        A = np.zeros((N, N)); b = np.zeros(N)
        for n in range(1, N + 1):
            A[n - 1, n - 1] = Vs[n - 1] + Ls[n]
            if n < N: A[n - 1, n] = -Vs[n]
            if n > 1: A[n - 1, n - 2] = -Ls[n - 1]
            else: b[0] += 20.0
        Y = np.linalg.solve(A, b)
        under, over, info = Leacher(prm(n_stages=3.0, underflow_model="constant_solvent", underflow=u))(feed, solv)
        np.testing.assert_allclose(float(under["F_sugar"]) / float(under["F_water"]), Y[-1], rtol=1e-12)
        np.testing.assert_allclose(float(over["F_sugar"]) / float(over["F_water"]), Y[0], rtol=1e-12)
        np.testing.assert_allclose(float(under["F_water"]), L, rtol=1e-14)
        np.testing.assert_allclose(float(over["F_water"]), V1, rtol=1e-14)


class TestMassBalance:
    @pytest.mark.slow
    @pytest.mark.parametrize("kw", [
        dict(underflow_model="constant_solution", underflow=1.6, n_stages=3.7),
        dict(underflow_model="constant_solvent", underflow=1.4, n_stages=2.5),
        dict(underflow_model="variable", n_stages=4, underflow_y=[0.0, 0.1, 0.2, 0.4],
             underflow_s=[1.5, 1.8, 2.2, 3.0]),
    ])
    def test_species_balance_to_1e_10(self, kw):
        feed, solv = streams(V=160.0, fresh_sugar=2.0)
        under, over, info = Leacher(prm(**kw))(feed, solv)
        for sp in ("pulp", "sugar", "water"):
            fin = float(feed.get(f"F_{sp}", 0.0)) + float(solv.get(f"F_{sp}", 0.0))
            fout = float(under.get(f"F_{sp}", 0.0)) + float(over.get(f"F_{sp}", 0.0))
            assert abs(fin - fout) < 1e-10, (sp, fin, fout)
        assert bool(info["converged"]) and bool(info["feasible"])


@pytest.mark.slow
class TestVariableUnderflow:
    Y = [0.0, 0.1, 0.2, 0.4]
    S = [1.5, 1.8, 2.2, 3.0]

    @pytest.mark.slow
    def test_constant_table_equals_closed_form(self):
        feed, solv = streams(V=130.0, fresh_sugar=1.0)
        a = Leacher(prm(n_stages=4.0, underflow=1.7))(feed, solv)[2]
        b = Leacher(prm(n_stages=4, underflow_model="variable", underflow_y=[0.0, 1.0],
                        underflow_s=[1.7, 1.7]))(feed, solv)[2]
        np.testing.assert_allclose(b["recovery"], a["recovery"], rtol=1e-11)
        np.testing.assert_allclose(b["stage_y"], a["stage_y"], rtol=1e-10)

    def test_matches_scipy_fsolve(self):
        from difflow.staged import _pchip_eval, _pchip_slopes
        ys, ss = jnp.asarray(self.Y), jnp.asarray(self.S)
        d = _pchip_slopes(ys, ss)
        s = lambda y: float(_pchip_eval(ys, ss, d, jnp.asarray(y)))   # noqa: E731
        B, L0, m0, V, N = 100.0, 200.0, 20.0, 150.0, 4

        def F(y):
            L = np.array([B * s(v) for v in y])
            V1 = V + L0 - L[-1]
            Vn1 = V1 - L0 + L
            yn = np.concatenate([y[1:], [0.0]])
            return Vn1 * yn - L * y - (V1 * y[0] - m0)

        y_ref = fsolve(F, np.full(N, 0.05), xtol=1e-14)
        assert np.max(np.abs(F(y_ref))) < 1e-9
        feed, solv = streams(V=150.0)
        _, _, info = Leacher(prm(n_stages=4, underflow_model="variable",
                                 underflow_y=self.Y, underflow_s=self.S))(feed, solv)
        np.testing.assert_allclose(info["stage_y"], y_ref, rtol=1e-9)
        # the variable underflow differs from the constant one at the mean ratio (it matters)
        c = Leacher(prm(n_stages=4.0, underflow=2.0))(feed, solv)[2]
        assert abs(float(c["recovery"]) - float(info["recovery"])) > 1e-3

    def test_one_stage_hand_result_for_any_law(self):
        feed, solv = streams(V=150.0)
        _, _, info = Leacher(prm(n_stages=1, underflow_model="variable",
                                 underflow_y=self.Y, underflow_s=self.S))(feed, solv)
        np.testing.assert_allclose(info["y_underflow"], 20.0 / 350.0, rtol=1e-12)

    def test_noninteger_stage_count_is_rejected(self):
        feed, solv = streams()
        with pytest.raises(ValueError, match="integer"):
            Leacher(prm(n_stages=3.5, underflow_model="variable", underflow_y=self.Y,
                        underflow_s=self.S))(feed, solv)

    def test_tabulate_underflow(self):
        ys, ss = tabulate_underflow(lambda y: 1.5 + 3.0 * y, 0.4, 5)
        np.testing.assert_allclose(ss, [1.5 + 3.0 * y for y in ys])

    def test_jit(self):
        feed, solv = streams()
        L = Leacher(prm(n_stages=3, underflow_model="variable", underflow_y=self.Y, underflow_s=self.S))
        f = jax.jit(lambda v: L(feed, make_stream({"water": v}, T, P))[2]["recovery"])
        np.testing.assert_allclose(f(150.0), L(feed, solv)[2]["recovery"], rtol=1e-12)


class TestDesign:
    def test_stages_round_trip(self):
        feed, solv = streams(V=150.0)
        ps = prm(underflow=1.5, recovery=0.8, design="stages")
        _, _, i = Leacher(ps)(feed, solv)
        assert bool(i["feasible"])
        _, _, j = Leacher(prm(n_stages=float(i["n_stages"]), underflow=1.5))(feed, solv)
        np.testing.assert_allclose(j["recovery"], 0.8, rtol=1e-10)

    def test_stages_r_equal_one(self):
        feed, solv = streams(V=200.0)
        _, _, i = Leacher(prm(underflow=2.0, recovery=0.8, design="stages"))(feed, solv)
        np.testing.assert_allclose(i["n_stages"], 4.0, rtol=1e-9)   # N/(N+1) = 0.8

    def test_unreachable_recovery_is_nan_and_flagged(self):
        feed, solv = streams(V=150.0)     # w = 0.75: infinite stages recover only 0.75
        _, _, i = Leacher(prm(underflow=2.0, recovery=0.9, design="stages"))(feed, solv)
        assert not bool(i["feasible"]) and np.isnan(float(i["n_stages"]))

    @pytest.mark.slow
    @pytest.mark.parametrize("kw", [
        dict(underflow=2.0),
        dict(underflow_model="variable", underflow_y=[0.0, 0.1, 0.2, 0.4], underflow_s=[1.5, 1.8, 2.2, 3.0]),
    ])
    def test_solvent_design(self, kw):
        feed, solv = streams(V=150.0)
        _, _, i = Leacher(prm(n_stages=3, recovery=0.9, design="solvent", **kw))(feed, solv)
        assert bool(i["feasible"])
        np.testing.assert_allclose(i["recovery"], 0.9, rtol=1e-9)
        _, _, j = Leacher(prm(n_stages=3, **kw))(feed, make_stream({"water": float(i["solvent_rate"])}, T, P))
        np.testing.assert_allclose(j["recovery"], 0.9, rtol=1e-9)

    def test_validation(self):
        with pytest.raises(ValueError):
            prm(design="stages")                          # needs a recovery
        with pytest.raises(ValueError):
            prm(underflow_model="variable")               # needs tables
        with pytest.raises(ValueError):
            LeacherParams("a", "a", "b")
        with pytest.raises(ValueError):
            prm(underflow_model="nope")


class TestGradients:
    feed, solv = streams(V=150.0)

    def rec(self, **kw):
        return Leacher(prm(**kw))(self.feed, self.solv)[2]["recovery"]

    def test_solvent_rate(self):
        f = lambda v: Leacher(prm(n_stages=3.0, underflow=1.5))(  # noqa: E731
            self.feed, make_stream({"water": v}, T, P))[2]["recovery"]
        check_grads(f, (150.0,), order=1, modes=["rev"])
        assert float(jax.grad(f)(150.0)) > 0.0

    def test_n_stages(self):
        f = lambda n: self.rec(n_stages=n, underflow=1.5)   # noqa: E731
        check_grads(f, (3.0,), order=1, modes=["rev"])
        assert float(jax.grad(f)(3.0)) > 0.0

    def test_underflow_ratio(self):
        f = lambda u: self.rec(n_stages=3.0, underflow=u)   # noqa: E731
        check_grads(f, (1.5,), order=1, modes=["rev"])
        assert float(jax.grad(f)(1.5)) < 0.0

    def test_at_lambda_one(self):
        feed, solv = streams(V=150.0)
        g = lambda n: Leacher(prm(n_stages=n, underflow=1.5))(feed, solv)[2]["recovery"]
        check_grads(g, (3.0,), order=1, modes=["rev"])

    @pytest.mark.slow
    def test_variable_wrt_solvent_and_table(self):
        Y = jnp.array([0.0, 0.1, 0.2, 0.4])

        def f(v, S):
            p = prm(n_stages=3, underflow_model="variable", underflow_y=Y, underflow_s=S)
            return Leacher(p)(self.feed, make_stream({"water": v}, T, P))[2]["recovery"]
        check_grads(f, (150.0, jnp.array([1.5, 1.8, 2.2, 3.0])), order=1, modes=["rev"], rtol=1e-4, atol=1e-6)

    @pytest.mark.slow
    def test_design_gradients(self):
        f = lambda r: Leacher(prm(underflow=1.5, recovery=r, design="stages"))(  # noqa: E731
            self.feed, self.solv)[2]["n_stages"]
        check_grads(f, (0.7,), order=1, modes=["rev"])
        g = lambda r: Leacher(prm(n_stages=3, underflow=1.5, recovery=r, design="solvent"))(  # noqa: E731
            self.feed, self.solv)[2]["solvent_rate"]
        check_grads(g, (0.8,), order=1, modes=["rev"], rtol=1e-4)
