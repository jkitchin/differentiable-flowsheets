"""Tests for batch distillation (#426).

What is and is not verified
---------------------------
Verified against independent oracles:

* ``rayleigh`` at constant alpha against the closed form
  ``ln(F/W) = [ln(x_F/x_W) + alpha ln((1-x_W)/(1-x_F))]/(alpha-1)`` written out
  here, and against ``scipy.integrate.quad`` of ``1/(y*-x)``; the quadrature path
  (tabulated curve) against the closed form.
* ``BatchDistillation`` against a separate NumPy/SciPy implementation: the
  batch ODEs ``dW/dt = -D``, ``d(W x_W)/dt = -D x_D`` integrated in *time* with
  ``solve_ivp``, ``x_D`` from a NumPy tray-to-tray march inverted with
  ``brentq`` (the library integrates in x_W with quadrature and never
  integrates in time), for constant reflux with and without Murphree efficiency;
  constant composition against the exact ``W/F = (x_D-x_F)/(x_D-x_W)`` and a
  NumPy quadrature for the time.
* Mass balances, ``N = 0`` reducing to Rayleigh, monotone reflux, the
  azeotrope flags, three-cut conservation, gradients by ``check_grads``.
* ``rayleigh_multicomponent`` against ``solve_ivp`` with a NumPy bubble point.

Published example: ``rayleigh`` on a tabulated curve reproduces Geankoplis
(5e) Example 11.3-2 (n-pentane/n-heptane, 101.3 kPa, 100 mol at x_F = 0.5,
40 mol distilled): x_W = 0.277 and y_av = 0.835 read graphically in the book,
0.279 and 0.832 here (``TestGeankoplis``). No published batch *rectifier*
example (McCabe-Smith-Harriott Ch. 21, Wankat Ch. 9) is reproduced. The ethanol-water NRTL
and Antoine constants are the ones used in ``tests/test_phase_diagrams.py``
(their provenance is documented there).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from scipy import integrate, optimize

from difflow.staged import constant_alpha, tabulated
from difflow.thermo import IdealThermo, SpeciesData
from difflow.units.batch_distillation import (
    BatchDistillation, BatchDistillationParams, batch_cuts, rayleigh,
    rayleigh_multicomponent,
)
from difflow.units.lle import NRTLParams

jax.config.update("jax_enable_x64", True)

ATM = 101325.0


def closed_form(alpha, xF, xW):
    return (np.log(xF / xW) + alpha * np.log((1 - xW) / (1 - xF))) / (alpha - 1)


def ystar(alpha, x):
    return alpha * x / (1 + (alpha - 1) * x)


def charge(a=50.0, b=50.0):
    return {"F_a": a, "F_b": b, "T": 350.0, "P": 1e5}


def np_march(alpha, xD, L, N, E=1.0):
    """NumPy tray-to-tray march: still composition for distillate xD."""
    y = xD
    for _ in range(N):
        if E == 1.0:
            x = y / (alpha - (alpha - 1) * y)
        else:
            f = lambda x_: (1 - E) * (L * x_ + (1 - L) * xD) + E * ystar(alpha, x_) - y
            x = optimize.brentq(f, 0.0, 1.0, xtol=1e-14)
        y = L * x + (1 - L) * xD
    return y / (alpha - (alpha - 1) * y)


def np_xD(alpha, xW, L, N, E=1.0):
    return optimize.brentq(lambda xd: np_march(alpha, xd, L, N, E) - xW, xW, 1 - 1e-12, xtol=1e-14)


def np_batch(alpha, F, xF, R, V, N, t_end, E=1.0):
    """Time integration of the batch balances (independent of the library)."""
    L = R / (R + 1.0)
    D = V / (R + 1.0)

    def rhs(t, y):
        W, M, Dl = y
        xW = M / W
        xd = np_xD(alpha, xW, L, N, E)
        return [-D, -D * xd, D * xd]

    sol = integrate.solve_ivp(rhs, [0, t_end], [F, F * xF, 0.0], rtol=1e-11, atol=1e-12, method="DOP853")
    W, M, Dl = sol.y[:, -1]
    return W, M / W, Dl


# ---------------------------------------------------------------------------
# rayleigh()
# ---------------------------------------------------------------------------


class TestRayleigh:
    @pytest.mark.parametrize("alpha,xF,xW", [(2.5, 0.5, 0.2), (1.5, 0.9, 0.3), (4.0, 0.3, 0.05), (1.2, 0.6, 0.5),
                                              (2.0, 0.99, 0.01)])
    def test_closed_form(self, alpha, xF, xW):
        r = rayleigh(constant_alpha(alpha), xF, x_W=xW)
        assert float(r["ln_F_over_W"]) == pytest.approx(closed_form(alpha, xF, xW), abs=1e-8)
        # and the defining integral, by an independent adaptive quadrature
        q, _ = integrate.quad(lambda x: 1.0 / (ystar(alpha, x) - x), xW, xF, epsabs=1e-13, epsrel=1e-13)
        assert float(r["ln_F_over_W"]) == pytest.approx(q, abs=1e-8)
        assert bool(r["ok"])
        w = np.exp(-closed_form(alpha, xF, xW))
        assert float(r["W_over_F"]) == pytest.approx(w, abs=1e-8)
        assert float(r["x_D_avg"]) == pytest.approx((xF - w * xW) / (1 - w), abs=1e-8)

    @pytest.mark.parametrize("alpha,xF,xW", [(2.5, 0.5, 0.2), (1.5, 0.9, 0.3), (4.0, 0.3, 0.05)])
    def test_solve_for_xW(self, alpha, xF, xW):
        w = float(np.exp(-closed_form(alpha, xF, xW)))
        r = rayleigh(constant_alpha(alpha), xF, W_over_F=w)
        assert float(r["x_W"]) == pytest.approx(xW, abs=1e-8)

    @pytest.mark.parametrize("alpha,xF,xW", [(2.5, 0.5, 0.2)])
    def test_quadrature_on_tabulated_curve(self, alpha, xF, xW):
        x = np.linspace(0, 1, 101)
        curve = tabulated(x, ystar(alpha, x))
        r = rayleigh(curve, xF, x_W=xW)
        assert float(r["ln_F_over_W"]) == pytest.approx(closed_form(alpha, xF, xW), abs=1e-5)
        w = np.exp(-closed_form(alpha, xF, xW))
        r2 = rayleigh(curve, xF, W_over_F=w)
        assert float(r2["x_W"]) == pytest.approx(xW, abs=1e-5)

    def test_exactly_one_spec(self):
        with pytest.raises(ValueError):
            rayleigh(constant_alpha(2.0), 0.5)
        with pytest.raises(ValueError):
            rayleigh(constant_alpha(2.0), 0.5, W_over_F=0.5, x_W=0.3)

    def test_gradients(self):
        c = lambda a, xf, xw: rayleigh(constant_alpha(a), xf, x_W=xw)["ln_F_over_W"]
        check_grads(c, (2.5, 0.5, 0.2), order=1, modes=["rev"])
        d = lambda a, xf, w: rayleigh(constant_alpha(a), xf, W_over_F=w)["x_W"]
        check_grads(d, (2.5, 0.5, 0.4), order=1, modes=["rev"], atol=1e-5, rtol=1e-5)
        # tabulated path
        x = np.linspace(0, 1, 101)
        e = lambda xf: rayleigh(tabulated(x, ystar(2.5, x)), xf, W_over_F=0.4)["x_W"]
        check_grads(e, (0.5,), order=1, modes=["rev"], atol=1e-5, rtol=1e-5)

    def test_flag_not_nan_past_an_azeotrope(self):
        # a minimum-boiling azeotrope at x = 0.5: y* - x changes sign there
        x = np.linspace(0, 1, 101)
        y = x + 0.3 * x * (1 - x) * (1 - 2 * x) * 4 * 0 + 0.4 * x * (1 - x) * (0.5 - x) * 4
        curve = tabulated(x, np.maximum.accumulate(y))
        r = rayleigh(curve, 0.8, x_W=0.2)
        assert not bool(r["ok"])
        assert np.isinf(float(r["ln_F_over_W"]))
        assert float(r["W_over_F"]) == 0.0
        assert np.isfinite(float(r["x_D_avg"]))


# ---------------------------------------------------------------------------
# BatchDistillation, constant reflux
# ---------------------------------------------------------------------------


ALPHA = 2.5


def make(mode="constant_reflux", **kw):
    base = dict(species_order=["a", "b"], n_stages=3, R=3.0, boilup=2.0, mode=mode)
    base.update(kw)
    return BatchDistillation(BatchDistillationParams(**base), curve=constant_alpha(ALPHA))


class TestConstantReflux:
    def test_against_independent_time_integration(self):
        E = 0.7                               # Murphree efficiency rides the same path
        u = make(stage_efficiency=E)
        still, dist, info = u(charge(), t_final=30.0, n_traj=5)
        W, xW, Dl = np_batch(ALPHA, 100.0, 0.5, 3.0, 2.0, 3, 30.0, E=E)
        assert float(info["t_batch"]) == pytest.approx(30.0, rel=1e-8)
        assert float(info["W_final"]) == pytest.approx(W, abs=1e-8)       # exact: t = (F-W)(R+1)/V
        assert float(info["x_W_final"]) == pytest.approx(xW, abs=1e-6)
        assert float(dist["F_a"]) == pytest.approx(Dl, abs=1e-5)
        assert float(still["F_a"]) == pytest.approx(W * xW, abs=1e-5)
        assert bool(info["feasible"]) and bool(info["stop_reached"])

    def test_zero_stages_is_rayleigh(self):
        u = make(n_stages=0, R=2.0)
        still, dist, info = u(charge(), stop={"distillate": 55.0}, n_traj=5)
        r = rayleigh(constant_alpha(ALPHA), 0.5, W_over_F=0.45)
        assert float(info["x_W_final"]) == pytest.approx(float(r["x_W"]), abs=1e-7)
        assert float(info["x_D_avg"]) == pytest.approx(float(r["x_D_avg"]), abs=1e-7)
        # and the instantaneous distillate is in equilibrium with the still
        assert np.asarray(info["x_D"])[-1] == pytest.approx(ystar(ALPHA, float(info["x_W_final"])), abs=1e-9)

    def test_validation(self):
        with pytest.raises(NotImplementedError):
            BatchDistillationParams(species_order=["a", "b"], holdup=1.0)
        with pytest.raises(ValueError):
            BatchDistillationParams(species_order=["a", "b", "c"])
        with pytest.raises(ValueError):
            make()(charge())
        with pytest.raises(ValueError):
            make()(charge(), stop={"bogus": 1.0})
        with pytest.raises(ValueError):
            BatchDistillation(BatchDistillationParams(species_order=["a", "b"]))


# ---------------------------------------------------------------------------
# constant composition
# ---------------------------------------------------------------------------


class TestConstantComposition:
    def test_composition_held_and_reflux_rises(self):
        u = make("constant_composition", x_D_spec=0.9)
        still, dist, info = u(charge(), stop={"x_W": 0.2}, n_traj=21)
        np.testing.assert_allclose(np.asarray(info["x_D"]), 0.9, atol=1e-6)
        R = np.asarray(info["R"])
        assert np.all(np.diff(R) > 0) and R[0] > 0
        assert float(info["x_D_avg"]) == pytest.approx(0.9, abs=1e-9)
        assert bool(info["feasible"])
        # every R(x_W) satisfies the stage relation for x_D = 0.9
        for r, xw in zip(R, np.asarray(info["x_W"])):
            assert np_march(ALPHA, 0.9, r / (r + 1), 3) == pytest.approx(xw, abs=1e-8)

    def test_balances_and_exact_W(self):
        u = make("constant_composition", x_D_spec=0.9)
        still, dist, info = u(charge(30.0, 70.0), stop={"x_W": 0.25}, n_traj=5)
        W = 100.0 * (0.9 - 0.3) / (0.9 - 0.25)                        # exact for constant x_D
        assert float(info["W_final"]) == pytest.approx(W, abs=1e-10)
        assert float(still["F_a"] + dist["F_a"]) == pytest.approx(30.0, abs=1e-10)
        assert float(still["F_b"] + dist["F_b"]) == pytest.approx(70.0, abs=1e-10)
        assert float(dist["F_a"]) == pytest.approx(0.9 * (100.0 - W), abs=1e-9)

# ---------------------------------------------------------------------------
# ethanol-water (NRTL): the azeotrope
# ---------------------------------------------------------------------------


def _sd(name, antoine, Tc):
    return SpeciesData(name=name, MW=1.0, Cp_coeffs=(0, 0, 0, 0), Hvap_coeffs=(1.0, 0.38, Tc),
                       antoine_coeffs=antoine)


@pytest.fixture(scope="module")
def ew():
    thermo = IdealThermo({
        "ethanol": _sd("ethanol", (10.24677, 1598.673, -46.424), 514.0),
        "water": _sd("water", (9.6543, 1435.264, -64.848), 647.0),
    })
    nrtl = NRTLParams(species=("ethanol", "water"),
                      a=jnp.array([[0.0, -0.8009], [3.4578, 0.0]]),
                      b=jnp.array([[0.0, 246.18], [-586.0809, 0.0]]),
                      alpha=jnp.array([[0.0, 0.3], [0.3, 0.0]]))
    return thermo, nrtl


class TestEthanolWater:
    def test_rayleigh_flags_azeotrope_crossing(self, ew):
        from difflow.units.batch_distillation import equilibrium_curve
        thermo, nrtl = ew
        curve = equilibrium_curve(thermo, ["ethanol", "water"], ATM, nrtl)
        assert bool(rayleigh(curve, 0.5, x_W=0.1)["ok"])
        r = rayleigh(curve, 0.5, x_W=0.95)             # would need to cross x_az
        assert not bool(r["ok"]) and np.isinf(float(r["ln_F_over_W"]))
        assert np.isfinite(float(r["x_D_avg"]))


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------


class TestGradients:
    def _unit(self, **kw):
        return make(n_stages=2, **kw)

    def test_constant_reflux_wrt_R_boilup_xF(self):
        def f(R, V, xF):
            u = BatchDistillation(BatchDistillationParams(species_order=["a", "b"], n_stages=2, R=R, boilup=V),
                                  curve=constant_alpha(ALPHA))
            ch = {"F_a": 100.0 * xF, "F_b": 100.0 * (1 - xF), "T": 350.0, "P": 1e5}
            _, _, info = u(ch, stop={"x_W": 0.2}, n_traj=3)
            return jnp.stack([info["recovery"], info["t_batch"]])

        check_grads(f, (2.5, 2.0, 0.5), order=1, modes=["rev"], atol=1e-4, rtol=1e-4)

    def test_jit(self):
        u = self._unit(R=2.0)
        f = jax.jit(lambda ch: u(ch, stop={"x_W": 0.2}, n_traj=3)[2]["t_batch"])
        assert np.isfinite(float(f(charge())))


# ---------------------------------------------------------------------------
# batch_cuts
# ---------------------------------------------------------------------------


class TestBatchCuts:
    def test_cut_equals_continuous_run(self):
        """A constant-reflux batch stopped and restarted is the same batch."""
        u = make(n_stages=3, R=3.0)
        one = u(charge(), stop={"x_W": 0.2}, n_traj=3)
        res = batch_cuts(u, charge(), [{"distillate": 20.0}, {"x_W": 0.2}])
        assert float(res["residue"]["F_a"]) == pytest.approx(float(one[0]["F_a"]), abs=1e-7)
        assert float(res["total_time"]) == pytest.approx(float(one[2]["t_batch"]), rel=1e-7)


# ---------------------------------------------------------------------------
# multicomponent simple distillation
# ---------------------------------------------------------------------------


class TestMulticomponentRayleigh:
    @pytest.fixture(scope="class")
    def thermo(self):
        return IdealThermo({
            "A": _sd("A", (9.0, 1100.0, -50.0), 500.0),
            "B": _sd("B", (9.0, 1250.0, -50.0), 520.0),
            "C": _sd("C", (9.0, 1450.0, -50.0), 540.0),
        })

    def test_against_scipy(self, thermo):
        sp = ("A", "B", "C")
        xF = np.array([0.3, 0.4, 0.3])
        P = 1e5

        def K_and_T(x):
            f = lambda T: sum(x[i] * float(thermo.Psat(s, T)) for i, s in enumerate(sp)) / P - 1.0
            T = optimize.brentq(f, 250.0, 600.0, xtol=1e-12)
            return np.array([float(thermo.Psat(s, T)) / P for s in sp]), T

        def rhs(u, x):
            K, _ = K_and_T(x)
            return K * x - x

        ref = integrate.solve_ivp(rhs, [0.0, np.log(0.5)], xF, rtol=1e-10, atol=1e-12, method="DOP853")
        r = rayleigh_multicomponent(thermo, sp, jnp.asarray(xF), 0.5, P, n_steps=60)
        np.testing.assert_allclose(np.asarray(r["x_W"]), ref.y[:, -1], atol=2e-6)
        assert float(jnp.sum(r["x_W"])) == pytest.approx(1.0, abs=1e-9)
        # component balance defines the average distillate
        w = 0.5
        np.testing.assert_allclose(np.asarray(r["x_D_avg"]) * (1 - w), xF - w * np.asarray(r["x_W"]), atol=1e-12)
        assert float(r["x_D_avg"][0]) > xF[0] > float(r["x_W"][0])

    def test_gradient(self, thermo):
        f = lambda w: rayleigh_multicomponent(thermo, ("A", "B", "C"), jnp.array([0.3, 0.4, 0.3]), w, 1e5,
                                              n_steps=6)["x_W"][0]
        check_grads(f, (0.5,), order=1, modes=["rev"], atol=1e-4, rtol=1e-4)


class TestGeankoplis:
    """Geankoplis 5e Example 11.3-2: simple (Rayleigh) distillation.

    100 mol of 50/50 n-pentane/n-heptane at 101.3 kPa; 40 mol distilled.
    The book integrates 1/(y - x) graphically from its equilibrium table and
    reports x_W = 0.277 and y_av = 0.835.
    """

    x = jnp.array([0.0, 0.059, 0.145, 0.254, 0.398, 0.594, 0.867, 1.0])
    y = jnp.array([0.0, 0.271, 0.521, 0.701, 0.836, 0.925, 0.984, 1.0])

    def test_example_11_3_2(self):
        r = rayleigh(tabulated(self.x, self.y), 0.5, W_over_F=0.6)
        assert float(r["x_W"]) == pytest.approx(0.277, abs=0.003)
        assert float(r["x_D_avg"]) == pytest.approx(0.835, abs=0.005)
        assert float(r["ln_F_over_W"]) == pytest.approx(np.log(1 / 0.6), rel=1e-12)

    def test_curve_is_not_shared_between_units(self):
        """The jitted body takes the curve as an argument (it is not in the value key)."""
        p = BatchDistillationParams(["a", "b"], n_stages=0, R=0.0)
        out = [BatchDistillation(p, curve=constant_alpha(a))(charge(), stop={"distillate": 40.0})[2]
               for a in (2.0, 4.0)]
        assert float(out[0]["x_D_avg"]) < float(out[1]["x_D_avg"]) - 0.05
