"""PIDController, LinearMPC/run_mpc and the flowsheet -> StateSpaceModel adapter (#520)."""

import jax
import jax.numpy as jnp
import pytest

from difflow import make_stream
from difflow.control import LinearMPC, box_qp, run_mpc
from difflow.dynamic import DynamicCSTR, DynamicFlowsheet, PIDController
from difflow.mhe import StateSpaceModel, linear_model

jax.config.update("jax_enable_x64", True)

V, F = 1.0, 10.0


def _rate(C, T, p):
    return jnp.array([1e-3 * jnp.exp(-4000.0 * (1 / T - 1 / 340.0)) * C["A"]])


def _loop(controller):
    """Jacketed CSTR whose coolant temperature comes from ``controller``."""
    fs = DynamicFlowsheet(["A", "B"])
    fs.add_feed("feed", make_stream({"A": 0.5, "B": F - 0.5}, T=330.0, P=2e5))
    cstr = DynamicCSTR(volume=V, rate_fn=_rate, stoich=jnp.array([[-1.0], [1.0]]),
                       species_order=["A", "B"], dH_rxn=jnp.array([-60e3]),
                       mode="jacket", UA=2e3, Tc="coolant", name="reactor")
    fs.add_unit(cstr, inlets={"inlet": "feed", "coolant": "Tc"}, outlets={"outlet": "product"})
    fs.add_unit(controller, inlets={"measurement": "product"}, outlets={"output": "Tc"})
    return fs


def _y0(fs):
    y0 = fs.initial_state()
    return y0.at[2].set(345.0)        # start the reactor off its setpoint


def _sim(fs, params=None):
    return fs.simulate((0.0, 600.0), y0=_y0(fs), method="RK4", n_steps=60, params=params)


def test_pi_closed_loop_runs():
    fs = _loop(PIDController(Kc=5.0, tau_I=200.0, setpoint=340.0, bias=330.0, name="TC"))
    assert fs.combined_state_spec().names[-1] == "TC.I"
    res = _sim(fs)
    assert bool(jnp.all(jnp.isfinite(res.trajectory.y)))
    hist = res.stream_history("Tc")
    assert hist["T"].shape == res.trajectory.t.shape
    with pytest.raises(KeyError):
        res.stream_history("nope")


def test_pid_with_derivative_and_limits_runs():
    fs = _loop(PIDController(Kc=5.0, tau_I=200.0, tau_D=20.0, setpoint=340.0, bias=330.0,
                             u_min=300.0, u_max=335.0, name="TC"))
    assert fs.combined_state_spec().names[-2:] == ["TC.I", "TC.y_f"]
    res = _sim(fs)
    assert bool(jnp.all(jnp.isfinite(res.trajectory.y)))
    Tc = res.stream_history("Tc")["T"]
    assert float(Tc.max()) <= 335.0 + 1e-12 and float(Tc.min()) >= 300.0 - 1e-12


def test_grad_wrt_gains_is_finite():
    fs = _loop(PIDController(Kc=5.0, tau_I=200.0, setpoint=340.0, bias=330.0, u_max=336.0, name="TC"))

    def ise(Kc):
        y = _sim(fs, params={"Kc": Kc}).trajectory.y
        return jnp.sum((y[:, 2] - 340.0) ** 2)

    g = jax.grad(ise)(5.0)
    assert jnp.isfinite(g)


def test_back_calculation_bounds_the_integral():
    def max_integral(tau_T):
        pi = PIDController(Kc=5.0, tau_I=200.0, setpoint=360.0, bias=330.0, u_max=331.0,
                           tau_T=tau_T, name="TC")
        return float(jnp.max(jnp.abs(_sim(_loop(pi)).trajectory.y[:, -1])))

    assert max_integral(50.0) < max_integral(jnp.inf)


def test_direct_action_and_setpoint_function():
    pi = PIDController(Kc=-5.0, tau_I=200.0, setpoint=lambda t: 340.0 + 0.0 * t, bias=330.0,
                       action="direct", name="TC")
    assert bool(jnp.all(jnp.isfinite(_sim(_loop(pi)).trajectory.y)))


@pytest.mark.parametrize("kwargs", [
    {"action": "sideways"}, {"tau_I": 0.0}, {"tau_D": -1.0}, {"tau_D": 1.0, "alpha": 0.0},
    {"tau_T": 0.0}, {"u_min": 5.0, "u_max": 1.0}
])
def test_pid_validation(kwargs):
    base = {"Kc": 1.0, "tau_I": 10.0, "setpoint": 1.0}
    with pytest.raises(ValueError):
        PIDController(**{**base, **kwargs})


def test_pid_measure_must_be_key_or_callable():
    with pytest.raises(TypeError):
        PIDController(Kc=1.0, tau_I=10.0, setpoint=1.0, measure=3)


def test_pid_needs_a_measurement_port():
    pi = PIDController(Kc=1.0, tau_I=10.0, setpoint=1.0)
    s = make_stream({"A": 1.0}, T=300.0, P=1e5)
    with pytest.raises(ValueError, match="measurement"):
        pi.outputs(0.0, jnp.zeros(1), {"a": s, "b": s})
    with pytest.raises(KeyError):
        PIDController(Kc=1.0, tau_I=10.0, setpoint=1.0, measure="F_Z").outputs(
            0.0, jnp.zeros(1), {"measurement": s})


def test_box_qp_matches_unconstrained_and_clips():
    H = jnp.array([[2.0, 0.5], [0.5, 1.0]])
    g = jnp.array([-1.0, -1.0])
    u, kkt = box_qp(H, g, jnp.full(2, -10.0), jnp.full(2, 10.0))
    assert jnp.allclose(u, jnp.linalg.solve(H, -g), atol=1e-12) and kkt < 1e-12
    u, kkt = box_qp(jnp.eye(2), jnp.array([-2.0, 1.0]), jnp.array([-1.0, 0.0]), jnp.ones(2))
    assert jnp.allclose(u, jnp.array([1.0, 0.0])) and kkt < 1e-12


def _first_order():
    return linear_model(jnp.array([[0.9]]), jnp.array([[1.0]]), b=jnp.array([[0.1]]))


def test_mpc_runs_and_respects_bounds():
    model = _first_order()
    mpc = LinearMPC.from_model(model, jnp.zeros(1), jnp.zeros(1), horizon=10, r_move=0.01,
                               u_min=-1.0, u_max=1.0)
    res = run_mpc(mpc, model, jnp.zeros(1), 30, 2.0)
    assert res.u.shape == (30, 1) and res.y.shape == (31, 1)
    assert bool(jnp.all(jnp.isfinite(res.y)))
    assert float(res.u.max()) <= 1.0 + 1e-12 and float(res.u.min()) >= -1.0 - 1e-12
    res = run_mpc(mpc, lambda x, u: 0.85 * x + 0.12 * u, jnp.zeros(1), 30, jnp.full((30, 1), 0.5))
    assert bool(jnp.all(jnp.isfinite(res.y)))


@pytest.mark.parametrize("kwargs", [{"horizon": 0}, {"u_min": 1.0, "u_max": 0.0}, {"q": -1.0}])
def test_mpc_validation(kwargs):
    base = {"A": [[0.9]], "B": [[0.1]], "C": [[1.0]], "horizon": 5}
    with pytest.raises(ValueError):
        LinearMPC(**{**base, **kwargs})


def test_run_mpc_validation():
    mpc = LinearMPC(A=[[0.9]], B=[[0.1]], C=[[1.0]], horizon=5)
    with pytest.raises(ValueError):
        run_mpc(mpc, lambda x, u: x, jnp.zeros(1), 4, jnp.zeros((3, 1)))
    with pytest.raises(TypeError):
        run_mpc(mpc, "plant", jnp.zeros(1), 4, 0.0)
    with pytest.raises(ValueError):
        run_mpc(mpc, lambda x, u: x, jnp.zeros(1), 0, 0.0)


def test_state_space_model_from_flowsheet():
    fs = DynamicFlowsheet(["A", "B"])
    fs.add_feed("feed", lambda t, args: make_stream({"A": 0.5, "B": F - 0.5}, T=330.0, P=2e5))
    fs.add_unit(DynamicCSTR(volume=V, rate_fn=_rate, stoich=jnp.array([[-1.0], [1.0]]),
                            species_order=["A", "B"], dH_rxn=jnp.array([-60e3]), mode="jacket",
                            UA=2e3, Tc=lambda t, args: args[0], name="reactor"),
                inlets={"inlet": "feed"}, outlets={"outlet": "product"})
    model = StateSpaceModel.from_flowsheet(fs, lambda x, u, th: x[2:3], n_y=1, n_u=1, dt=10.0)
    assert model.x_names == ["reactor.n_A", "reactor.n_B", "reactor.T"]
    x1 = model.step(fs.initial_state(args=jnp.array([320.0])), jnp.array([320.0]), jnp.zeros(3))
    assert bool(jnp.all(jnp.isfinite(x1)))
