"""EPANET INP reader and extended-period simulation: run, finite, differentiate, validate."""

import jax
import jax.numpy as jnp
import pytest

from difflow_water.network import NetworkModel, example_inp, read_inp, simulate

MINIMAL = """[JUNCTIONS]
 J1  10  5
[RESERVOIRS]
 R1  50
[PIPES]
 P1  R1  J1  100  150  110  0  Open
[OPTIONS]
 Units  LPS
 Headloss  {headloss}
[END]
"""


def test_read_inp_example():
    net, sch = read_inp(example_inp("Net1"))
    assert (len(net.junctions), len(net.pipes), len(net.pumps), len(net.tanks)) == (9, 12, 1, 1)
    assert len(sch.controls) == 2 and sch.duration == 24 * 3600


def test_read_minimal_inp_and_solve(tmp_path):
    p = tmp_path / "m.inp"
    p.write_text(MINIMAL.format(headloss="H-W"))
    net, _ = read_inp(p)
    assert net.junctions["J1"].demand == pytest.approx(5e-3)
    assert NetworkModel(net).solve()["converged"]


@pytest.mark.parametrize("bad", ["C-M", "rules", "units", "valve", "volcurve"])
def test_read_inp_rejects_unsupported(tmp_path, bad):
    text = MINIMAL.format(headloss="C-M" if bad == "C-M" else "H-W")
    if bad == "rules":
        text += "[RULES]\nRULE 1\nIF TANK 1 LEVEL ABOVE 3\nTHEN PUMP 9 STATUS IS CLOSED\n"
    elif bad == "units":
        text = text.replace("Units  LPS", "Units  BARRELS")
    elif bad == "valve":
        text = text.replace("[OPTIONS]", "[VALVES]\n V1 J1 R1 100 GPV 1 0\n[OPTIONS]")
    elif bad == "volcurve":
        text = text.replace("[OPTIONS]", "[TANKS]\n T1 20 1 0 5 10 0 VC1\n[OPTIONS]")
    p = tmp_path / "bad.inp"
    p.write_text(text)
    with pytest.raises(ValueError):
        read_inp(p)


def test_simulate_runs():
    net, sch = read_inp(example_inp("Net1"))
    res = simulate(net, sch, duration=4 * 3600)
    assert len(res["time"]) == 5
    for k in ("head", "pressure", "flow", "tank_level", "status"):
        assert jnp.all(jnp.isfinite(res[k])), k


def test_simulate_replay_gradient_is_finite():
    net, sch = read_inp(example_inp("Net1"))
    model = NetworkModel(net)
    res = simulate(net, sch, model=model, duration=4 * 3600)
    r0 = model.parameters()["roughness"]

    def level(rough):
        return simulate(net, sch, model=model, duration=4 * 3600, plan=res["plan"],
                        params={"roughness": rough})["tank_level"][-1, 0]
    assert jnp.all(jnp.isfinite(jax.grad(level)(r0)))


def test_energy_is_read_and_pump_energy_is_finite():
    net, sch = read_inp(example_inp("Net1"))
    assert sch.energy["efficiency"] == pytest.approx(0.75)
    res = simulate(net, sch, duration=3 * 3600)
    assert jnp.all(jnp.isfinite(res["pump_energy_kwh"])) and jnp.all(res["step_power"] >= 0)


def test_speed_profile_gradient_is_finite():
    net, sch = read_inp(example_inp("Net1"))
    sch.controls = []
    model = NetworkModel(net)
    ip = model.arrays.links.index("9")

    def run(s, plan=None):
        prof = jnp.ones((2, model.arrays.n_links)).at[:, ip].set(s)
        return simulate(net, sch, model=model, duration=2 * 3600, speed_profile=prof,
                        profile_step=3600.0, plan=plan)
    plan = run(jnp.array([1.0, 0.9]))["plan"]
    g = jax.grad(lambda s: jnp.sum(run(s, plan)["pump_energy_kwh"]))(jnp.array([1.0, 0.9]))
    assert jnp.all(jnp.isfinite(g))
