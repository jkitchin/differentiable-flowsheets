"""Tests for difflow_power.physics.

The branch model is checked against hand algebra rather than against
another implementation in this package. Getting it wrong is the classic
way to build a power-flow tool that is quietly a few percent off: the
``tap = 0`` sentinel, which end the tap sits at, the sign of the phase
shift, and whether the charging susceptance is a total or a half are
each an easy mistake with a plausible-looking result.
"""

import cmath
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import difflow_power as dp
from difflow_power import physics as ph


def test_plain_line_block_is_symmetric():
    """With no tap, off-diagonals are equal and are minus the series y."""
    r, x, b = 0.01, 0.1, 0.05
    yff, yft, ytf, ytt = ph.branch_admittances(r, x, b)
    ys = 1.0 / complex(r, x)
    assert complex(yft) == pytest.approx(complex(-ys))
    assert complex(ytf) == pytest.approx(complex(-ys))
    assert complex(yff) == pytest.approx(complex(ys + 1j * b / 2))
    assert complex(ytt) == pytest.approx(complex(yff))


def test_tap_scales_the_from_side_only():
    """``Yff`` carries ``1/tau^2``, ``Ytt`` carries none."""
    r, x, tau = 0.0, 0.1, 0.95
    plain = ph.branch_admittances(r, x, 0.0, 1.0, 0.0)
    tapped = ph.branch_admittances(r, x, 0.0, tau, 0.0)
    assert complex(tapped[0]) == pytest.approx(complex(plain[0]) / tau ** 2)
    assert complex(tapped[1]) == pytest.approx(complex(plain[1]) / tau)
    assert complex(tapped[2]) == pytest.approx(complex(plain[2]) / tau)
    assert complex(tapped[3]) == pytest.approx(complex(plain[3]))


def test_phase_shift_makes_the_block_non_reciprocal():
    """``Yft / Ytf = exp(2 j theta)``: equal magnitude, rotated phase.

    A reciprocal branch has ``Yft == Ytf``. A phase shifter does not,
    and that asymmetry is exactly what lets it control real power flow
    rather than just impede it.
    """
    theta = math.radians(15.0)
    yff, yft, ytf, ytt = ph.branch_admittances(0.01, 0.1, 0.0, 1.0, theta)
    assert abs(complex(yft)) == pytest.approx(abs(complex(ytf)))
    assert complex(yft) / complex(ytf) == pytest.approx(
        cmath.exp(2j * theta)
    )
    assert abs(complex(yft) - complex(ytf)) > 1e-3
    # The diagonal is untouched: a pure shift moves no magnitude.
    assert complex(yff) == pytest.approx(complex(ytt))


def test_ybus_matches_hand_assembly_on_a_triangle():
    net = dp.cases.case3()
    y = np.asarray(net.ybus())
    index = net.bus_index
    want = np.zeros((3, 3), dtype=complex)
    for br in net.branches.values():
        f, t = index[br.from_bus], index[br.to_bus]
        ys = 1.0 / complex(br.r, br.x)
        want[f, f] += ys + 1j * br.b / 2
        want[t, t] += ys + 1j * br.b / 2
        want[f, t] += -ys
        want[t, f] += -ys
    np.testing.assert_allclose(y, want, atol=1e-12)


def test_ybus_is_symmetric_without_phase_shifters():
    for case in ("case3", "case5", "case9", "case14"):
        y = np.asarray(dp.cases.load_case(case).ybus())
        np.testing.assert_allclose(y, y.T, atol=1e-12)


def test_ybus_rows_sum_to_the_shunts_when_charging_is_absent():
    """Kirchhoff: with no charging and no shunts, every row sums to zero."""
    net = dp.cases.radial_feeder()      # all b = 0, no shunts
    y = np.asarray(net.ybus())
    np.testing.assert_allclose(y.sum(axis=1), 0.0, atol=1e-12)


def test_bus_shunt_enters_the_diagonal_with_the_case_file_sign():
    """case14's 19 MVAr capacitor at bus 9 adds +j0.19 pu."""
    net = dp.cases.case14()
    i = net.bus_index["9"]
    with_shunt = np.asarray(net.ybus())[i, i]
    net.buses["9"].b_shunt_mvar = 0.0
    without = np.asarray(net.ybus())[i, i]
    assert (with_shunt - without) == pytest.approx(0.19j)


def test_polynomial_cost_and_its_derivative():
    coefficients = (0.11, 5.0, 150.0)
    assert float(ph.polynomial_cost(100.0, coefficients)) == pytest.approx(
        0.11 * 100 ** 2 + 5 * 100 + 150
    )
    assert float(ph.marginal_cost(100.0, coefficients)) == pytest.approx(
        2 * 0.11 * 100 + 5
    )


def test_marginal_cost_agrees_with_autodiff():
    coefficients = (0.002, 0.11, 5.0, 150.0)
    for p in (0.0, 37.5, 210.0):
        assert float(ph.marginal_cost(p, coefficients)) == pytest.approx(
            float(jax.grad(ph.polynomial_cost)(p, coefficients))
        )


def test_per_unit_round_trips():
    assert ph.base_impedance(230.0, 100.0) == pytest.approx(529.0)
    z = ph.ohms_to_pu(52.9, 230.0, 100.0)
    assert z == pytest.approx(0.1)
    assert ph.pu_to_ohms(z, 230.0, 100.0) == pytest.approx(52.9)
    assert ph.pu_to_mw(ph.mw_to_pu(250.0)) == pytest.approx(250.0)


def test_line_charging_from_capacitance():
    """b_total = 2 pi f C L, converted to per unit."""
    b = ph.line_charging_pu(100.0, 12.0, 230.0, 100.0, 60.0)
    want = 2 * math.pi * 60.0 * 12e-9 * 100.0 * ph.base_impedance(230.0, 100.0)
    assert float(b) == pytest.approx(want)


def test_branch_flows_lose_real_power_in_the_resistance():
    """A branch with a known current loses exactly I^2 R."""
    r, x = 0.02, 0.2
    yff, yft, ytf, ytt = ph.branch_admittances(r, x, 0.0)
    v = jnp.asarray([1.0 + 0j, 0.98 * jnp.exp(-0.05j)])
    s_f, s_t = ph.branch_power_flows(v, [0], [1], yff, yft, ytf, ytt)
    current = jnp.abs((v[0] - v[1]) / complex(r, x))
    assert float(jnp.real(s_f + s_t)[0]) == pytest.approx(
        float(current ** 2 * r)
    )


def test_apparent_power_squared_has_bounded_curvature_at_the_origin():
    """The reason thermal limits are posed on the square, not the modulus.

    ``|S| = sqrt(P^2 + Q^2)`` has curvature going as ``1 / |S|``, so an
    interior-point step taken near a lightly loaded branch --- exactly
    where early iterates sit --- sees an enormous Hessian entry. The
    square is a plain quadratic there.
    """

    def modulus(pq):
        return jnp.sqrt(pq[0] ** 2 + pq[1] ** 2)

    def squared(pq):
        return ph.apparent_power_squared(pq[0] + 1j * pq[1])

    near_origin = jnp.asarray([1e-6, 0.0])
    assert float(jnp.max(jnp.abs(jax.hessian(modulus)(near_origin)))) > 1e5
    assert float(
        jnp.max(jnp.abs(jax.hessian(squared)(near_origin)))
    ) == pytest.approx(2.0)


def test_admittances_differentiate_with_respect_to_reactance():
    def imag_yff(x):
        return jnp.imag(ph.branch_admittances(0.01, x, 0.05)[0])

    got = jax.grad(imag_yff)(0.1)
    step = 1e-7
    want = (imag_yff(0.1 + step) - imag_yff(0.1 - step)) / (2 * step)
    assert float(got) == pytest.approx(float(want), rel=1e-6)
