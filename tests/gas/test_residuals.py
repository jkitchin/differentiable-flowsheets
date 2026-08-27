"""Tests for difflow_gas.residuals.

The point of the module is to be the traceable twin of
:mod:`difflow_gas.verify`, so the load-bearing tests are the
entry-by-entry round trip against it and the demonstration that
``jax.jit`` and ``jax.jacobian`` work where ``verify`` cannot.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import difflow_gas as dg
from difflow_gas import verify
from difflow_gas.residuals import (
    gas_state_layout,
    network_residuals,
    residual_names,
)
from tests.gas.test_network import mixed_network, triangle

P_SLACK = 50.0e5
MIXED_RATIOS = {"cs_1": 1.2}
MIXED_CV_BAR = {"cv_1": 2.0}


def _solve(net, root, ratios=None, cv_drops_pa=None):
    """Solve a network and return ``(p_bar, q_kg_s, dec)``."""
    fs, dec = dg.build_network_flowsheet(
        net, root=root, p_slack_pa=P_SLACK,
        ratios=ratios, cv_drops_pa=cv_drops_pa,
    )
    streams = fs.solve(tol=1e-12, max_iter=500)
    return (
        verify.node_pressures_bar(streams, dec),
        verify.arc_flows_kg_s(streams, dec),
        dec,
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def triangle_state():
    net = triangle()
    p_bar, q, _ = _solve(net, "n0")
    return net, p_bar, q


@pytest.fixture(scope="module")
def mixed_state():
    net = mixed_network()
    cv_pa = {k: v * 1e5 for k, v in MIXED_CV_BAR.items()}
    p_bar, q, _ = _solve(net, "s1", ratios=MIXED_RATIOS, cv_drops_pa=cv_pa)
    return net, p_bar, q


# =============================================================================
# Agreement with verify
# =============================================================================


class TestRoundTripAgainstVerify:
    """Same equations, same numbers, different representation."""

    @staticmethod
    def _compare(net, p_bar, q, **kw):
        layout = gas_state_layout(net)
        x = layout.pack(p_bar, q, net.supply_kg_s)
        # eps_flow=0 recovers verify's exact non-smooth q|q|.
        got = network_residuals(x, net, layout, eps_flow=0.0, **kw)
        names = residual_names(net, layout)

        report = verify.residuals_from_values(
            p_bar, q, net, cv_drops_bar=kw.get("cv_drops_bar")
        )
        expected = {}
        expected.update(
            {f"balance_{k}": v for k, v in report.node_imbalance.items()}
        )
        expected.update(
            {f"resistance_{k}": v
             for k, v in report.resistance_residual_bar2.items()}
        )
        expected.update(
            {f"equality_{k}": v for k, v in report.equality_dp_bar.items()}
        )
        expected.update(
            {f"cv_{k}": v for k, v in report.control_valve_residual_bar.items()}
        )
        for i, name in enumerate(names):
            if name in expected:
                assert float(got[i]) == pytest.approx(
                    expected[name], abs=1e-12
                ), f"{name} disagrees with verify"
        return names, got

    def test_triangle(self, triangle_state):
        net, p_bar, q = triangle_state
        names, _ = self._compare(net, p_bar, q)
        assert names == [
            "balance_n0", "balance_n1", "balance_n2",
            "resistance_p01", "resistance_p02", "resistance_p12",
        ]

    def test_mixed_network_covers_every_arc_kind(self, mixed_state):
        """pipe, resistor, valve, short pipe, control valve, compressor."""
        net, p_bar, q = mixed_state
        names, _ = self._compare(
            net, p_bar, q, ratios=MIXED_RATIOS, cv_drops_bar=MIXED_CV_BAR
        )
        assert any(n.startswith("equality_") for n in names)
        assert any(n.startswith("cv_") for n in names)
        assert any(n.startswith("compressor_") for n in names)

    def test_residuals_vanish_at_the_sequential_solution(self, triangle_state):
        net, p_bar, q = triangle_state
        layout = gas_state_layout(net)
        x = layout.pack(p_bar, q, net.supply_kg_s)
        r = network_residuals(x, net, layout, eps_flow=0.0)
        assert float(jnp.max(jnp.abs(r))) < 1e-9


class TestCompressorResidual:
    """The block verify deliberately omits."""

    def test_present_and_zero_at_the_solution(self, mixed_state):
        net, p_bar, q = mixed_state
        layout = gas_state_layout(net)
        names = residual_names(net, layout)
        assert "compressor_cs_1" in names

        x = layout.pack(p_bar, q, net.supply_kg_s)
        r = network_residuals(
            x, net, layout, ratios=MIXED_RATIOS, cv_drops_bar=MIXED_CV_BAR
        )
        i = names.index("compressor_cs_1")
        assert float(r[i]) == pytest.approx(0.0, abs=1e-9)

    def test_nonzero_for_the_wrong_ratio(self, mixed_state):
        net, p_bar, q = mixed_state
        layout = gas_state_layout(net)
        names = residual_names(net, layout)
        x = layout.pack(p_bar, q, net.supply_kg_s)
        r = network_residuals(
            x, net, layout, ratios={"cs_1": 1.5}, cv_drops_bar=MIXED_CV_BAR
        )
        i = names.index("compressor_cs_1")
        assert abs(float(r[i])) > 1.0

    def test_verify_does_not_check_it(self, mixed_state):
        """verify reports ok for a ratio the compressor is not running.

        This is why a reconciliation formulation cannot reuse verify's
        equation set as-is.
        """
        net, p_bar, q = mixed_state
        report = verify.residuals_from_values(
            p_bar, q, net, cv_drops_bar=MIXED_CV_BAR
        )
        assert report.ok


# =============================================================================
# Traceability -- the reason the module exists
# =============================================================================


class TestTraceability:
    def test_jit(self, triangle_state):
        net, p_bar, q = triangle_state
        layout = gas_state_layout(net)
        x = layout.pack(p_bar, q, net.supply_kg_s)
        fn = jax.jit(lambda xx: network_residuals(xx, net, layout))
        assert bool(jnp.all(jnp.isfinite(fn(x))))

    def test_jacobian_is_finite_and_correctly_shaped(self, triangle_state):
        net, p_bar, q = triangle_state
        layout = gas_state_layout(net)
        x = layout.pack(p_bar, q, net.supply_kg_s)
        a = jax.jacobian(network_residuals)(x, net, layout)
        assert a.shape == (len(residual_names(net, layout)), layout.size)
        assert bool(jnp.all(jnp.isfinite(a)))

    def test_balance_rows_have_unit_supply_entries(self, triangle_state):
        """Each balance row carries a +1 in its own supply column.

        That is what gives the Jacobian full row rank; with the
        supplies held fixed the balance block is the incidence matrix,
        whose rank is only ``n_nodes - 1``.
        """
        net, p_bar, q = triangle_state
        layout = gas_state_layout(net)
        x = layout.pack(p_bar, q, net.supply_kg_s)
        a = np.asarray(jax.jacobian(network_residuals)(x, net, layout))
        names = residual_names(net, layout)
        for node in layout.nodes:
            row = names.index(f"balance_{node}")
            col = layout.index(f"s_{node}")
            assert a[row, col] == pytest.approx(1.0)
        assert np.linalg.matrix_rank(a) == a.shape[0]

    def test_gradient_wrt_efficiency(self, triangle_state):
        """A pipe coefficient passed as an argument stays differentiable."""
        net, p_bar, q = triangle_state
        layout = gas_state_layout(net)
        x = layout.pack(p_bar, q, net.supply_kg_s)
        names = residual_names(net, layout)
        i = names.index("resistance_p01")

        g = jax.grad(
            lambda e: network_residuals(
                x, net, layout, efficiencies={"p01": e}
            )[i]
        )(1.0)
        assert jnp.isfinite(g)
        # more resistance -> the law's residual falls
        assert float(g) < 0.0


# =============================================================================
# Layout
# =============================================================================


class TestLayout:
    def test_pack_unpack_round_trip(self, triangle_state):
        net, p_bar, q = triangle_state
        layout = gas_state_layout(net)
        x = layout.pack(p_bar, q, net.supply_kg_s)
        p2, q2, s2, extra = layout.unpack(x)

        assert extra == {}
        for n in layout.nodes:
            assert float(p2[n]) == pytest.approx(p_bar[n])
        for a in layout.arcs:
            assert float(q2[a]) == pytest.approx(q[a])
        for n in layout.supply_nodes:
            assert float(s2[n]) == pytest.approx(net.supply_kg_s[n])

    def test_names_match_size_and_order(self, triangle_state):
        net, _, _ = triangle_state
        layout = gas_state_layout(net, efficiency_arcs=["p01"])
        assert len(layout.names) == layout.size
        assert layout.names[0].startswith("p_")
        assert layout.names[-1] == "eta_p01"
        assert layout.index("eta_p01") == layout.size - 1
        assert layout.default_scale.shape == (layout.size,)

    def test_index_rejects_unknown_names(self, triangle_state):
        net, _, _ = triangle_state
        layout = gas_state_layout(net)
        with pytest.raises(KeyError, match="not a state variable"):
            layout.index("q_nope")

    def test_layout_rejects_wrong_arc_kinds(self, mixed_state):
        net, _, _ = mixed_state
        with pytest.raises(ValueError, match="only pipes and resistors"):
            gas_state_layout(net, efficiency_arcs=["cs_1"])
        with pytest.raises(ValueError, match="not a compressor"):
            gas_state_layout(net, ratio_arcs=["pipe_1"])

    def test_efficiency_as_state_and_as_argument_agree(self, triangle_state):
        """The same eta gives the same residual either way it is supplied.

        In the state it is *estimated*; as an argument it is a fixed
        parameter you can differentiate with respect to. The physics is
        identical.
        """
        net, p_bar, q = triangle_state
        state_layout = gas_state_layout(net, efficiency_arcs=["p01"])
        plain_layout = gas_state_layout(net)
        i = residual_names(net, plain_layout).index("resistance_p01")

        from_state = network_residuals(
            state_layout.pack(p_bar, q, net.supply_kg_s, {"eta_p01": 1.5}),
            net, state_layout, eps_flow=0.0,
        )[residual_names(net, state_layout).index("resistance_p01")]
        from_arg = network_residuals(
            plain_layout.pack(p_bar, q, net.supply_kg_s),
            net, plain_layout, efficiencies={"p01": 1.5}, eps_flow=0.0,
        )[i]
        assert float(from_state) == pytest.approx(float(from_arg), abs=1e-12)

    def test_state_efficiency_takes_precedence_over_the_argument(
        self, triangle_state
    ):
        """A variable being estimated ignores any fixed value passed in."""
        net, p_bar, q = triangle_state
        layout = gas_state_layout(net, efficiency_arcs=["p01"])
        i = residual_names(net, layout).index("resistance_p01")
        x = layout.pack(p_bar, q, net.supply_kg_s, {"eta_p01": 1.0})

        with_arg = network_residuals(
            x, net, layout, efficiencies={"p01": 9.0}, eps_flow=0.0
        )[i]
        without = network_residuals(x, net, layout, eps_flow=0.0)[i]
        assert float(with_arg) == pytest.approx(float(without), abs=1e-12)


# =============================================================================
# Smoothing
# =============================================================================


class TestSmoothing:
    def test_bias_is_negligible_at_the_solution(self, triangle_state):
        net, p_bar, q = triangle_state
        layout = gas_state_layout(net)
        x = layout.pack(p_bar, q, net.supply_kg_s)
        exact = network_residuals(x, net, layout, eps_flow=0.0)
        smooth = network_residuals(x, net, layout)
        assert float(jnp.max(jnp.abs(exact - smooth))) < 1e-8

    def test_zero_flow_row_degenerates_in_flow_but_not_pressure(
        self, triangle_state
    ):
        """A pipe at zero flow cannot tell you its flow.

        The resistance row's flow entry vanishes there, which is
        physical rather than an artefact: with no pressure drop there
        is nothing to infer a flow from. Smoothing keeps the row
        differentiable without pretending the information is there.
        """
        net, p_bar, q = triangle_state
        layout = gas_state_layout(net)
        zero_q = dict.fromkeys(q, 0.0)
        x = layout.pack(p_bar, zero_q, net.supply_kg_s)
        a = np.asarray(jax.jacobian(network_residuals)(x, net, layout))
        names = residual_names(net, layout)

        row = names.index("resistance_p01")
        col = layout.index("q_p01")
        # d/dq [q sqrt(q^2 + eps^2)] = eps at q = 0, so the entry is
        # exactly beta_bar2 * eps rather than zero -- small enough to
        # lose the flow, large enough to keep the Jacobian defined.
        beta_bar2 = net.beta["p01"] / 1e10
        assert a[row, col] == pytest.approx(
            -beta_bar2 * dg.EPS_FLOW, rel=1e-9
        )
        assert abs(a[row, col]) < 1e-3, "flow sensitivity is negligible at q = 0"
        assert abs(a[row, layout.index("p_n0")]) > 1.0, "pressure entry survives"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
