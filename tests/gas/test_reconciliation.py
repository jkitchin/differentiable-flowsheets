"""Data reconciliation of gas networks.

Covers the full story of ``examples/28_data_reconciliation.ipynb``:
reconciling noisy measurements, the precision the constraints buy,
finding a bad meter, estimating an unmeasured pipe efficiency, and the
observability boundary that estimation runs into.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import difflow_gas as dg
from difflow_gas import verify
from difflow_gas.reconcile import (
    measurement_sigma,
    network_residual_fn,
    perturb,
    reconcile_network,
    reconciled_values,
)
from difflow_gas.residuals import gas_state_layout
from difflow.reconciliation import (
    ReconciliationStructureError,
    global_test,
    measurement_test,
    sensor_ranking,
    serial_elimination,
    solve_reconciliation,
)
from tests.gas.test_network import triangle

RATIOS = {"cs1": 1.2}
P_SLACK_PA = 60.0e5
LOOP_VARS = ["q_p2", "q_p3", "q_p4", "p_b", "p_c", "p_d"]


def five_node() -> dg.GasNetwork:
    """Source, one compressor station, and a b-c-d loop.

    The network of ``examples/20_gas_network_flowsheets.ipynb``:
    realistic Weymouth coefficients, 60 bar, 120 kg/s.
    """
    return dg.GasNetwork(
        arcs={
            "p1": ("src", "a", "pipe"),
            "cs1": ("a", "b", "compressor"),
            "p2": ("b", "c", "pipe"),
            "p3": ("b", "d", "pipe"),
            "p4": ("c", "d", "pipe"),
        },
        beta={
            aid: dg.weymouth_beta(
                length_m=length, diameter_m=0.6, roughness_m=1e-4
            )
            for aid, length in [
                ("p1", 20e3), ("p2", 40e3), ("p3", 60e3), ("p4", 80e3)
            ]
        },
        supply_kg_s={"src": 120.0, "c": -50.0, "d": -70.0},
        pressure_bounds_bar={n: (30.0, 80.0) for n in "abcd" } | {"src": (30.0, 80.0)},
    )


def _solve(net, root, p_slack_pa, ratios=None):
    fs, dec = dg.build_network_flowsheet(
        net, root=root, p_slack_pa=p_slack_pa, ratios=ratios
    )
    streams = fs.solve(tol=1e-12, max_iter=500)
    return (
        verify.node_pressures_bar(streams, dec),
        verify.arc_flows_kg_s(streams, dec),
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def five():
    """``(net, layout, residual_fn, x_true, sigma)`` for the 5-node network."""
    net = five_node()
    p_bar, q = _solve(net, "src", P_SLACK_PA, RATIOS)
    layout = gas_state_layout(net)
    residual_fn = network_residual_fn(net, layout, ratios=RATIOS)
    x_true = layout.pack(p_bar, q, net.supply_kg_s)
    sigma = measurement_sigma(layout)
    return net, layout, residual_fn, x_true, sigma


@pytest.fixture(scope="module")
def five_noisy(five):
    net, layout, _, x_true, sigma = five
    y = perturb(x_true, sigma, jax.random.PRNGKey(0))
    return reconcile_network(net, y, sigma, layout, ratios=RATIOS), y


# =============================================================================
# The triangle, against its closed form
# =============================================================================


class TestTriangleAnalytic:
    def test_noise_free_state_is_a_fixed_point(self):
        """Reconciling consistent data must change nothing."""
        net = triangle()
        p_bar, q = _solve(net, "n0", 50.0e5)
        layout = gas_state_layout(net)
        x_true = layout.pack(p_bar, q, net.supply_kg_s)
        sigma = measurement_sigma(layout)

        res = reconcile_network(net, x_true, sigma, layout)

        assert res.converged
        assert res.objective == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_allclose(
            np.asarray(res.x), np.asarray(x_true), atol=1e-9
        )

    def test_recovers_the_closed_form_split(self):
        """x^2 - 200 x + 3400 = 0 for the p01 flow."""
        net = triangle()
        p_bar, q = _solve(net, "n0", 50.0e5)
        layout = gas_state_layout(net)
        sigma = measurement_sigma(layout)
        y = perturb(
            layout.pack(p_bar, q, net.supply_kg_s), sigma, jax.random.PRNGKey(2)
        )
        res = reconcile_network(net, y, sigma, layout)

        exact = 100.0 - math.sqrt(100.0**2 - 3400.0)
        assert res.x_named["q_p01"] == pytest.approx(exact, rel=0.05)
        assert res.converged


# =============================================================================
# The 5-node network
# =============================================================================


class TestFiveNodeReconciliation:
    def test_noisy_measurements_are_made_consistent(self, five, five_noisy):
        """Balances and pipe laws are O(1) before and ~0 after."""
        net, layout, _, _, _ = five
        res, y = five_noisy

        before = verify.residuals_from_values(
            {n: float(y[layout.index(f"p_{n}")]) for n in layout.nodes},
            {a: float(y[layout.index(f"q_{a}")]) for a in layout.arcs},
            net,
        )
        assert before.max_node_imbalance_kg_s > 0.1
        assert before.max_resistance_residual_bar2 > 1.0

        assert res.converged
        assert res.residual_norm < 1e-9

    def test_verify_accepts_the_reconciled_state(self, five, five_noisy):
        """Close the loop through the plugin's own equation checker."""
        net, layout, _, _, _ = five
        res, _ = five_noisy
        p_bar, q, supply = reconciled_values(res, layout)

        after = verify.residuals_from_values(p_bar, q, net)
        # verify uses the network's nominations, which are the truth
        # here; the reconciled supplies are what must balance.
        assert after.max_resistance_residual_bar2 < 1e-6
        assert abs(sum(supply.values())) < 1e-8, "nominations must close"

    def test_reconciliation_sharpens_every_estimate(self, five, five_noisy):
        net, layout, _, _, sigma = five
        res, _ = five_noisy
        for i, name in enumerate(layout.names):
            assert res.std[name] < float(sigma[i]), (
                f"{name}: sd {res.std[name]:.4f} not below sigma {sigma[i]:.4f}"
            )

    def test_the_chord_flow_gains_least(self, five, five_noisy):
        """q_p4 carries the least flow and is the least improved.

        Pinned because it is the hook for the sensor-placement section
        of the notebook.
        """
        _, _, _, _, sigma = five
        res, _ = five_noisy
        gain = {
            name: 1.0 - (res.std[name] / float(sigma[i])) ** 2
            for i, name in enumerate(res.names)
        }
        flows = {k: v for k, v in gain.items() if k.startswith("q_")}
        assert min(flows, key=flows.get) == "q_p4"
        assert flows["q_p4"] == pytest.approx(0.58, abs=0.05)
        assert flows["q_p3"] == pytest.approx(0.93, abs=0.05)

    def test_degrees_of_redundancy(self, five_noisy):
        res, _ = five_noisy
        assert res.structure.degree_of_redundancy == 10
        assert res.structure.solvable

    def test_clean_data_passes_the_global_test(self, five):
        net, layout, _, x_true, sigma = five
        res = reconcile_network(net, x_true, sigma, layout, ratios=RATIOS)
        assert not global_test(res).detected

    def test_gradient_through_the_reconciliation(self, five):
        """d(reconciled flow)/d(pipe coefficient) is finite and signed."""
        net, layout, residual_fn, x_true, sigma = five
        y = perturb(x_true, sigma, jax.random.PRNGKey(1))
        res = reconcile_network(net, y, sigma, layout, ratios=RATIOS)

        def q_p3_hat(efficiency):
            return solve_reconciliation(
                residual_fn, y, sigma, x0=x_true, scaling=res.scaling,
                params={"p3": efficiency}, n_steps=12,
            )[0][layout.index("q_p3")]

        g = jax.grad(q_p3_hat)(1.0)
        assert jnp.isfinite(g)
        assert float(g) < 0.0, "a dirtier pipe must carry less flow"


# =============================================================================
# Gross errors
# =============================================================================


class TestGrossErrors:
    def test_biased_flow_meter_is_identified(self, five):
        net, layout, _, x_true, sigma = five
        y = perturb(
            x_true, sigma, jax.random.PRNGKey(0),
            layout=layout, gross_errors={"q_p3": 8.0},
        )
        res = reconcile_network(net, y, sigma, layout, ratios=RATIOS)

        gt = global_test(res)
        assert gt.detected
        assert gt.statistic > gt.critical

        mt = measurement_test(res)
        assert mt.suspect == "q_p3"
        assert abs(mt.z["q_p3"]) > 5.0
        ranked = mt.ranked()
        assert abs(ranked[0][1]) > 2.0 * abs(ranked[1][1]), (
            "the suspect should stand clear of the next candidate"
        )

    def test_error_on_a_weakly_redundant_sensor_smears(self, five):
        """A bias on the least redundant meter may be misattributed.

        Documented rather than wished away: least squares spreads a
        gross error over its neighbours, and the thinner the redundancy
        the more the blame moves.
        """
        net, layout, _, x_true, sigma = five
        y = perturb(
            x_true, sigma, jax.random.PRNGKey(5),
            layout=layout, gross_errors={"q_p4": 8.0},
        )
        res = reconcile_network(net, y, sigma, layout, ratios=RATIOS)

        assert global_test(res).detected
        mt = measurement_test(res)
        assert mt.suspect in {"q_p4", "q_p2", "q_p3", "s_c", "s_d"}

    def test_serial_elimination_clears_the_data(self, five):
        net, layout, residual_fn, x_true, sigma = five
        y = perturb(
            x_true, sigma, jax.random.PRNGKey(0),
            layout=layout, gross_errors={"q_p3": 8.0},
        )
        steps = serial_elimination(
            residual_fn, y, sigma, names=layout.names,
            unmeasured_scale=layout.default_scale,
        )
        assert steps[0].detected
        assert steps[0].suspect == "q_p3"
        assert not steps[-1].detected
        assert steps[-1].removed == "q_p3"
        assert steps[-1].dof == steps[0].dof - 1, (
            "discarding a measurement costs a degree of redundancy"
        )


# =============================================================================
# Joint parameter estimation and observability
# =============================================================================


class TestEfficiencyEstimation:
    @staticmethod
    def _setup(net, unmeasured=(), sigma_eta=float("inf")):
        layout = gas_state_layout(net, efficiency_arcs=["p3"])
        residual_fn = network_residual_fn(net, layout, ratios=RATIOS)
        sigma = measurement_sigma(
            layout, sigma_eta=sigma_eta, unmeasured=unmeasured
        )
        return layout, residual_fn, sigma

    def test_clean_data_gives_unit_efficiency(self, five):
        net, _, _, _, _ = five
        p_bar, q = _solve(net, "src", P_SLACK_PA, RATIOS)
        layout, _, sigma = self._setup(net)
        x_true = layout.pack(p_bar, q, net.supply_kg_s, {"eta_p3": 1.0})

        res = reconcile_network(net, x_true, sigma, layout, ratios=RATIOS)
        assert res.x_named["eta_p3"] == pytest.approx(1.0, abs=1e-6)
        assert res.structure.degree_of_redundancy == 9, (
            "estimating a parameter costs one degree of redundancy"
        )
        assert res.std["eta_p3"] == pytest.approx(0.043, abs=0.005)

    def test_recovers_a_fouled_pipe(self, five):
        """A pipe 15% more resistive than the model is detected as such."""
        net, _, _, _, _ = five
        fouled = five_node()
        fouled.beta["p3"] = net.beta["p3"] * 1.15
        p_bar, q = _solve(fouled, "src", P_SLACK_PA, RATIOS)

        layout, _, sigma = self._setup(net)
        x_true = layout.pack(p_bar, q, fouled.supply_kg_s, {"eta_p3": 1.0})
        y = perturb(x_true, sigma, jax.random.PRNGKey(3))

        # the residual model uses the CLEAN beta, so eta must absorb 1.15
        res = reconcile_network(net, y, sigma, layout, ratios=RATIOS)
        eta = res.x_named["eta_p3"]
        assert abs(eta - 1.15) < 3.0 * res.std["eta_p3"], (
            f"eta = {eta:.4f} +- {res.std['eta_p3']:.4f}, expected 1.15"
        )

    def test_unmeasured_loop_plus_efficiency_is_unobservable(self, five):
        """The rank boundary: six unknowns fit, seven do not."""
        net, _, _, _, _ = five
        p_bar, q = _solve(net, "src", P_SLACK_PA, RATIOS)
        layout, _, sigma = self._setup(net, unmeasured=LOOP_VARS)
        x_true = layout.pack(p_bar, q, net.supply_kg_s, {"eta_p3": 1.0})

        with pytest.raises(ReconciliationStructureError) as exc:
            reconcile_network(net, x_true, sigma, layout, ratios=RATIOS)
        assert "eta_p3" in str(exc.value)

    def test_the_same_loop_is_observable_without_the_efficiency(self, five):
        """The compressor relation is what buys the loop its observability.

        With eta dropped, the six unmeasured loop variables are still
        determined, because ``p_b = ratio * p_a`` ties the loop to the
        measured pressure upstream of the station. That relation is the
        one :mod:`difflow_gas.verify` omits.
        """
        net, layout, _, x_true, _ = five
        sigma = measurement_sigma(layout, unmeasured=LOOP_VARS)
        res = reconcile_network(net, x_true, sigma, layout, ratios=RATIOS)

        assert res.converged
        assert res.structure.solvable
        assert res.structure.degree_of_redundancy == 4

    def test_a_prior_restores_solvability(self, five):
        """Shrinking a weakly identified parameter beats failing on it."""
        net, _, _, _, _ = five
        p_bar, q = _solve(net, "src", P_SLACK_PA, RATIOS)
        layout, _, sigma = self._setup(
            net, unmeasured=LOOP_VARS, sigma_eta=0.1
        )
        x_true = layout.pack(p_bar, q, net.supply_kg_s, {"eta_p3": 1.0})

        res = reconcile_network(net, x_true, sigma, layout, ratios=RATIOS)
        assert res.converged
        assert res.std["eta_p3"] <= 0.1 + 1e-9, (
            "the posterior cannot be looser than the prior"
        )

    def test_sensor_ranking_prefers_a_loop_flow_meter(self, five):
        """The best sensor for eta is a meter on the loop flows."""
        net, _, _, _, _ = five
        p_bar, q = _solve(net, "src", P_SLACK_PA, RATIOS)
        layout, residual_fn, sigma = self._setup(
            net, unmeasured=LOOP_VARS, sigma_eta=0.1
        )
        x_true = layout.pack(p_bar, q, net.supply_kg_s, {"eta_p3": 1.0})

        ranked = sensor_ranking(
            residual_fn, x_true, sigma, target="eta_p3",
            candidates=LOOP_VARS, candidate_sigma=1.0, names=layout.names,
        )
        assert ranked[0]["candidate"].startswith("q_")
        assert ranked[0]["variance_reduction"] > 0.3
        reductions = [d["variance_reduction"] for d in ranked]
        assert reductions == sorted(reductions, reverse=True)
        assert all(d["sd_after"] <= d["sd_before"] + 1e-12 for d in ranked)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
