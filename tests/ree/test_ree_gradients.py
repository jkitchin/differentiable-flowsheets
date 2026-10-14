"""jax.grad runs through the REE extractor and returns a finite value."""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from difflow.streams import get_flows, make_stream
from difflow_ree.units.extraction import REEExtractor, REEExtractorParams


def test_extractor_grad_is_finite():
    feed = make_stream({"H2O": 55.0, "Nd": 0.3, "Dy": 0.036}, T=298.15, P=101325.0)
    solvent = make_stream({"D2EHPA": 3.0, "kerosene": 9.0}, T=298.15, P=101325.0)

    def raffinate_nd(pH, a_nd):
        params = REEExtractorParams(
            n_stages=5.0, extractant="D2EHPA", elements=("Nd", "Dy"), pH=0.3,
            extractant_conc=1.0, coefficient_overrides={"Nd": {"a": a_nd}})
        raff, _, _ = REEExtractor(params)(feed, solvent, pH=pH)
        return get_flows(raff)["Nd"]

    g = jax.grad(raffinate_nd, argnums=(0, 1))(0.3, -5.0)
    assert all(jnp.isfinite(x) for x in g)
