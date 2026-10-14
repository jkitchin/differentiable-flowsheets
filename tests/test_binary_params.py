"""NRTL/UNIQUAC binary-parameter database (issue #442): lookup contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow.binary_params import (
    NRTL_BINARY_PARAMETERS,
    R_CAL,
    UNIQUAC_BINARY_PARAMETERS,
    UNIQUAC_RQ,
    MissingBinaryParametersWarning,
    binary_parameter_source,
    get_nrtl_params,
    get_uniquac_params,
    list_binary_pairs,
)

jax.config.update("jax_enable_x64", True)


def test_every_entry_has_a_source_and_valid_alpha():
    for (i, j), e in NRTL_BINARY_PARAMETERS.items():
        assert 0.0 < e.alpha < 1.0
        assert binary_parameter_source("nrtl", j, i) == e.source
    for (i, j), e in UNIQUAC_BINARY_PARAMETERS.items():
        assert i in UNIQUAC_RQ and j in UNIQUAC_RQ


def test_nrtl_convention_and_symmetry():
    p = get_nrtl_params(("water", "ethanol"))      # reversed from storage
    e = NRTL_BINARY_PARAMETERS[("ethanol", "water")]
    assert float(p.b[1, 0]) == pytest.approx(e.A_ij / R_CAL)  # ethanol->water
    assert float(p.b[0, 1]) == pytest.approx(e.A_ji / R_CAL)
    q = get_nrtl_params(("ethanol", "water"))
    g1 = p.gamma(jnp.array([0.3, 0.7]), 350.0)
    g2 = q.gamma(jnp.array([0.7, 0.3]), 350.0)
    assert jnp.allclose(g1, g2[::-1], rtol=1e-12)


def test_aliases_and_species_tuple():
    p = get_nrtl_params(["EtOH", "water"])
    assert p.species == ("EtOH", "water")
    q = get_uniquac_params(["IPA", "water"])
    assert float(q.r[0]) == UNIQUAC_RQ["2_propanol"][0]


def test_multicomponent_assembly():
    p = get_nrtl_params(("methanol", "ethanol", "water"))
    assert p.b.shape == (3, 3)
    g = p.gamma(jnp.array([0.2, 0.3, 0.5]), 350.0)
    assert jnp.all(jnp.isfinite(g))


def test_missing_pair_raises_by_default():
    with pytest.raises(KeyError, match="toluene"):
        get_nrtl_params(("toluene", "water"))


def test_missing_pair_ideal_warns_and_is_ideal():
    with pytest.warns(MissingBinaryParametersWarning, match="toluene/water"):
        p = get_nrtl_params(("toluene", "water"), missing="ideal")
    assert jnp.allclose(p.gamma(jnp.array([0.4, 0.6]), 350.0), 1.0)
    with pytest.warns(MissingBinaryParametersWarning):
        get_uniquac_params(("toluene", "water"), missing="ideal")


def test_uniquac_requires_r_q():
    with pytest.raises(KeyError, match="r, q"):
        get_uniquac_params(("cyclohexane", "ethanol"), missing="ideal")


def test_bad_arguments():
    with pytest.raises(ValueError):
        get_nrtl_params(("water",))
    with pytest.raises(ValueError):
        get_nrtl_params(("water", "water"))
    with pytest.raises(ValueError):
        get_nrtl_params(("ethanol", "water"), missing="zero")
    with pytest.raises(ValueError):
        list_binary_pairs("wilson")
    assert ("ethanol", "water") in list_binary_pairs()
    assert ("water", "ethanol") in list_binary_pairs("uniquac")


@pytest.mark.parametrize("getter", [get_nrtl_params, get_uniquac_params])
def test_gamma_grad_finite(getter):
    p = getter(("ethanol", "water"))
    g = jax.grad(lambda x1, T: jnp.log(p.gamma(jnp.stack([x1, 1.0 - x1]), T))[0],
                 argnums=(0, 1))(0.4, 350.0)
    assert all(np.isfinite(float(v)) for v in g)
