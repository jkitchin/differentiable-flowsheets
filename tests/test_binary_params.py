"""NRTL/UNIQUAC binary-parameter database (issue #442, Ch. 2).

Every parameter set is checked against a physical fact that does not come
from the set itself, using the database's own Antoine constants at 1 atm:

- azeotrope compositions and temperatures: CRC Handbook of Chemistry and
  Physics, "Azeotropic Data for Binary Mixtures" (after Horsley), with the
  weight fractions there converted to mole fractions;
- "no azeotrope" where none is known (methanol-water, acetone-benzene, ...);
- acetone-chloroform boils ABOVE both pure components (maximum-boiling);
- benzene-toluene is close to ideal (gamma near 1).

A set that fails such a check is not in the database (see the module
docstring of difflow.binary_params for the two that were dropped).
"""

import warnings

import jax
import jax.numpy as jnp
import pytest
from jax.test_util import check_grads

from difflow import IdealThermo
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
from difflow.database import get_species_data
from difflow.phase_diagrams import find_azeotrope
from difflow.units.lle import nrtl_activity_coefficients

jax.config.update("jax_enable_x64", True)

P_ATM = 101325.0

# (species pair as (1, 2), x1 at the azeotrope, T_az K, tol x, tol T K).
# CRC/Horsley data at 1 atm; weight % -> mole fraction with the database MWs.
NRTL_AZEOTROPES = [
    (("ethanol", "water"), 0.894, 351.3, 0.02, 0.5),
    (("ethyl_acetate", "ethanol"), 0.54, 345.0, 0.03, 0.5),   # 69.4 wt%, 71.8 C
    (("acetone", "methanol"), 0.80, 328.6, 0.03, 0.5),       # 88 wt%, 55.5 C
    (("ethanol", "benzene"), 0.448, 341.0, 0.03, 0.5),       # 32.4 wt%, 67.9 C
    (("methanol", "benzene"), 0.61, 331.0, 0.03, 1.0),       # 39 wt%, 57.5-58.3 C
    (("ethanol", "cyclohexane"), 0.445, 338.0, 0.03, 0.5),   # 30.5 wt%, 64.9 C
    (("acetone", "chloroform"), 0.34, 337.7, 0.03, 0.5),     # 20 wt%, 64.5 C
    # Heterogeneous azeotropes: the homogeneous search still lands on y = x
    # near the measured vapour composition.
    (("water", "1_butanol"), 0.753, 365.85, 0.05, 0.5),      # 42.5 wt% water, 92.7 C
    (("ethyl_acetate", "water"), 0.69, 343.55, 0.04, 0.5),   # 91.5 wt%, 70.4 C
]
NRTL_NO_AZEOTROPE = [("methanol", "water"), ("benzene", "toluene"),
                     ("acetone", "benzene"), ("methanol", "ethanol"),
                     ("methanol", "1_butanol"), ("ethanol", "2_propanol")]

UNIQUAC_AZEOTROPES = [
    (("ethanol", "water"), 0.894, 351.3, 0.025, 0.5),
    (("2_propanol", "water"), 0.675, 353.4, 0.025, 0.5),    # 87.4 wt%, 80.3 C
    (("1_propanol", "water"), 0.432, 360.85, 0.025, 0.5),   # 71.7 wt%, 87.7 C
    (("acetone", "methanol"), 0.80, 328.6, 0.03, 0.5),
    (("methanol", "benzene"), 0.61, 331.0, 0.03, 1.0),
    (("ethanol", "benzene"), 0.448, 341.0, 0.03, 0.5),
    (("methanol", "toluene"), 0.883, 336.95, 0.025, 0.5),   # 72.4 wt%, 63.8 C
    (("ethanol", "toluene"), 0.81, 349.85, 0.03, 0.5),      # 68 wt%, 76.7 C
    (("methanol", "chloroform"), 0.347, 326.65, 0.025, 0.5),  # 12.5 wt%, 53.5 C
    (("ethanol", "chloroform"), 0.163, 332.45, 0.025, 0.5),   # 7 wt%, 59.3 C
    (("ethanol", "n_hexane"), 0.332, 331.85, 0.03, 0.5),      # 21 wt%, 58.7 C
]
UNIQUAC_NO_AZEOTROPE = [("water", "methanol"), ("water", "acetone"),
                        ("benzene", "toluene"), ("acetone", "benzene"),
                        ("methanol", "ethanol")]


def _thermo(pair):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return IdealThermo({s: get_species_data(s) for s in pair})


def _az(pair, model):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Antoine range: scans touch the ends
        return find_azeotrope(_thermo(pair), pair, P=P_ATM, activity_model=model)


# ---------------------------------------------------------------- coverage


def test_every_nrtl_pair_is_verified():
    tested = {frozenset(p) for p, *_ in NRTL_AZEOTROPES} | {
        frozenset(p) for p in NRTL_NO_AZEOTROPE}
    assert {frozenset(k) for k in NRTL_BINARY_PARAMETERS} == tested


def test_every_uniquac_pair_is_verified():
    tested = {frozenset(p) for p, *_ in UNIQUAC_AZEOTROPES} | {
        frozenset(p) for p in UNIQUAC_NO_AZEOTROPE}
    assert {frozenset(k) for k in UNIQUAC_BINARY_PARAMETERS} == tested


def test_every_entry_has_a_source_and_valid_alpha():
    for (i, j), e in NRTL_BINARY_PARAMETERS.items():
        assert "DECHEMA" in e.source
        assert 0.0 < e.alpha < 1.0
        assert binary_parameter_source("nrtl", j, i) == e.source
    for (i, j), e in UNIQUAC_BINARY_PARAMETERS.items():
        assert "ChemSep" in e.source
        assert i in UNIQUAC_RQ and j in UNIQUAC_RQ


# ------------------------------------------------------------ physical facts


@pytest.mark.parametrize("pair,x_ref,T_ref,tol_x,tol_T", NRTL_AZEOTROPES)
def test_nrtl_azeotrope(pair, x_ref, T_ref, tol_x, tol_T):
    az = _az(pair, get_nrtl_params(pair))
    assert bool(az["found"])
    assert float(az["x"]) == pytest.approx(x_ref, abs=tol_x)
    assert float(az["T"]) == pytest.approx(T_ref, abs=tol_T)


@pytest.mark.parametrize("pair", NRTL_NO_AZEOTROPE)
def test_nrtl_no_azeotrope(pair):
    assert not bool(_az(pair, get_nrtl_params(pair))["found"])


@pytest.mark.parametrize("pair,x_ref,T_ref,tol_x,tol_T", UNIQUAC_AZEOTROPES)
def test_uniquac_azeotrope(pair, x_ref, T_ref, tol_x, tol_T):
    az = _az(pair, get_uniquac_params(pair))
    assert bool(az["found"])
    assert float(az["x"]) == pytest.approx(x_ref, abs=tol_x)
    assert float(az["T"]) == pytest.approx(T_ref, abs=tol_T)


@pytest.mark.parametrize("pair", UNIQUAC_NO_AZEOTROPE)
def test_uniquac_no_azeotrope(pair):
    assert not bool(_az(pair, get_uniquac_params(pair))["found"])


def test_acetone_chloroform_is_maximum_boiling():
    """Negative deviation: gamma < 1, and the azeotrope boils above both."""
    pair = ("acetone", "chloroform")
    p = get_nrtl_params(pair)
    g = p.gamma(jnp.array([0.5, 0.5]), 335.0)
    assert float(g[0]) < 1.0 and float(g[1]) < 1.0
    th = _thermo(pair)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # chloroform's Tb is its range's end
        Tb = [float(th.saturation_T(s, P_ATM)) for s in pair]
    assert float(_az(pair, p)["T"]) > max(Tb)


def test_ethanol_water_minimum_boiling():
    pair = ("ethanol", "water")
    th = _thermo(pair)
    T_az = float(_az(pair, get_nrtl_params(pair))["T"])
    assert T_az < min(float(th.saturation_T(s, P_ATM)) for s in pair)


@pytest.mark.parametrize("getter", [get_nrtl_params, get_uniquac_params])
def test_benzene_toluene_near_ideal(getter):
    p = getter(("benzene", "toluene"))
    for x1 in (1e-6, 0.5, 1 - 1e-6):
        g = p.gamma(jnp.array([x1, 1 - x1]), 365.0)
        assert jnp.all(jnp.abs(jnp.log(g)) < 0.05)


def test_water_butanol_splits_into_two_liquids():
    """The DECHEMA set (alpha 0.44) predicts the known miscibility gap.

    Water/1-butanol is partially miscible at 25 C (about 2 and 49 mol%
    butanol in the two layers): the Gibbs energy of mixing must be
    non-convex between them.
    """
    p = get_nrtl_params(("water", "1_butanol"))
    T = 298.15

    def gmix(x2):
        x = jnp.array([1 - x2, x2])
        return jnp.sum(x * jnp.log(x * p.gamma(x, T)))

    d2 = jax.vmap(jax.grad(jax.grad(gmix)))(jnp.linspace(0.05, 0.45, 41))
    assert float(jnp.min(d2)) < 0.0
    # and the one-phase ends are stable
    assert float(jax.grad(jax.grad(gmix))(0.005)) > 0.0
    assert float(jax.grad(jax.grad(gmix))(0.9)) > 0.0


# ----------------------------------------------------------------- lookups


def test_nrtl_convention_and_symmetry():
    p = get_nrtl_params(("water", "ethanol"))      # reversed from storage
    e = NRTL_BINARY_PARAMETERS[("ethanol", "water")]
    assert float(p.b[1, 0]) == pytest.approx(e.A_ij / R_CAL)  # ethanol->water
    assert float(p.b[0, 1]) == pytest.approx(e.A_ji / R_CAL)
    assert float(p.alpha[0, 1]) == float(p.alpha[1, 0]) == e.alpha
    assert jnp.all(p.a == 0.0)
    # Order of the species list does not change the physics.
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
    assert jnp.all(jnp.diag(p.b) == 0.0)
    g = p.gamma(jnp.array([0.2, 0.3, 0.5]), 350.0)
    assert jnp.all(jnp.isfinite(g)) and jnp.all(g > 0.0)


def test_missing_pair_raises_by_default():
    with pytest.raises(KeyError, match="toluene"):
        get_nrtl_params(("toluene", "water"))


def test_missing_pair_ideal_warns_and_is_ideal():
    with pytest.warns(MissingBinaryParametersWarning, match="toluene/water"):
        p = get_nrtl_params(("toluene", "water"), missing="ideal")
    g = p.gamma(jnp.array([0.4, 0.6]), 350.0)
    assert jnp.allclose(g, 1.0)
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


# --------------------------------------------------------------- gradients


def test_nrtl_gamma_gradients():
    p = get_nrtl_params(("ethanol", "water"))

    def f(x1, T):
        x = jnp.stack([x1, 1.0 - x1])
        return jnp.log(nrtl_activity_coefficients(x, T, p))[0]

    check_grads(f, (0.4, 350.0), order=1, modes=["rev", "fwd"])


def test_uniquac_gamma_gradients():
    p = get_uniquac_params(("2_propanol", "water"))

    def f(x1, T):
        x = jnp.stack([x1, 1.0 - x1])
        return jnp.log(p.gamma(x, T))[1]

    check_grads(f, (0.4, 353.0), order=1, modes=["rev", "fwd"])


@pytest.mark.slow
def test_azeotrope_composition_gradient_in_pressure():
    """The ethanol-water azeotrope shifts with P; the derivative is finite
    and matches a central difference (implicit differentiation through the
    database-assembled model)."""
    pair = ("ethanol", "water")
    th, p = _thermo(pair), get_nrtl_params(pair)

    def x_az(P):
        return find_azeotrope(th, pair, P=P, activity_model=p)["x"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = float(jax.grad(x_az)(P_ATM))
        h = 500.0
        fd = (float(x_az(P_ATM + h)) - float(x_az(P_ATM - h))) / (2 * h)
    assert jnp.isfinite(g)
    assert g == pytest.approx(fd, rel=1e-3, abs=1e-10)
