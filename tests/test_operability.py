"""Tests for difflow.operability — steady-state controllability screening.

The tests are organised around the four things that can be wrong with a
screen: the RGA arithmetic itself, the diagnosis of a plant that has no
inverse, the scaling that every magnitude depends on, and whether the gains
came out of AD correctly in the first place (checked against central
differences on a real CSTR + flash flowsheet, not on a toy linear map).
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from difflow import (
    CSTR,
    CSTRParams,
    IdealThermo,
    SpeciesData,
    get_flows,
    make_stream,
)
from difflow.units.flash import Flash, FlashParams
from difflow.operability import (
    OperabilityReport,
    OperabilityWarning,
    Scaling,
    condition_number,
    disturbance_condition_number,
    disturbance_gain,
    effective_rank,
    gain_matrix,
    min_singular_value,
    negative_pairings,
    pinv,
    required_input_move,
    rga,
    rga_number,
    screen,
    suggest_pairing,
)


# --------------------------------------------------------------------------
# A real flowsheet: A -> B in an isothermal CSTR, then a TP flash.
#
#   u = [reactor temperature, flash temperature]
#   y = [light in the vapour product, heavy in the liquid product]
#   d = [feed rate of light]
#
# Both units carry an inner solve (the CSTR's steady-state balance and the
# flash's Rachford-Rice root), so the Jacobians below are implicitly
# differentiated through them.  That is exactly what the FD comparison is
# testing.
# --------------------------------------------------------------------------

def _thermo():
    species = {
        "Light": SpeciesData(
            name="Light", MW=72.0, Cp_coeffs=(120.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(26000.0, 0.38, 470.0),
            antoine_coeffs=(10.422, 1687.537, -38.44), Hf=0.0),
        "Heavy": SpeciesData(
            name="Heavy", MW=114.0, Cp_coeffs=(190.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(35000.0, 0.38, 570.0),
            antoine_coeffs=(10.186, 2004.68, -60.53), Hf=0.0),
    }
    return IdealThermo(species)


def _rate_fn(C, T, params):
    k = params["A"] * jnp.exp(-params["Ea"] / (8.314 * T))
    return jnp.array([k * C["Light"]])


def _make_flowsheet(volume=2.0):
    """Return a pure callable ``fn(u, d) -> y`` for the CSTR + flash train."""
    thermo = _thermo()
    cstr = CSTR(
        CSTRParams(
            V=jnp.asarray(volume, dtype=float),
            rate_fn=_rate_fn,
            stoich=jnp.array([[-1.0], [1.0]]),
            rate_params={"A": jnp.array(1e3), "Ea": jnp.array(50000.0)},
            species_order=["Light", "Heavy"],
        ),
        thermo=thermo, mode="isothermal")
    flash = Flash(FlashParams(species_order=["Light", "Heavy"]), thermo=thermo)

    def fn(u, d):
        inlet = make_stream({"Light": d[0], "Heavy": 0.1},
                            T=320.0, P=101325.0)
        reacted, _ = cstr(inlet, T_spec=u[0])
        liquid, vapor, _ = flash(reacted, T=u[1], P=101325.0)
        return jnp.array([get_flows(vapor)["Light"],
                          get_flows(liquid)["Heavy"]])

    return fn


U0 = jnp.array([350.0, 380.0])
D0 = jnp.array([10.0])


@pytest.fixture(scope="module")
def flowsheet():
    return _make_flowsheet()


@pytest.fixture(scope="module")
def flowsheet_scaling():
    # 10 K of usable temperature swing on each unit; 0.2 mol/s of product
    # flow is the largest acceptable control error; the feed rate wanders by
    # 1 mol/s.
    return Scaling(u_span=[10.0, 10.0], y_span=[0.2, 0.2], d_span=[1.0])


def _central_difference_jacobian(fn, x0, h):
    cols = []
    for j in range(x0.shape[0]):
        e = jnp.zeros_like(x0).at[j].set(h)
        cols.append((np.asarray(fn(x0 + e)) - np.asarray(fn(x0 - e))) / (2 * h))
    return np.stack(cols, axis=1)


# --------------------------------------------------------------------------
# RGA
# --------------------------------------------------------------------------

def test_rga_2x2_matches_hand_arithmetic():
    """RGA of [[1, 2], [3, 4]]: det = -2, lambda_11 = 1*4/-2 = -2."""
    G = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(np.asarray(rga(G)),
                               [[-2.0, 3.0], [3.0, -2.0]], atol=1e-12)

    # A second, differently signed case: det = 3, lambda_11 = 2*2/3.
    G2 = jnp.array([[2.0, 1.0], [1.0, 2.0]])
    np.testing.assert_allclose(np.asarray(rga(G2)),
                               [[4 / 3, -1 / 3], [-1 / 3, 4 / 3]], atol=1e-12)


def test_skogestad_lv_column_matches_the_published_numbers():
    """External check against a case with textbook values.

    The distillation column of Skogestad and Postlethwaite, *Multivariable
    Feedback Control* 2nd ed. (Example 3.11 and section 3.5),
    ``G = [[0.878, -0.864], [1.082, -1.096]]``, is quoted there with
    ``lambda_11 = 35.1`` and a condition number of ``141.7``.  Reproducing
    both from this module's own code is the one check here that does not
    rest on arithmetic done in this repository.
    """
    G = jnp.array([[0.878, -0.864], [1.082, -1.096]])
    assert float(rga(G)[0, 0]) == pytest.approx(35.1, abs=0.05)
    assert float(condition_number(G, assume_scaled=True)) \
        == pytest.approx(141.7, abs=0.1)
    # sigma_min of the raw matrix is famously tiny: 0.0139.
    assert float(min_singular_value(G, assume_scaled=True)) \
        == pytest.approx(0.0139, abs=1e-4)

    # A 1% composition tolerance and unit reflux/boilup spans put sigma_min
    # just above 1 -- feasible, but only just, and the interaction warning
    # and directionality warning both stand.
    sc = Scaling(u_span=[1.0, 1.0], y_span=[0.01, 0.01])
    rep = screen(G, scaling=sc, u_names=["L", "V"], y_names=["x_D", "x_B"])
    assert float(rep.msv) == pytest.approx(1.391, abs=1e-3)
    assert float(rep.cond) == pytest.approx(141.7, abs=0.1)
    kinds = [f.kind for f in rep.findings]
    assert "directional" in kinds and "rga_interaction" in kinds
    assert "weak_direction" not in kinds
    assert "rga_negative" not in kinds       # 35.1 is positive, just large


def test_rga_of_diagonal_plant_is_identity():
    """A decoupled plant has no interaction whatever the gains are."""
    G = jnp.diag(jnp.array([7.0, -0.02, 1e4]))
    np.testing.assert_allclose(np.asarray(rga(G)), np.eye(3), atol=1e-10)


def test_rga_rows_and_columns_sum_to_one():
    """The defining invariant of the square RGA."""
    key = jax.random.PRNGKey(0)
    G = jax.random.normal(key, (4, 4)) + 2.0 * jnp.eye(4)
    R = np.asarray(rga(G))
    np.testing.assert_allclose(R.sum(axis=0), np.ones(4), atol=1e-9)
    np.testing.assert_allclose(R.sum(axis=1), np.ones(4), atol=1e-9)


def test_rga_is_invariant_to_diagonal_scaling():
    """The RGA is the one measure here that does not depend on units."""
    G = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    sc = Scaling(u_span=[1e3, 1e-2], y_span=[5.0, 1e4])
    np.testing.assert_allclose(np.asarray(rga(G)),
                               np.asarray(rga(sc.scale_gain(G))), atol=1e-10)


def test_non_square_rga_keeps_exactly_one_sum_rule():
    """Only one of the two sum rules survives, and which one is the shape.

    Row sums are ``diag(G G+)`` and column sums ``diag(G+ G)``.  A wide ``G``
    has full row rank, so ``G G+ = I`` and the *rows* sum to 1; a tall ``G``
    has full column rank, so ``G+ G = I`` and the *columns* do.  Getting this
    backwards is easy and is what the overactuated/underactuated findings
    tell the user, so it is pinned here with matrices big enough that the
    two rules genuinely differ (a 1-row matrix cannot tell them apart).
    """
    wide = jnp.array([[1.0, 2.0, 0.5], [0.3, 1.0, 2.0]])     # 2 x 3
    Rw = np.asarray(rga(wide))
    assert Rw.shape == (2, 3)
    np.testing.assert_allclose(Rw.sum(axis=1), np.ones(2), atol=1e-10)
    assert not np.allclose(Rw.sum(axis=0), np.ones(3), atol=1e-3)

    tall = np.asarray(wide).T                                # 3 x 2
    Rt = np.asarray(rga(jnp.asarray(tall)))
    assert Rt.shape == (3, 2)
    np.testing.assert_allclose(Rt.sum(axis=0), np.ones(2), atol=1e-10)
    assert not np.allclose(Rt.sum(axis=1), np.ones(3), atol=1e-3)

    # RGA(G.T) == RGA(G).T, which is what makes the two cases one statement.
    np.testing.assert_allclose(Rt, Rw.T, atol=1e-10)


def test_rga_is_not_transposed_anywhere_in_the_pipeline():
    """A 2x2 RGA is always symmetric, so orientation needs a 3x3 to test.

    Every check in this module that reads ``RGA[i, j]`` -- the pairing
    diagonal, the negative-pairing scan, ``suggest_pairing`` -- would be
    silently wrong on a transposed RGA and silently *right* on any 2x2.
    """
    G = jnp.array([[1.0, 2.0, 0.3], [0.4, 1.0, 3.0], [2.0, 0.1, 1.0]])
    R = np.asarray(rga(G))
    assert not np.allclose(R, R.T)          # the premise of the test

    # Hand value: RGA = G .* inv(G).T, computed independently with numpy.
    np.testing.assert_allclose(
        R, np.asarray(G) * np.linalg.inv(np.asarray(G)).T, atol=1e-12)
    assert R[0, 1] == pytest.approx(0.99009901, abs=1e-6)
    assert R[1, 0] == pytest.approx(-0.06966054, abs=1e-6)

    sc = Scaling(u_span=[1.0] * 3, y_span=[1.0] * 3)
    rep = screen(G, scaling=sc, u_names=["a", "b", "c"],
                 y_names=["p", "q", "r"], pairing=[1, 2, 0])
    np.testing.assert_allclose(np.asarray(rep.RGA), R, atol=1e-12)
    np.testing.assert_allclose(np.asarray(rep.rga_pairs),
                               [R[0, 1], R[1, 2], R[2, 0]], atol=1e-12)

    # Under this pairing every relative gain is near 1 and positive; under
    # the transposed RGA two of the three would be negative.
    assert negative_pairings(R, pairing=[1, 2, 0], is_rga=True) == []
    assert negative_pairings(R.T, pairing=[1, 2, 0], is_rga=True) != []
    assert suggest_pairing(G) == [1, 2, 0]
    assert rep.suggested_pairing() == [("p", "b"), ("q", "c"), ("r", "a")]


def test_rga_number_is_zero_for_a_decoupled_plant_and_pairing_aware():
    assert float(rga_number(jnp.eye(3))) == pytest.approx(0.0, abs=1e-12)
    # A plant that is perfectly decoupled under the *swapped* pairing.
    G = jnp.array([[0.0, 2.0], [3.0, 0.0]])
    assert float(rga_number(G)) == pytest.approx(4.0, abs=1e-9)
    assert float(rga_number(G, pairing=[1, 0])) == pytest.approx(0.0, abs=1e-9)


def test_negative_relative_gain_is_reported_as_an_error():
    """Diagonal pairing on [[1, 2], [3, 4]] has lambda = -2: flag it."""
    G = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    assert negative_pairings(G) == [(0, 0), (1, 1)]
    assert negative_pairings(G, pairing=[1, 0]) == []

    rep = screen(G, scaling=Scaling(u_span=[1.0, 1.0], y_span=[1.0, 1.0]))
    kinds = [f.kind for f in rep.findings]
    assert "rga_negative" in kinds
    assert all(f.severity == "error"
               for f in rep.findings if f.kind == "rga_negative")
    np.testing.assert_allclose(np.asarray(rep.rga_pairs), [-2.0, -2.0],
                               atol=1e-10)
    # ... and the greedy suggestion is the off-diagonal pairing instead.
    assert suggest_pairing(G) == [1, 0]
    assert rep.suggested_pairing() == [("y0", "u1"), ("y1", "u0")]


# --------------------------------------------------------------------------
# Singular and rank-deficient plants
# --------------------------------------------------------------------------

def test_singular_plant_is_diagnosed_not_silently_inverted():
    """Two inputs acting through one path: rank 1, and the RGA is not usable."""
    G = jnp.array([[1.0, 2.0], [2.0, 4.0]])          # exactly rank 1
    sc = Scaling(u_span=[1.0, 1.0], y_span=[1.0, 1.0])
    rep = screen(G, scaling=sc)

    assert float(rep.rank) == 1.0
    assert rep.singular
    assert float(rep.msv) == pytest.approx(0.0, abs=1e-12)
    assert float(rep.cond) > 1e12          # numerically infinite

    singular = [f for f in rep.findings if f.kind == "singular"]
    assert len(singular) == 1
    assert singular[0].severity == "error"
    assert "pseudo-inverse" in singular[0].detail

    # The pseudo-inverse RGA of a singular plant does *not* have unit row
    # sums -- that failure is the diagnosis, and it must not be papered over.
    rows = np.asarray(rep.RGA).sum(axis=1)
    assert not np.allclose(rows, 1.0)
    assert np.all(np.isfinite(np.asarray(rep.RGA)))


def test_effective_rank_and_pinv_agree_with_numpy():
    G = jnp.array([[1.0, 2.0], [2.0, 4.0000001], [0.0, 1.0]])
    assert float(effective_rank(G)) == 2.0
    np.testing.assert_allclose(np.asarray(pinv(G)),
                               np.linalg.pinv(np.asarray(G)), atol=1e-8)
    assert float(effective_rank(jnp.array([[1.0, 2.0], [2.0, 4.0]]))) == 1.0


def test_pinv_cutoff_keeps_an_exactly_singular_plant_finite():
    """The rcond cutoff is what stops 1/0 reaching the RGA.

    ``[[1, 2], [2, 4]]`` is singular only to rounding -- its smallest
    singular value is ~1e-16, and a pseudo-inverse with no cutoff merely
    returns 1e16 rather than nan.  A structurally singular matrix has an
    exactly zero singular value and is where a missing cutoff actually bites.
    """
    dead = jnp.array([[1.0, 0.0], [0.0, 0.0]])       # sigma = [1, 0] exactly
    P = np.asarray(pinv(dead))
    assert np.all(np.isfinite(P))
    np.testing.assert_allclose(P, np.linalg.pinv(np.asarray(dead)), atol=1e-12)
    assert np.all(np.isfinite(np.asarray(rga(dead))))
    assert float(effective_rank(dead)) == 1.0

    zero = jnp.zeros((2, 2))
    assert np.all(np.isfinite(np.asarray(pinv(zero))))
    np.testing.assert_allclose(np.asarray(pinv(zero)), np.zeros((2, 2)))
    assert float(effective_rank(zero)) == 0.0

    # And the cutoff is a real knob: a direction 1e-10 down is kept by
    # default and discarded by a loose rcond.
    weak = jnp.array([[1.0, 0.0], [0.0, 1e-10]])
    assert float(pinv(weak)[1, 1]) == pytest.approx(1e10, rel=1e-9)
    assert float(pinv(weak, rcond=1e-6)[1, 1]) == 0.0
    assert float(effective_rank(weak, rcond=1e-6)) == 1.0


def test_underactuated_and_overactuated_plants_are_flagged():
    """...and each finding states the sum rule that actually holds for it."""
    tall = screen(jnp.array([[1.0, 2.0], [0.0, 1.0], [1.0, 1.0]]),
                  scaling=Scaling(u_span=[1.0, 1.0], y_span=[1.0, 1.0, 1.0]))
    under = [f for f in tall.findings if f.kind == "underactuated"]
    assert len(under) == 1 and under[0].severity == "error"
    # Tall: columns sum to 1, rows do not.  The finding must say so.
    np.testing.assert_allclose(np.asarray(tall.RGA).sum(axis=0), [1.0, 1.0],
                               atol=1e-10)
    assert not np.allclose(np.asarray(tall.RGA).sum(axis=1), 1.0, atol=1e-3)
    assert "columns sum to 1" in under[0].detail

    wide = screen(jnp.array([[1.0, 2.0, 0.5], [0.3, 1.0, 2.0]]),
                  scaling=Scaling(u_span=[1.0] * 3, y_span=[1.0, 1.0]))
    over = [f for f in wide.findings if f.kind == "overactuated"]
    assert len(over) == 1 and over[0].severity == "warning"
    # Wide: rows sum to 1, columns do not.  The other way round.
    np.testing.assert_allclose(np.asarray(wide.RGA).sum(axis=1), [1.0, 1.0],
                               atol=1e-10)
    assert not np.allclose(np.asarray(wide.RGA).sum(axis=0), 1.0, atol=1e-3)
    assert "rows sum to 1" in over[0].detail


# --------------------------------------------------------------------------
# Scaling
# --------------------------------------------------------------------------

def test_scaling_moves_min_singular_value_the_right_way():
    """sigma_min scales exactly with the spans, in both directions."""
    G = jnp.array([[1.0, 0.0], [0.0, 0.5]])
    base = Scaling(u_span=[1.0, 1.0], y_span=[1.0, 1.0])
    msv_base = float(min_singular_value(G, base))
    assert msv_base == pytest.approx(0.5)

    # Twice the input travel -> the inputs can do twice as much.
    wider_u = Scaling(u_span=[2.0, 2.0], y_span=[1.0, 1.0])
    assert float(min_singular_value(G, wider_u)) == pytest.approx(1.0)

    # A tighter output tolerance is a harder problem: the same physical gain
    # now covers less of the (smaller) acceptable range... no: it covers
    # *more* of it, so sigma_min goes up.  Halving y_span doubles sigma_min.
    tighter_y = Scaling(u_span=[1.0, 1.0], y_span=[0.5, 0.5])
    assert float(min_singular_value(G, tighter_y)) == pytest.approx(1.0)

    # A looser tolerance on the weak output makes the plant look worse.
    looser_y = Scaling(u_span=[1.0, 1.0], y_span=[1.0, 2.0])
    assert float(min_singular_value(G, looser_y)) == pytest.approx(0.25)
    assert float(min_singular_value(G, looser_y)) < msv_base


def test_scaling_can_flip_the_controllability_verdict():
    """The same plant passes or fails depending only on the declared spans."""
    G = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    generous = Scaling(u_span=[10.0, 10.0], y_span=[1.0, 1.0])
    stingy = Scaling(u_span=[0.1, 0.1], y_span=[1.0, 1.0])
    assert float(min_singular_value(G, generous)) > 1.0
    assert float(min_singular_value(G, stingy)) < 1.0

    kinds = [f.kind for f in screen(G, scaling=stingy).findings]
    assert "weak_direction" in kinds
    assert screen(G, scaling=generous).ok


def test_condition_number_is_not_scaling_invariant():
    G = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    even = Scaling(u_span=[1.0, 1.0], y_span=[1.0, 1.0])
    skewed = Scaling(u_span=[1.0, 100.0], y_span=[1.0, 1.0])
    assert float(condition_number(G, even)) == pytest.approx(1.0)
    assert float(condition_number(G, skewed)) == pytest.approx(100.0)


def test_unscaled_use_is_loudly_caveated():
    G = jnp.array([[1.0, 0.0], [0.0, 1e5]])

    # 1. A metric called with no Scaling at all warns.
    with pytest.warns(OperabilityWarning, match="no Scaling"):
        min_singular_value(G)

    # 2. assume_scaled=True is the silent, deliberate opt-out.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        min_singular_value(G, assume_scaled=True)

    # 3. screen refuses to run without a Scaling at all: it is a required
    #    keyword, and anything that is not a Scaling is rejected by name.
    with pytest.raises(TypeError, match="scaling"):
        screen(G)
    with pytest.raises(TypeError, match="requires a Scaling"):
        screen(G, scaling={"u_span": [1.0, 1.0], "y_span": [1.0, 1.0]})

    # 4. Scaling.unscaled works but stamps the report.
    with pytest.warns(OperabilityWarning, match="UNSCALED"):
        rep = screen(G, scaling=Scaling.unscaled(2, 2))
    assert rep.scaled is False
    assert "unscaled" in [f.kind for f in rep.findings]
    assert rep.summary().startswith("!! UNSCALED")

    # An unscaled report must not claim anything about sigma_min.
    assert "weak_direction" not in [f.kind for f in rep.findings]


def test_scaling_validation_and_constructors():
    with pytest.raises(ValueError, match="strictly positive"):
        Scaling(u_span=[1.0, 0.0], y_span=[1.0])
    with pytest.raises(ValueError, match="finite"):
        Scaling(u_span=[1.0, np.inf], y_span=[1.0])

    sc = Scaling(u_span=[1.0, 2.0], y_span=[1.0])
    with pytest.raises(ValueError, match="Scaling is for"):
        sc.scale_gain(jnp.ones((2, 2)))
    with pytest.raises(ValueError, match="no d_span"):
        sc.scale_disturbance(jnp.ones((1, 1)))

    from_bounds = Scaling.from_bounds([0.0, 300.0], [1.0, 400.0],
                                      y_tol=[0.01], d_lb=[0.0], d_ub=[5.0])
    np.testing.assert_allclose(np.asarray(from_bounds.u_span), [1.0, 100.0])
    np.testing.assert_allclose(np.asarray(from_bounds.d_span), [5.0])
    assert from_bounds.explicit

    assert Scaling.unscaled(2, 3, 1).explicit is False
    assert Scaling.unscaled(2, 3, 1).caveat() is not None
    assert sc.caveat() is None


def test_scaling_round_trips_a_gain_matrix():
    G = jnp.array([[3.0, -1.0], [0.5, 2.0]])
    sc = Scaling(u_span=[10.0, 0.25], y_span=[2.0, 40.0])
    np.testing.assert_allclose(np.asarray(sc.unscale_gain(sc.scale_gain(G))),
                               np.asarray(G), atol=1e-12)


def test_scaling_from_block_uses_the_bounds():
    from difflow.planning import Block

    blk = Block(name="sep", fn=lambda u: jnp.array([u[0] * u[1]]),
                u_names=["recovery", "split"], y_names=["product"],
                lb=[0.0, 0.2], ub=[1.0, 0.8])
    sc = Scaling.from_block(blk, y_tol=[0.05])
    np.testing.assert_allclose(np.asarray(sc.u_span), [1.0, 0.6], atol=1e-12)

    unbounded = Block(name="open", fn=lambda u: jnp.array([u[0]]),
                      u_names=["x"], y_names=["y"])
    with pytest.raises(ValueError, match="unbounded"):
        Scaling.from_block(unbounded, y_tol=[1.0])


# --------------------------------------------------------------------------
# Disturbances
# --------------------------------------------------------------------------

def test_disturbance_condition_number_hits_its_known_bounds():
    """gamma_d = 1 along the strong direction, cond(G) along the weak one.

    ``sigma_max(G)`` is deliberately not 1 and the disturbance columns are
    deliberately not unit vectors, so the test fails if either the
    ``sigma_max`` factor or the normalisation of ``g_d`` is dropped -- both
    of which a plant with ``sigma_max = 1`` and unit-length ``g_d`` hides.
    """
    G = jnp.array([[5.0, 0.0], [0.0, 0.05]])       # sigma_max 5, cond 100
    assert float(condition_number(G, assume_scaled=True)) \
        == pytest.approx(100.0)

    strong = jnp.array([[3.0], [0.0]])             # norm 3, strong direction
    weak = jnp.array([[0.0], [7.0]])               # norm 7, weak direction
    assert float(disturbance_condition_number(G, strong,
                                              assume_scaled=True)[0]) \
        == pytest.approx(1.0)
    assert float(disturbance_condition_number(G, weak,
                                              assume_scaled=True)[0]) \
        == pytest.approx(100.0)

    # It is a measure of *direction*: scaling a disturbance column must not
    # change it, though it certainly changes the disturbance gain.
    both = jnp.array([[0.6], [0.8]])
    gamma = float(disturbance_condition_number(G, both, assume_scaled=True)[0])
    gamma_big = float(disturbance_condition_number(G, 1000.0 * both,
                                                   assume_scaled=True)[0])
    assert gamma_big == pytest.approx(gamma, rel=1e-10)
    assert 1.0 <= gamma <= float(condition_number(G, assume_scaled=True))

    # Hand value for the mixed direction: y_d = [0.6, 0.8],
    # pinv(G) y_d = [0.12, 16], |.| = 16.00045, gamma = 5 * 16.00045.
    assert gamma == pytest.approx(5.0 * np.linalg.norm([0.6 / 5.0, 0.8 / 0.05]))

    # A column of exact zeros is not a disturbance and is reported as 1,
    # not as nan.
    z = float(disturbance_condition_number(G, jnp.zeros((2, 1)),
                                           assume_scaled=True)[0])
    assert z == pytest.approx(1.0)


def test_strongly_directional_plant_is_flagged():
    """cond(G) > 10 raises `directional`, on its own, with no other finding."""
    sc = Scaling(u_span=[1.0, 1.0], y_span=[1.0, 1.0])

    # sigma_min = 1 exactly, so `weak_direction` cannot fire; only the
    # directionality is left to complain about.
    skewed = screen(jnp.array([[50.0, 0.0], [0.0, 1.0]]), scaling=sc)
    assert float(skewed.msv) == pytest.approx(1.0)
    assert float(skewed.cond) == pytest.approx(50.0)
    kinds = [f.kind for f in skewed.findings]
    assert kinds == ["directional"]
    assert skewed.findings[0].severity == "warning"
    assert "50" in skewed.findings[0].detail

    # Just under the threshold: nothing at all.
    even = screen(jnp.array([[9.0, 0.0], [0.0, 1.0]]), scaling=sc)
    assert float(even.cond) == pytest.approx(9.0)
    assert even.ok


def test_required_input_move_detects_a_disturbance_that_cannot_be_rejected():
    """The inputs do not span the direction the disturbance pushes."""
    # One input, two outputs pushed in different directions by one disturbance.
    G = jnp.array([[1.0], [0.0]])
    Gd = jnp.array([[0.0], [3.0]])
    sc = Scaling(u_span=[1.0], y_span=[1.0, 1.0], d_span=[1.0])
    rep = screen(G, Gd=Gd, scaling=sc, u_names=["valve"],
                 y_names=["T", "level"], d_names=["feed"])
    kinds = [f.kind for f in rep.findings]
    assert "disturbance_gain" in kinds        # |Gd| = 3 > 1
    # The single input can do nothing about the second output, so the least
    # squares move is zero -- the un-rejectable part is invisible here and is
    # caught by the underactuation finding instead.
    assert "underactuated" in kinds
    np.testing.assert_allclose(np.asarray(rep.u_required), [[0.0]], atol=1e-12)

    # Now a square plant that is simply too weak in the disturbed direction.
    G2 = jnp.array([[1.0, 0.0], [0.0, 0.1]])
    rep2 = screen(G2, Gd=jnp.array([[0.0], [1.0]]),
                  scaling=Scaling(u_span=[1.0, 1.0], y_span=[1.0, 1.0],
                                  d_span=[1.0]))
    need = np.asarray(rep2.u_required)
    assert need[1, 0] == pytest.approx(10.0)
    infeasible = [f for f in rep2.findings
                  if f.kind == "disturbance_infeasible"]
    assert len(infeasible) == 1
    assert infeasible[0].severity == "error"
    assert float(infeasible[0].value) == pytest.approx(10.0)


def test_a_feasible_disturbance_in_the_weak_direction_is_still_flagged():
    """The combination neither measure catches on its own.

    ``G = diag(100, 5)`` has plenty of gain, so the disturbance is rejectable
    with 40% of an input's travel and nothing is infeasible.  But it pushes
    along the direction the plant is 20x weaker in, so rejecting it uses the
    input combination the plant delivers least well -- a warning about
    sensitivity to model error, not about feasibility.
    """
    sc = Scaling(u_span=[1.0, 1.0], y_span=[1.0, 1.0], d_span=[1.0])
    rep = screen(jnp.diag(jnp.array([100.0, 5.0])),
                 Gd=jnp.array([[0.0], [2.0]]), scaling=sc, d_names=["feed_T"])

    assert float(rep.msv) == pytest.approx(5.0)          # not weak
    assert float(rep.dist_cond[0]) == pytest.approx(20.0)
    assert float(np.abs(np.asarray(rep.u_required)).max()) == pytest.approx(0.4)

    kinds = [f.kind for f in rep.findings]
    assert "disturbance_infeasible" not in kinds         # it *is* rejectable
    assert "disturbance_gain" in kinds                   # but not for free
    direction = [f for f in rep.findings
                 if f.kind == "disturbance_direction"]
    assert len(direction) == 1
    assert direction[0].variable == "feed_T"
    assert float(direction[0].value) == pytest.approx(20.0)

    # A disturbance of the same size along the *strong* direction raises the
    # gain warning and nothing about direction.
    aligned = screen(jnp.diag(jnp.array([100.0, 5.0])),
                     Gd=jnp.array([[2.0], [0.0]]), scaling=sc)
    assert float(aligned.dist_cond[0]) == pytest.approx(1.0)
    assert "disturbance_direction" not in [f.kind for f in aligned.findings]
    assert "disturbance_gain" in [f.kind for f in aligned.findings]


def test_small_disturbance_needs_no_control():
    """A disturbance inside the acceptable error raises nothing."""
    G = jnp.eye(2)
    Gd = jnp.array([[0.1], [0.05]])
    rep = screen(G, Gd=Gd, scaling=Scaling(u_span=[1.0, 1.0],
                                           y_span=[1.0, 1.0], d_span=[1.0]))
    assert rep.ok
    assert [f.kind for f in rep.findings] == []


def test_disturbance_gain_scaling_uses_d_span():
    Gd_raw = jnp.array([[3.0]])
    sc = Scaling(u_span=[1.0], y_span=[0.5], d_span=[2.0])
    np.testing.assert_allclose(
        np.asarray(disturbance_gain(Gd_raw, scaling=sc)), [[12.0]])


# --------------------------------------------------------------------------
# AD gains against central differences, on a real flowsheet
# --------------------------------------------------------------------------

def test_gain_matrix_matches_central_differences(flowsheet):
    """The AD gain of a CSTR + flash train, verified by finite differences.

    Both units contain an inner solve, so this is a test that implicit
    differentiation through them lands where a perturbation study would.
    """
    G_ad = np.asarray(gain_matrix(flowsheet, U0, D0))
    G_fd = _central_difference_jacobian(lambda u: flowsheet(u, D0), U0, 1e-4)

    assert G_ad.shape == (2, 2)
    scale = np.max(np.abs(G_ad))
    assert scale > 1e-6                       # the gains are not trivially 0
    assert np.max(np.abs(G_ad - G_fd)) / scale < 1e-6

    # And the plant really is coupled: an uncoupled one would make the rest
    # of this module vacuous here.
    assert np.abs(G_ad[0, 1]) / scale > 1e-3
    assert np.abs(G_ad[1, 0]) / scale > 1e-3


def test_disturbance_gain_matches_central_differences(flowsheet):
    Gd_ad = np.asarray(disturbance_gain(flowsheet, U0, D0))
    Gd_fd = _central_difference_jacobian(lambda d: flowsheet(U0, d), D0, 1e-5)
    assert Gd_ad.shape == (2, 1)
    assert np.max(np.abs(Gd_ad - Gd_fd)) / np.max(np.abs(Gd_ad)) < 1e-6


def test_scaled_gain_is_the_raw_gain_times_the_spans(flowsheet,
                                                     flowsheet_scaling):
    raw = np.asarray(gain_matrix(flowsheet, U0, D0))
    scaled = np.asarray(gain_matrix(flowsheet, U0, D0,
                                    scaling=flowsheet_scaling))
    expected = raw * np.array([10.0, 10.0])[None, :] \
        / np.array([0.2, 0.2])[:, None]
    np.testing.assert_allclose(scaled, expected, rtol=1e-12)


def test_screen_on_the_flowsheet_reports_a_coherent_picture(flowsheet,
                                                            flowsheet_scaling):
    rep = screen(flowsheet, U0, D0, scaling=flowsheet_scaling,
                 u_names=["T_reactor", "T_flash"],
                 y_names=["light_vapor", "heavy_liquid"],
                 d_names=["feed"])

    assert isinstance(rep, OperabilityReport)
    assert rep.G.shape == (2, 2) and rep.Gd.shape == (2, 1)
    assert rep.square and not rep.singular
    assert float(rep.rank) == 2.0

    # Internal consistency: the report's scalars are the metric functions
    # applied to its own scaled gain.
    assert float(rep.msv) == pytest.approx(
        float(min_singular_value(rep.G, assume_scaled=True)))
    assert float(rep.cond) == pytest.approx(
        float(condition_number(rep.G, assume_scaled=True)))
    np.testing.assert_allclose(np.asarray(rep.RGA),
                               np.asarray(rga(rep.G)), atol=1e-10)
    np.testing.assert_allclose(np.asarray(rep.u_required),
                               np.asarray(required_input_move(
                                   rep.G, rep.Gd, assume_scaled=True)),
                               atol=1e-10)
    np.testing.assert_allclose(np.asarray(rep.RGA).sum(axis=1),
                               [1.0, 1.0], atol=1e-8)

    text = rep.summary()
    assert "T_reactor" in text and "heavy_liquid" in text and "feed" in text
    assert "sigma_min" in text and "RGA" in text
    assert not text.startswith("!! UNSCALED")


def test_screen_verdict_responds_to_the_design(flowsheet_scaling):
    """A bigger reactor is a different plant, and the screen says so.

    This is the point of the module: the metric is a function of the design
    variables, so it can be compared across candidate designs.
    """
    small = screen(_make_flowsheet(volume=0.5), U0, D0,
                   scaling=flowsheet_scaling)
    large = screen(_make_flowsheet(volume=8.0), U0, D0,
                   scaling=flowsheet_scaling)
    assert float(small.msv) != pytest.approx(float(large.msv), rel=1e-3)


# --------------------------------------------------------------------------
# Traceability: the reason any of this can go inside a design loop
# --------------------------------------------------------------------------

def test_jit_matches_eager(flowsheet, flowsheet_scaling):
    def msv_of(u):
        return screen(flowsheet, u, D0, scaling=flowsheet_scaling).msv

    eager = float(msv_of(U0))
    jitted = float(jax.jit(msv_of)(U0))
    assert jitted == pytest.approx(eager, rel=1e-10)
    assert np.isfinite(eager) and eager > 0


def test_whole_screen_vmaps_over_operating_points(flowsheet,
                                                   flowsheet_scaling):
    """A screen per candidate design, in one batched call.

    ``screen`` returns a pytree, so ``vmap`` maps over the whole report and
    not just one scalar out of it -- which is what lets a design loop rank
    candidates without a Python loop over flowsheet solves.
    """
    U = jnp.stack([U0, U0 + jnp.array([5.0, 0.0]), U0 - jnp.array([0.0, 5.0])])

    def one(u):
        return screen(flowsheet, u, D0, scaling=flowsheet_scaling)

    batched = jax.jit(jax.vmap(one))(U)
    assert batched.G.shape == (3, 2, 2)
    assert batched.msv.shape == (3,)
    assert batched.u_required.shape == (3, 2, 1)
    for k in range(3):
        eager = one(U[k])
        assert float(batched.msv[k]) == pytest.approx(float(eager.msv),
                                                      rel=1e-9)
        np.testing.assert_allclose(np.asarray(batched.RGA[k]),
                                   np.asarray(eager.RGA), rtol=1e-9)
    # The three points are genuinely different plants, so the batch is not
    # accidentally broadcasting one answer.
    assert len(set(np.round(np.asarray(batched.msv), 6))) == 3


def test_forward_and_reverse_mode_agree_on_the_flowsheet(flowsheet):
    """AD mode is a cost choice, never an accuracy one."""
    fwd = np.asarray(gain_matrix(flowsheet, U0, D0, mode="fwd"))
    rev = np.asarray(gain_matrix(flowsheet, U0, D0, mode="rev"))
    auto = np.asarray(gain_matrix(flowsheet, U0, D0, mode="auto"))
    np.testing.assert_allclose(fwd, rev, rtol=1e-12)
    np.testing.assert_allclose(auto, rev, rtol=1e-12)


def test_disturbances_without_a_d_span_fail_loudly(flowsheet):
    """Asking for disturbance measures with no declared disturbance size."""
    no_d = Scaling(u_span=[10.0, 10.0], y_span=[0.2, 0.2])
    with pytest.raises(ValueError, match="no d_span"):
        screen(flowsheet, U0, D0, scaling=no_d)

    with pytest.raises(ValueError, match="both u0 and d0"):
        disturbance_gain(flowsheet, U0)


def test_metrics_are_jit_and_vmap_compatible():
    def metrics(u):
        G = gain_matrix(lambda x: jnp.array([x[0] + 0.9 * x[1],
                                             0.9 * x[0] + x[1] ** 2]), u,
                        scaling=Scaling(u_span=[1.0, 1.0], y_span=[1.0, 1.0]))
        return jnp.array([min_singular_value(G, assume_scaled=True),
                          condition_number(G, assume_scaled=True),
                          rga(G)[0, 0]])

    U = jnp.array([[0.3, 0.4], [1.0, 2.0], [-0.5, 0.7]])
    eager = np.stack([np.asarray(metrics(u)) for u in U])
    batched = np.asarray(jax.jit(jax.vmap(metrics))(U))
    np.testing.assert_allclose(batched, eager, rtol=1e-10)


def test_msv_is_differentiable_with_respect_to_a_design_variable():
    """sigma_min inside an objective: the whole motivation of issue #199."""
    sc = Scaling(u_span=[1.0, 1.0], y_span=[1.0, 1.0])

    def worst_direction_gain(design):
        def plant(u):
            return jnp.array([u[0] + design * u[1],
                              design * u[0] + u[1]])
        return screen(plant, jnp.zeros(2), scaling=sc).msv

    # G = [[1, a], [a, 1]] has singular values 1 + a and 1 - a, so
    # sigma_min = 1 - a and d(sigma_min)/da = -1 for 0 < a < 1.
    assert float(worst_direction_gain(0.4)) == pytest.approx(0.6)
    g = float(jax.grad(worst_direction_gain)(0.4))
    assert g == pytest.approx(-1.0, abs=1e-8)


def test_repeated_singular_values_are_documented_where_they_bite():
    """The two families of measures fail differently at a crossing.

    ``sigma_min`` and the condition number use ``compute_uv=False`` and keep
    a finite (one-sided) gradient.  Anything that needs the singular vectors
    -- ``pinv``, and so ``rga`` -- carries a ``1/(s_i^2 - s_j^2)`` term and
    returns nan.  That is a real trap for an objective built on the RGA, so
    it is pinned rather than left to be rediscovered.
    """
    def msv_of(a):
        return min_singular_value(jnp.array([[a, 0.0], [0.0, a]]),
                                  assume_scaled=True)

    def rga_of(a):
        return rga(jnp.array([[a, 0.0], [0.0, a]]))[0, 0]

    assert np.isfinite(float(jax.grad(msv_of)(2.0)))
    assert float(jax.grad(msv_of)(2.0)) == pytest.approx(1.0)
    assert np.isnan(float(jax.grad(rga_of)(2.0)))

    # Distinct singular values: both are fine.
    def msv_split(a):
        return min_singular_value(jnp.array([[a, 0.0], [0.0, 5.0]]),
                                  assume_scaled=True)

    def rga_split(a):
        return rga(jnp.array([[a, 0.3], [0.2, 5.0]]))[0, 0]

    assert float(jax.grad(msv_split)(1.0)) == pytest.approx(1.0)
    assert np.isfinite(float(jax.grad(rga_split)(1.0)))


def test_report_is_a_pytree_that_round_trips():
    rep = screen(jnp.array([[1.0, 0.2], [0.3, 1.0]]),
                 Gd=jnp.array([[0.5], [0.1]]),
                 scaling=Scaling(u_span=[1.0, 1.0], y_span=[1.0, 1.0],
                                 d_span=[1.0]),
                 u_names=["a", "b"], y_names=["p", "q"], d_names=["z"])
    leaves, treedef = jax.tree_util.tree_flatten(rep)
    assert len(leaves) == 11        # Gd, dist_cond and u_required all present
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert back.u_names == ("a", "b") and back.d_names == ("z",)
    assert float(back.msv) == pytest.approx(float(rep.msv))

    doubled = jax.tree_util.tree_map(lambda x: 2.0 * x, rep)
    np.testing.assert_allclose(np.asarray(doubled.G),
                               2.0 * np.asarray(rep.G), atol=1e-12)


def test_findings_refuse_to_be_evaluated_on_tracers():
    sc = Scaling(u_span=[1.0, 1.0], y_span=[1.0, 1.0])

    @jax.jit
    def bad(u):
        rep = screen(lambda x: jnp.array([x[0], x[1]]), u, scaling=sc)
        return len(rep.findings)

    with pytest.raises(ValueError, match="tracers"):
        bad(jnp.zeros(2))


def test_screen_rejects_inconsistent_arguments():
    sc = Scaling(u_span=[1.0, 1.0], y_span=[1.0, 1.0])
    G = jnp.eye(2)
    with pytest.raises(ValueError, match="expected 2 u names"):
        screen(G, scaling=sc, u_names=["only_one"])
    with pytest.raises(ValueError, match="pairing"):
        screen(G, scaling=sc, pairing=[0, 5])
    with pytest.raises(ValueError, match="Gd is for the matrix form"):
        screen(lambda u: u, jnp.zeros(2), scaling=sc, Gd=jnp.zeros((2, 1)))
