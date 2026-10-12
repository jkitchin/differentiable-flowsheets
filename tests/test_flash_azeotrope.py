"""Tests for the activity-coefficient (gamma-phi) flash path (issue #90).

Demonstrates that with an NRTL liquid model the flash reproduces
non-ideal VLE, including the ethanol-water azeotrope, which the ideal
Raoult's-law path cannot.

NRTL ethanol(1)-water(2) parameters (tau_ij = a_ij + b_ij/T, alpha = 0.3,
in the ``NRTLParams`` orientation: a[0, 1] = a_12 = -0.8009) are the widely
used Aspen/DECHEMA set (Gmehling, J., Onken, U.,
Vapor-Liquid Equilibrium Data Collection, DECHEMA, 1977). Antoine
coefficients are the NIST log10(P[bar]) form shifted by +5 to give Pa.
"""

import pytest
import jax
import jax.numpy as jnp

from difflow.thermo import IdealThermo, SpeciesData
from difflow.units.flash import Flash, FlashParams
from difflow.units.lle import nrtl_activity_coefficients, NRTLParams
from difflow.streams import make_stream, get_flows

jax.config.update("jax_enable_x64", True)


def _thermo():
    ethanol = SpeciesData(
        name="ethanol", MW=46.07, Cp_coeffs=(0, 0, 0, 0),
        Hvap_coeffs=(1.0, 0.38, 514.0), antoine_coeffs=(10.24677, 1598.673, -46.424),
    )
    water = SpeciesData(
        name="water", MW=18.02, Cp_coeffs=(0, 0, 0, 0),
        Hvap_coeffs=(1.0, 0.38, 647.0), antoine_coeffs=(9.6543, 1435.264, -64.848),
    )
    return IdealThermo({"ethanol": ethanol, "water": water})


def _nrtl():
    return NRTLParams(
        species=("ethanol", "water"),
        a=jnp.array([[0.0, -0.8009], [3.4578, 0.0]]),
        b=jnp.array([[0.0, 246.18], [-586.0809, 0.0]]),
        alpha=jnp.array([[0.0, 0.3], [0.3, 0.0]]),
    )


class TestNRTLRelativeVolatility:
    """The NRTL model produces an azeotrope; ideal Raoult does not."""

    def test_relative_volatility_crosses_unity(self):
        th, nrtl = _thermo(), _nrtl()
        T, P = 351.4, 101325.0
        K_ideal = jnp.array([th.Psat("ethanol", T), th.Psat("water", T)]) / P

        def alpha(x1):
            x = jnp.array([x1, 1 - x1])
            g = nrtl_activity_coefficients(x, T, nrtl)
            K = g * K_ideal
            return float(K[0] / K[1])

        # Ethanol-rich liquid: water salts out, alpha drops below 1 (azeotrope)
        assert alpha(0.2) > 1.0
        assert alpha(0.9) < 1.0
        # Ideal relative volatility is constant and never crosses 1
        ideal_alpha = float(K_ideal[0] / K_ideal[1])
        assert ideal_alpha > 1.0


class TestNRTLOrientation:
    """``_nrtl()`` is in the ``NRTLParams`` orientation (issue #444): the
    transposed matrices put the 5 mol% ethanol bubble point near 95 C and
    the azeotrope at 76.7 C."""

    @pytest.mark.parametrize("x1, T_meas_C, tol", [
        (0.05, 90.2, 1.0),     # dilute ethanol (about 90.2 C measured)
        (0.894, 78.15, 0.2),   # azeotrope at 1 atm
    ])
    def test_bubble_point(self, x1, T_meas_C, tol):
        from scipy.optimize import brentq

        th, nrtl = _thermo(), _nrtl()
        x = jnp.array([x1, 1 - x1])

        def resid(T):
            psat = jnp.array([th.Psat("ethanol", T), th.Psat("water", T)])
            return float(jnp.sum(x * nrtl_activity_coefficients(x, T, nrtl) * psat)) - 101325.0

        T_b = brentq(resid, 330.0, 380.0)
        assert T_b - 273.15 == pytest.approx(T_meas_C, abs=tol)


@pytest.mark.slow
class TestGammaPhiFlash:
    """Flash with an NRTL activity model (#90)."""

    def _feed(self, x_etoh):
        return make_stream({"ethanol": x_etoh, "water": 1 - x_etoh}, T=351.4, P=101325.0)

    def test_flash_runs_and_conserves_mass(self):
        flash = Flash(FlashParams(species_order=["ethanol", "water"]),
                      thermo=_thermo(), activity_model=_nrtl())
        feed = self._feed(0.5)
        liquid, vapor, info = flash(feed, T=351.4, P=101325.0)
        lf, vf = get_flows(liquid), get_flows(vapor)
        for s in ("ethanol", "water"):
            total = float(lf.get(s, 0.0)) + float(vf.get(s, 0.0))
            assert total == pytest.approx(float(get_flows(feed)[s]), rel=1e-6)

    def test_azeotrope_pinch_vs_ideal(self):
        """Near the azeotrope the gamma flash barely separates; ideal does."""
        params = FlashParams(species_order=["ethanol", "water"])
        gamma_flash = Flash(params, thermo=_thermo(), activity_model=_nrtl())
        ideal_flash = Flash(params, thermo=_thermo())

        feed = self._feed(0.85)  # near the ethanol-water azeotrope
        _, _, ig = gamma_flash(feed, T=351.4, P=101325.0)
        _, _, ii = ideal_flash(feed, T=351.4, P=101325.0)

        # x and y are close for the azeotropic (gamma) system...
        gap_gamma = abs(float(ig["x"]["ethanol"]) - float(ig["y"]["ethanol"]))
        # ...but the ideal model shows a large ethanol enrichment in vapor.
        gap_ideal = abs(float(ii["x"]["ethanol"]) - float(ii["y"]["ethanol"]))
        assert gap_gamma < gap_ideal

    def test_gamma_changes_k_values(self):
        """The activity path yields different K-values than ideal Raoult."""
        params = FlashParams(species_order=["ethanol", "water"])
        _, _, ig = Flash(params, thermo=_thermo(), activity_model=_nrtl())(
            self._feed(0.3), T=351.4, P=101325.0)
        _, _, ii = Flash(params, thermo=_thermo())(
            self._feed(0.3), T=351.4, P=101325.0)
        assert float(ig["K"]["ethanol"]) != pytest.approx(float(ii["K"]["ethanol"]), rel=1e-3)


class TestDiluteVolatileFeed:
    """Rachford-Rice Newton must not leave [0, 1] for a dilute volatile feed.

    5 mol% ethanol in water at 1 atm, NRTL in the orientation that puts the
    feed's bubble point near 364 K (experiment: about 362 K). With K_ethanol
    near 8, Newton from the interpolated guess overshot past the pole at
    V = 1/(1 - K) and the flash reported V ~ 1e-5 at 366.5-369.5 K, well
    inside the two-phase region (bubble and dew sums both above 1.1). The
    root is now bracketed by bisection on [0, 1] before Newton polishes it.
    """

    def _flash(self):
        nrtl = NRTLParams(
            species=("ethanol", "water"),
            a=jnp.array([[0.0, -0.8009], [3.4578, 0.0]]),
            b=jnp.array([[0.0, 246.18], [-586.0809, 0.0]]),
            alpha=jnp.array([[0.0, 0.3], [0.3, 0.0]]),
        )
        return Flash(FlashParams(species_order=["ethanol", "water"]),
                     thermo=_thermo(), activity_model=nrtl)

    def test_two_phase_split_is_found(self):
        flash = self._flash()
        feed = make_stream({"ethanol": 0.05, "water": 0.95}, T=360.0, P=101325.0)
        Vs = []
        for T in (365.5, 366.5, 367.5, 368.5):
            _, _, info = flash(feed, T=T, P=101325.0)
            V = float(info["V_frac"])
            K = float(info["K"]["ethanol"])
            x = float(info["x"]["ethanol"])
            # the returned split satisfies its own Rachford-Rice balance
            assert x == pytest.approx(0.05 / (1.0 + V * (K - 1.0)), rel=1e-3)
            Vs.append(V)
        assert Vs[2] > 0.1                      # was 1.4e-4 at 367.5 K
        assert all(b > a for a, b in zip(Vs, Vs[1:]))   # V rises with T

    @pytest.mark.slow
    def test_gradient_matches_finite_difference(self):
        flash = self._flash()
        feed = make_stream({"ethanol": 0.05, "water": 0.95}, T=360.0, P=101325.0)

        def V(T):
            return flash(feed, T=T, P=101325.0)[2]["V_frac"]

        g = float(jax.grad(V)(367.5))
        fd = float((V(367.51) - V(367.49)) / 0.02)
        assert g == pytest.approx(fd, rel=1e-3)
