"""C5/C6 light-naphtha isomerization (#311): species, reactor, unit, planning block.

Per commit: the species table against the database, the reaction network's
atom balance, the feeds, one reactor pass on each feed (balances, energy,
positivity, the integrator's own residual), one once-through unit solve
(the whole-unit balances), the hydrogen-starvation warning, and the
planning block's wiring into a gasoline pool.

``release`` (and ``slow`` where it solves the unit repeatedly): the
equilibrium trends the issue names (2,2-DMB falls with T; RON has a
maximum in the inlet temperature), the second feed, and the once-through
implicit gradients against central differences.

The DIH recycle is in ``test_isomerization_dih.py`` and
``test_isomerization_dih_gradients.py`` (release; each a file of its own so
that ``--dist loadfile`` can put them on separate workers). The IDAES
cross-check is in ``test_isomerization_validation.py`` (release) and
``test_isomerization_validation_file.py`` (per commit).
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow.database import get_critical_props
from difflow.planning import Network
from difflow_refinery.blending import BlendComponent, BlendPool
from difflow_refinery.isomerization import (
    OUTPUT_NAMES,
    IsomerizationHydrogenWarning,
    IsomerizationReactor,
    IsomerizationReactorParams,
    IsomerizationUnit,
    IsomerizationUnitParams,
    constructed_feed,
    family_equilibrium,
    isom_block,
    link_isom,
    nc6_fraction,
    with_nc6_fraction,
)
from difflow_refinery.isomerization import thermochem as tc
from difflow_refinery.isomerization.feed import SPECIATIONS, hydrocarbon_flows
from difflow_refinery.isomerization.plant import MIN_H2_HC_OUT
from difflow_refinery.isomerization.reactor import REACTIONS, assert_balanced, flows_of

jax.config.update("jax_enable_x64", True)

FEEDS = ("paraffinic", "benzene_rich")
T_BASE = 413.15


def charge(kind, H2_HC=0.3, T=T_BASE, P=30e5):
    """Fresh feed plus hydrogen at ``H2_HC``, at reactor inlet conditions."""
    f = constructed_feed(kind, 10.0)
    F = hydrocarbon_flows(f)
    F = F.at[tc.idx("hydrogen")].set(H2_HC * jnp.sum(F))
    return F, T, P


def idx(name):
    return OUTPUT_NAMES.index(name)


# -- per commit ---------------------------------------------------------------


class TestSpecies:
    def test_formulae_match_the_database(self):
        tc.check_formulae()

    def test_every_species_is_in_the_database(self):
        for n in tc.NAMES:
            assert get_critical_props(n).MW > 0, n

    def test_every_reaction_balances_carbon_and_hydrogen(self):
        assert_balanced()

    @pytest.mark.parametrize("family", sorted(tc.FAMILIES))
    def test_a_family_equilibrium_is_a_distribution(self, family):
        x = family_equilibrium(family, 450.0)
        assert set(x) == set(tc.FAMILIES[family])
        assert sum(float(v) for v in x.values()) == pytest.approx(1.0, abs=1e-14)
        assert all(float(v) > 0 for v in x.values())

    def test_octanes_are_flagged_for_verification(self):
        """The pure-component octanes are not from a primary source this
        module can cite page by page; every species says so."""
        assert all(s.note for s in tc.SPECIES)


class TestFeeds:
    @pytest.mark.parametrize("kind", FEEDS)
    def test_the_feed_has_its_mass_rate(self, kind):
        F = hydrocarbon_flows(constructed_feed(kind, 10.0))
        assert float(jnp.dot(F, tc.MW)) / 1e3 == pytest.approx(10.0, rel=1e-12)

    def test_the_benzene_rich_feed_is_benzene_rich(self):
        def w_bz(kind):
            return dict(SPECIATIONS[kind].whole_mass)["benzene"]
        assert w_bz("benzene_rich") > 3 * w_bz("paraffinic")

    def test_with_nc6_fraction_sets_it_and_keeps_the_moles(self):
        f = constructed_feed("paraffinic", 10.0)
        g = with_nc6_fraction(f, 0.2)
        assert float(nc6_fraction(g)) == pytest.approx(0.2, abs=1e-14)
        assert float(jnp.sum(hydrocarbon_flows(g))) == pytest.approx(
            float(jnp.sum(hydrocarbon_flows(f))), rel=1e-14)

    def test_an_unknown_species_is_refused(self):
        with pytest.raises(ValueError, match="not isomerization species"):
            flows_of({"F_toluene": 1.0, "T": 400.0, "P": 1e5})


class TestParams:
    def test_an_unknown_configuration_is_refused(self):
        with pytest.raises(ValueError, match="configuration"):
            IsomerizationUnitParams(configuration="tip")

    def test_an_unknown_catalyst_is_refused(self):
        with pytest.raises(ValueError, match="catalyst"):
            IsomerizationReactorParams(catalyst="platinum_on_wishes")


@pytest.fixture(scope="module", params=FEEDS)
def reactor_pass(request):
    F0, T0, P = charge(request.param)
    rx = IsomerizationReactor(IsomerizationReactorParams())
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # no convergence warning on the base case
        out = rx.run(F0, T0, P=P)
    return request.param, F0, T0, out


class TestReactorPass:
    def test_mass_atoms_and_carbon_numbers_close(self, reactor_pass):
        _, _, _, out = reactor_pass
        for k, v in out["info"]["balance"].items():
            assert abs(float(v)) < 1e-12, k

    def test_the_bed_is_adiabatic(self, reactor_pass):
        _, F0, T0, out = reactor_pass
        h_in = float(jnp.dot(F0, tc.enthalpy(T0)))
        h_out = float(jnp.dot(out["F"], tc.enthalpy(out["T"])))
        assert h_out == pytest.approx(h_in, rel=1e-10, abs=1e-6 * abs(h_in))

    def test_the_reactions_are_exothermic(self, reactor_pass):
        _, _, T0, out = reactor_pass
        assert float(out["T"]) > T0

    def test_no_flow_goes_negative(self, reactor_pass):
        _, F0, _, out = reactor_pass
        assert float(jnp.min(out["F"])) >= -1e-12 * float(jnp.sum(F0))

    def test_every_implicit_stage_converged(self, reactor_pass):
        _, _, _, out = reactor_pass
        assert float(out["info"]["stage_residual"]) < 1e-9

    def test_the_approach_to_equilibrium_is_reported(self, reactor_pass):
        """Not bounded by one: in an adiabatic bed isomer made upstream is
        past the equilibrium of the hotter outlet (benzene saturation, the
        fastest, ends at about 1.016), and the reverse reaction is what
        pulls it back."""
        _, _, _, out = reactor_pass
        a = out["info"]["approach"]
        assert set(a) == {r.name for r in REACTIONS if r.product}
        for name, v in a.items():
            assert 0.0 < float(v) < 1.05, name

    def test_benzene_is_saturated(self, reactor_pass):
        _, _, _, out = reactor_pass
        assert float(out["info"]["benzene_conversion"]) > 0.99


class TestHydrogenWarning:
    """The warning is the reactor's word that hydrocracking has run away;
    it must fire below the threshold and stay quiet above it."""

    def test_it_fires_when_the_outlet_is_starved(self):
        with pytest.warns(IsomerizationHydrogenWarning, match="hydrogen-starved"):
            IsomerizationUnit._check_hydrogen({"reactor": {"H2_HC_out": 0.5 * MIN_H2_HC_OUT}})

    def test_it_is_quiet_otherwise(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            IsomerizationUnit._check_hydrogen({"reactor": {"H2_HC_out": 2 * MIN_H2_HC_OUT}})

    def test_a_nan_counts_as_starved(self):
        with pytest.warns(IsomerizationHydrogenWarning):
            IsomerizationUnit._check_hydrogen({"reactor": {"H2_HC_out": float("nan")}})


@pytest.fixture(scope="module")
def once_through():
    unit = IsomerizationUnit(IsomerizationUnitParams())
    feed = constructed_feed("paraffinic", 10.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", IsomerizationHydrogenWarning)
        iso, gas, info = unit(feed)
    return unit, feed, iso, gas, info


class TestOnceThrough:
    def test_the_whole_unit_closes(self, once_through):
        unit, feed, _, _, info = once_through
        res = {"streams": info["streams"], "info": info}
        b = unit.balances(feed, res)
        assert set(b) >= {"mass", "C5", "C6"}
        for k, v in b.items():
            assert abs(float(v)) < 1e-8, k

    def test_the_separator_converged(self, once_through):
        info = once_through[-1]
        assert bool(info["separator"]["converged"])

    def test_every_output_is_finite(self, once_through):
        out = once_through[-1]["outputs"]
        assert set(out) == set(OUTPUT_NAMES)
        for k, v in out.items():
            assert np.isfinite(float(v)), k

    def test_the_isomerate_meets_its_rvp_spec(self, once_through):
        unit, _, _, _, info = once_through
        assert float(info["outputs"]["RVP"]) == pytest.approx(unit.params.stabilizer_rvp, rel=1e-6)

    def test_it_has_no_column_duties(self, once_through):
        out = once_through[-1]["outputs"]
        assert float(out["dih_duty"]) == 0.0 and float(out["dip_duty"]) == 0.0


class TestPlanningBlock:
    """Wiring only: the block's names, units and bounds, and its link into a
    gasoline pool. Its delta vectors are a release check below."""

    def test_names_units_bounds(self, once_through):
        unit, feed, *_ = once_through
        blk = isom_block(unit, feed, levers=("T_in", "LHSV", "x_nC6"),
                         outputs=("isomerate_V", "RON", "dih_duty"),
                         bounds={"T_in": (120.0, 160.0)})
        assert blk.u_names == ["T_in", "LHSV", "x_nC6"]
        assert blk.y_names == ["isomerate_V", "RON", "dih_duty"]
        assert blk.metadata["u_units"] == ["C", "1/h", "-"]
        assert blk.metadata["y_units"] == ["m3/h", "-", "MW"]
        assert list(blk.lb)[0] == 120.0 and list(blk.ub)[0] == 160.0
        assert blk.u0[0] == pytest.approx(T_BASE - 273.15)

    def test_bad_levers_outputs_and_bounds_are_refused(self, once_through):
        unit, feed, *_ = once_through
        with pytest.raises(ValueError, match="unknown levers"):
            isom_block(unit, feed, levers=("pressure",))
        with pytest.raises(ValueError, match="unknown outputs"):
            isom_block(unit, feed, outputs=("octane",))
        with pytest.raises(ValueError, match="not levers"):
            isom_block(unit, feed, levers=("T_in",), bounds={"LHSV": (1.0, 3.0)})
        with pytest.raises(ValueError, match="outside its bounds"):
            isom_block(unit, feed, bounds={"T_in": (150.0, 160.0)})

    def test_it_links_to_a_gasoline_pool(self, once_through):
        unit, feed, _, _, info = once_through
        blk = isom_block(unit, feed)
        reformate = BlendComponent.from_properties(
            "reformate", SG=0.82, RON=98.0, MON=88.0, RVP_psi=4.0, olefins_vol=0.0,
            aromatics_vol=60.0, benzene_vol=1.0)
        pool = BlendPool("gasoline").as_block(
            [unit.blend_component(info["outputs"]), reformate],
            u0=[float(info["outputs"]["isomerate_volume"]), 50.0])
        links = link_isom(blk, pool)
        assert links == [("isom.isomerate_V", "gasoline.isomerate_V")]
        Network([blk, pool], links)

    def test_a_pool_without_an_isomerate_lever_is_refused(self, once_through):
        unit, feed, *_ = once_through
        blk = isom_block(unit, feed)
        reformate = BlendComponent.from_properties(
            "reformate", SG=0.82, RON=98.0, MON=88.0, RVP_psi=4.0, olefins_vol=0.0,
            aromatics_vol=60.0, benzene_vol=1.0)
        pool = BlendPool("gasoline").as_block([reformate])
        with pytest.raises(ValueError, match="isomerate_V"):
            link_isom(blk, pool)

    def test_the_isomerate_blends(self, once_through):
        unit, _, _, _, info = once_through
        c = unit.blend_component(info["outputs"])
        assert float(c.properties["RON"]) == pytest.approx(float(info["outputs"]["RON"]))


# -- release ------------------------------------------------------------------


@pytest.mark.release
class TestEquilibriumTrends:
    def test_22dmb_falls_with_temperature(self):
        x = [float(family_equilibrium("C6P", T)["2_2_dimethylbutane"])
             for T in np.linspace(380.0, 560.0, 10)]
        assert all(b < a for a, b in zip(x, x[1:]))

    def test_isopentane_falls_with_temperature(self):
        x = [float(family_equilibrium("C5", T)["isopentane"])
             for T in np.linspace(380.0, 560.0, 10)]
        assert all(b < a for a, b in zip(x, x[1:]))

    def test_the_reactor_reaches_the_closed_form_at_high_activity(self):
        """Isothermal, no cracking, a very active bed: the C6 paraffins at
        the outlet are the closed-form family equilibrium."""
        F0, _, P = charge("paraffinic")
        T = 450.0
        rx = IsomerizationReactor(IsomerizationReactorParams(
            adiabatic=False, k_scale=1e4, crack_scale=0.0, n_steps=40, LHSV=1.0))
        F = rx.run(F0, T, P=P)["F"]
        names = tc.FAMILIES["C6P"]
        tot = sum(float(F[tc.idx(n)]) for n in names)
        x = family_equilibrium("C6P", T)
        for n in names:
            assert float(F[tc.idx(n)]) / tot == pytest.approx(float(x[n]), abs=1e-9), n


@pytest.mark.release
@pytest.mark.slow
class TestSecondFeed:
    def test_the_benzene_rich_feed_closes(self):
        unit = IsomerizationUnit(IsomerizationUnitParams())
        feed = constructed_feed("benzene_rich", 10.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", IsomerizationHydrogenWarning)
            _, _, info = unit(feed)
        assert bool(info["separator"]["converged"])
        b = unit.balances(feed, {"streams": info["streams"], "info": info})
        for k, v in b.items():
            assert abs(float(v)) < 1e-8, k
        assert float(info["outputs"]["benzene_conversion"]) > 0.99


@pytest.mark.release
@pytest.mark.slow
class TestRONMaximum:
    """Equilibrium favours the branched isomers at low T and kinetics
    favour high T; the product octane has a maximum between. The scan
    points sit inside the window where hydrocracking stays bounded (the
    benzene-rich feed runs away above about 155 C at LHSV 2 and H2/HC 0.3;
    see the docs)."""

    @pytest.mark.parametrize("kind,temps", [("paraffinic", (110.0, 150.0, 190.0)),
                                            ("benzene_rich", (100.0, 120.0, 150.0))])
    def test_ron_has_an_interior_maximum(self, kind, temps):
        unit = IsomerizationUnit(IsomerizationUnitParams())
        feed = constructed_feed(kind, 10.0)
        ron = [float(unit.outputs(feed, T + 273.15, 2.0)[idx("RON")]) for T in temps]
        assert ron[1] > ron[0] and ron[1] > ron[2], ron


# Not RVP, H2_makeup or stabilizer_c4_recovery: the stabilizer holds the
# RVP at its spec, a once-through unit's make-up is its charge hydrogen, and
# the benzene-rich feed's butane recovery sits at a bound; their derivatives
# are zero and a comparison of them compares round-off.
GRAD_OUTPUTS = ("RON", "MON", "yield_volume", "H2_consumption", "gas_make")
#: Central-difference steps: T_in (K), LHSV (1/h), x_nC6 (-). Small enough
#: that the O(h^2) truncation is under 1e-5 relative on every entry (the
#: tightest is the benzene-rich volume yield against LHSV, a derivative of
#: about 1e-5: 4e-4 relative at h = 2e-3, 4e-6 at 2e-4).
FD_STEPS = (0.02, 2e-4, 3e-5)


@pytest.mark.release
@pytest.mark.slow
class TestOnceThroughGradients:
    """jacfwd through the reactor's implicit stages and the separator and
    stabilizer's implicit solves, against central differences."""

    @pytest.mark.parametrize("kind", FEEDS)
    def test_gradients_match_central_differences(self, kind):
        unit = IsomerizationUnit(IsomerizationUnitParams())
        feed = constructed_feed(kind, 10.0)
        base = jnp.array([T_BASE, 2.0, float(nc6_fraction(feed))])

        def f(v):
            return unit.outputs(feed, v[0], v[1], x_nc6=v[2])

        J = jax.jacfwd(f)(base)
        for j, h in enumerate(FD_STEPS):
            e = jnp.zeros(3).at[j].set(h)
            fd = (f(base + e) - f(base - e)) / (2 * h)
            for n in GRAD_OUTPUTS:
                a, b = float(J[idx(n), j]), float(fd[idx(n)])
                assert b != 0.0, (j, n)
                assert abs(a - b) <= 1e-5 * abs(b), (j, n, a, b)
