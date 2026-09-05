"""Tests for the REE separation-train superstructure layer (#202).

Two limits of ``FullSeparationTrain`` are lifted here, and the tests are
built to discriminate between them being lifted and merely being claimed:

1. The topology is data. Two trains over the *same* module instances,
   differing only in their connectivity map, must give different and
   individually correct answers.
2. The organic loop is closed. It must converge, conserve every
   component to machine precision around the loop, and --- the central
   claim of #202 --- the residual loading an imperfectly stripped solvent
   carries must **measurably degrade** the raffinate purity relative to
   the open-loop assumption that the same circuit is fed fresh solvent.

Plus the two supporting handles: operating boundaries as inequality
constraints an optimizer can see, and the Fenske bound as a cheap
admissibility filter.
"""

import math

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

pytest.importorskip("difflow_ree")

from difflow.streams import get_flows, make_stream  # noqa: E402
from difflow_ree.equilibrium.distribution import (  # noqa: E402
    stages_fenske,
    stages_kremser,
)
from difflow_ree.flowsheets import (  # noqa: E402
    CeriumOxidationModule,
    ExtractScrubStripModule,
    ExtractScrubStripParams,
    OperatingLimits,
    PortMismatchError,
    PrecipitationModule,
    SaponificationModule,
    SeparationTrain,
    SolventRegenerationParams,
    SplitShellModule,
    SplitShellParams,
    TopologyError,
    list_modules,
    screen_separation,
    screen_train,
)
from difflow_ree.units.cerium import CeriumOxidizerParams  # noqa: E402
from difflow_ree.units.precipitation import PrecipitatorParams  # noqa: E402
from difflow_ree.units.saponification import SaponifierParams  # noqa: E402

# ---------------------------------------------------------------------
# The reference separation
#
# La / Dy on D2EHPA. Dy is extracted, La stays in the raffinate, so the
# raffinate is the light product and its purity is limited by Dy
# leakage -- which is exactly the quantity residual organic loading
# controls (#202). The stripping section is deliberately under-designed
# (one stage at pH 2.3), because a *perfectly* stripped circuit is the
# open-loop assumption and would make the comparison vacuous.
# ---------------------------------------------------------------------

ELEMENTS = ("La", "Dy")


def _feed(la: float = 3.0, dy: float = 3.0, water: float = 100.0):
    """The aqueous leach liquor used throughout.

    Args:
        la: La molar flow (mol/s).
        dy: Dy molar flow (mol/s).
        water: Water molar flow (mol/s).

    Returns:
        The stream.
    """
    return make_stream(
        {"H2O": water, "La": la, "Dy": dy}, 298.15, 101325.0
    )


def _ess_params(**overrides) -> ExtractScrubStripParams:
    """Reference extract-scrub-strip parameters.

    Args:
        **overrides: Fields to change.

    Returns:
        The parameters.
    """
    base = dict(
        extractant="D2EHPA",
        elements=ELEMENTS,
        target_elements=("Dy",),
        n_extraction_stages=10,
        n_scrubbing_stages=2,
        n_stripping_stages=1,
        extraction_pH=2.4,
        scrubbing_pH=2.3,
        stripping_pH=2.3,
        solvent_to_feed_ratio=1.0,
        scrub_to_solvent_ratio=0.1,
        strip_to_solvent_ratio=0.5,
    )
    base.update(overrides)
    return ExtractScrubStripParams(**base)


def _ess_module(
    name: str = "sep",
    limits: OperatingLimits | None = None,
    **overrides,
) -> ExtractScrubStripModule:
    """The reference module.

    Args:
        name: Module name.
        limits: Operating boundaries; None uses a 0.65 third-phase limit.
        **overrides: Parameter overrides.

    Returns:
        The module.
    """
    return ExtractScrubStripModule(
        name,
        _ess_params(**overrides),
        limits=limits or OperatingLimits(third_phase_loading=0.65),
    )


def _closed_train(module: ExtractScrubStripModule, feed=None) -> SeparationTrain:
    """A train whose only edge is the closed organic loop.

    Args:
        module: The circuit module.
        feed: Aqueous feed; None uses :func:`_feed`.

    Returns:
        The train.
    """
    train = SeparationTrain("closed")
    train.add_module(module)
    train.add_feed("leach", feed if feed is not None else _feed(),
                   f"{module.name}.feed")
    train.connect(f"{module.name}.barren_organic", f"{module.name}.solvent")
    return train


def _purity(stream, key: str, basis=ELEMENTS) -> float:
    """Mole fraction of ``key`` within the REE it shares a stream with.

    Args:
        stream: The stream.
        key: The element.
        basis: The elements the fraction is taken over.

    Returns:
        The mole fraction, or 0.0 for an empty stream.
    """
    flows = get_flows(stream)
    total = sum(float(flows.get(e, 0.0)) for e in basis)
    return float(flows.get(key, 0.0)) / total if total > 0 else 0.0


def _flow(stream, key: str) -> float:
    """One molar flow out of a stream.

    Args:
        stream: The stream.
        key: The species.

    Returns:
        The flow, 0.0 when absent.
    """
    return float(get_flows(stream).get(key, 0.0))


# =====================================================================
# 1. Typed ports: connectivity is checked, not assumed
# =====================================================================


def test_organic_outlet_into_an_aqueous_inlet_is_rejected():
    """The two liquid phases are the same Python type; the port isn't."""
    train = SeparationTrain("t")
    train.add_module(_ess_module())

    with pytest.raises(PortMismatchError, match="phase mismatch"):
        train.connect("sep.barren_organic", "sep.feed")

    with pytest.raises(PortMismatchError, match="phase mismatch"):
        train.connect("sep.raffinate", "sep.solvent")

    # ... and the failed connections left no edge behind.
    assert train.connections == []


def test_solid_outlet_cannot_feed_a_liquid_contactor():
    """A precipitate is a stream too, and must not be piped into a mixer."""
    train = SeparationTrain("t")
    train.add_module(_ess_module())
    train.add_module(PrecipitationModule(
        "precip", PrecipitatorParams(elements=ELEMENTS)
    ))
    with pytest.raises(PortMismatchError, match="phase mismatch"):
        train.connect("precip.solid", "sep.feed")


def test_species_loss_is_refused_unless_it_is_asked_for():
    """Dropping a component silently is a mass-balance error, so it raises."""
    train = SeparationTrain("t")
    train.add_module(_ess_module("wide"))
    train.add_module(ExtractScrubStripModule(
        "narrow",
        _ess_params(elements=("La",), target_elements=("La",)),
    ))
    with pytest.raises(PortMismatchError, match="Dy"):
        train.connect("wide.raffinate", "narrow.feed")

    # The escape hatch exists and is explicit.
    train.connect("wide.raffinate", "narrow.feed", allow_species_loss=True)
    assert len(train.connections) == 1


def test_an_inlet_takes_exactly_one_source():
    """Two sources on one inlet is a modelling error, not a mixer."""
    train = SeparationTrain("t")
    train.add_module(_ess_module())
    train.add_feed("leach", _feed(), "sep.feed")
    with pytest.raises(TopologyError, match="already has a source"):
        train.add_feed("second", _feed(), "sep.feed")


def test_an_unconnected_inlet_is_the_open_loop_and_is_rejected():
    """A dangling organic inlet is exactly the bug #202 is about."""
    train = SeparationTrain("t")
    train.add_module(_ess_module())
    train.add_feed("leach", _feed(), "sep.feed")
    with pytest.raises(TopologyError, match="sep.solvent.*no source"):
        train.validate()


# =====================================================================
# 2. The module library and its schema
# =====================================================================


def test_library_carries_the_five_modules_the_issue_names():
    """Extract-scrub-strip, split-shell, Ce oxidation, precipitation, sap."""
    assert set(list_modules()) == {
        "extract_scrub_strip",
        "split_shell",
        "cerium_oxidation",
        "precipitation",
        "saponification",
    }


def test_every_module_declares_typed_ports():
    """Each module in the library declares phases on both directions."""
    modules = [
        _ess_module(),
        SplitShellModule("shell", SplitShellParams(
            extractant="D2EHPA", elements=("La", "Nd", "Dy"),
            n_stages=12, split_points=(4, 8),
        )),
        CeriumOxidationModule("ce", CeriumOxidizerParams(
            elements=("La", "Ce", "Dy"))),
        PrecipitationModule("pr", PrecipitatorParams(elements=ELEMENTS)),
        SaponificationModule("sap", SaponifierParams(
            extractant="D2EHPA", elements=ELEMENTS)),
    ]
    for module in modules:
        assert module.ports.inlets, module.kind
        assert module.ports.outlets, module.kind
        for port in module.ports.inlets + module.ports.outlets:
            assert port.phase in ("aqueous", "organic", "solid")

    # The circuit's two phases are distinguished, which is the whole point.
    ess = modules[0]
    assert ess.ports.inlet("feed").phase == "aqueous"
    assert ess.ports.inlet("solvent").phase == "organic"
    assert ess.ports.outlet("barren_organic").phase == "organic"
    assert ess.ports.outlet("raffinate").phase == "aqueous"
    # And the solid products cannot be mistaken for liquids.
    assert modules[2].ports.outlet("ceo2").phase == "solid"
    assert modules[3].ports.outlet("solid").phase == "solid"


def test_ports_are_data_not_code():
    """Split-shell port count follows its split points, at runtime."""
    two = SplitShellModule("a", SplitShellParams(
        extractant="D2EHPA", elements=("La", "Nd", "Dy"),
        n_stages=12, split_points=(4, 8)))
    three = SplitShellModule("b", SplitShellParams(
        extractant="D2EHPA", elements=("La", "Nd", "Dy"),
        n_stages=16, split_points=(4, 8, 12)))
    assert len(two.ports.outlets) == 4      # 3 side-draws + raffinate
    assert len(three.ports.outlets) == 5    # 4 side-draws + raffinate


def test_describe_delegates_the_parameter_schema_to_difflow_catalog():
    """Parameters are derived once, in the core catalog, not restated."""
    from difflow.catalog import describe_class
    from difflow_ree.units.extraction import REEExtractor

    schema = _ess_module().describe()
    assert [op["name"] for op in schema["operations"]] == [
        "REEExtractor", "REEScrubber", "REEStripper"
    ]
    extractor = schema["operations"][0]
    assert extractor["params_class"] == "REEExtractorParams"
    # Identical to what the core catalog reports on its own.
    assert extractor == describe_class(
        REEExtractor, plugin="difflow_ree"
    ).to_dict()
    # The phase typing is the part the core cannot derive from a
    # signature, and it is present.
    phases = {p["name"]: p["phase"] for p in schema["ports"]["inlets"]}
    assert phases == {"feed": "aqueous", "solvent": "organic"}


# =====================================================================
# 3. Topology is data: same modules, different connectivity
# =====================================================================


@pytest.mark.slow
def test_two_topologies_over_the_same_modules_give_different_correct_answers():
    """Connectivity is a decision variable, not source code."""
    elements = ("La", "Nd", "Dy")
    feed = make_stream(
        {"H2O": 100.0, "La": 2.0, "Nd": 2.0, "Dy": 2.0}, 298.15, 101325.0
    )

    def circuit(name, target, ext_pH, strip_pH):
        return ExtractScrubStripModule(name, ExtractScrubStripParams(
            extractant="D2EHPA", elements=elements, target_elements=target,
            n_extraction_stages=8, n_scrubbing_stages=2,
            n_stripping_stages=1, extraction_pH=ext_pH,
            scrubbing_pH=ext_pH - 0.1, stripping_pH=strip_pH,
            solvent_to_feed_ratio=1.0, scrub_to_solvent_ratio=0.1,
            strip_to_solvent_ratio=0.5,
        ))

    def build(order):
        train = SeparationTrain(order)
        train.add_module(circuit("A", ("Dy",), 2.4, 2.3))
        train.add_module(circuit("B", ("Nd",), 3.3, 3.1))
        train.connect("A.barren_organic", "A.solvent")
        train.connect("B.barren_organic", "B.solvent")
        if order == "AB":
            train.add_feed("leach", feed, "A.feed")
            train.connect("A.raffinate", "B.feed")
            terminal = ["A.product", "A.scrub_liquor",
                        "B.product", "B.scrub_liquor", "B.raffinate"]
        else:
            train.add_feed("leach", feed, "B.feed")
            train.connect("B.raffinate", "A.feed")
            terminal = ["B.product", "B.scrub_liquor",
                        "A.product", "A.scrub_liquor", "A.raffinate"]
        return train, terminal

    results = {}
    for order in ("AB", "BA"):
        train, terminal = build(order)
        assert train.validate() == list(order)
        result = train.solve()
        assert result.converged
        # Each topology is individually correct: everything fed leaves.
        for element in elements:
            out = sum(_flow(result.stream(ref), element) for ref in terminal)
            assert out == pytest.approx(_flow(feed, element), rel=1e-9)
        results[order] = result

    # And the two topologies genuinely differ. Feeding the Dy circuit
    # first leaves the Nd circuit a Dy-depleted feed, so its product is
    # not the same product.
    dy_in_b_product = {
        order: _flow(results[order].stream("B.product"), "Dy")
        for order in ("AB", "BA")
    }
    assert dy_in_b_product["BA"] > 1.3 * dy_in_b_product["AB"]
    assert _flow(results["AB"].stream("A.product"), "Dy") != pytest.approx(
        _flow(results["BA"].stream("A.product"), "Dy"), rel=1e-6
    )


def test_train_json_round_trip_reproduces_the_solution():
    """A topology can be written, read back, and solved to the same answer."""
    train = _closed_train(_ess_module())
    clone = SeparationTrain.from_json(train.to_json())

    assert list(clone.modules) == list(train.modules)
    assert [(c.source, c.dest) for c in clone.connections] == \
           [(c.source, c.dest) for c in train.connections]

    original = train.solve()
    restored = clone.solve()
    for element in ELEMENTS:
        assert _flow(restored.stream("sep.raffinate"), element) == \
            pytest.approx(_flow(original.stream("sep.raffinate"), element),
                          rel=1e-12)


def test_round_trip_refuses_an_unknown_format_version():
    """The train file borrows difflow.serialize's version check."""
    from difflow.serialize import SerializationError

    data = _closed_train(_ess_module()).to_dict()
    data["format_version"] = 999
    with pytest.raises(SerializationError, match="format version"):
        SeparationTrain.from_dict(data)


# =====================================================================
# 4. The closed organic loop
# =====================================================================


def test_closed_loop_converges_and_conserves_every_component():
    """Nothing accumulates or vanishes around the torn organic loop."""
    feed = _feed()
    train = _closed_train(_ess_module(), feed)
    result = train.solve()

    assert result.converged
    assert result.residual < 1e-10
    assert result.tear_streams == ("sep.solvent.tear",)

    # Component balance: the aqueous outlets carry everything fed.
    for element in ELEMENTS:
        out = sum(
            _flow(result.stream(f"sep.{port}"), element)
            for port in ("raffinate", "scrub_liquor", "product")
        )
        assert out == pytest.approx(_flow(feed, element), abs=1e-9, rel=1e-11)

    # The tear is genuinely converged: what leaves the stripper is what
    # enters the extractor, component by component, including the
    # organic carriers.
    barren = get_flows(result.stream("sep.barren_organic"))
    torn = get_flows(result.stream("sep.solvent.tear"))
    for key, value in barren.items():
        assert float(value) == pytest.approx(float(torn[key]), abs=1e-9)

    # The solvent inventory is conserved exactly: an organic loop that
    # loses extractant would converge just as happily to nonsense.
    assert float(barren["D2EHPA"]) == pytest.approx(50.0, rel=1e-12)
    assert float(barren["kerosene"]) == pytest.approx(100.0, rel=1e-12)


def test_imperfect_stripping_degrades_raffinate_purity():
    """The central claim of #202, quantified.

    The open loop feeds the extraction section fresh, REE-free solvent,
    which is the assumption that stripping is perfect. Closing the loop
    returns the solvent the stripper actually produces; the Dy it still
    carries consumes free extractant, the extraction factor falls, and
    more Dy leaks into the La raffinate.
    """
    feed = _feed()
    module = _ess_module()

    # Open loop: exactly what ExtractScrubStripCircuit does today.
    open_raffinate = module(feed, module.fresh_solvent(feed))[0]
    open_purity = _purity(open_raffinate, "La")
    open_impurity = _flow(open_raffinate, "Dy")

    # Closed loop: the same module, one extra edge.
    result = _closed_train(module, feed).solve()
    assert result.converged
    closed_raffinate = result.stream("sep.raffinate")
    closed_purity = _purity(closed_raffinate, "La")
    closed_impurity = _flow(closed_raffinate, "Dy")

    # The recycled solvent is measurably loaded -- this is the mechanism.
    residual = _flow(result.stream("sep.barren_organic"), "Dy")
    assert residual > 0.5, "stripping must be imperfect for the test to bite"
    assert float(result.info["sep"]["extraction"]["theta_solvent"]) > 0.1

    # The consequence: worse raffinate purity, by a wide margin.
    assert closed_purity < open_purity
    assert open_purity - closed_purity > 0.04
    assert closed_impurity > 5.0 * open_impurity

    # Pinned so a change of mechanism cannot pass unnoticed.
    #
    # The closed-loop figures moved from 0.94137 / 6.37 when the extractor's
    # capacity limiter was corrected to saturate the *total* organic loading
    # rather than only the increment (#207 review): the recycled solvent enters
    # loaded, and bounding it properly leaks slightly more Dy. The open-loop
    # value is unchanged to every digit, because a REE-free solvent makes the
    # corrected expression identical to the previous one -- which is the
    # cleanest evidence that the correction touches only the loaded case.
    assert open_purity == pytest.approx(0.99032, abs=2e-4)
    assert closed_purity == pytest.approx(0.93952, abs=2e-4)
    assert closed_impurity / open_impurity == pytest.approx(6.59, rel=0.02)


def test_the_degradation_comes_from_the_residue_not_from_the_loop():
    """Control: with stripping near-complete, closing the loop changes nothing.

    Without this the previous test could be passing on an artefact of
    tearing rather than on the physics it claims to demonstrate.
    """
    feed = _feed()
    module = _ess_module(stripping_pH=1.5, n_stripping_stages=4)

    open_raffinate = module(feed, module.fresh_solvent(feed))[0]
    result = _closed_train(module, feed).solve()
    closed_raffinate = result.stream("sep.raffinate")

    assert _flow(result.stream("sep.barren_organic"), "Dy") < 1e-6
    assert _purity(closed_raffinate, "La") == pytest.approx(
        _purity(open_raffinate, "La"), abs=1e-9
    )


def test_regeneration_bleed_removes_the_accumulated_residue():
    """Saponification with solvent regeneration (#197) closes the loop physically.

    A loop with no bleed accumulates whatever the stripper misses; the
    bleed is what stops it, and turning it up must reduce the residual
    loading the extraction section sees.
    """
    feed = _feed()
    loadings = {}
    for bleed in (0.0, 0.05, 0.20):
        train = SeparationTrain("sap")
        train.add_module(_ess_module())
        train.add_module(SaponificationModule(
            "sap",
            SaponifierParams(extractant="D2EHPA", elements=ELEMENTS,
                             saponification_degree=0.2),
            regeneration=SolventRegenerationParams(bleed_fraction=bleed),
        ))
        train.add_feed("leach", feed, "sep.feed")
        train.connect("sep.barren_organic", "sap.organic")
        train.connect("sap.organic", "sep.solvent", allow_species_loss=True)
        assert train.validate() == ["sep", "sap"]

        result = train.solve()
        assert result.converged
        loadings[bleed] = float(
            result.info["sep"]["extraction"]["theta_solvent"]
        )

        # Everything fed still leaves, now including through the bleed.
        for element in ELEMENTS:
            out = sum(
                _flow(result.stream(ref), element)
                for ref in ("sep.raffinate", "sep.scrub_liquor",
                            "sep.product", "sap.bleed")
            )
            assert out == pytest.approx(_flow(feed, element),
                                        abs=1e-9, rel=1e-11)

        # The bleed conserves the solvent inventory: make-up replaces it.
        organic = get_flows(result.stream("sap.organic"))
        assert float(organic["D2EHPA"]) == pytest.approx(50.0, rel=1e-12)
        assert float(organic["kerosene"]) == pytest.approx(100.0, rel=1e-12)

    assert loadings[0.20] < loadings[0.05] < loadings[0.0]
    assert loadings[0.20] < 0.75 * loadings[0.0]


# =====================================================================
# 5. Operating boundaries as constraints
# =====================================================================


def test_constraints_are_numbers_an_optimizer_can_use():
    """Not info flags: a signed margin vector, feasible when >= 0."""
    train = _closed_train(_ess_module())
    result = train.solve()
    constraints = train.constraints(result)

    vector = constraints.vector()
    assert vector.shape == (len(constraints),)
    assert jnp.all(jnp.isfinite(vector))
    assert set(constraints.names) >= {"sep.third_phase", "sep.loading"}
    assert constraints.feasible
    assert float(constraints["sep.third_phase"].margin) > 0.0

    # Both loading margins are driven by the loading of the organic
    # *outlet*, which is the stream that would split, not by the loading
    # of the solvent entering.
    theta = float(result.info["sep"]["extraction"]["theta_total"])
    assert float(constraints["sep.third_phase"].value) == pytest.approx(theta)
    assert float(constraints["sep.loading"].value) == pytest.approx(theta)
    assert theta > float(
        result.info["sep"]["extraction"]["theta_solvent"]
    )


def test_a_design_past_third_phase_onset_is_reported_as_violating():
    """The optimizer must be told it crossed, not merely that it is close.

    And note which design crosses: the open-loop one sits inside the
    limit, and only the *closed* loop --- the honest one --- crosses it.
    An open loop hides the constraint violation as well as the purity
    loss (#202).
    """
    feed = _feed()
    limits = OperatingLimits(third_phase_loading=0.40)
    module = _ess_module(limits=limits)

    open_info = module(feed, module.fresh_solvent(feed))[4]
    open_constraints = module.constraints(open_info)
    assert open_constraints.feasible
    assert float(open_constraints["sep.third_phase"].margin) > 0.0

    train = _closed_train(_ess_module(limits=limits), feed)
    result = train.solve()
    closed = train.constraints(result)

    assert not closed.feasible
    violated = closed.violations()
    assert [c.qualified_name for c in violated] == ["sep.third_phase"]
    assert violated[0].violated is True
    assert float(violated[0].margin) < 0.0
    assert float(violated[0].value) > float(violated[0].limit)
    assert closed.worst().qualified_name == "sep.third_phase"
    assert "VIOLATED" in closed.summary()
    # The saturation constraint is a *different* boundary and still holds,
    # so a design cannot satisfy one by ignoring the other.
    assert float(closed["sep.loading"].margin) > 0.0


def test_hydraulic_and_phase_ratio_limits_are_reported_when_declared():
    """Undeclared limits are absent, not invented."""
    feed = _feed()
    bare = _ess_module(limits=OperatingLimits(
        third_phase_loading=None, max_loading=None
    ))
    assert len(bare.constraints(bare(feed, bare.fresh_solvent(feed))[4])) == 0

    full = _ess_module(limits=OperatingLimits(
        third_phase_loading=0.65, max_loading=1.0,
        hydraulic_capacity=200.0, min_phase_ratio=0.2, max_phase_ratio=5.0,
    ))
    constraints = full.constraints(full(feed, full.fresh_solvent(feed))[4])
    assert set(constraints.names) == {
        "sep.third_phase", "sep.loading", "sep.hydraulic",
        "sep.phase_ratio_min", "sep.phase_ratio_max",
    }
    # 250 mol/s of two-phase throughput against a 200 mol/s settler.
    assert float(constraints["sep.hydraulic"].margin) < 0.0
    assert not constraints.feasible


def test_a_constraint_margin_is_differentiable():
    """An optimizer needs the derivative of the boundary, not just its sign."""
    feed = _feed()
    module = _ess_module()

    def margin(organic_dy):
        """Third-phase margin as a function of the recycle's Dy loading."""
        solvent = dict(module.fresh_solvent(feed))
        solvent["F_Dy"] = jnp.asarray(organic_dy, dtype=jnp.float64)
        return module.constraints(module(feed, solvent)[4])[
            "sep.third_phase"
        ].margin

    value = margin(0.5)
    gradient = jax.grad(margin)(0.5)
    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient)
    # More REE already on the entering solvent means a higher loading and
    # so less room before the third phase.
    assert float(gradient) < 0.0


# =====================================================================
# 6. Screening before costing: the Fenske bound
# =====================================================================


def test_stages_fenske_matches_hand_arithmetic():
    """Three cases computed by hand from N_min = ln[odds product]/ln(alpha)."""
    # alpha = 100, both products 99% pure:
    #   N_min = ln(99 * 99) / ln(100) = 2 ln(99) / ln(100)
    assert float(stages_fenske(100.0, 0.99, 0.99)) == pytest.approx(
        2 * math.log(99) / math.log(100)
    )
    assert float(stages_fenske(100.0, 0.99, 0.99)) == pytest.approx(
        1.9956352, abs=1e-6
    )
    # alpha = 2, both products 99.9% pure: 2 ln(999) / ln(2)
    assert float(stages_fenske(2.0, 0.999, 0.999)) == pytest.approx(
        2 * math.log(999) / math.log(2)
    )
    assert float(stages_fenske(2.0, 0.999, 0.999)) == pytest.approx(
        19.9286817, abs=1e-6
    )
    # A very easy separation needs less than one theoretical stage, and
    # the bound says so rather than being floored at 1 the way
    # stages_kremser is: ln(81) / ln(1000).
    assert float(stages_fenske(1000.0, 0.9, 0.9)) == pytest.approx(
        math.log(81) / math.log(1000)
    )
    assert float(stages_fenske(1000.0, 0.9, 0.9)) == pytest.approx(
        0.6361617, abs=1e-6
    )
    assert float(stages_fenske(1000.0, 0.9, 0.9)) < 1.0


def test_stages_fenske_is_unordered_in_its_key_pair():
    """alpha and 1/alpha describe the same separation."""
    assert float(stages_fenske(20.0, 0.98, 0.95)) == pytest.approx(
        float(stages_fenske(1 / 20.0, 0.98, 0.95))
    )
    # alpha == 1 separates nothing at any stage count.
    assert math.isinf(float(stages_fenske(1.0, 0.99, 0.99)))


def _kremser_fraction_remaining(E: float, N: float) -> float:
    """Fraction of a component left in the raffinate.

    Written out here rather than imported, so the bound is checked
    against arithmetic this test owns.

    Args:
        E: Extraction factor ``D * S/F``.
        N: Number of stages.

    Returns:
        The raffinate fraction.
    """
    if abs(E - 1.0) < 1e-9:
        return 1.0 / (N + 1.0)
    return (E - 1.0) / (E ** (N + 1.0) - 1.0)


def test_stages_fenske_is_a_true_lower_bound_on_stages_kremser():
    """Across a sweep of splits and separation factors.

    For each case: ask Kremser how many stages the extract key needs for
    its target recovery, see what the raffinate key does at that stage
    count, and check the Fenske minimum for the split those two actually
    achieved never exceeds the stage count that achieved it.
    """
    worst_slack = math.inf
    cases = 0
    for D_A in (0.5, 1.0, 3.0, 10.0, 100.0):
        for alpha in (1.5, 2.0, 5.0, 20.0, 500.0):
            for S_F in (0.5, 1.0, 2.0):
                for recovery in (0.9, 0.99, 0.999, 0.9999):
                    E_A = D_A * S_F
                    if E_A <= 1.05:
                        continue  # unreachable at any stage count
                    N = float(stages_kremser(D_A, S_F, recovery))
                    r_A = 1.0 - _kremser_fraction_remaining(E_A, N)
                    r_B = 1.0 - _kremser_fraction_remaining(E_A / alpha, N)
                    n_min = float(stages_fenske(alpha, r_A, 1.0 - r_B))
                    assert n_min <= N + 1e-9, (
                        f"Fenske {n_min} exceeded Kremser {N} at "
                        f"D_A={D_A}, alpha={alpha}, S/F={S_F}, "
                        f"recovery={recovery}"
                    )
                    worst_slack = min(worst_slack, N - n_min)
                    cases += 1
    assert cases > 100
    # The bound is tight somewhere in the sweep, so it is a bound and not
    # merely a small number.
    assert worst_slack < 1e-6


def test_fenske_screen_rejects_a_topology_before_it_is_costed():
    """The filter must actually reject, and reject for the right reason."""
    # Two scrub stages are enough for a 99% Dy/La split on D2EHPA ...
    easy = screen_separation(
        "D2EHPA", "Dy", "La", installed_stages=2, purity=0.99, pH=2.3
    )
    assert easy.admissible
    assert easy.separation_factor > 100

    # ... and not enough for 99.9%.
    hard = screen_separation(
        "D2EHPA", "Dy", "La", installed_stages=2, purity=0.999, pH=2.3
    )
    assert not hard.admissible
    assert hard.minimum_stages > hard.installed_stages
    assert "unreachable" in hard.reason

    # The neighbouring-lanthanide split is the one that really costs
    # stages, and the bound says so without solving anything.
    neighbours = screen_separation(
        "D2EHPA", "Nd", "Pr", installed_stages=20, purity=0.99, pH=3.0
    )
    assert neighbours.minimum_stages > 5.0

    # And it works through a train.
    train = _closed_train(_ess_module())
    report = screen_train(train, {"sep": ("Dy", "La")}, purity=0.999)
    assert not report.admissible
    assert [v.name for v in report.rejected()] == ["sep"]
    assert "REJECTED" in report.summary()


# =====================================================================
# 7. Differentiability through the closed loop
# =====================================================================


def test_jit_and_grad_through_a_closed_loop_train_are_finite():
    """The loop is torn implicitly, so the whole train stays differentiable."""
    feed = _feed()

    def raffinate_impurity(strip_ratio):
        module = ExtractScrubStripModule(
            "sep", _ess_params(strip_to_solvent_ratio=strip_ratio)
        )
        train = SeparationTrain("g")
        train.add_module(module)
        train.add_feed("leach", feed, "sep.feed")
        train.connect("sep.barren_organic", "sep.solvent")
        streams = train.solve_differentiable()
        return get_flows(streams["sep.raffinate"])["Dy"]

    eager = float(raffinate_impurity(0.5))
    compiled = float(jax.jit(raffinate_impurity)(0.5))
    gradient = float(jax.grad(raffinate_impurity)(0.5))

    assert math.isfinite(eager) and eager > 0.0
    assert compiled == pytest.approx(eager, rel=1e-10)
    assert math.isfinite(gradient)
    # More strip solution strips the solvent harder, so less Dy comes
    # back on the recycle and less leaks into the raffinate.
    assert gradient < 0.0

    # The traceable solve agrees with the reported one; it is the same
    # graph, the same units and the same tear set.
    reported = _closed_train(_ess_module()).solve()
    assert eager == pytest.approx(
        _flow(reported.stream("sep.raffinate"), "Dy"), rel=1e-8
    )


# =====================================================================
# Review findings on PR #207 -- each fix pinned so it cannot regress
# =====================================================================

class TestReviewFindings207:
    """Mass-balance and topology defects found reviewing #202, and the
    pre-existing extractor capacity defect the closed loop newly exposed.

    Each of these was measured on the branch before it was fixed; the
    numbers in the docstrings are what the broken code actually produced.
    """

    def test_split_shell_does_not_replicate_the_organic_carrier(self):
        """Each organic side-draw got the *whole* inlet carrier, so a module
        fed 2.0 mol/s of D2EHPA emitted 6.0 across three product ports --
        3x the extractant inventory created from nothing, compounding every
        pass around a closed loop.
        """
        from difflow_ree.flowsheets.modules import SplitShellModule
        from difflow_ree.flowsheets.split_shell import SplitShellParams

        mod = SplitShellModule("shell", SplitShellParams(
            extractant="D2EHPA", elements=("La", "Nd", "Dy"),
            n_stages=12, split_points=(4, 8)))
        solvent_in = {"D2EHPA": 2.0, "kerosene": 18.0}
        out = mod(
            make_stream({"H2O": 100.0, "La": 1.0, "Nd": 1.0, "Dy": 1.0},
                        298.15, 101325.0),
            make_stream(solvent_in, 298.15, 101325.0),
        )
        n_organic_ports = len(out) - 2
        assert n_organic_ports == 3, "test needs more than one organic draw"

        totals: dict[str, float] = {}
        for stream in out[:-1]:
            for key, value in get_flows(stream).items():
                totals[key] = totals.get(key, 0.0) + float(value)
        for carrier, fed in solvent_in.items():
            assert totals[carrier] == pytest.approx(fed, rel=1e-12), (
                f"{carrier} not conserved across the side-draws"
            )

    def test_regeneration_bleed_partitions_the_inlet(self):
        """`kept + bled` exceeded the inlet by b*(1-eta)*v, manufacturing
        0.2% of the loop's REE inventory per pass at b=0.02, eta=0.9 -- a
        slow drift the tear solve converges to happily. The bleed must also
        equal what the module reports as removed.
        """
        from difflow_ree.flowsheets.modules import (
            SaponificationModule, SolventRegenerationParams,
        )
        from difflow_ree.units.saponification import SaponifierParams

        b, eta = 0.02, 0.9
        mod = SaponificationModule(
            "sap",
            SaponifierParams(extractant="D2EHPA", elements=("La", "Nd")),
            regeneration=SolventRegenerationParams(
                bleed_fraction=b, regeneration_efficiency=eta),
        )
        organic = make_stream(
            {"D2EHPA": 2.0, "kerosene": 18.0, "La": 0.5, "Nd": 0.5},
            298.15, 101325.0)
        out, _spent, bleed, info = mod(organic)

        f_in, f_out, f_bleed = (
            get_flows(organic), get_flows(out), get_flows(bleed))
        for elem in ("La", "Nd"):
            total_out = float(f_out.get(elem, 0.0)) + float(f_bleed.get(elem, 0.0))
            assert total_out == pytest.approx(float(f_in[elem]), rel=1e-12), (
                f"{elem} is created or destroyed by the bleed"
            )
        # ...and the reported removal is the amount that actually leaves.
        leaving = sum(float(f_bleed.get(e, 0.0)) for e in ("La", "Nd"))
        assert leaving == pytest.approx(
            float(info["ree_removed_by_regeneration"]), rel=1e-12)

    def test_two_tears_sharing_one_source_are_refused(self):
        """`Flowsheet.add_recycle` keys recycles by source, so a second tear
        from the same outlet silently overwrote the first: the dropped tear
        kept its seed value, was never iterated, and `solve` still reported
        convergence. Refused rather than converging on an unsolved tear.
        """
        from difflow_ree.flowsheets.train import SeparationTrain, TopologyError

        a = _ess_module("a")
        b = _ess_module("b")
        train = SeparationTrain("shared_tear")
        # b is added first, so the DFS opens b, descends into a, and both of
        # a's organic edges below close back onto an open node -- two genuine
        # tears sharing one source, which is the case the guard exists for.
        train.add_module(b)
        train.add_module(a)
        train.add_feed("leach", _feed(), "b.feed")
        train.connect("b.raffinate", "a.feed")
        # One outlet torn into two inlets: add_recycle keys recycles by source,
        # so the second would overwrite the first and its tear go unsolved
        # while solve() still reported convergence.
        train.connect("a.barren_organic", "a.solvent")
        train.connect("a.barren_organic", "b.solvent")
        with pytest.raises(TopologyError, match=r"feeds both"):
            train.to_flowsheet()


def test_capacity_bounds_total_organic_loading_not_just_the_increment():
    """Pre-existing in the extractor, newly exposed by the closed loop.

    The limiter scaled only what the section newly extracted and then added
    the REE already on the entering solvent back on top, so with a loaded
    solvent theta_total reached 2.18 while capacity_scale still read 0.78 --
    the limiter appearing to bind while the organic carried 2.18x the
    extractant inventory. An open circuit always fed REE-free solvent, so it
    never showed.
    """
    from difflow_ree.units.extraction import REEExtractor, REEExtractorParams
    from difflow_ree.database import get_extractant

    m = get_extractant("D2EHPA").monomers_per_ree
    extractor = REEExtractor(REEExtractorParams(
        n_stages=5, extractant="D2EHPA", elements=("Nd", "Dy"), pH=3.0))
    feed = make_stream({"H2O": 10.0, "Nd": 0.3, "Dy": 0.3}, 298.15, 101325.0)
    capacity = 1.0 / m

    loaded = make_stream(
        {"D2EHPA": 1.0, "kerosene": 5.0, "Nd": 0.10, "Dy": 0.10},
        298.15, 101325.0)
    raffinate, extract, info = extractor(feed, loaded)

    organic_ree = sum(float(get_flows(extract)[e]) for e in ("Nd", "Dy"))
    assert organic_ree <= capacity + 1e-12, (
        f"organic carries {organic_ree / capacity:.2f}x the extractant capacity"
    )
    assert float(info["theta_total"]) <= 1.0 + 1e-9

    # Mass balance survives the rejection of the excess to the aqueous.
    f_feed, f_solv = get_flows(feed), get_flows(loaded)
    f_raff, f_ext = get_flows(raffinate), get_flows(extract)
    for elem in ("Nd", "Dy"):
        assert float(f_raff[elem]) + float(f_ext[elem]) == pytest.approx(
            float(f_feed[elem]) + float(f_solv.get(elem, 0.0)), rel=1e-12)

    # With REE-free solvent the corrected expression is the previous one.
    fresh = make_stream({"D2EHPA": 1.0, "kerosene": 5.0}, 298.15, 101325.0)
    fresh_organic = sum(
        float(get_flows(extractor(feed, fresh)[1])[e]) for e in ("Nd", "Dy"))
    assert fresh_organic <= capacity + 1e-12
