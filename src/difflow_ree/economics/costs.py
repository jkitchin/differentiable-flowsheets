"""Cost estimation for REE separation processes.

Provides economic analysis including:
- REE product pricing (volatile market)
- Reagent and utility costs
- Capital and operating cost estimation
- Profitability metrics

All functions support JAX arrays for differentiable optimization.

PROVENANCE -- READ THIS BEFORE REPORTING A NUMBER FROM THIS MODULE
==================================================================

Most factors in this file are ESTIMATED in the sense of
``difflow_ree/data/sources.yaml:EST``: an indicative order of magnitude with
no identified source. The exception is the capital cost, which is anchored
to three published projects (``CAPEX_ANCHORS``, tagged ``DISCLOSED``);
:func:`capex_basis` reports which one an estimate came from, what it
enclosed, and the AACE accuracy range that follows. Specifically:

* **Prices** (:attr:`REEPricing.base_prices`) are the same indicative oxide
  values as ``elements.yaml:*.price_usd_kg`` (tagged ``EST`` there, with the
  warning that REE prices move by factors of several within a year and are
  not public for the thin-market elements). ``purity_premium`` and
  ``form_factors`` are invented multipliers, not observed premia. Nothing
  here knows about payability: a real offtake pays a *fraction* of contained
  oxide value, and for La/Ce that fraction is small because the market is in
  surplus.
* **CAPEX** is the one thing here with a citable basis.
  :func:`estimate_capex` takes its *level* from a disclosed project cost --
  Energy Fuels White Mesa Phase 1 as-built, Avalon's Geismar separation
  refinery PFS, or the Energy Fuels Phase 2 bankable study -- and moves it
  to the requested capacity by the 0.6 power law and to the requested year
  by CEPCI. **The scope you ask for is the whole estimate**: those three
  anchors differ by more than 10x at the same capacity because one is
  solvent extraction retrofitted into an existing licensed mill and another
  is a standalone refinery with its own buildings, civils, utilities and
  effluent treatment. Pass ``scope=`` deliberately and read
  :func:`capex_basis` before quoting a number.

  Scaling one project to another capacity is an AACE Class 5 method
  (capacity-factored, single-point analogy) however well defined the anchor
  was, so the honest accuracy on the output is about -50 % / +100 % --
  ``capex_basis(...)["derived_accuracy"]``. And the stage count moves only
  the stage-driven fraction of capital, and only if you give it a base case
  to move from (``n_stages_reference``); by default it does nothing,
  because no anchor discloses a stage count.

  The *section breakdown* is ESTIMATED: it splits the anchored total by
  conventional section shares that no anchor published in full. One
  section IS published, and it is what keeps the split honest -- Avalon
  discloses its solvent-extraction circuit, "over 1,000 mixer-settlers",
  at 33 % of total capital, US$101 million. difflow's split brackets that
  figure (mixer-settlers alone below it, the whole SX equipment group about
  a quarter above) and ``tests/ree/test_capex_anchors.py`` holds it there.
  That same figure prices a stage at no more than about US$101,000 in 2011
  dollars, near US$142,000 in 2024 -- worth knowing before accepting any
  per-stage price.

  Anchoring rather than factoring is a deliberate choice. A factored
  estimate needs a priced equipment list this package does not have, and
  it fails quietly when it is short: a per-stage equipment price and a
  handful of percentage add-ons can look like a complete estimate while
  applying a total multiplier of 2 to delivered equipment where
  fluid-processing practice is Lang 4.74 (``sources.yaml:LANG_FLUID``), and
  while enclosing only the extraction bay. If you do build a factored
  estimate here, check the multiplier and write down the battery limits.
* **OPEX unit rates** -- $2/kg REE extractant makeup, $1/kg acid, $0.5/kg
  base, $3/kg precipitant, 3 % of CAPEX for maintenance, the utility
  correlations -- are placeholders chosen for plausibility. They are also
  *per kg of REE*, which hides the fact that reagent consumption in a
  cation-exchange circuit is set by the acid/base equivalents the pH swing
  demands, not by the mass of product.

  There is exactly one published rate to check a total against: Avalon's
  study gives US$5,634 per tonne of separated REO at 10,000 t/yr, of which
  reagents are 70 % at US$3,934/t, and that covers labour, supplies,
  reagents and maintenance but no capital charge and no feed cost
  (``sources.yaml:AVALON_GEISMAR``). It is a rate for a large plant on a
  mixed feed, so it bounds an order of magnitude rather than validating a
  line item -- but a total that comes out several times away from it, once
  scale is allowed for, is telling you something.
* **Labor** is a headcount times an hourly rate. No shift schedule, no
  overhead burden, no supervision.

What this module IS good for: differentiating a cost through a flowsheet,
comparing two designs on the same basis, finding which term dominates, and
placing a capital cost within a factor of about two of a project somebody
actually built at a stated scope. What it is NOT good for: a bankable
figure, or any break-even price -- break-even runs on the OPEX rates and
the prices, and those have one published total to be checked against
between them. A statement of the form "this project is profitable" needs
dated quotations, a real equipment list, and an offtake, none of which are
in here.
"""

from dataclasses import dataclass, field
from typing import Literal

import jax.numpy as jnp
from jax import Array

from difflow_ree.database import get_ree_database, get_extractant_database


# =============================================================================
# REE Product Pricing
# =============================================================================

@dataclass
class REEPricing:
    """REE product pricing model.

    REE prices are highly volatile and depend on:
    - Purity (higher purity = premium)
    - Form (oxide, metal, compounds)
    - Market conditions

    Attributes:
        base_prices: Base prices in USD/kg (oxide basis)
        purity_premium: Premium factor for high purity
        form_factors: Price multipliers for different forms
    """
    base_prices: dict[str, float] = field(default_factory=lambda: {
        "La": 5.0,
        "Ce": 2.0,
        "Pr": 85.0,
        "Nd": 120.0,
        "Sm": 15.0,
        "Eu": 35.0,
        "Gd": 55.0,
        "Tb": 1500.0,
        "Dy": 450.0,
        "Y": 35.0,
    })

    purity_premium: dict[str, float] = field(default_factory=lambda: {
        "99%": 1.0,
        "99.9%": 1.3,
        "99.99%": 2.0,
        "99.999%": 5.0,
    })

    form_factors: dict[str, float] = field(default_factory=lambda: {
        "oxide": 1.0,
        "metal": 3.0,
        "chloride": 0.9,
        "nitrate": 0.85,
        "carbonate": 0.8,
    })

    def get_price(
        self,
        element: str,
        purity: str = "99%",
        form: str = "oxide",
    ) -> float:
        """Get price for specific REE product.

        Args:
            element: REE symbol
            purity: Purity grade
            form: Product form

        Returns:
            Price in USD/kg
        """
        base = self.base_prices.get(element, 50.0)
        purity_mult = self.purity_premium.get(purity, 1.0)
        form_mult = self.form_factors.get(form, 1.0)
        return base * purity_mult * form_mult

    def get_price_array(
        self,
        elements: list[str],
        purity: str = "99%",
        form: str = "oxide",
    ) -> Array:
        """Get prices as JAX array.

        Args:
            elements: List of element symbols
            purity: Purity grade
            form: Product form

        Returns:
            JAX array of prices
        """
        prices = [self.get_price(e, purity, form) for e in elements]
        return jnp.array(prices)

    def update_prices(self, new_prices: dict[str, float]):
        """Update base prices (for market scenarios).

        Args:
            new_prices: Dictionary of new prices
        """
        self.base_prices.update(new_prices)


# =============================================================================
# Reagent Costs
# =============================================================================

@dataclass
class ReagentCosts:
    """Reagent cost data for REE processing.

    Attributes:
        extractants: Extractant costs (USD/kg)
        acids: Acid costs (USD/kg)
        bases: Base costs (USD/kg)
        precipitants: Precipitant costs (USD/kg)
        diluents: Diluent costs (USD/L)
    """
    extractants: dict[str, float] = field(default_factory=lambda: {
        "D2EHPA": 8.0,
        "PC88A": 12.0,
        "Cyanex272": 25.0,
        "TBP": 4.0,
    })

    acids: dict[str, float] = field(default_factory=lambda: {
        "HCl": 0.15,  # USD/kg (30% solution)
        "H2SO4": 0.10,
        "HNO3": 0.40,
    })

    bases: dict[str, float] = field(default_factory=lambda: {
        "NaOH": 0.50,
        "NH4OH": 0.35,
        "Na2CO3": 0.30,
    })

    precipitants: dict[str, float] = field(default_factory=lambda: {
        "oxalic_acid": 2.0,
        "Na2CO3": 0.30,
        "NH4OH": 0.35,
    })

    diluents: dict[str, float] = field(default_factory=lambda: {
        "kerosene": 1.0,
        "n-heptane": 2.5,
        "Shellsol_D70": 1.5,
    })

    def extractant_makeup_cost(
        self,
        extractant: str,
        loss_rate: float = 0.001,  # kg lost per kg REE processed
        ree_throughput: float = 1.0,  # kg REE/hour
    ) -> float:
        """Calculate extractant makeup cost.

        Args:
            extractant: Extractant name
            loss_rate: Extractant loss (kg/kg REE)
            ree_throughput: REE throughput (kg/hour)

        Returns:
            Makeup cost (USD/hour)
        """
        unit_cost = self.extractants.get(extractant, 10.0)
        return unit_cost * loss_rate * ree_throughput

    def acid_cost(
        self,
        acid: str,
        consumption: float,  # kg/hour
    ) -> float:
        """Calculate acid cost.

        Args:
            acid: Acid type
            consumption: Consumption rate (kg/hour)

        Returns:
            Cost (USD/hour)
        """
        unit_cost = self.acids.get(acid, 0.20)
        return unit_cost * consumption


# =============================================================================
# Operating Costs
# =============================================================================

@dataclass
class OperatingCosts:
    """Operating cost estimation.

    Attributes:
        labor_rate: Labor cost (USD/hour/worker)
        electricity_rate: Electricity cost (USD/kWh)
        steam_rate: Steam cost (USD/kg)
        cooling_water_rate: Cooling water cost (USD/m³)
        maintenance_factor: Maintenance as fraction of CAPEX
    """
    labor_rate: float = 35.0  # USD/hour
    electricity_rate: float = 0.08  # USD/kWh
    steam_rate: float = 0.02  # USD/kg
    cooling_water_rate: float = 0.05  # USD/m³
    maintenance_factor: float = 0.03  # 3% of CAPEX

    def labor_cost(
        self,
        n_operators: int,
        hours_per_year: float = 8000,
    ) -> float:
        """Calculate annual labor cost.

        Args:
            n_operators: Total number of operators (all shifts combined)
            hours_per_year: Operating hours per year

        Returns:
            Annual labor cost (USD/year)
        """
        # n_operators is total headcount across all shifts
        # (e.g., 6 total for a small REE plant with ~2 per shift)
        return self.labor_rate * n_operators * hours_per_year

    def utility_cost(
        self,
        electricity_kw: float,
        steam_kg_hr: float,
        cooling_m3_hr: float,
        hours_per_year: float = 8000,
    ) -> float:
        """Calculate annual utility cost.

        Args:
            electricity_kw: Power consumption (kW)
            steam_kg_hr: Steam consumption (kg/hr)
            cooling_m3_hr: Cooling water (m³/hr)
            hours_per_year: Operating hours

        Returns:
            Annual utility cost (USD/year)
        """
        elec = self.electricity_rate * electricity_kw * hours_per_year
        steam = self.steam_rate * steam_kg_hr * hours_per_year
        cooling = self.cooling_water_rate * cooling_m3_hr * hours_per_year
        return elec + steam + cooling


# =============================================================================
# Capital cost
# =============================================================================

# Chemical Engineering Plant Cost Index, annual averages. Used only as a ratio,
# so the absolute base does not matter. Values after 2024 are HELD AT THE 2024
# VALUE because a verified published index for them was not retrieved -- a
# deliberate refusal to extrapolate, not a claim that the index was flat.
CEPCI = {
    2010: 550.8, 2012: 584.6, 2015: 556.8, 2018: 603.1, 2020: 596.2,
    2021: 708.0, 2022: 816.0, 2023: 800.0, 2024: 820.0,
    2025: 820.0, 2026: 820.0,
}

#: Capital-intensity anchors taken from PUBLIC REE separation projects.
#:
#: This is the part of a capital estimate that cannot be derived from first
#: principles: somebody has to have built one. Each entry records a real
#: project, the capacity it was costed at, the year of that cost, and -- the
#: field that actually matters -- what was and was not inside its battery
#: limits. `source` is a key in ``difflow_ree/data/sources.yaml``.
#:
#: Read `capacity_basis` carefully before comparing a number to your own
#: flowsheet. "REO feed" is the oxide-equivalent rare earth content entering
#: the separation circuit; "separated REO" is finished product leaving it.
#: For a bastnasite circuit these differ by a factor of several.
CAPEX_ANCHORS: dict[str, dict] = {
    "sx_retrofit": {
        "capex_usd": 16.0e6,
        "capacity_tpy": 4500.0,
        "capacity_basis": "REO feed",
        "year": 2024,
        "source": "EF_PH1",
        "aace_class": "as-built actual cost",
        "accuracy": (0.0, 0.0),
        "description": (
            "Energy Fuels White Mesa Phase 1 REE separation circuit: solvent "
            "extraction installed inside an already-licensed operating mill."
        ),
        "includes": (
            "mixer-settler trains, their tanks, pumps, piping and "
            "instrumentation, and installation of same",
        ),
        "excludes": (
            "cracking and leaching to a mixed carbonate",
            "buildings, civils and site preparation",
            "electrical supply, utilities and steam",
            "effluent and wastewater treatment",
            "licensing, permitting, land and owner's costs",
            "working capital and initial solvent inventory",
        ),
        "sections": (
            "mixer_settlers", "tanks_vessels", "pumps_piping",
            "instrumentation", "installation", "engineering", "contingency",
        ),
    },
    "separation_plant": {
        "capex_usd": 302.0e6,
        "capacity_tpy": 10000.0,
        "capacity_basis": "separated REO",
        # The release tabulates a product slate summing to exactly 10,000 t/yr
        # AND gives overall plant recovery as 98%, so feed and product differ
        # by 2%. This is the one anchor that reads the same on either basis.
        "capacity_note": (
            "10,000 t/yr of separated product; at the disclosed 98% overall "
            "recovery the feed is 10,200 t/yr, so the basis does not matter "
            "for this anchor"
        ),
        "year": 2012,
        "cost_basis": "fourth quarter 2011 price quotations",
        "source": "AVALON_GEISMAR",
        "aace_class": "AACE Class 4 (prefeasibility), +/-25% claimed",
        # The study's own stated intent, not a range imputed from its class:
        # "have an intended overall accuracy of +/-25%". A claim about a
        # study, not an outcome -- sources.yaml:LYNAS_KALGOORLIE is what
        # happened to an REE plant that got built.
        "accuracy": (-0.25, 0.25),
        "description": (
            "Avalon Rare Metals' proposed Nechalacho separation plant and "
            "refinery at Geismar, Louisiana: a standalone separation refinery "
            "on a greenfield site, fed with mixed concentrate from elsewhere. "
            "Prefeasibility study by SNC-Lavalin."
        ),
        # First entry is the release's own words; the rest expand it, and are
        # difflow's reading of "a complete separation plant facility".
        "includes": (
            "\"a complete separation plant facility; infrastructure, "
            "utilities and ancillary services; indirect costs; and "
            "contingency\"",
            "the full solvent-extraction house -- over 1,000 mixer-settlers, "
            "disclosed at 33% of the total, US$101 million",
            "reagent receipt, storage and make-up",
            "precipitation and calcination to saleable oxide",
            "buildings, civils, electrical and utilities",
            "engineering, offsites and contingency",
        ),
        "excludes": (
            "the mine and concentrator",
            "the sulphuric acid bake plant in the Northwest Territories",
            "the cracking plant co-located with the separation plant",
            "land, which had not been acquired when the estimate was priced",
            "owner's costs",
        ),
        "sections": (
            "mixer_settlers", "tanks_vessels", "pumps_piping",
            "instrumentation", "precipitation", "ce_removal", "installation",
            "civil_structures", "electrical_utilities", "effluent_treatment",
            "engineering", "contingency",
        ),
    },
    "integrated": {
        "capex_usd": 410.0e6,
        "capacity_tpy": 7554.0,
        "capacity_basis": "separated REO",
        "year": 2026,
        "source": "EF_PH2",
        "aace_class": "AACE Class 3 (bankable feasibility)",
        "accuracy": (-0.20, 0.30),
        "description": (
            "Energy Fuels White Mesa Phase 2 circuit: cracking and leaching "
            "through to separated NdPr, Tb, Dy, SEG and Ho+ concentrates, "
            "plus uranium recovery, on an existing licensed site."
        ),
        "includes": (
            "monazite cracking and leaching",
            "the full solvent-extraction house including heavies circuits",
            "precipitation, calcination and product finishing",
            "effluent treatment, utilities and infrastructure",
            "engineering and contingency to Class 3 definition",
        ),
        "excludes": (
            "the mineral-sand mines supplying the monazite",
            "land and owner's costs",
        ),
        "sections": (
            "mixer_settlers", "tanks_vessels", "pumps_piping",
            "instrumentation", "leaching", "precipitation", "ce_removal",
            "installation", "civil_structures", "electrical_utilities",
            "effluent_treatment", "engineering", "contingency",
        ),
    },
}

#: How the anchored total is split across line items.
#:
#: ESTIMATED, and the weakest part of this function. The anchors above publish
#: a total, not a breakdown, so these shares are conventional
#: fluid-processing-plant proportions rather than anything reported by the
#: projects. They are renormalised over whichever sections the chosen scope
#: actually contains, so the ITEMS are indicative while the TOTAL is anchored.
#: If you need a defensible breakdown, price an equipment list.
_SECTION_SHARES: dict[str, float] = {
    "mixer_settlers": 0.22,
    "tanks_vessels": 0.06,
    "pumps_piping": 0.09,
    "instrumentation": 0.07,
    "leaching": 0.12,
    "precipitation": 0.06,
    "ce_removal": 0.04,
    "installation": 0.13,
    "civil_structures": 0.09,
    "electrical_utilities": 0.07,
    "effluent_treatment": 0.04,
    "engineering": 0.08,
    "contingency": 0.15,
}

#: Sections whose cost tracks the number of mixer-settler stages.
_STAGE_DRIVEN = ("mixer_settlers", "tanks_vessels", "pumps_piping",
                 "instrumentation", "installation")

#: The mixer-settler house as equipment, before installation labour.
_SX_EQUIPMENT = ("mixer_settlers", "tanks_vessels", "pumps_piping",
                 "instrumentation")

#: The ONLY section-level figure any of difflow's anchors publishes: Avalon's
#: solvent-extraction circuit, "over 1,000 mixer-settlers", at 33 % of total
#: capital, US$101 million (``sources.yaml:AVALON_GEISMAR``).
#:
#: It is what keeps ``_SECTION_SHARES`` above from being unfalsifiable. The
#: release does not define where "the solvent extraction circuit" stops, so
#: the requirement is a bracket rather than a match: mixer-settlers alone must
#: come out below 33 % and the whole SX equipment group above it.
#: ``tests/ree/test_capex_anchors.py`` asserts exactly that.
#:
#: It also prices a stage. Over 1,000 units for US$101 million is at most
#: about US$101,000 each in 2011 dollars, near US$142,000 in 2024 dollars --
#: which is the check a per-stage equipment price should have been held to.
_DISCLOSED_SX_SHARE = 0.33


def capex_basis(scope: str = "separation_plant") -> dict:
    """Report where :func:`estimate_capex` gets its level for ``scope``.

    Returns the anchor project, its capacity and cost basis, its battery
    limits, its AACE estimate class and the accuracy range that class carries.
    Call this before quoting a CAPEX number anywhere.

    Args:
        scope: Key of :data:`CAPEX_ANCHORS`.

    Returns:
        A copy of the anchor record, plus ``scope``.

    Example:
        >>> basis = capex_basis("sx_retrofit")
        >>> basis["source"]
        'EF_PH1'
    """
    if scope not in CAPEX_ANCHORS:
        raise ValueError(
            f"unknown scope {scope!r}; choose one of "
            f"{sorted(CAPEX_ANCHORS)} -- and read its battery limits, because "
            f"they differ by more than an order of magnitude"
        )
    return {
        "scope": scope,
        # The anchor's own class describes the anchor. What estimate_capex
        # returns is that anchor moved to another capacity by a power law from
        # a SINGLE data point, which is a capacity-factored estimate however
        # good the anchor was -- AACE 18R-97 Class 5, the widest class there
        # is. A bankable anchor does not buy a bankable answer.
        "derived_class": "AACE Class 5 (capacity-factored from one project)",
        "derived_accuracy": (-0.50, 1.00),
        "derived_accuracy_note": (
            "AACE 18R-97 Table 1 gives Class 5 a low range of -20% to -50% "
            "and a high range of +30% to +100%. The wide end is quoted here "
            "because a single-point capacity factoring over a large capacity "
            "ratio has no better claim."
        ),
        **CAPEX_ANCHORS[scope],
    }


def estimate_capex(
    annual_ree_tonnes: float,
    n_stages_extraction: int,
    n_stages_scrubbing: int,
    n_stages_stripping: int,
    include_precipitation: bool = True,
    include_ce_removal: bool = False,
    year: int = 2024,
    *,
    scope: str = "separation_plant",
    n_stages_reference: int | None = None,
    scale_exponent: float = 0.6,
) -> dict[str, float]:
    """Estimate capital cost for an REE separation plant.

    The *level* is anchored to a public project (see :data:`CAPEX_ANCHORS` and
    :func:`capex_basis`) and scaled to ``annual_ree_tonnes`` by the usual
    power law. The *shape* -- how cost responds to stage count, and the split
    across line items -- is conventional and ESTIMATED. So the total is as
    good as the anchor and the scaling; the individual rows are indicative.

    **``scope`` is the most important argument.** The three anchors differ by
    more than an order of magnitude at the same capacity, because they are
    battery limits around different amounts of plant, not competing estimates
    of one plant:

    ``"sx_retrofit"``
        A solvent-extraction circuit dropped into a plant that already exists
        and already has buildings, power, utilities, effluent treatment and a
        licence. What a mixer-settler cascade costs, and nothing else.
    ``"separation_plant"`` (default)
        A standalone separation refinery on a greenfield site: the same
        cascade plus everything needed to stand it up and run it.
    ``"integrated"``
        The above plus cracking and leaching of concentrate to feed it.

    The default is the middle one deliberately: it is the answer to the
    question "what does an REE separation plant cost", and a caller who has
    not thought about battery limits should get the conservative number rather
    than the retrofit number.

    Args:
        annual_ree_tonnes: Plant capacity, tonnes/year. Compare against the
            anchor's ``capacity_basis`` -- feed or product -- before trusting
            the result; they differ by a factor of several.
        n_stages_extraction: Number of extraction stages.
        n_stages_scrubbing: Number of scrubbing stages.
        n_stages_stripping: Number of stripping stages.
        include_precipitation: Include a precipitation/calcination section.
            Ignored for ``scope="sx_retrofit"``, which has none, and mirror
            this in :func:`estimate_opex` so a plant is not charged for
            precipitant it has no capital for.
        include_ce_removal: Include a Ce oxidation section.

            Dropping either section **reduces the total** by that section's
            share of the anchor's scope, rather than reallocating it over the
            remaining rows. The deduction is therefore as good as the
            ESTIMATED section split, which is to say indicative -- but the
            alternative is quoting the price of a plant that has the section
            to a caller who said they do not want it.
        year: Cost basis year. The anchor is moved to it by CEPCI ratio.
        scope: Battery limits. See above and :func:`capex_basis`.
        n_stages_reference: Total stage count the anchor's cost is taken to
            correspond to. **Default None, meaning no stage adjustment at
            all** -- none of the anchors publishes a stage count, so assuming
            one would put an invented number into the level. Pass your own
            base-case stage count to get stage sensitivity: the anchor is then
            reproduced exactly at that base case, and only the stage-driven
            fraction of capital moves as you vary stages around it.
        scale_exponent: Capacity scaling exponent; 0.6 is the usual value.

    Returns:
        Line items in USD plus ``"total"``. Every value is a float, so the
        dict can be iterated and printed without special-casing; the
        provenance lives in :func:`capex_basis`, not in here.

    Example:
        >>> c = estimate_capex(500, 8, 4, 4, scope="sx_retrofit")
        >>> round(c["total"] / 1e6, 2)
        4.28
    """
    anchor = capex_basis(scope)

    if year not in CEPCI:
        raise ValueError(
            f"no CEPCI value for {year}; have {sorted(CEPCI)}. Add a published "
            f"index rather than extrapolating one"
        )
    cepci_ratio = CEPCI[year] / CEPCI[anchor["year"]]

    # Which sections this scope has at all.
    sections = list(anchor["sections"])
    if not include_precipitation and "precipitation" in sections:
        sections.remove("precipitation")
    if not include_ce_removal and "ce_removal" in sections:
        sections.remove("ce_removal")

    shares = {s: _SECTION_SHARES[s] for s in sections}
    share_sum = sum(shares.values())

    # Dropping a section has to leave the total, not be redistributed over the
    # rows that remain. The deduction is the dropped section's ESTIMATED share
    # of the anchor's full scope, so it carries that class -- but a caller who
    # asks for a plant without precipitation and gets the price of one with it
    # has been told something false, and the error would be invisible because
    # the row simply disappears.
    full_share_sum = sum(_SECTION_SHARES[s] for s in anchor["sections"])
    scope_factor = share_sum / full_share_sum

    # Fraction of in-scope capital that tracks stage count. For a retrofit
    # this is most of it; for a full refinery, about half.
    f_stage = sum(v for k, v in shares.items() if k in _STAGE_DRIVEN) / share_sum

    n_total = n_stages_extraction + n_stages_scrubbing + n_stages_stripping
    stage_ratio = 1.0 if n_stages_reference is None else n_total / n_stages_reference

    # Level: the anchor, scaled on capacity, moved to the cost basis year, and
    # adjusted for a stage count different from the anchor's. Only the
    # stage-driven fraction responds -- civils and a licence do not get more
    # expensive because the cascade grew. At n_total == n_stages_reference the
    # bracket is 1 and the anchor is reproduced exactly.
    capacity_ratio = (annual_ree_tonnes / anchor["capacity_tpy"]) ** scale_exponent
    stage_multiplier = (1.0 - f_stage) + f_stage * stage_ratio
    total = (anchor["capex_usd"] * capacity_ratio * cepci_ratio
             * stage_multiplier * scope_factor)

    # Shape: split the total across line items, with the stage-driven ones
    # carrying the stage adjustment so the rows still sum to the total.
    weights = {
        s: v * (stage_ratio if s in _STAGE_DRIVEN else 1.0)
        for s, v in shares.items()
    }
    weight_sum = sum(weights.values())
    breakdown = {s: total * w / weight_sum for s, w in weights.items()}
    breakdown["total"] = total
    return breakdown


def estimate_opex(
    annual_ree_tonnes: float,
    capex: float,
    extractant: str = "D2EHPA",
    product_elements: list[str] = None,
    include_precipitation: bool = True,
    n_operators: int = 6,
) -> dict[str, float]:
    """Estimate annual operating cost.

    See the module docstring: every unit rate below is indicative, and the
    total is useful for comparing designs rather than as a budget.

    Args:
        annual_ree_tonnes: Annual REE production
        capex: Total capital cost
        extractant: Primary extractant
        product_elements: Elements being produced
        include_precipitation: Charge for precipitant (oxalic acid). Mirrors
            the flag of the same name on :func:`estimate_capex`, which until
            now had no OPEX counterpart -- so a circuit costed with
            ``include_precipitation=False`` still paid $3/kg for a
            precipitation section it had no capital for. Set both the same
            way. A circuit whose product leaves as a loaded strip liquor for
            downstream precipitation elsewhere wants ``False``.
        n_operators: TOTAL operating headcount across all shifts, passed
            straight to :meth:`OperatingCosts.labor_cost`. The default 6 is
            deliberately the total, not per-shift: multiplying by a shift
            count here was bug #123, and
            ``tests/ree/test_ree_moderate_bugs.py::TestBug123_LaborCostOverestimate``
            pins it. Do not "fix" it back. Six people covering 8000 h/year is
            a thin crew for continuous operation -- four rotating crews is
            the usual answer -- so pass a larger total if you want realistic
            staffing rather than editing the default.

    Returns:
        Dictionary of operating cost components
    """
    if product_elements is None:
        product_elements = ["Nd", "Dy"]

    reagents = ReagentCosts()
    opex = OperatingCosts()

    # Reagent costs (rough estimates based on throughput)
    # Extractant makeup: ~$2/kg REE
    extractant_cost = annual_ree_tonnes * 1000 * 2.0

    # Acid consumption: ~$1/kg REE
    acid_cost = annual_ree_tonnes * 1000 * 1.0

    # Base consumption: ~$0.5/kg REE
    base_cost = annual_ree_tonnes * 1000 * 0.5

    # Precipitant: ~$3/kg REE (oxalic acid), only if there is a
    # precipitation section to consume it.
    if include_precipitation:
        precipitant_cost = annual_ree_tonnes * 1000 * 3.0
    else:
        precipitant_cost = 0.0

    # Labor. `labor_cost` wants TOTAL headcount, not per-shift -- the two
    # readings differ by the number of crews, so conflating them is the 4x
    # error that was bug #123. The stale "6 operators per shift" comment that
    # used to sit here described the pre-#123 code, not this call.
    labor_cost = opex.labor_cost(n_operators, 8000)

    # Utilities (estimate based on throughput)
    electricity_kw = 50 + annual_ree_tonnes * 0.5  # kW
    steam_kg_hr = annual_ree_tonnes * 0.1  # kg/hr
    cooling_m3_hr = annual_ree_tonnes * 0.2  # m³/hr
    utility_cost = opex.utility_cost(electricity_kw, steam_kg_hr, cooling_m3_hr)

    # Maintenance (3% of CAPEX)
    maintenance_cost = capex * opex.maintenance_factor

    # Total OPEX
    total_opex = (
        extractant_cost +
        acid_cost +
        base_cost +
        precipitant_cost +
        labor_cost +
        utility_cost +
        maintenance_cost
    )

    return {
        "extractant": extractant_cost,
        "acid": acid_cost,
        "base": base_cost,
        "precipitant": precipitant_cost,
        "labor": labor_cost,
        "utilities": utility_cost,
        "maintenance": maintenance_cost,
        "total": total_opex,
    }


# =============================================================================
# Profitability Analysis
# =============================================================================

def calculate_revenue(
    product_flows: dict[str, float],  # kg/year per element
    pricing: REEPricing | None = None,
    purity: str = "99%",
    form: str = "oxide",
) -> dict[str, float]:
    """Calculate annual revenue from REE products.

    Args:
        product_flows: Annual production of each element (kg/year)
        pricing: REE pricing model
        purity: Product purity
        form: Product form

    Returns:
        Revenue breakdown by element and total
    """
    if pricing is None:
        pricing = REEPricing()

    revenue = {}
    total = 0.0

    for element, production in product_flows.items():
        price = pricing.get_price(element, purity, form)
        elem_revenue = production * price
        revenue[element] = elem_revenue
        total += elem_revenue

    revenue["total"] = total
    return revenue


def calculate_profit(
    revenue: float,
    opex: float,
    capex: float,
    tax_rate: float = 0.25,
    depreciation_years: int = 10,
) -> dict[str, float]:
    """Calculate profitability metrics.

    Args:
        revenue: Annual revenue (USD/year)
        opex: Annual operating cost (USD/year)
        capex: Total capital cost (USD)
        tax_rate: Corporate tax rate
        depreciation_years: Depreciation period

    Returns:
        Dictionary of profitability metrics
    """
    # EBITDA
    ebitda = revenue - opex

    # Depreciation (straight-line)
    depreciation = capex / depreciation_years

    # EBIT
    ebit = ebitda - depreciation

    # Taxes
    taxes = max(0, ebit * tax_rate)

    # Net income
    net_income = ebit - taxes

    # Cash flow (add back depreciation)
    cash_flow = net_income + depreciation

    # Simple payback
    payback = capex / cash_flow if cash_flow > 0 else float('inf')

    # ROI
    roi = net_income / capex if capex > 0 else 0

    return {
        "revenue": revenue,
        "opex": opex,
        "ebitda": ebitda,
        "depreciation": depreciation,
        "ebit": ebit,
        "taxes": taxes,
        "net_income": net_income,
        "cash_flow": cash_flow,
        "payback_years": payback,
        "roi": roi,
    }


def minimum_selling_price(
    opex: float,
    capex: float,
    annual_production_kg: float,
    target_roi: float = 0.15,
    tax_rate: float = 0.25,
    depreciation_years: int = 10,
) -> float:
    """Calculate minimum selling price for target ROI.

    Args:
        opex: Annual operating cost (USD/year)
        capex: Total capital cost (USD)
        annual_production_kg: Annual production (kg/year)
        target_roi: Target return on investment
        tax_rate: Corporate tax rate
        depreciation_years: Depreciation period

    Returns:
        Minimum selling price (USD/kg)
    """
    # Target net income
    target_net = target_roi * capex

    # Required EBIT (before tax)
    required_ebit = target_net / (1 - tax_rate)

    # Required EBITDA
    depreciation = capex / depreciation_years
    required_ebitda = required_ebit + depreciation

    # Required revenue
    required_revenue = required_ebitda + opex

    # MSP
    msp = required_revenue / annual_production_kg

    return msp
