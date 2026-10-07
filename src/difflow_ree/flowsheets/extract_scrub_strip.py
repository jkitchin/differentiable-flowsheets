"""Industrial 3-section extract-scrub-strip circuit.

Three-section flowsheet:
1. Extraction: Load all REE onto organic
2. Scrubbing: Remove unwanted REE (typically lighter ones)
3. Stripping: Recover purified REE product

        Feed                    Scrub                   Product
          ↓                       ↓                        ↑
    ┌───────────┐         ┌───────────┐         ┌───────────┐
    │           │         │           │         │           │
    │ EXTRACTION│ ──Org──▶│ SCRUBBING │ ──Org──▶│ STRIPPING │
    │           │         │           │         │           │
    └───────────┘         └───────────┘         └───────────┘
          ↓                     ↓          ◀──Org──    ↓
      Raffinate            Scrub Liquor          Strip Acid
    (depleted)           (impurities)         (recycled)

The scrub liquor (containing rejected light REE) is typically
recycled back to the feed or processed in a separate circuit.

All operations are fully differentiable using JAX.
"""

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from difflow.numerics import safe_divide
from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, make_stream, get_flows
from difflow_ree.units.extraction import REEExtractor, REEExtractorParams
from difflow_ree.units.scrubbing import REEScrubber, ScrubberParams
from difflow_ree.units.stripping import REEStripper, StripperParams


@dataclass(repr=False)
class ExtractScrubStripParams(ParamsMixin):
    """Parameters for 3-section circuit.

    Attributes:
        extractant: Extractant name
        elements: All REE elements to track
        target_elements: Elements counted as the product. The three sections
            are driven by their pH values, stage counts and phase ratios, and
            every element in ``elements`` goes through the same equations
            whatever is listed here (#288). It selects which elements
            ``target_recovery``, ``target_purity`` and ``impurity_rejection``
            are computed over, and, for any section pH left as None, which
            boundary that pH is cut at (2026 operating-point audit, R1); with
            every pH given it moves no flow. Must be a subset of
            ``elements``.
        diluent: Organic diluent name (e.g., "kerosene", "n-dodecane")
        n_extraction_stages: Number of extraction stages
        n_scrubbing_stages: Number of scrubbing stages
        n_stripping_stages: Number of stripping stages
        extraction_pH: pH in extraction section. None (the default) is read
            off the extractant's D curves
            (:func:`difflow_ree.equilibrium.operating_points.cut_pHs`): with
            targets, where the geometric-mean ``D * (O/A)`` of the boundary
            pair (least extractable target, most extractable non-target) is
            one; without, where every element has ``D * (O/A) >= 10``.
        scrubbing_pH: pH in scrubbing section. None resolves to where the
            boundary pair's mean ``D`` equals the scrub aqueous/organic ratio
            (the extraction pH when there are no targets).
        stripping_pH: pH in stripping section. None resolves to where the
            most strongly held target (every element, without targets) has
            ``D * (O/A) = 0.1``. Heavy REE on D2EHPA need strong acid, below
            the fitted window; the distribution warns about the
            extrapolation when the section runs.
        extractant_conc: Extractant concentration (M)
        solvent_to_feed_ratio: O/A in extraction
        scrub_to_solvent_ratio: Scrub/O ratio
        strip_to_solvent_ratio: Strip/O ratio
        nitrate_conc: Aqueous nitrate concentration (M), required for solvating
            extractants such as TBP whose D is nitrate- rather than pH-driven
            (#195). None for the acidic cation-exchange extractants. Threaded
            to all three sections.
        mechanism: Explicit extraction-mechanism override passed to
            REEDistribution ("cation_exchange" / "solvating"). None takes the
            mechanism from the extractant record (#195). Threaded to all
            three sections, so a circuit never mixes mechanisms.
        capacity_sharpness: Sharpness k of the extraction section's smooth
            loading limiters; see REEExtractorParams (#193).
    """
    extractant: str
    elements: tuple[str, ...]
    target_elements: tuple[str, ...] = ()  # reporting label only, see #288
    diluent: str = "kerosene"
    n_extraction_stages: int = 10
    n_scrubbing_stages: int = 5
    n_stripping_stages: int = 5
    # None means "read it off the D curves for these targets" (see
    # __post_init__). The literals 3.5 / 2.0 / 0.5 were pre-#270; the window
    # fractions that replaced them (#270) did not sit between the groups
    # (audit R1).
    extraction_pH: float | None = None
    scrubbing_pH: float | None = None
    stripping_pH: float | None = None
    extractant_conc: float = 0.5
    solvent_to_feed_ratio: float = 1.0
    scrub_to_solvent_ratio: float = 0.2
    strip_to_solvent_ratio: float = 0.5
    nitrate_conc: float | None = None  # see #195
    mechanism: str | None = None  # see #195
    capacity_sharpness: int = 8  # see REEExtractorParams (#193)

    def __post_init__(self):
        """Resolve the record's pH defaults (#270); check the labels (#288)."""
        from difflow_ree.database import default_pH

        if isinstance(self.target_elements, str):
            raise TypeError(
                "ExtractScrubStripParams.target_elements must be a tuple of "
                f"element names, not the string {self.target_elements!r}; "
                f'write ("{self.target_elements}",)'
            )
        self.target_elements = tuple(self.target_elements)
        unknown = [e for e in self.target_elements if e not in tuple(self.elements)]
        if unknown:
            raise ValueError(
                f"ExtractScrubStripParams.target_elements {unknown} are not "
                f"in elements {tuple(self.elements)}. target_elements only "
                "selects which elements the recovery and purity metrics are "
                "reported over (#288), so an untracked name would silently "
                "contribute nothing; add them to elements or drop them from "
                "target_elements."
            )

        # Unset pHs come off the D curves for this circuit's own elements
        # and targets, at the phase ratios its units really run at (2026
        # operating-point audit, R1). The window fractions of default_pH
        # (#270) used before scrubbed at a pH that removed only part of the
        # La/Ce/Nd and stripped at one that left the heavies on the solvent:
        # targets Gd/Tb/Dy/Y on D2EHPA gave a product of 0.09 % target
        # purity and 0.2 % Y recovery. A pH the caller gives is kept.
        if None in (self.extraction_pH, self.scrubbing_pH, self.stripping_pH):
            from difflow_ree.equilibrium.operating_points import (
                circuit_phase_ratios, cut_pHs)

            ratios = circuit_phase_ratios(
                self.solvent_to_feed_ratio, self.scrub_to_solvent_ratio,
                self.strip_to_solvent_ratio, self.extractant_conc)
            cuts = cut_pHs(
                self.extractant, tuple(self.elements), self.target_elements,
                extraction_OA=ratios["extraction"],
                scrub_OA=ratios["scrubbing"], strip_OA=ratios["stripping"],
                extractant_conc=self.extractant_conc,
                nitrate_conc=self.nitrate_conc, mechanism=self.mechanism)
            for duty in ("extraction", "scrubbing", "stripping"):
                if getattr(self, f"{duty}_pH") is None:
                    value = (getattr(cuts, duty) if cuts is not None
                             else default_pH(self.extractant, duty))
                    setattr(self, f"{duty}_pH", value)


class ExtractScrubStripCircuit:
    """Industrial 3-section REE separation circuit.

    Produces purified REE product by:
    1. Extracting all REE from feed
    2. Scrubbing out lighter/unwanted REE
    3. Stripping purified heavy/target REE

    Example:
        >>> params = ExtractScrubStripParams(
        ...     extractant="D2EHPA",
        ...     elements=("La", "Ce", "Nd", "Dy"),
        ...     target_elements=("Nd", "Dy"),  # Heavy REE product
        ...     n_extraction_stages=10,
        ...     n_scrubbing_stages=5,
        ...     n_stripping_stages=5,
        ... )
        >>> circuit = ExtractScrubStripCircuit(params)
        >>> results = circuit(feed)
        >>> print(f"Nd purity: {results['product_purity']['Nd']:.1%}")
    """

    symbol = "Extract-Scrub-Strip"
    equations = [
        r"\text{feed} \xrightarrow{\text{Extract}} \text{loaded org.} \xrightarrow{\text{Scrub}} \text{purified org.} \xrightarrow{\text{Strip}} \text{product}",
        r"\mathrm{purity}_i = \frac{F_i^\mathrm{product}}{\sum_j F_j^\mathrm{product}}",
    ]
    assumptions = [
        "Three counter-current sections (extract, scrub, strip) in series.",
        "Target vs. non-target selectivity driven by pH difference between sections.",
    ]
    references = ["Xie, F., Zhang, T.A., Dreisinger, D., Doyle, F. Miner. Eng., 56, 10 (2014)."]
    parameter_symbols = {
        "n_extraction_stages": "N_E",
        "n_scrubbing_stages": "N_Sc",
        "n_stripping_stages": "N_S",
    }
    parameter_units = {
        "n_extraction_stages": "-",
        "n_scrubbing_stages": "-",
        "n_stripping_stages": "-",
        "extraction_pH": "-",
        "scrubbing_pH": "-",
        "stripping_pH": "-",
        "extractant_conc": "mol/L",
        "solvent_to_feed_ratio": "-",
        "scrub_to_solvent_ratio": "-",
        "strip_to_solvent_ratio": "-",
        "nitrate_conc": "mol/L",
        "capacity_sharpness": "-",
    }

    def __init__(self, params: ExtractScrubStripParams):
        """Initialize circuit.

        Args:
            params: Circuit parameters
        """
        self.params = params

        # Extraction section
        self._extractor = REEExtractor(REEExtractorParams(
            n_stages=params.n_extraction_stages,
            extractant=params.extractant,
            elements=params.elements,
            diluent=params.diluent,
            pH=params.extraction_pH,
            extractant_conc=params.extractant_conc,
            nitrate_conc=params.nitrate_conc,  # see #195
            mechanism=params.mechanism,  # see #195
            capacity_sharpness=params.capacity_sharpness,  # see #193
        ))

        # Scrubbing section
        self._scrubber = REEScrubber(ScrubberParams(
            n_stages=params.n_scrubbing_stages,
            extractant=params.extractant,
            elements=params.elements,
            target_elements=params.target_elements,
            diluent=params.diluent,
            pH=params.scrubbing_pH,
            extractant_conc=params.extractant_conc,
            nitrate_conc=params.nitrate_conc,  # see #195
            mechanism=params.mechanism,  # see #195
        ))

        # Stripping section
        self._stripper = REEStripper(StripperParams(
            n_stages=params.n_stripping_stages,
            extractant=params.extractant,
            elements=params.elements,
            diluent=params.diluent,
            pH=params.stripping_pH,
            extractant_conc=params.extractant_conc,
            nitrate_conc=params.nitrate_conc,  # see #195
            mechanism=params.mechanism,  # see #195
        ))

    def __call__(
        self,
        feed: Stream,
        T: Array | float = 298.15,
    ) -> dict:
        """Run 3-section circuit.

        Args:
            feed: Aqueous REE feed solution
            T: Operating temperature (K)

        Returns:
            Dictionary with:
            - raffinate: Depleted aqueous from extraction
            - scrub_liquor: Rejected REE from scrubbing
            - product: Purified REE product
            - barren_organic: Stripped organic (for recycle)
            - target_recovery: Recovery of target elements
            - product_purity: Purity of each element in product
            - section_info: Details from each section
        """
        p = self.params
        T = jnp.asarray(T)

        feed_flows = get_flows(feed)
        F_aq = feed_flows.get("H2O", 1.0)

        # Create fresh solvent
        F_org = F_aq * p.solvent_to_feed_ratio
        solvent_flows = {
            p.diluent: F_org,
            p.extractant: p.extractant_conc * F_org,
        }
        for elem in p.elements:
            solvent_flows[elem] = 0.0
        solvent = make_stream(solvent_flows, T, feed["P"])

        # EXTRACTION
        raffinate, loaded_org, ext_info = self._extractor(
            feed, solvent, T, pH=p.extraction_pH
        )

        # Create scrub solution
        F_scrub = F_org * p.scrub_to_solvent_ratio
        scrub_flows = {"H2O": F_scrub}
        for elem in p.elements:
            scrub_flows[elem] = 0.0
        scrub_soln = make_stream(scrub_flows, T, feed["P"])

        # SCRUBBING
        scrub_liquor, scrubbed_org, scrub_info = self._scrubber(
            loaded_org, scrub_soln, T, pH=p.scrubbing_pH
        )

        # Create strip acid
        F_strip = F_org * p.strip_to_solvent_ratio
        strip_flows = {"H2O": F_strip}
        for elem in p.elements:
            strip_flows[elem] = 0.0
        strip_acid = make_stream(strip_flows, T, feed["P"])

        # STRIPPING
        product, barren_org, strip_info = self._stripper(
            scrubbed_org, strip_acid, T, pH=p.stripping_pH
        )

        # Calculate metrics
        product_flows = get_flows(product)
        scrub_flows = get_flows(scrub_liquor)
        raff_flows = get_flows(raffinate)

        # Target element recovery
        target_recovery = {}
        for elem in p.target_elements:
            f_in = float(feed_flows.get(elem, 0.0))
            f_out = float(product_flows.get(elem, 0.0))
            target_recovery[elem] = safe_divide(f_out, f_in)

        # Product purity (mole fraction)
        total_product_ree = sum(float(product_flows.get(e, 0.0)) for e in p.elements)
        product_purity = {}
        for elem in p.elements:
            product_purity[elem] = safe_divide(float(product_flows.get(elem, 0.0)), total_product_ree)

        # Target purity (sum of target elements)
        target_purity = sum(product_purity.get(e, 0.0) for e in p.target_elements)

        # Impurity rejection
        impurity_elements = [e for e in p.elements if e not in p.target_elements]
        impurity_rejection = {}
        for elem in impurity_elements:
            f_in = float(feed_flows.get(elem, 0.0))
            f_product = float(product_flows.get(elem, 0.0))
            impurity_rejection[elem] = 1 - safe_divide(f_product, f_in)

        # Mass balance verification. The barren organic is an outlet too:
        # REE left on the stripped solvent is not lost mass.
        barren_flows = get_flows(barren_org)
        feed_total = {
            elem: jnp.asarray(float(feed_flows.get(elem, 0.0)))
            for elem in p.elements
        }
        output_total = {
            elem: (
                float(raff_flows.get(elem, 0.0))
                + float(scrub_flows.get(elem, 0.0))
                + float(product_flows.get(elem, 0.0))
                + float(barren_flows.get(elem, 0.0))
            )
            for elem in p.elements
        }
        mass_closure = {
            elem: safe_divide(output_total[elem], feed_total[elem])
            for elem in p.elements
        }

        return {
            "raffinate": raffinate,
            "scrub_liquor": scrub_liquor,
            "product": product,
            "barren_organic": barren_org,
            "target_recovery": target_recovery,
            "product_purity": product_purity,
            "target_purity": target_purity,
            "impurity_rejection": impurity_rejection,
            "extraction_info": ext_info,
            "scrubbing_info": scrub_info,
            "stripping_info": strip_info,
            "mass_balance": {
                "feed": feed_total,
                "output": output_total,
                "closure": mass_closure,
            },
        }

    def material_balance(self, feed: Stream, results: dict) -> dict:
        """Check material balance across circuit.

        Args:
            feed: Input feed
            results: Results from circuit run

        Returns:
            Material balance closure for each element
        """
        feed_flows = get_flows(feed)
        raff_flows = get_flows(results["raffinate"])
        scrub_flows = get_flows(results["scrub_liquor"])
        product_flows = get_flows(results["product"])
        barren_flows = get_flows(results["barren_organic"])

        balance = {}
        for elem in self.params.elements:
            f_in = float(feed_flows.get(elem, 0.0))
            f_out = (
                float(raff_flows.get(elem, 0.0)) +
                float(scrub_flows.get(elem, 0.0)) +
                float(product_flows.get(elem, 0.0)) +
                float(barren_flows.get(elem, 0.0))
            )
            closure = safe_divide(f_out, f_in)
            balance[elem] = {
                "in": f_in,
                "out": f_out,
                "closure": closure,
            }

        return balance


def design_extract_scrub_strip(
    feed_composition: dict[str, float],
    target_elements: tuple[str, ...],
    extractant: str,
    target_purity: float = 0.95,
    target_recovery: float = 0.90,
    nitrate_conc: float | None = None,
    mechanism: str | None = None,
    max_extraction_stages: int = 30,
    max_scrubbing_stages: int = 30,
    max_stripping_stages: int = 20,
    pH_search: float = 1.0,
    pH_step: float = 0.1,
) -> ExtractScrubStripParams:
    """Design a 3-section circuit that meets purity and recovery targets.

    Starts from the cut pHs for the target/rest boundary
    (:func:`difflow_ree.equilibrium.operating_points.cut_pHs`) and searches
    the extraction and scrubbing pH within ``pH_search`` of their cuts, and
    the three stage counts, for the smallest total stage count whose predicted
    product meets both targets. Every target element must reach
    ``target_recovery`` (the circuit reports recovery per element), and the
    target share of the product must reach ``target_purity``. The prediction
    uses the D values at each candidate pH through the same Kremser fractions
    the units evaluate, at the phase ratios the units really run at; it does
    not model the extraction section's loading limiter, so it holds for a
    feed well below the solvent's capacity.

    The circuit does not reflux the scrub liquor, so target REE scrubbed back
    to the aqueous are lost: the search trades scrub pH against extraction pH
    to balance that loss against purity. The strip pH is held at its cut and
    its stage count sized so each target strips to within a tenth of the
    allowed recovery loss.

    This replaced a design that returned 10/5/5 stages at the window-fraction
    pHs whatever the targets, and raised ``KeyError`` for naphthenic acid
    from a separation-factor lookup it never used (2026 operating-point
    audit, R1).

    Args:
        feed_composition: Element flows in feed (mol/s); only the ratios matter.
        target_elements: Elements to recover.
        extractant: Extractant to use.
        target_purity: Target product purity (target share of the product).
        target_recovery: Target recovery of every target element.
        nitrate_conc: Aqueous nitrate concentration (M), required for
            solvating extractants such as TBP (#195).
        mechanism: Explicit mechanism override; see REEDistribution (#195).
        max_extraction_stages: Largest extraction stage count searched.
        max_scrubbing_stages: Largest scrubbing stage count searched.
        max_stripping_stages: Largest stripping stage count searched.
        pH_search: Half-width of the pH search around each cut.
        pH_step: pH grid step.

    Returns:
        ExtractScrubStripParams with the pHs and stage counts.

    Warns:
        UserWarning: If no design in the search meets both targets; the
            design with the best worst-case ratio of achieved to required is
            returned.

    Example:
        >>> p = design_extract_scrub_strip(
        ...     {"Nd": 0.01, "Dy": 0.001}, ("Dy",), "PC88A")
        >>> p.target_elements
        ('Dy',)
    """
    import warnings

    import numpy as np

    from difflow_ree.equilibrium.distribution import REEDistribution
    from difflow_ree.equilibrium.operating_points import (
        circuit_phase_ratios, is_pH_driven, kremser_fraction, section_factors)

    elements = tuple(feed_composition.keys())
    target_elements = tuple(target_elements)
    template = ExtractScrubStripParams(
        extractant=extractant, elements=elements,
        target_elements=target_elements,
        nitrate_conc=nitrate_conc, mechanism=mechanism)
    oa = circuit_phase_ratios(
        template.solvent_to_feed_ratio, template.scrub_to_solvent_ratio,
        template.strip_to_solvent_ratio, template.extractant_conc)
    dist = REEDistribution(
        extractant, elements, concentration=template.extractant_conc,
        nitrate_conc=nitrate_conc, mechanism=mechanism,
        on_out_of_range="ignore")

    offsets = np.round(np.arange(-pH_search, pH_search + 0.5 * pH_step, pH_step), 10)
    if not is_pH_driven(dist, elements):
        offsets = np.zeros(1)  # no pH lever (a solvating extractant)
    ext_pHs = template.extraction_pH + offsets
    scr_pHs = template.scrubbing_pH + offsets

    f = np.array([float(feed_composition[e]) for e in elements])
    is_t = np.array([e in target_elements for e in elements])

    # Strip: at its cut, sized so each target strips to (1 - loss/10).
    S_strip = 1.0 / section_factors(dist, elements, [template.stripping_pH],
                                    oa["stripping"])[0]
    n_str_all = np.arange(1, max_stripping_stages + 1)
    stripped_all = 1.0 - kremser_fraction(S_strip[None, :], n_str_all[:, None])
    need = 1.0 - 0.1 * (1.0 - target_recovery)
    good = (stripped_all[:, is_t] >= need).all(axis=1)
    j = int(np.argmax(good)) if good.any() else len(n_str_all) - 1
    n_strip = int(n_str_all[j])
    stripped = stripped_all[j]                                       # (el,)

    n_ext = np.arange(1, max_extraction_stages + 1)
    n_scr = np.arange(0, max_scrubbing_stages + 1)
    E = section_factors(dist, elements, ext_pHs, oa["extraction"])     # (pe, el)
    S = 1.0 / section_factors(dist, elements, scr_pHs, oa["scrubbing"])  # (ps, el)
    extracted = 1.0 - kremser_fraction(E[:, None, :], n_ext[None, :, None])  # (pe, Ne, el)
    retained = kremser_fraction(S[:, None, :], n_scr[None, :, None])         # (ps, Ns, el)
    # product fraction of each element: (pe, Ne, ps, Ns, el)
    frac = (extracted[:, :, None, None, :] * retained[None, None, :, :, :]
            * stripped)
    recovery = frac[..., is_t].min(axis=-1)
    product = frac * f
    purity = product[..., is_t].sum(axis=-1) / np.maximum(product.sum(axis=-1), 1e-300)

    total = n_ext[None, :, None, None] + n_scr[None, None, None, :]
    distance = (np.abs(offsets)[:, None, None, None]
                + np.abs(offsets)[None, None, :, None])
    ok = (recovery >= target_recovery) & (purity >= target_purity)
    if ok.any():
        # fewest stages, then nearest the cuts
        score = np.where(ok, total + 1e-3 * distance, np.inf)
        idx = np.unravel_index(np.argmin(score), score.shape)
    else:
        merit = np.minimum(recovery / target_recovery, purity / target_purity)
        idx = np.unravel_index(np.argmax(merit), merit.shape)
        warnings.warn(
            f"design_extract_scrub_strip: no design within the search meets "
            f"purity {target_purity} and recovery {target_recovery} for "
            f"{target_elements} on {extractant}; the best found predicts "
            f"purity {purity[idx]:.4f} and minimum target recovery "
            f"{recovery[idx]:.4f}.", UserWarning, stacklevel=2)
    ie, ne, isc, ns = idx

    return ExtractScrubStripParams(
        extractant=extractant,
        elements=elements,
        target_elements=target_elements,
        n_extraction_stages=int(n_ext[ne]),
        n_scrubbing_stages=int(n_scr[ns]),
        n_stripping_stages=n_strip,
        extraction_pH=float(ext_pHs[ie]),
        scrubbing_pH=float(scr_pHs[isc]),
        stripping_pH=template.stripping_pH,
        nitrate_conc=nitrate_conc,  # see #195
        mechanism=mechanism,  # see #195
    )
