"""Basic extract-strip circuit for REE separation.

Two-section flowsheet:
1. Extraction: Load REE onto organic from aqueous feed
2. Stripping: Recover REE from organic into product solution

      Feed                    Product
        ↓                        ↑
    ┌───────────┐         ┌───────────┐
    │           │         │           │
    │ EXTRACTION│ ──Org──▶│ STRIPPING │
    │           │         │           │
    └───────────┘         └───────────┘
        ↓           ◀──Org──    ↓
    Raffinate              Strip Acid

All operations are fully differentiable using JAX.
"""

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from difflow.numerics import safe_divide
from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, make_stream, get_flows
from difflow_ree.units.extraction import REEExtractor, REEExtractorParams
from difflow_ree.units.stripping import REEStripper, StripperParams


@dataclass(repr=False)
class ExtractStripParams(ParamsMixin):
    """Parameters for extract-strip circuit.

    Attributes:
        extractant: Extractant name
        elements: REE elements to track
        diluent: Organic diluent name (e.g., "kerosene", "n-dodecane")
        n_extraction_stages: Number of extraction stages
        n_stripping_stages: Number of stripping stages
        extraction_pH: pH in extraction section. None (the default) resolves
            to the pH where the least extractable element has
            ``D * (O/A) = 10``, read off the extractant's D curves
            (:func:`difflow_ree.equilibrium.operating_points.cut_pHs`).
        stripping_pH: pH in stripping section. None (the default) resolves to
            the pH where the most strongly held element has
            ``D * (O/A) = 0.1``. For D2EHPA and the heavy REE this is below
            the fitted window (strong acid), and the distribution model warns
            about the extrapolation when the section runs.
        extractant_conc: Extractant concentration (M)
        solvent_to_feed_ratio: Organic/aqueous ratio in extraction
        strip_to_solvent_ratio: Strip acid/organic ratio
        nitrate_conc: Aqueous nitrate concentration (M), required for solvating
            extractants such as TBP whose D is nitrate- rather than pH-driven
            (#195). Threaded to both sections.
        mechanism: Explicit extraction-mechanism override passed to
            REEDistribution ("cation_exchange" / "solvating"). None takes the
            mechanism from the extractant record (#195). Threaded to both
            sections, so a circuit never mixes mechanisms.
        strip_nitrate_conc: Aqueous nitrate concentration (M) in the
            stripping section. For a solvating extractant (TBP) the strip is
            what lowers the nitrate, so None (the default) resolves to
            ``min(nitrate_conc, DEFAULT_STRIP_NITRATE)``, 1 M: the bottom of
            the 1-6 M window the TBP coefficients are documented for (see the
            TBP record in ``extractants.yaml``). A water strip goes lower
            still, but below 1 M the correlation is extrapolated. For a
            cation-exchange extractant nitrate does not enter ``D`` and None
            resolves to ``nitrate_conc``. One nitrate used to reach every
            section, so a TBP circuit's strip D equalled its extraction D and
            nothing stripped (2026 operating-point audit, R8).
        capacity_sharpness: Sharpness k of the extraction section's smooth
            loading limiters; see REEExtractorParams (#193).
    """
    extractant: str
    elements: tuple[str, ...]
    diluent: str = "kerosene"
    n_extraction_stages: int = 10
    n_stripping_stages: int = 5
    # None means "read it off the D curves" (see __post_init__). The literals
    # 3.5 / 0.5 were pre-#270; the window fractions that replaced them (#270)
    # stripped at a pH where the heavy REE stayed on the solvent (audit R1).
    extraction_pH: float | None = None
    stripping_pH: float | None = None
    extractant_conc: float = 0.5
    solvent_to_feed_ratio: float = 1.0
    strip_to_solvent_ratio: float = 0.5
    nitrate_conc: float | None = None  # see #195
    strip_nitrate_conc: float | None = None  # audit R8

    #: Default strip nitrate (M) for a solvating extractant (audit R8).
    DEFAULT_STRIP_NITRATE = 1.0
    mechanism: str | None = None  # see #195
    capacity_sharpness: int = 8  # see REEExtractorParams (#193)

    def __post_init__(self):
        """Resolve unset pHs from the D curves of this circuit's elements.

        A pH left as None is read off the extractant's ``D`` curves for the
        elements this circuit carries, at the phase ratios its units really
        run at (:mod:`difflow_ree.equilibrium.operating_points`): extraction
        where the least extractable element has ``D * (O/A) = 10``, stripping
        where the most strongly held one has ``D * (O/A) = 0.1``. The window
        fractions of :func:`~difflow_ree.database.default_pH` (#270) used
        before left 67 % of the Sm and 99.8 % of the Y on the barren organic
        of a default D2EHPA circuit (2026 operating-point audit, R1). A pH
        the caller gives is kept as given. An extractant whose ``D`` does not
        move with pH (TBP) keeps the window default, which then has no effect
        on ``D``.
        """
        from difflow_ree.database import default_pH

        # (audit R8) A solvating extractant strips by lowering the nitrate;
        # one nitrate for every section made strip D equal extraction D.
        if self.strip_nitrate_conc is None:
            from difflow_ree.database import get_extractant

            solvating = (self.mechanism or get_extractant(self.extractant).mechanism) == "solvating"
            self.strip_nitrate_conc = (
                min(self.nitrate_conc, self.DEFAULT_STRIP_NITRATE)
                if solvating and self.nitrate_conc is not None
                else self.nitrate_conc)
        from difflow_ree.equilibrium.operating_points import (
            circuit_phase_ratios, cut_pHs)

        if self.extraction_pH is None or self.stripping_pH is None:
            ratios = circuit_phase_ratios(
                self.solvent_to_feed_ratio, None, self.strip_to_solvent_ratio,
                self.extractant_conc)
            cuts = cut_pHs(
                self.extractant, tuple(self.elements),
                extraction_OA=ratios["extraction"], strip_OA=ratios["stripping"],
                extractant_conc=self.extractant_conc,
                nitrate_conc=self.nitrate_conc, mechanism=self.mechanism)
            if self.extraction_pH is None:
                self.extraction_pH = (cuts.extraction if cuts is not None
                                      else default_pH(self.extractant, "extraction"))
            if self.stripping_pH is None:
                self.stripping_pH = (cuts.stripping if cuts is not None
                                     else default_pH(self.extractant, "stripping"))


class ExtractStripCircuit:
    """Basic extract-strip circuit.

    Simple two-section solvent extraction circuit for
    recovering REE from aqueous solution.

    Example:
        >>> params = ExtractStripParams(
        ...     extractant="D2EHPA",
        ...     elements=("La", "Ce", "Nd", "Dy"),
        ...     n_extraction_stages=10,
        ...     n_stripping_stages=5,
        ... )
        >>> circuit = ExtractStripCircuit(params)
        >>> results = circuit(feed)
        >>> print(results["recovery"])
    """

    symbol = "Extract-Strip"
    equations = [
        r"\text{Aqueous feed} \xrightarrow{\text{Extract (pH 3-4)}} \text{loaded organic} \xrightarrow{\text{Strip (pH<1)}} \text{pregnant aqueous}",
        r"\mathrm{recovery} = 1 - \frac{[\mathrm{RE}]_\mathrm{raffinate}}{[\mathrm{RE}]_\mathrm{feed}}",
    ]
    assumptions = [
        "Two counter-current sections (extraction + stripping) in series.",
        "Constant phase ratios; no bleed/recycle.",
    ]
    references = ["Xie, F., Zhang, T.A., Dreisinger, D., Doyle, F. Miner. Eng., 56, 10 (2014)."]
    parameter_symbols = {
        "n_extraction_stages": "N_E",
        "n_stripping_stages": "N_S",
        "solvent_to_feed_ratio": "O/A_E",
        "strip_to_solvent_ratio": "A/O_S",
    }
    parameter_units = {
        "n_extraction_stages": "-",
        "n_stripping_stages": "-",
        "extraction_pH": "-",
        "stripping_pH": "-",
        "extractant_conc": "mol/L",
        "solvent_to_feed_ratio": "-",
        "strip_to_solvent_ratio": "-",
        "nitrate_conc": "mol/L",
        "strip_nitrate_conc": "mol/L",  # audit R8
        "capacity_sharpness": "-",
    }

    def __init__(self, params: ExtractStripParams):
        """Initialize circuit.

        Args:
            params: Circuit parameters
        """
        self.params = params

        # Create extraction section
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

        # Create stripping section
        self._stripper = REEStripper(StripperParams(
            n_stages=params.n_stripping_stages,
            extractant=params.extractant,
            elements=params.elements,
            diluent=params.diluent,
            pH=params.stripping_pH,
            extractant_conc=params.extractant_conc,
            nitrate_conc=params.strip_nitrate_conc,  # audit R8
            mechanism=params.mechanism,  # see #195
        ))

    def __call__(
        self,
        feed: Stream,
        T: Array | float = 298.15,
    ) -> dict:
        """Run extract-strip circuit.

        Args:
            feed: Aqueous REE feed solution
            T: Operating temperature (K)

        Returns:
            Dictionary with:
            - raffinate: Depleted aqueous from extraction
            - product: REE product solution from stripping
            - barren_organic: Stripped organic (for recycle)
            - recovery: Overall REE recovery
            - extraction_info: Extraction section details
            - stripping_info: Stripping section details
        """
        p = self.params
        T = jnp.asarray(T)

        feed_flows = get_flows(feed)
        F_aq = feed_flows.get("H2O", 1.0)

        # Create solvent stream
        F_org = F_aq * p.solvent_to_feed_ratio
        solvent_flows = {
            p.diluent: F_org,
            p.extractant: p.extractant_conc * F_org,
        }
        for elem in p.elements:
            solvent_flows[elem] = 0.0
        solvent = make_stream(solvent_flows, T, feed["P"])

        # Extraction
        raffinate, loaded_org, ext_info = self._extractor(
            feed, solvent, T, pH=p.extraction_pH
        )

        # Create strip acid stream
        F_strip = F_org * p.strip_to_solvent_ratio
        strip_flows = {"H2O": F_strip}
        for elem in p.elements:
            strip_flows[elem] = 0.0
        strip_acid = make_stream(strip_flows, T, feed["P"])

        # Stripping
        product, barren_org, strip_info = self._stripper(
            loaded_org, strip_acid, T, pH=p.stripping_pH
        )

        # Calculate overall recovery
        product_flows = get_flows(product)
        total_feed = sum(float(feed_flows.get(e, 0.0)) for e in p.elements)
        total_product = sum(float(product_flows.get(e, 0.0)) for e in p.elements)
        overall_recovery = safe_divide(total_product, total_feed)

        # Element-wise recovery
        element_recovery = {}
        for elem in p.elements:
            f_in = float(feed_flows.get(elem, 0.0))
            f_out = float(product_flows.get(elem, 0.0))
            element_recovery[elem] = safe_divide(f_out, f_in)

        # Mass balance verification. Every stream that leaves the circuit
        # counts, the barren organic included: REE the stripper does not
        # remove leaves on the solvent, and omitting it reported that as a
        # mass loss (Nd closure 0.94 on a D2EHPA circuit stripped at pH 0).
        raff_flows = get_flows(raffinate)
        barren_flows = get_flows(barren_org)
        feed_total = {
            elem: jnp.asarray(float(feed_flows.get(elem, 0.0)))
            for elem in p.elements
        }
        product_total = {
            elem: (
                float(product_flows.get(elem, 0.0))
                + float(raff_flows.get(elem, 0.0))
                + float(barren_flows.get(elem, 0.0))
            )
            for elem in p.elements
        }
        mass_closure = {
            elem: safe_divide(product_total[elem], feed_total[elem])
            for elem in p.elements
        }

        return {
            "raffinate": raffinate,
            "product": product,
            "barren_organic": barren_org,
            "recovery": overall_recovery,
            "element_recovery": element_recovery,
            "extraction_info": ext_info,
            "stripping_info": strip_info,
            "mass_balance": {
                "feed": feed_total,
                "output": product_total,
                "closure": mass_closure,
            },
        }

    def optimize_stages(
        self,
        feed: Stream,
        target_recovery: float = 0.99,
        max_extraction_stages: int = 20,
        max_stripping_stages: int = 10,
    ) -> dict:
        """Find minimum stages for target recovery.

        Args:
            feed: Feed stream
            target_recovery: Desired overall recovery
            max_extraction_stages: Maximum extraction stages to try
            max_stripping_stages: Maximum stripping stages to try

        Returns:
            Optimal configuration
        """
        best_config = None
        min_total_stages = float('inf')

        for n_ext in range(3, max_extraction_stages + 1):
            for n_strip in range(2, max_stripping_stages + 1):
                # Update params temporarily
                old_ext = self.params.n_extraction_stages
                old_strip = self.params.n_stripping_stages
                self.params.n_extraction_stages = n_ext
                self.params.n_stripping_stages = n_strip

                # Rebuild units
                self.__init__(self.params)

                # Run circuit
                results = self(feed)

                # Check if target met
                if results["recovery"] >= target_recovery:
                    total = n_ext + n_strip
                    if total < min_total_stages:
                        min_total_stages = total
                        best_config = {
                            "n_extraction_stages": n_ext,
                            "n_stripping_stages": n_strip,
                            "total_stages": total,
                            "recovery": results["recovery"],
                        }

                # Restore
                self.params.n_extraction_stages = old_ext
                self.params.n_stripping_stages = old_strip

        return best_config


def design_extract_strip(
    feed_composition: dict[str, float],
    extractant: str,
    target_recovery: float = 0.99,
    extraction_pH: float | None = None,
    nitrate_conc: float | None = None,
    mechanism: str | None = None,
    stripping_pH: float | None = None,
    max_extraction_stages: int = 40,
    max_stripping_stages: int = 20,
) -> ExtractStripParams:
    """Design an extract-strip circuit that meets a recovery target.

    The pHs come off the extractant's D curves for the feed's elements
    (:func:`difflow_ree.equilibrium.operating_points.cut_pHs`), and both stage
    counts are sized from the D values at those pHs through the same Kremser
    fractions the units evaluate, at the phase ratios the units really run at.
    The smallest total stage count whose predicted overall recovery (product
    over feed, feed-weighted) meets ``target_recovery`` is returned.

    This replaced a design that read the extraction pH off the window
    fraction, sized only the extraction and set ``n_strip = max(3, n_ext //
    2)`` whatever the strip D (2026 operating-point audit, R1). On D2EHPA it
    returned 3/3 stages at a strip pH where the heavy REE stayed on the
    solvent.

    Args:
        feed_composition: Element flows in feed (mol/s); only the ratios matter.
        extractant: Extractant to use.
        target_recovery: Target overall recovery fraction into the product.
        extraction_pH: Operating pH; None reads it off the D curves.
        nitrate_conc: Aqueous nitrate concentration (M), required for
            solvating extractants such as TBP (#195).
        mechanism: Explicit mechanism override; see REEDistribution (#195).
        stripping_pH: Strip pH; None reads it off the D curves.
        max_extraction_stages: Largest extraction stage count searched.
        max_stripping_stages: Largest stripping stage count searched.

    Returns:
        ExtractStripParams with the pHs and stage counts.

    Warns:
        UserWarning: If no stage count up to the maxima meets the target (the
            best design found is returned).

    Example:
        >>> p = design_extract_strip({"Nd": 0.01, "Dy": 0.001}, "PC88A")
        >>> p.n_extraction_stages >= 1
        True
    """
    import warnings

    import numpy as np

    from difflow_ree.equilibrium.distribution import REEDistribution
    from difflow_ree.equilibrium.operating_points import (
        kremser_fraction, section_factors)

    elements = tuple(feed_composition.keys())
    # Resolve the pHs the way the circuit would (the cut rule), keeping any
    # the caller gave.
    template = ExtractStripParams(
        extractant=extractant, elements=elements,
        extraction_pH=extraction_pH, stripping_pH=stripping_pH,
        nitrate_conc=nitrate_conc, mechanism=mechanism)
    from difflow_ree.equilibrium.operating_points import circuit_phase_ratios

    oa = circuit_phase_ratios(template.solvent_to_feed_ratio, None,
                              template.strip_to_solvent_ratio,
                              template.extractant_conc)
    dist = REEDistribution(
        extractant, elements, concentration=template.extractant_conc,
        nitrate_conc=nitrate_conc, mechanism=mechanism,
        on_out_of_range="ignore")
    E = section_factors(dist, elements, [template.extraction_pH], oa["extraction"])[0]
    S = 1.0 / section_factors(dist, elements, [template.stripping_pH], oa["stripping"])[0]
    f = np.array([float(feed_composition[e]) for e in elements])
    f = f / f.sum()

    n_ext = np.arange(1, max_extraction_stages + 1)
    n_str = np.arange(1, max_stripping_stages + 1)
    extracted = 1.0 - kremser_fraction(E[None, :], n_ext[:, None])   # (Ne, el)
    stripped = 1.0 - kremser_fraction(S[None, :], n_str[:, None])    # (Ns, el)
    recovery = np.einsum("ae,be,e->ab", extracted, stripped, f)
    total = n_ext[:, None] + n_str[None, :]
    ok = recovery >= target_recovery
    if ok.any():
        cost = np.where(ok, total, np.iinfo(int).max)
        i, j = np.unravel_index(np.argmin(cost), cost.shape)
    else:
        i, j = np.unravel_index(np.argmax(recovery), recovery.shape)
        warnings.warn(
            f"design_extract_strip: no design up to {max_extraction_stages} + "
            f"{max_stripping_stages} stages reaches recovery {target_recovery} "
            f"on {extractant}; the best found predicts {recovery[i, j]:.4f}.",
            UserWarning, stacklevel=2)

    return ExtractStripParams(
        extractant=extractant,
        elements=elements,
        n_extraction_stages=int(n_ext[i]),
        n_stripping_stages=int(n_str[j]),
        extraction_pH=template.extraction_pH,
        stripping_pH=template.stripping_pH,
        nitrate_conc=nitrate_conc,  # see #195
        mechanism=mechanism,  # see #195
    )
