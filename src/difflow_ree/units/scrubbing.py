"""REE scrubbing section unit operation.

Scrubbing removes unwanted elements (impurities or lighter REE)
from the loaded organic phase by contacting with a scrub solution.

The scrub solution is typically:
- Dilute acid (to maintain pH)
- REE solution (to displace lighter REE)
- Pure water (simple wash)

All operations are fully differentiable using JAX.
"""

import warnings
from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from jax import Array

from difflow.numerics import safe_divide
from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, make_stream, get_flows
from difflow_ree.equilibrium.distribution import REEDistribution
from difflow_ree.units.kremser import kremser_two_inlet
from difflow_ree.units.stripping import acid_consumption


class ScrubTypeDeprecationWarning(UserWarning):
    """``ScrubberParams.scrub_type`` is deprecated and never had an effect (#288).

    The field was declared as ``Literal["acid", "ree", "water"]`` and the
    class documented a "scrub-type-dependent boundary condition", but
    ``REEScrubber.__call__`` never read it: all three values gave bit-identical
    outlet streams. There is nothing left for it to select, because everything
    it claimed to switch on is already carried by arguments the scrubber does
    read:

    * ``"acid"`` vs. ``"water"`` is the scrub solution's acid strength, which
      is ``pH`` (and, for a solvating extractant, ``nitrate_conc``).
    * ``"ree"`` -- a scrub solution that carries target REE, as in a strip
      liquor refluxed to the scrub end -- is the REE content of the
      ``scrub_solution`` stream, which the two-inlet Kremser solve takes as a
      boundary condition through ``F_scrub_in`` (#284).

    So a mode flag could only ever contradict the stream it was describing.
    Setting it raises this warning; the field will be removed.
    """


@dataclass(repr=False)
class ScrubberParams(ParamsMixin):
    """Parameters for REE scrubbing section.

    Attributes:
        n_stages: Number of scrubbing stages
        extractant: Extractant name
        elements: REE elements to track
        target_elements: Elements the scrub is meant to retain in the organic.
            REPORTING ONLY (#288): it labels the diagnostics and does not
            enter the calculation. Every element in ``elements`` is scrubbed
            by its own D through the same two-inlet Kremser solve, so
            changing this changes ``info["target_retained"]`` /
            ``info["impurity_removed"]`` / the ``"is_target"`` flags and
            nothing else -- the outlet streams are identical. Optional;
            defaults to no labelling. Must be a subset of ``elements``,
            which is checked rather than silently dropping a name that is
            not tracked. What actually decides which elements stay in the
            organic is ``pH`` (and ``n_stages`` and the phase ratio).
        diluent: Organic diluent name (e.g., "kerosene", "n-dodecane")
        pH: Scrub solution pH (lower pH strips more)
        extractant_conc: Extractant concentration (M)
        scrub_type: DEPRECATED and ignored (#288). It never entered the
            calculation and there is nothing for it to select; see
            :class:`ScrubTypeDeprecationWarning`. Setting it warns.
        nitrate_conc: Aqueous nitrate concentration (M), required for solvating
            extractants such as TBP whose D is nitrate- rather than pH-driven
            (#195)
        mechanism: Explicit extraction-mechanism override passed to
            REEDistribution ("cation_exchange" / "solvating"). None takes the
            mechanism from the extractant record (#195).
        coefficient_overrides: Per-element replacements for the tabulated
            log10(D) correlation coefficients, ``{element: {"a": ...}}``,
            passed straight to
            :class:`~difflow_ree.equilibrium.distribution.REEDistribution`.
            May be JAX tracers: this is the supported way to put an
            uncertainty distribution on D and differentiate through it. See
            that class for the full description.
    """
    n_stages: int | float | Array
    extractant: str
    elements: tuple[str, ...]
    # Reporting label only -- see the Attributes entry above and #288. Kept
    # in its original positional slot so existing positional calls still work.
    target_elements: tuple[str, ...] = ()
    diluent: str = "kerosene"
    # (#270) None means "the extractant record's own default scrubbing pH", a
    # quarter of the way up its fitted validity window -- lower than the
    # extraction default, which is what a scrub needs, but read off the
    # record rather than pinned to a literal that only suited one of them.
    pH: float | Array | None = None
    extractant_conc: float = 0.5
    scrub_type: Literal["acid", "ree", "water"] | None = None  # deprecated, #288
    nitrate_conc: float | None = None  # see #195
    mechanism: str | None = None  # see #195
    # Per-element log10(D) coefficient overrides, possibly traced; passed to
    # REEDistribution. The supported way to put uncertainty on D.
    coefficient_overrides: dict | None = None

    def __post_init__(self):
        """Resolve the record's pH default (#270); check the labels (#288)."""
        if self.pH is None:
            from difflow_ree.database import default_pH

            self.pH = default_pH(self.extractant, "scrubbing")

        # (#288) target_elements is a reporting label, so a name that is not
        # tracked cannot show up as a wrong number -- it shows up as nothing
        # at all, an empty target_retained that reads like "no target was
        # retained". Reject it here instead: the student who hit this had
        # asked which elements go in target_elements, and got silence either
        # way.
        if isinstance(self.target_elements, str):
            raise TypeError(
                "ScrubberParams.target_elements must be a tuple of element "
                f"names, not the string {self.target_elements!r}; write "
                f'("{self.target_elements}",)'
            )
        self.target_elements = tuple(self.target_elements)
        unknown = [e for e in self.target_elements if e not in tuple(self.elements)]
        if unknown:
            raise ValueError(
                f"ScrubberParams.target_elements {unknown} are not in "
                f"elements {tuple(self.elements)}. target_elements only "
                "labels the scrubbing diagnostics (#288), so an untracked "
                "name would silently label nothing; add them to elements or "
                "drop them from target_elements."
            )

        if self.scrub_type is not None:
            allowed = ("acid", "ree", "water")
            if self.scrub_type not in allowed:
                raise ValueError(
                    f"ScrubberParams.scrub_type {self.scrub_type!r} is not one "
                    f"of {allowed}."
                )
            warnings.warn(
                "ScrubberParams.scrub_type is deprecated and ignored (#288): "
                "it never entered the calculation, and all three values give "
                "identical outlets. Set the scrub's acid strength with pH "
                "(and nitrate_conc for a solvating extractant), and pass a "
                "REE-bearing scrub_solution stream for a reflux scrub -- the "
                "two-inlet Kremser solve takes that as a boundary condition "
                "(#284). Remove the argument.",
                ScrubTypeDeprecationWarning,
                stacklevel=3,
            )


class REEScrubber:
    """Multi-stage REE scrubbing section.

    Removes unwanted (typically lighter) REE from the loaded organic
    by contacting with an acidic scrub solution.

    In a typical flowsheet:
    - Extract at pH 3-4: All REE go to organic
    - Scrub at pH 2: Light REE return to aqueous, heavy REE stay

    What sets the split is ``pH`` (through each element's own D), ``n_stages``
    and the scrub/organic phase ratio. ``target_elements`` labels the
    diagnostics and changes no flow (#288).

    Note:
        The outlets are rebuilt from ``{extractant, diluent} | elements``
        plus the aqueous carrier, so ``elements`` must name every REE present
        in either inlet: an REE that is not tracked is dropped from both
        outlets and the section does not conserve it. The same is true of any
        other species carried along (a second extractant such as TBP, a
        modifier, the strip acid). ``info["dropped_species"]`` reports what
        was left behind.

    Example:
        >>> params = ScrubberParams(
        ...     n_stages=5,
        ...     extractant="D2EHPA",
        ...     elements=("La", "Ce", "Nd", "Dy"),
        ...     target_elements=("Nd", "Dy"),  # labels the diagnostics only
        ...     pH=2.0,
        ... )
        >>> scrubber = REEScrubber(params)
        >>> scrub_liquor, scrubbed_org, info = scrubber(loaded_org, scrub_soln)
    """

    symbol = "REE Scrubbing"
    equations = [
        r"\overline{\mathrm{REA}_3} + 3\,\mathrm{H}^+ \rightleftharpoons \mathrm{RE}^{3+}_{(aq)} + 3\,\overline{\mathrm{HA}}\qquad \text{(reverse extraction at low pH)}",
        r"\frac{x_{N+1}}{x_1} = \frac{E^{-1}-1}{E^{-(N+1)}-1}\qquad \text{(Kremser, counter-current, reversed)}",
    ]
    assumptions = [
        "Counter-current equilibrium stages with constant phase flows.",
        "The scrub aqueous is described by its pH and by its own REE content; "
        "there is no separate scrub-type mode (#288).",
        "Separation factor between target and non-target REE is pH-dominated.",
    ]
    references = [
        "Xie, F., Zhang, T.A., Dreisinger, D., Doyle, F. Miner. Eng., 56, 10 (2014).",
    ]
    parameter_symbols = {"n_stages": "N", "pH": r"\mathrm{pH}", "extractant_conc": "[HA]"}
    parameter_units = {
        "n_stages": "-",
        "pH": "-",
        "extractant_conc": "mol/L",
        "nitrate_conc": "mol/L",
    }
    numerical_method = (
        "Two-inlet Kremser (kremser_two_inlet) applied to reverse extraction: "
        "REE arriving on the scrub solution is a boundary condition of the "
        "solve, so a refluxed strip liquor is handled by the stream that is "
        "passed rather than by a mode switch (#284, #288)."
    )

    def __init__(self, params: ScrubberParams):
        """Initialize scrubber.

        Args:
            params: Scrubber parameters
        """
        self.params = params
        self._distribution = REEDistribution(
            extractant=params.extractant,
            elements=params.elements,
            concentration=params.extractant_conc,
            nitrate_conc=params.nitrate_conc,
            mechanism=params.mechanism,
            coefficient_overrides=params.coefficient_overrides,
        )

    def __call__(
        self,
        loaded_organic: Stream,
        scrub_solution: Stream,
        T: Array | float = 298.15,
        pH: Array | float | None = None,
    ) -> tuple[Stream, Stream, dict]:
        """Perform multi-stage scrubbing.

        Counter-current: Organic flows from extraction → stripping
                        Scrub solution flows opposite direction

        Args:
            loaded_organic: Organic from extraction (loaded with REE)
            scrub_solution: Aqueous scrub solution
            T: Temperature (K)
            pH: Scrub pH (overrides params if provided)

        Returns:
            scrub_liquor: Aqueous outlet (contains scrubbed impurities)
            scrubbed_organic: Organic outlet (purified, target REE retained)
            info: Scrubbing diagnostics, including ``dropped_species``: inlet
                species that appear in neither outlet because they are
                neither the extractant, the diluent, the aqueous carrier nor
                one of ``elements`` (#288).
        """
        p = self.params
        pH = pH if pH is not None else p.pH
        pH = jnp.asarray(pH)
        T = jnp.asarray(T)

        org_flows = get_flows(loaded_organic)
        scrub_flows = get_flows(scrub_solution)

        F_extractant = org_flows.get(p.extractant, 0.0)
        F_diluent = org_flows.get(p.diluent, 1.0)
        F_org = F_extractant + F_diluent
        F_scrub = scrub_flows.get("H2O", 1.0)

        # Get D values at scrub pH (lower than extraction)
        D_values = self._distribution.get_D_all(pH, T)

        n_stages = jnp.asarray(p.n_stages, dtype=jnp.float64)

        scrub_liquor_flows = {"H2O": F_scrub}
        scrubbed_org_flows = {p.extractant: F_extractant, p.diluent: F_diluent}
        scrub_efficiency = {}

        for elem in p.elements:
            D = D_values[elem]
            F_org_in = jnp.asarray(org_flows.get(elem, 0.0))
            F_scrub_in = jnp.asarray(scrub_flows.get(elem, 0.0))

            # Scrub factor S = F_scrub / (D * F_org): S > 1 favors scrubbing.
            # Reported as a diagnostic below; the actual outlet flows go
            # through the two-inlet Kremser solve (#284), which handles REE
            # arriving on the scrub solution -- from a refluxed strip liquor
            # -- rather than lumping it in with the organic-borne REE.
            S = safe_divide(F_scrub, D * F_org)
            S_Np1 = jnp.power(S, n_stages + 1)
            frac_in_org = jnp.where(
                jnp.abs(S - 1.0) < 1e-6,
                1.0 / (n_stages + 1),
                safe_divide(S - 1.0, S_Np1 - 1.0)
            )
            frac_in_org = jnp.clip(frac_in_org, 0.0, 1.0)

            F_scrub_out, F_org_out = kremser_two_inlet(
                D, F_scrub, F_org, F_scrub_in, F_org_in, n_stages
            )

            scrubbed_org_flows[elem] = F_org_out
            scrub_liquor_flows[elem] = F_scrub_out

            # Track scrubbing efficiency (how much was removed)
            scrub_efficiency[elem] = {
                "D": D,
                "fraction_scrubbed": 1 - frac_in_org,
                "is_target": elem in p.target_elements,
            }

        # Acid (H+) consumption (#115). The reverse extraction that moves REE
        # from organic to the scrub liquor consumes 3 H+ per mole REE:
        #   RE(A)3(org) + 3 H+ -> RE3+(aq) + 3 HA(org)
        # Track the H+ balance and the resulting scrub-liquor pH so acid make-up
        # / recycle can be sized.
        ree_scrubbed_total = jnp.asarray(0.0)
        for elem in p.elements:
            moved = jnp.asarray(org_flows.get(elem, 0.0)) - scrubbed_org_flows[elem]
            ree_scrubbed_total = ree_scrubbed_total + jnp.maximum(moved, 0.0)
        acid_consumed = acid_consumption(ree_scrubbed_total, stoichiometry=3)
        # H+ supplied by the scrub solution at its pH ([H+] = 10^-pH), over the
        # scrub volumetric flow (F_scrub used as the aqueous flow basis).
        h_plus_supplied = jnp.power(10.0, -pH) * F_scrub
        h_plus_remaining = jnp.maximum(h_plus_supplied - acid_consumed, 0.0)
        pH_final = -jnp.log10(
            jnp.maximum(h_plus_remaining, 1e-30) / jnp.maximum(F_scrub, 1e-30)
        )

        P = loaded_organic["P"]
        scrub_liquor = make_stream(scrub_liquor_flows, T, P)
        scrubbed_organic = make_stream(scrubbed_org_flows, T, P)

        # Calculate selectivity metrics
        target_retained = {}
        impurity_removed = {}
        for elem, eff in scrub_efficiency.items():
            if eff["is_target"]:
                target_retained[elem] = 1.0 - jnp.asarray(eff["fraction_scrubbed"])
            else:
                impurity_removed[elem] = jnp.asarray(eff["fraction_scrubbed"])

        # (#288) Everything that is neither a tracked element nor a carrier
        # is dropped when the outlets are rebuilt above, and mass is not
        # conserved across the section for it. Report it rather than leaving
        # the user to notice the shortfall downstream.
        kept = set(scrub_liquor_flows) | set(scrubbed_org_flows)
        dropped_species = tuple(
            sorted((set(org_flows) | set(scrub_flows)) - kept)
        )

        info = {
            "n_stages": n_stages,
            "pH": pH,
            "T": T,
            "D_values": D_values,
            "scrub_efficiency": scrub_efficiency,
            "target_retained": target_retained,
            "impurity_removed": impurity_removed,
            # Acid balance (#115)
            "ree_scrubbed_total": ree_scrubbed_total,
            "acid_consumed": acid_consumed,
            "h_plus_supplied": h_plus_supplied,
            "h_plus_remaining": h_plus_remaining,
            "pH_final": pH_final,
            # Inlet species that appear in neither outlet (#288).
            "dropped_species": dropped_species,
        }

        return scrub_liquor, scrubbed_organic, info


def optimal_scrub_pH(
    extractant: str,
    target_element: str,
    impurity_element: str,
    min_target_retention: float = 0.95,
    pH_range: tuple[float, float] = (1.0, 4.0),
    n_points: int = 50,
    nitrate_conc: float | None = None,
    mechanism: str | None = None,
) -> tuple[float, float, float]:
    """Find optimal scrub pH for separation.

    Finds pH that maximizes impurity removal while retaining target.

    Note:
        Only meaningful for a cation-exchange extractant. A solvating
        extractant's D is nitrate- rather than pH-driven (#195), so the scan is
        flat in pH and the returned pH carries no information.

    Args:
        extractant: Extractant name
        target_element: Element to keep in organic
        impurity_element: Element to remove
        min_target_retention: Minimum fraction of target to retain
        pH_range: pH range to search
        n_points: Number of evaluation points
        nitrate_conc: Aqueous nitrate concentration (M), required for solvating
            extractants (#195)
        mechanism: Explicit mechanism override; see REEDistribution (#195)

    Returns:
        Tuple of (optimal_pH, target_D, impurity_D)
    """
    dist = REEDistribution(
        extractant=extractant,
        elements=(target_element, impurity_element),
        nitrate_conc=nitrate_conc,
        mechanism=mechanism,
    )

    best_pH = pH_range[0]
    best_ratio = 0.0

    for pH in jnp.linspace(pH_range[0], pH_range[1], n_points):
        D_target = float(dist.get_D(target_element, pH))
        D_impurity = float(dist.get_D(impurity_element, pH))

        # Want high D for target (stays in organic)
        # Want low D for impurity (goes to aqueous)
        # Ratio = D_target / D_impurity should be maximized

        if D_target > 1.0:  # Target should stay in organic
            ratio = D_target / (D_impurity + 0.01)
            if ratio > best_ratio:
                best_ratio = ratio
                best_pH = float(pH)

    D_target_opt = float(dist.get_D(target_element, best_pH))
    D_impurity_opt = float(dist.get_D(impurity_element, best_pH))

    return best_pH, D_target_opt, D_impurity_opt
