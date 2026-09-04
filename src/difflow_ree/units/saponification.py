"""The saponifier: contacting organic with base before the cascade (#197).

Industrial rare-earth circuits neutralize 30 to 50% of the acidic extractant
in a dedicated contactor *before* the organic enters the cascade,

.. math::

    \\overline{\\mathrm{HA}} + \\mathrm{MOH}
        \\rightarrow \\overline{\\mathrm{MA}} + \\mathrm{H_2O}

rather than dosing base into every mixer, which causes local pH excursions
that precipitate hydroxides and stabilize emulsions.

WHY IT IS A UNIT AND NOT A NUMBER
---------------------------------
A saponification degree stated as a parameter is an assumption; a saponifier
is a *duty*. Putting the contactor on the flowsheet is what makes the reagent
bill and the effluent load fall out of the same balance the cascade already
solves, instead of being asserted afterwards. It also makes the two things a
plant actually gets wrong visible: base that does not reach the organic
(``base_utilization``), and a solvent that comes back from stripping still
carrying counter-ion, which is a credit against the fresh base.

WHY THE CONTACT IS STOICHIOMETRIC AND NOT AN EQUILIBRIUM
--------------------------------------------------------
Unlike the cascade, this contact is not an interesting equilibrium. A strong
base against an extractant of pKa 3-6 goes essentially to completion, and a
plant sizes the saponifier so that it does. So the unit is a *stoichiometric*
contactor -- every equivalent that reaches the organic neutralizes an
extractant equivalent, capped at full neutralization -- and all the
equilibrium physics stays in
:class:`~difflow_ree.equilibrium.saponification.SaponifiedSection`, where the
organic re-equilibrates with the aqueous phase and the degree becomes an
output again. Modelling the saponifier itself with mass action would need a
hydroxide species the reaction networks deliberately do not carry, and would
buy nothing: the answer is "it all reacts".

Counter-ion is conserved exactly: what the base brings in leaves either bound
to the organic or in the spent aqueous phase, to round-off, whatever the
utilization.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, get_flows, make_stream
from difflow_ree.database import get_extractant
from difflow_ree.economics.saponification import (
    BaseReagent,
    base_for_counter_ion,
    get_base,
)
from difflow_ree.equilibrium.saponification import INDUSTRIAL_DEGREE_RANGE
from difflow_ree.equilibrium.schema import REEStreamSchema, counter_ion_charge_of


@dataclass(repr=False)
class SaponifierParams(ParamsMixin):
    """Parameters for a :class:`Saponifier` (#197).

    Attributes:
        extractant: Extractant name, e.g. ``"D2EHPA"``. Must be an acidic
            (cation-exchange) extractant: a neutral solvating extractant has
            no proton to neutralize.
        elements: REE symbols the stream schema tracks. A stripped solvent
            normally carries none, but a partially loaded one does.
        saponification_degree: Target fraction of the extractant's
            exchangeable protons to neutralize, 0 to 1. May be a tracer: this
            is the circuit's primary manipulated variable, and the whole point
            of the unit is that a control or RTO layer can move it.
            ``0.30-0.50`` is the industrial range.
        counter_ion: Cation the base leaves behind: ``"Na"``, ``"NH4"``,
            ``"Mg"`` or ``"K"``.
        base: Reagent key, e.g. ``"NaOH"``, ``"NH3"``, ``"Mg(OH)2"``. None
            picks the default for the counter-ion.
        base_utilization: Fraction of the dosed base that actually reaches the
            organic phase. The remainder leaves with the spent aqueous stream
            and is still paid for and still discharged, which is exactly why
            it is a parameter rather than an assumption of 1.
        diluent: Organic diluent species key.
        anion: Aqueous anion species key, for the schema.
        monomers_per_component: Extractant monomers per extractant component
            -- 2 for a dimeric acidic organophosphorus extractant. None reads
            the basis off the extractant record (#191).

    Example:
        >>> p = SaponifierParams(extractant="D2EHPA", saponification_degree=0.35)
        >>> p["counter_ion"]
        'Na'
    """

    extractant: str
    elements: tuple[str, ...] = ()
    saponification_degree: float | Array = 0.35
    counter_ion: str = "Na"
    base: str | None = None
    base_utilization: float = 1.0
    diluent: str = "kerosene"
    anion: str = "Cl"
    monomers_per_component: float | None = None

    def __post_init__(self) -> None:
        """Validate the reagent and the target degree.

        Raises:
            ValueError: On an unknown extractant, a solvating extractant, a
                concrete degree outside [0, 1], or a utilization outside
                (0, 1].
        """
        ext = get_extractant(self.extractant)
        if ext.mechanism != "cation_exchange":
            raise ValueError(
                f"Extractant {self.extractant!r} extracts by "
                f"{ext.mechanism!r} and has no acidic proton, so there is "
                f"nothing for a base to neutralize (#197)."
            )
        counter_ion_charge_of(self.counter_ion)
        if self.base is not None:
            reagent = get_base(self.base)
            if reagent.counter_ion != self.counter_ion:
                raise ValueError(
                    f"Base {self.base!r} carries the cation "
                    f"{reagent.counter_ion!r} but the saponifier declares "
                    f"counter_ion={self.counter_ion!r}. The base decides which "
                    f"cation ends up in the raffinate, so these cannot "
                    f"disagree (#197)."
                )
        if not 0.0 < float(self.base_utilization) <= 1.0:
            raise ValueError(
                f"base_utilization is the fraction of dosed base that reaches "
                f"the organic and must lie in (0, 1], got "
                f"{self.base_utilization} (#197)."
            )
        try:
            degree = float(self.saponification_degree)
        except Exception:  # pragma: no cover - tracer
            return
        if not 0.0 <= degree <= 1.0:
            raise ValueError(
                f"saponification_degree must lie in [0, 1], got {degree}. "
                f"Industrial circuits run at "
                f"{INDUSTRIAL_DEGREE_RANGE[0]}-{INDUSTRIAL_DEGREE_RANGE[1]} "
                f"(#197)."
            )


class Saponifier:
    """Contact organic with base to a target saponification degree (#197).

    Attributes:
        params: The :class:`SaponifierParams`.
        schema: The :class:`~difflow_ree.equilibrium.schema.REEStreamSchema`
            this unit reads and writes.
        base: The :class:`~difflow_ree.economics.saponification.BaseReagent`.
        monomers_per_component: Extractant monomers per component.

    Example:
        >>> sap = Saponifier(SaponifierParams(extractant="D2EHPA",
        ...                                   saponification_degree=0.35))
        >>> organic = sap.schema.make_organic(0.5, diluent_flow=4.0)
        >>> out, spent, info = sap(organic)
        >>> f"{float(info['saponification_degree']):.4f}"
        '0.3500'
        >>> f"{float(out['F_Na_org']):.6f}"
        '0.087500'
    """

    symbol = "Saponifier"
    equations = [
        r"\overline{\mathrm{HA}} + \mathrm{MOH} \rightarrow \overline{\mathrm{MA}} + \mathrm{H_2O}",
        r"n_{\mathrm{eq}} = S\,F_{\mathrm{ext}}/m",
        r"n_{\mathrm{base}} = n_{\mathrm{eq}} / (z_{\mathrm{base}}\,\eta)",
    ]
    assumptions = [
        "Strong base against a weak organic acid: the contact goes to completion.",
        "Neutralization is capped at the extractant inventory (S <= 1).",
        "Base that does not reach the organic leaves in the spent aqueous phase.",
        "No extractant is lost to the aqueous phase and no third phase forms.",
    ]
    references = [
        "Banda, R., Jeon, H., Lee, M. J. Ind. Eng. Chem. 21, 436 (2015). doi:10.1016/j.jiec.2014.03.002",
        "Liao, C. et al. J. Rare Earths 31, 331 (2013). doi:10.1016/S1002-0721(12)60281-6",
    ]
    numerical_method = "Closed-form stoichiometry; no iteration."

    def __init__(self, params: SaponifierParams):
        """Build the schema and resolve the reagent.

        Args:
            params: Saponifier parameters.
        """
        self.params = params
        self.schema = REEStreamSchema(
            elements=tuple(params.elements) or ("Nd",),
            extractant=params.extractant,
            diluent=params.diluent,
            counter_ion=params.counter_ion,
            anion=params.anion,
        )
        self.base: BaseReagent = (
            get_base(params.base) if params.base is not None
            else base_for_counter_ion(params.counter_ion)
        )
        if params.monomers_per_component is not None:
            self.monomers_per_component = float(params.monomers_per_component)
        else:
            ext = get_extractant(params.extractant)
            self.monomers_per_component = (
                2.0 if ext.stoichiometry_basis == "dimer" else 1.0
            )

    # -- duty ------------------------------------------------------------

    def extractant_equivalents(self, organic: Stream) -> Array:
        """Exchangeable extractant equivalents in a stream (eq/s).

        Args:
            organic: Organic stream.

        Returns:
            ``F_ext / m``: one equivalent per extractant component, which for
            a dimeric extractant is one per dimer, matching the single proton
            the dimer releases into the extracted complex.
        """
        f_ext = jnp.asarray(
            get_flows(organic).get(self.schema.extractant, 0.0),
            dtype=jnp.float64,
        )
        return f_ext / self.monomers_per_component

    def inlet_counter_ion(self, organic: Stream) -> Array:
        """Counter-ion already bound to the inlet organic (mol/s).

        A solvent returning from stripping is not always fully de-saponified,
        and what it carries is a credit against fresh base.

        Args:
            organic: Organic stream.

        Returns:
            Molar flow of bound counter-ion; zero when the stream declares
            none.
        """
        key = self.schema.organic_counter_ion
        return jnp.asarray(
            get_flows(organic).get(key, 0.0) if key else 0.0,
            dtype=jnp.float64,
        )

    def base_requirement(self, organic: Stream) -> Array:
        """Base needed to reach the target degree (mol/s of reagent).

        Args:
            organic: Organic stream entering the saponifier.

        Returns:
            Molar flow of the reagent, including the shortfall for
            ``base_utilization < 1``.
        """
        target = (
            jnp.asarray(self.params.saponification_degree, dtype=jnp.float64)
            * self.extractant_equivalents(organic)
        )
        have = self.inlet_counter_ion(organic) * self.schema.counter_ion_charge
        needed = jnp.maximum(target - have, 0.0)
        return needed / (
            self.base.equivalents_per_mole * float(self.params.base_utilization)
        )

    # -- the unit --------------------------------------------------------

    def __call__(
        self,
        organic: Stream,
        base_flow: Array | float | None = None,
    ) -> tuple[Stream, Stream, dict]:
        """Saponify an organic stream.

        Args:
            organic: Organic stream entering the saponifier. Its extractant
                key carries the **total** extractant on a monomer basis.
            base_flow: Reagent dosing (mol/s). None doses exactly
                :meth:`base_requirement`, which is the "set the degree" mode;
                pass a value to run the unit in the "this is what the pumps
                are doing" mode, in which the degree achieved is an output.

        Returns:
            ``(organic_out, spent_aqueous, info)``.

            ``organic_out`` is the inlet with its counter-ion key set to the
            bound counter-ion. ``spent_aqueous`` carries the counter-ion that
            did not transfer, its hydroxide, and the water the neutralization
            produced. ``info`` reports ``base_flow`` (mol/s),
            ``base_mass_flow`` (kg/s), ``base_equivalents_dosed`` and
            ``..._transferred``, ``saponification_degree`` achieved,
            ``counter_ion_in/out_organic/out_aqueous`` and
            ``counter_ion_imbalance``, which is zero to round-off by
            construction.
        """
        z = self.schema.counter_ion_charge
        eq_total = self.extractant_equivalents(organic)
        m_in = self.inlet_counter_ion(organic)
        eq_in = m_in * z

        dosed = (
            self.base_requirement(organic) if base_flow is None
            else jnp.asarray(base_flow, dtype=jnp.float64)
        )
        eq_dosed = dosed * self.base.equivalents_per_mole
        # Everything that reaches the organic reacts, up to full
        # neutralization: a strong base against a pKa 3-6 organic acid goes to
        # completion, and there is no more acid to take past S = 1.
        eq_transferred = jnp.minimum(
            eq_dosed * float(self.params.base_utilization),
            jnp.maximum(eq_total - eq_in, 0.0),
        )
        eq_out = eq_in + eq_transferred

        m_out_org = eq_out / z
        m_out_aq = dosed * self.base.counter_ion_per_mole - eq_transferred / z

        flows = dict(get_flows(organic))
        flows[self.schema.organic_counter_ion] = m_out_org
        organic_out = make_stream(flows, organic["T"], organic["P"])

        spent = make_stream(
            {
                self.schema.counter_ion: m_out_aq,
                # Un-transferred base leaves as free hydroxide, which is what
                # keeps the spent stream electroneutral.
                "OH": eq_dosed - eq_transferred,
                self.schema.water: eq_transferred,
            },
            organic["T"],
            organic["P"],
        )

        degree = eq_out / jnp.maximum(eq_total, 1e-300)
        info = {
            "base": self.base.name,
            "counter_ion": self.params.counter_ion,
            "base_flow": dosed,
            "base_mass_flow": dosed * self.base.molar_mass / 1000.0,
            "base_equivalents_dosed": eq_dosed,
            "base_equivalents_transferred": eq_transferred,
            "extractant_equivalents": eq_total,
            # An OUTPUT: it equals the target only while there is enough
            # extractant left to neutralize and enough base dosed to do it.
            "saponification_degree": degree,
            "saponification_degree_in": eq_in / jnp.maximum(eq_total, 1e-300),
            "counter_ion_in": m_in + dosed * self.base.counter_ion_per_mole,
            "counter_ion_out_organic": m_out_org,
            "counter_ion_out_aqueous": m_out_aq,
            "counter_ion_imbalance": (
                m_in + dosed * self.base.counter_ion_per_mole
                - m_out_org - m_out_aq
            ),
            "water_produced": eq_transferred,
        }
        return organic_out, spent, info
