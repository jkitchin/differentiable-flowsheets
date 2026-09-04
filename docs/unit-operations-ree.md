# Rare Earth Element (REE) Unit Operations

This document provides comprehensive documentation for the `difflow_ree` plugin, which provides specialized tools for modeling and optimizing rare earth element solvent extraction processes.

---

## Overview

The `difflow_ree` plugin provides:

- **Database** of 10 commercial REE properties (La, Ce, Pr, Nd, Sm, Eu, Gd, Tb, Dy, Y)
- **4 extractant systems**: D2EHPA, PC88A, Cyanex272, TBP
- **pH-dependent distribution coefficient models**
- **Loading and speciation corrections**
- **A mass-action equilibrium closure** with the reaction network carried as
  data, where pH is an output rather than a parameter -- see
  [Mass-Action Equilibrium Closure](mass-action-closure) (#196)
- **Unit operations**: extraction, scrubbing, stripping, precipitation
- **Pre-built flowsheet templates**
- **Economic analysis tools**

All operations are fully differentiable using JAX, enabling gradient-based optimization of separation processes.

---

## Installation

The REE plugin is included as an optional dependency:

```bash
pip install difflow[ree]
```

Or install with all extras:

```bash
pip install difflow[all]
```

---

## Database and Properties

(ree-element-database)=
### REE Element Database

Access REE properties using the database functions:

```python
from difflow_ree import get_element, list_ree_elements

# List available elements
print(list_ree_elements())
# ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Y']

# Get element properties
nd = get_element("Nd")
print(f"Atomic weight: {nd.atomic_weight}")
print(f"Ionic radius: {nd.ionic_radius} pm")
print(f"Price: ${nd.price_usd_kg}/kg")
```

#### Available Properties

| Property | Description | Units |
|----------|-------------|-------|
| `atomic_weight` | Atomic mass | g/mol |
| `ionic_radius` | Ionic radius (3+) | pm |
| `price_usd_kg` | Market price | USD/kg |
| `oxide_mw` | Oxide molecular weight | g/mol |
| `oxide_formula` | Oxide formula | - |

(extractant-database)=
### Extractant Database

Four industrial extractants are supported:

```python
from difflow_ree import get_extractant, list_extractants

print(list_extractants())
# ['D2EHPA', 'PC88A', 'Cyanex272', 'TBP']

d2ehpa = get_extractant("D2EHPA")
print(f"Full name: {d2ehpa.full_name}")
print(f"Reference concentration: {d2ehpa.reference_concentration} M")
```

| Extractant | Full Name | Primary Use |
|------------|-----------|-------------|
| D2EHPA | Di(2-ethylhexyl)phosphoric acid | Light/middle REE |
| PC88A | 2-ethylhexyl phosphonic acid mono-2-ethylhexyl ester | Middle REE |
| Cyanex272 | Bis(2,4,4-trimethylpentyl)phosphinic acid | Heavy REE, Co/Ni |
| TBP | Tri-n-butyl phosphate | Ce separation, nuclear |

Every extractant record declares a normalized **extraction mechanism**, and it
is the mechanism that decides which correlation drives `D` (#195):

| Mechanism | `type` values | Driving variable | Coefficient block |
|-----------|---------------|------------------|-------------------|
| `cation_exchange` | `acidic_phosphoric`, `acidic_phosphonic`, `acidic_phosphinic`, `acidic_carboxylic` | pH | `ph_coefficients` |
| `solvating` | `solvating_neutral` | nitrate concentration | `nitrate_coefficients` |

A record carries only the block its mechanism needs. TBP has **no**
`ph_coefficients` block (it was deleted — see below), so
`Extractant.ph_coefficients` is `dict | None`, and asking for TBP with
`mechanism="cation_exchange"` raises.

```python
from difflow_ree import get_extractant, normalize_mechanism

print(get_extractant("D2EHPA").mechanism)  # 'cation_exchange'
print(get_extractant("TBP").mechanism)     # 'solvating'
print(get_extractant("TBP").requires_nitrate, get_extractant("TBP").reference_nitrate)
# True 3.0
print(normalize_mechanism("acidic_phosphonic"))  # 'cation_exchange'
```

---

## Equilibrium Models

(distribution-coefficients)=
### Distribution Coefficients

The distribution coefficient D = [REE]_org / [REE]_aq is modeled as a function of the
mechanism's driving variable, temperature, and extractant concentration.

For a **cation-exchange** extractant (D2EHPA, PC88A, Cyanex272), the driving
variable is pH:

$$\log_{10}(D) = a + b \cdot pH + c \cdot pH^2 + \frac{\Delta H}{R \ln(10)} \left(\frac{1}{T} - \frac{1}{T_{ref}}\right)$$

```python
from difflow_ree import REEDistribution, get_distribution_coefficient

# Create distribution calculator
dist = REEDistribution(
    extractant="D2EHPA",
    elements=("La", "Ce", "Nd", "Dy"),
    concentration=0.5,  # M
)

# Get D value for Nd at pH 3.0
D_nd = dist.get_D("Nd", pH=3.0, T=298.15)
print(f"D(Nd) at pH 3.0: {D_nd:.2f}")

# Get all D values
D_all = dist.get_D_all(pH=3.0)
for elem, D in D_all.items():
    print(f"D({elem}): {D:.2f}")
```

(solvating-extractants)=
#### Solvating extractants (TBP)

A neutral extractant releases no protons; it extracts the neutral nitrate
complex, so its `D` rises with nitrate concentration and is comparatively flat
in pH. The correlation is referenced to `reference_nitrate` (3 M for TBP), so
`a` is $\log_{10}(D)$ **at** that concentration and `b` is the nitrate slope
$\mathrm{d}\log_{10}(D)/\mathrm{d}\log_{10}[\mathrm{NO_3^-}]$:

$$s = \log_{10}\!\left(\frac{[\mathrm{NO_3^-}]}{[\mathrm{NO_3^-}]_{ref}}\right),
\qquad \log_{10}(D) = a + b\,s + c\,s^2 + \frac{\Delta H}{R \ln(10)}\left(\frac{1}{T}-\frac{1}{T_{ref}}\right)$$

```python
tbp = REEDistribution(
    extractant="TBP", elements=("Nd",), nitrate_conc=3.0, concentration=1.1
)
print(float(tbp.get_D("Nd")))                    # at 3 M nitrate
print(float(tbp.get_D("Nd", nitrate_conc=6.0)))  # rises with nitrate
```

```{note}
**TBP's nitrate coefficients are refitted from primary literature (and are the
only literature-derived numbers in the extractant database).** D2EHPA, PC88A
and Cyanex272 remain **hand-tuned with no recorded source** — do not mistake one
for the other.

**Fit basis.** Kraikaew, Srinuttrakul & Chayavadhanakur (2005), "Solvent
Extraction Study of Rare Earths from Nitrate Medium by the Mixtures of TBP and
D2EHPA in Kerosene", *J. Metals, Materials and Minerals* **15**(2), 89-95
([S1](https://jmmm.material.chula.ac.th/index.php/jmmm/article/view/1345), no
DOI), Table 1 — 1.0 M TBP in kerosene, 0.2001 N free acidity, 35 ± 1 °C —
corrected to the record's reference (3.0 M NO₃⁻, 1.0 M TBP, 298.15 K) by

$$a = \log_{10}D_\mathrm{meas} + 3\log_{10}\!\frac{3.0}{6.06}
      - 2300\left(\frac{1}{308.15}-\frac{1}{298.15}\right)
    = \log_{10}D_\mathrm{meas} - 0.6658$$

The **6.06 M nitrate is derived, not reported**: it was computed from S1's ppm
feed table (RE nitrates plus 0.2 N free acid). That is the single largest
uncertainty — a 10% error in it shifts every `a` by 0.13 log units *in common*,
so relative selectivity survives but the absolute level does not.

**What this fixed.** Three defects in the previous, unsourced block:

| | Old (unsourced) | New (measured) |
|---|---|---|
| $D_\mathrm{Dy}/D_\mathrm{La}$ at the reference | 100 | **10.2** |
| $D_\mathrm{Dy}$ at the reference | 3.16 | **0.24** |
| Temperature coefficient `d` | negative (⇒ endothermic) | **+2300 K (⇒ exothermic)** |

At the reference the record now gives $D_\mathrm{La}=0.023$,
$D_\mathrm{Ce}=0.032$, $D_\mathrm{Pr}=0.060$, $D_\mathrm{Nd}=0.072$,
$D_\mathrm{Sm}=0.120$, $D_\mathrm{Eu}=0.138$, $D_\mathrm{Gd}=0.174$,
$D_\mathrm{Tb}=0.204$, $D_\mathrm{Dy}=0.240$, $D_\mathrm{Y}=0.214$ — every
value below 1, mean adjacent-pair separation factor 1.29 per unit atomic
number (La(57) -> Dy(66) is 9 steps, Pm included), which is why a TBP
separation needs very many stages.

**Per-element provenance.** La, Ce, Pr, Nd, Sm, Eu, Gd, Dy, Y are **derived**
from S1 Table 1. **Tb is interpolated** (log-linear in atomic number between Gd
and Dy) — no TBP-nitrate `D` for Tb exists in any retrieved source. **Y is a
special case**: its effective position slides from Ho-like at high acidity to
La-like at very low acidity (Poos & Wilhelm, ISC-695, 1954), so a single $a_Y$
is a fiction; the value recorded is the ~6 M, low-free-acid one.

**`b = 3.0` is stoichiometric, not fitted.** It is the 3 of RE(NO₃)₃ + 3 TBP.
No measured $\mathrm{d}\log D/\mathrm{d}\log[\mathrm{NO_3^-}]$ in a
neutral-salt system was found; the only corroboration is indirect — Ganesh &
Pandey (2019) measured the *TBP* order as 2.81 ($R^2 = 0.992$). The old
per-element 2.50–2.85 trend had no support and is removed.

**`d = +2300 K`: one measurement, nine assumptions.** From
$\Delta H_\mathrm{Sm} = -43.3$ kJ/mol (Ganesh & Pandey, *J. Rad. Nucl. Appl.*
**4**(2), 109-115 (2019); the DOI printed in the PDF, 10.18576/jrna/040205, is
**not registered in Crossref** — cite the [retrieved
PDF](https://www.naturalspublishing.com/files/published/q86xt25u4iq5o9.pdf))
via $d = -\Delta H/(2.303R)$. **Sm is measured; the other nine elements are
assumed equal to Sm and are no data.** That paper's Table 1 appears to have its
$\Delta H$ and slope columns transposed between its two rows, so
$\Delta H_\mathrm{TBP}$ is either −43.3 or −61.2 kJ/mol, i.e. $d = +2261$ or
$+3197$ K — a ±40% ambiguity that cannot be resolved from the paper. The
defensible range is **+2260 to +3200 K**. The *sign* is solid, and is
independently corroborated by Jorjani & Shahbazi, *Arab. J. Chem.* **9**,
S1532-S1539 (2016), [doi:10.1016/j.arabjc.2012.04.002](https://doi.org/10.1016/j.arabjc.2012.04.002),
whose extraction fell as T went 25 → 55 °C.

**Known gap — the highest-value follow-up.** Fidelis, "Temperature effect on the
extraction of lanthanides in the TBP-HNO₃ system", *J. Inorg. Nucl. Chem.*
**32**, 997-1003 (1970),
[doi:10.1016/0022-1902(70)80079-3](https://doi.org/10.1016/0022-1902(70)80079-3),
measured the whole series at 10/17/25/40 °C — exactly the per-element
$\Delta H$ assumed constant above. It is paywalled with no open-access copy and
was **not** used. Obtaining it would replace nine assumed `d` values with nine
measured ones.
```

```{warning}
**Validity window: `b = 3.0` holds only for nitrate supplied by a neutral
salt.** NH₄NO₃ / LiNO₃ / Ca(NO₃)₂ / Mg(NO₃)₂ / Al(NO₃)₃, **≤ 0.5 M free acid,
1–6 M NO₃⁻, 283–326 K.**

One `b` cannot cover more, and both counter-cases are measured:

* In **neat TBP with concentrated HNO₃** the slope is **~6**, not 3 (Topp &
  Weaver, ORNL-1811, 1954,
  [doi:10.2172/4398970](https://doi.org/10.2172/4398970), Tables II/IV,
  8.5–17.4 N).
* At **1.1 M TBP in HNO₃**, `D` actually *falls* with acidity, because the acid
  consumes the extractant as the TBP·HNO₃ adduct and starves the RE complex of
  free TBP — effective slope **~0 or negative** (Ganesh & Pandey, Fig. 4).

Two further caveats on the *level* (not the shape): the model's extractant term
uses **total** [TBP] while the measured cube law holds in **free** TBP; and S1's
solvent was at or over saturation (0.35 M organic loading needs 1.05 M TBP of
the 1.0 M available), so these are **loading-suppressed process** `D` values,
not trace `D`.

**Use for:** relative REE selectivity; why TBP needs many stages; trend and
sensitivity studies; the *direction* of the temperature and nitrate
dependences.

**Do not use for:** stage counts; solvent inventories; absolute recoveries;
HNO₃-supplied nitrate; loaded solvent; free acidity above ~0.5 M; regressing
equilibrium constants; or Y at any acidity other than S1's 0.2 N.
```

```{warning}
**`extractant_conc = 0.5` is meaningless for TBP.** Every Params class in
difflow_ree defaults to 0.5 M, which is a cation-exchange default. TBP's
`reference_concentration` is 1.0 M and its `concentration_exponent` is 3.0, so
0.5 M multiplies every `D` by $(0.5/1.0)^3 = 0.125$ — an 8x reduction, giving
$D_\mathrm{La} = 0.0029$ and $D_\mathrm{Nd} = 0.0091$. TBP is normally run at
~30% v/v, which for the recorded density (0.979 g/mL) and molecular weight
(266.31 g/mol) is **1.10 M** (neat TBP is 3.68 M). Pass
`concentration=1.1` / `extractant_conc=1.1` explicitly whenever you use TBP.
```

```{warning}
**Behaviour change (#195).** `REEDistribution(extractant="TBP", ...)` without a
`nitrate_conc` now **raises**. Before, it silently used TBP's `ph_coefficients`
block and modelled TBP as a weak cation exchanger, which gets the qualitative
dependence backwards. `nitrate_conc=0.0` raises for the same reason.

**TBP's `ph_coefficients` block has since been deleted**, so the opt-in that
used to reach it now **raises** as well:

    REEDistribution(extractant="TBP", elements=(...), mechanism="cation_exchange")
    # ValueError: ... 'TBP' ... carries no 'ph_coefficients' block ...

The block was removed because it was mechanistically indefensible, not merely
uncalibrated: it modelled TBP as a weak cation exchanger, but TBP's own record
has `pKa: null` and `stoichiometry.protons_released: 0` — there is *no proton to
exchange*. No retrieved source reports a pH-slope correlation for TBP, and the
block carried the same refuted 100× La-to-Dy spread as the old nitrate block.

The error message names TBP, says the block was deleted as mechanistically
unsupported, and points at the nitrate path. There is **no silent fall-back**
and no `AttributeError`/`KeyError` leaking out of a later `get_D`.

TBP in HNO₃ is a real system, but this database has no coefficients fitted for
it — and per the validity window above, `b = 3` would be wrong for it anyway
(ORNL-1811 measured ~6; Ganesh & Pandey measured ~0 or negative). Supply your
own via `create_custom_extractant()` rather than reaching for a deleted block.

Consequently `Extractant.ph_coefficients` is now `dict[str, PHCoefficients] |
None`. Any consumer must handle `None`; there is no empty-dict stand-in.
`ExtractantDatabase.add_element_to_extractant()` raises for a record with no
pH block rather than lazily creating one.
```

##### What "a medium its coefficients do not cover" actually means

Issue #195 asked for a raise when an extractant is used in a medium its
coefficients do not cover. There are **two distinct checks**, and it is worth
being precise about which is which, because only the second is a medium check:

1. **Driving-ion check (always on).** A `solvating` extractant's correlation is
   a function of `[NO3-]`, so that concentration must exist and be positive.
   `nitrate_conc=None` and `nitrate_conc=0.0` raise. A concrete array is checked
   at its minimum, so a per-stage profile containing a zero raises too.
2. **Medium check (only when you state a medium).** `REEDistribution` takes an
   optional `medium`, one of `AQUEOUS_MEDIA` (`"sulfate"`, `"chloride"`,
   `"nitrate"`, `"mixed"`). The extractant records carry exactly **one** medium
   constraint, `stoichiometry.requires_nitrate`, so exactly one thing is
   detected: a nitrate-requiring extractant declared to be operating in a
   medium that supplies no nitrate.

```python
from difflow_ree import REEDistribution

# Detected: TBP requires nitrate, chloride supplies none.
REEDistribution(extractant="TBP", elements=("Nd",),
                nitrate_conc=3.0, medium="chloride")   # ValueError

REEDistribution(extractant="TBP", elements=("Nd",),
                nitrate_conc=3.0, medium="nitrate")    # fine
```

Nothing in `data/extractants.yaml` declares a chloride or sulfate
*incompatibility* for the acidic extractants, so **no other medium combination
is rejected** — D2EHPA is accepted in every medium. `medium=None` (the default)
leaves the medium unstated and is not guessed at. If you want more than this,
the records have to carry more than `requires_nitrate`.

##### Which Params classes carry these fields

| Params class | `nitrate_conc` | `mechanism` |
|---|---|---|
| `REEExtractorParams` | yes | **no** |
| `MixerSettlerParams` | yes | **no** |
| `ScrubberParams` | yes | yes |
| `StripperParams` | yes | yes |
| `ExtractStripParams` | yes | yes |
| `SplitShellParams` | yes | yes |
| `ExtractScrubStripParams` | yes | yes |
| `SeparationTrainParams` | yes | yes |

`REEExtractorParams` and `MixerSettlerParams` each own a `REEDistribution` but
have **no** `mechanism` field, so the mechanism-override cannot reach an
extraction section: it takes the mechanism from the extractant record. The
flowsheets that contain an extraction section (`ExtractStripParams`,
`ExtractScrubStripParams`, `SeparationTrainParams`) therefore apply `mechanism`
to their scrubbing and stripping sections only. Adding a `mechanism` field to
`REEExtractorParams` and `MixerSettlerParams` would close the gap.

(activity-corrections)=
#### pH scale and activity corrections

`pH` is on the **concentration** scale, $pH = -\log_{10}[\mathrm{H^+}]$, matching
the header of `data/extractants.yaml`.

The tabulated correlations are *conditional constants*: their coefficients were
fitted at the ionic strength of their source experiments and already absorb the
activity coefficients there. So **`ionic_strength=None` (the default) is a
first-class option, not a fallback** — for a 2 to 4 M chloride leach liquor, a
conditional constant used at the liquor's own ionic strength is the standard and
defensible treatment, and every implemented activity model is out of range there.

Supplying `ionic_strength` requests a correction *to a different ionic strength*.
Because the $b \cdot pH$ term already carries the $[\mathrm{H^+}]^{-p}$ dependence
on the concentration scale, what remains is (#194)

$$D_\text{corr} = D \cdot \frac{\gamma_{\mathrm{RE^{3+}}}}{\gamma_{\mathrm{H^+}}^{\,p}},
\qquad p = \texttt{Extractant.stoichiometry\_protons}$$

not $\gamma_{\mathrm{RE^{3+}}}$ alone. For a solvating extractant $p = 0$ and
there is no proton term; its real ionic-strength dependence is a nitrate salting
effect on the *anion*, which an aqueous-cation model does not represent, so
difflow_ree says so rather than reusing the cation-exchange form silently.

```python
# Implemented activity models and their declared validity ranges
from difflow_ree import ACTIVITY_MODELS
print({k: v["max_ionic_strength"] for k, v in ACTIVITY_MODELS.items()})
# {'davies': 0.5, 'none': inf}

dist = REEDistribution(extractant="D2EHPA", elements=("Nd",))
dist.get_D("Nd", pH=3.0, ionic_strength=0.3)   # in range, silent
dist.get_D("Nd", pH=3.0, ionic_strength=3.0)   # UserWarning: outside Davies range

# Escalate to an error, or silence it, at construction time
REEDistribution(extractant="D2EHPA", elements=("Nd",), on_out_of_range="raise")
REEDistribution(extractant="D2EHPA", elements=("Nd",), on_out_of_range="ignore")

# Or state explicitly that no correction is wanted
REEDistribution(extractant="D2EHPA", elements=("Nd",), activity_model="none")
```

Bromley and SIT are deliberately **not** offered — difflow_ree does not carry
their ion-interaction parameters and will not invent them.

(davies-sign-inversion)=
##### Davies does not just lose accuracy above 0.5 M — it changes sign

```{danger}
Davies writes $\log_{10}\gamma = -A z^2 f(I)$ with
$f(I) = \sqrt{I}/(1+\sqrt{I}) - 0.3\,I$. That bracket is **not monotone**: it
peaks near $I = 0.4$ M and crosses zero at

$$I = 1.940363884733242\ \mathrm{M}$$

the root of $0.3 I + 0.3\sqrt{I} - 1 = 0$
(`speciation.DAVIES_SIGN_CHANGE_IONIC_STRENGTH`). Above it every Davies
$\gamma$ exceeds 1 and the correction
$\gamma_{\mathrm{RE}}/\gamma_{\mathrm{H}}^3 = 10^{-6Af}$ **inverts**:

| $I$ (M) | 0.1 | 1.0 | 1.9404 | 3.0 | 4.0 |
|---|---|---|---|---|---|
| $D_\text{corr}/D$ | 0.228 | 0.245 | 1.000 | **6.49** | **42.5** |

That is exactly the 2 to 4 M chloride regime #194 was filed about: raw Davies
does not merely extrapolate there, it multiplies `D` by 6.5.
```

The guard is therefore **arithmetic, not a warning**. The ionic strength handed
to the activity model is clamped at that model's `max_ionic_strength` (0.5 M for
Davies), so beyond the documented range the correction **saturates at its
end-of-range value instead of reversing**, and $\mathrm{d}D/\mathrm{d}I$ is
exactly zero there — the honest statement that the model carries no information
about that regime. Values inside the range are untouched.

```python
d = REEDistribution(extractant="D2EHPA", elements=("Nd",), on_out_of_range="ignore")
D0 = float(d.get_D("Nd", pH=3.0))
[float(d.get_D("Nd", pH=3.0, ionic_strength=I)) / D0 for I in (0.1, 1.0, 3.0)]
# [0.2280, 0.1560, 0.1560]   <- never above 1

# Raw, possibly inverted Davies is still available, but only on request:
e = REEDistribution(extractant="D2EHPA", elements=("Nd",),
                    on_out_of_range="ignore", extrapolate_activity_model=True)
float(e.get_D("Nd", pH=3.0, ionic_strength=3.0)) / D0     # 6.49
```

Why the clamp rather than refusing to trace, or an opt-in flag for traced
values: under `jit`, `grad` and `vmap` the ionic strength is an abstract tracer,
and **no** Python-level check can inspect it. Refusing to trace would break
gradient-based design studies outright; a per-tracer opt-in would leave the
inverted branch reachable by anyone who set the flag for an in-range sweep. A
clamp is arithmetic, so it holds identically for scalars, concrete arrays and
tracers. An inverted `D` is reachable only through
`extrapolate_activity_model=True`.

The reporting layer is separate from the guard, runs at the Python level, and
reports once per instance per condition, so it neither spams a stage loop nor
breaks `jit`/`grad`:

* a **concrete scalar or array** is inspected (arrays at their maximum) and
  reported against the model's range;
* an **abstract tracer** cannot be inspected, and that is itself reported —
  `on_out_of_range="raise"` raises on a traced ionic strength rather than
  silently skipping the check, which is what it used to do.

#### pH Dependence

Distribution coefficients are strongly pH-dependent. Higher pH generally increases extraction:

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt

dist = REEDistribution(extractant="D2EHPA", elements=("La", "Nd", "Dy"))

pH_range = jnp.linspace(1.0, 5.0, 50)
for elem in ["La", "Nd", "Dy"]:
    D_values = [float(dist.get_D(elem, pH)) for pH in pH_range]
    plt.semilogy(pH_range, D_values, label=elem)

plt.xlabel("pH")
plt.ylabel("Distribution Coefficient D")
plt.legend()
plt.grid(True)
```

(separation-factors)=
### Separation Factors

The separation factor SF = D1/D2 determines separation feasibility:

```python
from difflow_ree import REEDistribution

dist = REEDistribution(extractant="PC88A", elements=("Nd", "Pr"))

# Separation factor at pH 3.0
SF = dist.get_separation_factor("Nd", "Pr", pH=3.0)
print(f"SF(Nd/Pr) = {SF:.2f}")

# Find optimal pH for separation
opt_pH, max_SF = dist.optimal_pH_for_separation("Nd", "Pr", pH_range=(1.0, 5.0))
print(f"Optimal pH: {opt_pH:.2f}, Max SF: {max_SF:.2f}")
```


---

(mass-action-closure)=
## Mass-Action Equilibrium Closure (#196)

Everything above is the **correlation level (L1)**: `log10(D)` is evaluated at a
pH you specify, and loading and speciation are multiplicative corrections. This
section describes the **closed level (L2)**, where the stage solves conservation
laws for its own state instead of evaluating a correlation at specified
conditions.

Three limitations follow directly from having no closure, and none can be fixed
inside a correlation:

- **pH was a parameter, not a state.** Every extracted trivalent ion releases
  three protons, so a real cascade's pH profile is set by the extraction itself.
  A model that specifies pH per cascade cannot predict the profile.
- **Competitive loading was a correction rather than an outcome.** The elements
  share one finite extractant inventory. That should emerge from a single free
  extractant balance, not from multiplying independent `D` values by
  `(1 - theta)^3` (see #189, #190, #191).
- **Extractant selection was not physically grounded.** A fitted `D` cannot
  respond to loading or medium, which is exactly where the ordering between
  extractants actually changes.

### The reaction network is data

The design decision that determines whether this layer generalizes is that the
reaction network is carried as **data**, in
`src/difflow_ree/data/reaction_networks.yaml`. Cation exchange, saponified
cation exchange (#197), solvating extraction and anion exchange are rows in a
table, not four code paths.

Each network declares a **component** basis (a chemically independent set whose
totals are conserved) and the **species** formed from it, with integer
stoichiometry, a phase and one `log10 K`:

```yaml
cation_exchange_dimer:
  mechanism: cation_exchange
  extractant_basis: dimer
  components:
    - {name: "RE3+",  phase: aqueous, charge: 3,  role: rare_earth, per_element: true}
    - {name: "H+",    phase: aqueous, charge: 1,  role: proton}
    - {name: "M+",    phase: aqueous, charge: 1,  role: counter_ion}
    - {name: "X-",    phase: aqueous, charge: -1, role: anion}
    - {name: "(HA)2", phase: organic, charge: 0,  role: extractant}
  species:
    - name: "RE(HA2)3"
      phase: organic
      charge: 0
      per_element: true
      stoichiometry: {"RE3+": 1, "(HA)2": 3, "H+": -3}
      log10_K: null      # calibrated from the L1 correlation
```

Mass action is then `log10[S_j] = log10 K_j + sum_c nu_jc log10[C_c]`, and the
conserved total of component `c` over a stage is
`T_c = sum_j nu_jc [S_j] Q_phase(j)`, summed over every species including the
free components themselves.

Two things about this table are worth stating plainly:

- **Negative coefficients are normal.** `H+` appears with coefficient `-3`
  because the complex *releases* three protons. The `H+` component therefore
  means "proton in excess of the reference state in which the extractant is
  fully protonated", and a loaded organic phase carries a negative H component.
  That is exact bookkeeping, and it is what makes a recycled loaded solvent
  behave correctly.
- **Charge consistency is checked.** A species' declared charge must equal
  `sum_c nu_jc * charge_c`. A mistyped coefficient is otherwise invisible until
  the charge balance quietly drifts.

```python
from difflow_ree.equilibrium import list_networks, cation_exchange_network

list_networks()
# ['anion_exchange', 'cation_exchange_dimer', 'cation_exchange_monomer',
#  'solvating_nitrate']

net = cation_exchange_network("D2EHPA", ("Nd", "Dy"), calibration_pH=3.0)
print(net.describe())
```

Nothing in `mass_action.py` mentions cation exchange, which is checkable rather
than aspirational: selecting a solvating extractant selects a different row and
the same closure predicts different physics.

```python
from difflow_ree.equilibrium import MassActionParams, MassActionSection

tbp = MassActionSection(MassActionParams(
    n_stages=2, extractant="TBP", elements=("Nd", "Dy"),
    aqueous_volumetric_flow=1.0, organic_volumetric_flow=1.0,
    anion="NO3", extractant_conc=1.0,
))
tbp.network.name         # 'solvating_nitrate'
```

Two consequences fall out with no code change: the pH profile is *flat*,
because the complex `RE(NO3)3.3S` contains no proton, and `D` rises as the
**cube** of the free nitrate, because the anion is a conserved component that
the complex draws three of. The salting effect is a balance, not a correction.

```{warning}
The shipped networks declare a monovalent anion. Asking for `anion="SO4"`
raises rather than quietly running with the wrong charge: a divalent anion
needs its own network row with the charge, and for a solvating or
anion-exchange complex the stoichiometry, corrected.
```

#### How #197 (saponification) slots in

The counter-ion `M+` is already a conserved component in **every** shipped
network, even though nothing forms from it yet. Saponification adds exactly one
species row and no code:

```yaml
- name: "M(HA2)"
  phase: organic
  charge: 0
  stoichiometry: {"M+": 1, "(HA)2": 1, "H+": -1}
  log10_K: <saponification constant>
```

With that row present, sodium partitions between the phases, the saponification
degree becomes an *output* of the same component balances, and base addition is
an input through the M and H totals. `tests/ree/test_mass_action.py` exercises
exactly this: it adds the row and solves a section through the unchanged
`solve_section`.

### Unknowns, equations and how they are solved

**Unknowns per stage** are the natural logs of the free component
concentrations: free `[H+]`, free extractant, free anion, the counter-ion and
the aqueous concentration of each rare earth.

**Equations** are the component balances -- one per component, so the system is
square. The mass-action expressions are *substituted* rather than posed as
extra rows, which means they hold identically at every Newton iterate, not just
at convergence. Aqueous charge balance is then a *consequence* of the component
balances whenever the entering totals are electroneutral; it is reported as
`info["charge_imbalance"]` (a non-zero value is a statement about the feed, not
the solver) and can be used *in place of* the anion balance with
`anion_closure="charge"`.

Four choices, and why:

| Choice | Reason |
|---|---|
| **Solve at section scope**, not stage by stage | The whole section is one residual `r(z; theta, u) = 0` handed to `difflow.eo_solver.solve_residual_system`. The reverse-mode tape is constant size rather than proportional to stages times iterations; the section Jacobian falls out and is the object the linearization, back-off and estimation layers need; long-cascade conditioning becomes a residual-scaling question; and a recycle tear stops being a separate mechanism -- it is another row of `r`. |
| **Solve in log concentration** | Positivity is automatic (no clipping, so no dead gradient), the ten-plus orders of magnitude a real cascade spans stay conditioned, and mass action becomes *linear* in the unknowns. |
| **Initialize from the correlation** | Mass-action systems lose Newton from a poor start. The L1 Kremser profile is the starting point, which is what gives the correlation path a continuing purpose. |
| **Return soft failures** | One cannot raise from inside `vmap` or `scan`, so `solve_section` returns the solution *with* a residual norm and a boolean feasibility flag. No Python branch is ever taken on a traced value. |

**Globalization.** Undamped Newton from the correlation start proposes steps of
1e3 or more in log space, and `exp` of that is `inf` and then `NaN`. Neither
standard remedy is sufficient alone here: a damped monotone Newton stalls where
the linear model of a sum of exponentials is poor (notably near full
neutralization, where the proton total passes through zero), and
Levenberg-Marquardt alone converges to spurious least-squares minima on a
ten-element cascade. `difflow_ree.equilibrium.mass_action._globalize` runs
damped Newton, then a trust region, then damped Newton again -- each phase is
monotone or discarded, and each is a no-op if the previous one converged. The
whole globalization runs under `stop_gradient`; the answer and its derivative
come from the `optimistix` root find that follows.

**Tolerances.** `inner_tol` defaults to `1e-12` on the *scaled* (dimensionless)
component balances, and feasibility is declared at `1e-8`. Outer flowsheet
tolerances in difflow are 1e-6 to 1e-8, so the inner solve is four to six
orders tighter. Keep it that way: a loosely converged inner solve gives an
implicit-function gradient that is exact for the solution manifold but
inconsistent with the number the code actually returned, and the resulting
finite-difference disagreement is very hard to diagnose after the fact.

**Conservation is structural, not asymptotic.** The organic outlet is read from
the converged stage-0 organic phase and the aqueous outlet is formed as
`(everything in) - (organic out)`, componentwise on the tableau. Every component
therefore balances to floating-point round-off no matter how well the
equilibrium converged, and how well it converged is reported separately in
`info["residual_norm"]`. `info["equilibrium_closure"]` gives the
tolerance-sized gap against the aqueous phase the solve predicts, so the choice
is visible rather than hidden.

### Usage

```python
from difflow_ree.equilibrium import MassActionParams, MassActionSection

params = MassActionParams(
    n_stages=4,
    extractant="D2EHPA",
    elements=("Nd", "Dy"),
    # The closed model works in CONCENTRATIONS, so it needs the phase volumes
    # (L/s) that a flow ratio could stand in for at L1. There is no defensible
    # way to guess them from molar flows, so they are required.
    aqueous_volumetric_flow=1.0,
    organic_volumetric_flow=1.0,
    # NOT an operating specification: this is where the closed model and the
    # correlation are made to agree. The operating pH is an output.
    calibration_pH=3.0,
)
section = MassActionSection(params)

feed = section.schema.make_aqueous(
    {"Nd": 0.02, "Dy": 0.02}, acid=0.02, water=55.0
)
solvent = section.schema.make_organic(0.5, diluent_flow=4.0)

raffinate, extract, info = section(feed, solvent)

info["pH_profile"]      # an OUTPUT, one value per stage
info["theta"]           # organic loading fraction per stage
info["free_extractant"] # M, from the one shared balance
info["D"]               # per element, from the closed model
info["feasible"]        # boolean array -- consume with jnp.where, not `if`
info["residual_norm"]
info["charge_imbalance"]
```

`section.schema.make_aqueous` closes the anion by electroneutrality unless you
give one explicitly: a feed that is not electroneutral has no physical
realisation, and handing one to the closed model produces a free proton
concentration that silently absorbs the imbalance.

### Same interface, two things that are not hidden

`REEExtractor` reaches both levels, so cascade code does not have to know which
one it is running at:

```python
from difflow_ree import REEExtractor, REEExtractorParams

params = REEExtractorParams(
    n_stages=4, extractant="D2EHPA", elements=("Nd", "Dy"), pH=3.0,
)
raffinate, extract, info = REEExtractor(params)(feed, solvent)          # L1

closed = REEExtractor(params.update(
    model="mass_action",
    aqueous_volumetric_flow=1.0,
    organic_volumetric_flow=1.0,
))
raffinate, extract, info = closed(feed, solvent)                        # L2
closed.section          # the underlying MassActionSection
```

Two things genuinely differ between the levels and are deliberately **not**
hidden behind the shared interface.

**State width.** The closed model reads and writes an acid, counter-ion and
anion balance the correlation ignores. The vocabulary is declared once as a
superset in `difflow_ree.equilibrium.schema.REEStreamSchema` -- rare earths by
element, `H`, `Na`/`NH4`/`K`, `Cl`/`NO3`/`SO4`, water, extractant total (on a
**monomer** basis, free plus bound), loaded organic by element, co-extracted
acid, water in organic, and `T`. The correlation path passes through what it
does not use, as it always has.

**Degrees of freedom.** pH is an *input* to the correlation and an *output* of
the closed model, whose corresponding input is base addition (or, from #197,
saponification degree). A design specified at one level is therefore not
directly expressible at the other. Under `model="mass_action"` the `pH` field
becomes the *calibration* pH, and passing an explicit `pH` to the call raises
rather than being silently ignored.

The bridge is an explicit inverse problem:

```python
from difflow_ree.equilibrium import base_addition_for_pH, base_addition_bounds

b_lo, b_hi = base_addition_bounds(section, feed, solvent)
base, ok = base_addition_for_pH(section, feed, solvent, target_pH=2.5)

raffinate, extract, info = section(feed, solvent, base_addition=base)
float(info["pH"])          # 2.5
```

It is posed as an *augmented* root find -- the section's component balances
plus one extra unknown (the base rate) and one extra row ("the pH at this stage
equals the target") -- so `d(base)/d(pH*)` falls out of one implicit
differentiation, and it is `jax.grad`-able. Base addition is bounded: summing
the proton balance over the section shows there is no root at all once every
proton has been neutralized, and below, the counter-ion total cannot go
negative. A target outside those bounds comes back with `feasible=False` and
`b` clipped to the nearest bound; nothing is raised.

### Where the constants come from, and where the two levels part company

`log_K_from_correlation` inverts the L1 correlation at a stated reference
condition, which is the only source available in this repository. It therefore
inherits that source's provenance: **the `ph_coefficients` of D2EHPA, PC88A and
Cyanex272 are hand-tuned with no literature source** (see the header of
`data/extractants.yaml`), so a constant derived from them is illustrative, not
measured. Supply your own with `log10_K={"Nd": ..., "Dy": ...}` for design
numbers.

The calibration is exact only at the reference condition, and the residual
disagreement is a real statement about the correlation rather than a defect of
the closure. Mass action forces

```
d log10 D / d pH = protons_released = 3
```

while the tabulated pH slopes `b` are 2.20 to 2.90.
`correlation_ph_slope_defect(extractant, element)` returns `3 - b` so the gap
can be quoted rather than discovered:

```python
from difflow_ree.equilibrium import correlation_ph_slope_defect
correlation_ph_slope_defect("D2EHPA", "Nd")   # 0.55
```

Away from the calibration pH the two levels then differ by exactly
`(p - b)(pH - pH_ref) - c(pH^2 - pH_ref^2)`, which the test suite asserts to
seven digits.

### Validation

| Claim | Measured |
|---|---|
| Reduces to the correlation in the dilute limit | With rare-earth totals at `1e-6` of the free acid, `D` agrees with `REEDistribution.get_D` to better than **2e-5 relative**. What is left is not model error: it is the pH shift from the protons the trace extraction releases, and it scales exactly linearly with the dilution (a ten-fold more dilute feed gives a ten-fold smaller disagreement). |
| Independent check of the `[HA]` dependence | The correlation applies `n * log10(C/C_ref)` with `n = 3`; the closed model never sees `n` and gets the same dependence from three dimers in the tableau plus a free-extractant balance. Doubling the extractant moves both by exactly 8. |
| Every component conserved | To **machine precision** (`< 1e-15` relative), including the proton component, and including under a deliberately unconverged solve. |
| Gradients | `jax.test_util.check_grads` passes through the implicit solve; analytic and central-difference gradients agree to `1e-6` relative. `log10 K` is traced, so extractant selection is differentiable. |
| `jit` / `vmap` | Both work; a failing solve under `vmap` returns `feasible=False` rather than raising. |
| Conditioning | A six-element, eight-stage cascade whose concentrations span more than ten decades converges to a residual below `1e-10`. |
| pH responds to three protons per trivalent ion | The acid released equals `protons_released` times the rare earth extracted, to `1e-12` relative. |

An external benchmark worth reading for behaviour: Iloeje et al., *Environ. Sci.
Technol.* **53**, 8926 (2019), [doi:10.1021/acs.est.9b01718](https://doi.org/10.1021/acs.est.9b01718),
which poses rare-earth extraction as Gibbs energy minimization with activity
models in both phases.

### What is deliberately not modelled

Water dissociation and rare-earth hydrolysis (no `OH-` species), aqueous
complexation with the anion, non-idealities in either phase (the constants are
conditional constants at the medium's ionic strength -- the same convention the
correlations use, see #194), third-phase formation, and any temperature
dependence of `log10 K` beyond what the calibration point carries. Each of those
is a row in `reaction_networks.yaml` away, which is the point of carrying the
network as data.

---

## Unit Operations

(reeextractor)=
### REEExtractor

**Location**: `difflow_ree/units/extraction.py`

**Class**: `REEExtractor`

**Description**: Multi-stage counter-current extraction cascade using the Kremser equation.

#### Parameters

```python
@dataclass
class REEExtractorParams:
    n_stages: int              # Number of extraction stages
    extractant: str            # Extractant name (D2EHPA, PC88A, etc.)
    elements: tuple[str, ...]  # REE elements to track
    pH: float = 3.0            # Operating pH
    extractant_conc: float = 0.5  # Extractant concentration (M)
    nitrate_conc: float | None = None  # M; required for solvating extractants
    include_loading: bool = True  # Account for extractant loading capacity
    capacity_sharpness: int = 8   # Sharpness of the smooth loading limiters
    include_speciation: bool = False  # Account for aqueous speciation

    # Closed mass-action level (#196); ignored by the default correlation path.
    # See "Mass-Action Equilibrium Closure" above.
    model: str = "correlation"        # or "mass_action"
    aqueous_volumetric_flow: float | None = None   # L/s, required at L2
    organic_volumetric_flow: float | None = None   # L/s, required at L2
    counter_ion: str | None = "Na"
    anion: str = "Cl"
    reaction_network: str | None = None   # None picks it from the record
    log10_K: dict | None = None           # measured constants, by element
    base_addition: float = 0.0            # mol/s of strong base into the feed
```

```{note}
With `model="mass_action"` the `pH` field is the **calibration** pH, not an
operating specification: the operating pH is an output, in
`info["pH_profile"]`, and the input that replaces it is `base_addition`.
Passing an explicit `pH` to the call raises. See
[Mass-Action Equilibrium Closure](mass-action-closure).
```

#### Usage

```python
from difflow_ree import REEExtractor, REEExtractorParams
from difflow.streams import make_stream

# Create extractor
params = REEExtractorParams(
    n_stages=10,
    extractant="D2EHPA",
    elements=("La", "Ce", "Nd", "Dy"),
    pH=3.0,
)
extractor = REEExtractor(params)

# Create feed and solvent streams.
# The solvent must name the extractant and/or the diluent the extractor is
# configured with: those two species are the organic phase, everything else
# in a stream is aqueous. A stream carrying neither raises (#192).
feed = make_stream({"H2O": 1.0, "La": 0.1, "Ce": 0.2, "Nd": 0.15, "Dy": 0.05}, T=298.15, P=101325.0)
solvent = make_stream({"D2EHPA": 0.2, "kerosene": 1.0}, T=298.15, P=101325.0)

# Run extraction
raffinate, extract, info = extractor(feed, solvent, T=298.15, pH=3.0)

# Check recoveries
for elem, data in info["profiles"].items():
    print(f"{elem}: Recovery = {data['recovery']:.1%}")

# With include_loading=True, info also reports the capacity condition
print(info["theta_total"])                # organic loading, 1.0 = saturated
print(info["capacity"])                   # F_extractant / m, a molar flow
print(info["capacity_clamped_fraction"])  # how much the limiter removed
```

#### Governing Equations

**Kremser Equation** for counter-current extraction:

$$\frac{x_{out}}{x_{in}} = \frac{E - 1}{E^{N+1} - 1}$$

Where:
- $E = D \cdot (S/F)$ is the extraction factor
- $D$ is the distribution coefficient
- $S/F$ is the solvent-to-feed ratio
- $N$ is the number of stages

#### Phases, loading and capacity

The organic phase of any stream is the extractant plus the diluent; every
other species (water, acid, dissolved REE, spectators) is aqueous. `REEExtractor`
and `REEMixerSettler` share this one definition, so a single Kremser stage and
one 100%-efficient mixer-settler give the same recovery. A stream missing the
phase a unit needs raises a `ValueError` naming the species that were present,
rather than silently defaulting that phase's flow to 1.0.

Loading is always a **dimensionless fraction**,

$$\theta = \frac{m \, n_{\mathrm{REE,org}}}{n_{\mathrm{HA}}}$$

with $m$ the extractant monomer equivalents bound per REE, read from the
extraction mechanism declared in `data/extractants.yaml`
(`Extractant.monomers_per_ree`). It is 6 for the acidic organophosphorus
extractants, which are declared as three dimers, and 3 for TBP. The capacity is
$1/m$ mol REE per mol extractant, and the free-extractant exponent in
`LoadingIsotherm.apparent_D` is the same $m$, so the two cannot disagree.

:::{warning}
This changed exported API. `LoadingIsotherm.max_loading` was a constructor
field defaulting to 0.33; it is now a read-only property equal to $1/m$, so
`LoadingIsotherm(max_loading=0.33)` raises `TypeError` — pass `m=3.0` instead,
and note that the acidic organophosphorus extractants now derive $m=6$, halving
the capacity the old literal claimed. The `"stoichiometry"` and `"max_loading"`
keys were removed from the public `EXTRACTANT_CAPACITIES` dict (read
`get_extractant(name).monomers_per_ree` instead), and `loading_correction()`
now raises the free fraction to `isotherm.m` rather than a literal 3 against a
halved capacity, which moves its output by a factor of ~25 for D2EHPA.
:::

Free-extractant depletion, $D \propto [\mathrm{HA}]_\mathrm{free}^n$, is applied
in exactly one place: the concentration term inside `REEDistribution.get_D`.
`LoadingIsotherm.apparent_D` implements the same physics and is available for
callers holding a $D$ that does not already carry that term, but it is not
applied in the stage path, which would double the correction.

What the stage does enforce is finite capacity. The Kremser closed form can
predict extraction beyond what the extractant can physically hold, so the newly
extracted total is multiplied by the smooth saturation

$$s = \left[1 + \left(\frac{n_\mathrm{extracted}}{n_\mathrm{HA}/m}\right)^{k}\right]^{-1/k}$$

with $k$ = `capacity_sharpness`. Because the capacity is a flow of
extractant, a solvent stream that does not declare one raises when
`include_loading=True`, rather than silently returning zero recovery. This is
$C^\infty$, so `jax.grad`
is continuous at the capacity constraint an economic optimum sits on, unlike
the hard clamp it replaces. `info` reports `theta_total`, `theta_solvent`,
`free_fraction_in`, `capacity`, `uncapped_extracted`, `capacity_scale`,
`capacity_clamped_fraction` and `capacity_sharpness` so a converged design can
be told apart from one pinned against the capacity wall.

The **same** smoothing, driven by the same $k$, is applied to the free fraction
of the *entering* solvent, which multiplies $E$ when a partly loaded solvent is
recycled from the strip section:

$$f_\mathrm{free} = 1 - \theta_\mathrm{solvent}\left[1 + \theta_\mathrm{solvent}^{\,k}\right]^{-1/k}$$

A hard $\max(1-\theta, 0)$ here is worse than a kink: beyond $\theta = 1$ it is
identically zero, so $E$ is zero, so the derivative of every downstream
quantity with respect to the solvent loading is *exactly* zero and the lever is
dead — the vanishing-column failure `difflow.planning.health` reports. The
smooth form behaves as $1-\theta$ below saturation, equals $1 - 2^{-1/k}$ at
$\theta = 1$, and decays as $\theta^{-k}/k$ beyond it, so the gradient is
small but never zero. (It is evaluated as $-\mathrm{expm1}(-\mathrm{log1p}
(\theta^{-k})/k)$ above $\theta = 1$; the literal expression cancels to exactly
0 in float64 around $\theta \approx 100$, which would resurrect the dead lever.)

**Choosing $k$.** The smoothing costs a small unconditional haircut below
capacity where a hard clamp cost nothing:

| $n_\mathrm{extracted}/n_\mathrm{capacity}$ | $k=4$ | $k=8$ (default) | $k=16$ |
|---|---|---|---|
| 0.50 | 0.98496 | 0.99951 | 0.9999990 |
| 0.75 | 0.93358 | 0.98814 | 0.99938 |
| 1.00 | 0.84090 | 0.91700 | 0.95760 |

The log-log slope $\mathrm{d}\ln s/\mathrm{d}\ln r$ is bounded in $[-1, 0]$ for
every $k$, so $k$ does not change first-derivative magnitudes; what grows is
the curvature, $|\mathrm{d}^2\ln s/\mathrm{d}(\ln r)^2| = k/4$ at the crossing.
The default is 8: since `include_loading` defaults to `True`, every default
result carries this haircut, and 0.05% at half capacity is below the
uncertainty in the correlations themselves, where $k=4$ cost 1.5%. Raise it to
16 or 32 to approach `min()` once a solve has converged; lower it to 2–4 when
an optimizer is far away and needs a gentler surface.

The REE flowsheet Params (`ExtractStripParams` and friends) do not yet expose
`capacity_sharpness`; reach it through the extractor they build:

```python
circuit._extractor.params = circuit._extractor.params.update(
    capacity_sharpness=16)
```

**Zero-flow phases.** A phase whose species are present but whose flows sum to
zero (`{"H2O": 0.0, "D2EHPA": 1.0, "kerosene": 5.0}`) raises: the phase ratio
$D\,F_\mathrm{org}/F_\mathrm{aq}$ is undefined, and flooring it produced an
extraction factor of order $10^{10}$. Non-zero denominators are guarded by a
floor *relative* to the streams' own total flow rather than an absolute molar
flow, so recoveries are invariant to the unit the flows are expressed in over
the whole float64 range (an absolute $10^{-10}$ floor broke that below about
$10^{-11}$ mol/s).

(reemixersettler)=
### REEMixerSettler

**Description**: Single mixer-settler stage for REE extraction with efficiency factor.

```python
@dataclass
class MixerSettlerParams:
    extractant: str
    elements: tuple[str, ...]
    pH: float = 3.0
    extractant_conc: float = 0.5
    nitrate_conc: float | None = None  # M; required for solvating extractants
    mixer_residence_time: float = 120.0  # seconds
    settler_residence_time: float = 300.0  # seconds
    stage_efficiency: float = 0.95
    third_phase_loading_limit: float | None = None  # mol REE / mol extractant
```

When `third_phase_loading_limit` is set, the organic inlet must declare a flow
of the extractant — the loading is mol REE per mol extractant, and dividing by
a missing flow reported loadings of order $10^{29}$ and called them converged.
`info` then carries `organic_loading`, the
boolean `third_phase_formed`, and `third_phase_margin` = `limit - loading`. The
margin is smooth and signed (positive is feasible), so third-phase onset can be
posed as an inequality constraint in an optimization rather than only read as a
diagnostic; the boolean has no gradient, so an optimizer would otherwise walk
straight through the boundary because crossing it is profitable in the model.

(reescrubber)=
### REEScrubber

**Description**: Multi-stage scrubbing section for removing impurities from loaded organic.

Scrubbing uses lower pH to selectively strip lighter REE back to aqueous phase while retaining heavier REE in the organic.

```python
from difflow_ree import REEScrubber, ScrubberParams

params = ScrubberParams(
    n_stages=5,
    extractant="D2EHPA",
    elements=("La", "Ce", "Nd", "Dy"),
    target_elements=("Nd", "Dy"),  # Keep these in organic
    pH=2.0,  # Lower pH to reject La, Ce
)
scrubber = REEScrubber(params)
```

(reestripper)=
### REEStripper

**Description**: Multi-stage stripping section for product recovery.

Stripping uses very low pH (strong acid) to transfer all REE from organic back to aqueous phase.

```python
from difflow_ree import REEStripper, StripperParams

params = StripperParams(
    n_stages=5,
    extractant="D2EHPA",
    elements=("Nd", "Dy"),
    pH=0.5,  # Strong acid for complete stripping
)
stripper = REEStripper(params)
```

(ceriumoxidizer)=
### CeriumOxidizer

**Location**: `difflow_ree/units/cerium.py`

**Description**: Oxidizes Ce³⁺ to Ce⁴⁺ and precipitates as CeO₂.

Cerium is unique among lanthanides because it can be oxidized from Ce³⁺ to Ce⁴⁺, enabling selective removal.

#### Parameters

```python
@dataclass
class CeriumOxidizerParams:
    elements: tuple[str, ...]
    oxidant: str = "air"  # air, H2O2, NaOCl, electrolytic
    oxidant_excess: float = 2.0
    pH: float = 8.0  # Alkaline conditions favor oxidation
    temperature: float = 353.15  # 80°C typical
    ce_conversion: float = 0.95
```

#### Usage

```python
from difflow_ree import CeriumOxidizer, CeriumOxidizerParams

params = CeriumOxidizerParams(
    elements=("La", "Ce", "Pr", "Nd"),
    oxidant="air",
    pH=8.0,
    ce_conversion=0.95,
)
oxidizer = CeriumOxidizer(params)

# Run oxidation
filtrate, ceo2_solid, info = oxidizer(feed)

print(f"Ce conversion: {info['ce_conversion']:.1%}")
print(f"CeO2 produced: {info['ceo2_mass_kg_s']:.4f} kg/s")
```

---

(precipitation-operations)=
## Precipitation Operations

(oxalateprecipitator)=
### OxalatePrecipitator

**Description**: Precipitates REE as oxalate, which can be calcined to oxide.

**Reaction**: 2REE³⁺ + 3C₂O₄²⁻ → REE₂(C₂O₄)₃↓

```python
from difflow_ree import OxalatePrecipitator, PrecipitatorParams

params = PrecipitatorParams(
    elements=("Nd", "Dy"),
    precipitant_excess=1.5,  # 50% excess
    target_conversion=0.995,
)
precipitator = OxalatePrecipitator(params)

# Feed is stripped REE solution, precipitant is oxalic acid
filtrate, solid, info = precipitator(feed, oxalic_acid)

print(f"Total precipitated: {info['total_precipitated']:.4f} mol/s")
print(f"Solid composition: {info['solid_composition']}")
```

(carbonateprecipitator)=
### CarbonatePrecipitator

**Reaction**: 2REE³⁺ + 3CO₃²⁻ → REE₂(CO₃)₃↓

Used for group precipitation from leach solutions.

(hydroxideprecipitator)=
### HydroxidePrecipitator

**Reaction**: REE³⁺ + 3OH⁻ → REE(OH)₃↓

Hydroxide precipitation can be selective based on pH - heavy REE precipitate at lower pH than light REE.

```python
from difflow_ree import HydroxidePrecipitator, PrecipitatorParams

params = PrecipitatorParams(elements=("La", "Ce", "Nd", "Dy"))
precipitator = HydroxidePrecipitator(params)

# pH-selective precipitation
filtrate, solid, info = precipitator(feed, naoh_solution, pH=8.5)

# Find selective precipitation pH range
min_pH, max_pH = precipitator.selective_precipitation_pH("Dy", "La")
print(f"pH range for Dy/La separation: {min_pH:.1f} - {max_pH:.1f}")
```

---

(flowsheet-templates)=
## Flowsheet Templates

(extractstripcircuit)=
### ExtractStripCircuit

**Description**: Basic 2-section circuit for simple separations.

```
    Feed                    Product
      ↓                        ↑
┌───────────┐         ┌───────────┐
│           │         │           │
│ EXTRACTION│ ──Org──▶│ STRIPPING │
│           │         │           │
└───────────┘         └───────────┘
      ↓                     ↓
  Raffinate            Strip Acid
```

(extractscrubstripcircuit)=
### ExtractScrubStripCircuit

**Description**: Industrial 3-section circuit for high-purity separations.

```
    Feed                    Scrub                   Product
      ↓                       ↓                        ↑
┌───────────┐         ┌───────────┐         ┌───────────┐
│           │         │           │         │           │
│ EXTRACTION│ ──Org──▶│ SCRUBBING │ ──Org──▶│ STRIPPING │
│           │         │           │         │           │
└───────────┘         └───────────┘         └───────────┘
      ↓                     ↓          ◀──Org──    ↓
  Raffinate            Scrub Liquor          Strip Acid
```

#### Parameters

```python
@dataclass
class ExtractScrubStripParams:
    extractant: str
    elements: tuple[str, ...]
    target_elements: tuple[str, ...]  # Elements to recover
    n_extraction_stages: int = 10
    n_scrubbing_stages: int = 5
    n_stripping_stages: int = 5
    extraction_pH: float = 3.5
    scrubbing_pH: float = 2.0  # Lower pH rejects light REE
    stripping_pH: float = 0.5
    solvent_to_feed_ratio: float = 1.0
    scrub_to_solvent_ratio: float = 0.2
    strip_to_solvent_ratio: float = 0.5
```

#### Usage

```python
from difflow_ree import ExtractScrubStripCircuit, ExtractScrubStripParams
from difflow.streams import make_stream

params = ExtractScrubStripParams(
    extractant="D2EHPA",
    elements=("La", "Ce", "Nd", "Dy"),
    target_elements=("Nd", "Dy"),
    n_extraction_stages=10,
    n_scrubbing_stages=5,
    n_stripping_stages=5,
    extraction_pH=3.5,
    scrubbing_pH=2.0,
    stripping_pH=0.5,
)
circuit = ExtractScrubStripCircuit(params)

# Create feed
feed = make_stream({
    "H2O": 1.0,
    "La": 0.10,
    "Ce": 0.20,
    "Nd": 0.15,
    "Dy": 0.05,
}, T=298.15, P=101325.0)

# Run circuit
results = circuit(feed)

print(f"Target purity: {results['target_purity']:.1%}")
for elem, recovery in results['target_recovery'].items():
    print(f"{elem} recovery: {recovery:.1%}")
```

(splitshellcascade)=
### SplitShellCascade

**Description**: Multi-product split-shell cascade for producing multiple pure REE streams.

---

(custom-elements-and-data)=
## Custom Elements and Data

The built-in database covers 10 commercial REEs and 4 extractant systems, but many applications require elements or extractant data not included by default. The `difflow_ree` plugin provides a runtime API for adding your own literature data, following the same pattern as the existing `create_custom_extractant` / `add_extractant` workflow.

(adding-a-custom-element)=
### Adding a Custom Element

Use `create_custom_element` to build an `REEElement` from known physical properties, then register it with the element database. All physical constants (atomic weight, ionic radius, density, melting point) should come from standard references such as the CRC Handbook or Shannon (1976) ionic radii tables.

```python
from difflow_ree import create_custom_element, get_ree_database

# Create Holmium from literature data
ho = create_custom_element(
    symbol="Ho",
    name="Holmium",
    atomic_number=67,
    atomic_weight=164.930,   # g/mol, CRC Handbook
    ionic_radius_pm=90.1,    # pm, Shannon (1976), CN=6, 3+
    density=8.795,           # g/cm³
    melting_point=1734,      # K
    group="heavy",
    oxide_formula="Ho2O3",
    oxide_mw=377.86,         # g/mol
    price_usd_kg=60.0,      # approximate market price
)

# Register with the database
db = get_ree_database()
db.add_element("Ho", ho)

# Now Ho is available alongside built-in elements
print(db.get("Ho").ionic_radius_pm)  # 90.1
print(db.list_by_group("heavy"))     # [..., 'Ho']
```

Elements can also be updated or removed:

```python
db.update_element("Ho", updated_ho)  # replace with corrected data
db.remove_element("Ho")              # remove entirely
```

(adding-extractant-coefficients-for-a-new-element)=
### Adding Extractant Coefficients for a New Element

After registering an element, you need to provide its pH and temperature coefficients for at least one extractant before it can be used in extraction simulations. These coefficients are empirical and should come from published experimental correlations (e.g., Gupta & Krishnamurthy, 2005; Xie et al., 2014).

You only need to add data for the extractants you plan to use. For example, to add Ho data for PC88A only:

```python
from difflow_ree import get_extractant_database

ext_db = get_extractant_database()

# Add Ho coefficients to PC88A
# Model: log10(D) = a + b*pH + c*pH^2 + d/T
ext_db.add_element_to_extractant(
    "PC88A",
    "Ho",
    ph_coefficients={
        "a": -6.15,   # from your literature source
        "b": 2.95,
        "c": 0.010,
    },
    temperature_coefficient=-2350,  # K, for d*(1/T - 1/T_ref) correction
)

# Verify
extractant = ext_db.get("PC88A")
print("Ho" in extractant.ph_coefficients)  # True

# Other extractants are unaffected
print("Ho" in ext_db.get("D2EHPA").ph_coefficients)  # False
```

If you need to correct values, remove and re-add:

```python
ext_db.remove_element_from_extractant("PC88A", "Ho")
ext_db.add_element_to_extractant("PC88A", "Ho", ...)
```

(adding-separation-factors)=
### Adding Separation Factors

Separation factor data can be added incrementally. You can add individual pairs to existing extractants or create complete entries for new ones.

Adding pairs to an existing extractant:

```python
from difflow_ree import get_sf_database

sf_db = get_sf_database()

# Add Ho separation factors to PC88A (from literature)
sf_db.add_pair("PC88A", "Ho_Dy", 1.4, adjacent=True, stages_99=20)
sf_db.add_pair("PC88A", "Y_Ho", 0.9, adjacent=True)

# Add a non-adjacent group pair
sf_db.add_pair("PC88A", "Ho_Nd", 10.5, adjacent=False)

# Query the new data
print(sf_db.get_sf("PC88A", "Ho_Dy"))           # 1.4
print(sf_db.get_stages_needed("PC88A", "Ho_Dy"))  # 20
```

Creating a complete entry for a new or custom extractant:

```python
sf_db.add_separation_factors(
    extractant="MyExtractant",
    conditions={"pH": 3.0, "temperature_K": 298, "concentration_M": 0.5},
    adjacent_pairs={"Ho_Dy": 1.4, "Y_Ho": 0.9},
    group_pairs={"Ho_La": 50.0},
    stages_for_99_purity={"Ho_Dy": 20},
)
```

(custom-data-complete-workflow)=
### Complete Workflow

Here is a full example of adding Holmium and using it in a separation simulation. In a real application, the pH coefficients and separation factors should come from published experimental data for your specific extractant system.

```python
from difflow_ree import (
    create_custom_element,
    get_ree_database,
    get_extractant_database,
    get_sf_database,
    REEDistribution,
)

# 1. Register the element
ho = create_custom_element(
    symbol="Ho", name="Holmium", atomic_number=67,
    atomic_weight=164.930, ionic_radius_pm=90.1, density=8.795,
    melting_point=1734, group="heavy", oxide_formula="Ho2O3",
    oxide_mw=377.86, price_usd_kg=60.0,
)
get_ree_database().add_element("Ho", ho)

# 2. Add extraction coefficients (from your literature source)
get_extractant_database().add_element_to_extractant(
    "PC88A", "Ho",
    ph_coefficients={"a": -6.15, "b": 2.95, "c": 0.010},
    temperature_coefficient=-2350,
)

# 3. Add separation factor data
sf_db = get_sf_database()
sf_db.add_pair("PC88A", "Ho_Dy", 1.4, stages_99=20)
sf_db.add_pair("PC88A", "Ho_Gd", 2.5, adjacent=False)

# 4. Use in distribution calculations
dist = REEDistribution(
    extractant="PC88A",
    elements=("Gd", "Dy", "Ho", "Y"),
)
D_ho = dist.get_D("Ho", pH=3.5, T=298.15)
print(f"D(Ho) at pH 3.5: {D_ho:.2f}")
```

---

## Economics

The plugin includes economic analysis tools:

```python
from difflow_ree import (
    estimate_capex,
    estimate_opex,
    calculate_revenue,
    calculate_profit,
    minimum_selling_price,
)

# Equipment costs
capex = estimate_capex(
    n_mixer_settlers=20,
    mixer_settler_volume=5.0,  # m³
    precipitation_capacity=100.0,  # kg/hr
)

# Operating costs
opex = estimate_opex(
    ree_throughput=200.0,  # tonnes/year
    extractant_consumption=50.0,  # kg/year
    acid_consumption=1000.0,  # kg/year
)

# Revenue from products
revenue = calculate_revenue(
    nd_production=50.0,  # kg/year
    dy_production=10.0,
    nd_price=100.0,  # $/kg
    dy_price=350.0,
)

# Profitability
profit = calculate_profit(revenue, opex, capex, years=10)
```

---

## Examples

### Example 1: Simple Nd/Pr Separation

```python
from difflow_ree import (
    REEDistribution,
    ExtractScrubStripCircuit,
    ExtractScrubStripParams,
)
from difflow.streams import make_stream

# Analyze separation factors
dist = REEDistribution(extractant="PC88A", elements=("Pr", "Nd"))
opt_pH, max_SF = dist.optimal_pH_for_separation("Nd", "Pr")
print(f"Optimal pH for Nd/Pr: {opt_pH:.2f}, SF = {max_SF:.2f}")

# Design separation circuit
params = ExtractScrubStripParams(
    extractant="PC88A",
    elements=("Pr", "Nd"),
    target_elements=("Nd",),
    extraction_pH=opt_pH,
    scrubbing_pH=opt_pH - 1.0,
)
circuit = ExtractScrubStripCircuit(params)

# Run separation
feed = make_stream({"H2O": 1.0, "Pr": 0.3, "Nd": 0.7}, T=298.15, P=101325.0)
results = circuit(feed)

print(f"Nd purity: {results['product_purity']['Nd']:.1%}")
print(f"Nd recovery: {results['target_recovery']['Nd']:.1%}")
```

### Example 2: Cerium Removal from Bastnasite

```python
from difflow_ree import CeriumOxidizer, CeriumOxidizerParams
from difflow.streams import make_stream

# Bastnasite composition (typical)
feed = make_stream({
    "H2O": 1.0,
    "La": 0.25,
    "Ce": 0.50,  # 50% Ce typical
    "Pr": 0.05,
    "Nd": 0.15,
    "Sm": 0.03,
    "Gd": 0.02,
}, T=298.15, P=101325.0)

# Oxidize and remove Ce
params = CeriumOxidizerParams(
    elements=("La", "Ce", "Pr", "Nd", "Sm", "Gd"),
    oxidant="air",
    pH=8.0,
    ce_conversion=0.95,
)
oxidizer = CeriumOxidizer(params)

filtrate, ceo2, info = oxidizer(feed)

print(f"Ce removed: {info['ce_conversion']:.1%}")
print(f"CeO2 produced: {info['ceo2_mass_kg_s']*3600*24*365:.1f} kg/year")
print(f"Ce in filtrate: {info['ce_fraction_out']:.1%}")
```

### Example 3: Gradient-Based Optimization

```python
import jax
import jax.numpy as jnp
from difflow_ree import ExtractScrubStripCircuit, ExtractScrubStripParams
from difflow.streams import make_stream

def separation_objective(pH_values):
    """Objective: maximize Nd purity × recovery."""
    extraction_pH, scrubbing_pH = pH_values

    params = ExtractScrubStripParams(
        extractant="D2EHPA",
        elements=("La", "Ce", "Nd"),
        target_elements=("Nd",),
        extraction_pH=extraction_pH,
        scrubbing_pH=scrubbing_pH,
    )
    circuit = ExtractScrubStripCircuit(params)

    feed = make_stream({"H2O": 1.0, "La": 0.3, "Ce": 0.4, "Nd": 0.3}, T=298.15, P=101325.0)
    results = circuit(feed)

    purity = results['product_purity']['Nd']
    recovery = results['target_recovery']['Nd']

    return -(purity * recovery)  # Negative for minimization

# Compute gradients
grad_fn = jax.grad(separation_objective)
pH_init = jnp.array([3.5, 2.0])
gradients = grad_fn(pH_init)
print(f"Gradients: d/d(ext_pH) = {gradients[0]:.4f}, d/d(scrub_pH) = {gradients[1]:.4f}")
```

---

## See Also

- [Examples: 04_rare_earth_extraction.ipynb](../examples/04_rare_earth_extraction.ipynb) - Basic REE extraction
- [Examples: 09_ree_ndfeb_magnet.ipynb](../examples/09_ree_ndfeb_magnet.ipynb) - NdFeB magnet recycling
- [Examples: 10_bastnasite_separation.ipynb](../examples/10_bastnasite_separation.ipynb) - Bastnasite ore processing
