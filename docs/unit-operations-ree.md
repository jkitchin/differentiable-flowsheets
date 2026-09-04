# Rare Earth Element (REE) Unit Operations

This document provides comprehensive documentation for the `difflow_ree` plugin, which provides specialized tools for modeling and optimizing rare earth element solvent extraction processes.

---

## Overview

The `difflow_ree` plugin provides:

- **Database** of 10 commercial REE properties (La, Ce, Pr, Nd, Sm, Eu, Gd, Tb, Dy, Y)
- **4 extractant systems**: D2EHPA, PC88A, Cyanex272, TBP
- **pH-dependent distribution coefficient models**
- **Loading and speciation corrections**
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
