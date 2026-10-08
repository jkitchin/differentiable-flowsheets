# Rare Earth Element (REE) Unit Operations

![difflow_ree](../images/plugins/difflow-ree-logo.svg)

This document provides comprehensive documentation for the `difflow_ree` plugin, which provides specialized tools for modeling and optimizing rare earth element solvent extraction processes.

---

## Overview

The `difflow_ree` plugin provides:

- **Database** of 15 REE properties (La, Ce, Pr, Nd, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu, Y)
- **5 extractant systems**: D2EHPA, PC88A, Cyanex272, TBP, naphthenic acid
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
# ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Y',
#  'Ho', 'Er', 'Tm', 'Yb', 'Lu']

# Get element properties
nd = get_element("Nd")
print(f"Atomic weight: {nd.atomic_weight}")
print(f"Ionic radius: {nd.ionic_radius_pm} pm")
print(f"Price: ${nd.price_usd_kg}/kg")
```

#### Available Properties

| Property | Description | Units |
|----------|-------------|-------|
| `atomic_weight` | Atomic mass | g/mol |
| `ionic_radius_pm` | Ionic radius (3+) | pm |
| `price_usd_kg` | Market price | USD/kg |
| `oxide_mw` | Oxide molecular weight | g/mol |
| `oxide_formula` | Oxide formula | - |

(extractant-database)=
### Extractant Database

Four industrial extractants are supported:

```python
from difflow_ree import get_extractant, list_extractants

print(list_extractants())
# ['D2EHPA', 'PC88A', 'Cyanex272', 'TBP', 'naphthenic_acid']

d2ehpa = get_extractant("D2EHPA")
print(f"Full name: {d2ehpa.full_name}")
print(f"Reference concentration: {d2ehpa.reference_concentration} M")
```

| Extractant | Full Name | Primary Use | Elements covered |
|------------|-----------|-------------|------------------|
| D2EHPA | Di(2-ethylhexyl)phosphoric acid | Light/middle REE | 10/15 — La–Dy, Y |
| PC88A | 2-ethylhexyl phosphonic acid mono-2-ethylhexyl ester | Middle REE | 10/15 — La–Dy, Y |
| Cyanex272 | Bis(2,4,4-trimethylpentyl)phosphinic acid | Heavy REE, Co/Ni | 10/15 — La–Dy, Y |
| TBP | Tri-n-butyl phosphate | Ce separation, nuclear | 10/15 — La–Dy, Y |
| naphthenic_acid | Naphthenic acid (saponified) | Y purification, full series | **15/15** |

(extractant-basis)=
### Extractant concentration basis

Every `extractant_conc` of a unit, module or circuit, the `concentration` of
`REEDistribution`, and the extractant entry of a solvent stream are on the
**extractant record's own basis** (#374, which kept this basis):

| Extractant | Basis | `extractant_conc = 0.5` means |
|---|---|---|
| D2EHPA, PC88A, Cyanex272 | dimer, `(HA)2` | 0.5 M dimer = 1.0 M formal |
| TBP | molecule (monomer) | 0.5 M TBP |
| naphthenic_acid | molecule (monomer) | 0.5 M HA |

The record states it as `stoichiometry.basis` in `data/extractants.yaml`; its
`reference_concentration` and fitted coefficients are on the same basis. The
loading capacity is `extractant_conc / Extractant.basis_units_per_ree` (3 for
every shipped record), so 0.5 M D2EHPA holds at most 0.167 M REE.

The mass-action layer (`MassActionParams`, `log_K_from_correlation`,
`cation_exchange_network`, `REEStreamSchema` streams) works on the **formal
monomer** basis instead: 1.0 M there is the 0.5 M dimer above.
`Extractant.monomers_per_basis_unit` (2 for the dimeric records, 1 otherwise)
converts, and `REEExtractor(model="mass_action")` applies it to both
`extractant_conc` and the solvent stream, so one solvent is the same solvent
at both levels. Before #374 neither boundary converted: the closure saw half a
D2EHPA solvent's extractant, and calibrated its constants against the
correlation at twice the dimer charge it then ran at.

(element-coverage)=
### Which elements an extractant covers

Coverage is a property of the record, not of the package, and it is uneven. Ask
before a run rather than during one (#269):

```python
from difflow_ree import get_extractant_database

print(get_extractant_database().coverage().as_text())
#   extractant       covered  missing
#   D2EHPA            10/15   Ho, Er, Tm, Yb, Lu
#   PC88A             10/15   Ho, Er, Tm, Yb, Lu
#   Cyanex272         10/15   Ho, Er, Tm, Yb, Lu
#   TBP               10/15   Ho, Er, Tm, Yb, Lu
#   naphthenic_acid   15/15   -
```

Four of the five stop at Dy plus Y. `naphthenic_acid` is the exception, and its
fifteen elements come from one measured table (`Z1` Sec. 4.7, Table 4.36). Full
coverage is not the same as a trustworthy correlation, though: only the `a`
ladder is that table's, the pH slope `b` comes from `Q1` Eq. 2.88 and carries
the known saponification gap (#266), and `c`/`d` are declared zeros. Run
`difflow_ree.provenance.explain` before reading absolute `D` off it.

`coverage(elements)` answers the same question for a list you care about, and
`get_extractant("TBP").covered_elements` answers it for one record —
mechanism-aware, so it reads the nitrate block for a solvating extractant and
the pH block for a cation exchanger.

Naming an element an extractant has no coefficients for is now refused **when
the distribution is constructed**, with all of the missing elements at once and
the coverage that does exist beside them:

```python
from difflow_ree import REEDistribution

try:
    REEDistribution(extractant="D2EHPA", elements=("Y", "Ho", "Er", "Tm", "Yb", "Lu"))
except ValueError as err:
    print(err)
# ValueError: Extractant 'D2EHPA' has no coefficients for Ho, Er, Tm, Yb, Lu
# (mechanism='cation_exchange') ... It covers: La, Ce, Pr, Nd, Sm, Eu, Gd, Tb, Dy, Y.
```

It used to be a `KeyError` from mid-solve, raised while iterating stages and
naming one element at a time.

To look an element up without being refused, ask for `no_data="nan"`: `get_D`
answers NaN for an element with no coefficients, and also for the heavies the
record only extrapolates below its fitted pH window (`unmeasured_outside_window`:
Gd, Tb, Dy and Y on D2EHPA, where heavy-REE stripping has to run, #384).
`no_data="raise"` makes that second case an error. A NaN is never filtered
downstream, so a calculation that uses one returns NaN. The default, `"warn"`,
is the behaviour above.

```python
d = REEDistribution(extractant="D2EHPA", elements=("Nd", "Dy", "Ho"), no_data="nan")
print(float(d.get_D("Nd", 1.0)), float(d.get_D("Dy", -0.8)), float(d.get_D("Ho", 1.0)))
# a number, nan (below the window), nan (no coefficients)
```

Those five are not an arbitrary example. **Yttrium purification is Y against Ho,
Er, Tm, Yb and Lu** — Y(III)'s 90.0 pm ionic radius sits between Ho's 90.1 and
Er's 89.0, which is exactly why they are what it has to be told apart from. A
record whose coefficients stop at Dy plus Y cannot do that separation however
many elements it lists.

Filling such a gap is a different job per extractant, and none of it is
interpolation:

- **D2EHPA, Cyanex272** have no published source for the ten they
  already carry (`HAND_TUNED` in `sources.yaml`), so there is nothing to extend
  *from* — extending them is a **refit**, not an interpolation.
- **PC88A** was in that list until #270 and is not any more: its `a` values are
  fitted to T21 (Tanaka 2021). Extending it is still a refit, but now for the
  opposite reason — there *is* something to extend from, and T21 covers only
  La, Nd, Sm, Dy and Y, so the five it does not cover (Ce, Pr, Eu, Gd, Tb) are
  already interpolations and are tagged `DERIVED`, not `MEASURED`. Ho through
  Lu are not even that.
- **TBP**'s ten are literature-derived (`K05` Table 1), so extending it means
  finding a source covering the heavies at comparable conditions, with its own
  `sources.yaml` key.
- **`naphthenic_acid`** already covers all fifteen, from `Z1` Table 4.36 --- a
  coverage gap it does not have, whatever the caveats above on its slope.

Anything added should arrive with a citation, via `add_element_to_extractant`;
`tests/ree/test_provenance.py` fails on a data field that matches no rule,
which is what keeps a partial extension honest about which elements are
measured and which are not.

Every extractant record declares a normalized **extraction mechanism**, and it
is the mechanism that decides which correlation drives `D` (#195):

| Mechanism | `type` values | Driving variable | Coefficient block |
|-----------|---------------|------------------|-------------------|
| `cation_exchange` | `acidic_phosphoric`, `acidic_phosphonic`, `acidic_phosphinic`, `acidic_carboxylic` | pH | `ph_coefficients` |
| `solvating` | `solvating_neutral` | nitrate concentration | `nitrate_coefficients` |
| `counter_ion_exchange` | declared explicitly | counter-ion concentration | `counter_ion_coefficients` |

The third is what a **saponified** circuit actually runs — see
[Saponified circuits exchange a counter-ion, not a proton](#saponified-circuits-exchange-a-counter-ion-not-a-proton).

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

(saponified-correlations)=
### Saponified circuits exchange a counter-ion, not a proton

`cation_exchange` describes proton exchange (Q1 Eq. 2.87):

$$\mathrm{RE}^{3+} + 3\,\mathrm{HA}_{(o)} \rightleftharpoons \mathrm{REA}_3{}_{(o)} + 3\,\mathrm{H}^+$$

which is what a pH slope of about 3 means. Industrial rare-earth circuits
saponify 30–80 % of the extractant before the cascade, and the reaction that
then runs is a **counter-ion** exchange (Z1 Eq. 4.96, Q1 Eq. 2.120), preceded by
the saponification itself (Z1 Eq. 4.95):

$$\mathrm{HL}_{(o)} + \mathrm{NH_4OH} = \mathrm{NH_4L}_{(o)} + \mathrm{H_2O}$$
$$\mathrm{RE}^{3+} + 3\,\mathrm{NH_4L}_{(o)} = \mathrm{REL}_3{}_{(o)} + 3\,\mathrm{NH_4}^+$$

**No proton appears on either side of the second.** For a correlation fitted to
such a system, a `b` of 3 is not a slope that is too large — it is a slope on
the wrong axis. Measured saponified systems put the apparent pH slope near 0.3
(Z1 Table 4.36), and Z1 Fig. 4.43 shows about one decade of `D` over pH 4.0–5.4
where a slope of 3 demands 10⁴·².

`mechanism: counter_ion_exchange` is where such a correlation goes (#266):

$$\log_{10}(D) = a - p \cdot \log_{10}\!\left(\frac{[\mathrm{M}^+]}{[\mathrm{M}^+]_{ref}}\right) + \frac{\Delta H}{R\ln 10}\left(\frac{1}{T} - \frac{1}{T_{ref}}\right) + n\log_{10}\!\left(\frac{[\mathrm{HA}]}{[\mathrm{HA}]_{ref}}\right)$$

Three things are deliberate about the shape:

- **The slope `p` is record-level, not per element.** It is the counter-ion
  released per mol REE, which the stoichiometry already fixes. That is what
  makes a separation factor *exactly* independent of `[M⁺]` — the term cancels
  in `D_A/D_B` — so `get_separation_factor` does not ask for one at all. Stage
  counts driven by β are untouched by this whole question; solvent inventory and
  O/A driven by absolute `D` are not.
- **`reference_counter_ion` has no default.** The sources cited here report no
  anchor `[M⁺]` alongside their distribution data, and a plausible-looking one
  would scale every `D` under it. A block without it is refused, and
  `get_D` on this mechanism requires `counter_ion_conc` rather than assuming one.
- **No shipped record carries the block.** The mechanism exists so a record
  measured on a saponified system has somewhere to put its real correlation.

<!-- doc-test: skip: needs a hypothetical saponified record "MySaponified" -->
```python
dist = REEDistribution(extractant="MySaponified", elements=("Sm", "Nd"))
dist.get_D("Nd", counter_ion_conc=0.5)          # needs [M+]
dist.get_separation_factor("Sm", "Nd")          # does not: the term cancels
```

#### Which system were the coefficients measured on?

The saponification `degree` on a record is an **operating default for the
circuit** and says nothing about what its correlation was fitted to. So the
record states that separately, as data:

```yaml
saponification:
  counter_ion: Na
  degree: 0.35
  correlation_basis: unsaponified   # what ph_coefficients were measured on
```

Every record shipping with difflow_ree declares `unsaponified`, and its pH slope
means what a pH slope means. A record declaring `saponified` while still being
driven by `ph_coefficients` raises `SaponifiedCorrelationWarning`, naming the
anchor its absolute `D` is tied to and the fact that separation factors survive
it. Inferring this from `degree` instead would put a warning on every REE
calculation in the package, where nothing is wrong.

---

(ree-provenance)=
### Where the Numbers Come From

The data files mix numbers of very different pedigree. Some are copied from a
named table in a named book. Some are computed from other numbers in the same
file. Some were invented so that a demo would converge. Read as bare YAML they
look identical, and that is exactly how an invented number ends up behind a
published stage count.

Every field is tagged. `explain` resolves one field to a citation:

```python
from difflow_ree import explain

p = explain("extractants", "extractants.naphthenic_acid.ph_coefficients.Dy.a")
print(p.source, p.cls, p.publishable)
# Z1 MEASURED True
print(p.locus)
# Sec. 4.7, Table 4.36, system 1

p = explain("extractants", "extractants.D2EHPA.ph_coefficients.Dy.a")
print(p.source, p.cls, p.publishable)
# HAND_TUNED HAND_TUNED False
```

`cls` answers the only question that matters at the call site: **may a
published number rest on this?**

| class | meaning | publishable |
|---|---|---|
| `MEASURED` | reported as an experimental result in a named source | yes |
| `REFERENCE` | standard reference datum (atomic weights, ionic radii) | yes |
| `DERIVED` | computed from other tagged values in this database | yes |
| `CONVENTION` | a bookkeeping choice, not a fact about the world | yes |
| `CONSTRUCTED` | built to satisfy published constraints; not itself measured | **no** |
| `ESTIMATED` | indicative order of magnitude, no identified source | **no** |
| `HAND_TUNED` | chosen to make code behave; no external basis at all | **no** |

The survey, for the whole database or one file:

```python
from difflow_ree import coverage, audit, unsourced

coverage()
# {'HAND_TUNED': 123, 'ESTIMATED': 18, 'CONSTRUCTED': 6,
#  'CONVENTION': 156, 'DERIVED': 32, 'REFERENCE': 186, 'MEASURED': 94}

len(unsourced())            # 147 of 615 fields must not back a published number
audit(source="Z1")          # every field traceable to Zhang (2016)
audit(cls="HAND_TUNED")     # every number nobody can defend
```

or from the shell:

```bash
python -m difflow_ree.provenance
python -m difflow_ree.provenance --cls HAND_TUNED
python -m difflow_ree.provenance --dataset elements --explain Dy.ionic_radius_pm
```

**Three things this makes visible that were previously invisible.**

1. Of the five extractants, three carry coefficients traceable to a named
   source: `naphthenic_acid` (Zhang 2016 Table 4.36, all fifteen elements),
   `TBP` (Kraikaew 2005 for `a`, Ganesh & Pandey 2019 for `d`) and, since
   #270, `PC88A` (Tanaka 2021, 121 digitized points, five elements measured
   and five interpolated). **D2EHPA and Cyanex272 do not.** Their coefficients
   were invented to make demonstrations look right. They are fine for
   exercising the solver, testing gradients and teaching the API, and must not
   appear behind a design number.

   One consequence of fixing PC88A and not the other two is worth stating
   plainly, because the file now reads as though it says something it does not.
   Solving each record for the pH at which `D(Nd) = 1` gives PC88A 1.03,
   D2EHPA 3.10, Cyanex272 3.74 — i.e. that D2EHPA needs a pH two units
   *higher* than PC88A. **That is backwards.** D2EHPA is the stronger acid
   (pKa 3.24 against PC88A's 4.10) and extracts at the lower pH. The ordering
   is an artifact of one record being measured and two being invented, not a
   claim about chemistry. Separation factors, cascade behaviour and design pH
   **within** one record are unaffected; *any* comparison of absolute `D`
   **across** these three records is meaningless.

   *Superseded.* #270 sourced D2EHPA and Cyanex272 and #283 refit D2EHPA's
   level (below). The pH at which `D(Nd) = 1` is now D2EHPA 0.27, PC88A
   1.03, Cyanex272 2.14 at each record's reference charge --- the textbook
   order, strongest acid lowest.

2. **Nine of fifteen element prices are `ESTIMATED`, and every solvent cost
   is.** #270 gave La, Ce, Nd, Pr, Eu and Gd the USGS 2025 annual average
   oxide price (`USGS26`); Nd and Pr carry the same number, because USGS
   quotes didymium as one product and publishes no split, so a circuit that
   separates them earns nothing here for having done so. The remaining nine —
   Sm, Tb, Dy, Y, Ho, Er, Tm, Yb, Lu — are untouched and still indicative, and
   solvent inventory is one of the larger terms in a TEA, so any economic
   conclusion that survives only at those numbers is still an artifact.

3. `separation_factors.yaml` **no longer carries factors at all** (#265). It
   used to, and the two files disagreed: recomputing an adjacent pair from the
   `ph_coefficients` gave 1.3x to 2.8x the tabulated value (4.8x for Cyanex272
   Sm/Nd), and 4x to 8x in the *opposite* direction for every Y/Dy pair. Two
   independently hand-tuned descriptions of one set of physics, never
   reconciled. The file now selects which pairs to report and at what
   conditions, and the values — and, since #270, the Fenske stage counts —
   are computed from the coefficient block of the same extractant.

   The Y/Dy pair is the interesting one in hindsight. Both descriptions put Y
   *below* Dy on all three acidic extractants, so #265 had no way to tell that
   both were wrong about PC88A. T21 measures β(Y/Dy) = 3.14 there — Y between
   Dy and Ho, which is what the PC88A literature has always said. D2EHPA and
   Cyanex272 still say 0.21 and 0.10: right for Cyanex 272, whose industrial
   appeal *is* that Y falls out of the heavy group, and backwards for D2EHPA.
   Where Y sits is a property of the extractant, not of yttrium.

The tagging is a maintained obligation rather than a comment that rots: adding
a field to a data file without a matching rule fails CI, as does citing a
source key that is not in `data/sources.yaml`.

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

# Get D value for Nd at pH 1.0, inside D2EHPA's validity window of [0, 2]
D_nd = dist.get_D("Nd", pH=1.0, T=298.15)
print(f"D(Nd) at pH 1.0: {D_nd:.2f}")     # 115.61

# Get all D values
D_all = dist.get_D_all(pH=1.0)
# La 10.12, Ce 24.29, Nd 115.61, Dy 13128
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
**Every extractant record's distribution coefficients are now fitted against
named primary sources.** TBP, `naphthenic_acid`, PC88A, D2EHPA and Cyanex272
all carry `sources.yaml` keys for their `ph_coefficients` (or, for TBP, its
nitrate block); #270 closed the last three. Each record's fit basis, correction
arithmetic, per-element measured-vs-interpolated status, validity window and
known gaps are written into `extractants.yaml` beside the numbers.

**D2EHPA's level and shape (#283).** #270 left D2EHPA's absolute level on one
measured `D` (La in kerosene, X95) and its shape on chromatographic separation
factors (PPH63), with three routes to the level disagreeing by up to 1.7
decades. The record is now fitted to 54 points digitized from Mason (1976) ---
HDEHP in *n*-heptane, Y/Tm/Lu, a pH series at constant ionic strength and a
concentration series --- with Peppard (1957)'s spacing for the shape (2.48 per
step, against PPH63's 2.20). X95's point is held out and recovered to 0.105
decades. The disagreement was the concentration law: Mason measures
`d log D / d log C = 2.38`, not the cube, and the discordant routes had carried
a cube across two decades of charge. `concentration_exponent` is therefore
2.38 on D2EHPA; the stoichiometry, and so the capacity, is unchanged. Nd, Sm,
Gd and Dy are interpolated on Peppard's line; the record gives `se(a)` per
element and the digitized data live in the ree-database record
`mason1976extraction`.

The exception, tagged rather than papered over, is the
`temperature_coefficients` block on PC88A and Cyanex272 (D2EHPA's was sourced from X95's 10--50 °C table in #283, which flipped its sign): every source
behind the #270 refit is isothermal, so it carries no information about `dH`
and those entries stay `HAND_TUNED`. `python -m difflow_ree.provenance --cls
HAND_TUNED` lists them.

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
difflow_ree defaults to 0.5 M, which is a cation-exchange default (0.5 M dimer
for D2EHPA; TBP's basis is the molecule, see
[the concentration basis](#extractant-basis)). TBP's
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
try:
    REEDistribution(extractant="TBP", elements=("Nd",),
                    nitrate_conc=3.0, medium="chloride")
except ValueError as err:
    print("refused:", str(err)[:60])

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
dist.get_D("Nd", pH=1.0, ionic_strength=0.3)   # in range, silent
dist.get_D("Nd", pH=1.0, ionic_strength=3.0)   # UserWarning: outside Davies range

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
D0 = float(d.get_D("Nd", pH=1.0))
[float(d.get_D("Nd", pH=1.0, ionic_strength=I)) / D0 for I in (0.1, 1.0, 3.0)]
# [0.2280, 0.1560, 0.1560]   <- never above 1

# Raw, possibly inverted Davies is still available, but only on request:
e = REEDistribution(extractant="D2EHPA", elements=("Nd",),
                    on_out_of_range="ignore", extrapolate_activity_model=True)
float(e.get_D("Nd", pH=1.0, ionic_strength=3.0)) / D0     # 6.49
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

pH_range = jnp.linspace(0.0, 2.0, 50)   # D2EHPA's fitted window
for elem in ["La", "Nd", "Dy"]:
    D_values = [float(dist.get_D(elem, pH)) for pH in pH_range]
    plt.semilogy(pH_range, D_values, label=elem)

plt.xlabel("pH")
plt.ylabel("Distribution Coefficient D")
plt.legend()
plt.grid(True)
```

(ph-validity-range)=
##### The pH correlation has a validity window, and it is now enforced (#262)

Every cation-exchange record declares a `valid_ph_range` — the window its
`ph_coefficients` were meant to describe:

```python
from difflow_ree import get_extractant
{e: get_extractant(e).valid_ph_range
 for e in ("D2EHPA", "PC88A", "Cyanex272", "TBP", "naphthenic_acid")}
# {'D2EHPA': (0.0, 2.0), 'PC88A': (0.1, 2.5), 'Cyanex272': (1.5, 3.5),
#  'TBP': (0.5, 4.0), 'naphthenic_acid': (4.0, 5.0)}
```

Four of those five windows moved in #270, because the refit made them
*measured* spans rather than declarations. D2EHPA went from `[1, 5]` to
`[0, 2]`, Cyanex272 from `[3, 7]` to `[1.5, 3.5]`, PC88A from `[0.1, 5.5]` to
`[0.1, 2.5]`. **No single pH is inside all four cation-exchange windows any
more**, which is why the library stopped shipping pH literals at all; see
{ref}`record-derived-ph-defaults`.

**PC88A's window is now [0.1, 2.5], and it means something different from the
other four.** It is the span T21 (Tanaka 2021) actually measured -- log D
against equilibrium pH for La, Nd, Sm, Dy and Y in Shellsol D70, that record's
own diluent, dimer basis and three protons per RE(III), from pH 0.12 to 2.46 at
six extractant concentrations -- and since #270 it is a window over *measured*
`a` values rather than over invented ones. It used to read `[0.1, 5.5]`, a
floor bounded by T21 and a ceiling bounded by nothing, sitting under
coefficients the same source put about 2 pH units and six decades of `D` away.
Both ends now come from the same 121 points as the coefficients.

The narrowing is not a loss of capability. The old ceiling was never a place
you could compute at; it was a place where the answer was wrong quietly instead
of loudly. A companion source above pH 2.5 is wanted and does not exist yet.

TBP's window is recorded but never checked: it is solvating, its correlation is
driven by nitrate activity and carries no pH term at all, so there is nothing
to extrapolate.

Until #262 that field was loaded into `database.Extractant` and then read by
nothing, so a circuit could be operated at `stripping_pH=0.3` against a
quadratic fitted over `[1.5, 5.5]` (PC88A's window at the time) and get a
silent answer. That is not a small
error: with `b = 3` for PC88A, **one pH unit outside the window moves `D` by
three decades**, and the failure is quiet — `D` stays finite, positive
and plausible-looking all the way down.

```{warning}
`examples/10_bastnasite_separation.ipynb` did exactly this, at all three of its
section pH values. Its product stream came out at 1e-22 mol/s of Nd, a
`if nd_mass_yr > 0` guard let it through, and a break-even calculation divided
by it and printed a price of `$2252838774866486493184/kg`. The notebook now
operates inside the window and asserts that the check stays quiet.
```

`REEDistribution.get_D` now reports pH against that window through the same
`on_out_of_range` setting as the ionic-strength check:

```python
d = REEDistribution(extractant="PC88A", elements=("Nd",))
d.get_D("Nd", pH=2.0)      # inside [0.1, 2.5], silent
d.get_D("Nd", pH=0.05)     # UserWarning: pH minimum 0.05 is outside ... (#262)

REEDistribution(extractant="PC88A", elements=("Nd",), on_out_of_range="raise")
REEDistribution(extractant="PC88A", elements=("Nd",), on_out_of_range="ignore")
```

Three things about it differ from the ionic-strength check above, each
deliberately:

* **It is a report, not a guard. pH is never clamped.** Ionic strength is
  clamped because raw Davies *inverts* past 1.94 M, so extrapolating it is
  qualitatively wrong. Extrapolating the pH quadratic is merely inaccurate, and
  clamping pH would silently relocate a flowsheet's operating point — a worse
  failure than an out-of-range number the caller can see.
* **A concrete array is checked at both ends**, not just its maximum, so a
  per-stage pH profile that leaves the window anywhere is reported.
* **An abstract tracer is passed over in silence.** `ionic_strength` defaults to
  `None`, so its check is opt-in and a traced value means the caller asked for a
  correction they cannot verify. `pH` is mandatory and is this library's primary
  differentiation variable — pH and `n_stages` are both advertised as
  continuous, traceable decisions — so warning on a tracer would fire on every
  `grad`/`jit`/`vmap` of every REE circuit and train users to filter the
  category that also carries the concrete report.

```{note}
Until #270 the flowsheet templates shipped a default `stripping_pH=0.5` that
was *below* D2EHPA's window, so `ExtractStripCircuit` and
`ExtractScrubStripCircuit` warned on their own defaults --- the check working,
and a warning nobody could act on without picking a different literal. Every
such default is now taken from the record; the templates no longer warn on
themselves. `tests/ree/test_kremser_temp_bugs.py::TestStripperKremser::test_stripping_at_low_pH`
extrapolates on purpose and is still expected to warn.
```

Tests: `tests/ree/test_ph_validity_range.py`.

(record-derived-ph-defaults)=
##### Operating pH defaults come from the record, not from a literal (#270)

The #270 refit narrowed three of the four cation-exchange windows, and it left
the package with no pH literal that is legal everywhere: `[0, 2]`, `[0.1, 2.5]`,
`[1.5, 3.5]` and `[4, 5]` have empty intersection. The old defaults --- 3.5 for
extraction, 2.0 for scrubbing, 0.5 for stripping, 3.0 in the extractor ---
were each outside at least two of them, and at D2EHPA's refitted coefficients
`D(Nd)` at pH 0.5 is 5.1 (3.7 before #283), so the "strip" was still
extracting.

So the defaults became `None`, and `None` means *ask the record*:

```python
from difflow_ree.database import default_pH, get_extractant

default_pH("D2EHPA", "extraction")   # 2.0
default_pH("D2EHPA", "scrubbing")    # 0.5
default_pH("D2EHPA", "stripping")    # 0.0
default_pH("Cyanex272", "stripping") # 1.5
```

The policy is deliberately crude, and stated rather than tuned:

| duty | where in the window | why |
|---|---|---|
| extraction | the top | the most extracting condition the coefficients can speak to |
| scrubbing | a quarter of the way up | above the strip, below the extract, on the record's own scale |
| stripping | the bottom | the least extracting condition on record |

The endpoints are legal: `_check_ph_range` tests inclusively, so a default at a
window edge never warns. A solvating record has no cation-exchange duty pH ---
`Extractant.default_extraction_pH` and friends return `None` for TBP --- and
`default_pH` falls back to the midpoint of its declared window, which nothing
pH-driven reads.

Every params class that carried a pH literal now carries `None`:
`REEExtractorParams`, `MixerSettlerParams`, `ScrubberParams`, `StripperParams`,
`ExtractStripParams`, `ExtractScrubStripParams`, `SplitShellParams`. The single
units resolve it from the window as above, with one exception: an unset
`StripperParams.pH` is set by the strip acid, `-log10(acid_conc)` (4 M HCl by
default, pH -0.60). The window fractions were not good enough for the
circuits. Stripping at the bottom of D2EHPA's window left 67 % of the Sm and
99.8 % of the Y on the barren organic of a default `ExtractStripCircuit`, so
`ExtractStripParams` and `ExtractScrubStripParams` now read their unset pHs
off the `D` curves instead; see {ref}`circuit-operating-points`. Two of the
remaining classes resolve to something other than their own duty name, on
purpose:

* `SplitShellParams.pH` takes the **scrubbing** default when no
  `product_groups` are given. A split-shell cascade fractionates only where `D`
  straddles 1; at D2EHPA's extraction default both La and Dy are
  quantitatively extracted and the cascade separates nothing. With
  `product_groups`, each section gets its own cut instead; see
  {ref}`splitshellcascade`.
* The screening functions (`separation_factor`, `screen_separation`) take the
  record's **`reference_pH`** --- the condition #268 declares every derived
  constant at. The separation factor does not depend on the choice anyway (one
  shared `b = 3` makes `D_i/D_j` pH-independent); what it buys is that the `D`
  values quoted alongside the verdict are ones the coefficients were fitted for.

`FullSeparationTrain`'s internal pH values are read off the record's own `D`
curves rather than fixed, for the same reason; see {ref}`groupseparator`.

(separation-factors)=
### Separation Factors

The separation factor SF = D1/D2 determines separation feasibility:

```python
from difflow_ree import REEDistribution

dist = REEDistribution(extractant="PC88A", elements=("Nd", "Pr"))

# Separation factor at pH 1.0 (with one shared slope b = 3 the answer is
# 10**(a_Nd - a_Pr) = 2.14 at every pH; see #270)
SF = dist.get_separation_factor("Nd", "Pr", pH=1.0)
print(f"SF(Nd/Pr) = {SF:.2f}")

# An operating pH for the pair. The search range defaults to the record's
# fitted window ([0.1, 2.5] for PC88A).
opt_pH, SF = dist.optimal_pH_for_separation("Nd", "Pr")
print(f"Operating pH: {opt_pH:.2f}, SF: {SF:.2f}")   # 1.08, 2.14
```

Because every acidic record has the same pH slope for every element, the
separation factor does not depend on pH, and there is no SF maximum to find.
`optimal_pH_for_separation` used to return the argmax of the floating-point
noise in a flat scan (pH 4.88 for Cyanex272, outside its window; 1.12 for
naphthenic acid, where `D` is about 1e-10). When the scan is flat it now
returns the pair's extraction cut instead, the pH where the geometric mean of
their `D * (O/A)` is one (`phase_ratio`, default 1), kept inside the search
range, and raises a `SeparationFactorFlatWarning` that says so. For a record
whose SF does move with pH it still returns the SF maximum.

`difflow_ree.units.scrubbing.optimal_scrub_pH(extractant, target, impurity,
min_target_retention=0.95)` returns the lowest pH in the window at which a
counter-current scrub (`n_stages=5`, `phase_ratio=7.5`, the
`ExtractScrubStripCircuit` defaults) still keeps `min_target_retention` of the
target on the organic: the lowest pH removes the most impurity. It used to
maximise `D_t / (D_i + 0.01)`, which grows without bound with pH, so it always
returned the top of its range and never read `min_target_retention`.

#### The tabulated factors are derived from the same correlations

`get_sf_database()` reports a factor per pair at one declared set of conditions,
which is convenient for screening. Those numbers used to be authored by hand in
`separation_factors.yaml`, independently of the `ph_coefficients` in
`extractants.yaml` that every unit operation computes `D` from — two hand-tuned
descriptions of the same physics, never reconciled, disagreeing by up to 8x
(#265). The coefficients ran 1.3–2.8x high on 24 of 27 pairs; all three Y/Dy
pairs ran 4–8x low, which is a disagreement about that pair rather than a
calibration offset. Which number you got depended on which API you reached for.

Neither set was measured at the time, so there was no right one to keep --- and
since #270 the coefficients *are* fitted to named primary sources while
`separation_factors.yaml` never was, which settles it a second time. The tie is
broken by the coefficients being what the simulator actually runs on: a factor derived
from them describes the model you are about to solve, and one authored beside
them describes nothing else in the package. So `separation_factors.yaml` now
declares *which* pairs to report and *at what conditions*, and the values come
from the same `get_separation_factor` as the code above:

```python
from difflow_ree.database import get_sf_database

sf_db = get_sf_database()
data = sf_db.get("PC88A")
data.conditions            # {'pH': 1.33, 'temperature_K': 298, 'concentration_M': 0.5}
sf_db.get_sf("PC88A", "Nd_Pr")   # 2.14, the coefficients' own answer
"Nd_Pr" in data.derived    # True
```

`conditions` is not decoration: it is the point the coefficients are evaluated
at, so a factor quoted from this table is only the factor at that pH. For any
other pH, ask `REEDistribution` directly.

PC88A is the exception that proves it, and the exception is new. Its block used
to say pH 3.5, which is outside the `[0.1, 2.5]` window the #270 refit gave it,
so every factor under it was an extrapolation and loading the database warned
about it. It now says 1.33 — the record's own `reference_pH`. The *numbers* did
not move: since #270 every element on that record shares one slope, `b = 3`,
so `log10 β = a_i − a_j` and the pH, temperature and concentration terms cancel
identically. Moving PC88A's conditions moves `D`, not `β`. That is also what
killed the "optimal pH = 5.0" artifact — with per-element slopes, `β` drifted
with pH and an optimiser would climb it straight out of the fitted window.

A pair given an explicit value in the YAML is used as authored and left out of
`derived`. None ship with difflow_ree: an override is a claim that a measured
number exists which the correlations cannot reproduce, and it needs a citation
beside it.

**Stage counts are derived too (#270).** `separation_factors.yaml` carried an
18-entry `stages_for_99_purity:` table until then. It matched no `β` in the
package — solving Fenske backwards out of its own entries gives betas spanning
1.17–3.15, in an order that tracks neither description — and its `Y_Dy` rows
counted stages of a separation that runs the other way. `get_stages_needed`
now computes

$$N_{min} = \frac{2\ln 99}{\lvert \ln \beta \rvert}$$

— Fenske at total reflux for a 99 %/99 % split of an equimolar binary. It is a
**thermodynamic floor**: a real cascade at a finite solvent ratio needs several
times more, and it inherits every weakness of the `β` beneath it. An authored
value still wins, via `add_pair(..., stages_99=...)` or a
`stages_for_99_purity` block in your own YAML.

(free-extractant)=
### Free extractant, not total

`Q1` Eq. 2.88 — the working correlation the naphthenic-acid form is built on —
is

$$\lg D = \lg K_{ex} + 3\lg (\mathrm{HA})_o + 3\,\mathrm{pH}$$

and Eq. 2.89 says what `(HA)o` is:

$$(\mathrm{HA})_o = [\mathrm{HA}]_0 - 3\,(\mathrm{REA}_3)$$

**free** extractant — total charged minus three monomers per extracted RE(III).
`REEDistribution.get_D` applies the same functional form with `[HA]` the *total*
charged. The two agree on a clean solvent and diverge where the cascade works
hardest, and the error flatters the model: total overstates what is available,
so `D` is overpredicted exactly there (#267).

`solve_free_extractant` closes the loop rather than approximating it:

$$c_{org} = D\big([\mathrm{HA}]_{free}\big)\,c_{aq}, \qquad
[\mathrm{HA}]_{free} = [\mathrm{HA}]_0 - m\,c_{org}$$

Substituting leaves one scalar equation in `[HA]_free` that is strictly
decreasing and changes sign between `0` and `[HA]_0` — one root, bracketed, on a
smooth monotone function. It is solved with `optimistix.root_find`, so it
differentiates by the implicit function theorem rather than by unrolling.

```python
from difflow_ree import REEDistribution, solve_free_extractant

dist = REEDistribution(extractant="D2EHPA", elements=("Nd",), concentration=0.5)
r = solve_free_extractant(dist, "Nd", c_aq=0.03, pH=0.2)

r.D                 # 0.428 -- against free extractant
r.D_total_basis     # 0.638 -- what the correlation says against total
r.overprediction    # 1.49
r.loading_fraction  # 0.154
```

This also restores something #204's closing note recorded as lost: keeping the
correlation's fixed-parameter concentration term left `D` independent of stage
loading. The free extractant enters the *correlation* here — it is not a factor
applied afterwards — so this is not a reintroduction of the double count #190
found. `LoadingIsotherm.apparent_D` caps an answer after `D` is computed; this
changes the input that produced it. **Do not compose the two.**

#### An impossible loading is now impossible

`m * c_org > [HA]_0` leaves negative free extractant, which Eq. 2.89 duly
returns and which has no logarithm. A correlation written against total does not
notice — it takes `log10([HA]_0 / C_ref)` and hands back a finite `D`.

```python
from difflow_ree import implied_loading_fraction, check_loading_capacity

# Z1 Table 4.36 system 2: 0.13 mol/L naphthenic acid, ~0.066 mol/L RE organic
implied_loading_fraction("naphthenic_acid", 0.066, 0.13)   # 1.523
check_loading_capacity("naphthenic_acid", 0.066, 0.13)     # ExtractantCapacityWarning
```

That row is why system 2 was rejected as an anchor in favour of system 1, which
sits at 19 % of capacity — a rejection that previously had to be made by hand.
`action="raise"` turns it into a `ValueError`; the self-consistent solve above
cannot land there at all.

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
  `(1 - theta)^3` (see #189, #190, #191). #267 closes the balance for a
  *single* element at the correlation level (see
  [Free extractant, not total](#free-extractant-not-total)); sharing one
  inventory *between* elements is still what L2 is for.
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

# calibration_pH defaults to the record's own reference_pH (#268), which for
# D2EHPA is 0.82 -- inside the [0, 2] window its #270 refit was fitted over.
net = cation_exchange_network("D2EHPA", ("Nd", "Dy"))
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

#### How #197 (saponification) slotted in

The counter-ion `M+` is a conserved component in **every** shipped network even
when nothing forms from it. Saponification (#197) was therefore exactly one
species row and no change to `mass_action.py`:

```yaml
- name: "M(HA2)"
  phase: organic
  charge: 0
  stoichiometry: {"M+": 1, "(HA)2": 1, "H+": -1}
  log10_K: -2.2688
```

With that row present, sodium partitions between the phases and the
saponification degree becomes an *output* of the same component balances. It
ships as a separate network, `cation_exchange_dimer_saponified`, rather than as
an extra row on `cation_exchange_dimer`, so that the unsaponified network keeps
meaning unsaponified proton exchange: it is the `S = 0` reference the
saponification tests compare against, and a feed carrying sodium as a spectator
salt must not start neutralizing the organic merely because sodium is present.
See [Saponified extractants](#saponification) below.

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
    # correlation are made to agree. The operating pH is an output. Left
    # unset it is the record's own reference_pH (#268) -- 0.82 for D2EHPA.
    calibration_pH=None,
)
section = MassActionSection(params)

feed = section.schema.make_aqueous(
    {"Nd": 0.02, "Dy": 0.02}, acid=0.02, water=55.0
)
# Extractant flow on the FORMAL monomer basis of this layer: 0.5 mol/s at
# 1 L/s is 0.25 M D2EHPA dimer (see "Extractant concentration basis", #374).
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
    n_stages=4, extractant="D2EHPA", elements=("Nd", "Dy"), pH=1.0,
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
element, `H`, `Na`/`NH4`/`K`, `Cl`/`NO3`/`SO4`, water, extractant total (on the
formal **monomer** basis, free plus bound; a unit's solvent is on the record's
dimer basis and `REEExtractor` converts it, #374), loaded organic by element, co-extracted
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
inherits that source's provenance — since #270 that is a named primary source
for every extractant, so a constant derived from one is as defensible as the
fit behind it, no better and no worse. Read the record's own fit note in
`data/extractants.yaml` for the window it covers and the gaps it declares, and
supply your own with `log10_K={"Nd": ..., "Dy": ...}` where you have measured
constants.

The calibration is exact only at the reference condition. Mass action forces

```
d log10 D / d pH = protons_released = 3
```

and since #270 the tabulated pH slopes agree: every acidic record is fitted
with `b = 3` exactly, the stoichiometric slope, rather than a per-element
number floating free of the mechanism it is supposed to express.
`correlation_ph_slope_defect(extractant, element)` returns `3 - b` and now
reads zero everywhere, which is the point of keeping it:

```python
from difflow_ree.equilibrium import correlation_ph_slope_defect
correlation_ph_slope_defect("D2EHPA", "Nd")   # 0.0
```

It used to read 0.55 on that call and up to 0.80 elsewhere, and each of those
tenths was a decade of divergence between the two levels per pH unit away from
the calibration point. The general statement still holds — away from the
calibration pH the two levels differ by exactly
`(p - b)(pH - pH_ref) - c(pH^2 - pH_ref^2)`, which the test suite asserts to
seven digits — but with `p = b` and `c = 0` the first term vanishes
identically and the two levels now agree at every pH, not just at one.

### Validation

| Claim | Measured |
|---|---|
| Reduces to the correlation in the dilute limit | With rare-earth totals at `1e-6` of the free acid, `D` agrees with `REEDistribution.get_D` to better than **2e-5 relative**. What is left is not model error: it is the pH shift from the protons the trace extraction releases, and it scales exactly linearly with the dilution (a ten-fold more dilute feed gives a ten-fold smaller disagreement). |
| Independent check of the `[HA]` dependence | The correlation applies `n * log10(C/C_ref)` (`n = 2.38`, measured, for D2EHPA since #283); the closed model never sees `n` and gets an ideal cube from three dimers in the tableau plus a free-extractant balance. Doubling the extractant moves the closure by exactly 8 and the correlation by 5.2; calibrated at its own charge the closure reproduces the correlation. |
| One solvent, one basis | The same record-basis solvent through `REEExtractor` at both levels gives the same `D` and split (to the dilute-limit agreement above), the same free extractant and the same loading fraction (#374). |
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

(saponification)=
## Saponified Extractants and the Counter-Ion Balance (#197)

Industrial rare-earth circuits do not run on free acidic extractant, and they
do not dose base into every mixer -- that causes local pH excursions which
precipitate hydroxides and stabilize emulsions. They neutralize 30 to 50% of
the extractant *before* it enters the cascade,

```
HA_org + NaOH  ->  NaA_org + H2O
```

so extraction becomes a counter-ion exchange rather than a proton exchange:

```
RE3+ + 3 NaA_org  <->  RE(A)3_org + 3 Na+
```

### Why it changes the answer, not just the bookkeeping

A model without saponification predicts a **pH collapse down the extraction
section that a real plant does not have**. Every trivalent ion extracted
releases three protons into an aqueous phase with nothing to absorb them, so
the model under-predicts loading, over-predicts the stage count required, and
mis-ranks extractants -- while looking entirely plausible, because every stage
is internally consistent. It is the most likely way for a closed model to be
wrong and still pass inspection.

Here is the measurement, from `tests/ree/test_saponification.py`. Both sections
get the **same** feed, the same solvent inventory and the same number of base
equivalents; the only difference is where the base is.

| 8-stage section, D2EHPA, Nd + Dy | pH profile peak-to-peak | Nd + Dy extracted |
|---|---|---|
| base dosed into the aqueous feed | **0.77 pH units** | 0.0287 mol/s |
| same base pre-neutralized onto the organic | **0.26 pH units** | 0.0227 mol/s |

A factor of **2.9** flatter, and 2.3 after normalizing the excursion by the
rare earth actually moved (which is what releases the protons). The advantage
*grows* with the cascade: 2.6 at four stages, 2.9 at eight, 3.1 at twelve --
because the buffer spans every stage, so the longer the cascade the more of it
there is to spend. With no base anywhere at all -- the only thing `difflow_ree`
could express before #197 -- the same feed extracts **more than twenty times
less** Nd.

### The organic is the buffer

This is the mechanism, and it is why the acid-base equilibrium of the organic
is not optional. `(HA)2` and its counter-ion salt `M(HA2)` are a conjugate acid
/ base pair whose proton lives in the *aqueous* phase:

$$[\overline{\mathrm{M(HA_2)}}] = K\,\frac{[\mathrm{M}^+]\,[\overline{\mathrm{(HA)_2}}]}{[\mathrm{H}^+]}
\qquad\Longrightarrow\qquad
\mathrm{pH} = \mathrm{p}K + \log_{10}\frac{S}{1-S} - \log_{10}[\mathrm{M}^+]$$

with `S` the saponification degree. That is Henderson-Hasselbalch for the
organic phase (`organic_buffer_pH`), and its Van Slyke capacity

$$\beta = \ln(10)\,E_T\,S\,(1-S)$$

(`organic_buffer_capacity`, maximal at half neutralization) is what the
released protons are spent against. Perturbing the feed acid by 0.005 mol/s
shows it directly: **more than a third of the added acid is absorbed by the
organic**, released as counter-ion instead of appearing as free protons. An
unsaponified network cannot do that at all -- it has no conjugate base, so its
counter-ion release is identically zero.

```{note}
A free organic `A-` is deliberately **not** a species. A bare anion is not
stable in a low-dielectric diluent; it is always paired with its counter-ion,
and pairing it is exactly what `M(HA2)` is. So the `HA`/`A-` equilibrium is
present, in the only form in which it is physical, as one row of the tableau.
```

### The section

`SaponifiedSection` is a `MassActionSection` on the saponified network. It
overrides exactly two things, and nothing in `mass_action.py` changed:

1. the **solvent** contributes a counter-ion salt species, so a saponified
   solvent brings counter-ion in and, through the tableau, a *negative* proton
   component -- the protons the base removed. It enters at the solvent end of
   the cascade, which is not the same as dosing the equivalent base into the
   aqueous feed at the other end;
2. the **extract** carries the counter-ion still bound to the organic when it
   leaves, so the counter-ion is conserved across the unit's own interface and
   not merely inside the solver.

```python
from difflow_ree.equilibrium import SaponifiedParams, SaponifiedSection

section = SaponifiedSection(SaponifiedParams(
    n_stages=8, extractant="D2EHPA", elements=("Nd", "Dy"),
    aqueous_volumetric_flow=1.0, organic_volumetric_flow=1.0,
    extractant_conc=0.5, saponification_degree=0.35, counter_ion="Na",
))

feed    = section.schema.make_aqueous({"Nd": 0.02, "Dy": 0.02},
                                      acid=0.005, water=55.0)
solvent = section.schema.saponified_organic(
    0.5, 0.35, monomers_per_component=2.0, diluent_flow=4.0,
)

raffinate, extract, info = section(feed, solvent)
info["pH_profile"]                      # flat, and an OUTPUT
info["saponification_degree_profile"]   # also an OUTPUT: the organic re-equilibrates
info["counter_ion_released"]            # the reagent duty and the effluent load
info["pH_flatness"]                     # peak-to-peak span, in pH units
```

`S = 0` reproduces the unsaponified proton-exchange result **bit for bit**, so
the saponified network is a strict generalization rather than a different
model.

### Saponification degree is the manipulated variable

Along with phase ratio per section and scrub/strip acid strength, the degree is
what an operator actually adjusts. A control or RTO layer whose inputs are
stage pH setpoints is modelling a plant that does not exist. So the degree is a
real handle:

- it travels on the **stream**, written by `schema.saponified_organic` or by a
  `Saponifier`, so it can be a tracer -- `jit`, `grad` and `check_grads` all go
  through the section and the implicit solve;
- `saponification_degree_for_pH(section, feed, solvent, target_pH)` inverts the
  section for the degree that hits a pH specification, posed as one more row of
  the same root find (the organic-side twin of `base_addition_for_pH`), so the
  derivative comes out of one implicit differentiation.

```python
from difflow_ree.equilibrium import saponification_degree_for_pH

degree, ok = saponification_degree_for_pH(section, feed, solvent, target_pH=3.2)
```

### The Saponifier

A degree stated as a parameter is an assumption; a saponifier is a *duty*.
Putting the contactor on the flowsheet is what makes the reagent bill and the
effluent load fall out of the same balance the cascade already solves.

```python
from difflow_ree.units import Saponifier, SaponifierParams

unit = Saponifier(SaponifierParams(
    extractant="D2EHPA", saponification_degree=0.35,
    counter_ion="Na",          # "Na", "NH4", "Mg", "K"
    base=None,                 # None -> the default base for the counter-ion
    base_utilization=0.9,      # base that does not reach the organic
))

organic = unit.schema.make_organic(0.5, diluent_flow=4.0)
solvent, spent, info = unit(organic)

info["base_flow"], info["base_mass_flow"]   # mol/s and kg/s of reagent
info["saponification_degree"]               # achieved, an output
info["counter_ion_imbalance"]               # zero to round-off, by construction
```

The contact is *stoichiometric*, not an equilibrium: a strong base against an
extractant of pKa 3-6 goes essentially to completion, and a plant sizes the
saponifier so that it does. All the equilibrium physics stays in the section,
where the organic re-equilibrates and the degree becomes an output again. The
unit does model the two things a plant actually gets wrong -- base that does
not reach the organic, and a recycled solvent that still carries counter-ion,
which is a credit against fresh base -- and it neutralizes no further than the
extractant inventory allows.

### kg base per kg REO

The counter-ion balance that makes the cascade correct is the equation that
predicts the raffinate load, so the reagent and environmental metric costs no
extra machinery. `network.base_equivalents_per_mole_ree` reads **three
equivalents per mole of rare earth** off the tableau -- through the extractant
column, so it is an independent check on the proton column, and it gives 3 for
a divalent counter-ion too. The solved section satisfies the exact identity

```
dT_H(aqueous) + z * dT_M(aqueous) = 3 * (rare earth extracted)
```

to round-off: every extracted trivalent ion occupies three extractant
equivalents and gives back either a proton or a counter-ion.

```python
from difflow_ree.economics import saponification_duty, compare_counter_ions

duty = saponification_duty(
    info["base_flow"],
    {"Nd": extract["F_Nd"], "Dy": extract["F_Dy"]},
    base="NaOH",
    equivalents_per_mole_ree=section.network.base_equivalents_per_mole_ree,
)
print(duty.report())
```

The stoichiometric floor, for Nd (Nd2O3 is 336.48 g/mol for two Nd, i.e.
168.24 g REO per mol Nd):

| Base | kg base / kg REO | kg N / kg REO | kg salt / kg REO (chloride) |
|---|---|---|---|
| NaOH | 3 x 39.997 / 168.24 = **0.7132** | 0 | 1.042 (NaCl) |
| NH3 | 3 x 17.031 / 168.24 = **0.3037** | **0.2498** | 0.954 (NH4Cl) |
| Mg(OH)2 | 1.5 x 58.320 / 168.24 = **0.5200** | 0 | 0.849 (MgCl2) |

`compare_counter_ions` computes that table. It is the comparison the whole
feature exists to make possible: **ammonia saponification is the origin of the
ammonium-nitrogen effluent that is the industry's signature pollution problem,
and sodium trades it for a saline raffinate.** Neither number existed before
#197, because there was no counter-ion anywhere in the extraction path.
Nothing here prices the effluent -- an ammonium-nitrogen discharge limit is a
regulatory fact, not a correlation -- so the loads are reported and the
valuation is left to the caller. `REO` is not always `RE2O3`: the oxide mass is
taken from the element record's own formula, so `CeO2`, `Pr6O11` and `Tb4O7`
are handled correctly.

### The extractant record

```yaml
D2EHPA:
  ...
  saponification:
    counter_ion: Na           # H | Na | NH4 | Mg
    degree: 0.35              # an operating DEFAULT, not a property
    log10_K: null             # calibrated from the degree; see below
    reference_pH: 3.0
    reference_counter_ion: 0.1   # M
```

`Extractant` carries `counter_ion`, `saponification_degree` and
`saponification_log10_K`; `create_custom_extractant` takes all three. The
record rejects a degree outside `[0, 1]`, a degree with no counter-ion, a
degree with `counter_ion: H` (which means un-neutralized proton exchange), and
a degree on an extractant that releases no protons.

### Where the constant comes from

Nowhere measurable in this repository, and it is worth being blunt about that.
`saponification_log_K` inverts

$$K = \frac{S}{1-S}\,\frac{[\mathrm{H}^+]}{[\mathrm{M}^+]}$$

at a *stated* reference degree and condition, so the constant is a restatement
of a declared operating point rather than a number from a paper -- the same
discipline `log_K_from_correlation` follows for the extraction constants. The
YAML default, `log10 K = -2.2688`, is `S = 0.35` at pH 3.0 with 0.1 M
counter-ion, and `SaponifiedSection` recalibrates it from the extractant
record's own reference block. Supply a measured constant for design numbers.

### A divalent counter-ion is a different tableau

Magnesia saponification is real, and it is a *different network* for exactly
the reason a divalent anion is: the counter-ion component carries charge +2,
the salt neutralizes two extractant equivalents and releases two protons, so
the row is `M(HA2)2`, not `M(HA2)`. `divalent_counter_ion_template` derives it
from the shipped monovalent template, so the stoichiometry has one source and
the two cannot drift apart; `SaponifiedSection` builds it automatically for
`counter_ion="Mg"` and checks the schema's charge against the network's rather
than trusting either.

### Validation

| Claim | Measured |
|---|---|
| Counter-ion conserved across a section | Aqueous plus organic, in equals out to **machine precision**, for Na, NH4 and Mg |
| Saponified cascade holds a flatter pH profile | **0.26 against 0.77** pH units peak-to-peak on the same reagent, a factor of 2.9; 2.3 after normalizing by the rare earth moved; the advantage grows with the stage count |
| `S = 0` reproduces proton exchange | **Bit for bit**: pH profile identical to `1e-16`, raffinate and extract flows to `1e-12` relative |
| Three equivalents of base per mole of rare earth | Read off the tableau through the extractant column (3.0 for monovalent *and* divalent), and `dT_H + z dT_M = 3 dRE` on the solved section to **`1e-10` relative** |
| kg base per kg REO | Cross-checked against the hand calculation to `5e-5` absolute for NaOH, NH3 and Mg(OH)2 |
| The organic buffers | More than **a third** of an aqueous acid perturbation is absorbed by the organic; the buffered section's pH moves less than two thirds as far |
| `Saponifier` conservation | Counter-ion imbalance below `1e-18` at any utilization or overdose |
| Gradients | `grad` matches central differences to `1e-4` relative through the implicit solve; `check_grads` passes; `jit` works |

### What is deliberately not modelled

Hydroxide as a species (so the saponifier contact is stoichiometric rather than
an equilibrium), extractant loss to the aqueous phase, third-phase formation --
which a high saponification degree genuinely does cause -- water transfer into
the organic with the counter-ion, and any treatment cost for the effluent
loads. The counter-ion is also taken as a spectator in the aqueous phase: no
sodium complexation with the medium anion.

---

## Unit Operations

(reeextractor)=
### REEExtractor

**Location**: `difflow_ree/units/extraction.py`

**Class**: `REEExtractor`

**Description**: Multi-stage counter-current extraction cascade using the Kremser equation.

#### Parameters

<!-- doc-test: skip: Params signature listing, not runnable code -->
```python
@dataclass
class REEExtractorParams:
    n_stages: int              # Number of extraction stages
    extractant: str            # Extractant name (D2EHPA, PC88A, etc.)
    elements: tuple[str, ...]  # REE elements to track
    pH: float | None = None    # Operating pH; None = the extractant record's
                               # own default extraction pH, the top of its
                               # fitted validity window (#270)
    extractant_conc: float = 0.5  # M, record basis: 0.5 M D2EHPA dimer
                                  # = 1.0 M formal (#374)
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
    pH=1.0,   # inside D2EHPA's [0, 2]; omit it for the record's own default
)
extractor = REEExtractor(params)

# Create feed and solvent streams.
# The solvent must name the extractant and/or the diluent the extractor is
# configured with: those two species are the organic phase, everything else
# in a stream is aqueous. A stream carrying neither raises (#192).
feed = make_stream({"H2O": 1.0, "La": 0.1, "Ce": 0.2, "Nd": 0.15, "Dy": 0.05}, T=298.15, P=101325.0)
solvent = make_stream({"D2EHPA": 0.2, "kerosene": 1.0}, T=298.15, P=101325.0)

# Run extraction
raffinate, extract, info = extractor(feed, solvent, T=298.15, pH=1.0)

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

with $m$ the extractant units bound per REE, counted on the same basis as the
extractant concentration and flow, read from the extraction mechanism declared
in `data/extractants.yaml` (`Extractant.basis_units_per_ree`). It is 3 for the
acidic organophosphorus extractants, which are declared as three dimers (so
`extractant_conc = 0.5` is 0.5 M *dimer*, 1.0 M nominal), and 3 for TBP.
The capacity is $1/m$ mol REE per mol extractant, and the free-extractant
exponent in `LoadingIsotherm.apparent_D` is the same $m$, so the two cannot
disagree. (`Extractant.monomers_per_ree`, 6 for the dimers, is the monomer
count; it is not the divisor of a dimer-basis concentration. Dividing by it
halved the capacity until #374: 0.5 M dimer extracted at most 0.081 M REE per
litre of feed instead of 0.167 M.)

:::{warning}
This changed exported API. `LoadingIsotherm.max_loading` was a constructor
field defaulting to 0.33; it is now a read-only property equal to $1/m$, so
`LoadingIsotherm(max_loading=0.33)` raises `TypeError` — pass `m=3.0` instead,
and note that the acidic organophosphorus extractants derive $m=3$ (three
dimers), 1/3 mol REE per mol dimer, as the old literal 0.33 roughly claimed. The `"stoichiometry"` and `"max_loading"`
keys were removed from the public `EXTRACTANT_CAPACITIES` dict (read
`get_extractant(name).basis_units_per_ree` instead; since #268 that object is no
longer a `dict` at all but a mapping deriving its values on access), and
`loading_correction()`
now raises the free fraction to `isotherm.m` rather than a literal 3 against a
halved capacity, which moves its output by a factor of ~25 for D2EHPA.
:::

#### The Langmuir constants are derived, not stored (#268)

`EXTRACTANT_CAPACITIES` used to carry per-extractant, per-element `typical_K_L`
literals derived once from `data/extractants.yaml` and then hand-synced. In the
trace limit the Langmuir isotherm is $q = q_{max}K_Lc$ and the distribution
ratio gives $q = Dc$, so

$$K_L = \frac{D(\text{reference conditions})}{q_{max}}, \qquad
q_{max} = \frac{[\mathrm{HA}]_{ref}}{m}$$

— a quantity the YAML already determines. The literals had drifted out of step
with it. Asked what single pH would reconcile each stored table with the
coefficients it came from:

| extractant | best-fitting pH | rms log10 residual |
|---|---|---|
| D2EHPA | 3.02 | 0.62 |
| PC88A | 2.84 | 0.85 |
| Cyanex272 | 3.10 | 0.92 |

Off by factors of 4–8 at *any* pH, so not merely evaluated at a different
condition: they described an extractant the database no longer contained. And
nothing failed when that happened — the suite iterated `list_extractants()`, so
a *missing* extractant was caught and a *stale* one was not.

They are now computed at the record's own declared reference conditions, and
`EXTRACTANT_CAPACITIES` is a mapping that derives on access rather than a dict
of numbers. It indexes, iterates and `.items()` exactly as before, so no call
site changed; what changed is that there is nothing left to hand-sync, and
`tests/ree/test_capacity_derivation.py` asserts `K_L == D(reference)/q_max` for
every element of every extractant.

That derivation needs a **declared** basis, so every record states one:
`reference_concentration` for the charge, and `reference_pH` (cation exchange)
or `reference_nitrate` (solvating) for the driving variable. The `reference_pH`
values are the conditions `separation_factors.yaml` already treats as typical
operating for the same extractant, restated where a derived quantity can reach
them. Changing one moves every derived quantity under it — which is the point
of having it written down.

```python
from difflow_ree.equilibrium.loading import typical_K_L

typical_K_L("D2EHPA")                      # at the record's own reference
typical_K_L("D2EHPA", concentration=1.0)   # what they would be at 1 M
```

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

<!-- doc-test: skip: fragment, needs a built REE flowsheet circuit -->
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

<!-- doc-test: skip: Params signature listing, not runnable code -->
```python
@dataclass
class MixerSettlerParams:
    extractant: str
    elements: tuple[str, ...]
    pH: float | None = None    # None = the record's default extraction pH (#270)
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
    target_elements=("Nd", "Dy"),  # labels the diagnostics; see below
    pH=0.5,  # low enough to reject La, Ce, and inside D2EHPA's [0, 2];
             # omit it and the record's own scrubbing default is used
)
scrubber = REEScrubber(params)
```

#### `target_elements` is a reporting label (#288)

It does **not** steer the calculation. Every element in `elements` is scrubbed
through the same two-inlet Kremser solve on its own `D`, so changing
`target_elements` changes `info["target_retained"]`, `info["impurity_removed"]`
and the `"is_target"` flags --- and leaves both outlet streams bit-identical.
What decides which elements stay in the organic is `pH` (through each element's
`D`), `n_stages` and the scrub/organic phase ratio.

It is therefore optional, and defaults to no labelling. What it must not be is
wrong: a name that is not in `elements` labels nothing, which reads like "the
scrub retained none of the target", so it raises instead.

```python
from difflow_ree import ScrubberParams

try:
    ScrubberParams(n_stages=5, extractant="D2EHPA",
                   elements=("La", "Nd"), target_elements=("Y",))
except ValueError as err:
    print(err)
# ValueError: target_elements ['Y'] are not in elements ('La', 'Nd')
```

`ExtractScrubStripParams.target_elements` is the same kind of label --- it
selects which elements `target_recovery`, `target_purity` and
`impurity_rejection` are reported over --- and is checked the same way.

#### `scrub_type` is deprecated (#288)

The field was declared `Literal["acid", "ree", "water"]` and the class
advertised a "scrub-type-dependent boundary condition", but `__call__` never
read it: all three values gave identical outlets. There is nothing left for it
to select, because everything it claimed to switch on is already carried by
arguments the scrubber does read:

- `"acid"` against `"water"` is the scrub solution's acid strength, which is
  `pH` (and `nitrate_conc` for a solvating extractant).
- `"ree"` --- a scrub carrying target REE, as in a strip liquor refluxed to the
  scrub end --- is the REE content of the `scrub_solution` stream you pass. The
  two-inlet Kremser solve takes that as a boundary condition through
  `F_scrub_in` (#284), so it changes the answer where a mode flag never did.

Setting it raises `ScrubTypeDeprecationWarning`; drop the argument.

```{note}
`elements` must name **every** REE present in either inlet. An untracked REE
is dropped from both outlets and the section does not conserve it;
`info["dropped_species"]` lists it. Any other species (a saponification
counter-ion such as `Na_org`, a second extractant, a modifier, acid in the
scrub liquor) does not partition in this model and leaves with the phase it
came in. The same holds for `REEStripper`. Both used to drop these species
too, so a saponified solvent lost its counter-ion in the scrub.
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
    acid_conc=4.0,  # M strip acid; sets the strip pH to -log10(4) = -0.60
)
stripper = REEStripper(params)
```

The strip acid sets the strip pH. With `pH` unset, `acid_conc` (4 M by
default) gives `pH = -log10(acid_conc)`, counting one free proton per formula
unit of HCl, HNO3 or H2SO4. Give `pH` instead and `acid_conc` is filled in as
`10**-pH`; give both with different meanings and the constructor raises.
`acid_conc` used to be reported and never read, so 0.01 M and 8 M acid
stripped identically. Stripping Dy from D2EHPA needs strong acid; the pH it
takes is below D2EHPA's fitted window, and the distribution model warns about
the extrapolation.

An `acid_conc` that sets the pH may not exceed `max_strip_acid` (default
6 M, about the strongest HCl/HNO3 strip used in practice; concentrated HCl is
~12 M): a stronger one raises rather than being capped, because capping would
run a different strip from the one named. Raise `max_strip_acid` (or set it
to None) if the liquor is real. An explicit `pH` is never limited.

(ceriumoxidizer)=
### CeriumOxidizer

**Location**: `difflow_ree/units/cerium.py`

**Description**: Oxidizes Ce³⁺ to Ce⁴⁺ and precipitates as CeO₂.

Cerium is unique among lanthanides because it can be oxidized from Ce³⁺ to Ce⁴⁺, enabling selective removal.

#### Parameters

<!-- doc-test: skip: Params signature listing, not runnable code -->
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

The conversion is a screening model, `ce_conversion` x pH factor x
temperature factor x oxidant efficiency. The pH factor is 0 at pH 6 and 1 at
pH 8 and above (pH 8 is the documented operating point, where Ce(III) is
precipitated as the hydroxide that air oxidises); the temperature factor is 1
at 353.15 K; the oxidant efficiencies are air 0.85, H2O2 0.95, NaOCl 0.98 and
electrolytic 0.99. So `ce_conversion` is the conversion at the reference
conditions with an ideal oxidant, and the defaults give 0.95 x 0.85 = 0.81.
The pH ramp used to run to pH 10, so the documented default pH scored 0.5 and
the unit converted 40 % against a stated 95 %. Every non-REE species in the
feed (acid, impurity metals) passes through to the filtrate.

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

All three precipitators share three rules. Precipitation is capped by the
reagent supplied: no more REE than the oxalate or carbonate fed can bind (1.5
per REE), or than the base fed can pay for (3 OH- per REE, after it has
neutralised any HCl, HNO3 or H2SO4 present). Every non-REE species of both
inlets leaves in the filtrate, including the precipitant's water and its
unreacted excess; the bound reagent leaves in the solid. And an element with
no solubility product in the tables (Ho, Er, Tm, Yb, Lu) raises a
`ValueError` naming it when the unit is built; no constants are invented for
them. Before these rules, 0.5x oxalate precipitated 70.5 % of the REE, the
filtrate lost every species but water and REE, and the hydroxide route never
read its precipitant.

(oxalateprecipitator)=
### OxalatePrecipitator

**Description**: Precipitates REE as oxalate, which can be calcined to oxide.

**Reaction**: 2REE³⁺ + 3C₂O₄²⁻ → REE₂(C₂O₄)₃↓

```python
from difflow.streams import make_stream
from difflow_ree import OxalatePrecipitator, PrecipitatorParams

params = PrecipitatorParams(
    elements=("Nd", "Dy"),
    precipitant_excess=1.5,  # 50% excess
    target_conversion=0.995,
)
precipitator = OxalatePrecipitator(params)

# Feed is stripped REE solution, precipitant is oxalic acid
feed = make_stream({"H2O": 55.5, "Nd": 0.10, "Dy": 0.02}, 298.15, 101325.0)
oxalic_acid = make_stream({"H2O": 10.0, "C2O4": 0.25}, 298.15, 101325.0)
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

feed = make_stream({"H2O": 55.5, "La": 0.05, "Ce": 0.05, "Nd": 0.05, "Dy": 0.02},
                   298.15, 101325.0)
naoh_solution = make_stream({"H2O": 10.0, "OH": 0.8}, 298.15, 101325.0)

# pH-selective precipitation: the setpoint pH decides how much of each REE is
# above its hydroxide solubility, and the NaOH supplied has to pay for it
filtrate, solid, info = precipitator(feed, naoh_solution, pH=8.5)
info["reagent_scale"]   # < 1 when the base ran short
info["pH_final"]        # the setpoint, or the pH of the net acid/base excess

# Find selective precipitation pH range
min_pH, max_pH = precipitator.selective_precipitation_pH("Dy", "La")
print(f"pH range for Dy/La separation: {min_pH:.1f} - {max_pH:.1f}")
```

---

(flowsheet-templates)=
## Flowsheet Templates

Each circuit below returns one results dict: its outlet streams under
named keys beside recovery, purity and mass-balance figures. That suits a
script, not a flowsheet, which maps a unit's return value onto its outlets
by position. So the palette entries under these four names are
stream-returning wrappers from `difflow_ree.flowsheets.palette`, taking
the same `Params` and running the circuit unchanged:

| Palette name | Class | Inlets | Outlets, in order |
|---|---|---|---|
| `ExtractStripCircuit` | `ExtractStripUnit` | `feed` | `raffinate`, `product`, `barren_organic` |
| `ExtractScrubStripCircuit` | `ExtractScrubStripUnit` | `feed` | `raffinate`, `scrub_liquor`, `product`, `barren_organic` |
| `SplitShellCascade` | `SplitShellUnit` | `feed`, `solvent` | `product_1..n` (organic), `raffinate` (aqueous) |
| `FullSeparationTrain` | `SeparationTrainUnit` | `feed` | `light_REE`, `middle_REE`, `heavy_REE` (or `ce_depleted` without group separation), then `CeO2` when cerium removal runs |

The rest of the results dict comes back as the unit's `info`. Wiring a
circuit class itself into a `Flowsheet` raises a `ValueError` naming
these wrappers, rather than passing the dict downstream as a stream.
Like the circuits, the wrappers compute their metrics with Python
`float` and cannot be traced; for a traced train with the organic loop
closed, use the [modules](#separation-trains).

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

With its pHs unset it extracts where every element has `D * (O/A) >= 10`
and strips where every element has `D * (O/A) <= 0.1`; see
{ref}`circuit-operating-points`. `design_extract_strip(feed_composition,
extractant, target_recovery=0.99)` returns params at those pHs with both
stage counts sized from the actual `D` to meet the recovery.

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

<!-- doc-test: skip: Params signature listing, not runnable code -->
```python
@dataclass
class ExtractScrubStripParams:
    extractant: str
    elements: tuple[str, ...]
    target_elements: tuple[str, ...]  # Elements to recover
    n_extraction_stages: int = 10
    n_scrubbing_stages: int = 5
    n_stripping_stages: int = 5
    # None on any of the three: read off the D curves for these elements
    # and targets (see "Where the circuits operate" below).
    extraction_pH: float | None = None
    scrubbing_pH: float | None = None   # lower pH rejects light REE
    stripping_pH: float | None = None
    solvent_to_feed_ratio: float = 1.0
    scrub_to_solvent_ratio: float = 0.2
    strip_to_solvent_ratio: float = 0.5
    nitrate_conc: float | None = None        # solvating extractants (TBP)
    strip_nitrate_conc: float | None = None  # TBP: min(nitrate_conc, 1 M)
    recycle_scrub_liquor: bool = False       # return the scrub liquor to the extraction feed
    recycle_tol: float = 1e-10
    recycle_max_iter: int = 500
```

With `recycle_scrub_liquor=True` the scrub liquor, which carries the
co-extracted non-targets and some target, is mixed back into the extraction
feed, as in a plant (#377). The loop is closed with a tear stream and solved as
a fixed point (`results["recycle"]` reports `converged`, `iterations` and the
residual; the liquor in `results["scrub_liquor"]` is then internal, not an
outlet, and the mass balance does not count it). It gives the extraction
section reflux, which lifts the purity and recovery ceiling of the stand-alone
circuit. The group separator and `FullSeparationTrain` recycle by default
(`recycle_scrub_liquor=True`), which stops the default train sending 63 % of
the Gd to the light product.

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
    # Leave the three pH values unset and they are read off D2EHPA's D
    # curves for the Nd / Ce boundary: 0.34 / 0.10 / -1.02.
)
circuit = ExtractScrubStripCircuit(params)

# A leach liquor of about 0.09 M REE: 55.5 mol of water is one litre.
feed = make_stream({
    "H2O": 55.5,
    "La": 0.025,
    "Ce": 0.045,
    "Nd": 0.015,
    "Dy": 0.002,
}, T=298.15, P=101325.0)

# Run circuit
results = circuit(feed)

print(f"Target purity: {results['target_purity']:.1%}")   # 98.9%
for elem, recovery in results['target_recovery'].items():
    print(f"{elem} recovery: {recovery:.1%}")             # Nd 59.4%, Dy 99.8%
```

Concentration matters. The extractor caps the organic loading at the
extractant's capacity (one REE per `basis_units_per_ree` extractant units), so a feed
written as `{"H2O": 1.0, "La": 0.10, ...}`, which is several molar in REE,
saturates the solvent at any pH and sends most of the feed to the raffinate.
That was this example until the operating-point audit: purity 39 %, with 83 %
of the feed in the raffinate.

The Nd recovery is the price of a scrub that is not refluxed. The circuit
sends its scrub liquor out rather than back to the extraction section, so
the Nd it washes off with the Ce is lost; the stage counts and pHs trade that
loss against purity. `design_extract_scrub_strip` makes that trade for you:

```python
from difflow_ree.flowsheets.extract_scrub_strip import design_extract_scrub_strip

params = design_extract_scrub_strip(
    {"La": 0.025, "Ce": 0.045, "Nd": 0.015, "Dy": 0.002},
    target_elements=("Nd", "Dy"), extractant="D2EHPA",
    target_purity=0.95, target_recovery=0.75,
)
# 3 / 9 / 2 stages, extraction at pH 0.34 and scrub at 0.20; in simulation
# 95.8 % purity, Nd 76.2 % and Dy 99.0 % recovery
```

It searches the extraction and scrubbing pH around their cuts and all three
stage counts, predicting each candidate with the same Kremser fractions the
units evaluate, and returns the smallest total stage count that meets both
targets; it warns and returns its best attempt when none does. It does not
model the loading limiter, so it holds for feeds well below the solvent's
capacity.

(circuit-operating-points)=
#### Where the circuits operate

An element moves to the other phase of a counter-current section when its
factor `D * (O/A)` crosses one. A section therefore separates two groups only
at a pH where that crossing falls between them, and it extracts or strips
everything only where every element is well to one side. Any section pH left
as `None` in `ExtractStripParams` or `ExtractScrubStripParams` is read off the
extractant's `D` curves with that rule
(`difflow_ree.equilibrium.operating_points.cut_pHs`):

| Section | Condition |
|---|---|
| extraction, with targets | geometric-mean `D * (O/A)` of the boundary pair is 1; the pair is the least extractable target and the most extractable non-target |
| scrubbing, with targets | the same pair's mean `D` equals the scrub aqueous/organic ratio |
| extraction, no targets | the least extractable element has `D * (O/A) = 10` |
| stripping | the most strongly held target (every element, without targets) has `D * (O/A) = 0.1` |

The phase ratios are the ones the units actually run at: the organic flow is
the diluent volume (the extractant entry is a moles-per-volume charge and does
not count, #373), so at the defaults the extraction runs at O/A 1, the scrub
at 5 and the strip at 2, whatever the extractant concentration. An extractant whose `D` does
not move with pH (TBP) has no pH cut and keeps the window defaults; a TBP
circuit instead strips at its own `strip_nitrate_conc`, by default
`min(nitrate_conc, 1 M)`, the bottom of the 1 to 6 M window the TBP record
documents. A pH you give is used as given.

These used to be fixed fractions of the extractant's fitted window (#270).
After the refit they did not sit between the groups: a D2EHPA
`ExtractScrubStripCircuit` with targets Gd/Tb/Dy/Y scrubbed at pH 0.5 and
stripped at 0, and returned a product of 0.09 % target purity and 0.2 % Y
recovery. At the cuts it returns 99.6 % purity and 97.5 % Y recovery.
Stripping the heavy REE from D2EHPA needs strong acid, so the strip cut lies
below D2EHPA's fitted window and the distribution model warns about the
extrapolation instead of clamping the pH. When the targets are not the more
extractable group (no single pH extracts them and rejects the rest), an
`OperatingPointWarning` says so.

(strip-acid-floor)=
#### The strip acid floor

For the heavies on D2EHPA the cut is pH -1.17, 14.6 M H+ on the plugin's
concentration scale: beyond any real strip liquor (concentrated HCl is about
12 M, and plants strip the heavies with 4 to 6 M) and 1.2 pH units below the
`[0, 2]` window the coefficients were fitted over. So a strip pH the code
chooses is floored at `-log10(max_strip_acid)`, with `max_strip_acid = 6 M`
by default (pH -0.78), on `ExtractStripParams`, `ExtractScrubStripParams`,
`GroupSeparator`, `SeparationTrainParams` and the two design functions.

| D2EHPA, targets Gd/Tb/Dy/Y | strip pH | Y recovery | Y left on the barren organic |
|---|---|---|---|
| the cut (`max_strip_acid=None`) | -1.17 (14.6 M) | 97.5 % | 0 |
| the 6 M floor (default) | -0.78 | 63.6 % | 34.7 % of the Y entering the strip |

What the floor leaves on the solvent is reported, not hidden: the circuits
return `results["strip_retained"]` (the fraction of each element entering the
strip that leaves on the barren organic), a `FullSeparationTrain` counts it as
solvent holdup, and the params raise a `StripAcidLimitWarning` naming the
elements held. `design_extract_strip` and `design_extract_scrub_strip` size
the strip at the floor with the same Kremser fractions, so where an element
still strips there (`D * O/A < 1`, as Dy does) they add strip stages
instead (Nd/Dy: 4 strip stages at the floor against 2 at the cut, same
99 % recovery); where it does not (Y), no stage count can, and they warn
with `StripAcidLimitWarning`. PC88A (cut pH -0.47, 2.9 M) and Cyanex272
(0.59) cut inside the limit and are unchanged.

A `stripping_pH` you pass is used as given, below the floor or not. The floor
is a stopgap: what is missing is D2EHPA distribution data down to strong acid
(#384).

(splitshellcascade)=
### SplitShellCascade

**Description**: Multi-product split-shell cascade for producing multiple pure REE streams.

The aqueous passes the sections in order (`split_points` divide `n_stages`
into them) and each section's organic extract is a product; the raffinate is
what is left. The solvent enters at the raffinate end, so REE it already
carries enter the last section as its organic inlet, and the mass balance
counts feed plus solvent. `split_points` must be strictly increasing and each
in `[1, n_stages - 1]`.

```python
from difflow.streams import make_stream
from difflow_ree.flowsheets.split_shell import SplitShellCascade, SplitShellParams

comp = dict(La=0.025, Ce=0.045, Pr=0.005, Nd=0.015, Sm=0.002, Eu=0.0005,
            Gd=0.001, Tb=0.0002, Dy=0.0005, Y=0.002)
params = SplitShellParams(
    extractant="D2EHPA", elements=tuple(comp), n_stages=20, split_points=(10,),
    product_groups={"heavy": ("Gd", "Tb", "Dy", "Y"), "middle": ("Sm", "Eu"),
                    "light": ("La", "Ce", "Pr", "Nd")},
)
feed = make_stream({"H2O": 55.5, **comp}, 298.15, 101325.0)
solvent = make_stream({"kerosene": 55.5, "D2EHPA": 27.75}, 298.15, 101325.0)
result = SplitShellCascade(params)(feed, solvent)
result["section_pHs"]          # (-0.26, 0.07): one cut per section
result["products"]["product_1"]["flows"]   # the heavies
```

`product_groups` lists one group per section in the order the aqueous meets
them, most extractable first, optionally with one more for the raffinate.
With groups and no `pH`, each section runs at its own cut at the cascade's
actual O/A: where the mean `D * (O/A)` of its least extractable member and the
most extractable element still to come is one. Above, more than 95 % of each
heavy leaves in `product_1` and the lights leave in the raffinate. The middle
product stays impure: a section with no scrub cannot reject the light REE it
co-extracts. Without groups every section shares one pH (`pH`, or the
record's scrubbing default) and the cascade makes one useful split;
`section_pHs` sets the pHs explicitly. `product_groups` used to be stored and
never read.

---

(groupseparator)=
### GroupSeparator

**Location**: `difflow_ree/flowsheets/full_train.py`

**Class**: `GroupSeparator`

**Description**: Splits a mixed REE feed into light, middle and heavy groups with two `ExtractScrubStripCircuit`s in series.

Individual REE separation is hard because adjacent lanthanides have
separation factors near 1.5. Group separation is the first cut that is
*not* hard: the distribution coefficients spread monotonically across the
series, so heavies, middles and lights can be taken apart at a cost per
stage that individual separation cannot match. Almost every real plant
does this first and separates individual elements only within a group.

Two circuits do it, each splitting its feed at a boundary between two
groups:

1. **Heavy circuit**: separates the heavies (Gd, Tb, Dy, Y) from everything
   else. They leave in the strip product. The rest leave in the raffinate
   (not extracted) and the scrub liquor (co-extracted, then washed back).
2. **Middle circuit**: takes the heavy circuit's raffinate plus scrub
   liquor as its feed and separates the middles (Sm, Eu, to its product)
   from the lights (La, Ce, Pr, Nd, to its raffinate plus scrub liquor).

A circuit whose target group has no element in the feed is skipped, and its
product is empty. Each circuit's solvent leaves through the barren organic,
and any REE still on it is reported as `info["solvent_holdup"]`. In a plant
that REE recycles with the solvent, so it is neither product nor loss.
`FullSeparationTrain` counts it in its mass balance: `closure` includes it
and `recovery` does not.

**Where each section runs.** An element moves to the other phase of a
section when $D \cdot (O/A)$ crosses one. A section therefore separates two
groups only at a pH where that crossing falls between them, that is between
the least extractable target and the most extractable element to reject.
`GroupSeparator` leaves each circuit's pHs unset, so each
`ExtractScrubStripCircuit` reads them off the extractant's own `D` curves for
its group boundary (see {ref}`circuit-operating-points`); the chosen values
are stored as `operating_pH`:

| Section | Condition | Boundary |
|---|---|---|
| extraction | geometric mean of $D \cdot$ O/A for the boundary pair $= 1$ (O/A 1) | lightest target, heaviest rejected |
| scrubbing | the same mean $D$ equals the scrub aqueous/organic ratio (O/A 5) | same pair |
| stripping | the largest target $D \cdot$ O/A $= 0.1$ (strip O/A 2) | most strongly held target |

On D2EHPA this gives extraction at pH −0.20, scrubbing at −0.43 and
stripping at −1.17 for the heavy circuit, and 0.13 / −0.10 / −0.57 for the
middle circuit. Both circuits run in roughly 1 to 2 M acid, and the heavies
strip only from strong acid, as they do in practice. The heavy strip's −1.17
is 14.6 M acid, so it runs at the 6 M `max_strip_acid` floor, pH −0.78,
instead and warns; what it leaves on the solvent (mostly Y) is the
`solvent_holdup` (see [the strip acid floor](#strip-acid-floor)). These pH values lie
below the window the D2EHPA coefficients were fitted over (`[0, 2]`), and
the distribution model says so with an extrapolation warning rather than
clamping the pH. An extractant whose `D` does not move with pH (a solvating
extractant such as TBP, driven by nitrate) has no pH cut, and the sections
fall back to fixed fractions of its window.

The pH values used to be fixed fractions of the window (#270). After the
D2EHPA refit those no longer sat between the groups. The heavy circuit
extracted the lights at $D \geq 10$, and stripping at the bottom of the
window left the heavies on the solvent at $D \sim 100$. The "heavy"
product was light REE, and the heavies never came off. The circuit also
passed on only its scrub liquor, dropping the raffinate where most of the
light REE goes.

The stage counts are fixed inside the unit. To move a boundary, or to
optimize it (both circuits are differentiable, so this is possible), build
the two [`ExtractScrubStripCircuit`](#extractscrubstripcircuit)s yourself
with the pH values as parameters.

```python
from difflow_ree import GroupSeparator

separator = GroupSeparator(
    elements=("La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Dy", "Y"),
    extractant="D2EHPA",
    diluent="kerosene",
    light_elements=("La", "Ce", "Pr", "Nd"),
    middle_elements=("Sm", "Eu"),
    heavy_elements=("Gd", "Tb", "Dy", "Y"),
)

light, middle, heavy, info = separator(feed, T=298.15)
info["group_compositions"]["heavy"]   # mole fractions in the heavy product
```

`nitrate_conc` and `mechanism` are threaded into every section of both
circuits, so a solvating extractant such as TBP works here as it does on
a single extractor (see [Solvating extractants](#solvating-extractants));
`capacity_sharpness` likewise reaches both extraction sections' loading
limiters.

`elements` has no default, so the editor cannot drop a `GroupSeparator`
from the palette alone --- the code context has to supply the element
tuple.

---

(fullseparationtrain)=
### FullSeparationTrain

**Location**: `difflow_ree/flowsheets/full_train.py`

**Class**: `FullSeparationTrain`

**Description**: Cerium removal, then group separation, as one prebuilt plant with an overall mass balance.

```
feed ──▶ CeriumOxidizer ──▶ GroupSeparator ──┬──▶ light REE
          (optional)          (optional)     ├──▶ middle REE
             │                               └──▶ heavy REE
             └──▶ CeO2 (solid)
```

Cerium comes first because it is the one element with an easy handle:
oxidised to Ce(IV) it precipitates as CeO2 and leaves the circuit
entirely, and since bastnasite feeds are often half cerium, removing it
ahead of the extraction shrinks everything downstream. The oxidizer runs at
its own temperature (80 °C) and pH 8, and removes 81 % of the Ce at its
defaults; the train's `T` is the solvent-extraction temperature and is not
passed to it. It used to be, and at 298.15 K the oxidizer's Arrhenius factor
cut a default train's Ce removal to 8 %.

#### Parameters

<!-- doc-test: skip: Params signature listing, not runnable code -->
```python
@dataclass
class SeparationTrainParams:
    elements: tuple = ("La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Y")
    extractant: str = "D2EHPA"
    secondary_extractant: str = "PC88A"   # For Nd/Pr separation
    diluent: str = "kerosene"
    include_ce_removal: bool = True
    group_separation: bool = True
    individual_separation: bool = False   # Not yet implemented
    nitrate_conc: float = None            # Required for solvating extractants
    mechanism: str = None                 # Overrides the extractant's default
    capacity_sharpness: int = 8
    recycle_scrub_liquor: bool = True     # each group circuit refluxes its scrub liquor (#377)
    target_purities: dict = {"Nd": 0.99, "Dy": 0.99, "Y": 0.95}
```

`include_ce_removal` only takes effect if `"Ce"` is in `elements`, and the
group memberships are intersected with `elements`, so a train over a
feed with no heavies simply has no heavy product.

#### Outputs

A single dict, not a tuple of streams:

| Key | Contents |
|---|---|
| `products` | `"CeO2"`, `"light_REE"`, `"middle_REE"`, `"heavy_REE"` --- whichever steps ran |
| `intermediates` | `"ce_depleted"`, the feed to group separation |
| `info` | `"ce_removal"` and `"group_separation"`, each step's own info |
| `holdup` | `"solvent"`: REE left on the stripped solvent |
| `mass_balance` | `total_in`, `total_out`, `holdup`, `closure` (products plus holdup over feed) and `recovery` (products over feed) |

```python
from difflow_ree import FullSeparationTrain, SeparationTrainParams

train = FullSeparationTrain(SeparationTrainParams(
    elements=("La", "Ce", "Pr", "Nd", "Sm", "Gd", "Dy", "Y"),
    include_ce_removal=True,
    group_separation=True,
))
results = train(feed)
results["mass_balance"]["closure"]      # (products + holdup) / REE in
```

`closure` is the mass-balance check: every outlet, the products and the REE
held up on the stripped solvent, over the feed, which should be 1.
`recovery` leaves the holdup out.

`difflow_ree.flowsheets.full_train.design_separation_train(feed_analysis,
target_products, ...)` returns a recommended `SeparationTrainParams` for
a feed assay --- it switches Ce
removal on above 30% Ce and flags individual separation when a
high-value element (Nd, Pr, Eu, Tb) is a target.

#### What it is not

`FullSeparationTrain` is a **fixed sequence**: the topology is decided in
`__init__` by direct calls in a set order, there is no decision variable
over connectivity, `individual_separation=True` is accepted but not yet
implemented, and the `barren_organic` each circuit returns is not
recycled, so every circuit assumes its solvent comes back perfectly
stripped. The circuits and the palette wrappers (`ExtractStripUnit`,
`ExtractScrubStripUnit`, `SplitShellUnit`, `SeparationTrainUnit`) are
traceable: they run under `jax.jit` and `jax.grad` (#381).

For a train whose topology *is* data --- modules plus a connectivity map,
with the organic loop actually closed --- use
`difflow_ree.flowsheets.train.SeparationTrain`, described in
[Separation Trains](#separation-trains) below.

---

(separation-trains)=
## Separation Trains: Topology as Data, with the Organic Loop Closed (#202)

`FullSeparationTrain` above is a **fixed sequence**: cerium removal, then
group separation, then individual separations, by direct calls in a set
order. The topology is decided at import time and there is no decision
variable over connectivity. And the `barren_organic` every circuit
returns "for recycle" is never actually recycled, so every circuit
silently assumes its solvent comes back perfectly stripped.

`difflow_ree.flowsheets.train.SeparationTrain` lifts both limits. A train
is **module instances plus a connectivity map**, and the organic loop is
closed by adding one edge.

### The typed module library

Each module declares aqueous, organic and solid ports with the species
they carry:

| kind | wraps | inlets | outlets |
|---|---|---|---|
| `extract_scrub_strip` | `REEExtractor`, `REEScrubber`, `REEStripper` | `feed` (aq), `solvent` (org) | `raffinate` (aq), `scrub_liquor` (aq), `product` (aq), `barren_organic` (org) |
| `split_shell` | `SplitShellCascade` | `feed` (aq), `solvent` (org) | `product_1..n` (org), `raffinate` (aq) |
| `cerium_oxidation` | `CeriumOxidizer` | `feed` (aq) | `filtrate` (aq), `ceo2` (solid) |
| `precipitation` | `OxalatePrecipitator` / `Carbonate…` / `Hydroxide…` | `feed` (aq) | `filtrate` (aq), `solid` (solid) |
| `saponification` | `Saponifier` (#197) + solvent regeneration | `organic` (org) | `organic` (org), `spent_aqueous` (aq), `bleed` (org) |

The port count is data, not code: a `split_shell` module with three
split points has one more organic outlet than one with two.

Parameter schemas are **not** restated here. `module.describe()` calls
`difflow.catalog.describe_class` on each wrapped unit, so the parameter
half of the schema is derived once, in the core, from the `Params`
dataclasses. What the core catalog cannot supply is the *phase*: both
liquid phases are `difflow.streams.Stream`, so nothing in
`(feed: Stream, solvent: Stream) -> ...` says which inlet is organic.
That is what `difflow_ree.flowsheets.ports.Port` adds, and it is what
lets a wrong connection be refused:

<!-- doc-test: skip: fragment; illustrates a deliberately refused connection on a built train -->
```python
train.connect("sep.barren_organic", "sep.feed")
# PortMismatchError: phase mismatch: sep.barren_organic carries the
# organic phase but sep.feed expects the aqueous phase.
```

A connection that would drop a component the destination does not
declare is refused too, because a flowsheet that quietly loses a
component still converges. Pass `allow_species_loss=True` when the loss
is deliberate.

### Closing the organic loop

```python
from difflow.streams import make_stream
from difflow_ree.flowsheets import (
    ExtractScrubStripModule, ExtractScrubStripParams,
    OperatingLimits, SeparationTrain,
)

# A concentrated La/Nd liquor, deliberately run near the solvent's capacity.
feed = make_stream({"H2O": 100.0, "La": 3.0, "Nd": 3.0}, 298.15, 101325.0)

# Cyanex 272: its fitted window [1.5, 3.5] holds the whole circuit. The strip
# is deliberately one stage, so the solvent comes back loaded.
sep = ExtractScrubStripModule("sep", ExtractScrubStripParams(
    extractant="Cyanex272", elements=("La", "Nd"), target_elements=("Nd",),
    n_extraction_stages=10, n_scrubbing_stages=2, n_stripping_stages=1,
    extraction_pH=2.4, scrubbing_pH=2.3, stripping_pH=1.9,
    extractant_conc=0.3, solvent_to_feed_ratio=1.0,
    scrub_to_solvent_ratio=0.1, strip_to_solvent_ratio=0.5,
), limits=OperatingLimits(third_phase_loading=0.65))

train = SeparationTrain("nd_circuit")
train.add_module(sep)
train.add_feed("leach", feed, "sep.feed")
train.connect("sep.barren_organic", "sep.solvent")   # <- the whole change

result = train.solve()
```

Nothing in `SeparationTrain` iterates. `to_flowsheet()` emits a real
`difflow.flowsheet.Flowsheet` — units in topological order, feeds,
`add_recycle` for every back edge the depth-first pass finds — and
`solve()` calls its Anderson tear solver. The torn organic inlet is
initialised with the fresh, REE-free solvent the open-loop circuit
would have synthesised, which is both the sensible starting point and
literally the assumption the closed loop exists to test.

### What closing the loop costs you

The open loop is not a small approximation. For the circuit above,
solved both ways:

| | raffinate La purity | Nd in raffinate (mol/s) | loaded-organic θ |
|---|---|---|---|
| open loop (fresh solvent every pass) | 0.9928 | 0.0186 | 0.685 |
| closed loop | 0.9055 | 0.2772 | 0.819 |

Fifteen times the impurity, and nine percentage points of raffinate
purity, from one edge. The mechanism is the one #202 names: the barren
organic still carries Nd, that Nd occupies extractant, the
free-extractant fraction and hence the extraction factor fall, and more
Nd leaks past the extraction section into the La raffinate. Strip the
solvent properly (`stripping_pH=1.5, n_stripping_stages=6`) and the two
answers coincide to 1e-9: the difference is the residue, not the
tearing. These are the numbers `tests/ree/test_separation_train.py`
pins. This example used to run La/Dy on D2EHPA at pH 2.4 / 2.3 / 2.3,
outside D2EHPA's fitted window of [0, 2], where the strip stripped
nothing; its table predated the #270 refit.

The loop conserves every component to machine precision: what leaves the
stripper equals what enters the extractor, and the extractant and
diluent inventory is invariant.

**A closed loop with no bleed is an accumulator.** Whatever the stripper
misses circulates for ever, so put a `saponification` module in the loop
and give it a `SolventRegenerationParams(bleed_fraction=...)`. The bleed
diverts a fraction of the circulating organic to regeneration and
replaces it with fresh solvent of the same carrier flow, so the
inventory is unchanged and the residual loading falls.

### Operating boundaries as constraints, not flags

An optimizer walks up to a boundary it is only warned about and then
crosses it, because in the model crossing is profitable. Every module
reports a `ConstraintSet` whose margins are **feasible when ≥ 0**:

```python
constraints = train.constraints(result)
constraints.vector()          # g(x) >= 0, a JAX array
constraints.feasible          # False if anything is violated
constraints.violations()      # worst first
print(constraints.summary())
```

With the third-phase limit above (0.65) the closed loop is feasible, at
a margin of +0.075. Tighten it to `OperatingLimits(third_phase_loading=0.40)`
and the same solve reports:

```
sep.third_phase  third_phase  value=+0.575235  limit=+0.4  margin=-0.175235  VIOLATED
sep.loading      loading      value=+0.575235  limit=+1    margin=+0.424765  ok
```

Four boundaries are available, each only when its limit is declared on
`OperatingLimits`:

- `third_phase` — organic loading below the onset of a second organic
  layer, built on the signed margin #193 added.
- `loading` — extractant saturation, θ ≤ 1. A *different* wall from the
  third phase, which usually sits below it, so a design cannot satisfy
  one by ignoring the other.
- `hydraulic` — two-phase throughput below the installed settler
  capacity. Expressed in the same units as the stream flows, on purpose:
  a litres-per-second capacity would smuggle a hidden unit back into a
  package that is otherwise invariant to the flow unit (#189).
- `phase_ratio` — O/A inside the dispersion band.

Every margin is a traced JAX scalar, so `jax.grad` of a margin with
respect to a design variable works. Note which design crosses in the
example above: at a 0.40 third-phase limit the *open-loop* circuit sits
inside the wall (θ = 0.357) and only the honest closed-loop one crosses
it. An open loop hides the constraint violation as well as the purity
loss.

### Screening before costing: the Fenske bound

`equilibrium.distribution.stages_fenske` is the companion to
`stages_kremser`, and where Kremser is an operating estimate Fenske is a
rigorous lower bound — the total-reflux limit, so no finite
solvent-to-feed ratio beats it:

$$N_\min = \frac{\ln\!\left[\frac{s_E}{1-s_E}\cdot\frac{s_R}{1-s_R}\right]}{\ln \alpha}$$

with `s_E` the fraction of the extract key reporting to the extract and
`s_R` the fraction of the raffinate key reporting to the raffinate (for
an equimolar binary feed, the two product purities). It costs two
logarithms, so a candidate topology whose installed stage count is below
the bound can be struck out before it is solved, let alone costed:

```python
from difflow_ree.flowsheets import screen_separation, screen_train

screen_separation("D2EHPA", "Dy", "La", installed_stages=2,
                  purity=0.99, pH=2.3).admissible     # True
screen_separation("D2EHPA", "Dy", "La", installed_stages=2,
                  purity=0.999, pH=2.3).admissible    # False

screen_train(train, {"sep": ("Dy", "La")}, purity=0.999).summary()
```

Unlike `stages_kremser`, the result is floored at 0 rather than 1: a
separation needing less than one theoretical stage is information a
screening filter should keep, and flooring at 1 would turn a genuine
lower bound into one that is sometimes wrong in the direction that
matters.

### What is deliberately not here

**No discrete search.** difflow avoids Pyomo and MINLP solvers, and #202
keeps it that way. What is provided is the graph representation, the
closed recycle, the constraint handles and the cheap screening bound, so
an *external* discrete layer can drive it. `SeparationTrain.to_dict()` /
`from_dict()` write the connectivity map explicitly, which is the form a
topology enumerator wants.

Route out through `difflow.solvers` (#203), whose `as_nlp` / `as_residual`
wrap a train's continuous subproblem. One thing to know when you do:
discopt's `CustomCall` cannot carry binaries *around* a wrapped
flowsheet — the wrapped call is opaque to the MILP — so the discrete
layer must **decompose**: enumerate or branch on the topology outside and
call in for each fixed topology, rather than hoping the solver will see
connectivity variables through the wrapper.

**Traceability.** `solve()` uses `Flowsheet.solve`, which records its
convergence diagnostics as Python floats and therefore cannot be traced.
`solve_differentiable()` runs the *same* flowsheet graph — same units,
same order, same tear set — through `optimistix.fixed_point`, and gets
implicit differentiation through the converged loop:

<!-- doc-test: skip: fragment; objective is defined in the surrounding discussion, not here -->
```python
jax.grad(objective)(0.5)   # finite through the closed organic loop
```

Its adjoint uses a least-squares linear solve deliberately. A closed
organic loop is structurally singular in its carrier coordinates: the
extractant and diluent come out exactly as they went in, so *any*
solvent inventory is a fixed point. The inventory is a design degree of
freedom, not something the loop determines, and the minimum-norm
solution is the one that holds it fixed.

The `split_shell` and `cerium_oxidation` modules wrap units that still
concretise their diagnostics, so they are correct eager graph nodes but
are not traceable.

---

(custom-elements-and-data)=
## Custom Elements and Data

The built-in database covers 15 REEs and 5 extractant systems, but coverage is uneven -- only naphthenic acid carries coefficients for all fifteen; the three acidic extractants stop at Dy and Y -- and many applications require elements or extractant data not included by default. The `difflow_ree` plugin provides a runtime API for adding your own literature data, following the same pattern as the existing `create_custom_extractant` / `add_extractant` workflow.

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

# Register with the database. Ho already ships in the built-in table, so
# set the shipped record aside first (restored at the end of this section).
db = get_ree_database()
shipped_ho = db.get("Ho")
db.remove_element("Ho")
db.add_element("Ho", ho)

# Now Ho is available alongside built-in elements
print(db.get("Ho").ionic_radius_pm)  # 90.1
print(db.list_by_group("heavy"))     # [..., 'Ho']
```

Elements can also be updated or removed:

```python
from dataclasses import replace

updated_ho = replace(ho, price_usd_kg=65.0)  # corrected data
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
ext_db.add_element_to_extractant(
    "PC88A", "Ho",
    ph_coefficients={"a": -6.20, "b": 2.95, "c": 0.010},  # corrected values
    temperature_coefficient=-2350,
)
```

(adding-separation-factors)=
### Adding Separation Factors

Separation factor data can be added incrementally. You can add individual pairs to existing extractants or create complete entries for new ones.

Adding pairs to an existing extractant:

Anything added this way is an **authored** factor, used as given rather than
derived from the extractant's correlations (#265). Reach for it when you have a
*measured* number the correlations cannot reproduce; when they can, adding the
element to the extractant (`add_element_to_extractant`, above) keeps one
description of the physics instead of two, and every pair involving it follows.

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
ree_db = get_ree_database()
ree_db.add_element("Ho", ho)

# 2. Add extraction coefficients (from your literature source). The sections
# above already added Ho data to the shared database, so clear it first.
get_extractant_database().remove_element_from_extractant("PC88A", "Ho")
get_extractant_database().add_element_to_extractant(
    "PC88A", "Ho",
    ph_coefficients={"a": -6.15, "b": 2.95, "c": 0.010},
    temperature_coefficient=-2350,
)

# 3. Add separation factor data
sf_db = get_sf_database()
for pair in ("Ho_Dy", "Y_Ho", "Ho_Nd"):   # added by the examples above
    sf_db.remove_pair("PC88A", pair)
sf_db.add_pair("PC88A", "Ho_Dy", 1.4, stages_99=20)
sf_db.add_pair("PC88A", "Ho_Gd", 2.5, adjacent=False)

# 4. Use in distribution calculations
dist = REEDistribution(
    extractant="PC88A",
    elements=("Gd", "Dy", "Ho", "Y"),
)
D_ho = dist.get_D("Ho", pH=2.0, T=298.15)   # inside PC88A's [0.1, 2.5]
print(f"D(Ho) at pH 2.0: {D_ho:.2f}")
```

The database objects are process-wide singletons, so put back what the two
Holmium examples changed if you carry on in the same session:

```python
get_extractant_database().remove_element_from_extractant("PC88A", "Ho")
sf_db.remove_pair("PC88A", "Ho_Dy")
sf_db.remove_pair("PC88A", "Ho_Gd")
sf_db.remove_separation_factors("MyExtractant")
ree_db.remove_element("Ho")
ree_db.add_element("Ho", shipped_ho)
```

---

## Economics

The plugin includes economic analysis tools:

```python
from difflow_ree import (
    estimate_capex,
    capex_basis,
    estimate_opex,
    calculate_revenue,
    calculate_profit,
    minimum_selling_price,
)

# Capital cost. `scope` is the most consequential argument -- see below.
capex = estimate_capex(
    annual_ree_tonnes=500.0,
    n_stages_extraction=8,
    n_stages_scrubbing=4,
    n_stages_stripping=4,
    include_precipitation=False,   # selling concentrate, not finished oxide
    year=2024,
    scope="separation_plant",
)
print(capex["total"] / 1e6, "M$")

# Operating cost. Mirror include_precipitation, or the plant is charged for
# precipitant it has no capital for.
opex = estimate_opex(
    annual_ree_tonnes=500.0,
    capex=capex["total"],
    extractant="PC88A",
    include_precipitation=False,
)

# Revenue from a product stream (mol/s by element, contained-oxide basis)
revenue = calculate_revenue({"Nd": 0.02, "Pr": 0.006})

# Profitability
profit = calculate_profit(revenue["total"], opex["total"], capex["total"])
msp = minimum_selling_price(opex["total"], capex["total"],
                            annual_production_kg=500e3, target_roi=0.15)
```

### Battery limits decide the answer

`estimate_capex` takes its *level* from a disclosed project cost and moves it
to the requested capacity by the 0.6 power law and to the requested year by a
CEPCI ratio. Three anchors are available through `scope`, and at the same
capacity they differ by more than a factor of ten -- not because they disagree
about one plant, but because they draw the battery limits around different
amounts of plant:

| `scope` | Anchor | Encloses |
|---|---|---|
| `"sx_retrofit"` | Energy Fuels White Mesa Phase 1, $16 M as-built for 4,500 t/yr REO **feed** | The mixer-settler trains and their tanks, pumps, piping and installation. Buildings, power, utilities, effluent treatment and the licence already existed. |
| `"separation_plant"` (default) | Avalon Nechalacho Geismar PFS, US$302 M (Q4-2011 quotations, ±25 % claimed) for 10,000 t/yr **separated** REO -- 98 % recovery, so feed and product coincide | A standalone separation refinery: the cascade plus precipitation and calcination, reagent handling, effluent treatment, civils, electrical, utilities, engineering and contingency. No mine, no concentrator, no cracking plant. |
| `"integrated"` | Energy Fuels Phase 2, $410 M Class 3 BFS for ~7,554 t/yr separated products | The above plus monazite cracking and leaching. |

`capex_basis(scope)` returns the anchor, its citation key in
`difflow_ree/data/sources.yaml`, its capacity basis, its `includes` and
`excludes` lists, and the accuracy range. Read it before quoting a number.

Two traps the arguments cannot protect you from:

- **Capacity basis.** The retrofit anchor is quoted per tonne of REO *fed*;
  the other two per tonne of separated product. For a bastnasite circuit those
  differ by a factor of several, so scaling a product tonnage against a feed
  anchor silently undersizes the plant.
- **Estimate class.** Capacity-factoring a single project is an AACE 18R-97
  Class 5 method however well defined the anchor was. `capex_basis()` reports
  `derived_class` and `derived_accuracy` (about -50 % / +100 %) alongside the
  anchor's own tighter class, and the derived one is the one to quote.

Dropping a section with `include_precipitation=False` or
`include_ce_removal=False` reduces the total by that section's share of the
anchor's scope. It does not reallocate the money over the rows that remain --
that would quote the price of a plant with the section to a caller who asked
for one without, and the rows would still sum to the total, so nothing would
look wrong.

Stage count moves only the stage-driven fraction of capital -- civils and a
licence do not get more expensive because the cascade grew -- and only if you
pass `n_stages_reference`, the base-case stage count the anchor is taken to
correspond to. It defaults to `None`, meaning no stage adjustment at all,
because none of the anchors publishes a stage count and assuming one would put
an invented number into the level. Pass your own base case to get stage
sensitivity; the anchor is then reproduced exactly at that base case.

### What the section split is checked against

The anchors publish totals. Between them they publish exactly one section:
Avalon's solvent-extraction circuit, "over 1,000 mixer-settlers", at 33 % of
total capital and US$101 million. difflow's section shares are `ESTIMATED` and
are not derived from it, but they are held to it -- mixer-settlers alone come
out below 33 %, the whole SX equipment group about a quarter above, and
`tests/ree/test_capex_anchors.py` fails if that bracket breaks. The same figure
prices a mixer-settler at no more than about US$101,000 in 2011 dollars, which
is worth knowing before accepting any per-stage price.

The same release gives the one operating rate available for a cross-check:
US$5,634 per tonne of separated REO at 10,000 t/yr, reagents 70 % of it,
covering labour, supplies, reagents and maintenance but no capital charge and
no feed cost.

Everything else in `difflow_ree.economics` is `ESTIMATED` in the sense of
`sources.yaml`: the prices, the payability (there is none -- a real offtake pays
a fraction of contained value), the reagent and utility unit rates, the labour
model, and the *section breakdown* of the anchored CAPEX total. Break-even
prices and profitability statements run on those, not on the anchored total, so
what they carry is one order-of-magnitude check, not a citable basis. The
module docstring in `difflow_ree/economics/costs.py` says which is which.

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

# The separation factor is 2.14 at every pH (one shared slope); the pH
# returned is the Nd/Pr extraction cut, with a warning saying so.
dist = REEDistribution(extractant="PC88A", elements=("Pr", "Nd"))
cut_pH, SF = dist.optimal_pH_for_separation("Nd", "Pr")
print(f"Nd/Pr cut at pH {cut_pH:.2f}, SF = {SF:.2f}")   # 1.08, 2.14

# Leave the pHs unset: the circuit cuts at the Nd/Pr boundary itself.
params = ExtractScrubStripParams(
    extractant="PC88A",
    elements=("Pr", "Nd"),
    target_elements=("Nd",),
)
circuit = ExtractScrubStripCircuit(params)

# 0.1 M REE (55.5 mol of water is one litre)
feed = make_stream({"H2O": 55.5, "Pr": 0.03, "Nd": 0.07}, T=298.15, P=101325.0)
results = circuit(feed)

print(f"Nd purity: {results['product_purity']['Nd']:.1%}")      # 95.8%
print(f"Nd recovery: {results['target_recovery']['Nd']:.1%}")   # 35.0%
```

At a separation factor of 2.1 a single pass without reflux buys purity with
recovery: the scrub that removes the Pr removes Nd too, and this circuit does
not return its scrub liquor. `design_extract_scrub_strip({"Pr": 0.03, "Nd":
0.07}, ("Nd",), "PC88A", target_purity=0.97, target_recovery=0.30)` meets
both (97.3 % / 30.4 %); ask for 90 % purity at 50 % recovery and it warns that
no design in its search reaches it. Plants separate Nd from Pr with refluxed
cascades of many tens of stages. This example used to run at a total REE
concentration of 1 M, where the solvent saturated and Nd recovery was 8 %.

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

The extractor is differentiable with respect to its operating pH. (The
`ExtractScrubStripCircuit` reports its diagnostics as Python floats, so
differentiate the individual units, or use a flowsheet's
`solve_differentiable()` as described above.)

```python
import jax
from difflow_ree import REEExtractor, REEExtractorParams
from difflow.streams import make_stream, get_flows

feed = make_stream({"H2O": 1.0, "La": 0.3, "Ce": 0.4, "Nd": 0.3}, T=298.15, P=101325.0)
solvent = make_stream({"D2EHPA": 0.2, "kerosene": 1.0}, T=298.15, P=101325.0)

def separation_objective(pH):
    """Objective: maximize Nd purity x recovery in the loaded organic."""
    params = REEExtractorParams(n_stages=5, extractant="D2EHPA",
                                elements=("La", "Ce", "Nd"), pH=pH)
    raffinate, loaded, info = REEExtractor(params)(feed, solvent)
    org = get_flows(loaded)
    recovery = org["Nd"] / 0.3
    purity = org["Nd"] / (org["La"] + org["Ce"] + org["Nd"])
    return -(purity * recovery)  # Negative for minimization

# Compute the gradient with respect to the extraction pH
grad_fn = jax.grad(separation_objective)
print(f"d(objective)/d(pH) = {grad_fn(0.5):.4f}")
```

---

## See Also

- [Examples: 04_rare_earth_extraction.ipynb](../examples/04_rare_earth_extraction.ipynb) - Basic REE extraction
- [Examples: 09_ree_ndfeb_magnet.ipynb](../examples/09_ree_ndfeb_magnet.ipynb) - NdFeB magnet recycling
- [Examples: 10_bastnasite_separation.ipynb](../examples/10_bastnasite_separation.ipynb) - Bastnasite ore processing
