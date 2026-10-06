"""Cost of goods against a published process: Petrides (2015), section 11.6.3.

D. Petrides, "Bioprocess Design and Economics", Intelligen, Inc., March
2015, pp. 11-63 to 11-71 (an improved version is Chapter 11 of Harrison,
Todd, Rudge and Petrides, Bioseparations Science and Engineering, 2nd ed.,
Oxford University Press, 2015):
https://www.intelligen.com/wp-content/uploads/2020/05/BioProcessDesignAndEconomics_March_2015.pdf

The process is encoded in ``data/petrides2015_mab.yaml``, every number tagged
with its page. Only what the source's stated inputs determine is compared:
production, cycles per batch, elution buffer volumes, resin replacement cost
and media cost. The source gives labor, facility-dependent, QC and
miscellaneous costs as totals without their inputs, so the total cost per
gram ($84/g, Table 11.17) is not a target here.
"""

from pathlib import Path

import pytest

from difflow_bio.economics import cogs_breakdown, load_cost_model
from difflow_bio.economics.cogs import REFERENCE

BENCHMARK = Path(REFERENCE).parent / "petrides2015_mab.yaml"

#: Table 11.17, $/yr
CONSUMABLES = 23.58e6
RAW_MATERIALS = 16.67e6
#: Table 11.15, kg/batch (aqueous buffers, about 1 kg/L)
PROTEIN_A_ELUTION = 10_002.42
HIC_ELUTION = 2_990.00


@pytest.fixture(scope="module")
def result():
    process, basis = load_cost_model(BENCHMARK)
    return process, cogs_breakdown(process, basis)


def test_production_matches(result):
    """19.3 kg per batch and 1,544 kg per year (pp. 11-66, 11-67)."""
    _, out = result
    assert float(out["product_per_batch_g"]) == pytest.approx(19.3e3, rel=5e-3)
    assert float(out["annual_product_g"]) == pytest.approx(1544e3, rel=5e-3)


@pytest.mark.parametrize("step, stated", [("protein_a", 4), ("iex", 3), ("hic", 3)])
def test_cycles_per_batch_match(result, step, stated):
    """The source runs whole cycles (4, 3, 3); load / (DBC x CV) gives
    3.85, 3.01 and 3.00 from its stated loads, capacities and volumes."""
    _, out = result
    assert float(out["steps"][step]["cycles"]) == pytest.approx(stated, abs=0.2)


@pytest.mark.parametrize("step, cv_total, published", [
    ("protein_a", 5 + 14, PROTEIN_A_ELUTION),
    ("hic", 5 + 12, HIC_ELUTION),
])
def test_elution_buffer_matches_table_11_15(result, step, cv_total, published):
    process, out = result
    batches = process.batches_per_year
    per_batch = float(out["steps"][step]["buffer_L"]) / batches
    elution = per_batch * 5 / cv_total
    assert elution == pytest.approx(published, rel=0.06)


def test_resin_cost_is_the_stated_inputs_and_fits_consumables(result):
    """Replacement cost = CV x price x cycles / lifetime x batches, from the
    stated resins (p. 11-69); it must fit inside Consumables (Table 11.17),
    which also holds membranes, filters and bags."""
    process, out = result
    by_hand = 0.0
    for step in process.steps:
        if hasattr(step, "resin_cost_usd_L"):
            cycles = float(out["steps"][step.name]["cycles"])
            by_hand += (step.column_volume_L * step.resin_cost_usd_L * cycles
                        / step.resin_lifetime_cycles * process.batches_per_year)
    assert float(out["resin"]) == pytest.approx(by_hand, rel=1e-9)
    assert 0.0 < float(out["resin"]) < CONSUMABLES


def test_media_cost_fits_raw_materials(result):
    """466.79 kg/batch of media at $300/kg (Table 11.15, p. 11-69) is
    $11.2M/yr, inside Raw Materials ($16.67M/yr, Table 11.17)."""
    _, out = result
    assert float(out["media"]) == pytest.approx(466.79 * 300.0 * 80, rel=1e-4)
    assert float(out["media"]) < RAW_MATERIALS


def test_every_number_in_the_benchmark_says_where_it_came_from():
    for line in BENCHMARK.read_text().splitlines():
        body = line.split("#", 1)[0]
        if ":" not in body:
            continue
        value = body.split(":", 1)[1].strip()
        try:
            float(value)
        except ValueError:
            continue                      # a name or a section, not a number
        assert "[stated p." in line or "[derived]" in line \
            or "[not stated]" in line, line
