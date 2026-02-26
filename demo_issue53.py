"""Demo: REEExtractor mass conservation fix (issue #53)."""
from difflow_ree import REEExtractor, REEExtractorParams
from difflow.streams import make_stream, get_flows

params = REEExtractorParams(
    n_stages=1,
    extractant="D2EHPA",
    elements=("Dy", "Nd"),
    pH=1.6,
)
extractor = REEExtractor(params)

feed = make_stream(
    flows={"H2O": 10, "Nd": 0.2, "Dy": 0.143827799, "Fe": 0.553},
    T=298.15,
    P=101325.0,
)
solvent = make_stream(
    flows={"Organic": 10.0, "n-Heptane": 20.0, "Nd": 0.0, "Dy": 0.0, "Fe": 0.0},
    T=298.15,
    P=101325.0,
)

raffinate, extract, info = extractor(feed, solvent)

feed_flows = get_flows(feed)
solvent_flows = get_flows(solvent)
raff_flows = get_flows(raffinate)
ext_flows = get_flows(extract)

all_species = sorted(set(feed_flows.keys()) | set(solvent_flows.keys()))

print(f"{'Species':<12} {'Feed':>10} {'Solvent':>10} {'Raffinate':>10} {'Extract':>10} {'Balance':>10}")
print("-" * 64)
for sp in all_species:
    f_in = float(feed_flows.get(sp, 0.0))
    s_in = float(solvent_flows.get(sp, 0.0))
    r_out = float(raff_flows.get(sp, 0.0))
    e_out = float(ext_flows.get(sp, 0.0))
    balance = (f_in + s_in) - (r_out + e_out)
    print(f"{sp:<12} {f_in:10.4f} {s_in:10.4f} {r_out:10.4f} {e_out:10.4f} {balance:10.2e}")
