# difflow_bio

Notes for Claude Code, loaded when working in this directory (moved out of
the root CLAUDE.md, which keeps only what every session needs).

## `difflow_bio` - Bio manufacturing

- Bioreactors: `ContinuousBioreactor`, `FedBatchBioreactor`
- Separation: `Centrifuge`, `DiscStackCentrifuge`
- Filtration: `Ultrafiltration`, `Diafiltration`, `TFF`
- Chromatography: `ProteinAChromatography`, `IonExchangeChromatography`, `SizeExclusionChromatography`
- Economics (#358): `economics.cogs_breakdown` is the cost-of-goods model;
  costs follow the process (per-step resin, cycles = load/(DBC x CV) with a
  softplus floor at one so it stays differentiable, labor from batches and
  steps, buffers, QC, failed batches). All inputs are data
  (`load_cost_model`, `data/mab_reference.yaml`); every number in that file
  carries a `[database]`/`[carried over]`/`[placeholder]` tag and a test
  refuses an untagged one. Do not present the reference result as a
  benchmark. The benchmark is `data/petrides2015_mab.yaml` (Petrides 2015,
  Intelligen "Bioprocess Design and Economics", sec. 11.6.3), checked by
  `tests/bio/test_economics_benchmark.py`: production, cycles per batch,
  elution volumes, resin and media cost reproduce the source's stated
  inputs. Its labor, facility, QC and total $/g are NOT reproduced (the
  source gives them as totals without inputs); do not claim they are.
