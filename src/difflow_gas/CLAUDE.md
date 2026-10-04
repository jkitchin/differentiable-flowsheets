# difflow_gas

Notes for Claude Code, loaded when working in this directory (moved out of
the root CLAUDE.md, which keeps only what every session needs).

## `difflow_gas` -- Gas transmission networks

- Network model: `GasNetwork` (pipes, compressor stations, valves, control valves, resistors, short pipes; signed flows)
- Decomposition: `decompose` computes the spanning tree, tear set and balance schedule from the topology
- Units: `GasPipe`, `BackPipe`, `PipePressure`, `PressureDrivenPipe`, `Compressor`, `CompressorBoost`, `OpenValve`, `PressureEqual`, `ControlValveDrop`, `SourceHead`, `AffineFlow`, `Junction`, splits
- Flowsheets: `GasNetworkFlowsheet` (signed-flow Anderson + damped differentiable tear solve), `build_network_flowsheet`
- Physics: `weymouth_beta`, `resistor_xi`, `compressor_power`, `smoothed_power_w`, GasLib unit conversions
- Verification: full equation-oriented residual checks (`difflow_gas.verify`)
- Equations: `difflow_gas.residuals.network_residuals` is the single JAX-traceable definition of the equation set; `verify` is the reporting layer over it
- Plotting: `dg.draw_network(net, pos=..., pressures=..., flows=..., highlight=...)` draws a network schematic
- Reconciliation: `reconcile_network`, `monitor_network` (a campaign against a fixed
  model), `reconcile_network_multi` (pool periods sharing a parameter); all three
  fill in the layout's names and scales (see `difflow.reconciliation`)
- Gotchas encoded in docs: solve with `clip_negative_flows=False` (signed flows), damp the tear map (alpha ~ 0.3), pose optimization pressure constraints in squared pressure
