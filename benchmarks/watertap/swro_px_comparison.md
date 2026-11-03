Reference: WaterTAP 1.8.0, IDAES 2.13.0, Pyomo 6.10.1; RO1D with 160 finite elements. difflow: RK4 with 80 steps.

### Case `base`

**RO stage alone**

| Quantity | WaterTAP | difflow | rel. diff |
|---|---|---|---|
| `recovery_mass_water` | 0.42383 | 0.423827 | -0.00% |
| `recovery_vol` | 0.419961 | 0.419959 | -0.00% |
| `salinity_permeate` | 0.000242628 | 0.000242319 | -0.13% |
| `salinity_concentrate` | 0.0590632 | 0.0590631 | -0.00% |
| `deltaP` | -212208 | -212689 | -0.23% |
| `flux_mass_water_avg` | 0.00837265 | 0.00837259 | -0.00% |
| `Re_in` | 266.878 | 266.878 | +0.00% |
| `Re_out` | 149.103 | 149.103 | +0.00% |

**Plant (RO + pressure exchanger + pumps + costing)**

| Quantity | WaterTAP | difflow | rel. diff |
|---|---|---|---|
| `LCOW` | 0.410379 | 0.410409 | +0.01% |
| `SEC` | 2.4696 | 2.46983 | +0.01% |
| `work_P1` | 3355.26 | 3355.23 | -0.00% |
| `work_P2` | 378.44 | 378.773 | +0.09% |
| `px_hp_pressure` | 5.97847e+06 | 5.97801e+06 | -0.01% |
| `total_capital` | 19483.9 | 19485.1 | +0.01% |
| `total_operating` | 2946.48 | 2946.69 | +0.01% |
| `px_flow_vol_lp` | 0.000580506 | 0.000580509 | +0.00% |
| `ro_feed_salinity` | 0.035 | 0.035 | +0.00% |

### Case `high_pressure`

**RO stage alone**

| Quantity | WaterTAP | difflow | rel. diff |
|---|---|---|---|
| `recovery_mass_water` | 0.591697 | 0.592205 | +0.09% |
| `recovery_vol` | 0.586311 | 0.586814 | +0.09% |
| `salinity_permeate` | 0.00034953 | 0.000348832 | -0.20% |
| `salinity_concentrate` | 0.0811551 | 0.0812485 | +0.12% |
| `deltaP` | -264912 | -265662 | -0.28% |
| `flux_mass_water_avg` | 0.00730551 | 0.00731178 | +0.09% |
| `Re_in` | 266.878 | 266.878 | +0.00% |
| `Re_out` | 102.487 | 102.346 | -0.14% |

**Plant (RO + pressure exchanger + pumps + costing)**

| Quantity | WaterTAP | difflow | rel. diff |
|---|---|---|---|
| `LCOW` | 0.439895 | 0.439794 | -0.02% |
| `SEC` | 2.71774 | 2.7176 | -0.01% |
| `work_P1` | 5414.27 | 5418.91 | +0.09% |
| `work_P2` | 322.12 | 322.099 | -0.01% |
| `px_hp_pressure` | 6.8784e+06 | 6.87769e+06 | -0.01% |
| `total_capital` | 28287 | 28302.7 | +0.06% |
| `total_operating` | 4496.58 | 4499.6 | +0.07% |
| `px_flow_vol_lp` | 0.000414569 | 0.000414067 | -0.12% |
| `ro_feed_salinity` | 0.035 | 0.035 | +0.00% |

### Case `brackish_like`

**RO stage alone**

| Quantity | WaterTAP | difflow | rel. diff |
|---|---|---|---|
| `recovery_mass_water` | 0.319856 | 0.31916 | -0.22% |
| `recovery_vol` | 0.318658 | 0.317965 | -0.22% |
| `salinity_permeate` | 0.000136398 | 0.000136479 | +0.06% |
| `salinity_concentrate` | 0.0218383 | 0.0218165 | -0.10% |
| `deltaP` | -252742 | -253320 | -0.23% |
| `flux_mass_water_avg` | 0.0052871 | 0.00527559 | -0.22% |
| `Re_in` | 305.858 | 305.858 | +0.00% |
| `Re_out` | 206.364 | 206.581 | +0.10% |

**Plant (RO + pressure exchanger + pumps + costing)**

| Quantity | WaterTAP | difflow | rel. diff |
|---|---|---|---|
| `LCOW` | 0.300893 | 0.301389 | +0.16% |
| `SEC` | 1.29218 | 1.29351 | +0.10% |
| `work_P1` | 1154.4 | 1151.89 | -0.22% |
| `work_P2` | 327.956 | 328.758 | +0.24% |
| `px_hp_pressure` | 2.61496e+06 | 2.61441e+06 | -0.02% |
| `total_capital` | 11881.4 | 11877.6 | -0.03% |
| `total_operating` | 1535.09 | 1534.03 | -0.07% |
| `px_flow_vol_lp` | 0.000681399 | 0.000682093 | +0.10% |
| `ro_feed_salinity` | 0.015 | 0.015 | +0.00% |

### Case `px_leak_mix`

**RO stage alone**

| Quantity | WaterTAP | difflow | rel. diff |
|---|---|---|---|
| `recovery_mass_water` | 0.42383 | 0.423827 | -0.00% |
| `recovery_vol` | 0.419961 | 0.419959 | -0.00% |
| `salinity_permeate` | 0.000242628 | 0.000242319 | -0.13% |
| `salinity_concentrate` | 0.0590632 | 0.0590631 | -0.00% |
| `deltaP` | -212208 | -212689 | -0.23% |
| `flux_mass_water_avg` | 0.00837265 | 0.00837259 | -0.00% |
| `Re_in` | 266.878 | 266.878 | +0.00% |
| `Re_out` | 149.103 | 149.103 | +0.00% |

**Plant (RO + pressure exchanger + pumps + costing)**

| Quantity | WaterTAP | difflow | rel. diff |
|---|---|---|---|
| `LCOW` | 0.416341 | 0.416371 | +0.01% |
| `SEC` | 2.50541 | 2.50563 | +0.01% |
| `work_P1` | 3353.95 | 3353.93 | -0.00% |
| `work_P2` | 379.515 | 379.844 | +0.09% |
| `px_hp_pressure` | 5.97713e+06 | 5.97668e+06 | -0.01% |
| `total_capital` | 19483.6 | 19484.8 | +0.01% |
| `total_operating` | 2946.35 | 2946.55 | +0.01% |
| `px_flow_vol_lp` | 0.000580669 | 0.000580672 | +0.00% |
| `ro_feed_salinity` | 0.0355026 | 0.0355026 | +0.00% |
