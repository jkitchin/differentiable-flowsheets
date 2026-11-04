# MVC: difflow_water vs WaterTAP 1.8.0 (IDAES 2.13.0, Pyomo 6.10.1)

## Simulation

**optimum** (T_brine = 348.15 K, pressure ratio 1.6048, preheaters 199.2 / 174.4 m^2)

| output | WaterTAP | difflow | rel. diff |
|---|---|---|---|
| evaporator_area | 789.135 | 789.135 | -2.6e-07 |
| evaporator_heat | 5.2668e+07 | 5.2668e+07 | -8.8e-09 |
| evaporator_lmtd | 22.2472 | 22.2472 | +2.5e-07 |
| T_evaporator_feed | 342.014 | 342.014 | +8.1e-09 |
| P_brine | 32448.2 | 32448.2 | +8.3e-10 |
| T_compressed_vapor | 399.404 | 399.404 | +1.1e-12 |
| P_compressed_vapor | 52073.5 | 52073.5 | +8.3e-10 |
| compressor_work | 1.77926e+06 | 1.77926e+06 | -4.0e-11 |
| T_condensate | 355.484 | 355.484 | +1.4e-08 |
| T_distillate_out | 311.293 | 311.293 | +1.2e-08 |
| T_brine_out | 305.134 | 305.134 | -9.3e-13 |
| pump_feed_work | 362.401 | 362.401 | -7.8e-16 |
| pump_brine_work | 1532.6 | 1532.6 | -4.5e-10 |
| pump_distillate_work | 1145.14 | 1145.14 | -1.1e-07 |
| product_flow_vol | 0.0223819 | 0.0223819 | -1.1e-07 |
| SEC | 22.1198 | 22.1198 | +1.1e-07 |
| LCOW | 4.51853 | 4.51853 | +1.5e-08 |
| total_capital | 1.12788e+07 | 1.12788e+07 | -1.9e-07 |
| total_operating | 1.74449e+06 | 1.74449e+06 | -3.6e-08 |
| capital pump_feed | 1382.92 | 1382.92 | -6.6e-16 |
| capital pump_brine | 5848.39 | 5848.39 | -4.5e-10 |
| capital pump_distillate | 4369.86 | 4369.86 | -1.1e-07 |
| capital hx_distillate | 604606 | 604606 | -1.9e-16 |
| capital hx_brine | 529307 | 529307 | +0.0e+00 |
| capital mixer_feed | 30508.9 | 30508.9 | +1.5e-09 |
| capital evaporator | 7.98268e+06 | 7.98267e+06 | -2.6e-07 |
| capital compressor | 2.12012e+06 | 2.12012e+06 | +0.0e+00 |

**sim_lower_PR** (T_brine = 348.15 K, pressure ratio 1.5800, preheaters 199.0 / 174.0 m^2)

| output | WaterTAP | difflow | rel. diff |
|---|---|---|---|
| evaporator_area | 965.699 | 965.698 | -3.9e-07 |
| evaporator_heat | 5.28693e+07 | 5.28693e+07 | -8.5e-09 |
| evaporator_lmtd | 18.2491 | 18.2491 | +3.8e-07 |
| T_evaporator_feed | 340.805 | 340.805 | +7.9e-09 |
| P_brine | 32448.2 | 32448.2 | -1.1e-16 |
| T_compressed_vapor | 397.566 | 397.566 | -1.4e-15 |
| P_compressed_vapor | 51268.2 | 51268.2 | -2.8e-16 |
| compressor_work | 1.7217e+06 | 1.7217e+06 | -1.7e-14 |
| T_condensate | 352.707 | 352.707 | +1.4e-08 |
| T_distillate_out | 310.659 | 310.659 | +1.1e-08 |
| T_brine_out | 305.152 | 305.152 | +1.9e-16 |
| pump_feed_work | 362.401 | 362.401 | -7.8e-16 |
| pump_brine_work | 1532.6 | 1532.6 | +0.0e+00 |
| pump_distillate_work | 1143.08 | 1143.08 | -1.1e-07 |
| product_flow_vol | 0.0223767 | 0.0223767 | -1.1e-07 |
| SEC | 21.4103 | 21.4103 | +1.1e-07 |
| LCOW | 4.80635 | 4.80635 | -5.3e-08 |
| total_capital | 1.30301e+07 | 1.30301e+07 | -2.9e-07 |
| total_operating | 1.75162e+06 | 1.75162e+06 | -6.5e-08 |
| capital pump_feed | 1382.92 | 1382.92 | -6.6e-16 |
| capital pump_brine | 5848.39 | 5848.39 | +0.0e+00 |
| capital pump_distillate | 4361.98 | 4361.98 | -1.1e-07 |
| capital hx_distillate | 603909 | 603909 | -1.9e-16 |
| capital hx_brine | 528041 | 528041 | +0.0e+00 |
| capital mixer_feed | 30488.6 | 30488.6 | +1.5e-09 |
| capital evaporator | 9.76875e+06 | 9.76874e+06 | -3.9e-07 |
| capital compressor | 2.08734e+06 | 2.08734e+06 | -5.6e-16 |

## LCOW optimum

| decision | WaterTAP (IPOPT) | difflow (SLSQP + jax.grad) | rel. diff |
|---|---|---|---|
| T_brine (K) | 348.15 | 348.15 | +1.5e-09 |
| pressure_ratio | 1.604816 | 1.604816 | -7.5e-09 |
| A_hx_distillate (m^2) | 199.2295 | 199.2295 | -2.4e-08 |
| A_hx_brine (m^2) | 174.4172 | 174.4171 | -8.3e-08 |
| LCOW (USD/m^3) | 4.518535 | 4.518535 | +1.0e-07 |

SLSQP: Optimization terminated successfully (42 iterations) from the sim_lower_PR design.
