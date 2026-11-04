# Activated sludge (ASM1, BSM1 layout): difflow_water vs references

## WaterTAP 1.8.0 BSM1 example (split clarifier)

Largest relative difference over all 7 streams x 13 states: **1.3e-07**

| stream | state | WaterTAP | difflow | rel. diff |
|---|---|---|---|---|
| R5 | S_I | 30 | 30 | +7.1e-16 |
| R5 | S_S | 0.893035 | 0.893035 | +2.2e-10 |
| R5 | X_I | 1148.92 | 1148.92 | -4.7e-15 |
| R5 | X_S | 49.5136 | 49.5136 | +4.6e-10 |
| R5 | X_BH | 2558.93 | 2558.93 | -4.1e-12 |
| R5 | X_BA | 149.283 | 149.283 | -1.1e-10 |
| R5 | X_P | 452.04 | 452.04 | -7.6e-12 |
| R5 | S_O | 0.449 | 0.449 | +0.0e+00 |
| R5 | S_NO | 10.1476 | 10.1476 | +1.8e-13 |
| R5 | S_NH | 1.85966 | 1.85966 | +2.3e-09 |
| R5 | S_ND | 0.688134 | 0.688134 | -4.6e-10 |
| R5 | X_ND | 3.54137 | 3.54137 | -1.5e-09 |
| R5 | S_ALK | 4.15372 | 4.15372 | -2.4e-09 |
| effluent | S_I | 30 | 30 | +7.1e-16 |
| effluent | S_S | 0.893035 | 0.893035 | +2.2e-10 |
| effluent | X_I | 4.38859 | 4.38859 | -4.5e-15 |
| effluent | X_S | 0.18913 | 0.18913 | -7.1e-09 |
| effluent | X_BH | 9.77449 | 9.77449 | -4.1e-12 |
| effluent | X_BA | 0.570225 | 0.570225 | -1.4e-09 |
| effluent | X_P | 1.72668 | 1.72668 | -7.6e-12 |
| effluent | S_O | 0.449 | 0.449 | -2.2e-09 |
| effluent | S_NO | 10.1476 | 10.1476 | +1.8e-13 |
| effluent | S_NH | 1.85966 | 1.85966 | +2.3e-09 |
| effluent | S_ND | 0.688134 | 0.688134 | -4.6e-10 |
| effluent | X_ND | 0.0135272 | 0.0135272 | -1.3e-07 |
| effluent | S_ALK | 4.15372 | 4.15372 | -2.4e-09 |
| wastage | S_I | 30 | 30 | +9.5e-16 |
| wastage | S_S | 0.893035 | 0.893035 | +2.2e-10 |
| wastage | X_I | 2246.63 | 2246.63 | -4.7e-15 |
| wastage | X_S | 96.8203 | 96.8203 | +4.6e-10 |
| wastage | X_BH | 5003.81 | 5003.81 | -4.1e-12 |
| wastage | X_BA | 291.913 | 291.913 | -1.1e-10 |
| wastage | X_P | 883.932 | 883.932 | -7.6e-12 |
| wastage | S_O | 0.449 | 0.449 | -2.2e-09 |
| wastage | S_NO | 10.1476 | 10.1476 | +1.8e-13 |
| wastage | S_NH | 1.85966 | 1.85966 | +2.3e-09 |
| wastage | S_ND | 0.688134 | 0.688134 | -4.6e-10 |
| wastage | X_ND | 6.92491 | 6.92491 | -1.5e-09 |
| wastage | S_ALK | 4.15372 | 4.15372 | -2.4e-09 |

## BSM1 report open-loop steady state (Takacs settler)

Largest relative difference over Tables 6-7 (70 reactor values, 10 layers): **3.2e-03** (the tables print 3-4 digits); sludge age 9.14 d

| reactor | state | BSM1 report | difflow | rel. diff |
|---|---|---|---|---|
| 5 | S_I | 30 | 30 | +3.6e-16 |
| 5 | S_S | 0.889 | 0.8895 | +5.5e-04 |
| 5 | X_I | 1149 | 1149 | +1.1e-04 |
| 5 | X_S | 49.3 | 49.31 | +1.1e-04 |
| 5 | X_BH | 2559 | 2559 | +1.3e-04 |
| 5 | X_BA | 150 | 149.8 | -1.4e-03 |
| 5 | X_P | 452 | 452.2 | +4.7e-04 |
| 5 | S_O | 0.491 | 0.4909 | -1.2e-04 |
| 5 | S_NO | 10.4 | 10.42 | +1.5e-03 |
| 5 | S_NH | 1.73 | 1.733 | +1.9e-03 |
| 5 | S_ND | 0.688 | 0.6883 | +4.1e-04 |
| 5 | X_ND | 3.53 | 3.527 | -8.0e-04 |
| 5 | S_ALK | 4.13 | 4.126 | -1.1e-03 |
| 5 | TSS | 3270 | 3270 | -5.0e-05 |

| settler layer (bottom = 1) | BSM1 report | difflow | rel. diff |
|---|---|---|---|
| 1 | 6394 | 6394 | -2.4e-06 |
| 2 | 356 | 356.1 | +2.1e-04 |
| 3 | 356 | 356.1 | +2.1e-04 |
| 4 | 356 | 356.1 | +2.1e-04 |
| 5 | 356 | 356.1 | +2.1e-04 |
| 6 | 356 | 356.1 | +2.1e-04 |
| 7 | 69 | 68.98 | -3.2e-04 |
| 8 | 29.5 | 29.54 | +1.4e-03 |
| 9 | 18.1 | 18.11 | +7.3e-04 |
| 10 | 12.5 | 12.5 | -2.4e-04 |
