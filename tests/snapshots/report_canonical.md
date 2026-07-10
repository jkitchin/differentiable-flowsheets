# Flowsheet Report

## Topology

- Units: split
- Species order: methane, ethane

**Connections**

| stream | source | target |
| --- | --- | --- |
| feed | (feed) | split |

## Unit Operations

### split — Splitter (core)

Simple stream splitter (no phase equilibrium).

**Inlets:** feed  
**Outlets:** top, bot

**Governing equations**

- $$F_{i,k}^{\mathrm{out}} = \phi_k\, F_{i}^{\mathrm{in}}\qquad \sum_k \phi_k = 1$$
- $$T^{\mathrm{out}} = T^{\mathrm{in}},\quad P^{\mathrm{out}} = P^{\mathrm{in}}$$

**Assumptions**

- No phase change; each outlet has the same composition as the inlet.
- Isothermal, isobaric split.

**References**

- Seider, Seader, Lewin, Widagdo. Product & Process Design Principles, 4e.

## Species and Thermophysical Data

| species | MW (g/mol) | Tc (K) | Pc (Pa) | ω | Hf (J/mol) | source |
| --- | --- | --- | --- | --- | --- | --- |
| methane | 16.04 | 190.6 | 4.599e+06 | 0.011 | -7.487e+04 | NIST Chemistry WebBook; Perry's 9e; Yaws; DIPPR 801 |
| ethane | 30.07 | 305.3 | 4.872e+06 | 0.099 | -8.4e+04 | NIST Chemistry WebBook; Perry's 9e; Yaws; DIPPR 801 |

## Feed Streams

### feed

- T = 300 K
- P = 1.013e+05 Pa

| species | F (mol/s) |
| --- | --- |
| methane | 1 |
| ethane | 0.5 |

## Solved Streams

### top

- T = 300 K
- P = 1.013e+05 Pa

| species | F (mol/s) |
| --- | --- |
| methane | 0.6 |
| ethane | 0.3 |

### bot

- T = 300 K
- P = 1.013e+05 Pa

| species | F (mol/s) |
| --- | --- |
| methane | 0.4 |
| ethane | 0.2 |

## Mass Balance Closure

| species | feed total | outlet total | residual |
| --- | --- | --- | --- |
| methane | 1 | 1 | +0 |
| ethane | 0.5 | 0.5 | +0 |

## Recycle Convergence

- Method: direct
- Tear streams: (none)
- Iterations: 0
- Final residual: 0
- Tolerance: 1e-08
- Converged: yes
