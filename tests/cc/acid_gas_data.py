"""Measured acid-gas solubility in aqueous MDEA and DEA (#471).

Each set records its source table. Pressures are acid-gas partial
pressures in kPa; loadings are mol acid gas / mol amine; ``w`` is the amine
mass fraction of the acid-gas-free solution.

* AUSTGEN_CO2_MDEA: Austgen (1989), PhD dissertation, UT Austin, Table 5.3
  (his own measurements; also Austgen et al. 1991, Ind. Eng. Chem. Res.
  30:543), OSTI 6346766, 40 C. The table gives 2.0 and 4.28 kmol/m3 MDEA; the mass
  fractions here use solution densities of 1.010 and 1.040 g/cm3
  (ILLUSTRATIVE, typical of 20-50 wt% MDEA at 40 C; a 1 % density error
  moves the molality by about 1 %). Replicate loadings are averaged.
* HUTTENHUIS_CO2_MDEA: Huttenhuis, Agrawal, Hogendoorn & Versteeg (2007),
  J. Pet. Sci. Eng. 55:122-134, doi:10.1016/j.petrol.2006.04.018, Table 1 (35 wt% MDEA, methane make-up gas,
  6.9 bar total). The 50 wt% table (Table 2) is not monotonic as
  extracted and is left out.
* HUTTENHUIS_H2S_MDEA: the same paper, Tables 3 (35 wt%) and 4 (50 wt%),
  6.9 bar total (the lowest make-up pressure, nearest an ideal gas).
* BARREAU_CO2_DEA, BARREAU_H2S_DEA (and LITERATURE_*_DEA, held out): Barreau, Blanchon le Bouhelec,
  Habchi Tounsi, Mougin & Lecomte (2006), Oil Gas Sci. Technol. 61:345-361, doi:10.2516/ogst:2006038a,
  Tables 1-4 (open access, CC-BY; reported in MPa). Table 1 is 2 mol/L
  DEA, converted at 1.005 g/cm3 (ILLUSTRATIVE, a typical 20 wt% DEA
  density). Points above 1 MPa are left out (gas non-ideality), and the
  first point of Table 3 (0.4 kPa at 0.100, against 21 kPa at 0.224) is
  left out as inconsistent with its neighbours.
"""

#: (T K, w, alpha, P kPa)
AUSTGEN_CO2_MDEA = [
    (313.15, 2.0 * 119.163 / 1010.0, a, p) for p, a in [
        (0.0056, 0.006015), (0.0151, 0.01175), (0.0452, 0.02155), (0.177, 0.0457),
        (0.419, 0.0740), (0.887, 0.113), (6.95, 0.3565), (92.8, 0.840)]
] + [
    (313.15, 4.28 * 119.163 / 1040.0, a, p) for p, a in [
        (0.0102, 0.003225), (0.118, 0.01405), (0.585, 0.03635), (3.04, 0.105),
        (93.6, 0.662)]
]

HUTTENHUIS_CO2_MDEA = [
    (298.0, 0.35, 0.048, 0.169), (298.0, 0.35, 0.143, 1.001), (298.0, 0.35, 0.270, 3.037),
    (283.0, 0.35, 0.048, 0.054), (283.0, 0.35, 0.143, 0.314), (283.0, 0.35, 0.276, 1.000),
    (283.0, 0.35, 0.325, 1.324),
]

HUTTENHUIS_H2S_MDEA = [
    (283.0, 0.35, 0.052, 0.141), (283.0, 0.35, 0.171, 1.057),
    (298.0, 0.35, 0.042, 0.156), (298.0, 0.35, 0.165, 1.495),
    (283.0, 0.50, 0.081, 0.486), (283.0, 0.50, 0.125, 1.137),
    (298.0, 0.50, 0.028, 0.153), (298.0, 0.50, 0.062, 0.609),
    (298.0, 0.50, 0.083, 1.024), (298.0, 0.50, 0.105, 1.484),
]

_W2M = 2.0 * 105.136 / 1005.0
BARREAU_CO2_DEA = [
    (323.15, _W2M, a, p) for a, p in [
        (0.45, 14.0), (0.55, 36.0), (0.66, 101.0), (0.75, 226.0), (0.81, 382.0),
        (0.87, 626.0)]
] + [
    (338.50, 0.25, a, p) for a, p in [
        (0.098, 1.0), (0.199, 4.0), (0.288, 8.0), (0.378, 18.0), (0.464, 38.0),
        (0.565, 103.0), (0.659, 247.0), (0.738, 493.0), (0.799, 803.0)]
] + [
    (366.90, 0.25, a, p) for a, p in [
        (0.224, 21.0), (0.283, 40.0), (0.400, 126.0), (0.497, 289.0), (0.581, 556.0),
        (0.650, 923.0)]
]

BARREAU_H2S_DEA = [
    (339.0, 0.25, a, p) for a, p in [
        (0.115, 3.4), (0.169, 5.5), (0.221, 7.7), (0.285, 11.5), (0.333, 14.8),
        (0.428, 23.4), (0.482, 29.9), (0.525, 36.1), (0.548, 40.1), (0.626, 56.7),
        (0.677, 71.3), (0.717, 85.7), (0.752, 101.9), (0.789, 122.7), (0.823, 147.1),
        (0.893, 222.3), (0.932, 282.9), (0.972, 365.0)]
]

# Held out of the fit: the smoothed literature values Barreau et al. (2006)
# tabulate beside their own (Table 1: Lee, Otto & Mather 1972; Tables 2-4:
# Lawson & Garst 1976), at the same T, w and loading; below 1 MPa.
LITERATURE_CO2_DEA = [
    (323.15, _W2M, a, p) for a, p in [
        (0.45, 7.0), (0.55, 28.0), (0.66, 89.0), (0.75, 212.0), (0.81, 376.0),
        (0.87, 617.0)]
] + [
    (338.50, 0.25, a, p) for a, p in [
        (0.464, 32.0), (0.565, 91.0), (0.659, 216.0), (0.738, 415.0), (0.799, 658.0)]
] + [
    (366.90, 0.25, a, p) for a, p in [
        (0.400, 115.0), (0.497, 265.0), (0.581, 488.0), (0.650, 767.0)]
]

LITERATURE_H2S_DEA = [
    (339.0, 0.25, a, p) for a, p in [
        (0.169, 3.9), (0.221, 6.6), (0.285, 10.7), (0.333, 14.5), (0.428, 24.1),
        (0.482, 31.1), (0.525, 37.6), (0.548, 41.5), (0.626, 57.6), (0.677, 71.5),
        (0.717, 84.9), (0.752, 99.8), (0.789, 119.1), (0.823, 142.1), (0.893, 217.1),
        (0.932, 286.2), (0.972, 390.6)]
]
