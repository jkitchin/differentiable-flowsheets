"""Sulfur species, the wide-range Cp set and liquid sulfur (#467).

Every number below was read from the primary NIST-JANAF Thermochemical
Tables, 4th ed. (Chase, M.W., J. Phys. Chem. Ref. Data Monograph 9, 1998),
the table files at janaf.nist.gov/tables/<ID>.txt; the table ID is with each
value. The one measurement not from JANAF is the sulfur point (Stimson 1961,
see ``tc.SULFUR_NBP_MEASURED``).
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from difflow_refinery import thermochemistry as tc

LN10 = math.log(10.0)

# JANAF 298.15 K: Hf (kJ/mol), S0 (J/mol/K), table ID.
JANAF_298 = {
    "sulfur_s2": (128.600, 228.165, "S-012"),
    "sulfur_s3": (141.461, 269.517, "S-016"),
    "sulfur_s4": (145.771, 310.646, "S-017"),
    "sulfur_s5": (109.370, 308.638, "S-018"),
    "sulfur_s6": (101.922, 354.076, "S-019"),
    "sulfur_s7": (113.679, 407.673, "S-020"),
    "sulfur_s8": (100.416, 430.311, "S-021"),
    "carbonyl_sulfide": (-138.407, 231.581, "C-094"),
    "carbon_disulfide": (116.943, 237.977, "C-099"),
}

# JANAF tabulated Cp (J/mol/K) at 298.15, 600, 1000, 1500, 1800 K.
JANAF_CP = {
    "hydrogen_sulfide": (34.192, 38.936, 45.786, 51.476, 53.589),   # H-080
    "sulfur_s2": (32.490, 35.781, 37.277, 38.818, 39.661),          # S-012
    "sulfur_s8": (156.043, 175.248, 180.064, 181.636, 182.025),     # S-021
    "carbonyl_sulfide": (41.497, 51.250, 56.990, 60.208, 61.266),   # C-094
    "carbon_disulfide": (45.664, 54.616, 59.283, 61.588, 62.340),   # C-099
}
CP_T = (298.15, 600.0, 1000.0, 1500.0, 1800.0)

# JANAF log10 Kf at 600, 800, 1000, 1200, 1500 K. A reaction's
# ln K = ln10 * sum(nu * log Kf): every table at one T shares one element
# reference (S(l) below 882.117 K, S2(g) above), which a balanced reaction
# cancels.
LOGKF_T = (600.0, 800.0, 1000.0, 1200.0, 1500.0)
LOGKF = {
    "H2S": (3.691, 2.984, 2.141, 1.355, 0.566),       # H-080
    "SO2": (26.144, 19.482, 15.081, 11.931, 8.784),   # O-034
    "S2": (-3.092, -0.655, 0.0, 0.0, 0.0),            # S-012
    "H2O": (18.631, 13.287, 10.060, 7.897, 5.724),    # H-064
    "CO2": (34.404, 25.829, 20.679, 17.242, 13.800),  # C-095
    "COS": (16.600, 13.416, 11.102, 9.340, 7.575),    # C-094
    "CS2": (-1.824, 0.383, 0.904, 0.815, 0.726),      # C-099
    "S8": (-1.220, -0.220, -2.763, -6.151, -9.477),   # S-021
    "S6": (-1.640, -0.347, -1.940, -4.279, -6.580),   # S-019
}

REACTIONS = {
    "claus_S2": {"H2S": -2, "SO2": -1, "S2": 1.5, "H2O": 2},
    "S8_to_S2": {"S8": -1, "S2": 4},
    "S6_to_S2": {"S6": -1, "S2": 3},
    "COS_hydrolysis": {"COS": -1, "H2O": -1, "CO2": 1, "H2S": 1},
    "CS2_hydrolysis": {"CS2": -1, "H2O": -2, "CO2": 1, "H2S": 2},
}


class TestRows:
    @pytest.mark.parametrize("key", list(JANAF_298))
    def test_janaf_pins(self, key):
        Hf, S0, table = JANAF_298[key]
        s = tc.species(key)
        assert s.Hf == pytest.approx(Hf * 1e3, abs=1e-6)
        assert s.S0 == pytest.approx(S0, abs=1e-9)
        assert s.Hf_source == s.S0_source == "JANAF_TABLE"
        assert table in s.note

    def test_aliases(self):
        for n in range(2, 9):
            assert tc.resolve(f"S{n}") == f"sulfur_s{n}"
        assert tc.resolve("COS") == "carbonyl_sulfide"
        assert tc.resolve("CS2") == "carbon_disulfide"
        assert tc.species("S8").MW == pytest.approx(8 * 32.06)

    @pytest.mark.parametrize("key", [k for k in JANAF_CP if k != "hydrogen_sulfide"])
    def test_table_cp_against_janaf(self, key):
        s = tc.species(key)
        for T, c in zip(CP_T, JANAF_CP[key]):
            if T <= s.cp_range[1]:
                assert float(tc.cp(key, T)) == pytest.approx(c, rel=s.cp_max_dev + 1e-4)


class TestWideCp:
    def test_every_entry_is_a_table_species(self):
        for k in tc.WIDE_CP:
            assert k in tc.TABLE

    @pytest.mark.parametrize("key", list(JANAF_CP))
    def test_against_janaf_to_1800(self, key):
        C, dev, _ = tc.WIDE_CP[key]
        for T, c in zip(CP_T, JANAF_CP[key]):
            assert tc._cp(np.asarray(C), T) == pytest.approx(c, rel=dev + 1e-4)

    def test_fits_are_close_and_monotone(self):
        T = np.linspace(*tc.WIDE_CP_RANGE, 400)
        for k, (C, dev, _) in tc.WIDE_CP.items():
            assert dev < 0.025, k
            c = np.asarray(tc._cp(np.asarray(C), T))
            assert np.all(np.diff(c) > -1e-9), k

    def test_wide_beats_the_1000K_fit_where_it_extrapolates(self):
        """The reason for the wide set: H2S's 1000 K cubic is 5.6 % low at
        1500 K (JANAF H-080: 51.476)."""
        narrow = float(tc.cp("H2S", 1500.0))
        wide = float(tc.IdealGasSet(["H2S"], cp_basis="wide").cp(1500.0)[0])
        assert abs(narrow / 51.476 - 1) > 0.05
        assert abs(wide / 51.476 - 1) < 0.006

    def test_basis(self):
        g = tc.IdealGasSet(["H2", "CH4"], cp_basis="wide")
        assert np.allclose(g.HF, tc.IdealGasSet(["H2", "CH4"]).HF)
        with pytest.raises(KeyError):
            tc.IdealGasSet(["benzene"], cp_basis="wide")
        with pytest.raises(ValueError):
            tc.IdealGasSet(["H2"], cp_basis="hot")


class TestEquilibria:
    @pytest.mark.parametrize("basis", ["table", "wide"])
    @pytest.mark.parametrize("name", list(REACTIONS))
    def test_ln_K_against_janaf(self, name, basis):
        """ln K from the table's Hf, S0 and Cp cubic against JANAF's own
        tabulated log Kf. The table's H2S and SO2 are CODATA (0.10 and 0.03
        kJ/mol from JANAF) and its cubics deviate up to ~1 %, so the standard
        Gibbs energy of reaction agrees to < 1 kJ/mol, not exactly."""
        nu = REACTIONS[name]
        gas = tc.IdealGasSet(list(nu), cp_basis=basis)
        v = np.array(list(nu.values()), dtype=float)
        for i, T in enumerate(LOGKF_T):
            if basis == "table" and T > min(tc.species(k).cp_range[1] for k in nu):
                continue
            ref = LN10 * sum(n * LOGKF[k][i] for k, n in nu.items())
            got = float(gas.ln_K(v, T))
            assert abs(got - ref) * tc.R * T < 1000.0, (name, T, got, ref)

    def test_claus_equilibrium_falls_with_temperature(self):
        """2 H2S + SO2 -> 3/n Sn + 2 H2O to S8 is exothermic: K falls with T,
        which is why the catalytic stages run cool."""
        nu = {"H2S": -2, "SO2": -1, "S8": 3 / 8, "H2O": 2}
        assert float(tc.ln_K(nu, 500.0)) > float(tc.ln_K(nu, 600.0)) > 0


# JANAF S-004 rows the model leaves out of its nodes (H - H(298) kJ/mol, S).
SL_OFF_NODE = {300.0: (0.042, 36.965), 882.117: (19.456, 72.624)}
SL_GEF = {400.0: 37.829, 432.02: 38.442, 500.0: 40.106, 1000.0: 53.407, 1500.0: 63.420}


class TestLiquidSulfur:
    def test_nodes_are_the_table(self):
        H = np.asarray(tc.liquid_sulfur_enthalpy(tc._SL_T)) - tc._SL_HF
        S = np.asarray(tc.liquid_sulfur_entropy(tc._SL_T))
        assert np.allclose(H, tc._SL_H, atol=1e-6)
        assert np.allclose(S, tc._SL_S, atol=1e-9)

    def test_off_node_rows(self):
        for T, (H, S) in SL_OFF_NODE.items():
            assert float(tc.liquid_sulfur_enthalpy(T)) - 1854.0 == pytest.approx(H * 1e3, abs=3.0)
            assert float(tc.liquid_sulfur_entropy(T)) == pytest.approx(S, abs=4e-3)

    def test_gibbs_against_janaf_gef(self):
        """JANAF gef = -[G - H(298.15)]/T, so G = 1854 - T gef on the table's basis."""
        for T, gef in SL_GEF.items():
            # JANAF's own rounding: S - [H - H(298)]/T differs from gef by up to 0.0015
            assert float(tc.liquid_sulfur_gibbs(T)) == pytest.approx(1854.0 - T * gef,
                                                                     abs=0.0015 * T + 0.1)

    def test_consistency(self):
        T = jnp.linspace(300.0, 1490.0, 47)
        dG = jax.vmap(jax.grad(tc.liquid_sulfur_gibbs))(T)
        assert np.allclose(dG, -np.asarray(tc.liquid_sulfur_entropy(T)), atol=1e-8)
        check_grads(tc.liquid_sulfur_gibbs, (jnp.array(615.0),), order=2, modes=["fwd", "rev"])

    def test_S2_reference_changeover(self):
        """JANAF changes the sulfur reference from S(l) to S2(g) at 882.117 K,
        where 1/2 S2(g, 1 bar) and S(l) have equal Gibbs energy."""
        def f(T):
            return float(tc.gibbs("S2", T) - 2 * tc.liquid_sulfur_gibbs(T))
        lo, hi = 860.0, 900.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
        assert 0.5 * (lo + hi) == pytest.approx(882.117, abs=1.0)

    @pytest.mark.parametrize("i,T", [(0, 600.0), (1, 800.0)])
    def test_vapour_pressures_against_janaf(self, i, T):
        """Below 882 K the JANAF reference is S(l): p_Sn/bar = 10^logKf(Sn)
        straight from the gas tables. The model's Sn(g) Gibbs energy comes
        from a Cp cubic integrated twice (1-2 % Cp error on the large rings),
        so it is held to 0.4 kJ/mol of Sn, i.e. RT|ln(p/p_JANAF)| < 400."""
        logkf = {2: (-3.092, -0.655), 6: (-1.640, -0.347), 8: (-1.220, -0.220)}
        for basis in ("table", "wide"):
            p = np.asarray(tc.sulfur_vapor_pressures(T, cp_basis=basis)) / tc.P_STD
            for n, lk in logkf.items():
                assert tc.R * T * abs(math.log(p[n - 2]) - LN10 * lk[i]) < 400.0, (basis, n)

    def test_boiling_point_discrepancy_is_documented(self):
        """The JANAF sulfur data put the 1 atm point ~27 K above the measured
        sulfur point (444.6 degC, Stimson 1961). Pinned so nobody mistakes the
        JANAF vapour for a vapour-pressure correlation."""
        def P(T):
            return float(jnp.sum(tc.sulfur_vapor_pressures(T))) - 101325.0
        lo, hi = 650.0, 800.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if P(mid) > 0 else (mid, hi)
        nbp = 0.5 * (lo + hi)
        assert tc.SULFUR_NBP_MEASURED == pytest.approx(717.75)
        assert 735.0 < nbp < 755.0
