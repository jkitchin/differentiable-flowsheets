"""Tests for thermodynamic property database."""

import pytest
import jax.numpy as jnp

from difflow.database import (
    get_species_data,
    get_critical_props,
    get_species_info,
    list_species,
    get_alkanes,
    get_btex,
    get_common_solvents,
    resolve_alias,
)
from difflow import IdealThermo, PengRobinson


class TestGetCriticalProps:
    """Tests for critical property retrieval."""

    def test_get_methane(self):
        """Test retrieving methane critical properties."""
        props = get_critical_props("methane")
        assert props.Tc == pytest.approx(190.6, rel=0.01)
        assert props.Pc == pytest.approx(4.599e6, rel=0.01)
        assert props.omega == pytest.approx(0.011, rel=0.1)
        assert props.MW == pytest.approx(16.04, rel=0.01)

    def test_get_water(self):
        """Test retrieving water critical properties."""
        props = get_critical_props("water")
        assert props.Tc == pytest.approx(647.1, rel=0.01)
        assert props.Pc == pytest.approx(22.064e6, rel=0.01)

    def test_case_insensitive(self):
        """Test that lookup is case-insensitive."""
        props1 = get_critical_props("Methane")
        props2 = get_critical_props("METHANE")
        props3 = get_critical_props("methane")
        assert props1.Tc == props2.Tc == props3.Tc

    def test_species_not_found(self):
        """Test error for unknown species."""
        with pytest.raises(KeyError) as exc_info:
            get_critical_props("unobtainium")
        assert "not in database" in str(exc_info.value)


class TestGetSpeciesData:
    """Tests for SpeciesData retrieval."""

    def test_get_methanol(self):
        """Test retrieving methanol species data."""
        data = get_species_data("methanol")
        assert data.MW == pytest.approx(32.04, rel=0.01)
        assert data.name == "methanol"

    def test_get_water(self):
        """Test retrieving water species data."""
        data = get_species_data("water")
        assert data.MW == pytest.approx(18.015, rel=0.01)
        assert data.Hf == pytest.approx(-285830.0, rel=0.01)  # Liquid-phase value

    def test_species_not_found(self):
        """Test error for unknown species."""
        with pytest.raises(KeyError):
            get_species_data("unobtainium")


class TestAliases:
    """Tests for alias resolution."""

    def test_co2_alias(self):
        """Test CO2 -> carbon_dioxide alias."""
        assert resolve_alias("CO2") == "carbon_dioxide"
        props = get_critical_props("CO2")
        assert props.name == "carbon_dioxide"

    def test_butane_alias(self):
        """Test butane -> n_butane alias."""
        assert resolve_alias("butane") == "n_butane"
        props = get_critical_props("butane")
        assert props.Tc == pytest.approx(425.1, rel=0.01)

    def test_isopropanol_alias(self):
        """Test isopropanol -> 2_propanol alias."""
        data = get_species_data("isopropanol")
        assert data.name == "2_propanol"

    def test_meoh_alias(self):
        """Test MeOH -> methanol alias."""
        data = get_species_data("MeOH")
        assert data.name == "methanol"


class TestListSpecies:
    """Tests for species listing."""

    def test_list_returns_list(self):
        """Test that list_species returns a list."""
        species = list_species()
        assert isinstance(species, list)
        assert len(species) > 30  # Should have many species

    def test_list_is_sorted(self):
        """Test that species list is sorted."""
        species = list_species()
        assert species == sorted(species)

    def test_common_species_present(self):
        """Test that common species are in the list."""
        species = list_species()
        for name in ["water", "methane", "benzene", "ethanol"]:
            assert name in species


class TestGetSpeciesInfo:
    """Tests for complete species info retrieval."""

    def test_info_with_both_datasets(self):
        """Test species present in both databases."""
        info = get_species_info("methane")
        assert "critical" in info
        assert "ideal_thermo" in info
        assert info["critical"]["Tc"] == pytest.approx(190.6, rel=0.01)

    def test_info_critical_only(self):
        """Test species with only critical properties."""
        info = get_species_info("helium")
        assert "critical" in info
        assert "ideal_thermo" not in info

    def test_info_not_found(self):
        """Test error for unknown species."""
        with pytest.raises(KeyError):
            get_species_info("unobtainium")


class TestConvenienceFunctions:
    """Tests for group retrieval functions."""

    def test_get_alkanes(self):
        """Test retrieving alkane series."""
        alkanes = get_alkanes(4)
        assert len(alkanes) == 4
        assert "methane" in alkanes
        assert "n_butane" in alkanes

    def test_get_alkanes_limit(self):
        """Test alkane limit parameter."""
        alkanes_4 = get_alkanes(4)
        alkanes_8 = get_alkanes(8)
        assert len(alkanes_4) < len(alkanes_8)

    def test_get_btex(self):
        """Test retrieving BTEX aromatics."""
        btex = get_btex()
        assert "benzene" in btex
        assert "toluene" in btex
        assert "ethylbenzene" in btex
        assert "m_xylene" in btex

    def test_get_common_solvents(self):
        """Test retrieving common solvents."""
        solvents = get_common_solvents()
        assert "water" in solvents
        assert "methanol" in solvents
        assert "acetone" in solvents


class TestIntegration:
    """Tests for integration with difflow models."""

    def test_ideal_thermo_integration(self):
        """Test using database with IdealThermo."""
        thermo = IdealThermo({
            "methanol": get_species_data("methanol"),
            "water": get_species_data("water"),
        })
        assert thermo.n_species == 2

        # Test property calculation
        Cp = thermo.Cp("methanol", 300.0)
        assert float(Cp) > 0

    def test_peng_robinson_integration(self):
        """Test using database with PengRobinson EOS."""
        eos = PengRobinson({
            "methane": get_critical_props("methane"),
            "ethane": get_critical_props("ethane"),
        })
        assert eos.n_species == 2

        # Test property calculation
        T = jnp.array(250.0)
        P = jnp.array(1e6)
        y = jnp.array([0.5, 0.5])
        Z = eos.solve_Z(T, P, y, phase="vapor")
        assert float(Z) > 0

    def test_alkanes_with_eos(self):
        """Test using get_alkanes directly with EOS."""
        eos = PengRobinson(get_alkanes(4))
        assert eos.n_species == 4

    def test_common_solvents_with_ideal_thermo(self):
        """Test using get_common_solvents with IdealThermo."""
        thermo = IdealThermo(get_common_solvents())
        assert thermo.n_species >= 5


class TestDataConsistency:
    """Tests for data quality and consistency."""

    def test_critical_temperature_order(self):
        """Test that Tc increases with molecular weight for alkanes."""
        alkanes = ["methane", "ethane", "propane", "n_butane", "n_pentane"]
        Tc_values = [get_critical_props(a).Tc for a in alkanes]

        for i in range(len(Tc_values) - 1):
            assert Tc_values[i] < Tc_values[i + 1], \
                f"Tc should increase: {alkanes[i]} -> {alkanes[i+1]}"

    def test_acentric_factor_range(self):
        """Test that acentric factors are in reasonable range."""
        for name in list_species():
            try:
                props = get_critical_props(name)
                assert -0.5 < props.omega < 2.0, \
                    f"Unusual omega for {name}: {props.omega}"
            except KeyError:
                pass  # Species might only have ideal thermo data

    def test_molecular_weight_positive(self):
        """Test that all molecular weights are positive."""
        for name in list_species():
            try:
                data = get_species_data(name)
                assert data.MW > 0, f"MW should be positive for {name}"
            except KeyError:
                pass  # Species might only have critical data


class TestEthylAcetate:
    """Ethyl acetate's entry against the measurements it was taken from.

    Each number was entered by hand from the NIST WebBook, so each is
    checked against a value it should reproduce rather than against itself.
    """

    @pytest.fixture(scope="class")
    def thermo(self):
        return IdealThermo({"ethyl_acetate": get_species_data("ethyl_acetate")})

    def test_normal_boiling_point(self, thermo):
        # Tb = 350.2 K (NIST average of 58 values): Psat there is 1 atm.
        assert float(thermo.Psat("ethyl_acetate", 350.2)) == pytest.approx(101325, rel=0.005)

    def test_vapor_pressure_at_25C(self, thermo):
        # About 12.6 kPa at 298.15 K.
        assert float(thermo.Psat("ethyl_acetate", 298.15)) == pytest.approx(12600, rel=0.02)

    def test_heat_of_vaporization(self, thermo):
        # 31.94 kJ/mol at 350.3 K (Majer & Svoboda); 35 +/- 2 at 298 K.
        assert float(thermo.Hvap("ethyl_acetate", 350.3)) == pytest.approx(31940, rel=0.002)
        assert float(thermo.Hvap("ethyl_acetate", 298.15)) == pytest.approx(35000, abs=2000)

    def test_critical_properties(self):
        props = get_critical_props("ethyl_acetate")
        assert props.Tc == pytest.approx(523.2)
        assert props.Pc == pytest.approx(38.82e5)
        assert props.MW == pytest.approx(88.106)

    def test_alias(self):
        assert resolve_alias("etoac") == "ethyl_acetate"


class TestHeatOfVaporization:
    """The database's Watson coefficients against measured latent heats.

    The table once held each species' Hvap at its boiling point in the
    place of Watson's prefactor, which put every Hvap 30-40% low at every
    temperature (water: 29.3 kJ/mol at 100 C). Nothing failed, because
    nothing compared the correlation with the number it was built from.
    """

    @pytest.mark.parametrize("name", list_species())
    def test_every_entry_reproduces_its_own_measurement(self, name):
        from difflow.database import _IDEAL_THERMO_DATA
        if name not in _IDEAL_THERMO_DATA:
            pytest.skip("no ideal-thermo record")
        d = _IDEAL_THERMO_DATA[name]
        if "Hvap_T" not in d:
            # Watson's A stored directly; checked against the CRC dHvap(298)
            # in test_database_refinery_species.py instead.
            pytest.skip("no anchor temperature")
        thermo = IdealThermo({name: get_species_data(name)})
        assert float(thermo.Hvap(name, d["Hvap_T"])) == pytest.approx(d["Hvap"][0], rel=1e-9)

    @pytest.mark.parametrize("name, T, nist", [
        # Independent of the anchor: NIST WebBook fluid tables (reference
        # equations of state), H_vapor - H_liquid on the saturation line.
        ("water", 373.124, 40650.0),
        ("water", 298.15, 43973.0),
        ("ammonia", 298.15, 19855.0),
        ("carbon_dioxide", 273.15, 10161.0),
        ("carbon_dioxide", 250.0, 12733.0),
        ("methane", 111.67, 8195.0),
    ])
    def test_against_reference_equations_of_state(self, name, T, nist):
        thermo = IdealThermo({name: get_species_data(name)})
        assert float(thermo.Hvap(name, T)) == pytest.approx(nist, rel=0.03)

    def test_watson_coeffs_refuses_a_reference_past_critical(self):
        from difflow.database import watson_coeffs
        with pytest.raises(ValueError):
            watson_coeffs(10000.0, 0.38, 300.0, 316.0)


class TestC4Olefins:
    """cis-2-butene, trans-2-butene and isobutylene (FCC LPG olefins).

    Each check is against a number published independently of the one
    stored, so a transcription or fitting slip shows up here.
    """

    # TRC (1997) ideal-gas Cp tables, as the NIST WebBook gives them (J/mol/K)
    TRC_CP = {
        "cis_2_butene": {200: 61.73, 298.15: 80.15, 400: 102.73, 500: 123.64,
                         600: 141.91, 700: 157.66, 800: 171.27},
        "trans_2_butene": {200: 69.41, 298.15: 87.67, 400: 108.53, 500: 128.08,
                           600: 145.43, 700: 160.56, 800: 173.75},
        "isobutylene": {200: 67.34, 298.15: 88.09, 400: 109.79, 500: 129.35,
                        600: 146.48, 700: 161.35, 800: 174.30},
    }
    # NIST WebBook normal boiling points (averages of 7, 10 and 25 values)
    TB = {"cis_2_butene": 276.84, "trans_2_butene": 274.2, "isobutylene": 266.7}

    @pytest.mark.parametrize("name", sorted(TRC_CP))
    def test_ideal_gas_cp_matches_the_trc_table(self, name):
        a, b, c, d = get_species_data(name).Cp_coeffs
        for T, cp in self.TRC_CP[name].items():
            assert a + b * T + c * T**2 + d * T**3 == pytest.approx(cp, rel=0.012)

    @pytest.mark.parametrize("name", sorted(TB))
    def test_antoine_boils_at_the_normal_boiling_point(self, name):
        A, B, C = get_species_data(name).antoine_coeffs
        T = self.TB[name]
        assert 10 ** (A - B / (T + C)) == pytest.approx(101325.0, rel=0.03)

    @pytest.mark.parametrize("name", sorted(TB))
    def test_critical_constants_agree_with_tsonopoulos_ambrose(self, name):
        # The stored constants are Lemmon & Ihmels (2005); Tsonopoulos &
        # Ambrose (1996), via the NIST WebBook, is the independent check.
        ta = {"cis_2_butene": (435.5, 42.1e5), "trans_2_butene": (428.6, 41.0e5),
              "isobutylene": (417.9, 40.0e5)}[name]
        p = get_critical_props(name)
        assert p.Tc == pytest.approx(ta[0], abs=0.5)
        assert p.Pc == pytest.approx(ta[1], rel=0.025)
        assert 0.18 < p.omega < 0.22

    def test_aliases_and_sources(self):
        from difflow.database import SOURCE_CITATIONS, resolve_alias
        assert resolve_alias("isobutene") == "isobutylene"
        for name in self.TB:
            assert "Lemmon" in SOURCE_CITATIONS[name]

    def test_volatility_order_from_the_critical_constants(self):
        # Normal boiling points order isobutylene < trans < cis; Wilson's
        # K (from Tc, Pc, omega alone) must order them the same way.
        eos = PengRobinson({n: get_critical_props(n) for n in
                            ("isobutylene", "trans_2_butene", "cis_2_butene")})
        K = eos.K_values_wilson(300.0, 5e5)
        assert K[0] > K[1] > K[2]


class TestC6Isomers:
    """The branched hexanes and C6 naphthenes of light-naphtha
    isomerization (#311), each checked against a number published
    independently of the one stored."""

    # NIST WebBook ideal-gas Cp tables (J/mol/K): Scott (1974) for the
    # branched hexanes, TRC (1997) for methylcyclopentane, Dorofeeva et al.
    # (1986) for cyclohexane -- the tables the stored cubics were fitted to.
    T = (273.15, 298.15, 400.0, 500.0, 600.0, 700.0, 800.0)
    NIST_CP = {
        "2_methylpentane": (131.88, 142.2, 183.51, 219.83, 251.04, 277.40, 300.41),
        "3_methylpentane": (129.83, 140.1, 181.17, 217.48, 248.95, 275.73, 298.74),
        "2_2_dimethylbutane": (131.08, 141.5, 183.13, 220.33, 253.13, 281.58, 306.69),
        "2_3_dimethylbutane": (128.32, 139.4, 181.71, 218.36, 250.2, 277.4, 301.67),
        "methylcyclopentane": (99.62, 109.5, 151.5, 188.9, 220.3, 246.6, 268.6),
        "cyclohexane": (95.2, 105.3, 148.64, 188.68, 223.38, 252.62, 277.05),
    }
    # NIST WebBook normal boiling points (K)
    TB = {"2_methylpentane": 333.4, "3_methylpentane": 336.4,
          "2_2_dimethylbutane": 322.9, "2_3_dimethylbutane": 331.2,
          "methylcyclopentane": 345.0, "cyclohexane": 353.9}
    # Passut & Danner (1973) critical constants (K, Pa): independent of the
    # stored PSRK-appendix values
    PASSUT_DANNER = {"2_methylpentane": (497.5, 3.010e6), "3_methylpentane": (504.43, 3.124e6),
                     "2_2_dimethylbutane": (488.78, 3.081e6),
                     "2_3_dimethylbutane": (499.98, 3.127e6),
                     "methylcyclopentane": (532.79, 3.785e6),
                     "cyclohexane": (553.54, 4.075e6)}

    @pytest.mark.parametrize("name", sorted(NIST_CP))
    def test_ideal_gas_cp_matches_the_nist_table(self, name):
        a, b, c, d = get_species_data(name).Cp_coeffs
        for T, cp in zip(self.T, self.NIST_CP[name]):
            assert a + b * T + c * T**2 + d * T**3 == pytest.approx(cp, rel=0.006)

    @pytest.mark.parametrize("name", sorted(TB))
    def test_antoine_boils_at_the_normal_boiling_point(self, name):
        A, B, C = get_species_data(name).antoine_coeffs
        T = self.TB[name]
        assert 10 ** (A - B / (T + C)) == pytest.approx(101325.0, rel=0.01)

    @pytest.mark.parametrize("name", sorted(PASSUT_DANNER))
    def test_critical_constants_agree_with_passut_danner(self, name):
        Tc, Pc = self.PASSUT_DANNER[name]
        p = get_critical_props(name)
        assert p.Tc == pytest.approx(Tc, abs=0.5)
        assert p.Pc == pytest.approx(Pc, rel=0.012)
        assert 0.2 < p.omega < 0.3

    def test_aliases_and_sources(self):
        from difflow.database import SOURCE_CITATIONS
        assert resolve_alias("2,2-dimethylbutane") == "2_2_dimethylbutane"
        assert resolve_alias("neohexane") == "2_2_dimethylbutane"
        assert resolve_alias("MCP") == "methylcyclopentane"
        assert get_critical_props("2,3-dimethylbutane").Tc == pytest.approx(500.0)
        for name in self.TB:
            assert "Horstmann" in SOURCE_CITATIONS[name]

    def test_volatility_order_from_the_critical_constants(self):
        # boiling points order 22DMB < 23DMB < 2MP < 3MP < MCP < CH; Wilson's
        # K from Tc, Pc and omega alone must order them the same way
        names = sorted(self.TB, key=self.TB.get)
        eos = PengRobinson({n: get_critical_props(n) for n in names})
        K = eos.K_values_wilson(340.0, 1e5)
        assert all(K[i] > K[i + 1] for i in range(len(names) - 1))


class TestHydrotreatingSpecies:
    """H2, H2S and thiophene are in both databases (#391)."""

    @pytest.mark.parametrize("name", ["hydrogen", "hydrogen_sulfide", "thiophene"])
    def test_present_in_both(self, name):
        from difflow.database import get_critical_props, get_species_data

        assert get_species_data(name).name == name
        assert get_critical_props(name).Tc > 0

    def test_thiophene_critical_properties(self):
        from difflow.database import get_critical_props

        c = get_critical_props("thiophene")
        assert c.Tc == pytest.approx(579.4)
        assert c.Pc == pytest.approx(5.70e6)
        assert c.omega == pytest.approx(0.200)

    @pytest.mark.parametrize("name, Tb, Hvap", [
        ("hydrogen", 20.39, 900.0),
        ("hydrogen_sulfide", 213.6, 18670.0),
        ("thiophene", 357.15, 31480.0),
    ])
    def test_antoine_and_watson_hit_the_normal_boiling_point(self, name, Tb, Hvap):
        from difflow.database import get_species_data
        from difflow.thermo import IdealThermo

        thermo = IdealThermo({name: get_species_data(name)})
        assert float(thermo.Psat(name, Tb)) == pytest.approx(101325.0, rel=0.05)
        assert float(thermo.Hvap(name, Tb)) == pytest.approx(Hvap, rel=1e-9)

    def test_formation_enthalpies(self):
        from difflow.database import get_species_data

        assert get_species_data("hydrogen").Hf == 0.0
        assert get_species_data("hydrogen_sulfide").Hf == -20600.0
        assert get_species_data("thiophene").Hf == 114900.0
