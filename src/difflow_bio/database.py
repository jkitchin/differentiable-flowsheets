"""Database module for Bio Manufacturing plugin.

Provides access to properties of cell lines, chromatography resins,
and membrane materials for biopharmaceutical process simulation.

All property data includes literature-validated parameters with
appropriate references for monoclonal antibody manufacturing.

References:
    Cell culture:
        - Ozturk SS (2006). Biotechnol Bioeng 94:147
        - Wlaschin KF, Hu WS (2006). Trends Biotechnol 24:10
    Chromatography:
        - Carta G, Jungbauer A (2010). Protein Chromatography.
        - GE Healthcare (2020). Affinity Chromatography Handbook.
    Membrane:
        - Zydney AL (2016). Biotechnol Bioeng 113:465
"""

__all__ = [
    # Cell Lines
    "CellLine",
    "CellLineDatabase",
    "get_cell_line",
    "list_cell_lines",
    # Resins
    "Resin",
    "ResinDatabase",
    "get_resin",
    "list_resins",
    # Membranes
    "BioMembrane",
    "BioMembraneDatabase",
    "get_bio_membrane",
    "list_bio_membranes",
    # JAX accessors
    "get_kinetic_params_array",
    "get_resin_capacity_array",
]

from dataclasses import dataclass
import jax.numpy as jnp
from jax import Array


# =============================================================================
# Cell Line Data
# =============================================================================

@dataclass(frozen=True)
class CellLine:
    """Properties of a mammalian cell line for biomanufacturing.

    Attributes:
        name: Cell line identifier (e.g., 'CHO-K1')
        full_name: Full name
        organism: Source organism
        cell_type: Cell type (e.g., 'CHO', 'HEK293', 'NS0')
        mu_max: Maximum specific growth rate (1/h)
        K_s: Monod saturation constant for glucose (g/L)
        Y_xs: Yield coefficient, cells/glucose (g/g)
        k_d: Death rate constant (1/h)
        m_s: Maintenance coefficient (g glucose/g cells/h)
        q_p_max: Maximum specific productivity (pg/cell/day)
        typical_titer: Typical product titer (g/L)
        doubling_time: Doubling time (h)
        viable_density_max: Maximum viable cell density (cells/mL)
        optimal_temperature: Optimal culture temperature (K)
        optimal_pH: Optimal culture pH
        reference: Literature reference
    """
    name: str
    full_name: str
    organism: str
    cell_type: str
    mu_max: float
    K_s: float
    Y_xs: float
    k_d: float
    m_s: float
    q_p_max: float
    typical_titer: float
    doubling_time: float
    viable_density_max: float
    optimal_temperature: float
    optimal_pH: float
    reference: str


# Default cell line data
_CELL_LINE_DATA = {
    "CHO-K1": CellLine(
        name="CHO-K1",
        full_name="Chinese Hamster Ovary K1",
        organism="Cricetulus griseus",
        cell_type="CHO",
        mu_max=0.03,  # 1/h
        K_s=0.5,  # g/L glucose
        Y_xs=0.4,  # g cells / g glucose
        k_d=0.001,  # 1/h
        m_s=0.02,  # g glucose / g cells / h
        q_p_max=50.0,  # pg/cell/day
        typical_titer=5.0,  # g/L
        doubling_time=24.0,  # h
        viable_density_max=2e7,  # cells/mL
        optimal_temperature=310.15,  # K (37C)
        optimal_pH=7.0,
        reference="Ozturk SS (2006). Biotechnol Bioeng 94:147",
    ),
    "CHO-DG44": CellLine(
        name="CHO-DG44",
        full_name="Chinese Hamster Ovary DG44 (DHFR-)",
        organism="Cricetulus griseus",
        cell_type="CHO",
        mu_max=0.028,
        K_s=0.4,
        Y_xs=0.38,
        k_d=0.0012,
        m_s=0.018,
        q_p_max=60.0,
        typical_titer=6.0,
        doubling_time=26.0,
        viable_density_max=2.5e7,
        optimal_temperature=310.15,
        optimal_pH=7.0,
        reference="Wurm FM (2004). Nat Biotechnol 22:1393",
    ),
    "CHO-S": CellLine(
        name="CHO-S",
        full_name="Chinese Hamster Ovary Suspension",
        organism="Cricetulus griseus",
        cell_type="CHO",
        mu_max=0.032,
        K_s=0.45,
        Y_xs=0.42,
        k_d=0.001,
        m_s=0.022,
        q_p_max=55.0,
        typical_titer=5.5,
        doubling_time=22.0,
        viable_density_max=2.2e7,
        optimal_temperature=310.15,
        optimal_pH=7.0,
        reference="Thermo Fisher Scientific",
    ),
    "HEK293": CellLine(
        name="HEK293",
        full_name="Human Embryonic Kidney 293",
        organism="Homo sapiens",
        cell_type="HEK293",
        mu_max=0.035,
        K_s=0.6,
        Y_xs=0.35,
        k_d=0.0015,
        m_s=0.025,
        q_p_max=30.0,
        typical_titer=1.0,
        doubling_time=20.0,
        viable_density_max=3e6,
        optimal_temperature=310.15,
        optimal_pH=7.4,
        reference="ATCC CRL-1573",
    ),
    "NS0": CellLine(
        name="NS0",
        full_name="Mouse Myeloma NS0",
        organism="Mus musculus",
        cell_type="NS0",
        mu_max=0.025,
        K_s=0.5,
        Y_xs=0.36,
        k_d=0.002,
        m_s=0.03,
        q_p_max=40.0,
        typical_titer=3.0,
        doubling_time=28.0,
        viable_density_max=1e7,
        optimal_temperature=310.15,
        optimal_pH=7.2,
        reference="Barnes LM et al. (2000). Cytotechnology 32:109",
    ),
    "Sp2/0": CellLine(
        name="Sp2/0",
        full_name="Mouse Myeloma Sp2/0-Ag14",
        organism="Mus musculus",
        cell_type="Sp2/0",
        mu_max=0.024,
        K_s=0.55,
        Y_xs=0.34,
        k_d=0.0018,
        m_s=0.028,
        q_p_max=35.0,
        typical_titer=2.5,
        doubling_time=30.0,
        viable_density_max=8e6,
        optimal_temperature=310.15,
        optimal_pH=7.2,
        reference="ATCC CRL-1581",
    ),
}


class CellLineDatabase:
    """Database of mammalian cell line properties.

    Example:
        >>> db = CellLineDatabase()
        >>> cho = db.get('CHO-K1')
        >>> print(f"CHO mu_max: {cho.mu_max} 1/h")
    """

    def __init__(self):
        """Initialize cell line database."""
        self._cell_lines = _CELL_LINE_DATA.copy()

    def get(self, name: str) -> CellLine:
        """Get cell line by name."""
        if name not in self._cell_lines:
            raise KeyError(
                f"Unknown cell line: {name}. "
                f"Available: {list(self._cell_lines.keys())}"
            )
        return self._cell_lines[name]

    def __getitem__(self, name: str) -> CellLine:
        return self.get(name)

    def list_cell_lines(self) -> list[str]:
        """List all available cell line names."""
        return list(self._cell_lines.keys())

    def list_by_type(self, cell_type: str) -> list[str]:
        """List cell lines by type (CHO, HEK293, NS0, etc.)."""
        return [
            name for name, cl in self._cell_lines.items()
            if cl.cell_type == cell_type
        ]


# =============================================================================
# Chromatography Resin Data
# =============================================================================

@dataclass(frozen=True)
class Resin:
    """Properties of a chromatography resin.

    Attributes:
        name: Resin identifier
        full_name: Full commercial name
        manufacturer: Manufacturer
        resin_type: protein_a, cation_exchange, anion_exchange, sec, hic
        ligand: Affinity ligand or functional group
        base_matrix: Base matrix material
        particle_size: Mean particle diameter (um)
        pore_size: Mean pore size (nm)
        q_max: Static binding capacity (g protein / L resin)
        K_d: Dissociation constant (g/L) for affinity resins
        pH_range: Operating pH range
        max_pressure: Maximum operating pressure (bar)
        cycles: Expected cycle lifetime
        cost_usd_L: Cost (USD/L resin)
        reference: Literature/manufacturer reference
    """
    name: str
    full_name: str
    manufacturer: str
    resin_type: str
    ligand: str
    base_matrix: str
    particle_size: float
    pore_size: float
    q_max: float
    K_d: float
    pH_range: tuple[float, float]
    max_pressure: float
    cycles: int
    cost_usd_L: float
    reference: str


_RESIN_DATA = {
    # Protein A resins
    "MabSelect_SuRe": Resin(
        name="MabSelect_SuRe",
        full_name="MabSelect SuRe LX",
        manufacturer="Cytiva",
        resin_type="protein_a",
        ligand="Protein A (alkaline-stabilized)",
        base_matrix="Highly cross-linked agarose",
        particle_size=85.0,  # um
        pore_size=None,  # Not typically specified
        q_max=35.0,  # g/L at 10% breakthrough
        K_d=0.05,  # g/L, high affinity
        pH_range=(3.0, 12.0),
        max_pressure=3.0,  # bar
        cycles=200,
        cost_usd_L=15000.0,
        reference="Cytiva Data Sheet 29-0490-01",
    ),
    "MabSelect_PrismA": Resin(
        name="MabSelect_PrismA",
        full_name="MabSelect PrismA",
        manufacturer="Cytiva",
        resin_type="protein_a",
        ligand="Protein A (engineered)",
        base_matrix="Highly cross-linked agarose",
        particle_size=60.0,
        pore_size=None,
        q_max=65.0,  # Higher capacity
        K_d=0.03,
        pH_range=(3.0, 12.0),
        max_pressure=5.0,
        cycles=250,
        cost_usd_L=20000.0,
        reference="Cytiva Data Sheet 29-1504-64",
    ),
    "Protein_A_Sepharose": Resin(
        name="Protein_A_Sepharose",
        full_name="Protein A Sepharose 4 Fast Flow",
        manufacturer="Cytiva",
        resin_type="protein_a",
        ligand="Protein A (native)",
        base_matrix="Cross-linked agarose",
        particle_size=90.0,
        pore_size=None,
        q_max=30.0,
        K_d=0.1,
        pH_range=(2.0, 11.0),
        max_pressure=1.0,
        cycles=100,
        cost_usd_L=12000.0,
        reference="Cytiva Data Sheet 71-5000-14",
    ),
    # Cation Exchange resins
    "SP_Sepharose_FF": Resin(
        name="SP_Sepharose_FF",
        full_name="SP Sepharose Fast Flow",
        manufacturer="Cytiva",
        resin_type="cation_exchange",
        ligand="Sulfopropyl (strong cation)",
        base_matrix="Cross-linked agarose",
        particle_size=90.0,
        pore_size=None,
        q_max=55.0,  # mg lysozyme/mL
        K_d=0.3,
        pH_range=(4.0, 13.0),
        max_pressure=1.0,
        cycles=200,
        cost_usd_L=3000.0,
        reference="Cytiva Data Sheet 18-1023-25",
    ),
    "Capto_S_ImpAct": Resin(
        name="Capto_S_ImpAct",
        full_name="Capto S ImpAct",
        manufacturer="Cytiva",
        resin_type="cation_exchange",
        ligand="Sulfonate (strong cation)",
        base_matrix="Highly cross-linked agarose",
        particle_size=40.0,
        pore_size=None,
        q_max=85.0,
        K_d=0.25,
        pH_range=(4.0, 12.0),
        max_pressure=3.0,
        cycles=300,
        cost_usd_L=5000.0,
        reference="Cytiva Data Sheet 29-0366-19",
    ),
    # Anion Exchange resins
    "Q_Sepharose_FF": Resin(
        name="Q_Sepharose_FF",
        full_name="Q Sepharose Fast Flow",
        manufacturer="Cytiva",
        resin_type="anion_exchange",
        ligand="Quaternary amine (strong anion)",
        base_matrix="Cross-linked agarose",
        particle_size=90.0,
        pore_size=None,
        q_max=50.0,  # mg BSA/mL
        K_d=0.4,
        pH_range=(2.0, 12.0),
        max_pressure=1.0,
        cycles=200,
        cost_usd_L=3000.0,
        reference="Cytiva Data Sheet 18-1023-25",
    ),
    "Capto_Q": Resin(
        name="Capto_Q",
        full_name="Capto Q",
        manufacturer="Cytiva",
        resin_type="anion_exchange",
        ligand="Quaternary amine",
        base_matrix="Highly cross-linked agarose",
        particle_size=90.0,
        pore_size=None,
        q_max=70.0,
        K_d=0.35,
        pH_range=(2.0, 12.0),
        max_pressure=3.0,
        cycles=300,
        cost_usd_L=4500.0,
        reference="Cytiva Data Sheet 28-9365-00",
    ),
    # SEC resins
    "Superdex_200": Resin(
        name="Superdex_200",
        full_name="Superdex 200 Increase",
        manufacturer="Cytiva",
        resin_type="sec",
        ligand="None (size exclusion)",
        base_matrix="Cross-linked agarose/dextran",
        particle_size=8.6,  # um, prep grade is larger
        pore_size=None,  # Fractionation range 10-600 kDa
        q_max=0.0,  # Not applicable for SEC
        K_d=0.0,
        pH_range=(3.0, 12.0),
        max_pressure=40.0,  # Higher for analytical
        cycles=1000,
        cost_usd_L=8000.0,
        reference="Cytiva Data Sheet 29-0487-00",
    ),
}


class ResinDatabase:
    """Database of chromatography resin properties.

    Example:
        >>> db = ResinDatabase()
        >>> proa = db.get('MabSelect_SuRe')
        >>> print(f"Binding capacity: {proa.q_max} g/L")
    """

    def __init__(self):
        """Initialize resin database."""
        self._resins = _RESIN_DATA.copy()

    def get(self, name: str) -> Resin:
        """Get resin by name."""
        if name not in self._resins:
            raise KeyError(
                f"Unknown resin: {name}. "
                f"Available: {list(self._resins.keys())}"
            )
        return self._resins[name]

    def __getitem__(self, name: str) -> Resin:
        return self.get(name)

    def list_resins(self) -> list[str]:
        """List all available resin names."""
        return list(self._resins.keys())

    def list_by_type(self, resin_type: str) -> list[str]:
        """List resins by type (protein_a, cation_exchange, etc.)."""
        return [
            name for name, r in self._resins.items()
            if r.resin_type == resin_type
        ]


# =============================================================================
# Membrane Data
# =============================================================================

@dataclass(frozen=True)
class BioMembrane:
    """Properties of a bioprocess membrane.

    Attributes:
        name: Membrane identifier
        full_name: Full commercial name
        manufacturer: Manufacturer
        membrane_type: uf, mf, vf (ultrafiltration, microfiltration, virus)
        material: Membrane material
        MWCO: Molecular weight cutoff (kDa) for UF
        pore_size: Pore size (um) for MF
        Lp: Clean water permeability (L/m^2/h/bar)
        protein_retention: Typical protein retention (%)
        max_TMP: Maximum transmembrane pressure (bar)
        max_temperature: Maximum temperature (C)
        pH_range: Operating pH range
        cost_usd_m2: Cost (USD/m^2)
        reference: Manufacturer reference
    """
    name: str
    full_name: str
    manufacturer: str
    membrane_type: str
    material: str
    MWCO: float | None
    pore_size: float | None
    Lp: float
    protein_retention: float
    max_TMP: float
    max_temperature: float
    pH_range: tuple[float, float]
    cost_usd_m2: float
    reference: str


_MEMBRANE_DATA = {
    # Ultrafiltration membranes
    "Pellicon_3_30kDa": BioMembrane(
        name="Pellicon_3_30kDa",
        full_name="Pellicon 3 Ultracel 30 kDa",
        manufacturer="MilliporeSigma",
        membrane_type="uf",
        material="Regenerated cellulose",
        MWCO=30.0,
        pore_size=None,
        Lp=65.0,  # L/m^2/h/bar
        protein_retention=99.5,  # % for mAb
        max_TMP=4.0,
        max_temperature=50.0,
        pH_range=(2.0, 14.0),
        cost_usd_m2=400.0,
        reference="MilliporeSigma P3C030C01",
    ),
    "Pellicon_3_10kDa": BioMembrane(
        name="Pellicon_3_10kDa",
        full_name="Pellicon 3 Ultracel 10 kDa",
        manufacturer="MilliporeSigma",
        membrane_type="uf",
        material="Regenerated cellulose",
        MWCO=10.0,
        pore_size=None,
        Lp=45.0,
        protein_retention=99.9,
        max_TMP=4.0,
        max_temperature=50.0,
        pH_range=(2.0, 14.0),
        cost_usd_m2=400.0,
        reference="MilliporeSigma P3C010C01",
    ),
    "Biomax_50kDa": BioMembrane(
        name="Biomax_50kDa",
        full_name="Biomax 50 kDa PES",
        manufacturer="MilliporeSigma",
        membrane_type="uf",
        material="Polyethersulfone",
        MWCO=50.0,
        pore_size=None,
        Lp=80.0,
        protein_retention=98.0,
        max_TMP=5.0,
        max_temperature=55.0,
        pH_range=(1.0, 14.0),
        cost_usd_m2=350.0,
        reference="MilliporeSigma PBQK0MP04",
    ),
    "Kvick_30kDa": BioMembrane(
        name="Kvick_30kDa",
        full_name="Kvick Flow 30 kDa",
        manufacturer="Cytiva",
        membrane_type="uf",
        material="Polyethersulfone",
        MWCO=30.0,
        pore_size=None,
        Lp=70.0,
        protein_retention=99.0,
        max_TMP=4.0,
        max_temperature=50.0,
        pH_range=(1.0, 14.0),
        cost_usd_m2=380.0,
        reference="Cytiva UFP-30-C-4X2MA",
    ),
    # Microfiltration membranes
    "Pellicon_3_0.45um": BioMembrane(
        name="Pellicon_3_0.45um",
        full_name="Pellicon 3 Durapore 0.45 um",
        manufacturer="MilliporeSigma",
        membrane_type="mf",
        material="PVDF",
        MWCO=None,
        pore_size=0.45,  # um
        Lp=500.0,
        protein_retention=0.0,  # Product passes through
        max_TMP=2.0,
        max_temperature=40.0,
        pH_range=(1.0, 14.0),
        cost_usd_m2=300.0,
        reference="MilliporeSigma P3GVPP001",
    ),
    "Pellicon_3_0.22um": BioMembrane(
        name="Pellicon_3_0.22um",
        full_name="Pellicon 3 Durapore 0.22 um",
        manufacturer="MilliporeSigma",
        membrane_type="mf",
        material="PVDF",
        MWCO=None,
        pore_size=0.22,
        Lp=350.0,
        protein_retention=0.0,
        max_TMP=2.0,
        max_temperature=40.0,
        pH_range=(1.0, 14.0),
        cost_usd_m2=320.0,
        reference="MilliporeSigma P3GVPP001",
    ),
    # Virus filtration
    "Viresolve_Pro": BioMembrane(
        name="Viresolve_Pro",
        full_name="Viresolve Pro",
        manufacturer="MilliporeSigma",
        membrane_type="vf",
        material="PVDF",
        MWCO=None,
        pore_size=0.020,  # 20 nm nominal
        Lp=30.0,  # Lower due to tight pores
        protein_retention=0.0,  # mAb passes through
        max_TMP=3.5,
        max_temperature=40.0,
        pH_range=(3.0, 10.0),
        cost_usd_m2=2000.0,  # Expensive
        reference="MilliporeSigma VPM60001HODA",
    ),
    "Planova_20N": BioMembrane(
        name="Planova_20N",
        full_name="Planova 20N",
        manufacturer="Asahi Kasei",
        membrane_type="vf",
        material="Cuprammonium regenerated cellulose",
        MWCO=None,
        pore_size=0.020,
        Lp=25.0,
        protein_retention=0.0,
        max_TMP=1.0,
        max_temperature=40.0,
        pH_range=(4.0, 8.0),
        cost_usd_m2=2500.0,
        reference="Asahi Kasei Bioprocess",
    ),
}


class BioMembraneDatabase:
    """Database of bioprocess membrane properties.

    Example:
        >>> db = BioMembraneDatabase()
        >>> mem = db.get('Pellicon_3_30kDa')
        >>> print(f"MWCO: {mem.MWCO} kDa")
    """

    def __init__(self):
        """Initialize membrane database."""
        self._membranes = _MEMBRANE_DATA.copy()

    def get(self, name: str) -> BioMembrane:
        """Get membrane by name."""
        if name not in self._membranes:
            raise KeyError(
                f"Unknown membrane: {name}. "
                f"Available: {list(self._membranes.keys())}"
            )
        return self._membranes[name]

    def __getitem__(self, name: str) -> BioMembrane:
        return self.get(name)

    def list_membranes(self) -> list[str]:
        """List all available membrane names."""
        return list(self._membranes.keys())

    def list_by_type(self, membrane_type: str) -> list[str]:
        """List membranes by type (uf, mf, vf)."""
        return [
            name for name, m in self._membranes.items()
            if m.membrane_type == membrane_type
        ]


# =============================================================================
# Global Database Instances (lazy loaded)
# =============================================================================

_cell_line_db: CellLineDatabase | None = None
_resin_db: ResinDatabase | None = None
_membrane_db: BioMembraneDatabase | None = None


def get_cell_line_database() -> CellLineDatabase:
    """Get the cell line database (singleton)."""
    global _cell_line_db
    if _cell_line_db is None:
        _cell_line_db = CellLineDatabase()
    return _cell_line_db


def get_resin_database() -> ResinDatabase:
    """Get the resin database (singleton)."""
    global _resin_db
    if _resin_db is None:
        _resin_db = ResinDatabase()
    return _resin_db


def get_membrane_database() -> BioMembraneDatabase:
    """Get the membrane database (singleton)."""
    global _membrane_db
    if _membrane_db is None:
        _membrane_db = BioMembraneDatabase()
    return _membrane_db


# =============================================================================
# Convenience Functions
# =============================================================================

def get_cell_line(name: str) -> CellLine:
    """Get cell line properties.

    Args:
        name: Cell line name (e.g., 'CHO-K1', 'HEK293')

    Returns:
        CellLine dataclass with all properties

    Example:
        >>> cho = get_cell_line('CHO-K1')
        >>> print(f"Max growth rate: {cho.mu_max} 1/h")
    """
    return get_cell_line_database().get(name)


def get_resin(name: str) -> Resin:
    """Get chromatography resin properties.

    Args:
        name: Resin name (e.g., 'MabSelect_SuRe', 'SP_Sepharose_FF')

    Returns:
        Resin dataclass with all properties

    Example:
        >>> proa = get_resin('MabSelect_SuRe')
        >>> print(f"Binding capacity: {proa.q_max} g/L")
    """
    return get_resin_database().get(name)


def get_bio_membrane(name: str) -> BioMembrane:
    """Get bioprocess membrane properties.

    Args:
        name: Membrane name (e.g., 'Pellicon_3_30kDa')

    Returns:
        BioMembrane dataclass with all properties

    Example:
        >>> mem = get_bio_membrane('Pellicon_3_30kDa')
        >>> print(f"MWCO: {mem.MWCO} kDa")
    """
    return get_membrane_database().get(name)


def list_cell_lines() -> list[str]:
    """List all available cell lines."""
    return get_cell_line_database().list_cell_lines()


def list_resins() -> list[str]:
    """List all available chromatography resins."""
    return get_resin_database().list_resins()


def list_bio_membranes() -> list[str]:
    """List all available bioprocess membranes."""
    return get_membrane_database().list_membranes()


# =============================================================================
# JAX-Compatible Data Accessors
# =============================================================================

def get_kinetic_params_array(cell_line: str) -> dict[str, Array]:
    """Get cell kinetic parameters as JAX arrays.

    Args:
        cell_line: Cell line name

    Returns:
        Dictionary with kinetic parameter arrays
    """
    cl = get_cell_line(cell_line)
    return {
        "mu_max": jnp.array(cl.mu_max),
        "K_s": jnp.array(cl.K_s),
        "Y_xs": jnp.array(cl.Y_xs),
        "k_d": jnp.array(cl.k_d),
        "m_s": jnp.array(cl.m_s),
    }


def get_resin_capacity_array(resins: list[str]) -> Array:
    """Get resin binding capacities as JAX array.

    Args:
        resins: List of resin names

    Returns:
        JAX array of binding capacities (g/L)
    """
    db = get_resin_database()
    capacities = [db.get(r).q_max for r in resins]
    return jnp.array(capacities)
