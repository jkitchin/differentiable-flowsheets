"""Visual styles for unit operations in flowsheet visualization."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class UnitStyle:
    """Visual style definition for a unit operation type.

    Attributes:
        color: Fill color (CSS color string)
        border_color: Border/stroke color
        shape: Node shape ('circle', 'rect', 'diamond', 'hexagon')
        size: Base size for the node
        icon: Optional unicode icon or emoji
        label_position: Position of label ('center', 'bottom', 'top')
    """
    color: str = "#6baed6"
    border_color: str = "#2171b5"
    shape: str = "circle"
    size: int = 40
    icon: Optional[str] = None
    label_position: str = "bottom"


# Default styles for different unit operation types
UNIT_STYLES: dict[str, UnitStyle] = {
    # Core chemical operations
    "CSTR": UnitStyle(
        color="#74c476",
        border_color="#238b45",
        shape="circle",
        size=50,
        icon="⚗",
    ),
    "PFR": UnitStyle(
        color="#74c476",
        border_color="#238b45",
        shape="rect",
        size=45,
        icon="→",
    ),
    "Flash": UnitStyle(
        color="#9ecae1",
        border_color="#3182bd",
        shape="diamond",
        size=45,
        icon="◇",
    ),
    "Distillation": UnitStyle(
        color="#9ecae1",
        border_color="#3182bd",
        shape="rect",
        size=55,
        icon="⫘",
    ),
    "HeatExchanger": UnitStyle(
        color="#fdae6b",
        border_color="#e6550d",
        shape="rect",
        size=40,
        icon="≋",
    ),
    "Mixer": UnitStyle(
        color="#c7e9c0",
        border_color="#74c476",
        shape="circle",
        size=30,
        icon="+",
    ),
    "Splitter": UnitStyle(
        color="#c7e9c0",
        border_color="#74c476",
        shape="diamond",
        size=30,
        icon="⋔",
    ),

    # Liquid-liquid extraction
    "LLE": UnitStyle(
        color="#bcbddc",
        border_color="#756bb1",
        shape="rect",
        size=45,
        icon="⇌",
    ),
    "Cascade": UnitStyle(
        color="#bcbddc",
        border_color="#756bb1",
        shape="rect",
        size=55,
        icon="⋯",
    ),

    # Bio manufacturing - upstream
    "Bioreactor": UnitStyle(
        color="#a1d99b",
        border_color="#31a354",
        shape="circle",
        size=55,
        icon="🧫",
    ),
    "ContinuousBioreactor": UnitStyle(
        color="#a1d99b",
        border_color="#31a354",
        shape="circle",
        size=55,
        icon="⟳",
    ),
    "FedBatchBioreactor": UnitStyle(
        color="#74c476",
        border_color="#238b45",
        shape="circle",
        size=55,
        icon="⊕",
    ),

    # Bio manufacturing - harvest
    "Centrifuge": UnitStyle(
        color="#fdd0a2",
        border_color="#f16913",
        shape="circle",
        size=45,
        icon="◎",
    ),
    "DiscStackCentrifuge": UnitStyle(
        color="#fdd0a2",
        border_color="#f16913",
        shape="circle",
        size=50,
        icon="⊚",
    ),

    # Bio manufacturing - filtration
    "Ultrafiltration": UnitStyle(
        color="#c6dbef",
        border_color="#4292c6",
        shape="rect",
        size=45,
        icon="▥",
    ),
    "Diafiltration": UnitStyle(
        color="#c6dbef",
        border_color="#4292c6",
        shape="rect",
        size=45,
        icon="▤",
    ),
    "TFF": UnitStyle(
        color="#9ecae1",
        border_color="#2171b5",
        shape="rect",
        size=50,
        icon="⧈",
    ),

    # Bio manufacturing - chromatography
    "ProteinAChromatography": UnitStyle(
        color="#dadaeb",
        border_color="#9e9ac8",
        shape="rect",
        size=50,
        icon="⫿",
    ),
    "IonExchangeChromatography": UnitStyle(
        color="#dadaeb",
        border_color="#807dba",
        shape="rect",
        size=50,
        icon="±",
    ),
    "SizeExclusionChromatography": UnitStyle(
        color="#dadaeb",
        border_color="#6a51a3",
        shape="rect",
        size=50,
        icon="○",
    ),

    # Feed and product nodes
    "feed": UnitStyle(
        color="#fee5d9",
        border_color="#fb6a4a",
        shape="circle",
        size=25,
        icon="→",
        label_position="top",
    ),
    "product": UnitStyle(
        color="#deebf7",
        border_color="#3182bd",
        shape="circle",
        size=25,
        icon="⊙",
        label_position="top",
    ),
    "waste": UnitStyle(
        color="#f0f0f0",
        border_color="#969696",
        shape="circle",
        size=20,
        icon="×",
        label_position="top",
    ),

    # Generic fallback
    "generic": UnitStyle(
        color="#d9d9d9",
        border_color="#737373",
        shape="circle",
        size=40,
    ),
}


def get_unit_style(unit_type: str) -> UnitStyle:
    """Get the visual style for a unit operation type.

    Args:
        unit_type: The type of unit operation

    Returns:
        UnitStyle for the given type, or generic style if not found
    """
    return UNIT_STYLES.get(unit_type, UNIT_STYLES["generic"])


# Color palettes for streams
STREAM_COLORS = {
    "default": "#636363",
    "feed": "#e6550d",
    "product": "#3182bd",
    "recycle": "#31a354",
    "vapor": "#9ecae1",
    "liquid": "#6baed6",
    "organic": "#fdae6b",
    "aqueous": "#a1d99b",
}


def get_stream_color(stream_type: str = "default") -> str:
    """Get color for a stream type."""
    return STREAM_COLORS.get(stream_type, STREAM_COLORS["default"])
