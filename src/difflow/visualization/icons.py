"""SVG icons for chemical engineering unit operations.

Icons are based on ISO 10628 P&ID symbols, simplified for clarity.
Each icon is provided as an SVG string and as a data URI for use in
visualization libraries.
"""

import base64
from typing import Callable

# Icon dimensions (viewBox)
ICON_SIZE = 60


def _svg_to_data_uri(svg: str) -> str:
    """Convert SVG string to data URI for embedding in HTML/CSS."""
    encoded = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
    return f"data:image/svg+xml;base64,{encoded}"


# =============================================================================
# Core Unit Operation Icons
# =============================================================================

def _cstr_svg(fill: str = "#4a90d9", stroke: str = "#2c5aa0") -> str:
    """CSTR - Continuous Stirred Tank Reactor (tank with agitator)."""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ICON_SIZE} {ICON_SIZE}">
  <rect x="10" y="15" width="40" height="35" rx="3" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
  <line x1="30" y1="5" x2="30" y2="20" stroke="{stroke}" stroke-width="2"/>
  <line x1="22" y1="30" x2="38" y2="30" stroke="{stroke}" stroke-width="2"/>
  <line x1="25" y1="36" x2="35" y2="36" stroke="{stroke}" stroke-width="2"/>
  <ellipse cx="30" cy="50" rx="12" ry="3" fill="{fill}" stroke="{stroke}" stroke-width="1"/>
</svg>'''


def _pfr_svg(fill: str = "#5ba55b", stroke: str = "#3d7a3d") -> str:
    """PFR - Plug Flow Reactor (horizontal tube)."""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ICON_SIZE} {ICON_SIZE}">
  <rect x="5" y="20" width="50" height="20" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
  <line x1="15" y1="25" x2="15" y2="35" stroke="{stroke}" stroke-width="1.5"/>
  <line x1="25" y1="25" x2="25" y2="35" stroke="{stroke}" stroke-width="1.5"/>
  <line x1="35" y1="25" x2="35" y2="35" stroke="{stroke}" stroke-width="1.5"/>
  <line x1="45" y1="25" x2="45" y2="35" stroke="{stroke}" stroke-width="1.5"/>
</svg>'''


def _flash_svg(fill: str = "#9b59b6", stroke: str = "#7d3c98") -> str:
    """Flash drum - Vapor-liquid separator (vertical vessel with V/L)."""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ICON_SIZE} {ICON_SIZE}">
  <ellipse cx="30" cy="12" rx="15" ry="6" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
  <rect x="15" y="12" width="30" height="36" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
  <ellipse cx="30" cy="48" rx="15" ry="6" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
  <line x1="15" y1="30" x2="45" y2="30" stroke="{stroke}" stroke-width="1" stroke-dasharray="3,2"/>
  <text x="30" y="22" text-anchor="middle" font-size="8" fill="{stroke}">V</text>
  <text x="30" y="42" text-anchor="middle" font-size="8" fill="{stroke}">L</text>
</svg>'''


def _mixer_svg(fill: str = "#3498db", stroke: str = "#2471a3") -> str:
    """Mixer - Stream combining (converging triangle)."""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ICON_SIZE} {ICON_SIZE}">
  <polygon points="10,15 10,45 50,30" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
</svg>'''


def _splitter_svg(fill: str = "#e67e22", stroke: str = "#ba6318") -> str:
    """Splitter - Stream dividing (diverging triangle)."""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ICON_SIZE} {ICON_SIZE}">
  <polygon points="10,30 50,15 50,45" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
</svg>'''


def _distillation_svg(fill: str = "#1abc9c", stroke: str = "#16a085") -> str:
    """Distillation column (tall column with trays)."""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ICON_SIZE} {ICON_SIZE}">
  <ellipse cx="30" cy="8" rx="12" ry="4" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
  <rect x="18" y="8" width="24" height="44" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
  <ellipse cx="30" cy="52" rx="12" ry="4" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
  <line x1="20" y1="18" x2="40" y2="18" stroke="{stroke}" stroke-width="1"/>
  <line x1="20" y1="26" x2="40" y2="26" stroke="{stroke}" stroke-width="1"/>
  <line x1="20" y1="34" x2="40" y2="34" stroke="{stroke}" stroke-width="1"/>
  <line x1="20" y1="42" x2="40" y2="42" stroke="{stroke}" stroke-width="1"/>
</svg>'''


def _heater_svg(fill: str = "#e74c3c", stroke: str = "#c0392b") -> str:
    """Heater (box with flame symbol)."""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ICON_SIZE} {ICON_SIZE}">
  <rect x="10" y="15" width="40" height="30" rx="3" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
  <path d="M30 22 Q25 28 30 34 Q35 28 30 22" fill="#f39c12" stroke="#d68910" stroke-width="1"/>
  <path d="M24 26 Q21 30 24 34 Q27 30 24 26" fill="#f39c12" stroke="#d68910" stroke-width="1"/>
  <path d="M36 26 Q33 30 36 34 Q39 30 36 26" fill="#f39c12" stroke="#d68910" stroke-width="1"/>
</svg>'''


def _cooler_svg(fill: str = "#3498db", stroke: str = "#2980b9") -> str:
    """Cooler (box with snowflake/wave symbol)."""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ICON_SIZE} {ICON_SIZE}">
  <rect x="10" y="15" width="40" height="30" rx="3" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
  <path d="M20 25 Q25 22 30 25 Q35 28 40 25" fill="none" stroke="white" stroke-width="2"/>
  <path d="M20 30 Q25 27 30 30 Q35 33 40 30" fill="none" stroke="white" stroke-width="2"/>
  <path d="M20 35 Q25 32 30 35 Q35 38 40 35" fill="none" stroke="white" stroke-width="2"/>
</svg>'''


def _heat_exchanger_svg(fill: str = "#9b59b6", stroke: str = "#7d3c98") -> str:
    """Shell-and-tube heat exchanger."""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ICON_SIZE} {ICON_SIZE}">
  <ellipse cx="10" cy="30" rx="6" ry="15" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
  <ellipse cx="50" cy="30" rx="6" ry="15" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
  <rect x="10" y="15" width="40" height="30" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
  <line x1="15" y1="22" x2="45" y2="22" stroke="{stroke}" stroke-width="1"/>
  <line x1="15" y1="30" x2="45" y2="30" stroke="{stroke}" stroke-width="1"/>
  <line x1="15" y1="38" x2="45" y2="38" stroke="{stroke}" stroke-width="1"/>
</svg>'''


def _fed_batch_svg(fill: str = "#4a90d9", stroke: str = "#2c5aa0") -> str:
    """Fed-batch reactor (tank with feed arrow)."""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ICON_SIZE} {ICON_SIZE}">
  <rect x="10" y="18" width="40" height="32" rx="3" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
  <line x1="30" y1="8" x2="30" y2="18" stroke="{stroke}" stroke-width="2"/>
  <polygon points="26,12 30,18 34,12" fill="{stroke}"/>
  <line x1="22" y1="32" x2="38" y2="32" stroke="{stroke}" stroke-width="2"/>
  <line x1="25" y1="38" x2="35" y2="38" stroke="{stroke}" stroke-width="2"/>
</svg>'''


def _cascade_svg(fill: str = "#16a085", stroke: str = "#0e6655") -> str:
    """Multistage cascade (LLE or extraction)."""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ICON_SIZE} {ICON_SIZE}">
  <rect x="5" y="20" width="14" height="20" rx="2" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>
  <rect x="23" y="20" width="14" height="20" rx="2" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>
  <rect x="41" y="20" width="14" height="20" rx="2" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>
  <line x1="19" y1="30" x2="23" y2="30" stroke="{stroke}" stroke-width="1.5"/>
  <line x1="37" y1="30" x2="41" y2="30" stroke="{stroke}" stroke-width="1.5"/>
</svg>'''


def _bioreactor_svg(fill: str = "#27ae60", stroke: str = "#1e8449") -> str:
    """Bioreactor (vessel with bubbles)."""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ICON_SIZE} {ICON_SIZE}">
  <ellipse cx="30" cy="12" rx="15" ry="6" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
  <rect x="15" y="12" width="30" height="36" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
  <ellipse cx="30" cy="48" rx="15" ry="6" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
  <circle cx="24" cy="35" r="3" fill="white" opacity="0.6"/>
  <circle cx="32" cy="28" r="2" fill="white" opacity="0.6"/>
  <circle cx="36" cy="38" r="2.5" fill="white" opacity="0.6"/>
  <circle cx="26" cy="42" r="2" fill="white" opacity="0.6"/>
</svg>'''


def _centrifuge_svg(fill: str = "#8e44ad", stroke: str = "#6c3483") -> str:
    """Centrifuge (disk with rotation arrows)."""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ICON_SIZE} {ICON_SIZE}">
  <circle cx="30" cy="30" r="20" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
  <circle cx="30" cy="30" r="8" fill="white" stroke="{stroke}" stroke-width="1"/>
  <path d="M30 8 A22 22 0 0 1 52 30" fill="none" stroke="{stroke}" stroke-width="2" marker-end="url(#arrow)"/>
  <defs>
    <marker id="arrow" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="{stroke}"/>
    </marker>
  </defs>
</svg>'''


def _filtration_svg(fill: str = "#2980b9", stroke: str = "#1a5276") -> str:
    """Filtration unit (funnel shape with filter)."""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ICON_SIZE} {ICON_SIZE}">
  <polygon points="10,15 50,15 40,35 20,35" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
  <rect x="20" y="35" width="20" height="15" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
  <line x1="15" y1="25" x2="45" y2="25" stroke="white" stroke-width="2" stroke-dasharray="2,2"/>
</svg>'''


def _chromatography_svg(fill: str = "#d35400", stroke: str = "#a04000") -> str:
    """Chromatography column (column with gradient)."""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ICON_SIZE} {ICON_SIZE}">
  <defs>
    <linearGradient id="chromGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#f39c12"/>
      <stop offset="50%" style="stop-color:#e74c3c"/>
      <stop offset="100%" style="stop-color:#9b59b6"/>
    </linearGradient>
  </defs>
  <rect x="20" y="8" width="20" height="44" rx="3" fill="url(#chromGrad)" stroke="{stroke}" stroke-width="2"/>
  <ellipse cx="30" cy="8" rx="10" ry="3" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
  <ellipse cx="30" cy="52" rx="10" ry="3" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
</svg>'''


def _feed_svg(fill: str = "#2ecc71", stroke: str = "#229954") -> str:
    """Feed stream source (circle with arrow out)."""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ICON_SIZE} {ICON_SIZE}">
  <circle cx="25" cy="30" r="15" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
  <line x1="40" y1="30" x2="55" y2="30" stroke="{stroke}" stroke-width="3"/>
  <polygon points="50,25 55,30 50,35" fill="{stroke}"/>
</svg>'''


def _product_svg(fill: str = "#e74c3c", stroke: str = "#c0392b") -> str:
    """Product stream sink (circle with arrow in)."""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ICON_SIZE} {ICON_SIZE}">
  <circle cx="35" cy="30" r="15" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
  <line x1="5" y1="30" x2="20" y2="30" stroke="{stroke}" stroke-width="3"/>
  <polygon points="15,25 20,30 15,35" fill="{stroke}"/>
</svg>'''


def _generic_svg(fill: str = "#95a5a6", stroke: str = "#7f8c8d") -> str:
    """Generic unit operation (simple box)."""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ICON_SIZE} {ICON_SIZE}">
  <rect x="10" y="15" width="40" height="30" rx="5" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
</svg>'''


# =============================================================================
# Icon Registry
# =============================================================================

# Map operation functions/names to their icon generators
ICON_REGISTRY: dict[str, Callable[[], str]] = {
    # Core reactors
    "CSTR": _cstr_svg,
    "cstr": _cstr_svg,
    "PFR": _pfr_svg,
    "pfr": _pfr_svg,
    "GasPFR": _pfr_svg,
    "FedBatchReactor": _fed_batch_svg,
    "fed_batch": _fed_batch_svg,
    "SemiBatchReactor": _fed_batch_svg,

    # Separations
    "Flash": _flash_svg,
    "flash": _flash_svg,
    "Mixer": _mixer_svg,
    "mixer": _mixer_svg,
    "Splitter": _splitter_svg,
    "splitter": _splitter_svg,

    # Distillation
    "ShortcutColumn": _distillation_svg,
    "DistillationColumn": _distillation_svg,
    "distillation": _distillation_svg,

    # Heat exchange
    "Heater": _heater_svg,
    "heater": _heater_svg,
    "Cooler": _cooler_svg,
    "cooler": _cooler_svg,
    "CounterCurrentHX": _heat_exchanger_svg,
    "CoCurrentHX": _heat_exchanger_svg,
    "heat_exchanger": _heat_exchanger_svg,

    # LLE
    "MultistageCascade": _cascade_svg,
    "cascade": _cascade_svg,
    "DifferentialContactor": _cascade_svg,
    "LLEEquilibrium": _cascade_svg,

    # Bio units
    "ContinuousBioreactor": _bioreactor_svg,
    "FedBatchBioreactor": _bioreactor_svg,
    "bioreactor": _bioreactor_svg,
    "Centrifuge": _centrifuge_svg,
    "DiscStackCentrifuge": _centrifuge_svg,
    "centrifuge": _centrifuge_svg,
    "Ultrafiltration": _filtration_svg,
    "Diafiltration": _filtration_svg,
    "TFF": _filtration_svg,
    "filtration": _filtration_svg,
    "ProteinAChromatography": _chromatography_svg,
    "IonExchangeChromatography": _chromatography_svg,
    "SizeExclusionChromatography": _chromatography_svg,
    "chromatography": _chromatography_svg,

    # Special nodes
    "feed": _feed_svg,
    "Feed": _feed_svg,
    "product": _product_svg,
    "Product": _product_svg,
    "generic": _generic_svg,
}


def get_icon_svg(operation_type: str) -> str:
    """Get SVG string for a given operation type.

    Args:
        operation_type: Name of the operation (e.g., 'CSTR', 'Flash', 'Mixer')

    Returns:
        SVG string for the icon
    """
    icon_fn = ICON_REGISTRY.get(operation_type, _generic_svg)
    return icon_fn()


def get_icon_data_uri(operation_type: str) -> str:
    """Get data URI for a given operation type.

    Args:
        operation_type: Name of the operation (e.g., 'CSTR', 'Flash', 'Mixer')

    Returns:
        Data URI string (data:image/svg+xml;base64,...)
    """
    svg = get_icon_svg(operation_type)
    return _svg_to_data_uri(svg)


def list_available_icons() -> list[str]:
    """List all available icon types."""
    return sorted(set(ICON_REGISTRY.keys()))


def register_icon(name: str, icon_fn: Callable[[], str]) -> None:
    """Register a custom icon.

    Args:
        name: Name for the icon (will be matched against operation names)
        icon_fn: Function that returns an SVG string
    """
    ICON_REGISTRY[name] = icon_fn
