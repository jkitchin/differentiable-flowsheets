"""Write the plugin logos in images/plugins/ (``python scripts/make_plugin_logos.py``).

Each plugin gets the difflow badge -- a 64x64 rounded square with a white
glyph -- in its own gradient, as an icon and as a wordmark logo matching
images/difflow-logo.svg. Edit the table below and rerun; the SVGs are
generated, not hand-edited.
"""

import math
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "images" / "plugins"
FONT = "'Helvetica Neue', Helvetica, Arial, sans-serif"
W = 'fill="none" stroke="#ffffff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"'


def helix():
    """bio: a double helix."""
    def strand(sign):
        pts = []
        for i in range(41):
            y = 10 + i
            x = 32 + sign * 12 * math.sin((y - 10) / 44 * 2 * math.pi * 1.25)
            pts.append(f"{x:.1f},{y:.1f}")
        return " ".join(pts)

    rungs = []
    for y in (14, 20, 32, 38, 50):
        dx = 12 * math.sin((y - 10) / 44 * 2 * math.pi * 1.25)
        if abs(dx) > 4:
            rungs.append(f'<line x1="{32 - dx + 2.5 * (1 if dx > 0 else -1):.1f}" y1="{y}" '
                         f'x2="{32 + dx - 2.5 * (1 if dx > 0 else -1):.1f}" y2="{y}" stroke-width="2.2" opacity="0.8"/>')
    return (f'<g {W}><polyline points="{strand(1)}"/><polyline points="{strand(-1)}"/>'
            + "".join(rungs) + "</g>")


def settler():
    """ree: a two-phase settler, organic over aqueous, with drops crossing."""
    return (f'<rect x="13" y="20" width="38" height="30" rx="5" fill="#ffffff" fill-opacity="0.18"/>'
            f'<path d="M13 36 H51 V45 a5 5 0 0 1 -5 5 H18 a5 5 0 0 1 -5 -5 Z" fill="#ffffff" fill-opacity="0.75"/>'
            f'<rect x="13" y="20" width="38" height="30" rx="5" {W}/>'
            f'<line x1="13" y1="36" x2="51" y2="36" stroke="#ffffff" stroke-width="2" stroke-dasharray="3 2.5"/>'
            '<g fill="#ffffff"><circle cx="22" cy="29" r="2.4"/><circle cx="31" cy="25" r="1.8"/>'
            '<circle cx="40" cy="30" r="2.2"/></g>'
            f'<path d="M32 8 V17" {W}/><path d="M28.5 13.5 L32 17.5 L35.5 13.5" {W}/>')


def co2():
    """cc: a CO2 molecule dropping into a capture vessel."""
    return (f'<g {W}>'
            '<line x1="22" y1="16" x2="27" y2="16"/><line x1="22" y1="22" x2="27" y2="22"/>'
            '<line x1="37" y1="16" x2="42" y2="16"/><line x1="37" y1="22" x2="42" y2="22"/>'
            '<path d="M32 30 V38"/><path d="M28.5 34.5 L32 38.5 L35.5 34.5"/>'
            '<path d="M14 38 V48 a6 6 0 0 0 6 6 H44 a6 6 0 0 0 6 -6 V38"/></g>'
            '<path d="M14 46 H50 V48 a6 6 0 0 1 -6 6 H20 a6 6 0 0 1 -6 -6 Z" fill="#ffffff" fill-opacity="0.6"/>'
            '<g fill="#ffffff"><circle cx="15" cy="19" r="6.5"/><circle cx="32" cy="19" r="5.5"/>'
            '<circle cx="49" cy="19" r="6.5"/></g>'
            '<circle cx="32" cy="19" r="2.2" fill="{accent}"/>')


def pipeline():
    """gas: a pipeline network with a compressor station."""
    nodes = [(13, 18), (13, 46), (51, 18), (51, 46)]
    edges = "".join(f'<line x1="{x}" y1="{y}" x2="32" y2="32"/>' for x, y in nodes)
    return (f'<g {W}>{edges}<line x1="13" y1="18" x2="13" y2="46"/>'
            '<line x1="51" y1="18" x2="51" y2="46"/></g>'
            '<polygon points="25,24 39,28 39,36 25,40" fill="#ffffff" stroke="#ffffff" '
            'stroke-width="2" stroke-linejoin="round"/>'
            '<g fill="#ffffff">' + "".join(f'<circle cx="{x}" cy="{y}" r="4.6"/>' for x, y in nodes)
            + "</g>" + "".join(f'<circle cx="{x}" cy="{y}" r="1.8" fill="{{accent}}"/>' for x, y in nodes))


def bolt():
    """power: a lightning bolt over a bus bar."""
    return ('<polygon points="37,7 17,36 30,36 25,57 47,26 34,26 40,7" fill="#ffffff" '
            'stroke="#ffffff" stroke-width="2" stroke-linejoin="round"/>')


def column():
    """refinery: a distillation column with trays, a feed and side draws."""
    trays = "".join(f'<line x1="26" y1="{y}" x2="38" y2="{y}" stroke-width="2" opacity="0.85"/>'
                    for y in (17, 24, 31, 38, 45, 52))
    draws = "".join(f'<path d="M40 {y} H52"/><path d="M48.5 {y - 3.5} L52.5 {y} L48.5 {y + 3.5}"/>'
                    for y in (14, 28, 42))
    return (f'<g {W}><rect x="23" y="7" width="18" height="51" rx="9"/>{trays}{draws}'
            '<path d="M9 35 H22"/><path d="M18.5 31.5 L22.5 35 L18.5 38.5"/></g>')


PLUGINS = {
    # key: (subtitle, gradient stops, accent for the inner details, glyph)
    "bio": ("BIOMANUFACTURING", ("#059669", "#10b981", "#a3e635"), "#059669", helix),
    "ree": ("RARE EARTH SEPARATIONS", ("#9d174d", "#db2777", "#f9a8d4"), "#db2777", settler),
    "cc": ("CARBON CAPTURE", ("#0f766e", "#14b8a6", "#5eead4"), "#0f766e", co2),
    "gas": ("GAS TRANSMISSION NETWORKS", ("#c2410c", "#f97316", "#fbbf24"), "#ea580c", pipeline),
    "power": ("ELECTRICAL GRIDS", ("#312e81", "#6d28d9", "#a78bfa"), "#6d28d9", bolt),
    "refinery": ("PETROLEUM REFINING", ("#1e293b", "#475569", "#d97706"), "#475569", column),
}


def gradient(gid, stops):
    s = "".join(f'<stop offset="{o}" stop-color="{c}"/>' for o, c in zip((0, 0.55, 1), stops))
    return (f'<linearGradient id="{gid}" x1="0" y1="0" x2="64" y2="64" '
            f'gradientUnits="userSpaceOnUse">{s}</linearGradient>')


def badge(key):
    _, stops, accent, glyph = PLUGINS[key]
    return (f'<rect x="0" y="0" width="64" height="64" rx="15" fill="url(#bg-{key})"/>'
            + glyph().replace("{accent}", accent))


def icon(key):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="64" height="64" '
            f'role="img" aria-label="difflow_{key} icon"><defs>{gradient("bg-" + key, PLUGINS[key][1])}'
            f'</defs>{badge(key)}</svg>\n')


def logo(key):
    sub, stops, accent, _ = PLUGINS[key]
    width = 112 + 23 * len("difflow " + key)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} 96" width="{width}" '
            f'height="96" role="img" aria-label="difflow_{key} -- {sub.title()}"><defs>'
            f'{gradient("bg-" + key, stops)}'
            '<linearGradient id="ink" x1="96" y1="0" x2="300" y2="0" gradientUnits="userSpaceOnUse">'
            '<stop offset="0" stop-color="#4f46e5"/><stop offset="1" stop-color="#0ea5e9"/>'
            '</linearGradient></defs>'
            f'<g transform="translate(16,16)">{badge(key)}</g>'
            f'<text x="98" y="52" font-family="{FONT}" font-size="40" font-weight="700" '
            f'letter-spacing="-1"><tspan fill="url(#ink)">difflow</tspan>'
            f'<tspan fill="{accent}" font-weight="400" dx="8">{key}</tspan></text>'
            f'<text x="100" y="74" font-family="{FONT}" font-size="12.5" font-weight="600" '
            f'letter-spacing="2.4" fill="#64748b">{sub}</text></svg>\n')


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for key in PLUGINS:
        (OUT / f"difflow-{key}-icon.svg").write_text(icon(key))
        (OUT / f"difflow-{key}-logo.svg").write_text(logo(key))
        print(f"wrote difflow-{key}-icon.svg, difflow-{key}-logo.svg")
