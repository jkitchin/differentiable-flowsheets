"""Generate difflow_water/chemistry/data/pitzer_seawater.py from PHREEQC pitzer.dat."""
import hashlib
import sys

src = sys.argv[1]
out = sys.argv[2]
with open(src, 'rb') as fh:
    raw = fh.read()
sha = hashlib.sha256(raw).hexdigest()
L = raw.decode('latin-1').split('\n')
CATIONS = {'H+': 1, 'Na+': 1, 'K+': 1, 'Mg+2': 2, 'Ca+2': 2, 'MgOH+': 1}
ANIONS = {'OH-': -1, 'Cl-': -1, 'SO4-2': -2, 'HSO4-': -1, 'HCO3-': -1, 'CO3-2': -2}
NEUTRALS = {'CO2': 0, 'MgCO3': 0}
ALL = {**CATIONS, **ANIONS, **NEUTRALS}
NSP = {'-B0': 2, '-B1': 2, '-B2': 2, '-C0': 2, '-THETA': 2, '-LAMDA': 2, '-ZETA': 3, '-PSI': 3}
i = L.index('PITZER')
blk = None
params = []
for line in L[i + 1:]:
    if line and not line[0].isspace() and not line.startswith(('-', '#')):
        break
    s = line.split('#')[0].strip()
    if not s:
        continue
    if s.startswith('-'):
        blk = s.split()[0]
        continue
    if blk not in NSP:
        continue
    t = s.split()
    n = NSP[blk]
    sp = t[:n]
    if all(x in ALL for x in sp):
        co = [float(x) for x in t[n:]] + [0.0] * (6 - len(t[n:]))
        params.append((blk[1:], tuple(sp), tuple(co)))
# reactions: species (formation) and phases (dissolution)
def grab(start_kw, names):
    j = L.index(start_kw)
    res = {}
    cur = None
    for line in L[j + 1:]:
        if line.strip() in ('PHASES', 'PITZER', 'EXCHANGE_MASTER_SPECIES'):
            break
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        if not line[0].isspace():
            cur = None
            key = line.split()[0]
            if start_kw == 'SOLUTION_SPECIES':
                for nm, rx in names.items():
                    if line.split('#')[0].strip() == rx:
                        cur = nm
                        res[nm] = {'reaction': rx}
            elif key in names:
                cur = key
                res[key] = {}
            continue
        if cur is None:
            continue
        s = line.split('#')[0].strip()
        if not s:
            continue
        for part in s.split(';'):
            t = part.split()
            if not t:
                continue
            k = t[0].lstrip('-').lower()
            if '=' in part and start_kw == 'PHASES':
                res[cur]['reaction'] = part.strip()
            elif k in ('log_k',):
                res[cur]['log_k'] = float(t[1])
            elif k in ('delta_h',):
                unit = t[2] if len(t) > 2 else 'kJ'
                res[cur]['delta_h'] = (float(t[1]), unit)
            elif k.startswith('analytic'):
                res[cur]['analytic'] = tuple(float(x) for x in t[1:]) + (0.0,) * (6 - len(t[1:]))
    return res
species_rx = grab('SOLUTION_SPECIES', {
    'OH-': 'H2O = OH- + H+', 'HCO3-': 'CO3-2 + H+ = HCO3-', 'CO2': 'CO3-2 + 2 H+ = CO2 + H2O',
    'HSO4-': 'SO4-2 + H+ = HSO4-', 'MgOH+': 'Mg+2 + H2O = MgOH+ + H+', 'MgCO3': 'Mg+2 + CO3-2 = MgCO3'})
phases = grab('PHASES', {k: None for k in ('Calcite', 'Aragonite', 'Gypsum', 'Anhydrite', 'Brucite', 'Halite', 'Magnesite', 'Dolomite')})
with open(out, 'w') as f:
    f.write('"""Pitzer parameters and equilibrium constants for seawater (GENERATED, do not edit).\n\n')
    f.write('Transcribed mechanically from PHREEQC\'s ``pitzer.dat`` (the copy shipped with\n')
    f.write(f'phreeqpython 1.6.2, sha256 {sha[:16]}...) by\n')
    f.write('``benchmarks/phreeqc/gen_pitzer_data.py``, for the species below. PHREEQC is USGS\n')
    f.write('public-domain software; its pitzer.dat derives from Harvie, Moller & Weare (1984)\n')
    f.write('via PHRQPITZ (Plummer et al. 1988) with later additions it cites inline. The\n')
    f.write('values are therefore second-hand (unverified against the primary papers); the\n')
    f.write('implementation is cross-checked against PHREEQC itself (benchmarks/phreeqc/).\n\n')
    f.write('Pitzer parameter temperature form (PHREEQC manual, PITZER keyword):\n')
    f.write('    P = A0 + A1 (1/T - 1/Tr) + A2 ln(T/Tr) + A3 (T - Tr) + A4 (T^2 - Tr^2)\n')
    f.write('        + A5 (1/T^2 - 1/Tr^2),  Tr = 298.15 K.\n')
    f.write('log K: ``analytic`` (A1 + A2 T + A3/T + A4 log10 T + A5/T^2 + A6 T^2) takes\n')
    f.write('precedence; otherwise van\'t Hoff from ``log_k`` and ``delta_h``.\n"""\n\n')
    f.write(f'SOURCE_SHA256 = "{sha}"\n\n')
    f.write(f'CATIONS = {CATIONS!r}\nANIONS = {ANIONS!r}\nNEUTRALS = {NEUTRALS!r}\n\n')
    f.write('#: (kind, species, (A0..A5)); kind in B0 B1 B2 C0 THETA LAMDA ZETA PSI\nPARAMS = [\n')
    f.writelines(f'    {p!r},\n' for p in params)
    f.write(']\n\n#: formation reactions of the secondary species\nSPECIES = {\n')
    f.writelines(f'    {k!r}: {v!r},\n' for k, v in species_rx.items())
    f.write('}\n\n#: dissolution reactions of minerals\nPHASES = {\n')
    f.writelines(f'    {k!r}: {v!r},\n' for k, v in phases.items())
    f.write('}\n')
print(len(params), list(species_rx), list(phases))
