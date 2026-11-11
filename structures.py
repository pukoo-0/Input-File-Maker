"""
structures.py — reading structures and the geometry helpers the input builders need.

Readers: .xyz (plain and extended), POSCAR/CONTCAR (VASP 5 format), .cif (with symmetry
operations). If the ASE library is installed it is tried first for CIF/POSCAR, because it
handles unusual files well; the built-in readers below are the fallback.
"""

import io
import math
import re
import shlex
from dataclasses import dataclass, field

import numpy as np

from elements import normalize_symbol


@dataclass
class Structure:
    symbols: list                     # element symbol per atom
    positions: np.ndarray             # Cartesian, Å, shape (n, 3)
    cell: np.ndarray = None           # 3×3 rows = lattice vectors in Å, or None for a molecule
    fixed: np.ndarray = None          # (n, 3) bool, True = coordinate fixed (from POSCAR selective dynamics)
    source: str = ""
    notes: list = field(default_factory=list)

    def __post_init__(self):
        self.positions = np.asarray(self.positions, dtype=float).reshape(-1, 3)
        if self.fixed is None:
            self.fixed = np.zeros((len(self.symbols), 3), dtype=bool)

    @property
    def n(self):
        return len(self.symbols)

    @property
    def periodic(self):
        return self.cell is not None

    def formula(self):
        counts = {}
        for s in self.symbols:
            counts[s] = counts.get(s, 0) + 1
        return "".join(f"{s}{c if c > 1 else ''}" for s, c in counts.items())

    def fractional(self):
        return np.linalg.solve(self.cell.T, self.positions.T).T

    def cell_parameters(self):
        a, b, c = (np.linalg.norm(v) for v in self.cell)
        ang = lambda u, v: math.degrees(math.acos(np.clip(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))
        return a, b, c, ang(self.cell[1], self.cell[2]), ang(self.cell[0], self.cell[2]), ang(self.cell[0], self.cell[1])

    def volume(self):
        return abs(np.linalg.det(self.cell))


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------

def cell_from_parameters(a, b, c, alpha, beta, gamma):
    al, be, ga = (math.radians(x) for x in (alpha, beta, gamma))
    va = [a, 0.0, 0.0]
    vb = [b * math.cos(ga), b * math.sin(ga), 0.0]
    cx = c * math.cos(be)
    cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
    cz = math.sqrt(max(c ** 2 - cx ** 2 - cy ** 2, 0.0))
    return np.array([va, vb, [cx, cy, cz]])


def kgrid_from_spacing(cell, spacing, slab_axis=None):
    """Monkhorst–Pack grid from a k-point spacing in Å⁻¹ (2π included), like VASP KSPACING."""
    rec = 2 * np.pi * np.linalg.inv(cell).T
    grid = [max(1, int(math.ceil(np.linalg.norm(b) / spacing))) for b in rec]
    if slab_axis is not None:
        grid[slab_axis] = 1
    return grid


def boxed(struct: Structure, vacuum=10.0):
    """Put a molecule in a cubic box with at least `vacuum` Å of space on every side, centred."""
    pos = struct.positions
    size = float(np.max(pos.max(axis=0) - pos.min(axis=0))) + 2 * vacuum
    centre = (pos.max(axis=0) + pos.min(axis=0)) / 2
    new = pos - centre + size / 2
    return Structure(list(struct.symbols), new, np.eye(3) * size, struct.fixed.copy(),
                     struct.source, struct.notes + [f"Placed in a {size:.2f} Å cubic box."])


def vacuum_axis(struct: Structure, min_gap=6.0):
    """Index of a cell axis with a large empty gap (a slab's vacuum direction), or None."""
    if not struct.periodic:
        return None
    frac = struct.fractional() % 1.0
    best, best_gap = None, 0
    for ax in range(3):
        f = np.sort(frac[:, ax])
        gaps = np.diff(np.concatenate([f, [f[0] + 1]]))
        gap = gaps.max() * np.linalg.norm(struct.cell[ax])
        if gap > best_gap:
            best, best_gap = ax, gap
    return best if best_gap >= min_gap else None


def band_path(cell, npoints_per_segment=30):
    """
    High-symmetry path for the given cell, using ASE if it is installed.
    Returns a list of (label, (k1, k2, k3) in crystal coordinates, weight) or raises RuntimeError.
    """
    try:
        from ase.cell import Cell
    except Exception as e:
        raise RuntimeError("Automatic band paths need the ASE library (pip install ase).") from e
    bp = Cell(np.asarray(cell)).bandpath()
    special = bp.special_points
    out = []
    for seg in bp.path.split(","):
        labels = re.findall(r"[A-Z][a-z]*[0-9]*", seg)
        for i, lab in enumerate(labels):
            last = i == len(labels) - 1
            out.append((lab, tuple(float(x) for x in special[lab]), 1 if last else npoints_per_segment))
    return out


# ---------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------

def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def read_xyz(text: str, name="") -> Structure:
    lines = [l for l in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    try:
        n = int(lines[0].split()[0])
    except Exception:
        raise ValueError("An .xyz file must start with the number of atoms.")
    comment = lines[1] if len(lines) > 1 else ""
    symbols, pos = [], []
    for l in lines[2:2 + n]:
        p = l.split()
        if len(p) < 4:
            raise ValueError(f"Couldn’t read this atom line: “{l.strip()}”")
        symbols.append(normalize_symbol(p[0]))
        pos.append([float(p[1]), float(p[2]), float(p[3])])
    if len(symbols) != n:
        raise ValueError(f"The file says {n} atoms but contains {len(symbols)}.")
    cell = None
    m = re.search(r'Lattice\s*=\s*"([^"]+)"', comment)
    if m:
        vals = [float(v) for v in m.group(1).split()]
        if len(vals) == 9:
            cell = np.array(vals).reshape(3, 3)
    return Structure(symbols, np.array(pos), cell, source=name)


def read_poscar(text: str, name="") -> Structure:
    lines = [l.strip() for l in text.splitlines()]
    lines = [l for l in lines if l != ""] if len(lines) > 8 else lines
    scale = float(lines[1].split()[0])
    lat = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)])
    factor = (abs(scale) / abs(np.linalg.det(lat))) ** (1 / 3) if scale < 0 else scale
    lat *= factor
    idx = 5
    tokens = lines[idx].split()
    if not tokens[0].isalpha():
        raise ValueError("This POSCAR has no element line (old VASP 4 format). Add a line with the element "
                         "symbols above the atom counts.")
    species = [normalize_symbol(t.split("/")[0].split("_")[0]) for t in tokens]
    idx += 1
    counts = [int(x) for x in lines[idx].split()]
    idx += 1
    selective = lines[idx][:1].lower() == "s"
    if selective:
        idx += 1
    direct = lines[idx][:1].lower() == "d"
    idx += 1
    symbols, pos, fixed = [], [], []
    for sp, cnt in zip(species, counts):
        for _ in range(cnt):
            p = lines[idx].split()
            idx += 1
            v = np.array([float(x) for x in p[:3]])
            pos.append(v @ lat if direct else v * factor)
            symbols.append(sp)
            fixed.append([f.upper().startswith("F") for f in p[3:6]] if selective and len(p) >= 6 else [False] * 3)
    return Structure(symbols, np.array(pos), lat, np.array(fixed, dtype=bool), source=name)


_NUM = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")


def _cif_float(v):
    m = _NUM.match(str(v).strip())
    if not m:
        raise ValueError(f"Not a number in the CIF: “{v}”")
    return float(m.group(1))


def _cif_blocks(text):
    """Very small CIF tokenizer: returns a list of blocks, each {'tags': {}, 'loops': [ {tag: [values]} ]}."""
    blocks, cur = [], None
    lines = text.splitlines()
    i = 0

    def new_block():
        b = {"tags": {}, "loops": []}
        blocks.append(b)
        return b

    def tokens_of(line):
        try:
            return shlex.split(line, posix=True)
        except ValueError:
            return line.split()

    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        if line.lower().startswith("data_"):
            cur = new_block()
            i += 1
            continue
        if cur is None:
            cur = new_block()
        if line.lower().startswith("loop_"):
            i += 1
            tags = []
            while i < len(lines) and lines[i].strip().startswith("_"):
                tags.append(lines[i].strip().split()[0].lower())
                i += 1
            values = []
            while i < len(lines):
                l = lines[i].strip()
                if not l or l.startswith("#"):
                    i += 1
                    continue
                if l.startswith("_") or l.lower().startswith(("loop_", "data_")):
                    break
                if l.startswith(";"):
                    txt = [l[1:]]
                    i += 1
                    while i < len(lines) and not lines[i].startswith(";"):
                        txt.append(lines[i])
                        i += 1
                    i += 1
                    values.append("\n".join(txt))
                    continue
                values += tokens_of(l)
                i += 1
            if tags:
                n = len(tags)
                cur["loops"].append({t: values[k::n] for k, t in enumerate(tags)})
            continue
        if line.startswith("_"):
            parts = tokens_of(line)
            tag = parts[0].lower()
            if len(parts) > 1:
                cur["tags"][tag] = " ".join(parts[1:])
                i += 1
            else:
                i += 1
                if i < len(lines) and lines[i].startswith(";"):
                    txt = [lines[i][1:]]
                    i += 1
                    while i < len(lines) and not lines[i].startswith(";"):
                        txt.append(lines[i])
                        i += 1
                    i += 1
                    cur["tags"][tag] = "\n".join(txt)
                elif i < len(lines):
                    cur["tags"][tag] = " ".join(tokens_of(lines[i]))
                    i += 1
            continue
        i += 1
    return blocks


_SYMOP_OK = re.compile(r"^[xyzXYZ0-9+\-*/. ]+$")


def _symop(text):
    """'x+1/2, -y, z' -> (W 3×3, t 3)."""
    parts = [p.strip() for p in text.replace("'", "").replace('"', "").split(",")]
    if len(parts) != 3 or not all(_SYMOP_OK.match(p) for p in parts):
        raise ValueError(f"Couldn’t understand the symmetry operation “{text}”.")
    W, t = np.zeros((3, 3)), np.zeros(3)
    for r, expr in enumerate(parts):
        e = expr.lower()
        f = lambda x, y, z: eval(e, {"__builtins__": {}}, {"x": x, "y": y, "z": z})
        t[r] = f(0.0, 0.0, 0.0)
        W[r] = [f(1.0, 0, 0) - t[r], f(0, 1.0, 0) - t[r], f(0, 0, 1.0) - t[r]]
    return W, t


def read_cif(text: str, name="") -> Structure:
    blocks = _cif_blocks(text)
    for b in blocks:
        site_loop = next((L for L in b["loops"] if "_atom_site_fract_x" in L), None)
        if site_loop is None:
            continue
        tg = b["tags"]
        try:
            params = [_cif_float(tg[k]) for k in ("_cell_length_a", "_cell_length_b", "_cell_length_c",
                                                   "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma")]
        except KeyError as e:
            raise ValueError(f"The CIF is missing {e.args[0]}.")
        cell = cell_from_parameters(*params)

        ops = None
        for L in b["loops"]:
            for key in ("_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz"):
                if key in L:
                    ops = [_symop(s) for s in L[key]]
        notes = []
        if ops is None:
            ops = [(np.eye(3), np.zeros(3))]
            if any(k in tg for k in ("_symmetry_space_group_name_h-m", "_space_group_name_h-m_alt")):
                hm = tg.get("_symmetry_space_group_name_h-m", tg.get("_space_group_name_h-m_alt", ""))
                if hm.replace(" ", "").upper() not in ("P1", "'P1'"):
                    notes.append(f"The CIF names space group {hm} but lists no symmetry operations, so only the "
                                 "listed atoms were used. Check the atom count.")

        n_sites = len(site_loop["_atom_site_fract_x"])
        if "_atom_site_type_symbol" in site_loop:
            elems = [normalize_symbol(s) for s in site_loop["_atom_site_type_symbol"]]
        else:
            elems = [normalize_symbol(s) for s in site_loop["_atom_site_label"]]
        occ = site_loop.get("_atom_site_occupancy", ["1"] * n_sites)
        if any(_cif_float(o) < 0.999 for o in occ if o not in ("?", ".")):
            notes.append("Some sites are partially occupied. All of them were kept; edit the structure if "
                         "that’s not what you want.")

        frac_all, sym_all = [], []
        for k in range(n_sites):
            r = np.array([_cif_float(site_loop[c][k]) for c in
                          ("_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z")])
            for W, t in ops:
                new = (W @ r + t) % 1.0
                dup = False
                for old in frac_all:
                    d = new - old
                    d -= np.round(d)
                    if np.linalg.norm(d @ cell) < 1e-2:
                        dup = True
                        break
                if not dup:
                    frac_all.append(new)
                    sym_all.append(elems[k])
        frac_all = np.array(frac_all)
        frac_all[np.isclose(frac_all, 1.0, atol=1e-6)] = 0.0
        return Structure(sym_all, frac_all @ cell, cell, source=name, notes=notes)
    raise ValueError("No atom positions (_atom_site_fract_x …) found in this CIF.")


def _read_with_ase(raw: bytes, fmt: str, name: str):
    from ase.io import read
    atoms = read(io.StringIO(_decode(raw)), format=fmt)
    cell = np.array(atoms.cell) if atoms.pbc.any() and atoms.cell.rank == 3 else None
    fixed = None
    try:
        from ase.constraints import FixAtoms
        fixed = np.zeros((len(atoms), 3), dtype=bool)
        for c in atoms.constraints:
            if isinstance(c, FixAtoms):
                fixed[c.index] = True
    except Exception:
        fixed = None
    return Structure(atoms.get_chemical_symbols(), atoms.positions, cell, fixed, source=name,
                     notes=["Read with ASE."])


def read_structure(raw: bytes, filename: str) -> Structure:
    """Read .xyz, .cif or POSCAR/CONTCAR/.vasp/.poscar."""
    name = filename.lower()
    if name.endswith(".xyz") or name.endswith(".extxyz"):
        return read_xyz(_decode(raw), filename)
    if name.endswith(".cif"):
        fmt, builtin = "cif", read_cif
    elif "poscar" in name or "contcar" in name or name.endswith(".vasp"):
        fmt, builtin = "vasp", read_poscar
    else:
        raise ValueError("Use an .xyz, .cif, POSCAR, CONTCAR or .vasp file.")
    # CIF: ASE first (handles unusual files well). POSCAR: built-in first (keeps partial selective dynamics).
    readers = [lambda: _read_with_ase(raw, fmt, filename), lambda: builtin(_decode(raw), filename)]
    if fmt == "vasp":
        readers.reverse()
    last = None
    for read in readers:
        try:
            return read()
        except Exception as e:
            last = e
    raise ValueError(str(last))


# ---------------------------------------------------------------------
# Structure tools: symmetry, primitive cell, supercell, slab, vacuum
# ---------------------------------------------------------------------

def _spglib_cell(struct):
    from elements import Z
    return (np.asarray(struct.cell), struct.fractional() % 1.0, [Z[s] for s in struct.symbols])


def symmetry_info(struct, symprec=1e-3):
    """Space group as text, e.g. 'Pm-3m (221)'. Needs spglib; raises RuntimeError without it."""
    try:
        import spglib
    except Exception as e:
        raise RuntimeError("Symmetry tools need the spglib library (pip install spglib).") from e
    sg = spglib.get_spacegroup(_spglib_cell(struct), symprec=symprec)
    if sg is None:
        raise RuntimeError("spglib couldn’t determine the space group. Try a looser tolerance.")
    return sg


def primitive_cell(struct, symprec=1e-3):
    """Standardised primitive cell (spglib). Fewer atoms means a much cheaper calculation."""
    try:
        import spglib
    except Exception as e:
        raise RuntimeError("Symmetry tools need the spglib library (pip install spglib).") from e
    from elements import SYMBOLS
    res = spglib.standardize_cell(_spglib_cell(struct), to_primitive=True, no_idealize=False, symprec=symprec)
    if res is None:
        raise RuntimeError("spglib couldn’t find a primitive cell. Try a looser tolerance.")
    lat, frac, nums = res
    lat = np.asarray(lat)
    return Structure([SYMBOLS[int(n) - 1] for n in nums], np.asarray(frac) @ lat, lat,
                     source=struct.source, notes=[f"Primitive cell ({len(nums)} atoms, was {struct.n})."])


def supercell(struct, n1, n2, n3):
    """Repeat the cell n1 × n2 × n3 times."""
    reps = [(i, j, k) for i in range(n1) for j in range(n2) for k in range(n3)]
    syms, pos, fixed = [], [], []
    for i, j, k in reps:
        shift = i * struct.cell[0] + j * struct.cell[1] + k * struct.cell[2]
        syms += list(struct.symbols)
        pos.append(struct.positions + shift)
        fixed.append(struct.fixed)
    cell = np.array([struct.cell[0] * n1, struct.cell[1] * n2, struct.cell[2] * n3])
    return Structure(syms, np.vstack(pos), cell, np.vstack(fixed), struct.source,
                     notes=[f"{n1}×{n2}×{n3} supercell ({len(syms)} atoms)."])


def add_vacuum(struct, axis, vacuum):
    """Make the cell longer along one axis so there is `vacuum` Å of empty space, and centre the atoms."""
    frac = struct.fractional()
    vec = struct.cell[axis]
    length = np.linalg.norm(vec)
    unit = vec / length
    proj = struct.positions @ unit
    thickness = proj.max() - proj.min()
    new_len = thickness + vacuum
    cell = struct.cell.copy()
    cell[axis] = unit * new_len
    frac[:, axis] = (proj - proj.min()) / new_len + (vacuum / 2) / new_len
    pos = frac @ cell
    return Structure(list(struct.symbols), pos, cell, struct.fixed.copy(), struct.source,
                     notes=[f"Axis {axis + 1} set to {new_len:.3f} Å ({vacuum:.1f} Å of vacuum)."])


def make_slab(struct, miller, layers, vacuum):
    """Cut a slab with ASE: Miller indices (h, k, l), number of layers, vacuum in Å."""
    try:
        from ase import Atoms
        from ase.build import surface
    except Exception as e:
        raise RuntimeError("Building slabs needs the ASE library (pip install ase).") from e
    atoms = Atoms(struct.symbols, positions=struct.positions, cell=struct.cell, pbc=True)
    slab = surface(atoms, tuple(int(x) for x in miller), int(layers), vacuum=float(vacuum) / 2)
    return Structure(slab.get_chemical_symbols(), slab.positions, np.array(slab.cell), source=struct.source,
                     notes=[f"({''.join(str(int(x)) for x in miller)}) slab, {layers} layers, {vacuum:.1f} Å vacuum "
                            f"({len(slab)} atoms)."])


def viewer_html(struct, highlight=(), repeat=(1, 1, 1), height=420, labels=False):
    """HTML for an interactive 3D view with 3Dmol.js. Highlighted atoms (e.g. fixed) are drawn larger and red."""
    import json
    s = struct
    show = s
    if s.periodic and tuple(repeat) != (1, 1, 1):
        show = supercell(s, *repeat)
    xyz = "\n".join([str(show.n), "structure"] +
                    [f"{el} {x:.6f} {y:.6f} {z:.6f}" for el, (x, y, z) in zip(show.symbols, show.positions)])
    n0 = s.n
    hl = sorted({int(i) + k * n0 for i in highlight for k in range(max(1, show.n // max(n0, 1)))})
    edges = []
    if s.periodic:
        o = np.zeros(3)
        a, b, c = s.cell
        corners = [o, a, b, c, a + b, a + c, b + c, a + b + c]
        pairs = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)]
        edges = [[list(map(float, corners[i])), list(map(float, corners[j]))] for i, j in pairs]
    label_js = ""
    if labels:
        label_js = ("m.selectedAtoms({}).forEach(function(at){v.addLabel(String(at.index),{position:at,"
                    "fontSize:10,fontColor:'black',backgroundOpacity:0.0,inFront:true});});")
    return f"""
<div id="mol" style="width:100%;height:{height}px;position:relative;border:1px solid #ddd;border-radius:6px"></div>
<script>
function start() {{
  var v = $3Dmol.createViewer(document.getElementById("mol"), {{backgroundColor: "white"}});
  var m = v.addModel({json.dumps(xyz)}, "xyz");
  v.setStyle({{}}, {{stick: {{radius: 0.12}}, sphere: {{scale: 0.28}}}});
  var hl = {json.dumps(hl)};
  if (hl.length) v.setStyle({{index: hl}}, {{stick: {{radius: 0.12}}, sphere: {{scale: 0.4, color: "red"}}}});
  var edges = {json.dumps(edges)};
  edges.forEach(function(e) {{
    v.addCylinder({{start: {{x: e[0][0], y: e[0][1], z: e[0][2]}}, end: {{x: e[1][0], y: e[1][1], z: e[1][2]}},
                   radius: 0.03, color: "gray", fromCap: 1, toCap: 1}});
  }});
  {label_js}
  v.zoomTo(); v.render();
}}
function load(src, next) {{
  var s = document.createElement("script"); s.src = src; s.onload = start; s.onerror = next;
  document.head.appendChild(s);
}}
load("https://3Dmol.org/build/3Dmol-min.js", function() {{
  load("https://cdn.jsdelivr.net/npm/3dmol@2.4.0/build/3Dmol-min.js", function() {{
    document.getElementById("mol").innerText = "The 3D viewer couldn’t load (no internet connection?).";
  }});
}});
</script>
"""
