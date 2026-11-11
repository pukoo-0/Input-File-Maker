import os
import subprocess
import sys
import tempfile

import numpy as np

import qe_input as Q
import structures as S
from data import CSPBBR3_CIF, NI_SLAB_POSCAR

BULK = S.read_cif(CSPBBR3_CIF)
SLAB = S.read_poscar(NI_SLAB_POSCAR)


def build(c, st=BULK, labels=None, species=None, final=None):
    labels = labels or list(st.symbols)
    species = species or Q.species_table(st, labels)
    return Q.build_qe(c, st, species, labels, st.fixed, final)


def test_fortran_values():
    assert Q.fortran(True) == ".true." and Q.fortran(1e-8) == "1d-8" and Q.fortran("a'b") == "'a''b'"
    assert Q.fortran(50.0) == "50.0" and Q.fortran(3) == "3"


def test_scf_input_structure():
    files, errors, _, commands = build({"prefix": "t"})
    assert not errors
    t = files["t.scf.in"]
    for part in ("&CONTROL", "&SYSTEM", "&ELECTRONS", "ATOMIC_SPECIES", "CELL_PARAMETERS angstrom",
                 "ATOMIC_POSITIONS crystal", "K_POINTS automatic", "nat = 5", "ntyp = 3"):
        assert part in t
    assert commands == ["$RUN pw.x -in t.scf.in > t.scf.out"]


def test_cutoff_and_spin_checks():
    _, errors, _, _ = build({"ecutwfc": 50.0, "ecutrho": 150.0})
    assert any("ecutrho" in e for e in errors)
    _, errors, _, _ = build({"spin": "collinear"})
    assert any("starting magnetisation" in e for e in errors)


def test_relax_writes_fixed_flags_and_hubbard():
    labels = ["Ni1", "Ni2", "Ni1", "Ni2", "O"]
    sp = Q.species_table(SLAB, labels)
    sp[0].update({"Start. magnetisation": 0.5, "Hubbard U (eV)": 6.0})
    sp[1].update({"Start. magnetisation": -0.5})
    files, errors, _, _ = build({"calculation": "relax", "spin": "collinear", "kgrid": (6, 6, 1), "prefix": "n"},
                                SLAB, labels, sp)
    t = files["n.relax.in"]
    assert not errors
    assert "  0 0 0" in t and "1 1 0" in t
    assert "HUBBARD ortho-atomic" in t and "U Ni1-3d 6.0" in t
    assert "starting_magnetization(2) = -0.5" in t


def test_wrong_ion_dynamics_is_an_error():
    _, errors, _, _ = build({"calculation": "relax", "ion_dynamics": "verlet"})
    assert errors


def test_workflows_make_all_files():
    path = [("G", (0, 0, 0), 20), ("X", (0, 0.5, 0), 1)]
    expected = {"bands": ["p.scf.in", "p.bands.in", "p.bands_pp.in"],
                "dos": ["p.scf.in", "p.nscf.in", "p.dos.in", "p.projwfc.in"],
                "pp": ["p.scf.in", "p.pp.in"],
                "phonon": ["p.scf.in", "p.ph.in", "p.q2r.in", "p.matdyn.in", "p.matdyn_dos.in"],
                "md": ["p.md.in"]}
    for wf, names in expected.items():
        files, errors, _, _ = build({"workflow": wf, "prefix": "p", "path": path, "nbnd": 30})
        assert not errors, (wf, errors)
        for n in names:
            assert n in files, (wf, n)
    files, _, _, _ = build({"workflow": "bands", "prefix": "p", "path": path})
    assert "K_POINTS crystal_b" in files["p.bands.in"]


def test_md_time_step_conversion():
    files, _, warnings, _ = build({"workflow": "md", "prefix": "m", "md_dt_fs": 1.0, "ion_temperature": "svr"})
    t = files["m.md.in"]
    dt = float(next(l for l in t.splitlines() if l.strip().startswith("dt =")).split("=")[1])
    assert abs(dt * Q.RY_AU_FS - 1.0) < 1e-6
    assert "ion_temperature = 'svr'" in t and any("nosym" in w for w in warnings)


def test_neb_input():
    moved = S.Structure(BULK.symbols, BULK.positions + np.array([[0.2, 0, 0]] + [[0, 0, 0]] * 4), BULK.cell)
    files, errors, _, _ = build({"workflow": "neb", "prefix": "r"}, final=moved)
    t = files["r.neb.in"]
    assert not errors and t.startswith("BEGIN") and "FIRST_IMAGE" in t and "LAST_IMAGE" in t


def test_convergence_series_and_collector():
    files, errors, _, commands = build({"workflow": "convergence", "prefix": "c", "conv_ecut_start": 30.0,
                                        "conv_ecut_stop": 50.0, "conv_ecut_step": 10.0,
                                        "conv_kspacings": (0.4, 0.3)})
    assert not errors
    assert {"c.ecut30.in", "c.ecut40.in", "c.ecut50.in", "collect_convergence.py"} <= set(files)
    assert commands[-1] == "python3 collect_convergence.py"
    with tempfile.TemporaryDirectory() as d:
        for name, text in files.items():
            open(os.path.join(d, name), "w").write(text)
        for i, e in enumerate((-100.0, -100.01, -100.0101)):
            open(os.path.join(d, f"c.ecut{30 + 10 * i}.out"), "w").write(f"!    total energy              =   {e} Ry\n")
        r = subprocess.run([sys.executable, "collect_convergence.py"], cwd=d, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert os.path.exists(os.path.join(d, "convergence.csv"))
