import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "web"))

import bridge  # noqa: E402
from data import CSPBBR3_CIF, NI_SLAB_POSCAR, WATER_XYZ  # noqa: E402


def build(p):
    return json.loads(bridge.build(json.dumps(p)))


def qe_defaults():
    o = json.loads(bridge.options())
    return dict(o["qe"]["defaults"], kspacing=0.25, kgrid_auto=True, ecutrho_auto=True, vacuum=10.0,
                pseudo_template="{El}.upf")


def test_options_cover_both_programs():
    o = json.loads(bridge.options())
    assert "SP" in o["orca"]["jobs"] and "bands" in o["qe"]["workflows"] and "scheduler" in o["job"]


def test_orca_build_and_electrons():
    o = json.loads(bridge.options())
    r = build({"program": "ORCA", "structure": {"name": "w.xyz", "text": WATER_XYZ}, "orca": o["orca"]["defaults"]})
    assert not r["errors"] and "job.inp" in r["files"] and r["info"]["electrons"] == 10


def test_qe_species_edits_are_kept():
    q = qe_defaults()
    first = build({"program": "QE", "structure": {"name": "c.cif", "text": CSPBBR3_CIF}, "qe": q})
    sp = first["species"]
    sp[1].update({"Pseudopotential": "Pb.mine.UPF", "auto": False, "Start. magnetisation": 0.2})
    r = build({"program": "QE", "structure": {"name": "c.cif", "text": CSPBBR3_CIF}, "qe": dict(q, spin="collinear"),
               "atoms": first["atoms"], "species": sp})
    text = r["files"]["pwscf.scf.in"]
    assert "Pb.mine.UPF" in text and "starting_magnetization(2) = 0.2" in text


def test_qe_fixed_atoms_from_poscar_are_highlighted():
    r = build({"program": "QE", "structure": {"name": "POSCAR", "text": NI_SLAB_POSCAR}, "qe": qe_defaults()})
    assert r["view"]["highlight"] == [0, 1, 4] and r["info"]["slab_axis"] == 2 and r["info"]["kgrid_auto"][2] == 1


def test_structure_errors_are_reported():
    r = build({"program": "QE", "structure": {"name": "x.cif", "text": "nonsense"}, "qe": qe_defaults()})
    assert r["errors"] and not r["files"]


def test_built_page_is_up_to_date():
    """docs/index.html must contain the current Python modules: run `python build_web.py` after editing them."""
    html = open(os.path.join(ROOT, "docs", "index.html"), encoding="utf-8").read()
    blob = re.search(r"const PY_FILES = (\{.*?\});\n", html, re.S).group(1).replace("<\\/", "</")
    embedded = json.loads(blob)
    for name, src in embedded.items():
        path = os.path.join(ROOT, "web" if name == "bridge.py" else "", name)
        assert open(path, encoding="utf-8").read() == src, f"{name} changed: run python build_web.py"
