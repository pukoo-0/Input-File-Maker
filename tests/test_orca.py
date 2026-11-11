import numpy as np

import orca_input as O
import structures as S
from data import WATER_XYZ

WATER = S.read_xyz(WATER_XYZ)


def bang(files, name="job.inp"):
    return next(l for l in files[name].splitlines() if l.startswith("!"))


def test_default_input_is_valid():
    files, errors, warnings = O.build_orca({}, WATER)
    assert not errors
    text = files["job.inp"]
    assert "* xyz 0 1" in text and text.strip().endswith("*")
    assert "B3LYP" in bang(files) and "D3BJ" in bang(files) and "def2/J" in bang(files)


def test_multiplicity_parity():
    _, errors, _ = O.build_orca({"mult": 2}, WATER)
    assert errors
    _, errors, _ = O.build_orca({"charge": 1, "mult": 2}, WATER)
    assert not errors


def test_built_in_dispersion_not_doubled():
    files, _, warnings = O.build_orca({"functional": "wB97X-V"}, WATER)
    assert "D3BJ" not in bang(files) and warnings


def test_dlpno_gets_correlation_auxiliary_basis():
    files, errors, _ = O.build_orca({"family": "Wavefunction", "wavefunction": "DLPNO-CCSD(T)", "basis": "cc-pVTZ"}, WATER)
    assert not errors and "cc-pVTZ/C" in bang(files)


def test_heavy_elements_and_pople_basis():
    pb = S.Structure(["Pb", "I", "I"], np.array([[0, 0, 0], [2.8, 0, 0], [-2.8, 0, 0]]))
    _, errors, _ = O.build_orca({"basis": "6-31G(d)"}, pb)
    assert errors
    files, errors, _ = O.build_orca({"relativistic": "ZORA"}, pb)
    assert 'NewGTO Pb "SARC-ZORA-TZVP"' in files["job.inp"]


def test_scan_and_constraints():
    files, errors, _ = O.build_orca({"job": "Scan", "scan_atoms": "0 1", "fix_atoms": "2"}, WATER)
    assert not errors
    t = files["job.inp"]
    assert "B 0 1 = 1.0, 2.0, 10" in t and "{C 2 C}" in t
    _, errors, _ = O.build_orca({"job": "Scan", "scan_atoms": "0 7"}, WATER)
    assert errors


def test_neb_needs_final_structure():
    _, errors, _ = O.build_orca({"job": "NEB-CI"}, WATER)
    assert errors
    files, errors, _ = O.build_orca({"job": "NEB-CI"}, WATER, WATER)
    assert not errors and "job_final.xyz" in files


def test_md_block():
    files, errors, _ = O.build_orca({"job": "MD", "family": "xTB"}, WATER)
    t = files["job.inp"]
    assert not errors and "%md" in t and "Thermostat CSVR" in t and "Run 2000" in t


def test_solvation_and_tddft():
    files, _, _ = O.build_orca({"solvation": "SMD", "solvent": "water", "tddft": True, "nroots": 5}, WATER)
    t = files["job.inp"]
    assert "CPCM(water)" in t and "smd true" in t and "NRoots 5" in t
    _, errors, _ = O.build_orca({"solvation": "ALPB"}, WATER)
    assert errors
