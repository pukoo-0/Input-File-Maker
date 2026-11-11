import numpy as np
import pytest

import structures as S
from data import CSPBBR3_CIF, NI_SLAB_POSCAR, RUTILE_CIF, WATER_XYZ


def test_xyz():
    s = S.read_xyz(WATER_XYZ)
    assert s.symbols == ["O", "H", "H"]
    assert not s.periodic
    assert np.isclose(s.positions[1, 1], 0.7572)


def test_extended_xyz_cell():
    text = '2\nLattice="4 0 0 0 4 0 0 0 4" Properties=species:S:1:pos:R:3\nNa 0 0 0\nCl 2 2 2\n'
    s = S.read_xyz(text)
    assert s.periodic and np.allclose(s.cell, np.eye(3) * 4)


def test_cif_applies_symmetry():
    s = S.read_cif(CSPBBR3_CIF)
    assert s.n == 5
    assert sorted(s.symbols) == ["Br", "Br", "Br", "Cs", "Pb"]
    assert np.allclose(s.cell_parameters()[:3], 5.874)


def test_cif_rutile_and_text_fields():
    s = S.read_cif(RUTILE_CIF)
    assert s.formula() == "Ti2O4"
    assert np.isclose(s.cell_parameters()[2], 2.959)


def test_poscar_selective_dynamics():
    s = S.read_poscar(NI_SLAB_POSCAR)
    assert s.formula() == "Ni4O"
    assert s.fixed[:2].all()
    assert s.fixed[4].tolist() == [False, False, True]


def test_read_structure_by_name():
    assert S.read_structure(WATER_XYZ.encode(), "w.xyz").n == 3
    assert S.read_structure(NI_SLAB_POSCAR.encode(), "POSCAR").n == 5
    with pytest.raises(ValueError):
        S.read_structure(b"x", "file.pdb")


def test_kgrid_and_slab_detection():
    slab = S.read_poscar(NI_SLAB_POSCAR)
    assert S.vacuum_axis(slab) == 2
    grid = S.kgrid_from_spacing(slab.cell, 0.25, slab_axis=2)
    assert grid[2] == 1 and grid[0] == grid[1] > 1
    assert S.vacuum_axis(S.read_cif(CSPBBR3_CIF)) is None


def test_boxed_molecule_is_centred():
    b = S.boxed(S.read_xyz(WATER_XYZ), 8.0)
    centre = (b.positions.max(0) + b.positions.min(0)) / 2
    assert np.allclose(centre, b.cell.diagonal() / 2)


def test_supercell_and_vacuum():
    bulk = S.read_cif(CSPBBR3_CIF)
    sc = S.supercell(bulk, 2, 2, 1)
    assert sc.n == 20 and np.isclose(sc.volume(), 4 * bulk.volume())
    slab = S.read_poscar(NI_SLAB_POSCAR)
    v = S.add_vacuum(slab, 2, 15.0)
    assert np.isclose(np.linalg.norm(v.cell[2]), 15.0 + np.ptp(slab.positions[:, 2]))
    d = lambda st: np.linalg.norm(st.positions[4] - st.positions[2])
    assert np.isclose(d(slab), d(v))


def test_viewer_html_marks_fixed_atoms():
    h = S.viewer_html(S.read_poscar(NI_SLAB_POSCAR), highlight=[0, 1])
    assert "3Dmol" in h and "[0, 1]" in h


def test_band_path_with_ase():
    pytest.importorskip("ase")
    path = S.band_path(np.eye(3) * 4.0, 20)
    assert path[0][0] == "G" and all(len(p[1]) == 3 for p in path)


def test_primitive_cell_with_spglib():
    pytest.importorskip("spglib")
    fcc_conv = S.Structure(["Cu"] * 4, np.array([[0, 0, 0], [0, 1.8, 1.8], [1.8, 0, 1.8], [1.8, 1.8, 0]]),
                           np.eye(3) * 3.6)
    assert S.primitive_cell(fcc_conv).n == 1
    assert "225" in S.symmetry_info(fcc_conv)
