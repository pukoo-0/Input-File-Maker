"""Small test structures."""

WATER_XYZ = """3
water
O   0.000000   0.000000   0.117300
H   0.000000   0.757200  -0.469200
H   0.000000  -0.757200  -0.469200
"""

# Cubic perovskite with a partial list of symmetry operations: the 3 Br must be generated from 1.
CSPBBR3_CIF = """data_CsPbBr3
_cell_length_a 5.874(2)
_cell_length_b 5.874
_cell_length_c 5.874
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_symmetry_equiv_pos_as_xyz
'x, y, z'
'z, x, y'
'y, z, x'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Cs1 Cs 0.5 0.5 0.5
Pb1 Pb 0 0 0
Br1 Br 0.5 0 0
"""

RUTILE_CIF = """data_rutile
_cell_length_a 4.594
_cell_length_b 4.594
_cell_length_c 2.959
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_publ_section_title
;
 A multi-line
 title
;
loop_
_space_group_symop_operation_xyz
'x,y,z'
'-x,-y,z'
'-y+1/2,x+1/2,z+1/2'
'y+1/2,-x+1/2,z+1/2'
'-x+1/2,y+1/2,-z+1/2'
'x+1/2,-y+1/2,-z+1/2'
'y,x,-z'
'-y,-x,-z'
'-x,-y,-z'
'x,y,-z'
'y+1/2,-x+1/2,-z+1/2'
'-y+1/2,x+1/2,-z+1/2'
'x+1/2,-y+1/2,z+1/2'
'-x+1/2,y+1/2,z+1/2'
'-y,-x,z'
'y,x,z'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Ti1 0.0 0.0 0.0
O1 0.3053(2) 0.3053(2) 0.0
"""

NI_SLAB_POSCAR = """Ni slab
1.0
3.52 0 0
0 3.52 0
0 0 20.0
Ni O
4 1
Selective dynamics
Direct
0 0 0 F F F
0.5 0.5 0 F F F
0.5 0 0.088 T T T
0 0.5 0.088 T T T
0.25 0.25 0.2 T T F
"""
