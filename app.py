"""
DFT Input Maker — Streamlit app for ORCA and Quantum ESPRESSO input files.
Run with:  streamlit run app.py
The logic lives in the other modules; this file is only the interface.
"""

import io
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import jobscript as J
import orca_input as O
import pseudolib as PL
import qe_input as Q
import recipes as R
import settings_io as SIO
import structures as S

st.set_page_config(page_title="DFT Input Maker", page_icon="⚛️", layout="wide")
ss = st.session_state

try:
    import ase  # noqa: F401
    HAVE_ASE = True
except Exception:
    HAVE_ASE = False
try:
    import spglib  # noqa: F401
    HAVE_SPGLIB = True
except Exception:
    HAVE_SPGLIB = False

EXAMPLE_XYZ = """3
water
O   0.000000   0.000000   0.117300
H   0.000000   0.757200  -0.469200
H   0.000000  -0.757200  -0.469200
"""
EXAMPLE_CIF = """data_CsPbBr3
_cell_length_a 5.874
_cell_length_b 5.874
_cell_length_c 5.874
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Cs1 Cs 0.5 0.5 0.5
Pb1 Pb 0.0 0.0 0.0
Br1 Br 0.5 0.0 0.0
Br2 Br 0.0 0.5 0.0
Br3 Br 0.0 0.0 0.5
"""


# ---------------------------------------------------------------------
# Widget helpers. They only pass a default when the key has no value yet,
# so recipes and loaded settings can set values without warnings.
# ---------------------------------------------------------------------
def select(label, options, default, key, **kw):
    options = list(options)
    if key in ss and ss[key] not in options:
        del ss[key]
    if key not in ss:
        kw["index"] = options.index(default) if default in options else 0
    return st.selectbox(label, options, key=key, **kw)


def radio(label, options, default, key, **kw):
    options = list(options)
    if key in ss and ss[key] not in options:
        del ss[key]
    if key not in ss:
        kw["index"] = options.index(default) if default in options else 0
    return st.radio(label, options, key=key, **kw)


def num(label, min_value, max_value, value, key, **kw):
    kind = type(min_value)
    if key in ss:
        try:
            ss[key] = min(max(kind(ss[key]), min_value), max_value)
        except Exception:
            del ss[key]
    if key not in ss:
        kw["value"] = kind(value)
    return st.number_input(label, min_value=min_value, max_value=max_value, key=key, **kw)


def chk(label, value, key, **kw):
    if key not in ss:
        kw["value"] = value
    return st.checkbox(label, key=key, **kw)


def txt(label, value, key, **kw):
    if key not in ss:
        kw["value"] = value
    return st.text_input(label, key=key, **kw)


def area(label, value, key, **kw):
    if key not in ss:
        kw["value"] = value
    return st.text_area(label, key=key, **kw)


def apply_values(values):
    """Callback: set many widget values at once (recipes, loaded settings)."""
    for k, v in values.items():
        ss[k] = v


def zip_files(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, text in files.items():
            info = zipfile.ZipInfo(name)
            info.external_attr = (0o755 if name.endswith((".sh", ".py", ".slurm", ".pbs")) else 0o644) << 16
            z.writestr(info, text)
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def read_upload(name, raw):
    return S.read_structure(raw, name)


# ---------------------------------------------------------------------
# Sidebar: settings file, program, structure
# ---------------------------------------------------------------------
with st.sidebar:
    with st.expander("Load saved settings"):
        sfile = st.file_uploader("Settings file (.json)", type=["json"], key="up_settings")
        if sfile is not None:
            try:
                loaded = SIO.load(sfile.getvalue().decode("utf-8"))

                def _apply_loaded(data=loaded):
                    apply_values(data["widgets"])
                    ss["program"] = data.get("program", "ORCA")
                    ss["loaded_tables"] = data["tables"]
                    ss["loaded_token"] = ss.get("loaded_token", 0) + 1

                st.button("Apply these settings", on_click=_apply_loaded, type="primary")
            except Exception as e:
                st.error(str(e))

    st.header("1. Program")
    program = radio("Program", ["ORCA", "Quantum ESPRESSO"], "ORCA", "program", horizontal=True,
                    label_visibility="collapsed")

    st.header("2. Structure")
    up = st.file_uploader("Structure file (.xyz, .cif, POSCAR, CONTCAR, .vasp)", type=None, key="up_struct",
                          help="POSCAR and CONTCAR files without an extension are fine too.")
    use_example = st.checkbox("Use an example structure", value=up is None)

base, struct_err = None, None
if up is not None:
    try:
        base = read_upload(up.name, up.getvalue())
    except Exception as e:
        struct_err = f"Couldn’t read {up.name}: {e}"
elif use_example:
    base = (S.read_xyz(EXAMPLE_XYZ, "example water.xyz") if program == "ORCA"
            else S.read_cif(EXAMPLE_CIF, "example CsPbBr3.cif"))

# Structure tools: a list of operations applied to the uploaded structure, in order.
ops_key = f"ops::{base.source}::{base.n}" if base is not None else "ops::none"
ss.setdefault(ops_key, [])
struct, op_err = base, None
if base is not None:
    try:
        for op in ss[ops_key]:
            if op[0] == "primitive":
                struct = S.primitive_cell(struct, op[1])
            elif op[0] == "supercell":
                struct = S.supercell(struct, *op[1:])
            elif op[0] == "vacuum":
                struct = S.add_vacuum(struct, *op[1:])
            elif op[0] == "slab":
                struct = S.make_slab(struct, op[1], op[2], op[3])
    except Exception as e:
        op_err = str(e)
        ss[ops_key] = []
        struct = base
    if ss[ops_key]:
        struct.source = base.source + " → " + " → ".join(o[0] for o in ss[ops_key])

with st.sidebar:
    if struct_err:
        st.error(struct_err)
    if struct is not None:
        st.markdown(f"**{struct.formula()}**, {struct.n} atoms  \n{struct.source}")
        if struct.periodic:
            a, b, c_, al, be, ga = struct.cell_parameters()
            st.caption(f"a = {a:.4f}, b = {b:.4f}, c = {c_:.4f} Å  \nα = {al:.2f}°, β = {be:.2f}°, γ = {ga:.2f}°  \n"
                       f"Volume {struct.volume():.2f} Å³")
        else:
            st.caption("Molecule (no unit cell)")
        for n in struct.notes:
            st.caption(n)

    st.markdown("**Final structure** (only for NEB)")
    up_final = st.file_uploader("Final structure, same atoms in the same order", type=None, key="up_final",
                                label_visibility="collapsed")
final = None
if up_final is not None:
    try:
        final = read_upload(up_final.name, up_final.getvalue())
    except Exception as e:
        st.sidebar.error(f"Couldn’t read the final structure: {e}")

st.title("DFT Input Maker")
st.caption("Build ORCA and Quantum ESPRESSO input files, with checks for common mistakes. Pick your settings, "
           "then download everything at the bottom of the page.")

if struct is None:
    st.info("Upload a structure in the sidebar, or tick **Use an example structure**.")
    st.stop()
if op_err:
    st.error(f"Structure tool failed: {op_err}")

# ---------------------------------------------------------------------
# Structure view and tools
# ---------------------------------------------------------------------
view_col, tool_col = st.columns([1.4, 1])
with tool_col:
    with st.expander("Quick-start recipes"):
        book = R.ORCA if program == "ORCA" else R.QE
        pick = st.selectbox("Recipe", list(book), key="recipe_pick")
        st.caption(book[pick][0])
        st.button("Apply recipe", on_click=apply_values, args=(book[pick][1],), type="primary")
        st.caption("A recipe only sets a starting point. Everything stays editable.")

    if base.periodic:
        with st.expander("Structure tools"):
            try:
                st.markdown(f"Space group: **{S.symmetry_info(struct)}**")
            except RuntimeError as e:
                st.caption(str(e))
            t1, t2 = st.columns(2)
            symprec = t1.number_input("Symmetry tolerance (Å)", 1e-5, 0.5, 1e-3, format="%.0e", key="tool_symprec")
            t2.write("")
            t2.button("Use primitive cell", on_click=lambda: ss[ops_key].append(("primitive", symprec)),
                      disabled=not HAVE_SPGLIB,
                      help="Fewer atoms, same crystal: often several times cheaper." if HAVE_SPGLIB
                      else "Needs spglib (pip install spglib).")
            st.markdown("**Supercell**")
            s1, s2, s3, s4 = st.columns(4)
            n1 = s1.number_input("n1", 1, 10, 1, key="tool_n1")
            n2 = s2.number_input("n2", 1, 10, 1, key="tool_n2")
            n3 = s3.number_input("n3", 1, 10, 1, key="tool_n3")
            s4.write("")
            s4.button("Make", key="tool_sc", on_click=lambda: ss[ops_key].append(("supercell", n1, n2, n3)))
            st.markdown("**Slab from bulk** (Miller indices)")
            h1, h2, h3, h4, h5 = st.columns(5)
            hkl = (h1.number_input("h", -9, 9, 1, key="tool_h"), h2.number_input("k", -9, 9, 0, key="tool_k"),
                   h3.number_input("l", -9, 9, 0, key="tool_l"))
            layers = h4.number_input("Layers", 1, 30, 4, key="tool_layers")
            svac = h5.number_input("Vacuum (Å)", 0.0, 60.0, 15.0, key="tool_svac")
            st.button("Build slab", disabled=not HAVE_ASE,
                      on_click=lambda: ss[ops_key].append(("slab", hkl, layers, svac)),
                      help=None if HAVE_ASE else "Needs ASE (pip install ase).")
            st.markdown("**Add vacuum**")
            v1, v2, v3 = st.columns(3)
            vax = v1.selectbox("Axis", [1, 2, 3], index=2, key="tool_vax")
            vvac = v2.number_input("Vacuum (Å)", 0.0, 80.0, 15.0, key="tool_vvac")
            v3.write("")
            v3.button("Add", key="tool_addvac", on_click=lambda: ss[ops_key].append(("vacuum", vax - 1, vvac)))
            if ss[ops_key]:
                st.caption("Applied: " + " → ".join(o[0] for o in ss[ops_key]))
                st.button("Back to the uploaded structure", on_click=lambda: ss[ops_key].clear())

    with st.expander("Atom numbers"):
        st.caption("ORCA counts atoms from 0; Quantum ESPRESSO lists them in this order.")
        st.dataframe(pd.DataFrame({"ORCA #": range(struct.n), "Element": struct.symbols,
                                   "x (Å)": struct.positions[:, 0], "y (Å)": struct.positions[:, 1],
                                   "z (Å)": struct.positions[:, 2]}).round(5), hide_index=True)
view_slot = view_col.container()

loaded_tables = ss.get("loaded_tables") or {}
token = ss.get("loaded_token", 0)

# =====================================================================
# ORCA
# =====================================================================
if program == "ORCA":
    s = {}
    t_method, t_job, t_env, t_scf, t_adv, t_res = st.tabs(
        ["Method", "Job type", "Solvent & properties", "SCF & accuracy", "Advanced", "Resources & output"])

    with t_method:
        c1, c2 = st.columns(2)
        with c1:
            fam = select("Method family", ["DFT", "HF", "Composite", "xTB", "Wavefunction", "CASSCF"], "DFT", "o_fam",
                         format_func=lambda f: {"DFT": "DFT", "HF": "Hartree–Fock", "Composite": "Composite (3c methods)",
                                                "xTB": "Semi-empirical (GFN-xTB)",
                                                "Wavefunction": "Wavefunction (MP2, coupled cluster)",
                                                "CASSCF": "Multireference (CASSCF / NEVPT2)"}[f])
            s["family"] = fam
            if fam == "DFT":
                grp = select("Functional type", list(O.FUNCTIONALS), "Hybrid", "o_fgrp")
                s["functional"] = select("Functional", O.FUNCTIONALS[grp], O.FUNCTIONALS[grp][0], "o_func")
            elif fam == "Composite":
                s["composite"] = select("Composite method", O.COMPOSITE, "r2SCAN-3c", "o_comp",
                                        help="Comes with its own basis set and dispersion correction.")
            elif fam == "xTB":
                s["xtb"] = select("xTB method", O.XTB, "XTB2", "o_xtb")
            elif fam == "Wavefunction":
                s["wavefunction"] = select("Method", O.WAVEFUNCTION, "DLPNO-CCSD(T)", "o_wf")
                if s["wavefunction"].startswith("DLPNO"):
                    s["pno"] = select("PNO accuracy", ["LoosePNO", "NormalPNO", "TightPNO"], "NormalPNO", "o_pno",
                                      help="TightPNO is recommended for reaction energies and weak interactions.")
            if fam in ("DFT", "HF"):
                s["dispersion"] = select("Dispersion correction", ["D3BJ", "D4", "D3ZERO", "NL", "none"], "D3BJ", "o_disp")
        with c2:
            if fam in ("DFT", "HF", "Wavefunction", "CASSCF"):
                bgrp = select("Basis set family", list(O.BASIS_SETS), "Karlsruhe (def2)", "o_bgrp")
                s["basis"] = select("Basis set", O.BASIS_SETS[bgrp], O.BASIS_SETS[bgrp][1], "o_basis")
                s["ri"] = select("RI approximation", ["auto", "RIJCOSX", "RI", "RIJK", "RIJONX", "NoRI"], "auto", "o_ri",
                                 help="“auto” picks RI for pure functionals and RIJCOSX for hybrids and wavefunction methods.")
                s["aux_j"] = txt("Auxiliary basis for RI (auto = chosen for you)", "auto", "o_auxj")
                if fam == "Wavefunction" or (fam == "DFT" and O.FUNCTIONAL_GROUP.get(s.get("functional")) == "Double hybrid"):
                    s["aux_c"] = txt("Correlation auxiliary basis (/C)", "auto", "o_auxc")
                s["relativistic"] = select("Scalar relativistic", ["none", "ZORA", "DKH", "X2C"], "none", "o_rel",
                                           help="Only needed for heavy elements without effective core potentials.")
            if fam in ("DFT", "HF", "Wavefunction"):
                s["reference"] = select("Reference", ["auto", "restricted", "unrestricted", "restricted-open"], "auto",
                                        "o_ref", help="auto: restricted for closed shells, unrestricted for open shells.")
        st.markdown("**Charge and spin**")
        c1, c2, c3 = st.columns(3)
        with c1:
            s["charge"] = num("Charge", -50, 50, 0, "o_charge", step=1)
        with c2:
            s["mult"] = num("Multiplicity (2S + 1)", 1, 50, 1, "o_mult", step=1)
        n_el = O.electron_count(struct.symbols, s["charge"])
        ok = (n_el + s["mult"]) % 2 == 1
        c3.metric("Electrons", n_el, "multiplicity OK" if ok else "multiplicity doesn’t match",
                  delta_color="normal" if ok else "inverse")

    with t_job:
        s["job"] = select("Job type", list(O.JOBS), "SP", "o_job", format_func=lambda j_: O.JOBS[j_])
        job = s["job"]
        if job in ("Opt", "OptFreq", "OptTS", "OptTSFreq", "Scan"):
            c1, c2 = st.columns(2)
            with c1:
                s["opt_conv"] = select("Optimisation convergence", ["loose", "normal", "tight", "verytight"], "normal", "o_optc")
            with c2:
                s["opt_maxiter"] = num("Max optimisation cycles (0 = ORCA default)", 0, 1000, 0, "o_optmax")
        if job in ("OptTS", "OptTSFreq"):
            s["ts_recalc_hess"] = num("Recalculate the Hessian every N steps (0 = never)", 0, 100, 0, "o_recalc")
        if job in ("Freq", "OptFreq", "OptTSFreq"):
            c1, c2, c3 = st.columns(3)
            with c1:
                s["numfreq"] = chk("Numerical frequencies", False, "o_numfreq",
                                   help="Needed for some methods and for Raman intensities.")
                s["raman"] = chk("Raman intensities", False, "o_raman")
            with c2:
                s["temperature"] = num("Temperature (K)", 0.0, 10000.0, 298.15, "o_temp")
            with c3:
                s["pressure"] = num("Pressure (atm)", 0.0, 1e6, 1.0, "o_press")
        if job == "Scan":
            c1, c2, c3 = st.columns(3)
            with c1:
                s["scan_relaxed"] = radio("Scan type", [True, False], True, "o_scanrel",
                                          format_func=lambda r: "Relaxed (optimise the rest)" if r else "Rigid (single points)")
            with c2:
                s["scan_type"] = select("Coordinate", ["B", "A", "D"], "B", "o_scant",
                                        format_func=lambda t: {"B": "Bond length (Å)", "A": "Angle (°)", "D": "Dihedral (°)"}[t])
            with c3:
                s["scan_atoms"] = txt("Atom numbers (from 0)", "0 1", "o_scana")
            c1, c2, c3 = st.columns(3)
            with c1:
                s["scan_start"] = num("Start", -360.0, 360.0, 1.0, "o_scan1")
            with c2:
                s["scan_end"] = num("End", -360.0, 360.0, 2.0, "o_scan2")
            with c3:
                s["scan_steps"] = num("Number of points", 2, 200, 10, "o_scann")
        if job == "IRC":
            c1, c2 = st.columns(2)
            with c1:
                s["irc_direction"] = select("Direction", ["both", "forward", "backward"], "both", "o_ircdir")
            with c2:
                s["irc_maxiter"] = num("Max steps per direction", 1, 500, 50, "o_ircmax")
            st.caption("Start the IRC from an optimised transition state.")
        if job in ("NEB-CI", "NEB-TS"):
            s["neb_images"] = num("Number of images", 3, 50, 8, "o_nebimg")
            if final is None:
                st.info("Upload the final structure in the sidebar.")
        if job == "MD":
            c1, c2, c3 = st.columns(3)
            with c1:
                s["md_timestep"] = num("Time step (fs)", 0.05, 5.0, 0.5, "o_mdts")
                s["md_steps"] = num("Number of steps", 1, 10_000_000, 2000, "o_mdsteps")
            with c2:
                s["md_temp"] = num("Temperature (K)", 0.0, 10000.0, 300.0, "o_mdtemp")
                s["md_thermostat"] = select("Thermostat", ["CSVR", "Berendsen", "NHC", "none"], "CSVR", "o_mdthermo",
                                            help="CSVR gives a correct canonical ensemble; “none” means constant energy (NVE).")
            with c3:
                s["md_timecon"] = num("Thermostat time constant (fs)", 1.0, 10000.0, 10.0, "o_mdtc")
                s["md_stride"] = num("Save every N steps", 1, 100000, 1, "o_mdstride")
            st.caption(f"Total time: {s['md_steps'] * s['md_timestep'] / 1000:.3g} ps")
        st.markdown("**Constraints** (optimisations and scans)")
        c1, c2 = st.columns(2)
        with c1:
            s["fix_atoms"] = txt("Fix these atoms completely (numbers from 0)", "", "o_fix", help="Example: 0 1 5")
        with c2:
            s["constraints"] = area("Other constraints, one per line", "", "o_cons", height=100,
                                    help="Keep a bond: {B 0 1 C}. Angle: {A 0 1 2 C}. Dihedral: {D 0 1 2 3 C}. "
                                         "Set a value: {B 0 1 1.50 C}.")

    with t_env:
        c1, c2 = st.columns(2)
        with c1:
            s["solvation"] = select("Solvation model", ["none", "CPCM", "SMD", "ALPB"], "none", "o_solv",
                                    help="SMD is usually best for solvation free energies; CPCM is faster. ALPB is for xTB.")
            if s["solvation"] != "none":
                sv = select("Solvent", O.SOLVENTS + ["other…"], "water", "o_solvent")
                s["solvent"] = txt("Solvent name", "", "o_solvent_other") if sv == "other…" else sv
        with c2:
            s["tddft"] = chk("Excited states (TD-DFT / CIS)", False, "o_tddft")
            if s["tddft"]:
                s["nroots"] = num("Number of excited states", 1, 200, 10, "o_nroots")
                s["tda"] = chk("Tamm–Dancoff approximation (TDA)", True, "o_tda")
                s["triplets"] = chk("Also triplets", False, "o_trip")
                s["iroot"] = num("Optimise / follow this excited state (0 = ground state)", 0, 200, 0, "o_iroot")
        st.markdown("**Properties**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            s["nmr"] = chk("NMR shieldings", False, "o_nmr")
            s["epr"] = chk("EPR g-tensor", False, "o_epr")
        with c2:
            s["hirshfeld"] = chk("Hirshfeld charges", False, "o_hirsh")
            s["chelpg"] = chk("CHELPG charges", False, "o_chelpg")
        with c3:
            s["nbo"] = chk("NBO analysis", False, "o_nbo")
            s["polar"] = chk("Polarisability", False, "o_polar")
        with c4:
            s["largeprint"] = chk("Detailed output (LargePrint)", False, "o_large")

    with t_scf:
        c1, c2 = st.columns(2)
        with c1:
            s["scf_conv"] = select("SCF convergence", ["LooseSCF", "NormalSCF", "TightSCF", "VeryTightSCF", "default"],
                                   "TightSCF", "o_scfc", help="TightSCF is recommended for optimisations and frequencies.")
            s["slowconv"] = select("Help difficult convergence", ["none", "SlowConv", "VerySlowConv"], "none", "o_slow",
                                   help="Useful for transition metals and open-shell systems.")
            s["scf_maxiter"] = num("Max SCF iterations (0 = ORCA default)", 0, 5000, 0, "o_scfmax")
            s["grid"] = select("Integration grid (DFT)", ["default", "DefGrid1", "DefGrid2", "DefGrid3"], "default", "o_grid",
                               help="DefGrid3 is more accurate, e.g. for small imaginary frequencies.")
        with c2:
            s["trah"] = chk("Allow TRAH (robust automatic SCF fallback)", True, "o_trah")
            s["stability"] = chk("Check SCF stability", False, "o_stab",
                                 help="Checks whether a lower-energy solution exists, e.g. for diradicals.")
            s["brokensym"] = txt("Broken symmetry (e.g. 1,1; leave empty if not needed)", "", "o_bs")
            s["guess"] = select("Initial guess", ["default", "PModel", "HCore", "Hueckel", "PAtom", "MORead"], "default", "o_guess")
            if s["guess"] == "MORead":
                s["moread_file"] = txt("Orbital file to read (.gbw)", "previous.gbw", "o_gbw")

    with t_adv:
        if fam == "CASSCF":
            st.markdown("**CASSCF active space**")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                s["cas_nel"] = num("Active electrons", 1, 50, 2, "o_casel")
            with c2:
                s["cas_norb"] = num("Active orbitals", 1, 50, 2, "o_casorb")
            with c3:
                s["cas_nroots"] = num("Roots", 1, 50, 1, "o_casroots")
            with c4:
                s["cas_pt"] = select("Perturbation correction", ["none", "SC_NEVPT2", "FIC_NEVPT2"], "none", "o_caspt")
        st.markdown("**Anything else** (added exactly as written)")
        s["extra_keywords"] = txt("Extra ! keywords", "", "o_extrakw", help="Example: PrintBasis KeepDens")
        s["extra_blocks"] = area("Extra % blocks", "", "o_extrablk", height=150,
                                 help="Any ORCA block, for example %output Print[P_Hirshfeld] 1 end")

    with t_res:
        c1, c2, c3 = st.columns(3)
        with c1:
            s["nprocs"] = num("CPU cores (%pal)", 1, 1024, 4, "o_np")
        with c2:
            s["maxcore"] = num("Memory per core (%maxcore, MB)", 100, 1_000_000, 2000, "o_mem", step=250,
                               help="Use about 75% of the real memory per core; ORCA can exceed this value.")
        with c3:
            s["basename"] = txt("File name", "job", "o_name")
        s["coords"] = radio("Coordinates", ["inline", "file"], "inline", "o_coords", horizontal=True,
                            format_func=lambda c_: "Inside the input file" if c_ == "inline" else "Separate .xyz file")

    files, errors, warnings = O.build_orca(s, struct, final)
    commands = [f"$(which orca) {s['basename']}.inp > {s['basename']}.out"]
    zip_name = f"{s['basename']}_orca.zip"
    program_tag = "orca"
    highlight = [int(x) for x in str(s.get("fix_atoms", "")).replace(",", " ").split() if x.isdigit()]
    view_struct = struct
    tables = {}

# =====================================================================
# Quantum ESPRESSO
# =====================================================================
else:
    c = {}
    qs, qfinal = struct, final
    with st.sidebar:
        if not struct.periodic:
            vac = num("Vacuum around the molecule (Å)", 4.0, 40.0, 10.0, "q_vac", step=1.0,
                      help="Quantum ESPRESSO needs a cell, so the molecule is placed in a cubic box.")
            qs = S.boxed(struct, vac)
            if final is not None and not final.periodic:
                shift = qs.positions[0] - struct.positions[0]
                qfinal = S.Structure(final.symbols, final.positions + shift, qs.cell)
    slab_axis = S.vacuum_axis(qs)
    akey = f"atoms::{struct.source}::{qs.n}::{token}"

    t_wf, t_atoms, t_sp, t_el, t_func, t_k, t_ions, t_post, t_extra = st.tabs(
        ["Workflow", "Atoms", "Species & pseudopotentials", "Electrons & spin", "Functional & corrections",
         "k-points", "Relax / MD / cell", "Post-processing", "Extra"])

    # ---------- workflow
    with t_wf:
        c1, c2 = st.columns(2)
        with c1:
            c["workflow"] = select("What do you want to calculate?", list(Q.WORKFLOWS), "single", "q_wf",
                                   format_func=lambda w: Q.WORKFLOWS[w])
            c["calculation"] = select("Calculation", Q.CALCULATIONS, "scf", "q_calc") if c["workflow"] == "single" else "scf"
            c["coord_units"] = select("Atomic positions in", ["crystal", "angstrom"], "crystal", "q_units",
                                      format_func=lambda u: "Fractional (crystal)" if u == "crystal" else "Ångström")
        with c2:
            c["prefix"] = txt("prefix (name for all files)", "pwscf", "q_prefix")
            c["outdir"] = txt("outdir (scratch folder)", "./out", "q_outdir")
            c["pseudo_dir"] = txt("pseudo_dir (folder with the .UPF files)", "./pseudo", "q_pdir")
            c["verbosity"] = select("verbosity", ["low", "high"], "high", "q_verb",
                                    help="“high” prints all eigenvalues and more detail; use “low” for MD.")
        if not struct.periodic:
            st.info(f"Your molecule was placed in a {qs.cell[0, 0]:.2f} Å cubic box. For isolated molecules, use "
                    "K_POINTS gamma and the Martyna–Tuckerman correction (Functional & corrections tab).")
        if c["workflow"] == "convergence":
            st.markdown("**Convergence tests**")
            c["conv_mode"] = radio("Test", ["both", "ecut", "kpoints"], "both", "q_convmode", horizontal=True,
                                   format_func=lambda m: {"both": "Cutoff and k-points", "ecut": "Cutoff only",
                                                          "kpoints": "k-points only"}[m])
            e1, e2, e3 = st.columns(3)
            with e1:
                c["conv_ecut_start"] = num("ecutwfc from (Ry)", 10.0, 300.0, 30.0, "q_cstart", step=5.0)
            with e2:
                c["conv_ecut_stop"] = num("to (Ry)", 10.0, 400.0, 80.0, "q_cstop", step=5.0)
            with e3:
                c["conv_ecut_step"] = num("step (Ry)", 1.0, 100.0, 10.0, "q_cstep", step=1.0)
            ks = txt("k-point spacings to test (Å⁻¹)", "0.5 0.4 0.3 0.25 0.2 0.15", "q_ckspacings",
                     help="Each spacing gives a grid; duplicates are skipped.")
            try:
                c["conv_kspacings"] = tuple(float(x) for x in ks.replace(",", " ").split())
            except ValueError:
                st.error("Write the spacings as numbers separated by spaces.")
            st.caption("The cutoff series uses the k-grid from the k-points tab; the k-point series uses the cutoffs "
                       "from the Electrons tab. Run the cutoff series first, set the converged cutoff, then run the "
                       "k-point series.")

    # ---------- atoms
    with t_atoms:
        st.caption("Change a **Label** to split one element into different species (for example Fe1 and Fe2 for "
                   "antiferromagnetic order or different U values). Tick **Fix** to freeze a coordinate during "
                   "relaxation or MD. Fixed atoms are shown in red in the 3D view.")
        if akey not in ss:
            frac = qs.fractional()
            lab0, fix0 = list(qs.symbols), qs.fixed.copy()
            la = loaded_tables.get("atoms")
            if la and len(la) == qs.n:
                lab0 = [r["Label"] for r in la]
                fix0 = np.array([[r["Fix x"], r["Fix y"], r["Fix z"]] for r in la], dtype=bool)
            ss[akey] = pd.DataFrame({
                "#": range(1, qs.n + 1), "Element": qs.symbols, "Label": lab0,
                "a (frac)": frac[:, 0].round(6), "b (frac)": frac[:, 1].round(6), "c (frac)": frac[:, 2].round(6),
                "Fix x": fix0[:, 0], "Fix y": fix0[:, 1], "Fix z": fix0[:, 2]})
            ss[akey + "::v"] = 0
        c1, c2, c3 = st.columns([1.2, 1, 1])
        with c1:
            below = num("Fix all atoms with fractional c below", 0.0, 1.0, 0.0, "q_fixbelow", step=0.01)
        if c2.button("Fix them"):
            df = ss[akey].copy()
            df.loc[df["c (frac)"] < below, ["Fix x", "Fix y", "Fix z"]] = True
            ss[akey] = df
            ss[akey + "::v"] += 1
        if c3.button("Unfix all"):
            df = ss[akey].copy()
            df[["Fix x", "Fix y", "Fix z"]] = False
            ss[akey] = df
            ss[akey + "::v"] += 1
        atoms = st.data_editor(ss[akey], key=f"ed_{akey}::{ss[akey + '::v']}", hide_index=True,
                               disabled=["#", "Element", "a (frac)", "b (frac)", "c (frac)"])
        labels = [str(x).strip() or el for x, el in zip(atoms["Label"], atoms["Element"])]
        fixed = atoms[["Fix x", "Fix y", "Fix z"]].to_numpy(bool)

    # ---------- species and pseudopotential library
    with t_sp:
        c["pseudo_type"] = radio("Pseudopotential type", ["US/PAW", "NC"], "US/PAW", "q_ptype", horizontal=True,
                                 format_func=lambda t: "Ultrasoft / PAW" if t == "US/PAW" else "Norm-conserving",
                                 help="Sets the recommended ecutrho and the checks.")
        mult = 8 if c["pseudo_type"] == "US/PAW" else 4
        with st.expander("Use a pseudopotential library (SSSP, PseudoDojo or your own CSV)"):
            st.caption("Upload the library’s description file: SSSP .json (from the SSSP website), PseudoDojo .djson, "
                       "or a CSV with columns element, filename, ecutwfc, ecutrho (Ry). The app fills in the file names "
                       "and the recommended cutoffs (the highest among your elements).")
            libf = st.file_uploader("Library file", type=["json", "djson", "csv"], key="up_pslib")
            if libf is not None:
                try:
                    lib = PL.parse_library(libf.getvalue(), libf.name)
                    rows, missing, wfc, rho = PL.recommend(lib, qs.symbols)
                    st.dataframe(pd.DataFrame(rows), hide_index=True)
                    if missing:
                        st.warning(f"Not in this library: {', '.join(missing)}. Set their files by hand.")
                    st.markdown(f"Recommended for this structure: ecutwfc = **{wfc} Ry**, ecutrho = **{rho} Ry**")

                    def _apply_lib(lib=lib, wfc=wfc, rho=rho, mult=mult):
                        ss["pseudo_override"] = {el: d["file"] for el, d in lib.items()}
                        if wfc:
                            ss["q_ecut"] = float(wfc)
                            ss[f"q_ecutrho::{mult}::{float(wfc)}"] = float(rho) if rho else float(mult * wfc)

                    st.button("Apply to this calculation", on_click=_apply_lib, type="primary")
                except Exception as e:
                    st.error(str(e))
        template = txt("Pseudopotential file name pattern ({El} is replaced by the element)", "{El}.upf", "q_ptemplate",
                       help="Names must match the files in pseudo_dir exactly. You can still edit each one in the table.")
        override = ss.get("pseudo_override", {})
        template_changed = ss.get("species_template_prev", template) != template
        ss["species_template_prev"] = template
        base_rows = Q.species_table(qs, labels)
        skey = (f"species::{akey}::{'|'.join(r['Label'] for r in base_rows)}::{template}::"
                f"{hash(tuple(sorted(override.items())))}")
        if skey not in ss:
            last = dict(ss.get("species_last", {}))
            for r in loaded_tables.get("species") or []:
                last.setdefault(r["Label"], r)
            for r in base_rows:
                keep = last.get(r["Label"])
                if keep and keep.get("Element") == r["Element"]:
                    r.update({k: keep[k] for k in ("Mass", "Start. magnetisation", "Hubbard U (eV)", "U manifold")
                              if k in keep})
                if override.get(r["Element"]):
                    r["Pseudopotential"] = override[r["Element"]]
                elif keep and keep.get("Pseudopotential") and not template_changed:
                    r["Pseudopotential"] = keep["Pseudopotential"]
                else:
                    r["Pseudopotential"] = template.replace("{El}", r["Element"])
            ss[skey] = pd.DataFrame(base_rows)
        sp_df = st.data_editor(ss[skey], key=f"ed_{skey}", hide_index=True, disabled=["Label", "Element"],
                               column_config={
                                   "Start. magnetisation": st.column_config.NumberColumn(
                                       help="Between −1 and 1. Only used with spin polarisation."),
                                   "Hubbard U (eV)": st.column_config.NumberColumn(help="0 = no Hubbard U."),
                                   "U manifold": st.column_config.TextColumn(help="For example 3d, 4f, 2p."),
                               })
        species = sp_df.to_dict("records")
        ss["species_last"] = {r["Label"]: r for r in species}

    # ---------- electrons
    with t_el:
        c1, c2, c3 = st.columns(3)
        with c1:
            c["ecutwfc"] = num("ecutwfc (Ry)", 10.0, 500.0, 50.0, "q_ecut", step=5.0)
            c["ecutrho"] = num(f"ecutrho (Ry), usually {mult}× ecutwfc", 40.0, 5000.0, float(mult * c["ecutwfc"]),
                               f"q_ecutrho::{mult}::{float(c['ecutwfc'])}", step=10.0)
            c["occupations"] = select("occupations", ["smearing", "fixed", "tetrahedra", "tetrahedra_opt"], "smearing", "q_occ",
                                      help="Smearing for metals and unknown systems; fixed only for insulators.")
            if c["occupations"] == "smearing":
                c["smearing"] = select("smearing", ["mv", "gaussian", "mp", "fd"], "mv", "q_smear")
                c["degauss"] = num("degauss (Ry)", 0.0001, 0.1, 0.01, "q_deg", step=0.005, format="%.4f")
        with c2:
            c["spin"] = select("Spin", ["none", "collinear", "noncollinear"], "none", "q_spin",
                               format_func=lambda x: {"none": "Not spin-polarised", "collinear": "Spin-polarised (nspin = 2)",
                                                      "noncollinear": "Non-collinear"}[x])
            if c["spin"] == "noncollinear":
                c["lspinorb"] = chk("Spin–orbit coupling", False, "q_soc")
            if c["spin"] == "collinear" and chk("Fix total magnetisation", False, "q_totmagon"):
                c["tot_magnetization"] = num("tot_magnetization (μB)", -100.0, 100.0, 0.0, "q_totmag")
            st.caption("Set starting magnetisations per species in the Species tab.")
            c["nbnd"] = num("nbnd (0 = automatic)", 0, 100000, 0, "q_nbnd")
            c["tot_charge"] = num("tot_charge (+1 = one electron removed)", -100.0, 100.0, 0.0, "q_charge", step=1.0)
        with c3:
            c["conv_thr"] = num("conv_thr (Ry)", 1e-14, 1e-3, 1e-8, "q_conv", format="%.0e")
            c["mixing_beta"] = num("mixing_beta", 0.01, 1.0, 0.4, "q_beta", step=0.05,
                                   help="Lower it (0.1–0.2) if the SCF doesn’t converge.")
            c["mixing_mode"] = select("mixing_mode", ["plain", "TF", "local-TF"], "plain", "q_mixmode",
                                      help="local-TF helps slabs and inhomogeneous systems.")
            c["electron_maxstep"] = num("electron_maxstep", 1, 10000, 100, "q_maxstep")
            c["diagonalization"] = select("diagonalization", ["david", "cg", "ppcg", "paro", "rmm-davidson"], "david", "q_diag")
            c["startingwfc"] = select("startingwfc", ["atomic+random", "atomic", "random", "file"], "atomic+random", "q_swfc")
            c["scf_must_converge"] = chk("Stop if the SCF doesn’t converge", True, "q_mustconv")

    # ---------- functional and corrections
    with t_func:
        c1, c2 = st.columns(2)
        with c1:
            c["functional"] = select("Exchange–correlation functional", list(Q.FUNCTIONALS), "(from pseudopotential)", "q_func",
                                     help="“From pseudopotential” uses the functional the pseudopotentials were made with.")
            if Q.FUNCTIONALS[c["functional"]] in Q.HYBRIDS:
                c["exx_nq"] = num("q-grid for exact exchange (nqx1 = nqx2 = nqx3)", 1, 12, 1, "q_nqx")
                c["ecutfock"] = num("ecutfock (Ry, 0 = default)", 0.0, 2000.0, 0.0, "q_efock")
                c["exxdiv"] = select("exxdiv_treatment", ["gygi-baldereschi", "vcut_spherical", "vcut_ws", "none"],
                                     "gygi-baldereschi", "q_exxdiv")
            c["vdw"] = select("Dispersion correction (vdw_corr)", ["none", "grimme-d2", "grimme-d3", "ts", "mbd", "xdm"],
                              "none", "q_vdw")
            if c["vdw"] == "grimme-d3":
                c["d3_version"] = select("D3 damping", [4, 3], 4, "q_d3",
                                         format_func=lambda v: "Becke–Johnson (BJ)" if v == 4 else "Zero damping")
            st.markdown("**DFT+U**")
            c["hubbard_style"] = radio("Hubbard input format", ["card", "legacy"], "card", "q_hubstyle", horizontal=True,
                                       format_func=lambda h: "HUBBARD card (QE 7.1+)" if h == "card" else "Legacy (older QE)")
            c["hubbard_projector"] = select("Projector", ["ortho-atomic", "atomic"], "ortho-atomic", "q_hubproj")
            st.caption("Set U values per species in the Species tab.")
        with c2:
            st.markdown("**Isolated systems and slabs**")
            c["isolated"] = select("assume_isolated", ["none", "mt", "makov-payne", "esm"],
                                   "mt" if not struct.periodic else "none", "q_iso",
                                   format_func=lambda x: {"none": "None (periodic)", "mt": "Martyna–Tuckerman (molecules)",
                                                          "makov-payne": "Makov–Payne (charged molecules, cubic cells)",
                                                          "esm": "ESM (slabs)"}[x])
            if c["isolated"] == "esm":
                c["esm_bc"] = select("esm_bc", ["bc1", "bc2", "bc3"], "bc1", "q_esmbc")
            c["dipole"] = chk("Dipole correction (slabs)", False, "q_dip")
            if c["dipole"]:
                c["edir"] = select("Direction normal to the slab (edir)", [1, 2, 3],
                                   (slab_axis + 1) if slab_axis is not None else 3, "q_edir")
                c["emaxpos"] = num("emaxpos (fraction, inside the vacuum)", 0.0, 1.0, 0.9, "q_emax")
                c["eopreg"] = num("eopreg", 0.0, 1.0, 0.05, "q_eop")
            st.markdown("**Symmetry**")
            c["nosym"] = chk("Turn off symmetry (nosym)", False, "q_nosym", help="Recommended for MD.")
            c["noinv"] = chk("No time-reversal symmetry (noinv)", False, "q_noinv")

    # ---------- k-points
    with t_k:
        c["k_mode"] = select("k-points", ["automatic", "gamma"], "gamma" if not struct.periodic else "automatic", "q_kmode",
                             format_func=lambda m: "Monkhorst–Pack grid" if m == "automatic" else "Γ point only (gamma)")
        if c["k_mode"] == "automatic":
            c1, c2 = st.columns(2)
            with c1:
                spacing = num("k-point spacing (Å⁻¹, 2π included)", 0.05, 2.0, 0.25, "q_kspace", step=0.05,
                              help="Smaller = denser. 0.2–0.3 is typical; metals need denser grids.")
                auto = S.kgrid_from_spacing(qs.cell, spacing, slab_axis)
                st.caption(f"Suggested grid: {auto[0]} × {auto[1]} × {auto[2]}"
                           + (f" (1 along the vacuum direction, axis {slab_axis + 1})" if slab_axis is not None else ""))
            with c2:
                g = st.columns(3)
                kg = []
                for i in range(3):
                    with g[i]:
                        kg.append(num(f"n{i + 1}", 1, 100, auto[i], f"q_k{i}::{auto[i]}"))
                c["kgrid"] = tuple(kg)
                h = st.columns(3)
                ksh = []
                for i in range(3):
                    with h[i]:
                        ksh.append(select(f"shift {i + 1}", [0, 1], 0, f"q_ks{i}"))
                c["kshift"] = tuple(ksh)
        needs_path = c["workflow"] in ("bands", "phonon") or c.get("calculation") == "bands"
        path_rows = None
        if needs_path:
            st.markdown("**High-symmetry path** (bands and phonon dispersion)")
            c["path_points"] = num("Points per segment", 2, 500, 30, "q_pathpts")
            path = None
            if HAVE_ASE:
                try:
                    path = S.band_path(qs.cell, int(c["path_points"]))
                    st.caption("Found automatically with ASE: " + " → ".join(p[0] for p in path)
                               + ". Edit it below if you want a different path.")
                except Exception as e:
                    st.warning(f"Couldn’t find the path automatically: {e}")
            else:
                st.info("Install ASE (pip install ase) for automatic paths. Until then, enter the path by hand in "
                        "crystal coordinates of the reciprocal lattice.")
            rows = loaded_tables.get("path") or [
                {"Label": p[0], "k1": p[1][0], "k2": p[1][1], "k3": p[1][2], "Points to next": p[2]} for p in (path or [])
            ] or [{"Label": "G", "k1": 0.0, "k2": 0.0, "k3": 0.0, "Points to next": int(c["path_points"])},
                  {"Label": "X", "k1": 0.0, "k2": 0.5, "k3": 0.0, "Points to next": 1}]
            pkey = f"ed_path::{akey}::{int(c['path_points'])}::{bool(path)}"
            path_df = st.data_editor(pd.DataFrame(rows), key=pkey, num_rows="dynamic", hide_index=True)
            st.caption("Set “Points to next” to 1 where the path jumps (a | in the usual notation).")
            path_rows = path_df.dropna().to_dict("records")
            c["path"] = [(str(r["Label"]), (float(r["k1"]), float(r["k2"]), float(r["k3"])), int(r["Points to next"]))
                         for r in path_rows] or None

    # ---------- relaxation, MD, cell
    with t_ions:
        wf = c["workflow"]
        calc = c["calculation"] if wf == "single" else ("md" if wf == "md" else "scf")
        if wf == "md":
            st.markdown("**Molecular dynamics** (Born–Oppenheimer, pw.x)")
            m1, m2, m3 = st.columns(3)
            with m1:
                c["md_ensemble"] = radio("Ensemble", ["NVE", "NVT", "NPT"], "NVT", "q_mdens", horizontal=True,
                                         help="NVE: constant energy. NVT: thermostat. NPT: variable cell (vc-md).")
                c["md_dt_fs"] = num("Time step (fs)", 0.05, 10.0, 1.0, "q_mddt", step=0.25,
                                    help=f"Converted to Rydberg atomic units for QE (1 a.u. = {Q.RY_AU_FS} fs).")
                c["nstep"] = num("Number of steps (nstep)", 1, 10_000_000, 2000, "q_nstep")
                st.caption(f"Total time: {c['nstep'] * c['md_dt_fs'] / 1000:.3g} ps")
            with m2:
                if c["md_ensemble"] == "NVT":
                    c["ion_dynamics"] = select("Integrator", ["verlet", "langevin"], "verlet", "q_mdint")
                if c["md_ensemble"] != "NVE":
                    if c.get("ion_dynamics") != "langevin":
                        c["ion_temperature"] = select("Thermostat (ion_temperature)",
                                                      ["svr", "berendsen", "andersen", "rescaling", "rescale-v",
                                                       "rescale-T", "reduce-T", "initial"], "svr", "q_mdthermo",
                                                      help="svr (stochastic velocity rescaling) samples the canonical "
                                                           "ensemble correctly; berendsen is fast but approximate.")
                    c["tempw"] = num("Temperature (K)", 0.0, 10000.0, 300.0, "q_tempw")
            with m3:
                it = c.get("ion_temperature", "")
                if it in ("rescale-v", "rescale-T", "reduce-T", "berendsen", "andersen", "svr"):
                    c["md_nraise"] = num("nraise (thermostat strength)", 1, 100000, 20, "q_nraise",
                                         help="For svr/berendsen the time constant is nraise × dt.")
                if it == "rescaling":
                    c["md_tolp"] = num("tolp (K)", 0.1, 1000.0, 50.0, "q_tolp")
                if it in ("rescale-T", "reduce-T"):
                    c["md_delta_t"] = num("delta_t", -1000.0, 1000.0, 1.0, "q_deltat")
            if c["md_ensemble"] == "NPT":
                n1_, n2_ = st.columns(2)
                with n1_:
                    c["cell_dynamics"] = select("cell_dynamics", ["pr", "w"], "pr", "q_mdcell",
                                                format_func=lambda x: {"pr": "Parrinello–Rahman", "w": "Wentzcovitch"}[x])
                with n2_:
                    c["press"] = num("Pressure (kbar)", -1000.0, 10000.0, 0.0, "q_mdpress")
            st.caption("Car–Parrinello MD uses a different program (cp.x) and isn’t covered here.")
        else:
            if calc not in ("relax", "vc-relax", "md", "vc-md"):
                st.info("These settings are used for relax, vc-relax, md and vc-md calculations, and the MD workflow.")
            c1, c2, c3 = st.columns(3)
            with c1:
                c["nstep"] = num("nstep (max ionic steps)", 1, 10_000_000, 100, "q_nstep")
                c["etot_conv_thr"] = num("etot_conv_thr (Ry)", 1e-10, 1e-1, 1e-5, "q_etot", format="%.0e")
                c["forc_conv_thr"] = num("forc_conv_thr (Ry/Bohr)", 1e-8, 1e-1, 1e-4, "q_forc", format="%.0e",
                                         help="1e-4 Ry/Bohr ≈ 0.0026 eV/Å; 1e-3 is a common looser value.")
            with c2:
                ion_opts = {"relax": ["bfgs", "damp", "fire"], "vc-relax": ["bfgs", "damp"], "md": ["verlet", "langevin"],
                            "vc-md": ["beeman"]}.get(calc, ["bfgs"])
                c["ion_dynamics"] = select("ion_dynamics", ion_opts, ion_opts[0], f"q_iondyn::{calc}")
                if calc in ("md", "vc-md"):
                    c["dt"] = num("dt (Rydberg a.u.; 1 = 0.0484 fs)", 1.0, 200.0, 20.0, "q_dt")
                    c["ion_temperature"] = select("ion_temperature", ["not_controlled", "svr", "berendsen", "andersen",
                                                                      "rescaling", "initial"], "not_controlled", "q_iontemp")
                    c["tempw"] = num("tempw (K)", 0.0, 10000.0, 300.0, "q_tempw")
            with c3:
                cell_opts = {"vc-relax": ["bfgs", "damp-w", "damp-pr"], "vc-md": ["pr", "w"]}.get(calc, ["bfgs"])
                c["cell_dynamics"] = select("cell_dynamics", cell_opts, cell_opts[0], f"q_celldyn::{calc}")
                c["press"] = num("press (kbar)", -1000.0, 10000.0, 0.0, "q_press")
                c["press_conv_thr"] = num("press_conv_thr (kbar)", 0.01, 50.0, 0.5, "q_pressthr")
                c["cell_dofree"] = select("cell_dofree (which cell parameters can change)",
                                          ["all", "ibrav", "shape", "volume", "x", "y", "z", "xy", "xz", "yz", "xyz",
                                           "2Dxy", "2Dshape", "epitaxial_ab", "epitaxial_ac", "epitaxial_bc"],
                                          "2Dxy" if slab_axis == 2 else "all", "q_dofree",
                                          help="For slabs, 2Dxy relaxes only the in-plane cell.")
        c["max_seconds"] = num("max_seconds (0 = no limit)", 0.0, 1e7, 0.0, "q_maxsec",
                               help="Stops cleanly before your cluster’s time limit, so you can restart.")

    # ---------- post-processing
    with t_post:
        wf = c["workflow"]
        if wf in ("single", "md", "convergence"):
            st.info("Choose the band structure, DOS, plot, phonon or NEB workflow to set up post-processing.")
        if wf == "dos":
            st.markdown("**nscf grid and DOS energy window**")
            g = st.columns(3)
            dens = tuple(max(1, 2 * int(x)) for x in c.get("kgrid", (4, 4, 4)))
            if slab_axis is not None:
                dens = tuple(1 if i == slab_axis else v for i, v in enumerate(dens))
            nk = []
            for i in range(3):
                with g[i]:
                    nk.append(num(f"nscf n{i + 1}", 1, 200, dens[i], f"q_nk{i}::{dens[i]}"))
            c["nscf_kgrid"] = tuple(nk)
            e = st.columns(3)
            with e[0]:
                c["dos_emin"] = num("Emin (eV)", -1000.0, 1000.0, -20.0, "q_emin")
            with e[1]:
                c["dos_emax"] = num("Emax (eV)", -1000.0, 1000.0, 10.0, "q_emax2")
            with e[2]:
                c["dos_de"] = num("DeltaE (eV)", 0.001, 1.0, 0.01, "q_de")
        if wf == "pp":
            c["pp_plot"] = select("What to plot (plot_num)", list(Q.PP_PLOTS), 0, "q_plotnum",
                                  format_func=lambda p: f"{p}: {Q.PP_PLOTS[p]}")
            if c["pp_plot"] == 7:
                k1, k2 = st.columns(2)
                with k1:
                    c["pp_kpoint"] = num("k-point number", 1, 100000, 1, "q_ppk")
                with k2:
                    c["pp_kband"] = num("Band number", 1, 100000, 1, "q_ppb")
            if c["pp_plot"] == 10:
                k1, k2 = st.columns(2)
                with k1:
                    c["pp_emin"] = num("emin (eV)", -1000.0, 1000.0, -1.0, "q_ppemin")
                with k2:
                    c["pp_emax"] = num("emax (eV)", -1000.0, 1000.0, 0.0, "q_ppemax")
            c["pp_format"] = radio("Output format", [6, 5], 6, "q_ppfmt", horizontal=True,
                                   format_func=lambda f: "Gaussian cube (.cube)" if f == 6 else "XCrySDen (.xsf)")
        if wf == "phonon":
            c["ph_gamma_only"] = chk("Only the Γ point (no dispersion)", False, "q_phgamma")
            if not c["ph_gamma_only"]:
                g = st.columns(3)
                qg = []
                for i in range(3):
                    with g[i]:
                        qg.append(num(f"nq{i + 1}", 1, 20, 2, f"q_nq{i}"))
                c["ph_qgrid"] = tuple(qg)
                c["ph_dispersion"] = chk("Phonon dispersion along the path", True, "q_phdisp")
                c["ph_dos"] = chk("Phonon DOS", True, "q_phdos")
                if c["ph_dos"]:
                    g = st.columns(3)
                    dg = []
                    for i in range(3):
                        with g[i]:
                            dg.append(num(f"DOS grid n{i + 1}", 1, 200, 20, f"q_pdg{i}"))
                    c["ph_dos_grid"] = tuple(dg)
            c["ph_tr2"] = num("tr2_ph", 1e-20, 1e-8, 1e-14, "q_tr2", format="%.0e")
            c["ph_epsil"] = chk("Dielectric tensor and Born charges (insulators only)", False, "q_epsil")
            c["ph_asr"] = select("Acoustic sum rule", ["crystal", "simple", "one-dim", "zero-dim", "no"], "crystal", "q_asr")
        if wf == "neb":
            if final is None:
                st.info("Upload the final structure in the sidebar.")
            n1_, n2_, n3_ = st.columns(3)
            with n1_:
                c["neb_images"] = num("num_of_images (including end points)", 3, 100, 7, "q_nebimg")
                c["neb_opt"] = select("opt_scheme", ["broyden", "broyden2", "quick-min", "sd", "langevin"], "broyden", "q_nebopt")
            with n2_:
                c["neb_nstep"] = num("nstep_path", 1, 10000, 100, "q_nebstep")
                c["neb_ci"] = select("Climbing image", ["auto", "no-CI", "manual"], "auto", "q_nebci")
            with n3_:
                c["neb_thr"] = num("path_thr (eV/Å)", 0.001, 1.0, 0.05, "q_nebthr")
                c["neb_ds"] = num("ds", 0.1, 5.0, 1.0, "q_nebds")

    # ---------- extra
    with t_extra:
        st.markdown("**Any other variable** (written exactly as typed, in Fortran syntax)")
        st.caption("Examples: SYSTEM · starting_charge(1) · 0.5  —  ELECTRONS · conv_thr_init · 1d-3  —  "
                   "INPUTPH · electron_phonon · 'interpolated'")
        extra_rows = loaded_tables.get("extra") or [{"Namelist": "SYSTEM", "Variable": "", "Value": ""}]
        extra = st.data_editor(
            pd.DataFrame(extra_rows), num_rows="dynamic", hide_index=True, key=f"ed_extra::{token}",
            column_config={"Namelist": st.column_config.SelectboxColumn(
                options=["CONTROL", "SYSTEM", "ELECTRONS", "IONS", "CELL", "INPUTPH", "DOS", "PROJWFC", "INPUTPP",
                         "PLOT", "BANDS", "PATH"])})
        c["extra_vars"] = extra.dropna().to_dict("records")
        c["extra_cards"] = area("Extra cards for pw.x (added at the end)", "", "q_cards", height=120,
                                help="For example CONSTRAINTS, OCCUPATIONS, ADDITIONAL_K_POINTS or SOLVENTS.")

    files, errors, warnings, commands = Q.build_qe(c, qs, species, labels, fixed, qfinal)
    zip_name = f"{c['prefix']}_qe.zip"
    program_tag = "qe"
    highlight = [i for i in range(qs.n) if fixed[i].any()]
    view_struct = qs
    tables = {"atoms": atoms[["Label", "Fix x", "Fix y", "Fix z"]].to_dict("records"), "species": species,
              "extra": c["extra_vars"], "path": path_rows}

# ---------------------------------------------------------------------
# 3D view (filled now that the fixed atoms are known)
# ---------------------------------------------------------------------
with view_slot:
    v1, v2, v3 = st.columns(3)
    show_labels = v1.checkbox("Show atom numbers", key="view_labels")
    rep = v2.selectbox("Repeat cell", ["1×1×1", "2×2×1", "2×2×2", "3×3×1"], key="view_rep",
                       disabled=not view_struct.periodic)
    v3.caption("Red = fixed atoms" if highlight else "Drag to rotate, scroll to zoom")
    components.html(S.viewer_html(view_struct, highlight, tuple(int(x) for x in rep.split("×")), height=400,
                                  labels=show_labels), height=410)
    if show_labels:
        st.caption("Labels count from 0, like ORCA." if program == "ORCA"
                   else "Labels count from 0; QE’s atom list (Atoms tab) counts from 1.")

# ---------------------------------------------------------------------
# Cluster script, results and downloads
# ---------------------------------------------------------------------
st.divider()
with st.expander("Run on a cluster (SLURM / PBS job script)"):
    j = {}
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        j["scheduler"] = select("Scheduler", ["slurm", "pbs", "none"], "slurm", "j_sched",
                                format_func=lambda x: {"slurm": "SLURM", "pbs": "PBS / Torque", "none": "None (plain script)"}[x])
        j["job_name"] = txt("Job name", "dft", "j_name")
    with c2:
        j["nodes"] = num("Nodes", 1, 10000, 1, "j_nodes")
        j["ntasks_per_node"] = num("MPI tasks per node", 1, 1024,
                                   int(ss.get("o_np", 4)) if program == "ORCA" else 4, "j_ntasks")
        j["cpus_per_task"] = num("Threads per task", 1, 256, 1, "j_cpus")
    with c3:
        j["time"] = txt("Time limit (HH:MM:SS)", "24:00:00", "j_time")
        j["partition"] = txt("Partition / queue", "", "j_part")
        j["account"] = txt("Account / project", "", "j_acct")
    with c4:
        j["memory"] = txt("Memory (e.g. 64G, optional)", "", "j_mem")
        j["email"] = txt("Email for notifications (optional)", "", "j_email")
        j["launcher"] = select("MPI launcher", ["mpirun", "srun"], "mpirun", "j_launcher")
    j["modules"] = area("Modules to load, one per line", "", "j_modules", height=80,
                        help="For example quantum-espresso/7.3 or orca/6.0. Lines starting with module, source, "
                             "export or conda are kept as they are.")
    j["setup"] = area("Other setup lines (optional)", "", "j_setup", height=60)
    jname, jtext, submit, jwarn = J.make_script(j, commands, program_tag)
    files[jname] = jtext
    warnings += jwarn
    if program == "ORCA" and int(j["nodes"]) * int(j["ntasks_per_node"]) != int(ss.get("o_np", 4)):
        warnings.append("The job script’s MPI tasks (nodes × tasks per node) don’t match ORCA’s %pal nprocs.")
    st.caption(f"Submit with: `{submit}`. Save your settings (below) to reuse this cluster setup next time.")

st.subheader("Your input files")
if errors:
    for e in errors:
        st.error(e)
else:
    st.success("No problems found. Still read through the files before running them.")
for w in warnings:
    st.warning(w)

if files:
    d1, d2 = st.columns(2)
    d1.download_button("Download all files (.zip)", zip_files(files), file_name=zip_name, mime="application/zip",
                       type="primary")
    d2.download_button("Save these settings (.json)", SIO.dump(dict(ss), program, tables).encode("utf-8"),
                       file_name="dft_input_settings.json", mime="application/json",
                       help="Load it later from the sidebar to get exactly the same setup.")
    ftabs = st.tabs(list(files))
    for tab, (name, text) in zip(ftabs, files.items()):
        with tab:
            st.code(text, language="bash" if name.endswith((".sh", ".slurm", ".pbs")) else
                    "python" if name.endswith(".py") else None)
            st.download_button(f"Download {name}", text.encode("utf-8"), file_name=name, mime="text/plain",
                               key=f"dl::{name}")
