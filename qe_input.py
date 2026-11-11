"""
qe_input.py — builds Quantum ESPRESSO inputs (pw.x, bands.x, dos.x, projwfc.x, pp.x, ph.x,
q2r.x, matdyn.x, neb.x) from a settings dictionary, and checks them.

build_qe(settings, structure, final_structure=None) -> (files, errors, warnings)
"""

import numpy as np

from elements import MASSES, normalize_symbol

WORKFLOWS = {
    "single": "Single pw.x calculation",
    "bands": "Band structure (scf → bands → bands.x)",
    "dos": "DOS and projected DOS (scf → nscf → dos.x + projwfc.x)",
    "pp": "Charge density / potential maps (scf → pp.x)",
    "phonon": "Phonons (scf → ph.x → q2r.x → matdyn.x)",
    "neb": "Reaction path, NEB (neb.x)",
    "md": "Molecular dynamics (pw.x, Born–Oppenheimer)",
    "convergence": "Convergence tests (cutoff and k-points)",
}

RY_AU_FS = 0.048378  # one Rydberg atomic unit of time, in fs

CALCULATIONS = ["scf", "relax", "vc-relax", "nscf", "bands", "md", "vc-md"]

FUNCTIONALS = {
    "(from pseudopotential)": "",
    "PBE": "pbe", "PBEsol": "pbesol", "LDA (PZ)": "pz", "revPBE": "revpbe", "BLYP": "blyp",
    "SCAN (needs libxc)": "scan", "r2SCAN (needs libxc)": "r2scan",
    "PBE0 (hybrid)": "pbe0", "HSE06 (hybrid)": "hse", "B3LYP (hybrid)": "b3lyp",
    "vdW-DF (nonlocal)": "vdw-df", "vdW-DF2 (nonlocal)": "vdw-df2", "vdW-DF2-B86R (nonlocal)": "vdw-df2-b86r",
    "rVV10 (nonlocal)": "rvv10",
}
HYBRIDS = {"pbe0", "hse", "b3lyp"}

PP_PLOTS = {
    0: "Electron (pseudo-)charge density",
    1: "Total potential (V_bare + V_H + V_xc)",
    2: "Local ionic potential (V_bare)",
    6: "Spin polarisation (ρ↑ − ρ↓)",
    7: "|ψ|² of one band at one k-point",
    8: "Electron localisation function (ELF)",
    10: "Integrated local DOS between two energies",
    11: "V_bare + V_H (for work functions)",
    17: "All-electron valence charge (PAW only)",
}

DEFAULTS = dict(
    workflow="single", calculation="scf", prefix="pwscf", outdir="./out", pseudo_dir="./pseudo",
    verbosity="high", coord_units="crystal", vacuum=10.0,
    ecutwfc=50.0, ecutrho=400.0, pseudo_type="US/PAW",
    occupations="smearing", smearing="mv", degauss=0.01, nbnd=0,
    spin="none", lspinorb=False, tot_magnetization=None, tot_charge=0.0,
    functional="(from pseudopotential)", exx_nq=1, ecutfock=0.0, exxdiv="gygi-baldereschi",
    vdw="none", d3_version=4,
    hubbard_style="card", hubbard_projector="ortho-atomic",
    isolated="none", esm_bc="bc1", dipole=False, edir=3, emaxpos=0.9, eopreg=0.05,
    nosym=False, noinv=False,
    conv_thr=1e-8, mixing_beta=0.4, mixing_mode="plain", electron_maxstep=100,
    diagonalization="david", startingwfc="atomic+random", scf_must_converge=True,
    etot_conv_thr=1e-5, forc_conv_thr=1e-4, nstep=100, max_seconds=0, disk_io="default",
    ion_dynamics="bfgs", dt=20.0, tempw=300.0, ion_temperature="not_controlled",
    cell_dynamics="bfgs", press=0.0, press_conv_thr=0.5, cell_dofree="all",
    k_mode="automatic", kgrid=(4, 4, 4), kshift=(0, 0, 0),
    path=None, path_points=30,
    nscf_kgrid=(8, 8, 8), dos_emin=-20.0, dos_emax=10.0, dos_de=0.01,
    pp_plot=0, pp_kpoint=1, pp_kband=1, pp_emin=-1.0, pp_emax=0.0, pp_format=6,
    ph_qgrid=(2, 2, 2), ph_gamma_only=False, ph_tr2=1e-14, ph_epsil=False, ph_asr="crystal",
    ph_dos=True, ph_dos_grid=(20, 20, 20), ph_dispersion=True,
    md_ensemble="NVT", md_dt_fs=1.0, md_nraise=20, md_tolp=50.0, md_delta_t=1.0,
    conv_mode="both", conv_ecut_start=30.0, conv_ecut_stop=80.0, conv_ecut_step=10.0,
    conv_kspacings=(0.5, 0.4, 0.3, 0.25, 0.2, 0.15),
    neb_images=7, neb_nstep=100, neb_opt="broyden", neb_ci="auto", neb_thr=0.05, neb_ds=1.0,
    nprocs=4, extra_vars=None, extra_cards="",
)

HARTREE = {"ry": 1.0}


# ---------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------

def fortran(v):
    if isinstance(v, bool) or isinstance(v, np.bool_):
        return ".true." if v else ".false."
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e6):
            mant, exp = f"{v:.6e}".split("e")
            mant = mant.rstrip("0").rstrip(".") if "." in mant else mant
            return f"{mant}d{int(exp)}"
        return repr(v) if "." in repr(v) or "e" in repr(v) else repr(v) + ".0"
    if isinstance(v, RawValue):
        return v.text
    return "'" + str(v).replace("'", "''") + "'"


class RawValue:
    """A value written exactly as given (used for extra variables typed by the user)."""
    def __init__(self, text):
        self.text = str(text).strip()


def namelist(name, items):
    lines = [f"&{name}"]
    for k, v in items:
        if v is None:
            continue
        lines.append(f"  {k} = {fortran(v)}")
    lines.append("/")
    return "\n".join(lines)


def default_manifold(element):
    """Usual Hubbard manifold for an element: 3d/4d/5d metals, 4f/5f, or p for O, N, etc."""
    from elements import Z
    z = Z[element]
    if 21 <= z <= 30:
        return "3d"
    if 39 <= z <= 48:
        return "4d"
    if 72 <= z <= 80 or z == 57:
        return "5d"
    if 58 <= z <= 71:
        return "4f"
    if 89 <= z <= 103:
        return "5f"
    if element in ("C", "N", "O", "F"):
        return "2p"
    if element in ("Si", "P", "S", "Cl"):
        return "3p"
    return "?"


def species_table(struct, labels=None):
    """Default species table rows: one per label (element unless the user split species)."""
    labels = labels or list(struct.symbols)
    rows, seen = [], []
    for lab, el in zip(labels, struct.symbols):
        if lab in seen:
            continue
        seen.append(lab)
        rows.append({"Label": lab, "Element": el, "Mass": MASSES[el], "Pseudopotential": f"{el}.upf",
                     "Start. magnetisation": 0.0, "Hubbard U (eV)": 0.0, "U manifold": default_manifold(el)})
    return rows


# ---------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------

def _cards_species(species):
    lines = ["ATOMIC_SPECIES"]
    for sp in species:
        lines.append(f"  {sp['Label']:<4s} {float(sp['Mass']):10.4f}  {sp['Pseudopotential']}")
    return "\n".join(lines)


def _cards_cell(struct):
    lines = ["CELL_PARAMETERS angstrom"]
    lines += [f"  {v[0]:14.8f} {v[1]:14.8f} {v[2]:14.8f}" for v in struct.cell]
    return "\n".join(lines)


def _cards_positions(struct, labels, fixed, units, show_flags, header=True):
    lines = [f"ATOMIC_POSITIONS {units}"] if header else []
    coords = struct.fractional() if units == "crystal" else struct.positions
    for lab, c, fx in zip(labels, coords, fixed):
        flags = "  " + " ".join("0" if f else "1" for f in fx) if show_flags else ""
        lines.append(f"  {lab:<4s} {c[0]:14.8f} {c[1]:14.8f} {c[2]:14.8f}{flags}")
    return "\n".join(lines)


def _cards_kpoints(mode, grid=None, shift=None, path=None):
    if mode == "gamma":
        return "K_POINTS gamma"
    if mode == "automatic":
        return "K_POINTS automatic\n  " + " ".join(str(int(x)) for x in grid) + "  " + " ".join(str(int(x)) for x in shift)
    if mode == "path":
        lines = ["K_POINTS crystal_b", f"  {len(path)}"]
        for lab, k, w in path:
            lines.append(f"  {k[0]:12.8f} {k[1]:12.8f} {k[2]:12.8f}  {int(w)}  ! {lab}")
        return "\n".join(lines)
    raise ValueError(mode)


def _hubbard_card(species, projector):
    rows = [sp for sp in species if float(sp["Hubbard U (eV)"]) > 0]
    if not rows:
        return ""
    lines = [f"HUBBARD {projector}"]
    lines += [f"  U {sp['Label']}-{sp['U manifold']} {float(sp['Hubbard U (eV)'])}" for sp in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------------
# pw.x
# ---------------------------------------------------------------------

def _extra(c, nl):
    rows = c.get("extra_vars") or []
    return [(r["Variable"], RawValue(r["Value"])) for r in rows
            if str(r.get("Namelist", "")).upper() == nl and str(r.get("Variable", "")).strip()]


def pw_namelists(c, struct, species, calc, nbnd=None, occupations=None, engine=False):
    """The &CONTROL, &SYSTEM, &ELECTRONS (and &IONS, &CELL) namelists as text."""
    ctrl = []
    if not engine:
        ctrl.append(("calculation", calc))
    ctrl += [("prefix", c["prefix"]), ("outdir", c["outdir"]), ("pseudo_dir", c["pseudo_dir"]),
             ("verbosity", c["verbosity"])]
    if calc in ("relax", "vc-relax", "md", "vc-md") or engine:
        ctrl += [("tprnfor", True)]
    if calc in ("vc-relax", "vc-md"):
        ctrl += [("tstress", True)]
    if calc in ("relax", "vc-relax", "md", "vc-md"):
        ctrl += [("nstep", int(c["nstep"])), ("etot_conv_thr", float(c["etot_conv_thr"])),
                 ("forc_conv_thr", float(c["forc_conv_thr"]))]
    if calc in ("md", "vc-md"):
        ctrl += [("dt", float(c["dt"]))]
    if float(c["max_seconds"]) > 0:
        ctrl += [("max_seconds", float(c["max_seconds"]))]
    if c["disk_io"] != "default":
        ctrl += [("disk_io", c["disk_io"])]
    if c["dipole"]:
        ctrl += [("tefield", True), ("dipfield", True)]
    ctrl += _extra(c, "CONTROL")

    sysl = [("ibrav", 0), ("nat", struct.n), ("ntyp", len(species)),
            ("ecutwfc", float(c["ecutwfc"])), ("ecutrho", float(c["ecutrho"]))]
    occ = occupations or c["occupations"]
    sysl += [("occupations", occ)]
    if occ == "smearing":
        sysl += [("smearing", c["smearing"]), ("degauss", float(c["degauss"]))]
    nb = nbnd if nbnd is not None else int(c["nbnd"])
    if nb > 0:
        sysl += [("nbnd", nb)]
    if abs(float(c["tot_charge"])) > 0:
        sysl += [("tot_charge", float(c["tot_charge"]))]
    if c["spin"] == "collinear":
        sysl += [("nspin", 2)]
    elif c["spin"] == "noncollinear":
        sysl += [("noncolin", True)]
        if c["lspinorb"]:
            sysl += [("lspinorb", True)]
    if c["spin"] != "none":
        for i, sp in enumerate(species, 1):
            m = float(sp["Start. magnetisation"])
            if m != 0:
                sysl += [(f"starting_magnetization({i})", m)]
        if c["tot_magnetization"] is not None and c["spin"] == "collinear":
            sysl += [("tot_magnetization", float(c["tot_magnetization"]))]
    func = FUNCTIONALS.get(c["functional"], c["functional"])
    if func:
        sysl += [("input_dft", func)]
    if func in HYBRIDS:
        nq = int(c["exx_nq"])
        sysl += [("nqx1", nq), ("nqx2", nq), ("nqx3", nq), ("exxdiv_treatment", c["exxdiv"])]
        if float(c["ecutfock"]) > 0:
            sysl += [("ecutfock", float(c["ecutfock"]))]
    if c["vdw"] != "none":
        sysl += [("vdw_corr", c["vdw"])]
        if c["vdw"] == "grimme-d3":
            sysl += [("dftd3_version", int(c["d3_version"]))]
    if c["hubbard_style"] == "legacy" and any(float(sp["Hubbard U (eV)"]) > 0 for sp in species):
        sysl += [("lda_plus_u", True), ("lda_plus_u_kind", 0), ("U_projection_type", c["hubbard_projector"])]
        for i, sp in enumerate(species, 1):
            if float(sp["Hubbard U (eV)"]) > 0:
                sysl += [(f"Hubbard_U({i})", float(sp["Hubbard U (eV)"]))]
    if c["isolated"] != "none":
        sysl += [("assume_isolated", c["isolated"])]
        if c["isolated"] == "esm":
            sysl += [("esm_bc", c["esm_bc"])]
    if c["dipole"]:
        sysl += [("edir", int(c["edir"])), ("emaxpos", float(c["emaxpos"])), ("eopreg", float(c["eopreg"])),
                 ("eamp", 0.0)]
    if c["nosym"]:
        sysl += [("nosym", True)]
    if c["noinv"]:
        sysl += [("noinv", True)]
    sysl += _extra(c, "SYSTEM")

    el = [("conv_thr", float(c["conv_thr"])), ("mixing_beta", float(c["mixing_beta"])),
          ("mixing_mode", c["mixing_mode"]), ("electron_maxstep", int(c["electron_maxstep"])),
          ("diagonalization", c["diagonalization"])]
    if c["startingwfc"] != "atomic+random":
        el += [("startingwfc", c["startingwfc"])]
    if not c["scf_must_converge"]:
        el += [("scf_must_converge", False)]
    el += _extra(c, "ELECTRONS")

    parts = [namelist("CONTROL", ctrl), namelist("SYSTEM", sysl), namelist("ELECTRONS", el)]
    if calc in ("relax", "vc-relax", "md", "vc-md"):
        ions = [("ion_dynamics", c["ion_dynamics"])]
        if calc in ("md", "vc-md"):
            it = c["ion_temperature"]
            if c["ion_dynamics"] == "langevin":
                ions += [("tempw", float(c["tempw"]))]
            else:
                ions += [("ion_temperature", it)]
                if it != "not_controlled":
                    ions += [("tempw", float(c["tempw"]))]
                if it == "rescaling":
                    ions += [("tolp", float(c["md_tolp"]))]
                if it in ("rescale-v", "rescale-T", "reduce-T", "berendsen", "andersen", "svr"):
                    ions += [("nraise", int(c["md_nraise"]))]
                if it in ("rescale-T", "reduce-T"):
                    ions += [("delta_t", float(c["md_delta_t"]))]
        ions += _extra(c, "IONS")
        parts.append(namelist("IONS", ions))
    if calc in ("vc-relax", "vc-md"):
        cell = [("cell_dynamics", c["cell_dynamics"]), ("press", float(c["press"])),
                ("press_conv_thr", float(c["press_conv_thr"])), ("cell_dofree", c["cell_dofree"])]
        cell += _extra(c, "CELL")
        parts.append(namelist("CELL", cell))
    return "\n".join(parts)


def pw_input(c, struct, species, labels, fixed, calc, kcard, nbnd=None, occupations=None, conv_extra=False):
    show_flags = calc in ("relax", "vc-relax", "md", "vc-md") and bool(np.any(fixed))
    nl = pw_namelists(c, struct, species, calc, nbnd, occupations)
    if conv_extra:
        nl = nl.replace("  verbosity = ", "  tprnfor = .true.\n  tstress = .true.\n  verbosity = ", 1)
    parts = [nl,
             _cards_species(species), _cards_cell(struct),
             _cards_positions(struct, labels, fixed, c["coord_units"], show_flags), kcard]
    if c["hubbard_style"] == "card":
        h = _hubbard_card(species, c["hubbard_projector"])
        if h:
            parts.append(h)
    if str(c.get("extra_cards", "")).strip():
        parts.append(str(c["extra_cards"]).strip())
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------
# Post-processing inputs
# ---------------------------------------------------------------------

def bands_x_input(c):
    return namelist("BANDS", [("prefix", c["prefix"]), ("outdir", c["outdir"]),
                              ("filband", f"{c['prefix']}.bands.dat"), ("lsym", True)] + _extra(c, "BANDS")) + "\n"


def dos_x_input(c):
    return namelist("DOS", [("prefix", c["prefix"]), ("outdir", c["outdir"]), ("fildos", f"{c['prefix']}.dos"),
                            ("Emin", float(c["dos_emin"])), ("Emax", float(c["dos_emax"])),
                            ("DeltaE", float(c["dos_de"]))] + _extra(c, "DOS")) + "\n"


def projwfc_input(c):
    return namelist("PROJWFC", [("prefix", c["prefix"]), ("outdir", c["outdir"]),
                                ("filpdos", f"{c['prefix']}.pdos"), ("Emin", float(c["dos_emin"])),
                                ("Emax", float(c["dos_emax"])), ("DeltaE", float(c["dos_de"]))]
                    + _extra(c, "PROJWFC")) + "\n"


def pp_input(c):
    p = int(c["pp_plot"])
    tag = {0: "charge", 1: "vtot", 2: "vbare", 6: "spin", 7: "psi2", 8: "elf", 10: "ildos", 11: "vh_bare", 17: "ae_charge"}.get(p, f"plot{p}")
    inpp = [("prefix", c["prefix"]), ("outdir", c["outdir"]), ("filplot", f"{c['prefix']}.{tag}"),
            ("plot_num", p)]
    if p == 7:
        inpp += [("kpoint(1)", int(c["pp_kpoint"])), ("kband(1)", int(c["pp_kband"]))]
    if p == 10:
        inpp += [("emin", float(c["pp_emin"])), ("emax", float(c["pp_emax"]))]
    inpp += _extra(c, "INPUTPP")
    ext = {6: "cube", 5: "xsf"}[int(c["pp_format"])]
    plot = [("nfile", 1), (f"filepp(1)", f"{c['prefix']}.{tag}"), ("weight(1)", 1.0), ("iflag", 3),
            ("output_format", int(c["pp_format"])), ("fileout", f"{c['prefix']}_{tag}.{ext}")] + _extra(c, "PLOT")
    return namelist("INPUTPP", inpp) + "\n" + namelist("PLOT", plot) + "\n"


def ph_input(c):
    items = [("prefix", c["prefix"]), ("outdir", c["outdir"]), ("fildyn", f"{c['prefix']}.dyn"),
             ("tr2_ph", float(c["ph_tr2"]))]
    if c["ph_epsil"]:
        items += [("epsil", True)]
    if not c["ph_gamma_only"]:
        q = c["ph_qgrid"]
        items += [("ldisp", True), ("nq1", int(q[0])), ("nq2", int(q[1])), ("nq3", int(q[2]))]
    items += _extra(c, "INPUTPH")
    text = f"phonons of {c['prefix']}\n" + namelist("INPUTPH", items) + "\n"
    if c["ph_gamma_only"]:
        text += "0.0 0.0 0.0\n"
    return text


def q2r_input(c):
    return namelist("input", [("fildyn", f"{c['prefix']}.dyn"), ("zasr", c["ph_asr"]),
                              ("flfrc", f"{c['prefix']}.fc")]) + "\n"


def matdyn_disp_input(c, path):
    head = namelist("input", [("asr", c["ph_asr"]), ("flfrc", f"{c['prefix']}.fc"),
                              ("flfrq", f"{c['prefix']}.freq"), ("q_in_band_form", True),
                              ("q_in_cryst_coord", True)])
    lines = [str(len(path))] + [f"  {k[0]:12.8f} {k[1]:12.8f} {k[2]:12.8f}  {int(w)}  ! {lab}" for lab, k, w in path]
    return head + "\n" + "\n".join(lines) + "\n"


def matdyn_dos_input(c):
    g = c["ph_dos_grid"]
    return namelist("input", [("asr", c["ph_asr"]), ("flfrc", f"{c['prefix']}.fc"), ("dos", True),
                              ("fldos", f"{c['prefix']}.phdos"), ("nk1", int(g[0])), ("nk2", int(g[1])),
                              ("nk3", int(g[2]))]) + "\n"


def neb_input(c, struct, final, species, labels, fixed, kcard):
    path = namelist("PATH", [("string_method", "neb"), ("nstep_path", int(c["neb_nstep"])),
                             ("num_of_images", int(c["neb_images"])), ("opt_scheme", c["neb_opt"]),
                             ("CI_scheme", c["neb_ci"]), ("path_thr", float(c["neb_thr"])),
                             ("ds", float(c["neb_ds"]))] + _extra(c, "PATH"))
    show_flags = bool(np.any(fixed))
    eng = [pw_namelists(c, struct, species, "scf", engine=True), _cards_species(species),
           "BEGIN_POSITIONS", "FIRST_IMAGE",
           _cards_positions(struct, labels, fixed, c["coord_units"], show_flags),
           "LAST_IMAGE",
           _cards_positions(final, labels, fixed, c["coord_units"], show_flags),
           "END_POSITIONS", kcard, _cards_cell(struct)]
    if c["hubbard_style"] == "card":
        h = _hubbard_card(species, c["hubbard_projector"])
        if h:
            eng.append(h)
    return ("BEGIN\nBEGIN_PATH_INPUT\n" + path + "\nEND_PATH_INPUT\nBEGIN_ENGINE_INPUT\n" + "\n".join(eng)
            + "\nEND_ENGINE_INPUT\nEND\n")


# ---------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------

def check(c, struct, species, labels, kgrid_used, slab_axis):
    errors, warnings = [], []
    for sp in species:
        lab = str(sp["Label"])
        if not lab or len(lab) > 3:
            errors.append(f"Species label “{lab}” must be 1–3 characters (QE’s limit), like Fe or Fe1.")
        try:
            if normalize_symbol(lab) != sp["Element"] and not lab.startswith(sp["Element"]):
                warnings.append(f"Label {lab} doesn’t start with its element {sp['Element']}. QE may misread it.")
        except ValueError:
            errors.append(f"Species label “{lab}” must start with an element symbol.")
        if not str(sp["Pseudopotential"]).strip():
            errors.append(f"Choose a pseudopotential file for {lab}.")
    ew, er = float(c["ecutwfc"]), float(c["ecutrho"])
    if er < 4 * ew:
        errors.append(f"ecutrho ({er}) must be at least 4 × ecutwfc ({4 * ew}).")
    elif c["pseudo_type"] == "US/PAW" and er < 8 * ew:
        warnings.append(f"Ultrasoft and PAW pseudopotentials usually need ecutrho of 8–12 × ecutwfc "
                        f"({8 * ew:.0f}–{12 * ew:.0f} Ry here). Check the recommendation for your pseudopotentials.")
    if ew < 25:
        warnings.append("ecutwfc below 25 Ry is very low for most pseudopotentials. Run a convergence test.")
    mags = [float(sp["Start. magnetisation"]) for sp in species]
    if c["spin"] != "none" and not any(m != 0 for m in mags) and c["tot_magnetization"] is None:
        errors.append("A spin-polarised run needs a non-zero starting magnetisation for at least one species "
                      "(or a total magnetisation). QE stops otherwise.")
    if c["spin"] == "none" and any(m != 0 for m in mags):
        warnings.append("Starting magnetisations are set but spin polarisation is off, so they’re ignored.")
    if c["spin"] == "collinear" and c["occupations"] == "fixed" and c["tot_magnetization"] is None:
        errors.append("Fixed occupations with spin polarisation need a total magnetisation (tot_magnetization).")
    if any(abs(m) > 1 for m in mags):
        errors.append("Starting magnetisation must be between −1 and 1 (it’s a fraction of the valence electrons).")
    if c["spin"] == "noncollinear" and c["lspinorb"]:
        warnings.append("Spin–orbit coupling needs fully relativistic pseudopotentials (often named “rel” or “fr”).")
    for sp in species:
        if float(sp["Hubbard U (eV)"]) > 0 and str(sp["U manifold"]).strip() in ("", "?"):
            errors.append(f"Set the Hubbard manifold for {sp['Label']} (for example 3d, 4f or 2p).")
    if any(float(sp["Hubbard U (eV)"]) > 0 for sp in species) and c["hubbard_style"] == "card":
        warnings.append("The HUBBARD card needs QE 7.1 or newer. For older versions choose the legacy Hubbard format.")
    func = FUNCTIONALS.get(c["functional"], c["functional"])
    if func in HYBRIDS:
        warnings.append("Hybrid functionals are 10–100× more expensive. Start from a converged PBE run, and keep "
                        "the q-grid for exact exchange (nqx) small at first.")
        if c["workflow"] in ("bands", "dos", "phonon"):
            warnings.append("Band structures, nscf DOS and phonons don’t work directly with hybrid functionals in pw.x. "
                            "Use Wannier interpolation, or a PBE run for those steps.")
    if not struct.periodic and c["k_mode"] != "gamma":
        warnings.append("For a molecule in a box, K_POINTS gamma is faster and is all you need.")
    if slab_axis is not None and kgrid_used is not None and kgrid_used[slab_axis] > 1:
        warnings.append(f"The cell has a vacuum gap along axis {slab_axis + 1}, but the k-grid has "
                        f"{kgrid_used[slab_axis]} points there. 1 is enough for a slab.")
    if slab_axis is not None and not c["dipole"] and c["isolated"] == "none":
        warnings.append("This looks like a slab. If the two sides differ (adsorbates, polar surface), turn on the "
                        "dipole correction.")
    if c["occupations"] == "fixed" and c["workflow"] in ("dos",):
        warnings.append("Fixed occupations only work for insulators. Metals need smearing.")
    if c["calculation"] == "vc-relax" and c["workflow"] == "single" and ew < 60:
        warnings.append("Variable-cell relaxations need a higher cutoff than usual (Pulay stress). Consider "
                        "raising ecutwfc by ~30%.")
    calc = c["calculation"] if c["workflow"] == "single" else "scf"
    md_ok = {"md": ("verlet", "langevin", "langevin-smc"), "vc-md": ("beeman",),
             "relax": ("bfgs", "damp", "fire"), "vc-relax": ("bfgs", "damp")}
    if calc in md_ok and c["ion_dynamics"] not in md_ok[calc]:
        errors.append(f"ion_dynamics = {c['ion_dynamics']} doesn’t work with calculation = {calc}. "
                      f"Use one of: {', '.join(md_ok[calc])}.")
    if calc == "vc-md" and c["cell_dynamics"] not in ("pr", "w"):
        errors.append("vc-md needs cell_dynamics = pr or w.")
    if calc == "vc-relax" and c["cell_dynamics"] not in ("bfgs", "damp-pr", "damp-w"):
        errors.append("vc-relax needs cell_dynamics = bfgs, damp-pr or damp-w.")
    if c["workflow"] == "phonon":
        if c["ph_epsil"] and c["occupations"] == "smearing":
            errors.append("ph.x can’t compute the dielectric tensor (epsil) for metals. Turn off epsil, or use fixed "
                          "occupations for an insulator.")
        if float(c["conv_thr"]) > 1e-10:
            warnings.append("Phonons need a very tight SCF. The app sets conv_thr = 1e-12 for the scf step.")
        if func in ("vdw-df", "vdw-df2", "vdw-df2-b86r", "rvv10") or c["vdw"] in ("ts", "mbd", "xdm"):
            warnings.append("Check that ph.x supports your functional / dispersion correction in your QE version.")
    if c["dipole"] and c["edir"] - 1 != slab_axis and slab_axis is not None:
        warnings.append(f"The dipole correction direction (edir = {c['edir']}) isn’t the vacuum direction "
                        f"(axis {slab_axis + 1}).")
    return errors, warnings


# ---------------------------------------------------------------------
# Workflows
# ---------------------------------------------------------------------

def build_qe(c, struct, species, labels, fixed, final=None):
    c = {**DEFAULTS, **c}
    files, errors, warnings = {}, [], []
    if struct is None:
        return files, ["Upload a structure first."], warnings, []
    if not struct.periodic:
        errors.append("Internal error: the structure needs a cell. (Molecules are boxed automatically.)")
        return files, errors, warnings, []
    from structures import vacuum_axis, kgrid_from_spacing  # noqa: F401
    slab_axis = vacuum_axis(struct)
    p = c["prefix"]
    wf = c["workflow"]

    def kcard_main():
        if c["k_mode"] == "gamma":
            return _cards_kpoints("gamma"), None
        return _cards_kpoints("automatic", c["kgrid"], c["kshift"]), list(c["kgrid"])

    kcard, kused = kcard_main()
    e2, w2 = check(c, struct, species, labels, kused, slab_axis)
    errors += e2
    warnings += w2
    run = ["#!/bin/bash", "# Quantum ESPRESSO workflow generated by the DFT Input Maker", f"NP={int(c['nprocs'])}",
           'RUN="mpirun -np $NP"', ""]

    if wf in ("bands", "phonon") and c["path"] is None and (wf == "bands" or c["ph_dispersion"]):
        errors.append("This workflow needs a k-point path. Install ASE for automatic paths, or enter one by hand.")

    if wf == "single":
        calc = c["calculation"]
        if calc in ("nscf", "bands"):
            warnings.append(f"A {calc} run needs an earlier scf run with the same prefix and outdir.")
        if calc == "bands" and c["path"]:
            kcard = _cards_kpoints("path", path=c["path"])
        files[f"{p}.{calc}.in"] = pw_input(c, struct, species, labels, fixed, calc, kcard)
        run.append(f"$RUN pw.x -in {p}.{calc}.in > {p}.{calc}.out")
    elif wf == "bands":
        files[f"{p}.scf.in"] = pw_input(c, struct, species, labels, fixed, "scf", kcard)
        if c["path"]:
            nb = int(c["nbnd"]) if int(c["nbnd"]) > 0 else None
            files[f"{p}.bands.in"] = pw_input(c, struct, species, labels, fixed, "bands",
                                              _cards_kpoints("path", path=c["path"]), nbnd=nb)
        files[f"{p}.bands_pp.in"] = bands_x_input(c)
        run += [f"$RUN pw.x -in {p}.scf.in > {p}.scf.out", f"$RUN pw.x -in {p}.bands.in > {p}.bands.out",
                f"$RUN bands.x -in {p}.bands_pp.in > {p}.bands_pp.out"]
        if int(c["nbnd"]) == 0:
            warnings.append("Set nbnd (number of bands) to include some empty bands above the Fermi level; by default "
                            "pw.x computes only a few for insulators.")
    elif wf == "dos":
        files[f"{p}.scf.in"] = pw_input(c, struct, species, labels, fixed, "scf", kcard)
        nscf_k = _cards_kpoints("automatic", c["nscf_kgrid"], (0, 0, 0))
        files[f"{p}.nscf.in"] = pw_input(c, struct, species, labels, fixed, "nscf", nscf_k,
                                         occupations="tetrahedra_opt",
                                         nbnd=int(c["nbnd"]) if int(c["nbnd"]) > 0 else None)
        files[f"{p}.dos.in"] = dos_x_input(c)
        files[f"{p}.projwfc.in"] = projwfc_input(c)
        run += [f"$RUN pw.x -in {p}.scf.in > {p}.scf.out", f"$RUN pw.x -in {p}.nscf.in > {p}.nscf.out",
                f"$RUN dos.x -in {p}.dos.in > {p}.dos.out", f"$RUN projwfc.x -in {p}.projwfc.in > {p}.projwfc.out"]
        if any(n <= m for n, m in zip(c["nscf_kgrid"], c["kgrid"])) and c["k_mode"] == "automatic":
            warnings.append("The nscf k-grid should be denser than the scf grid for a smooth DOS.")
    elif wf == "pp":
        files[f"{p}.scf.in"] = pw_input(c, struct, species, labels, fixed, "scf", kcard)
        files[f"{p}.pp.in"] = pp_input(c)
        run += [f"$RUN pw.x -in {p}.scf.in > {p}.scf.out", f"$RUN pp.x -in {p}.pp.in > {p}.pp.out"]
        if int(c["pp_plot"]) == 6 and c["spin"] == "none":
            errors.append("A spin-polarisation map needs a spin-polarised calculation.")
        if int(c["pp_plot"]) == 17:
            warnings.append("The all-electron charge (plot_num 17) only works with PAW pseudopotentials.")
    elif wf == "phonon":
        tight = {**c, "conv_thr": min(float(c["conv_thr"]), 1e-12)}
        files[f"{p}.scf.in"] = pw_input(tight, struct, species, labels, fixed, "scf", kcard)
        files[f"{p}.ph.in"] = ph_input(c)
        run += [f"$RUN pw.x -in {p}.scf.in > {p}.scf.out", f"$RUN ph.x -in {p}.ph.in > {p}.ph.out"]
        if not c["ph_gamma_only"]:
            files[f"{p}.q2r.in"] = q2r_input(c)
            run.append(f"q2r.x < {p}.q2r.in > {p}.q2r.out")
            if c["ph_dispersion"] and c["path"]:
                files[f"{p}.matdyn.in"] = matdyn_disp_input(c, c["path"])
                run.append(f"matdyn.x < {p}.matdyn.in > {p}.matdyn.out")
            if c["ph_dos"]:
                files[f"{p}.matdyn_dos.in"] = matdyn_dos_input(c)
                run.append(f"matdyn.x < {p}.matdyn_dos.in > {p}.matdyn_dos.out")
        if np.any(fixed):
            warnings.append("Phonons should be computed on a fully relaxed structure. Fixed atoms are ignored here.")
    elif wf == "md":
        ens = c["md_ensemble"]
        calc = "vc-md" if ens == "NPT" else "md"
        md = {**c, "dt": float(c["md_dt_fs"]) / RY_AU_FS}
        if ens == "NVE":
            md["ion_temperature"] = "not_controlled"
        if calc == "vc-md":
            md["ion_dynamics"], md["cell_dynamics"] = "beeman", c["cell_dynamics"] if c["cell_dynamics"] in ("pr", "w") else "pr"
        elif md["ion_dynamics"] not in ("verlet", "langevin"):
            md["ion_dynamics"] = "verlet"
        files[f"{p}.md.in"] = pw_input(md, struct, species, labels, fixed, calc, kcard)
        run.append(f"$RUN pw.x -in {p}.md.in > {p}.md.out")
        total_ps = int(c["nstep"]) * float(c["md_dt_fs"]) / 1000
        warnings.append(f"This run is {int(c['nstep'])} steps × {float(c['md_dt_fs'])} fs = {total_ps:.3g} ps.")
        if not c["nosym"]:
            warnings.append("Turn on nosym for MD (Functional & corrections tab). Symmetry can stop atoms from moving "
                            "the way they should.")
        if float(c["md_dt_fs"]) > 2.0 and "H" in struct.symbols:
            warnings.append("Time steps above ~1 fs are risky with hydrogen. Use 0.5–1 fs.")
        if ens == "NVT" and c["ion_temperature"] == "not_controlled" and c["ion_dynamics"] != "langevin":
            warnings.append("NVT needs a thermostat: choose ion_temperature (svr or berendsen are good choices).")
        if c["verbosity"] == "high":
            warnings.append("verbosity = 'high' makes MD output files very large. 'low' is usually better for MD.")
    elif wf == "convergence":
        nat = struct.n
        tests = []
        dual = float(c["ecutrho"]) / float(c["ecutwfc"])
        if c["conv_mode"] in ("ecut", "both"):
            e = float(c["conv_ecut_start"])
            while e <= float(c["conv_ecut_stop"]) + 1e-9:
                tag = f"ecut{int(round(e))}"
                cc = {**c, "ecutwfc": e, "ecutrho": e * dual, "prefix": f"{p}_{tag}"}
                files[f"{p}.{tag}.in"] = pw_input(cc, struct, species, labels, fixed, "scf", kcard, conv_extra=True)
                tests.append(("ecutwfc (Ry)", e, f"{p}.{tag}.out"))
                run.append(f"$RUN pw.x -in {p}.{tag}.in > {p}.{tag}.out")
                e += float(c["conv_ecut_step"])
            if len(tests) < 3:
                warnings.append("Use at least 3 cutoffs to see convergence.")
        if c["conv_mode"] in ("kpoints", "both"):
            grids = []
            for sp_ in sorted(c["conv_kspacings"], reverse=True):
                g = kgrid_from_spacing(struct.cell, float(sp_), slab_axis)
                if g not in grids:
                    grids.append(g)
            for g in grids:
                tag = "k" + "x".join(map(str, g))
                cc = {**c, "prefix": f"{p}_{tag}"}
                files[f"{p}.{tag}.in"] = pw_input(cc, struct, species, labels, fixed, "scf",
                                                  _cards_kpoints("automatic", g, c["kshift"]), conv_extra=True)
                tests.append(("k-grid", "x".join(map(str, g)), f"{p}.{tag}.out"))
                run.append(f"$RUN pw.x -in {p}.{tag}.in > {p}.{tag}.out")
        run.append("python3 collect_convergence.py")
        files["collect_convergence.py"] = convergence_script(tests, nat)
        warnings.append("Convergence tests use the settings above for everything else. Converge the cutoff first, "
                        "then the k-points with the converged cutoff.")
    elif wf == "neb":
        if final is None:
            errors.append("NEB needs a final structure. Upload it in the sidebar.")
        elif final.symbols != struct.symbols:
            errors.append("The initial and final structures must have the same atoms in the same order.")
        else:
            fin = final
            if not fin.periodic:
                from structures import Structure
                fin = Structure(final.symbols, final.positions - final.positions.mean(0) + struct.positions.mean(0),
                                struct.cell)
            files[f"{p}.neb.in"] = neb_input(c, struct, fin, species, labels, fixed, kcard)
            run.append(f"$RUN neb.x -inp {p}.neb.in > {p}.neb.out")
            disp = np.linalg.norm(fin.positions - struct.positions, axis=1).max()
            if disp > 5:
                warnings.append(f"One atom moves {disp:.1f} Å between the end points. Check that both structures "
                                "use the same atom order and the same cell.")
    files["run_qe.sh"] = "\n".join(run) + "\n"
    commands = [l for l in run if l and not l.startswith(("#", "NP=", "RUN="))]
    return files, errors, warnings, commands


def convergence_script(tests, nat):
    """A small Python script that reads the convergence-test outputs and reports the changes."""
    import json
    return f'''#!/usr/bin/env python3
"""Collect total energies, forces and pressures from the convergence tests and show how they change."""
import csv, os, re

RY_TO_MEV = 13605.693
NAT = {nat}
TESTS = {json.dumps(tests)}   # (parameter, value, output file)


def read(path):
    if not os.path.exists(path):
        return None, None, None
    text = open(path, errors="ignore").read()
    e = re.findall(r"^!\\s+total energy\\s+=\\s+(-?[\\d.]+)\\s+Ry", text, re.M)
    f = re.findall(r"Total force\\s+=\\s+([\\d.]+)", text)
    p = re.findall(r"P=\\s*(-?[\\d.]+)", text)
    return (float(e[-1]) if e else None, float(f[-1]) if f else None, float(p[-1]) if p else None)


rows = []
for param, value, out in TESTS:
    e, f, p = read(out)
    rows.append([param, value, out, e, f, p])

for param in dict.fromkeys(r[0] for r in rows):
    group = [r for r in rows if r[0] == param and r[3] is not None]
    if not group:
        print(f"No finished runs for {{param}} yet.")
        continue
    ref = group[-1][3]
    print(f"\\n{{param}}: energy change per atom compared with the last (most converged) run")
    print(f"{{'value':>10}} {{'E (Ry)':>18}} {{'dE (meV/atom)':>15}} {{'force (Ry/Bohr)':>16}} {{'P (kbar)':>10}}")
    for r in group:
        de = (r[3] - ref) * RY_TO_MEV / NAT
        r.append(de)
        print(f"{{str(r[1]):>10}} {{r[3]:18.8f}} {{de:15.3f}} {{(r[4] if r[4] is not None else float('nan')):16.6f}} "
              f"{{(r[5] if r[5] is not None else float('nan')):10.2f}}")

with open("convergence.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["parameter", "value", "output", "energy_Ry", "total_force_Ry_per_Bohr", "pressure_kbar",
                "dE_meV_per_atom"])
    w.writerows(rows)
print("\\nSaved convergence.csv. Converged usually means |dE| below 1 meV/atom.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    params = [p for p in dict.fromkeys(r[0] for r in rows) if any(r[3] is not None for r in rows if r[0] == p)]
    fig, axes = plt.subplots(1, len(params), figsize=(5 * len(params), 3.6), squeeze=False)
    for ax, param in zip(axes[0], params):
        group = [r for r in rows if r[0] == param and r[3] is not None]
        ax.plot(range(len(group)), [r[-1] for r in group], "o-")
        ax.set_xticks(range(len(group)))
        ax.set_xticklabels([str(r[1]) for r in group], rotation=45)
        ax.axhline(1, ls="--", c="gray"); ax.axhline(-1, ls="--", c="gray")
        ax.set_xlabel(param); ax.set_ylabel("dE (meV/atom)")
    fig.tight_layout()
    fig.savefig("convergence.png", dpi=150)
    print("Saved convergence.png")
except Exception:
    pass
'''
