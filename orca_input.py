"""
orca_input.py — builds ORCA input files from a settings dictionary, and checks them.

build_orca(settings, structure, final_structure=None) -> (files, errors, warnings)
  files    : {filename: text}
  errors   : problems that would make ORCA stop or give wrong results
  warnings : things worth a second look
"""

from elements import Z

# ---------------------------------------------------------------------
# Option lists (shown in the app)
# ---------------------------------------------------------------------

FUNCTIONALS = {
    "GGA": ["PBE", "BP86", "BLYP", "revPBE", "PW91", "B97-D3", "OLYP"],
    "meta-GGA": ["r2SCAN", "SCAN", "TPSS", "M06L", "B97M-V", "B97M-D4"],
    "Hybrid": ["B3LYP", "PBE0", "TPSSh", "r2SCANh", "M06", "M06-2X", "BHandHLYP", "X3LYP", "B3PW91"],
    "Range-separated hybrid": ["wB97X", "wB97X-D3", "wB97X-D4", "wB97X-V", "wB97M-V", "wB97M-D4",
                               "CAM-B3LYP", "LC-BLYP", "wr2SCAN"],
    "Double hybrid": ["B2PLYP", "mPW2PLYP", "DSD-PBEP86", "revDSD-PBEP86-D4", "DSD-BLYP", "wB2PLYP", "PWPB95"],
}
FUNCTIONAL_GROUP = {f: g for g, fs in FUNCTIONALS.items() for f in fs}
BUILT_IN_DISPERSION = {"B97-D3", "B97M-V", "B97M-D4", "wB97X-D3", "wB97X-D4", "wB97X-V", "wB97M-V",
                       "wB97M-D4", "revDSD-PBEP86-D4"}

COMPOSITE = ["r2SCAN-3c", "B97-3c", "PBEh-3c", "HF-3c", "wB97X-3c"]
XTB = ["XTB2", "XTB1"]
WAVEFUNCTION = ["MP2", "RI-MP2", "SCS-MP2", "CCSD", "CCSD(T)", "DLPNO-MP2", "DLPNO-CCSD", "DLPNO-CCSD(T)",
                "DLPNO-CCSD(T1)"]

BASIS_SETS = {
    "Karlsruhe (def2)": ["def2-SVP", "def2-TZVP", "def2-TZVPP", "def2-QZVPP", "def2-SVPD", "def2-TZVPD",
                         "def2-TZVPPD", "ma-def2-SVP", "ma-def2-TZVP", "def2-TZVP(-f)"],
    "Dunning (cc)": ["cc-pVDZ", "cc-pVTZ", "cc-pVQZ", "aug-cc-pVDZ", "aug-cc-pVTZ", "aug-cc-pVQZ",
                     "cc-pwCVTZ"],
    "Pople": ["STO-3G", "3-21G", "6-31G", "6-31G(d)", "6-31G(d,p)", "6-31+G(d,p)", "6-311G(d,p)",
              "6-311+G(d,p)", "6-311++G(2d,2p)"],
    "Jensen (pcseg)": ["pcseg-0", "pcseg-1", "pcseg-2", "pcseg-3", "aug-pcseg-1", "aug-pcseg-2"],
    "Relativistic all-electron": ["ZORA-def2-SVP", "ZORA-def2-TZVP", "DKH-def2-SVP", "DKH-def2-TZVP",
                                  "x2c-SVPall", "x2c-TZVPall", "SARC-ZORA-TZVP", "SARC-DKH-TZVP"],
}

SOLVENTS = ["water", "acetonitrile", "methanol", "ethanol", "dmso", "dmf", "thf", "dichloromethane",
            "chloroform", "acetone", "toluene", "benzene", "hexane", "cyclohexane", "diethylether",
            "pyridine", "octanol", "ccl4", "nitromethane", "ethylacetate"]

JOBS = {
    "SP": "Single point energy",
    "Opt": "Geometry optimisation",
    "Freq": "Frequencies only",
    "OptFreq": "Optimisation + frequencies",
    "OptTS": "Transition state search",
    "OptTSFreq": "Transition state search + frequencies",
    "Scan": "Coordinate scan (relaxed or rigid)",
    "IRC": "Intrinsic reaction coordinate (IRC)",
    "NEB-CI": "NEB, climbing image (needs a final structure)",
    "NEB-TS": "NEB followed by TS optimisation (needs a final structure)",
    "GOAT": "Conformer search, GOAT (ORCA 6+)",
    "MD": "Molecular dynamics (ORCA 5+)",
}

DEFAULTS = dict(
    basename="job", job="SP", family="DFT", functional="B3LYP", composite="r2SCAN-3c", xtb="XTB2",
    wavefunction="DLPNO-CCSD(T)", pno="NormalPNO", basis="def2-TZVP", dispersion="D3BJ",
    ri="RIJCOSX", aux_j="auto", aux_c="auto", relativistic="none", reference="auto",
    charge=0, mult=1, scf_conv="TightSCF", slowconv="none", scf_maxiter=0, trah=True,
    stability=False, brokensym="", guess="default", moread_file="", grid="default",
    opt_conv="normal", opt_maxiter=0, ts_recalc_hess=0, numfreq=False, raman=False,
    temperature=298.15, pressure=1.0, solvation="none", solvent="water",
    tddft=False, nroots=10, tda=True, triplets=False, iroot=0,
    nmr=False, epr=False, hirshfeld=False, chelpg=False, nbo=False, polar=False, largeprint=False,
    casscf=False, cas_nel=2, cas_norb=2, cas_nroots=1, cas_pt="none",
    scan_relaxed=True, scan_type="B", scan_atoms="0 1", scan_start=1.0, scan_end=2.0, scan_steps=10,
    constraints="", fix_atoms="", irc_direction="both", irc_maxiter=50,
    neb_images=8, md_timestep=0.5, md_temp=300.0, md_thermostat="CSVR", md_timecon=10.0, md_steps=2000,
    md_stride=1, nprocs=4, maxcore=2000, coords="inline", extra_keywords="", extra_blocks="",
)


def _aux_for(basis, kind):
    b = basis.replace("ma-", "")
    if kind == "J":
        return "SARC/J" if b.startswith(("ZORA", "DKH", "SARC")) else "def2/J"
    if kind == "JK":
        return "def2/JK"
    if kind == "C":
        if b.startswith("def2-") or b.startswith(("cc-p", "aug-cc-p")):
            return b.split("(")[0] + "/C"
        return "AutoAux"
    return ""


def electron_count(symbols, charge):
    return sum(Z[s] for s in symbols) - int(charge)


def build_orca(s: dict, struct, final=None):
    s = {**DEFAULTS, **s}
    errors, warnings = [], []
    kw = []           # ! line keywords
    blocks = []       # % blocks

    # ---------------- structure checks
    if struct is None:
        errors.append("Upload a structure first.")
        return {}, errors, warnings
    if struct.periodic:
        warnings.append("This structure has a unit cell, but ORCA is a molecular code: the cell is ignored and "
                        "the atoms are treated as an isolated cluster.")
    n_el = electron_count(struct.symbols, s["charge"])
    mult = int(s["mult"])
    if n_el <= 0:
        errors.append(f"The charge {s['charge']} leaves {n_el} electrons.")
    elif mult < 1 or mult > n_el + 1:
        errors.append(f"Multiplicity {mult} isn’t possible with {n_el} electrons.")
    elif (n_el + mult) % 2 == 0:
        errors.append(f"Charge and multiplicity don’t match: {n_el} electrons needs an "
                      f"{'odd' if n_el % 2 == 0 else 'even'} multiplicity "
                      f"({'1, 3, 5…' if n_el % 2 == 0 else '2, 4, 6…'}).")
    heavy = sorted({x for x in struct.symbols if Z[x] > 36}, key=lambda x: Z[x])

    # ---------------- method
    fam = s["family"]
    basis = s["basis"]
    uses_basis = fam in ("DFT", "HF", "Wavefunction", "CASSCF")
    if fam == "DFT":
        kw.append(s["functional"])
        group = FUNCTIONAL_GROUP.get(s["functional"], "")
        if s["dispersion"] != "none":
            if s["functional"] in BUILT_IN_DISPERSION:
                warnings.append(f"{s['functional']} already includes dispersion, so {s['dispersion']} was not added.")
            else:
                kw.append(s["dispersion"])
    elif fam == "HF":
        kw.append("HF")
        if s["dispersion"] != "none":
            kw.append(s["dispersion"])
    elif fam == "Composite":
        kw.append(s["composite"])
    elif fam == "xTB":
        kw.append(s["xtb"])
    elif fam == "Wavefunction":
        kw.append(s["wavefunction"])
        if s["wavefunction"].startswith("DLPNO"):
            kw.append(s["pno"])

    # reference (restricted / unrestricted)
    ref = s["reference"]
    if s["brokensym"] and ref == "auto":
        ref = "unrestricted"
    if ref != "auto" and fam in ("DFT", "HF", "Wavefunction"):
        dft = fam == "DFT"
        kw.append({"restricted": "RKS" if dft else "RHF", "unrestricted": "UKS" if dft else "UHF",
                   "restricted-open": "ROKS" if dft else "ROHF"}[ref])
        if ref == "restricted" and mult > 1:
            errors.append("A restricted (closed-shell) calculation needs multiplicity 1.")

    # basis sets and RI
    if uses_basis:
        b = basis
        if s["relativistic"] == "ZORA" and b.startswith("def2-"):
            b = "ZORA-" + b
            warnings.append(f"Switched the basis to {b}, the version recontracted for ZORA.")
        if s["relativistic"] == "DKH" and b.startswith("def2-"):
            b = "DKH-" + b
            warnings.append(f"Switched the basis to {b}, the version recontracted for DKH.")
        if s["relativistic"] == "X2C" and not b.lower().startswith("x2c"):
            warnings.append("X2C should be used with x2c basis sets (for example x2c-TZVPall).")
        kw.append(b)
        if s["relativistic"] != "none":
            kw.append(s["relativistic"])

        group = FUNCTIONAL_GROUP.get(s["functional"], "") if fam == "DFT" else ""
        ri = s["ri"]
        if ri == "auto":
            ri = "RIJCOSX" if (fam in ("HF", "Wavefunction", "CASSCF") or group in (
                "Hybrid", "Range-separated hybrid", "Double hybrid")) else "RI"
        if ri != "NoRI":
            kw.append(ri)
            aux = s["aux_j"]
            if aux == "auto":
                aux = _aux_for(b, "JK" if ri == "RIJK" else "J")
            if aux:
                kw.append(aux)
        else:
            kw.append("NoRI")
        needs_c = (fam == "Wavefunction" and ("RI" in s["wavefunction"] or "DLPNO" in s["wavefunction"]
                                              or s["wavefunction"] == "SCS-MP2")) or group == "Double hybrid"
        if needs_c:
            aux_c = s["aux_c"] if s["aux_c"] != "auto" else _aux_for(b, "C")
            kw.append(aux_c)

        if heavy:
            if b.startswith(("6-", "3-21", "STO")):
                errors.append(f"The {b} basis set isn’t defined for {', '.join(heavy)}. Use a def2 basis set, which "
                              "includes effective core potentials for heavy elements.")
            elif b.startswith("def2") and s["relativistic"] == "none":
                warnings.append(f"def2 basis sets automatically use effective core potentials for "
                                f"{', '.join(heavy)} (elements beyond Kr). That’s normally what you want.")
            elif b.startswith(("cc-p", "aug-cc")):
                warnings.append(f"For {', '.join(heavy)}, plain cc-pVnZ basis sets may not exist. ORCA needs the "
                                "-PP versions with effective core potentials, or a def2 basis.")
        if s["relativistic"] in ("ZORA", "DKH") and heavy and not b.startswith("SARC"):
            sarc = "SARC-ZORA-TZVP" if s["relativistic"] == "ZORA" else "SARC-DKH-TZVP"
            very_heavy = [el for el in heavy if Z[el] >= 57]
            middle = [el for el in heavy if Z[el] < 57]
            if very_heavy:
                blocks.append("%basis\n" + "\n".join(f'  NewGTO {el} "{sarc}" end' for el in very_heavy) + "\nend")
                warnings.append(f"{b} only covers elements up to Kr, so these get the all-electron {sarc} basis "
                                f"(added in a %basis block): {', '.join(very_heavy)}.")
            if middle:
                warnings.append(f"{b} doesn’t cover {', '.join(middle)}. Add an all-electron basis for these elements in a "
                                "%basis block (Extra settings), choosing one your ORCA version provides; see the "
                                "basis set section of the ORCA manual.")
        if s["relativistic"] != "none" and heavy == [] and all(Z[x] <= 18 for x in struct.symbols):
            warnings.append("Scalar-relativistic methods are only needed for heavier elements. They’re harmless "
                            "here but cost a little time.")
    elif fam in ("Composite", "xTB") and s["relativistic"] != "none":
        warnings.append(f"Relativistic settings aren’t used with {fam} methods and were ignored.")

    if fam == "Wavefunction" and s["wavefunction"].startswith(("CCSD", "DLPNO")) and basis.startswith(("6-", "STO", "3-21")):
        warnings.append("Coupled-cluster results are much better with correlation-consistent (cc-pVTZ) or def2-TZVPP "
                        "basis sets than with Pople basis sets.")
    if fam == "DFT" and s["functional"] == "B3LYP" and s["dispersion"] == "none":
        warnings.append("B3LYP without dispersion correction gives poor geometries and interaction energies. "
                        "D3BJ or D4 is strongly recommended.")

    # ---------------- accuracy
    if fam not in ("xTB",):
        if s["scf_conv"] != "default":
            kw.append(s["scf_conv"])
        if s["slowconv"] != "none":
            kw.append(s["slowconv"])
        if s["grid"] != "default" and fam in ("DFT", "Composite"):
            kw.append(s["grid"])

    # ---------------- job type
    job = s["job"]
    opt_kw = {"normal": "Opt", "loose": "LooseOpt", "tight": "TightOpt", "verytight": "VeryTightOpt"}[s["opt_conv"]]
    freq_kw = "NumFreq" if s["numfreq"] else "Freq"
    geom = []
    if job == "SP":
        kw.append("SP")
    elif job == "Opt":
        kw.append(opt_kw)
    elif job == "Freq":
        kw.append(freq_kw)
    elif job == "OptFreq":
        kw += [opt_kw, freq_kw]
    elif job in ("OptTS", "OptTSFreq"):
        kw.append("OptTS")
        if job == "OptTSFreq":
            kw.append(freq_kw)
        geom += ["  Calc_Hess true"]
        if int(s["ts_recalc_hess"]) > 0:
            geom.append(f"  Recalc_Hess {int(s['ts_recalc_hess'])}")
    elif job == "Scan":
        kw.append(opt_kw if s["scan_relaxed"] else "SP")
        atoms = [int(x) for x in str(s["scan_atoms"]).replace(",", " ").split()]
        need = {"B": 2, "A": 3, "D": 4}[s["scan_type"]]
        if len(atoms) != need:
            errors.append(f"A {'bond' if need == 2 else 'angle' if need == 3 else 'dihedral'} scan needs {need} atom "
                          f"numbers, but {len(atoms)} were given.")
        elif max(atoms) >= struct.n:
            errors.append(f"Scan atom numbers must be between 0 and {struct.n - 1} (ORCA counts from 0).")
        geom += ["  Scan", f"    {s['scan_type']} {' '.join(map(str, atoms))} = "
                           f"{s['scan_start']}, {s['scan_end']}, {int(s['scan_steps'])}", "  end"]
    elif job == "IRC":
        kw.append("IRC")
        blocks.append(f"%irc\n  MaxIter {int(s['irc_maxiter'])}\n  Direction {s['irc_direction']}\n"
                      f"  InitHess calc_anfreq\nend")
    elif job in ("NEB-CI", "NEB-TS"):
        kw.append(job)
        if final is None:
            errors.append("NEB needs a final structure. Upload it in the sidebar.")
        elif final.symbols != struct.symbols:
            errors.append("The initial and final structures must have the same atoms in the same order.")
        blocks.append(f'%neb\n  NEB_End_XYZFile "{s["basename"]}_final.xyz"\n  NImages {int(s["neb_images"])}\nend')
    elif job == "MD":
        kw.append("MD")
        md = [f"  Timestep {float(s['md_timestep'])}_fs", f"  Initvel {float(s['md_temp'])}_K"]
        if s["md_thermostat"] != "none":
            md.append(f"  Thermostat {s['md_thermostat']} {float(s['md_temp'])}_K Timecon {float(s['md_timecon'])}_fs")
        md += [f'  Dump Position Stride {int(s["md_stride"])} Filename "{s["basename"]}_traj.xyz"',
               f"  Run {int(s['md_steps'])}"]
        blocks.append("%md\n" + "\n".join(md) + "\nend")
        ps = int(s["md_steps"]) * float(s["md_timestep"]) / 1000
        warnings.append(f"This MD run is {int(s['md_steps'])} steps × {s['md_timestep']} fs = {ps:.3g} ps.")
        if fam not in ("xTB", "Composite"):
            warnings.append("MD needs thousands of energy and force evaluations. XTB2 or a 3c method is usually the "
                            "practical choice; DFT with a large basis will be very slow.")
        if float(s["md_timestep"]) > 1.0 and "H" in struct.symbols:
            warnings.append("Time steps above ~1 fs are risky with hydrogen. 0.5 fs is safer.")
    elif job == "GOAT":
        kw.append("GOAT")
        warnings.append("GOAT needs ORCA 6 or newer, and is usually run with a cheap method such as XTB2.")

    if job in ("Opt", "OptFreq", "OptTS", "OptTSFreq", "Scan") and int(s["opt_maxiter"]) > 0:
        geom.append(f"  MaxIter {int(s['opt_maxiter'])}")
    if job in ("OptTS", "OptTSFreq", "IRC") and fam == "xTB":
        warnings.append("Transition states from xTB are rough. Refine them with DFT.")

    # constraints and fixed atoms
    cons = [l.strip() for l in str(s["constraints"]).splitlines() if l.strip()]
    fix = [int(x) for x in str(s["fix_atoms"]).replace(",", " ").split() if x.strip().lstrip("-").isdigit()]
    if any(i < 0 or i >= struct.n for i in fix):
        errors.append(f"Fixed-atom numbers must be between 0 and {struct.n - 1}.")
    cons += [f"{{C {i} C}}" for i in fix]
    if cons:
        if job not in ("Opt", "OptFreq", "OptTS", "OptTSFreq", "Scan"):
            warnings.append("Constraints only matter for optimisations and scans.")
        geom += ["  Constraints"] + [f"    {c}" for c in cons] + ["  end"]
    if geom:
        blocks.append("%geom\n" + "\n".join(geom) + "\nend")

    if job in ("Freq", "OptFreq", "OptTSFreq"):
        if abs(float(s["temperature"]) - 298.15) > 1e-6 or abs(float(s["pressure"]) - 1.0) > 1e-9:
            blocks.append(f"%freq\n  Temp {s['temperature']}\n  Pressure {s['pressure']}\nend")
        if job == "Freq" and fam not in ("xTB",):
            warnings.append("Frequencies are only meaningful at a geometry optimised with the same method and basis set.")
    if s["raman"]:
        blocks.append("%elprop\n  Polar 1\nend")
        if job not in ("Freq", "OptFreq", "OptTSFreq"):
            warnings.append("Raman intensities need a frequency calculation.")
        if not s["numfreq"]:
            warnings.append("Raman intensities need numerical frequencies. Tick “numerical frequencies”.")

    # ---------------- SCF details
    scf = []
    if int(s["scf_maxiter"]) > 0:
        scf.append(f"  MaxIter {int(s['scf_maxiter'])}")
    if not s["trah"]:
        kw.append("NoTRAH")
    if s["stability"]:
        scf.append("  STABPerform true")
    if s["brokensym"]:
        scf.append(f"  BrokenSym {s['brokensym']}")
    if scf:
        blocks.append("%scf\n" + "\n".join(scf) + "\nend")
    if s["guess"] == "MORead":
        kw.append("MORead")
        if not s["moread_file"]:
            errors.append("MORead needs the name of a .gbw file to read the orbitals from.")
        blocks.append(f'%moinp "{s["moread_file"]}"')
    elif s["guess"] != "default":
        kw.append(s["guess"])

    # ---------------- solvation
    if s["solvation"] == "CPCM":
        kw.append(f"CPCM({s['solvent']})")
    elif s["solvation"] == "SMD":
        kw.append(f"CPCM({s['solvent']})")
        blocks.append(f'%cpcm\n  smd true\n  SMDsolvent "{s["solvent"]}"\nend')
    elif s["solvation"] == "ALPB":
        if fam != "xTB":
            errors.append("ALPB solvation only works with xTB methods. Use CPCM or SMD for DFT.")
        kw.append(f"ALPB({s['solvent']})")

    # ---------------- excited states
    if s["tddft"]:
        if fam not in ("DFT", "HF"):
            warnings.append("TD-DFT/CIS excited states need a DFT or HF method.")
        td = [f"  NRoots {int(s['nroots'])}", f"  TDA {'true' if s['tda'] else 'false'}"]
        if s["triplets"]:
            td.append("  Triplets true")
        if int(s["iroot"]) > 0:
            td.append(f"  IRoot {int(s['iroot'])}")
            if int(s["iroot"]) > int(s["nroots"]):
                errors.append("IRoot must not be larger than the number of roots.")
        blocks.append("%tddft\n" + "\n".join(td) + "\nend")

    # ---------------- properties
    if s["nmr"]:
        kw.append("NMR")
    if s["epr"]:
        blocks.append("%eprnmr\n  GTensor true\nend")
        if mult == 1:
            warnings.append("EPR g-tensors need an open-shell system (multiplicity above 1).")
    if s["hirshfeld"]:
        kw.append("Hirshfeld")
    if s["chelpg"]:
        kw.append("CHELPG")
    if s["nbo"]:
        kw.append("NBO")
        warnings.append("NBO analysis needs the separate NBO program to be installed and linked to ORCA.")
    if s["polar"] and not s["raman"]:
        blocks.append("%elprop\n  Polar 1\nend")
    if s["largeprint"]:
        kw.append("LargePrint")

    # ---------------- CASSCF
    if fam == "CASSCF":
        cas = [f"  NEl {int(s['cas_nel'])}", f"  NOrb {int(s['cas_norb'])}", f"  Mult {mult}",
               f"  NRoots {int(s['cas_nroots'])}"]
        if s["cas_pt"] != "none":
            cas.append(f"  PTMethod {s['cas_pt']}")
        blocks.append("%casscf\n" + "\n".join(cas) + "\nend")
        if int(s["cas_nel"]) > 2 * int(s["cas_norb"]):
            errors.append("The active space can’t hold more than 2 electrons per orbital.")
        if int(s["cas_nel"]) > n_el:
            errors.append("The active space has more electrons than the molecule.")

    # ---------------- resources
    nprocs, maxcore = int(s["nprocs"]), int(s["maxcore"])
    if nprocs > 1:
        blocks.insert(0, f"%pal\n  nprocs {nprocs}\nend")
    blocks.insert(1 if nprocs > 1 else 0, f"%maxcore {maxcore}")
    if maxcore < 500:
        warnings.append("%maxcore is in MB per core. Values below 500 MB often make ORCA slow or fail.")

    # ---------------- extras
    extra_kw = str(s["extra_keywords"]).strip()
    if extra_kw:
        kw += extra_kw.lstrip("!").split()
    extra_blocks = str(s["extra_blocks"]).strip()
    if extra_blocks:
        blocks.append(extra_blocks)

    # ---------------- assemble
    seen, line_kw = set(), []
    for k in kw:
        if k and k.lower() not in seen:
            seen.add(k.lower())
            line_kw.append(k)
    head = [f"# {s['basename']}: {JOBS.get(job, job)}", f"# {struct.formula()}, {struct.n} atoms, "
            f"charge {s['charge']}, multiplicity {mult}", ""]
    bang = []
    line = "!"
    for k in line_kw:
        if len(line) + len(k) + 1 > 80:
            bang.append(line)
            line = "!"
        line += " " + k
    bang.append(line)

    files = {}
    xyz_lines = [f"  {sym:<3s} {x:14.8f} {y:14.8f} {z:14.8f}" for sym, (x, y, z) in zip(struct.symbols, struct.positions)]
    if s["coords"] == "file":
        coord = f"* xyzfile {s['charge']} {mult} {s['basename']}.xyz"
        files[f"{s['basename']}.xyz"] = to_xyz(struct, f"{s['basename']} initial structure")
    else:
        coord = f"* xyz {s['charge']} {mult}\n" + "\n".join(xyz_lines) + "\n*"
    text = "\n".join(head + bang) + "\n\n" + "\n\n".join(blocks) + "\n\n" + coord + "\n"
    files = {f"{s['basename']}.inp": text, **files}
    if job in ("NEB-CI", "NEB-TS") and final is not None:
        files[f"{s['basename']}_final.xyz"] = to_xyz(final, f"{s['basename']} final structure")
    files["run_orca.sh"] = (
        "#!/bin/bash\n"
        "# ORCA must be started with its full path for parallel runs.\n"
        f"$(which orca) {s['basename']}.inp > {s['basename']}.out\n")
    return files, errors, warnings


def to_xyz(struct, comment=""):
    lines = [str(struct.n), comment]
    lines += [f"{sym:<3s} {x:14.8f} {y:14.8f} {z:14.8f}" for sym, (x, y, z) in zip(struct.symbols, struct.positions)]
    return "\n".join(lines) + "\n"
