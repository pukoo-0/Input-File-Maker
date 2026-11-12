# DFT Input Maker ⚛️

[![tests](../../actions/workflows/tests.yml/badge.svg)](../../actions/workflows/tests.yml)

A Streamlit app that builds input files for **ORCA** and **Quantum ESPRESSO**, with checks for common mistakes,
a 3D structure viewer, and cluster job scripts.

## Two versions

- **Web version** (`docs/index.html`): a fully custom interface with a live preview of every input file, an interactive
  3D viewer and editable tables. Python runs inside the browser (Pyodide), so nothing is sent to a server. Host it free
  on **GitHub Pages** (Settings → Pages → branch `main`, folder `/docs`), deploy `streamlit_web.py` on Streamlit, or
  simply open the file. The first start takes 10–20 seconds.
- **Streamlit version** (`app.py`): the original interface. It also has the primitive-cell tool, which needs spglib
  (that library may not be available in the browser).

Both versions use the same Python input builders, so their input files are identical.

## Highlights

- **3D viewer** of the structure (rotate, zoom, repeat the cell), with fixed atoms shown in red
- **Structure tools:** space group, primitive cell, supercells, slabs from Miller indices, adding vacuum
- **Quick-start recipes** for common jobs (optimisation, TS search, UV-Vis, DLPNO-CCSD(T), band structure, DOS,
  slabs, DFT+U, MD, phonons, convergence tests)
- **Convergence tests:** a cutoff and k-point series plus a script that collects the energies, makes a table and a plot
- **Molecular dynamics** in both programs (QE: NVE/NVT/NPT with thermostats; ORCA: the %md module)
- **Pseudopotential libraries:** upload the SSSP .json, a PseudoDojo .djson or your own CSV, and the app fills in
  file names and recommended cutoffs
- **Cluster scripts** for SLURM and PBS, with modules, partitions and accounts
- **Save and load settings** as a small JSON file, to repeat a setup exactly or share it

## Structures

Upload `.xyz` (plain or extended), `.cif` (symmetry operations are applied) or VASP `POSCAR`/`CONTCAR`
(selective dynamics flags become fixed coordinates). Molecules are boxed automatically for Quantum ESPRESSO.

## ORCA

- **Methods:** DFT (GGA to double hybrids), HF, 3c composite methods, GFN-xTB, MP2 / CCSD(T) / DLPNO-CCSD(T),
  CASSCF / NEVPT2, with automatic RI auxiliary basis sets
- **Basis sets:** def2, Dunning, Pople, Jensen and relativistic (ZORA, DKH, X2C, SARC)
- **Jobs:** single point, optimisation, frequencies, TS search, relaxed/rigid scans, IRC, NEB, GOAT, MD
- **Extras:** dispersion, CPCM / SMD / ALPB, TD-DFT, NMR, EPR, charges, constraints, broken symmetry, stability,
  MORead, grids, and free-form keywords and blocks
- **Checks:** charge/multiplicity, basis sets missing for heavy elements, NEB end points, Raman settings, and more

## Quantum ESPRESSO

- **Workflows:** single pw.x run (scf, relax, vc-relax, nscf, bands, md, vc-md), band structure, DOS/PDOS, pp.x maps,
  phonons (ph.x → q2r.x → matdyn.x), NEB, MD, and convergence tests. Each comes with a run script.
- **Settings:** cutoffs, smearing, spin (collinear, non-collinear, spin–orbit), species splitting for antiferromagnets,
  DFT+U, hybrids, nonlocal vdW, D2/D3/TS/MBD/XDM, isolated-system corrections, dipole correction, fixed atoms,
  k-grids from a spacing, automatic band paths (ASE), and any extra namelist variable or card
- **Checks:** ecutrho vs ecutwfc, missing pseudopotentials, spin settings QE would reject, label length, mismatched
  dynamics, slab k-points, MD time steps, hybrid limitations, and more

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Updating the web version

The web page embeds the Python modules. After changing any of them, rebuild it:

```bash
python build_web.py
```

A test checks that `docs/index.html` matches the current modules, so GitHub will tell you if you forget.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests run automatically on GitHub for every push (see `.github/workflows/tests.yml`).

## Files

| File | What it does |
|---|---|
| `app.py` | Streamlit interface |
| `docs/index.html` | Web version (built file, don’t edit by hand) |
| `web/index.src.html` | Web version source: layout, styling and interface code |
| `web/bridge.py` | Connects the web page to the Python builders |
| `build_web.py` | Builds `docs/index.html` |
| `streamlit_web.py` | Serves the web version from Streamlit |
| `orca_input.py` | ORCA input builder and checks |
| `qe_input.py` | Quantum ESPRESSO input builders, workflows and checks |
| `structures.py` | Structure readers, structure tools, k-grids, band paths, 3D viewer |
| `jobscript.py` | SLURM / PBS job scripts |
| `pseudolib.py` | Pseudopotential library readers (SSSP, PseudoDojo, CSV) |
| `recipes.py` | Quick-start recipes |
| `settings_io.py` | Saving and loading settings |
| `elements.py` | Atomic numbers and masses |
| `tests/` | Automated tests |

## Always check before running

The app catches common mistakes, but it can’t know your system. Read each file before submitting it, make sure your
pseudopotential file names match the files in `pseudo_dir`, and run a quick test on a small system first.
