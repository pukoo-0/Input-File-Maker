"""
recipes.py — quick-start settings. Each recipe is a set of widget values (by key) applied in one click,
plus a short description. You can change anything afterwards.
"""

ORCA = {
    "Fast optimisation + frequencies (r2SCAN-3c)": (
        "A cheap, reliable composite method for geometries and thermochemistry of molecules.",
        {"o_fam": "Composite", "o_comp": "r2SCAN-3c", "o_job": "OptFreq", "o_scfc": "TightSCF"}),
    "Accurate single-point energy (DLPNO-CCSD(T))": (
        "Near coupled-cluster accuracy for a geometry optimised with DFT.",
        {"o_fam": "Wavefunction", "o_wf": "DLPNO-CCSD(T)", "o_pno": "TightPNO", "o_bgrp": "Karlsruhe (def2)",
         "o_basis": "def2-TZVPP", "o_job": "SP"}),
    "Transition state search (B3LYP-D3BJ/def2-SVP)": (
        "Finds a saddle point and checks it has exactly one imaginary frequency.",
        {"o_fam": "DFT", "o_fgrp": "Hybrid", "o_func": "B3LYP", "o_disp": "D3BJ", "o_bgrp": "Karlsruhe (def2)",
         "o_basis": "def2-SVP", "o_job": "OptTSFreq"}),
    "UV-Vis spectrum in water (TD-DFT)": (
        "20 excited states with a range-separated hybrid and implicit water.",
        {"o_fam": "DFT", "o_fgrp": "Range-separated hybrid", "o_func": "CAM-B3LYP", "o_disp": "D3BJ",
         "o_bgrp": "Karlsruhe (def2)", "o_basis": "def2-TZVP", "o_job": "SP", "o_tddft": True, "o_nroots": 20,
         "o_solv": "CPCM", "o_solvent": "water"}),
    "Transition-metal complex (TPSSh-D3BJ)": (
        "A robust setup for metal complexes, with help for difficult SCF convergence.",
        {"o_fam": "DFT", "o_fgrp": "Hybrid", "o_func": "TPSSh", "o_disp": "D3BJ", "o_bgrp": "Karlsruhe (def2)",
         "o_basis": "def2-TZVP", "o_job": "Opt", "o_slow": "SlowConv", "o_stab": True}),
    "Conformer search (GOAT with XTB2)": (
        "Finds low-energy conformers quickly. Needs ORCA 6.",
        {"o_fam": "xTB", "o_xtb": "XTB2", "o_job": "GOAT"}),
    "Molecular dynamics (XTB2, 300 K)": (
        "A 10 ps run with a CSVR thermostat.",
        {"o_fam": "xTB", "o_xtb": "XTB2", "o_job": "MD", "o_mdts": 0.5, "o_mdtemp": 300.0, "o_mdthermo": "CSVR",
         "o_mdsteps": 20000}),
}

QE = {
    "Quick SCF check": (
        "One self-consistent run to make sure the structure and pseudopotentials work.",
        {"q_wf": "single", "q_calc": "scf", "q_occ": "smearing", "q_conv": 1e-8}),
    "Convergence tests (do this first)": (
        "Cutoff and k-point series with a script that collects the energies.",
        {"q_wf": "convergence", "q_convmode": "both"}),
    "Full crystal relaxation (vc-relax)": (
        "Relaxes atoms and the cell together. Uses a higher cutoff to avoid Pulay stress.",
        {"q_wf": "single", "q_calc": "vc-relax", "q_ecut": 60.0, "q_forc": 1e-4, "q_etot": 1e-5}),
    "Semiconductor band structure": (
        "scf followed by bands along the high-symmetry path.",
        {"q_wf": "bands", "q_occ": "fixed"}),
    "DOS and projected DOS": (
        "scf, a denser nscf with the tetrahedron method, then dos.x and projwfc.x.",
        {"q_wf": "dos"}),
    "Slab with adsorbate (relax)": (
        "Relaxation with a dipole correction and local-TF mixing. Fix the bottom layers in the Atoms tab.",
        {"q_wf": "single", "q_calc": "relax", "q_dip": True, "q_mixmode": "local-TF", "q_beta": 0.2}),
    "Magnetic oxide with DFT+U": (
        "Spin-polarised. Set starting magnetisation and U values in the Species tab.",
        {"q_wf": "single", "q_calc": "scf", "q_spin": "collinear", "q_hubstyle": "card",
         "q_hubproj": "ortho-atomic"}),
    "Ab initio MD (NVT, 300 K)": (
        "Born–Oppenheimer MD with an SVR thermostat and symmetry turned off.",
        {"q_wf": "md", "q_mdens": "NVT", "q_mddt": 1.0, "q_nstep": 2000, "q_mdthermo": "svr", "q_tempw": 300.0,
         "q_nosym": True, "q_verb": "low"}),
    "Phonons at Γ (IR / Raman ready)": (
        "Γ-point phonons with the dielectric tensor and Born charges (insulators).",
        {"q_wf": "phonon", "q_phgamma": True, "q_epsil": True, "q_occ": "fixed"}),
}
