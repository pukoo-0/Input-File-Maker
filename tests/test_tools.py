import json

import numpy as np
import pytest

import jobscript as J
import pseudolib as PL
import recipes as R
import settings_io as SIO


def test_slurm_script():
    name, text, submit, warnings = J.make_script({"nodes": 2, "ntasks_per_node": 16, "partition": "cpu",
                                                  "modules": "qe/7.3", "launcher": "srun"},
                                                 ["$RUN pw.x -in a.in > a.out"], "qe")
    assert name.endswith(".slurm") and submit.startswith("sbatch")
    assert "#SBATCH --nodes=2" in text and "module load qe/7.3" in text and 'RUN="srun"' in text
    assert not warnings


def test_pbs_script_and_time_check():
    name, text, submit, warnings = J.make_script({"scheduler": "pbs", "time": "1 day"}, ["x"], "orca")
    assert "#PBS -l nodes=1:ppn=4" in text and "cd $PBS_O_WORKDIR" in text
    assert warnings


def test_sssp_library():
    data = {"Fe": {"filename": "Fe.UPF", "cutoff_wfc": 90.0, "cutoff_rho": 1080.0},
            "O": {"filename": "O.UPF", "cutoff": 50.0, "dual": 8}}
    lib = PL.parse_library(json.dumps(data).encode(), "sssp.json")
    rows, missing, wfc, rho = PL.recommend(lib, ["Fe", "O", "H"])
    assert missing == ["H"] and wfc == 90.0 and rho == 1080.0
    assert lib["O"]["ecutrho"] == 400.0


def test_dojo_and_csv_libraries():
    dojo = {"pseudos_metadata": {"Si": {"basename": "Si.psp8", "hints": {"normal": {"ecut": 16}}}}}
    lib = PL.parse_library(json.dumps(dojo).encode(), "x.djson")
    assert lib["Si"]["ecutwfc"] == 32.0 and lib["Si"]["file"] == "Si.upf"
    lib = PL.parse_library(b"element,filename,ecutwfc\nO,O.upf,50\n", "x.csv")
    assert lib["O"]["file"] == "O.upf"
    with pytest.raises(ValueError):
        PL.parse_library(b"not json", "x.json")


def test_settings_round_trip():
    state = {"o_func": "PBE0", "q_k0::5": np.int64(5), "q_beta": np.float64(0.3), "ignored": 1, "q_bad": object()}
    text = SIO.dump(state, "ORCA", {"species": [{"Label": "Fe", "Mass": np.float64(55.8)}]})
    data = SIO.load(text)
    assert data["widgets"] == {"o_func": "PBE0", "q_k0::5": 5, "q_beta": 0.3}
    assert data["tables"]["species"][0]["Mass"] == 55.8
    with pytest.raises(ValueError):
        SIO.load(json.dumps({"something": "else"}))


def test_recipes_use_known_keys():
    for book in (R.ORCA, R.QE):
        for name, (desc, values) in book.items():
            assert desc and values
            assert all(k.startswith(("o_", "q_")) for k in values), name
