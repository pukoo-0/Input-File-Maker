"""
bridge.py — the connection between the web page (JavaScript) and the Python input builders.

The page runs this file inside Pyodide (Python in the browser). Every time a setting changes,
JavaScript calls build(payload_json) and gets back the input files, the checks, and the data
needed to draw the structure and the editable tables.
"""

import json

import numpy as np

import jobscript as J
import orca_input as O
import pseudolib as PL
import qe_input as Q
import structures as S
from elements import Z

_cache = {}


def _json(obj):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, tuple):
            return list(o)
        raise TypeError(type(o))
    return json.dumps(obj, default=conv, ensure_ascii=False)


def options():
    """Everything the page needs to draw its menus, taken from the builders so there is one source of truth."""
    qd = {k: (list(v) if isinstance(v, tuple) else v) for k, v in Q.DEFAULTS.items()}
    return _json({
        "orca": {"functionals": O.FUNCTIONALS, "basis_sets": O.BASIS_SETS, "composite": O.COMPOSITE, "xtb": O.XTB,
                 "wavefunction": O.WAVEFUNCTION, "solvents": O.SOLVENTS, "jobs": O.JOBS, "defaults": O.DEFAULTS},
        "qe": {"workflows": Q.WORKFLOWS, "calculations": Q.CALCULATIONS, "functionals": list(Q.FUNCTIONALS),
               "hybrids": [k for k, v in Q.FUNCTIONALS.items() if v in Q.HYBRIDS], "pp_plots": Q.PP_PLOTS,
               "defaults": qd, "ry_au_fs": Q.RY_AU_FS},
        "job": J.DEFAULTS,
    })


def _parse(name, text):
    key = (name, hash(text))
    if key not in _cache:
        if len(_cache) > 8:
            _cache.clear()
        _cache[key] = S.read_structure(text.encode("utf-8"), name)
    return _cache[key]


def _apply_ops(st, ops):
    for op in ops or []:
        kind = op[0]
        if kind == "primitive":
            st = S.primitive_cell(st, float(op[1]))
        elif kind == "supercell":
            st = S.supercell(st, int(op[1]), int(op[2]), int(op[3]))
        elif kind == "vacuum":
            st = S.add_vacuum(st, int(op[1]), float(op[2]))
        elif kind == "slab":
            st = S.make_slab(st, [int(x) for x in op[1]], int(op[2]), float(op[3]))
    return st


def _info(st):
    info = {"formula": st.formula(), "n": st.n, "periodic": st.periodic, "notes": list(st.notes),
            "source": st.source, "spacegroup": None}
    if st.periodic:
        a, b, c, al, be, ga = st.cell_parameters()
        info.update(cell_params=[a, b, c, al, be, ga], volume=st.volume())
        try:
            info["spacegroup"] = S.symmetry_info(st)
        except Exception:
            pass
    return info


def _view(st, highlight):
    return {"symbols": list(st.symbols), "positions": st.positions.tolist(),
            "cell": st.cell.tolist() if st.periodic else None, "highlight": sorted(set(int(i) for i in highlight))}


def library(name, text, elements_json):
    """Read a pseudopotential library file and recommend files and cutoffs for these elements."""
    try:
        lib = PL.parse_library(text.encode("utf-8"), name)
        rows, missing, wfc, rho = PL.recommend(lib, json.loads(elements_json))
        return _json({"ok": True, "rows": rows, "missing": missing, "ecutwfc": wfc, "ecutrho": rho,
                      "files": {el: d["file"] for el, d in lib.items()}})
    except Exception as e:
        return _json({"ok": False, "error": str(e)})


def band_path(cell_json, points):
    try:
        path = S.band_path(np.array(json.loads(cell_json)), int(points))
        return _json({"ok": True, "rows": [{"label": l, "k1": k[0], "k2": k[1], "k3": k[2], "n": w} for l, k, w in path]})
    except Exception as e:
        return _json({"ok": False, "error": str(e)})


def have(module):
    try:
        __import__(module)
        return True
    except Exception:
        return False


def build(payload):
    p = json.loads(payload)
    out = {"files": {}, "errors": [], "warnings": [], "info": None, "view": None, "submit": None}
    s_in = p.get("structure")
    if not s_in:
        out["errors"].append("Upload a structure, or pick an example, to start.")
        return _json(out)
    try:
        base = _parse(s_in["name"], s_in["text"])
    except Exception as e:
        out["errors"].append(f"Couldn’t read the structure: {e}")
        return _json(out)
    try:
        st = _apply_ops(base, p.get("ops"))
    except Exception as e:
        out["errors"].append(f"Structure tool failed, so the original structure is used: {e}")
        out["ops_failed"] = True
        st = base
    final = None
    f_in = p.get("final")
    if f_in:
        try:
            final = _parse(f_in["name"], f_in["text"])
        except Exception as e:
            out["errors"].append(f"Couldn’t read the final structure: {e}")
    out["info"] = _info(st)

    if p["program"] == "ORCA":
        s = dict(p.get("orca") or {})
        files, errors, warnings = O.build_orca(s, st, final)
        out["info"]["electrons"] = O.electron_count(st.symbols, s.get("charge", 0))
        base_name = s.get("basename", "job")
        commands = [f"$(which orca) {base_name}.inp > {base_name}.out"]
        fix = [int(x) for x in str(s.get("fix_atoms", "")).replace(",", " ").split() if x.isdigit()]
        out["view"] = _view(st, [i for i in fix if i < st.n])
        tag = "orca"
    else:
        c = dict(p.get("qe") or {})
        qs, qfinal = st, final
        if not st.periodic:
            qs = S.boxed(st, float(c.get("vacuum", 10.0)))
            if final is not None and not final.periodic:
                shift = qs.positions[0] - st.positions[0]
                qfinal = S.Structure(final.symbols, final.positions + shift, qs.cell)
        slab_axis = S.vacuum_axis(qs)
        out["info"].update(slab_axis=slab_axis, box=qs.cell[0][0] if not st.periodic else None,
                           cell=qs.cell.tolist())

        atoms = p.get("atoms") or []
        if len(atoms) == qs.n:
            labels = [str(a.get("label") or el).strip() or el for a, el in zip(atoms, qs.symbols)]
            fixed = np.array([[bool(a.get("fx")), bool(a.get("fy")), bool(a.get("fz"))] for a in atoms])
        else:
            labels, fixed = list(qs.symbols), qs.fixed.copy()
        frac = qs.fractional()
        out["atoms"] = [{"el": el, "label": lab, "fx": bool(f[0]), "fy": bool(f[1]), "fz": bool(f[2]),
                         "a": float(fr[0]), "b": float(fr[1]), "c": float(fr[2])}
                        for el, lab, f, fr in zip(qs.symbols, labels, fixed, frac)]

        template = c.get("pseudo_template") or "{El}.upf"
        user = {r.get("Label"): r for r in (p.get("species") or [])}
        species = []
        for r in Q.species_table(qs, labels):
            u = user.get(r["Label"])
            if u and u.get("Element") == r["Element"]:
                for k in ("Mass", "Start. magnetisation", "Hubbard U (eV)", "U manifold"):
                    if k in u and u[k] not in (None, ""):
                        r[k] = u[k]
                if u.get("Pseudopotential") and not u.get("auto", True):
                    r["Pseudopotential"] = u["Pseudopotential"]
                    r["auto"] = False
                else:
                    r["Pseudopotential"] = template.replace("{El}", r["Element"])
                    r["auto"] = True
            else:
                r["Pseudopotential"] = template.replace("{El}", r["Element"])
                r["auto"] = True
            for k in ("Mass", "Start. magnetisation", "Hubbard U (eV)"):
                try:
                    r[k] = float(r[k])
                except (TypeError, ValueError):
                    r[k] = 0.0
            species.append(r)
        out["species"] = species

        mult = 8 if c.get("pseudo_type", "US/PAW") == "US/PAW" else 4
        if c.get("ecutrho_auto", True):
            c["ecutrho"] = mult * float(c.get("ecutwfc", 50.0))
        auto = S.kgrid_from_spacing(qs.cell, float(c.get("kspacing", 0.25)), slab_axis)
        out["info"]["kgrid_auto"] = auto
        if c.get("kgrid_auto", True):
            c["kgrid"] = auto
        dens = [max(1, 2 * int(x)) for x in c.get("kgrid", auto)]
        if slab_axis is not None:
            dens[slab_axis] = 1
        out["info"]["nscf_auto"] = dens
        if c.get("nscf_auto", True):
            c["nscf_kgrid"] = dens
        if not c.get("tot_mag_on"):
            c["tot_magnetization"] = None
        rows = p.get("path") or []
        needs_path = c.get("workflow") in ("bands", "phonon") or (c.get("workflow") == "single" and c.get("calculation") == "bands")
        if needs_path and p.get("path_placeholder"):
            out["warnings"].append("The k-point path is still the placeholder Γ → X. Use “Find automatically” in the "
                                   "k-points tab, or enter the right path for your lattice.")
        c["path"] = [(str(r["label"]), (float(r["k1"]), float(r["k2"]), float(r["k3"])), int(r["n"]))
                     for r in rows if str(r.get("label", "")).strip()] or None
        c["extra_vars"] = [{"Namelist": r.get("nl", ""), "Variable": r.get("var", ""), "Value": r.get("val", "")}
                           for r in (p.get("extra") or [])]
        if c.get("workflow") == "convergence":
            try:
                c["conv_kspacings"] = tuple(float(x) for x in str(c.get("conv_kspacings_text", "")).replace(",", " ").split())
            except ValueError:
                out["errors"].append("Write the k-point spacings as numbers separated by spaces.")
        species_for_build = [{k: v for k, v in r.items() if k != "auto"} for r in species]
        files, errors, warnings, commands = Q.build_qe(c, qs, species_for_build, labels, fixed, qfinal)
        out["view"] = _view(qs, [i for i in range(qs.n) if fixed[i].any()])
        tag = "qe"

    job = p.get("job") or {}
    if job.get("enabled"):
        name, text, submit, jw = J.make_script(job, commands, tag)
        files[name] = text
        warnings += jw
        out["submit"] = submit
        if tag == "orca":
            ntasks = int(job.get("nodes", 1)) * int(job.get("ntasks_per_node", 1))
            if ntasks != int((p.get("orca") or {}).get("nprocs", 4)):
                warnings.append("The job script’s MPI tasks (nodes × tasks per node) don’t match ORCA’s %pal nprocs.")
    out["files"] = files
    out["errors"] += errors
    out["warnings"] += warnings
    return _json(out)
