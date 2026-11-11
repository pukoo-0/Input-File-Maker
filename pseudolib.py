"""
pseudolib.py — read a pseudopotential library description and suggest files and cutoffs.

Supported:
  * SSSP JSON (efficiency or precision), from the SSSP website. Cutoffs in Ry, either
    "cutoff_wfc"/"cutoff_rho" or the older "cutoff"/"dual".
  * PseudoDojo .djson (hints in Hartree, converted to Ry; the "normal" hint is used).
  * A simple CSV with columns: element, filename, ecutwfc, ecutrho (cutoffs in Ry; ecutrho optional).

parse_library(raw_bytes, filename) -> {element: {"file": str, "ecutwfc": float|None, "ecutrho": float|None}}
"""

import csv
import io
import json

from elements import normalize_symbol


def _from_sssp(data):
    out = {}
    for el, d in data.items():
        if not isinstance(d, dict) or "filename" not in d:
            continue
        wfc = d.get("cutoff_wfc", d.get("cutoff"))
        rho = d.get("cutoff_rho")
        if rho is None and wfc is not None and "dual" in d:
            rho = float(wfc) * float(d["dual"])
        out[normalize_symbol(el)] = {"file": d["filename"], "ecutwfc": float(wfc) if wfc is not None else None,
                                     "ecutrho": float(rho) if rho is not None else None}
    return out


def _from_dojo(data, hint="normal"):
    out = {}
    for el, d in data["pseudos_metadata"].items():
        base = d.get("basename") or d.get("filename") or f"{el}.upf"
        stem = base.rsplit(".", 1)[0]
        hints = d.get("hints", {})
        h = hints.get(hint) or hints.get("normal") or next(iter(hints.values()), {})
        ecut_ha = h.get("ecut") if isinstance(h, dict) else None
        wfc = float(ecut_ha) * 2 if ecut_ha is not None else None
        out[normalize_symbol(el)] = {"file": stem + ".upf", "ecutwfc": wfc, "ecutrho": 4 * wfc if wfc else None}
    return out


def _from_csv(text):
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise ValueError("The CSV is empty.")
    cols = {k.lower().strip(): k for k in rows[0].keys()}
    need = ["element", "filename"]
    if not all(n in cols for n in need):
        raise ValueError("The CSV needs at least the columns: element, filename (and optionally ecutwfc, ecutrho).")
    out = {}
    for r in rows:
        el = normalize_symbol(r[cols["element"]])
        num = lambda k: float(r[cols[k]]) if k in cols and str(r[cols[k]]).strip() else None
        out[el] = {"file": r[cols["filename"]].strip(), "ecutwfc": num("ecutwfc"), "ecutrho": num("ecutrho")}
    return out


def parse_library(raw: bytes, filename: str):
    text = raw.decode("utf-8-sig", errors="replace")
    name = filename.lower()
    if name.endswith(".csv"):
        lib = _from_csv(text)
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"This isn’t valid JSON ({e}). Use an SSSP .json, a PseudoDojo .djson or a CSV.")
        lib = _from_dojo(data) if isinstance(data, dict) and "pseudos_metadata" in data else _from_sssp(data)
    if not lib:
        raise ValueError("No pseudopotentials found in this file.")
    return lib


def recommend(lib, elements):
    """Per-element table and the cutoffs for the whole structure (the highest of its elements)."""
    rows, missing = [], []
    for el in sorted(set(elements)):
        d = lib.get(el)
        if d is None:
            missing.append(el)
            continue
        rows.append({"Element": el, "File": d["file"], "ecutwfc (Ry)": d["ecutwfc"], "ecutrho (Ry)": d["ecutrho"]})
    wfcs = [r["ecutwfc (Ry)"] for r in rows if r["ecutwfc (Ry)"]]
    rhos = [r["ecutrho (Ry)"] for r in rows if r["ecutrho (Ry)"]]
    return rows, missing, (max(wfcs) if wfcs else None), (max(rhos) if rhos else None)
