"""
settings_io.py — save the app’s settings to a small JSON file and load them back.

Widget values are stored by their keys (prefixes o_ for ORCA, q_ for Quantum ESPRESSO, j_ for the
cluster script). Tables (atom labels, fixed atoms, species, extra variables, band path) are stored
separately because they aren’t simple widget values.
"""

import json

import numpy as np

PREFIXES = ("o_", "q_", "j_")
VERSION = 1


def _plain(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, tuple):
        return list(v)
    return v


def dump(state: dict, program: str, tables: dict) -> str:
    widgets = {}
    for k, v in state.items():
        if isinstance(k, str) and k.startswith(PREFIXES):
            v = _plain(v)
            try:
                json.dumps(v)
            except TypeError:
                continue
            widgets[k] = v
    clean_tables = {}
    for name, rows in (tables or {}).items():
        clean_tables[name] = [{k: _plain(v) for k, v in r.items()} for r in rows] if rows is not None else None
    return json.dumps({"app": "DFT Input Maker", "version": VERSION, "program": program,
                       "widgets": widgets, "tables": clean_tables}, indent=1, ensure_ascii=False)


def load(text: str) -> dict:
    data = json.loads(text)
    if not isinstance(data, dict) or data.get("app") != "DFT Input Maker":
        raise ValueError("This isn’t a settings file from the DFT Input Maker.")
    if int(data.get("version", 0)) > VERSION:
        raise ValueError("This settings file comes from a newer version of the app.")
    data.setdefault("widgets", {})
    data.setdefault("tables", {})
    return data
