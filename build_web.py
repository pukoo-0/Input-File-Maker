"""
build_web.py — makes the single-file web version (docs/index.html) from web/index.src.html.

The Python modules are embedded in the page so it works from one file: on GitHub Pages,
inside Streamlit, or opened directly from your computer.

Run after changing any of the Python modules:  python build_web.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MODULES = ["elements.py", "structures.py", "orca_input.py", "qe_input.py", "jobscript.py", "pseudolib.py",
           os.path.join("web", "bridge.py")]


def main():
    files = {}
    for rel in MODULES:
        with open(os.path.join(HERE, rel), encoding="utf-8") as fh:
            files[os.path.basename(rel)] = fh.read()
    with open(os.path.join(HERE, "web", "index.src.html"), encoding="utf-8") as fh:
        template = fh.read()
    blob = json.dumps(files, ensure_ascii=False).replace("</", "<\\/")
    marker = "/*__PY_FILES__*/{}"
    assert marker in template, "placeholder not found in the template"
    html = template.replace(marker, blob, 1)
    os.makedirs(os.path.join(HERE, "docs"), exist_ok=True)
    out = os.path.join(HERE, "docs", "index.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Wrote {out} ({len(html) // 1024} KB, {len(files)} Python modules embedded)")


if __name__ == "__main__":
    main()
