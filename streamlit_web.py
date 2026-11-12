"""
Streamlit wrapper for the web version (docs/index.html).
Deploy with this file as the main file to get the new interface inside Streamlit.
The page itself runs Python in the browser (Pyodide), so Streamlit only serves it.
"""
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="DFT Input Maker", page_icon="⚛️", layout="wide")
st.markdown("<style>.block-container{padding:0 !important;max-width:100% !important}"
            "[data-testid='stHeader']{display:none}</style>", unsafe_allow_html=True)
components.html((Path(__file__).parent / "docs" / "index.html").read_text(encoding="utf-8"), height=1500, scrolling=True)
