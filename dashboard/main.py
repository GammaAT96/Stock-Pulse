import logging
import os
import sys
from pathlib import Path

# Streamlit sets sys.path[0] to this folder; project root must be on path for `import dashboard`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if not logging.root.handlers:
    _lvl = getattr(logging, os.getenv("LOG_LEVEL", "WARNING").upper(), logging.WARNING)
    logging.basicConfig(level=_lvl, format="%(levelname)s %(name)s: %(message)s")

import streamlit as st

from dashboard.views import ml_strategy, stock_analysis

st.set_page_config(page_title="StockPulse Dashboard", layout="wide")

theme = st.sidebar.radio("Theme", ["Dark", "Light"], index=0, horizontal=True)
st.session_state["theme"] = theme.lower()

if theme == "Dark":
    st.markdown(
        """
<style>
:root { color-scheme: dark; }
[data-testid="stAppViewContainer"] { background-color: #0e1117; color: #fafafa; }
section[data-testid="stSidebar"] { background-color: #111827; }
</style>
""",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
<style>
:root { color-scheme: light; }
[data-testid="stAppViewContainer"] { background-color: #ffffff; color: #111827; }
section[data-testid="stSidebar"] { background-color: #f8fafc; }

/* Make BaseWeb widgets readable in light mode */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {
  background-color: #ffffff !important;
  color: #111827 !important;
}

div[data-baseweb="select"] *,
div[data-baseweb="input"] *,
div[data-baseweb="textarea"] *,
label, p, span, div, small {
  color: #111827;
}

/* Date input + number input inner text */
input, textarea {
  color: #111827 !important;
  background-color: #ffffff !important;
}

/* Slider track */
div[data-baseweb="slider"] div {
  color: #111827 !important;
}
</style>
""",
        unsafe_allow_html=True,
    )

st.sidebar.title("Navigation")

page = st.sidebar.selectbox(
    "Go to",
    ["Stock Analysis", "ML Strategy"],
)

if page == "Stock Analysis":
    stock_analysis.show()
elif page == "ML Strategy":
    ml_strategy.show()