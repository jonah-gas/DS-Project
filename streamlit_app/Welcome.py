# Run: streamlit run streamlit_app\app.py
# (important: streamlit run app.py will mess up relative paths for imports)

import os
import sys

root_path = os.path.abspath(os.path.join('')) # <- adjust such that root_path always points at the root project dir
if root_path not in sys.path:
    sys.path.append(root_path)

import streamlit as st

import numpy as np
import pandas as pd

import streamlit_app.app_functions as appf # <- contains functions used in our app

### page setup (visual) ###
st.set_page_config(initial_sidebar_state='expanded', page_title="Welcome!", layout="wide")
appf.hide_image_fullscreen_option()
appf.show_app_logo_sidebar(vertical_pos='top')

# draw app logo on main page
l, m, r = st.columns([1, 1.5, 1])
with m:
    appf.show_app_logo(use_column_width=True)

### initialize session state ###
appf.init_session_state()

### content ###
appf.header_txt("Top3Bets - THE place for soccer predictions!", lvl=1, align="center", color=None)
st.divider()
appf.header_txt('Welcome!', lvl=3, align="center", color=None)
appf.aligned_text("Please use the sidebar to navigate between pages.", align="center")



### end of loading cycle - sidebar & other stuff ###

