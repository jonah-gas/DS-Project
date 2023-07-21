import os
import sys

root_path = os.path.abspath(os.path.join('')) # <- adjust such that root_path always points at the root project dir
if root_path not in sys.path:
    sys.path.append(root_path)

import streamlit as st

import numpy as np
import pandas as pd

import streamlit_app.app_functions as appf # <- contains functions used in our app

### reset certain session state variables for other pages ###
st.session_state['trad_ml_skip_pred_button'] = True

### Entry page ###
st.set_page_config(
    page_title="Dev Tools",
    initial_sidebar_state='expanded'
)

appf.show_app_logo_sidebar(vertical_pos='top')

appf.header_txt("Developer Tools", lvl=1, align="center", color=None)
st.write('') # spacing

st.markdown(
    """
    - DB connection status?
    - Scraping, cleaning and inserting new data
    - Maybe require authentication?
    """
)

#appf.keep_sidebar_extended() # only use if necessary (causes slight jitter in sidebar menu)