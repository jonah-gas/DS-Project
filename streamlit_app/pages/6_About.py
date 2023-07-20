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

st.set_page_config(
    page_title="About",
    initial_sidebar_state='expanded'
)

appf.show_app_logo_sidebar(vertical_pos='top')

st.write("# Project info...")