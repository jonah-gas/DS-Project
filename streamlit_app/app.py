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


# set page config
#st.set_page_config(initial_sidebar_state='expanded')


### initialize session state ###
appf.init_session_state()
# skip submit button when (re-)loading the prediction page
st.session_state['trad_ml_skip_pred_button'] = True

### page ###

# draw app logo on page
appf.show_app_logo_sidebar(vertical_pos='top')
l, m, r = st.columns([1, 1, 1])
with m:
    appf.show_app_logo(use_column_width=True)

st.write("# Header - Our Soccer Prediction App (Wow!)")
#st.sidebar.success("Navigate pages above.")

# debug
st.write(st.session_state)


### end of loading cycle - sidebar stuff ###
appf.keep_sidebar_extended()
