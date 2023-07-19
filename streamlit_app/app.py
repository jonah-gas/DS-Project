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





### initialize session state ###
appf.init_session_state()
# skip submit button when (re-)loading the prediction page
st.session_state['trad_ml_skip_pred_button'] = True

### page ###
st.write("# Header - Our Soccer Prediction App (Wow!)")
st.sidebar.success("Navigate pages above.")

# debug
st.write(st.session_state)