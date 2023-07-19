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
# db connection object
if 'conn' not in st.session_state:
    st.session_state['conn'] = appf.get_db_conn()
# teams data
if 'teams_df' not in st.session_state:
    st.session_state['teams_df'], st.session_state['teams_name2id'], st.session_state['teams_id2name'] = appf.get_teams_data() # cached
# tradml preds: selected team ids
if 'trad_ml_home_team_select_id' not in st.session_state:
    appf.update_session_state_tradml_selections(home_id=135, away_id=122)
# skip submit button when (re-)loading the prediction page
st.session_state['trad_ml_skip_pred_button'] = True

### page ###
st.write("# Header - Our Soccer Prediction App (Wow!)")
st.sidebar.success("Select page above.")

# debug
st.write(st.session_state)