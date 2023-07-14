import os
import sys

root_path = os.path.abspath(os.path.join('..')) # <- adjust such that root_path always points at the root project dir (i.e. if current file is two folders deep, use '../..'). 
if root_path not in sys.path:
    sys.path.append(root_path)

import database_server.db_utilities as dbu 

import numpy as np
import pandas as pd

import streamlit as st
import streamlit_app.app_functions as appf # <- contains functions used in our app



# debug
#st.write(f"root_path: {root_path}")

### Entry page ###

### streamlit setup ###


### initialize session state with cached function calls ###
# db connection object
if 'conn' not in st.session_state:
    st.session_state['conn'] = appf.get_db_conn()
# teams data
if 'teams_df' not in st.session_state:
    st.session_state['teams_df'], st.session_state['teams_name2id'], st.session_state['teams_id2name'] = appf.get_teams_data() # cached

# load teams data



st.write("# Header - Our Soccer Prediction App (Wow!)")
st.sidebar.success("Select page above.")

st.write(st.session_state)