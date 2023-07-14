import os
import sys

root_path = os.path.abspath(os.path.join('..')) # <- adjust such that root_path always points at the root project dir (i.e. if current file is two folders deep, use '../..'). 
if root_path not in sys.path:
    sys.path.append(root_path)

import streamlit as st
import database_server.db_utilities as dbu
from models.trad_ml.feature_generation import FeatureGen



####################
# cached functions #
####################

# get db connection object
@st.cache_resource(ttl=60*60*24*7, max_entries=1, show_spinner="Creating database connection...")
def get_db_conn():
    return dbu.get_conn(config=st.secrets)

# load teams data (e.g. for prediction selection)
@st.cache_data(ttl=60*60*24*7, max_entries=1, show_spinner="Retrieving teams data...")
def get_teams_data():
    # load full teams table
    teams_df = dbu.select_query("SELECT * FROM teams;", conn=st.session_state['conn'])
    # create matching dicts
    teams_name2id = {row[0]: row[1] for row in teams_df[['name', 'id']].itertuples(index=False)}
    teams_id2name = {row[1]: row[0] for row in teams_df[['name', 'id']].itertuples(index=False)}

    return teams_df, teams_name2id, teams_id2name

# create feature gen instance and load full db dataset 
@st.cache_resource(ttl=60*60*24*7, max_entries=1, show_spinner="Retrieving match data...")
def get_feature_gen_instance():
    # create feature gen instance
    fg = FeatureGen(conn=st.session_state['conn'], dpo_path=os.path.join(root_path, 'models', 'trad_ml', 'saved_data_prep_objects') ) # note: no params dict provided yet -> need to call set_params() later
    # load full dataset
    fg.load_data()
    return fg

#####################
# styling utilities #
#####################

# aligned text
def aligned_text(text, align='center', bold=False, color=None, font_size=None):
    if bold:
        text = f"<b>{text}</b>"
    if color:
        text = f"<span style='color:{color};'>{text}</span>"
    if font_size:
        text = f"<span style='font-size:{font_size};'>{text}</span>"
    return st.markdown(f'<div style="text-align: {align};">{text}</div>', unsafe_allow_html=True)








