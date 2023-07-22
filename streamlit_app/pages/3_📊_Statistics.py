import os
import sys

root_path = os.path.abspath(os.path.join('')) # <- adjust such that root_path always points at the root project dir
if root_path not in sys.path:
    sys.path.append(root_path)

import streamlit as st

import numpy as np
import pandas as pd

import streamlit_app.app_functions as appf # <- contains functions used in our app
import database_server.db_utilities as dbu

### page setup (visual) ###
st.set_page_config(initial_sidebar_state='expanded', layout='wide')
appf.hide_image_fullscreen_option()
appf.show_app_logo_sidebar(vertical_pos='top')

### session state updates ###
appf.init_session_state()

### content ###
appf.header_txt("Statistics", lvl=1, align="center", color=None)
st.write('') # spacing



## stat: team-level aggregate stats by league and season -> displayed in table
appf.header_txt("Season aggregates by team (selected statistics)", lvl=3, align="left", color=None)
with st.container():
    t1, t2, t3, t4, t5, t6 = st.columns(6) 
    with t1:
        league_ids, league_names = appf.get_league_options()
        league_name_selection = st.selectbox('League:', options=league_names, index=0)
    with t2:
        season_str_options = appf.get_season_str_options()
        season_str_selection = st.selectbox('Season:', options=season_str_options, index=0)
    with t3:
        agg_type = st.selectbox('Aggregation type:', ['AVG', 'SUM', 'MIN', 'MAX'], index=0)

    # display dataframe
    st.write("(sort by clicking on column headers)")
    st.dataframe(appf.get_aggregated_stats(agg_type, season_str_selection, league_ids[league_names.index(league_name_selection)]), hide_index=True)


# stat: 


### end of loading cycle - sidebar stuff ###
#appf.keep_sidebar_extended() # only use if necessary (causes slight jitter in sidebar menu)
