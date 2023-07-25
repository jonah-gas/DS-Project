import os
import sys

root_path = os.path.abspath(os.path.join('')) # <- adjust such that root_path always points at the root project dir
if root_path not in sys.path:
    sys.path.append(root_path)

import streamlit as st

import numpy as np
import pandas as pd

import streamlit_app.app_functions as appf # <- contains functions used in our app


### page setup (visual)
st.set_page_config(initial_sidebar_state='expanded')
appf.show_app_logo_sidebar(vertical_pos='top')


appf.header_txt("Developer Tools", lvl=1, align="center", color=None)
st.write('') # spacing

db_online = appf.check_db_connection()

# get current timestamp and format it
timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

st.write("### Database connection status:")
if db_online:
    st.success(f"✅ Database connection is online. (Last checked: {timestamp})")
else:
    st.error("❌ Database connection is offline. (Last checked: {timestamp})")
# streamlit button, if submitted: recreate db connection object and try again
if st.button("Retry", key="recheck_db_connection"): # refresh db connection status
    # clear cache for db object 
    appf.get_db_conn.clear()
    # update session state with new conn object
    st.session_state['db_conn'] = appf.get_db_conn()



st.divider()

st.write("### Scraping & Inserting New Data")

st.warning("⚠️ This functionality is currently disabled to prevent unauthorized database manipulation. Please refer to the sections on scraping and database inserts in the [github repo's](https://github.com/jonah-gas/DS-Project) ``README.md``.")


#appf.keep_sidebar_extended() # only use if necessary (causes slight jitter in sidebar menu)