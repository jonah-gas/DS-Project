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
st.set_page_config(initial_sidebar_state='expanded')
appf.hide_image_fullscreen_option()
appf.show_app_logo_sidebar(vertical_pos='top')

### session state updates ###
appf.init_session_state()

appf.header_txt("Statistics", lvl=1, align="center", color=None)
st.write('') # spacing

st.markdown(
    """
    Stats based on our collected data.
    """
)

### end of loading cycle - sidebar stuff ###
#appf.keep_sidebar_extended() # only use if necessary (causes slight jitter in sidebar menu)
