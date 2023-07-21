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


appf.header_txt("Project Info, FAQ & Disclaimer", lvl=1, align="center", color=None)
st.write('') # spacing

