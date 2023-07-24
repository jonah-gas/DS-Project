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

about_text = """
### Welcome to TOP3BETS - Your Sports Data Science Web Application!

At TOP3BETS, our mission is to provide sports enthusiasts and bettors like you with accurate predictions and comprehensive statistics for the top 5 soccer leagues of Europe. 
Developed by Data Science Master students - Jan, Jonah, and Paul, from Tübingen University. Our platform harnesses the power of data science and machine learning to enhance your soccer betting experience.

**What We Offer**:
Accurate Predictions: With our advanced machine learning models, TOP3BETS provides predictions for match outcomes and the number of goals scored in soccer matches across Europe's top 5 leagues.
Hypothetical Matchups: Curious about how your favorite teams would perform against each other? Explore hypothetical scenarios by simulating matchups and get projected results with TOP3BETS.
Transparency and Model Explanations: We believe in transparency. That's why TOP3BETS provides detailed information and explanations concerning model parameters and performance metrics.
Comprehensive Soccer Statistics: In addition to predictions, TOP3BETS offers an extensive database of soccer statistics for the top 5 European leagues. Access valuable insights on team performance and more to inform your betting decisions.

Embrace data-driven sports predictions and delve into a wealth of soccer statistics with TOP3BETS! Explore our user-friendly web interface, accessible on all devices, and level up your approach to soccer data and predictions.
"""
st.markdown(about_text) # spacing

