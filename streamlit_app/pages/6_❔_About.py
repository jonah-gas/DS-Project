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


appf.header_txt("About this Project", lvl=1, align="center", color=None)
### formatted text ###

st.markdown("""At TOP3BETS, our mission is to provide sports enthusiasts and bettors like you with accurate predictions and comprehensive statistics for the top 5 soccer leagues of Europe. 
Developed by three Data Science students - Jan Jacobsen, Jonah Gasparro, and Paul Aspacher, from Tübingen University. Our platform harnesses the power of data science and machine learning to enhance your soccer betting experience.""")

st.divider()

appf.header_txt("What we offer", lvl=4, align="center", color=None)
st.markdown("""**Predictions**:""")
st.markdown("""- TOP3BETS provides predictions for match outcomes and the number of goals scored in soccer matches across Europe's top 5 leagues, based on our advanced machine learning models.""")
st.markdown("""**Hypothetical Matchups**:""")
st.markdown("""- Curious about how your favorite teams would perform against each other? Explore hypothetical scenarios by simulating matchups and get projected results with TOP3BETS.""")
st.markdown("""**Transparency and Model Explanations**:""")
st.markdown("""- We believe in transparency. That's why TOP3BETS provides detailed information and explanations concerning model parameters and performance metrics.""")
st.markdown("""**Comprehensive Soccer Statistics**:""")
st.markdown("""- In addition to predictions, TOP3BETS offers an extensive database of soccer statistics for the top 5 European leagues. Access valuable insights on team performance and more to inform your betting decisions.""")
st.write("") # spacing
st.markdown("""Embrace data-driven sports predictions and delve into a wealth of soccer statistics with TOP3BETS! Explore our user-friendly web interface, accessible on all devices, and elevate your approach to soccer data and predictions.""")

st.divider()

appf.header_txt("Disclaimers", lvl=4, align="center", color=None)

st.markdown("""**Prediction Accuracy**""")
st.markdown("""- While we do our best to improve and test our models, we cannot make any guarantees about model performance on matches in the future.
            We do not recommend users to bet on matches based on our predictions alone. Please use our predictions as a reference only and combine them with your own research and knowledge.""")

st.markdown("""**Responsible Betting**""")
st.markdown("""- TOP3BETS is not a betting platform. We do not offer any betting services or accept bets. We are not responsible for any losses incurred by our users on other, non-affiliated, platforms.
              If gambling is affecting your life adversely, please refer to one of the following resources for help: [USA](https://www.ncpgambling.org/help-treatment/national-helpline-1-800-522-4700/), [UK](https://www.begambleaware.org/), [Germany](https://www.spielen-mit-verantwortung.de/hilfe-und-beratung/)""")

st.markdown("""**Team Logos**""")
st.markdown("""- All team logos used on this website are the property of their respective owners.""")

st.divider()

appf.header_txt("Github Repo", lvl=4, align="center", color=None)
st.markdown("""The source code for this project is available on [Github](https://github.com/jonah-gas/DS-Project).""")




