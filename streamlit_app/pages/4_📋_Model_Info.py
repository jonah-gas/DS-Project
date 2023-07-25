import os
import sys

root_path = os.path.abspath(os.path.join('')) # <- adjust such that root_path always points at the root project dir (i.e. if current file is two folders deep, use '../..'). 
if root_path not in sys.path:
    sys.path.append(root_path)

import streamlit as st
import streamlit_app.app_functions as appf # <- contains functions used in our app


### page setup (visual) ###
st.set_page_config(initial_sidebar_state='expanded')
appf.hide_image_fullscreen_option()
appf.show_app_logo_sidebar(vertical_pos='top')

# make sure models are loaded
appf.load_trad_ml_models()
appf.load_lstm_models()

appf.header_txt("Model Information & Metrics", lvl=1, align="center", color=None)
st.write('') # spacing

# model (type) 1
name_1 = 'XGBoost'
st.write(f"## XGBoost")
st.markdown("""**XGBoost**, short for "Extreme Gradient Boosting," is an advanced algorithm used for classification tasks. It employs a boosting technique where weak learners, like small decision trees, are iteratively trained to correct the mistakes of their predecessors.""")

with st.expander("View parameters", expanded=False):
    st.write("### Feature generation parameters:")
    st.write(st.session_state['trad_ml_models'][name_1]['info']['fg_config'])
    st.write("### Model parameters:")
    st.write(st.session_state['trad_ml_models'][name_1]['info']['model_config'])
with st.expander("View performance metrics", expanded=False):
    st.write(f"{name_1} performance metrics:")
    st.write(st.session_state['trad_ml_models'][name_1]['info']['metrics'])


# model (type) 2
name_2 = 'RF'
st.write(f"## Random Forest (RF)")
st.markdown("""**Random Forest** is a powerful ensemble model employed for classification tasks as well. Instead of relying on one decision tree, it combines multiple decision trees to make predictions. Each tree individually votes on the class for the input, and the final result is determined by the majority vote among all the trees.""")

with st.expander("View parameters", expanded=False):    
    st.write("### Feature generation parameters:")
    st.write(st.session_state['trad_ml_models'][name_2]['info']['fg_config'])
    st.write("### Model parameters:")
    st.write(st.session_state['trad_ml_models'][name_2]['info']['model_config'])
with st.expander("View performance metrics", expanded=False):
    st.write(f"{name_1} performance metrics:")
    st.write(st.session_state['trad_ml_models'][name_2]['info']['metrics'])

# model (type) 3
name_3 = 'LogReg'
st.write("## Logistic Regression (LogReg)")
st.markdown("""**Logistic Regression** is a versatile model used for classification tasks, which involve assigning input data to one of several categories via regression. 
            It works by calculating the probabilities of each class and then selects the most likely one.""")

with st.expander("View parameters", expanded=False):    
    st.write("### Feature generation parameters:")
    st.write(st.session_state['trad_ml_models'][name_3]['info']['fg_config'])
    st.write("### Model parameters:")
    st.write(st.session_state['trad_ml_models'][name_3]['info']['model_config'])
with st.expander("View performance metrics", expanded=False):
    st.write(f"{name_1} performance metrics:")
    st.write(st.session_state['trad_ml_models'][name_3]['info']['metrics'])


# LSTM
name_4 = 'Long Short-Term Memory (LSTM)'
st.write("## LSTM")
st.markdown("""**LSTM**s are usually effective for classification tasks where the input data has a temporal component. LSTMs have a unique ability to retain information from previous time steps and selectively forget or update that information as new data comes in. This memory capability allows LSTM to capture long-term dependencies and patterns in the data, making it well-suited for tasks requiring context and memory over time.""")
with st.expander("View parameters", expanded=False):
    st.write("### Feature generation parameters:")

    st.write("### Model parameters:")



### end of loading cycle - sidebar & other stuff ###
appf.keep_sidebar_extended()
appf.hide_image_fullscreen_option()