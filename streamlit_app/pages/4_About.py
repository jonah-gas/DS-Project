import streamlit as st

### reset certain session state variables for other pages ###
st.session_state['trad_ml_skip_pred_button'] = True

st.set_page_config(
    page_title="About"
)

st.write("# Project info...")