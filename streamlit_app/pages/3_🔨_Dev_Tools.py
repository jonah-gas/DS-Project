import streamlit as st

### reset certain session state variables for other pages ###
st.session_state['trad_ml_skip_pred_button'] = True

### Entry page ###
st.set_page_config(
    page_title="Dev Tools"
)

st.write("# Header - Dev Tools")

st.markdown(
    """
    - DB connection status?
    - Scraping, cleaning and inserting new data
    - Maybe require authentication?
    """
)