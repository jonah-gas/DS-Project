import streamlit as st

### reset certain session state variables for other pages ###
st.session_state['trad_ml_skip_pred_button'] = True


### Entry page ###
st.set_page_config(
    page_title="Statistics TEST",
    page_icon="📊"
)

st.write("# Header - Statistics")

st.markdown(
    """
    Stats based on our collected data.
    """
)
