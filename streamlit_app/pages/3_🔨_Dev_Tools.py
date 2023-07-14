import streamlit as st


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