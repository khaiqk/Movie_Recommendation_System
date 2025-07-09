import streamlit as st

st.set_page_config(page_title="Movie Recommendation System")

if "user_id" not in st.session_state:
    st.switch_page("pages/login.py")
else:
    st.switch_page("pages/home.py")