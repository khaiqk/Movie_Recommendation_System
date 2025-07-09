import streamlit as st

st.set_page_config(page_title="🔐 Login")

st.title("🔐 User Login")

user_id = st.text_input("Enter your user_id", placeholder="Example: 1")

if st.button("Login"):
    if user_id.strip().isdigit():
        st.session_state.user_id = int(user_id.strip())
        st.success(f"Login successful with user_id: {user_id}")
        st.switch_page("pages/home.py")
    else:
        st.error("❌ user_id must be an integer. Please try again.")
