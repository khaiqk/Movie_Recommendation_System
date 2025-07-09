import streamlit as st
import random
import pandas as pd

import os
import sys
from dotenv import load_dotenv

load_dotenv()

project_root = os.getenv("ROOT")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
    
    
if "user_id" not in st.session_state:
    st.warning("⚠️ You need to log in first.")
    st.switch_page("pages/Login.py")

with st.sidebar:
    st.markdown(f"👤 User ID: `{st.session_state.user_id}`")
    if st.button("🚪 Log out"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("✅ Logged out.")
        st.switch_page("pages/Login.py")
        
        

if "env" not in st.session_state:
    from DQN.model import DQL_model
    from DQN.env import Env
    from processed_data.movies_data import data_movies
    from processed_data.users_data import data_users

    model = DQL_model()
    env = Env(user_id=st.session_state.user_id, data_movies=data_movies, data_users=data_users, model=model)
    st.session_state.env = env

env = st.session_state.env



if "data_movies" not in st.session_state or st.session_state.data_movies is None:
    try:
        suggestions = env.suggest_next_movie()
        if isinstance(suggestions, pd.DataFrame) and not suggestions.empty:
            st.session_state.data_movies = suggestions.reset_index(drop=True)
        else:
            raise ValueError("suggest_next_movie() returns None or empty.")
    except Exception as e:
        st.error("❌ Unable to load movie list.")
        st.code(str(e))
        st.stop()



if st.button("🔄 Movie Refresh"):
    try:
        suggestions = env.suggest_next_movie()
        if isinstance(suggestions, pd.DataFrame) and not suggestions.empty:
            st.session_state.data_movies = suggestions.reset_index(drop=True)
            st.success("✅ Refreshed movie list.")
        else:
            st.warning("⚠️ There are no new movies to show.")
    except Exception as e:
        st.error("❌ Error while refreshing.")
        st.code(str(e))
        
        

movies = st.session_state.get("data_movies", pd.DataFrame())

st.title("🎬 Recommended movies list")

if not movies.empty:
    cols_per_row = 3
    for i in range(0, len(movies), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            if i + j < len(movies):
                movie = movies.iloc[i + j]
                with cols[j]:
                    with st.container(border=True):
                        st.markdown(
                            f"""
                            <div style='text-align:center; padding: 30px 0; font-weight: bold'>
                                {movie['title']}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        if st.button("See details", key=f"btn_{movie['movieId']}"):
                            st.session_state.selected_movie_id = movie["movieId"]
                            st.session_state.selected_index = i + j
                            st.switch_page("pages/rating_movie.py")
else:
    st.warning("❌ There are no movies to display.")