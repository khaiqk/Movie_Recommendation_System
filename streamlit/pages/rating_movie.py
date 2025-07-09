import streamlit as st
import pandas as pd
import numpy as np

import os
import sys
from dotenv import load_dotenv

load_dotenv()

project_root = os.getenv("ROOT")

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from processed_data.users_data import data_users
from processed_data.ratings_data import data_ratings

user_id = st.session_state.get("user_id", None)
if user_id is None:
    st.warning("⚠️ You need to log in first.")
    st.stop()
    
with st.sidebar:
    st.markdown(f"👤 User ID: `{st.session_state.user_id}`")
    
env = st.session_state.get("env", None)
if env is None:
    st.error("❌ Environment (env) not found. Please return to main page.")
    st.stop()

if "user_ratings_df" not in st.session_state:
    st.session_state.user_ratings_df = data_ratings.copy()
user_ratings_df = st.session_state.user_ratings_df

selected_index = st.session_state.get("selected_index", None)
movie_id = st.session_state.get("selected_movie_id", None)
data_movies = st.session_state.get("data_movies", None)

if movie_id is not None and data_movies is not None:
    if movie_id in data_movies["movieId"].values:
        movie = data_movies[data_movies["movieId"] == movie_id].iloc[0]

        if st.button("🏠 Back to list"):
            st.switch_page("pages/home.py")

        st.title(movie["title"])

        genres = movie.iloc[2:-1]  
        selected_genres = genres[genres == 1].index.tolist()
        st.markdown(f"**Genres:** {' | '.join(selected_genres)}")

        existing_rating_row = user_ratings_df[
            (user_ratings_df["userId"] == user_id) & (user_ratings_df["movieId"] == movie_id)
        ]

        if not existing_rating_row.empty:
            rated_score = existing_rating_row.iloc[0]["rating"]
            st.success(f"⭐ You (user {user_id}) rated this movie: {rated_score}/10.")
        else:
            
            rating_options = ["-- Select rating --"] + list(np.arange(1, 5.5, 0.5))
            selected_rating = st.selectbox("Select the score you want to rate:", rating_options, index=0)
            
            if not selected_rating == "-- Select rating --":
                if st.button("💾 Save"):

                    env.step(index=selected_index)
                    env.rating_movie(rating=selected_rating)
                    
                    new_row = pd.DataFrame([{"userId": user_id, "movieId": movie_id, "rating": selected_rating}])
                    st.session_state.user_ratings_df = pd.concat([user_ratings_df, new_row], ignore_index=True)
                    st.success(f"✅ Saved: {selected_rating}/10")
                    st.rerun()
    else:
        st.warning("Movie not found.")
else:
    st.warning("No movies selected.")