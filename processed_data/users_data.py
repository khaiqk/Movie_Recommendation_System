import pandas as pd

import os
import sys
from dotenv import load_dotenv

load_dotenv()

project_root = os.getenv("ROOT")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from processed_data.movies_data import data_movies
from processed_data.ratings_data import data_ratings 

if isinstance(data_movies, pd.DataFrame) and isinstance(data_ratings, pd.DataFrame):
    data_movies_no_title =  data_movies.copy()
    data_movies_no_title.drop(columns=['title'], inplace=True)
    data_users = pd.merge(data_movies_no_title, data_ratings, on='movieId', how='inner')