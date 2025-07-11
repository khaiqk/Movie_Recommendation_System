import pandas as pd

data_movies = None

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    df['genres_split'] = df['genres'].str.split('|')
    genres_dumies = df['genres_split'].explode().str.get_dummies().groupby(level=0).sum()
    genres_dumies = genres_dumies.astype(int)
    df = df.join(genres_dumies)
    df.drop(columns=['genres', 'genres_split', '(no genres listed)'], inplace=True)
    return df


file_path = "data/movies.csv"
data = load_data(file_path=file_path)
data_movies = process_data(df=data)
