import pandas as pd

data_ratings = None

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df = df.head(100) 
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(columns=['timestamp'], inplace=True)
    return df


file_path = "data/ratings.csv"
data = load_data(file_path=file_path)
data_ratings = process_data(df=data)
