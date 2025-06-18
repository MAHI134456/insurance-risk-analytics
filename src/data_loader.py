import pandas as pd
import os

RAW_DATA_PATH = '../data/raw/MachineLearningRating_v3.txt'
PROCESSED_DATA_PATH = '../data/processed/MachineLearningRating_v3_processed.txt'

def load_data(file_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, sep='|', encoding='utf-8')
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()

def save_data(df: pd.DataFrame, file_path: str = PROCESSED_DATA_PATH) -> bool:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, sep='|', index=False, encoding='utf-8')
        print(f"Data saved successfully to {file_path}.")
        return True
    except Exception as e:
        print(f"Error saving data to {file_path}: {e}")
        return False

if __name__ == "__main__":
    df = load_data()
    if not df.empty:
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
        save_data(df)
    else:
        print("No data loaded.")
