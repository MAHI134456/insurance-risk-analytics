import pandas as pd
import numpy as np
from data_loader import load_data, save_data
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'MachineLearningRating_v3.txt')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'MachineLearningRating_v3_processed.txt')

def handle_missing_values(df: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
    # Drop columns with > threshold missing values
    missing_pct = df.isna().mean()
    to_drop = missing_pct[missing_pct > threshold].index.tolist()
    if to_drop:
        print(f"Dropping columns >{int(threshold*100)}% missing: {to_drop}")
        df.drop(columns=to_drop, inplace=True)

    for col in df.columns:
        pct = df[col].isna().mean()
        if pct == 0:
            continue

        if pct < 0.05:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode().iat[0], inplace=True)
        else:
            if col in ['PolicyID', 'PostalCode']:
                df.dropna(subset=[col], inplace=True)
            else:
                df[col].fillna('Unknown' if df[col].dtype == 'object' else 0, inplace=True)

    return df

def correct_data_types(df: pd.DataFrame) -> pd.DataFrame:
    # Dates
    for col in ['TransactionMonth', 'VehicleIntroDate']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', format='%Y-%m')
            # If column was dropped earlier, skip
    # Numerics
    for col in df.select_dtypes(include=[object]).columns:
        df[col] = df[col].str.replace(',', '').astype(str)
    for col in df.columns:
        if col not in ['TransactionMonth', 'VehicleIntroDate'] and pd.api.types.is_numeric_dtype(df[col]):
            continue
        try:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        except:
            pass

    # Convert obvious categories
    for col in ['Gender', 'Province', 'VehicleType', 'CoverType', 'MaritalStatus']:
        if col in df.columns:
            df[col] = df[col].astype('category')

    return df

def handle_outliers(df: pd.DataFrame, method: str = 'iqr', cap=True) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        if method == 'iqr':
            iqr = q3 - q1
            lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
        else:
            mean, std = df[col].mean(), df[col].std()
            lower, upper = mean - 3*std, mean + 3*std

        if cap:
            df[col] = df[col].clip(lower=lower, upper=upper)
        else:
            df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df

def standardize_categorical(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip().str.title()
        if col == 'Gender':
            df[col] = df[col].replace({'M': 'Male', 'F': 'Female'})
        elif col == 'Province':
            df[col] = df[col].replace({
                'Gau': 'Gauteng', 'Wc': 'Western Cape', 'Kzn': 'Kwazulu-Natal',
                'Ec': 'Eastern Cape', 'Fs': 'Free State', 'Mp': 'Mpumalanga',
                'Lp': 'Limpopo', 'Nc': 'Northern Cape', 'Nw': 'North West'
            })
    return df

def clean_insurance_data() -> bool:
    print(" Starting cleaning...")
    df = load_data(RAW_DATA_PATH)
    if df.empty:
        print("❌ No data loaded, aborting.")
        return False

    df = handle_missing_values(df, threshold=0.6)
    df = correct_data_types(df)
    df = handle_outliers(df)
    df = standardize_categorical(df)

    if save_data(df, PROCESSED_DATA_PATH):
        print("✅ Cleaned data saved!")
        return True
    else:
        print("❌ Failed to save cleaned data.")
        return False

if __name__ == "__main__":
    clean_insurance_data()
