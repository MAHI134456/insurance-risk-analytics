import os
import pandas as pd
import numpy as np
from datetime import datetime

# Paths to raw and processed files
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'MachineLearningRating_v3.txt')
CLEANED_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'MachineLearningRating_v3_cleaned3.txt')

def load_data(path=RAW_PATH):
    try:
        df = pd.read_csv(path, sep='|', encoding='utf-8', low_memory=False)
        return df
    except Exception as e:
        print(f"[ERROR] loading data: {e}")
        return pd.DataFrame()

def drop_high_missing(df, pct=0.6):
    mn = df.isna().mean()
    drop_cols = mn[mn > pct].index.tolist()
    if drop_cols:
        print(f"Dropping {len(drop_cols)} cols with >{pct*100:.0f}% missing: {drop_cols}")
        df = df.drop(columns=drop_cols)
    return df

def handle_missing(df):
    for col in df.columns:
        if df[col].isna().mean() == 0:
            continue
        if df[col].dtype.kind in 'biufc':  # numeric
            med = df[col].median()
            df[col] = df[col].fillna(med)
        else:
            mode = df[col].mode(dropna=True)
            fill = mode[0] if not mode.empty else 'Unknown'
            df[col] = df[col].fillna(fill)
    return df

def fix_types(df):
    df = df.copy()
    # date fields
    for c in ['TransactionMonth', 'VehicleIntroDate']:
        if c in df: df[c] = pd.to_datetime(df[c], errors='coerce')
    # numeric currency/amount fields
    for c in ['TotalPremium', 'TotalClaims', 'CustomValueEstimate', 'CubicCapacity', 'Kilowatts']:
        if c in df: df[c] = pd.to_numeric(df[c], errors='coerce')
    # categorical
    for c in ['Gender', 'Province', 'VehicleType', 'CoverType', 'MaritalStatus']:
        if c in df: df[c] = df[c].astype('category')
    return df

def outlier_treatment(df, columns=None, method='iqr', cap=True):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in columns:
        if col not in df or not np.issubdtype(df[col].dtype, np.number):
            continue
        Q1, Q3 = df[col].quantile([0.25,0.75])
        IQR = Q3 - Q1
        lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        if cap:
            df[col] = df[col].clip(lo, hi)
        else:
            df = df[ df[col].between(lo, hi) ]
    return df

def save_data(df, path=CLEANED_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep='|', index=False, encoding='utf-8')
    print(f"✅ Cleaned data saved to: {path}")

def main():
    print("🔁 Starting cleaning pipeline...")
    df = load_data()
    if df.empty:
        print("[ERROR] No data loaded ➝ aborting.")
        return

    df = drop_high_missing(df)
    df = handle_missing(df)
    df = fix_types(df)
    df = outlier_treatment(df, cap=True)
    save_data(df)

    print("✅ Cleaning completed successfully.")
    print(df.describe(include='all').T[['count']].head(10))
    print(df.info())

if __name__ == "__main__":
    main()
