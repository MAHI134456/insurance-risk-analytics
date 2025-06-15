# this script is to provide cleaning functions for the data.
import pandas as pd
import numpy as np
from datetime import datetime
from data_loader import load_data, save_data  # Import from your data_loader.py

def handle_missing_values(df: pd.DataFrame, strategy: dict = None, drop_threshold: float = 0.6) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame based on specified strategies.

    Parameters:
    - df: Input DataFrame containing insurance data.
    - strategy: Dictionary mapping column names to strategies ('mean', 'median', 'mode', 'drop', 'unknown').
    - drop_threshold: Proportion of missing values above which columns will be dropped (default: 0.6).

    Returns:
    - Cleaned DataFrame with handled missing values.
    """
    df = df.copy()
    if strategy is None:
        strategy = {}

    # Step 1: Drop columns with missingness above threshold
    high_missing_cols = df.columns[df.isna().mean() > drop_threshold]
    if not high_missing_cols.empty:
        print(f"Dropping columns with >{int(drop_threshold*100)}% missing values: {list(high_missing_cols)}")
        df.drop(columns=high_missing_cols, inplace=True)

    # Step 2: Handle remaining missing values
    for col in df.columns:
        missing_pct = df[col].isna().mean()
        if missing_pct == 0:
            continue
        if col in strategy:
            if strategy[col] == 'mean' and df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy[col] == 'median' and df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy[col] == 'mode':
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif strategy[col] == 'drop':
                df = df.dropna(subset=[col])
            elif strategy[col] == 'unknown':
                df[col].fillna('Unknown', inplace=True)
        else:
            if missing_pct < 0.05:
                if df[col].dtype in ['float64', 'int64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                if col in ['PolicyID', 'PostalCode']:
                    df = df.dropna(subset=[col])
                else:
                    df[col].fillna('Unknown' if df[col].dtype == 'object' else 0, inplace=True)

    return df

def correct_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Correct data types for insurance dataset columns to support analysis.

    Parameters:
    - df: Input DataFrame containing insurance data.

    Returns:
    - DataFrame with corrected data types.
    """
    df = df.copy()
    date_cols = ['TransactionMonth', 'VehicleIntroDate']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', format='%Y-%m')

    num_cols = ['TotalPremium', 'TotalClaims', 'CubicCapacity', 'Kilowatts', 'CustomValueEstimate']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    cat_cols = ['Gender', 'Province', 'VehicleType', 'CoverType', 'MaritalStatus']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    return df


def handle_outliers(df: pd.DataFrame, columns: list = None, method: str = 'iqr', cap: bool = True) -> pd.DataFrame:
    """
    Detect and handle outliers in numerical columns using IQR or Z-score.

    Parameters:
    - df: Input DataFrame containing insurance data.
    - columns: List of numerical columns to check for outliers. If None, defaults to key financial columns.
    - method: Outlier detection method ('iqr' or 'zscore').
    - cap: If True, cap outliers at thresholds; if False, remove rows with outliers.

    Returns:
    - DataFrame with handled outliers.
    """
    df = df.copy()
    if columns is None:
        columns = ['TotalPremium', 'TotalClaims', 'CustomValueEstimate']

    for col in columns:
        if col not in df.columns or df[col].dtype not in ['float64', 'int64']:
            continue

        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            lower_bound = df[col].mean() - 3 * df[col].std()
            upper_bound = df[col].mean() + 3 * df[col].std()

        if cap:
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        else:
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            df = df[~outliers]

    return df

def standardize_categorical(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Standardize categorical columns by fixing inconsistencies and formatting.

    Parameters:
    - df: Input DataFrame containing insurance data.
    - columns: List of categorical columns to standardize. If None, defaults to key categorical columns.

    Returns:
    - DataFrame with standardized categorical columns.
    """
    df = df.copy()
    if columns is None:
        columns = ['Gender', 'Province', 'VehicleType', 'CoverType']

    for col in columns:
        if col not in df.columns or df[col].dtype not in ['object', 'category']:
            continue
        df[col] = df[col].astype(str).str.strip().str.title()
        if col == 'Gender':
            df[col] = df[col].replace({'M': 'Male', 'F': 'Female', 'Unknown': 'Unknown'})
        elif col == 'Province':
            province_map = {
                'Gau': 'Gauteng', 'Wc': 'Western Cape', 'Kzn': 'Kwazulu-Natal',
                'Ec': 'Eastern Cape', 'Fs': 'Free State', 'Mp': 'Mpumalanga',
                'Lp': 'Limpopo', 'Nc': 'Northern Cape', 'Nw': 'North West'
            }
            df[col] = df[col].replace(province_map)

    return df

def clean_insurance_data(input_path: str, output_path: str) -> bool:
    """
    Load, clean, and save the insurance dataset using a pipeline.

    Parameters:
    - input_path: Path to the input pipe-separated CSV file.
    - output_path: Path to save the cleaned pipe-separated CSV file.

    Returns:
    - True if cleaning and saving were successful, False otherwise.
    """
    try:
        print("Starting data cleaning pipeline...")
        # Load data using data_loader.py
        df = load_data(input_path)
        if df is None:
            print("Failed to load data. Aborting cleaning pipeline.")
            return False

        # Apply cleaning steps
        df = handle_missing_values(df)
        df = correct_data_types(df)
        df = remove_duplicates(df)
        df = handle_outliers(df)
        df = standardize_categorical(df)

        # Save cleaned data
        success = save_data(df, output_path)
        if success:
            print(f"Cleaned data saved to {output_path}")
        else:
            print("Failed to save cleaned data.")
        return success

    except Exception as e:
        print(f"Error in cleaning pipeline: {e}")
        return False

if __name__ == "__main__":
    input_file = "data/raw/MachineLearningRating_v3.txt"
    output_file = "data/processed/MachineLearningRating_v3_cleaned.txt"
    success = clean_insurance_data(input_file, output_file)
    if success:
        print("Data cleaning pipeline completed successfully.")
    else:
        print("Data cleaning pipeline failed.")
