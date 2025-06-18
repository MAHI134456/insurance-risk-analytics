import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_loader import load_data  # Ensure this imports from your src folder

# Define base directory and processed data path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'MachineLearningRating_v3_processed3.txt')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'eda_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")



def calculate_loss_ratio(df):
    overall = df['TotalClaims'].sum() / df['TotalPremium'].sum()
    by_province = df.groupby('Province').apply(lambda x: x['TotalClaims'].sum() / x['TotalPremium'].sum()).reset_index(name='LossRatio')
    by_vehicle = df.groupby('VehicleType').apply(lambda x: x['TotalClaims'].sum() / x['TotalPremium'].sum()).reset_index(name='LossRatio')
    by_gender = df.groupby('Gender').apply(lambda x: x['TotalClaims'].sum() / x['TotalPremium'].sum()).reset_index(name='LossRatio')
    return overall, by_province, by_vehicle, by_gender


def plot_loss_ratios(by_province, by_vehicle, by_gender):
    fig, axes = plt.subplots(3, 1, figsize=(12, 16))

    sns.barplot(data=by_province, x='LossRatio', y='Province', ax=axes[0])
    axes[0].set_title('Loss Ratio by Province')

    sns.barplot(data=by_vehicle, x='LossRatio', y='VehicleType', ax=axes[1])
    axes[1].set_title('Loss Ratio by Vehicle Type')

    sns.barplot(data=by_gender, x='LossRatio', y='Gender', ax=axes[2])
    axes[2].set_title('Loss Ratio by Gender')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_ratios.png'))
    plt.close()


def distribution_plots(df):
    for col in ['TotalPremium', 'TotalClaims']:
        if col not in df.columns:
            continue
        plt.figure(figsize=(10, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.savefig(os.path.join(OUTPUT_DIR, f'distribution_{col}.png'))
        plt.close()

        plt.figure(figsize=(10, 4))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.savefig(os.path.join(OUTPUT_DIR, f'boxplot_{col}.png'))
        plt.close()


def correlation_matrix(df):
    num = df.select_dtypes(include='number')
    corr = num.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'))
    plt.close()


def monthly_trends(df):
    if 'TransactionMonth' in df.columns:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
        df = df.dropna(subset=['TransactionMonth'])

        trend = df.groupby(pd.Grouper(key='TransactionMonth', freq='M')).agg({
            'TotalClaims': ['sum', 'mean'], 'TotalPremium': ['sum', 'mean']
        })
        trend.columns = ['_'.join(col) for col in trend.columns]
        trend.reset_index(inplace=True)

        plt.figure(figsize=(12, 5))
        sns.lineplot(x='TransactionMonth', y='TotalClaims_sum', data=trend, label='Total Claims')
        sns.lineplot(x='TransactionMonth', y='TotalPremium_sum', data=trend, label='Total Premium')
        plt.title("Monthly Claim and Premium Trends")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'monthly_trends.png'))
        plt.close()


def top_vehicle_makes(df):
    if 'make' in df.columns:
        makes = df.groupby('make')['TotalClaims'].sum().sort_values(ascending=False).head(10)
        plt.figure(figsize=(10, 5))
        sns.barplot(x=makes.values, y=makes.index)
        plt.title('Top 10 Vehicle Makes by Total Claims')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'top_vehicle_makes.png'))
        plt.close()


def run_eda():
    print("Starting EDA...")
    df = load_data(PROCESSED_DATA_PATH)
    if df is None or df.empty:
        print(f"Error loading data from {PROCESSED_DATA_PATH}")
        return

    data_summary(df)
    data_quality_plot(df)

    print("Calculating loss ratios...")
    overall, by_province, by_vehicle, by_gender = calculate_loss_ratio(df)
    print(f"Overall Loss Ratio: {overall:.4f}")
    plot_loss_ratios(by_province, by_vehicle, by_gender)

    print("Creating distribution and box plots...")
    distribution_plots(df)

    print("Generating correlation matrix...")
    correlation_matrix(df)

    print("Plotting monthly trends...")
    monthly_trends(df)

    print("Analyzing vehicle claims...")
    top_vehicle_makes(df)

    print("EDA completed and all outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    run_eda()
