"""
data_pipeline.py
Housing Affordability DSS — CS7P01 MSc Project
London Metropolitan University

Data loading and cleaning functions.
Refactored from notebooks/01_eda.ipynb for modular use.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ── Region constants ──────────────────────────────────────────
REGIONS = [
    'East Midlands',
    'East of England',
    'London',
    'North East',
    'North West',
    'South East',
    'South West',
    'West Midlands Region',
    'Yorkshire and The Humber',
    'Wales'
]

PROCESSED_PATH = 'data/processed/'
RAW_PATH       = 'data/raw/'


def load_master(processed_path: str = PROCESSED_PATH) -> pd.DataFrame:
    """
    Load the master affordability dataset.

    Returns:
        DataFrame with 290 rows × 18 cols (10 regions, 1997–2025)
    """
    path = os.path.join(processed_path, 'master_affordability.csv')
    df   = pd.read_csv(path)
    df['year'] = df['year'].astype(int)
    return df


def load_features(processed_path: str = PROCESSED_PATH) -> pd.DataFrame:
    """
    Load the feature-engineered dataset including CAI and
    risk classifications.

    Returns:
        DataFrame with 260 rows × 32 cols (2000–2025)
    """
    path = os.path.join(processed_path, 'features_engineered.csv')
    df   = pd.read_csv(path)
    df['year']       = df['year'].astype(int)
    df['risk_class'] = df['risk_class'].astype(str)
    return df


def load_forecasts(processed_path: str = PROCESSED_PATH) -> pd.DataFrame:
    """
    Load ARIMA forecast outputs (2026–2030).

    Returns:
        DataFrame with 50 rows (10 regions × 5 forecast years)
    """
    path = os.path.join(processed_path, 'arima_forecasts.csv')
    df   = pd.read_csv(path)
    df['year'] = df['year'].astype(int)
    return df


def load_hpi(raw_path: str = RAW_PATH) -> pd.DataFrame:
    """
    Load and clean the Land Registry HPI Full File.
    Filters to 10 modelling regions, 1995–2025.

    Returns:
        DataFrame with 3720 rows (10 regions × 372 months)
    """
    path = os.path.join(raw_path, 'land_registry_hpi_full_2025.csv')
    hpi  = pd.read_csv(path, encoding='utf-8')

    hpi['Date'] = pd.to_datetime(hpi['Date'], dayfirst=True)
    hpi['Year'] = hpi['Date'].dt.year
    hpi['Month']= hpi['Date'].dt.month

    hpi = hpi[hpi['RegionName'].isin(REGIONS)].copy()
    hpi = hpi[hpi['Year'] >= 1995].copy()

    hpi = hpi[[
        'Date', 'Year', 'Month', 'RegionName',
        'AveragePrice', 'AveragePriceSA',
        'SalesVolume', '12m%Change', 'IndexSA'
    ]].copy()

    hpi.columns = [
        'date', 'year', 'month', 'region',
        'avg_price', 'avg_price_sa',
        'sales_volume', 'yoy_pct_change', 'index_sa'
    ]

    for col in ['avg_price', 'avg_price_sa', 'sales_volume',
                'yoy_pct_change', 'index_sa']:
        hpi[col] = pd.to_numeric(hpi[col], errors='coerce')

    return hpi.sort_values(
        ['region', 'date']
    ).reset_index(drop=True)


def get_region_series(
    df: pd.DataFrame,
    region: str,
    column: str = 'affordability_ratio'
) -> pd.Series:
    """
    Extract a time series for a single region.

    Args:
        df     : master or features dataframe
        region : region name string
        column : column to extract

    Returns:
        pandas Series indexed by year
    """
    return (df[df['region'] == region]
            .sort_values('year')
            .set_index('year')[column])


if __name__ == '__main__':
    print("── data_pipeline.py — module test")
    master = load_master()
    print(f"   Master loaded    : {master.shape}")
    features = load_features()
    print(f"   Features loaded  : {features.shape}")
    forecasts = load_forecasts()
    print(f"   Forecasts loaded : {forecasts.shape}")
    print("✓ All data loads successful")