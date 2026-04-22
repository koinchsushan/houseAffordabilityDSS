"""
models.py
Housing Affordability DSS — CS7P01 MSc Project
London Metropolitan University

Model wrappers for OLS Fixed Effects and Random Forest.
Refactored from notebooks/03_models.ipynb for modular use.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error

# ── Final feature specification ───────────────────────────────
# Determined through VIF analysis in notebook 03
FEATURES_FINAL = [
    'real_house_price',
    'ratio_lag1',
    'yoy_price_change_pct'
]

TARGET = 'affordability_ratio'
PROCESSED_PATH = 'data/processed/'


def prepare_model_data(
    features_df: pd.DataFrame,
    start_year: int = 2003
) -> tuple:
    """
    Prepare feature matrix and target vector for modelling.

    Args:
        features_df : features_engineered.csv as DataFrame
        start_year  : first year with complete lag features

    Returns:
        Tuple of (model_df, X_raw, X_scaled, y, scaler,
                  region_dummies, X_fe)
    """
    model_df = (features_df[features_df['year'] >= start_year]
                .dropna(subset=FEATURES_FINAL + [TARGET])
                .copy()
                .reset_index(drop=True))

    X_raw  = model_df[FEATURES_FINAL]
    y      = model_df[TARGET]

    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_raw),
        columns=FEATURES_FINAL,
        index=X_raw.index
    )

    region_dummies = pd.get_dummies(
        model_df['region'], drop_first=True, dtype=float
    )
    X_fe = pd.concat([X_scaled, region_dummies], axis=1)

    return model_df, X_raw, X_scaled, y, scaler, \
           region_dummies, X_fe


def load_saved_models(processed_path: str = PROCESSED_PATH):
    """
    Load pre-fitted models from disk.

    Returns:
        Tuple of (ols_model, rf_model, scaler)
    """
    ols = joblib.load(
        os.path.join(processed_path, 'ols_fe_model.pkl')
    )
    rf  = joblib.load(
        os.path.join(processed_path, 'rf_model.pkl')
    )
    sc  = joblib.load(
        os.path.join(processed_path, 'feature_scaler.pkl')
    )
    return ols, rf, sc


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5
) -> dict:
    """
    Evaluate model performance with in-sample and CV metrics.

    Args:
        model    : fitted sklearn model
        X        : feature matrix
        y        : target vector
        cv_folds : number of cross-validation folds

    Returns:
        Dict with in_r2, in_rmse, cv_r2, cv_rmse
    """
    kf      = KFold(n_splits=cv_folds,
                    shuffle=True, random_state=42)
    y_pred  = model.predict(X)

    cv_r2   = cross_val_score(
        model, X, y, cv=kf, scoring='r2'
    )
    cv_rmse = cross_val_score(
        model, X, y, cv=kf,
        scoring='neg_root_mean_squared_error'
    )

    return {
        'in_r2':   round(r2_score(y, y_pred), 4),
        'in_rmse': round(
            np.sqrt(mean_squared_error(y, y_pred)), 4),
        'cv_r2':   round(cv_r2.mean(), 4),
        'cv_rmse': round(-cv_rmse.mean(), 4)
    }


if __name__ == '__main__':
    print("── models.py — module test")
    print(f"   Features : {FEATURES_FINAL}")
    print(f"   Target   : {TARGET}")
    try:
        ols, rf, sc = load_saved_models()
        print(f"✓ Saved models loaded successfully")
    except FileNotFoundError:
        print("   Run notebook 03 first to generate model files")