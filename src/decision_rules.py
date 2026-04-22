"""
decision_rules.py
Housing Affordability DSS — CS7P01 MSc Project
London Metropolitan University

Decision logic layer — risk classification, trajectory
assessment, and policy recommendation functions.
Refactored from notebooks/05_decision_support.ipynb.

Academic justification:
    Timmermans (1997) — DSS supports not replaces decisions
    Power (2002) — model-driven DSS framework
    Busuioc (2021) — algorithmic accountability in policy
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

# ── Risk tier thresholds ──────────────────────────────────────
RISK_THRESHOLDS = {
    'Low':      (0,  25),
    'Medium':   (25, 50),
    'High':     (50, 75),
    'Critical': (75, 100)
}

RISK_ORDER = ['Low', 'Medium', 'High', 'Critical']

# ── Policy action lookup table ────────────────────────────────
POLICY_ACTIONS = {
    ('Low',      'Improving'):  'No action required',
    ('Low',      'Stable'):     'No action required',
    ('Low',      'Worsening'):  'Monitor — early warning',
    ('Medium',   'Improving'):  'Monitor — improving trend',
    ('Medium',   'Stable'):     'Monitor — no change expected',
    ('Medium',   'Worsening'):  'Intervene — deteriorating',
    ('High',     'Improving'):  'Intervene — recovery underway',
    ('High',     'Stable'):     'Intervene — sustained stress',
    ('High',     'Worsening'):  'Urgent intervention required',
    ('Critical', 'Improving'):  'Intervene — recovery underway',
    ('Critical', 'Stable'):     'Urgent intervention required',
    ('Critical', 'Worsening'):  'Urgent intervention required',
}


def get_risk_class(cai: float) -> str:
    """
    Classify CAI score into risk tier.

    Args:
        cai : Composite Affordability Index score (0–100)

    Returns:
        Risk tier string: 'Low', 'Medium', 'High', 'Critical'
    """
    if cai < 25:
        return 'Low'
    elif cai < 50:
        return 'Medium'
    elif cai < 75:
        return 'High'
    else:
        return 'Critical'


def get_forecast_trajectory(
    current_ratio: float,
    forecast_ratio: float,
    threshold: float = 0.20
) -> str:
    """
    Classify forecast trajectory from current to forecast ratio.

    Args:
        current_ratio  : current affordability ratio
        forecast_ratio : projected affordability ratio
        threshold      : minimum change to classify as
                         improving/worsening (default 0.20)

    Returns:
        'Improving', 'Stable', or 'Worsening'
    """
    change = forecast_ratio - current_ratio
    if change <= -threshold:
        return 'Improving'
    elif change >= threshold:
        return 'Worsening'
    else:
        return 'Stable'


def get_policy_action(
    risk_class: str,
    trajectory: str
) -> str:
    """
    Return policy recommendation from risk class and trajectory.

    Args:
        risk_class : current risk tier
        trajectory : forecast trajectory

    Returns:
        Policy action string
    """
    key = (risk_class, trajectory)
    return POLICY_ACTIONS.get(key, 'Review required')


def get_risk_alert(
    region: str,
    current_risk: str,
    forecast_risk: str,
    trajectory: str
) -> str:
    """
    Generate a policy alert based on risk tier changes.

    Args:
        region        : region name
        current_risk  : current risk tier
        forecast_risk : projected risk tier
        trajectory    : forecast trajectory

    Returns:
        Alert string with ALERT or INFO prefix
    """
    curr_idx = RISK_ORDER.index(current_risk)
    fc_idx   = RISK_ORDER.index(forecast_risk)

    if current_risk == 'Critical':
        return f"ALERT: {region} remains Critical risk by 2030"
    elif fc_idx > curr_idx:
        return (f"ALERT: {region} projected to deteriorate "
                f"from {current_risk} to {forecast_risk} by 2030")
    elif fc_idx < curr_idx:
        return (f"INFO: {region} projected to improve "
                f"from {current_risk} to {forecast_risk} by 2030")
    else:
        return f"INFO: {region} risk tier stable at {current_risk}"


def classify_all_regions(
    features_df: pd.DataFrame,
    forecasts_df: pd.DataFrame,
    year: int = 2025
) -> pd.DataFrame:
    """
    Run full DSS assessment for all regions.

    Args:
        features_df  : features_engineered.csv loaded as DataFrame
        forecasts_df : arima_forecasts.csv loaded as DataFrame
        year         : assessment year (default 2025)

    Returns:
        DataFrame with full DSS assessment for all regions,
        sorted by risk rank (1 = least affordable)
    """
    records = []

    for region in sorted(features_df['region'].unique()):
        row = features_df[
            (features_df['region'] == region) &
            (features_df['year']   == year)
        ]
        if len(row) == 0:
            continue

        row    = row.iloc[0]
        fc_row = forecasts_df[
            (forecasts_df['region'] == region) &
            (forecasts_df['year']   == 2030)
        ]
        if len(fc_row) == 0:
            continue

        ratio      = float(row['affordability_ratio'])
        cai        = float(row['cai'])
        risk_class = str(row['risk_class'])
        risk_rank  = int(row['risk_rank'])
        fc_2030    = float(fc_row['forecast'].values[0])
        ci_lower   = float(fc_row['ci_lower'].values[0])
        ci_upper   = float(fc_row['ci_upper'].values[0])

        trajectory   = get_forecast_trajectory(ratio, fc_2030)
        forecast_risk= get_risk_class(
            cai * (fc_2030 / ratio) if ratio > 0 else cai
        )
        policy       = get_policy_action(risk_class, trajectory)
        alert        = get_risk_alert(
            region, risk_class, forecast_risk, trajectory
        )

        records.append({
            'rank':          risk_rank,
            'region':        region,
            'ratio_2025':    round(ratio, 2),
            'cai':           round(cai, 2),
            'risk_class':    risk_class,
            'forecast_2030': round(fc_2030, 2),
            'ci_lower':      round(ci_lower, 2),
            'ci_upper':      round(ci_upper, 2),
            'trajectory':    trajectory,
            'forecast_risk': forecast_risk,
            'policy_action': policy,
            'alert':         alert
        })

    return (pd.DataFrame(records)
            .sort_values('rank')
            .reset_index(drop=True))


if __name__ == '__main__':
    print("── decision_rules.py — module test")
    print(f"   get_risk_class(78.1)       : {get_risk_class(78.1)}")
    print(f"   get_risk_class(17.4)       : {get_risk_class(17.4)}")
    print(f"   get_forecast_trajectory"
          f"(10.61, 10.06) : "
          f"{get_forecast_trajectory(10.61, 10.06)}")
    print(f"   get_policy_action"
          f"(Critical, Improving) : "
          f"{get_policy_action('Critical', 'Improving')}")
    print("✓ All decision rules functions operational")