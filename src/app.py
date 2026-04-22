"""
app.py
Housing Affordability DSS — CS7P01 MSc Project
London Metropolitan University

Command-line runner for the Housing Affordability DSS.
Loads all processed outputs and prints the DSS report.

Usage:
    python src/app.py
    python src/app.py --region London
    python src/app.py --year 2024
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))

from src.data_pipeline import load_features, load_forecasts
from src.decision_rules import classify_all_regions, \
    get_risk_class, get_forecast_trajectory, get_policy_action


def print_dss_report(year: int = 2025) -> None:
    """Print full DSS regional risk assessment report."""

    features  = load_features()
    forecasts = load_forecasts()
    report    = classify_all_regions(features, forecasts, year)

    print("=" * 80)
    print("  HOUSING AFFORDABILITY DECISION SUPPORT SYSTEM")
    print("  CS7P01 MSc Project | London Metropolitan University")
    print(f"  Assessment Year: {year} | Forecast Horizon: 2030")
    print("=" * 80)

    print(f"\n{'Rank':<5} {'Region':<28} {'Ratio':>7} "
          f"{'CAI':>6} {'Risk':>10} {'2030fc':>8} "
          f"{'Trend':>11} {'Policy Action'}")
    print("-" * 90)

    for _, row in report.iterrows():
        print(
            f"{int(row['rank']):<5} "
            f"{row['region']:<28} "
            f"{row['ratio_2025']:>7.2f} "
            f"{row['cai']:>6.1f} "
            f"{row['risk_class']:>10} "
            f"{row['forecast_2030']:>8.2f} "
            f"{row['trajectory']:>11}  "
            f"{row['policy_action']}"
        )

    print("-" * 90)
    print(f"\n── POLICY ALERTS:")
    for _, row in report.iterrows():
        prefix = "🔴" if "ALERT" in row['alert'] else "🟡"
        print(f"   {prefix} {row['alert']}")

    # Summary
    risk_counts = report['risk_class'].value_counts()
    print(f"\n── RISK SUMMARY:")
    for tier in ['Critical', 'High', 'Medium', 'Low']:
        n = risk_counts.get(tier, 0)
        if n > 0:
            regions = report[
                report['risk_class'] == tier
            ]['region'].tolist()
            print(f"   {tier:<10}: {n} region(s) — {regions}")


def print_region_report(region: str, year: int = 2025) -> None:
    """Print DSS assessment for a single region."""

    features  = load_features()
    forecasts = load_forecasts()

    row = features[
        (features['region'] == region) &
        (features['year']   == year)
    ]
    fc_row = forecasts[
        (forecasts['region'] == region) &
        (forecasts['year']   == 2030)
    ]

    if len(row) == 0:
        print(f"Error: No data for region '{region}' in {year}")
        print(f"Available regions: "
              f"{sorted(features['region'].unique())}")
        return

    row    = row.iloc[0]
    fc_row = fc_row.iloc[0]

    ratio      = float(row['affordability_ratio'])
    cai        = float(row['cai'])
    risk_class = str(row['risk_class'])
    fc_2030    = float(fc_row['forecast'])
    trajectory = get_forecast_trajectory(ratio, fc_2030)
    policy     = get_policy_action(risk_class, trajectory)

    print(f"\n{'='*55}")
    print(f"  DSS ASSESSMENT REPORT — {region.upper()}")
    print(f"{'='*55}")
    print(f"\n  CURRENT STATUS ({year})")
    print(f"  Affordability Ratio  : {ratio:.2f}×")
    print(f"  Composite Index (CAI): {cai:.1f} / 100")
    print(f"  Risk Classification  : {risk_class}")
    print(f"  Regional Rank        : "
          f"{int(row['risk_rank'])} / 10")
    print(f"\n  FORECAST (2030)")
    print(f"  Projected Ratio      : {fc_2030:.2f}×")
    print(f"  95% CI               : "
          f"{float(fc_row['ci_lower']):.2f}× "
          f"– {float(fc_row['ci_upper']):.2f}×")
    print(f"  Trajectory           : {trajectory}")
    print(f"\n  RECOMMENDATION")
    print(f"  Policy Action        : {policy}")
    print(f"{'='*55}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Housing Affordability DSS'
    )
    parser.add_argument(
        '--region', type=str, default=None,
        help='Specific region to query'
    )
    parser.add_argument(
        '--year', type=int, default=2025,
        help='Assessment year (default: 2025)'
    )
    args = parser.parse_args()

    if args.region:
        print_region_report(args.region, args.year)
    else:
        print_dss_report(args.year)