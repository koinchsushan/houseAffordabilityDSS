# Housing Affordability Decision Support System
### CS7P01 MSc Project | London Metropolitan University | MSc Data Analytics 2025/26

A data-driven Decision Support System for assessing and forecasting housing affordability risk across ten regions of England and Wales. The system integrates five UK government datasets into a unified analytical pipeline, constructs a Composite Affordability Index, applies OLS regression and Random Forest modelling to identify structural drivers of affordability change, generates five-year ARIMA forecasts, and produces structured policy recommendations through a rule-based decision logic layer.

---

## Project Structure

```
housing-dss/
├── data/
│   ├── raw/                          # Original downloaded datasets
│   └── processed/                    # Cleaned CSVs, model files, plots
├── notebooks/
│   ├── 01_eda.ipynb                  # Data loading, EDA, stationarity
│   ├── 02_features.ipynb             # CAI, risk classification, features
│   ├── 03_models.ipynb               # OLS, Random Forest, SHAP
│   ├── 04_forecasting.ipynb          # ARIMA forecasting, validation
│   └── 05_decision_support.ipynb     # DSS logic, dashboard, report
├── src/
│   ├── data_pipeline.py              # Dataset loading functions
│   ├── decision_rules.py             # Risk classification and policy rules
│   ├── models.py                     # OLS and RF model wrappers
│   └── app.py                        # Command-line DSS interface
├── requirements.txt
└── README.md
```

---

## Setup and Installation

### Prerequisites
- Python 3.10 or higher
- pip

### 1. Clone the repository
```bash
git clone https://github.com/koinchsushan/houseAffordabilityDSS
cd houseAffordabilityDSS
```

### 2. Create and activate virtual environment
```bash
# Create
python -m venv housing-dss-env

# Activate — Mac/Linux
source housing-dss-env/bin/activate

# Activate — Windows
housing-dss-env\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Register Jupyter kernel
```bash
python -m ipykernel install --user \
  --name=housing-dss-env \
  --display-name "Housing DSS (Python)"
```

---

## Data Sources

All datasets are publicly available under Open Government Licence. Download each file and save to `data/raw/` using the filenames below.

| File | Source | URL |
|---|---|---|
| `land_registry_hpi_full_2025.csv` | Land Registry | [UK HPI Data Downloads](https://www.gov.uk/government/collections/uk-house-price-index) |
| `ons_hp_earnings_ratio.xlsx` | ONS | [Housing Affordability 2024](https://www.ons.gov.uk/peoplepopulationandcommunity/housing/bulletins/housingaffordabilityinenglandandwales/2024) |
| `ons_cpi_mm23.csv` | ONS | [Consumer Price Indices](https://www.ons.gov.uk/economy/inflationandpriceindices/datasets/consumerpriceindices) |
| `earn01_average_weekly_earnings.xls` | ONS | [Average Weekly Earnings EARN01](https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/earningsandworkinghours/datasets/averageweeklyearningsearn01) |

---

## Running the Analysis

Run the notebooks in sequence. Each notebook is self-contained and reproducible.

```bash
# Open VS Code with Jupyter extension and select kernel:
# "Housing DSS (Python)"

# Or launch Jupyter directly
jupyter notebook
```

| Notebook | Run time | Output |
|---|---|---|
| `01_eda.ipynb` | ~1 min | master_affordability.csv, 7 plots |
| `02_features.ipynb` | ~1 min | features_engineered.csv, 2 plots |
| `03_models.ipynb` | ~2 min | model files (.pkl), 5 plots |
| `04_forecasting.ipynb` | ~2 min | arima_forecasts.csv, 3 plots |
| `05_decision_support.ipynb` | ~1 min | dss_report.csv, dashboard plot |

---

## Command-Line DSS Interface

The DSS can be queried directly from the terminal without opening any notebook.

```bash
# Full 10-region risk assessment report (current year: 2025)
python src/app.py

# Single region assessment
python src/app.py --region London
python src/app.py --region "North East"

# Historical year assessment
python src/app.py --year 2024
python src/app.py --region "South East" --year 2023
```

### Example output — full report
```
================================================================================
  HOUSING AFFORDABILITY DECISION SUPPORT SYSTEM
  CS7P01 | London Metropolitan University
  Assessment Year: 2025 | Forecast Horizon: 2030
================================================================================
Rank  Region                          Ratio    CAI       Risk   2030fc       Trend  Policy Action
1     London                          10.61   78.1   Critical    10.06   Improving  Intervene — recovery underway
2     South East                       9.58   62.2       High     9.52      Stable  Intervene — sustained stress
3     East of England                  9.02   55.6       High     8.91      Stable  Intervene — sustained stress
...
10    North East                       5.00   17.4        Low     4.97      Stable  No action required
```

### Example output — single region
```
=======================================================
  DSS ASSESSMENT REPORT — LONDON
=======================================================
  CURRENT STATUS (2025)
  Affordability Ratio  : 10.61×
  Composite Index (CAI): 78.1 / 100
  Risk Classification  : Critical
  Regional Rank        : 1 / 10

  FORECAST (2030)
  Projected Ratio      : 10.06×
  95% CI               : 6.52× – 13.61×
  Trajectory           : Improving

  RECOMMENDATION
  Policy Action        : Intervene — recovery underway
=======================================================
```

---

## Key Results

| Finding | Value |
|---|---|
| Regions above ONS threshold (5×) in 2025 | 10 / 10 |
| London affordability ratio 2025 | 10.61× |
| London affordability ratio 2021 peak | 12.9× |
| North East affordability ratio 2025 | 5.0× |
| London–rest gap 2025 | 3.5× |
| OLS Fixed Effects CV R² | 0.9805 |
| Random Forest CV R² (comparator) | 0.9676 |
| ARIMA walk-forward MAE (post-shock) | 0.436 ratio points |
| Regions projected to worsen by 2030 | 1 / 10 (Yorkshire, +0.01×) |

---

## Model Summary

### Regression — OLS with Regional Fixed Effects (Primary Model)
- **Target**: median affordability ratio
- **Features**: real house price, ratio lag 1 year, year-on-year price change %
- **Specification**: standardised features, East Midlands as reference region
- **CV R²**: 0.9805 | **CV RMSE**: 0.2530 | **Durbin-Watson**: 2.078
- **Key finding**: OLS Fixed Effects outperforms Random Forest on cross-validated R² — interpretable model chosen without accuracy cost

### Forecasting — ARIMA(1,1,0) per Region
- **Specification**: uniform (1,1,0) across all 10 regions
- **Horizon**: 2026–2030 with 95% confidence intervals
- **Validation**: walk-forward (train 1997–2019, test 2020–2025)
- **Post-shock MAE**: 0.436 ratio points | **Post-shock MAPE**: 5.40%

### Risk Classification — Composite Affordability Index
| Tier | CAI Range | Typical Ratio | 2025 Regions |
|---|---|---|---|
| Low | 0–24 | Below 5.5× | North East |
| Medium | 25–49 | 5.5×–7.5× | N. West, Wales, Yorks, W. Mids, E. Mids, S. West |
| High | 50–74 | 7.5×–10× | South East, East of England |
| Critical | 75–100 | Above 10× | London |

---

## Dependencies

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
statsmodels>=0.14
scikit-learn>=1.3
pmdarima>=2.0
shap>=0.42
joblib>=1.3
openpyxl>=3.1
xlrd>=2.0
ipykernel>=6.0
```

Full pinned versions in `requirements.txt`.

---

## Academic Context

- **Module**: CS7P01 MSc Project
- **Programme**: MSc Data Analytics
- **Institution**: London Metropolitan University
- **Academic Year**: 2025/26
- **Submission**: May 2026

### Key References
- Box, G.E.P. and Jenkins, G.M. (1976) *Time Series Analysis: Forecasting and Control*
- Busuioc, M. (2021) Accountable artificial intelligence. *Public Administration Review*, 81(5)
- Haffner, M.E.A. and Hulse, K. (2021) A fresh look at housing affordability. *International Journal of Urban Sciences*, 25(S1)
- Lundberg, S.M. and Lee, S.I. (2017) A unified approach to interpreting model predictions. *NeurIPS*, 30
- Meen, G. (2002) The time-series behavior of house prices. *Journal of Housing Economics*, 11(1)
- ONS (2025) *Housing affordability in England and Wales: 2024*

---

## Licence

Data sources used under [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

Project code: MIT Licence.
