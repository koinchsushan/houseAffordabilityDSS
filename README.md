# Housing Affordability Decision Support System

**MSc Data Analytics | FC7P01 | London Metropolitan University**

A data-driven DSS for assessing and forecasting housing affordability risk across UK regions.

## Project Structure
- `data/raw/` — original downloaded datasets (not tracked by Git)
- `data/processed/` — cleaned, merged datasets (not tracked by Git)
- `notebooks/` — Jupyter notebooks for each analytical stage
- `src/` — reusable Python modules
- `app/` — optional Streamlit dashboard

## Setup
```bash
python -m venv housing-dss-env
source housing-dss-env/bin/activate
pip install -r requirements.txt
```

## Notebooks
1. `01_eda.ipynb` — Data loading, cleaning, exploratory analysis
2. `02_features.ipynb` — Feature engineering, affordability index
3. `03_models.ipynb` — Regression and statistical modelling
4. `04_forecasting.ipynb` — ARIMA/SARIMA forecasting
