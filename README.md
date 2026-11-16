# PM2.5 Prediction in South African Cities Using Machine Learning (Phase 1)

This repository contains our Phase 1 reproduction of the university Machine Learning semester project for Section AI-5B at FAST NUCES Islamabad, taught by Sir Ahmad Raza.

**Group 46:**
- Shaheed Khan (24I-0037)
- Abuzar Ali (24I-0041)

**Submission Date:** 19 November 2026

**GitHub Repository:** `https://github.com/MSK456/south-africa-pm25-prediction-phase1`

---

## 1. Project Description
This project reproduces the core methodology and experiments from the research paper:
> **Tshepang Duncan Morapedi and Ibidun Christiana Obagbuwa**, *"Air pollution particulate matter (PM2.5) prediction in South African cities using machine learning techniques"*, **Frontiers in Artificial Intelligence**, vol. 6, p. 1230087, 2023. DOI: `10.3389/frai.2023.1230087`.

The paper predicts daily particulate matter ($\text{PM}_{2.5}$) levels across 12 South African cities using gradient boosting models (CatBoost and XGBoost), classifies Air Quality Index (AQI) health status using five classifiers, predicts pollution levels for unmonitored cities, and forecasts future pollution through recursive multi-step prediction.

---

## 2. Repository Structure
```text
south-africa-pm25-prediction-phase1/
├── data/
│   ├── raw/                  # Raw CSV files from aqicn.org (gitignored)
│   ├── processed/            # Cleaned data and predictions (gitignored)
│   └── paper_reported.csv    # Benchmark numbers reported in the base paper
├── reference/                # Base paper PDF (gitignored)
├── src/
│   ├── config.py             # Global constants, coordinates, seed, parameters
│   ├── load_data.py          # Reads and filters raw WAQI CSV files
│   ├── preprocess.py         # Temporal reindexing and city mean imputation
│   ├── features.py           # Calendar features and 52-week annual lags
│   ├── cv_regression.py      # 5-fold spatial GroupKFold and TimeSeriesSplit
│   ├── datesplit_eval.py     # Holdout evaluation (< 2022 vs >= 2022)
│   ├── no_station.py         # Prediction for 6 unmonitored cities + LOCO CV
│   ├── forecast.py           # Recursive daily forecast (2022-2023) + sanity check
│   ├── aqi_classifiers.py    # 5 AQI status classifiers + interactive demo
│   ├── make_figures.py       # Generates all figures and syncs report assets
│   └── compare_with_paper.py # Builds side-by-side comparison tables
├── results/
│   ├── figures/              # Generated high-resolution PNG plots (200 dpi)
│   ├── tables/               # Generated CSV and LaTeX (.tex) tables
│   └── deviations.md         # Documented assumptions and paper deviations
├── report/                   # Single-column LaTeX project (Overleaf-ready)
│   ├── main.tex              # Main LaTeX file with cover page
│   ├── sections/             # Modular section files (01 to 09)
│   ├── figures/              # Synced figure assets
│   ├── tables/               # Synced LaTeX table assets
│   └── references.bib        # Verified bibliography entries
├── run_all.py                # Single script to run the whole pipeline
├── requirements.txt          # Pinned Python package dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # Project documentation
```

---

## 3. Dataset Download Instructions
Due to the terms of service of the World Air Quality Index (WAQI) platform, the raw data files cannot be redistributed in this public repository.

To download the raw data:
1. Visit the WAQI COVID-19 open data platform: [https://aqicn.org/data-platform/covid19/](https://aqicn.org/data-platform/covid19/)
2. Accept the terms of service and download the annual CSV archives for the years:
   - `waqi-covid19-airqualitydata-2019.csv`
   - `waqi-covid19-airqualitydata-2020.csv`
   - `waqi-covid19-airqualitydata-2021.csv`
   - `waqi-covid19-airqualitydata-2022.csv`
3. Place these 4 CSV files directly inside the `data/raw/` directory.

> **Note on Units:** The values in this dataset are US EPA Air Quality Index (AQI) values, not raw physical concentrations in $\mu\text{g/m}^3$.

---

## 4. Environment Setup (PowerShell)

We use Python 3.11 with an isolated virtual environment (`.venv`).

In Windows PowerShell, run:
```powershell
# 1. Create the virtual environment
python -m venv .venv

# 2. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install pinned dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. How to Run the Complete Pipeline

Once the raw data files are in `data/raw/`, run everything from end to end with one command:
```powershell
python run_all.py
```

> **Note on Environment & Dependency Versions:** Always run the code inside the `.venv` with the pinned package versions in `requirements.txt` (Python 3.11.9, scikit-learn 1.9.1). Logistic Regression's accuracy changes slightly with other scikit-learn versions (99.879 percent with version 1.8.0 versus 99.900 percent with version 1.9.1).

The script will automatically execute all steps in sequence:
1. `preprocess.py`: Cleans raw records, drops UTC edge dates, and reindexes each city between its own first and last dates with mean imputation.
2. `cv_regression.py`: Runs 5-fold spatial `GroupKFold` (held-out cities) and per-city `TimeSeriesSplit`.
3. `datesplit_eval.py`: Trains on dates before 2022-01-01 and evaluates on 2022-01-01 onward.
4. `no_station.py`: Computes leave-one-city-out (LOCO) CV and predicts 2022 levels for 6 unmonitored towns.
5. `forecast.py`: Trains on data up to 2022-11-25 and forecasts recursively through 2023-12-31, comparing with real late-2022 data as a sanity check.
6. `aqi_classifiers.py`: Evaluates 5 classifiers on 6 AQI health categories and tests the 125.55 demo sample.
7. `make_figures.py`: Generates all PNG plots at 200 dpi and copies figures and tables into `report/`.
8. `compare_with_paper.py`: Compares our results with `data/paper_reported.csv` and outputs side-by-side tables.

Running `run_all.py` twice with seed 42 will produce identical numerical results.

---

## 6. Global Seed and Determinism
A global random seed of `42` is set in `src/config.py` for Python's `random`, `numpy`, scikit-learn estimators, CatBoost (`random_seed=42`), and XGBoost (`random_state=42`).

---

## 7. How to Build the Report
The `report/` folder contains a single-column academic report formatted for Overleaf or local MiKTeX/pdfLaTeX:
1. Zip the contents of the `report/` folder (or open it directly in a LaTeX editor).
2. Upload to Overleaf or compile locally using `pdflatex` or `latexmk`:
   ```powershell
   cd report
   pdflatex main.tex
   bibtex main
   pdflatex main.tex
   pdflatex main.tex
   ```
3. All tables inside `report/tables/` are generated directly by the Python code and included via `\input{}`, ensuring the report numbers always match the code.

---

## 8. Honest Note on Approximations and Paper Inconsistencies
Full details are recorded in `results/deviations.md`. The key points are:
- **Unmonitored Cities**: The base paper mentioned Google Earth Engine and Sentinel-5P layers without providing code or band definitions. We approximated this using spatial coordinates and evaluated reliability via Leave-One-City-Out validation.
- **Lag Features**: The paper showed 3 lag features in Figure 12 without defining their shift intervals. We used annual 52-week shifts (364, 728, 1092 days).
- **Label Leakage in Classification**: The paper achieved near-100% accuracy because the AQI category label was directly computed from the single input feature (`Median_PM25`). This is a deterministic mapping rather than true predictive power.
- **Model Estimator Discrepancy**: The paper text states 1,000 estimators for XGBoost, but Figure 8 shows `n_estimators=500`. We used 1,000 estimators.
- **Recursive Forecasting**: The paper used an hourly dataframe with frozen lags. We implemented a daily recursive forecast where predicted values dynamically update future lags.

