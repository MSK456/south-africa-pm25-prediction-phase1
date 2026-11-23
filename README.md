# PM2.5 Prediction in South African Cities Using Machine Learning (Phase 1)

This repository contains our Phase 1 reproduction of the university Machine Learning semester project for Section AI-5B at FAST NUCES Islamabad, taught by Sir Ahmad Raza.

> **Note on Base Paper Identity:** The DOI link in the assignment sheet (`https://doi.org/10.3390/s21041064`) points to an MDPI *Sensors* article by Wardana et al. on edge deep learning, which differs from the paper title assigned. We implemented the paper matching the assigned title: *"Air pollution particulate matter (PM2.5) prediction in South African cities using machine learning techniques"* by Tshepang Duncan Morapedi and Ibidun Christiana Obagbuwa, *Frontiers in Artificial Intelligence*, vol. 6, p. 1230087, 2023 (DOI: `10.3389/frai.2023.1230087`).

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
│   ├── raw/                  # Raw CSV files from aqicn.org (excluded from repo/ZIP, gitignored)
│   ├── processed/            # Cleaned data and predictions (included in submission ZIP, gitignored)
│   └── paper_reported.csv    # Benchmark numbers reported in the base paper (tracked in Git)
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
│   ├── extra_stats.py        # Generates extra_stats.csv and aqi_precision_recall.csv
│   ├── make_figures.py       # Generates all figures and syncs report assets
│   └── compare_with_paper.py # Builds side-by-side comparison tables
├── results/
│   ├── figures/              # Generated high-resolution PNG plots (300 dpi)
│   ├── tables/               # Generated CSV and LaTeX tables (including extra_stats.csv and aqi_precision_recall.csv)
│   └── deviations.md         # Documented assumptions and paper deviations
├── report/                   # Single-column LaTeX project (Overleaf-ready)
│   ├── main.pdf              # Compiled 10-page academic report
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

## 3. Dataset Download Instructions and Access Terms

### Why Raw Data Files Are Excluded
In compliance with project submission instructions and data licensing agreements, the four raw annual CSV archives are **not included** in this GitHub repository or in the final submission ZIP:
1. **File Size Limits:** The four raw global CSV archives total approximately 490 MB (467.40 MiB). Each individual file exceeds GitHub's standard 100 MiB file-size limit for regular Git repositories.
2. **Platform Terms of Use:** The World Air Quality Index (WAQI) project terms of use ([https://aqicn.org/data-platform/tos/](https://aqicn.org/data-platform/tos/)) govern access for non-commercial research and educational use upon user registration, and restrict direct bulk re-hosting or redistribution of the raw global archives. Processed, cleaned subsets relevant to this project are retained in `data/processed/` for direct inspection and verification.

### Official Data Source & Access Instructions
Evaluators, researchers, and users can obtain the dataset directly from the official provider:
- **Official WAQI Portal:** [https://aqicn.org/data-platform/covid19/](https://aqicn.org/data-platform/covid19/)
- **Terms of Service:** [https://aqicn.org/data-platform/tos/](https://aqicn.org/data-platform/tos/) (governed under non-commercial research and educational use).

To obtain the data for replication:
1. Visit the WAQI COVID-19 open data platform: [https://aqicn.org/data-platform/covid19/](https://aqicn.org/data-platform/covid19/)
2. Request access through the platform's email registration form and agree to the WAQI Data Use Agreement.
3. Download the individual yearly CSV archives for the four study years (2019, 2020, 2021, and 2022).
4. Save the downloaded CSV files directly inside the `data/raw/` directory with these exact filenames:
   - `data/raw/waqi-covid19-airqualitydata-2019.csv`
   - `data/raw/waqi-covid19-airqualitydata-2020.csv`
   - `data/raw/waqi-covid19-airqualitydata-2021.csv`
   - `data/raw/waqi-covid19-airqualitydata-2022.csv`

> **Note on Data Units:** All PM$_{2.5}$ measurements in the WAQI dataset are reported as US EPA Air Quality Index (AQI) values, not raw physical concentrations in $\mu\text{g/m}^3$.

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

Once the raw data files are in `data/raw/`, run everything from end to end using the virtual environment Python interpreter (`.\.venv\Scripts\python.exe`):
```powershell
.\.venv\Scripts\python.exe run_all.py
```

> **Note on Environment & Dependency Versions:** Always run the code inside the `.venv` with the pinned package versions in `requirements.txt` (Python 3.11.9, scikit-learn 1.9.1). Logistic Regression's accuracy changes slightly with other scikit-learn versions (99.879 percent with version 1.8.0 versus 99.900 percent with version 1.9.1).

The script will automatically execute all 9 steps in sequence:
1. `preprocess.py`: Cleans raw records, drops UTC edge dates, and reindexes each city between its own first and last dates with mean imputation.
2. `cv_regression.py`: Runs 5-fold spatial `GroupKFold` (held-out cities) and per-city `TimeSeriesSplit`.
3. `datesplit_eval.py`: Trains on dates before 2022-01-01 and evaluates on 2022-01-01 onward.
4. `no_station.py`: Computes leave-one-city-out (LOCO) CV and predicts 2022 levels for 6 unmonitored towns.
5. `forecast.py`: Trains on data up to 2022-11-25 and forecasts recursively through 2023-12-31, saving predictions to `data/processed/` and comparing with real late-2022 data as a sanity check.
6. `aqi_classifiers.py`: Evaluates 5 classifiers on 6 AQI health categories and tests the 125.55 demo sample.
7. `extra_stats.py`: Computes overall mean PM2.5, rows per city, 999 readings, class distributions, per-classifier misclassifications and macro precision/recall/F1, and forecast maxima, writing `results/tables/extra_stats.csv` and `results/tables/aqi_precision_recall.csv`.
8. `make_figures.py`: Generates all PNG plots at high resolution (300 dpi) and copies figures and tables into `report/`.
9. `compare_with_paper.py`: Compares our results with `data/paper_reported.csv` and outputs side-by-side tables.

Running `run_all.py` twice with seed 42 will produce identical numerical results.

---

## 6. Execution Run Time

Measured on an **Intel Core i5-8250U CPU @ 1.60GHz (4 cores, 8 threads), 16 GB RAM, Windows 11 Pro** running Python 3.11.9 within `.venv`:

- **Total Pipeline Execution (`run_all.py`):** The full run takes about 3 to 7 minutes (measured 3:14 and 3:16 with `run_all.py`, and 6:35 in an earlier run under heavier load).

### Per-Script Breakdown

| Script | Stage Description | Runtime |
| :--- | :--- | :---: |
| `src/preprocess.py` | Raw data loading, edge date pruning, and mean imputation | 25.43 s |
| `src/cv_regression.py` | 5-Fold GroupKFold and per-city TimeSeriesSplit cross-validation | 144.71 s |
| `src/datesplit_eval.py` | Pre-2022 train and 2022 test temporal holdout evaluation | 3.90 s |
| `src/no_station.py` | Leave-one-city-out CV and spatial predictions for unmonitored cities | 19.35 s |
| `src/forecast.py` | Multi-step recursive daily forecast through 2023 with sanity check | 146.24 s |
| `src/aqi_classifiers.py` | 5 AQI classification models, metrics evaluation, and demo test | 11.81 s |
| `src/make_figures.py` | High-resolution figure generation and asset synchronization | 32.03 s |
| `src/compare_with_paper.py` | Side-by-side comparison tables against published paper baselines | 0.16 s |
| **Total** | **Full end-to-end execution (`python run_all.py`)** | **394.90 s** |

> **Note on Execution Timing:** The sum of the individual script runtimes is approximately 383.63 seconds. The ~11.27 second difference from the total measured pipeline execution time (394.90 seconds via `Measure-Command`) is attributable to pipeline execution overhead: Python interpreter startup time across subprocess invocations, multi-stage module imports (specifically heavy machine learning libraries like CatBoost, XGBoost, and Scikit-Learn that initialize C/C++ backends upon import), and intermediate disk I/O between consecutive pipeline stages.

---

## 7. Global Seed and Determinism
A global random seed of `42` is set in `src/config.py` for Python's `random`, `numpy`, scikit-learn estimators, CatBoost (`random_seed=42`), and XGBoost (`random_state=42`).

---

## 8. How to Build the Report
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

## 9. Honest Note on Approximations and Paper Inconsistencies
Full details are recorded in `results/deviations.md`. The key points are:
- **Unmonitored Cities**: The base paper mentioned Google Earth Engine and Sentinel-5P layers without providing code or band definitions. We approximated this using spatial coordinates and evaluated reliability via Leave-One-City-Out validation.
- **Lag Features**: The paper showed 3 lag features in Figure 12 without defining their shift intervals. We used annual 52-week shifts (364, 728, 1092 days).
- **Label Leakage in Classification**: The paper achieved near-100% accuracy because the AQI category label was directly computed from the single input feature (`Median_PM25`). This is a deterministic mapping rather than true predictive power.
- **Model Estimator Discrepancy**: The paper text states 1,000 estimators for XGBoost, but Figure 8 shows `n_estimators=500`. We used 1,000 estimators.
- **Recursive Forecasting**: The paper used an hourly dataframe with frozen lags. We implemented a daily recursive forecast where predicted values dynamically update future lags.

