import random
from pathlib import Path
import numpy as np

SEED = 42

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "results" / "figures"
TABLES_DIR = BASE_DIR / "results" / "tables"
REPORT_FIGURES_DIR = BASE_DIR / "report" / "figures"
REPORT_TABLES_DIR = BASE_DIR / "report" / "tables"
PAPER_REPORTED_CSV = BASE_DIR / "data" / "paper_reported.csv"

CITY_COORDS = {
    "Johannesburg": (-26.2044, 28.0416),
    "Pretoria": (-25.7479, 28.2293),
    "Cape Town": (-33.9249, 18.4241),
    "Durban": (-29.8587, 31.0218),
    "Bloemfontein": (-29.0852, 26.1596),
    "Port Elizabeth": (-33.9608, 25.6022),
    "East London": (-33.0153, 27.9116),
    "Vereeniging": (-26.6731, 27.9261),
    "Klerksdorp": (-26.8521, 26.6667),
    "Middelburg": (-25.7751, 29.4648),
    "Richards Bay": (-28.7830, 32.0377),
    "Worcester": (-33.6465, 19.4485),
}

UNMONITORED_CITIES = {
    "Kimberley": (-28.7282, 24.7499),
    "Kuruman": (-27.4524, 23.4327),
    "Lebowakgomo": (-24.2, 29.5),
    "Mahikeng": (-25.8560, 25.6403),
    "Upington": (-28.4478, 21.2561),
    "Vryburg": (-26.9550, 24.7280),
}

CATBOOST_PARAMS = {
    "iterations": 1000,
    "loss_function": "RMSE",
    "early_stopping_rounds": 100,
    "verbose": False,
    "random_seed": SEED,
}

XGBOOST_PARAMS = {
    "booster": "gbtree",
    "base_score": 0.5,
    "n_estimators": 1000,
    "max_depth": 3,
    "learning_rate": 0.01,
    "objective": "reg:squarederror",
    "early_stopping_rounds": 50,
    "random_state": SEED,
}

def escape_latex(s):
    # Escape special characters for LaTeX table cells
    return str(s).replace("%", r"\%").replace("_", r"\_")

def save_latex_tabular(df, filepath, align=None):
    cols = list(df.columns)
    header = " & ".join(escape_latex(c) for c in cols) + r" \\ \midrule"
    rows = []
    for _, row in df.iterrows():
        row_str = " & ".join(escape_latex(v) for v in row.values) + r" \\"
        rows.append(row_str)
    if align is None:
        align = "l" + "c" * (len(cols) - 1)
    content = [
        r"\begin{tabular}{" + align + r"}",
        r"\toprule",
        header,
        "\n".join(rows),
        r"\bottomrule",
        r"\end{tabular}"
    ]
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(content) + "\n")
