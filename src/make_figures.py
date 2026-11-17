import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit

from config import (
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    REPORT_FIGURES_DIR,
    REPORT_TABLES_DIR,
    TABLES_DIR,
)

def set_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["axes.labelsize"] = 10

def plot_figure1_workflow():
    fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def draw_box(x, y, w, h, text, color="#e8f0fe", ec="#1a73e8"):
        rect = patches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.8,rounding_size=2",
            facecolor=color, edgecolor=ec, linewidth=1.2
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=8.2, fontweight="medium", linespacing=1.2)

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color="#3c4043", lw=1.2, mutation_scale=10)
        )

    # Column 1: Ingestion & Filter
    draw_box(12, 75, 18, 28, "Raw WAQI CSVs\n(2019-2022 Data)")
    draw_box(12, 25, 18, 28, "Filter ZA & pm25\n(South Africa)")
    draw_arrow(12, 61, 12, 39)

    # Column 2: Cleaning & Features
    draw_box(36, 25, 18, 28, "Per-City Reindex\n& Mean Imputation")
    draw_box(36, 75, 18, 28, "Feature Engineering\n(7 Date Feats, 3 Lags)")
    draw_arrow(21, 25, 27, 25)
    draw_arrow(36, 39, 36, 61)

    # Column 3: Three model branches
    draw_box(63, 83, 22, 22, "Regression (CB & XGB)\nGroupKFold, TS, DateSplit", color="#e6f4ea", ec="#137333")
    draw_box(63, 50, 22, 22, "AQI Classification\n5 Classifiers & Demo", color="#e6f4ea", ec="#137333")
    draw_box(63, 17, 22, 22, "Forecast & Spatial\nRecursive, 6 Cities", color="#e6f4ea", ec="#137333")

    # Arrows from Feature Engineering to 3 branches
    draw_arrow(45, 78, 52, 83)
    draw_arrow(45, 75, 52, 50)
    draw_arrow(45, 72, 52, 17)

    # Column 4: Comparison & Results
    draw_box(88, 50, 18, 38, "Results &\nPaper Comparison\n(Tables & Figures)", color="#fef7e0", ec="#b06000")

    # Arrows from 3 branches to Results
    draw_arrow(74, 83, 80, 58)
    draw_arrow(74, 50, 79, 50)
    draw_arrow(74, 17, 80, 42)

    plt.tight_layout()
    out = FIGURES_DIR / "figure1_workflow.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

def plot_figure2_city_and_month_pm25(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    city_order = [
        "Middelburg", "Pretoria", "East London", "Johannesburg",
        "Cape Town", "Vereeniging", "Klerksdorp", "Richards Bay",
        "Port Elizabeth", "Worcester", "Bloemfontein", "Durban"
    ]
    cities = [c for c in city_order if c in df["City"].values]
    
    sns.barplot(
        data=df,
        x="City",
        y="Median_PM25",
        order=cities,
        hue="City",
        legend=False,
        palette="tab10",
        capsize=0.1,
        err_kws={"linewidth": 1.2, "color": "black"},
        ax=ax1
    )
    ax1.set_title("(a) Mean PM2.5 AQI by Monitoring City", fontsize=11, fontweight="bold")
    ax1.set_xlabel("City", fontsize=10)
    ax1.set_ylabel("PM2.5 Median (AQI)", fontsize=10)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    
    df_m = df.copy()
    df_m["Date"] = pd.to_datetime(df_m["Date"])
    df_m["Month"] = df_m["Date"].dt.month
    
    sns.lineplot(
        data=df_m,
        x="Month",
        y="Median_PM25",
        color="#1f77b4",
        errorbar=("ci", 95),
        linewidth=2.0,
        ax=ax2
    )
    ax2.set_title("(b) Monthly Profile (Winter Peak)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Month of Year", fontsize=10)
    ax2.set_ylabel("PM2.5 Median (AQI)", fontsize=10)
    ax2.set_xticks(range(1, 13))
    
    plt.tight_layout()
    out = FIGURES_DIR / "figure2_city_and_month_pm25.png"
    plt.savefig(out, dpi=200)
    plt.close()

def plot_figure3_city_bar(df):
    plt.figure(figsize=(10, 5))
    city_order = [
        "Middelburg", "Pretoria", "East London", "Johannesburg",
        "Cape Town", "Vereeniging", "Klerksdorp", "Richards Bay",
        "Port Elizabeth", "Worcester", "Bloemfontein", "Durban"
    ]
    cities = [c for c in city_order if c in df["City"].values]
    
    sns.barplot(
        data=df,
        x="City",
        y="Median_PM25",
        order=cities,
        hue="City",
        legend=False,
        palette="tab10",
        capsize=0.1,
        err_kws={"linewidth": 1.2, "color": "black"}
    )
    plt.title("PM2.5 Median by City in South Africa (2019-2022)", fontweight="bold", fontsize=12)
    plt.xlabel("City", fontsize=11)
    plt.ylabel("PM2.5 Median (AQI)", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.tight_layout()
    out = FIGURES_DIR / "figure3_city_bar.png"
    plt.savefig(out, dpi=200)
    plt.close()

def plot_figure3_pm25_by_date(df):
    plt.figure(figsize=(12, 4.5))
    df_sorted = df.sort_values("Date").copy()
    df_sorted["Date"] = pd.to_datetime(df_sorted["Date"])
    
    daily_stats = df_sorted.groupby("Date")["Median_PM25"].agg(["median", "min", "max"]).reset_index()
    
    plt.plot(daily_stats["Date"], daily_stats["median"], color="#1f77b4", linewidth=1.0, label="Daily Median Across Cities")
    plt.fill_between(daily_stats["Date"], daily_stats["min"], daily_stats["max"], color="#1f77b4", alpha=0.2, label="City Min-Max Range")
    
    # Clip y-axis to sensible range 0 to 400 to eliminate outlier squash
    plt.ylim(0, 400)
    plt.title("PM2.5 Concentration Over Time Across South African Stations (Clipped to 0-400)", fontweight="bold", fontsize=12)
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("PM2.5 Median (AQI)", fontsize=11)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    out = FIGURES_DIR / "figure3_pm25_by_date.png"
    plt.savefig(out, dpi=200)
    plt.close()

def plot_figure3_pm25_by_month(df):
    plt.figure(figsize=(8, 4.5))
    df_m = df.copy()
    df_m["Date"] = pd.to_datetime(df_m["Date"])
    df_m["Month"] = df_m["Date"].dt.month
    
    sns.lineplot(
        data=df_m,
        x="Month",
        y="Median_PM25",
        color="#1f77b4",
        errorbar=("ci", 95),
        linewidth=2.0
    )
    plt.title("Seasonal PM2.5 Pattern by Month (Winter Surge)", fontweight="bold", fontsize=12)
    plt.xlabel("Month of Year", fontsize=11)
    plt.ylabel("PM2.5 Median (AQI)", fontsize=11)
    plt.xticks(range(1, 13), fontsize=10)
    plt.tight_layout()
    out = FIGURES_DIR / "figure3_pm25_by_month.png"
    plt.savefig(out, dpi=200)
    plt.close()

def plot_figure4_location_scatter(df):
    plt.figure(figsize=(8, 6.2))
    city_means = df.groupby(["City", "Lat", "Long"])["Median_PM25"].mean().reset_index()
    
    scatter = plt.scatter(
        city_means["Long"],
        city_means["Lat"],
        c=city_means["Median_PM25"],
        cmap="viridis",
        s=130,
        edgecolors="black",
        linewidths=1.2
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label("Mean PM2.5 (AQI)", fontsize=11)
    
    # Manual offsets to prevent overlapping city labels
    offsets = {
        "Pretoria": (+0.30, +0.35),
        "Middelburg": (+0.35, -0.20),
        "Johannesburg": (-2.80, +0.30),
        "Vereeniging": (+0.30, -0.45),
        "Klerksdorp": (-2.80, -0.35),
        "Cape Town": (-2.50, -0.35),
        "Worcester": (+0.35, +0.25),
        "Bloemfontein": (+0.35, -0.10),
        "Durban": (+0.35, +0.10),
        "Richards Bay": (+0.35, +0.15),
        "East London": (+0.35, -0.25),
        "Port Elizabeth": (-3.10, -0.35),
    }
    
    for _, r in city_means.iterrows():
        dx, dy = offsets.get(r["City"], (+0.3, +0.1))
        plt.annotate(
            r["City"],
            (r["Long"] + dx, r["Lat"] + dy),
            fontsize=9,
            fontweight="bold"
        )
        
    plt.title("South African Monitoring Stations Spatial Distribution", fontweight="bold", fontsize=12)
    plt.xlabel("Longitude (°E)", fontsize=11)
    plt.ylabel("Latitude (°S)", fontsize=11)
    plt.xlim(16, 35)
    plt.ylim(-36, -23)
    plt.tight_layout()
    out = FIGURES_DIR / "figure4_location_scatter.png"
    plt.savefig(out, dpi=200)
    plt.close()

def plot_figure1_timeseries_cv(df):
    plt.figure(figsize=(10, 8))
    jhb = df[df["City"] == "Johannesburg"].sort_values("Date").reset_index(drop=True)
    jhb["Date"] = pd.to_datetime(jhb["Date"])
    
    tscv = TimeSeriesSplit(n_splits=5, test_size=150, gap=24)
    
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(jhb)):
        plt.subplot(5, 1, fold + 1)
        train_dates = jhb.iloc[tr_idx]["Date"]
        train_vals = jhb.iloc[tr_idx]["Median_PM25"]
        test_dates = jhb.iloc[te_idx]["Date"]
        test_vals = jhb.iloc[te_idx]["Median_PM25"]
        
        plt.plot(train_dates, train_vals, color="#1f77b4", linewidth=0.9, label="Train" if fold == 0 else "")
        plt.plot(test_dates, test_vals, color="#d62728", linewidth=1.1, label="Test" if fold == 0 else "")
        plt.axvline(test_dates.iloc[0], color="black", linestyle="--", linewidth=0.8)
        
        plt.title(f"Johannesburg TimeSeriesSplit Fold {fold}", fontsize=9, fontweight="bold")
        plt.ylabel("PM2.5", fontsize=8)
        if fold == 0:
            plt.legend(loc="upper left", fontsize=8)
        if fold < 4:
            plt.xticks([])
            
    plt.xlabel("Date", fontsize=9)
    plt.tight_layout()
    out = FIGURES_DIR / "figure1_timeseries_cv_folds.png"
    plt.savefig(out, dpi=200)
    plt.close()

def plot_figure2_datesplit(df):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    split_date = pd.Timestamp("2022-01-01")
    
    for idx, city in enumerate(["Vereeniging", "Cape Town"]):
        ax = axes[idx]
        sub = df[df["City"] == city].sort_values("Date").copy()
        sub["Date"] = pd.to_datetime(sub["Date"])
        
        train_part = sub[sub["Date"] < split_date]
        test_part = sub[sub["Date"] >= split_date]
        
        ax.plot(train_part["Date"], train_part["Median_PM25"], color="#1f77b4", linewidth=1.0, label="Training Set (< 2022-01-01)")
        ax.plot(test_part["Date"], test_part["Median_PM25"], color="#d62728", linewidth=1.1, label="Test Set (>= 2022-01-01)")
        ax.axvline(split_date, color="black", linestyle="--", linewidth=1.0)
        
        ax.set_title(f"{city} Data Train/Test Split", fontweight="bold", fontsize=11)
        ax.set_ylabel("PM2.5 (AQI)", fontsize=10)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
        
    axes[1].set_xlabel("Date", fontsize=10)
    plt.tight_layout()
    out = FIGURES_DIR / "figure2_datesplit_split.png"
    plt.savefig(out, dpi=200)
    plt.close()

def plot_figure7_catboost_johannesburg():
    oof_path = PROCESSED_DATA_DIR / "pm25_oof_preds.csv"
    if not oof_path.exists():
        return
    df_oof = pd.read_csv(oof_path)
    df_oof["Date"] = pd.to_datetime(df_oof["Date"])
    
    jhb = df_oof[df_oof["City"] == "Johannesburg"].sort_values("Date").copy()
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Smoothed time series (7-day rolling)
    jhb["Actual_smooth"] = jhb["Median_PM25"].rolling(7, min_periods=1).mean()
    jhb["Pred_smooth"] = jhb["preds_cb"].rolling(7, min_periods=1).mean()
    
    axes[0].plot(jhb["Date"], jhb["Actual_smooth"], color="#1f77b4", linewidth=1.4, label="Actual (7-day smooth)")
    axes[0].plot(jhb["Date"], jhb["Pred_smooth"], color="#ff7f0e", linewidth=1.4, label="Predicted CatBoost (7-day smooth)")
    axes[0].set_title("Time Series Plot for Johannesburg (Actual vs Predicted)", fontweight="bold", fontsize=11)
    axes[0].set_ylabel("PM2.5 (AQI)", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # Scatter plot actual vs predicted
    axes[1].scatter(jhb["Median_PM25"], jhb["preds_cb"], alpha=0.3, color="#1f77b4", s=18)
    max_val = max(jhb["Median_PM25"].max(), jhb["preds_cb"].max(), 200)
    axes[1].plot([0, max_val], [0, max_val], color="red", linestyle="--", linewidth=1.5, label="y = x (Perfect Prediction)")
    axes[1].set_title("CatBoost Regressor: Actual vs Predicted PM2.5 (Johannesburg)", fontweight="bold", fontsize=11)
    axes[1].set_xlabel("Actual Median_PM25", fontsize=10)
    axes[1].set_ylabel("Predicted preds", fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    out = FIGURES_DIR / "figure7_catboost_johannesburg.png"
    plt.savefig(out, dpi=200)
    plt.close()

def plot_figure9_xgboost_klerksdorp():
    oof_path = PROCESSED_DATA_DIR / "pm25_oof_preds.csv"
    if not oof_path.exists():
        return
    df_oof = pd.read_csv(oof_path)
    df_oof["Date"] = pd.to_datetime(df_oof["Date"])
    
    kld = df_oof[df_oof["City"] == "Klerksdorp"].sort_values("Date").copy()
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Scatter plot
    axes[0].scatter(kld["Median_PM25"], kld["preds_xgb"], alpha=0.35, color="#2ca02c", s=20)
    max_val = max(kld["Median_PM25"].max(), kld["preds_xgb"].max(), 120)
    axes[0].plot([0, max_val], [0, max_val], color="red", linestyle="--", linewidth=1.5, label="y = x (Perfect Prediction)")
    axes[0].set_title("XGBoost Regressor: Actual vs Predicted PM2.5 (Klerksdorp)", fontweight="bold", fontsize=11)
    axes[0].set_xlabel("Actual Median_PM25", fontsize=10)
    axes[0].set_ylabel("Predicted preds", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # Smoothed time series
    kld["Actual_smooth"] = kld["Median_PM25"].rolling(7, min_periods=1).mean()
    kld["Pred_smooth"] = kld["preds_xgb"].rolling(7, min_periods=1).mean()
    
    axes[1].plot(kld["Date"], kld["Actual_smooth"], color="#1f77b4", linewidth=1.4, label="Actual (7-day smooth)")
    axes[1].plot(kld["Date"], kld["Pred_smooth"], color="#ff7f0e", linewidth=1.4, label="Predicted XGBoost (7-day smooth)")
    axes[1].set_title("Time Series Plot for Klerksdorp (Actual vs Predicted)", fontweight="bold", fontsize=11)
    axes[1].set_ylabel("PM2.5 (AQI)", fontsize=10)
    axes[1].set_xlabel("Date", fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    out = FIGURES_DIR / "figure9_xgboost_klerksdorp.png"
    plt.savefig(out, dpi=200)
    plt.close()

def sync_report_assets():
    REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    for f in FIGURES_DIR.glob("*.png"):
        shutil.copy(f, REPORT_FIGURES_DIR / f.name)
    
    for t in TABLES_DIR.glob("*.tex"):
        shutil.copy(t, REPORT_TABLES_DIR / t.name)
        
    for c in TABLES_DIR.glob("*.csv"):
        shutil.copy(c, REPORT_TABLES_DIR / c.name)
        
    print(f"Synced figures to {REPORT_FIGURES_DIR} and tables to {REPORT_TABLES_DIR}")

def make_all_figures():
    set_style()
    clean_path = PROCESSED_DATA_DIR / "pm25_clean.csv"
    df = pd.read_csv(clean_path)
    
    print("\n=== Generating All Figures ===")
    plot_figure1_workflow()
    plot_figure2_city_and_month_pm25(df)
    plot_figure3_city_bar(df)
    plot_figure3_pm25_by_date(df)
    plot_figure3_pm25_by_month(df)
    plot_figure4_location_scatter(df)
    plot_figure1_timeseries_cv(df)
    plot_figure2_datesplit(df)
    plot_figure7_catboost_johannesburg()
    plot_figure9_xgboost_klerksdorp()
    
    sync_report_assets()
    print("All figures successfully created and synchronized.")

if __name__ == "__main__":
    make_all_figures()
