from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

from config import (
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    TABLES_DIR,
    XGBOOST_PARAMS,
    save_latex_tabular,
)
from features import add_date_features

def run_future_forecast():
    print("\n=== Future Forecasting (2022-11-26 to 2023-12-31) ===")
    
    clean_path = PROCESSED_DATA_DIR / "pm25_clean.csv"
    df = pd.read_csv(clean_path)
    df["Date"] = pd.to_datetime(df["Date"])
    
    features = [
        "dayofweek", "quarter", "month", "year",
        "dayofyear", "dayofmonth", "weekofyear",
        "lag1", "lag2", "lag3"
    ]
    target = "Median_PM25"
    
    # Train cutoff: up to 2022-11-25
    train_cutoff = pd.Timestamp("2022-11-25")
    forecast_start = pd.Timestamp("2022-11-26")
    forecast_end = pd.Timestamp("2023-12-31")
    
    forecast_dates = pd.date_range(forecast_start, forecast_end, freq="D")
    
    all_forecast_records = []
    sanity_results = []
    
    cities = sorted(df["City"].unique())
    
    for city in cities:
        city_history = df[df["City"] == city].sort_values("Date").copy()
        
        # Build training set strictly before cutoff
        train_df = city_history[city_history["Date"] <= train_cutoff].copy()
        
        # Compute lags for training
        train_df["lag1"] = train_df["Median_PM25"].shift(364)
        train_df["lag2"] = train_df["Median_PM25"].shift(728)
        train_df["lag3"] = train_df["Median_PM25"].shift(1092)
        train_df = add_date_features(train_df)
        
        # Train XGBoost on city history
        xgb_p = XGBOOST_PARAMS.copy()
        xgb_p.pop("early_stopping_rounds", None)
        model = XGBRegressor(**xgb_p)
        model.fit(train_df[features], train_df[target], verbose=False)
        
        # Value map initialized with history for lag lookup
        history_map = dict(zip(train_df["Date"].dt.strftime("%Y-%m-%d"), train_df["Median_PM25"]))
        
        city_preds = []
        for cur_date in forecast_dates:
            d_str = cur_date.strftime("%Y-%m-%d")
            
            # Lookup lags
            l1_date = (cur_date - timedelta(days=364)).strftime("%Y-%m-%d")
            l2_date = (cur_date - timedelta(days=728)).strftime("%Y-%m-%d")
            l3_date = (cur_date - timedelta(days=1092)).strftime("%Y-%m-%d")
            
            lag1_val = history_map.get(l1_date, np.nan)
            lag2_val = history_map.get(l2_date, np.nan)
            lag3_val = history_map.get(l3_date, np.nan)
            
            row = pd.DataFrame([{
                "Date": cur_date,
                "City": city,
                "lag1": lag1_val,
                "lag2": lag2_val,
                "lag3": lag3_val,
            }])
            row = add_date_features(row)
            
            pred_val = float(model.predict(row[features])[0])
            history_map[d_str] = pred_val
            
            city_preds.append({
                "Date": d_str,
                "City": city,
                "Predicted_PM25": pred_val,
                "lag1": lag1_val,
                "lag2": lag2_val,
                "lag3": lag3_val
            })
            
        city_pred_df = pd.DataFrame(city_preds)
        all_forecast_records.append(city_pred_df)
        
        # Sanity check: compare against actual 2022-11-26 to 2022-12-31 data if available
        actual_late = city_history[(city_history["Date"] >= forecast_start) & (city_history["Date"] <= pd.Timestamp("2022-12-31"))]
        if len(actual_late) > 0:
            merged = pd.merge(
                actual_late[["Date", "Median_PM25"]].assign(Date=lambda x: x["Date"].dt.strftime("%Y-%m-%d")),
                city_pred_df[["Date", "Predicted_PM25"]],
                on="Date"
            )
            rmse = np.sqrt(mean_squared_error(merged["Median_PM25"], merged["Predicted_PM25"]))
            mae = mean_absolute_error(merged["Median_PM25"], merged["Predicted_PM25"])
            sanity_results.append({
                "City": city,
                "Eval Days": len(merged),
                "Sanity RMSE": f"{rmse:.3f}",
                "Sanity MAE": f"{mae:.3f}",
                "raw_rmse": rmse,
                "raw_mae": mae
            })
            print(f"  {city:15s} | Forecasted 401 days | Late-2022 Eval ({len(merged)} days) RMSE: {rmse:.3f}, MAE: {mae:.3f}")
        else:
            sanity_results.append({
                "City": city,
                "Eval Days": 0,
                "Sanity RMSE": "-",
                "Sanity MAE": "-",
                "raw_rmse": np.nan,
                "raw_mae": np.nan
            })
            print(f"  {city:15s} | Forecasted 401 days | No late-2022 data (ended earlier)")
            
    full_forecast_df = pd.concat(all_forecast_records, ignore_index=True)
    full_forecast_df.to_csv(PROCESSED_DATA_DIR / "future_forecast_2022_2023.csv", index=False)
    
    # Sanity table
    sanity_df = pd.DataFrame(sanity_results)
    valid_rmses = [r["raw_rmse"] for r in sanity_results if not np.isnan(r["raw_rmse"])]
    valid_maes = [r["raw_mae"] for r in sanity_results if not np.isnan(r["raw_mae"])]
    mean_sanity = {
        "City": "Average (Available)",
        "Eval Days": int(np.sum([r["Eval Days"] for r in sanity_results])),
        "Sanity RMSE": f"{float(np.mean(valid_rmses)):.3f}",
        "Sanity MAE": f"{float(np.mean(valid_maes)):.3f}"
    }
    sanity_tex = pd.concat([
        sanity_df[["City", "Eval Days", "Sanity RMSE", "Sanity MAE"]],
        pd.DataFrame([mean_sanity])
    ], ignore_index=True)
    
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    sanity_tex.to_csv(TABLES_DIR / "forecast_sanity_rmse.csv", index=False)
    save_latex_tabular(sanity_tex, TABLES_DIR / "forecast_sanity_rmse.tex")
    
    # Plot future forecasts in a 2-column by 6-row layout
    import matplotlib.dates as mdates
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(6, 2, figsize=(6.3, 8.0), sharex=True, dpi=300)
    axes = axes.flatten()
    
    for idx, city in enumerate(cities):
        ax = axes[idx]
        sub = full_forecast_df[full_forecast_df["City"] == city].copy()
        sub["Date"] = pd.to_datetime(sub["Date"])
        ax.plot(sub["Date"], sub["Predicted_PM25"], color="#1f77b4", linewidth=1.0)
        ax.set_title(city, fontsize=8.5, fontweight="bold")
        ax.set_ylabel("AQI", fontsize=8)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)
        if idx >= 10:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.tick_params(axis="x", rotation=0, labelsize=8)
        
    plt.tight_layout()
    fig_path = FIGURES_DIR / "figure13_future_forecast_grid.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved 2x6 forecast grid to {fig_path}")
    
    return full_forecast_df, sanity_tex

if __name__ == "__main__":
    run_future_forecast()

