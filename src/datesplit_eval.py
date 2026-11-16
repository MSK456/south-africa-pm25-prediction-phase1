import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

from config import (
    CATBOOST_PARAMS,
    PROCESSED_DATA_DIR,
    REPORT_TABLES_DIR,
    TABLES_DIR,
    XGBOOST_PARAMS,
    save_latex_tabular,
)
from features import build_features

def evaluate_date_split(df=None):
    if df is None:
        df = build_features()
        
    features = [
        "dayofweek", "quarter", "month", "year",
        "dayofyear", "dayofmonth", "weekofyear",
        "lag1", "lag2", "lag3"
    ]
    target = "Median_PM25"
    
    train_mask = df["Date"] < "2022-01-01"
    test_mask = df["Date"] >= "2022-01-01"
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"\n=== Date Split Evaluation ===")
    print(f"Train samples (< 2022-01-01): {len(train_df)}")
    print(f"Test samples  (>= 2022-01-01): {len(test_df)}")
    
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    # CatBoost
    cb = CatBoostRegressor(**CATBOOST_PARAMS)
    cb.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=100, verbose=False)
    preds_cb = cb.predict(X_test)
    rmse_cb = np.sqrt(mean_squared_error(y_test, preds_cb))
    mae_cb = mean_absolute_error(y_test, preds_cb)
    
    # XGBoost
    xgb = XGBRegressor(**XGBOOST_PARAMS)
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds_xgb = xgb.predict(X_test)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, preds_xgb))
    mae_xgb = mean_absolute_error(y_test, preds_xgb)
    
    print(f"Overall Test CatBoost RMSE: {rmse_cb:.3f}, MAE: {mae_cb:.3f}")
    print(f"Overall Test XGBoost RMSE:  {rmse_xgb:.3f}, MAE: {mae_xgb:.3f}")
    
    test_df["preds_cb"] = preds_cb
    test_df["preds_xgb"] = preds_xgb
    
    # Per-city breakdown
    city_rows = []
    all_cities = sorted(df["City"].unique())
    for city in all_cities:
        city_test = test_df[test_df["City"] == city]
        if len(city_test) == 0:
            print(f"  {city:15s} | No 2022 test records (skipped)")
            city_rows.append({
                "City": city,
                "Test Records": 0,
                "CatBoost RMSE": "-",
                "CatBoost MAE": "-",
                "XGBoost RMSE": "-",
                "XGBoost MAE": "-"
            })
        else:
            c_rmse_cb = np.sqrt(mean_squared_error(city_test[target], city_test["preds_cb"]))
            c_mae_cb = mean_absolute_error(city_test[target], city_test["preds_cb"])
            c_rmse_xgb = np.sqrt(mean_squared_error(city_test[target], city_test["preds_xgb"]))
            c_mae_xgb = mean_absolute_error(city_test[target], city_test["preds_xgb"])
            print(f"  {city:15s} ({len(city_test):4d} records) | CB RMSE: {c_rmse_cb:.3f} | XGB RMSE: {c_rmse_xgb:.3f}")
            city_rows.append({
                "City": city,
                "Test Records": len(city_test),
                "CatBoost RMSE": f"{c_rmse_cb:.3f}",
                "CatBoost MAE": f"{c_mae_cb:.3f}",
                "XGBoost RMSE": f"{c_rmse_xgb:.3f}",
                "XGBoost MAE": f"{c_mae_xgb:.3f}"
            })
            
    summary_row = {
        "City": "Overall",
        "Test Records": len(test_df),
        "CatBoost RMSE": f"{rmse_cb:.3f}",
        "CatBoost MAE": f"{mae_cb:.3f}",
        "XGBoost RMSE": f"{rmse_xgb:.3f}",
        "XGBoost MAE": f"{mae_xgb:.3f}"
    }
    city_rows.append(summary_row)
    
    metrics_df = pd.DataFrame(city_rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(TABLES_DIR / "datesplit_metrics.csv", index=False)
    
    # Export LaTeX table with compact column headers for table*
    tex_df = metrics_df.copy()
    tex_df.columns = ["City", "Records", "CB RMSE", "CB MAE", "XGB RMSE", "XGB MAE"]
    save_latex_tabular(tex_df, TABLES_DIR / "datesplit_metrics.tex")
    
    # Export compact overall table for single-column report
    overall_df = pd.DataFrame([
        {"Model": "CatBoost", "Test Records": f"{len(test_df):,}", "RMSE": f"{rmse_cb:.3f}", "MAE": f"{mae_cb:.3f}"},
        {"Model": "XGBoost", "Test Records": f"{len(test_df):,}", "RMSE": f"{rmse_xgb:.3f}", "MAE": f"{mae_xgb:.3f}"}
    ])
    save_latex_tabular(overall_df, TABLES_DIR / "report_datesplit_overall.tex")
    REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    save_latex_tabular(overall_df, REPORT_TABLES_DIR / "report_datesplit_overall.tex")
    
    # Save predictions for plotting
    test_df.to_csv(PROCESSED_DATA_DIR / "datesplit_preds.csv", index=False)
    
    return metrics_df, test_df

if __name__ == "__main__":
    evaluate_date_split()

