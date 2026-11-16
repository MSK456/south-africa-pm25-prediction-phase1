import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

from config import (
    CATBOOST_PARAMS,
    PROCESSED_DATA_DIR,
    TABLES_DIR,
    UNMONITORED_CITIES,
    XGBOOST_PARAMS,
    save_latex_tabular,
)
from features import add_date_features

def run_no_station_experiment():
    print("\n=== Cities Without Stations Experiment (Approximation) ===")
    
    # Load clean data
    data_path = PROCESSED_DATA_DIR / "pm25_clean.csv"
    df = pd.read_csv(data_path)
    df = add_date_features(df)
    
    features = [
        "dayofweek", "quarter", "month", "year",
        "dayofyear", "dayofmonth", "weekofyear",
        "Lat", "Long"
    ]
    target = "Median_PM25"
    
    # Determine best model from Step B cv_fold_rmse.csv
    cv_table_path = TABLES_DIR / "cv_fold_rmse.csv"
    best_model_name = "XGBoost"
    if cv_table_path.exists():
        cv_df = pd.read_csv(cv_table_path)
        mean_row = cv_df[cv_df["fold"] == "Mean"]
        if not mean_row.empty:
            cb_rmse = float(mean_row["cb_rmse"].iloc[0])
            xgb_rmse = float(mean_row["xgb_rmse"].iloc[0])
            best_model_name = "CatBoost" if cb_rmse < xgb_rmse else "XGBoost"
            print(f"Selected best model from CV: {best_model_name} (CB: {cb_rmse:.3f}, XGB: {xgb_rmse:.3f})")
            
    # Leave-One-City-Out Validation across 12 station cities
    print("\nComputing Leave-One-City-Out (LOCO) CV to assess spatial generalization:")
    loco_results = []
    cities = sorted(df["City"].unique())
    
    for left_out in cities:
        train_df = df[df["City"] != left_out]
        test_df = df[df["City"] == left_out]
        
        X_tr, y_tr = train_df[features], train_df[target]
        X_te, y_te = test_df[features], test_df[target]
        
        if best_model_name == "CatBoost":
            model = CatBoostRegressor(**CATBOOST_PARAMS)
            model.fit(X_tr, y_tr, eval_set=(X_te, y_te), early_stopping_rounds=100, verbose=False)
        else:
            model = XGBRegressor(**XGBOOST_PARAMS)
            model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
            
        preds = model.predict(X_te)
        loco_rmse = np.sqrt(mean_squared_error(y_te, preds))
        loco_mae = mean_absolute_error(y_te, preds)
        
        loco_results.append({
            "Left-Out City": left_out,
            "LOCO RMSE": f"{loco_rmse:.3f}",
            "LOCO MAE": f"{loco_mae:.3f}",
            "raw_rmse": loco_rmse,
            "raw_mae": loco_mae
        })
        print(f"  LOCO {left_out:15s} | RMSE: {loco_rmse:.3f} | MAE: {loco_mae:.3f}")
        
    loco_df = pd.DataFrame(loco_results)
    mean_loco_rmse = float(np.mean([r["raw_rmse"] for r in loco_results]))
    mean_loco_mae = float(np.mean([r["raw_mae"] for r in loco_results]))
    
    loco_summary = {
        "Left-Out City": "Average",
        "LOCO RMSE": f"{mean_loco_rmse:.3f}",
        "LOCO MAE": f"{mean_loco_mae:.3f}"
    }
    loco_tex_df = pd.concat([
        loco_df[["Left-Out City", "LOCO RMSE", "LOCO MAE"]],
        pd.DataFrame([loco_summary])
    ], ignore_index=True)
    
    loco_tex_df.to_csv(TABLES_DIR / "loco_cv_rmse.csv", index=False)
    save_latex_tabular(loco_tex_df, TABLES_DIR / "loco_cv_rmse.tex")
    print(f"\nAverage LOCO RMSE: {mean_loco_rmse:.3f}, MAE: {mean_loco_mae:.3f}")
    
    # Train model on all 12 cities
    print(f"\nTraining {best_model_name} on all 12 station cities...")
    X_all = df[features]
    y_all = df[target]
    
    if best_model_name == "CatBoost":
        final_model = CatBoostRegressor(**CATBOOST_PARAMS)
        final_model.fit(X_all, y_all, verbose=False)
    else:
        # For training on all data without a separate eval set
        xgb_p = XGBOOST_PARAMS.copy()
        xgb_p.pop("early_stopping_rounds", None)
        final_model = XGBRegressor(**xgb_p)
        final_model.fit(X_all, y_all, verbose=False)
        
    # Generate daily records for 2022 for the 6 unmonitored cities
    dates_2022 = pd.date_range("2022-01-01", "2022-12-31", freq="D")
    pred_rows = []
    
    for city, (lat, lon) in UNMONITORED_CITIES.items():
        city_df = pd.DataFrame({
            "Date": dates_2022.strftime("%Y-%m-%d"),
            "City": city,
            "Lat": lat,
            "Long": lon
        })
        city_df = add_date_features(city_df)
        preds = final_model.predict(city_df[features])
        city_df["Predicted_PM25"] = preds
        pred_rows.append(city_df)
        
    pred_all_df = pd.concat(pred_rows, ignore_index=True)
    
    # Compute mean predicted PM2.5 per city
    city_means = pred_all_df.groupby("City")["Predicted_PM25"].mean().reset_index()
    city_means.columns = ["City", "Mean Predicted PM2.5"]
    city_means["Latitude"] = city_means["City"].map(lambda c: UNMONITORED_CITIES[c][0])
    city_means["Longitude"] = city_means["City"].map(lambda c: UNMONITORED_CITIES[c][1])
    city_means["Mean Predicted PM2.5"] = city_means["Mean Predicted PM2.5"].apply(lambda v: f"{v:.3f}")
    
    city_means = city_means[["City", "Latitude", "Longitude", "Mean Predicted PM2.5"]]
    
    print("\nPredicted Daily 2022 Mean PM2.5 for Unmonitored Cities:")
    print(city_means.to_string(index=False))
    
    city_means.to_csv(TABLES_DIR / "no_station_predictions.csv", index=False)
    
    # Export LaTeX table with coordinates formatted to 4 decimal places
    tex_means = city_means.copy()
    tex_means["Latitude"] = tex_means["City"].map(lambda c: f"{UNMONITORED_CITIES[c][0]:.4f}")
    tex_means["Longitude"] = tex_means["City"].map(lambda c: f"{UNMONITORED_CITIES[c][1]:.4f}")
    save_latex_tabular(tex_means, TABLES_DIR / "no_station_predictions.tex")
    
    pred_all_df.to_csv(PROCESSED_DATA_DIR / "no_station_daily_preds_2022.csv", index=False)
    return city_means, loco_tex_df

if __name__ == "__main__":
    run_no_station_experiment()

