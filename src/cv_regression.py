import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from config import (
    CATBOOST_PARAMS,
    PROCESSED_DATA_DIR,
    TABLES_DIR,
    XGBOOST_PARAMS,
    save_latex_tabular,
)
from features import build_features

def run_group_cv(df):
    features = [
        "dayofweek", "quarter", "month", "year",
        "dayofyear", "dayofmonth", "weekofyear",
        "lag1", "lag2", "lag3"
    ]
    target = "Median_PM25"
    
    gkf = GroupKFold(n_splits=5)
    groups = df["City"].values
    
    oof_cb = np.zeros(len(df))
    oof_xgb = np.zeros(len(df))
    
    cb_fold_rmses = []
    xgb_fold_rmses = []
    cb_fold_maes = []
    xgb_fold_maes = []
    fold_details = []
    
    print("\n=== Primary Protocol: 5-Fold GroupKFold (Held-Out Cities) ===")
    for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=groups)):
        train_cities = df.iloc[train_idx]["City"].unique()
        test_cities = df.iloc[test_idx]["City"].unique()
        
        X_train, y_train = df.iloc[train_idx][features], df.iloc[train_idx][target]
        X_test, y_test = df.iloc[test_idx][features], df.iloc[test_idx][target]
        
        # Train CatBoost
        cb = CatBoostRegressor(**CATBOOST_PARAMS)
        cb.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=100, verbose=False)
        preds_cb = cb.predict(X_test)
        oof_cb[test_idx] = preds_cb
        rmse_cb = np.sqrt(mean_squared_error(y_test, preds_cb))
        mae_cb = mean_absolute_error(y_test, preds_cb)
        cb_fold_rmses.append(rmse_cb)
        cb_fold_maes.append(mae_cb)
        
        # Train XGBoost
        xgb = XGBRegressor(**XGBOOST_PARAMS)
        xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        preds_xgb = xgb.predict(X_test)
        oof_xgb[test_idx] = preds_xgb
        rmse_xgb = np.sqrt(mean_squared_error(y_test, preds_xgb))
        mae_xgb = mean_absolute_error(y_test, preds_xgb)
        xgb_fold_rmses.append(rmse_xgb)
        xgb_fold_maes.append(mae_xgb)
        
        print(f"Fold {fold}: Train cities ({len(train_cities)}), Test cities ({len(test_cities)}): {list(test_cities)}")
        print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        print(f"  CatBoost RMSE: {rmse_cb:.3f} | XGBoost RMSE: {rmse_xgb:.3f}")
        
        fold_details.append({
            "fold": f"Fold {fold}",
            "train_cities_count": len(train_cities),
            "test_cities_count": len(test_cities),
            "test_cities": ", ".join(test_cities),
            "cb_rmse": rmse_cb,
            "cb_mae": mae_cb,
            "xgb_rmse": rmse_xgb,
            "xgb_mae": mae_xgb,
        })
        
    df_out = df.copy()
    df_out["preds_cb"] = oof_cb
    df_out["preds_xgb"] = oof_xgb
    
    oof_rmse_cb = np.sqrt(mean_squared_error(df[target], oof_cb))
    oof_rmse_xgb = np.sqrt(mean_squared_error(df[target], oof_xgb))
    
    print("\n--- GroupKFold Summary ---")
    print(f"Mean Fold CatBoost RMSE: {np.mean(cb_fold_rmses):.3f}")
    print(f"Mean Fold XGBoost RMSE:  {np.mean(xgb_fold_rmses):.3f}")
    print(f"Overall OOF CatBoost RMSE: {oof_rmse_cb:.3f}")
    print(f"Overall OOF XGBoost RMSE:  {oof_rmse_xgb:.3f}")
    
    # Save fold table
    fold_df = pd.DataFrame(fold_details)
    summary_row = {
        "fold": "Mean",
        "train_cities_count": "-",
        "test_cities_count": "-",
        "test_cities": "All",
        "cb_rmse": float(np.mean(cb_fold_rmses)),
        "cb_mae": float(np.mean(cb_fold_maes)),
        "xgb_rmse": float(np.mean(xgb_fold_rmses)),
        "xgb_mae": float(np.mean(xgb_fold_maes)),
    }
    fold_df = pd.concat([fold_df, pd.DataFrame([summary_row])], ignore_index=True)
    
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(TABLES_DIR / "cv_fold_rmse.csv", index=False)
    
    # Format and save LaTeX table
    tex_df = fold_df[["fold", "train_cities_count", "test_cities_count", "cb_rmse", "xgb_rmse"]].copy()
    tex_df.columns = ["Fold", "Train Cities", "Test Cities", "CB RMSE", "XGB RMSE"]
    tex_df["CB RMSE"] = tex_df["CB RMSE"].apply(lambda v: f"{v:.3f}")
    tex_df["XGB RMSE"] = tex_df["XGB RMSE"].apply(lambda v: f"{v:.3f}")
    save_latex_tabular(tex_df, TABLES_DIR / "cv_fold_rmse.tex")
        
    return df_out, fold_df

def run_all_cv():
    df = build_features()
    df_oof, fold_df = run_group_cv(df)
    
    # Save out-of-fold predictions
    oof_path = PROCESSED_DATA_DIR / "pm25_oof_preds.csv"
    df_oof.to_csv(oof_path, index=False)
    print(f"\nSaved OOF predictions to {oof_path}")
    return df_oof, fold_df

if __name__ == "__main__":
    run_all_cv()
