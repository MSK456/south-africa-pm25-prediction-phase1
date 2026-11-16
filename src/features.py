import pandas as pd
from config import PROCESSED_DATA_DIR

def add_date_features(df):
    dt = pd.to_datetime(df["Date"])
    df["dayofweek"] = dt.dt.dayofweek
    df["quarter"] = dt.dt.quarter
    df["month"] = dt.dt.month
    df["year"] = dt.dt.year
    df["dayofyear"] = dt.dt.dayofyear
    df["dayofmonth"] = dt.dt.day
    df["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    return df

def add_lag_features(df):
    # Group by city and compute annual 52-week lag shifts
    dfs = []
    for city, grp in df.groupby("City", as_index=False):
        grp = grp.sort_values("Date").copy()
        grp["lag1"] = grp["Median_PM25"].shift(364)
        grp["lag2"] = grp["Median_PM25"].shift(728)
        grp["lag3"] = grp["Median_PM25"].shift(1092)
        dfs.append(grp)
    return pd.concat(dfs, ignore_index=True)

def build_features(df=None):
    if df is None:
        data_path = PROCESSED_DATA_DIR / "pm25_clean.csv"
        df = pd.read_csv(data_path)
        
    df = add_date_features(df)
    df = add_lag_features(df)
    return df

if __name__ == "__main__":
    feat_df = build_features()
    print("Features created successfully. Columns:", list(feat_df.columns))
    print(feat_df[["Date", "City", "Median_PM25", "lag1", "lag2", "lag3", "dayofweek", "month"]].head())

