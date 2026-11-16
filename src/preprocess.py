import pandas as pd
from config import CITY_COORDS, PROCESSED_DATA_DIR
from load_data import load_raw_data

def preprocess_data(raw_df=None):
    if raw_df is None:
        raw_df = load_raw_data()
        
    df = raw_df.copy()
    
    # Keep only the target date range 2019-01-01 to 2022-12-31
    df = df[(df["Date"] >= "2019-01-01") & (df["Date"] <= "2022-12-31")]
    
    # Rename median to Median_PM25 and ensure numeric type
    df = df.rename(columns={"median": "Median_PM25"})
    df["Median_PM25"] = pd.to_numeric(df["Median_PM25"], errors="coerce")
    
    # Keep only relevant columns
    df = df[["Date", "City", "Median_PM25"]]
    
    # Drop duplicates if any
    df = df.drop_duplicates(subset=["Date", "City"])
    
    clean_dfs = []
    
    for city in sorted(df["City"].unique()):
        sub = df[df["City"] == city].copy()
        sub["Date"] = pd.to_datetime(sub["Date"])
        sub = sub.sort_values("Date")
        
        # Reindex between city's own first and last available dates
        city_start = sub["Date"].min()
        city_end = sub["Date"].max()
        full_dates = pd.date_range(start=city_start, end=city_end, freq="D")
        
        sub = sub.set_index("Date").reindex(full_dates)
        sub["City"] = city
        
        # Add coordinates
        lat, lon = CITY_COORDS[city]
        sub["Lat"] = lat
        sub["Long"] = lon
        
        # Impute missing values with that city's mean
        city_mean = sub["Median_PM25"].mean()
        sub["Median_PM25"] = sub["Median_PM25"].fillna(city_mean)
        
        sub = sub.reset_index().rename(columns={"index": "Date"})
        sub["Date"] = sub["Date"].dt.strftime("%Y-%m-%d")
        
        print(f"{city}: {len(sub)} daily rows ({sub['Date'].min()} to {sub['Date'].max()}), mean PM2.5 = {city_mean:.2f}")
        clean_dfs.append(sub)
        
    clean_df = pd.concat(clean_dfs, ignore_index=True)
    clean_df = clean_df.sort_values(["City", "Date"]).reset_index(drop=True)
    
    # Ensure processed directory exists and save
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "pm25_clean.csv"
    clean_df.to_csv(output_path, index=False)
    print(f"\nSaved clean data with {len(clean_df)} rows to {output_path}")
    
    return clean_df

if __name__ == "__main__":
    preprocess_data()
