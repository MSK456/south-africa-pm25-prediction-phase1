import glob
import pandas as pd
from config import RAW_DATA_DIR

def load_raw_data():
    files = sorted(glob.glob(str(RAW_DATA_DIR / "*.csv")))
    dfs = []
    
    for f in files:
        df = pd.read_csv(f, comment="#", skipinitialspace=True, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        
        # Keep only South Africa and PM2.5 observations
        filtered = df[(df["Country"] == "ZA") & (df["Specie"] == "pm25")]
        dfs.append(filtered)
        print(f"Loaded {len(filtered)} PM2.5 rows from {f}")
        
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total raw South Africa PM2.5 records: {len(combined)}")
    print("\nRecords per city:")
    print(combined["City"].value_counts())
    return combined

if __name__ == "__main__":
    load_raw_data()
