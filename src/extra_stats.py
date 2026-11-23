import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

from config import PROCESSED_DATA_DIR, TABLES_DIR, SEED
from aqi_classifiers import assign_aqi_class

def run_extra_statistics():
    print("\n=== Computing Extra Statistics and Precision/Recall ===")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    clean_path = PROCESSED_DATA_DIR / "pm25_clean.csv"
    if not clean_path.exists():
        raise FileNotFoundError(f"Clean data not found at {clean_path}")
    df = pd.read_csv(clean_path)
    
    extra_rows = []
    
    # 1. Overall mean Median_PM25 and rows per city
    overall_mean = float(df["Median_PM25"].mean())
    extra_rows.append({
        "category": "overall_mean_pm25",
        "item": "all_cities",
        "value": f"{overall_mean:.6f}",
        "min_val": f"{df['Median_PM25'].min():.6f}",
        "max_val": f"{df['Median_PM25'].max():.6f}",
        "details": f"Total cleaned rows: {len(df)}"
    })
    
    city_counts = df["City"].value_counts()
    for city, count in city_counts.items():
        extra_rows.append({
            "category": "rows_per_city",
            "item": city,
            "value": str(count),
            "min_val": "",
            "max_val": "",
            "details": f"Mean PM2.5: {df[df['City'] == city]['Median_PM25'].mean():.2f}"
        })
        
    # 2. Readings equal to 999 with city and date
    readings_999 = df[df["Median_PM25"] == 999.0]
    for _, r in readings_999.iterrows():
        extra_rows.append({
            "category": "reading_999",
            "item": r["City"],
            "value": "999.0",
            "min_val": "999.0",
            "max_val": "999.0",
            "details": f"Date: {r['Date']}"
        })
        
    # 3. AQI class counts in train and test parts (test_size=0.30, random_state=42)
    df["AQI_Class"] = df["Median_PM25"].apply(assign_aqi_class)
    X = df[["Median_PM25"]]
    y = df["AQI_Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=SEED
    )
    
    train_counts = y_train.value_counts().sort_index()
    for c in range(6):
        cnt = int(train_counts.get(c, 0))
        extra_rows.append({
            "category": "aqi_train_class_count",
            "item": f"Class {c}",
            "value": str(cnt),
            "min_val": "",
            "max_val": "",
            "details": f"Train partition (N={len(y_train)})"
        })
        
    test_counts = y_test.value_counts().sort_index()
    for c in range(6):
        cnt = int(test_counts.get(c, 0))
        extra_rows.append({
            "category": "aqi_test_class_count",
            "item": f"Class {c}",
            "value": str(cnt),
            "min_val": "",
            "max_val": "",
            "details": f"Test partition (N={len(y_test)})"
        })
        
    # 4. Classifiers: macro precision, recall, F1 and misclassifications
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "SVM": SVC(random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED)
    }
    
    pr_rows = []
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        
        pr_rows.append({
            "Model": name,
            "Macro Precision": f"{prec:.4f}",
            "Macro Recall": f"{rec:.4f}",
            "Macro F1": f"{f1:.4f}"
        })
        
        extra_rows.append({
            "category": "aqi_macro_metric",
            "item": name,
            "value": f"{f1:.4f}",
            "min_val": f"{prec:.4f}",
            "max_val": f"{rec:.4f}",
            "details": f"Macro Prec={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}"
        })
        
        eval_df = X_test.copy()
        eval_df["true_class"] = y_test
        eval_df["pred_class"] = y_pred
        misc = eval_df[eval_df["true_class"] != eval_df["pred_class"]]
        
        extra_rows.append({
            "category": "classifier_misclassification_total",
            "item": name,
            "value": str(len(misc)),
            "min_val": "",
            "max_val": "",
            "details": f"Total misclassified out of {len(y_test)}"
        })
        
        for (tc, pc), grp in misc.groupby(["true_class", "pred_class"]):
            min_p = float(grp["Median_PM25"].min())
            max_p = float(grp["Median_PM25"].max())
            extra_rows.append({
                "category": "classifier_misclassification_pair",
                "item": name,
                "value": str(len(grp)),
                "min_val": f"{min_p:.4f}",
                "max_val": f"{max_p:.4f}",
                "details": f"True Class {tc} -> Predicted Class {pc} (count={len(grp)})"
            })
            
    # 5. Maximum forecast value per city
    forecast_path = PROCESSED_DATA_DIR / "future_forecast_2022_2023.csv"
    if forecast_path.exists():
        fdf = pd.read_csv(forecast_path)
        city_maxes = fdf.groupby("City")["Predicted_PM25"].max()
        for city, max_val in city_maxes.items():
            extra_rows.append({
                "category": "forecast_max_pm25",
                "item": city,
                "value": f"{max_val:.4f}",
                "min_val": f"{max_val:.2f}",
                "max_val": f"{max_val:.4f}",
                "details": f"2022-11-26 to 2023-12-31 recursive forecast (rounded: {max_val:.2f})"
            })
            
    # 6. Additional split counts and parameters for report traceability
    split_date = pd.to_datetime("2022-01-01")
    df["Date_dt"] = pd.to_datetime(df["Date"])
    train_datesplit = len(df[df["Date_dt"] < split_date])
    test_datesplit = len(df[df["Date_dt"] >= split_date])
    extra_rows.append({
        "category": "datesplit_partition_counts",
        "item": "train_pre_2022",
        "value": str(train_datesplit),
        "min_val": "",
        "max_val": "",
        "details": "Train rows (< 2022-01-01)"
    })
    extra_rows.append({
        "category": "datesplit_partition_counts",
        "item": "test_post_2022",
        "value": str(test_datesplit),
        "min_val": "",
        "max_val": "",
        "details": "Test rows (>= 2022-01-01)"
    })
    extra_rows.append({
        "category": "forecast_parameters",
        "item": "horizon_days",
        "value": "401",
        "min_val": "2022-11-26",
        "max_val": "2023-12-31",
        "details": "401 forecast days from 2022-11-26 to 2023-12-31"
    })
            
    # Save aqi_precision_recall.csv
    pr_df = pd.DataFrame(pr_rows)
    pr_path = TABLES_DIR / "aqi_precision_recall.csv"
    pr_df.to_csv(pr_path, index=False)
    print(f"Saved AQI precision/recall to {pr_path}")
    print(pr_df.to_string(index=False))
    
    # Save extra_stats.csv
    extra_df = pd.DataFrame(extra_rows)
    extra_path = TABLES_DIR / "extra_stats.csv"
    extra_df.to_csv(extra_path, index=False)
    print(f"\nSaved extra stats ({len(extra_df)} rows) to {extra_path}")
    
    return extra_df, pr_df

if __name__ == "__main__":
    run_extra_statistics()
