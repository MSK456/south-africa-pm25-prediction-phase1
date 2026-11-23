import sys
from pathlib import Path

# Add src folder to module search path
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocess import preprocess_data
from cv_regression import run_all_cv
from datesplit_eval import evaluate_date_split
from no_station import run_no_station_experiment
from forecast import run_future_forecast
from aqi_classifiers import run_aqi_classification
from extra_stats import run_extra_statistics
from make_figures import make_all_figures
from compare_with_paper import compare_results

def run_pipeline():
    print("=" * 60)
    print("STARTING END-TO-END PM2.5 SOUTH AFRICA PIPELINE")
    print("=" * 60)
    
    # Step 1: Preprocess raw WAQI data
    print("\n[1/9] Running Preprocessing...")
    preprocess_data()
    
    # Step 2: Cross-validation regression (GroupKFold and TimeSeriesSplit)
    print("\n[2/9] Running Cross-Validation Regression...")
    run_all_cv()
    
    # Step 3: Date split evaluation (< 2022 vs >= 2022)
    print("\n[3/9] Running Date Split Evaluation...")
    evaluate_date_split()
    
    # Step 4: Cities without recording stations experiment
    print("\n[4/9] Running Cities Without Stations Experiment...")
    run_no_station_experiment()
    
    # Step 5: Future recursive forecasting
    print("\n[5/9] Running Future Forecasting (2022-2023)...")
    run_future_forecast()
    
    # Step 6: AQI status classification & demo
    print("\n[6/9] Running AQI Status Classification & Demo...")
    run_aqi_classification()
    
    # Step 7: Extra statistics and precision/recall
    print("\n[7/9] Running Extra Statistics & Precision/Recall...")
    run_extra_statistics()
    
    # Step 8: Figure generation and asset synchronization
    print("\n[8/9] Generating Report Figures and Syncing Tables...")
    make_all_figures()
    
    # Step 9: Comparison with base paper
    print("\n[9/9] Comparing with Base Paper...")
    compare_results()
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline()

